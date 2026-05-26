"""
MoiraiForecastModel — обёртка над Salesforce Moirai-2 в нашем интерфейсе ForecastModel.

Конфигурация через переменные окружения (читаются в __init__ если параметры
не переданы явно):
  MOIRAI_2_SIZE              — small / base / large (default: small)
  MOIRAI_2_CONTEXT_LENGTH    — макс контекст (default: 4096)
  MOIRAI_2_PREDICTION_LENGTH — макс горизонт (default: 64)
  MOIRAI_2_PATCH_SIZE        — 'auto' или int (default: auto)
  MOIRAI_2_NUM_SAMPLES       — число samples (default: 100)

Динамические опции из model_options в /forecast/backtest пока не применяются
(TODO: добавить _maybe_reconfigure хук); для смены размера модели нужно
перезапустить сервер с другой env.

Реализует все методы predict_*, get_info, get_adapter_modules, get_lora_target_modules.

Стратегия для OHLC:
  Moirai-2 поддерживает multivariate напрямую через target_dim, но интеграция чище
  если обрабатывать каждый OHLC канал как отдельный univariate series и батчить их
  в один forward (4 канала × B окон = 4B входов в одном вызове). Это совпадает
  с тем как PatchTSTFM делает внутри (rearrange B T N → (B N) T).

OHLC физическая корректность:
  После inference применяем тот же декодер что и для PatchTST — обеспечиваем
  high >= max(open,close), low <= min(open,close) на уровне q50 (см. patchtst_decoder).
"""
from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from ..base import ForecastModel, OHLCQuantileForecast, QuantileForecast
from ..patchtst.patchtst_decoder import PatchTSTDecoder
from .moirai_runtime import (
    MoiraiRuntime, DEFAULT_QUANTILE_LEVELS,
    DEFAULT_CONTEXT_LENGTH, DEFAULT_PREDICTION_LENGTH,
    DEFAULT_PATCH_SIZE, DEFAULT_NUM_SAMPLES,
)

logger = logging.getLogger(__name__)


OHLC_CHANNELS = ("open", "high", "low", "close")


class MoiraiForecastModel(ForecastModel):
    """
    Foundation model Moirai-2 для time series forecasting.

    Параметры (через MoiraiModelOptions):
      size              — small / base / large
      context_length    — макс длина контекста (default 4096)
      prediction_length — макс длина прогноза (default 64)
      patch_size        — 'auto' или конкретный int
      num_samples       — число samples для квантильной оценки

    Загрузка ленивая — первый вызов predict_* триггерит ensure_loaded.
    Соответственно первый forecast будет медленнее (~10-30 сек на скачивание весов).
    """

    model_name = "Moirai-2"
    model_version = "2.0-R"

    def __init__(
            self,
            size: str | None = None,
            context_length: int | None = None,
            prediction_length: int | None = None,
            patch_size: str | int | None = None,
            num_samples: int | None = None,
    ) -> None:
        # Если параметр не передан явно — берём из ENV или дефолта runtime'а.
        # Это позволяет переключать size/context без правки кода — просто
        # `MOIRAI_2_SIZE=base uvicorn ...`.
        size              = size              or os.getenv("MOIRAI_2_SIZE", "small")
        context_length    = context_length    or int(os.getenv("MOIRAI_2_CONTEXT_LENGTH", DEFAULT_CONTEXT_LENGTH))
        prediction_length = prediction_length or int(os.getenv("MOIRAI_2_PREDICTION_LENGTH", DEFAULT_PREDICTION_LENGTH))
        patch_size        = patch_size        or os.getenv("MOIRAI_2_PATCH_SIZE", DEFAULT_PATCH_SIZE)
        num_samples       = num_samples       or int(os.getenv("MOIRAI_2_NUM_SAMPLES", DEFAULT_NUM_SAMPLES))

        self._runtime = MoiraiRuntime(
            size=size,
            context_length=context_length,
            prediction_length=prediction_length,
            patch_size=patch_size,
            num_samples=num_samples,
        )
        # OHLC decoder для гарантии корректности свечей (high >= max, low <= min)
        self._ohlc_decoder = PatchTSTDecoder()
        # Последнее окно — для дебага и info-эндпоинта
        self._last_input_window_info: dict[str, Any] = {}

    # ==================================================================
    # Inference methods
    # ==================================================================

    def predict_line_exact(
            self,
            candles: list[dict[str, float]],
            horizon: int,
            context: dict | None = None,
    ) -> list[float]:
        """Точечный прогноз close — медиана из samples."""
        quantiles = self.predict_line_quantiles(candles, horizon, context)
        return list(quantiles.q50)

    def predict_line_quantiles(
            self,
            candles: list[dict[str, float]],
            horizon: int,
            context: dict | None = None,
    ) -> QuantileForecast:
        """Квантильный прогноз close — sampling + percentiles."""
        self._runtime.ensure_loaded()
        close_series = [float(c["close"]) for c in candles]
        samples = self._forecast_samples(close_series, horizon)  # shape (num_samples, horizon)
        q = self._samples_to_quantiles(samples)
        self._record_input_info(len(candles), horizon, channels=1)
        return q

    def predict_ohlc_exact(
            self,
            candles: list[dict[str, float]],
            horizon: int,
            context: dict | None = None,
    ) -> list[dict[str, float]]:
        """Точечный OHLC прогноз — медианы по каждому каналу + корректность."""
        ohlc_q = self.predict_ohlc_quantiles(candles, horizon, context)
        return ohlc_q.median_candles()

    def predict_ohlc_quantiles(
            self,
            candles: list[dict[str, float]],
            horizon: int,
            context: dict | None = None,
    ) -> OHLCQuantileForecast:
        """
        OHLC квантильный прогноз — каждый канал прогнозируется отдельно через Moirai,
        результаты собираются в OHLCQuantileForecast.

        Для скорости батчим все 4 канала в один forward вместо 4 последовательных.
        """
        self._runtime.ensure_loaded()
        # Собираем 4 univariate серии в один batch
        series_per_channel = {
            ch: [float(c[ch]) for c in candles] for ch in OHLC_CHANNELS
        }
        samples_per_channel = self._forecast_samples_batch(
            list(series_per_channel.values()), horizon,
        )  # list of 4 tensors, каждый (num_samples, horizon)

        quantiles_per_channel = {
            ch: self._samples_to_quantiles(samples_per_channel[i])
            for i, ch in enumerate(OHLC_CHANNELS)
        }
        # Применяем физическую корректность ко всем 5 квантилям OHLC
        # (high >= max(o,c,l), low <= min(o,c,h)) — переиспользуем PatchTSTDecoder
        open_q, high_q, low_q, close_q = self._ohlc_decoder.enforce_ohlc_consistency(
            quantiles_per_channel["open"],
            quantiles_per_channel["high"],
            quantiles_per_channel["low"],
            quantiles_per_channel["close"],
        )
        ohlc = OHLCQuantileForecast(
            open=open_q, high=high_q, low=low_q, close=close_q,
            volume=None,   # Moirai 2 не предсказывает volume — отдельная задача
        )

        self._record_input_info(len(candles), horizon, channels=4)
        return ohlc

    def get_info(self) -> dict:
        return {
            "name": self.model_name,
            "version": self.model_version,
            "model_id": self._runtime.model_id,
            "size": self._runtime.size,
            "loaded": self._runtime.is_loaded,
            "device": self._runtime.device,
            "type": "multivariate-quantile-forecast-model",
            "context_length": self._runtime.context_length,
            "prediction_length": self._runtime.prediction_length,
            "patch_size": self._runtime.patch_size,
            "num_samples": self._runtime.num_samples,
            "quantile_levels": self._runtime.quantile_levels,
            "supports_quantiles": True,
            "supports_ohlc": True,
            "include_volume_in_forecast": False,
            "last_input_window_info": self._last_input_window_info,
            "load_error": self._runtime.load_error,
        }

    # ==================================================================
    # Adapter modules (для head/input training и LoRA)
    # ==================================================================

    def get_adapter_modules(self) -> dict:
        """Делегируем runtime'у — он знает структуру MoiraiModule."""
        return self._runtime.get_adapter_modules()

    def get_lora_target_modules(self) -> list[str]:
        return self._runtime.get_lora_target_modules()

    # ==================================================================
    # Internal forecast machinery
    # ==================================================================

    def _forecast_samples(self, series: list[float], horizon: int):
        """
        Прогон одной univariate серии. Возвращает torch tensor (num_samples, horizon).
        """
        return self._forecast_samples_batch([series], horizon)[0]

    def _forecast_samples_batch(self, series_list: list[list[float]], horizon: int):
        """
        Батчевый прогон нескольких univariate серий ОДНИМ forward'ом.

        Возвращает список тензоров (по одному на серию), каждый shape (num_samples, horizon).

        Все серии должны иметь одинаковую длину (батч-тензор).
        """
        import torch

        if not series_list:
            return []
        n_series = len(series_list)
        ctx_len = len(series_list[0])
        if any(len(s) != ctx_len for s in series_list):
            raise ValueError(
                "[Moirai] _forecast_samples_batch: все серии должны иметь одинаковую длину."
            )

        # Moirai API ожидает (batch, ctx_len, target_dim=1)
        past_target = torch.tensor(series_list, dtype=torch.float32, device=self._runtime.device)
        past_target = past_target.unsqueeze(-1)  # (B, T, 1)
        past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
        past_is_pad = torch.zeros((n_series, ctx_len), dtype=torch.bool, device=self._runtime.device)

        # Если у нас другой horizon чем у forecaster — пересоздадим (быстрее
        # чем grow forecaster для каждого вызова, но мы храним один на runtime).
        if horizon != self._runtime.prediction_length:
            self._rebuild_forecaster(horizon=horizon)

        with torch.no_grad():
            samples = self._runtime.forecaster(
                past_target=past_target,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad,
            )
            # samples shape: (B, num_samples, prediction_length, target_dim=1)
            samples = samples.squeeze(-1)   # (B, num_samples, prediction_length)

        # Триммим до запрошенного horizon (на случай если forecaster внутри округлил)
        samples = samples[:, :, :horizon]
        return [samples[i] for i in range(n_series)]   # list of (num_samples, horizon)

    def _samples_to_quantiles(self, samples) -> QuantileForecast:
        """samples: (num_samples, horizon) → QuantileForecast."""
        import torch

        quantiles_tensor = torch.quantile(
            samples,
            torch.tensor(DEFAULT_QUANTILE_LEVELS, device=samples.device),
            dim=0,
        )  # (5, horizon)
        as_lists = quantiles_tensor.cpu().tolist()
        return QuantileForecast(
            q10=as_lists[0],
            q25=as_lists[1],
            q50=as_lists[2],
            q75=as_lists[3],
            q90=as_lists[4],
        )

    def _rebuild_forecaster(self, horizon: int) -> None:
        """
        Если запрошен другой prediction_length — пересобираем forecaster wrapper.
        Сам MoiraiModule (главный вес) не меняется, только wrapper'овые параметры.
        """
        from uni2ts.model.moirai import MoiraiForecast

        self._runtime.prediction_length = horizon
        self._runtime.forecaster = MoiraiForecast(
            module=self._runtime.module,
            prediction_length=horizon,
            context_length=self._runtime.context_length,
            patch_size=self._runtime.patch_size,
            num_samples=self._runtime.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(self._runtime.device)
        self._runtime.forecaster.eval()

    def _record_input_info(self, context_len: int, horizon: int, channels: int) -> None:
        """Метаинфа последнего вызова — для get_info()."""
        self._last_input_window_info = {
            "requested_horizon": horizon,
            "context_length_provided": context_len,
            "channels": channels,
            "model_context_length": self._runtime.context_length,
            "model_prediction_length": self._runtime.prediction_length,
        }
