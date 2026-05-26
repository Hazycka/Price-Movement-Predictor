"""
PatchTST FM Foundation Model для прогнозирования временных рядов.
"""
import contextlib
import logging
import os

import pandas as pd

from ..base import ForecastModel, QuantileForecast, OHLCQuantileForecast
from .patchtst_runtime import PatchTSTRuntime, QUANTILE_LEVELS
from .patchtst_decoder import PatchTSTDecoder
from ...services.market_data.common import DEFAULT_FREQ

logger = logging.getLogger(__name__)


# Логгеры HF/tsfm которые мы подавляем при detailed_logs=False.
# Имена иерархические — повышение уровня корня каскадно глушит потомков.
_HF_LOGGER_NAMES = (
    "tsfm_public",
    "transformers",
    # На случай если внутри tsfm используются короткие имена
    "time_series_forecasting_pipeline",
    "modeling_patchtst_fm",
)


@contextlib.contextmanager
def _suppress_hf_logs(active: bool):
    """
    Context manager: при active=True временно поднимает уровень HF/tsfm логгеров
    до WARNING. При выходе восстанавливает оригинальные уровни.
    active=False — no-op.
    """
    if not active:
        yield
        return
    loggers = [logging.getLogger(name) for name in _HF_LOGGER_NAMES]
    saved = [(lg, lg.level) for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for lg, lvl in saved:
            lg.setLevel(lvl)

DEFAULT_MODEL_ID = "ibm-granite/granite-timeseries-patchtst-fm-r1"
OHLC_CHANNELS  = ["open", "high", "low", "close"]
OHLCV_CHANNELS = ["open", "high", "low", "close", "volume"]
TIMESTAMP_COL  = "timestamp"
SERIES_ID_COL  = "series_id"

# Формат колонок квантилей в выводе pipeline: {channel}_q{quantile}
# Пример: open_q0.1, close_q0.5
QUANTILE_COL_TEMPLATE = "{channel}_q{q}"

# Размер микро-батча для batched инференса. Контролирует VRAM:
# при context=8192 и batch=16 расход ~ десятки MB интермедиатов.
DEFAULT_INFERENCE_BATCH_SIZE = 16

# Число потоков-воркеров HF DataLoader для CPU-предобработки в TSFM pipeline.
# Подобрано под 8-ядерные CPU (например Ryzen 7 7800X3D): 6 воркеров оставляют
# ядра для главного потока и system overhead. Можно переопределить через
# env-переменную PATCHTST_FM_NUM_WORKERS если нужно.
DEFAULT_NUM_WORKERS = int(os.getenv("PATCHTST_FM_NUM_WORKERS", "6"))


class PatchTSTForecastModel(ForecastModel):

    def __init__(
            self,
            model_id: str = DEFAULT_MODEL_ID,
            model_version: str = "patchtst-fm-r1",
            target_column: str = "close",
            include_volume_in_forecast: bool = False,
    ) -> None:
        self.model_name = "PatchTST-FM"
        self.model_version = model_version
        self.model_id = os.getenv("PATCHTST_FM_MODEL_ID", model_id)
        self.target_column = os.getenv("PATCHTST_FM_TARGET_COLUMN", target_column)
        self.include_volume_in_forecast = include_volume_in_forecast
        self._runtime = PatchTSTRuntime(model_id=self.model_id)
        self._last_input_window_info: dict = {}

    def _candles_to_df(
            self,
            candles: list[dict[str, float]],
            forecast_columns: list[str],
            freq: str,
    ) -> pd.DataFrame:
        df = pd.DataFrame(candles)
        for col in forecast_columns:
            if col not in df.columns:
                df[col] = 0.0
        df[TIMESTAMP_COL] = pd.date_range(start="2000-01-01", periods=len(df), freq=freq)
        return df[[TIMESTAMP_COL] + forecast_columns]

    def _build_future_timestamps(
            self,
            context_df: pd.DataFrame,
            output_length: int,
            freq: str,
    ) -> pd.DataFrame:
        """
        Строит DataFrame с будущими временными метками.
        Pipeline требует future_time_series как DataFrame с timestamp колонкой — не int.

        output_length должен совпадать с prediction_length модели — TSFM pipeline
        интерпретирует длину future_time_series как ожидаемую длину выхода.
        Если запрошен меньший horizon — нарезка происходит уже после инференса.
        """
        last_ts = context_df[TIMESTAMP_COL].iloc[-1]
        future_timestamps = pd.date_range(start=last_ts, periods=output_length + 1, freq=freq)[1:]
        return pd.DataFrame({TIMESTAMP_COL: future_timestamps})

    # ------------------------------------------------------------------
    # Общие helpers для single и batched путей
    # ------------------------------------------------------------------

    def _resolve_forecast_columns(self, sample_candles: list[dict]) -> list[str]:
        """Определяет какие каналы прогнозируем: OHLC или OHLCV."""
        if self.include_volume_in_forecast and any("volume" in c for c in sample_candles):
            return OHLCV_CHANNELS
        return OHLC_CHANNELS

    def _truncate_to_max_history(
            self,
            candles_lengths: list[int],
            horizon: int,
    ) -> tuple[int, bool]:
        """
        Считает общий start_index_used для всех окон (длина одна).

        max_history = context_length модели. В Granite PatchTST FM context_length
        (8192, макс. длина входа) и prediction_length (64, макс. длина выхода) —
        независимые параметры. Делить бюджет между ними не нужно — это пережиток
        старых реализаций PatchTST, где они шарили embedding-бюджет.

        Возвращает (start_index_used, trimmed). Caller должен сам обрезать
        свои candles по start_index_used.

        horizon в сигнатуре оставлен для будущих моделей где он действительно
        отнимается от max_input (если такая модель появится — переопределить).
        """
        max_history = self._runtime.context_length
        first_len = candles_lengths[0]
        if first_len > max_history:
            return first_len - max_history, True
        return 0, False

    def _validate_horizon(self, horizon: int) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be > 0.")
        if horizon > self._runtime.prediction_length:
            raise ValueError(
                f"horizon={horizon} превышает prediction_length={self._runtime.prediction_length} "
                f"модели PatchTST FM. Используйте horizon ≤ {self._runtime.prediction_length}."
            )

    def _update_last_input_window_info(
            self,
            used_len: int,
            start_index_used: int,
            trimmed: bool,
            horizon: int,
            batch_size: int | None = None,
    ) -> None:
        """Заполняет диагностический dict о последнем инференсе."""
        info = {
            "original_length":   used_len + start_index_used,
            "used_length":       used_len,
            "trimmed":           trimmed,
            "start_index_used":  start_index_used,
            "requested_horizon": horizon,
            "prediction_length": self._runtime.prediction_length,
        }
        if batch_size is not None:
            info["batch_size"] = batch_size
        self._last_input_window_info = info

    def _run_pipeline(
            self,
            candles: list[dict[str, float]],
            horizon: int,
            context: dict | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Single-window инференс. Подходит для одиночного forecast-запроса.
        Для бэктеста с множеством окон используй _run_pipeline_batch — он
        пакует окна в один forward pass на GPU.
        """
        if not candles:
            raise ValueError("Input candles are empty.")
        if len(candles) < 16:
            raise ValueError(f"Для PatchTST FM требуется минимум 16 точек истории, получено {len(candles)}.")

        self._runtime.ensure_loaded()
        self._validate_horizon(horizon)

        freq = (context or {}).get("freq", DEFAULT_FREQ)
        verbose = (context or {}).get("detailed_logs", True)
        forecast_columns = self._resolve_forecast_columns(candles)

        start_index_used, trimmed = self._truncate_to_max_history([len(candles)], horizon)
        if trimmed:
            candles = candles[start_index_used:]

        self._update_last_input_window_info(
            used_len=len(candles),
            start_index_used=start_index_used,
            trimmed=trimmed,
            horizon=horizon,
        )

        prediction_length = self._runtime.prediction_length
        if verbose:
            logger.info(
                "[PatchTST FM] Инференс: horizon=%d (model_pred_len=%d) channels=%s history=%d trimmed=%s freq=%s",
                horizon, prediction_length, forecast_columns, len(candles), trimmed, freq,
            )

        context_df = self._candles_to_df(candles, forecast_columns, freq)
        future_df  = self._build_future_timestamps(context_df, prediction_length, freq)

        with _suppress_hf_logs(active=not verbose):
            forecast_df = self._runtime.pipeline(
                context_df,
                future_time_series=future_df,
                timestamp_column=TIMESTAMP_COL,
                target_columns=forecast_columns,
                id_columns=[],
                freq=freq,
            )

        if verbose:
            logger.info(
                "[PatchTST FM] Pipeline вернул DataFrame: shape=%s columns=%s",
                forecast_df.shape, list(forecast_df.columns),
            )

        return forecast_df, forecast_columns

    def _extract_channel_quantiles(
            self,
            forecast_df: pd.DataFrame,
            channel: str,
            horizon: int,
    ) -> QuantileForecast:
        """
        Извлекает квантили для одного канала из forecast_df.
        Pipeline возвращает колонки в формате: {channel}_q{quantile}
        Например: open_q0.1, close_q0.5
        """
        def _get(q: float) -> list[float]:
            col = QUANTILE_COL_TEMPLATE.format(channel=channel, q=q)
            if col not in forecast_df.columns:
                available = [c for c in forecast_df.columns if channel in c]
                raise ValueError(
                    f"Колонка '{col}' не найдена. Колонки с '{channel}': {available}. "
                    f"Все колонки: {list(forecast_df.columns)}"
                )
            return [float(v) for v in forecast_df[col].tolist()[:horizon]]

        q_map = dict(zip(["q10", "q25", "q50", "q75", "q90"], QUANTILE_LEVELS))
        return QuantileForecast(**{name: _get(q) for name, q in q_map.items()})

    def predict_line_exact(self, candles, horizon, context=None) -> list[float]:
        try:
            forecast_df, _ = self._run_pipeline(candles, horizon, context)
            return self._extract_channel_quantiles(forecast_df, self.target_column, horizon).q50
        except Exception as ex:
            raise RuntimeError(f"Ошибка predict_line_exact PatchTST FM: {ex}") from ex

    def predict_line_quantiles(self, candles, horizon, context=None) -> QuantileForecast:
        try:
            forecast_df, _ = self._run_pipeline(candles, horizon, context)
            return self._extract_channel_quantiles(forecast_df, self.target_column, horizon)
        except Exception as ex:
            raise RuntimeError(f"Ошибка predict_line_quantiles PatchTST FM: {ex}") from ex

    def predict_ohlc_exact(self, candles, horizon, context=None) -> list[dict[str, float]]:
        return self.predict_ohlc_quantiles(candles, horizon, context).median_candles()

    def predict_ohlc_quantiles(self, candles, horizon, context=None) -> OHLCQuantileForecast:
        try:
            forecast_df, forecast_columns = self._run_pipeline(candles, horizon, context)
            return self._build_ohlc_qf_from_df(forecast_df, forecast_columns, horizon)
        except Exception as ex:
            raise RuntimeError(f"Ошибка predict_ohlc_quantiles PatchTST FM: {ex!r}") from ex

    def _build_ohlc_qf_from_df(
            self,
            forecast_df: pd.DataFrame,
            forecast_columns: list[str],
            horizon: int,
    ) -> OHLCQuantileForecast:
        """Извлекает все квантильные каналы из forecast_df и собирает OHLCQuantileForecast."""
        open_qf  = self._extract_channel_quantiles(forecast_df, "open",  horizon)
        high_qf  = self._extract_channel_quantiles(forecast_df, "high",  horizon)
        low_qf   = self._extract_channel_quantiles(forecast_df, "low",   horizon)
        close_qf = self._extract_channel_quantiles(forecast_df, "close", horizon)

        open_qf, high_qf, low_qf, close_qf = PatchTSTDecoder.enforce_ohlc_consistency(
            open_qf, high_qf, low_qf, close_qf
        )

        volume_qf = None
        if self.include_volume_in_forecast and "volume" in forecast_columns:
            try:
                volume_qf = self._extract_channel_quantiles(forecast_df, "volume", horizon)
            except ValueError:
                pass

        return OHLCQuantileForecast(open=open_qf, high=high_qf, low=low_qf, close=close_qf, volume=volume_qf)

    # ------------------------------------------------------------------
    # Batched инференс: один pipeline-вызов для N окон одинаковой длины
    # ------------------------------------------------------------------

    def _run_pipeline_batch(
            self,
            candles_list: list[list[dict[str, float]]],
            horizon: int,
            context: dict | None = None,
    ) -> tuple[list[pd.DataFrame], list[str]]:
        """
        Прогоняет N окон одним вызовом TSFM pipeline через id_columns=[SERIES_ID_COL].

        Требования к caller:
          - все candles_list[i] имеют одинаковую длину (одну контекстную форму)
          - horizon должен быть ≤ prediction_length модели

        Возвращает список forecast_df (по одному на каждое окно входа) и
        forecast_columns (channels).
        """
        if not candles_list:
            return [], []

        N = len(candles_list)
        first_len = len(candles_list[0])
        if first_len < 16:
            raise ValueError(f"Для PatchTST FM требуется минимум 16 точек истории, получено {first_len}.")
        for i, c in enumerate(candles_list):
            if len(c) != first_len:
                raise ValueError(
                    f"Batched inference требует одинаковую длину всех окон. "
                    f"candles_list[0] имеет {first_len}, candles_list[{i}] имеет {len(c)}."
                )

        self._runtime.ensure_loaded()
        self._validate_horizon(horizon)

        freq = (context or {}).get("freq", DEFAULT_FREQ)
        verbose = (context or {}).get("detailed_logs", True)
        batch_size = max(1, int((context or {}).get("inference_batch_size", DEFAULT_INFERENCE_BATCH_SIZE)))
        forecast_columns = self._resolve_forecast_columns(candles_list[0])

        start_index_used, trimmed = self._truncate_to_max_history([first_len], horizon)
        if trimmed:
            candles_list = [c[start_index_used:] for c in candles_list]

        used_len = len(candles_list[0])
        prediction_length = self._runtime.prediction_length

        self._update_last_input_window_info(
            used_len=used_len,
            start_index_used=start_index_used,
            trimmed=trimmed,
            horizon=horizon,
            batch_size=N,
        )

        if verbose:
            logger.info(
                "[PatchTST FM] Batched инференс: N=%d batch_size=%d horizon=%d (pred_len=%d) "
                "channels=%s history=%d trimmed=%s freq=%s",
                N, batch_size, horizon, prediction_length, forecast_columns, used_len, trimmed, freq,
            )

        # Прогоняем чанками по batch_size — защита от OOM при больших N.
        results: list[pd.DataFrame] = []
        for chunk_start in range(0, N, batch_size):
            chunk = candles_list[chunk_start: chunk_start + batch_size]
            chunk_dfs = self._run_pipeline_chunk(
                chunk, forecast_columns, prediction_length, freq,
                start_series_id=chunk_start, verbose=verbose,
            )
            results.extend(chunk_dfs)

        return results, forecast_columns

    def _run_pipeline_chunk(
            self,
            candles_chunk: list[list[dict[str, float]]],
            forecast_columns: list[str],
            prediction_length: int,
            freq: str,
            start_series_id: int,
            verbose: bool,
    ) -> list[pd.DataFrame]:
        """
        Один вызов TSFM pipeline для чанка окон (≤ batch_size).
        Концатенирует все окна с series_id, делает 1 forward pass на GPU,
        возвращает список индивидуальных DataFrame'ов.
        """
        chunk_contexts = []
        chunk_futures = []
        for i, candles in enumerate(candles_chunk):
            sid = start_series_id + i
            ctx_df = self._candles_to_df(candles, forecast_columns, freq)
            fut_df = self._build_future_timestamps(ctx_df, prediction_length, freq)
            ctx_df = ctx_df.copy()
            fut_df = fut_df.copy()
            ctx_df[SERIES_ID_COL] = sid
            fut_df[SERIES_ID_COL] = sid
            chunk_contexts.append(ctx_df)
            chunk_futures.append(fut_df)

        big_context = pd.concat(chunk_contexts, ignore_index=True)
        big_future = pd.concat(chunk_futures, ignore_index=True)

        with _suppress_hf_logs(active=not verbose):
            # batch_size + num_workers — это говорит HF pipeline:
            # - сложить ВСЕ серии в один тензор и сделать один forward pass на GPU
            # - использовать N параллельных потоков для CPU-предобработки
            forecast_df = self._runtime.pipeline(
                big_context,
                future_time_series=big_future,
                timestamp_column=TIMESTAMP_COL,
                target_columns=forecast_columns,
                id_columns=[SERIES_ID_COL],
                freq=freq,
                batch_size=len(candles_chunk),
                num_workers=DEFAULT_NUM_WORKERS,
            )

        # Разделяем результат по series_id
        per_series: list[pd.DataFrame] = []
        for i in range(len(candles_chunk)):
            sid = start_series_id + i
            sub = forecast_df[forecast_df[SERIES_ID_COL] == sid].reset_index(drop=True)
            per_series.append(sub)
        return per_series

    def predict_ohlc_quantiles_batch(
            self,
            candles_list: list[list[dict[str, float]]],
            horizon: int,
            context: dict | None = None,
    ) -> list[OHLCQuantileForecast]:
        if not candles_list:
            return []
        try:
            forecast_dfs, forecast_columns = self._run_pipeline_batch(candles_list, horizon, context)
            return [
                self._build_ohlc_qf_from_df(df, forecast_columns, horizon)
                for df in forecast_dfs
            ]
        except Exception as ex:
            raise RuntimeError(f"Ошибка predict_ohlc_quantiles_batch PatchTST FM: {ex!r}") from ex

    def predict_line_exact_batch(
            self,
            candles_list: list[list[dict[str, float]]],
            horizon: int,
            context: dict | None = None,
    ) -> list[list[float]]:
        if not candles_list:
            return []
        try:
            forecast_dfs, _ = self._run_pipeline_batch(candles_list, horizon, context)
            return [
                self._extract_channel_quantiles(df, self.target_column, horizon).q50
                for df in forecast_dfs
            ]
        except Exception as ex:
            raise RuntimeError(f"Ошибка predict_line_exact_batch PatchTST FM: {ex!r}") from ex

    def get_info(self) -> dict:
        return {
            "name": self.model_name,
            "version": self.model_version,
            "model_id": self.model_id,
            "loaded": self._runtime.is_loaded,
            "device": self._runtime.device,
            "type": "multivariate-quantile-forecast-model",
            "target_column": self.target_column,
            "context_length": self._runtime.context_length,
            "prediction_length": self._runtime.prediction_length,
            "quantile_levels": self._runtime.quantile_levels,
            "supports_quantiles": True,
            "supports_ohlc": True,
            "include_volume_in_forecast": self.include_volume_in_forecast,
            "last_input_window_info": self._last_input_window_info,
            "load_error": self._runtime.load_error,
        }

    # ------------------------------------------------------------------
    # Adapter points (см. ForecastModel.get_adapter_modules / get_lora_target_modules)
    #
    # Структура PatchTSTFMForPrediction:
    #   PatchTSTFMForPrediction
    #     └─ backbone: PatchTSTFMModel
    #          ├─ in_layer:  ResidualBlock — patch embedding (input projection)
    #          ├─ pos_embed
    #          ├─ blocks:    ModuleList[TransformerBlock] (для LoRA attention)
    #          ├─ out_layer: ResidualBlock — projection → квантильные выходы (head)
    #          └─ norm_fn:   RevIN
    # ------------------------------------------------------------------

    def get_adapter_modules(self) -> dict:
        """Возвращает {"head": backbone.out_layer, "input": backbone.in_layer}."""
        self._runtime.ensure_loaded()
        backbone = self._runtime.model.backbone
        return {
            "head":  backbone.out_layer,
            "input": backbone.in_layer,
        }

    def get_lora_target_modules(self) -> list[str]:
        """
        Стандартные attention-проекции внутри TransformerBlock'ов PatchTSTFM.
        PEFT находит их по имени во всех слоях.
        """
        return ["q_proj", "k_proj", "v_proj", "out_proj"]