"""
PatchTST FM Foundation Model для прогнозирования временных рядов.
"""
import logging
import os

import pandas as pd

from ..base import ForecastModel, QuantileForecast, OHLCQuantileForecast
from .patchtst_runtime import PatchTSTRuntime, QUANTILE_LEVELS
from .patchtst_decoder import PatchTSTDecoder

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "ibm-granite/granite-timeseries-patchtst-fm-r1"
OHLC_CHANNELS  = ["open", "high", "low", "close"]
OHLCV_CHANNELS = ["open", "high", "low", "close", "volume"]
TIMESTAMP_COL  = "timestamp"
TIMESTAMP_FREQ = "h"


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

    def _candles_to_df(self, candles: list[dict[str, float]], forecast_columns: list[str]) -> pd.DataFrame:
        df = pd.DataFrame(candles)
        for col in forecast_columns:
            if col not in df.columns:
                df[col] = 0.0
        df[TIMESTAMP_COL] = pd.date_range(start="2000-01-01", periods=len(df), freq=TIMESTAMP_FREQ)
        return df[[TIMESTAMP_COL] + forecast_columns]

    def _build_future_timestamps(self, context_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Строит DataFrame с будущими временными метками.
        Pipeline требует future_time_series как DataFrame с timestamp колонкой — не int.
        """
        last_ts = context_df[TIMESTAMP_COL].iloc[-1]
        future_timestamps = pd.date_range(start=last_ts, periods=horizon + 1, freq=TIMESTAMP_FREQ)[1:]
        return pd.DataFrame({TIMESTAMP_COL: future_timestamps})

    def _run_pipeline(self, candles: list[dict[str, float]], horizon: int, context: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
        if not candles:
            raise ValueError("Input candles are empty.")
        if horizon <= 0:
            raise ValueError("horizon must be > 0.")
        if len(candles) < 16:
            raise ValueError(f"Для PatchTST FM требуется минимум 16 точек истории, получено {len(candles)}.")

        self._runtime.ensure_loaded()

        forecast_columns = (
            OHLCV_CHANNELS if self.include_volume_in_forecast and any("volume" in c for c in candles)
            else OHLC_CHANNELS
        )

        max_history = self._runtime.context_length - horizon
        trimmed = False
        start_index_used = 0

        if len(candles) > max_history:
            start_index_used = len(candles) - max_history
            candles = candles[start_index_used:]
            trimmed = True

        self._last_input_window_info = {
            "context_length": self._runtime.context_length,
            "original_length": len(candles) + start_index_used,
            "used_length": len(candles),
            "trimmed": trimmed,
            "start_index_used": start_index_used,
            "horizon": horizon,
        }

        logger.info("[PatchTST FM] Инференс: horizon=%d channels=%s history=%d trimmed=%s",
                    horizon, forecast_columns, len(candles), trimmed)

        context_df = self._candles_to_df(candles, forecast_columns)
        future_df  = self._build_future_timestamps(context_df, horizon)

        forecast_df = self._runtime.pipeline(
            context_df,
            future_time_series=future_df,
            timestamp_column=TIMESTAMP_COL,
            target_columns=forecast_columns,
        )

        logger.info("[PatchTST FM] Pipeline вернул DataFrame: shape=%s columns=%s",
                    forecast_df.shape, list(forecast_df.columns))

        return forecast_df, forecast_columns

    def _extract_channel_quantiles(self, forecast_df: pd.DataFrame, channel: str, horizon: int) -> QuantileForecast:
        def _get(q: float) -> list[float]:
            col = f"{channel}_{q}"
            if col not in forecast_df.columns:
                available = [c for c in forecast_df.columns if channel in c]
                raise ValueError(
                    f"Колонка '{col}' не найдена. Колонки с '{channel}': {available}. "
                    f"Все колонки: {list(forecast_df.columns)}"
                )
            return [float(v) for v in forecast_df[col].tolist()[:horizon]]

        return QuantileForecast(q10=_get(0.1), q25=_get(0.25), q50=_get(0.5), q75=_get(0.75), q90=_get(0.9))

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
        except Exception as ex:
            raise RuntimeError(f"Ошибка predict_ohlc_quantiles PatchTST FM: {ex}") from ex

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
            "quantile_levels": self._runtime.quantile_levels,
            "supports_quantiles": True,
            "supports_ohlc": True,
            "include_volume_in_forecast": self.include_volume_in_forecast,
            "last_input_window_info": self._last_input_window_info,
            "load_error": self._runtime.load_error,
        }