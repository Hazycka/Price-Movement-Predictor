import os
import numpy as np

from ..base import ForecastModel
from ..features import FeaturePipeline
from .patchtst_runtime import PatchTSTRuntime
from .patchtst_preprocessor import PatchTSTPreprocessor
from .patchtst_decoder import PatchTSTDecoder
from .patchtst_guards import PatchTSTGuards


class PatchTSTForecastModel(ForecastModel):
    def __init__(
            self,
            model_id: str = "ibm-research/patchtst-fm-r1",
            model_version: str = "patchtst-fm-r1",
            target_column: str = "close"
    ) -> None:
        self.model_name = "PatchTST"
        self.model_version = model_version
        self.model_id = os.getenv("PATCHTST_MODEL_ID", model_id)
        self.target_column = os.getenv("PATCHTST_TARGET_COLUMN", target_column)

        self._runtime = PatchTSTRuntime(model_id=self.model_id)
        self._feature_pipeline = FeaturePipeline()
        self._last_plugins_used: list[str] = []
        self._last_input_window_info: dict = {}

    def predict(self, series: list[float], horizon: int, context: dict | None = None) -> list[float]:
        candles = [{"open": v, "high": v, "low": v, "close": v, "volume": 0.0} for v in series]
        return self.predict_multivariate(candles=candles, horizon=horizon, context=context)

    def predict_multivariate(
            self,
            candles: list[dict[str, float]],
            horizon: int,
            context: dict | None = None
    ) -> list[float]:
        if not candles:
            raise ValueError("Input candles are empty.")
        if horizon <= 0:
            raise ValueError("horizon must be > 0.")
        if len(candles) < 32:
            raise ValueError("Для PatchTST желательно минимум 32+ точки истории.")

        self._runtime.ensure_loaded()
        torch = self._runtime.torch

        try:
            pipeline_result = self._feature_pipeline.build(candles=candles, context=context)
            df = pipeline_result.df
            feature_columns = pipeline_result.feature_columns
            self._last_plugins_used = pipeline_result.plugins_used

            model_input = PatchTSTPreprocessor.to_model_input(df, feature_columns)
            model_input, win_info = PatchTSTPreprocessor.apply_context_window(
                model_input, self._runtime.required_context_length
            )
            self._last_input_window_info = win_info

            x = torch.tensor(model_input, dtype=torch.float32, device=self._runtime.device).unsqueeze(0)
            with torch.no_grad():
                outputs = self._runtime.model(past_values=x)

            arr = PatchTSTDecoder.extract_prediction_array(outputs)
            result = PatchTSTDecoder.decode_close(arr, feature_columns, self.target_column)
            result = PatchTSTGuards.trim_to_horizon(result, horizon)

            last_close = float(candles[-1]["close"])
            if PatchTSTGuards.has_non_finite(result):
                result = [last_close] * len(result)

            return result
        except Exception as ex:
            raise RuntimeError(f"Ошибка инференса PatchTST: {ex}") from ex

    def predict_ohlc_multivariate(
            self,
            candles: list[dict[str, float]],
            horizon: int,
            context: dict | None = None
    ) -> list[dict[str, float]]:
        if not candles:
            raise ValueError("Input candles are empty.")
        if horizon <= 0:
            raise ValueError("horizon must be > 0.")
        if len(candles) < 32:
            raise ValueError("Для PatchTST желательно минимум 32+ точки истории.")

        self._runtime.ensure_loaded()
        torch = self._runtime.torch

        try:
            pipeline_result = self._feature_pipeline.build(candles=candles, context=context)
            df = pipeline_result.df
            feature_columns = pipeline_result.feature_columns
            self._last_plugins_used = pipeline_result.plugins_used

            for required in ("open", "high", "low", "close"):
                if required not in feature_columns:
                    raise ValueError(f"Для OHLC-прогноза отсутствует канал '{required}' во feature pipeline.")

            model_input = PatchTSTPreprocessor.to_model_input(df, feature_columns)
            model_input, win_info = PatchTSTPreprocessor.apply_context_window(
                model_input, self._runtime.required_context_length
            )
            self._last_input_window_info = win_info

            x = torch.tensor(model_input, dtype=torch.float32, device=self._runtime.device).unsqueeze(0)
            with torch.no_grad():
                outputs = self._runtime.model(past_values=x)

            arr = PatchTSTDecoder.extract_prediction_array(outputs)
            channels = PatchTSTDecoder.decode_ohlc(arr, feature_columns)

            open_f = PatchTSTGuards.trim_to_horizon(channels["open"], horizon)
            high_f = PatchTSTGuards.trim_to_horizon(channels["high"], horizon)
            low_f = PatchTSTGuards.trim_to_horizon(channels["low"], horizon)
            close_f = PatchTSTGuards.trim_to_horizon(channels["close"], horizon)

            result: list[dict[str, float]] = []
            for o, h, l, c in zip(open_f, high_f, low_f, close_f):
                result.append({
                    "open": float(o),
                    "high": float(max(o, h, l, c)),
                    "low": float(min(o, h, l, c)),
                    "close": float(c)
                })

            if PatchTSTGuards.has_non_finite_ohlc(result):
                raise ValueError("OHLC-прогноз содержит NaN/Inf.")

            return result
        except Exception as ex:
            raise RuntimeError(f"Ошибка OHLC-инференса PatchTST: {ex}") from ex

    def get_info(self) -> dict:
        return {
            "name": self.model_name,
            "version": self.model_version,
            "model_id": self.model_id,
            "loaded": self._runtime.is_loaded,
            "device": self._runtime.device,
            "type": "multivariate-forecast-model",
            "target_column": self.target_column,
            "feature_plugins_used": self._last_plugins_used,
            "supports_ohlc_forecast": True,
            "required_context_length": self._runtime.required_context_length,
            "last_input_window_info": self._last_input_window_info,
            "load_error": self._runtime.load_error
        }
