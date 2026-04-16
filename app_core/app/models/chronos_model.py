import os
import numpy as np
from .base import ForecastModel


class ChronosForecastModel(ForecastModel):
    def __init__(
            self,
            model_id: str = "amazon/chronos-t5-large",
            model_version: str = "chronos-t5-large",
            num_samples: int = 64
    ) -> None:
        self.model_name = "Chronos"
        self.model_version = model_version
        self.model_id = os.getenv("CHRONOS_MODEL_ID", model_id)
        self.num_samples = int(os.getenv("CHRONOS_NUM_SAMPLES", str(num_samples)))

        self.is_loaded = False
        self.device = "cpu"
        self.load_error: str | None = None
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self.is_loaded and self._pipeline is not None:
            return

        try:
            import torch
            from chronos import ChronosPipeline

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype_ = torch.float16 if self.device == "cuda" else torch.float32

            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
                dtype=dtype_
            )

            self.is_loaded = True
            self.load_error = None
        except Exception as ex:
            self.is_loaded = False
            self._pipeline = None
            self.load_error = str(ex)
            raise RuntimeError(f"Не удалось загрузить Chronos модель '{self.model_id}': {ex}") from ex

    def predict(self, series: list[float], horizon: int, context: dict | None = None) -> list[float]:
        if not series:
            raise ValueError("Input series is empty.")
        if horizon <= 0:
            raise ValueError("horizon must be > 0.")

        self._ensure_loaded()

        try:
            import torch

            prediction_samples = self.num_samples
            if context and isinstance(context.get("num_samples"), int):
                prediction_samples = max(1, int(context["num_samples"]))

            ts = torch.tensor(series, dtype=torch.float32)

            forecast = self._pipeline.predict(
                inputs=ts,
                prediction_length=horizon,
                num_samples=prediction_samples
            )

            arr = forecast.detach().cpu().numpy() if hasattr(forecast, "detach") else np.asarray(forecast)
            arr = np.squeeze(arr)

            while arr.ndim > 2:
                arr = arr[0]

            if arr.ndim == 1:
                point_forecast = arr
            elif arr.ndim == 2:
                point_forecast = np.median(arr, axis=0)
            else:
                raise RuntimeError(f"Неожиданная форма прогноза Chronos: {arr.shape}")

            result = [float(x) for x in point_forecast.tolist()]
            if len(result) != horizon:
                result = result[:horizon]
                if len(result) < horizon and result:
                    result.extend([result[-1]] * (horizon - len(result)))

            return result
        except Exception as ex:
            raise RuntimeError(f"Ошибка инференса Chronos: {ex}") from ex

    def predict_multivariate(
            self,
            candles: list[dict[str, float]],
            horizon: int,
            context: dict | None = None
    ) -> list[float]:
        close_series = [float(candle["close"]) for candle in candles]
        return self.predict(series=close_series, horizon=horizon, context=context)

    def get_info(self) -> dict:
        return {
            "name": self.model_name,
            "version": self.model_version,
            "model_id": self.model_id,
            "loaded": self.is_loaded,
            "device": self.device,
            "num_samples": self.num_samples,
            "type": "close-only-forecast-model",
            "load_error": self.load_error
        }