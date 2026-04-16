from abc import ABC, abstractmethod


class ForecastModel(ABC):
    @abstractmethod
    def predict(self, series: list[float], horizon: int, context: dict | None = None) -> list[float]:
        pass

    def predict_multivariate(
            self,
            candles: list[dict[str, float]],
            horizon: int,
            context: dict | None = None
    ) -> list[float]:
        """
        Унифицированный multivariate-интерфейс.
        По умолчанию откатываемся к close-only, чтобы старые модели работали без изменений.
        """
        if not candles:
            raise ValueError("Input candles are empty.")
        close_series = [float(candle["close"]) for candle in candles]
        return self.predict(series=close_series, horizon=horizon, context=context)

    def predict_ohlc_multivariate(
            self,
            candles: list[dict[str, float]],
            horizon: int,
            context: dict | None = None
    ) -> list[dict[str, float]]:
        raise ValueError(f"Модель '{self.__class__.__name__}' не поддерживает OHLC-прогноз.")

    @abstractmethod
    def get_info(self) -> dict:
        pass

    def fit_adapter(self, data, config: dict | None = None) -> None:
        return None
