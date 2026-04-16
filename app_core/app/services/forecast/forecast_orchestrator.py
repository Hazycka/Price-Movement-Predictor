from ..chart import calculate_indicators
from ...schemas import ForecastResponse


class ForecastOrchestrator:
    def __init__(self, model, context_builder, metadata_builder) -> None:
        self.model = model
        self.context_builder = context_builder
        self.metadata_builder = metadata_builder

    @staticmethod
    def can_render_candles(candles: list[dict[str, float]]) -> bool:
        required_keys = {"open", "high", "low", "close"}
        if not candles:
            return False
        return all(required_keys.issubset(candle.keys()) for candle in candles)

    def run(self, request, source: str, dates: list[str], candles: list[dict[str, float]]) -> ForecastResponse:
        close_series = [candle["close"] for candle in candles]
        indicators = calculate_indicators(close_series, request.indicators)

        if request.chart_type_history == "candlestick" and not self.can_render_candles(candles):
            raise ValueError("Запрошен history candlestick, но входные данные не содержат полный OHLC.")

        context = self.context_builder.build(request)

        if request.chart_type_forecast == "candlestick":
            forecast_candles = self.model.predict_ohlc_multivariate(
                candles=candles,
                horizon=request.horizon,
                context=context
            )
        elif request.chart_type_forecast == "line":
            forecast_close = self.model.predict_multivariate(
                candles=candles,
                horizon=request.horizon,
                context=context
            )
            forecast_candles = [{"open": v, "high": v, "low": v, "close": v} for v in forecast_close]
        else:
            raise ValueError(f"Неподдерживаемый chart_type_forecast='{request.chart_type_forecast}'.")

        model_info = self.model.get_info()
        window_info = model_info.get("last_input_window_info", {}) if isinstance(model_info, dict) else {}
        start_index_used = window_info.get("start_index_used")
        start_date_used = None
        if isinstance(start_index_used, int) and dates and 0 <= start_index_used < len(dates):
            start_date_used = dates[start_index_used]

        horizon_mismatch_count = len(forecast_candles) - request.horizon

        metadata = self.metadata_builder.build(
            request=request,
            dates=dates,
            close_series=close_series,
            window_info=window_info,
            start_date_used=start_date_used
        )

        return ForecastResponse(
            source=source,
            model=model_info,
            chart_type_history=request.chart_type_history,
            chart_type_forecast=request.chart_type_forecast,
            candles=candles,
            forecast_candles=forecast_candles,
            indicators=indicators,
            dates=dates,
            interval=request.interval,
            model_input_start_date_used=start_date_used,
            metadata=metadata,
            horizon_mismatch_count=horizon_mismatch_count
        )