from ..chart import calculate_indicators
from ...schemas import ForecastResponse, OHLCQuantileForecastSchema, QuantileForecastSchema


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

        # ------------------------------------------------------------------
        # Выбор метода прогноза по запрошенному chart_type_forecast.
        # ------------------------------------------------------------------

        forecast_ohlc_quantiles: OHLCQuantileForecastSchema | None = None

        if request.chart_type_forecast == "candlestick":
            # Основной и предпочтительный путь — OHLC с квантилями.
            # Модели без поддержки поднимают NotImplementedError → HTTP 400.
            try:
                qf = self.model.predict_ohlc_quantiles(
                    candles=candles,
                    horizon=request.horizon,
                    context=context
                )
            except NotImplementedError as ex:
                raise ValueError(
                    f"Модель '{self.model.get_info().get('name')}' не поддерживает "
                    f"режим candlestick прогноза с квантилями. "
                    f"Используйте chart_type_forecast='line' или выберите другую модель. "
                    f"Детали: {ex}"
                ) from ex

            # Конвертируем dataclass → pydantic schema для ответа
            forecast_ohlc_quantiles = OHLCQuantileForecastSchema(
                open=QuantileForecastSchema(**qf.open.to_dict()),
                high=QuantileForecastSchema(**qf.high.to_dict()),
                low=QuantileForecastSchema(**qf.low.to_dict()),
                close=QuantileForecastSchema(**qf.close.to_dict()),
                volume=QuantileForecastSchema(**qf.volume.to_dict()) if qf.volume is not None else None,
            )

            # Медианные свечи как основной прогноз для отображения
            forecast_candles = qf.median_candles()

        elif request.chart_type_forecast == "line":
            # Line прогноз — только точечный close, квантили не нужны.
            try:
                close_forecast = self.model.predict_line_exact(
                    candles=candles,
                    horizon=request.horizon,
                    context=context
                )
            except NotImplementedError as ex:
                raise ValueError(
                    f"Модель '{self.model.get_info().get('name')}' не поддерживает "
                    f"режим line прогноза. "
                    f"Детали: {ex}"
                ) from ex

            forecast_candles = [
                {"open": v, "high": v, "low": v, "close": v}
                for v in close_forecast
            ]

        else:
            raise ValueError(
                f"Неподдерживаемый chart_type_forecast='{request.chart_type_forecast}'. "
                f"Допустимые значения: 'candlestick', 'line'."
            )

        # ------------------------------------------------------------------
        # Метаданные окна модели
        # ------------------------------------------------------------------

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

        # ------------------------------------------------------------------
        # Visual-trim истории для ответа.
        #
        # max_chart_history_candles управляет ТОЛЬКО тем, сколько исторических
        # свечей возвращается клиенту (в JSON-ответе и в HTML-графике).
        # Модель к этому моменту уже отработала на ПОЛНОЙ истории — обрезка
        # здесь на инференс никак не влияет. В metadata остаётся исходная
        # длина (history_length, model_input_history_length_used) — это для
        # аудита того, что реально получила модель.
        #
        # Замечание про индексы: model_input_start_index_used считался от
        # полной истории; после обрезки он становится неточным. Корректным
        # ориентиром после trim'а остаётся model_input_start_date_used (дата).
        # ------------------------------------------------------------------
        max_n = getattr(request, "max_chart_history_candles", None)
        if max_n is not None and len(candles) > max_n:
            candles_out = candles[-max_n:]
            dates_out = dates[-max_n:] if dates else dates
            indicators_out = {
                name: (values[-max_n:] if values else values)
                for name, values in indicators.items()
            }
        else:
            candles_out = candles
            dates_out = dates
            indicators_out = indicators

        return ForecastResponse(
            source=source,
            model=model_info,
            chart_type_history=request.chart_type_history,
            chart_type_forecast=request.chart_type_forecast,
            candles=candles_out,
            forecast_candles=forecast_candles,
            forecast_ohlc_quantiles=forecast_ohlc_quantiles,
            indicators=indicators_out,
            dates=dates_out,
            interval=request.interval,
            model_input_start_date_used=start_date_used,
            metadata=metadata,
            horizon_mismatch_count=horizon_mismatch_count
        )