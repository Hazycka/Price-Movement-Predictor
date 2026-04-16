from .plotly_renderer import build_forecast_chart_html


class ChartService:
    @staticmethod
    def build(result) -> str:
        title = f"{result.source} | model={result.model['name']}"
        return build_forecast_chart_html(
            title=title,
            labels=result.dates,
            candles=result.candles,
            forecast_candles=result.forecast_candles,
            indicators=result.indicators,
            chart_type_history=result.chart_type_history,
            chart_type_forecast=result.chart_type_forecast,
            interval=result.interval,
            model_input_start_date=result.model_input_start_date_used
        )