from .plotly_renderer import build_forecast_chart_html


class ChartService:
    @staticmethod
    def build(result) -> str:
        title = f"{result.source} | model={result.model['name']}"

        # Конвертируем pydantic schema в plain dict для рендерера
        forecast_ohlc_quantiles = None
        if result.forecast_ohlc_quantiles is not None:
            q = result.forecast_ohlc_quantiles
            forecast_ohlc_quantiles = {
                ch: {
                    "q10": getattr(getattr(q, ch), "q10"),
                    "q25": getattr(getattr(q, ch), "q25"),
                    "q50": getattr(getattr(q, ch), "q50"),
                    "q75": getattr(getattr(q, ch), "q75"),
                    "q90": getattr(getattr(q, ch), "q90"),
                }
                for ch in ("open", "high", "low", "close")
            }
            if q.volume is not None:
                forecast_ohlc_quantiles["volume"] = {
                    "q10": q.volume.q10,
                    "q25": q.volume.q25,
                    "q50": q.volume.q50,
                    "q75": q.volume.q75,
                    "q90": q.volume.q90,
                }

        return build_forecast_chart_html(
            title=title,
            labels=result.dates,
            candles=result.candles,
            forecast_candles=result.forecast_candles,
            forecast_ohlc_quantiles=forecast_ohlc_quantiles,
            indicators=result.indicators,
            chart_type_history=result.chart_type_history,
            chart_type_forecast=result.chart_type_forecast,
            interval=result.interval,
            model_input_start_date=result.model_input_start_date_used,
            max_history_candles=300
        )