from ..market_data.common import INTERVAL_TO_FREQ, DEFAULT_FREQ


class ForecastContextBuilder:
    @staticmethod
    def build(request) -> dict:
        return {
            "indicators_enabled": request.indicators,
            "chart_type_history": request.chart_type_history,
            "chart_type_forecast": request.chart_type_forecast,
            "model_options": request.model_options,
            "feature_plugins": request.feature_plugins,
            "freq": INTERVAL_TO_FREQ.get(request.interval, DEFAULT_FREQ),
            "future_extensions": {
                "feature_plugins": True,
                "custom_training_hooks": True,
                "close_only_forecast": False
            }
        }