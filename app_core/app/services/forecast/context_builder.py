class ForecastContextBuilder:
    @staticmethod
    def build(request) -> dict:
        return {
            "indicators_enabled": request.indicators,
            "chart_type_history": request.chart_type_history,
            "chart_type_forecast": request.chart_type_forecast,
            "num_samples": request.num_samples,
            "feature_plugins": request.feature_plugins,
            "future_extensions": {
                "feature_plugins": True,
                "custom_training_hooks": True,
                "close_only_forecast": False
            }
        }