class ForecastMetadataBuilder:
    @staticmethod
    def build(
            request,
            dates: list[str],
            close_series: list[float],
            window_info: dict,
            start_date_used: str | None
    ) -> dict:
        start_index_used = window_info.get("start_index_used") if isinstance(window_info, dict) else None
        return {
            "history_length": len(close_series),
            "requested_horizon": request.horizon,
            "history_period": request.history_period,
            "history_up_to": request.history_up_to,
            "requested_chart_type_history": request.chart_type_history,
            "requested_chart_type_forecast": request.chart_type_forecast,
            "model_options": request.model_options,
            "feature_plugins": request.feature_plugins,
            "forecast_representation": "ohlc_candles",
            "model_required_context_length": window_info.get("required_context_length") if isinstance(window_info, dict) else None,
            "model_input_history_length_original": window_info.get("original_length") if isinstance(window_info, dict) else None,
            "model_input_history_length_used": window_info.get("used_length") if isinstance(window_info, dict) else None,
            "model_input_trimmed": window_info.get("trimmed") if isinstance(window_info, dict) else None,
            "model_input_start_index_used": start_index_used
        }