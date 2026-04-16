class ModelWindowUsageExtractor:
    def __init__(self, model) -> None:
        self.model = model

    def extract(self, dates: list[str], train_end: int) -> dict:
        model_info = self.model.get_info() if hasattr(self.model, "get_info") else {}
        window_info = model_info.get("last_input_window_info", {}) if isinstance(model_info, dict) else {}

        start_index_used = window_info.get("start_index_used")
        train_dates = dates[:train_end] if dates else []
        start_date_used = None
        if isinstance(start_index_used, int) and train_dates and 0 <= start_index_used < len(train_dates):
            start_date_used = train_dates[start_index_used]

        return {
            "required_context_length": window_info.get("required_context_length"),
            "original_length": window_info.get("original_length"),
            "used_length": window_info.get("used_length"),
            "trimmed": window_info.get("trimmed"),
            "start_index_used": start_index_used,
            "start_date_used": start_date_used
        }