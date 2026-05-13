class ForecastMetadataBuilder:
    """
    Собирает metadata-блок ForecastResponse.

    Не влияет на вычисления — только для диагностики и аудита запроса.
    Объединяет параметры из ForecastRequest и post-inference информацию
    модели (last_input_window_info из model.get_info()).

    Поля результирующего dict:
      history_length                    — количество исторических свечей в ответе.

      requested_horizon                 — горизонт прогноза из запроса (в барах).

      history_period                    — строка периода истории из запроса (например "1y", "6mo").

      history_up_to                     — дата конца исторического окна из запроса.
                                          None означает «до текущего момента».

      requested_chart_type_history      — тип графика истории из запроса ("line" | "candlestick").

      requested_chart_type_forecast     — тип графика прогноза из запроса ("line" | "candlestick").

      model_options                     — опции модели переданные в запросе (сериализованный объект
                                          ChronosModelOptions / PatchTSTModelOptions или None).

      feature_plugins                   — список feature-плагинов из запроса.

      forecast_representation           — тип внутреннего представления прогноза. Всегда
                                          "ohlc_candles" — медианные свечи q50 по каждому каналу.

      model_required_context_length     — минимальная длина контекста, которую модель объявила
                                          обязательной. None для Foundation Models (PatchTST FM,
                                          Chronos) — они принимают любой объём истории от 16 до
                                          max_context_length баров и не имеют жёсткого требования.
                                          Будет заполнено моделями с фиксированным контекстом
                                          через поле "required_context_length" в last_input_window_info.

      model_input_history_length_original — длина истории до обрезки под контекстное окно модели
                                            (= len(candles) до trimming).

      model_input_history_length_used   — длина истории, реально скормленной модели
                                          (после обрезки, если она была).

      model_input_trimmed               — True если история была обрезана из-за ограничения
                                          context_length модели. False или None иначе.

      model_input_start_index_used      — индекс первого бара из candles[], скормленного модели.
                                          0 если trimming не было. Совпадает с start_index_used
                                          в last_input_window_info.
    """

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