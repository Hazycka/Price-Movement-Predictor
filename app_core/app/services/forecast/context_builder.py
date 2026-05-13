from ..market_data.common import INTERVAL_TO_FREQ, DEFAULT_FREQ


class ForecastContextBuilder:
    """
    Собирает контекст вызова модели из ForecastRequest.

    Контекст — это dict, который передаётся в model.predict_*() и используется
    внутри модели для настройки инференса. Не является частью API-ответа.

    Поля контекста:
      indicators_enabled      — список запрошенных индикаторов (например ["rsi", "ema_20"]).
                                Передаётся для справки; индикаторы рассчитываются отдельно
                                в ForecastOrchestrator.

      chart_type_history      — тип графика истории из запроса ("line" | "candlestick").
                                Модели могут учитывать, нужно ли готовить OHLC или только close.

      chart_type_forecast     — тип графика прогноза из запроса ("line" | "candlestick").
                                Управляет выбором метода модели: "candlestick" → predict_ohlc_quantiles,
                                "line" → predict_line_exact.

      model_options           — специфичные опции модели (ChronosModelOptions, PatchTSTModelOptions
                                или None). Модель извлекает нужные ей параметры через
                                context.get("model_options").

      feature_plugins         — список feature-плагинов для мультивариативных моделей.
                                Используется моделями, которые принимают дополнительные фичи
                                (например технические индикаторы как входные каналы).

      freq                    — pandas-совместимая строка частоты (например "B" для рабочих дней,
                                "h" для часовых данных). Вычисляется из interval запроса.
                                Нужна для построения временных меток в pipeline.

      future_extensions       — зарезервированные флаги для будущей функциональности.
                                Не используются в production; позволяют моделям проверять
                                наличие поддерживаемых расширений без ломки интерфейса.
    """

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