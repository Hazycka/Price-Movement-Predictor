from pydantic import BaseModel, Field, model_validator
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Provider options
# ---------------------------------------------------------------------------

class TInvestProviderOptions(BaseModel):
    figi: str | None = Field(default=None, description="FIGI инструмента")
    ticker: str | None = Field(default=None, description="Например: AAPL")
    class_code: str | None = Field(default=None, description="Код класса (например TQBR)")
    token: str | None = Field(default=None, description="Опционально: токен (предпочтительно env TINVEST_TOKEN)")


class YahooProviderOptions(BaseModel):
    ticker: str = Field(default=None, description="Например: AAPL")
    auto_adjust: bool = Field(default=False, description="Автокорректировка цен в yfinance")


class CsvProviderOptions(BaseModel):
    csv_path: str = Field(default=None, description="Путь к CSV в папке data или абсолютный путь")
    date_column: str = Field(default="Date", description="Название колонки с датой/временем")
    open_column: str = Field(default="Open", description="Название колонки с ценой открытия")
    high_column: str = Field(default="High", description="Название колонки с максимальной ценой")
    low_column: str = Field(default="Low", description="Название колонки с минимальной ценой")
    close_column: str = Field(default="Close", description="Название колонки с ценой закрытия")
    volume_column: str | None = Field(default="Volume", description="Название колонки с объёмом. None чтобы не загружать объём")


# ---------------------------------------------------------------------------
# Model options
# ---------------------------------------------------------------------------

class PatchTSTModelOptions(BaseModel):
    pass

class ChronosModelOptions(BaseModel):
    num_samples: int = Field(
        default=64, ge=1, le=256,
        description="Количество сэмплов в вероятностном прогнозе"
    )

ProviderOptions = TInvestProviderOptions | YahooProviderOptions | CsvProviderOptions
ModelOptions = PatchTSTModelOptions | ChronosModelOptions


# ---------------------------------------------------------------------------
# Quantile schemas
#
# QuantileForecastSchema  — квантили одного канала (например close)
# OHLCQuantileForecastSchema — квантили по всем OHLC каналам + опциональный volume
# ---------------------------------------------------------------------------

class QuantileForecastSchema(BaseModel):
    """
    Квантильный прогноз для одного канала на горизонт N точек.
    Каждое поле — список длиной horizon.

    [q10, q90] — 80% доверительный интервал (широкий band на графике)
    [q25, q75] — 50% доверительный интервал (узкий band на графике)
    q50        — медиана, центральный прогноз
    """
    q10: list[float]
    q25: list[float]
    q50: list[float]
    q75: list[float]
    q90: list[float]


class OHLCQuantileForecastSchema(BaseModel):
    """
    Квантильный прогноз полной OHLC свечи на горизонт N точек.

    volume: None в zero-shot режиме.
    Заполняется после дообучения модели на конкретном инструменте.
    """
    open:   QuantileForecastSchema
    high:   QuantileForecastSchema
    low:    QuantileForecastSchema
    close:  QuantileForecastSchema
    volume: QuantileForecastSchema | None = Field(
        default=None,
        description="None в zero-shot. Заполняется после дообучения на объёме."
    )


# ---------------------------------------------------------------------------
# Forecast request / response
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    model_name: Literal["chronos", "patchtst"] | None = Field(
        default=None,
        description=(
            "Выбор модели. None — использует модель по умолчанию из конфигурации сервиса. "
            "'patchtst' — IBM Granite PatchTST FM (мультивариативный, OHLC + квантили). "
            "'chronos' — Amazon Chronos (univariate, только close)."
        )
    )
    model_options: ModelOptions | None = Field(
        default=None,
        description=(
            "Специфичные опции модели. Тип должен соответствовать model_name: "
            "PatchTSTModelOptions для 'patchtst', ChronosModelOptions для 'chronos'. "
            "None — использовать настройки по умолчанию."
        )
    )

    data_source: Literal["yfinance", "t_invest", "csv"] = Field(
        default="t_invest",
        description=(
            "Источник рыночных данных. "
            "'t_invest' — Т-Инвестиции API (provider_options: TInvestProviderOptions). "
            "'yfinance' — Yahoo Finance (provider_options: YahooProviderOptions). "
            "'csv' — локальный CSV-файл (provider_options: CsvProviderOptions)."
        )
    )
    provider_options: ProviderOptions | None = Field(
        default=None,
        description=(
            "Параметры источника данных. Тип должен соответствовать data_source. "
            "None — для t_invest используются переменные окружения (TINVEST_TOKEN, TINVEST_FIGI)."
        )
    )

    chart_type_history: Literal["line", "candlestick"] = Field(
        default="candlestick",
        description=(
            "'candlestick' — история возвращается как OHLC свечи. "
            "'line' — только close (candles будут содержать только поле close)."
        )
    )
    chart_type_forecast: Literal["line", "candlestick"] = Field(
        default="candlestick",
        description=(
            "'candlestick' — прогноз строится как OHLC свечи с квантилями; "
            "forecast_ohlc_quantiles будет заполнен. "
            "'line' — точечный прогноз только по close; forecast_ohlc_quantiles = None."
        )
    )
    horizon: int = Field(
        default=5, ge=1, le=64,
        description=(
            "Горизонт прогноза в барах. Например, при interval='1d' и horizon=5 — прогноз на 5 торговых дней. "
            "Верхняя граница (64) равна prediction_length модели PatchTST FM по умолчанию. "
            "Точное значение для конкретной загруженной модели смотри в response.model.prediction_length."
        )
    )
    history_period: str = Field(
        default="1y",
        description="Период загружаемой истории. Форматы: '6mo', '1y', '2y', '5y'. Чем длиннее — тем точнее индикаторы."
    )
    history_up_to: str | None = Field(
        default=None,
        description="Дата конца исторического окна в формате YYYY-MM-DD. None — загружать до текущего момента."
    )
    interval: str = Field(
        default="1d",
        description="Временной интервал одного бара. Поддерживаемые значения: '1d', '1h', '15m' и др. (зависит от провайдера)."
    )

    indicators: list[str] = Field(
        default_factory=list,
        description=(
            "Список технических индикаторов для расчёта и включения в ответ. "
            "Примеры: ['rsi_14', 'ema_20', 'macd']. Возвращаются в поле indicators ответа."
        )
    )
    feature_plugins: list[str] = Field(
        default_factory=list,
        description="Список feature-плагинов для multivariate моделей"
    )

    @model_validator(mode="after")
    def validate_options(self):
        if self.data_source == "csv":
            if self.provider_options is not None and not isinstance(self.provider_options, CsvProviderOptions):
                raise ValueError("Для csv_path параметр provider_options должен иметь тип CsvProviderOptions.")
        else:
            if self.data_source == "t_invest":
                if self.provider_options is not None and not isinstance(self.provider_options, TInvestProviderOptions):
                    raise ValueError("Для data_source='t_invest' provider_options должен иметь тип TInvestProviderOptions.")
            if self.data_source == "yfinance":
                if self.provider_options is not None and not isinstance(self.provider_options, YahooProviderOptions):
                    raise ValueError("Для data_source='yfinance' provider_options должен иметь тип YahooProviderOptions.")

        if self.model_name == "patchtst":
            if self.model_options is not None and not isinstance(self.model_options, PatchTSTModelOptions):
                raise ValueError("Для model_name='patchtst' model_options должен иметь тип PatchTSTModelOptions.")
        else:
            if self.model_options is not None:
                raise ValueError("model_options задан, но model_name != 'patchtst'.")

        return self


class ForecastResponse(BaseModel):
    source: str = Field(description="Идентификатор источника данных: тикер, FIGI или путь к CSV-файлу.")
    model: dict[str, Any] = Field(description=(
        "Информация о модели из model.get_info(). Содержит: name, version, model_id, device, "
        "context_length (макс. длина входа), prediction_length (нативная длина выхода — "
        "максимальный поддерживаемый horizon), quantile_levels, supports_quantiles, supports_ohlc, "
        "last_input_window_info (параметры последнего инференса включая requested_horizon) и др."
    ))
    chart_type_history: Literal["line", "candlestick"] = Field(
        description="Тип графика истории, применённый при построении ответа."
    )
    chart_type_forecast: Literal["line", "candlestick"] = Field(
        description="Тип графика прогноза, применённый при построении ответа."
    )
    candles: list[dict[str, float]] = Field(description=(
        "Исторические свечи, переданные на вход модели. "
        "Каждый элемент: {open, high, low, close} и опционально volume. "
        "Длина совпадает с len(dates)."
    ))
    forecast_candles: list[dict[str, float]] = Field(description=(
        "Медианные прогнозные свечи (q50 по каждому каналу OHLC). "
        "Для chart_type_forecast='line' все поля свечи равны close_q50. "
        "Длина в норме равна horizon; при несовпадении см. horizon_mismatch_count."
    ))
    forecast_ohlc_quantiles: OHLCQuantileForecastSchema | None = Field(
        default=None,
        description=(
            "Полный квантильный прогноз OHLC: q10/q25/q50/q75/q90 для каждого канала. "
            "None если chart_type_forecast='line' или модель не поддерживает квантили."
        )
    )
    indicators: dict[str, list[float | None]] = Field(description=(
        "Рассчитанные технические индикаторы. "
        "Ключ — название индикатора (например 'rsi_14', 'ema_20'). "
        "Значение — список длиной history_length; None на позициях, где индикатор не определён "
        "(например первые N баров для скользящей средней)."
    ))
    dates: list[str] = Field(description=(
        "Даты исторических баров в формате ISO 8601. "
        "Длина совпадает с len(candles)."
    ))
    interval: str = Field(description="Временной интервал баров из запроса (например '1d', '1h', '15m').")
    model_input_start_date_used: str | None = Field(description=(
        "Дата первого бара, скормленного модели (ISO 8601). "
        "None если модель использовала всю доступную историю без обрезки."
    ))
    horizon_mismatch_count: int = Field(description=(
        "Разница len(forecast_candles) − horizon из запроса. "
        "В норме 0. Отрицательное значение означает, что модель вернула меньше точек чем запрошено."
    ))
    metadata: dict[str, Any] = Field(description=(
        "Диагностические метаданные запроса и инференса. "
        "Содержит: history_length, requested_horizon, history_period, history_up_to, "
        "model_options, feature_plugins, forecast_representation, "
        "model_required_context_length, model_input_history_length_original/used, "
        "model_input_trimmed, model_input_start_index_used. "
        "Подробнее — см. ForecastMetadataBuilder."
    ))


# ---------------------------------------------------------------------------
# Backtest request / response
# ---------------------------------------------------------------------------

class BacktestBaseRequest(BaseModel):
    """
    Общие параметры одиночного бэктеста и sweep'а.

    BacktestRequest и BacktestSweepRequest наследуются от этого класса и
    добавляют только специфичные поля:
      BacktestRequest      — train_window_size, persist, detailed_logs
      BacktestSweepRequest — параметры sweep-логики (enable_refinement, ranking_metric и т.п.)

    Валидатор согласованности data_source ↔ provider_options и model_name ↔
    model_options живёт здесь, наследуется обоими.
    """
    model_name: Literal["chronos", "patchtst"] | None = Field(
        default=None,
        description=(
            "Выбор модели. None — модель по умолчанию из конфигурации сервиса. "
            "'patchtst' — IBM Granite PatchTST FM. 'chronos' — Amazon Chronos."
        )
    )
    model_options: ModelOptions | None = Field(
        default=None,
        description=(
            "Специфичные опции модели. Тип должен соответствовать model_name: "
            "PatchTSTModelOptions для 'patchtst', ChronosModelOptions для 'chronos'. "
            "None — настройки по умолчанию."
        )
    )

    data_source: Literal["yfinance", "t_invest", "csv"] = Field(
        default="t_invest",
        description=(
            "Источник рыночных данных. "
            "'t_invest' — Т-Инвестиции API. "
            "'yfinance' — Yahoo Finance. "
            "'csv' — локальный CSV-файл."
        )
    )
    provider_options: ProviderOptions | None = Field(
        default=None,
        description="Параметры источника данных. Тип должен соответствовать data_source."
    )

    values: list[float] | None = Field(
        default=None,
        description=(
            "Произвольный временной ряд (только close) вместо загрузки через провайдера. "
            "Если задан — data_source и provider_options игнорируются."
        )
    )

    history_period: str = Field(
        default="1y",
        description="Период загружаемой истории. Форматы: '6mo', '1y', '2y'."
    )
    history_up_to: str | None = Field(
        default=None,
        description="Дата конца исторического окна в формате YYYY-MM-DD. None — до текущего момента."
    )
    interval: str = Field(
        default="1d",
        description="Временной интервал одного бара (например '1d', '1h')."
    )

    horizon: int = Field(
        default=64, ge=1, le=64,
        description=(
            "Горизонт прогноза в барах. Каждое окно предсказывает следующие horizon баров. "
            "Верхняя граница (64) равна prediction_length модели PatchTST FM по умолчанию. "
            "Точное значение для текущей модели смотри в response.model.prediction_length."
        )
    )
    feature_plugins: list[str] = Field(
        default_factory=list,
        description="Список feature-плагинов для multivariate моделей"
    )
    backtest_target: Literal["close", "ohlc"] = Field(
        default="ohlc",
        description=(
            "Режим бэктеста. "
            "'ohlc' — квантильный прогноз OHLC; точечные и квантильные метрики по всем каналам. "
            "'close' — точечный прогноз close; только MAE/RMSE/MAPE/directional_acc."
        )
    )

    train_window_mode: Literal["sliding", "growing"] = Field(
        default="sliding",
        description=(
            "Режим обучающего окна walk-forward. "
            "'sliding' (по умолчанию) — фиксированный размер контекста, окно скользит. "
            "'growing' — окно растёт от начала истории до train_end."
        )
    )
    step: int | None = Field(
        default=None, ge=1, le=8192,
        description=(
            "Шаг сдвига walk-forward окна в барах. "
            "None (по умолчанию) → step = horizon, окна не пересекаются."
        )
    )
    max_windows: int | None = Field(
        default=None, ge=1, le=10000,
        description="Опциональный потолок числа окон. None — без ограничения."
    )

    evaluation_weights: Literal["uniform", "exponential", "linear"] = Field(
        default="exponential",
        description=(
            "Схема весов для оценки многошагового прогноза. "
            "'exponential' — w_i = ratio^(-i/(H-1)); первый бар важнее последнего. "
            "'linear' — линейное убывание от 1 до 1/ratio. "
            "'uniform' — все веса равны."
        )
    )
    weight_first_to_last_ratio: float = Field(
        default=32.0, ge=1.0, le=1000.0,
        description=(
            "Во сколько раз первый бар горизонта важнее последнего. "
            "ratio=1.0 эквивалентно evaluation_weights='uniform'."
        )
    )

    bootstrap_iterations: int = Field(
        default=1000, ge=0, le=10000,
        description=(
            "Число итераций bootstrap для расчёта 95% CI и LCB. "
            "0 — пропустить расчёт. 1000 — стабильные оценки."
        )
    )
    ci_z_score: float = Field(
        default=1.96, ge=0.5, le=3.0,
        description="Z-коэффициент для LCB = mean - z * std. 1.96 ≈ 95% односторонняя нижняя граница."
    )

    inference_batch_size: int = Field(
        default=64, ge=1, le=256,
        description=(
            "Размер микро-батча для batched-инференса (GPU). "
            "Все walk-forward окна одного контекста собираются в чанки по N и "
            "прогоняются одним forward pass'ом."
        )
    )

    detailed_logs: bool = Field(
        default=False,
        description=(
            "Подробность ответа и логов. "
            "False (по умолчанию): "
            "  • single-backtest: details=[] (без per-window данных), "
            "    подавлены логи TSFM/HF инференса. "
            "  • sweep: каждый конфиг содержит только ~5 ключевых метрик, "
            "    без metrics_ci/metrics_lcb/cv_metrics_mean/cv_metrics_std. "
            "    Полные данные конкретного run всегда доступны через "
            "    GET /backtest/runs/{primary_run_id}. "
            "True: полные данные (per-window + полный набор метрик + CI/LCB) и подробные логи."
        )
    )

    @model_validator(mode="after")
    def validate_options(self):
        if self.data_source == "csv":
            if self.provider_options is not None and not isinstance(self.provider_options, CsvProviderOptions):
                raise ValueError("Для csv_path параметр provider_options должен иметь тип CsvProviderOptions.")
        else:
            if self.data_source == "t_invest":
                if self.provider_options is not None and not isinstance(self.provider_options, TInvestProviderOptions):
                    raise ValueError("Для data_source='t_invest' provider_options должен иметь тип TInvestProviderOptions.")
            if self.data_source == "yfinance":
                if self.provider_options is not None and not isinstance(self.provider_options, YahooProviderOptions):
                    raise ValueError("Для data_source='yfinance' provider_options должен иметь тип YahooProviderOptions.")

        if self.model_name == "patchtst":
            if self.model_options is not None and not isinstance(self.model_options, PatchTSTModelOptions):
                raise ValueError("Для model_name='patchtst' model_options должен иметь тип PatchTSTModelOptions.")
        else:
            if self.model_options is not None:
                raise ValueError("model_options задан, но model_name != 'patchtst'.")

        return self


class BacktestRequest(BacktestBaseRequest):
    """
    Одиночный walk-forward бэктест для конкретного train_window_size.

    Наследует все общие параметры от BacktestBaseRequest, добавляет
    train_window_size (контекст модели) и переключатель persist.
    """
    train_window_size: int = Field(
        default=512, ge=16, le=8192,
        description=(
            "Размер обучающего окна (= размер контекста модели). "
            "Для 'sliding' это размер каждого окна; для 'growing' — начальный размер. "
            "Должен быть ≤ model.context_length. Кратность 16 рекомендуется для PatchTST."
        )
    )
    persist: bool = Field(
        default=True,
        description=(
            "Сохранять результат бэктеста в БД (таблица backtest_runs). "
            "True — каждый запуск получает run_id и попадает в историю. "
            "False — результат не сохраняется, run_id=None."
        )
    )


RankingMetric = Literal[
    "skill_mae_close",     # backtest_target="ohlc": MAE_close vs naive (по умолчанию)
    "skill_mae",            # backtest_target="close": MAE vs naive
    "pinball_mean",         # общее качество квантильного прогноза (инвертируется: меньше=лучше)
    "directional_acc",      # точность направления
]


class BacktestSweepRequest(BacktestBaseRequest):
    """
    Sweep по разным размерам контекста (train_window_size).

    Наследует все общие параметры от BacktestBaseRequest. train_window_size НЕ
    задаётся — выбирается автоматически в ходе sweep.

    Алгоритм:
      1. Coarse pass — лог-spaced точки [128, 256, 512, 1024, 2048, 4096, 8192]
         (отсекаются те что больше истории/контекста модели).
      2. Refinement pass (если enable_refinement) — точки с фактором 1.25 вокруг
         лучшего coarse-результата для уточнения локального максимума.
      3. LCB-ранжирование по выбранной ranking_metric.
      4. CV-подтверждение для кандидатов, чьи CI перекрываются с top-1
         (до cv_max_candidates штук).
      5. Финальный recommended config: лучший CV-LCB, при равенстве — меньший контекст.
    """
    enable_refinement: bool = Field(
        default=True,
        description=(
            "Включить refinement pass — добавление точек с фактором 1.25 вокруг "
            "лучшего coarse-результата. False — только coarse (быстрее, грубее)."
        )
    )
    ranking_metric: RankingMetric = Field(
        default="skill_mae_close",
        description=(
            "Метрика для ранжирования конфигов. Для target='close' лучше 'skill_mae'. "
            "Для приоритизации направления — 'directional_acc'. "
            "pinball_mean инвертируется (меньше = лучше) автоматически."
        )
    )
    cv_folds: int = Field(
        default=3, ge=1, le=10,
        description=(
            "Максимальное число фолдов для CV-подтверждения. "
            "1 — CV отключена. Фактическое K может быть меньше если истории не хватает."
        )
    )
    cv_max_candidates: int = Field(
        default=7, ge=1, le=20,
        description="Потолок количества кандидатов для CV (защита от 'все перекрываются')."
    )
    lcb_tie_tolerance: float = Field(
        default=0.02, ge=0.0, le=1.0,
        description=(
            "Порог в единицах LCB, ниже которого считаем результаты статистически "
            "равными. При равенстве выбираем меньший train_window_size."
        )
    )


class SweepConfigResult(BaseModel):
    """
    Результат одного train_window_size в составе sweep.

    В кратком режиме (detailed_response=False) heavy-словари (metrics_ci, metrics_lcb,
    cv_metrics_mean, cv_metrics_std) равны None или пусты, а в metrics остаются только
    ~4 базовые метрики. Это сжимает ответ в ~10 раз.
    Полные данные конкретного run доступны через GET /backtest/runs/{primary_run_id}.
    """
    train_window_size: int
    pass_type: Literal["coarse", "refinement"]
    primary_run_id: int
    windows_count: int
    metrics: dict[str, float]
    metrics_ci: dict[str, list[float]] | None = None
    metrics_lcb: dict[str, float] | None = None
    ranking_metric_value: float
    ranking_metric_lcb: float
    cv_status: Literal["completed", "skipped_short_history", "not_selected"]
    cv_folds_used: int | None
    cv_metrics_mean: dict[str, float] | None = None
    cv_metrics_std: dict[str, float] | None = None
    cv_ranking_metric_lcb: float | None


class RecommendedConfig(BaseModel):
    train_window_size: int
    reason: Literal[
        "highest_cv_lcb",
        "tied_lcb_smaller_context",
        "no_cv_fallback_to_primary",
    ]
    ranking_metric: RankingMetric
    lcb: float
    lcb_margin: float


class BacktestSweepResponse(BaseModel):
    sweep_id: int = Field(description="Уникальный id sweep'а, по которому можно получить все его runs.")
    ticker: str
    source: str
    interval: str
    model_name: str
    history_length: int
    ranking_metric: RankingMetric
    configs: list[SweepConfigResult] = Field(
        description="Все протестированные конфиги (coarse + refinement), отсортированные по LCB."
    )
    recommended: RecommendedConfig
    cv_summary: dict[str, int] = Field(
        description="Сводка по CV: completed/skipped/not_selected (число кандидатов в каждой категории)."
    )


class BacktestResponse(BaseModel):
    source: str = Field(description="Идентификатор источника данных: тикер, FIGI или путь к CSV-файлу.")
    model: dict[str, Any] = Field(description=(
        "Информация о модели из model.get_info(). Включает context_length (макс. длина входа) "
        "и prediction_length (нативная длина выхода = максимальный horizon). "
        "Поле last_input_window_info отражает параметры ПОСЛЕДНЕГО окна бэктеста, не агрегат."
    ))
    run_id: int | None = Field(
        default=None,
        description=(
            "ID записи в таблице backtest_runs. None при persist=False. "
            "По этому id можно получить полные данные через GET /backtest/runs/{id}."
        )
    )
    metrics: dict[str, float] = Field(description=(
        "Агрегированные метрики (mean) по всем окнам бэктеста. "
        "Для backtest_target='close': mae, rmse, mape, directional_acc, "
        "naive_mae/rmse/mape, skill_mae (>0 → модель лучше наивного прогноза). "
        "Для backtest_target='ohlc': mae/rmse/mape по каждому каналу OHLC + агрегаты, "
        "квантильные метрики (pinball_mean, pinball_q50, coverage_q25_q75, coverage_q10_q90, "
        "winkler_q25_q75, winkler_q10_q90, directional_acc), naive baseline, skill_mae_close. "
        "Все метрики ВЗВЕШЕННЫЕ по схеме evaluation_weights."
    ))
    metrics_ci: dict[str, list[float]] = Field(description=(
        "95% доверительные интервалы [lower, upper] для каждой метрики, "
        "посчитанные bootstrap-ресэмплингом окон. "
        "Если bootstrap_iterations=0 — CI=[mean, mean]."
    ))
    metrics_lcb: dict[str, float] = Field(description=(
        "Lower Confidence Bound (mean − z * std) для каждой метрики. "
        "Это пессимистическая оценка: «насколько мы уверены, что результат хотя бы такой». "
        "Используется как основной критерий ранжирования в sweep."
    ))
    windows_count: int = Field(description=(
        "Фактическое число выполненных окон walk-forward. "
        "Зависит от длины истории, train_window_size, step и max_windows."
    ))
    horizon: int = Field(description="Горизонт прогноза из запроса (в барах).")
    history_length: int = Field(description="Длина исходного временного ряда (количество свечей).")
    details: list[dict[str, Any]] = Field(description=(
        "Детали по каждому окну walk-forward. Пустой список при detailed_logs=False (по умолчанию). "
        "При detailed_logs=True каждый элемент содержит: train_end_index, train_end_date, horizon, "
        "model_input_window, per-window метрики, naive baseline, actual_ohlc/forecast_ohlc "
        "(или actual_close/forecast_close)."
    ))
    metadata: dict[str, Any] = Field(description=(
        "Параметры бэктеста и диагностические сведения: "
        "train_window_mode, train_window_size, step, max_windows, evaluation_weights, "
        "weight_first_to_last_ratio, bootstrap_iterations, feature_plugins, backtest_target, "
        "trimmed_windows_count/share, metric_note."
    ))