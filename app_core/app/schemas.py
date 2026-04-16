from pydantic import BaseModel, Field, model_validator
from typing import Any, Literal


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
    date_column: str = Field(default="Date")
    open_column: str = Field(default="Open")
    high_column: str = Field(default="High")
    low_column: str = Field(default="Low")
    close_column: str = Field(default="Close")
    volume_column: str | None = Field(default="Volume")


class PatchTSTModelOptions(BaseModel):
    pass


ProviderOptions = TInvestProviderOptions | YahooProviderOptions | CsvProviderOptions
ModelOptions = PatchTSTModelOptions


class ForecastRequest(BaseModel):
    model_name: Literal["chronos", "patchtst"] | None = Field(default=None,
                                                              description="Выбор модели для конкретного запроса")
    model_options: ModelOptions | None = Field(default=None)

    data_source: Literal["yfinance", "t_invest", "csv"] = Field(default="t_invest")
    provider_options: ProviderOptions | None = Field(default=None)

    chart_type_history: Literal["line", "candlestick"] = Field(default="candlestick")
    chart_type_forecast: Literal["line", "candlestick"] = Field(default="candlestick")
    horizon: int = Field(default=5, ge=1, le=60)
    history_period: str = Field(default="1y", description="Период для истории, например 6mo, 1y, 2y")
    history_up_to: str | None = Field(default=None, description="Дата конца окна истории, например 2026-03-20")
    interval: str = Field(default="1d", description="Интервал данных, например 1d, 1h")

    indicators: list[str] = Field(default_factory=lambda: ["sma_20", "ema_20", "rsi_14"])
    num_samples: int = Field(default=64, ge=1, le=256,
                             description="Количество сэмплов в вероятностном прогнозе Chronos")
    feature_plugins: list[str] = Field(default_factory=list,
                                       description="Список feature-плагинов для multivariate моделей")

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
    source: str
    model: dict[str, Any]
    chart_type_history: Literal["line", "candlestick"]
    chart_type_forecast: Literal["line", "candlestick"]
    candles: list[dict[str, float]]
    forecast_candles: list[dict[str, float]]
    indicators: dict[str, list[float | None]]
    dates: list[str]
    interval: str
    model_input_start_date_used: str | None
    horizon_mismatch_count: int
    metadata: dict[str, Any]


class BacktestRequest(BaseModel):
    model_name: Literal["chronos", "patchtst"] | None = Field(default=None,
                                                              description="Выбор модели для конкретного запроса")
    model_options: ModelOptions | None = Field(default=None)

    data_source: Literal["yfinance", "t_invest", "csv"] = Field(default="t_invest")
    provider_options: ProviderOptions | None = Field(default=None)

    values: list[float] | None = Field(default=None, description="Пользовательский временной ряд")

    history_period: str = Field(default="1y")
    history_up_to: str | None = Field(default=None)
    interval: str = Field(default="1d")

    horizon: int = Field(default=5, ge=1, le=60)
    num_samples: int = Field(default=64, ge=1, le=256)
    feature_plugins: list[str] = Field(
        default_factory=list,
        description="Список feature-плагинов для multivariate моделей"
    )
    backtest_target: Literal["close", "ohlc"] = Field(
        default="ohlc",
        description="Что валидировать в бэктесте: только close или полный OHLC"
    )

    min_train_size: int = Field(default=120, ge=20, le=5000, description="Минимальная длина обучающего окна")
    step: int = Field(default=5, ge=1, le=200, description="Шаг окна walk-forward")
    max_windows: int = Field(default=50, ge=1, le=1000, description="Ограничение числа окон бэктеста")

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


class BacktestResponse(BaseModel):
    source: str
    model: dict[str, Any]
    metrics: dict[str, float]
    windows_count: int
    horizon: int
    history_length: int
    details: list[dict[str, Any]]
    metadata: dict[str, Any]