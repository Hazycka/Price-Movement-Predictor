from ..market_data import load_ohlc_from_ticker, load_ohlc_from_csv
from ...schemas import ForecastRequest, BacktestRequest, CsvProviderOptions, TInvestProviderOptions, YahooProviderOptions


class InputResolver:

    @staticmethod
    def resolve(request: ForecastRequest | BacktestRequest) -> tuple[str, list[str], list[dict[str, float]]]:
        options = request.provider_options
        
        if isinstance(options, CsvProviderOptions):
            dates, candles = load_ohlc_from_csv(
                provider_options = options
            )

            if len(candles) < 10:
                raise ValueError("Слишком мало данных для прогнозирования (минимум 10 свечей).")

            return options.csv_path, dates, candles

        if isinstance(options, TInvestProviderOptions | YahooProviderOptions):
            provider = "t_invest" if request.data_source == "t_invest" else "yfinance"

            dates, candles = load_ohlc_from_ticker(
                history_period=request.history_period,
                interval=request.interval,
                history_up_to=request.history_up_to,
                provider=provider,
                provider_options=options
            )

            if len(candles) < 10:
                raise ValueError("Слишком мало данных для прогнозирования (минимум 10 свечей).")
            
            return options.ticker.upper(), dates, candles

        raise ValueError("Нужно передать либо ticker, либо values, либо csv_path.")