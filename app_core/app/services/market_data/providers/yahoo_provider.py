from __future__ import annotations

import pandas as pd
import yfinance as yf

from ..base import MarketDataProvider, MarketDataRequest
from ..common import resolve_history_window
from ....schemas import YahooProviderOptions


def _normalize_ohlc_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        else:
            data.columns = data.columns.get_level_values(0)

    data.columns = [str(column) for column in data.columns]
    return data


class YahooMarketDataProvider(MarketDataProvider):
    def load_ohlc(self, request: MarketDataRequest) -> tuple[list[str], list[dict[str, float]]]:
        options = request.provider_options
        if not isinstance(options, YahooProviderOptions):
            raise TypeError(f"Ожидались настройки Yahoo, но получено {type(options).__name__}")
        
        ticker = options.ticker
        if not ticker:
            raise ValueError("Для Yahoo provider требуется ticker.")

        start, end = resolve_history_window(request.history_period, request.history_up_to)

        data = yf.download(
            tickers=ticker,
            period=None if start else request.history_period,
            start=start,
            end=end,
            interval=request.interval,
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            raise ValueError(f"Не удалось загрузить данные по тикеру '{ticker}'.")

        data = _normalize_ohlc_columns(data, ticker.upper())

        required_columns = ["Open", "High", "Low", "Close"]
        missing_columns = [column for column in required_columns if column not in data.columns]
        if missing_columns:
            raise ValueError(
                f"В данных по тикеру '{ticker}' отсутствуют колонки: {missing_columns}. "
                f"Фактические колонки: {list(data.columns)}"
            )

        data = data.dropna(subset=required_columns)
        dates = [str(idx) for idx in data.index.tolist()]

        candles: list[dict[str, float]] = []
        for _, row in data.iterrows():
            candles.append({
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]) if "Volume" in data.columns and not pd.isna(row["Volume"]) else 0.0
            })

        return dates, candles