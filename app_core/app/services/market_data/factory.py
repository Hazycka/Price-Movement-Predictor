from __future__ import annotations

from .base import MarketDataProvider
from .providers import CsvMarketDataProvider
from .providers import TInvestMarketDataProvider
from .providers import YahooMarketDataProvider


def get_market_data_provider(source: str) -> MarketDataProvider:
    normalized = (source or "").strip().lower()
    if normalized == "yfinance":
        return YahooMarketDataProvider()
    if normalized == "csv":
        return CsvMarketDataProvider()
    if normalized in ("t_invest", "tinvest"):
        return TInvestMarketDataProvider()
    raise ValueError(f"Неизвестный источник данных: {source}")