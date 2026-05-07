from __future__ import annotations

from .base import MarketDataRequest
from .candle_cache_service import CandleCacheService
from .factory import get_market_data_provider
from ...schemas import ProviderOptions, CsvProviderOptions


_cache_service = CandleCacheService()


def load_ohlc_from_ticker(
        history_period: str,
        interval: str,
        provider_options: ProviderOptions,
        history_up_to: str | None = None,
        provider: str = "t_invest",
) -> tuple[list[str], list[dict[str, float]]]:
    """
    Загружает свечи через CandleCacheService.
    Сначала проверяет БД, дозапрашивает только недостающее.
    """
    return _cache_service.load_ohlc(
        history_period=history_period,
        interval=interval,
        provider_options=provider_options,
        history_up_to=history_up_to,
        provider=provider,
    )


def load_ohlc_from_csv(
        provider_options: ProviderOptions,
) -> tuple[list[str], list[dict[str, float]]]:
    request = MarketDataRequest(provider_options=provider_options)
    return get_market_data_provider("csv").load_ohlc(request)