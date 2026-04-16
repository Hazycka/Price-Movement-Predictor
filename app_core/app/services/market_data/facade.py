from __future__ import annotations

from .base import MarketDataRequest
from .factory import get_market_data_provider
from ...schemas import ProviderOptions


def load_ohlc_from_ticker(
        history_period: str,
        interval: str,
        provider_options: ProviderOptions,
        history_up_to: str | None = None,
        provider: str = "t_invest"
) -> tuple[list[str], list[dict[str, float]]]:
    request = MarketDataRequest(
        history_period=history_period,
        interval=interval,
        history_up_to=history_up_to,
        provider_options=provider_options
    )
    return get_market_data_provider(provider).load_ohlc(request)


def load_ohlc_from_csv(
        provider_options: ProviderOptions
) -> tuple[list[str], list[dict[str, float]]]:
    request = MarketDataRequest(
        provider_options=provider_options
    )
    return get_market_data_provider("csv").load_ohlc(request)