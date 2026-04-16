from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from ...schemas import ProviderOptions


@dataclass
class MarketDataRequest:
    provider_options: ProviderOptions
    history_period: str = "1y"
    interval: str = "1d"
    history_up_to: str | None = None


class MarketDataProvider(ABC):
    @abstractmethod
    def load_ohlc(self, request: MarketDataRequest) -> tuple[list[str], list[dict[str, float]]]:
        pass