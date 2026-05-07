from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
        """
        Загружает свечи по history_period + history_up_to.
        Используется для обратной совместимости.
        """
        pass

    @abstractmethod
    def load_ohlc_range(
            self,
            provider_options: ProviderOptions,
            interval: str,
            from_dt: str,
            to_dt: str,
    ) -> tuple[list[str], list[dict[str, float]]]:
        """
        Загружает свечи за конкретный диапазон дат [from_dt, to_dt].
        Используется CandleCacheService для точечной догрузки данных.
        from_dt, to_dt — ISO строки с timezone (UTC).
        """
        pass