from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _RangeInfo:
    from_dt: str
    to_dt: str
    reason: str | None = None


class DataUnavailableError(Exception):
    """
    Возникает когда провайдер не смог вернуть данные за один или несколько
    запрошенных диапазонов. Несёт информацию о том какие диапазоны
    недоступны (с причиной) и какие — наоборот, доступны (можно запросить).

    Перехватывается в API-слое и превращается в HTTP 422 со структурированным
    телом ответа.
    """

    def __init__(
            self,
            ticker: str,
            source: str,
            interval: str,
            unavailable: list[_RangeInfo],
            available: list[_RangeInfo],
    ) -> None:
        self.ticker = ticker
        self.source = source
        self.interval = interval
        self.unavailable = unavailable
        self.available = available

        unavail_brief = ", ".join(
            f"[{r.from_dt[:10]} → {r.to_dt[:10]}]" for r in unavailable
        )
        avail_brief = (
            ", ".join(f"[{r.from_dt[:10]} → {r.to_dt[:10]}]" for r in available)
            if available else "нет доступных диапазонов"
        )
        super().__init__(
            f"Провайдер '{source}' не вернул данные для {ticker} {interval} "
            f"за диапазоны: {unavail_brief}. "
            f"Доступные диапазоны: {avail_brief}."
        )
