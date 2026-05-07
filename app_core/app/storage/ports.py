from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

@dataclass
class ModelArtifact:
    symbol: str
    market: str | None
    interval: str
    model_name: str
    training_type: str  # lora | linear_probe | full_ft
    version: str
    status: str
    artifact_path: str
    metrics: dict[str, Any] | None = None
    params: dict[str, Any] | None = None


class ModelRegistryPort(Protocol):
    def upsert(self, item: ModelArtifact) -> None: ...
    def find_ready(
            self,
            symbol: str,
            interval: str,
            model_name: str,
            training_type: str,
            market: str | None = None,
    ) -> list[ModelArtifact]: ...


# ---------------------------------------------------------------------------
# Provider State
# ---------------------------------------------------------------------------

class ProviderStatePort(Protocol):
    def set_state(self, provider: str, key: str, value: dict[str, Any]) -> None: ...
    def get_state(self, provider: str, key: str) -> dict[str, Any] | None: ...


# ---------------------------------------------------------------------------
# Candle Repository
#
# Хранит рыночные свечи и покрытые диапазоны загрузки.
#
# Концепция покрытий (candle_coverage):
#   При загрузке данных за диапазон [from_dt, to_dt] сохраняем этот факт.
#   Перекрывающиеся диапазоны автоматически мержатся в один.
#   Это позволяет при следующем запросе определить какие данные уже есть
#   и дозапросить только недостающие куски.
#
# Атомарность:
#   upsert_candles_batch и upsert_coverage должны вызываться
#   в одном UnitOfWork — тогда они в одной транзакции.
# ---------------------------------------------------------------------------

@dataclass
class CandleRow:
    ticker: str
    source: str
    interval: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class CoverageRange:
    ticker: str
    source: str
    interval: str
    from_dt: str
    to_dt: str


@dataclass
class TickerInfo:
    ticker: str
    source: str
    interval: str
    from_dt: str
    to_dt: str
    candles_count: int


class CandleRepositoryPort(Protocol):

    def upsert_candles_batch(
            self,
            ticker: str,
            source: str,
            interval: str,
            rows: list[CandleRow],
    ) -> None:
        """
        Сохраняет батч свечей. INSERT OR REPLACE по (ticker, source, interval, timestamp).
        Вызывать в одном UnitOfWork вместе с upsert_coverage для атомарности.
        """
        ...

    def get_candles(
            self,
            ticker: str,
            source: str,
            interval: str,
            from_dt: str,
            to_dt: str,
    ) -> list[CandleRow]:
        """
        Возвращает свечи за диапазон [from_dt, to_dt] включительно.
        Отсортированы по timestamp ASC.
        """
        ...

    def get_coverage(
            self,
            ticker: str,
            source: str,
            interval: str,
    ) -> list[CoverageRange]:
        """
        Возвращает список покрытых диапазонов для данного инструмента.
        Диапазоны не перекрываются (мержатся при записи).
        Отсортированы по from_dt ASC.
        """
        ...

    def upsert_coverage(
            self,
            ticker: str,
            source: str,
            interval: str,
            from_dt: str,
            to_dt: str,
    ) -> None:
        """
        Добавляет покрытый диапазон и мержит с существующими.
        Алгоритм:
          1. Загрузить все существующие диапазоны для (ticker, source, interval)
          2. Добавить новый
          3. Отсортировать по from_dt
          4. Смержить перекрывающиеся и соседние
          5. Удалить старые, вставить смерженные
        Вызывать в одном UnitOfWork вместе с upsert_candles_batch.
        """
        ...

    def get_available_tickers(self) -> list[TickerInfo]:
        """
        Возвращает список всех доступных инструментов с метаданными.
        Используется эндпоинтом GET /market/tickers.
        """
        ...


# ---------------------------------------------------------------------------
# Unit of Work
# ---------------------------------------------------------------------------

class UnitOfWorkPort(Protocol):
    model_registry: ModelRegistryPort
    provider_state: ProviderStatePort
    candle_repository: CandleRepositoryPort

    def __enter__(self) -> "UnitOfWorkPort": ...
    def __exit__(self, exc_type, exc, tb) -> None: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...