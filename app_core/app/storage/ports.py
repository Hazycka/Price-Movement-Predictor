from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

@dataclass
class ModelArtifact:
    """
    Дообученный артефакт поверх foundation-модели.

    Идентичность артефакта (UNIQUE-ключ в БД):
        (symbol, source, interval, model_name, training_components, train_window_size, version)

    Поля:
      symbol  — тикер инструмента (SBER, AAPL и т.д.).
      source  — провайдер данных, на которых обучали (t_invest / yahoo / csv).
                Часть UNIQUE-ключа, NOT NULL.
      market  — биржевая секция / код (TQBR, SPBXM, NASDAQ и т.д.). Информативное
                поле, в UNIQUE-ключе НЕ участвует.
      interval — таймфрейм данных (1h, 1d).
      model_name — имя foundation-модели (patchtst, chronos и т.д.).

      training_components — список компонентов которые применяются при загрузке:
         ["lora"]         — только LoRA-адаптеры (низкоранговые матрицы)
         ["head"]         — только новая output head (linear probing)
         ["lora", "head"] — комбо: LoRA + новая head обучаются вместе
         ["full_ft"]      — полное дообучение (не планируем, но архитектурно возможно)

      train_window_size — контекст на котором обучен артефакт. Часть «идентичности»
        артефакта: разные контексты дают разные внутренние представления, на (ticker,
        interval) может быть несколько артефактов с разными train_window_size.

      artifact_path — путь к директории с файлами:
         data/artifacts/{id}/
            adapter_model.safetensors    (если "lora" в components)
            head.pt                       (если "head" в components)
            metadata.json                 (всегда — снимок params + базовая инфа)

      params  — конфиг обучения (LR, epochs, lora_r/alpha, training_loss и т.д.).
      metrics — val-метрики после обучения (val_pinball, val_skill, val_dir_acc).
    """
    symbol: str
    source: str                        # NOT NULL: провайдер (t_invest / yahoo / csv)
    market: str | None                 # nullable: биржевая секция (TQBR, SPBXM, ...)
    interval: str
    model_name: str
    training_components: list[str]
    train_window_size: int
    version: str
    status: str
    artifact_path: str
    metrics: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    id: int | None = None
    created_at: str | None = None


class ModelRegistryPort(Protocol):
    def upsert(self, item: ModelArtifact) -> int:
        """Сохраняет/обновляет артефакт, возвращает его id."""
        ...

    def get_by_id(self, artifact_id: int) -> ModelArtifact | None:
        """Возвращает артефакт по id или None."""
        ...

    def find_ready(
            self,
            symbol: str,
            interval: str,
            model_name: str,
            source: str | None = None,
    ) -> list[ModelArtifact]:
        """Возвращает все артефакты со статусом 'ready' для данного инструмента.
        source опционален — без него поиск идёт по всем провайдерам."""
        ...

    def list_all(self) -> list[ModelArtifact]:
        """Все артефакты в реестре (для GET /artifacts эндпоинта)."""
        ...


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
class UnavailableRange:
    """
    Диапазон, за который провайдер вернул пустой ответ или ошибку.
    Эфемерный — повторно проверяется при каждом запросе того же диапазона.
    """
    ticker: str
    source: str
    interval: str
    from_dt: str
    to_dt: str
    reason: str
    recorded_at: str


@dataclass
class TickerInfo:
    """
    Сводка по инструменту в локальной БД.

    coverage_periods    — все диапазоны где у нас есть свечи. Может быть
                          несколько фрагментов с гэпами между ними.
    unavailable_periods — диапазоны где провайдер не вернул данные.
    candles_count       — общее число свечей в market_candles по ключу
                          (ticker, source, interval), может быть >0
                          даже если coverage_periods пуст (orphan-свечи).
    """
    ticker: str
    source: str
    interval: str
    candles_count: int
    coverage_periods: list[CoverageRange]
    unavailable_periods: list[UnavailableRange]


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
        Каждый TickerInfo включает список unavailable_periods — диапазонов,
        за которые провайдер не вернул данных. Используется эндпоинтом GET /market/tickers.
        """
        ...

    # ------------------------------------------------------------------
    # Unavailable ranges
    #
    # Хранят диапазоны, за которые провайдер вернул пустой ответ.
    # В отличие от candle_coverage — эфемерны: при следующем запросе
    # того же диапазона провайдер опрашивается снова (вдруг данные
    # появились). Если опять пусто — recorded_at обновляется.
    # ------------------------------------------------------------------

    def upsert_unavailable_range(
            self,
            ticker: str,
            source: str,
            interval: str,
            from_dt: str,
            to_dt: str,
            reason: str,
    ) -> None:
        """
        Записывает или обновляет диапазон, за который провайдер вернул пустой ответ.
        При совпадающем (ticker, source, interval, from_dt, to_dt) обновляет reason
        и recorded_at — это позволяет видеть когда последний раз пытались получить данные.
        """
        ...

    def get_unavailable_ranges(
            self,
            ticker: str,
            source: str,
            interval: str,
    ) -> list[UnavailableRange]:
        """
        Возвращает все недоступные диапазоны для (ticker, source, interval).
        Отсортированы по from_dt ASC.
        """
        ...

    def delete_unavailable_range_overlap(
            self,
            ticker: str,
            source: str,
            interval: str,
            from_dt: str,
            to_dt: str,
    ) -> None:
        """
        Удаляет все unavailable-записи, пересекающиеся с диапазоном [from_dt, to_dt].
        Вызывается когда провайдер вернул реальные данные за этот диапазон —
        предыдущая отметка о недоступности больше не актуальна.
        """
        ...


# ---------------------------------------------------------------------------
# Backtest Runs Repository
#
# Хранит результаты walk-forward бэктестов (одиночных и в составе sweep).
# Главные сценарии использования:
#   1. После любого /backtest (с persist=True) — сохраняем один run
#   2. /backtest/sweep — сохраняем N primary runs + K*M CV runs со связями
#   3. Просмотр истории / выбор лучшего конфига для тикера
# ---------------------------------------------------------------------------

@dataclass
class BacktestRunRecord:
    """
    Снапшот одного walk-forward бэктеста (или его CV-фолда).

    Идентификация артефакта (если использовался адаптер):
      artifact_id         — id записи в model_artifacts, None для base модели
      applied_components  — список компонентов которые были применены
                            (["lora"], ["head"], ["lora", "head"]) или []  для base

    Связи runs:
      sweep_id      — None для standalone runs; одинаковый id для всех runs
                      одного sweep (включая CV-фолды)
      parent_run_id — для CV-фолдов: ссылка на primary run, который проверяется.
                      None для primary runs.
      cv_fold_index — 0..K-1 для CV-фолдов; None для primary runs.
    """
    # Идентификация
    model_name: str
    ticker: str
    source: str
    interval: str
    artifact_id: int | None
    applied_components: list[str]

    # Параметры бэктеста (snapshot)
    train_window_mode: str
    train_window_size: int
    horizon: int
    step: int
    backtest_target: str
    evaluation_weights: str
    weight_first_to_last_ratio: float
    bootstrap_iterations: int
    ci_z_score: float
    history_period: str
    history_up_to: str | None
    history_length: int
    feature_plugins: list[str]

    # Результаты (JSON-сериализуемые)
    windows_count: int
    metrics: dict[str, float]
    metrics_ci: dict[str, list[float]]
    metrics_lcb: dict[str, float]
    metadata: dict[str, Any]

    # Связи и метаданные
    sweep_id: int | None = None
    parent_run_id: int | None = None
    cv_fold_index: int | None = None
    id: int | None = None                # назначается после save_run
    created_at: str | None = None         # назначается БД


class BacktestRepositoryPort(Protocol):
    def save_run(self, record: BacktestRunRecord) -> int:
        """
        Сохраняет run и возвращает присвоенный id.
        Также записывает id обратно в record.id для удобства caller'а.
        """
        ...

    def get_run(self, run_id: int) -> BacktestRunRecord | None:
        """Возвращает run по id или None если не найден."""
        ...

    def get_runs(
            self,
            model_name: str | None = None,
            ticker: str | None = None,
            source: str | None = None,
            interval: str | None = None,
            artifact_id: int | None = None,
            sweep_id: int | None = None,
            limit: int = 100,
            offset: int = 0,
    ) -> list[BacktestRunRecord]:
        """
        Список runs с фильтрами. Все фильтры опциональны.
        Возвращает в порядке убывания id (новые первыми).
        """
        ...

    def get_sweep_runs(self, sweep_id: int) -> list[BacktestRunRecord]:
        """Все runs одного sweep'а (primary + CV) в порядке создания."""
        ...

    def get_next_sweep_id(self) -> int:
        """Возвращает следующий уникальный sweep_id для группировки нового sweep'а."""
        ...


# ---------------------------------------------------------------------------
# Unit of Work
# ---------------------------------------------------------------------------

class UnitOfWorkPort(Protocol):
    model_registry: ModelRegistryPort
    provider_state: ProviderStatePort
    candle_repository: CandleRepositoryPort
    backtest_repository: BacktestRepositoryPort

    def __enter__(self) -> "UnitOfWorkPort": ...
    def __exit__(self, exc_type, exc, tb) -> None: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...