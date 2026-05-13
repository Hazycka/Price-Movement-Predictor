from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ..ports import (
    ModelArtifact,
    CandleRow,
    CoverageRange,
    TickerInfo,
    UnavailableRange,
    BacktestRunRecord,
    UnitOfWorkPort,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_sqlite_schema(db_path: str) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path.as_posix(), timeout=30.0)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.executescript("""
                           CREATE TABLE IF NOT EXISTS model_artifacts (
                                                                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                          symbol TEXT NOT NULL,
                                                                          market TEXT,
                                                                          interval TEXT NOT NULL,
                                                                          model_name TEXT NOT NULL,
                                                                          training_type TEXT NOT NULL,
                                                                          version TEXT NOT NULL,
                                                                          status TEXT NOT NULL DEFAULT 'ready',
                                                                          metrics_json TEXT,
                                                                          params_json TEXT,
                                                                          artifact_path TEXT NOT NULL,
                                                                          created_at TEXT NOT NULL DEFAULT (datetime('now')),
                               updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                               UNIQUE(symbol, market, interval, model_name, training_type, version)
                               );

                           CREATE TABLE IF NOT EXISTS provider_state (
                                                                         provider TEXT NOT NULL,
                                                                         state_key TEXT NOT NULL,
                                                                         state_value_json TEXT NOT NULL,
                                                                         updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                               PRIMARY KEY(provider, state_key)
                               );

                           CREATE TABLE IF NOT EXISTS market_candles (
                                                                         ticker   TEXT NOT NULL,
                                                                         source   TEXT NOT NULL,
                                                                         interval TEXT NOT NULL,
                                                                         timestamp TEXT NOT NULL,
                                                                         open  REAL NOT NULL,
                                                                         high  REAL NOT NULL,
                                                                         low   REAL NOT NULL,
                                                                         close REAL NOT NULL,
                                                                         volume REAL NOT NULL DEFAULT 0.0,
                                                                         PRIMARY KEY (ticker, source, interval, timestamp)
                               );

                           CREATE TABLE IF NOT EXISTS candle_coverage (
                                                                          ticker   TEXT NOT NULL,
                                                                          source   TEXT NOT NULL,
                                                                          interval TEXT NOT NULL,
                                                                          from_dt  TEXT NOT NULL,
                                                                          to_dt    TEXT NOT NULL,
                                                                          PRIMARY KEY (ticker, source, interval, from_dt)
                               );

                           CREATE TABLE IF NOT EXISTS candle_unavailable_ranges (
                                                                          ticker      TEXT NOT NULL,
                                                                          source      TEXT NOT NULL,
                                                                          interval    TEXT NOT NULL,
                                                                          from_dt     TEXT NOT NULL,
                                                                          to_dt       TEXT NOT NULL,
                                                                          reason      TEXT NOT NULL,
                                                                          recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
                                                                          PRIMARY KEY (ticker, source, interval, from_dt, to_dt)
                               );

                           CREATE TABLE IF NOT EXISTS backtest_runs (
                                                                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                          model_name TEXT NOT NULL,
                                                                          ticker TEXT NOT NULL,
                                                                          source TEXT NOT NULL,
                                                                          interval TEXT NOT NULL,
                                                                          has_lora INTEGER NOT NULL DEFAULT 0,
                                                                          lora_artifact_id INTEGER,
                                                                          train_window_mode TEXT NOT NULL,
                                                                          train_window_size INTEGER NOT NULL,
                                                                          horizon INTEGER NOT NULL,
                                                                          step INTEGER NOT NULL,
                                                                          backtest_target TEXT NOT NULL,
                                                                          evaluation_weights TEXT NOT NULL,
                                                                          weight_first_to_last_ratio REAL NOT NULL,
                                                                          bootstrap_iterations INTEGER NOT NULL,
                                                                          ci_z_score REAL NOT NULL,
                                                                          history_period TEXT NOT NULL,
                                                                          history_up_to TEXT,
                                                                          history_length INTEGER NOT NULL,
                                                                          feature_plugins TEXT NOT NULL DEFAULT '[]',
                                                                          windows_count INTEGER NOT NULL,
                                                                          metrics_json TEXT NOT NULL,
                                                                          metrics_ci_json TEXT NOT NULL,
                                                                          metrics_lcb_json TEXT NOT NULL,
                                                                          metadata_json TEXT NOT NULL,
                                                                          sweep_id INTEGER,
                                                                          parent_run_id INTEGER,
                                                                          cv_fold_index INTEGER,
                                                                          created_at TEXT NOT NULL DEFAULT (datetime('now'))
                               );

                           CREATE INDEX IF NOT EXISTS idx_model_artifacts_lookup
                               ON model_artifacts(symbol, interval, model_name, training_type, status);

                           CREATE INDEX IF NOT EXISTS idx_market_candles_lookup
                               ON market_candles(ticker, source, interval, timestamp);

                           CREATE INDEX IF NOT EXISTS idx_candle_unavailable_lookup
                               ON candle_unavailable_ranges(ticker, source, interval);

                           CREATE INDEX IF NOT EXISTS idx_backtest_runs_lookup
                               ON backtest_runs(model_name, ticker, source, interval, has_lora);

                           CREATE INDEX IF NOT EXISTS idx_backtest_runs_sweep
                               ON backtest_runs(sweep_id);
                           """)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Model Registry Repository
# ---------------------------------------------------------------------------

class SQLiteModelRegistryRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    @staticmethod
    def _to_json(v: dict[str, Any] | None) -> str | None:
        return None if v is None else json.dumps(v, ensure_ascii=False)

    @staticmethod
    def _from_json(v: str | None) -> dict[str, Any] | None:
        return None if not v else json.loads(v)

    def upsert(self, item: ModelArtifact) -> None:
        self.conn.execute(
            """
            INSERT INTO model_artifacts (
                symbol, market, interval, model_name, training_type, version,
                status, metrics_json, params_json, artifact_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, market, interval, model_name, training_type, version)
            DO UPDATE SET
                status=excluded.status,
                                   metrics_json=excluded.metrics_json,
                                   params_json=excluded.params_json,
                                   artifact_path=excluded.artifact_path,
                                   updated_at=datetime('now')
            """,
            (
                item.symbol, item.market, item.interval, item.model_name,
                item.training_type, item.version, item.status,
                self._to_json(item.metrics), self._to_json(item.params),
                item.artifact_path,
            ),
        )

    def find_ready(
            self,
            symbol: str,
            interval: str,
            model_name: str,
            training_type: str,
            market: str | None = None,
    ) -> list[ModelArtifact]:
        rows = self.conn.execute(
            """
            SELECT * FROM model_artifacts
            WHERE symbol=? AND interval=? AND model_name=? AND training_type=?
              AND status='ready'
              AND (market=? OR (?  IS NULL AND market IS NULL))
            ORDER BY updated_at DESC
            """,
            (symbol, interval, model_name, training_type, market, market),
        ).fetchall()

        return [
            ModelArtifact(
                symbol=r["symbol"], market=r["market"], interval=r["interval"],
                model_name=r["model_name"], training_type=r["training_type"],
                version=r["version"], status=r["status"],
                artifact_path=r["artifact_path"],
                metrics=self._from_json(r["metrics_json"]),
                params=self._from_json(r["params_json"]),
            )
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Provider State Repository
# ---------------------------------------------------------------------------

class SQLiteProviderStateRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def set_state(self, provider: str, key: str, value: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO provider_state(provider, state_key, state_value_json)
            VALUES (?, ?, ?)
                ON CONFLICT(provider, state_key)
            DO UPDATE SET
                state_value_json=excluded.state_value_json,
                                   updated_at=datetime('now')
            """,
            (provider, key, json.dumps(value, ensure_ascii=False)),
        )

    def get_state(self, provider: str, key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT state_value_json FROM provider_state WHERE provider=? AND state_key=?",
            (provider, key),
        ).fetchone()
        return None if not row else json.loads(row["state_value_json"])


# ---------------------------------------------------------------------------
# Candle Repository
# ---------------------------------------------------------------------------

class SQLiteCandleRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def upsert_candles_batch(
            self,
            ticker: str,
            source: str,
            interval: str,
            rows: list[CandleRow],
    ) -> None:
        if not rows:
            return
        self.conn.executemany(
            """
            INSERT INTO market_candles
                (ticker, source, interval, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, source, interval, timestamp)
            DO UPDATE SET
                                   open=excluded.open, high=excluded.high,
                                   low=excluded.low,   close=excluded.close,
                                   volume=excluded.volume
            """,
            [
                (ticker, source, interval, r.timestamp,
                 r.open, r.high, r.low, r.close, r.volume)
                for r in rows
            ],
        )

    def get_candles(
            self,
            ticker: str,
            source: str,
            interval: str,
            from_dt: str,
            to_dt: str,
    ) -> list[CandleRow]:
        rows = self.conn.execute(
            """
            SELECT ticker, source, interval, timestamp,
                open, high, low, close, volume
            FROM market_candles
            WHERE ticker=? AND source=? AND interval=?
              AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            """,
            (ticker, source, interval, from_dt, to_dt),
        ).fetchall()

        return [
            CandleRow(
                ticker=r["ticker"], source=r["source"], interval=r["interval"],
                timestamp=r["timestamp"], open=r["open"], high=r["high"],
                low=r["low"], close=r["close"], volume=r["volume"],
            )
            for r in rows
        ]

    def get_coverage(
            self,
            ticker: str,
            source: str,
            interval: str,
    ) -> list[CoverageRange]:
        rows = self.conn.execute(
            """
            SELECT ticker, source, interval, from_dt, to_dt
            FROM candle_coverage
            WHERE ticker=? AND source=? AND interval=?
            ORDER BY from_dt ASC
            """,
            (ticker, source, interval),
        ).fetchall()

        return [
            CoverageRange(
                ticker=r["ticker"], source=r["source"], interval=r["interval"],
                from_dt=r["from_dt"], to_dt=r["to_dt"],
            )
            for r in rows
        ]

    def upsert_coverage(
            self,
            ticker: str,
            source: str,
            interval: str,
            from_dt: str,
            to_dt: str,
    ) -> None:
        """
        Добавляет диапазон и мержит с существующими.
        Алгоритм:
          1. Загрузить все существующие диапазоны
          2. Добавить новый
          3. Отсортировать по from_dt
          4. Смержить перекрывающиеся и соседние
          5. Удалить старые, вставить смерженные
        """
        existing = self.get_coverage(ticker, source, interval)
        ranges = [(r.from_dt, r.to_dt) for r in existing]
        ranges.append((from_dt, to_dt))
        ranges.sort(key=lambda x: x[0])

        merged: list[tuple[str, str]] = []
        for start, end in ranges:
            if merged and start <= merged[-1][1]:
                # Перекрывается или вплотную — расширяем последний
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Удаляем все старые покрытия для этого ключа
        self.conn.execute(
            "DELETE FROM candle_coverage WHERE ticker=? AND source=? AND interval=?",
            (ticker, source, interval),
        )

        # Вставляем смерженные
        self.conn.executemany(
            """
            INSERT INTO candle_coverage (ticker, source, interval, from_dt, to_dt)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(ticker, source, interval, s, e) for s, e in merged],
        )

    def get_available_tickers(self) -> list[TickerInfo]:
        """
        Возвращает список инструментов с детализацией покрытия.

        Не агрегируем coverage в один диапазон — возвращаем все фрагменты
        как есть. Если coverage у инструмента состоит из нескольких кусков
        с гэпами — это явно видно в coverage_periods.

        Три отдельных запроса:
          1. Все диапазоны из candle_coverage (фрагменты)
          2. COUNT свечей по ключу из market_candles
          3. Все диапазоны из candle_unavailable_ranges

        Список инструментов формируется как UNION ключей из coverage и
        unavailable — это позволяет показывать инструменты, по которым
        пока нет данных, но уже есть отметка о неудачной попытке.
        """
        coverage_rows = self.conn.execute(
            """
            SELECT ticker, source, interval, from_dt, to_dt
            FROM candle_coverage
            ORDER BY ticker, source, interval, from_dt
            """
        ).fetchall()

        count_rows = self.conn.execute(
            """
            SELECT ticker, source, interval, COUNT(timestamp) AS candles_count
            FROM market_candles
            GROUP BY ticker, source, interval
            """
        ).fetchall()

        unavailable_rows = self.conn.execute(
            """
            SELECT ticker, source, interval, from_dt, to_dt, reason, recorded_at
            FROM candle_unavailable_ranges
            ORDER BY ticker, source, interval, from_dt
            """
        ).fetchall()

        coverage_by_key: dict[tuple[str, str, str], list[CoverageRange]] = {}
        for r in coverage_rows:
            key = (r["ticker"], r["source"], r["interval"])
            coverage_by_key.setdefault(key, []).append(
                CoverageRange(
                    ticker=r["ticker"], source=r["source"], interval=r["interval"],
                    from_dt=r["from_dt"], to_dt=r["to_dt"],
                )
            )

        count_by_key: dict[tuple[str, str, str], int] = {
            (r["ticker"], r["source"], r["interval"]): r["candles_count"]
            for r in count_rows
        }

        unavailable_by_key: dict[tuple[str, str, str], list[UnavailableRange]] = {}
        for r in unavailable_rows:
            key = (r["ticker"], r["source"], r["interval"])
            unavailable_by_key.setdefault(key, []).append(
                UnavailableRange(
                    ticker=r["ticker"], source=r["source"], interval=r["interval"],
                    from_dt=r["from_dt"], to_dt=r["to_dt"],
                    reason=r["reason"], recorded_at=r["recorded_at"],
                )
            )

        all_keys = sorted(set(coverage_by_key) | set(unavailable_by_key))
        return [
            TickerInfo(
                ticker=key[0], source=key[1], interval=key[2],
                candles_count=count_by_key.get(key, 0),
                coverage_periods=coverage_by_key.get(key, []),
                unavailable_periods=unavailable_by_key.get(key, []),
            )
            for key in all_keys
        ]

    # ------------------------------------------------------------------
    # Unavailable ranges
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
        self.conn.execute(
            """
            INSERT INTO candle_unavailable_ranges
                (ticker, source, interval, from_dt, to_dt, reason, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(ticker, source, interval, from_dt, to_dt)
            DO UPDATE SET
                reason=excluded.reason,
                recorded_at=excluded.recorded_at
            """,
            (ticker, source, interval, from_dt, to_dt, reason),
        )

    def get_unavailable_ranges(
            self,
            ticker: str,
            source: str,
            interval: str,
    ) -> list[UnavailableRange]:
        rows = self.conn.execute(
            """
            SELECT ticker, source, interval, from_dt, to_dt, reason, recorded_at
            FROM candle_unavailable_ranges
            WHERE ticker=? AND source=? AND interval=?
            ORDER BY from_dt ASC
            """,
            (ticker, source, interval),
        ).fetchall()

        return [
            UnavailableRange(
                ticker=r["ticker"], source=r["source"], interval=r["interval"],
                from_dt=r["from_dt"], to_dt=r["to_dt"],
                reason=r["reason"], recorded_at=r["recorded_at"],
            )
            for r in rows
        ]

    def delete_unavailable_range_overlap(
            self,
            ticker: str,
            source: str,
            interval: str,
            from_dt: str,
            to_dt: str,
    ) -> None:
        """
        Удаляет все unavailable-записи, пересекающиеся с [from_dt, to_dt].
        Пересечение: NOT (record.to_dt <= from_dt OR record.from_dt >= to_dt).
        """
        self.conn.execute(
            """
            DELETE FROM candle_unavailable_ranges
            WHERE ticker=? AND source=? AND interval=?
              AND NOT (to_dt <= ? OR from_dt >= ?)
            """,
            (ticker, source, interval, from_dt, to_dt),
        )


# ---------------------------------------------------------------------------
# Backtest Runs Repository
# ---------------------------------------------------------------------------

class SQLiteBacktestRepository:
    """Реализация BacktestRepositoryPort для SQLite."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    @staticmethod
    def _row_to_record(r: sqlite3.Row) -> BacktestRunRecord:
        return BacktestRunRecord(
            id=r["id"],
            model_name=r["model_name"],
            ticker=r["ticker"],
            source=r["source"],
            interval=r["interval"],
            has_lora=bool(r["has_lora"]),
            lora_artifact_id=r["lora_artifact_id"],
            train_window_mode=r["train_window_mode"],
            train_window_size=r["train_window_size"],
            horizon=r["horizon"],
            step=r["step"],
            backtest_target=r["backtest_target"],
            evaluation_weights=r["evaluation_weights"],
            weight_first_to_last_ratio=r["weight_first_to_last_ratio"],
            bootstrap_iterations=r["bootstrap_iterations"],
            ci_z_score=r["ci_z_score"],
            history_period=r["history_period"],
            history_up_to=r["history_up_to"],
            history_length=r["history_length"],
            feature_plugins=json.loads(r["feature_plugins"]),
            windows_count=r["windows_count"],
            metrics=json.loads(r["metrics_json"]),
            metrics_ci=json.loads(r["metrics_ci_json"]),
            metrics_lcb=json.loads(r["metrics_lcb_json"]),
            metadata=json.loads(r["metadata_json"]),
            sweep_id=r["sweep_id"],
            parent_run_id=r["parent_run_id"],
            cv_fold_index=r["cv_fold_index"],
            created_at=r["created_at"],
        )

    def save_run(self, record: BacktestRunRecord) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO backtest_runs (
                model_name, ticker, source, interval, has_lora, lora_artifact_id,
                train_window_mode, train_window_size, horizon, step,
                backtest_target, evaluation_weights, weight_first_to_last_ratio,
                bootstrap_iterations, ci_z_score,
                history_period, history_up_to, history_length, feature_plugins,
                windows_count, metrics_json, metrics_ci_json, metrics_lcb_json, metadata_json,
                sweep_id, parent_run_id, cv_fold_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.model_name, record.ticker, record.source, record.interval,
                1 if record.has_lora else 0, record.lora_artifact_id,
                record.train_window_mode, record.train_window_size, record.horizon, record.step,
                record.backtest_target, record.evaluation_weights, record.weight_first_to_last_ratio,
                record.bootstrap_iterations, record.ci_z_score,
                record.history_period, record.history_up_to, record.history_length,
                json.dumps(record.feature_plugins, ensure_ascii=False),
                record.windows_count,
                json.dumps(record.metrics, ensure_ascii=False),
                json.dumps(record.metrics_ci, ensure_ascii=False),
                json.dumps(record.metrics_lcb, ensure_ascii=False),
                json.dumps(record.metadata, ensure_ascii=False, default=str),
                record.sweep_id, record.parent_run_id, record.cv_fold_index,
            ),
        )
        record.id = cursor.lastrowid
        return record.id

    def get_run(self, run_id: int) -> BacktestRunRecord | None:
        row = self.conn.execute(
            "SELECT * FROM backtest_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        return self._row_to_record(row) if row else None

    def get_runs(
            self,
            model_name: str | None = None,
            ticker: str | None = None,
            source: str | None = None,
            interval: str | None = None,
            has_lora: bool | None = None,
            sweep_id: int | None = None,
            limit: int = 100,
            offset: int = 0,
    ) -> list[BacktestRunRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if model_name is not None:
            clauses.append("model_name=?")
            params.append(model_name)
        if ticker is not None:
            clauses.append("ticker=?")
            params.append(ticker)
        if source is not None:
            clauses.append("source=?")
            params.append(source)
        if interval is not None:
            clauses.append("interval=?")
            params.append(interval)
        if has_lora is not None:
            clauses.append("has_lora=?")
            params.append(1 if has_lora else 0)
        if sweep_id is not None:
            clauses.append("sweep_id=?")
            params.append(sweep_id)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM backtest_runs {where} ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_sweep_runs(self, sweep_id: int) -> list[BacktestRunRecord]:
        rows = self.conn.execute(
            "SELECT * FROM backtest_runs WHERE sweep_id=? ORDER BY id ASC",
            (sweep_id,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_next_sweep_id(self) -> int:
        """
        Возвращает следующий sweep_id. Используем MAX+1 (не AUTOINCREMENT,
        потому что sweep_id — отдельное измерение, не PK).
        """
        row = self.conn.execute("SELECT COALESCE(MAX(sweep_id), 0) AS m FROM backtest_runs").fetchone()
        return int(row["m"]) + 1


# ---------------------------------------------------------------------------
# Unit of Work
# ---------------------------------------------------------------------------

class SQLiteUnitOfWork(UnitOfWorkPort):
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: sqlite3.Connection | None = None
        self.model_registry: SQLiteModelRegistryRepo | None = None
        self.provider_state: SQLiteProviderStateRepo | None = None
        self.candle_repository: SQLiteCandleRepository | None = None
        self.backtest_repository: SQLiteBacktestRepository | None = None

    def __enter__(self) -> "SQLiteUnitOfWork":
        self.conn = sqlite3.connect(self.db_path.as_posix(), timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.model_registry = SQLiteModelRegistryRepo(self.conn)
        self.provider_state = SQLiteProviderStateRepo(self.conn)
        self.candle_repository = SQLiteCandleRepository(self.conn)
        self.backtest_repository = SQLiteBacktestRepository(self.conn)
        return self

    def commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        self.conn.rollback()

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type:
            self.rollback()
        else:
            self.commit()
        self.conn.close()