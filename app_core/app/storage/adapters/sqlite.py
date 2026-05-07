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

                           CREATE INDEX IF NOT EXISTS idx_model_artifacts_lookup
                               ON model_artifacts(symbol, interval, model_name, training_type, status);

                           CREATE INDEX IF NOT EXISTS idx_market_candles_lookup
                               ON market_candles(ticker, source, interval, timestamp);
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
        Возвращает список инструментов с покрытием и количеством свечей.
        JOIN candle_coverage + COUNT из market_candles.
        """
        rows = self.conn.execute(
            """
            SELECT
                cc.ticker, cc.source, cc.interval,
                MIN(cc.from_dt) as from_dt,
                MAX(cc.to_dt)   as to_dt,
                COUNT(mc.timestamp) as candles_count
            FROM candle_coverage cc
                     LEFT JOIN market_candles mc
                               ON mc.ticker=cc.ticker
                                   AND mc.source=cc.source
                                   AND mc.interval=cc.interval
            GROUP BY cc.ticker, cc.source, cc.interval
            ORDER BY cc.ticker, cc.source, cc.interval
            """
        ).fetchall()

        return [
            TickerInfo(
                ticker=r["ticker"], source=r["source"], interval=r["interval"],
                from_dt=r["from_dt"], to_dt=r["to_dt"],
                candles_count=r["candles_count"],
            )
            for r in rows
        ]


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

    def __enter__(self) -> "SQLiteUnitOfWork":
        self.conn = sqlite3.connect(self.db_path.as_posix(), timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.model_registry = SQLiteModelRegistryRepo(self.conn)
        self.provider_state = SQLiteProviderStateRepo(self.conn)
        self.candle_repository = SQLiteCandleRepository(self.conn)
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