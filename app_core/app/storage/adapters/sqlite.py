from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ..ports import ModelArtifact, UnitOfWorkPort


def init_sqlite_schema(db_path: str) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path.as_posix(), timeout=30.0)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.executescript(
            """
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

            CREATE INDEX IF NOT EXISTS idx_model_artifacts_lookup
                ON model_artifacts(symbol, interval, model_name, training_type, status);
            """
        )
        conn.commit()
    finally:
        conn.close()


class SQLiteModelRegistryRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    @staticmethod
    def _to_json(v: dict[str, Any] | None) -> str | None:
        if v is None:
            return None
        return json.dumps(v, ensure_ascii=False)

    @staticmethod
    def _from_json(v: str | None) -> dict[str, Any] | None:
        if not v:
            return None
        return json.loads(v)

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
                item.symbol,
                item.market,
                item.interval,
                item.model_name,
                item.training_type,
                item.version,
                item.status,
                self._to_json(item.metrics),
                self._to_json(item.params),
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
            SELECT *
            FROM model_artifacts
            WHERE symbol = ?
              AND interval = ?
              AND model_name = ?
              AND training_type = ?
              AND status = 'ready'
              AND (market = ? OR (? IS NULL AND market IS NULL))
            ORDER BY updated_at DESC
            """,
            (symbol, interval, model_name, training_type, market, market),
        ).fetchall()

        result: list[ModelArtifact] = []
        for r in rows:
            result.append(
                ModelArtifact(
                    symbol=r["symbol"],
                    market=r["market"],
                    interval=r["interval"],
                    model_name=r["model_name"],
                    training_type=r["training_type"],
                    version=r["version"],
                    status=r["status"],
                    artifact_path=r["artifact_path"],
                    metrics=self._from_json(r["metrics_json"]),
                    params=self._from_json(r["params_json"]),
                )
            )
        return result


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
            """
            SELECT state_value_json
            FROM provider_state
            WHERE provider = ? AND state_key = ?
            """,
            (provider, key),
        ).fetchone()
        if not row:
            return None
        return json.loads(row["state_value_json"])


class SQLiteUnitOfWork(UnitOfWorkPort):
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: sqlite3.Connection | None = None
        self.model_registry = None
        self.provider_state = None

    def __enter__(self) -> "SQLiteUnitOfWork":
        self.conn = sqlite3.connect(self.db_path.as_posix(), timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.model_registry = SQLiteModelRegistryRepo(self.conn)
        self.provider_state = SQLiteProviderStateRepo(self.conn)
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