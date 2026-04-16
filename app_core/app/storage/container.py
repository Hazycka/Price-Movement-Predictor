from __future__ import annotations

import os
from typing import Callable

from .adapters.sqlite import SQLiteUnitOfWork, init_sqlite_schema
from .ports import UnitOfWorkPort

_uow_factory: Callable[[], UnitOfWorkPort] | None = None


def init_storage() -> None:
    global _uow_factory
    driver = os.getenv("APP_DB_DRIVER", "sqlite").strip().lower()

    if driver == "sqlite":
        db_path = os.getenv("APP_DB_PATH", "data/app.sqlite3")
        init_sqlite_schema(db_path)
        _uow_factory = lambda: SQLiteUnitOfWork(db_path)
        return

    if driver == "postgres":
        raise NotImplementedError("Postgres adapter пока не реализован")

    raise ValueError(f"Unsupported APP_DB_DRIVER: {driver}")


def get_uow_factory() -> Callable[[], UnitOfWorkPort]:
    if _uow_factory is None:
        init_storage()
    return _uow_factory