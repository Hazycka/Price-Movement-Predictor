from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


SUPPORTED_INTERVALS = (
    "1min", "2min", "3min", "5min", "10min", "15min", "30min",
    "1h", "2h", "4h",
    "1d", "1w", "1mo"
)

INTERVAL_TO_FREQ: dict[str, str] = {
    "1min":  "min",
    "2min":  "2min",
    "3min":  "3min",
    "5min":  "5min",
    "10min": "10min",
    "15min": "15min",
    "30min": "30min",
    "1h":    "h",
    "2h":    "2h",
    "4h":    "4h",
    "1d":    "B",
    "1w":    "W",
    "1mo":   "MS",
}

DEFAULT_FREQ = "h"


def parse_history_period(period: str) -> pd.DateOffset:
    value = int(period[:-1])
    unit = period[-1]

    if unit == "d":
        return pd.DateOffset(days=value)
    if unit == "w":
        return pd.DateOffset(weeks=value)
    if unit == "m":
        return pd.DateOffset(months=value)
    if unit == "y":
        return pd.DateOffset(years=value)

    raise ValueError(f"Неподдерживаемый формат history_period: {period}")


def resolve_history_window(
        history_period: str,
        history_up_to: str | None
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if not history_up_to:
        return None, None

    end = pd.to_datetime(history_up_to, errors="raise")
    start = end - parse_history_period(history_period)
    return start, end


def resolve_end_dt(history_up_to: str | None) -> datetime:
    if history_up_to:
        dt = pd.to_datetime(history_up_to, errors="raise").to_pydatetime()
    else:
        dt = datetime.now(timezone.utc)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt