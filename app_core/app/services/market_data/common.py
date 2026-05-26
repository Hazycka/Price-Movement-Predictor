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


def is_absolute_date(s: str) -> bool:
    """
    True если строка похожа на абсолютную дату (YYYY-MM-DD или ISO 8601).
    Эвристика: содержит '-' (отделитель в YYYY-MM-DD).
    Относительные периоды ('1y', '6mo', '14d', '2w') символ '-' не содержат.
    """
    return "-" in s


def parse_history_period(period: str) -> pd.DateOffset:
    """
    Парсит ОТНОСИТЕЛЬНЫЙ период ('1y', '6mo', '14d', '2w') в pandas DateOffset.

    Для абсолютной даты использовать is_absolute_date()/resolve_history_window().
    """
    if is_absolute_date(period):
        raise ValueError(
            f"parse_history_period получил абсолютную дату '{period}'. "
            f"Используй resolve_history_window() для смешанной логики."
        )

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
    """
    Резолвит history_period + history_up_to в (start, end).

    Поддерживает два формата history_period:
      - Относительный: '1y', '6mo', '14d', '2w' — отсчитывается от history_up_to
        (или возвращает (None, None) если history_up_to не задан — провайдер
        сам решит)
      - Абсолютный: '2021-05-22' (YYYY-MM-DD) или полная ISO 8601 — start =
        указанная дата. end = history_up_to или текущий момент UTC.
    """
    end_dt = pd.to_datetime(history_up_to, errors="raise") if history_up_to else None

    if is_absolute_date(history_period):
        start_dt = pd.to_datetime(history_period, errors="raise")
        # Для абсолютной даты end должен существовать всегда — иначе диапазон
        # неопределён. Берём now() в UTC.
        if end_dt is None:
            end_dt = pd.Timestamp.now(tz=timezone.utc)
        return start_dt, end_dt

    # Относительный путь: без history_up_to оставляем как было — провайдер решает
    if end_dt is None:
        return None, None

    return end_dt - parse_history_period(history_period), end_dt


def resolve_end_dt(history_up_to: str | None) -> datetime:
    if history_up_to:
        dt = pd.to_datetime(history_up_to, errors="raise").to_pydatetime()
    else:
        dt = datetime.now(timezone.utc)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt