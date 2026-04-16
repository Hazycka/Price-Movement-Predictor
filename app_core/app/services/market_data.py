import os

import pandas as pd
import yfinance as yf


def _normalize_ohlc_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        else:
            data.columns = data.columns.get_level_values(0)

    data.columns = [str(column) for column in data.columns]
    return data


def _parse_history_period(period: str) -> pd.DateOffset:
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


def _resolve_history_window(history_period: str, history_up_to: str | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if not history_up_to:
        return None, None

    end = pd.to_datetime(history_up_to, errors="raise")
    start = end - _parse_history_period(history_period)
    return start, end


def load_ohlc_from_ticker(
        ticker: str,
        history_period: str,
        interval: str,
        history_up_to: str | None = None
) -> tuple[list[str], list[dict[str, float]]]:
    start, end = _resolve_history_window(history_period, history_up_to)

    data = yf.download(
        tickers=ticker,
        period=None if start else history_period,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(f"Не удалось загрузить данные по тикеру '{ticker}'.")

    data = _normalize_ohlc_columns(data, ticker.upper())

    required_columns = ["Open", "High", "Low", "Close"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            f"В данных по тикеру '{ticker}' отсутствуют колонки: {missing_columns}. "
            f"Фактические колонки: {list(data.columns)}"
        )

    data = data.dropna(subset=required_columns)

    if len(data) < 10:
        raise ValueError("Слишком мало данных для прогнозирования.")

    dates = [str(idx) for idx in data.index.tolist()]
    candles: list[dict[str, float]] = []

    for _, row in data.iterrows():
        candle = {
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"])
        }
        if "Volume" in data.columns and not pd.isna(row["Volume"]):
            candle["volume"] = float(row["Volume"])
        else:
            candle["volume"] = 0.0
        candles.append(candle)

    return dates, candles


def load_ohlc_from_csv(
        csv_path: str,
        date_column: str = "Date",
        open_column: str = "Open",
        high_column: str = "High",
        low_column: str = "Low",
        close_column: str = "Close",
        volume_column: str | None = "Volume"
) -> tuple[list[str], list[dict[str, float]]]:
    normalized_path = csv_path.strip().strip('"').strip("'")

    if not os.path.isabs(normalized_path):
        normalized_path = os.path.join("data", normalized_path)

    if not os.path.exists(normalized_path):
        raise ValueError(f"CSV файл не найден: {normalized_path}")

    data = pd.read_csv(normalized_path)

    required_columns = [date_column, open_column, high_column, low_column, close_column]
    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"В CSV отсутствует колонка '{column}'.")

    data = data[required_columns + ([volume_column] if volume_column and volume_column in data.columns else [])].dropna()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    for column in [open_column, high_column, low_column, close_column]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if volume_column and volume_column in data.columns:
        data[volume_column] = pd.to_numeric(data[volume_column], errors="coerce")

    data = data.dropna()

    if data.empty:
        raise ValueError("CSV не содержит валидных данных после очистки.")

    if len(data) < 10:
        raise ValueError("Слишком мало данных для прогнозирования.")

    dates = [str(v) for v in data[date_column].tolist()]
    candles: list[dict[str, float]] = []

    for _, row in data.iterrows():
        candle = {
            "open": float(row[open_column]),
            "high": float(row[high_column]),
            "low": float(row[low_column]),
            "close": float(row[close_column]),
            "volume": float(row[volume_column]) if volume_column and volume_column in data.columns else 0.0
        }
        candles.append(candle)

    return dates, candles