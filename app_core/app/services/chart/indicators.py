import pandas as pd


def _to_nullable_list(series: pd.Series) -> list[float | None]:
    result: list[float | None] = []
    for value in series.tolist():
        if pd.isna(value):
            result.append(None)
        else:
            result.append(float(value))
    return result


def calculate_indicators(values: list[float], indicators: list[str]) -> dict[str, list[float | None]]:
    series = pd.Series(values, dtype=float)
    df = pd.DataFrame({"close": series})
    result: dict[str, list[float | None]] = {}

    for indicator in indicators:
        if indicator == "sma_20":
            result[indicator] = _to_nullable_list(df["close"].rolling(window=20).mean())

        elif indicator == "ema_20":
            result[indicator] = _to_nullable_list(df["close"].ewm(span=20, adjust=False).mean())

        elif indicator == "rsi_14":
            delta = df["close"].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = (-delta.clip(upper=0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            result[indicator] = _to_nullable_list(rsi)

    return result