import numpy as np
import pandas as pd

from .base import FeaturePlugin


class OhlcvBasePlugin(FeaturePlugin):
    name = "ohlcv_base"

    def apply(self, df: pd.DataFrame, context: dict | None = None) -> pd.DataFrame:
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df


class ReturnsVolatilityPlugin(FeaturePlugin):
    name = "returns_volatility"

    def apply(self, df: pd.DataFrame, context: dict | None = None) -> pd.DataFrame:
        df["log_return"] = np.log(df["close"].replace(0, np.nan)).diff().fillna(0.0)
        df["rolling_vol_5"] = df["log_return"].rolling(5).std().fillna(0.0)
        df["rolling_vol_20"] = df["log_return"].rolling(20).std().fillna(0.0)
        return df


class VolumeFeaturesPlugin(FeaturePlugin):
    name = "volume_features"

    def apply(self, df: pd.DataFrame, context: dict | None = None) -> pd.DataFrame:
        df["volume_ratio_20"] = df["volume"] / (df["volume"].rolling(20).mean().replace(0, np.nan))
        df["volume_ratio_20"] = df["volume_ratio_20"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

        df["volume_zscore_20"] = (
                (df["volume"] - df["volume"].rolling(20).mean()) /
                df["volume"].rolling(20).std().replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df


class TechnicalIndicatorsPlugin(FeaturePlugin):
    name = "technical_indicators"

    def apply(self, df: pd.DataFrame, context: dict | None = None) -> pd.DataFrame:
        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()
        df["ema_spread_12_26"] = (ema_fast - ema_slow).fillna(0.0)

        delta = df["close"].diff().fillna(0.0)
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

        prev_close = df["close"].shift(1).fillna(df["close"])
        tr = pd.concat([
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean().fillna(0.0)

        ma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        upper = ma20 + 2.0 * std20
        lower = ma20 - 2.0 * std20
        df["bollinger_width_20"] = ((upper - lower) / ma20.replace(0, np.nan)).fillna(0.0)

        roll_max_20 = df["high"].rolling(20).max()
        roll_min_20 = df["low"].rolling(20).min()
        df["dist_to_local_max_20"] = ((roll_max_20 - df["close"]) / df["close"].replace(0, np.nan)).fillna(0.0)
        df["dist_to_local_min_20"] = ((df["close"] - roll_min_20) / df["close"].replace(0, np.nan)).fillna(0.0)
        return df


class CalendarRegimePlugin(FeaturePlugin):
    name = "calendar_regime"

    def apply(self, df: pd.DataFrame, context: dict | None = None) -> pd.DataFrame:
        idx = pd.RangeIndex(start=0, stop=len(df), step=1)
        df["day_of_week"] = (idx % 5).astype(float)
        df["month"] = ((idx // 21) % 12 + 1).astype(float)
        df["quarter"] = (((df["month"] - 1) // 3) + 1).astype(float)

        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()
        trend = (ema_fast > ema_slow).astype(float)

        high_vol = (df["rolling_vol_20"] > df["rolling_vol_20"].rolling(60).median().fillna(df["rolling_vol_20"])).astype(float)
        range_flag = (1.0 - trend).clip(lower=0.0)

        df["regime_trend"] = trend
        df["regime_range"] = range_flag
        df["regime_high_vol"] = high_vol
        return df