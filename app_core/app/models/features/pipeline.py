from dataclasses import dataclass
import pandas as pd

from .base import FeaturePlugin
from .plugins import (
    OhlcvBasePlugin,
    ReturnsVolatilityPlugin,
    VolumeFeaturesPlugin,
    TechnicalIndicatorsPlugin,
    CalendarRegimePlugin,
)


@dataclass
class FeaturePipelineResult:
    df: pd.DataFrame
    feature_columns: list[str]
    plugins_used: list[str]


class FeaturePipeline:
    def __init__(self) -> None:
        self._registry: dict[str, FeaturePlugin] = {
            "ohlcv_base": OhlcvBasePlugin(),
            "returns_volatility": ReturnsVolatilityPlugin(),
            "volume_features": VolumeFeaturesPlugin(),
            "technical_indicators": TechnicalIndicatorsPlugin(),
            "calendar_regime": CalendarRegimePlugin(),
        }
        self._default_order = [
            "ohlcv_base",
            "returns_volatility",
            "volume_features",
            "technical_indicators",
            "calendar_regime",
        ]

    def build(self, candles: list[dict[str, float]], context: dict | None = None) -> FeaturePipelineResult:
        df = pd.DataFrame(candles).copy()
        requested = (context or {}).get("feature_plugins") or self._default_order

        plugins_used: list[str] = []
        for name in requested:
            plugin = self._registry.get(name)
            if plugin is None:
                continue
            df = plugin.apply(df, context=context)
            plugins_used.append(name)

        feature_columns = [c for c in df.columns if c not in {"date", "timestamp"}]
        return FeaturePipelineResult(df=df, feature_columns=feature_columns, plugins_used=plugins_used)