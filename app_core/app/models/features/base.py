from abc import ABC, abstractmethod
import pandas as pd


class FeaturePlugin(ABC):
    name: str

    @abstractmethod
    def apply(self, df: pd.DataFrame, context: dict | None = None) -> pd.DataFrame:
        pass