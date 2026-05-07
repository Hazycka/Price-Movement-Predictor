import os

from .base import ForecastModel
from .chronos_model import ChronosForecastModel
from .patchtst import PatchTSTForecastModel


def create_model(model_name: str | None = None) -> ForecastModel:
    name = (model_name or os.getenv("FORECAST_MODEL", "patchtst")).strip().lower()

    if name == "chronos":
        return ChronosForecastModel()

    if name == "patchtst":
        return PatchTSTForecastModel()

    raise ValueError(
        f"Неизвестная модель '{name}'. "
        f"Поддерживается: chronos, patchtst"
    )