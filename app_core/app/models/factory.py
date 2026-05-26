import os
import threading

from .base import ForecastModel
from .chronos_model import ChronosForecastModel
from .patchtst import PatchTSTForecastModel
from .moirai import MoiraiForecastModel


def _resolve_name(model_name: str | None) -> str:
    return (model_name or os.getenv("FORECAST_MODEL", "patchtst")).strip().lower()


def create_model(model_name: str | None = None) -> ForecastModel:
    """
    Создаёт НОВЫЙ инстанс модели каждый вызов.

    Когда использовать: для training. Тренировка модифицирует веса (head/LoRA),
    нельзя возвращать общий синглтон — это бы корраптило кеш для inference.
    """
    name = _resolve_name(model_name)

    if name == "chronos":
        return ChronosForecastModel()

    if name == "patchtst":
        return PatchTSTForecastModel()

    if name in ("moirai_2", "moirai", "moirai2"):
        # Все три варианта дают одну модель — moirai-2 (актуальная версия).
        # Если в будущем понадобится moirai-1 — добавить отдельный alias.
        return MoiraiForecastModel()

    raise ValueError(
        f"Неизвестная модель '{name}'. "
        f"Поддерживается: chronos, patchtst, moirai_2"
    )


# ---------------------------------------------------------------------------
# Inference singleton
#
# Для путей которые НЕ модифицируют веса модели (/forecast, /backtest) —
# держим один загруженный инстанс на каждое имя модели и переиспользуем.
# Раньше create_model() вызывался на каждый запрос — это означало повторную
# инициализацию веса (быстрая операция при холодном кэше торча, но всё равно
# выделение нового CUDA-буфера, лишний GC и сериализация инстансов модели).
#
# Тренинг по-прежнему вызывает create_model() напрямую и получает свежую
# модель — чтобы случайно не записать тренируемые градиенты в синглтон.
# ---------------------------------------------------------------------------

_INFERENCE_MODELS: dict[str, ForecastModel] = {}
_INFERENCE_LOCK = threading.Lock()


def get_inference_model(model_name: str | None = None) -> ForecastModel:
    """
    Возвращает синглтон модели для inference-путей.

    Thread-safe (FastAPI BackgroundTasks работает в thread pool, /forecast тоже
    может попадать в несколько worker thread'ов uvicorn).
    """
    name = _resolve_name(model_name)
    cached = _INFERENCE_MODELS.get(name)
    if cached is not None:
        return cached
    with _INFERENCE_LOCK:
        # double-check после захвата лока
        cached = _INFERENCE_MODELS.get(name)
        if cached is not None:
            return cached
        model = create_model(name)
        _INFERENCE_MODELS[name] = model
        return model