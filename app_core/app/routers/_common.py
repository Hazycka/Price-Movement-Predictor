"""
Общие хелперы для роутеров.

Главная утилита: декоратор @endpoint_errors который снимает дублирующийся
try/except паттерн со всех POST/GET ручек:
  - DataUnavailableError → HTTP 422 со структурированным телом
  - ValueError/RuntimeError → HTTP 400 (валидация / ожидаемая ошибка)
  - любая другая → HTTP 500 с префиксом
"""
from __future__ import annotations

from functools import wraps
from typing import Callable

from fastapi import HTTPException

from ..models.factory import create_model
from ..services.forecast_service import ForecastService
from ..services.market_data.exceptions import DataUnavailableError


def service_for_model(model_name: str | None) -> ForecastService:
    """Создаёт ForecastService с моделью по имени (или дефолтной если None)."""
    return ForecastService(model=create_model(model_name))


def data_unavailable_to_http(ex: DataUnavailableError) -> HTTPException:
    """
    Превращает DataUnavailableError в HTTP 422 со структурированным телом:
      detail.error          — машинный код ошибки
      detail.message        — человекочитаемое описание
      detail.ticker / source / interval
      detail.unavailable    — список диапазонов, за которые провайдер не вернул данных
      detail.available      — список диапазонов, по которым данные есть в БД
    """
    return HTTPException(
        status_code=422,
        detail={
            "error":    "data_unavailable",
            "message":  str(ex),
            "ticker":   ex.ticker,
            "source":   ex.source,
            "interval": ex.interval,
            "unavailable": [
                {"from_dt": r.from_dt, "to_dt": r.to_dt, "reason": r.reason}
                for r in ex.unavailable
            ],
            "available": [
                {"from_dt": r.from_dt, "to_dt": r.to_dt}
                for r in ex.available
            ],
        },
    )


def endpoint_errors(error_prefix: str) -> Callable:
    """
    Декоратор для FastAPI endpoints, который снимает повторяющийся try/except.

    error_prefix — префикс для HTTP 500 сообщения, например "Forecast failed".

    Порядок обработки:
      DataUnavailableError → HTTP 422 (структурированный JSON)
      ValueError / RuntimeError → HTTP 400 (детали как str)
      HTTPException → пробрасывается как есть (для случаев когда endpoint
                      сам кидает 404/etc до того как декоратор увидел исключение)
      Exception → HTTP 500 с префиксом

    Использование:
        @router.post("/forecast")
        @endpoint_errors("Forecast failed")
        def forecast(request: ForecastRequest) -> ForecastResponse:
            return service_for_model(request.model_name).run_forecast(request)
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except DataUnavailableError as ex:
                raise data_unavailable_to_http(ex) from ex
            except HTTPException:
                raise
            except (ValueError, RuntimeError) as ex:
                raise HTTPException(status_code=400, detail=str(ex)) from ex
            except Exception as ex:
                raise HTTPException(status_code=500, detail=f"{error_prefix}: {ex}") from ex
        return wrapper
    return decorator
