"""
Точка входа FastAPI приложения.

Содержит только сборку приложения: загрузку .env, инициализацию хранилища,
подключение роутеров и lifecycle (startup/shutdown). Логика эндпоинтов —
в app/routers/.
"""
from fastapi import FastAPI
from dotenv import load_dotenv, find_dotenv

from .routers import health, forecast, backtest, market, training
from .services.backtest.parallel_pool import shutdown_pool
from .storage import init_storage


load_dotenv(find_dotenv())
app = FastAPI(title="App Core", version="0.3.0")


@app.on_event("startup")
def on_startup() -> None:
    init_storage()


@app.on_event("shutdown")
def on_shutdown() -> None:
    # Останавливаем worker-процессы parallel sweep пула если он был создан.
    # No-op если пул не активировался (parallel_workers=1 во всех запросах).
    shutdown_pool()


app.include_router(health.router)
app.include_router(forecast.router)
app.include_router(backtest.router)
app.include_router(market.router)
app.include_router(training.router)
