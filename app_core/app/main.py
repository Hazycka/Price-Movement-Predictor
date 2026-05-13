"""
Точка входа FastAPI приложения.

Содержит только сборку приложения: загрузку .env, инициализацию хранилища
и подключение роутеров. Логика эндпоинтов — в app/routers/.
"""
from fastapi import FastAPI
from dotenv import load_dotenv, find_dotenv

from .routers import health, forecast, backtest, market
from .storage import init_storage


load_dotenv(find_dotenv())
app = FastAPI(title="App Core", version="0.3.0")


@app.on_event("startup")
def on_startup() -> None:
    init_storage()


app.include_router(health.router)
app.include_router(forecast.router)
app.include_router(backtest.router)
app.include_router(market.router)
