from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv, find_dotenv

from .schemas import (
    ForecastRequest, ForecastResponse,
    BacktestRequest, BacktestResponse,
    TInvestProviderOptions, PatchTSTModelOptions,
)
from .models.factory import create_model
from .services.forecast_service import ForecastService
from .services.market_data.market_data_query_service import MarketDataQueryService
from .storage import init_storage

load_dotenv(find_dotenv())
app = FastAPI(title="App Core", version="0.2.0")

_market_query = MarketDataQueryService()


@app.on_event("startup")
def on_startup() -> None:
    init_storage()


def _service_for_model(model_name: str | None) -> ForecastService:
    return ForecastService(model=create_model(model_name))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "default_model": create_model(None).get_info()}


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    try:
        return _service_for_model(request.model_name).run_forecast(request)
    except (ValueError, RuntimeError) as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {ex}") from ex


@app.post("/forecast/backtest", response_model=BacktestResponse)
def forecast_backtest(request: BacktestRequest) -> BacktestResponse:
    try:
        return _service_for_model(request.model_name).run_backtest(request)
    except (ValueError, RuntimeError) as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {ex}") from ex


@app.post("/forecast/chart", response_class=HTMLResponse)
def forecast_chart(request: ForecastRequest) -> HTMLResponse:
    try:
        return HTMLResponse(content=_service_for_model(request.model_name).build_chart(request))
    except (ValueError, RuntimeError) as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Chart build failed: {ex}") from ex


@app.get("/forecast/chart/test", response_class=HTMLResponse)
def forecast_chart_test(history_up_to: str | None = None) -> HTMLResponse:
    try:
        request = ForecastRequest(
            model_name="patchtst",
            data_source="t_invest",
            provider_options=TInvestProviderOptions(ticker="AAPL", class_code="SPBXM"),
            history_up_to=history_up_to,
            chart_type_history="candlestick",
            chart_type_forecast="candlestick",
            history_period="3y",
            interval="1h",
            horizon=64,
        )
        return HTMLResponse(content=_service_for_model(request.model_name).build_chart(request))
    except (ValueError, RuntimeError) as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Test chart build failed: {ex}") from ex


# ---------------------------------------------------------------------------
# Market data cache
# ---------------------------------------------------------------------------

@app.get("/market/tickers")
def market_tickers() -> dict:
    """
    Список всех инструментов с данными в локальной БД.
    Показывает покрытый диапазон дат и количество свечей по каждому.
    """
    try:
        tickers = _market_query.get_available_tickers()
        return {
            "count": len(tickers),
            "tickers": [
                {
                    "ticker":        t.ticker,
                    "source":        t.source,
                    "interval":      t.interval,
                    "from_dt":       t.from_dt,
                    "to_dt":         t.to_dt,
                    "candles_count": t.candles_count,
                }
                for t in tickers
            ],
        }
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Ошибка получения списка тикеров: {ex}") from ex


@app.get("/market/candles")
def market_candles(
        ticker:   str = Query(..., description="Тикер инструмента, например AAPL"),
        source:   str = Query(..., description="Источник: t_invest, yfinance"),
        interval: str = Query(..., description="Интервал: 1h, 4h, 1d и т.д."),
        from_dt:  str = Query(..., description="Начало диапазона ISO, например 2024-01-01"),
        to_dt:    str = Query(..., description="Конец диапазона ISO, например 2025-01-01"),
) -> dict:
    """
    Сырые свечи из локальной БД. Данные не дозапрашиваются у провайдера.
    """
    try:
        rows = _market_query.get_candles(ticker, source, interval, from_dt, to_dt)
        return {
            "ticker":        ticker.upper(),
            "source":        source,
            "interval":      interval,
            "from_dt":       from_dt,
            "to_dt":         to_dt,
            "candles_count": len(rows),
            "candles": [
                {
                    "timestamp": r.timestamp,
                    "open":  r.open, "high": r.high,
                    "low":   r.low,  "close": r.close,
                    "volume": r.volume,
                }
                for r in rows
            ],
        }
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Ошибка получения свечей: {ex}") from ex


@app.get("/market/candles/chart", response_class=HTMLResponse)
def market_candles_chart(
        ticker:   str = Query(..., description="Тикер инструмента, например AAPL"),
        source:   str = Query(..., description="Источник: t_invest, yfinance"),
        interval: str = Query(..., description="Интервал: 1h, 4h, 1d и т.д."),
        from_dt:  str = Query(..., description="Начало диапазона ISO, например 2024-01-01"),
        to_dt:    str = Query(..., description="Конец диапазона ISO, например 2025-01-01"),
) -> HTMLResponse:
    """
    HTML график свечей из локальной БД.
    Данные не дозапрашиваются — только то что есть в БД.
    """
    try:
        html = _market_query.build_candles_chart(ticker, source, interval, from_dt, to_dt)
        return HTMLResponse(content=html)
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Ошибка построения графика: {ex}") from ex