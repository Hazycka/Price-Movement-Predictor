from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv, find_dotenv

from .schemas import (
    ForecastRequest,
    ForecastResponse,
    BacktestRequest,
    BacktestResponse,
    TInvestProviderOptions,
    PatchTSTModelOptions
)
from .models.factory import create_model
from .services.forecast_service import ForecastService
from .storage import init_storage


load_dotenv(find_dotenv())
app = FastAPI(title="App Core", version="0.1.0")

@app.on_event("startup")
def on_startup() -> None:
    init_storage()

def _service_for_model(model_name: str | None) -> ForecastService:
    model = create_model(model_name)
    return ForecastService(model=model)


@app.get("/health")
def health() -> dict:
    model = create_model(None)
    return {
        "status": "ok",
        "default_model": model.get_info()
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    try:
        service = _service_for_model(request.model_name)
        return service.run_forecast(request)
    except (ValueError, RuntimeError) as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {ex}") from ex


@app.post("/forecast/backtest", response_model=BacktestResponse)
def forecast_backtest(request: BacktestRequest) -> BacktestResponse:
    try:
        service = _service_for_model(request.model_name)
        return service.run_backtest(request)
    except (ValueError, RuntimeError) as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {ex}") from ex


@app.post("/forecast/chart", response_class=HTMLResponse)
def forecast_chart(request: ForecastRequest) -> HTMLResponse:
    try:
        service = _service_for_model(request.model_name)
        html = service.build_chart(request)
        return HTMLResponse(content=html)
    except (ValueError, RuntimeError) as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Chart build failed: {ex}") from ex


# дата формата 2026-03-20
@app.get("/forecast/chart/test", response_class=HTMLResponse)
def forecast_chart_test(history_up_to: str | None = None) -> HTMLResponse:
    try:
        request = ForecastRequest(
            model_name="patchtst",
            data_source="t_invest",
            provider_options=TInvestProviderOptions(
                ticker="AAPL",
                class_code="SPBXM"
                # token="..." лучше через env TINVEST_TOKEN
            ),
            history_up_to=history_up_to,
            chart_type_history="candlestick",
            chart_type_forecast="candlestick",
            history_period="3y",
            interval="1h",
            horizon=10,
            num_samples=64
        )
        service = _service_for_model(request.model_name)
        html = service.build_chart(request)
        return HTMLResponse(content=html)
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Test chart build failed: {ex}") from ex