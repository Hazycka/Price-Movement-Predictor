"""
Прогноз / графики прогноза.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from ..schemas import ForecastRequest, ForecastResponse, TInvestProviderOptions
from ._common import service_for_model, endpoint_errors

router = APIRouter(tags=["forecast"])


@router.post("/forecast", response_model=ForecastResponse)
@endpoint_errors("Forecast failed")
def forecast(request: ForecastRequest) -> ForecastResponse:
    return service_for_model(request.model_name).run_forecast(request)


@router.post("/forecast/chart", response_class=HTMLResponse)
@endpoint_errors("Chart build failed")
def forecast_chart(request: ForecastRequest) -> HTMLResponse:
    return HTMLResponse(content=service_for_model(request.model_name).build_chart(request))


@router.get("/forecast/chart/test", response_class=HTMLResponse)
@endpoint_errors("Test chart build failed")
def forecast_chart_test(
        history_up_to: str | None = None,
        history_period: str = "2y",
) -> HTMLResponse:
    """
    Утилита для быстрой визуальной проверки конфигурации: SBER через T-Invest,
    1h интервал, horizon=64. Удобно открывать в браузере для дебага рендерера.
    """
    request = ForecastRequest(
        model_name="patchtst",
        data_source="t_invest",
        provider_options=TInvestProviderOptions(ticker="SBER", figi="BBG004730N88"),
        history_up_to=history_up_to,
        chart_type_history="candlestick",
        chart_type_forecast="candlestick",
        history_period=history_period,
        interval="1h",
        horizon=64,
    )
    return HTMLResponse(content=service_for_model(request.model_name).build_chart(request))
