"""
Кеш рыночных данных.
"""
from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from ..services.market_data.market_data_query_service import MarketDataQueryService
from ._common import endpoint_errors

router = APIRouter(prefix="/market", tags=["market"])

_market_query = MarketDataQueryService()


@router.get("/tickers")
@endpoint_errors("Ошибка получения списка тикеров")
def market_tickers() -> dict:
    """
    Список всех инструментов в локальной БД.
    coverage_periods    — фрагменты покрытия (могут быть гэпы между ними);
    unavailable_periods — диапазоны куда ходили и получили пустоту;
    candles_count       — общее число свечей в market_candles по ключу.
    """
    tickers = _market_query.get_available_tickers()
    return {
        "count": len(tickers),
        "tickers": [
            {
                "ticker":        t.ticker,
                "source":        t.source,
                "interval":      t.interval,
                "candles_count": t.candles_count,
                "coverage_periods": [
                    {"from_dt": c.from_dt, "to_dt": c.to_dt}
                    for c in t.coverage_periods
                ],
                "unavailable_periods": [
                    {
                        "from_dt":     u.from_dt,
                        "to_dt":       u.to_dt,
                        "reason":      u.reason,
                        "recorded_at": u.recorded_at,
                    }
                    for u in t.unavailable_periods
                ],
            }
            for t in tickers
        ],
    }


@router.get("/candles")
@endpoint_errors("Ошибка получения свечей")
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


@router.get("/candles/chart", response_class=HTMLResponse)
@endpoint_errors("Ошибка построения графика")
def market_candles_chart(
        ticker:   str = Query(..., description="Тикер инструмента, например AAPL"),
        source:   str = Query(..., description="Источник: t_invest, yfinance"),
        interval: str = Query(..., description="Интервал: 1h, 4h, 1d и т.д."),
        from_dt:  str = Query(..., description="Начало диапазона ISO"),
        to_dt:    str = Query(..., description="Конец диапазона ISO"),
) -> HTMLResponse:
    """
    HTML график свечей из локальной БД. Данные не дозапрашиваются.
    """
    html = _market_query.build_candles_chart(ticker, source, interval, from_dt, to_dt)
    return HTMLResponse(content=html)
