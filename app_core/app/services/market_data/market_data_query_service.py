"""
MarketDataQueryService — сервис запросов к кешированным рыночным данным.

Отвечает за чтение данных из локальной БД и построение графиков.
Не загружает данные у провайдеров — только читает что уже есть.
Для загрузки используется CandleCacheService через ForecastService.
"""
from __future__ import annotations

import logging

from ...storage import get_uow_factory
from ...storage.ports import CandleRow, TickerInfo
from ..chart.plotly_renderer import build_forecast_chart_html

logger = logging.getLogger(__name__)


class MarketDataQueryService:

    @staticmethod
    def get_available_tickers() -> list[TickerInfo]:
        """
        Возвращает список всех инструментов с данными в БД.
        """
        with get_uow_factory()() as uow:
            return uow.candle_repository.get_available_tickers()

    @staticmethod
    def get_candles(
            ticker: str,
            source: str,
            interval: str,
            from_dt: str,
            to_dt: str,
    ) -> list[CandleRow]:
        """
        Возвращает свечи из БД за указанный диапазон.
        Не дозапрашивает у провайдера.
        """
        with get_uow_factory()() as uow:
            rows = uow.candle_repository.get_candles(
                ticker.upper(), source, interval, from_dt, to_dt
            )

        logger.info(
            "[MarketDataQuery] get_candles: ticker=%s source=%s interval=%s "
            "[%s → %s] → %d свечей",
            ticker.upper(), source, interval,
            from_dt[:10], to_dt[:10], len(rows),
        )
        return rows

    @staticmethod
    def build_candles_chart(
            ticker: str,
            source: str,
            interval: str,
            from_dt: str,
            to_dt: str,
    ) -> str:
        """
        Строит HTML график свечей из БД.
        Raises ValueError если данных нет.
        """
        with get_uow_factory()() as uow:
            rows = uow.candle_repository.get_candles(
                ticker.upper(), source, interval, from_dt, to_dt
            )

        if not rows:
            raise ValueError(
                f"Нет данных для {ticker.upper()} [{source}] {interval} "
                f"[{from_dt[:10]} → {to_dt[:10]}]. "
                f"Сначала выполните прогнозный запрос чтобы загрузить данные."
            )

        logger.info(
            "[MarketDataQuery] build_candles_chart: ticker=%s interval=%s → %d свечей",
            ticker.upper(), interval, len(rows),
        )

        dates = [r.timestamp for r in rows]
        candles = [
            {"open": r.open, "high": r.high, "low": r.low,
             "close": r.close, "volume": r.volume}
            for r in rows
        ]

        return build_forecast_chart_html(
            title=f"{ticker.upper()} | {source} | {interval} | {from_dt[:10]} → {to_dt[:10]}",
            labels=dates,
            candles=candles,
            forecast_candles=[],
            forecast_ohlc_quantiles=None,
            indicators={},
            chart_type_history="candlestick",
            chart_type_forecast="line",
            interval=interval,
        )