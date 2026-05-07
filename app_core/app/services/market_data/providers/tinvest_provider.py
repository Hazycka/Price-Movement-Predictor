from __future__ import annotations

import logging
import math
import os
from datetime import timedelta, timezone

import pandas as pd

from ..base import MarketDataProvider, MarketDataRequest
from ..common import parse_history_period, resolve_end_dt
from ....schemas import TInvestProviderOptions, ProviderOptions
from ....storage import get_uow_factory

logger = logging.getLogger(__name__)


def _quotation_to_float(value) -> float:
    return float(value.units) + float(value.nano) / 1_000_000_000.0


class TInvestMarketDataProvider(MarketDataProvider):
    INTERVAL_MAP = {
        "1min": "CANDLE_INTERVAL_1_MIN", "2min": "CANDLE_INTERVAL_2_MIN",
        "3min": "CANDLE_INTERVAL_3_MIN", "5min": "CANDLE_INTERVAL_5_MIN",
        "10min": "CANDLE_INTERVAL_10_MIN", "15min": "CANDLE_INTERVAL_15_MIN",
        "30min": "CANDLE_INTERVAL_30_MIN", "1h": "CANDLE_INTERVAL_HOUR",
        "2h": "CANDLE_INTERVAL_2_HOUR", "4h": "CANDLE_INTERVAL_4_HOUR",
        "1d": "CANDLE_INTERVAL_DAY", "1w": "CANDLE_INTERVAL_WEEK",
        "1mo": "CANDLE_INTERVAL_MONTH",
    }

    MAX_CHUNK_DAYS = {
        "1min": 1, "2min": 2, "3min": 3, "5min": 7, "10min": 14,
        "15min": 30, "30min": 60, "1h": 180, "2h": 365, "4h": 365,
        "1d": 2000, "1w": 4000, "1mo": 6000,
    }

    PROVIDER_NAME = "t_invest"
    CHUNK_DAYS_STATE_PREFIX = "chunk_days:"

    def __init__(self) -> None:
        self._chunk_days_cache: dict[str, int] = dict(self.MAX_CHUNK_DAYS)

    @staticmethod
    def _resolve_figi(client, provider_options: TInvestProviderOptions) -> str:
        ticker = provider_options.ticker
        figi = provider_options.figi
        class_code = provider_options.class_code
        if figi:
            return figi
        if not ticker:
            raise ValueError("Для T-Invest требуется ticker или figi.")
        response = client.instruments.find_instrument(query=ticker)
        candidates = [i for i in response.instruments if i.ticker.upper() == ticker.upper()]
        if class_code:
            candidates = [i for i in candidates if i.class_code == class_code]
        if not candidates:
            raise ValueError(f"Инструмент не найден: ticker={ticker}, class_code={class_code}")
        return candidates[0].figi

    @staticmethod
    def _is_max_period_error(ex: Exception) -> bool:
        msg = str(ex).lower()
        return "30014" in msg and "maximum request period" in msg

    def _get_candles_adaptive(self, client, figi, interval_enum, from_dt, to_dt,
                              min_chunk=timedelta(hours=1)) -> tuple[list, int | None]:
        candles = []
        stack = [(from_dt, to_dt)]
        max_success_span_days = 0.0

        while stack:
            left, right = stack.pop()
            if left >= right:
                continue
            try:
                resp = client.market_data.get_candles(
                    figi=figi, from_=left, to=right, interval=interval_enum,
                )
                candles.extend(resp.candles)
                span_days = (right - left).total_seconds() / 86400.0
                max_success_span_days = max(max_success_span_days, span_days)
            except Exception as ex:
                if not self._is_max_period_error(ex):
                    raise
                span = right - left
                if span <= min_chunk:
                    raise RuntimeError(
                        f"T-Invest ограничил период даже для минимального чанка "
                        f"{left.isoformat()}..{right.isoformat()}: {ex}"
                    ) from ex
                mid = left + span / 2
                if mid <= left or mid >= right:
                    raise RuntimeError(
                        f"Не удалось разделить диапазон "
                        f"{left.isoformat()}..{right.isoformat()}: {ex}"
                    ) from ex
                logger.warning(
                    "[T-Invest] ⚠ Период слишком большой для одного запроса (30014) "
                    "[%s → %s] — делим чанк пополам и повторяем. "
                    "Сервис продолжает загрузку меньшими кусками.",
                    left.strftime("%Y-%m-%d"), right.strftime("%Y-%m-%d"),
                )
                stack.append((mid, right))
                stack.append((left, mid))

        suggested_days = max(1, int(math.floor(max_success_span_days))) if max_success_span_days > 0 else None
        return candles, suggested_days

    def _get_cached_chunk_days(self, interval: str) -> int:
        in_mem = self._chunk_days_cache.get(interval)
        if in_mem is not None:
            return in_mem
        try:
            uow_factory = get_uow_factory()
            with uow_factory() as uow:
                state = uow.provider_state.get_state(
                    self.PROVIDER_NAME, f"{self.CHUNK_DAYS_STATE_PREFIX}{interval}"
                )
            if state:
                value = max(1, int(state.get("value")))
                self._chunk_days_cache[interval] = value
                return value
        except Exception:
            pass
        fallback = self.MAX_CHUNK_DAYS[interval]
        self._chunk_days_cache[interval] = fallback
        return fallback

    def _set_cached_chunk_days(self, interval: str, days: int) -> None:
        safe = max(1, int(days))
        self._chunk_days_cache[interval] = safe
        try:
            uow_factory = get_uow_factory()
            with uow_factory() as uow:
                uow.provider_state.set_state(
                    self.PROVIDER_NAME,
                    f"{self.CHUNK_DAYS_STATE_PREFIX}{interval}",
                    {"value": safe},
                )
        except Exception:
            pass

    def _load_range(self, provider_options: TInvestProviderOptions,
                    interval: str, start_dt, end_dt) -> tuple[list[str], list[dict]]:
        token = provider_options.token or os.getenv("TINVEST_TOKEN")
        if not token:
            raise ValueError("Не задан токен T-Invest. Используйте env TINVEST_TOKEN.")
        if interval not in self.INTERVAL_MAP:
            raise ValueError(f"Интервал '{interval}' не поддерживается T-Invest.")

        from t_tech.invest import Client, CandleInterval
        interval_enum = getattr(CandleInterval, self.INTERVAL_MAP[interval])

        total_days = max(1, int((end_dt - start_dt).total_seconds() / 86400))
        chunk_days = self._get_cached_chunk_days(interval)
        total_chunks_est = max(1, math.ceil(total_days / chunk_days))

        logger.info(
            "[T-Invest] Начало загрузки: interval=%s [%s → %s] "
            "~%d дней, чанк=%d дней, ~%d запросов",
            interval, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"),
            total_days, chunk_days, total_chunks_est,
        )

        all_candles = []
        chunks_done = 0

        with Client(token) as client:
            figi = self._resolve_figi(client, provider_options)
            logger.info("[T-Invest] FIGI определён: %s", figi)

            cursor = start_dt
            while cursor < end_dt:
                to_dt_chunk = min(cursor + timedelta(days=chunk_days), end_dt)
                chunk_candles, suggested_days = self._get_candles_adaptive(
                    client=client, figi=figi, interval_enum=interval_enum,
                    from_dt=cursor, to_dt=to_dt_chunk,
                )
                all_candles.extend(chunk_candles)
                chunks_done += 1

                if suggested_days is not None and suggested_days < chunk_days:
                    logger.info(
                        "[T-Invest] → Адаптируем размер чанка: %d → %d дней",
                        chunk_days, suggested_days,
                    )
                    chunk_days = suggested_days
                    self._set_cached_chunk_days(interval, suggested_days)

                elapsed = (to_dt_chunk - start_dt).total_seconds() / 86400
                pct = min(100, int(elapsed / total_days * 100))
                logger.info(
                    "[T-Invest] Прогресс: %3d%% | чанк %d [%s → %s] | свечей: %d",
                    pct, chunks_done,
                    cursor.strftime("%Y-%m-%d"), to_dt_chunk.strftime("%Y-%m-%d"),
                    len(all_candles),
                )
                cursor = to_dt_chunk

        logger.info("[T-Invest] ✓ Загрузка завершена: %d сырых свечей", len(all_candles))

        if not all_candles:
            return [], []

        rows = [{
            "time": c.time, "open": _quotation_to_float(c.open),
            "high": _quotation_to_float(c.high), "low": _quotation_to_float(c.low),
            "close": _quotation_to_float(c.close), "volume": float(c.volume),
        } for c in all_candles]

        data = (
            pd.DataFrame(rows)
            .sort_values("time")
            .drop_duplicates(subset=["time"], keep="last")
            .dropna(subset=["open", "high", "low", "close"])
        )

        dates = [str(v) for v in data["time"].tolist()]
        candles = data[["open", "high", "low", "close", "volume"]].to_dict(orient="records")
        logger.info("[T-Invest] ✓ После очистки: %d свечей", len(candles))
        return dates, candles

    def load_ohlc(self, request: MarketDataRequest) -> tuple[list[str], list[dict]]:
        options = request.provider_options
        if not isinstance(options, TInvestProviderOptions):
            raise TypeError(f"Ожидались настройки T-Invest, но получено {type(options).__name__}")
        end_dt = resolve_end_dt(request.history_up_to)
        start_dt = (pd.Timestamp(end_dt) - parse_history_period(request.history_period)).to_pydatetime()
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        return self._load_range(options, request.interval, start_dt, end_dt)

    def load_ohlc_range(self, provider_options: ProviderOptions, interval: str,
                        from_dt: str, to_dt: str) -> tuple[list[str], list[dict]]:
        if not isinstance(provider_options, TInvestProviderOptions):
            raise TypeError(f"Ожидались настройки T-Invest, но получено {type(provider_options).__name__}")
        start_dt = pd.to_datetime(from_dt, utc=True).to_pydatetime()
        end_dt   = pd.to_datetime(to_dt,   utc=True).to_pydatetime()
        return self._load_range(provider_options, interval, start_dt, end_dt)