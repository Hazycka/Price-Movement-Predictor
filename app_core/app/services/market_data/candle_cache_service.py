"""
CandleCacheService — сервис кеширования рыночных свечей в БД.

Логика работы:
  1. Конвертируем history_period + history_up_to → конкретные даты [from_dt, to_dt]
  2. Читаем покрытия из БД для (ticker, source, interval)
  3. Вычисляем непокрытые диапазоны внутри [from_dt, to_dt]
  4. Для каждого непокрытого диапазона — запрашиваем провайдера
  5а. Если провайдер вернул данные:
      - сохраняем свечи + обновляем coverage в одной транзакции (атомарно)
      - удаляем все unavailable-записи, пересекающиеся с этим диапазоном
  5б. Если провайдер вернул 0 свечей или ошибку:
      - записываем диапазон в candle_unavailable_ranges с причиной
      - продолжаем по остальным missing диапазонам
  6. После цикла, если были unavailable-диапазоны — бросаем DataUnavailableError
     с информацией о доступных периодах. Запрос прерывается на верхнем уровне.
  7. Иначе читаем финальный результат из БД и возвращаем.

Прозрачность:
  Записи в candle_unavailable_ranges НЕ являются постоянными — при следующем
  запросе того же диапазона провайдер опрашивается снова (вдруг данные появились).

Атомарность:
  upsert_candles_batch + upsert_coverage + delete_unavailable_range_overlap —
  в одном UnitOfWork блоке.
"""
from __future__ import annotations

import logging
from datetime import timezone

import pandas as pd

from .common import is_absolute_date, parse_history_period
from .exceptions import DataUnavailableError, _RangeInfo
from .factory import get_market_data_provider
from ...storage import get_uow_factory
from ...storage.ports import CandleRow
from ...schemas import ProviderOptions, TInvestProviderOptions, YahooProviderOptions

logger = logging.getLogger(__name__)


_GAP_TOLERANCE = pd.Timedelta(days=7)


def _has_significant_gap(earlier_dt: str, later_dt: str) -> bool:
    """
    True если разрыв между двумя датами больше _GAP_TOLERANCE.

    Используется для отличия нормальных пропусков (выходные, праздники)
    от реальных гэпов в данных, когда инструмент не торговался месяцами/годами.
    """
    diff = pd.to_datetime(later_dt, utc=True) - pd.to_datetime(earlier_dt, utc=True)
    return diff > _GAP_TOLERANCE


def _resolve_date_range(history_period: str, history_up_to: str | None) -> tuple[str, str]:
    """
    Конвертирует history_period + history_up_to в конкретные даты (ISO строки).

    Поддерживает два формата history_period:
      - Относительный ('1y', '6mo', '14d', '2w') — отсчитывается от to_dt
      - Абсолютный ('2021-05-22' или ISO 8601) — start берётся как есть
    """
    if history_up_to:
        to_dt = pd.to_datetime(history_up_to, utc=True)
    else:
        to_dt = pd.Timestamp.now(tz=timezone.utc)

    if is_absolute_date(history_period):
        from_dt = pd.to_datetime(history_period, utc=True)
    else:
        from_dt = to_dt - parse_history_period(history_period)

    return from_dt.isoformat(), to_dt.isoformat()


def _find_missing_ranges(
        coverage: list,
        from_dt: str,
        to_dt: str,
) -> list[tuple[str, str]]:
    """
    Находит непокрытые диапазоны внутри [from_dt, to_dt].

    Пример:
      Запрошено:  [2022-01-01 ........... 2025-01-01]
      Покрыто:    [2022-01-01 .. 2023-06-01]  [2024-01-01 .. 2025-01-01]
      Пропуск:                  [2023-06-01 .. 2024-01-01]
    """
    if not coverage:
        return [(from_dt, to_dt)]

    missing = []
    cursor = from_dt

    for cov in coverage:
        if cov.to_dt <= cursor:
            # Это покрытие уже позади курсора — пропускаем
            continue
        if cov.from_dt > cursor:
            # Дыра между курсором и началом покрытия
            gap_end = min(cov.from_dt, to_dt)
            missing.append((cursor, gap_end))
        # Двигаем курсор за конец покрытия
        cursor = max(cursor, cov.to_dt)
        if cursor >= to_dt:
            break

    # Хвост после всех покрытий
    if cursor < to_dt:
        missing.append((cursor, to_dt))

    return missing


def _source_from_provider(provider: str) -> str:
    """Нормализует название провайдера в строку источника для БД."""
    normalized = provider.strip().lower()
    if normalized in ("t_invest", "tinvest"):
        return "t_invest"
    if normalized == "yfinance":
        return "yfinance"
    return normalized


def _ticker_from_options(provider_options: ProviderOptions) -> str:
    if isinstance(provider_options, TInvestProviderOptions | YahooProviderOptions):
        return (provider_options.ticker or "").upper()
    return "unknown"


class CandleCacheService:
    """
    Сервис кеширования свечей. Стоит между провайдерами данных и сервисами прогноза.
    Провайдер остаётся чистым — он только загружает данные, не знает о кеше.
    """

    def load_ohlc(
            self,
            history_period: str,
            interval: str,
            provider_options: ProviderOptions,
            history_up_to: str | None = None,
            provider: str = "t_invest",
    ) -> tuple[list[str], list[dict[str, float]]]:

        source = _source_from_provider(provider)
        ticker = _ticker_from_options(provider_options)
        from_dt, to_dt = _resolve_date_range(history_period, history_up_to)

        logger.info(
            "[CandleCache] Запрос: ticker=%s source=%s interval=%s [%s → %s]",
            ticker, source, interval,
            from_dt[:10], to_dt[:10],
        )

        uow_factory = get_uow_factory()

        # --- Шаг 1: читаем покрытие из БД ---
        with uow_factory() as uow:
            coverage = uow.candle_repository.get_coverage(ticker, source, interval)

        logger.info(
            "[CandleCache] Покрытий в БД: %d диапазонов",
            len(coverage),
        )

        # --- Шаг 2: находим что нужно дозапросить ---
        missing = _find_missing_ranges(coverage, from_dt, to_dt)

        if not missing:
            logger.info("[CandleCache] ✓ Все данные есть в БД — провайдер не нужен")
        else:
            logger.info(
                "[CandleCache] Непокрытых диапазонов: %d — дозапрашиваем у провайдера",
                len(missing),
            )

        # --- Шаг 3: дозапрашиваем непокрытые диапазоны ---
        data_provider = get_market_data_provider(provider)
        unavailable_collected: list[_RangeInfo] = []

        for range_from, range_to in missing:
            logger.info(
                "[CandleCache] → Запрос к провайдеру: [%s → %s]",
                range_from[:10], range_to[:10],
            )

            failure_reason: str | None = None
            dates: list[str] = []
            candles: list[dict[str, float]] = []

            try:
                dates, candles = data_provider.load_ohlc_range(
                    provider_options=provider_options,
                    interval=interval,
                    from_dt=range_from,
                    to_dt=range_to,
                )
            except Exception as ex:
                failure_reason = f"Ошибка провайдера: {ex}"
                logger.error(
                    "[CandleCache] ✗ Ошибка загрузки [%s → %s]: %s",
                    range_from[:10], range_to[:10], ex,
                )

            if failure_reason is None and not candles:
                failure_reason = "Провайдер вернул 0 свечей"

            if failure_reason is not None:
                # --- Шаг 4а: весь запрошенный диапазон недоступен ---
                with uow_factory() as uow:
                    uow.candle_repository.upsert_unavailable_range(
                        ticker, source, interval, range_from, range_to, failure_reason,
                    )
                logger.warning(
                    "[CandleCache] ⚠ Помечен недоступным [%s → %s]: %s",
                    range_from[:10], range_to[:10], failure_reason,
                )
                unavailable_collected.append(
                    _RangeInfo(from_dt=range_from, to_dt=range_to, reason=failure_reason)
                )
                continue

            # --- Шаг 4б: данные есть. Считаем РЕАЛЬНОЕ покрытие по факту возврата ---
            # Провайдер мог вернуть данные не за весь запрошенный [range_from, range_to],
            # а только за часть (например, AAPL на SPBXM приостановили в 2022, данные
            # появляются только с 2024). Гэпы по краям помечаем как недоступные.
            actual_from = dates[0]
            actual_to   = dates[-1]
            pre_gap     = _has_significant_gap(range_from, actual_from)
            post_gap    = _has_significant_gap(actual_to, range_to)

            candle_rows = [
                CandleRow(
                    ticker=ticker,
                    source=source,
                    interval=interval,
                    timestamp=dates[i],
                    open=candles[i]["open"],
                    high=candles[i]["high"],
                    low=candles[i]["low"],
                    close=candles[i]["close"],
                    volume=candles[i].get("volume", 0.0),
                )
                for i in range(len(candles))
            ]

            with uow_factory() as uow:
                uow.candle_repository.upsert_candles_batch(ticker, source, interval, candle_rows)
                uow.candle_repository.upsert_coverage(
                    ticker, source, interval, actual_from, actual_to,
                )
                uow.candle_repository.delete_unavailable_range_overlap(
                    ticker, source, interval, actual_from, actual_to,
                )
                if pre_gap:
                    reason = f"Провайдер не вернул данные ранее {actual_from[:10]}"
                    uow.candle_repository.upsert_unavailable_range(
                        ticker, source, interval, range_from, actual_from, reason,
                    )
                if post_gap:
                    reason = f"Провайдер не вернул данные позднее {actual_to[:10]}"
                    uow.candle_repository.upsert_unavailable_range(
                        ticker, source, interval, actual_to, range_to, reason,
                    )

            logger.info(
                "[CandleCache] ✓ Сохранено: %d свечей, покрытие [%s → %s]",
                len(candle_rows), actual_from[:10], actual_to[:10],
            )

            if pre_gap:
                msg = f"Провайдер не вернул данные ранее {actual_from[:10]}"
                logger.warning("[CandleCache] ⚠ Гэп [%s → %s]: %s",
                               range_from[:10], actual_from[:10], msg)
                unavailable_collected.append(
                    _RangeInfo(from_dt=range_from, to_dt=actual_from, reason=msg)
                )
            if post_gap:
                msg = f"Провайдер не вернул данные позднее {actual_to[:10]}"
                logger.warning("[CandleCache] ⚠ Гэп [%s → %s]: %s",
                               actual_to[:10], range_to[:10], msg)
                unavailable_collected.append(
                    _RangeInfo(from_dt=actual_to, to_dt=range_to, reason=msg)
                )

        # --- Шаг 5: если были недоступные диапазоны — прерываем запрос ---
        if unavailable_collected:
            with uow_factory() as uow:
                final_coverage = uow.candle_repository.get_coverage(ticker, source, interval)
            available = [
                _RangeInfo(from_dt=c.from_dt, to_dt=c.to_dt) for c in final_coverage
            ]
            raise DataUnavailableError(
                ticker=ticker,
                source=source,
                interval=interval,
                unavailable=unavailable_collected,
                available=available,
            )

        # --- Шаг 6: читаем финальный результат из БД ---
        with uow_factory() as uow:
            result_rows = uow.candle_repository.get_candles(ticker, source, interval, from_dt, to_dt)

        logger.info(
            "[CandleCache] ✓ Итого свечей из БД: %d",
            len(result_rows),
        )

        if not result_rows:
            return [], []

        dates_out = [r.timestamp for r in result_rows]
        candles_out = [
            {"open": r.open, "high": r.high, "low": r.low,
             "close": r.close, "volume": r.volume}
            for r in result_rows
        ]

        return dates_out, candles_out