"""
Метрики бэктеста.

Все метрики ВЗВЕШЕННЫЕ: на горизонте H каждой точке i присваивается вес w_i,
и метрика считается как взвешенное среднее ошибок/индикаторов.
При weights=[1, 1, ..., 1] (uniform) поведение идентично невзвешенному среднему.

Группы метрик:
  point         — MAE, RMSE, MAPE (на одном канале — например close)
  point OHLC    — те же по каждому каналу + агрегаты
  quantile      — pinball, coverage, winkler, directional_acc
  naive         — last-value baseline
"""
from __future__ import annotations

import math


class BacktestMetrics:

    # ------------------------------------------------------------------
    # Примитивы
    # ------------------------------------------------------------------

    @staticmethod
    def _pinball(y_true: float, y_pred: float, q: float) -> float:
        err = y_true - y_pred
        return max(q * err, (q - 1) * err)

    @staticmethod
    def _wmean(values: list[float], weights: list[float]) -> float:
        if not values:
            return 0.0
        total_w = sum(weights[: len(values)])
        if total_w == 0:
            return 0.0
        return sum(v * w for v, w in zip(values, weights)) / total_w

    @staticmethod
    def _wcoverage(actual: list[float], lower: list[float], upper: list[float], weights: list[float]) -> float:
        """Взвешенная доля точек, попавших в [lower_i, upper_i]."""
        if not actual:
            return 0.0
        total_w = sum(weights[: len(actual)])
        if total_w == 0:
            return 0.0
        hit_w = sum(
            w for y, lo, hi, w in zip(actual, lower, upper, weights)
            if lo <= y <= hi
        )
        return hit_w / total_w

    @staticmethod
    def _wwinkler(actual: list[float], lower: list[float], upper: list[float],
                  alpha: float, weights: list[float]) -> float:
        """Взвешенный Winkler score. Меньше — лучше."""
        if not actual:
            return 0.0
        total_w = sum(weights[: len(actual)])
        if total_w == 0:
            return 0.0
        score = 0.0
        for y, lo, hi, w in zip(actual, lower, upper, weights):
            width = hi - lo
            if y < lo:
                s = width + (2.0 / alpha) * (lo - y)
            elif y > hi:
                s = width + (2.0 / alpha) * (y - hi)
            else:
                s = width
            score += w * s
        return score / total_w

    @staticmethod
    def _wdirectional_accuracy(actual: list[float], forecast: list[float],
                                last_known: float, weights: list[float]) -> float:
        """Взвешенная доля шагов с правильным знаком (прогноз - last_known)."""
        if not actual:
            return 0.0
        total_w = sum(weights[: len(actual)])
        if total_w == 0:
            return 0.0
        correct_w = sum(
            w for y, p, w in zip(actual, forecast, weights)
            if (y - last_known) * (p - last_known) > 0
        )
        return correct_w / total_w

    # ------------------------------------------------------------------
    # Per-window метрики
    # ------------------------------------------------------------------

    @staticmethod
    def close_metrics(
            actual: list[float],
            forecast: list[float],
            weights: list[float],
    ) -> dict[str, float]:
        """
        MAE / RMSE / MAPE для close-канала на одном окне.
        """
        if not actual:
            return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

        abs_errors: list[float] = []
        sq_errors:  list[float] = []
        pct_errors: list[float] = []
        for y_true, y_pred in zip(actual, forecast):
            err = y_pred - y_true
            abs_errors.append(abs(err))
            sq_errors.append(err * err)
            if y_true != 0:
                pct_errors.append((abs(err) / abs(y_true)) * 100.0)
            else:
                pct_errors.append(0.0)

        mae  = BacktestMetrics._wmean(abs_errors, weights)
        rmse = math.sqrt(BacktestMetrics._wmean(sq_errors, weights))
        mape = BacktestMetrics._wmean(pct_errors, weights)
        return {"mae": mae, "rmse": rmse, "mape": mape}

    @staticmethod
    def ohlc_metrics(
            actual_ohlc: list[dict],
            forecast_ohlc: list[dict],
            weights: list[float],
            channels=("open", "high", "low", "close"),
    ) -> dict[str, dict[str, float]]:
        """
        {channel: {mae, rmse, mape}} для каждого канала на одном окне.
        """
        result: dict[str, dict[str, float]] = {}
        for ch in channels:
            actual_ch = [float(row[ch]) for row in actual_ohlc]
            forecast_ch = [float(row[ch]) for row in forecast_ohlc]
            result[ch] = BacktestMetrics.close_metrics(actual_ch, forecast_ch, weights)
        return result

    @staticmethod
    def quantile_metrics_close(
            actual: list[float],
            q10: list[float],
            q25: list[float],
            q50: list[float],
            q75: list[float],
            q90: list[float],
            last_known_close: float,
            weights: list[float],
    ) -> dict[str, float]:
        """
        Квантильные метрики для close на одном окне.

        pinball_mean     — средний pinball loss по 5 квантилям (≈ CRPS)
        pinball_q50      — MAE медианы в pinball-форме (= 0.5 * MAE_q50)
        coverage_q25_q75 — норма ≈ 0.50
        coverage_q10_q90 — норма ≈ 0.80
        winkler_*        — штраф за широкие/промахивающиеся интервалы (меньше = лучше)
        directional_acc  — доля шагов с правильным направлением по медиане
        """
        if not actual:
            return {
                "pinball_mean": 0.0, "pinball_q50": 0.0,
                "coverage_q25_q75": 0.0, "coverage_q10_q90": 0.0,
                "winkler_q25_q75": 0.0, "winkler_q10_q90": 0.0,
                "directional_acc": 0.0,
            }

        levels = [(0.1, q10), (0.25, q25), (0.5, q50), (0.75, q75), (0.9, q90)]

        per_q_pinball: list[float] = []
        pinball_q50_value = 0.0
        for q, preds in levels:
            pb_at_each_step = [BacktestMetrics._pinball(y, p, q) for y, p in zip(actual, preds)]
            wm = BacktestMetrics._wmean(pb_at_each_step, weights)
            per_q_pinball.append(wm)
            if q == 0.5:
                pinball_q50_value = wm

        pinball_mean = sum(per_q_pinball) / len(per_q_pinball)

        return {
            "pinball_mean":     pinball_mean,
            "pinball_q50":      pinball_q50_value,
            "coverage_q25_q75": BacktestMetrics._wcoverage(actual, q25, q75, weights),
            "coverage_q10_q90": BacktestMetrics._wcoverage(actual, q10, q90, weights),
            "winkler_q25_q75":  BacktestMetrics._wwinkler(actual, q25, q75, alpha=0.5, weights=weights),
            "winkler_q10_q90":  BacktestMetrics._wwinkler(actual, q10, q90, alpha=0.2, weights=weights),
            "directional_acc":  BacktestMetrics._wdirectional_accuracy(actual, q50, last_known_close, weights),
        }

    @staticmethod
    def naive_close_metrics(
            actual: list[float],
            last_known_close: float,
            weights: list[float],
    ) -> dict[str, float]:
        """
        Метрики naive last-value baseline (forecast = last_known_close на всём горизонте).
        """
        forecast = [last_known_close] * len(actual)
        m = BacktestMetrics.close_metrics(actual, forecast, weights)
        return {
            "naive_mae":  m["mae"],
            "naive_rmse": m["rmse"],
            "naive_mape": m["mape"],
        }
