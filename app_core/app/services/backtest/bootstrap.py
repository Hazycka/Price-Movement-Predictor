"""
Bootstrap-доверительные интервалы для метрик бэктеста.

Единица ресэмплинга — окно walk-forward.  Имея массив per_window данных,
агрегатор `aggregator_fn(windows) -> dict[str, float]` пересчитывает
все интересующие метрики на любом подмножестве окон.

Алгоритм:
  1. На полном множестве окон считаем «primary» значения метрик
  2. N_iters раз ресэмплируем окна с возвращением (bootstrap)
  3. На каждом ресэмпле пересчитываем агрегатор
  4. Из распределения N_iters значений каждой метрики берём:
       - 95% CI: percentiles [2.5, 97.5]
       - std: для расчёта LCB = mean - z * std

LCB (Lower Confidence Bound) — это пессимистическая оценка метрики:
«насколько мы уверены, что результат хотя бы такой». Используется как
основной критерий ранжирования в sweep.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Sequence


AggregatorFn = Callable[[Sequence[Any]], dict[str, float]]


def _percentile(sorted_values: list[float], pct: float) -> float:
    """
    Линейная интерполяция percentile по отсортированному массиву.
    pct ∈ [0, 100].
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def bootstrap_metrics(
        windows: Sequence[Any],
        aggregator: AggregatorFn,
        n_iters: int,
        z_score: float,
        random_seed: int | None = None,
) -> tuple[dict[str, float], dict[str, list[float]], dict[str, float]]:
    """
    Возвращает три словаря с одинаковыми ключами:
       primary       — значения метрик на полном множестве окон
       ci_by_metric  — [lower_2.5, upper_97.5] для каждой метрики
       lcb_by_metric — mean - z_score * std для каждой метрики

    n_iters=0 → CI и LCB вычисляются вырожденно: CI=[primary, primary], LCB=primary.
    Это поведение полезно для отладки или когда CI не нужен.
    """
    primary = aggregator(windows)

    if n_iters <= 0 or len(windows) < 2:
        # Вырожденный случай: невозможно сделать осмысленный bootstrap
        ci = {name: [value, value] for name, value in primary.items()}
        lcb = dict(primary)
        return primary, ci, lcb

    rng = random.Random(random_seed)
    n = len(windows)
    samples_by_metric: dict[str, list[float]] = {name: [] for name in primary}

    for _ in range(n_iters):
        sample = [windows[rng.randrange(n)] for _ in range(n)]
        metrics = aggregator(sample)
        for name, value in metrics.items():
            samples_by_metric.setdefault(name, []).append(value)

    ci: dict[str, list[float]] = {}
    lcb: dict[str, float] = {}
    for name, samples in samples_by_metric.items():
        samples_sorted = sorted(samples)
        ci[name] = [
            _percentile(samples_sorted, 2.5),
            _percentile(samples_sorted, 97.5),
        ]
        mean = sum(samples) / len(samples)
        variance = sum((s - mean) ** 2 for s in samples) / len(samples)
        std = variance ** 0.5
        lcb[name] = primary[name] - z_score * std

    return primary, ci, lcb
