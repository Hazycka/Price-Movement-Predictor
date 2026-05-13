"""
Веса для оценки многошагового прогноза.

При горизонте H модель предсказывает H точек. Считать качество равномерно
по всем H точкам — некорректно: ошибка на 1-м шаге в типовом use-case важнее
ошибки на H-м (мы быстрее по нему действуем).

Поэтому метрики становятся ВЗВЕШЕННЫМИ:
    metric = sum(w_i * loss_i) / sum(w_i)

Поддерживаемые схемы:
  uniform     — все веса равны (бэкап для отладки и сравнения)
  exponential — w_i = ratio^(-i / (H - 1)); первый бар в `ratio` раз важнее последнего
  linear      — w_i убывает линейно от 1 до 1/ratio

Параметр weight_first_to_last_ratio имеет прозрачный смысл:
во сколько раз вес первого бара больше веса последнего.
"""
from __future__ import annotations

from typing import Literal


WeightScheme = Literal["uniform", "exponential", "linear"]


def compute_weights(
        horizon: int,
        scheme: WeightScheme,
        first_to_last_ratio: float = 16.0,
) -> list[float]:
    """
    Возвращает список длиной horizon: w[0] = вес первого предсказанного бара,
    w[horizon-1] = вес последнего.

    Веса нормализованы так, что sum(w) == horizon — взвешенные метрики тогда
    численно сопоставимы с невзвешенным средним при scheme="uniform".
    """
    if horizon <= 0:
        raise ValueError(f"horizon должен быть положительным, получено {horizon}")
    if first_to_last_ratio < 1.0:
        raise ValueError(
            f"first_to_last_ratio должен быть >= 1.0 (первый бар не менее важен "
            f"чем последний), получено {first_to_last_ratio}"
        )

    if scheme == "uniform" or horizon == 1 or first_to_last_ratio == 1.0:
        return [1.0] * horizon

    if scheme == "exponential":
        # w_i = ratio^(-i / (H-1))
        # i=0:        ratio^0 = 1
        # i=H-1: ratio^(-1) = 1/ratio
        denom = horizon - 1
        raw = [first_to_last_ratio ** (-i / denom) for i in range(horizon)]
    elif scheme == "linear":
        # w_i убывает линейно от 1 до 1/ratio
        # w_i = 1 - (i / (H-1)) * (1 - 1/ratio)
        denom = horizon - 1
        end = 1.0 / first_to_last_ratio
        raw = [1.0 - (i / denom) * (1.0 - end) for i in range(horizon)]
    else:
        raise ValueError(f"Неподдерживаемая schema весов: {scheme}")

    # Нормализация: sum(w) = horizon
    total = sum(raw)
    factor = horizon / total
    return [w * factor for w in raw]


def weighted_mean(values: list[float], weights: list[float]) -> float:
    """Среднее с весами. sum(w*v) / sum(w)."""
    if not values:
        return 0.0
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w
