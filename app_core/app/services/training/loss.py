"""
Weighted pinball loss для обучения адаптеров.

Та же логика что в backtest.metrics.BacktestMetrics._pinball, только векторизованная
через torch для backward pass и со взвешиванием по горизонту (первые бары важнее).

Формула pinball для квантиля q ∈ (0,1), истины y, прогноза p:
  loss = max(q*(y-p), (q-1)*(y-p))
       = max(q*err, (q-1)*err)  где err = y - p

При q=0.5 получаем 0.5*|err| — половина MAE.
При q≠0.5 штраф асимметричен: недопредсказание / перепредсказание весят по-разному.

Loss = взвешенное среднее pinball по [batch × horizon × num_quantiles × channels],
где веса по горизонту определяют относительную важность баров (см. weights.py).
"""
from __future__ import annotations

from typing import Sequence

try:
    import torch
except ImportError:
    torch = None


def weighted_pinball_loss(
        predicted_quantiles,  # (B, H, num_quantiles, C) или (B, H, num_quantiles)
        actual,                # (B, H, C) или (B, H)
        weights,               # (H,) или 1D-tensor длины H
        quantile_levels: Sequence[float],
):
    """
    Returns scalar loss tensor.

    predicted_quantiles: torch.Tensor shape (B, H, Q [,C]) — модель прогноза квантилей
    actual:              torch.Tensor shape (B, H [,C])    — реальные значения
    weights:             torch.Tensor shape (H,)           — веса по горизонту
    quantile_levels:     список Q квантилей, например [0.1, 0.25, 0.5, 0.75, 0.9]

    Если есть размерность каналов (C) — pinball считается по каждому каналу и усредняется.
    """
    if torch is None:
        raise RuntimeError("PyTorch не установлен.")

    if not isinstance(quantile_levels, torch.Tensor):
        q = torch.tensor(quantile_levels, dtype=predicted_quantiles.dtype,
                         device=predicted_quantiles.device)
    else:
        q = quantile_levels

    # Расширяем actual под shape predicted_quantiles
    # predicted_quantiles: (B, H, Q [,C])
    # actual:              (B, H [,C])
    # → нужно: actual.unsqueeze(2) (B, H, 1 [,C]) для broadcast по Q
    while actual.ndim < predicted_quantiles.ndim:
        actual = actual.unsqueeze(2)  # после: (B, H, 1) или (B, H, 1, C)

    err = actual - predicted_quantiles  # (B, H, Q [,C])

    # Broadcast квантилей: q shape (Q,) → (1, 1, Q, [1])
    while q.ndim < err.ndim:
        q = q.unsqueeze(0)
    # Теперь q: (1, 1, Q) или (1, 1, Q, 1)
    # При необходимости добавить C-dim вручную чтобы соответствовать err
    # (но q в Q-измерении, err также имеет Q, остальные совпадут через broadcasting)

    loss_per_point = torch.maximum(q * err, (q - 1.0) * err)  # (B, H, Q [,C])

    # Усредняем по Q (квантилям) и C (каналам если есть)
    while loss_per_point.ndim > 2:
        loss_per_point = loss_per_point.mean(dim=-1)
    # Теперь shape (B, H)

    # Взвешенное среднее по горизонту H с весами weights (H,)
    weights = weights.to(loss_per_point.device).to(loss_per_point.dtype)
    weighted = (loss_per_point * weights.unsqueeze(0)).sum(dim=-1) / weights.sum()  # (B,)
    return weighted.mean()
