"""
PyTorch Dataset для walk-forward обучения адаптеров над PatchTST FM.

Для каждой обучающей пары (x, y):
  x = окно длины train_window_size (контекст)
  y = горизонт длины horizon (целевой OHLC)

При итерации __getitem__ возвращает кортеж тензоров:
  (past_values, future_values)
  past_values:   (T_ctx, C)  где C = число каналов (OHLC = 4)
  future_values: (H, C)

Внутренние представления: float32. Нормализация (RevIN) применяется
внутри модели — здесь сырые значения.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


OHLC_CHANNELS = ("open", "high", "low", "close")


@dataclass
class WalkForwardSample:
    """Одна обучающая пара."""
    train_end: int   # индекс конца контекста (exclusive) в исходном candles
    past: np.ndarray   # (T_ctx, C)
    future: np.ndarray  # (H, C)


class WalkForwardDataset(Dataset):
    """
    Walk-forward dataset поверх списка свечей.

    Параметры:
      candles            — list[dict] с ключами OHLC (+volume опционально)
      train_window_size  — длина контекста (T_ctx)
      horizon            — длина прогноза (H)
      step               — шаг walk-forward (между концами окон)
      channels           — какие каналы использовать (default OHLC)
      val_split          — доля окон отводимая в валидацию из хвоста истории.
                           0.0 — нет валидации. 0.15 = последние 15% окон → val.
      mode               — 'train' или 'val'. Используется вместе с val_split:
                           'train' → первые (1 - val_split) окон,
                           'val'   → последние val_split окон.

    Использование:
      ds_train = WalkForwardDataset(candles, 8192, 64, step=64, val_split=0.15, mode='train')
      ds_val   = WalkForwardDataset(candles, 8192, 64, step=64, val_split=0.15, mode='val')
      DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=4)
    """

    def __init__(
            self,
            candles: list[dict[str, float]],
            train_window_size: int,
            horizon: int,
            step: int,
            channels: Iterable[str] = OHLC_CHANNELS,
            val_split: float = 0.0,
            mode: str = "train",
    ) -> None:
        if torch is None:
            raise RuntimeError(
                "PyTorch не установлен. Установите torch для использования training-функциональности."
            )
        if mode not in ("train", "val"):
            raise ValueError(f"mode должен быть 'train' или 'val', получено '{mode}'")
        if not 0.0 <= val_split < 1.0:
            raise ValueError(f"val_split должен быть в [0, 1), получено {val_split}")

        self.candles = candles
        self.train_window_size = train_window_size
        self.horizon = horizon
        self.step = step
        self.channels = tuple(channels)
        self.mode = mode

        # Список всех возможных train_end-индексов, для которых влезает (контекст + горизонт)
        all_indices = list(range(
            train_window_size,
            len(candles) - horizon + 1,
            step,
        ))
        if not all_indices:
            raise ValueError(
                f"Недостаточно данных: len(candles)={len(candles)} < "
                f"train_window_size={train_window_size} + horizon={horizon}."
            )

        # Делим на train/val
        if val_split > 0:
            split_idx = int(len(all_indices) * (1.0 - val_split))
            self.indices = all_indices[:split_idx] if mode == "train" else all_indices[split_idx:]
        else:
            self.indices = all_indices if mode == "train" else []

        # Преобразуем candles в numpy-матрицу (N, C) один раз для быстрого slicing'а
        self._matrix = np.array(
            [[c.get(ch, 0.0) for ch in self.channels] for c in candles],
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        train_end = self.indices[idx]
        past = self._matrix[train_end - self.train_window_size: train_end]   # (T_ctx, C)
        future = self._matrix[train_end: train_end + self.horizon]            # (H, C)
        return (
            torch.from_numpy(past).float(),
            torch.from_numpy(future).float(),
        )
