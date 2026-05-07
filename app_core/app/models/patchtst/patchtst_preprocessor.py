"""
Препроцессор для PatchTSTFMForPrediction.

Отличие от классического PatchTST:
- Вход модели = конкатенация [история + маски горизонта]
- Общая длина тензора = context_length (из config)
- Если история длиннее context_length - horizon → обрезаем с начала
- Если история короче → паддим средним значения по каналу (как в оригинале)
- Позиции горизонта заполняются нулями (маски)

Нормализация: instance normalization (mean=0, std=1 по каждому каналу)
применяется к истории; горизонт нормализуется теми же параметрами.
"""
import numpy as np
import pandas as pd


class PatchTSTPreprocessor:

    @staticmethod
    def normalize(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Instance normalization по каждому каналу.
        arr: [T, C]
        Возвращает: нормализованный arr, mean [1, C], std [1, C]
        """
        mean = np.mean(arr, axis=0, keepdims=True)   # [1, C]
        std  = np.std(arr,  axis=0, keepdims=True)   # [1, C]
        std  = np.where(std < 1e-8, 1.0, std)
        return (arr - mean) / std, mean, std

    @staticmethod
    def to_model_input(
            df: pd.DataFrame,
            feature_columns: list[str],
            horizon: int,
            context_length: int
    ) -> tuple[np.ndarray, dict]:
        """
        Подготавливает входной тензор для PatchTSTFM.

        Возвращает:
            input_tensor: np.ndarray [1, context_length, num_channels]
                          (batch=1, time, channels)
            window_info:  dict с метаинформацией об использованном окне
        """
        arr = df[feature_columns].to_numpy(dtype=np.float32)  # [T, C]
        total_len = int(arr.shape[0])
        num_channels = arr.shape[1]

        # Длина истории которую мы можем подать = context_length - horizon
        # (горизонт займёт оставшееся место в контексте модели)
        max_history = context_length - horizon

        if max_history <= 0:
            raise ValueError(
                f"horizon={horizon} >= context_length={context_length}. "
                f"Уменьшите горизонт прогноза."
            )

        # --- Определяем какой кусок истории использовать ---
        trimmed = False
        start_index_used = 0

        if total_len > max_history:
            # Обрезаем с начала — берём последние max_history точек
            start_index_used = total_len - max_history
            history_arr = arr[start_index_used:, :]
            used_length = max_history
            trimmed = True
        elif total_len < max_history:
            # История короче — будем паддить
            history_arr = arr
            used_length = total_len
        else:
            history_arr = arr
            used_length = total_len

        # --- Нормализация по истории ---
        history_norm, mean, std = PatchTSTPreprocessor.normalize(history_arr)

        # --- Паддинг если история короче max_history ---
        if used_length < max_history:
            pad_len = max_history - used_length
            # Паддим средним значением (нормализованным → 0.0)
            pad = np.zeros((pad_len, num_channels), dtype=np.float32)
            history_norm = np.concatenate([pad, history_norm], axis=0)  # [max_history, C]

        # --- Добавляем маску горизонта (нули = замаскированные позиции) ---
        horizon_mask = np.zeros((horizon, num_channels), dtype=np.float32)

        # Конкатенируем: [max_history, C] + [horizon, C] = [context_length, C]
        input_seq = np.concatenate([history_norm, horizon_mask], axis=0)  # [context_length, C]

        # Добавляем batch dimension: [1, context_length, C]
        input_tensor = input_seq[np.newaxis, :, :]

        window_info = {
            "context_length": context_length,
            "max_history_used": max_history,
            "original_length": total_len,
            "used_length": used_length,
            "trimmed": trimmed,
            "start_index_used": start_index_used,
            "padded": used_length < max_history,
            "pad_length": max(0, max_history - used_length),
            "horizon": horizon,
        }

        return input_tensor, window_info