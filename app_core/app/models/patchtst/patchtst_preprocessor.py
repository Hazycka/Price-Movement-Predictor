import numpy as np
import pandas as pd


class PatchTSTPreprocessor:
    @staticmethod
    def to_model_input(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
        arr = df[feature_columns].to_numpy(dtype=np.float32)
        mean = np.mean(arr, axis=0, keepdims=True)
        std = np.std(arr, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return (arr - mean) / std

    @staticmethod
    def apply_context_window(arr: np.ndarray, required_context_length: int | None) -> tuple[np.ndarray, dict]:
        total_len = int(arr.shape[0])

        if required_context_length is None:
            return arr, {
                "required_context_length": None,
                "original_length": total_len,
                "used_length": total_len,
                "trimmed": False,
                "start_index_used": 0
            }

        required = int(required_context_length)
        if total_len < required:
            raise ValueError(
                f"История слишком мала: получено {total_len}, требуется минимум {required}."
            )

        if total_len > required:
            start = total_len - required
            used = arr[start:, :]
            return used, {
                "required_context_length": required,
                "original_length": total_len,
                "used_length": required,
                "trimmed": True,
                "start_index_used": start
            }

        return arr, {
            "required_context_length": required,
            "original_length": total_len,
            "used_length": total_len,
            "trimmed": False,
            "start_index_used": 0
        }
