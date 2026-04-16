import numpy as np


class PatchTSTDecoder:
    @staticmethod
    def extract_prediction_array(outputs):
        pred = None
        for attr in ("prediction_outputs", "regression_outputs", "logits"):
            if hasattr(outputs, attr):
                pred = getattr(outputs, attr)
                break
        if pred is None:
            if isinstance(outputs, tuple) and len(outputs) > 0:
                pred = outputs[0]
            else:
                raise RuntimeError("PatchTST вернул неожиданный формат выхода.")

        arr = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)
        return np.squeeze(arr)

    @staticmethod
    def decode_close(arr: np.ndarray, feature_columns: list[str], target_column: str) -> list[float]:
        if arr.ndim == 1:
            point_forecast = arr
        elif arr.ndim == 2:
            point_forecast = arr[0] if arr.shape[0] > 1 else arr.flatten()
        elif arr.ndim >= 3:
            target_idx = feature_columns.index(target_column) if target_column in feature_columns else 3
            point_forecast = arr[0, :, target_idx] if arr.shape[-1] > target_idx else arr[0, :, 0]
        else:
            raise RuntimeError(f"Неожиданная форма прогноза PatchTST: {arr.shape}")

        return [float(x) for x in np.asarray(point_forecast).flatten().tolist()]

    @staticmethod
    def decode_ohlc(arr: np.ndarray, feature_columns: list[str]) -> dict[str, list[float]]:
        if arr.ndim < 3:
            raise ValueError("Модель вернула не-многоканальный прогноз. Для OHLC нужен каналовый выход.")

        ch_open = feature_columns.index("open")
        ch_high = feature_columns.index("high")
        ch_low = feature_columns.index("low")
        ch_close = feature_columns.index("close")

        if arr.shape[-1] <= max(ch_open, ch_high, ch_low, ch_close):
            raise ValueError("В выходе модели недостаточно каналов для OHLC.")

        return {
            "open": arr[0, :, ch_open].astype(float).tolist(),
            "high": arr[0, :, ch_high].astype(float).tolist(),
            "low": arr[0, :, ch_low].astype(float).tolist(),
            "close": arr[0, :, ch_close].astype(float).tolist(),
        }
