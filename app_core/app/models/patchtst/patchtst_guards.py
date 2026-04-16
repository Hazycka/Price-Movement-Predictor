import numpy as np


class PatchTSTGuards:
    @staticmethod
    def trim_to_horizon(values: list[float], horizon: int) -> list[float]:
        return values[:horizon]
    
    @staticmethod
    def pad_or_trim(values: list[float], horizon: int) -> list[float]:
        result = values[:horizon]
        if len(result) < horizon and result:
            result.extend([result[-1]] * (horizon - len(result)))
        return result

    @staticmethod
    def has_non_finite(values: list[float]) -> bool:
        return any(np.isnan(values)) or any(np.isinf(values))

    @staticmethod
    def has_non_finite_ohlc(rows: list[dict[str, float]]) -> bool:
        return any(any(np.isnan(v) or np.isinf(v) for v in row.values()) for row in rows)
