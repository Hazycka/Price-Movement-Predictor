"""
Декодер для PatchTST FM.

Pipeline возвращает pandas DataFrame — декодирование квантилей
происходит в patchtst_model.py через _extract_channel_quantiles.

Этот модуль содержит только утилиту коррекции физических ограничений свечей,
которая гарантирует high >= max(o,c) и low <= min(o,c) для каждого квантиля.
"""
from ..base import QuantileForecast


class PatchTSTDecoder:

    @staticmethod
    def enforce_ohlc_consistency(
            open_qf: QuantileForecast,
            high_qf: QuantileForecast,
            low_qf: QuantileForecast,
            close_qf: QuantileForecast,
    ) -> tuple[QuantileForecast, QuantileForecast, QuantileForecast, QuantileForecast]:
        """
        Корректирует high и low чтобы свечи были физически корректны.

        Модель channel-independent — она не знает об ограничениях
        high >= max(open, close) и low <= min(open, close).
        Применяем коррекцию для каждого квантиля независимо.
        """
        for q_name in ("q10", "q25", "q50", "q75", "q90"):
            o = getattr(open_qf,  q_name)
            h = getattr(high_qf,  q_name)
            l = getattr(low_qf,   q_name)
            c = getattr(close_qf, q_name)

            setattr(high_qf, q_name, [max(o[i], h[i], l[i], c[i]) for i in range(len(o))])
            setattr(low_qf,  q_name, [min(o[i], h[i], l[i], c[i]) for i in range(len(o))])

        return open_qf, high_qf, low_qf, close_qf