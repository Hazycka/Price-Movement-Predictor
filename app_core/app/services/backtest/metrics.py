import math


class BacktestMetrics:
    @staticmethod
    def close_metrics(actual: list[float], forecast: list[float]) -> tuple[dict, list[float], list[float], list[float]]:
        local_abs: list[float] = []
        local_sq: list[float] = []
        local_pct: list[float] = []

        for y_true, y_pred in zip(actual, forecast):
            err = y_pred - y_true
            aerr = abs(err)
            serr = err * err
            local_abs.append(aerr)
            local_sq.append(serr)
            if y_true != 0:
                local_pct.append((aerr / abs(y_true)) * 100.0)

        return {
            "mae": float(sum(local_abs) / len(local_abs)) if local_abs else 0.0,
            "rmse": float(math.sqrt(sum(local_sq) / len(local_sq))) if local_sq else 0.0,
            "mape": float(sum(local_pct) / len(local_pct)) if local_pct else 0.0,
        }, local_abs, local_sq, local_pct

    @staticmethod
    def ohlc_metrics(actual_ohlc: list[dict], forecast_ohlc: list[dict], channels=("open", "high", "low", "close")):
        local_abs = {ch: [] for ch in channels}
        local_sq = {ch: [] for ch in channels}
        local_pct = {ch: [] for ch in channels}

        for a_row, f_row in zip(actual_ohlc, forecast_ohlc):
            for ch in channels:
                y_true = float(a_row[ch])
                y_pred = float(f_row[ch])
                err = y_pred - y_true
                aerr = abs(err)
                serr = err * err
                local_abs[ch].append(aerr)
                local_sq[ch].append(serr)
                if y_true != 0:
                    local_pct[ch].append((aerr / abs(y_true)) * 100.0)

        per_channel = {
            ch: {
                "mae": float(sum(local_abs[ch]) / len(local_abs[ch])) if local_abs[ch] else 0.0,
                "rmse": float(math.sqrt(sum(local_sq[ch]) / len(local_sq[ch]))) if local_sq[ch] else 0.0,
                "mape": float(sum(local_pct[ch]) / len(local_pct[ch])) if local_pct[ch] else 0.0
            } for ch in channels
        }

        agg = {
            "mae_mean_ohlc": float(sum(per_channel[ch]["mae"] for ch in channels) / len(channels)),
            "rmse_mean_ohlc": float(sum(per_channel[ch]["rmse"] for ch in channels) / len(channels)),
            "mape_mean_ohlc": float(sum(per_channel[ch]["mape"] for ch in channels) / len(channels))
        }

        return per_channel, agg, local_abs, local_sq, local_pct