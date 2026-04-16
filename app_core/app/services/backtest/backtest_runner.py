from ...schemas import BacktestResponse
from .window_usage import ModelWindowUsageExtractor
from .metrics import BacktestMetrics


class BacktestRunner:
    def __init__(self, model) -> None:
        self.model = model
        self.window_usage = ModelWindowUsageExtractor(model=model)

    def run(self, request, source: str, dates: list[str], candles: list[dict[str, float]]) -> BacktestResponse:
        close_series = [candle["close"] for candle in candles]
        if len(close_series) < request.min_train_size + request.horizon:
            raise ValueError(
                f"Недостаточно данных для бэктеста: нужно минимум {request.min_train_size + request.horizon}, "
                f"получено {len(close_series)}."
            )

        if request.backtest_target == "close":
            return self._run_close(request, source, dates, candles, close_series)
        if request.backtest_target == "ohlc":
            return self._run_ohlc(request, source, dates, candles, close_series)
        raise ValueError(f"Неподдерживаемый backtest_target='{request.backtest_target}'.")

    def _run_close(self, request, source, dates, candles, close_series) -> BacktestResponse:
        trimmed_windows_count = 0
        required_context_length_seen = None
        all_abs, all_sq, all_pct = [], [], []
        details = []

        windows_used = 0
        start_train_end = request.min_train_size
        last_train_end = len(close_series) - request.horizon

        for train_end in range(start_train_end, last_train_end + 1, request.step):
            if windows_used >= request.max_windows:
                break

            train_candles = candles[:train_end]
            actual = close_series[train_end:train_end + request.horizon]
            forecast = self.model.predict_multivariate(
                candles=train_candles,
                horizon=request.horizon,
                context={
                    "num_samples": request.num_samples,
                    "feature_plugins": request.feature_plugins,
                    "mode": "backtest",
                    "backtest_target": "close"
                }
            )

            if len(forecast) != len(actual):
                min_len = min(len(forecast), len(actual))
                forecast = forecast[:min_len]
                actual = actual[:min_len]

            window_info = self.window_usage.extract(dates=dates, train_end=train_end)
            if window_info.get("trimmed"):
                trimmed_windows_count += 1
            if required_context_length_seen is None:
                required_context_length_seen = window_info.get("required_context_length")

            metrics, local_abs, local_sq, local_pct = BacktestMetrics.close_metrics(actual, forecast)
            all_abs.extend(local_abs)
            all_sq.extend(local_sq)
            all_pct.extend(local_pct)

            details.append({
                "train_end_index": train_end,
                "train_end_date": dates[train_end - 1] if dates and train_end - 1 < len(dates) else None,
                "horizon": len(actual),
                "actual_close": actual,
                "forecast_close": forecast,
                "model_input_window": window_info,
                **metrics
            })
            windows_used += 1

        import math
        mae = float(sum(all_abs) / len(all_abs)) if all_abs else 0.0
        rmse = float(math.sqrt(sum(all_sq) / len(all_sq))) if all_sq else 0.0
        mape = float(sum(all_pct) / len(all_pct)) if all_pct else 0.0

        return BacktestResponse(
            source=source,
            model=self.model.get_info(),
            metrics={"mae": mae, "rmse": rmse, "mape": mape},
            windows_count=windows_used,
            horizon=request.horizon,
            history_length=len(close_series),
            details=details,
            metadata={
                "step": request.step,
                "min_train_size": request.min_train_size,
                "max_windows": request.max_windows,
                "num_samples": request.num_samples,
                "feature_plugins": request.feature_plugins,
                "backtest_target": "close",
                "model_required_context_length": required_context_length_seen,
                "trimmed_windows_count": trimmed_windows_count,
                "trimmed_windows_share": float(trimmed_windows_count / windows_used) if windows_used else 0.0,
                "metric_note": "MAPE считается только там, где actual != 0"
            }
        )

    def _run_ohlc(self, request, source, dates, candles, close_series) -> BacktestResponse:
        channels = ("open", "high", "low", "close")
        trimmed_windows_count = 0
        required_context_length_seen = None

        all_abs = {ch: [] for ch in channels}
        all_sq = {ch: [] for ch in channels}
        all_pct = {ch: [] for ch in channels}
        details = []

        windows_used = 0
        start_train_end = request.min_train_size
        last_train_end = len(close_series) - request.horizon

        for train_end in range(start_train_end, last_train_end + 1, request.step):
            if windows_used >= request.max_windows:
                break

            train_candles = candles[:train_end]
            actual_ohlc = candles[train_end:train_end + request.horizon]
            forecast_ohlc = self.model.predict_ohlc_multivariate(
                candles=train_candles,
                horizon=request.horizon,
                context={
                    "num_samples": request.num_samples,
                    "feature_plugins": request.feature_plugins,
                    "mode": "backtest",
                    "backtest_target": "ohlc"
                }
            )

            if len(forecast_ohlc) != len(actual_ohlc):
                min_len = min(len(forecast_ohlc), len(actual_ohlc))
                forecast_ohlc = forecast_ohlc[:min_len]
                actual_ohlc = actual_ohlc[:min_len]

            window_info = self.window_usage.extract(dates=dates, train_end=train_end)
            if window_info.get("trimmed"):
                trimmed_windows_count += 1
            if required_context_length_seen is None:
                required_context_length_seen = window_info.get("required_context_length")

            per_channel, agg, local_abs, local_sq, local_pct = BacktestMetrics.ohlc_metrics(
                actual_ohlc, forecast_ohlc, channels=channels
            )
            for ch in channels:
                all_abs[ch].extend(local_abs[ch])
                all_sq[ch].extend(local_sq[ch])
                all_pct[ch].extend(local_pct[ch])

            details.append({
                "train_end_index": train_end,
                "train_end_date": dates[train_end - 1] if dates and train_end - 1 < len(dates) else None,
                "horizon": len(actual_ohlc),
                "actual_ohlc": actual_ohlc,
                "forecast_ohlc": forecast_ohlc,
                "model_input_window": window_info,
                "metrics_by_channel": per_channel,
                "metrics_aggregate": agg
            })
            windows_used += 1

        import math
        global_by_channel = {
            ch: {
                "mae": float(sum(all_abs[ch]) / len(all_abs[ch])) if all_abs[ch] else 0.0,
                "rmse": float(math.sqrt(sum(all_sq[ch]) / len(all_sq[ch]))) if all_sq[ch] else 0.0,
                "mape": float(sum(all_pct[ch]) / len(all_pct[ch])) if all_pct[ch] else 0.0
            } for ch in channels
        }

        mae_mean_ohlc = float(sum(global_by_channel[ch]["mae"] for ch in channels) / len(channels))
        rmse_mean_ohlc = float(sum(global_by_channel[ch]["rmse"] for ch in channels) / len(channels))
        mape_mean_ohlc = float(sum(global_by_channel[ch]["mape"] for ch in channels) / len(channels))

        return BacktestResponse(
            source=source,
            model=self.model.get_info(),
            metrics={
                "mae_mean_ohlc": mae_mean_ohlc,
                "rmse_mean_ohlc": rmse_mean_ohlc,
                "mape_mean_ohlc": mape_mean_ohlc,
                "mae_open": global_by_channel["open"]["mae"],
                "rmse_open": global_by_channel["open"]["rmse"],
                "mape_open": global_by_channel["open"]["mape"],
                "mae_high": global_by_channel["high"]["mae"],
                "rmse_high": global_by_channel["high"]["rmse"],
                "mape_high": global_by_channel["high"]["mape"],
                "mae_low": global_by_channel["low"]["mae"],
                "rmse_low": global_by_channel["low"]["rmse"],
                "mape_low": global_by_channel["low"]["mape"],
                "mae_close": global_by_channel["close"]["mae"],
                "rmse_close": global_by_channel["close"]["rmse"],
                "mape_close": global_by_channel["close"]["mape"]
            },
            windows_count=windows_used,
            horizon=request.horizon,
            history_length=len(close_series),
            details=details,
            metadata={
                "step": request.step,
                "min_train_size": request.min_train_size,
                "max_windows": request.max_windows,
                "num_samples": request.num_samples,
                "feature_plugins": request.feature_plugins,
                "backtest_target": "ohlc",
                "model_required_context_length": required_context_length_seen,
                "trimmed_windows_count": trimmed_windows_count,
                "trimmed_windows_share": float(trimmed_windows_count / windows_used) if windows_used else 0.0,
                "metric_mode": "ohlc_channel_wise_plus_aggregate",
                "metric_note": "MAPE считается только там, где actual != 0"
            }
        )