"""
BacktestRunner — выполняет walk-forward бэктест для одной конфигурации.

Архитектура:
  1. Подготовка: веса по горизонту, разрешение step и max_windows
  2. Walk-forward цикл: для каждого train_end получаем train_candles
     (sliding или growing), запрашиваем модель, складываем per-window
     данные в список (НЕ агрегируем по ходу)
  3. Агрегация: на полном множестве окон считаем primary метрики
     через _aggregate_close / _aggregate_ohlc
  4. Bootstrap: ресэмплируем окна N раз, считаем CI и LCB
  5. Формируем BacktestResponse

Per-window данные хранятся как простые dict'ы — это позволяет
bootstrap-агрегатору пересчитывать метрики на любом подмножестве окон.
"""
from __future__ import annotations

from typing import Any

from ...schemas import BacktestResponse
from .window_usage import ModelWindowUsageExtractor
from .metrics import BacktestMetrics
from .weights import compute_weights
from .bootstrap import bootstrap_metrics


class BacktestRunner:
    def __init__(self, model) -> None:
        self.model = model
        self.window_usage = ModelWindowUsageExtractor(model=model)

    # ------------------------------------------------------------------
    # Точка входа
    # ------------------------------------------------------------------

    def run(
            self,
            request,
            source: str,
            dates: list[str],
            candles: list[dict[str, float]],
            precomputed_predictions: list | None = None,
    ) -> BacktestResponse:
        """
        Запуск walk-forward бэктеста.

        precomputed_predictions: если задан — пропускаем вызов модели, используем
        переданные предсказания. Длина должна совпадать с количеством walk-forward
        окон, иначе ValueError.

        Этот режим нужен для cross-fold батчинга в sweep CV: предсказания всех
        фолдов одного контекста собираются в ОДИН batched-вызов модели, а потом
        раздаются по фолдам для per-fold агрегации метрик.
        """
        close_series = [candle["close"] for candle in candles]

        # --- разрешение умолчаний ---
        horizon = request.horizon
        step = request.step if request.step is not None else horizon
        train_window_size = request.train_window_size
        mode = request.train_window_mode

        # --- веса по горизонту ---
        weights = compute_weights(
            horizon=horizon,
            scheme=request.evaluation_weights,
            first_to_last_ratio=request.weight_first_to_last_ratio,
        )

        # --- валидация ---
        if train_window_size > len(close_series) - horizon:
            raise ValueError(
                f"Недостаточно данных: train_window_size={train_window_size} + horizon={horizon} "
                f"> len(history)={len(close_series)}. Уменьшите train_window_size или horizon, "
                f"или загрузите более длинную историю."
            )

        # --- запуск ---
        if request.backtest_target == "close":
            return self._run_close(
                request, source, dates, candles, close_series,
                weights=weights, step=step, train_window_size=train_window_size, mode=mode,
                precomputed_predictions=precomputed_predictions,
            )
        if request.backtest_target == "ohlc":
            return self._run_ohlc(
                request, source, dates, candles, close_series,
                weights=weights, step=step, train_window_size=train_window_size, mode=mode,
                precomputed_predictions=precomputed_predictions,
            )
        raise ValueError(f"Неподдерживаемый backtest_target='{request.backtest_target}'.")

    # ------------------------------------------------------------------
    # Walk-forward iterator: возвращает (train_start, train_end) для каждого окна
    # ------------------------------------------------------------------

    def _get_predictions(
            self,
            target: str,
            train_candles_list: list[list[dict[str, float]]],
            horizon: int,
            request,
            precomputed_predictions: list | None,
    ) -> list:
        """
        Получает прогнозы либо из precomputed_predictions, либо вызовом модели.

        target — 'close' или 'ohlc', определяет какой метод модели вызывать.
        Validation: precomputed_predictions длина должна совпадать с числом окон.
        """
        if precomputed_predictions is not None:
            if len(precomputed_predictions) != len(train_candles_list):
                raise ValueError(
                    f"precomputed_predictions длина {len(precomputed_predictions)} "
                    f"не совпадает с walk-forward окон {len(train_candles_list)}."
                )
            return precomputed_predictions

        ctx = {
            "model_options":        request.model_options,
            "feature_plugins":      request.feature_plugins,
            "mode":                 "backtest",
            "backtest_target":      target,
            "detailed_logs":        request.detailed_logs,
            "inference_batch_size": request.inference_batch_size,
        }
        try:
            if target == "ohlc":
                return self.model.predict_ohlc_quantiles_batch(train_candles_list, horizon, ctx)
            return self.model.predict_line_exact_batch(train_candles_list, horizon, ctx)
        except NotImplementedError as ex:
            other = "ohlc" if target == "close" else "close"
            raise ValueError(
                f"Модель '{self.model.get_info().get('name')}' не поддерживает "
                f"backtest_target='{target}'. Используйте backtest_target='{other}'. Детали: {ex}"
            ) from ex

    @staticmethod
    def _walk_forward_indices(
            history_len: int,
            horizon: int,
            train_window_size: int,
            step: int,
            mode: str,
            max_windows: int | None,
    ) -> list[tuple[int, int]]:
        """
        Возвращает список (train_start, train_end) для каждого окна.
        Окно использует candles[train_start:train_end] для прогноза candles[train_end:train_end+horizon].

        sliding: train_start = train_end - train_window_size (фиксированный размер)
        growing: train_start = 0 (растёт от 0 до train_end)
        """
        first_train_end = train_window_size
        last_train_end = history_len - horizon
        windows: list[tuple[int, int]] = []
        for train_end in range(first_train_end, last_train_end + 1, step):
            train_start = (train_end - train_window_size) if mode == "sliding" else 0
            windows.append((train_start, train_end))
            if max_windows is not None and len(windows) >= max_windows:
                break
        return windows

    # ------------------------------------------------------------------
    # Close (точечный прогноз)
    # ------------------------------------------------------------------

    def _run_close(
            self, request, source, dates, candles, close_series,
            weights: list[float], step: int, train_window_size: int, mode: str,
            precomputed_predictions: list | None = None,
    ) -> BacktestResponse:
        horizon = request.horizon
        verbose = request.detailed_logs

        windows_indices = self._walk_forward_indices(
            history_len=len(close_series),
            horizon=horizon,
            train_window_size=train_window_size,
            step=step,
            mode=mode,
            max_windows=request.max_windows,
        )

        # Собираем все train_candles одним списком.
        train_candles_list = [candles[ts:te] for ts, te in windows_indices]
        forecasts = self._get_predictions(
            "close", train_candles_list, horizon, request, precomputed_predictions,
        )

        per_window: list[dict[str, Any]] = []
        trimmed_count = 0
        required_context_seen = None

        for (train_start, train_end), forecast in zip(windows_indices, forecasts):
            actual = close_series[train_end:train_end + horizon]
            last_known_close = float(close_series[train_end - 1])

            if len(forecast) != len(actual):
                min_len = min(len(forecast), len(actual))
                forecast = forecast[:min_len]
                actual = actual[:min_len]

            window_info = self.window_usage.extract(dates=dates, train_end=train_end)
            if window_info.get("trimmed"):
                trimmed_count += 1
            if required_context_seen is None:
                required_context_seen = window_info.get("required_context_length")

            per_window.append({
                "train_start":      train_start,
                "train_end":        train_end,
                "train_end_date":   dates[train_end - 1] if dates and train_end - 1 < len(dates) else None,
                "actual":           actual,
                "forecast":         forecast,
                "last_known_close": last_known_close,
                "window_info":      window_info,
            })

        if not per_window:
            raise ValueError(
                f"После применения параметров walk-forward (train_window_size={train_window_size}, "
                f"step={step}, max_windows={request.max_windows}) ни одного окна не получилось."
            )

        # --- Предвычисление per-window метрик ОДИН раз ---
        # Bootstrap затем работает на этих скалярах, не пересчитывая от сырых баров.
        per_window_metrics = [self._compute_window_metrics_close(w, weights) for w in per_window]

        primary, ci, lcb = bootstrap_metrics(
            windows=per_window_metrics,
            aggregator=self._aggregate_close_precomputed,
            n_iters=request.bootstrap_iterations,
            z_score=request.ci_z_score,
        )

        details = []
        if verbose:
            for w in per_window:
                details.append({
                    "train_start":        w["train_start"],
                    "train_end_index":    w["train_end"],
                    "train_end_date":     w["train_end_date"],
                    "horizon":            len(w["actual"]),
                    "model_input_window": w["window_info"],
                    "actual_close":       w["actual"],
                    "forecast_close":     w["forecast"],
                })

        return BacktestResponse(
            source=source,
            model=self.model.get_info(),
            run_id=None,  # заполняется в сервисе если persist=True
            metrics=primary,
            metrics_ci=ci,
            metrics_lcb=lcb,
            windows_count=len(per_window),
            horizon=horizon,
            history_length=len(close_series),
            details=details,
            metadata=self._metadata_common(
                request=request, step=step, train_window_size=train_window_size, mode=mode,
                trimmed_count=trimmed_count, required_context_seen=required_context_seen,
                target="close",
            ),
        )

    # ------------------------------------------------------------------
    # Predcomputed per-window metrics (close)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_window_metrics_close(w: dict, weights: list[float]) -> dict[str, float]:
        """
        Считает все метрики ОДНОГО окна один раз. Возвращает плоский dict скаляров.
        Bootstrap будет ресэмплировать эти dict'ы и усреднять — никакого пересчёта
        от сырых баров.
        """
        actual = w["actual"]
        forecast = w["forecast"]
        last_known_close = w["last_known_close"]
        cur_horizon = len(actual)
        w_slice = weights[:cur_horizon]

        m = BacktestMetrics.close_metrics(actual, forecast, w_slice)
        dir_acc = BacktestMetrics._wdirectional_accuracy(actual, forecast, last_known_close, w_slice)
        naive = BacktestMetrics.naive_close_metrics(actual, last_known_close, w_slice)
        return {
            "mae":              m["mae"],
            "rmse":             m["rmse"],
            "mape":             m["mape"],
            "directional_acc":  dir_acc,
            "naive_mae":        naive["naive_mae"],
            "naive_rmse":       naive["naive_rmse"],
            "naive_mape":       naive["naive_mape"],
        }

    @staticmethod
    def _aggregate_close_precomputed(window_metrics: list[dict]) -> dict[str, float]:
        """
        Усредняет per-window метрики между окнами + считает derived metrics
        (skill_mae). Очень быстро: ~10 операций на окно.
        """
        if not window_metrics:
            return {
                "mae": 0.0, "rmse": 0.0, "mape": 0.0,
                "directional_acc": 0.0,
                "naive_mae": 0.0, "naive_rmse": 0.0, "naive_mape": 0.0,
                "skill_mae": 0.0,
            }
        n = len(window_metrics)
        out: dict[str, float] = {}
        for key in ("mae", "rmse", "mape", "directional_acc",
                    "naive_mae", "naive_rmse", "naive_mape"):
            out[key] = sum(m[key] for m in window_metrics) / n
        out["skill_mae"] = (1.0 - out["mae"] / out["naive_mae"]) if out["naive_mae"] else 0.0
        return out

    # ------------------------------------------------------------------
    # OHLC (квантильный прогноз)
    # ------------------------------------------------------------------

    def _run_ohlc(
            self, request, source, dates, candles, close_series,
            weights: list[float], step: int, train_window_size: int, mode: str,
            precomputed_predictions: list | None = None,
    ) -> BacktestResponse:
        horizon = request.horizon
        verbose = request.detailed_logs

        windows_indices = self._walk_forward_indices(
            history_len=len(close_series),
            horizon=horizon,
            train_window_size=train_window_size,
            step=step,
            mode=mode,
            max_windows=request.max_windows,
        )

        # Собираем все train_candles одним списком.
        train_candles_list = [candles[ts:te] for ts, te in windows_indices]
        qfs = self._get_predictions(
            "ohlc", train_candles_list, horizon, request, precomputed_predictions,
        )

        per_window: list[dict[str, Any]] = []
        trimmed_count = 0
        required_context_seen = None

        for (train_start, train_end), qf in zip(windows_indices, qfs):
            actual_ohlc = candles[train_end:train_end + horizon]
            actual_close = [float(c["close"]) for c in actual_ohlc]
            last_known_close = float(close_series[train_end - 1])
            forecast_ohlc = qf.median_candles()

            if len(forecast_ohlc) != len(actual_ohlc):
                min_len = min(len(forecast_ohlc), len(actual_ohlc))
                forecast_ohlc = forecast_ohlc[:min_len]
                actual_ohlc   = actual_ohlc[:min_len]
                actual_close  = actual_close[:min_len]

            window_info = self.window_usage.extract(dates=dates, train_end=train_end)
            if window_info.get("trimmed"):
                trimmed_count += 1
            if required_context_seen is None:
                required_context_seen = window_info.get("required_context_length")

            per_window.append({
                "train_start":      train_start,
                "train_end":        train_end,
                "train_end_date":   dates[train_end - 1] if dates and train_end - 1 < len(dates) else None,
                "actual_ohlc":      actual_ohlc,
                "forecast_ohlc":    forecast_ohlc,
                "actual_close":     actual_close,
                "last_known_close": last_known_close,
                "qf_close_q10":     list(qf.close.q10[: len(actual_ohlc)]),
                "qf_close_q25":     list(qf.close.q25[: len(actual_ohlc)]),
                "qf_close_q50":     list(qf.close.q50[: len(actual_ohlc)]),
                "qf_close_q75":     list(qf.close.q75[: len(actual_ohlc)]),
                "qf_close_q90":     list(qf.close.q90[: len(actual_ohlc)]),
                "window_info":      window_info,
            })

        if not per_window:
            raise ValueError(
                f"После применения параметров walk-forward (train_window_size={train_window_size}, "
                f"step={step}, max_windows={request.max_windows}) ни одного окна не получилось."
            )

        # Предвычисление per-window метрик
        per_window_metrics = [self._compute_window_metrics_ohlc(w, weights) for w in per_window]

        primary, ci, lcb = bootstrap_metrics(
            windows=per_window_metrics,
            aggregator=self._aggregate_ohlc_precomputed,
            n_iters=request.bootstrap_iterations,
            z_score=request.ci_z_score,
        )

        details = []
        if verbose:
            for w in per_window:
                details.append({
                    "train_start":        w["train_start"],
                    "train_end_index":    w["train_end"],
                    "train_end_date":     w["train_end_date"],
                    "horizon":            len(w["actual_ohlc"]),
                    "model_input_window": w["window_info"],
                    "actual_ohlc":        w["actual_ohlc"],
                    "forecast_ohlc":      w["forecast_ohlc"],
                })

        return BacktestResponse(
            source=source,
            model=self.model.get_info(),
            run_id=None,
            metrics=primary,
            metrics_ci=ci,
            metrics_lcb=lcb,
            windows_count=len(per_window),
            horizon=horizon,
            history_length=len(close_series),
            details=details,
            metadata=self._metadata_common(
                request=request, step=step, train_window_size=train_window_size, mode=mode,
                trimmed_count=trimmed_count, required_context_seen=required_context_seen,
                target="ohlc",
            ),
        )

    # ------------------------------------------------------------------
    # Precomputed per-window metrics (ohlc)
    # ------------------------------------------------------------------

    _OHLC_CHANNELS = ("open", "high", "low", "close")
    _OHLC_PER_WINDOW_KEYS = (
        "mae_open", "rmse_open", "mape_open",
        "mae_high", "rmse_high", "mape_high",
        "mae_low",  "rmse_low",  "mape_low",
        "mae_close", "rmse_close", "mape_close",
        "pinball_mean", "pinball_q50",
        "coverage_q25_q75", "coverage_q10_q90",
        "winkler_q25_q75", "winkler_q10_q90",
        "directional_acc",
        "naive_mae", "naive_rmse", "naive_mape",
    )

    @staticmethod
    def _compute_window_metrics_ohlc(w: dict, weights: list[float]) -> dict[str, float]:
        """
        Считает все метрики ОДНОГО окна (точечные по 4 каналам + квантильные по close
        + naive baseline) один раз. Возвращает плоский dict скаляров.
        """
        cur_h = len(w["actual_ohlc"])
        w_slice = weights[:cur_h]

        per_ch = BacktestMetrics.ohlc_metrics(
            w["actual_ohlc"], w["forecast_ohlc"], w_slice, BacktestRunner._OHLC_CHANNELS,
        )
        qm = BacktestMetrics.quantile_metrics_close(
            actual=w["actual_close"],
            q10=w["qf_close_q10"], q25=w["qf_close_q25"], q50=w["qf_close_q50"],
            q75=w["qf_close_q75"], q90=w["qf_close_q90"],
            last_known_close=w["last_known_close"],
            weights=w_slice,
        )
        naive = BacktestMetrics.naive_close_metrics(w["actual_close"], w["last_known_close"], w_slice)

        out: dict[str, float] = {}
        for ch in BacktestRunner._OHLC_CHANNELS:
            out[f"mae_{ch}"]  = per_ch[ch]["mae"]
            out[f"rmse_{ch}"] = per_ch[ch]["rmse"]
            out[f"mape_{ch}"] = per_ch[ch]["mape"]
        for k in ("pinball_mean", "pinball_q50",
                  "coverage_q25_q75", "coverage_q10_q90",
                  "winkler_q25_q75", "winkler_q10_q90",
                  "directional_acc"):
            out[k] = qm[k]
        for k in ("naive_mae", "naive_rmse", "naive_mape"):
            out[k] = naive[k]
        return out

    @staticmethod
    def _aggregate_ohlc_precomputed(window_metrics: list[dict]) -> dict[str, float]:
        """
        Усредняет per-window метрики + считает derived (mae_mean_ohlc, skill_mae_close).
        """
        if not window_metrics:
            out = {k: 0.0 for k in BacktestRunner._OHLC_PER_WINDOW_KEYS}
            out["mae_mean_ohlc"] = 0.0
            out["rmse_mean_ohlc"] = 0.0
            out["mape_mean_ohlc"] = 0.0
            out["skill_mae_close"] = 0.0
            return out

        n = len(window_metrics)
        out: dict[str, float] = {}
        for key in BacktestRunner._OHLC_PER_WINDOW_KEYS:
            out[key] = sum(m[key] for m in window_metrics) / n

        channels = BacktestRunner._OHLC_CHANNELS
        out["mae_mean_ohlc"]  = sum(out[f"mae_{ch}"]  for ch in channels) / len(channels)
        out["rmse_mean_ohlc"] = sum(out[f"rmse_{ch}"] for ch in channels) / len(channels)
        out["mape_mean_ohlc"] = sum(out[f"mape_{ch}"] for ch in channels) / len(channels)
        out["skill_mae_close"] = (1.0 - out["mae_close"] / out["naive_mae"]) if out["naive_mae"] else 0.0
        return out

    # ------------------------------------------------------------------
    # Метаданные ответа
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata_common(
            request, step: int, train_window_size: int, mode: str,
            trimmed_count: int, required_context_seen, target: str,
    ) -> dict[str, Any]:
        return {
            "train_window_mode":         mode,
            "train_window_size":         train_window_size,
            "step":                      step,
            "max_windows":               request.max_windows,
            "horizon":                   request.horizon,
            "evaluation_weights":        request.evaluation_weights,
            "weight_first_to_last_ratio": request.weight_first_to_last_ratio,
            "bootstrap_iterations":      request.bootstrap_iterations,
            "ci_z_score":                request.ci_z_score,
            "feature_plugins":           request.feature_plugins,
            "backtest_target":           target,
            "model_required_context_length": required_context_seen,
            "trimmed_windows_count":     trimmed_count,
            "metric_note": (
                "Все метрики ВЗВЕШЕННЫЕ по схеме evaluation_weights. "
                "Метрики ranking: skill_mae(_close) — выше=лучше; pinball, winkler — ниже=лучше; "
                "coverage_q25_q75 норма ≈ 0.50; coverage_q10_q90 норма ≈ 0.80. "
                "LCB = mean − z_score × std (нижняя граница доверительного интервала)."
            ),
        }
