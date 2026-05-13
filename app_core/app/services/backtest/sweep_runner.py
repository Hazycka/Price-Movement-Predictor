"""
BacktestSweepRunner — поиск оптимального train_window_size (контекста модели)
для данного инструмента и параметров.

Алгоритм:
  Phase A (Coarse pass)
    Лог-spaced grid [128, 256, 512, 1024, 2048, 4096, 8192], отсекаются точки
    которые больше model.context_length или len(history) - horizon.

  Phase B (Refinement pass, опционально)
    Берём top-1 из coarse, добавляем точки вокруг него с фактором 1.25.

  Phase C (Ranking & CV candidate selection)
    LCB-ранжирование по выбранной ranking_metric.
    Кандидаты на CV: те у кого ranking_lcb >= top_lcb - lcb_tie_tolerance
    (до cv_max_candidates штук).

  Phase D (Cross-validation)
    Для каждого кандидата: делим историю на K фолдов (K = min(cv_folds,
    K_max_possible_for_this_context)). Если K < 2 → cv_skipped. Иначе для
    каждого фолда запускаем walk-forward и считаем mean/std по фолдам.

  Phase E (Final recommendation)
    select_recommended: лучший cv_lcb (или primary lcb если CV пропущена),
    при равенстве — меньший train_window_size.

Все runs сохраняются в backtest_runs с общим sweep_id.
"""
from __future__ import annotations

from typing import Any

from ...schemas import (
    BacktestRequest,
    BacktestSweepRequest,
    BacktestSweepResponse,
    SweepConfigResult,
    RecommendedConfig,
    RankingMetric,
)
from ...storage import get_uow_factory
from ...storage.ports import BacktestRunRecord
from .backtest_runner import BacktestRunner
from .ranking import ranking_value, ranking_lcb, overlapping_with_top, select_recommended


# Coarse grid — log-spaced с фактором 2
_COARSE_GRID = [128, 256, 512, 1024, 2048, 4096, 8192]
# Минимальное число окон чтобы фолд CV имел смысл
_MIN_WINDOWS_PER_FOLD = 2
# Фактор для refinement точек вокруг лучшего
_REFINEMENT_FACTOR = 1.25
# Округлять контекст до patch size = 16 (требование PatchTST)
_CONTEXT_ALIGN = 16
# Ключевые метрики для brief-режима ответа sweep. Остальные ~20 метрик
# (per-channel mae/rmse/mape, winkler, naive_*) доступны через
# GET /backtest/runs/{primary_run_id}.
_BRIEF_METRIC_KEYS = (
    "skill_mae_close", "skill_mae",
    "directional_acc",
    "coverage_q10_q90",
    "pinball_mean",
)


def _align_context(c: int) -> int:
    """Округляем контекст до ближайшего кратного _CONTEXT_ALIGN, не меньше align."""
    return max(_CONTEXT_ALIGN, (c // _CONTEXT_ALIGN) * _CONTEXT_ALIGN)


class BacktestSweepRunner:
    def __init__(self, model) -> None:
        self.model = model
        self.runner = BacktestRunner(model=model)

    # ------------------------------------------------------------------
    # Точка входа
    # ------------------------------------------------------------------

    def run(
            self,
            request: BacktestSweepRequest,
            source: str,
            dates: list[str],
            candles: list[dict[str, float]],
    ) -> BacktestSweepResponse:
        history_len = len(candles)
        horizon = request.horizon
        step = request.step if request.step is not None else horizon
        ranking_metric: RankingMetric = request.ranking_metric

        model_info = self.model.get_info()
        model_max_context = int(model_info.get("context_length", 8192))

        # --- Phase 0: новый sweep_id ---
        with get_uow_factory()() as uow:
            sweep_id = uow.backtest_repository.get_next_sweep_id()

        # --- Phase A: coarse pass ---
        coarse_grid = self._build_coarse_grid(history_len, horizon, step, model_max_context)
        if not coarse_grid:
            raise ValueError(
                f"Невозможно построить coarse grid: история {history_len} баров слишком "
                f"коротка для horizon={horizon} с минимальным контекстом {_COARSE_GRID[0]}."
            )

        config_results: list[dict[str, Any]] = []
        for ctx in coarse_grid:
            result = self._run_single(
                request, source, dates, candles, train_window_size=ctx,
                pass_type="coarse", sweep_id=sweep_id, ranking_metric=ranking_metric,
            )
            config_results.append(result)

        # --- Phase B: refinement ---
        if request.enable_refinement and config_results:
            refinement_contexts = self._build_refinement_grid(
                config_results, coarse_grid, history_len, horizon, step, model_max_context,
            )
            for ctx in refinement_contexts:
                result = self._run_single(
                    request, source, dates, candles, train_window_size=ctx,
                    pass_type="refinement", sweep_id=sweep_id, ranking_metric=ranking_metric,
                )
                config_results.append(result)

        # --- Phase C+D: отбор CV-кандидатов и сам CV ---
        # Раньше эти две фазы были разделены, и из-за того что _run_single
        # инициализирует cv_status="not_selected" по умолчанию (как метку
        # «ещё не оценивался CV»), в Phase D проверка `if cv_status == "not_selected"`
        # пропускала ВСЕ конфиги — включая реальных кандидатов. Баг.
        # Теперь идём в один проход с явной проверкой членства по id().
        cv_candidates = overlapping_with_top(
            configs=config_results,
            ranking_metric=ranking_metric,
            lcb_tie_tolerance=request.lcb_tie_tolerance,
            max_candidates=request.cv_max_candidates,
        )
        cv_candidate_ids = {id(c) for c in cv_candidates}

        cv_summary = {"completed": 0, "skipped_short_history": 0, "not_selected": 0}
        for c in config_results:
            if id(c) not in cv_candidate_ids:
                c["cv_status"] = "not_selected"
                cv_summary["not_selected"] += 1
                continue
            # Это кандидат — запускаем CV. _run_cv сам обновит cv_status
            # в "completed" или "skipped_short_history".
            self._run_cv(
                request, source, dates, candles, sweep_id=sweep_id,
                config=c, ranking_metric=ranking_metric, step=step,
            )
            if c["cv_status"] == "completed":
                cv_summary["completed"] += 1
            elif c["cv_status"] == "skipped_short_history":
                cv_summary["skipped_short_history"] += 1

        # --- Phase E: финальная рекомендация ---
        recommended_config, reason, margin = select_recommended(
            configs=config_results,
            ranking_metric=ranking_metric,
            lcb_tie_tolerance=request.lcb_tie_tolerance,
        )
        recommended_lcb = recommended_config.get("cv_ranking_metric_lcb")
        if recommended_lcb is None:
            recommended_lcb = recommended_config["ranking_metric_lcb"]

        # --- Сборка ответа ---
        detailed = request.detailed_logs
        configs_response = [self._to_sweep_config_result(c, detailed=detailed) for c in config_results]
        # Сортировка по effective LCB для удобства чтения
        configs_response.sort(
            key=lambda c: c.cv_ranking_metric_lcb if c.cv_ranking_metric_lcb is not None else c.ranking_metric_lcb,
            reverse=True,
        )

        return BacktestSweepResponse(
            sweep_id=sweep_id,
            ticker=source,
            source=request.data_source,
            interval=request.interval,
            model_name=str(model_info.get("name", "unknown")),
            history_length=history_len,
            ranking_metric=ranking_metric,
            configs=configs_response,
            recommended=RecommendedConfig(
                train_window_size=recommended_config["train_window_size"],
                reason=reason,
                ranking_metric=ranking_metric,
                lcb=recommended_lcb,
                lcb_margin=margin,
            ),
            cv_summary=cv_summary,
        )

    # ------------------------------------------------------------------
    # Phase A: построение coarse grid
    # ------------------------------------------------------------------

    @staticmethod
    def _max_context_for_history(history_len: int, horizon: int, step: int) -> int:
        """
        Максимальный train_window_size при котором умещается хотя бы 1 окно.
        Окно требует train_window_size + horizon <= history_len.
        """
        return history_len - horizon

    def _build_coarse_grid(
            self, history_len: int, horizon: int, step: int, model_max_context: int,
    ) -> list[int]:
        max_useful = min(model_max_context, self._max_context_for_history(history_len, horizon, step))
        grid = [c for c in _COARSE_GRID if c <= max_useful]
        if not grid and max_useful >= _CONTEXT_ALIGN:
            # История очень короткая, но хоть один маленький контекст влезает
            grid = [_align_context(max_useful)]
        return grid

    # ------------------------------------------------------------------
    # Phase B: построение refinement grid
    # ------------------------------------------------------------------

    def _build_refinement_grid(
            self,
            coarse_results: list[dict],
            coarse_grid: list[int],
            history_len: int,
            horizon: int,
            step: int,
            model_max_context: int,
    ) -> list[int]:
        """
        Берём top-1 coarse результат, добавляем 2-3 точки вокруг него с фактором 1.25.
        Отсекаем дубликаты, точки выходящие за рамки coarse grid (бессмысленно),
        и точки превышающие model_max_context / history.
        """
        if not coarse_results:
            return []
        max_useful = min(model_max_context, self._max_context_for_history(history_len, horizon, step))

        top = max(coarse_results, key=lambda c: c["ranking_metric_lcb"])
        base = top["train_window_size"]

        candidates_raw = [
            int(base * _REFINEMENT_FACTOR),
            int(base / _REFINEMENT_FACTOR),
            int(base * (_REFINEMENT_FACTOR ** 2)),
            int(base / (_REFINEMENT_FACTOR ** 2)),
        ]
        existing = set(coarse_grid)
        result: list[int] = []
        for raw in candidates_raw:
            aligned = _align_context(raw)
            if aligned <= 0 or aligned > max_useful:
                continue
            if aligned in existing or aligned in result:
                continue
            result.append(aligned)
        result.sort()
        return result

    # ------------------------------------------------------------------
    # Запуск одного бэктеста + сохранение в БД с sweep_id
    # ------------------------------------------------------------------

    def _build_request_for_context(
            self, sweep_req: BacktestSweepRequest, train_window_size: int,
    ) -> BacktestRequest:
        """
        Создаёт обычный BacktestRequest из BacktestSweepRequest для конкретного контекста.
        persist=False — сохраняем сами через _save_run с sweep_id.
        """
        return BacktestRequest(
            model_name=sweep_req.model_name,
            model_options=sweep_req.model_options,
            data_source=sweep_req.data_source,
            provider_options=sweep_req.provider_options,
            history_period=sweep_req.history_period,
            history_up_to=sweep_req.history_up_to,
            interval=sweep_req.interval,
            horizon=sweep_req.horizon,
            feature_plugins=sweep_req.feature_plugins,
            backtest_target=sweep_req.backtest_target,
            train_window_mode=sweep_req.train_window_mode,
            train_window_size=train_window_size,
            step=sweep_req.step,
            max_windows=sweep_req.max_windows,
            evaluation_weights=sweep_req.evaluation_weights,
            weight_first_to_last_ratio=sweep_req.weight_first_to_last_ratio,
            bootstrap_iterations=sweep_req.bootstrap_iterations,
            ci_z_score=sweep_req.ci_z_score,
            inference_batch_size=sweep_req.inference_batch_size,
            persist=False,
            # Прокидываем флаг подробности в inner backtests:
            # detailed_logs=True → per-window details сохранятся в БД (полезно для дебага CV),
            # но инференс будет «шумным» — много логов TSFM.
            detailed_logs=sweep_req.detailed_logs,
        )

    def _save_run(
            self,
            sweep_req: BacktestSweepRequest,
            response,
            source: str,
            train_window_size: int,
            sweep_id: int,
            parent_run_id: int | None = None,
            cv_fold_index: int | None = None,
    ) -> int:
        meta = response.metadata
        record = BacktestRunRecord(
            model_name=response.model.get("name", "unknown"),
            ticker=source,
            source=sweep_req.data_source,
            interval=sweep_req.interval,
            has_lora=False,
            lora_artifact_id=None,
            train_window_mode=meta.get("train_window_mode", sweep_req.train_window_mode),
            train_window_size=train_window_size,
            horizon=sweep_req.horizon,
            step=meta.get("step", sweep_req.horizon),
            backtest_target=sweep_req.backtest_target,
            evaluation_weights=sweep_req.evaluation_weights,
            weight_first_to_last_ratio=sweep_req.weight_first_to_last_ratio,
            bootstrap_iterations=sweep_req.bootstrap_iterations,
            ci_z_score=sweep_req.ci_z_score,
            history_period=sweep_req.history_period,
            history_up_to=sweep_req.history_up_to,
            history_length=response.history_length,
            feature_plugins=list(sweep_req.feature_plugins),
            windows_count=response.windows_count,
            metrics=response.metrics,
            metrics_ci=response.metrics_ci,
            metrics_lcb=response.metrics_lcb,
            metadata=response.metadata,
            sweep_id=sweep_id,
            parent_run_id=parent_run_id,
            cv_fold_index=cv_fold_index,
        )
        with get_uow_factory()() as uow:
            return uow.backtest_repository.save_run(record)

    def _run_single(
            self,
            sweep_req: BacktestSweepRequest,
            source: str,
            dates: list[str],
            candles: list[dict[str, float]],
            train_window_size: int,
            pass_type: str,
            sweep_id: int,
            ranking_metric: RankingMetric,
    ) -> dict[str, Any]:
        """Один primary run для конкретного контекста."""
        bt_req = self._build_request_for_context(sweep_req, train_window_size)
        response = self.runner.run(bt_req, source, dates, candles)
        run_id = self._save_run(sweep_req, response, source, train_window_size, sweep_id)

        mean_val = response.metrics.get(ranking_metric, 0.0)
        ci = response.metrics_ci.get(ranking_metric, [mean_val, mean_val])

        return {
            "train_window_size":    train_window_size,
            "pass_type":            pass_type,
            "primary_run_id":       run_id,
            "windows_count":        response.windows_count,
            "metrics":              response.metrics,
            "metrics_ci":           response.metrics_ci,
            "metrics_lcb":          response.metrics_lcb,
            "ranking_metric_value": ranking_value(ranking_metric, mean_val),
            "ranking_metric_lcb":   ranking_lcb(ranking_metric, mean_val, ci[0], ci[1]),
            "cv_status":            "not_selected",
            "cv_folds_used":        None,
            "cv_metrics_mean":      None,
            "cv_metrics_std":       None,
            "cv_ranking_metric_lcb": None,
        }

    # ------------------------------------------------------------------
    # Phase D: CV для одного кандидата
    # ------------------------------------------------------------------

    def _run_cv(
            self,
            sweep_req: BacktestSweepRequest,
            source: str,
            dates: list[str],
            candles: list[dict[str, float]],
            sweep_id: int,
            config: dict[str, Any],
            ranking_metric: RankingMetric,
            step: int,
    ) -> None:
        """
        Запускает CV для одного кандидата. Обновляет config in-place.

        Оптимизация: вместо K независимых вызовов модели (по одному на фолд)
        собираем все train_candles ВСЕХ фолдов в один список и делаем ОДИН
        batched-вызов predict_*_batch. Затем раздаём предсказания по фолдам и
        агрегируем метрики per-fold.

        Это экономит K-1 проходов TSFM pre/postprocessing на CPU — главный
        источник «idle gap» между GPU-пиками.
        """
        history_len = len(candles)
        horizon = sweep_req.horizon
        train_window_size = config["train_window_size"]
        mode = sweep_req.train_window_mode

        # --- Подбираем K (уменьшаем пока не помещается) ---
        K = sweep_req.cv_folds
        chosen_K = 0
        while K >= 2:
            fold_size = history_len // K
            min_required = train_window_size + horizon + step * (_MIN_WINDOWS_PER_FOLD - 1)
            if fold_size >= min_required:
                chosen_K = K
                break
            K -= 1

        if chosen_K < 2:
            config["cv_status"] = "skipped_short_history"
            config["cv_folds_used"] = 0
            return

        bt_req = self._build_request_for_context(sweep_req, train_window_size)
        fold_size = history_len // chosen_K

        # --- Phase 1: для каждого фолда строим indices и train_candles_list ---
        # Сохраняем достаточно информации чтобы потом сагрегировать per-fold.
        fold_meta: list[dict[str, Any]] = []
        all_train_candles: list[list[dict[str, float]]] = []

        for fold_idx in range(chosen_K):
            start = fold_idx * fold_size
            end = history_len if fold_idx == chosen_K - 1 else start + fold_size
            fold_candles = candles[start:end]
            fold_dates = dates[start:end]

            fold_indices = BacktestRunner._walk_forward_indices(
                history_len=len(fold_candles),
                horizon=horizon,
                train_window_size=train_window_size,
                step=step,
                mode=mode,
                max_windows=sweep_req.max_windows,
            )
            if not fold_indices:
                continue

            fold_train_candles = [fold_candles[ts:te] for ts, te in fold_indices]
            fold_meta.append({
                "fold_idx":          fold_idx,
                "fold_dates":        fold_dates,
                "fold_candles":      fold_candles,
                "n_windows":         len(fold_train_candles),
                "offset_in_batch":   len(all_train_candles),
            })
            all_train_candles.extend(fold_train_candles)

        if not all_train_candles or len(fold_meta) < 2:
            config["cv_status"] = "skipped_short_history"
            config["cv_folds_used"] = len(fold_meta)
            return

        # --- Phase 2: ОДИН batched-вызов модели на все окна всех фолдов ---
        ctx = {
            "model_options":         sweep_req.model_options,
            "feature_plugins":       sweep_req.feature_plugins,
            "mode":                  "backtest",
            "backtest_target":       sweep_req.backtest_target,
            "detailed_logs":         False,
            "inference_batch_size":  sweep_req.inference_batch_size,
        }
        try:
            if sweep_req.backtest_target == "ohlc":
                all_predictions = self.model.predict_ohlc_quantiles_batch(
                    all_train_candles, horizon, ctx,
                )
            else:
                all_predictions = self.model.predict_line_exact_batch(
                    all_train_candles, horizon, ctx,
                )
        except NotImplementedError:
            # Модель не поддерживает batched-метод. Fallback: пер-фолд по старому.
            all_predictions = None

        # --- Phase 3: распределяем предсказания по фолдам, агрегируем per-fold ---
        per_fold_metrics: list[dict[str, float]] = []
        for fm in fold_meta:
            if all_predictions is not None:
                start_off = fm["offset_in_batch"]
                end_off = start_off + fm["n_windows"]
                fold_predictions = all_predictions[start_off:end_off]
            else:
                fold_predictions = None  # fallback: runner вызовет модель сам

            try:
                fold_response = self.runner.run(
                    bt_req, source, fm["fold_dates"], fm["fold_candles"],
                    precomputed_predictions=fold_predictions,
                )
            except ValueError:
                continue

            self._save_run(
                sweep_req, fold_response, source, train_window_size, sweep_id,
                parent_run_id=config["primary_run_id"], cv_fold_index=fm["fold_idx"],
            )
            per_fold_metrics.append(fold_response.metrics)

        if len(per_fold_metrics) < 2:
            config["cv_status"] = "skipped_short_history"
            config["cv_folds_used"] = len(per_fold_metrics)
            return

        # Агрегация фолдов: mean и std по каждой метрике
        mean_by_name: dict[str, float] = {}
        std_by_name: dict[str, float] = {}
        all_keys = set()
        for m in per_fold_metrics:
            all_keys.update(m.keys())
        for key in all_keys:
            vals = [m.get(key, 0.0) for m in per_fold_metrics]
            n = len(vals)
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n
            mean_by_name[key] = mean
            std_by_name[key] = var ** 0.5

        # CV-LCB по ranking metric: mean - z * std
        cv_mean = mean_by_name.get(ranking_metric, 0.0)
        cv_std = std_by_name.get(ranking_metric, 0.0)
        cv_ci_low = cv_mean - sweep_req.ci_z_score * cv_std
        cv_ci_high = cv_mean + sweep_req.ci_z_score * cv_std

        config["cv_status"] = "completed"
        config["cv_folds_used"] = len(per_fold_metrics)
        config["cv_metrics_mean"] = mean_by_name
        config["cv_metrics_std"] = std_by_name
        config["cv_ranking_metric_lcb"] = ranking_lcb(ranking_metric, cv_mean, cv_ci_low, cv_ci_high)

    # ------------------------------------------------------------------
    # Конвертация словаря-конфига в Pydantic-модель ответа
    # ------------------------------------------------------------------

    @staticmethod
    def _to_sweep_config_result(c: dict[str, Any], detailed: bool) -> SweepConfigResult:
        """
        Конвертирует внутренний словарь конфига в Pydantic-модель ответа.

        detailed=False — heavy-словари обнуляются, в metrics остаются только
        ключевые метрики (skill_mae_close, directional_acc, coverage_q10_q90,
        pinball_mean). Это даёт ответ ~10× компактнее.
        Полные данные доступны через GET /backtest/runs/{primary_run_id}.
        """
        if detailed:
            metrics = c["metrics"]
            metrics_ci = c["metrics_ci"]
            metrics_lcb = c["metrics_lcb"]
            cv_metrics_mean = c["cv_metrics_mean"]
            cv_metrics_std = c["cv_metrics_std"]
        else:
            metrics = {k: v for k, v in c["metrics"].items() if k in _BRIEF_METRIC_KEYS}
            metrics_ci = None
            metrics_lcb = None
            cv_metrics_mean = None
            cv_metrics_std = None

        return SweepConfigResult(
            train_window_size=c["train_window_size"],
            pass_type=c["pass_type"],
            primary_run_id=c["primary_run_id"],
            windows_count=c["windows_count"],
            metrics=metrics,
            metrics_ci=metrics_ci,
            metrics_lcb=metrics_lcb,
            ranking_metric_value=c["ranking_metric_value"],
            ranking_metric_lcb=c["ranking_metric_lcb"],
            cv_status=c["cv_status"],
            cv_folds_used=c["cv_folds_used"],
            cv_metrics_mean=cv_metrics_mean,
            cv_metrics_std=cv_metrics_std,
            cv_ranking_metric_lcb=c["cv_ranking_metric_lcb"],
        )
