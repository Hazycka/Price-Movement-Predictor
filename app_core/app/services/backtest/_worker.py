"""
Top-level функции для worker-процессов параллельного sweep.

Каждый worker-процесс:
  1. При первом таске инициализирует storage (init_storage) — для записи runs в БД
  2. При первом таске лениво загружает свою копию модели + BacktestRunner
  3. Получает payload через ProcessPoolExecutor.submit, прогоняет backtest,
     сохраняет run в БД (если sweep_id передан), возвращает компактный результат

Все функции/классы здесь — top-level (не nested), потому что spawn-метод
multiprocessing требует pickle-сериализации для передачи в child процесс.

State worker'а (модель, факт инициализации storage) хранится в module-level
переменных. У каждого процесса свой экземпляр модуля → свои переменные.
"""
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any


# Per-worker singletons. Каждый процесс имеет свою копию этого модуля.
_RUNNER = None
_STORAGE_INITIALIZED = False
# Shared multiprocessing.Lock устанавливается parallel_pool._worker_initializer
# при старте каждого воркера. Используется в _ensure_runner для сериализации
# загрузки модели между параллельными воркерами (защита от VRAM-спайка).
# None — лок не передавался (одиночный воркер или legacy путь).
_MODEL_LOAD_LOCK = None

# Лог-файл для диагностики worker-крашей. Каждый воркер пишет в него ID PID
# и текущий шаг — если воркер умрёт на C-уровне (CUDA TDR / segfault),
# в логе будет видна последняя строчка ДО краша.
_WORKER_LOG = Path("worker_diagnostic.log")


def _log_step(msg: str) -> None:
    """Атомарная запись строки в файл worker_diagnostic.log."""
    try:
        with open(_WORKER_LOG, "a", encoding="utf-8") as f:
            f.write(f"[pid={os.getpid()} t={time.strftime('%H:%M:%S')}] {msg}\n")
            f.flush()
    except Exception:
        pass  # не падаем из-за логирования


def init_worker() -> None:
    """
    Идемпотентная инициализация storage. Вызывается из _worker_initializer
    (один раз при старте воркера) и из run_backtest_task (защита если воркер
    создан без initializer'а — теоретический legacy путь).
    """
    global _STORAGE_INITIALIZED
    if _STORAGE_INITIALIZED:
        return
    _log_step("init_worker: starting init_storage")
    from ...storage import init_storage
    init_storage()
    _STORAGE_INITIALIZED = True
    _log_step("init_worker: done")


def _ensure_runner(model_name: str | None):
    """
    Лениво создаёт BacktestRunner с моделью при первом вызове.
    Кешируется per-worker — повторные таски используют ту же модель.

    Если установлен _MODEL_LOAD_LOCK (shared между всеми воркерами пула),
    загрузка модели сериализуется — N воркеров грузят модели по очереди,
    а не одновременно. Это критично для GPU с малым VRAM: одновременная
    загрузка 2-3 моделей легко вызывает CUDA OOM.

    Lock берётся ТОЛЬКО на время первой загрузки в данном воркере. После
    этого _RUNNER уже инициализирован и последующие вызовы lock не трогают.
    """
    global _RUNNER
    if _RUNNER is not None:
        return _RUNNER

    if _MODEL_LOAD_LOCK is not None:
        _log_step(f"_ensure_runner: waiting for model_load_lock (model={model_name})")
        with _MODEL_LOAD_LOCK:
            _log_step("_ensure_runner: acquired lock, loading model")
            # double-checked locking: другой воркер мог успеть загрузить пока ждали
            if _RUNNER is None:
                _RUNNER = _load_runner(model_name)
            _log_step("_ensure_runner: model loaded, releasing lock")
    else:
        _log_step(f"_ensure_runner: no lock, loading model (model={model_name})")
        _RUNNER = _load_runner(model_name)
        _log_step("_ensure_runner: model loaded")

    return _RUNNER


def _load_runner(model_name: str | None):
    """Фактическая загрузка модели + BacktestRunner. Под защитой lock."""
    _log_step(f"_load_runner: importing factories")
    from ...models.factory import create_model
    from .backtest_runner import BacktestRunner
    _log_step(f"_load_runner: create_model({model_name})")
    model = create_model(model_name)
    _log_step(f"_load_runner: BacktestRunner init")
    runner = BacktestRunner(model=model)
    _log_step(f"_load_runner: done")
    return runner


def run_backtest_task(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Запускает один backtest в worker-процессе и (опционально) сохраняет run в БД.

    payload содержит:
      request_dict       — dict от BacktestRequest.model_dump()
      source             — тикер / FIGI / csv-путь (из InputResolver)
      dates              — список дат
      candles            — список свечей
      ticker             — для сохранения в record.ticker
      sweep_id           — int | None
      parent_run_id      — int | None (для CV-фолдов)
      cv_fold_index      — int | None

    Возвращает dict с двумя ключами:
      response — dict от BacktestResponse.model_dump()
      run_id   — int | None (если sweep_id задан и save_run прошёл)
    """
    ctx_label = f"ctx={payload.get('request_dict', {}).get('train_window_size', '?')}"
    _log_step(f"run_backtest_task START {ctx_label}")
    try:
        init_worker()

        from ...schemas import BacktestRequest
        request = BacktestRequest.model_validate(payload["request_dict"])

        # Внутри worker'а persist всегда False — сохранение делаем сами с sweep_id.
        request_local = request.model_copy(update={"persist": False})

        _log_step(f"run_backtest_task ensuring runner {ctx_label} artifact_id={request.artifact_id}")
        runner = _ensure_runner(request_local.model_name)

        _log_step(f"run_backtest_task calling runner.run {ctx_label}")
        response = runner.run(
            request=request_local,
            source=payload["source"],
            dates=payload["dates"],
            candles=payload["candles"],
        )
        _log_step(f"run_backtest_task runner.run done {ctx_label}")

        run_id: int | None = None
        if payload.get("sweep_id") is not None:
            _log_step(f"run_backtest_task saving run {ctx_label}")
            run_id = _save_run_in_worker(request, response, payload)
            response.run_id = run_id
            _log_step(f"run_backtest_task saved run_id={run_id} {ctx_label}")

        _log_step(f"run_backtest_task DONE {ctx_label}")
        return {
            "response": response.model_dump(),
            "run_id":   run_id,
        }
    except BaseException as ex:
        # Логируем ЛЮБОЕ исключение, включая SystemExit / KeyboardInterrupt,
        # чтобы если воркер умирает по любой причине — у нас был след в файле.
        _log_step(f"run_backtest_task EXCEPTION {ctx_label}: {type(ex).__name__}: {ex}")
        _log_step(traceback.format_exc())
        raise


def _save_run_in_worker(request, response, payload: dict[str, Any]) -> int:
    """Сохраняет результат backtest в backtest_runs с sweep-метаданными."""
    from ...storage import get_uow_factory
    from ...storage.ports import BacktestRunRecord

    meta = response.metadata

    # Подтягиваем applied_components из артефакта (если был)
    artifact_id = getattr(request, "artifact_id", None)
    applied_components: list[str] = []
    if artifact_id is not None:
        with get_uow_factory()() as uow:
            artifact = uow.model_registry.get_by_id(artifact_id)
        if artifact is not None:
            applied_components = list(artifact.training_components or [])

    record = BacktestRunRecord(
        model_name=response.model.get("name", "unknown"),
        ticker=payload["ticker"],
        source=request.data_source,
        interval=request.interval,
        artifact_id=artifact_id,
        applied_components=applied_components,
        train_window_mode=meta.get("train_window_mode", request.train_window_mode),
        train_window_size=meta.get("train_window_size", request.train_window_size),
        horizon=request.horizon,
        step=meta.get("step", request.horizon),
        backtest_target=request.backtest_target,
        evaluation_weights=request.evaluation_weights,
        weight_first_to_last_ratio=request.weight_first_to_last_ratio,
        bootstrap_iterations=request.bootstrap_iterations,
        ci_z_score=request.ci_z_score,
        history_period=request.history_period,
        history_up_to=request.history_up_to,
        history_length=response.history_length,
        feature_plugins=list(request.feature_plugins),
        windows_count=response.windows_count,
        metrics=response.metrics,
        metrics_ci=response.metrics_ci,
        metrics_lcb=response.metrics_lcb,
        metadata=response.metadata,
        sweep_id=payload["sweep_id"],
        parent_run_id=payload.get("parent_run_id"),
        cv_fold_index=payload.get("cv_fold_index"),
    )
    with get_uow_factory()() as uow:
        return uow.backtest_repository.save_run(record)
