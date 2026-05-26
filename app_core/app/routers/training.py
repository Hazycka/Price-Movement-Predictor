"""
Training endpoints (async): head, head→input, head→[input]→lora.

  POST /training/head     → job_id (head или head→input цепочка)
  POST /training/lora     → job_id (head→[input]→lora цепочка с reuse артефактов)
  GET  /training/jobs                       → список текущих job'ов
  GET  /training/jobs/{job_id}              → статус + multi-stage progress + логи
  POST /training/jobs/{job_id}/cancel       → запросить отмену
  GET  /artifacts                           → список артефактов в БД
  GET  /artifacts/{artifact_id}             → детали одного

JobStore — in-memory. При перезапуске сервера выполняемые задачи теряются
(статус становится 'orphan'), но артефакты в БД останутся.
"""
from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from ..schemas import HeadTrainingRequest, LoraTrainingRequest
from ..services.training.exceptions import (
    ArtifactHyperparamsConflictError, TrainingCancelledException,
)
from ..services.training.service import TrainingService, TrainingRequestInternal
from ..services.training.job_store import TrainingJobStore
from ..storage import get_uow_factory
from ._common import endpoint_errors

router = APIRouter(tags=["training"])

logger = logging.getLogger(__name__)
_store = TrainingJobStore()


# --------------------------------------------------------------------------
# Training (async)
# --------------------------------------------------------------------------

@router.post("/training/head", status_code=202)
@endpoint_errors("Head training submission failed")
def train_head(request: HeadTrainingRequest, background_tasks: BackgroundTasks) -> dict:
    """
    Запускает обучение head в background.
    Если train_input_too=True — цепочка head→input (2 стадии).
    """
    total_stages = 2 if request.train_input_too else 1
    job_id = _store.create(
        "head", request_summary=_summary(request), total_stages=total_stages,
    )
    background_tasks.add_task(_run_head_job, job_id, request)
    return {"job_id": job_id, "status": "pending", "kind": "head", "total_stages": total_stages}


@router.post("/training/lora", status_code=202)
@endpoint_errors("LoRA training submission failed")
def train_lora(request: LoraTrainingRequest, background_tasks: BackgroundTasks) -> dict:
    """
    Запускает обучение LoRA в background. Полная цепочка:
      [find_or_train head] → [find_or_train input?] → train lora.

    total_stages динамическое: 1..3 (зависит от того, нашлись ли head/input).
    Сейчас на момент submit'а ещё не знаем — поэтому ставим верхнюю оценку
    (3 если train_input_too=True, иначе 2), а в processing'е оркестратор
    обновит реальное количество выполненных стейджей.
    """
    total_stages = 3 if request.train_input_too else 2
    job_id = _store.create(
        "lora", request_summary=_summary(request), total_stages=total_stages,
    )
    background_tasks.add_task(_run_lora_job, job_id, request)
    return {"job_id": job_id, "status": "pending", "kind": "lora", "total_stages": total_stages}


# --------------------------------------------------------------------------
# Status / cancel
# --------------------------------------------------------------------------

@router.get("/training/jobs")
def list_jobs() -> dict:
    """Все training-задачи в текущей сессии сервера (RAM, теряется при рестарте)."""
    jobs = _store.list_all()
    return {
        "count": len(jobs),
        "jobs": [_job_to_dict(j, with_logs=False) for j in jobs],
    }


@router.get("/training/jobs/{job_id}")
def get_job(job_id: str, with_logs: bool = Query(True)) -> dict:
    """Статус задачи: pending → running → completed/failed/cancelled."""
    job = _store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job_id={job_id} не найден (или сервер был перезапущен)")
    return _job_to_dict(job, with_logs=with_logs)


@router.post("/training/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """
    Запрашивает отмену job'а. Trainer корректно остановится между батчами
    (~5-30 секунд). Уже завершённые стейджи (head/input артефакты) остаются
    в БД как ready — их можно переиспользовать при повторном запуске.

    Возвращает:
      cancel_accepted=true  — запрос принят, job будет остановлен
      cancel_accepted=false — job уже завершён (completed/failed/cancelled) или не найден
    """
    job = _store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job_id={job_id} не найден")
    accepted = _store.request_cancel(job_id)
    if accepted:
        _store.append_log(job_id, "Cancel запрошен пользователем — trainer остановится на ближайшей проверке между батчами")
    return {
        "job_id":          job_id,
        "cancel_accepted": accepted,
        "current_status":  job.status,
        "current_stage":   job.stage,
    }


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------

@router.get("/artifacts")
@endpoint_errors("Ошибка получения списка артефактов")
def list_artifacts(
        symbol:   str | None = Query(None, description="Фильтр по тикеру"),
        source:   str | None = Query(None, description="Фильтр по провайдеру (t_invest/yahoo/csv)"),
        interval: str | None = Query(None, description="Фильтр по интервалу"),
        status:   str | None = Query(None, description="Фильтр по статусу (ready/training/failed)"),
) -> dict:
    """Все артефакты в реестре. Фильтры опциональны."""
    with get_uow_factory()() as uow:
        records = uow.model_registry.list_all()
    filtered = [
        r for r in records
        if (symbol is None or r.symbol.upper() == symbol.upper())
        and (source is None or r.source == source)
        and (interval is None or r.interval == interval)
        and (status is None or r.status == status)
    ]
    return {
        "count": len(filtered),
        "artifacts": [_artifact_to_dict(r) for r in filtered],
    }


@router.get("/artifacts/{artifact_id}")
@endpoint_errors("Ошибка получения артефакта")
def get_artifact(artifact_id: int) -> dict:
    """Один артефакт с полным набором полей (params, metrics)."""
    with get_uow_factory()() as uow:
        record = uow.model_registry.get_by_id(artifact_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"artifact_id={artifact_id} не найден")
    return _artifact_to_dict(record, detailed=True)


# --------------------------------------------------------------------------
# Background job runners
# --------------------------------------------------------------------------

def _make_progress_callback(job_id: str):
    """
    Превращает структурированный payload trainer'а в строку лога + обновляет
    stage_progress в JobStore.

    Payload может содержать поле "stage" (head/input/lora) — оно проставляется
    оркестратором через _wrap_progress (trainer об этом не знает).
    """
    def _cb(payload: dict) -> None:
        phase = payload.get("phase", "?")
        stage = payload.get("stage")
        stage_prefix = f"[{stage}] " if stage else ""

        # Обновляем stage_progress внутри текущего стейджа
        if phase == "train":
            stage_total_epochs = payload.get("stage_total_epochs") or payload.get("total_epochs") or 1
            epoch = payload.get("epoch", 1)
            batch = payload.get("batch", 1)
            total_batches_per_epoch = payload.get("total_batches_per_epoch", 1)
            # Доля прогресса: завершённые эпохи + доля текущей эпохи
            progress = ((epoch - 1) + batch / max(total_batches_per_epoch, 1)) / max(stage_total_epochs, 1)
            _store.update_stage_progress(job_id, progress)
        elif phase == "epoch_end":
            stage_total_epochs = payload.get("stage_total_epochs") or payload.get("total_epochs") or 1
            epoch = payload.get("epoch", 1)
            _store.update_stage_progress(job_id, epoch / max(stage_total_epochs, 1))

        # Формат строки лога
        if phase == "start":
            line = (
                f"{stage_prefix}START | epochs={payload.get('total_epochs')} | "
                f"total_batches={payload.get('total_batches')} | "
                f"trainable={payload.get('trainable_params')} | "
                f"device={payload.get('device')}"
            )
        elif phase == "train":
            line = (
                f"{stage_prefix}epoch {payload['epoch']}/{payload['total_epochs']} "
                f"batch {payload['batch']}/{payload['total_batches_per_epoch']} "
                f"(overall {payload['global_batch']}/{payload['total_batches']} "
                f"{100 * payload['global_batch'] / payload['total_batches']:.0f}%) "
                f"loss={payload['loss']:.4f} running={payload['running_avg_loss']:.4f} "
                f"elapsed={_fmt_dur(payload['elapsed_s'])} eta={_fmt_dur(payload['eta_s'])}"
            )
        elif phase == "epoch_end":
            line = (
                f"{stage_prefix}epoch {payload['epoch']}/{payload['total_epochs']} done | "
                f"train_loss={payload['train_loss']:.4f} val_loss={payload['val_loss']:.4f} | "
                f"elapsed={_fmt_dur(payload['elapsed_s'])}"
            )
        else:
            line = f"{stage_prefix}{phase}: {payload}"
        _store.append_log(job_id, line)

    return _cb


def _make_stage_callback(job_id: str):
    """Оркестратор вызывает begin_stage(name) когда переходит к новому компоненту."""
    def _cb(stage_name: str) -> None:
        _store.begin_stage(job_id, stage_name)
        _store.append_log(job_id, f"=== Begin stage: {stage_name} ===")
    return _cb


def _make_stage_finished_callback(job_id: str):
    """Оркестратор вызывает finish_stage(name, artifact_id) когда стейдж сохранён в БД."""
    def _cb(stage_name: str, artifact_id: int) -> None:
        _store.finish_stage(job_id, artifact_id)
        _store.append_log(job_id, f"=== Stage {stage_name} done → artifact_id={artifact_id} ===")
    return _cb


def _make_cancel_check(job_id: str):
    """Trainer'ы дёргают это между батчами; True означает — нужно остановиться."""
    def _cb() -> bool:
        return _store.is_cancel_requested(job_id)
    return _cb


def _fmt_dur(sec: float) -> str:
    if sec < 60:
        return f"{int(sec)}s"
    if sec < 3600:
        return f"{int(sec // 60)}m{int(sec % 60):02d}s"
    return f"{int(sec // 3600)}h{int((sec % 3600) // 60):02d}m"


def _run_head_job(job_id: str, request: HeadTrainingRequest) -> None:
    """Background-задача обучения head (опционально с цепочкой input)."""
    _store.mark_running(job_id)
    _store.append_log(job_id, f"Starting head training (train_input_too={request.train_input_too})")
    try:
        req_int = _to_internal(request)
        result = TrainingService.train_head(
            req_int,
            train_input_too=request.train_input_too,
            progress_callback=_make_progress_callback(job_id),
            cancel_check=_make_cancel_check(job_id),
            stage_callback=_make_stage_callback(job_id),
            stage_finished_callback=_make_stage_finished_callback(job_id),
        )
        _store.append_log(job_id, f"Done: artifact_id={result['artifact_id']}")
        _store.mark_completed(job_id, artifact_id=result["artifact_id"])
    except TrainingCancelledException as ex:
        _store.append_log(job_id, f"CANCELLED: {ex}")
        _store.mark_cancelled(job_id)
    except Exception as ex:
        logger.exception("[train_head_job] failed")
        _store.append_log(job_id, f"FAILED: {ex}")
        _store.append_log(job_id, traceback.format_exc())
        _store.mark_failed(job_id, error=str(ex))


def _run_lora_job(job_id: str, request: LoraTrainingRequest) -> None:
    """Background-задача обучения LoRA (цепочка head→[input]→lora)."""
    _store.mark_running(job_id)
    _store.append_log(
        job_id,
        f"Starting LoRA pipeline (r={request.lora_r}, alpha={request.lora_alpha}, "
        f"train_input_too={request.train_input_too}, force_new_head_or_input={request.force_new_head_or_input})",
    )
    try:
        req_int = _to_internal(request)
        result = TrainingService.train_lora(
            req_int, request,
            progress_callback=_make_progress_callback(job_id),
            cancel_check=_make_cancel_check(job_id),
            stage_callback=_make_stage_callback(job_id),
            stage_finished_callback=_make_stage_finished_callback(job_id),
        )
        _store.append_log(
            job_id,
            f"Done: lora_artifact_id={result['artifact_id']}, "
            f"head_artifact_id={result.get('head_artifact_id')}, "
            f"input_artifact_id={result.get('input_artifact_id')}",
        )
        _store.mark_completed(job_id, artifact_id=result["artifact_id"])
    except ArtifactHyperparamsConflictError as ex:
        # Доменное 409 — пользователь должен явно решить
        logger.warning("[train_lora_job] hyperparams conflict: %s", ex)
        _store.append_log(job_id, f"CONFLICT 409: {ex}")
        # Помечаем как failed с понятной структурированной ошибкой
        _store.mark_failed(
            job_id,
            error=(
                f"Hyperparams conflict on {ex.component} artifact #{ex.existing_artifact_id}. "
                f"Mismatched: {ex.mismatched_fields}. "
                f"Existing params: {ex.existing_params}. "
                f"Requested: {ex.requested_params}. "
                f"To force retraining: pass force_new_head_or_input=true + new_head_or_input_version='vX'."
            ),
        )
    except TrainingCancelledException as ex:
        _store.append_log(job_id, f"CANCELLED: {ex}")
        _store.mark_cancelled(job_id)
    except Exception as ex:
        logger.exception("[train_lora_job] failed")
        _store.append_log(job_id, f"FAILED: {ex}")
        _store.append_log(job_id, traceback.format_exc())
        _store.mark_failed(job_id, error=str(ex))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _to_internal(request) -> TrainingRequestInternal:
    return TrainingRequestInternal(
        model_name=request.model_name,
        data_source=request.data_source,
        provider_options=request.provider_options,
        interval=request.interval,
        history_period=request.history_period,
        history_up_to=request.history_up_to,
        train_window_size=request.train_window_size,
        horizon=request.horizon,
        step=request.step,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        num_epochs=request.num_epochs,
        val_split=request.val_split,
        evaluation_weights=request.evaluation_weights,
        weight_first_to_last_ratio=request.weight_first_to_last_ratio,
        version=request.version,
        num_workers=request.num_workers,
    )


def _summary(request) -> dict:
    """Краткое описание запроса для записи в JobStore.request_summary."""
    base = {
        "model_name":        request.model_name,
        "data_source":       request.data_source,
        "interval":          request.interval,
        "history_period":    request.history_period,
        "train_window_size": request.train_window_size,
        "horizon":           request.horizon,
        "num_epochs":        request.num_epochs,
        "version":           request.version,
    }
    if hasattr(request, "train_input_too"):
        base["train_input_too"] = request.train_input_too
    if hasattr(request, "lora_r"):
        base["lora_r"]                     = request.lora_r
        base["lora_alpha"]                 = request.lora_alpha
        base["force_new_head_or_input"]    = request.force_new_head_or_input
        base["new_head_or_input_version"]  = request.new_head_or_input_version
    return base


def _job_to_dict(job, with_logs: bool) -> dict:
    base = {
        "job_id":      job.job_id,
        "kind":        job.kind,
        "status":      job.status,
        "started_at":  _iso_ts(job.started_at),
        "finished_at": _iso_ts(job.finished_at),
        "duration_s":  _duration(job.started_at, job.finished_at),
        "artifact_id": job.artifact_id,
        "error":       job.error,
        "request":     job.request_summary,
        # Multi-stage progress
        "stage":            job.stage,
        "stage_progress":   round(job.stage_progress, 4),
        "stages_completed": job.stages_completed,
        "total_stages":     job.total_stages,
        "completed_artifact_ids": list(job.completed_artifact_ids),
        # Cancel
        "cancel_requested": job.cancel_requested,
    }
    if with_logs:
        base["logs"] = list(job.logs)
    return base


def _iso_ts(epoch_sec: float | None) -> str | None:
    if epoch_sec is None:
        return None
    return datetime.fromtimestamp(epoch_sec, tz=timezone.utc).isoformat(timespec="seconds")


def _duration(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return round(end - start, 3)


def _artifact_to_dict(record, detailed: bool = False) -> dict:
    base = {
        "id":                  record.id,
        "symbol":              record.symbol,
        "source":              record.source,
        "market":              record.market,
        "interval":            record.interval,
        "model_name":          record.model_name,
        "training_components": record.training_components,
        "train_window_size":   record.train_window_size,
        "version":             record.version,
        "status":              record.status,
        "created_at":          record.created_at,
    }
    if detailed:
        base.update({
            "artifact_path": record.artifact_path,
            "metrics":       record.metrics,
            "params":        record.params,
        })
    return base
