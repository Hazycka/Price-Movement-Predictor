"""
Training endpoints (async): head, LoRA, combo.

Все обучающие эндпоинты теперь работают через FastAPI BackgroundTasks:
  POST /training/head → возвращает {job_id, status: "pending"} немедленно
  GET  /training/jobs/{job_id} → status + artifact_id + logs

In-memory JobStore хранит состояние. При перезапуске сервера выполняемые
задачи теряются (статус становится 'orphan'), но артефакты в БД останутся.

Также эндпоинты просмотра артефактов: GET /artifacts, GET /artifacts/{id}.
"""
from __future__ import annotations

import logging
import traceback
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from ..schemas import HeadTrainingRequest, LoraTrainingRequest
from ..services.training.service import TrainingService, TrainingRequestInternal
from ..services.training.job_store import TrainingJobStore
from ..storage import get_uow_factory
from ._common import endpoint_errors

router = APIRouter(tags=["training"])

logger = logging.getLogger(__name__)
_store = TrainingJobStore()


# --------------------------------------------------------------------------
# Обучение (async)
# --------------------------------------------------------------------------

@router.post("/training/head", status_code=202)
@endpoint_errors("Head training submission failed")
def train_head(request: HeadTrainingRequest, background_tasks: BackgroundTasks) -> dict:
    """
    Запускает обучение новой output head (linear probing) в **background task**.
    Возвращает job_id для опроса статуса через GET /training/jobs/{job_id}.

    На SBER 1h × 5 лет с 5 эпохами занимает ~5-15 минут на RTX 4070 Ti.
    """
    job_id = _store.create("head", request_summary=_summary(request))
    background_tasks.add_task(_run_head_job, job_id, request)
    return {"job_id": job_id, "status": "pending", "kind": "head"}


@router.post("/training/lora", status_code=202)
@endpoint_errors("LoRA training submission failed")
def train_lora(request: LoraTrainingRequest, background_tasks: BackgroundTasks) -> dict:
    """
    Запускает обучение LoRA-адаптера (+опционально новой head) в **background task**.
    Возвращает job_id для опроса статуса.

    На SBER 1h × 5 лет с 5 эпохами и rank=8 занимает ~10-25 минут.
    """
    job_id = _store.create("lora", request_summary=_summary(request))
    background_tasks.add_task(_run_lora_job, job_id, request)
    return {"job_id": job_id, "status": "pending", "kind": "lora"}


# --------------------------------------------------------------------------
# Опрос статуса
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
    """
    Статус задачи: pending → running → completed/failed.
    При completed: artifact_id для дальнейшего использования.
    При failed: error с описанием.
    """
    job = _store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job_id={job_id} не найден (или сервер был перезапущен)")
    return _job_to_dict(job, with_logs=with_logs)


# --------------------------------------------------------------------------
# Просмотр артефактов
# --------------------------------------------------------------------------

@router.get("/artifacts")
@endpoint_errors("Ошибка получения списка артефактов")
def list_artifacts(
        symbol:   str | None = Query(None, description="Фильтр по тикеру"),
        interval: str | None = Query(None, description="Фильтр по интервалу"),
        status:   str | None = Query(None, description="Фильтр по статусу (ready/training/failed)"),
) -> dict:
    """Все артефакты в реестре. Фильтры опциональны."""
    with get_uow_factory()() as uow:
        records = uow.model_registry.list_all()
    filtered = [
        r for r in records
        if (symbol is None or r.symbol.upper() == symbol.upper())
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
# Background job functions (выполняются в worker thread'е FastAPI)
# --------------------------------------------------------------------------

def _make_progress_callback(job_id: str):
    """
    Создаёт callback который trainer вызывает на каждый прогресс-event.
    Конвертирует структурированный payload в человеко-читаемую строку и пишет
    в JobStore (доступно через GET /training/jobs/{id}?with_logs=true).
    """
    def _cb(payload: dict) -> None:
        phase = payload.get("phase", "?")
        if phase == "start":
            line = (
                f"START | epochs={payload.get('total_epochs')} | "
                f"total_batches={payload.get('total_batches')} | "
                f"trainable={payload.get('trainable_params')} | "
                f"device={payload.get('device')}"
            )
        elif phase == "train":
            line = (
                f"epoch {payload['epoch']}/{payload['total_epochs']} "
                f"batch {payload['batch']}/{payload['total_batches_per_epoch']} "
                f"(global {payload['global_batch']}/{payload['total_batches']} "
                f"{100 * payload['global_batch'] / payload['total_batches']:.0f}%) "
                f"loss={payload['loss']:.4f} running={payload['running_avg_loss']:.4f} "
                f"elapsed={_fmt_dur(payload['elapsed_s'])} eta={_fmt_dur(payload['eta_s'])}"
            )
        elif phase == "epoch_end":
            line = (
                f"epoch {payload['epoch']}/{payload['total_epochs']} done | "
                f"train_loss={payload['train_loss']:.4f} val_loss={payload['val_loss']:.4f} | "
                f"elapsed={_fmt_dur(payload['elapsed_s'])}"
            )
        else:
            line = f"{phase}: {payload}"
        _store.append_log(job_id, line)
    return _cb


def _fmt_dur(sec: float) -> str:
    if sec < 60:
        return f"{int(sec)}s"
    if sec < 3600:
        return f"{int(sec // 60)}m{int(sec % 60):02d}s"
    return f"{int(sec // 3600)}h{int((sec % 3600) // 60):02d}m"


def _run_head_job(job_id: str, request: HeadTrainingRequest) -> None:
    """Background-задача обучения head."""
    _store.mark_running(job_id)
    _store.append_log(job_id, "Starting head training")
    try:
        req_int = _to_internal(request)
        result = TrainingService.train_head(
            req_int, progress_callback=_make_progress_callback(job_id),
        )
        _store.append_log(job_id, f"Done: artifact_id={result['artifact_id']}, metrics={result['metrics']}")
        _store.mark_completed(job_id, artifact_id=result["artifact_id"])
    except Exception as ex:
        logger.exception("[train_head_job] failed")
        _store.append_log(job_id, f"FAILED: {ex}")
        _store.append_log(job_id, traceback.format_exc())
        _store.mark_failed(job_id, error=str(ex))


def _run_lora_job(job_id: str, request: LoraTrainingRequest) -> None:
    """Background-задача обучения LoRA."""
    _store.mark_running(job_id)
    _store.append_log(job_id, f"Starting LoRA training (r={request.lora_r}, alpha={request.lora_alpha}, "
                              f"train_head_too={request.train_head_too})")
    try:
        req_int = _to_internal(request)
        result = TrainingService.train_lora(
            req_int, request, progress_callback=_make_progress_callback(job_id),
        )
        _store.append_log(job_id, f"Done: artifact_id={result['artifact_id']}, "
                                  f"trainable_params={result['metrics'].get('trainable_params')}")
        _store.mark_completed(job_id, artifact_id=result["artifact_id"])
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
    """Краткое описание запроса для записи в JobStore."""
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
    if hasattr(request, "lora_r"):
        base["lora_r"] = request.lora_r
        base["lora_alpha"] = request.lora_alpha
        base["train_head_too"] = request.train_head_too
    return base


def _job_to_dict(job, with_logs: bool) -> dict:
    base = {
        "job_id":      job.job_id,
        "kind":        job.kind,
        "status":      job.status,
        "started_at":  job.started_at,
        "finished_at": job.finished_at,
        "artifact_id": job.artifact_id,
        "error":       job.error,
        "request":     job.request_summary,
    }
    if with_logs:
        base["logs"] = list(job.logs)
    return base


def _artifact_to_dict(record, detailed: bool = False) -> dict:
    base = {
        "id":                  record.id,
        "symbol":              record.symbol,
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
