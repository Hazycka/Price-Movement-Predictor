"""
TrainingJobStore — in-memory реестр running/completed/failed/cancelled training-задач.

Простейшая thread-safe реализация: dict + lock. Один процесс приложения,
restart обнуляет состояние (это OK — artifact останется в БД, статус восстановим
из model_artifacts при необходимости).

Каждая задача имеет:
  job_id           — uuid строка для опроса
  status           — pending | running | completed | failed | cancelled
  started_at / finished_at — float Unix-timestamp (форматируется в ISO в router'е)
  artifact_id      — заполняется когда задача завершилась успешно
  error            — текст ошибки если failed
  logs             — последние строки (кольцевой буфер N последних)
  request_summary  — снимок входного запроса

Multi-stage progress (для оркестрированных цепочек head → input → lora):
  stage             — head / input / lora / done — текущий обучаемый компонент
  stage_progress    — 0.0..1.0 прогресс ТЕКУЩЕГО стейджа (0.5 = середина эпохи 3/5)
  total_stages      — общее число стейджей в цепочке (1, 2 или 3)
  stages_completed  — сколько стейджей уже доделано (== len(completed_artifact_ids))
  completed_artifact_ids — id артефактов завершённых стейджей (в порядке: head, input, lora)

Cancel-механизм:
  cancel_requested — флаг, выставляется через POST /training/jobs/{id}/cancel.
                     Trainer'ы периодически (между батчами) вызывают
                     job_store.is_cancel_requested(job_id) и при True бросают
                     TrainingCancelledException. Job помечается status='cancelled',
                     уже завершённые стейджи (артефакты) остаются как ready в БД.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
JobStage = Literal["head", "input", "lora", "done"]


@dataclass
class TrainingJob:
    job_id: str
    kind: str                          # "head" | "lora"
    status: JobStatus = "pending"
    started_at: float | None = None
    finished_at: float | None = None
    artifact_id: int | None = None     # id ФИНАЛЬНОГО артефакта (последний в цепочке)
    error: str | None = None
    logs: deque = field(default_factory=lambda: deque(maxlen=500))
    request_summary: dict = field(default_factory=dict)

    # Multi-stage progress
    stage: JobStage | None = None
    stage_progress: float = 0.0
    total_stages: int = 1
    stages_completed: int = 0
    completed_artifact_ids: list[int] = field(default_factory=list)

    # Cancel
    cancel_requested: bool = False


class TrainingJobStore:
    """Singleton-стор training-задач. Доступ thread-safe."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._jobs: dict[str, TrainingJob] = {}
                    inst._jobs_lock = threading.Lock()
                    cls._instance = inst
        return cls._instance

    def create(self, kind: str, request_summary: dict | None = None,
               total_stages: int = 1) -> str:
        """Создаёт новую задачу со статусом 'pending'. Возвращает job_id."""
        job_id = str(uuid.uuid4())
        with self._jobs_lock:
            self._jobs[job_id] = TrainingJob(
                job_id=job_id,
                kind=kind,
                request_summary=request_summary or {},
                total_stages=total_stages,
            )
        return job_id

    def get(self, job_id: str) -> TrainingJob | None:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def list_all(self) -> list[TrainingJob]:
        with self._jobs_lock:
            return list(self._jobs.values())

    def mark_running(self, job_id: str) -> None:
        with self._jobs_lock:
            j = self._jobs.get(job_id)
            if j:
                j.status = "running"
                j.started_at = time.time()

    def mark_completed(self, job_id: str, artifact_id: int) -> None:
        with self._jobs_lock:
            j = self._jobs.get(job_id)
            if j:
                j.status = "completed"
                j.finished_at = time.time()
                j.artifact_id = artifact_id
                j.stage = "done"
                j.stage_progress = 1.0

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._jobs_lock:
            j = self._jobs.get(job_id)
            if j:
                j.status = "failed"
                j.finished_at = time.time()
                j.error = error

    def mark_cancelled(self, job_id: str) -> None:
        """Job был отменён через cancel-endpoint и trainer корректно остановился."""
        with self._jobs_lock:
            j = self._jobs.get(job_id)
            if j:
                j.status = "cancelled"
                j.finished_at = time.time()

    def append_log(self, job_id: str, line: str) -> None:
        """Добавляет строку в логи задачи (кольцевой буфер на N последних)."""
        with self._jobs_lock:
            j = self._jobs.get(job_id)
            if j:
                j.logs.append(line)

    # ------------------------------------------------------------------
    # Multi-stage progress
    # ------------------------------------------------------------------

    def begin_stage(self, job_id: str, stage: JobStage) -> None:
        """Помечает что job вошёл в новый стейдж (head/input/lora). Прогресс = 0."""
        with self._jobs_lock:
            j = self._jobs.get(job_id)
            if j:
                j.stage = stage
                j.stage_progress = 0.0

    def update_stage_progress(self, job_id: str, progress: float) -> None:
        """Обновляет прогресс текущего стейджа (0.0..1.0)."""
        with self._jobs_lock:
            j = self._jobs.get(job_id)
            if j:
                j.stage_progress = max(0.0, min(1.0, progress))

    def finish_stage(self, job_id: str, artifact_id: int) -> None:
        """Стейдж завершён, его артефакт сохранён в БД, можно идти к следующему."""
        with self._jobs_lock:
            j = self._jobs.get(job_id)
            if j:
                j.stages_completed += 1
                j.stage_progress = 1.0
                j.completed_artifact_ids.append(artifact_id)

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def request_cancel(self, job_id: str) -> bool:
        """
        Помечает cancel_requested=True. Возвращает True если job существует и
        ещё running/pending, False иначе (job уже завершён или не найден).
        Сам trainer периодически проверяет is_cancel_requested и сам останавливается.
        """
        with self._jobs_lock:
            j = self._jobs.get(job_id)
            if j is None:
                return False
            if j.status not in ("pending", "running"):
                return False
            j.cancel_requested = True
            return True

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._jobs_lock:
            j = self._jobs.get(job_id)
            return bool(j and j.cancel_requested)
