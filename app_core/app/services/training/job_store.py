"""
TrainingJobStore — in-memory реестр running/completed/failed training-задач.

Простейшая thread-safe реализация: dict + lock. Один процесс приложения,
restart обнуляет состояние (это OK — artifact останется в БД, статус восстановим
из model_artifacts при необходимости).

Каждая задача имеет:
  job_id    — uuid строка для опроса
  status    — pending | running | completed | failed
  started_at, finished_at — timestamps
  artifact_id — заполняется когда задача завершилась успешно
  error     — текст ошибки если failed
  logs      — последние строки (последние N собранные через append_log)
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

JobStatus = Literal["pending", "running", "completed", "failed"]


@dataclass
class TrainingJob:
    job_id: str
    kind: str                          # "head" | "lora"
    status: JobStatus = "pending"
    started_at: float | None = None
    finished_at: float | None = None
    artifact_id: int | None = None
    error: str | None = None
    logs: deque = field(default_factory=lambda: deque(maxlen=200))
    request_summary: dict = field(default_factory=dict)


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

    def create(self, kind: str, request_summary: dict | None = None) -> str:
        """Создаёт новую задачу со статусом 'pending'. Возвращает job_id."""
        job_id = str(uuid.uuid4())
        with self._jobs_lock:
            self._jobs[job_id] = TrainingJob(
                job_id=job_id,
                kind=kind,
                request_summary=request_summary or {},
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

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._jobs_lock:
            j = self._jobs.get(job_id)
            if j:
                j.status = "failed"
                j.finished_at = time.time()
                j.error = error

    def append_log(self, job_id: str, line: str) -> None:
        """Добавляет строку в логи задачи (кольцевой буфер на N последних)."""
        with self._jobs_lock:
            j = self._jobs.get(job_id)
            if j:
                j.logs.append(line)
