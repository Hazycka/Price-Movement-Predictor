"""
Persistent ProcessPoolExecutor для параллельных sweep'ов.

Дизайн:
  - Singleton-пул на уровне приложения.
  - Лениво создаётся при первом запросе с parallel_workers > 1.
  - Переиспользуется между sweep-запросами (амортизация загрузки модели:
    каждый worker грузит PatchTST один раз и хранит до shutdown).
  - При запросе с другим размером — пул пересоздаётся.
  - Останавливается через `shutdown_pool()` (FastAPI lifespan + atexit).

CUDA + multiprocessing:
  Используем mp_context="spawn" — обязательно для CUDA на Windows (а на Linux
  тоже работает). Spawn создаёт чистый child-процесс без наследования CUDA-
  состояния родителя. Каждый worker инициализирует CUDA сам по факту первого
  использования модели.

Сериализация загрузки модели:
  При parallel_workers=N все N воркеров получают свои первые таски примерно
  одновременно → все N начинают грузить модель в VRAM **одновременно** →
  пиковый запрос VRAM = N × model_size. На 12GB GPU это легко вызывает OOM.

  Решение: shared multiprocessing.Lock(), переданный в воркеры через initargs.
  Каждый воркер при первой загрузке модели берёт lock → модели грузятся
  ПО ОЧЕРЕДИ, не симультанно. После загрузки воркер кеширует модель локально
  и больше lock не трогает.

Ограничения:
  - На Windows нет MPS → процессы конкурируют за GPU, GPU-работа сериализуется.
    Реальный выигрыш — на CPU-фазе (TSFM pre/post).
  - VRAM × N: каждый воркер держит свою копию модели (~3-4 GB).
  - На крахе главного процесса воркеры могут остаться висеть; в этом случае
    нужно вручную убить python.exe (TaskManager / `taskkill /F /IM python.exe`).
    atexit покрывает только нормальное завершение.
"""
from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)


_pool: ProcessPoolExecutor | None = None
_pool_size: int = 0
_atexit_registered: bool = False


def _worker_initializer(model_load_lock) -> None:
    """
    Initializer ProcessPoolExecutor: вызывается ОДИН раз в каждом воркер-процессе
    при его старте. Сохраняет shared lock в module-level переменной _worker.py
    чтобы _ensure_runner мог использовать его при первой загрузке модели.

    Также инициализирует storage в воркере для записи runs в БД.
    """
    from . import _worker
    _worker._MODEL_LOAD_LOCK = model_load_lock
    _worker.init_worker()


def get_pool(workers: int) -> ProcessPoolExecutor:
    """
    Возвращает singleton-пул нужного размера. Создаёт при первом вызове
    или если изменился запрошенный размер.
    """
    global _pool, _pool_size, _atexit_registered

    if workers < 1:
        raise ValueError(f"workers должно быть >= 1, получено {workers}")

    if _pool is not None and _pool_size == workers:
        return _pool

    if _pool is not None:
        logger.info("[parallel_pool] Пересоздаём пул: %d → %d воркеров", _pool_size, workers)
        _pool.shutdown(wait=True)
        _pool = None

    ctx = mp.get_context("spawn")
    # Shared lock сериализует загрузку модели между воркерами при первом таске.
    # Lock создаётся в том же mp_context что и пул — обязательно для совместимости.
    model_load_lock = ctx.Lock()
    _pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_worker_initializer,
        initargs=(model_load_lock,),
    )
    _pool_size = workers
    logger.info(
        "[parallel_pool] Создан пул на %d воркеров (spawn, model-load lock активен)",
        workers,
    )

    # Регистрируем shutdown один раз — на случай если приложение завершится
    # не через FastAPI lifespan (например при крахе или прямом Ctrl+C).
    if not _atexit_registered:
        atexit.register(shutdown_pool)
        _atexit_registered = True

    return _pool


def shutdown_pool() -> None:
    """
    Останавливает пул и освобождает ресурсы. Вызывается из FastAPI shutdown
    и из atexit. Идемпотентна.
    """
    global _pool, _pool_size
    if _pool is None:
        return
    logger.info("[parallel_pool] Останавливаем пул (%d воркеров)", _pool_size)
    try:
        _pool.shutdown(wait=True)
    except Exception as ex:
        logger.warning("[parallel_pool] Ошибка при shutdown: %s", ex)
    _pool = None
    _pool_size = 0


def is_active() -> bool:
    """Истина если пул создан и не остановлен."""
    return _pool is not None
