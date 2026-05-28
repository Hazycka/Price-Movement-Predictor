"""
ArtifactLoader — применяет training-артефакты поверх foundation-модели.

Use case:
  artifact = uow.model_registry.get_by_id(artifact_id)
  model = ArtifactLoader().apply(base_model, artifact)
  # base_model теперь имеет применённые LoRA / head / input

Дизайн:
  - Каждый component в artifact.training_components обрабатывается соответствующим
    хендлером (_apply_lora, _apply_head, _apply_input).
  - Хендлеры идемпотентны: ArtifactLoader НЕ кеширует — caller отвечает за кеш.
  - Если component не поддерживается → ValueError.

ПОДГРУЗКА СВЯЗАННЫХ КОМПОНЕНТОВ:
  LoRA-артефакт может ссылаться на head/input артефакты через поля
  params.base_head_artifact_id / params.base_input_artifact_id (записываются
  оркестратором при цепочечном обучении). При apply() LoRA-артефакта мы:
    1. Загружаем head артефакт по base_head_artifact_id и применяем его
    2. Загружаем input артефакт по base_input_artifact_id и применяем его
    3. Применяем сам LoRA
  Это нужно потому что LoRA-веса обучались поверх конкретной head/input —
  без них поведение модели будет неконсистентным.

КОНТРАКТ ПЕРЕСБОРКИ ПОСЛЕ LoRA:
  Inference у каждой FM идёт через свой wrapper (PatchTST — TSFM pipeline,
  Moirai2 — Moirai2Forecast). После оборачивания модели в PEFT этот wrapper
  ОБЯЗАН быть пересобран на peft-модели, иначе адаптер не участвует в инференсе.
  Поэтому каждый runtime, поддерживающий LoRA, реализует rebuild_after_lora().
  Loader НЕ знает про конкретные wrapper'ы и НЕ делает fallback — отсутствие
  метода или ошибка пересборки приводят к явному исключению.

Архитектурные хуки для других foundation-моделей:
  get_adapter_modules() (на самой модели) — единственная точка привязки head/input
  к структуре конкретной FM. rebuild_after_lora() (на runtime) — единственная
  точка привязки LoRA-инференса. Loader остаётся model-agnostic.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ...storage import get_uow_factory
from ...storage.ports import ModelArtifact

logger = logging.getLogger(__name__)


class ArtifactLoader:
    """Применяет training-артефакт к базовой модели."""

    def apply(self, base_model, artifact: ModelArtifact):
        """
        Применяет компоненты артефакта к base_model (in-place мутация состояния
        weights). Возвращает «адаптированную» модель.

        Если артефакт является LoRA — сначала проверяет params.base_head_artifact_id
        и params.base_input_artifact_id, и подгружает их (через рекурсивный apply)
        ПЕРЕД применением LoRA. Это гарантирует консистентность с обучением.
        """
        if not artifact.training_components:
            logger.info(
                "[ArtifactLoader] Артефакт id=%s не имеет компонентов — "
                "используем base model как есть.", artifact.id,
            )
            return base_model

        artifact_dir = Path(artifact.artifact_path)
        if not artifact_dir.exists():
            raise FileNotFoundError(
                f"Файлы артефакта id={artifact.id} не найдены: {artifact_dir}. "
                f"БД-запись существует но артефакт удалён."
            )

        # Если LoRA — подгружаем связанные head/input ПЕРЕД ним
        if "lora" in artifact.training_components:
            self._apply_linked_components(base_model, artifact)

        model = base_model
        for component in artifact.training_components:
            handler = self._get_handler(component)
            model = handler(model, artifact_dir, artifact.params or {})
            logger.info(
                "[ArtifactLoader] Применён компонент '%s' из артефакта id=%s",
                component, artifact.id,
            )
        return model

    def _apply_linked_components(self, base_model, lora_artifact: ModelArtifact) -> None:
        """
        Для LoRA-артефакта подгружает связанные head/input артефакты (если в
        params указаны base_head_artifact_id / base_input_artifact_id) и применяет
        их к модели до того как накатим саму LoRA.
        """
        params = lora_artifact.params or {}
        head_id = params.get("base_head_artifact_id")
        input_id = params.get("base_input_artifact_id")

        if head_id is None and input_id is None:
            logger.info(
                "[ArtifactLoader] LoRA-артефакт id=%s не имеет ссылок на head/input — "
                "применяем только LoRA (это zero-shot базис под LoRA)",
                lora_artifact.id,
            )
            return

        with get_uow_factory()() as uow:
            head_art = uow.model_registry.get_by_id(head_id) if head_id else None
            input_art = uow.model_registry.get_by_id(input_id) if input_id else None

        # Порядок: head → input → (затем сама LoRA в основном apply())
        if head_art is not None:
            logger.info(
                "[ArtifactLoader] LoRA #%d → подгружаем связанный head артефакт #%d",
                lora_artifact.id, head_art.id,
            )
            self._apply_head(base_model, Path(head_art.artifact_path), head_art.params or {})
        elif head_id is not None:
            logger.warning(
                "[ArtifactLoader] LoRA #%d ссылается на head #%d, но он не найден в БД — "
                "продолжаем без head (поведение может отличаться от обучения)",
                lora_artifact.id, head_id,
            )

        if input_art is not None:
            logger.info(
                "[ArtifactLoader] LoRA #%d → подгружаем связанный input артефакт #%d",
                lora_artifact.id, input_art.id,
            )
            self._apply_input(base_model, Path(input_art.artifact_path), input_art.params or {})
        elif input_id is not None:
            logger.warning(
                "[ArtifactLoader] LoRA #%d ссылается на input #%d, но он не найден в БД",
                lora_artifact.id, input_id,
            )

    def _get_handler(self, component: str):
        """Возвращает функцию-хендлер для component'а."""
        handlers = {
            "lora":    self._apply_lora,
            "head":    self._apply_head,
            "input":   self._apply_input,
            "full_ft": self._apply_full_ft,
        }
        if component not in handlers:
            raise ValueError(
                f"Неизвестный компонент артефакта: '{component}'. "
                f"Поддерживаемые: {list(handlers)}."
            )
        return handlers[component]

    # ------------------------------------------------------------------
    # Обработчики компонентов
    # ------------------------------------------------------------------

    def _apply_lora(self, model, artifact_dir: Path, params: dict):
        """
        Применяет LoRA-адаптер из artifact_dir/adapter/ через PEFT и пересобирает
        inference-стек runtime'а на peft-модели.

        FAIL-FAST: пересборка делегируется runtime.rebuild_after_lora(). Если
        метода нет — это ошибка конфигурации (адаптер не участвовал бы в инференсе).
        Ошибки пересборки НЕ глушатся: лучше явно упасть, чем тихо отдавать
        прогнозы base-модели без LoRA.
        """
        try:
            from peft import PeftModel
        except ImportError as ex:
            raise RuntimeError(
                "peft не установлен — невозможно применить LoRA-артефакт. "
                "Установите: `pip install peft`."
            ) from ex

        adapter_dir = artifact_dir / "adapter"
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"Директория LoRA-адаптера не найдена: {adapter_dir} — артефакт повреждён."
            )

        rt = model._runtime
        torch_model = self._unwrap_to_torch(model)
        peft_model = PeftModel.from_pretrained(torch_model, adapter_dir.as_posix())
        rt.model = peft_model

        # Контракт: runtime ОБЯЗАН уметь пересобрать свой inference-стек на
        # peft-модели. Без этого LoRA молча не применилась бы к инференсу.
        rebuild = getattr(rt, "rebuild_after_lora", None)
        if not callable(rebuild):
            raise RuntimeError(
                f"Runtime {type(rt).__name__} не реализует rebuild_after_lora(), "
                f"но к модели применяется LoRA-артефакт. Без пересборки inference-стека "
                f"адаптер был бы проигнорирован. Реализуй rebuild_after_lora(peft_model) "
                f"в этом runtime (см. PatchTSTRuntime / MoiraiRuntime)."
            )

        # Намеренно БЕЗ try/except: ошибка пересборки должна всплыть как есть.
        rebuild(peft_model)
        logger.info(
            "[ArtifactLoader] LoRA применена, %s.rebuild_after_lora() выполнен.",
            type(rt).__name__,
        )
        return model

    def _apply_head(self, model, artifact_dir: Path, params: dict):
        """
        Загружает head state_dict в адаптерный модуль (для PatchTSTFM —
        backbone.out_layer, для Moirai — param_proj и т.п.).

        Поиск модуля делегирован самой модели через `model.get_adapter_modules()`.
        Это убирает зависимость loader'а от конкретной структуры FM.

        ВНИМАНИЕ: мутирует base_model in-place.
        """
        import torch

        head_path = artifact_dir / "head.pt"
        if not head_path.exists():
            raise FileNotFoundError(
                f"Файл head.pt не найден в {artifact_dir} — артефакт повреждён."
            )

        adapter_modules = model.get_adapter_modules()
        if "head" not in adapter_modules:
            raise RuntimeError(
                f"Модель {type(model).__name__} не предоставляет 'head' "
                f"в get_adapter_modules() — head артефакт неприменим. "
                f"Доступны: {list(adapter_modules)}."
            )
        head_module = adapter_modules["head"]
        state_dict = torch.load(head_path, map_location=model._runtime.device, weights_only=True)
        head_module.load_state_dict(state_dict)
        logger.info(
            "[ArtifactLoader] Head загружен из %s в module=%s",
            head_path, type(head_module).__name__,
        )
        return model

    def _apply_input(self, model, artifact_dir: Path, params: dict):
        """Симметрично _apply_head — для input projection."""
        import torch

        input_path = artifact_dir / "input.pt"
        if not input_path.exists():
            raise FileNotFoundError(
                f"Файл input.pt не найден в {artifact_dir} — артефакт повреждён."
            )

        adapter_modules = model.get_adapter_modules()
        if "input" not in adapter_modules:
            raise RuntimeError(
                f"Модель {type(model).__name__} не предоставляет 'input' "
                f"в get_adapter_modules() — input артефакт неприменим. "
                f"Доступны: {list(adapter_modules)}."
            )
        input_module = adapter_modules["input"]
        state_dict = torch.load(input_path, map_location=model._runtime.device, weights_only=True)
        input_module.load_state_dict(state_dict)
        logger.info(
            "[ArtifactLoader] Input загружен из %s в module=%s",
            input_path, type(input_module).__name__,
        )
        return model

    def _apply_full_ft(self, model, artifact_dir: Path, params: dict):
        raise NotImplementedError(
            "Full fine-tuning component не реализован и не планируется в краткой перспективе."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_to_torch(model):
        """
        Достаёт raw torch-модель из wrapper'а (PatchTSTForecastModel /
        MoiraiForecastModel и т.д.). Используется в _apply_lora где
        PEFT-обёртке нужен raw torch-объект, а не наш wrapper.
        """
        rt = getattr(model, "_runtime", None)
        if rt is not None and rt.model is not None:
            return rt.model
        base = getattr(model, "base_model", None)
        if base is not None:
            return base
        return model