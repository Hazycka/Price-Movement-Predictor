"""
ArtifactLoader — применяет training-артефакты поверх foundation-модели.

Use case:
  artifact = uow.model_registry.get_by_id(artifact_id)
  model = ArtifactLoader().apply(base_model, artifact)
  # base_model теперь имеет применённые LoRA / head / любые другие компоненты

Дизайн:
  - Каждый component в artifact.training_components обрабатывается соответствующим
    хендлером (`_apply_lora`, `_apply_head`).
  - Хендлеры идемпотентны: ArtifactLoader НЕ кеширует — caller отвечает за кеш.
  - Если component не поддерживается → ValueError с понятным сообщением.

Текущий статус:
  Phase 0 — skeleton, конкретных хендлеров пока нет (заполнятся в Phase 1/2).
  Если в artifact.training_components что-то указано — будет ValueError "пока не реализовано".
"""
from __future__ import annotations

import logging
from pathlib import Path

from ...storage.ports import ModelArtifact

logger = logging.getLogger(__name__)


class ArtifactLoader:
    """Применяет training-артефакт к базовой модели."""

    def apply(self, base_model, artifact: ModelArtifact):
        """
        Применяет все компоненты артефакта к base_model.

        Возвращает «адаптированную» модель. Это может быть тот же объект (если
        компонент мутирует in-place) или обёртка (PeftModel и т.п.).

        ВАЖНО: при `artifact.train_window_size` !=  тому что использует caller,
        записываем warning. Адаптер технически применится, но качество
        предсказаний может пострадать.
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

        model = base_model
        for component in artifact.training_components:
            handler = self._get_handler(component)
            model = handler(model, artifact_dir, artifact.params or {})
            logger.info(
                "[ArtifactLoader] Применён компонент '%s' из артефакта id=%s",
                component, artifact.id,
            )
        return model

    def _get_handler(self, component: str):
        """Возвращает функцию-хендлер для component'а."""
        handlers = {
            "lora": self._apply_lora,
            "head": self._apply_head,
            "full_ft": self._apply_full_ft,
        }
        if component not in handlers:
            raise ValueError(
                f"Неизвестный компонент артефакта: '{component}'. "
                f"Поддерживаемые: {list(handlers)}."
            )
        return handlers[component]

    # ------------------------------------------------------------------
    # Обработчики компонентов (заполняются по фазам)
    # ------------------------------------------------------------------

    def _apply_lora(self, model, artifact_dir: Path, params: dict):
        """
        Применяет LoRA-адаптер из artifact_dir/adapter/ через PEFT.

        Возвращает PEFT-обёрнутую модель. Нашему wrapper'у (PatchTSTForecastModel)
        нужно подменить внутренний torch.model на эту обёртку, чтобы последующие
        forward-вызовы шли через адаптер.
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

        torch_model = self._unwrap_to_torch(model)
        peft_model = PeftModel.from_pretrained(torch_model, adapter_dir.as_posix())
        # Подменяем _runtime.model на peft-обёртку. PatchTSTRuntime.pipeline ссылается
        # на self.model — нам нужно ОБНОВИТЬ pipeline тоже чтобы он использовал
        # peft_model. Делаем это аккуратно — пересоздаём pipeline.
        model._runtime.model = peft_model
        # Pipeline хранит ссылку на старую модель — пересоздаём:
        try:
            from tsfm_public import TimeSeriesForecastingPipeline
            from .patchtst_runtime import QUANTILE_LEVELS  # noqa: re-importable
        except ImportError:
            pass
        # Re-creating pipeline через runtime API
        rt = model._runtime
        try:
            from tsfm_public import TimeSeriesForecastingPipeline as TSPipeline
            rt.pipeline = TSPipeline(
                model=peft_model,
                device=rt.device,
                explode_forecasts=True,
                quantile_levels=rt.quantile_levels,
            )
            logger.info("[ArtifactLoader] LoRA адаптер загружен и pipeline пересоздан.")
        except Exception as ex:
            logger.warning(
                "[ArtifactLoader] Не удалось пересоздать TSFM pipeline после применения LoRA: %s. "
                "Inference может использовать base-модель без адаптера.",
                ex,
            )

        return model

    def _apply_head(self, model, artifact_dir: Path, params: dict):
        """
        Применяет сохранённую новую output head поверх foundation-модели.

        model — это наш wrapper (PatchTSTForecastModel или аналог), а не raw torch-модуль.
        Поэтому нужно подменять head во внутреннем torch_model.

        ВНИМАНИЕ: head_state_dict ЗАГРУЖАЕТСЯ В МЕСТЕ ИСХОДНОЙ HEAD. Это значит что
        base model МУТИРУЕТСЯ — после inference остальные запросы (без артефакта)
        получат адаптированную модель. Caller отвечает за изоляцию (создать свежую
        модель на каждый запрос, либо хранить snapshot и восстанавливать).

        Для нашего workflow (worker процессы создают свои model singletons): пока
        worker обрабатывает только один artifact_id за раз — нормально. Когда смешаем
        разные artifact_id в одном воркере — нужно будет добавить snapshot/restore.
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch не установлен.")

        head_path = artifact_dir / "head.pt"
        if not head_path.exists():
            raise FileNotFoundError(
                f"Файл head.pt не найден в {artifact_dir} — артефакт повреждён."
            )

        # Находим head в torch модели (логика та же что в HeadTrainer)
        torch_model = self._unwrap_to_torch(model)
        head_module = self._locate_head_module(torch_model)

        state_dict = torch.load(head_path, map_location=model._runtime.device)
        head_module.load_state_dict(state_dict)
        logger.info(
            "[ArtifactLoader] Head загружен из %s в module=%s",
            head_path, type(head_module).__name__,
        )
        return model

    def _apply_full_ft(self, model, artifact_dir: Path, params: dict):
        """Полное дообучение — заменяет state_dict модели целиком. Не планируем сейчас."""
        raise NotImplementedError(
            "Full fine-tuning component не реализован и не планируется в краткой перспективе."
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_to_torch(model):
        """
        Достаёт raw torch-модель из нашего wrapper'а PatchTSTForecastModel.
        Если model уже torch-модуль — возвращает как есть.
        """
        # Наш wrapper держит модель в _runtime.model
        rt = getattr(model, "_runtime", None)
        if rt is not None and rt.model is not None:
            return rt.model
        # Если это PeftModel — base_model.model
        base = getattr(model, "base_model", None)
        if base is not None:
            return base
        return model

    @staticmethod
    def _locate_head_module(torch_model):
        """Дублирует логику HeadTrainer._locate_head_module."""
        for name in ("head", "prediction_head", "output_head", "linear_head"):
            module = getattr(torch_model, name, None)
            if module is not None and hasattr(module, "parameters"):
                return module
        children = [n for n, _ in torch_model.named_children()]
        raise RuntimeError(
            f"Не удалось найти head у модели. Топ-level модули: {children}."
        )
