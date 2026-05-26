"""
ArtifactSaver — записывает обученные компоненты на диск + регистрирует в БД.

Структура артефакта на диске:
  data/artifacts/{artifact_id}/
      metadata.json                — снимок params + base модель + дата обучения
      head.pt                       — если "head" в training_components
      adapter/                      — если "lora" в training_components
          adapter_config.json       — конфиг LoRA от PEFT
          adapter_model.safetensors

Сохранение происходит в две стадии:
  1. upsert_artifact() — записать в БД с status='training' (получить id)
  2. сохранение файлов в data/artifacts/{id}/
  3. update_artifact_status() — пометить status='ready'

Если файлы сохранить не удалось — оставляем status='failed' с описанием в metrics_json.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:
    torch = None

from ...storage import get_uow_factory
from ...storage.ports import ModelArtifact

logger = logging.getLogger(__name__)

ARTIFACTS_ROOT = Path(os.getenv("ARTIFACTS_DIR", "data/artifacts"))


@dataclass
class SaveContext:
    """
    Контекст одного training-run'а который сохраняем как артефакт.

    Используется чтобы передать в saver все нужные параметры в одном объекте,
    а не длинной портянкой аргументов.

    source — провайдер данных (t_invest / yahoo / csv), часть UNIQUE-ключа.
    market — биржевая секция (TQBR, SPBXM, ...), информативно, в UNIQUE НЕ входит.
    """
    symbol: str
    source: str
    market: str | None
    interval: str
    model_name: str
    training_components: list[str]   # ["head"] / ["lora"] / ["lora", "head"]
    train_window_size: int
    horizon: int
    version: str
    base_model_id: str
    base_model_version: str
    training_params: dict[str, Any]   # снимок гиперпараметров (LR, epochs и т.п.)


class ArtifactSaver:
    """
    Сохранение артефактов на диск + регистрация в БД.
    """

    @staticmethod
    def save_head(
            head_state_dict: dict,
            metrics: dict[str, float],
            context: SaveContext,
    ) -> int:
        """Сохраняет head state_dict в head.pt + регистрирует в БД."""
        return ArtifactSaver._save_components(
            components={"head.pt": ("head_torch", head_state_dict)},
            metrics=metrics,
            context=context,
        )

    @staticmethod
    def save_input(
            input_state_dict: dict,
            metrics: dict[str, float],
            context: SaveContext,
    ) -> int:
        """Сохраняет input projection state_dict в input.pt + регистрирует в БД."""
        return ArtifactSaver._save_components(
            components={"input.pt": ("input_torch", input_state_dict)},
            metrics=metrics,
            context=context,
        )

    @staticmethod
    def save_lora(
            peft_model,
            metrics: dict[str, float],
            context: SaveContext,
    ) -> int:
        """
        Сохраняет LoRA-адаптер через peft_model.save_pretrained(adapter_dir).
        Возвращает artifact_id.
        """
        return ArtifactSaver._save_components(
            components={"adapter/": ("lora_peft", peft_model)},
            metrics=metrics,
            context=context,
        )

    @staticmethod
    def save_combo(
            head_state_dict: dict | None,
            peft_model,
            metrics: dict[str, float],
            context: SaveContext,
    ) -> int:
        """LoRA + head в одном артефакте."""
        components = {}
        if peft_model is not None:
            components["adapter/"] = ("lora_peft", peft_model)
        if head_state_dict is not None:
            components["head.pt"] = ("head_torch", head_state_dict)
        return ArtifactSaver._save_components(
            components=components,
            metrics=metrics,
            context=context,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _save_components(
            components: dict[str, tuple[str, Any]],
            metrics: dict[str, float],
            context: SaveContext,
    ) -> int:
        """
        components: { filename_or_subdir: (writer_type, payload) }
          writer_type ∈ {"head_torch", "lora_peft"}.

        Алгоритм:
          1. Создаём запись в БД с пустым artifact_path и status='training' — получаем id
          2. Создаём data/artifacts/{id}/ и пишем туда файлы
          3. Записываем metadata.json
          4. Обновляем запись: artifact_path + status='ready' + metrics
        """
        if torch is None:
            raise RuntimeError("PyTorch не установлен.")

        # Stage 1: создать запись с status='training'
        placeholder = ModelArtifact(
            symbol=context.symbol,
            source=context.source,
            market=context.market,
            interval=context.interval,
            model_name=context.model_name,
            training_components=context.training_components,
            train_window_size=context.train_window_size,
            version=context.version,
            status="training",
            artifact_path="",  # заполним после получения id
            params=context.training_params,
            metrics=None,
        )
        with get_uow_factory()() as uow:
            artifact_id = uow.model_registry.upsert(placeholder)

        # Stage 2: создаём директорию и пишем файлы
        artifact_dir = ARTIFACTS_ROOT / str(artifact_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        try:
            for relpath, (writer_type, payload) in components.items():
                target = artifact_dir / relpath
                if writer_type in ("head_torch", "input_torch"):
                    # state_dict сохраняется одинаково через torch.save
                    torch.save(payload, target)
                elif writer_type == "lora_peft":
                    # payload — peft_model
                    target.mkdir(parents=True, exist_ok=True)
                    payload.save_pretrained(target.as_posix())
                else:
                    raise ValueError(f"Неизвестный writer_type: {writer_type}")

            # metadata.json
            metadata = {
                "artifact_id": artifact_id,
                "version": context.version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "base_model": {
                    "name": context.model_name,
                    "model_id": context.base_model_id,
                    "version": context.base_model_version,
                },
                "trained_on": {
                    "symbol": context.symbol,
                    "source": context.source,
                    "market": context.market,
                    "interval": context.interval,
                },
                "training_components": context.training_components,
                "training_config": {
                    "train_window_size": context.train_window_size,
                    "horizon": context.horizon,
                    **context.training_params,
                },
                "training_metrics": metrics,
            }
            (artifact_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Stage 3: обновляем status='ready' + сохраняем metrics
            final = ModelArtifact(
                id=artifact_id,
                symbol=context.symbol,
                source=context.source,
                market=context.market,
                interval=context.interval,
                model_name=context.model_name,
                training_components=context.training_components,
                train_window_size=context.train_window_size,
                version=context.version,
                status="ready",
                artifact_path=str(artifact_dir),
                metrics=metrics,
                params=context.training_params,
            )
            with get_uow_factory()() as uow:
                uow.model_registry.upsert(final)

            logger.info(
                "[ArtifactSaver] Сохранён артефакт id=%d в %s, status=ready",
                artifact_id, artifact_dir,
            )
            return artifact_id

        except Exception as ex:
            # При ошибке — помечаем status='failed', записываем причину
            logger.exception("[ArtifactSaver] Ошибка сохранения артефакта id=%d", artifact_id)
            failed = ModelArtifact(
                id=artifact_id,
                symbol=context.symbol,
                source=context.source,
                market=context.market,
                interval=context.interval,
                model_name=context.model_name,
                training_components=context.training_components,
                train_window_size=context.train_window_size,
                version=context.version,
                status="failed",
                artifact_path=str(artifact_dir),
                metrics={"error": str(ex)},
                params=context.training_params,
            )
            with get_uow_factory()() as uow:
                uow.model_registry.upsert(failed)
            raise
