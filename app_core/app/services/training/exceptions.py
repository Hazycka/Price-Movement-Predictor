"""
Исключения тренировочного слоя.

TrainingCancelledException — поднимается из progress callback'а когда job
получил cancel-запрос (через POST /training/jobs/{id}/cancel). Trainer ловит
и завершает обучение чисто, не сохраняя недотренированный артефакт.

ArtifactHyperparamsConflictError — поднимается ArtifactMatcher'ом когда есть
артефакт с подходящим shape, но другими training-гиперпараметрами (lr, epochs,
batch_size, evaluation_weights, weight_first_to_last_ratio). Пользователь должен
явно решить — переиспользовать (то есть оставить старые hyperparams) или создать
новый артефакт (force_new_head_or_input=true + new_head_or_input_version).
"""
from __future__ import annotations

from typing import Any


class TrainingCancelledException(Exception):
    """Job был отменён через cancel endpoint. Поднимается trainer'ом между батчами."""

    def __init__(self, stage: str | None = None) -> None:
        self.stage = stage
        super().__init__(
            f"Training was cancelled by user request"
            + (f" (stage: {stage})" if stage else "")
        )


class ArtifactHyperparamsConflictError(Exception):
    """
    Найден совместимый по shape артефакт, но его training-hyperparams отличаются
    от запрошенных.

    Поля:
      component — какой компонент в конфликте ('head' / 'input')
      existing_artifact_id — id найденного артефакта
      existing_params — параметры в существующем артефакте
      requested_params — параметры в текущем запросе
      mismatched_fields — список полей, по которым отличие
    """

    def __init__(
            self,
            component: str,
            existing_artifact_id: int,
            existing_params: dict[str, Any],
            requested_params: dict[str, Any],
            mismatched_fields: list[str],
    ) -> None:
        self.component = component
        self.existing_artifact_id = existing_artifact_id
        self.existing_params = existing_params
        self.requested_params = requested_params
        self.mismatched_fields = mismatched_fields
        msg = (
            f"Найден совместимый по shape {component}-артефакт "
            f"#{existing_artifact_id}, но training-hyperparams отличаются. "
            f"Несоответствие в полях: {mismatched_fields}. "
            f"Чтобы переиспользовать старый — повтори запрос с теми же hyperparams "
            f"(см. existing_params). Чтобы создать новый — добавь "
            f"force_new_head_or_input=true и new_head_or_input_version='vX'."
        )
        super().__init__(msg)
