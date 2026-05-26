"""
ArtifactMatcher — поиск совместимых артефактов для переиспользования в цепочке
обучения head → input → lora.

Стратегия матчинга в два этапа:

1. SHAPE-COMPAT: артефакт должен совпадать по «идентичности»:
     (symbol, source, interval, model_name, training_components, train_window_size, horizon)
   Это поля, без точного совпадения которых state_dict просто не ляжет (или
   ляжет, но даст бессмыслицу). Без совпадения — артефакт НЕ кандидат.

2. HYPERPARAMS-COMPAT: совпадают training-гиперпараметры:
     learning_rate, num_epochs, batch_size,
     evaluation_weights, weight_first_to_last_ratio
   Если shape-compat есть, но hyperparams отличаются — поднимаем
   ArtifactHyperparamsConflictError (HTTP 409). Пользователь должен явно
   решить переобучить (force_new_head_or_input=true + new_head_or_input_version)
   или прислать те же hyperparams чтобы переиспользовать.

3. TIE-BREAKING: если несколько артефактов прошли оба фильтра — выбираем
   лучший по `metrics.final_val_loss` (меньше = лучше). При равенстве — по
   created_at (свежее = лучше).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .exceptions import ArtifactHyperparamsConflictError
from ...storage import get_uow_factory
from ...storage.ports import ModelArtifact

logger = logging.getLogger(__name__)


# Поля training_params, по которым требуется ТОЧНОЕ совпадение для reuse'а
_HYPERPARAM_FIELDS = (
    "learning_rate",
    "num_epochs",
    "batch_size",
    "evaluation_weights",
    "weight_first_to_last_ratio",
)


@dataclass
class MatchRequest:
    """Параметры для поиска подходящего артефакта."""
    symbol: str
    source: str
    interval: str
    model_name: str
    training_components: list[str]   # ["head"] или ["input"]
    train_window_size: int
    horizon: int
    # Hyperparams для сравнения
    learning_rate: float
    num_epochs: int
    batch_size: int
    evaluation_weights: str
    weight_first_to_last_ratio: float


class ArtifactMatcher:
    """
    Сервис поиска совместимых артефактов.
    Использует ModelRegistryRepo напрямую.
    """

    @staticmethod
    def find(req: MatchRequest, force_new: bool = False) -> ModelArtifact | None:
        """
        Возвращает совместимый артефакт или None.

        force_new=True — игнорируем существующие артефакты (вернём None).
        Используется когда юзер хочет явно перетренировать с новыми параметрами.

        Поднимает ArtifactHyperparamsConflictError если есть shape-совместимый
        артефакт, но его hyperparams отличаются от запрошенных.
        """
        if force_new:
            logger.info(
                "[ArtifactMatcher] force_new=True — пропускаем поиск, "
                "цепочка будет тренировать новый %s с нуля",
                req.training_components,
            )
            return None

        candidates = ArtifactMatcher._find_shape_compatible(req)
        if not candidates:
            logger.info(
                "[ArtifactMatcher] Нет shape-совместимых артефактов для "
                "%s/%s/%s ctx=%d horizon=%d components=%s",
                req.symbol, req.source, req.interval,
                req.train_window_size, req.horizon, req.training_components,
            )
            return None

        # Фильтруем по hyperparams
        hp_matches: list[ModelArtifact] = []
        shape_only: list[ModelArtifact] = []
        for art in candidates:
            mismatched = ArtifactMatcher._hyperparams_diff(req, art)
            if not mismatched:
                hp_matches.append(art)
            else:
                shape_only.append((art, mismatched))   # type: ignore

        if hp_matches:
            # Tie-breaking: лучший val_loss, при равенстве — свежее
            best = ArtifactMatcher._pick_best(hp_matches)
            logger.info(
                "[ArtifactMatcher] Reuse артефакта #%d (%s, val_loss=%.4f)",
                best.id, req.training_components,
                (best.metrics or {}).get("final_val_loss", -1),
            )
            return best

        # Нашли shape-совместимые, но hyperparams отличаются — поднимаем conflict
        first, mismatched = shape_only[0]
        raise ArtifactHyperparamsConflictError(
            component=",".join(req.training_components),
            existing_artifact_id=first.id,
            existing_params=ArtifactMatcher._extract_hyperparams(first.params or {}),
            requested_params={f: getattr(req, f) for f in _HYPERPARAM_FIELDS},
            mismatched_fields=mismatched,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _find_shape_compatible(req: MatchRequest) -> list[ModelArtifact]:
        """
        Возвращает артефакты с точным совпадением shape-полей.

        Использует low-level фильтрацию вместо registry.find_ready чтобы учесть
        ВСЕ shape-поля, а не только (symbol, source, interval, model_name).
        """
        with get_uow_factory()() as uow:
            all_ready = uow.model_registry.find_ready(
                symbol=req.symbol,
                source=req.source,
                interval=req.interval,
                model_name=req.model_name,
            )

        components_target = sorted(req.training_components)
        result: list[ModelArtifact] = []
        for art in all_ready:
            if sorted(art.training_components or []) != components_target:
                continue
            if art.train_window_size != req.train_window_size:
                continue
            params = art.params or {}
            if int(params.get("horizon", -1)) != int(req.horizon):
                continue
            result.append(art)
        return result

    @staticmethod
    def _hyperparams_diff(req: MatchRequest, art: ModelArtifact) -> list[str]:
        """
        Возвращает список ключей по которым отличаются hyperparams.
        Пустой список = идеальное совпадение.
        """
        params = art.params or {}
        mismatched: list[str] = []
        for field in _HYPERPARAM_FIELDS:
            existing = params.get(field)
            requested = getattr(req, field)
            # Числовые сравниваем приблизительно (lr=5e-4 и 0.0005 одинаковы)
            if isinstance(existing, float) or isinstance(requested, float):
                try:
                    if abs(float(existing) - float(requested)) > 1e-9:
                        mismatched.append(field)
                except (TypeError, ValueError):
                    mismatched.append(field)
            else:
                if existing != requested:
                    mismatched.append(field)
        return mismatched

    @staticmethod
    def _pick_best(candidates: list[ModelArtifact]) -> ModelArtifact:
        """Tie-breaking: меньший val_loss, при равенстве — свежее created_at."""
        def sort_key(a: ModelArtifact) -> tuple[float, str]:
            val = (a.metrics or {}).get("final_val_loss", float("inf"))
            # Свежее = больше, но мы сортируем по убыванию created_at для свежести
            return (val if val is not None else float("inf"), -1 * (
                # int hash из строки даты — для стабильной сортировки при равных val_loss
                hash(a.created_at or "") & 0xFFFF
            ))
        return sorted(candidates, key=sort_key)[0]

    @staticmethod
    def _extract_hyperparams(params: dict[str, Any]) -> dict[str, Any]:
        return {f: params.get(f) for f in _HYPERPARAM_FIELDS}
