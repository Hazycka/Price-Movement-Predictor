"""
TrainingService — оркестрация обучения: head, head→input цепочка, head→[input]→lora.

Логика выбора компонентов:

  POST /training/head
    train_input_too=False → train_head_chain → артефакт ['head']
    train_input_too=True  → train_head_chain (head) → train_input_chain →
                             два артефакта: ['head'] и ['input']

  POST /training/lora
    Цепочка ВСЕГДА начинается с head (LP-FT принцип Kumar et al. 2022).
    ArtifactMatcher ищет существующие head/input артефакты:
      - force_new_head_or_input=False (default):
          * нашёлся compatible head + (input если train_input_too) → reuse
          * shape совпадает, hyperparams другие → 409 Conflict
          * не нашлось вообще → silent-create под req.version (без конфликта)
      - force_new_head_or_input=True:
          * игнорим существующие, тренируем новые с new_head_or_input_version
    После head/input — тренируется LoRA → артефакт ['lora']

Multi-stage прогресс:
  Каждый компонент в цепочке — отдельный «стейдж». В JobStore трекаются:
    stage (head/input/lora), stage_progress (0..1), stages_completed, total_stages.
  Прогресс внутри стейджа считается как (epoch_done + batch_in_epoch_frac) / total_epochs.

Cancel:
  Trainer'ы вызывают cancel_check каждые ~5 батчей. При True бросают
  TrainingCancelledException. Оркестратор ловит — НЕ сохраняет недотренированный
  стейдж (договорённость: только полностью завершённые остаются как ready).
  Уже завершённые предыдущие стейджи остаются в БД как ready.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from ...models.base import ForecastModel
from ...models.factory import create_model
from ...schemas import (
    ForecastRequest, ProviderOptions,
    TInvestProviderOptions, YahooProviderOptions, CsvProviderOptions,
    LoraTrainingRequest,
)
from ..forecast.input_resolver import InputResolver
from .artifact_matcher import ArtifactMatcher, MatchRequest
from .exceptions import TrainingCancelledException
from .head_trainer import HeadTrainer, HeadTrainingConfig
from .input_trainer import InputTrainer, InputTrainingConfig
from .lora_trainer import LoraTrainer, LoraTrainingConfig
from .saver import ArtifactSaver, SaveContext


ProgressCallback = Callable[[dict], None]
CancelCheck = Callable[[], bool]
StageCallback = Callable[[str], None]   # begin_stage(name)
StageFinishedCallback = Callable[[str, int], None]   # finish_stage(name, artifact_id)

logger = logging.getLogger(__name__)


@dataclass
class TrainingRequestInternal:
    """
    Внутренний контекст запроса. Не Pydantic — мы конвертим публичный
    HeadTrainingRequest / LoraTrainingRequest из API в это.
    """
    model_name: str | None
    data_source: str
    provider_options: ProviderOptions
    interval: str
    history_period: str
    history_up_to: str | None
    # training params
    train_window_size: int
    horizon: int
    step: int
    batch_size: int
    learning_rate: float
    num_epochs: int
    val_split: float
    evaluation_weights: str
    weight_first_to_last_ratio: float
    version: str
    num_workers: int = 0


@dataclass
class _LoadedArtifact:
    """Внутренняя структура — описание загруженного/только что обученного компонента."""
    artifact_id: int
    component: str   # 'head' | 'input' | 'lora'
    state_dict: dict | None = None   # для head/input
    peft_model: Any = None           # для lora


class TrainingService:
    """
    Оркестратор обучения. Методы train_head / train_lora принимают
    request_internal + флаги цепочки, возвращают финальный artifact_id.
    """

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    @staticmethod
    def train_head(
            req: TrainingRequestInternal,
            train_input_too: bool = False,
            progress_callback: ProgressCallback | None = None,
            cancel_check: CancelCheck | None = None,
            stage_callback: StageCallback | None = None,
            stage_finished_callback: StageFinishedCallback | None = None,
    ) -> dict[str, Any]:
        """
        Endpoint POST /training/head.

        train_input_too=False → одна стадия (head). Возвращает один artifact_id.
        train_input_too=True  → две стадии (head → input). Возвращает финальный
                                artifact_id (input), а head_artifact_id отдельно
                                в полях ответа.
        """
        forecast_like_req = TrainingService._build_resolver_request(req)
        source, _dates, candles = InputResolver.resolve(forecast_like_req)
        logger.info(
            "[TrainingService] train_head: source=%s len(candles)=%d train_input_too=%s",
            source, len(candles), train_input_too,
        )

        # Шаг 1: загружаем base модель (свежий инстанс — не из inference-кеша,
        # потому что мы будем модифицировать веса head/input)
        model = create_model(req.model_name)
        model._runtime.ensure_loaded()

        # Стадия head
        head_loaded = TrainingService._train_and_save_head(
            req=req, model=model, candles=candles, source=source,
            progress_callback=progress_callback, cancel_check=cancel_check,
            stage_callback=stage_callback,
            stage_finished_callback=stage_finished_callback,
        )

        result: dict[str, Any] = {
            "artifact_id":    head_loaded.artifact_id,
            "status":         "ready",
            "training_components": ["head"],
            "head_artifact_id":   head_loaded.artifact_id,
            "input_artifact_id":  None,
        }

        if not train_input_too:
            return result

        # Стадия input — head уже применена к модели (state_dict в памяти,
        # модуль backbone.out_layer обучен и не разморажен), переходим к input
        input_loaded = TrainingService._train_and_save_input(
            req=req, model=model, candles=candles, source=source,
            progress_callback=progress_callback, cancel_check=cancel_check,
            stage_callback=stage_callback,
            stage_finished_callback=stage_finished_callback,
        )
        result["input_artifact_id"] = input_loaded.artifact_id
        result["artifact_id"] = input_loaded.artifact_id  # последний в цепочке
        result["training_components"] = ["input"]
        return result

    @staticmethod
    def train_lora(
            req: TrainingRequestInternal,
            lora_request: LoraTrainingRequest,
            progress_callback: ProgressCallback | None = None,
            cancel_check: CancelCheck | None = None,
            stage_callback: StageCallback | None = None,
            stage_finished_callback: StageFinishedCallback | None = None,
    ) -> dict[str, Any]:
        """
        Endpoint POST /training/lora.

        Полная цепочка: [find_or_train head] → [find_or_train input?] → train lora.

        train_input_too управляет включением input.
        force_new_head_or_input=True → тренируем head/input заново под new_head_or_input_version.
        force_new_head_or_input=False → ищем существующие; при несовместимости — 409.
        """
        forecast_like_req = TrainingService._build_resolver_request(req)
        source, _dates, candles = InputResolver.resolve(forecast_like_req)
        logger.info(
            "[TrainingService] train_lora: source=%s len(candles)=%d "
            "train_input_too=%s force_new_head_or_input=%s",
            source, len(candles), lora_request.train_input_too,
            lora_request.force_new_head_or_input,
        )

        # Базовая модель — fresh, потому что веса будут модифицироваться
        model = create_model(req.model_name)
        model._runtime.ensure_loaded()

        ticker, source_name, market = TrainingService._extract_ticker_info(req)
        model_info = model.get_info()
        model_name_resolved = model_info.get("name", "unknown")

        # Шаг 1: head (reuse либо train)
        head_loaded = TrainingService._resolve_or_train_head(
            req=req, lora_request=lora_request,
            ticker=ticker, source_name=source_name, market=market,
            model_name=model_name_resolved, model=model, candles=candles, source=source,
            progress_callback=progress_callback, cancel_check=cancel_check,
            stage_callback=stage_callback,
            stage_finished_callback=stage_finished_callback,
        )
        TrainingService._apply_head_to_model(model, head_loaded.state_dict)

        # Шаг 2 (опциональный): input
        input_loaded: _LoadedArtifact | None = None
        if lora_request.train_input_too:
            input_loaded = TrainingService._resolve_or_train_input(
                req=req, lora_request=lora_request,
                ticker=ticker, source_name=source_name, market=market,
                model_name=model_name_resolved, model=model, candles=candles, source=source,
                progress_callback=progress_callback, cancel_check=cancel_check,
                stage_callback=stage_callback,
                stage_finished_callback=stage_finished_callback,
            )
            TrainingService._apply_input_to_model(model, input_loaded.state_dict)

        # Шаг 3: LoRA. Передаём id уже обученных head/input — оркестратор
        # сохранит их в params LoRA-артефакта, чтобы ArtifactLoader потом
        # знал какие компоненты нужно подгрузить ВМЕСТЕ с LoRA.
        lora_loaded = TrainingService._train_and_save_lora(
            req=req, lora_request=lora_request, model=model, candles=candles, source=source,
            base_head_artifact_id=head_loaded.artifact_id,
            base_input_artifact_id=input_loaded.artifact_id if input_loaded else None,
            progress_callback=progress_callback, cancel_check=cancel_check,
            stage_callback=stage_callback,
            stage_finished_callback=stage_finished_callback,
        )

        return {
            "artifact_id":          lora_loaded.artifact_id,
            "status":               "ready",
            "training_components":  ["lora"],
            "head_artifact_id":     head_loaded.artifact_id,
            "input_artifact_id":    input_loaded.artifact_id if input_loaded else None,
            "lora_artifact_id":     lora_loaded.artifact_id,
        }

    # ==================================================================
    # STAGE: head
    # ==================================================================

    @staticmethod
    def _train_and_save_head(
            *,
            req: TrainingRequestInternal,
            model: ForecastModel,
            candles: list[dict[str, float]],
            source: str,
            progress_callback: ProgressCallback | None,
            cancel_check: CancelCheck | None,
            stage_callback: StageCallback | None,
            stage_finished_callback: StageFinishedCallback | None,
            version_override: str | None = None,
    ) -> _LoadedArtifact:
        if stage_callback:
            stage_callback("head")

        trainer = HeadTrainer(base_model=model)
        config = HeadTrainingConfig(
            train_window_size=req.train_window_size,
            horizon=req.horizon,
            step=req.step,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            num_epochs=req.num_epochs,
            val_split=req.val_split,
            evaluation_weights=req.evaluation_weights,
            weight_first_to_last_ratio=req.weight_first_to_last_ratio,
            num_workers=req.num_workers,
        )
        t_start = time.time()
        result = trainer.train(
            candles, config,
            progress_callback=TrainingService._wrap_progress(progress_callback, "head", config.num_epochs),
            cancel_check=cancel_check,
        )
        elapsed = time.time() - t_start

        ticker, source_name, market = TrainingService._extract_ticker_info(req)
        model_info = model.get_info()
        save_ctx = SaveContext(
            symbol=ticker, source=source_name, market=market,
            interval=req.interval,
            model_name=model_info.get("name", "unknown"),
            training_components=["head"],
            train_window_size=req.train_window_size,
            horizon=req.horizon,
            version=version_override or req.version,
            base_model_id=model_info.get("model_id", ""),
            base_model_version=model_info.get("version", ""),
            training_params=TrainingService._snapshot_params(req),
        )
        metrics = TrainingService._build_head_input_metrics(result, candles, elapsed)
        artifact_id = ArtifactSaver.save_head(
            head_state_dict=result.head_state_dict, metrics=metrics, context=save_ctx,
        )

        if stage_finished_callback:
            stage_finished_callback("head", artifact_id)

        return _LoadedArtifact(
            artifact_id=artifact_id, component="head",
            state_dict=result.head_state_dict,
        )

    @staticmethod
    def _resolve_or_train_head(
            *,
            req: TrainingRequestInternal,
            lora_request: LoraTrainingRequest,
            ticker: str, source_name: str, market: str | None,
            model_name: str,
            model: ForecastModel,
            candles: list[dict[str, float]],
            source: str,
            progress_callback: ProgressCallback | None,
            cancel_check: CancelCheck | None,
            stage_callback: StageCallback | None,
            stage_finished_callback: StageFinishedCallback | None,
    ) -> _LoadedArtifact:
        """Ищет совместимый head в БД, либо тренирует новый."""
        match_req = MatchRequest(
            symbol=ticker, source=source_name, interval=req.interval,
            model_name=model_name,
            training_components=["head"],
            train_window_size=req.train_window_size, horizon=req.horizon,
            learning_rate=req.learning_rate, num_epochs=req.num_epochs,
            batch_size=req.batch_size, evaluation_weights=req.evaluation_weights,
            weight_first_to_last_ratio=req.weight_first_to_last_ratio,
        )
        existing = ArtifactMatcher.find(
            match_req, force_new=lora_request.force_new_head_or_input,
        )
        if existing is not None:
            state_dict = TrainingService._load_head_state(existing)
            logger.info("[TrainingService] head reused from artifact #%d", existing.id)
            return _LoadedArtifact(
                artifact_id=existing.id, component="head", state_dict=state_dict,
            )

        # Не нашли (или force_new) — тренируем
        version_override = lora_request.new_head_or_input_version if lora_request.force_new_head_or_input else None
        return TrainingService._train_and_save_head(
            req=req, model=model, candles=candles, source=source,
            progress_callback=progress_callback, cancel_check=cancel_check,
            stage_callback=stage_callback,
            stage_finished_callback=stage_finished_callback,
            version_override=version_override,
        )

    # ==================================================================
    # STAGE: input
    # ==================================================================

    @staticmethod
    def _train_and_save_input(
            *,
            req: TrainingRequestInternal,
            model: ForecastModel,
            candles: list[dict[str, float]],
            source: str,
            progress_callback: ProgressCallback | None,
            cancel_check: CancelCheck | None,
            stage_callback: StageCallback | None,
            stage_finished_callback: StageFinishedCallback | None,
            version_override: str | None = None,
    ) -> _LoadedArtifact:
        if stage_callback:
            stage_callback("input")

        trainer = InputTrainer(base_model=model)
        config = InputTrainingConfig(
            train_window_size=req.train_window_size,
            horizon=req.horizon,
            step=req.step,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            num_epochs=req.num_epochs,
            val_split=req.val_split,
            evaluation_weights=req.evaluation_weights,
            weight_first_to_last_ratio=req.weight_first_to_last_ratio,
            num_workers=req.num_workers,
        )
        t_start = time.time()
        result = trainer.train(
            candles, config,
            progress_callback=TrainingService._wrap_progress(progress_callback, "input", config.num_epochs),
            cancel_check=cancel_check,
        )
        elapsed = time.time() - t_start

        ticker, source_name, market = TrainingService._extract_ticker_info(req)
        model_info = model.get_info()
        save_ctx = SaveContext(
            symbol=ticker, source=source_name, market=market,
            interval=req.interval,
            model_name=model_info.get("name", "unknown"),
            training_components=["input"],
            train_window_size=req.train_window_size,
            horizon=req.horizon,
            version=version_override or req.version,
            base_model_id=model_info.get("model_id", ""),
            base_model_version=model_info.get("version", ""),
            training_params=TrainingService._snapshot_params(req),
        )
        metrics = TrainingService._build_head_input_metrics(result, candles, elapsed)
        artifact_id = ArtifactSaver.save_input(
            input_state_dict=result.input_state_dict, metrics=metrics, context=save_ctx,
        )

        if stage_finished_callback:
            stage_finished_callback("input", artifact_id)

        return _LoadedArtifact(
            artifact_id=artifact_id, component="input",
            state_dict=result.input_state_dict,
        )

    @staticmethod
    def _resolve_or_train_input(
            *,
            req: TrainingRequestInternal,
            lora_request: LoraTrainingRequest,
            ticker: str, source_name: str, market: str | None,
            model_name: str,
            model: ForecastModel,
            candles: list[dict[str, float]],
            source: str,
            progress_callback: ProgressCallback | None,
            cancel_check: CancelCheck | None,
            stage_callback: StageCallback | None,
            stage_finished_callback: StageFinishedCallback | None,
    ) -> _LoadedArtifact:
        match_req = MatchRequest(
            symbol=ticker, source=source_name, interval=req.interval,
            model_name=model_name,
            training_components=["input"],
            train_window_size=req.train_window_size, horizon=req.horizon,
            learning_rate=req.learning_rate, num_epochs=req.num_epochs,
            batch_size=req.batch_size, evaluation_weights=req.evaluation_weights,
            weight_first_to_last_ratio=req.weight_first_to_last_ratio,
        )
        existing = ArtifactMatcher.find(
            match_req, force_new=lora_request.force_new_head_or_input,
        )
        if existing is not None:
            state_dict = TrainingService._load_input_state(existing)
            logger.info("[TrainingService] input reused from artifact #%d", existing.id)
            return _LoadedArtifact(
                artifact_id=existing.id, component="input", state_dict=state_dict,
            )

        version_override = lora_request.new_head_or_input_version if lora_request.force_new_head_or_input else None
        return TrainingService._train_and_save_input(
            req=req, model=model, candles=candles, source=source,
            progress_callback=progress_callback, cancel_check=cancel_check,
            stage_callback=stage_callback,
            stage_finished_callback=stage_finished_callback,
            version_override=version_override,
        )

    # ==================================================================
    # STAGE: lora
    # ==================================================================

    @staticmethod
    def _train_and_save_lora(
            *,
            req: TrainingRequestInternal,
            lora_request: LoraTrainingRequest,
            model: ForecastModel,
            candles: list[dict[str, float]],
            source: str,
            base_head_artifact_id: int | None,
            base_input_artifact_id: int | None,
            progress_callback: ProgressCallback | None,
            cancel_check: CancelCheck | None,
            stage_callback: StageCallback | None,
            stage_finished_callback: StageFinishedCallback | None,
    ) -> _LoadedArtifact:
        if stage_callback:
            stage_callback("lora")

        trainer = LoraTrainer(base_model=model)
        config = LoraTrainingConfig(
            train_window_size=req.train_window_size,
            horizon=req.horizon,
            step=req.step,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            num_epochs=req.num_epochs,
            val_split=req.val_split,
            evaluation_weights=req.evaluation_weights,
            weight_first_to_last_ratio=req.weight_first_to_last_ratio,
            num_workers=req.num_workers,
            lora_r=lora_request.lora_r,
            lora_alpha=lora_request.lora_alpha,
            lora_dropout=lora_request.lora_dropout,
            lora_target_modules=list(lora_request.lora_target_modules),
            # train_head_too УБРАН — head всегда уже загружена до этой стадии
            train_head_too=False,
        )
        t_start = time.time()
        result = trainer.train(
            candles, config,
            progress_callback=TrainingService._wrap_progress(progress_callback, "lora", config.num_epochs),
            cancel_check=cancel_check,
        )
        elapsed = time.time() - t_start

        ticker, source_name, market = TrainingService._extract_ticker_info(req)
        model_info = model.get_info()
        save_ctx = SaveContext(
            symbol=ticker, source=source_name, market=market,
            interval=req.interval,
            model_name=model_info.get("name", "unknown"),
            training_components=["lora"],
            train_window_size=req.train_window_size,
            horizon=req.horizon,
            version=req.version,
            base_model_id=model_info.get("model_id", ""),
            base_model_version=model_info.get("version", ""),
            training_params={
                **TrainingService._snapshot_params(req),
                "lora_r":       lora_request.lora_r,
                "lora_alpha":   lora_request.lora_alpha,
                "lora_dropout": lora_request.lora_dropout,
                "lora_target_modules": list(lora_request.lora_target_modules),
                # Link на head/input артефакты — ArtifactLoader увидит и
                # подгрузит их перед применением самой LoRA. None если не было.
                "base_head_artifact_id":  base_head_artifact_id,
                "base_input_artifact_id": base_input_artifact_id,
            },
        )
        metrics = {
            "final_train_loss":         result.final_train_loss,
            "final_val_loss":           result.final_val_loss,
            "epochs_completed":         float(result.epochs_completed),
            "train_history":            result.train_history,
            "val_history":              result.val_history,
            "trainable_params":         float(result.trainable_params),
            "total_params":             float(result.total_params),
            "total_candles_trained_on": len(candles),
            "data_range_from":          (candles[0].get("date") if candles else None),
            "data_range_to":            (candles[-1].get("date") if candles else None),
            "training_duration_s":      round(elapsed, 3),
        }
        artifact_id = ArtifactSaver.save_lora(
            peft_model=result.peft_model, metrics=metrics, context=save_ctx,
        )

        if stage_finished_callback:
            stage_finished_callback("lora", artifact_id)

        return _LoadedArtifact(
            artifact_id=artifact_id, component="lora", peft_model=result.peft_model,
        )

    # ==================================================================
    # HELPERS
    # ==================================================================

    @staticmethod
    def _build_resolver_request(req: TrainingRequestInternal):
        """InputResolver принимает ForecastRequest-подобный объект для резолва данных."""
        return ForecastRequest(
            model_name=req.model_name,
            data_source=req.data_source,
            provider_options=req.provider_options,
            history_period=req.history_period,
            history_up_to=req.history_up_to,
            interval=req.interval,
            horizon=req.horizon,
        )

    @staticmethod
    def _extract_ticker_info(req: TrainingRequestInternal) -> tuple[str, str, str | None]:
        """Triple (ticker, source, market) для записи в model_artifacts."""
        opts = req.provider_options
        if isinstance(opts, TInvestProviderOptions):
            return (opts.ticker or "").upper(), "t_invest", opts.class_code
        if isinstance(opts, YahooProviderOptions):
            return (opts.ticker or "").upper(), "yahoo", None
        if isinstance(opts, CsvProviderOptions):
            return opts.csv_path, "csv", None
        return "unknown", req.data_source or "unknown", None

    @staticmethod
    def _snapshot_params(req: TrainingRequestInternal) -> dict[str, Any]:
        """Снимок training-гиперпараметров для записи в metrics_json артефакта."""
        return {
            "horizon":                    req.horizon,
            "step":                       req.step,
            "batch_size":                 req.batch_size,
            "learning_rate":              req.learning_rate,
            "num_epochs":                 req.num_epochs,
            "val_split":                  req.val_split,
            "evaluation_weights":         req.evaluation_weights,
            "weight_first_to_last_ratio": req.weight_first_to_last_ratio,
            "history_period":             req.history_period,
            "history_up_to":              req.history_up_to,
        }

    @staticmethod
    def _build_head_input_metrics(
            result, candles: list[dict[str, float]], elapsed: float,
    ) -> dict[str, Any]:
        """Расширенные метрики для head/input артефактов."""
        return {
            "final_train_loss":         result.final_train_loss,
            "final_val_loss":           result.final_val_loss,
            "epochs_completed":         float(result.epochs_completed),
            "train_history":            result.train_history,
            "val_history":              result.val_history,
            "total_candles_trained_on": len(candles),
            "data_range_from":          (candles[0].get("date") if candles else None),
            "data_range_to":            (candles[-1].get("date") if candles else None),
            "training_duration_s":      round(elapsed, 3),
        }

    @staticmethod
    def _wrap_progress(
            cb: ProgressCallback | None, stage: str, total_epochs: int,
    ) -> ProgressCallback | None:
        """
        Оборачивает progress_callback из trainer'а так чтобы добавить stage в payload.
        Trainer не знает что он часть многостадийной цепочки — это знает оркестратор.
        """
        if cb is None:
            return None

        def _wrapped(payload: dict) -> None:
            # Дописываем stage в payload, чтобы router'овский логгер мог
            # дифференцировать сообщения от разных стейджей в одном job'е.
            payload = {**payload, "stage": stage, "stage_total_epochs": total_epochs}
            cb(payload)

        return _wrapped

    # ------------------------------------------------------------------
    # Load/apply state dicts of existing artifacts
    # ------------------------------------------------------------------

    @staticmethod
    def _load_head_state(artifact) -> dict:
        return TrainingService._load_state_pt(artifact, filename="head.pt")

    @staticmethod
    def _load_input_state(artifact) -> dict:
        return TrainingService._load_state_pt(artifact, filename="input.pt")

    @staticmethod
    def _load_state_pt(artifact, filename: str) -> dict:
        """Грузит state_dict из data/artifacts/{id}/{filename}."""
        import torch
        from pathlib import Path
        path = Path(artifact.artifact_path) / filename
        if not path.exists():
            raise RuntimeError(
                f"Артефакт #{artifact.id} есть в БД, но файл {filename} не найден "
                f"по пути {path}. Возможно артефакт повреждён."
            )
        return torch.load(path, map_location="cpu", weights_only=True)

    @staticmethod
    def _apply_head_to_model(model: ForecastModel, head_state: dict) -> None:
        """Накатывает head state_dict в adapter-модуль 'head' модели."""
        adapter_modules = model.get_adapter_modules()
        adapter_modules["head"].load_state_dict(head_state)
        logger.info("[TrainingService] head state_dict применён в adapter['head']")

    @staticmethod
    def _apply_input_to_model(model: ForecastModel, input_state: dict) -> None:
        """Накатывает input state_dict в adapter-модуль 'input' модели."""
        adapter_modules = model.get_adapter_modules()
        adapter_modules["input"].load_state_dict(input_state)
        logger.info("[TrainingService] input state_dict применён в adapter['input']")
