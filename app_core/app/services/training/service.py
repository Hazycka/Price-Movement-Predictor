"""
TrainingService — оркестрация обучения: head / lora / комбо.

Высокоуровневый поток:
  1. Принять TrainingRequest
  2. Зарезолвить данные (через InputResolver) — те же candles что и для backtest
  3. Создать base модель через ModelFactory
  4. Выбрать тип обучения (head / lora / combo) → запустить trainer
  5. Сохранить результат через ArtifactSaver → получить artifact_id
  6. Вернуть TrainingResult

Это синхронный путь — endpoint блокируется до конца обучения. В Phase 3 обернётся
в background task для возврата 202 Accepted сразу.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from ...models.factory import create_model
from ...schemas import (
    ForecastRequest, ProviderOptions,
    TInvestProviderOptions, YahooProviderOptions, CsvProviderOptions,
)
from ..forecast.input_resolver import InputResolver
from .head_trainer import HeadTrainer, HeadTrainingConfig
from .lora_trainer import LoraTrainer, LoraTrainingConfig
from .saver import ArtifactSaver, SaveContext


# Тип progress callback: получает dict от trainer'а с полями
# phase / epoch / batch / loss / elapsed_s / eta_s / etc.
ProgressCallback = Callable[[dict], None]

logger = logging.getLogger(__name__)


@dataclass
class TrainingRequestInternal:
    """
    Внутренний контекст запроса. Не Pydantic — мы конвертим публичный
    HeadTrainingRequest из API в это.
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


class TrainingService:

    @staticmethod
    def train_head(
            req: TrainingRequestInternal,
            progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """
        Запускает обучение новой output head.
        Возвращает dict: {artifact_id, metrics, status}.

        progress_callback (если задан) транслирует каждое событие trainer'а вверх
        (для записи в JobStore). Формат — dict, см. HeadTrainer.train().
        """
        # 1. Резолвим данные через InputResolver
        forecast_like_req = TrainingService._build_resolver_request(req)
        source, _dates, candles = InputResolver.resolve(forecast_like_req)
        logger.info(
            "[TrainingService] head: source=%s len(candles)=%d",
            source, len(candles),
        )

        # 2. Создаём base модель
        model = create_model(req.model_name)
        # Заставляем загрузить веса сейчас (а не лениво при первом inference)
        model._runtime.ensure_loaded()

        # 3. Запускаем head trainer
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
        result = trainer.train(candles, config, progress_callback=progress_callback)

        # 4. Сохраняем артефакт
        ticker, market = TrainingService._extract_ticker_market(req)
        model_info = model.get_info()
        save_ctx = SaveContext(
            symbol=ticker,
            market=market,
            interval=req.interval,
            model_name=model_info.get("name", "unknown"),
            training_components=["head"],
            train_window_size=req.train_window_size,
            horizon=req.horizon,
            version=req.version,
            base_model_id=model_info.get("model_id", ""),
            base_model_version=model_info.get("version", ""),
            training_params={
                "horizon": req.horizon,
                "step": req.step,
                "batch_size": req.batch_size,
                "learning_rate": req.learning_rate,
                "num_epochs": req.num_epochs,
                "val_split": req.val_split,
                "evaluation_weights": req.evaluation_weights,
                "weight_first_to_last_ratio": req.weight_first_to_last_ratio,
                "history_period": req.history_period,
                "history_up_to": req.history_up_to,
            },
        )
        metrics = {
            "final_train_loss": result.final_train_loss,
            "final_val_loss":   result.final_val_loss,
            "epochs_completed": float(result.epochs_completed),
            "train_history":    result.train_history,
            "val_history":      result.val_history,
        }
        artifact_id = ArtifactSaver.save_head(
            head_state_dict=result.head_state_dict,
            metrics=metrics,
            context=save_ctx,
        )

        return {
            "artifact_id":   artifact_id,
            "status":        "ready",
            "training_components": ["head"],
            "metrics":       metrics,
        }

    @staticmethod
    def train_lora(
            req: TrainingRequestInternal,
            lora_request,
            progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """
        Обучает LoRA-адаптеры (+опционально новую head).
        lora_request — Pydantic LoraTrainingRequest со специфичными полями.
        progress_callback — см. train_head().
        Возвращает dict: {artifact_id, metrics, status, training_components}.
        """
        forecast_like_req = TrainingService._build_resolver_request(req)
        source, _dates, candles = InputResolver.resolve(forecast_like_req)
        logger.info("[TrainingService] lora: source=%s len(candles)=%d", source, len(candles))

        model = create_model(req.model_name)
        model._runtime.ensure_loaded()

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
            train_head_too=lora_request.train_head_too,
        )
        result = trainer.train(candles, config, progress_callback=progress_callback)

        ticker, market = TrainingService._extract_ticker_market(req)
        model_info = model.get_info()
        components = ["lora", "head"] if lora_request.train_head_too else ["lora"]

        save_ctx = SaveContext(
            symbol=ticker,
            market=market,
            interval=req.interval,
            model_name=model_info.get("name", "unknown"),
            training_components=components,
            train_window_size=req.train_window_size,
            horizon=req.horizon,
            version=req.version,
            base_model_id=model_info.get("model_id", ""),
            base_model_version=model_info.get("version", ""),
            training_params={
                "horizon": req.horizon,
                "step": req.step,
                "batch_size": req.batch_size,
                "learning_rate": req.learning_rate,
                "num_epochs": req.num_epochs,
                "val_split": req.val_split,
                "evaluation_weights": req.evaluation_weights,
                "weight_first_to_last_ratio": req.weight_first_to_last_ratio,
                "history_period": req.history_period,
                "history_up_to": req.history_up_to,
                "lora_r": lora_request.lora_r,
                "lora_alpha": lora_request.lora_alpha,
                "lora_dropout": lora_request.lora_dropout,
                "lora_target_modules": list(lora_request.lora_target_modules),
                "train_head_too": lora_request.train_head_too,
            },
        )
        metrics = {
            "final_train_loss": result.final_train_loss,
            "final_val_loss":   result.final_val_loss,
            "epochs_completed": float(result.epochs_completed),
            "train_history":    result.train_history,
            "val_history":      result.val_history,
            "trainable_params": float(result.trainable_params),
            "total_params":     float(result.total_params),
        }

        if lora_request.train_head_too:
            artifact_id = ArtifactSaver.save_combo(
                head_state_dict=result.head_state_dict,
                peft_model=result.peft_model,
                metrics=metrics,
                context=save_ctx,
            )
        else:
            artifact_id = ArtifactSaver.save_lora(
                peft_model=result.peft_model,
                metrics=metrics,
                context=save_ctx,
            )

        return {
            "artifact_id":   artifact_id,
            "status":        "ready",
            "training_components": components,
            "metrics":       metrics,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_resolver_request(req: TrainingRequestInternal):
        """
        InputResolver принимает ForecastRequest/BacktestRequest/SweepRequest —
        мы создаём минимальный совместимый объект для резолва данных.
        """
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
    def _extract_ticker_market(req: TrainingRequestInternal) -> tuple[str, str | None]:
        """Достаёт ticker и market для записи в model_artifacts."""
        opts = req.provider_options
        if isinstance(opts, TInvestProviderOptions):
            return (opts.ticker or "").upper(), opts.class_code
        if isinstance(opts, YahooProviderOptions):
            return (opts.ticker or "").upper(), None
        if isinstance(opts, CsvProviderOptions):
            return opts.csv_path, None
        return "unknown", None
