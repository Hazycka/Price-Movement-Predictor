"""
InputTrainer — обучение только input projection (`backbone.in_layer`).

Логика идентична HeadTrainer, но обучаемый модуль — `backbone.in_layer`
(ResidualBlock: patch embedding, d_patch*2 → d_model). Это «адаптер на входе»,
который учит модель «правильно читать» сырые патчи OHLC.

Каноничный порядок применения в pipeline:
  1. head — линейная пробинг (классический LP)
  2. input — адаптируется поверх обученной head'ы
  3. lora — поверх обоих

Этот модуль НЕ имеет публичного API — вызывается только из TrainingService
оркестратора (в составе head→input или head→input→lora цепочки).

Подробно про мотивацию двойной адаптации и порядок — см. design-обсуждение
в комментариях TrainingService.

lr по умолчанию = 1e-4 (в 5× меньше чем у head): изменения input «расходятся»
сильнее по сети, осторожнее с шагом.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    DataLoader = None

from .dataset import WalkForwardDataset
from .exceptions import TrainingCancelledException
from .head_trainer import _fmt_elapsed
from .loss import weighted_pinball_loss
from ..backtest.weights import compute_weights

logger = logging.getLogger(__name__)


ProgressCallback = Callable[[dict], None]
CancelCheck = Callable[[], bool]


@dataclass
class InputTrainingConfig:
    """Гиперпараметры обучения input projection."""
    train_window_size: int
    horizon: int
    step: int = 64
    batch_size: int = 16
    learning_rate: float = 1e-4   # < head'ы — input влияет на всю сеть глубже
    num_epochs: int = 5
    val_split: float = 0.15
    evaluation_weights: str = "exponential"
    weight_first_to_last_ratio: float = 16.0
    num_workers: int = 0
    log_every_n_batches: int = 5


@dataclass
class InputTrainingResult:
    input_state_dict: dict
    final_train_loss: float
    final_val_loss: float
    epochs_completed: int
    train_history: list[float]
    val_history: list[float]


class InputTrainer:
    """
    Тренирует только `backbone.in_layer` модели PatchTST FM. Все остальное
    заморожено (включая head, если она уже загружена из артефакта).

    Если в base_model уже загружена обученная head — это нормально и желательно.
    Loss + градиенты пройдут через неё (она заморожена → не обновится), что даст
    input'у «правильную» цель: подавать репрезентации, которые head хорошо
    декодирует.
    """

    def __init__(self, base_model) -> None:
        if torch is None:
            raise RuntimeError("PyTorch не установлен.")
        self.base_model = base_model
        base_model._runtime.ensure_loaded()
        self.torch_model = base_model._runtime.model
        self.quantile_levels = base_model._runtime.quantile_levels
        self.device = base_model._runtime.device

    def train(
            self,
            candles: list[dict[str, float]],
            config: InputTrainingConfig,
            progress_callback: ProgressCallback | None = None,
            cancel_check: CancelCheck | None = None,
    ) -> InputTrainingResult:
        """
        Возвращает InputTrainingResult. progress_callback — формат идентичен
        HeadTrainer. cancel_check — функция без аргументов: True означает
        «job отменён», бросаем TrainingCancelledException.
        """
        def _emit(payload: dict) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(payload)
                except Exception as ex:
                    logger.warning("[InputTrainer] progress_callback failed: %s", ex)

        def _check_cancel() -> None:
            if cancel_check is not None and cancel_check():
                raise TrainingCancelledException(stage="input")

        # 1) Получаем input через абстракцию модели
        adapter_modules = self.base_model.get_adapter_modules()
        if "input" not in adapter_modules:
            raise RuntimeError(
                f"Модель {type(self.base_model).__name__} не предоставляет 'input' "
                f"в get_adapter_modules(). Доступны: {list(adapter_modules)}."
            )
        input_module = adapter_modules["input"]
        logger.info(
            "[InputTrainer] input module: %s (%d params)",
            type(input_module).__name__,
            sum(p.numel() for p in input_module.parameters()),
        )
        self._freeze_all_except(input_module)

        # 2) Готовим datasets (идентично HeadTrainer)
        ds_train = WalkForwardDataset(
            candles=candles, train_window_size=config.train_window_size,
            horizon=config.horizon, step=config.step,
            val_split=config.val_split, mode="train",
        )
        ds_val = WalkForwardDataset(
            candles=candles, train_window_size=config.train_window_size,
            horizon=config.horizon, step=config.step,
            val_split=config.val_split, mode="val",
        )

        loader_train = DataLoader(
            ds_train, batch_size=config.batch_size,
            shuffle=True, num_workers=config.num_workers,
        )
        loader_val = DataLoader(
            ds_val, batch_size=config.batch_size,
            shuffle=False, num_workers=config.num_workers,
        ) if len(ds_val) > 0 else None

        # 3) Веса горизонта
        weights_list = compute_weights(
            horizon=config.horizon,
            scheme=config.evaluation_weights,
            first_to_last_ratio=config.weight_first_to_last_ratio,
        )
        weights_tensor = torch.tensor(weights_list, dtype=torch.float32, device=self.device)

        # 4) Optimizer — только параметры input
        trainable_params = [p for p in self.torch_model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("После замораживания не осталось обучаемых параметров.")
        optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

        total_batches_per_epoch = len(loader_train)
        total_batches = total_batches_per_epoch * config.num_epochs
        logger.info(
            "[InputTrainer] Starting: %d epochs × %d train batches = %d total | "
            "%d val batches | %d trainable params | device=%s",
            config.num_epochs, total_batches_per_epoch, total_batches,
            len(loader_val) if loader_val else 0,
            sum(p.numel() for p in trainable_params), self.device,
        )
        _emit({
            "phase": "start",
            "total_epochs": config.num_epochs,
            "total_batches": total_batches,
            "trainable_params": sum(p.numel() for p in trainable_params),
            "device": str(self.device),
        })

        # 5) Training loop
        train_history: list[float] = []
        val_history: list[float] = []
        t0 = time.perf_counter()
        global_batch = 0
        for epoch in range(config.num_epochs):
            self.torch_model.train()
            epoch_train_losses = []
            for batch_idx, (past, future) in enumerate(loader_train):
                _check_cancel()
                past = past.to(self.device)
                future = future.to(self.device)

                optimizer.zero_grad()
                pred_quantiles = self._forward_quantiles(past, config.horizon)
                loss = weighted_pinball_loss(
                    pred_quantiles, future, weights_tensor, self.quantile_levels,
                )
                loss.backward()
                optimizer.step()
                epoch_train_losses.append(loss.item())

                global_batch += 1
                if (batch_idx + 1) % config.log_every_n_batches == 0 or batch_idx == total_batches_per_epoch - 1:
                    elapsed = time.perf_counter() - t0
                    eta = (elapsed / global_batch) * (total_batches - global_batch) if global_batch else 0.0
                    running_avg = sum(epoch_train_losses) / len(epoch_train_losses)
                    logger.info(
                        "[InputTrainer] epoch %d/%d batch %d/%d (overall %d/%d %.0f%%) "
                        "loss=%.4f running=%.4f elapsed=%s eta=%s",
                        epoch + 1, config.num_epochs,
                        batch_idx + 1, total_batches_per_epoch,
                        global_batch, total_batches,
                        100 * global_batch / total_batches,
                        loss.item(), running_avg,
                        _fmt_elapsed(elapsed), _fmt_elapsed(eta),
                    )
                    _emit({
                        "phase": "train",
                        "epoch": epoch + 1, "total_epochs": config.num_epochs,
                        "batch": batch_idx + 1, "total_batches_per_epoch": total_batches_per_epoch,
                        "global_batch": global_batch, "total_batches": total_batches,
                        "loss": float(loss.item()),
                        "running_avg_loss": float(running_avg),
                        "elapsed_s": elapsed, "eta_s": eta,
                    })

            avg_train = sum(epoch_train_losses) / len(epoch_train_losses)
            train_history.append(avg_train)

            avg_val = 0.0
            if loader_val is not None:
                self.torch_model.eval()
                with torch.no_grad():
                    val_losses = []
                    for past, future in loader_val:
                        past = past.to(self.device)
                        future = future.to(self.device)
                        pred_quantiles = self._forward_quantiles(past, config.horizon)
                        l = weighted_pinball_loss(
                            pred_quantiles, future, weights_tensor, self.quantile_levels,
                        )
                        val_losses.append(l.item())
                    avg_val = sum(val_losses) / len(val_losses) if val_losses else 0.0
                val_history.append(avg_val)

            elapsed = time.perf_counter() - t0
            logger.info(
                "[InputTrainer] Epoch %d/%d done: train_loss=%.4f val_loss=%.4f elapsed=%s",
                epoch + 1, config.num_epochs, avg_train, avg_val, _fmt_elapsed(elapsed),
            )
            _emit({
                "phase": "epoch_end",
                "epoch": epoch + 1, "total_epochs": config.num_epochs,
                "train_loss": float(avg_train), "val_loss": float(avg_val),
                "elapsed_s": elapsed,
            })

        # 6) Снимок state_dict
        input_state = {k: v.detach().cpu() for k, v in input_module.state_dict().items()}

        return InputTrainingResult(
            input_state_dict=input_state,
            final_train_loss=train_history[-1] if train_history else 0.0,
            final_val_loss=val_history[-1] if val_history else 0.0,
            epochs_completed=config.num_epochs,
            train_history=train_history,
            val_history=val_history,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _freeze_all_except(self, input_module) -> None:
        """Замораживает все параметры кроме input_module."""
        input_params = set(id(p) for p in input_module.parameters())
        frozen = 0
        trainable = 0
        for p in self.torch_model.parameters():
            if id(p) in input_params:
                p.requires_grad = True
                trainable += p.numel()
            else:
                p.requires_grad = False
                frozen += p.numel()
        logger.info(
            "[InputTrainer] Заморожено %d параметров, обучаемо %d (%.2f%%)",
            frozen, trainable, 100 * trainable / (frozen + trainable),
        )

    def _forward_quantiles(self, past_values, horizon: int):
        """См. HeadTrainer._forward_quantiles — логика идентична."""
        outputs = self.torch_model(
            past_values=past_values,
            prediction_length=horizon,
            quantile_levels=list(self.quantile_levels),
        )
        quantile_preds = getattr(outputs, "quantile_outputs", None)
        if quantile_preds is not None:
            return quantile_preds.permute(0, 2, 1, 3)
        raise RuntimeError(
            f"Не удалось извлечь quantile_outputs из модели. "
            f"type={type(outputs).__name__}"
        )
