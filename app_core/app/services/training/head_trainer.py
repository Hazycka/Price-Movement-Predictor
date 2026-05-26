"""
HeadTrainer — обучение только новой output head поверх замороженной PatchTST FM.

Linear probing — самый простой способ адаптации foundation-модели:
  1. Замораживаем ВСЕ параметры base модели (requires_grad=False)
  2. Заменяем head на свежеинициализированную (или сохраняем оригинальную и обучаем
     с нуля, в зависимости от подхода)
  3. Тренируем только head'у на weighted pinball loss с walk-forward данными
  4. Сохраняем state_dict head в head.pt

Преимущества над LoRA:
  - Меньше параметров для обучения (~10K vs ~1M у LoRA)
  - Быстрее
  - Проще ничего не сломать

Недостатки:
  - Меньше выразительности — внутренние представления модели не меняются
  - Может не дать значительного улучшения если base модель плохо ловит паттерны

Использовать как baseline ПЕРЕД LoRA.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    DataLoader = None

from .dataset import WalkForwardDataset
from .exceptions import TrainingCancelledException
from .loss import weighted_pinball_loss
from ..backtest.weights import compute_weights

logger = logging.getLogger(__name__)


# Тип progress callback: получает dict с метриками текущего шага.
# Caller использует для записи в JobStore + консольный лог.
ProgressCallback = Callable[[dict], None]

# Тип cancel check: функция без аргументов, True означает «job отменён».
# Trainer проверяет между батчами и бросает TrainingCancelledException.
CancelCheck = Callable[[], bool]


def _fmt_elapsed(sec: float) -> str:
    """1m23s / 45s / 1h05m формат."""
    if sec < 60:
        return f"{int(sec)}s"
    if sec < 3600:
        return f"{int(sec // 60)}m{int(sec % 60):02d}s"
    return f"{int(sec // 3600)}h{int((sec % 3600) // 60):02d}m"


@dataclass
class HeadTrainingConfig:
    """Гиперпараметры обучения head."""
    train_window_size: int
    horizon: int
    step: int = 64
    batch_size: int = 16
    learning_rate: float = 5e-4
    num_epochs: int = 5
    val_split: float = 0.15
    evaluation_weights: str = "exponential"
    weight_first_to_last_ratio: float = 16.0
    num_workers: int = 0   # для DataLoader (0 — без подпроцессов, безопаснее на Windows)
    log_every_n_batches: int = 5   # частота progress-логов внутри эпохи


@dataclass
class HeadTrainingResult:
    """Результат обучения head."""
    head_state_dict: dict
    final_train_loss: float
    final_val_loss: float
    epochs_completed: int
    train_history: list[float]
    val_history: list[float]


class HeadTrainer:
    """
    Обучает только output head модели PatchTST FM на walk-forward данных.

    Использование:
        trainer = HeadTrainer(base_model)
        result = trainer.train(candles, HeadTrainingConfig(train_window_size=8192, horizon=64))
        saver.save(result.head_state_dict, ...)
    """

    def __init__(self, base_model) -> None:
        """
        base_model — PatchTSTForecastModel (наш wrapper) с уже загруженной моделью.
        """
        if torch is None:
            raise RuntimeError("PyTorch не установлен.")
        self.base_model = base_model
        # Достаём внутренний torch-модуль через runtime
        base_model._runtime.ensure_loaded()
        self.torch_model = base_model._runtime.model
        self.quantile_levels = base_model._runtime.quantile_levels
        self.device = base_model._runtime.device

    def train(
            self,
            candles: list[dict[str, float]],
            config: HeadTrainingConfig,
            progress_callback: ProgressCallback | None = None,
            cancel_check: CancelCheck | None = None,
    ) -> HeadTrainingResult:
        """
        Обучает новую output head на candles, возвращает state_dict + метрики.

        progress_callback (если задан) вызывается каждые config.log_every_n_batches
        батчей и в конце каждой эпохи. Получает dict вида:
          {"phase": "train"|"val"|"epoch_end", "epoch": int, "total_epochs": int,
           "batch": int, "total_batches": int, "loss": float,
           "elapsed_s": float, "eta_s": float}
        Caller (TrainingService) транслирует это в JobStore + консольный лог.

        cancel_check (если задан) — функция без аргументов, вызывается между батчами.
        Если возвращает True — поднимаем TrainingCancelledException и обучение
        останавливается. Caller ловит и помечает job как cancelled.
        """
        def _emit(payload: dict) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(payload)
                except Exception as ex:  # noqa - не валим training из-за progress-callback
                    logger.warning("[HeadTrainer] progress_callback failed: %s", ex)

        def _check_cancel() -> None:
            if cancel_check is not None and cancel_check():
                raise TrainingCancelledException(stage="head")

        # 1) Получаем head через абстракцию модели (PatchTSTFM = backbone.out_layer,
        # Moirai = param_proj и т.д.). См. ForecastModel.get_adapter_modules.
        adapter_modules = self.base_model.get_adapter_modules()
        if "head" not in adapter_modules:
            raise RuntimeError(
                f"Модель {type(self.base_model).__name__} не предоставляет 'head' "
                f"в get_adapter_modules(). Доступны: {list(adapter_modules)}."
            )
        head_module = adapter_modules["head"]
        logger.info(
            "[HeadTrainer] head module: %s (%d params)",
            type(head_module).__name__,
            sum(p.numel() for p in head_module.parameters()),
        )
        self._freeze_all_except(head_module)

        # 2) Готовим datasets
        ds_train = WalkForwardDataset(
            candles=candles,
            train_window_size=config.train_window_size,
            horizon=config.horizon,
            step=config.step,
            val_split=config.val_split,
            mode="train",
        )
        ds_val = WalkForwardDataset(
            candles=candles,
            train_window_size=config.train_window_size,
            horizon=config.horizon,
            step=config.step,
            val_split=config.val_split,
            mode="val",
        )

        loader_train = DataLoader(
            ds_train, batch_size=config.batch_size,
            shuffle=True, num_workers=config.num_workers,
        )
        loader_val = DataLoader(
            ds_val, batch_size=config.batch_size,
            shuffle=False, num_workers=config.num_workers,
        ) if len(ds_val) > 0 else None

        # 3) Веса горизонта (те же что в backtest)
        weights_list = compute_weights(
            horizon=config.horizon,
            scheme=config.evaluation_weights,
            first_to_last_ratio=config.weight_first_to_last_ratio,
        )
        weights_tensor = torch.tensor(weights_list, dtype=torch.float32, device=self.device)

        # 4) Optimizer — только параметры head (где requires_grad=True)
        trainable_params = [p for p in self.torch_model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError(
                "После замораживания не осталось обучаемых параметров. "
                "Проверь _locate_head_module()."
            )
        optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

        total_batches_per_epoch = len(loader_train)
        total_batches = total_batches_per_epoch * config.num_epochs
        logger.info(
            "[HeadTrainer] Starting: %d epochs × %d train batches = %d total | "
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
                # Forward — нужно вернуть квантильные прогнозы
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
                        "[HeadTrainer] epoch %d/%d batch %d/%d (overall %d/%d %.0f%%) "
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
                "[HeadTrainer] Epoch %d/%d done: train_loss=%.4f val_loss=%.4f elapsed=%s",
                epoch + 1, config.num_epochs, avg_train, avg_val, _fmt_elapsed(elapsed),
            )
            _emit({
                "phase": "epoch_end",
                "epoch": epoch + 1, "total_epochs": config.num_epochs,
                "train_loss": float(avg_train), "val_loss": float(avg_val),
                "elapsed_s": elapsed,
            })

        # 6) Готовим state_dict head'ы для сохранения
        head_state = {k: v.detach().cpu() for k, v in head_module.state_dict().items()}

        return HeadTrainingResult(
            head_state_dict=head_state,
            final_train_loss=train_history[-1] if train_history else 0.0,
            final_val_loss=val_history[-1] if val_history else 0.0,
            epochs_completed=config.num_epochs,
            train_history=train_history,
            val_history=val_history,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _freeze_all_except(self, head_module) -> None:
        """Замораживает все параметры модели кроме параметров head_module."""
        head_params = set(id(p) for p in head_module.parameters())
        frozen = 0
        trainable = 0
        for p in self.torch_model.parameters():
            if id(p) in head_params:
                p.requires_grad = True
                trainable += p.numel()
            else:
                p.requires_grad = False
                frozen += p.numel()
        logger.info(
            "[HeadTrainer] Заморожено %d параметров, обучаемо %d (%.2f%%)",
            frozen, trainable, 100 * trainable / (frozen + trainable),
        )

    def _forward_quantiles(self, past_values, horizon: int):
        """
        Прогон past_values через модель, возврат квантильных прогнозов в shape
        совместимом с weighted_pinball_loss: (B, H, Q, C).

        Внутреннее устройство PatchTSTFMForPrediction:
          - модель возвращает PatchTSTFMPredictionOutput с двумя полями:
              * prediction_outputs: (B, H, N)         — point forecast (mean по Q)
              * quantile_outputs:   (B, Q, H, N)      — квантильные прогнозы
          - prediction_length управляет длиной прогноза (если не передать —
            возьмётся model.config.prediction_length, который может быть != horizon)

        Для обучения head'ы нам нужен именно quantile_outputs:
          1) shape (B, Q, H, N) → permute в (B, H, Q, N) — ожидаемый формат loss'а
          2) horizon явно передаём в prediction_length, иначе получим неверную длину

        past_values: (B, T, C) где C = OHLC (4 канала)
        return:      (B, H, Q, C) тензор с requires_grad
        """
        # quantile_levels обязательно передаём — модель имеет внутренний набор из ~99
        # квантилей, который в forward фильтруется до запрошенных. Без этого получим
        # 99-мерный Q вместо 5, и weighted_pinball_loss упадёт по broadcast'у.
        outputs = self.torch_model(
            past_values=past_values,
            prediction_length=horizon,
            quantile_levels=list(self.quantile_levels),
        )

        # Основной путь: PatchTSTFMPredictionOutput.quantile_outputs
        quantile_preds = getattr(outputs, "quantile_outputs", None)
        if quantile_preds is not None:
            # (B, Q, H, N) → (B, H, Q, N), C совпадает с N для нашего univariate-по-каналам случая
            return quantile_preds.permute(0, 2, 1, 3)

        # Fallback: пробуем альтернативные имена (для совместимости с другими версиями)
        for attr in ("quantile_predictions", "predictions", "logits"):
            preds = getattr(outputs, attr, None)
            if preds is not None:
                logger.warning(
                    "[HeadTrainer] quantile_outputs не найден, использую outputs.%s "
                    "(shape=%s) — проверь совместимость с loss'ом",
                    attr, tuple(preds.shape),
                )
                return preds

        # Совсем экзотика — dict/tuple
        if isinstance(outputs, dict):
            for key in ("quantile_outputs", "quantile_predictions", "predictions"):
                if key in outputs:
                    return outputs[key]
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]

        raise RuntimeError(
            f"Не удалось извлечь квантильные прогнозы из output модели. "
            f"Тип output: {type(outputs).__name__}, доступные поля: "
            f"{[a for a in dir(outputs) if not a.startswith('_')]}"
        )
