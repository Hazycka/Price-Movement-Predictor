"""
LoraTrainer — обучение LoRA-адаптеров через HuggingFace PEFT.

LoRA (Low-Rank Adaptation) — это пара низкоранговых матриц A (d×r) и B (r×d),
добавляемых параллельно к существующим linear-слоям модели:
    new_weight(x) = original_weight(x) + B @ A @ x * (alpha / r)

Параметры A, B обучаются (~r × d × 2 параметров на слой), оригинальные веса
ЗАМОРОЖЕНЫ. Это в десятки раз меньше параметров чем full fine-tuning, но
позволяет адаптировать внутренние представления модели — не только output.

Реализация через peft (pip install peft):
  1. Конфиг: LoraConfig(r=8, alpha=16, target_modules=["q_proj", ...])
  2. Оборачиваем base model: peft_model = get_peft_model(model, config)
  3. Тренируем как обычно — PEFT сам управляет requires_grad
  4. Сохраняем: peft_model.save_pretrained(adapter_dir)

При train_head_too=True — head также unfrozen для совместного обучения.
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
from .head_trainer import _fmt_elapsed  # переиспользуем форматирование
from .loss import weighted_pinball_loss
from ..backtest.weights import compute_weights

logger = logging.getLogger(__name__)


ProgressCallback = Callable[[dict], None]


@dataclass
class LoraTrainingConfig:
    """Гиперпараметры обучения LoRA."""
    train_window_size: int
    horizon: int
    step: int = 64
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 5
    val_split: float = 0.15
    evaluation_weights: str = "exponential"
    weight_first_to_last_ratio: float = 16.0
    num_workers: int = 0
    log_every_n_batches: int = 5
    # LoRA-specific
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = None
    train_head_too: bool = False


@dataclass
class LoraTrainingResult:
    """Результат обучения LoRA."""
    peft_model: object   # peft.PeftModel — сохраняется через .save_pretrained(dir)
    head_state_dict: dict | None  # если train_head_too=True
    final_train_loss: float
    final_val_loss: float
    epochs_completed: int
    train_history: list[float]
    val_history: list[float]
    trainable_params: int
    total_params: int


class LoraTrainer:
    """
    Обучает LoRA-адаптеры через PEFT поверх PatchTST FM.

    Использование:
        trainer = LoraTrainer(base_model)
        result = trainer.train(candles, LoraTrainingConfig(train_window_size=8192, ...))
        ArtifactSaver.save_lora(result.peft_model, metrics, save_ctx)
    """

    def __init__(self, base_model) -> None:
        if torch is None:
            raise RuntimeError("PyTorch не установлен.")
        try:
            import peft  # noqa: F401
        except ImportError as ex:
            raise RuntimeError(
                "peft не установлен. Установите его: `pip install peft`."
            ) from ex
        self.base_model = base_model
        base_model._runtime.ensure_loaded()
        self.torch_model = base_model._runtime.model
        self.quantile_levels = base_model._runtime.quantile_levels
        self.device = base_model._runtime.device

    def train(
            self,
            candles: list[dict[str, float]],
            config: LoraTrainingConfig,
            progress_callback: ProgressCallback | None = None,
    ) -> LoraTrainingResult:
        """
        Обучает LoRA-адаптеры. progress_callback (если задан) получает per-batch
        и per-epoch updates — caller использует для записи в JobStore.

        Структура payload в progress_callback идентична HeadTrainer.
        """
        def _emit(payload: dict) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(payload)
                except Exception as ex:  # noqa
                    logger.warning("[LoraTrainer] progress_callback failed: %s", ex)

        from peft import LoraConfig, get_peft_model, TaskType

        target_modules = config.lora_target_modules or ["q_proj", "k_proj", "v_proj", "out_proj"]

        # 1) Создаём PEFT-конфиг и оборачиваем модель.
        # task_type='FEATURE_EXTRACTION' — generic, не language modeling.
        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        peft_model = get_peft_model(self.torch_model, lora_cfg)

        # 2) Опционально размораживаем head
        if config.train_head_too:
            head_module = self._locate_head_module(peft_model)
            for p in head_module.parameters():
                p.requires_grad = True
            logger.info("[LoraTrainer] train_head_too=True — head разморожен")

        # 3) Datasets
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

        # 4) Веса горизонта
        weights_list = compute_weights(
            horizon=config.horizon,
            scheme=config.evaluation_weights,
            first_to_last_ratio=config.weight_first_to_last_ratio,
        )
        weights_tensor = torch.tensor(weights_list, dtype=torch.float32, device=self.device)

        # 5) Optimizer — собираем все trainable параметры (LoRA + head если есть)
        trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
        total = sum(p.numel() for p in peft_model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        if not trainable_params:
            raise RuntimeError("После применения LoRA не осталось обучаемых параметров.")
        optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

        total_batches_per_epoch = len(loader_train)
        total_batches = total_batches_per_epoch * config.num_epochs
        logger.info(
            "[LoraTrainer] Starting: %d epochs × %d train batches = %d total | "
            "%d val batches | %d trainable / %d total (%.4f%%) | device=%s | target=%s",
            config.num_epochs, total_batches_per_epoch, total_batches,
            len(loader_val) if loader_val else 0,
            trainable_count, total, 100 * trainable_count / total,
            self.device, target_modules,
        )
        _emit({
            "phase": "start",
            "total_epochs": config.num_epochs,
            "total_batches": total_batches,
            "trainable_params": trainable_count,
            "total_params": total,
            "lora_target_modules": target_modules,
            "device": str(self.device),
        })

        # 6) Training loop
        train_history: list[float] = []
        val_history: list[float] = []
        t0 = time.perf_counter()
        global_batch = 0
        for epoch in range(config.num_epochs):
            peft_model.train()
            epoch_train = []
            for batch_idx, (past, future) in enumerate(loader_train):
                past = past.to(self.device)
                future = future.to(self.device)

                optimizer.zero_grad()
                pred_q = self._forward_quantiles(peft_model, past)
                loss = weighted_pinball_loss(
                    pred_q, future, weights_tensor, self.quantile_levels,
                )
                loss.backward()
                optimizer.step()
                epoch_train.append(loss.item())

                global_batch += 1
                if (batch_idx + 1) % config.log_every_n_batches == 0 or batch_idx == total_batches_per_epoch - 1:
                    elapsed = time.perf_counter() - t0
                    eta = (elapsed / global_batch) * (total_batches - global_batch) if global_batch else 0.0
                    running_avg = sum(epoch_train) / len(epoch_train)
                    logger.info(
                        "[LoraTrainer] epoch %d/%d batch %d/%d (overall %d/%d %.0f%%) "
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

            avg_train = sum(epoch_train) / len(epoch_train)
            train_history.append(avg_train)

            avg_val = 0.0
            if loader_val is not None:
                peft_model.eval()
                with torch.no_grad():
                    vs = []
                    for past, future in loader_val:
                        past = past.to(self.device)
                        future = future.to(self.device)
                        pred_q = self._forward_quantiles(peft_model, past)
                        l = weighted_pinball_loss(
                            pred_q, future, weights_tensor, self.quantile_levels,
                        )
                        vs.append(l.item())
                    avg_val = sum(vs) / len(vs) if vs else 0.0
                val_history.append(avg_val)

            elapsed = time.perf_counter() - t0
            logger.info(
                "[LoraTrainer] Epoch %d/%d done: train_loss=%.4f val_loss=%.4f elapsed=%s",
                epoch + 1, config.num_epochs, avg_train, avg_val, _fmt_elapsed(elapsed),
            )
            _emit({
                "phase": "epoch_end",
                "epoch": epoch + 1, "total_epochs": config.num_epochs,
                "train_loss": float(avg_train), "val_loss": float(avg_val),
                "elapsed_s": elapsed,
            })

        # 7) Снимок head если обучали
        head_state = None
        if config.train_head_too:
            head_module = self._locate_head_module(peft_model)
            head_state = {k: v.detach().cpu() for k, v in head_module.state_dict().items()}

        return LoraTrainingResult(
            peft_model=peft_model,
            head_state_dict=head_state,
            final_train_loss=train_history[-1] if train_history else 0.0,
            final_val_loss=val_history[-1] if val_history else 0.0,
            epochs_completed=config.num_epochs,
            train_history=train_history,
            val_history=val_history,
            trainable_params=trainable_count,
            total_params=total,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _locate_head_module(model):
        # PEFT оборачивает модель — head доступна через model.base_model.model.head
        # или прямой атрибут (зависит от версии PEFT)
        candidates_paths = [
            ("base_model", "model", "head"),
            ("base_model", "head"),
            ("head",),
        ]
        for path in candidates_paths:
            obj = model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "parameters"):
                return obj
        # Дополнительный поиск через base_model
        base = getattr(model, "base_model", None)
        if base is not None:
            for name in ("head", "prediction_head", "output_head"):
                module = getattr(base, name, None)
                if module is None and hasattr(base, "model"):
                    module = getattr(base.model, name, None)
                if module is not None:
                    return module
        raise RuntimeError(
            "Не удалось найти head у PEFT-обёрнутой модели. "
            "Возможно нужна корректировка путей под текущую версию peft."
        )

    def _forward_quantiles(self, peft_model, past_values):
        """Прогон через peft-обёрнутую модель, возврат квантильных прогнозов."""
        outputs = peft_model(past_values=past_values)
        for attr in ("prediction_outputs", "quantile_predictions", "predictions", "logits"):
            preds = getattr(outputs, attr, None)
            if preds is not None:
                return preds
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        if isinstance(outputs, dict):
            for key in ("prediction_outputs", "quantile_predictions", "predictions"):
                if key in outputs:
                    return outputs[key]
        raise RuntimeError(
            f"Не извлечь квантильные прогнозы из output PEFT-модели. type={type(outputs).__name__}"
        )
