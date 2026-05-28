"""
Runtime для Salesforce Moirai-2.0 (uni2ts).

ВАЖНО про версии namespace в uni2ts:
  - uni2ts.model.moirai      → Moirai 1.0 (encoder-only, masked, mixture-distr) — НЕ ИСПОЛЬЗУЕМ
  - uni2ts.model.moirai_moe  → Moirai-MoE (mixture-of-experts вариант)
  - uni2ts.model.moirai2     → Moirai 2.0 (decoder-only, quantile loss)  ← НАШ

Moirai 2.0 — это decoder-only transformer foundation model для time series:
  - Patch-based tokenization (single patch size, в отличие от multi-patch в 1.0)
  - Quantile forecasting НАПРЯМУЮ (quantile loss), без mixture-of-distributions
    и без sampling — модель сразу выдаёт значения по квантилям.
  - Multi-token prediction
  - Multivariate через target_dim параметр в Moirai2Forecast wrapper'е

Размеры и model_id:
  small  → Salesforce/moirai-2.0-R-small   (официально подтверждён на HF)
  base   → Salesforce/moirai-2.0-R-base    (НЕ ПОДТВЕРЖДЁН — может не существовать)
  large  → Salesforce/moirai-2.0-R-large   (НЕ ПОДТВЕРЖДЁН — может не существовать)

  В отличие от Moirai 1.0, для 2.0 Salesforce публично выложил только small
  ("less is more"). Перед использованием base/large — проверь, что репозиторий
  реально существует на huggingface.co/Salesforce.

Архитектура внутри `Moirai2Module` (ПОДТВЕРЖДЕНО против moirai-2.0-R-small):
  named_children() == ['scaler', 'in_proj', 'encoder', 'out_proj']
    scaler   — нормализация входа (instance norm), без обучаемого адаптера
    in_proj  — входная проекция (patch embedding)          ← наш "input"
    encoder  — стек decoder-блоков (цель для LoRA attention)
    out_proj — выходная проекция в квантили (quantile loss) ← наш "head"
  В отличие от 1.0 здесь нет mask_encoding и param_proj — 2.0 проще и выдаёт
  квантили напрямую через out_proj.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


DEFAULT_PREDICTION_LENGTH = 64
DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_PATCH_SIZE = "auto"
DEFAULT_NUM_SAMPLES = 100
DEFAULT_QUANTILE_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]


# Размеры, чьё существование на HF подтверждено публично.
_CONFIRMED_SIZES = {"small"}

# Полный маппинг (base/large — спекулятивны, см. докстринг модуля).
_SIZE_TO_MODEL_ID = {
    "small": "Salesforce/moirai-2.0-R-small",
    "base":  "Salesforce/moirai-2.0-R-base",
    "large": "Salesforce/moirai-2.0-R-large",
}


def _resolve_model_id(size: str) -> str:
    """
    Маппит логический размер в полный HF model_id.

    Для неподтверждённых размеров (base/large) бросаем явную ошибку вместо
    того чтобы отдать model_id который даст 404 при snapshot_download —
    так пользователь сразу понимает причину, а не ловит сетевую ошибку.
    """
    size = size.strip().lower()
    if size not in _SIZE_TO_MODEL_ID:
        raise ValueError(
            f"Неизвестный размер Moirai-2 '{size}'. "
            f"Поддерживаются: {list(_SIZE_TO_MODEL_ID)}."
        )
    if size not in _CONFIRMED_SIZES:
        raise ValueError(
            f"Размер Moirai-2 '{size}' не подтверждён на HuggingFace. "
            f"Публично доступен только 'small' (Salesforce/moirai-2.0-R-small). "
            f"Если репозиторий '{_SIZE_TO_MODEL_ID[size]}' реально существует — "
            f"добавь '{size}' в _CONFIRMED_SIZES в moirai_runtime.py."
        )
    return _SIZE_TO_MODEL_ID[size]


class MoiraiRuntime:
    """
    Lazy-loaded Moirai-2.0 runtime.

    Структура:
      self.module — Moirai2Module (raw torch model)
      self.forecaster — Moirai2Forecast (high-level predictor wrapper)
      self.device — 'cuda' / 'cpu' / 'mps'

    Используем module для adapter training (нужен прямой доступ к input/output
    проекциям), forecaster — для inference (он умеет квантили напрямую).
    """

    def __init__(
            self,
            size: str = "small",
            context_length: int = DEFAULT_CONTEXT_LENGTH,
            prediction_length: int = DEFAULT_PREDICTION_LENGTH,
            patch_size: str | int = DEFAULT_PATCH_SIZE,
            num_samples: int = DEFAULT_NUM_SAMPLES,
    ) -> None:
        self.size = size
        self.model_id = _resolve_model_id(size)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.quantile_levels = list(DEFAULT_QUANTILE_LEVELS)
        self.num_quantile = len(self.quantile_levels)

        self.is_loaded = False
        self.device = "cpu"
        self.load_error: str | None = None
        self.module = None       # Moirai2Module — для adapter training
        self.forecaster = None   # Moirai2Forecast — для inference
        # Backward-совместимый alias: ArtifactLoader._unwrap_to_torch ищет
        # _runtime.model — пусть это будет наш raw module.
        self.model = None

    def ensure_loaded(self) -> None:
        if self.is_loaded and self.module is not None:
            return
        try:
            import torch
            try:
                from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
            except ImportError as ex:
                raise RuntimeError(
                    "uni2ts (Moirai-2.0) не установлен или версия не содержит "
                    "moirai2 namespace. Установите/обновите: `pip install -U uni2ts`. "
                    "Подробнее: https://github.com/SalesforceAIResearch/uni2ts"
                ) from ex

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Шаг 1: грузим raw transformer module
            self.module = Moirai2Module.from_pretrained(self.model_id)
            self.module = self.module.to(self.device)
            self.module.eval()

            # Шаг 2: оборачиваем в predictor для inference.
            # Сигнатура Moirai2Forecast (подтверждена inspect):
            #   (prediction_length, target_dim, feat_dynamic_real_dim,
            #    past_feat_dynamic_real_dim, context_length,
            #    module_kwargs=None, module=None)
            # patch_size в 2.0 НЕ принимается (single patch фиксирован внутри).
            self.forecaster = Moirai2Forecast(
                prediction_length=self.prediction_length,
                target_dim=1,                    # univariate per channel
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
                context_length=self.context_length,
                module=self.module,
            ).to(self.device)
            self.forecaster.eval()

            # Alias для ArtifactLoader._unwrap_to_torch
            self.model = self.module

            self.is_loaded = True
            self.load_error = None
            logger.info(
                "[Moirai2] loaded model_id=%s on device=%s (ctx=%d pred=%d patch=%s)",
                self.model_id, self.device, self.context_length,
                self.prediction_length, self.patch_size,
            )

        except Exception as ex:
            self.is_loaded = False
            self.module = None
            self.forecaster = None
            self.model = None
            self.load_error = str(ex)
            raise RuntimeError(
                f"Не удалось загрузить Moirai-2.0 '{self.model_id}': {ex}"
            ) from ex

    # ------------------------------------------------------------------
    # Adapter modules — точки расширения для head/input training
    # ------------------------------------------------------------------

    def get_adapter_modules(self) -> dict:
        """
        Возвращает {"head": out_proj, "input": in_proj} для Moirai2Module.

        Имена ПОДТВЕРЖДЕНЫ против реального Moirai2Module
        (Salesforce/moirai-2.0-R-small): named_children() даёт
        ['scaler', 'in_proj', 'encoder', 'out_proj'].

        Trainers/ArtifactLoader обращаются исключительно через эту функцию.
        """
        if not self.is_loaded:
            self.ensure_loaded()

        # Имена подтверждены против реального Moirai2Module:
        #   named_children() == ['scaler', 'in_proj', 'encoder', 'out_proj']
        #   in_proj  — входная проекция (patch embedding)        → "input"
        #   out_proj — выходная проекция в квантили (quantile loss) → "head"
        # scaler/encoder для adapter-обучения не используются.
        attr_map = {"input": "in_proj", "head": "out_proj"}

        modules = {}
        for adapter_name, attr in attr_map.items():
            module = getattr(self.module, attr, None)
            if module is None:
                kids = [n for n, _ in self.module.named_children()]
                raise RuntimeError(
                    f"[Moirai2] adapter-модуль '{adapter_name}' ожидался под "
                    f"атрибутом '{attr}', но его нет. Реальные top-level модули: {kids}. "
                    f"Структура Moirai2Module изменилась — поправь attr_map в "
                    f"moirai_runtime.get_adapter_modules()."
                )
            modules[adapter_name] = module
        return modules

    def get_lora_target_modules(self) -> list[str]:
        """
        Имена attention-проекций внутри Moirai2Module.encoder для PEFT.

        ПОДТВЕРЖДЕНО против moirai-2.0-R-small. Все Linear-слои модели:
          ['fc1','fc2','fc_gate','hidden_layer','k_proj','out_proj',
           'output_layer','q_proj','residual_layer','v_proj']

        Берём ТОЛЬКО q/k/v_proj. out_proj НАМЕРЕННО исключён:
          - на top-level out_proj — это головная проекция в квантили (наш "head"),
            которую обучает отдельный head-trainer;
          - PEFT матчит target по суффиксу имени по ВСЕЙ модели, поэтому "out_proj"
            зацепил бы и attention-выходы, и головную проекцию одновременно —
            коллизия с head-обучением. q/k/v_proj такой коллизии не имеют.
        fc1/fc2/fc_gate/hidden_layer/residual_layer — FFN/MLP части, в LoRA
        стандартно не включаются.
        """
        return ["q_proj", "k_proj", "v_proj"]

    # ------------------------------------------------------------------
    # Hook для пересборки после применения LoRA
    # ------------------------------------------------------------------

    def rebuild_after_lora(self, peft_model) -> None:
        """
        Пересобирает forecaster после того как ArtifactLoader обернул module в
        PEFT. В отличие от PatchTST (который использует tsfm_public pipeline),
        Moirai2 использует свой Moirai2Forecast wrapper.

        ArtifactLoader должен звать ЭТОТ метод вместо хардкода TSFM pipeline,
        чтобы LoRA-адаптер реально участвовал в инференсе.
        """
        from uni2ts.model.moirai2 import Moirai2Forecast

        self.model = peft_model
        self.module = peft_model
        self.forecaster = Moirai2Forecast(
            prediction_length=self.prediction_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=self.context_length,
            module=peft_model,
        ).to(self.device)
        self.forecaster.eval()
        logger.info("[Moirai2] forecaster пересобран после применения LoRA.")