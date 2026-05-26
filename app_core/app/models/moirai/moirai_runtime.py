"""
Runtime для Salesforce Moirai-2 (uni2ts).

Moirai — это encoder-only transformer foundation model для time series:
  - Patch-based tokenization (как PatchTST)
  - Mixture-of-distributions output (Student-t + Normal + ...) → квантили через samples
  - Поддержка multivariate через target_dim параметр в MoiraiForecast wrapper'е

Размеры и model_id:
  small  → Salesforce/moirai-2.0-R-small   ~17M params
  base   → Salesforce/moirai-2.0-R-base    ~88M params
  large  → Salesforce/moirai-2.0-R-large   ~311M params

Архитектура внутри `MoiraiModule` (нужно знать для adapter training):
  - in_proj           — patch embedding (входная проекция) ← наш "input"
  - mask_encoding     — special token для prediction-маски
  - encoder           — TransformerEncoder (ModuleList Layer'ов)
  - param_proj        — выходная проекция в параметры distribution ← наш "head"
  - distr_output      — distribution mixture (не модуль с весами, а конфиг)

Имена атрибутов выше — точно из публичного uni2ts API на момент написания.
Если в новых версиях изменятся — поправить _adapter_paths.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


DEFAULT_PREDICTION_LENGTH = 64
DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_PATCH_SIZE = "auto"
DEFAULT_NUM_SAMPLES = 100
DEFAULT_QUANTILE_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]


def _resolve_model_id(size: str) -> str:
    """Маппит логический размер в полный HF model_id."""
    size = size.strip().lower()
    mapping = {
        "small": "Salesforce/moirai-2.0-R-small",
        "base":  "Salesforce/moirai-2.0-R-base",
        "large": "Salesforce/moirai-2.0-R-large",
    }
    if size not in mapping:
        raise ValueError(
            f"Неизвестный размер Moirai-2 '{size}'. "
            f"Поддерживаются: {list(mapping)}."
        )
    return mapping[size]


class MoiraiRuntime:
    """
    Lazy-loaded Moirai-2 runtime.

    Структура:
      self.module — MoiraiModule (raw torch model)
      self.forecaster — MoiraiForecast (high-level predictor wrapper)
      self.device — 'cuda' / 'cpu' / 'mps'

    Используем module для adapter training (нужен прямой доступ к in_proj/param_proj),
    forecaster — для inference (он умеет квантили через sampling).
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
        self.module = None       # MoiraiModule — для adapter training
        self.forecaster = None   # MoiraiForecast — для inference
        # Backward-совместимый alias: ArtifactLoader._unwrap_to_torch ищет
        # _runtime.model — пусть это будет наш raw module.
        self.model = None

    def ensure_loaded(self) -> None:
        if self.is_loaded and self.module is not None:
            return
        try:
            import torch
            try:
                from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
            except ImportError as ex:
                raise RuntimeError(
                    "uni2ts (Moirai-2) не установлен. Установите: "
                    "`pip install uni2ts`. Подробнее: https://github.com/SalesforceAIResearch/uni2ts"
                ) from ex

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Шаг 1: грузим raw transformer module
            self.module = MoiraiModule.from_pretrained(self.model_id)
            self.module = self.module.to(self.device)
            self.module.eval()

            # Шаг 2: оборачиваем в predictor для inference
            self.forecaster = MoiraiForecast(
                module=self.module,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                patch_size=self.patch_size,
                num_samples=self.num_samples,
                target_dim=1,                    # univariate per channel
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            ).to(self.device)
            self.forecaster.eval()

            # Alias для ArtifactLoader._unwrap_to_torch
            self.model = self.module

            self.is_loaded = True
            self.load_error = None
            logger.info(
                "[Moirai] loaded model_id=%s on device=%s (ctx=%d pred=%d patch=%s samples=%d)",
                self.model_id, self.device, self.context_length, self.prediction_length,
                self.patch_size, self.num_samples,
            )

        except Exception as ex:
            self.is_loaded = False
            self.module = None
            self.forecaster = None
            self.model = None
            self.load_error = str(ex)
            raise RuntimeError(
                f"Не удалось загрузить Moirai-2 '{self.model_id}': {ex}"
            ) from ex

    # ------------------------------------------------------------------
    # Adapter modules — точки расширения для head/input training
    # ------------------------------------------------------------------

    def get_adapter_modules(self) -> dict:
        """
        Возвращает {"head": param_proj, "input": in_proj} для MoiraiModule.

        Внимание: если в новых версиях uni2ts эти имена изменятся —
        здесь же будет легко найти и поправить. Trainers/ArtifactLoader
        обращаются исключительно через эту функцию.
        """
        if not self.is_loaded:
            self.ensure_loaded()

        modules = {}
        for adapter_name, attr_name in (("head", "param_proj"), ("input", "in_proj")):
            module = getattr(self.module, attr_name, None)
            if module is None:
                # Альтернативные имена на случай вариаций
                alt = {"head": ("out_proj", "param_head"), "input": ("embed_in", "patch_embed")}
                for alt_name in alt.get(adapter_name, ()):
                    module = getattr(self.module, alt_name, None)
                    if module is not None:
                        break
            if module is None:
                kids = [n for n, _ in self.module.named_children()]
                raise RuntimeError(
                    f"[Moirai] Не нашёл adapter-модуль '{adapter_name}'. "
                    f"Доступные top-level модули MoiraiModule: {kids}. "
                    f"Возможно изменилась структура uni2ts — обнови moirai_runtime.get_adapter_modules()."
                )
            modules[adapter_name] = module
        return modules

    def get_lora_target_modules(self) -> list[str]:
        """
        Имена attention-проекций внутри MoiraiModule.encoder для PEFT.
        Стандартный transformer attention naming.
        """
        return ["q_proj", "k_proj", "v_proj", "out_proj"]
