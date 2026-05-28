"""
Runtime для ibm-granite/granite-timeseries-patchtst-fm-r1.
"""

QUANTILE_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]
MAX_CONTEXT_LENGTH = 8192
DEFAULT_PREDICTION_LENGTH = 64


class PatchTSTRuntime:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.is_loaded = False
        self.device = "cpu"
        self.load_error: str | None = None
        self.model = None
        self.pipeline = None
        self.context_length: int = MAX_CONTEXT_LENGTH
        self.prediction_length: int = DEFAULT_PREDICTION_LENGTH
        self.num_quantile: int = len(QUANTILE_LEVELS)
        self.quantile_levels: list[float] = QUANTILE_LEVELS

    def ensure_loaded(self) -> None:
        if self.is_loaded and self.pipeline is not None:
            return
        try:
            import torch
            from tsfm_public import PatchTSTFMForPrediction, TimeSeriesForecastingPipeline

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.model = PatchTSTFMForPrediction.from_pretrained(self.model_id)
            self.model = self.model.to(self.device)
            self.model.eval()

            try:
                self.context_length = int(
                    getattr(self.model.config, "context_length", MAX_CONTEXT_LENGTH)
                )
            except Exception:
                self.context_length = MAX_CONTEXT_LENGTH

            try:
                self.prediction_length = int(
                    getattr(self.model.config, "prediction_length", DEFAULT_PREDICTION_LENGTH)
                )
            except Exception:
                self.prediction_length = DEFAULT_PREDICTION_LENGTH

            self.pipeline = TimeSeriesForecastingPipeline(
                model=self.model,
                device=self.device,
                explode_forecasts=True,
                quantile_levels=QUANTILE_LEVELS,
            )

            self.is_loaded = True
            self.load_error = None

        except ImportError as ex:
            self.is_loaded = False
            self.model = None
            self.pipeline = None
            self.load_error = str(ex)
            raise RuntimeError(
                "Не удалось импортировать tsfm_public. "
                "Установите зависимость: pip install granite-tsfm\n"
                f"Детали: {ex}"
            ) from ex

        except Exception as ex:
            self.is_loaded = False
            self.model = None
            self.pipeline = None
            self.load_error = str(ex)
            raise RuntimeError(
                f"Не удалось загрузить PatchTST FM модель '{self.model_id}': {ex}"
            ) from ex

    # ------------------------------------------------------------------
    # Hook для пересборки inference-стека после применения LoRA
    # ------------------------------------------------------------------

    def rebuild_after_lora(self, peft_model) -> None:
        """
        Пересобирает TSFM pipeline после того как ArtifactLoader обернул model
        в PEFT. Без этого pipeline продолжал бы держать ссылку на base-модель
        и LoRA-адаптер молча игнорировался бы на инференсе.

        Контракт (см. ArtifactLoader._apply_lora): любой runtime, к которому
        применяется LoRA, ОБЯЗАН реализовать этот метод. Ошибки НЕ глушатся —
        если pipeline не пересобрался, должно упасть, а не отдавать base-прогнозы.
        """
        from tsfm_public import TimeSeriesForecastingPipeline

        self.model = peft_model
        self.pipeline = TimeSeriesForecastingPipeline(
            model=peft_model,
            device=self.device,
            explode_forecasts=True,
            quantile_levels=self.quantile_levels,
        )