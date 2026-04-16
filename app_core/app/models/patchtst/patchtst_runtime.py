import os


class PatchTSTRuntime:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.is_loaded = False
        self.device = "cpu"
        self.load_error: str | None = None
        self.model = None
        self.torch = None
        self.required_context_length: int | None = None

    def ensure_loaded(self) -> None:
        if self.is_loaded and self.model is not None:
            return
        try:
            import torch
            from transformers import PatchTSTForPrediction

            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.model = PatchTSTForPrediction.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

            ctx_len = getattr(self.model.config, "context_length", None)
            self.required_context_length = int(ctx_len) if ctx_len is not None else None

            self.is_loaded = True
            self.load_error = None
        except Exception as ex:
            self.is_loaded = False
            self.model = None
            self.load_error = str(ex)
            raise RuntimeError(f"Не удалось загрузить PatchTST модель '{self.model_id}': {ex}") from ex
