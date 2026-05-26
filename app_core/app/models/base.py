from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Dataclasses результатов прогноза
# ---------------------------------------------------------------------------

@dataclass
class QuantileForecast:
    q10: list[float]
    q25: list[float]
    q50: list[float]
    q75: list[float]
    q90: list[float]

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "q10": self.q10,
            "q25": self.q25,
            "q50": self.q50,
            "q75": self.q75,
            "q90": self.q90,
        }

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> "QuantileForecast":
        return cls(
            q10=data["q10"],
            q25=data["q25"],
            q50=data["q50"],
            q75=data["q75"],
            q90=data["q90"],
        )


@dataclass
class OHLCQuantileForecast:
    """
    Квантильный прогноз для полной OHLC свечи на горизонт N точек.

    Каждый канал содержит QuantileForecast — распределение для своего канала.
    volume опционален: None в zero-shot режиме, заполняется после дообучения
    на конкретном инструменте где объём является значимым каналом.
    """
    open:   QuantileForecast
    high:   QuantileForecast
    low:    QuantileForecast
    close:  QuantileForecast
    volume: QuantileForecast | None = field(default=None)

    def to_dict(self) -> dict[str, dict[str, list[float]] | None]:
        return {
            "open":   self.open.to_dict(),
            "high":   self.high.to_dict(),
            "low":    self.low.to_dict(),
            "close":  self.close.to_dict(),
            "volume": self.volume.to_dict() if self.volume is not None else None,
        }

    def median_candles(self) -> list[dict[str, float]]:
        """
        Список свечей из медианных значений (q50) по каждому каналу.
        Используется как основной точечный прогноз для отображения.
        """
        horizon = len(self.close.q50)
        candles = []
        for i in range(horizon):
            candle = {
                "open":  self.open.q50[i],
                "high":  self.high.q50[i],
                "low":   self.low.q50[i],
                "close": self.close.q50[i],
            }
            if self.volume is not None:
                candle["volume"] = self.volume.q50[i]
            candles.append(candle)
        return candles


# ---------------------------------------------------------------------------
# Базовый класс модели
# ---------------------------------------------------------------------------

class ForecastModel(ABC):
    """
    Базовый интерфейс для всех прогнозных моделей.

    Соглашения:
    - Все методы принимают candles: list[dict[str, float]] — список свечей
      с ключами open/high/low/close/volume.
    - horizon: int — количество точек прогноза.
    - context: dict | None — произвольный контекст (num_samples, feature_plugins и т.д.).
    - Модели которые не поддерживают конкретный метод поднимают NotImplementedError
      с понятным сообщением.
    """

    @abstractmethod
    def predict_line_exact(
            self,
            candles: list[dict[str, float]],
            horizon: int,
            context: dict | None = None
    ) -> list[float]:
        """
        Точечный прогноз close как числового ряда.
        Возвращает список длиной horizon.
        """
        pass

    @abstractmethod
    def predict_line_quantiles(
            self,
            candles: list[dict[str, float]],
            horizon: int,
            context: dict | None = None
    ) -> QuantileForecast:
        """
        Квантильный прогноз close.
        Возвращает QuantileForecast с q10/q25/q50/q75/q90, каждый длиной horizon.
        q50 соответствует медианному (центральному) прогнозу.
        """
        pass

    @abstractmethod
    def predict_ohlc_exact(
            self,
            candles: list[dict[str, float]],
            horizon: int,
            context: dict | None = None
    ) -> list[dict[str, float]]:
        """
        Точечный прогноз полных OHLC свечей.
        Возвращает список словарей с ключами open/high/low/close длиной horizon.
        Гарантирует: high >= max(open, close), low <= min(open, close).
        """
        pass

    @abstractmethod
    def predict_ohlc_quantiles(
            self,
            candles: list[dict[str, float]],
            horizon: int,
            context: dict | None = None
    ) -> OHLCQuantileForecast:
        """
        Квантильный прогноз полных OHLC свечей — основной метод лаборатории.
        Возвращает OHLCQuantileForecast с квантилями по каждому каналу.
        Гарантирует физическую корректность свечей на уровне медианных значений.
        volume заполняется только если модель дообучена на объёме.
        """
        pass

    @abstractmethod
    def get_info(self) -> dict:
        pass

    def fit_adapter(self, data, config: dict | None = None) -> None:
        return None

    # ------------------------------------------------------------------
    # Точки расширения для дообучения и применения артефактов
    #
    # Каждая foundation-модель знает свою внутреннюю структуру лучше, чем
    # universal-trainer'ы. Эти методы — единственное место, где «привязка»
    # к конкретной архитектуре. Trainers и ArtifactLoader используют их
    # вместо собственного _locate_*_module.
    #
    # Дефолт — NotImplementedError. Модели, поддерживающие дообучение,
    # переопределяют. Модели чисто zero-shot (например, для quick prototyping)
    # могут не реализовывать — тогда соответствующие endpoint'ы /training/*
    # выдадут понятную ошибку.
    # ------------------------------------------------------------------

    def get_adapter_modules(self) -> dict:
        """
        Возвращает словарь {"head": nn.Module, "input": nn.Module, ...} —
        модули, которые можно тренировать как отдельные адаптеры
        (head training / input projection tuning).

        Имена ключей:
          'head'  — выходная проекция (для PatchTSTFM = backbone.out_layer,
                    для Moirai = param_proj и т.п.)
          'input' — входная проекция / patch embedding
                    (PatchTSTFM = backbone.in_layer, Moirai = in_proj)
          Любые дополнительные имена — модель-специфичные.

        Гарантия: возвращаемые модули — реальные nn.Module внутри загруженной
        модели (не копии), их state_dict загружается/сохраняется напрямую.
        """
        raise NotImplementedError(
            f"{type(self).__name__} не поддерживает adapter training "
            f"(head/input fine-tuning). Переопредели get_adapter_modules() "
            f"чтобы включить эту функциональность."
        )

    def get_lora_target_modules(self) -> list[str]:
        """
        Возвращает список имён модулей, в которые PEFT инжектит LoRA-матрицы.

        Стандартный набор для transformer-based моделей: q_proj/k_proj/v_proj/out_proj.
        Для модели с нестандартным naming (например, MoE с expert routing)
        список будет другим.

        Используется как default для LoraTrainingConfig.lora_target_modules;
        пользователь может переопределить в запросе если хочет точечно настроить.
        """
        raise NotImplementedError(
            f"{type(self).__name__} не задаёт LoRA target modules. "
            f"Переопредели get_lora_target_modules() чтобы включить LoRA training."
        )

    # ------------------------------------------------------------------
    # Опциональные батчевые методы (для ускорения walk-forward бэктеста)
    #
    # Дефолтная реализация — sequential loop через одноразовые методы.
    # Модели, способные к настоящему батчингу (например PatchTST FM с
    # multi-series pipeline), переопределяют их для прогона нескольких
    # окон ОДНИМ forward pass'ом на GPU.
    #
    # Все candles_list[i] должны иметь ОДИНАКОВУЮ длину (один контекст) —
    # иначе нельзя сложить в один тензор. Caller отвечает за это.
    # ------------------------------------------------------------------

    def predict_ohlc_quantiles_batch(
            self,
            candles_list: list[list[dict[str, float]]],
            horizon: int,
            context: dict | None = None,
    ) -> list[OHLCQuantileForecast]:
        """
        Батчевый OHLC квантильный прогноз. По умолчанию — sequential.
        Модели, поддерживающие настоящий батчинг, должны переопределить.
        """
        return [self.predict_ohlc_quantiles(c, horizon, context) for c in candles_list]

    def predict_line_exact_batch(
            self,
            candles_list: list[list[dict[str, float]]],
            horizon: int,
            context: dict | None = None,
    ) -> list[list[float]]:
        """
        Батчевый точечный прогноз close. По умолчанию — sequential.
        """
        return [self.predict_line_exact(c, horizon, context) for c in candles_list]