from ..schemas import (
    ForecastRequest, ForecastResponse,
    BacktestRequest, BacktestResponse,
    BacktestSweepRequest, BacktestSweepResponse,
)
from ..models.base import ForecastModel
from ..storage import get_uow_factory
from ..storage.ports import BacktestRunRecord

from .forecast import (
    InputResolver,
    ForecastContextBuilder,
    ForecastMetadataBuilder,
    ForecastOrchestrator,
)
from .backtest import BacktestRunner, BacktestSweepRunner
from .chart import ChartService


class ForecastService:
    def __init__(self, model: ForecastModel) -> None:
        self.model = model
        self._input_resolver = InputResolver()
        self._forecast_orchestrator = ForecastOrchestrator(
            model=model,
            context_builder=ForecastContextBuilder(),
            metadata_builder=ForecastMetadataBuilder()
        )
        self._backtest_runner = BacktestRunner(model=model)
        self._sweep_runner = BacktestSweepRunner(model=model)
        self._chart_service = ChartService()

    def run_forecast(self, request: ForecastRequest) -> ForecastResponse:
        source, dates, candles = self._input_resolver.resolve(request)
        return self._forecast_orchestrator.run(
            request=request,
            source=source,
            dates=dates,
            candles=candles
        )

    def run_backtest(self, request: BacktestRequest) -> BacktestResponse:
        source, dates, candles = self._input_resolver.resolve(request)
        response = self._backtest_runner.run(
            request=request,
            source=source,
            dates=dates,
            candles=candles
        )
        if request.persist:
            response.run_id = self._persist_run(request, response, source)
        return response

    def run_sweep(self, request: BacktestSweepRequest) -> BacktestSweepResponse:
        source, dates, candles = self._input_resolver.resolve(request)
        return self._sweep_runner.run(
            request=request,
            source=source,
            dates=dates,
            candles=candles,
        )

    def build_chart(self, request: ForecastRequest) -> str:
        result = self.run_forecast(request)
        return self._chart_service.build(
            result,
            max_chart_history_candles=request.max_chart_history_candles,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_run(
            self,
            request: BacktestRequest,
            response: BacktestResponse,
            source: str,
    ) -> int:
        """
        Сохраняет результат бэктеста в БД. Возвращает присвоенный run_id.

        source — идентификатор инструмента (тикер или csv путь);
        request.data_source — тип источника ('t_invest', 'yfinance', 'csv').

        artifact_id и applied_components наследуются из request и резолва артефакта
        (resolved заранее в BacktestRunner.run; здесь читаем из request.artifact_id).
        """
        meta = response.metadata
        artifact_id = request.artifact_id
        applied_components: list[str] = []
        if artifact_id is not None:
            with get_uow_factory()() as uow:
                artifact = uow.model_registry.get_by_id(artifact_id)
            if artifact is not None:
                applied_components = _resolve_applied_components(artifact)

        record = BacktestRunRecord(
            model_name=response.model.get("name", "unknown"),
            ticker=source,
            source=request.data_source,
            interval=request.interval,
            artifact_id=artifact_id,
            applied_components=applied_components,
            train_window_mode=meta.get("train_window_mode", "sliding"),
            train_window_size=meta.get("train_window_size", request.train_window_size),
            horizon=request.horizon,
            step=meta.get("step", request.horizon),
            backtest_target=request.backtest_target,
            evaluation_weights=request.evaluation_weights,
            weight_first_to_last_ratio=request.weight_first_to_last_ratio,
            bootstrap_iterations=request.bootstrap_iterations,
            ci_z_score=request.ci_z_score,
            history_period=request.history_period,
            history_up_to=request.history_up_to,
            history_length=response.history_length,
            feature_plugins=list(request.feature_plugins),
            windows_count=response.windows_count,
            metrics=response.metrics,
            metrics_ci=response.metrics_ci,
            metrics_lcb=response.metrics_lcb,
            metadata=response.metadata,
        )
        with get_uow_factory()() as uow:
            return uow.backtest_repository.save_run(record)


# ---------------------------------------------------------------------------
# applied_components resolution
# ---------------------------------------------------------------------------

# Канонический порядок применения компонентов (тот же что в ArtifactLoader.apply).
# Используется для нормализации applied_components чтобы всегда был стабильный
# порядок head → input → lora → full_ft, независимо от того в каком порядке
# их перечислили в БД.
_APPLY_ORDER = ("head", "input", "lora", "full_ft")


def _resolve_applied_components(artifact) -> list[str]:
    """
    Собирает «реально применённые компоненты» с учётом transitive-ссылок.

    Прямые компоненты артефакта (artifact.training_components) расширяются:
      - Если артефакт = LoRA и в params есть base_head_artifact_id → добавить 'head'
      - Если есть base_input_artifact_id → добавить 'input'

    Результат сортируется в каноническом порядке применения. Это даёт точный
    answer на вопрос «какая модель реально применялась в этом backtest run'е»:
      LoRA #Z с base_head=X, base_input=Y → applied_components=["head","input","lora"]
    Это видно в /backtest/runs и в dashboard, понятно при анализе результатов.
    """
    direct = set(artifact.training_components or [])
    params = artifact.params or {}
    if "lora" in direct:
        if params.get("base_head_artifact_id"):
            direct.add("head")
        if params.get("base_input_artifact_id"):
            direct.add("input")
    return [c for c in _APPLY_ORDER if c in direct]
