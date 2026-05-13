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
        return self._chart_service.build(result)

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

        source здесь это идентификатор инструмента (тикер или csv путь),
        request.data_source — тип источника ('t_invest', 'yfinance', 'csv').
        """
        meta = response.metadata
        record = BacktestRunRecord(
            model_name=response.model.get("name", "unknown"),
            ticker=source,
            source=request.data_source,
            interval=request.interval,
            has_lora=False,                   # пока без LoRA; добавится в Step 4
            lora_artifact_id=None,
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
