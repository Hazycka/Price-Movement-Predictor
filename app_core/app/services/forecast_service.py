from ..schemas import ForecastRequest, ForecastResponse, BacktestRequest, BacktestResponse
from ..models.base import ForecastModel

from .forecast import (
    InputResolver,
    ForecastContextBuilder,
    ForecastMetadataBuilder,
    ForecastOrchestrator,
)
from .backtest import BacktestRunner
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
        return self._backtest_runner.run(
            request=request,
            source=source,
            dates=dates,
            candles=candles
        )

    def build_chart(self, request: ForecastRequest) -> str:
        result = self.run_forecast(request)
        return self._chart_service.build(result)