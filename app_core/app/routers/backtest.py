"""
Backtest: одиночный запуск, sweep, просмотр истории, HTML дашборд.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from ..schemas import BacktestRequest, BacktestResponse, BacktestSweepRequest, BacktestSweepResponse
from ..services.backtest import build_dashboard_html
from ..storage import get_uow_factory
from ._common import service_for_model, endpoint_errors

router = APIRouter(prefix="/backtest", tags=["backtest"])


# --------------------------------------------------------------------------
# Запуск бэктестов
# --------------------------------------------------------------------------

@router.post("", response_model=BacktestResponse)
@endpoint_errors("Backtest failed")
def backtest(request: BacktestRequest) -> BacktestResponse:
    return service_for_model(request.model_name).run_backtest(request)


@router.post("/sweep", response_model=BacktestSweepResponse)
@endpoint_errors("Sweep failed")
def backtest_sweep(request: BacktestSweepRequest) -> BacktestSweepResponse:
    """
    Запускает sweep по train_window_size для поиска оптимального контекста.
    Возвращает sweep_id, кривую (configs), рекомендованный конфиг и сводку CV.

    Все промежуточные runs сохраняются в backtest_runs с общим sweep_id.
    Полную раскадровку sweep'а можно получить через GET /backtest/sweep/{sweep_id}.
    """
    return service_for_model(request.model_name).run_sweep(request)


# --------------------------------------------------------------------------
# Просмотр истории запусков
# --------------------------------------------------------------------------

# Ключевые метрики которые показываем в кратком виде (для списков и UI-таблиц).
# Полная картина с CI/LCB/metadata доступна через detailed=True.
_KEY_METRICS = (
    "skill_mae_close", "skill_mae", "mae_close", "mae",
    "directional_acc", "coverage_q10_q90", "pinball_mean",
)


def _run_to_dict(record, detailed: bool) -> dict:
    """
    Преобразует BacktestRunRecord в словарь для JSON-ответа.

    detailed=False: краткая выдача (для списков, /runs):
       id, ключевые параметры, агрегированные mean-метрики, lcb по skill.
    detailed=True: полный снимок (для /runs/{id}):
       все поля включая metrics_ci, metrics_lcb, metadata.
    """
    base = {
        "id":                  record.id,
        "model_name":          record.model_name,
        "ticker":              record.ticker,
        "source":              record.source,
        "interval":            record.interval,
        "artifact_id":         record.artifact_id,
        "applied_components":  record.applied_components,
        "train_window_mode":   record.train_window_mode,
        "train_window_size":   record.train_window_size,
        "horizon":             record.horizon,
        "step":                record.step,
        "backtest_target":     record.backtest_target,
        "windows_count":       record.windows_count,
        "history_length":      record.history_length,
        "sweep_id":            record.sweep_id,
        "parent_run_id":       record.parent_run_id,
        "cv_fold_index":       record.cv_fold_index,
        "created_at":          record.created_at,
    }
    if detailed:
        base.update({
            "evaluation_weights":         record.evaluation_weights,
            "weight_first_to_last_ratio": record.weight_first_to_last_ratio,
            "bootstrap_iterations":       record.bootstrap_iterations,
            "ci_z_score":                 record.ci_z_score,
            "history_period":             record.history_period,
            "history_up_to":              record.history_up_to,
            "feature_plugins":            record.feature_plugins,
            "metrics":                    record.metrics,
            "metrics_ci":                 record.metrics_ci,
            "metrics_lcb":                record.metrics_lcb,
            "metadata":                   record.metadata,
        })
    else:
        base["metrics"]     = {k: v for k, v in record.metrics.items() if k in _KEY_METRICS}
        base["metrics_lcb"] = {k: v for k, v in record.metrics_lcb.items() if k in _KEY_METRICS}
    return base


@router.get("/runs")
@endpoint_errors("Ошибка получения runs")
def list_runs(
        model_name:  str | None = Query(None, description="Фильтр по имени модели"),
        ticker:      str | None = Query(None, description="Фильтр по тикеру"),
        source:      str | None = Query(None, description="Фильтр по источнику (t_invest/yfinance/csv)"),
        interval:    str | None = Query(None, description="Фильтр по интервалу"),
        artifact_id: int | None = Query(None, description="Фильтр по id дообученного артефакта; null/0 → только base модель"),
        sweep_id:    int | None = Query(None, description="Только runs одного sweep'а"),
        limit:       int = Query(100, ge=1, le=1000),
        offset:      int = Query(0, ge=0),
        detailed:    bool = Query(False, description="Полная выдача со всеми метриками и metadata"),
) -> dict:
    """
    Список runs с фильтрами. По умолчанию краткая выдача — для UI-таблиц.
    detailed=true — полная структура каждого run'а (тяжелее, для отладки).
    """
    with get_uow_factory()() as uow:
        records = uow.backtest_repository.get_runs(
            model_name=model_name, ticker=ticker, source=source, interval=interval,
            artifact_id=artifact_id, sweep_id=sweep_id, limit=limit, offset=offset,
        )
    return {
        "count": len(records),
        "runs": [_run_to_dict(r, detailed=detailed) for r in records],
    }


@router.get("/runs/{run_id}")
@endpoint_errors("Ошибка получения run")
def get_run(run_id: int) -> dict:
    """Один run с полным набором полей."""
    with get_uow_factory()() as uow:
        record = uow.backtest_repository.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"run_id={run_id} не найден")
    return _run_to_dict(record, detailed=True)


@router.get("/sweep/{sweep_id}")
@endpoint_errors("Ошибка получения sweep")
def get_sweep(sweep_id: int, detailed: bool = Query(False)) -> dict:
    """Все runs одного sweep'а (primary + CV) в порядке выполнения."""
    with get_uow_factory()() as uow:
        records = uow.backtest_repository.get_sweep_runs(sweep_id)
    if not records:
        raise HTTPException(status_code=404, detail=f"sweep_id={sweep_id} не найден")
    return {
        "sweep_id": sweep_id,
        "count": len(records),
        "runs": [_run_to_dict(r, detailed=detailed) for r in records],
    }


# --------------------------------------------------------------------------
# HTML дашборд
# --------------------------------------------------------------------------

@router.get("/dashboard", response_class=HTMLResponse)
@endpoint_errors("Dashboard failed")
def dashboard(
        ticker:     str | None = Query(None, description="Фильтр по тикеру"),
        source:     str | None = Query(None, description="Фильтр по источнику"),
        interval:   str | None = Query(None, description="Фильтр по интервалу"),
        model_name: str | None = Query(None, description="Фильтр по модели"),
        sweep_ids:  str | None = Query(
            None,
            description=(
                "Список sweep_id через запятую: '2' для одного sweep'а, '20,21,22' для "
                "сравнения нескольких. Если задан — остальные фильтры игнорируются, "
                "каждый sweep рисуется отдельной кривой; для нескольких внизу появляется "
                "diff-таблица с Δ метрики."
            )
        ),
) -> HTMLResponse:
    """
    HTML страница с кривой train_window_size → метрика и таблицей всех runs.

    Два режима:
      A) sweep_ids='2' или '20,21' — конкретные sweep'ы
      B) Без sweep_ids — фильтрация по ticker/source/interval/model_name.
    """
    parsed_sweep_ids = None
    if sweep_ids:
        try:
            parsed_sweep_ids = [int(s.strip()) for s in sweep_ids.split(",") if s.strip()]
        except ValueError as ex:
            raise HTTPException(
                status_code=400,
                detail=f"sweep_ids должен быть списком целых через запятую (например '2' или '20,21,22'), а не '{sweep_ids}'",
            ) from ex
    html = build_dashboard_html(
        ticker=ticker, source=source, interval=interval, model_name=model_name,
        sweep_ids=parsed_sweep_ids,
    )
    return HTMLResponse(content=html)
