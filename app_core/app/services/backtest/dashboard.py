"""
HTML-дашборд для просмотра backtest_runs.

Содержит:
  - Curve: train_window_size → ranking_metric с error bars (CI)
  - Таблица: все runs по фильтру с ключевыми метриками
  - Контекст: тикер/интервал/модель, число sweep'ов, последний run

Использует Plotly для графиков, простой HTML+CSS для остального.
"""
from __future__ import annotations

import json
from typing import Iterable

import plotly.graph_objects as go

from ...storage import get_uow_factory
from ...storage.ports import BacktestRunRecord


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Backtest Dashboard — {title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ font-family: ui-sans-serif, system-ui, sans-serif; background: #0f1115; color: #e6e6e6; margin: 24px; }}
  h1 {{ font-size: 20px; font-weight: 600; margin: 0 0 8px; }}
  h2 {{ font-size: 16px; font-weight: 600; margin: 24px 0 8px; color: #9aa0a6; }}
  .meta {{ color: #9aa0a6; font-size: 13px; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #2a2d33; }}
  th {{ background: #1a1d22; text-align: right; color: #c8cdd3; font-weight: 600; }}
  th.left, td.left {{ text-align: left; }}
  tr.recommended {{ background: rgba(102, 204, 102, 0.12); }}
  .badge {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 11px; }}
  .badge.coarse {{ background: #2d4a73; color: #cfe2ff; }}
  .badge.refinement {{ background: #5a3c7a; color: #e0d0f0; }}
  .badge.completed {{ background: #2e6e2e; color: #d4f4d4; }}
  .badge.skipped {{ background: #6e6e2e; color: #f4f4d4; }}
  .badge.not_selected {{ background: #4a4a4a; color: #c0c0c0; }}
  .empty {{ color: #6c7280; font-style: italic; padding: 16px; }}
</style>
</head>
<body>
<h1>Backtest Dashboard</h1>
<div class="meta">{meta}</div>

<h2>Кривая: train_window_size → метрика (LCB)</h2>
<div id="curve"></div>

{diff_section}

<h2>Все runs ({count})</h2>
{table}

<script>
const curveData = {curve_data};
const curveLayout = {curve_layout};
if (curveData.length > 0) {{
  Plotly.newPlot('curve', curveData, curveLayout, {{displayModeBar: false}});
}} else {{
  document.getElementById('curve').innerHTML = '<div class="empty">Нет данных для построения кривой.</div>';
}}
</script>
</body>
</html>
"""


def _format_value(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✓" if v else "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _ranking_metric_for_target(target: str) -> str:
    """Дефолтная ranking-метрика для отображения на кривой."""
    return "skill_mae_close" if target == "ohlc" else "skill_mae"


def _build_curve(records: list[BacktestRunRecord]) -> tuple[list[dict], dict]:
    """
    Кривая: ось X = train_window_size, Y = ranking_metric (mean) с error bars (CI).
    Включаем только primary runs (cv_fold_index IS NULL), группируем по
    (sweep_id, train_window_size) для красоты.
    """
    primary = [r for r in records if r.cv_fold_index is None]
    if not primary:
        return [], {}

    # Метрика выбирается по target первого run (все обычно одинаковые)
    metric_name = _ranking_metric_for_target(primary[0].backtest_target)

    # Группируем по sweep_id чтобы каждый sweep был отдельной линией
    by_sweep: dict[int | None, list[BacktestRunRecord]] = {}
    for r in primary:
        by_sweep.setdefault(r.sweep_id, []).append(r)

    traces: list[dict] = []
    for sweep_id, runs in by_sweep.items():
        runs_sorted = sorted(runs, key=lambda r: r.train_window_size)
        xs = [r.train_window_size for r in runs_sorted]
        ys = [r.metrics.get(metric_name, 0.0) for r in runs_sorted]
        ci_low = [r.metrics_ci.get(metric_name, [y, y])[0] for r, y in zip(runs_sorted, ys)]
        ci_high = [r.metrics_ci.get(metric_name, [y, y])[1] for r, y in zip(runs_sorted, ys)]
        err_minus = [y - lo for y, lo in zip(ys, ci_low)]
        err_plus = [hi - y for y, hi in zip(ys, ci_high)]

        label = f"sweep {sweep_id}" if sweep_id is not None else "standalone"
        traces.append({
            "type":  "scatter",
            "mode":  "lines+markers",
            "x":     xs,
            "y":     ys,
            "name":  label,
            "error_y": {
                "type": "data",
                "symmetric": False,
                "array": err_plus,
                "arrayminus": err_minus,
                "visible": True,
                "color": "rgba(150,150,150,0.5)",
            },
            "marker": {"size": 8},
            "line":   {"width": 2},
        })

    layout = {
        "xaxis": {"title": "train_window_size (контекст)", "type": "log", "gridcolor": "#2a2d33"},
        "yaxis": {"title": metric_name, "gridcolor": "#2a2d33"},
        "paper_bgcolor": "#0f1115",
        "plot_bgcolor":  "#15181d",
        "font": {"color": "#e6e6e6"},
        "hovermode": "x unified",
        "margin": {"t": 30, "r": 20, "b": 50, "l": 60},
        "height": 420,
    }
    return traces, layout


def _build_table(records: list[BacktestRunRecord], recommended_sweep_ids: set[int]) -> str:
    if not records:
        return '<div class="empty">Нет runs по выбранным фильтрам.</div>'

    target = records[0].backtest_target
    metric_name = _ranking_metric_for_target(target)

    rows_html: list[str] = []
    for r in records:
        kind = "CV" if r.cv_fold_index is not None else "primary"
        sweep_cell = f"#{r.sweep_id}" if r.sweep_id is not None else "—"
        cv_cell = f"fold {r.cv_fold_index}" if r.cv_fold_index is not None else "—"
        metric_val = r.metrics.get(metric_name, 0.0)
        metric_lcb = r.metrics_lcb.get(metric_name, 0.0)
        dir_acc = r.metrics.get("directional_acc", 0.0)
        cov = r.metrics.get("coverage_q10_q90", 0.0)
        rows_html.append(
            f"<tr>"
            f"<td class='left'>{r.id}</td>"
            f"<td class='left'>{r.created_at or ''}</td>"
            f"<td>{r.train_window_size}</td>"
            f"<td>{r.horizon}</td>"
            f"<td>{r.windows_count}</td>"
            f"<td class='left'>{kind}</td>"
            f"<td class='left'>{sweep_cell}</td>"
            f"<td class='left'>{cv_cell}</td>"
            f"<td>{_format_value(metric_val)}</td>"
            f"<td>{_format_value(metric_lcb)}</td>"
            f"<td>{_format_value(dir_acc)}</td>"
            f"<td>{_format_value(cov)}</td>"
            f"</tr>"
        )

    return f"""
<table>
<thead><tr>
  <th class="left">id</th>
  <th class="left">created_at</th>
  <th>ctx</th>
  <th>horizon</th>
  <th>windows</th>
  <th class="left">kind</th>
  <th class="left">sweep</th>
  <th class="left">cv</th>
  <th>{metric_name}</th>
  <th>{metric_name}_lcb</th>
  <th>dir_acc</th>
  <th>cov_q10_q90</th>
</tr></thead>
<tbody>{''.join(rows_html)}</tbody>
</table>
"""


def build_dashboard_html(
        ticker: str | None = None,
        source: str | None = None,
        interval: str | None = None,
        model_name: str | None = None,
        sweep_ids: list[int] | None = None,
) -> str:
    """
    Собирает HTML-страницу со всеми runs.

    Два режима фильтрации:
      A) sweep_ids — если задан, показываем ТОЛЬКО эти sweep'ы (для сравнения,
         например base vs LoRA). Остальные фильтры игнорируются. Каждый sweep
         рисуется отдельной кривой; в таблице — все runs всех sweep'ов с
         подписями к какому sweep'у принадлежит.
      B) ticker/source/interval/model_name — фильтрация по конкретному
         инструменту/модели. Показывает все runs прошедшие фильтр.
         Использовать когда нужен общий обзор без фокуса на конкретные sweep'ы.
    """
    with get_uow_factory()() as uow:
        if sweep_ids:
            records: list[BacktestRunRecord] = []
            for sid in sweep_ids:
                records.extend(uow.backtest_repository.get_sweep_runs(sid))
        else:
            records = uow.backtest_repository.get_runs(
                model_name=model_name, ticker=ticker, source=source, interval=interval,
                limit=1000, offset=0,
            )

    if sweep_ids:
        title = f"sweeps {','.join(str(s) for s in sweep_ids)}"
        meta = f"Сравнение sweep'ов: {', '.join(str(s) for s in sweep_ids)} · runs: {len(records)}"
    else:
        title_parts = [p for p in (ticker, interval, model_name) if p]
        title = " · ".join(title_parts) if title_parts else "all runs"
        meta = (
            f"ticker={ticker or 'все'}, source={source or 'все'}, "
            f"interval={interval or 'все'}, model={model_name or 'все'} · "
            f"найдено runs: {len(records)}"
        )

    curve_traces, curve_layout = _build_curve(records)
    table_html = _build_table(records, recommended_sweep_ids=set())
    diff_html = _build_diff_table(records) if sweep_ids and len(sweep_ids) >= 2 else ""

    return _HTML_TEMPLATE.format(
        title=title,
        meta=meta,
        count=len(records),
        table=table_html,
        diff_section=diff_html,
        curve_data=json.dumps(curve_traces),
        curve_layout=json.dumps(curve_layout),
    )


def _build_diff_table(records: list[BacktestRunRecord]) -> str:
    """
    Дифф-таблица: для каждого train_window_size показывает значения метрики
    в каждом sweep'е и diff между ними.
    """
    primary = [r for r in records if r.cv_fold_index is None]
    if not primary:
        return ""

    metric_name = _ranking_metric_for_target(primary[0].backtest_target)

    # Группируем: ctx -> {sweep_id -> value}
    by_ctx: dict[int, dict[int | None, float]] = {}
    sweep_set: set[int | None] = set()
    for r in primary:
        sweep_set.add(r.sweep_id)
        by_ctx.setdefault(r.train_window_size, {})[r.sweep_id] = r.metrics.get(metric_name, 0.0)

    sweep_ids_sorted = sorted([s for s in sweep_set if s is not None])
    if len(sweep_ids_sorted) < 2:
        return ""

    # Базовый sweep — первый в списке. Остальные сравниваются с ним.
    base_id = sweep_ids_sorted[0]

    header_cols = ["ctx"]
    for sid in sweep_ids_sorted:
        header_cols.append(f"sweep #{sid}")
    for sid in sweep_ids_sorted[1:]:
        header_cols.append(f"Δ vs #{base_id}")

    rows: list[str] = []
    for ctx in sorted(by_ctx):
        cells = [str(ctx)]
        base_val = by_ctx[ctx].get(base_id)
        for sid in sweep_ids_sorted:
            v = by_ctx[ctx].get(sid)
            cells.append(_format_value(v))
        for sid in sweep_ids_sorted[1:]:
            v = by_ctx[ctx].get(sid)
            if v is not None and base_val is not None:
                d = v - base_val
                color = "color:#7cd97c" if d > 0 else ("color:#d97c7c" if d < 0 else "")
                cells.append(f'<span style="{color}">{d:+.4f}</span>')
            else:
                cells.append("—")
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    head = "".join(f"<th>{c}</th>" for c in header_cols)
    return f"""
<h2>Сравнение sweep'ов по метрике <code>{metric_name}</code></h2>
<table>
<thead><tr>{head}</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
"""
