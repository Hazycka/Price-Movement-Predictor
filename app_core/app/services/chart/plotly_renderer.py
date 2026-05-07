"""
Рендерер прогнозных графиков с квантильными bands.

Макет зависит от наличия прогноза:
  С прогнозом  → два субплота горизонтально (узкий/широкий диапазон) + объём
  Без прогноза → один субплот с историей + объём

Ось X: категориальная (type="category") — свечи равномерно без пробелов,
как в TradingView. Реальные даты отображаются в hover через customdata.

Каждая зона прогноза содержит 3 полупрозрачных свечи на одном x:
  - нижний квантиль (q25 или q10) — светлее
  - медиана q50 — белая полая
  - верхний квантиль (q75 или q90) — темнее

Вертикальные линии:
  - красная пунктирная: начало окна данных используемых моделью
  - белая пунктирная:   начало зоны прогноза (только если есть прогноз)
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Цветовая палитра
# ---------------------------------------------------------------------------

BULL_LIGHT  = "rgba(102, 204, 102, 0.35)"
BULL_MEDIAN = "rgba(255, 255, 255, 0.55)"
BULL_DARK   = "rgba(0,  140,  70,  0.50)"

BEAR_LIGHT  = "rgba(255, 153, 153, 0.35)"
BEAR_MEDIAN = "rgba(255, 255, 255, 0.55)"
BEAR_DARK   = "rgba(178,  34,  34, 0.50)"


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _make_category_labels(dates: list, prefix: str = "") -> list[str]:
    """
    Конвертирует список дат/строк в категориальные метки для оси X.
    Категориальная ось убирает пробелы выходных — свечи идут непрерывно.
    Реальные даты передаются в hover через customdata.
    """
    return [f"{prefix}{i}" for i in range(len(dates))]


def _candle_direction(close: float, prev_close: float) -> str:
    return "bull" if close >= prev_close else "bear"


def _add_history(
        fig: go.Figure,
        x_history: list[str],
        dates_hover: list[str],
        candles: list[dict],
        history_volume: list[float],
        indicators: dict,
        chart_type: str,
        row: int,
        col: int,
        vol_row: int,
        show_legend: bool,
) -> None:
    close_series = [c["close"] for c in candles]

    if chart_type == "candlestick":
        fig.add_trace(go.Candlestick(
            x=x_history,
            open=[c["open"]  for c in candles],
            high=[c["high"]  for c in candles],
            low=[c["low"]   for c in candles],
            close=[c["close"] for c in candles],
            customdata=dates_hover,
            text=dates_hover,
            name="История",
            showlegend=show_legend,
            legendgroup="history",
        ), row=row, col=col)
    else:
        fig.add_trace(go.Scatter(
            x=x_history,
            y=close_series,
            customdata=dates_hover,
            mode="lines+markers",
            name="История",
            showlegend=show_legend,
            legendgroup="history",
            line=dict(width=2),
            marker=dict(size=4),
        ), row=row, col=col)

    for name, values in indicators.items():
        fig.add_trace(go.Scatter(
            x=x_history,
            y=values,
            mode="lines",
            name=name,
            showlegend=show_legend,
            legendgroup=f"ind_{name}",
            line=dict(width=1),
        ), row=row, col=col)

    if any(v > 0 for v in history_volume):
        volume_colors = [
            "rgba(38,166,154,0.45)" if candles[i]["close"] >= candles[i]["open"]
            else "rgba(239,83,80,0.45)"
            for i in range(len(candles))
        ]
        fig.add_trace(go.Bar(
            x=x_history,
            y=history_volume,
            name="Объём (история)",
            showlegend=show_legend,
            legendgroup="vol_history",
            marker=dict(color=volume_colors),
        ), row=vol_row, col=col)


def _add_quantile_candles(
        fig: go.Figure,
        x_forecast: list[str],
        dates_hover: list[str],
        forecast_candles: list[dict],
        quantiles_low: dict[str, list],
        quantiles_high: dict[str, list],
        label_low: str,
        label_high: str,
        last_history_close: float,
        row: int,
        col: int,
        show_legend: bool,
) -> None:
    horizon = len(forecast_candles)
    prev_closes = [last_history_close] + [forecast_candles[i]["close"] for i in range(horizon - 1)]

    layers = [
        ("light",  label_low,  quantiles_low),
        ("median", "q50 (медиана)", {
            "open":  [forecast_candles[i]["open"]  for i in range(horizon)],
            "high":  [forecast_candles[i]["high"]  for i in range(horizon)],
            "low":   [forecast_candles[i]["low"]   for i in range(horizon)],
            "close": [forecast_candles[i]["close"] for i in range(horizon)],
        }),
        ("dark",   label_high, quantiles_high),
    ]

    for role, label, ohlc in layers:
        if role == "median":
            increasing = dict(line=dict(color=BULL_MEDIAN, width=1.5), fillcolor="rgba(0,0,0,0)")
            decreasing = dict(line=dict(color=BEAR_MEDIAN, width=1.5), fillcolor="rgba(0,0,0,0)")
        else:
            fill      = BULL_LIGHT if role == "light" else BULL_DARK
            fill_bear = BEAR_LIGHT if role == "light" else BEAR_DARK
            increasing = dict(line=dict(color=fill,      width=0.8), fillcolor=fill)
            decreasing = dict(line=dict(color=fill_bear, width=0.8), fillcolor=fill_bear)

        fig.add_trace(go.Candlestick(
            x=x_forecast,
            open=ohlc["open"],
            high=ohlc["high"],
            low=ohlc["low"],
            close=ohlc["close"],
            customdata=dates_hover,
            text=dates_hover,
            name=label,
            showlegend=show_legend,
            legendgroup=f"fc_{role}",
            increasing=increasing,
            decreasing=decreasing,
        ), row=row, col=col)


def _add_vertical_line(
        fig: go.Figure,
        x_val: str,
        y_min: float,
        y_max: float,
        color: str,
        dash: str,
        name: str,
        show_legend: bool,
        row: int,
        col: int,
) -> None:
    if x_val is None:
        return
    fig.add_trace(go.Scatter(
        x=[x_val, x_val],
        y=[y_min, y_max],
        mode="lines",
        name=name,
        showlegend=show_legend,
        legendgroup=name,
        line=dict(width=1.5, dash=dash, color=color),
        hovertemplate=f"{name}<extra></extra>",
    ), row=row, col=col)


def _find_category_label(
        x_labels: list[str],
        dt_str: str,
        dates_list: list[str],
) -> str | None:
    """
    Находит категориальную метку для заданной даты.
    Ищет ближайшую дату в dates_list и возвращает соответствующий x_label.
    """
    if not dt_str or not dates_list:
        return None
    try:
        target = pd.to_datetime(dt_str, utc=True, errors="coerce")
        if pd.isna(target):
            return None
        parsed = pd.to_datetime(dates_list, utc=True, errors="coerce")
        diffs = (parsed - target).abs()
        idx = diffs.argmin()
        return x_labels[idx]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def build_forecast_chart_html(
        title: str,
        labels: list[str] | None,
        candles: list[dict[str, float]],
        forecast_candles: list[dict[str, float]],
        forecast_ohlc_quantiles: dict | None,
        indicators: dict[str, list[float | None]],
        chart_type_history: str = "candlestick",
        chart_type_forecast: str = "candlestick",
        interval: str = "1d",
        model_input_start_date: str | None = None,
        max_history_candles: int | None = None,
) -> str:
    if not candles:
        raise ValueError("Candles series is empty.")

    # --- Пункт 4: ограничение истории для отрисовки ---
    if max_history_candles is not None and len(candles) > max_history_candles:
        display_candles = candles[-max_history_candles:]
        display_labels  = (labels[-max_history_candles:] if labels and len(labels) == len(candles) else labels)
    else:
        display_candles = candles
        display_labels  = labels

    has_forecast = bool(forecast_candles)
    horizon      = len(forecast_candles) if has_forecast else 0

    # --- Пункт 5: категориальные метки вместо дат — убирает пробелы выходных ---
    # История
    x_history    = [str(i) for i in range(len(display_candles))]
    dates_history = [str(d) for d in (display_labels or x_history)]

    # Прогноз — продолжаем нумерацию с конца истории
    history_len  = len(display_candles)
    x_forecast   = [str(history_len + i) for i in range(horizon)] if has_forecast else []

    # Даты прогноза для hover — генерируем через date_range если нет реальных
    if has_forecast:
        if display_labels:
            last_real_dt = pd.to_datetime(display_labels[-1], errors="coerce")
        else:
            last_real_dt = pd.Timestamp("2000-01-01")
        freq_map = {"1d": "B", "1h": "h", "4h": "4h", "1w": "W", "1mo": "MS"}
        freq = freq_map.get(interval, "h")
        future_dts = pd.date_range(start=last_real_dt, periods=horizon + 1, freq=freq)[1:]
        dates_forecast = [str(d) for d in future_dts]
    else:
        dates_forecast = []

    history_volume = [c.get("volume", 0.0) for c in display_candles]

    # --- Ценовой диапазон для вертикальных линий ---
    all_prices = (
            [c["high"] for c in display_candles] +
            [c["low"]  for c in display_candles] +
            ([c["high"] for c in forecast_candles] if has_forecast else []) +
            ([c["low"]  for c in forecast_candles] if has_forecast else [])
    )
    y_min = min(all_prices)
    y_max = max(all_prices)
    price_pad = (y_max - y_min) * 0.05
    y_min -= price_pad
    y_max += price_pad

    last_history_close = float(display_candles[-1]["close"])

    has_quantiles = (
            has_forecast
            and forecast_ohlc_quantiles is not None
            and chart_type_forecast == "candlestick"
    )

    if has_quantiles:
        def _extract(q_key: str, ch: str) -> list[float]:
            return forecast_ohlc_quantiles[ch][q_key]
        narrow_low  = {ch: _extract("q25", ch) for ch in ("open", "high", "low", "close")}
        narrow_high = {ch: _extract("q75", ch) for ch in ("open", "high", "low", "close")}
        wide_low    = {ch: _extract("q10", ch) for ch in ("open", "high", "low", "close")}
        wide_high   = {ch: _extract("q90", ch) for ch in ("open", "high", "low", "close")}

    # --- Пункт 2: макет зависит от наличия прогноза ---
    if has_forecast:
        num_cols = 2
        subplot_titles = [
            "Узкий диапазон [q25 / q50 / q75]",
            "Широкий диапазон [q10 / q50 / q90]",
            "", "",
        ]
        column_widths = [0.5, 0.5]
    else:
        num_cols = 1
        subplot_titles = ["История", ""]
        column_widths = [1.0]

    fig = make_subplots(
        rows=2, cols=num_cols,
        shared_xaxes=False,
        row_heights=[0.85, 0.15],
        column_widths=column_widths,
        horizontal_spacing=0.04,
        vertical_spacing=0.02,
        subplot_titles=subplot_titles,
    )

    cols_to_draw = (1, 2) if has_forecast else (1,)

    for col in cols_to_draw:
        show_legend = (col == 1)

        _add_history(
            fig=fig,
            x_history=x_history,
            dates_hover=dates_history,
            candles=display_candles,
            history_volume=history_volume,
            indicators=indicators,
            chart_type=chart_type_history,
            row=1, col=col,
            vol_row=2,
            show_legend=show_legend,
        )

        if has_forecast:
            if has_quantiles:
                low_q   = narrow_low  if col == 1 else wide_low
                high_q  = narrow_high if col == 1 else wide_high
                label_l = "q25" if col == 1 else "q10"
                label_h = "q75" if col == 1 else "q90"

                _add_quantile_candles(
                    fig=fig,
                    x_forecast=x_forecast,
                    dates_hover=dates_forecast,
                    forecast_candles=forecast_candles,
                    quantiles_low=low_q,
                    quantiles_high=high_q,
                    label_low=label_l,
                    label_high=label_h,
                    last_history_close=last_history_close,
                    row=1, col=col,
                    show_legend=show_legend,
                )
            else:
                fig.add_trace(go.Candlestick(
                    x=x_forecast,
                    open=[c["open"]  for c in forecast_candles],
                    high=[c["high"]  for c in forecast_candles],
                    low=[c["low"]   for c in forecast_candles],
                    close=[c["close"] for c in forecast_candles],
                    customdata=dates_forecast,
                    name="Прогноз (q50)",
                    showlegend=show_legend,
                    legendgroup="fc_median",
                ), row=1, col=col)

            # Вертикаль начала прогноза (белая) — граница история/прогноз
            _add_vertical_line(
                fig=fig,
                x_val=x_forecast[0] if x_forecast else None,
                y_min=y_min, y_max=y_max,
                color="rgba(255, 255, 255, 0.7)",
                dash="dash",
                name="Начало прогноза",
                show_legend=show_legend,
                row=1, col=col,
            )

        # Вертикаль начала окна модели (красная)
        if model_input_start_date and display_labels:
            x_model_start = _find_category_label(x_history, model_input_start_date, dates_history)
            if x_model_start:
                _add_vertical_line(
                    fig=fig,
                    x_val=x_model_start,
                    y_min=y_min, y_max=y_max,
                    color="rgba(255, 80, 80, 0.8)",
                    dash="dot",
                    name="Начало окна модели",
                    show_legend=show_legend,
                    row=1, col=col,
                )

    # --- Пункт 1 + 3: настройки осей ---
    # Категориальная ось X — непрерывный график без пробелов выходных
    xaxis_price_cfg = dict(
        showgrid=False,
        zeroline=False,
        type="category",          # ← ключевое: категории вместо дат
        fixedrange=False,
        tickangle=-45,
        showticklabels=False,     # скрываем категориальные индексы, даты в hover
        rangeslider=dict(
            visible=False,        # убираем нижний мини-график/rangeslider
        ),
    )

    # Пункт 1: объёмные оси привязаны к ценовым через matches
    # Пункт 3: yaxis_price fixedrange=True — зум только по X, ширина свечей меняется
    yaxis_price_cfg = dict(showgrid=True, zeroline=False, automargin=True, fixedrange=True)
    yaxis_vol_cfg   = dict(showgrid=False, zeroline=False, title="Объём",
                           fixedrange=True, automargin=True)

    if has_forecast:
        fig.update_layout(
            xaxis=xaxis_price_cfg,
            xaxis2=xaxis_price_cfg,
            xaxis3={**xaxis_price_cfg, "matches": "x"},   # объём левый → за ценовым левым
            xaxis4={**xaxis_price_cfg, "matches": "x2"},  # объём правый → за ценовым правым
            yaxis=yaxis_price_cfg,
            yaxis2=yaxis_price_cfg,
            yaxis3=yaxis_vol_cfg,
            yaxis4=yaxis_vol_cfg,
        )
    else:
        fig.update_layout(
            xaxis=xaxis_price_cfg,
            xaxis2={**xaxis_price_cfg, "matches": "x"},   # объём → за ценовым
            yaxis=yaxis_price_cfg,
            yaxis2=yaxis_vol_cfg,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(
            orientation="v",
            x=1.01,
            y=1.0,
            xanchor="left",
            bgcolor="rgba(30,30,30,0.8)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=30, r=180, t=60, b=30),
        autosize=True,
        bargap=0.05,
        dragmode="pan",
    )

    html_content = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={
            "responsive":  True,
            "scrollZoom":  True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["toggleSpikelines"],
        }
    )

    return f"""
    <html>
        <head>
            <style>
                body, html {{
                    margin: 0; padding: 0; height: 100%;
                    overflow: hidden; background: #111;
                    font-family: sans-serif;
                }}
                #chart {{ height: 100vh; width: 100vw; }}
            </style>
        </head>
        <body>
            <div id="chart">{html_content}</div>
        </body>
    </html>
    """