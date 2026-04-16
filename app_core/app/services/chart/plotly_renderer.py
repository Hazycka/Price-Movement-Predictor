import pandas as pd
import plotly.graph_objects as go


def _forecast_dates(last_date: pd.Timestamp, horizon: int, interval: str) -> list[pd.Timestamp]:
    if interval.endswith("d"):
        dates = pd.bdate_range(start=last_date, periods=horizon + 1)
        return list(dates[1:])

    if interval.endswith("h"):
        dates = pd.date_range(start=last_date, periods=horizon + 1, freq=interval)
        return list(dates[1:])

    if interval.endswith("m"):
        minutes = interval.replace("m", "")
        dates = pd.date_range(start=last_date, periods=horizon + 1, freq=f"{minutes}min")
        return list(dates[1:])

    dates = pd.bdate_range(start=last_date, periods=horizon + 1)
    return list(dates[1:])


def build_forecast_chart_html(
        title: str,
        labels: list[str] | None,
        candles: list[dict[str, float]],
        forecast_candles: list[dict[str, float]],
        indicators: dict[str, list[float | None]],
        chart_type_history: str = "candlestick",
        chart_type_forecast: str = "candlestick",
        interval: str = "1d",
        model_input_start_date: str | None = None
) -> str:
    if not candles:
        raise ValueError("Candles series is empty.")
    if not forecast_candles:
        raise ValueError("Forecast candles series is empty.")

    history_close = [candle["close"] for candle in candles]
    history_volume = [candle.get("volume", 0.0) for candle in candles]

    if labels and len(labels) == len(candles):
        parsed = pd.to_datetime(labels, errors="coerce")
        x_history = list(parsed) if not parsed.isna().any() else list(
            pd.date_range(start="2000-01-01", periods=len(candles), freq="D"))
    else:
        x_history = list(pd.date_range(start="2000-01-01", periods=len(candles), freq="D"))

    x_forecast = _forecast_dates(x_history[-1], len(forecast_candles), interval)
    fig = go.Figure()

    if chart_type_history == "candlestick":
        fig.add_trace(go.Candlestick(
            x=x_history,
            open=[c["open"] for c in candles],
            high=[c["high"] for c in candles],
            low=[c["low"] for c in candles],
            close=[c["close"] for c in candles],
            name="History"
        ))
    elif chart_type_history == "line":
        fig.add_trace(go.Scatter(
            x=x_history,
            y=history_close,
            mode="lines+markers",
            name="History",
            line=dict(width=2),
            marker=dict(size=5)
        ))
    else:
        raise ValueError(f"Неподдерживаемый chart_type_history='{chart_type_history}'.")

    if chart_type_forecast == "candlestick":
        fig.add_trace(go.Candlestick(
            x=x_forecast,
            open=[c["open"] for c in forecast_candles],
            high=[c["high"] for c in forecast_candles],
            low=[c["low"] for c in forecast_candles],
            close=[c["close"] for c in forecast_candles],
            name="Forecast"
        ))
    elif chart_type_forecast == "line":
        fig.add_trace(go.Scatter(
            x=x_forecast,
            y=[c["close"] for c in forecast_candles],
            mode="lines+markers",
            name="Forecast Close",
            line=dict(width=2, dash="dash"),
            marker=dict(size=6)
        ))
    else:
        raise ValueError(f"Неподдерживаемый chart_type_forecast='{chart_type_forecast}'.")

    if model_input_start_date:
        marker_dt = pd.to_datetime(model_input_start_date, errors="coerce")
        if not pd.isna(marker_dt):
            y_min = min([c["low"] for c in candles] + [c["low"] for c in forecast_candles])
            y_max = max([c["high"] for c in candles] + [c["high"] for c in forecast_candles])
            fig.add_trace(go.Scatter(
                x=[marker_dt, marker_dt],
                y=[y_min, y_max],
                mode="lines",
                name="Model Input Start",
                line=dict(width=2, dash="dot", color="red"),
                hovertemplate="Начало окна данных, используемого моделью<extra></extra>"
            ))

    if any(v > 0 for v in history_volume):
        volume_colors = [
            "rgba(38, 166, 154, 0.45)" if candle["close"] >= candle["open"] else "rgba(239, 83, 80, 0.45)"
            for candle in candles
        ]

        fig.add_trace(go.Bar(
            x=x_history,
            y=history_volume,
            name="Volume",
            marker=dict(color=volume_colors),
            yaxis="y2"
        ))

    for name, values in indicators.items():
        fig.add_trace(go.Scatter(
            x=x_history,
            y=values,
            mode="lines",
            name=name,
            line=dict(width=1)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        hovermode="x unified",
        legend_title_text="Series",
        margin=dict(l=30, r=20, t=50, b=30),
        autosize=True,
        bargap=0.05,
        dragmode="pan",
        yaxis=dict(domain=[0.13, 1.0], fixedrange=False),
        yaxis2=dict(domain=[0.0, 0.1], title="Volume", showgrid=False, fixedrange=False)
    )

    fig.update_xaxes(
        showgrid=True,
        zeroline=False,
        rangeslider_visible=False,
        automargin=True,
        type="date",
        tickformatstops=[
            dict(dtickrange=[None, 172800000], value="%H:%M"),
            dict(dtickrange=[172800000, 604800000], value="%d %b"),
            dict(dtickrange=[604800000, 2419200000], value="W%V (%d %b)"),
            dict(dtickrange=[2419200000, None], value="%b %Y")
        ],
        hoverformat="%d %b %Y",
        ticklabelmode="period",
        fixedrange=False,
        rangebreaks=[
            dict(bounds=["sat", "mon"])
        ]
    )
    fig.update_yaxes(showgrid=True, zeroline=False, automargin=True, fixedrange=True)

    html_content = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={
            "responsive": True,
            "scrollZoom": True,
            "displaylogo": False
        }
    )

    return f"""
    <html>
        <head>
            <style>
                body, html {{ margin: 0; padding: 0; height: 100%; overflow: hidden; background: #111; }}
                #chart {{ height: 100vh; width: 100vw; }}
            </style>
        </head>
        <body>
            <div id="chart">{html_content}</div>
        </body>
    </html>
    """