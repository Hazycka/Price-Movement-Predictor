"""
HTTP-роутеры. Каждый файл — отдельный APIRouter, подключаемый в main.py.

  health.py    — /health
  forecast.py  — /forecast, /forecast/chart, /forecast/chart/test
  backtest.py  — /backtest, /backtest/sweep, /backtest/runs, /backtest/dashboard
  market.py    — /market/tickers, /market/candles, /market/candles/chart
  _common.py   — общие хелперы (преобразование DataUnavailableError → HTTP, и т.п.)
"""
