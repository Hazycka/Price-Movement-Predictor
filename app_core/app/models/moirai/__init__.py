"""
Salesforce Moirai-2 foundation model integration.

Установка:
    pip install uni2ts

Поддерживаемые варианты (model_id):
    Salesforce/moirai-2.0-R-small  (default)
    Salesforce/moirai-2.0-R-base
    Salesforce/moirai-2.0-R-large

Использование через API:
    POST /forecast
    {
      "model_name": "moirai_2",
      "model_options": {
        "size": "small",
        "num_samples": 100
      },
      ...
    }
"""
from .moirai_model import MoiraiForecastModel

__all__ = ["MoiraiForecastModel"]
