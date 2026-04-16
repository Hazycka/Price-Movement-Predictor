from __future__ import annotations

import os
import pandas as pd

from ..base import MarketDataProvider, MarketDataRequest
from ....schemas import CsvProviderOptions


class CsvMarketDataProvider(MarketDataProvider):
    def load_ohlc(self, request: MarketDataRequest) -> tuple[list[str], list[dict[str, float]]]:
        options = request.provider_options
        if not isinstance(options, CsvProviderOptions):
            raise TypeError(f"Ожидались настройки Csv, но получено {type(options).__name__}")
        
        if not options.csv_path:
            raise ValueError("Для CSV provider требуется csv_path.")

        normalized_path = options.csv_path.strip().strip('"').strip("'")
        if not os.path.isabs(normalized_path):
            normalized_path = os.path.join("data", normalized_path)

        if not os.path.exists(normalized_path):
            raise ValueError(f"CSV файл не найден: {normalized_path}")

        csv_date_column = options.date_column
        csv_open_column = options.open_column
        csv_high_column = options.high_column
        csv_low_column = options.low_column
        csv_close_column = options.close_column
        csv_volume_column = options.volume_column

        data = pd.read_csv(normalized_path)

        required_columns = [
            csv_date_column,
            csv_open_column,
            csv_high_column,
            csv_low_column,
            csv_close_column
        ]
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"В CSV отсутствует колонка '{column}'.")

        volume_exists = csv_volume_column and csv_volume_column in data.columns
        data = data[required_columns + ([csv_volume_column] if volume_exists else [])].dropna()

        data[csv_date_column] = pd.to_datetime(data[csv_date_column], errors="coerce")
        for column in [
            csv_open_column,
            csv_high_column,
            csv_low_column,
            csv_close_column
        ]:
            data[column] = pd.to_numeric(data[column], errors="coerce")

        if volume_exists:
            data[csv_volume_column] = pd.to_numeric(data[csv_volume_column], errors="coerce")

        data = data.dropna()

        dates = [str(v) for v in data[csv_date_column].tolist()]
        candles = [{
            "open": float(row[csv_open_column]),
            "high": float(row[csv_high_column]),
            "low": float(row[csv_low_column]),
            "close": float(row[csv_close_column]),
            "volume": float(row[csv_volume_column]) if volume_exists else 0.0
        } for _, row in data.iterrows()]

        return dates, candles