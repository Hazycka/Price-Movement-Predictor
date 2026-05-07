from __future__ import annotations

import os
import pandas as pd

from ..base import MarketDataProvider, MarketDataRequest
from ....schemas import CsvProviderOptions, ProviderOptions


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

        c = options
        data = pd.read_csv(normalized_path)
        required = [c.date_column, c.open_column, c.high_column, c.low_column, c.close_column]
        for col in required:
            if col not in data.columns:
                raise ValueError(f"В CSV отсутствует колонка '{col}'.")

        vol_exists = c.volume_column and c.volume_column in data.columns
        data = data[required + ([c.volume_column] if vol_exists else [])].dropna()
        data[c.date_column] = pd.to_datetime(data[c.date_column], errors="coerce")
        for col in [c.open_column, c.high_column, c.low_column, c.close_column]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        if vol_exists:
            data[c.volume_column] = pd.to_numeric(data[c.volume_column], errors="coerce")
        data = data.dropna()

        if data.empty:
            raise ValueError("CSV не содержит валидных данных после очистки.")

        dates = [str(v) for v in data[c.date_column].tolist()]
        candles = [
            {
                "open":   float(row[c.open_column]),
                "high":   float(row[c.high_column]),
                "low":    float(row[c.low_column]),
                "close":  float(row[c.close_column]),
                "volume": float(row[c.volume_column]) if vol_exists else 0.0,
            }
            for _, row in data.iterrows()
        ]
        return dates, candles

    def load_ohlc_range(
            self,
            provider_options: ProviderOptions,
            interval: str,
            from_dt: str,
            to_dt: str,
    ) -> tuple[list[str], list[dict[str, float]]]:
        """
        Заглушка: CSV провайдер не поддерживает загрузку по диапазону дат.
        CSV содержит статичный файл — нет смысла в дозагрузке.
        """
        raise NotImplementedError(
            "CsvMarketDataProvider.load_ohlc_range не поддерживается. "
            "CSV является статичным источником данных — используйте load_ohlc."
        )