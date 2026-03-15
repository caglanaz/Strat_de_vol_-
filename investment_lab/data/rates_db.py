from datetime import datetime
from pathlib import Path

import pandas as pd

from investment_lab.data.data_loader import DataLoader

def _data_root() -> Path:
    """Return a stable data root regardless of current working directory."""
    module_root = Path(__file__).resolve().parents[2]
    return module_root / "data"


class USRatesLoader(DataLoader):
    @classmethod
    def _get_path(cls) -> str:
        return str(_data_root() / "par-yield-curve-rates-2020-2023.csv")

    @classmethod
    def _get_valid_date_range(cls) -> tuple[datetime, datetime]:
        return (datetime(2020, 1, 2), datetime(2023, 12, 30))

    @classmethod
    def _process_loaded_data(cls, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"], format="mixed")
        df = df.ffill().set_index("date").sort_index() / 100
        return df.reset_index()
