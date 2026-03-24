from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from investment_lab.data.data_loader import DataLoader

def _data_root() -> Path:
    """Return a stable data root regardless of current working directory."""
    module_root = Path(__file__).resolve().parents[2]
    return module_root / "data"


class OptionLoader(DataLoader):
    @classmethod
    def _get_path(cls) -> str:
        """Return the default parquet path for the full options dataset."""
        return str(_data_root() / "optiondb_2016_2023.parquet")

    @classmethod
    def _get_valid_date_range(cls) -> tuple[datetime, datetime]:
        """Return the supported date range for this dataset."""
        return (datetime(2016, 1, 2), datetime(2023, 12, 30))

    @classmethod
    def _process_loaded_data(
        cls, df: pd.DataFrame, *, ticker: str | Sequence[str], **kwargs
    ) -> pd.DataFrame:
        """Filter by ticker, clean volume, and apply expiration payoff rules."""
        if isinstance(ticker, str):
            ticker = [ticker]
        else:
            ticker = [t for t in ticker]
        df = df[df["ticker"].isin(ticker)]
        df["volume"] = df["volume"].fillna(0)
        return cls._compute_final_payoff(df)

    @classmethod
    def _add_extra_fields(cls, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Add derived features used in option analysis."""
        df["day_to_expiration"] = (df["expiration"] - df["date"]).dt.days
        df["moneyness"] = df["strike"] / df["spot"]
        return df

    @staticmethod
    def _compute_final_payoff(df_option: pd.DataFrame) -> pd.DataFrame:
        """Replace quotes with intrinsic value on expiration dates."""
        expiring_filter = df_option["date"] == df_option["expiration"].copy()
        expiring_calls_filter = expiring_filter & (df_option["call_put"] == "C")
        expiring_puts_filter = expiring_filter & (df_option["call_put"] == "P")

        call_payoff = (df_option["spot"] - df_option["strike"]).clip(lower=0)
        put_payoff = (df_option["strike"] - df_option["spot"]).clip(lower=0)

        for c in ("mid", "bid", "ask"):
            df_option[c] = np.where(
                expiring_calls_filter,
                call_payoff,
                np.where(expiring_puts_filter, put_payoff, df_option[c]),
            )
        return df_option


class SPYOptionLoader(OptionLoader):
    @classmethod
    def _get_path(cls) -> str:
        """Return the parquet path for SPY options."""
        return str(_data_root() / "spy_2020_2022.parquet")

    @classmethod
    def _get_valid_date_range(cls) -> tuple[datetime, datetime]:
        """Return the supported SPY options date range."""
        return (datetime(2020, 1, 2), datetime(2022, 12, 30))


class AAPLOptionLoader(OptionLoader):
    @classmethod
    def _get_path(cls) -> str:
        """Return the parquet path for AAPL options."""
        return str(_data_root() / "aapl_2016_2023.parquet")

    @classmethod
    def _get_valid_date_range(cls) -> tuple[datetime, datetime]:
        """Return the supported AAPL options date range."""
        return (datetime(2016, 1, 2), datetime(2023, 12, 31))


def extract_spot_from_options(df_options: pd.DataFrame) -> pd.DataFrame:
    """Extract a daily spot price series from options rows."""
    return (
        df_options[["date", "spot"]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
