import pandas as pd
import numpy as np


def compute_linear_weights(
    spreads: pd.Series,
    w_min: float = 0.0,
    w_max: float = 1.0,
    spread_min: float | None = None,
    spread_max: float | None = None,
) -> pd.Series:
    spread = spreads.values.astype(float)
    spread_min = spread_min if spread_min is not None else float(np.percentile(spread, 5))
    spread_max = spread_max if spread_max is not None else float(np.percentile(spread, 95))
    spread_clipped = np.clip(spread, spread_min, spread_max)
    weights = w_min + (w_max - w_min) * (spread_clipped - spread_min) / (spread_max - spread_min)
    return pd.Series(weights, index=spreads.index, name="dynamic_weight")


def rescale_positions_with_signal(
    df_positions: pd.DataFrame,
    dynamic_weights: pd.Series,
    date_col: str = "date",
    weight_col: str = "weight",
) -> pd.DataFrame:
    df_out = df_positions.copy()
    all_dates = df_out[date_col].sort_values().unique()
    dw = dynamic_weights.reindex(all_dates).ffill().fillna(1.0)
    date_to_weight = dw.to_dict()
    df_out[weight_col] = df_out[weight_col] * df_out[date_col].map(date_to_weight).fillna(1.0)
    return df_out