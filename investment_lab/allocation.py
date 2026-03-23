import pandas as pd
import numpy as np


def compute_linear_weights(
    spread: pd.Series,
    w_min: float = 0.0,
    w_max: float = 1.0,
    window : int = 60,
) -> pd.Series:
    s = spread.astype(float)
 
    s_min = s.expanding(min_periods=window).quantile(0.05)
    s_max = s.expanding(min_periods=window).quantile(0.95)
 
    denom = (s_max - s_min).replace(0, np.nan)
    weights = w_min + (w_max - w_min) * (s.clip(s_min, s_max) - s_min) / denom
    weights = weights.fillna(w_min)
 
    return weights.rename("dynamic_weight")

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