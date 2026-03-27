from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from investment_lab.metrics.volatility import rolling_realized_volatility
from investment_lab.stochastic.heston_mle import fit_heston_params_rolling
from investment_lab.stochastic.heston_ssm import HestonStateSpaceModel
from investment_lab.stochastic.ukf import ScalarUnscentedKalmanFilter
from investment_lab.util import check_is_true


@dataclass
class VolTimingConfig:
    rv_method: Literal["rolling_std", "heston_ukf"] = "rolling_std"
    rolling_window: int = 126
    iv_col: str = "implied_volatility"
    ret_col: str = "log_return"
    min_obs: int = 90
    measurement_var: float = 1e-8
    ukf_alpha: float = 0.1
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0
    ukf_init_state: float = 0.04
    ukf_init_var: float = 0.01
    base_weight: float = 1.0
    slope: float = 5.0
    min_weight: float = 0.0
    max_weight: float = 2.0


def build_iv_rv_timing_signal(df: pd.DataFrame, config: VolTimingConfig | None = None) -> pd.DataFrame:
    """Estimate realized volatility with rolling window and build timing spread.

    Expected columns: ['date', ret_col, iv_col].
    Returns columns: ['date', 'rv_hat', 'spread', 'timing_weight'].
    """
    cfg = config or VolTimingConfig()
    check_is_true(cfg.rolling_window > 1, "rolling_window must be > 1")
    check_is_true(cfg.min_obs > 1, "min_obs must be > 1")
    check_is_true(cfg.rolling_window >= cfg.min_obs, "rolling_window must be >= min_obs")
    check_is_true(cfg.measurement_var >= 0, "measurement_var must be >= 0")
    check_is_true(cfg.ukf_alpha > 0, "ukf_alpha must be > 0")
    check_is_true(cfg.ukf_beta >= 0, "ukf_beta must be >= 0")
    check_is_true(cfg.ukf_init_state > 0, "ukf_init_state must be > 0")
    check_is_true(cfg.ukf_init_var > 0, "ukf_init_var must be > 0")
    check_is_true(cfg.min_weight <= cfg.max_weight, "min_weight must be <= max_weight")
    check_is_true(
        cfg.rv_method in {"rolling_std", "heston_ukf"},
        f"Unsupported rv_method: {cfg.rv_method}",
    )
    required = {"date", cfg.ret_col, cfg.iv_col}
    missing = required.difference(df.columns)
    check_is_true(not missing, f"Missing columns in input dataframe: {missing}")

    cp = df[["date", cfg.ret_col, cfg.iv_col]].dropna().copy().sort_values("date")
    cp["date"] = pd.to_datetime(cp["date"], errors="coerce")
    cp = cp.dropna(subset=["date"])
    cp = cp.reset_index(drop=True)

    if cfg.rv_method == "rolling_std":
        cp["rv_hat"] = rolling_realized_volatility(
            cp[cfg.ret_col],
            window=cfg.rolling_window,
            volatility_type="std",
        )
        cp.loc[cp.index < cfg.min_obs, "rv_hat"] = np.nan
    elif cfg.rv_method == "heston_ukf":
        rv_hat = np.full(len(cp), np.nan)
        ukf = ScalarUnscentedKalmanFilter(
            alpha=cfg.ukf_alpha,
            beta=cfg.ukf_beta,
            kappa=cfg.ukf_kappa,
        )
        for i in range(cfg.rolling_window, len(cp)):
            window = cp.iloc[i - cfg.rolling_window : i]
            if len(window) < cfg.min_obs:
                continue
            returns_window = window[cfg.ret_col].to_numpy(dtype=float)
            fit = fit_heston_params_rolling(
                returns=returns_window,
                init_state=cfg.ukf_init_state,
                init_var=cfg.ukf_init_var,
                measurement_var=cfg.measurement_var,
                min_obs=cfg.min_obs,
                ukf=ukf,
            )
            model = HestonStateSpaceModel(fit.params)
            filt = ukf.filter(
                observations=returns_window,
                model=model,
                init_state=cfg.ukf_init_state,
                init_var=cfg.ukf_init_var,
                measurement_var=cfg.measurement_var,
            )
            rv_hat[i] = np.sqrt(max(float(filt.filtered_state[-1]), 1e-12))
        cp["rv_hat"] = rv_hat
    cp["spread"] = cp[cfg.iv_col] - cp["rv_hat"]
    cp["timing_weight"] = np.clip(
        cfg.base_weight + cfg.slope * cp["spread"],
        cfg.min_weight,
        cfg.max_weight,
    )
    return cp[["date", "rv_hat", "spread", "timing_weight"]]


def apply_timing_weight_to_positions(df_positions: pd.DataFrame, df_signal: pd.DataFrame) -> pd.DataFrame:
    """Scale strategy weights by the daily timing signal."""
    required = {"date", "weight"}
    missing = required.difference(df_positions.columns)
    check_is_true(not missing, f"Missing columns in positions dataframe: {missing}")

    check_is_true("timing_weight" in df_signal.columns, "Signal dataframe must contain 'timing_weight'.")

    merged = df_positions.merge(df_signal[["date", "timing_weight"]], on="date", how="left")
    merged["timing_weight"] = merged["timing_weight"].fillna(1.0)
    merged["weight"] = merged["weight"] * merged["timing_weight"]
    return merged.drop(columns=["timing_weight"])
