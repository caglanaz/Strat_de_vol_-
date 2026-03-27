"""
Plotting utilities for the volatility timing strategy notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from investment_lab.backtest import StrategyBacktester
from investment_lab.metrics.volatility import rolling_realized_volatility


def _clip_for_display(series: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    """Winsorize a series for visualization only (does not affect backtests)."""
    s = series.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return s
    lo = float(s.quantile(lower_q))
    hi = float(s.quantile(upper_q))
    return s.clip(lower=lo, upper=hi)


def plot_results(
    backtester_static: StrategyBacktester,
    backtester_dynamic: StrategyBacktester,
    spread: pd.Series,
    dynamic_weights: pd.Series,
    sigma_filtered: pd.Series,
    log_returns: pd.Series,
    rv_window: int = 21,
    display_clip_quantiles: tuple[float, float] = (0.01, 0.99),
) -> None:
    """Plot NAV, spread + weights, and volatility comparison.

    Parameters:
        backtester_static  : fitted static StrategyBacktester
        backtester_dynamic : fitted dynamic StrategyBacktester
        spread             : IV-RV spread series s_t
        dynamic_weights    : allocation weight series w_t
        sigma_filtered     : UKF filtered vol series
        log_returns        : daily log-return series (for rolling RV)
        rv_window          : rolling window for realised vol (default 21 days)
        display_clip_quantiles : winsorization quantiles for spread/sigma display
    """
    lower_q, upper_q = display_clip_quantiles
    spread_plot = _clip_for_display(spread, lower_q, upper_q)
    sigma_plot = _clip_for_display(sigma_filtered, lower_q, upper_q)

    fig, axes = plt.subplots(3, 1, figsize=(13, 12))

    # --- NAV ---
    ax = axes[0]
    ax.plot(backtester_static.nav.index,  backtester_static.nav["NAV"],  label="Static")
    ax.plot(backtester_dynamic.nav.index, backtester_dynamic.nav["NAV"], label="Dynamic", linestyle="--")
    ax.set_title("NAV — Static vs Dynamic delta-hedged carry")
    ax.set_ylabel("NAV")
    ax.legend()

    # --- Spread + weights (twin axis) ---
    ax = axes[1]
    ax.plot(spread_plot.index, spread_plot.values, label="s_t = IV - sigma_hat", color="steelblue")
    ax.axhline(0, color="red", lw=0.8, linestyle="--")
    ax.fill_between(spread_plot.index, spread_plot.values, 0,
                    where=(spread_plot.values > 0), alpha=0.25, color="green")
    ax.fill_between(spread_plot.index, spread_plot.values, 0,
                    where=(spread_plot.values < 0), alpha=0.25, color="red")
    ax.set_title("IV-RV Spread and Dynamic Weight (display-clipped)")
    ax.set_ylabel("Spread (annualised vol)")
    ax2 = ax.twinx()
    ax2.plot(dynamic_weights.index, dynamic_weights.values,
             color="purple", lw=1, alpha=0.6, label="w_t")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Weight")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # --- Volatilities ---
    ax = axes[2]
    rv = rolling_realized_volatility(log_returns, window=rv_window, volatility_type="std")
    ax.plot(sigma_plot.index, sigma_plot.values, label="UKF sigma_hat", color="navy")
    ax.plot(rv.index,             rv.values,             label=f"{rv_window}d RV", color="grey", linestyle=":")
    ax.set_title("Filtered vs Realised Volatility (display-clipped sigma)")
    ax.set_ylabel("Annualised vol")
    ax.legend()

    plt.tight_layout()
    plt.show()
