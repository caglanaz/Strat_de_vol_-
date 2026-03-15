import numpy as np
import pandas as pd

from projet.investment_lab.stochastic.base import StochasticProcess
from projet.investment_lab.constants import TRADING_DAYS_PER_YEAR


class GeometricBrownianMotion(StochasticProcess):
    """
    dS_t = mu * S_t * dt + sigma * S_t * dW_t

    Parameters
    ----------
    mu    : annualised drift (risk-neutral: set to risk_free_rate)
    sigma : annualised volatility
    """

    def __init__(self, mu: float = 0.0, sigma: float = 0.20) -> None:
        self.mu = mu
        self.sigma = sigma

    def simulate(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        dt: float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Simulate GBM paths using the exact discretisation scheme.

        Parameters
        ----------
        S0      : initial spot price
        T       : total horizon in years
        n_steps : number of time steps
        n_paths : number of Monte-Carlo paths
        dt      : step size (overrides T / n_steps if provided)
        seed    : random seed for reproducibility

        Returns
        -------
        np.ndarray of shape (n_paths, n_steps + 1)
        """
        rng = np.random.default_rng(seed)
        dt = dt if dt is not None else T / n_steps
        Z = rng.standard_normal((n_paths, n_steps))

        increments = np.exp(
            (self.mu - 0.5 * self.sigma**2) * dt
            + self.sigma * np.sqrt(dt) * Z
        )
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = S0
        paths[:, 1:] = S0 * np.cumprod(increments, axis=1)
        return paths

    def calibrate(self, prices: pd.Series | np.ndarray) -> "GeometricBrownianMotion":
        """
        MLE calibration from a price (or index) series.

        Estimates mu and sigma from log-returns assuming daily frequency
        (TRADING_DAYS_PER_YEAR = 252).

        Parameters
        ----------
        prices : observed price series (chronological order)

        Returns
        -------
        self  (allows chaining)
        """
        prices = np.asarray(prices, dtype=float)
        log_returns = np.diff(np.log(prices))
        mu_daily = log_returns.mean()
        sigma_daily = log_returns.std(ddof=1)

        self.sigma = sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
        self.mu = mu_daily * TRADING_DAYS_PER_YEAR + 0.5 * self.sigma**2
        return self