from dataclasses import dataclass

import numpy as np

from investment_lab.constants import TRADING_DAYS_PER_YEAR
from investment_lab.util import check_is_true


@dataclass(frozen=True)
class HestonParams:
    """Heston parameters under a simple Euler discretization."""

    kappa: float
    theta: float
    xi: float
    rho: float
    mu: float = 0.0


class HestonStateSpaceModel:
    """One-factor latent variance state-space model using Heston dynamics."""

    def __init__(self, params: HestonParams, dt: float = 1.0 / TRADING_DAYS_PER_YEAR) -> None:
        """Validate and store Heston parameters and time step."""
        check_is_true(dt > 0, "dt must be > 0")
        check_is_true(params.kappa > 0, "kappa must be > 0")
        check_is_true(params.theta > 0, "theta must be > 0")
        check_is_true(params.xi > 0, "xi must be > 0")
        check_is_true(-1.0 < params.rho < 1.0, "rho must be in (-1, 1)")
        self.params = params
        self.dt = dt

    def transition_mean(self, v_prev: float) -> float:
        """E[v_t | v_{t-1}] under Euler discretization."""
        p = self.params
        return max(v_prev + p.kappa * (p.theta - v_prev) * self.dt, 1e-12)

    def transition_var(self, v_prev: float) -> float:
        """Var[v_t | v_{t-1}] under Euler discretization."""
        p = self.params
        return max((p.xi**2) * max(v_prev, 1e-12) * self.dt, 1e-12)

    def observe_mean(self, v_t: float) -> float:
        """E[r_t | v_t] for log-return increment."""
        p = self.params
        return (p.mu - 0.5 * max(v_t, 1e-12)) * self.dt

    def observe_var(self, v_t: float, measurement_var: float = 0.0) -> float:
        """Var[r_t | v_t], with optional additive observation noise."""
        return max(max(v_t, 1e-12) * self.dt + measurement_var, 1e-12)

    def correlated_return_shock(self, state_shock: float, orthogonal_shock: float) -> float:
        """Build return shock correlated with the variance shock."""
        rho = self.params.rho
        return rho * state_shock + np.sqrt(max(1.0 - rho**2, 0.0)) * orthogonal_shock

    def observe_with_correlated_shocks(self, v_t: float, state_shock: float, orthogonal_shock: float) -> float:
        """Observe one return using shocks with Corr(dW1, dW2) = rho."""
        correlated_shock = self.correlated_return_shock(state_shock, orthogonal_shock)
        return self.observe_mean(v_t) + np.sqrt(max(v_t, 1e-12) * self.dt) * correlated_shock

    def transition(self, v_prev: float, shock: float) -> float:
        """Simulate one latent-variance step."""
        return max(self.transition_mean(v_prev) + np.sqrt(self.transition_var(v_prev)) * shock, 1e-12)

    def observe(self, v_t: float, shock: float, measurement_var: float = 0.0) -> float:
        """Simulate one log-return observation."""
        return self.observe_mean(v_t) + np.sqrt(self.observe_var(v_t, measurement_var=measurement_var)) * shock
