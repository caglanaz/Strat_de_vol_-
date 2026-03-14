from dataclasses import dataclass

import numpy as np

from investment_lab.constants import TRADING_DAYS_PER_YEAR


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

    def transition(self, v_prev: float, shock: float) -> float:
        """Simulate one latent-variance step."""
        return max(self.transition_mean(v_prev) + np.sqrt(self.transition_var(v_prev)) * shock, 1e-12)

    def observe(self, v_t: float, shock: float, measurement_var: float = 0.0) -> float:
        """Simulate one log-return observation."""
        return self.observe_mean(v_t) + np.sqrt(self.observe_var(v_t, measurement_var=measurement_var)) * shock
