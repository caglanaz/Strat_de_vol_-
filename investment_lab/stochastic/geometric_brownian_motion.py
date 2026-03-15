from investment_lab.stochastic.base import StochasticProcess
import numpy as np
import pandas as pd


class GeometricBrownianMotion(StochasticProcess):
    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 0.2,
        s0: float = 100.0,
        dt: float = 1.0 / 252.0,
    ) -> None:
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.s0 = float(s0)
        self.dt = float(dt)

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        s0: float | None = None,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Simulate GBM price paths with Euler-exact lognormal dynamics."""
        if n_steps <= 0:
            raise ValueError("n_steps must be > 0")
        if n_paths <= 0:
            raise ValueError("n_paths must be > 0")

        s0_val = self.s0 if s0 is None else float(s0)
        rng = np.random.default_rng(random_state)
        z = rng.standard_normal((n_steps, n_paths))

        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * z
        log_returns = drift + diffusion

        paths = np.empty((n_steps + 1, n_paths), dtype=float)
        paths[0, :] = s0_val
        paths[1:, :] = s0_val * np.exp(np.cumsum(log_returns, axis=0))
        return paths

    def calibrate(self, returns: np.ndarray | pd.Series, dt: float | None = None) -> tuple[float, float]:
        """Calibrate mu and sigma from simple returns."""
        r = np.asarray(returns, dtype=float)
        if r.size == 0:
            raise ValueError("returns must contain at least one observation")
        step = self.dt if dt is None else float(dt)
        if step <= 0:
            raise ValueError("dt must be > 0")

        log_r = np.log1p(r)
        sigma_hat = float(log_r.std(ddof=1) / np.sqrt(step))
        mu_hat = float(log_r.mean() / step + 0.5 * sigma_hat**2)

        self.mu = mu_hat
        self.sigma = sigma_hat
        return self.mu, self.sigma
