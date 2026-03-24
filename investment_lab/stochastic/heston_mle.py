from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from investment_lab.stochastic.heston_ssm import HestonParams, HestonStateSpaceModel
from investment_lab.stochastic.ukf import ScalarUnscentedKalmanFilter
from investment_lab.util import check_is_true


@dataclass
class HestonFitResult:
    """Container for one Heston calibration result."""

    params: HestonParams
    success: bool
    fun: float


def _to_params(x: np.ndarray) -> HestonParams:
    """Convert an optimization vector into `HestonParams`."""
    return HestonParams(kappa=x[0], theta=x[1], xi=x[2], rho=x[3], mu=x[4])


def fit_heston_params_rolling(
    returns: np.ndarray,
    initial_guess: tuple[float, float, float, float, float] = (2.0, 0.04, 0.4, -0.5, 0.0),
    bounds: tuple[tuple[float, float], ...] = ((1e-3, 20.0), (1e-6, 2.0), (1e-4, 5.0), (-0.99, 0.99), (-1.0, 1.0)),
    init_state: float = 0.04,
    init_var: float = 0.01,
    measurement_var: float = 1e-8,
    min_obs: int = 30,
    ukf: ScalarUnscentedKalmanFilter | None = None,
) -> HestonFitResult:
    """Fit Heston parameters by maximizing UKF log-likelihood on one window."""

    r = np.asarray(returns, dtype=float)
    check_is_true(min_obs > 1, "min_obs must be > 1")
    check_is_true(len(r) >= min_obs, f"returns must contain at least {min_obs} observations")
    check_is_true(init_state > 0, "init_state must be > 0")
    check_is_true(init_var > 0, "init_var must be > 0")
    check_is_true(measurement_var >= 0, "measurement_var must be >= 0")
    kf = ukf or ScalarUnscentedKalmanFilter()

    def objective(x: np.ndarray) -> float:
        """Return the negative UKF log-likelihood for candidate parameters."""
        params = _to_params(x)
        model = HestonStateSpaceModel(params)
        res = kf.filter(r, model=model, init_state=init_state, init_var=init_var, measurement_var=measurement_var)
        return -res.loglikelihood

    opt = minimize(objective, x0=np.asarray(initial_guess, dtype=float), method="L-BFGS-B", bounds=bounds)
    return HestonFitResult(params=_to_params(opt.x), success=bool(opt.success), fun=float(opt.fun))
