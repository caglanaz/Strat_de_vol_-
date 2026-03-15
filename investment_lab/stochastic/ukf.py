from dataclasses import dataclass

import numpy as np

from investment_lab.stochastic.heston_ssm import HestonStateSpaceModel
from investment_lab.util import check_is_true


@dataclass
class UKFResult:
    filtered_state: np.ndarray
    filtered_var: np.ndarray
    pred_obs_mean: np.ndarray
    pred_obs_var: np.ndarray
    loglikelihood: float


class ScalarUnscentedKalmanFilter:
    """Lightweight scalar UKF for latent variance filtering."""

    def __init__(self, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0) -> None:
        check_is_true(alpha > 0, "alpha must be > 0")
        check_is_true(beta >= 0, "beta must be >= 0")
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def _sigma_points(self, mean: float, var: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = 1
        lam = self.alpha**2 * (n + self.kappa) - n
        c = n + lam
        check_is_true(c > 0, "Invalid UKF scaling: n + lambda must be > 0")
        sqrt_term = np.sqrt(max(c * var, 1e-16))

        points = np.array([mean, mean + sqrt_term, mean - sqrt_term])
        wm = np.array([lam / c, 1.0 / (2.0 * c), 1.0 / (2.0 * c)])
        wc = np.array([lam / c + (1.0 - self.alpha**2 + self.beta), 1.0 / (2.0 * c), 1.0 / (2.0 * c)])
        return points, wm, wc

    def filter(
        self,
        observations: np.ndarray,
        model: HestonStateSpaceModel,
        init_state: float,
        init_var: float,
        measurement_var: float = 0.0,
    ) -> UKFResult:
        obs = np.asarray(observations, dtype=float)
        n = len(obs)

        filtered_state = np.zeros(n)
        filtered_var = np.zeros(n)
        pred_obs_mean = np.zeros(n)
        pred_obs_var = np.zeros(n)
        loglik = 0.0

        m_prev = max(init_state, 1e-12)
        p_prev = max(init_var, 1e-12)

        for t in range(n):
            x_pts, wm, wc = self._sigma_points(m_prev, p_prev)

            x_pred_pts = np.array([model.transition_mean(x) for x in x_pts])
            m_pred = float(np.sum(wm * x_pred_pts))
            p_pred = float(np.sum(wc * (x_pred_pts - m_pred) ** 2))
            p_pred += model.transition_var(m_prev)
            p_pred = max(p_pred, 1e-12)

            x2_pts, wm2, wc2 = self._sigma_points(m_pred, p_pred)
            y_pts = np.array([model.observe_mean(x) for x in x2_pts])
            y_mean = float(np.sum(wm2 * y_pts))

            y_var = float(np.sum(wc2 * (y_pts - y_mean) ** 2))
            y_var += model.observe_var(m_pred, measurement_var=measurement_var)
            y_var = max(y_var, 1e-12)

            cov_xy = float(np.sum(wc2 * (x2_pts - m_pred) * (y_pts - y_mean)))
            k_gain = cov_xy / y_var

            innov = obs[t] - y_mean
            m_post = max(m_pred + k_gain * innov, 1e-12)
            p_post = max(p_pred - (k_gain**2) * y_var, 1e-12)

            filtered_state[t] = m_post
            filtered_var[t] = p_post
            pred_obs_mean[t] = y_mean
            pred_obs_var[t] = y_var

            loglik += -0.5 * (np.log(2.0 * np.pi * y_var) + (innov**2) / y_var)

            m_prev, p_prev = m_post, p_post

        return UKFResult(
            filtered_state=filtered_state,
            filtered_var=filtered_var,
            pred_obs_mean=pred_obs_mean,
            pred_obs_var=pred_obs_var,
            loglikelihood=float(loglik),
        )
