from dataclasses import dataclass

import numpy as np

from investment_lab.stochastic.heston_ssm import HestonStateSpaceModel
from investment_lab.util import check_is_true


@dataclass
class UKFResult:
    """Outputs produced by one scalar UKF filtering pass."""

    filtered_state: np.ndarray
    filtered_var: np.ndarray
    pred_obs_mean: np.ndarray
    pred_obs_var: np.ndarray
    loglikelihood: float


class ScalarUnscentedKalmanFilter:
    """Scalar UKF for latent variance filtering."""

    def __init__(self, alpha: float = 0.1, beta: float = 2.0, kappa: float = 0.0) -> None:
        """Store UKF scaling parameters and validate their ranges."""
        check_is_true(alpha > 0, "alpha must be > 0")
        check_is_true(beta >= 0, "beta must be >= 0")
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    @staticmethod
    def _stable_cholesky(cov: np.ndarray, scale: float) -> np.ndarray:
        """Return chol(scale * cov) with small diagonal jitter if needed."""
        n = int(cov.shape[0])
        cov_sym = 0.5 * (cov + cov.T)
        eye = np.eye(n)
        jitter = 1e-12

        for _ in range(6):
            try:
                return np.linalg.cholesky(scale * cov_sym)
            except np.linalg.LinAlgError:
                cov_sym = cov_sym + eye * jitter
                jitter *= 10.0

        check_is_true(False, "Sigma-point covariance is not positive definite")
        return np.zeros((n, n), dtype=float)

    def _sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build sigma points and UKF weights around `mean` from covariance `cov`."""
        n = int(mean.shape[0])
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        scale = n + lambda_
        check_is_true(scale > 0, "Invalid UKF scaling: n + lambda must be > 0")

        sqrt_cov = self._stable_cholesky(cov, scale)

        # For each state dimension, add and subtract one scaled covariance direction.
        offsets = sqrt_cov.T
        points = np.empty((2 * n + 1, n), dtype=float)
        points[0] = mean
        points[1 : n + 1] = mean + offsets
        points[n + 1 :] = mean - offsets

        wm = np.full(2 * n + 1, 1.0 / (2.0 * scale), dtype=float)
        wc = np.full(2 * n + 1, 1.0 / (2.0 * scale), dtype=float)
        wm[0] = lambda_ / scale
        wc[0] = lambda_ / scale + (1.0 - self.alpha**2 + self.beta)
        return points, wm, wc

    def filter(
        self,
        observations: np.ndarray,
        model: HestonStateSpaceModel,
        init_state: float,
        init_var: float,
        measurement_var: float = 0.0,
    ) -> UKFResult:
        """Run UKF filtering and return states, prediction moments, and log-likelihood.

        Uses an augmented-state transform with process and observation shocks to preserve
        Heston correlation Corr(dW1, dW2) = rho during filtering.
        """
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
            # Augmented state: [v_{t-1}, z_state, z_obs_orth]
            aug_mean = np.array([m_prev, 0.0, 0.0], dtype=float)
            aug_cov = np.diag([p_prev, 1.0, 1.0])
            aug_pts, wm, wc = self._sigma_points(aug_mean, aug_cov)

            x_pred_pts = np.zeros(len(aug_pts), dtype=float)
            y_pts = np.zeros(len(aug_pts), dtype=float)
            for i, pt in enumerate(aug_pts):
                v_prev_i, z_state_i, z_obs_orth_i = float(pt[0]), float(pt[1]), float(pt[2])
                v_t_i = model.transition(v_prev_i, z_state_i)
                y_t_i = model.observe_with_correlated_shocks(v_t_i, z_state_i, z_obs_orth_i)
                x_pred_pts[i] = v_t_i
                y_pts[i] = y_t_i

            m_pred = float(np.sum(wm * x_pred_pts))
            p_pred = float(np.sum(wc * (x_pred_pts - m_pred) ** 2))
            p_pred = max(p_pred, 1e-12)

            y_mean = float(np.sum(wm * y_pts))
            y_var = float(np.sum(wc * (y_pts - y_mean) ** 2))
            if measurement_var > 0.0:
                y_var += measurement_var
            y_var = max(y_var, 1e-12)

            cov_xy = float(np.sum(wc * (x_pred_pts - m_pred) * (y_pts - y_mean)))
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
