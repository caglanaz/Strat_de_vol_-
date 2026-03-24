import numpy as np
import pandas as pd


def mse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Return the mean squared error between true and predicted values."""
    return float(np.mean((y_true - y_pred) ** 2))


def sse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Return the sum of squared errors between true and predicted values."""
    return np.sum((y_true - y_pred) ** 2)
