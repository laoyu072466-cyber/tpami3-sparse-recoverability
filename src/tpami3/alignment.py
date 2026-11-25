"""Standardize matched phase rows and fit the original 32-dimensional map."""

from dataclasses import dataclass

import numpy as np


DIMENSIONS = 32
RIDGE = 0.01


def _matrix(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != DIMENSIONS or not len(result) or not np.isfinite(result).all():
        raise ValueError("Expected a nonempty, finite matrix with 32 columns.")
    return result


def _stats(values):
    return values.mean(axis=0), np.where(np.ptp(values, axis=0) == 0, 1.0, values.std(axis=0))


@dataclass(frozen=True)
class PhaseMap:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    W_P: np.ndarray

    def __post_init__(self):
        for key, shape in (
            ("x_mean", (DIMENSIONS,)), ("x_scale", (DIMENSIONS,)),
            ("y_mean", (DIMENSIONS,)), ("y_scale", (DIMENSIONS,)),
            ("W_P", (DIMENSIONS, DIMENSIONS)),
        ):
            value = np.asarray(getattr(self, key), dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all() or (key.endswith("scale") and np.any(value <= 0)):
                raise ValueError(f"Invalid phase-map component {key}.")
            object.__setattr__(self, key, value)

    def transform(self, x):
        normalized = (_matrix(x) - self.x_mean) / self.x_scale
        return self.y_mean + (normalized @ self.W_P) * self.y_scale


def fit_p32(x, y):
    """Fit on matched representation rows from training data supplied by the caller."""
    x, y = _matrix(x), _matrix(y)
    if x.shape != y.shape:
        raise ValueError("The source and target phase representations must align by row.")
    x_mean, x_scale = _stats(x)
    y_mean, y_scale = _stats(y)
    xn, yn = (x - x_mean) / x_scale, (y - y_mean) / y_scale
    w = np.linalg.solve(xn.T @ xn / len(x) + RIDGE * np.eye(DIMENSIONS), xn.T @ yn / len(x))
    return PhaseMap(x_mean, x_scale, y_mean, y_scale, w)
