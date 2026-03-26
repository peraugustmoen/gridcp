"""Shared helpers for score implementations."""

import numpy as np
from gridcp.typing import ArrayLike


def as_obs(x: ArrayLike, n_features: int) -> np.ndarray:
    """Normalize one observation to a 1D float64 vector."""
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if x_arr.size != n_features:
        raise ValueError(
            f"Expected observation of size {n_features}, got {x_arr.size}."
        )
    return x_arr


def inv_sqrtm_pd(X: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Compute inverse square root of a positive-definite matrix."""
    eigvals, eigvecs = np.linalg.eigh(X)
    inv_sqrt_eigvals = 1.0 / np.sqrt(np.clip(eigvals, eps, None))
    return (eigvecs * inv_sqrt_eigvals) @ eigvecs.T
