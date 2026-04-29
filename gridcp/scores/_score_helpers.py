"""Shared helpers for score implementations."""

import numba as nb
import numpy as np
from gridcp.typing import ArrayLike


def as_obs(x: ArrayLike, n_features: int) -> np.ndarray:
    """Normalize one observation to a 1-D ``float64`` vector.

    Parameters
    ----------
    x : ArrayLike
        Observation input. Converted to ``float64`` and flattened.
    n_features : int
        Required vector length.

    Returns
    -------
    np.ndarray
        Flattened observation with shape ``(n_features,)`` and
        dtype ``float64``.

    Raises
    ------
    ValueError
        If the flattened input size differs from ``n_features``.
    """
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if x_arr.size != n_features:
        raise ValueError(
            f"Expected observation of size {n_features}, got {x_arr.size}."
        )
    return x_arr


def inv_sqrtm_pd(X: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Compute an inverse square root for a symmetric PSD/PD matrix.

    Eigenvalues are clipped below ``eps`` before inversion to avoid numerical
    blowups at near-singular boundaries.

    Parameters
    ----------
    X : np.ndarray
        Input square matrix.
    eps : float, default=1e-15
        Minimum eigenvalue used in clipping.

    Returns
    -------
    np.ndarray
        Matrix approximation to ``X^{-1/2}``.
    """
    eigvals, eigvecs = np.linalg.eigh(X)
    inv_sqrt_eigvals = 1.0 / np.sqrt(np.clip(eigvals, eps, None))
    return (eigvecs * inv_sqrt_eigvals) @ eigvecs.T


@nb.njit(cache=True)
def inv_sqrtm_pd_nb(X: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Numba-compiled inverse square root with eigenvalue clipping.

    Behavior matches :func:`inv_sqrtm_pd`: eigenvalues below ``eps`` are
    clipped before inversion.
    """
    eigvals, eigvecs = np.linalg.eigh(X)
    q = eigvals.shape[0]
    inv_sqrt_eigvals = np.empty(q, dtype=np.float64)
    for i in range(q):
        v = eigvals[i]
        if v < eps:
            v = eps
        inv_sqrt_eigvals[i] = 1.0 / np.sqrt(v)
    return (eigvecs * inv_sqrt_eigvals) @ eigvecs.T
