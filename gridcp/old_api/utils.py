import numba as nb
import math
import numpy as np

LOG2 = math.log(2.0)


@nb.njit(fastmath=True, cache=True)
def numba_log(x):
    return math.log(x)


# Backwards-compatible alias for old notebooks.
fastlog = numba_log


@nb.njit(fastmath=False, cache=True)
def v2_numba(r: int) -> int:
    """
    Compute the exponent of the largest power of 2 that divides r.
    """
    if r == 0:
        return -1
    if r < 0:
        r = -r
    c = 0
    while (r & 1) == 0:
        c += 1
        r >>= 1
    return c


@nb.njit(cache=True)
def logdet_spd(A):
    # A is assumed SPD; we compute log(det(A)) via Cholesky
    L = np.linalg.cholesky(A)
    # log(det(A)) = 2 * sum(log(diag(L)))
    logdet = 0.0
    for i in range(L.shape[0]):
        logdet += np.log(L[i, i])
    return 2.0 * logdet


@nb.njit(cache=True)
def inv_sqrtm_pd(X, eps=1e-15):
    # Eigen-decomposition: X = Q Λ Q^T
    eigvals, eigvecs = np.linalg.eigh(X)

    # Inverse square root of eigenvalues
    inv_sqrt_eigvals = 1.0 / np.sqrt(np.clip(eigvals, eps, None))

    # Reconstruct: X^{-1/2} = Q Λ^{-1/2} Q^T
    return (eigvecs * inv_sqrt_eigvals) @ eigvecs.T


@nb.njit(fastmath=False, cache=True)
def CUSUM(S, cumsum, g, t):
    # cumsum = sum up to (t-g)
    # S = cumulative sum up to t

    res = math.sqrt(1.0 * g / (t * (t - g))) * cumsum
    res = res - math.sqrt(1.0 * (t - g) / t / g) * (S - cumsum)
    return res * res
