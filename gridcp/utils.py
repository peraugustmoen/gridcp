"""Utility functions."""

import math
import numba as nb
import numpy as np

LOG2 = math.log(2.0)


@nb.njit(fastmath=True, cache=True)
def fastlog(x):
    """Compute a fast approximation to log(x).

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        Approximation to log(x).
    """
    return math.log(x)


def v2(r: int) -> int:
    """Compute the exponent of the largest power of 2 that divides r.

    This Python implementation is used in the new API hot path
    (``GridDetector.update``) to avoid Python<->Numba call overhead.

    Parameters
    ----------
    r : int
        Input integer.

    Returns
    -------
    int
        Exponent of the largest power of 2 that divides `r`.
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


@nb.njit(fastmath=False, cache=True)
def get_changeloc_grid(t):
    r"""
    Construct the grid $t - G^{(t)}$ for time step $t$ (non-recursive).

    The grid is ordered from smallest to largest, so the most recent candidate changepoint is at the end of the array.

    Parameters
    ----------
    t : int
        Current time step. Must be a positive integer ($t \ge 1$).

    Returns
    -------
    np.ndarray
        One-dimensional ``int64`` NumPy array containing the grid $t - G^{(t)}$.
        For $t = 1$ this returns ``array([0])``.

    Raises
    ------
    ValueError
        If ``t < 1``.

    Notes
    -----
    This function is not used by the main detection algorithm.
    It is provided for testing and debugging purposes, to verify that the grid
    is being updated correctly by `update_grid_numba`.
    """
    if t < 1:
        raise ValueError("t must be a positive integer (>= 1)")

    Gt = get_G_grid(t)
    # reverse Gt:
    Gt = Gt[::-1]

    return t - Gt


@nb.njit(fastmath=False, cache=True)
def get_G_grid(t):
    r"""
    Construct the grid $G^{(t)}$ for time step $t$ (non-recursive). The grid is ordered from smallest to largest.

    Parameters
    ----------
    t : int
        Current time step. Must be a positive integer ($t \ge 1$).

    Returns
    -------
    np.ndarray
        One-dimensional ``int64`` NumPy array containing the grid $G^{(t)}$.
        For $t = 1$ this returns ``array([1])``.

    Raises
    ------
    ValueError
        If ``t < 1``.

    Notes
    -----
    This function is not used by the main detection algorithm.
    It is provided for testing and debugging purposes, to verify that the grid
    is being updated correctly by `update_grid_numba`.
    """
    if t < 1:
        raise ValueError("t must be a positive integer (>= 1)")

    if t == 1:
        out = np.empty(1, dtype=np.int64)
        out[0] = 1
        return out

    upper1 = math.floor(math.log((t - 1) / 3.0) / LOG2) + 1
    upper2 = math.floor(math.log(t - 1) / LOG2) - 1

    length = 1 + (upper1 if upper1 > 0 else 0) + (upper2 if upper2 > 0 else 0)
    Gt = np.empty(length, dtype=np.int64)

    Gt[0] = 1
    counter = 1

    if upper1 >= 1:
        for j in range(1, upper1 + 1):
            gL = (1 << j) + ((t - 1) % (1 << (j - 1)))
            Gt[counter] = gL
            counter += 1

            if j <= upper2:
                Gt[counter] = gL + (1 << (j - 1))
                counter += 1

    return Gt
