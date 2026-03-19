"""Utility functions."""

import math
import numba as nb


@nb.njit(fastmath=True, cache=True)
def fastlog(x):
    """Compute A fast approximation to log(x).

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
