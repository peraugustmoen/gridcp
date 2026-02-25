import numba as nb
import math


LOG2 = math.log(2.0)


@nb.njit(fastmath=True, cache=True)
def fastlog(x):
    return math.log(x)


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
