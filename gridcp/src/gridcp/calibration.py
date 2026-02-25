import numpy as np
import numba as nb
from numba.typed import List
import math
from time import perf_counter
import matplotlib.pyplot as plt
from numba.core.registry import CPUDispatcher

from .core import init_state, update_func, is_numba_func, update_numba


def draw_samples(
    K,
    N,
    p,
    seed,
    pre_change_dist,
    pre_change_args=(),
    pre_change_kwargs={},
    post_change_dist=None,
    post_change_args=(),
    post_change_kwargs={},
    changepoint_loc=None,
):
    """
    Draw K samples of streams of length N with dimension p from a user-specified pre- and post-change
    distributions, with a single changepoint at changepoint_loc, return them as an array of shape (K, N, p).
    If no post-change distribution is provided, or if changepoint_loc = None, then all samples are
    drawn from pre-change distribution

    Parameters
    ----------
    K : int
        Number of samples
    N : int
        Stream length
    p : int
        Dimension of data
    seed : int
        Seed for reproducibility. Only compatible with random_func that uses numpy's random generator internally.
    pre_change_dist : callable
        Function that, when called as pre_change_dist(*pre_change_args, **pre_change_kwargs),
        returns either:
          - a scalar, or
          - a 1-D array of length p.
    pre_change_args, pre_change_kwargs :
        Passed through to pre_change_dist.
    post_change_dist : callable
        Function that, when called as post_change_dist(*post_change_args, **post_change_kwargs),
        returns either:
          - a scalar, or
          - a 1-D array of length p.
    pre_change_args, pre_change_kwargs :
        Passed through to post_change_dist.
    changepoint_loc : int
        First index to follow post-change distribution

    Returns
    -------
    X : np.ndarray
        Array of shape (K, N, p) containing the samples.
    """
    # check if changepoint_loc = None or less than 1
    if changepoint_loc is None:
        changepoint_loc = N
    elif changepoint_loc < 1:
        ## Error!
        pass
    elif post_change_dist is None:
        # error!
        pass

    np.random.seed(seed)
    X = np.empty((K, N, p), dtype=np.float64)

    for k in range(K):
        for i in range(changepoint_loc):
            x = pre_change_dist(*pre_change_args, **pre_change_kwargs)
            x = np.asarray(x, dtype=np.float64)

            if x.ndim == 0:
                # scalar -> broadcast to length p
                X[k, i, :] = x
            elif x.ndim == 1 and x.shape[0] == p:
                X[k, i, :] = x
            else:
                raise ValueError(
                    f"Sampler output has wrong shape {x.shape}, expected scalar or (p,)."
                )
        if N > changepoint_loc:
            for i in range(changepoint_loc, N):
                x = post_change_dist(*post_change_args, **post_change_kwargs)
                x = np.asarray(x, dtype=np.float64)

                if x.ndim == 0:
                    # scalar -> broadcast to length p
                    X[k, i, :] = x
                elif x.ndim == 1 and x.shape[0] == p:
                    X[k, i, :] = x
                else:
                    raise ValueError(
                        f"Sampler output has wrong shape {x.shape}, expected scalar or (p,)."
                    )

    return X


@nb.njit(fastmath=False, cache=True)
def mc_max_statistics_numba_driver(X, p, v, penalty_constant, h, f, penalty):
    """
    Numba driver for Monte Carlo max statistic computation.

    Parameters
    ----------
    X : np.ndarray, shape (K, N, p), float64
        Pre-generated null data.
    p : int
        Dimension of data.
    v : tuple(int, ...)
        Shape of h(x) output (same as state["v"]).
    penalty_constant : float
        As in your online algorithm.
    h, f, penalty : numba compiled callables.

    Returns
    -------
    max_values : np.ndarray, shape (K,)
        Max statistic for each MC sample.
    """
    K, N, _ = X.shape
    max_values = np.empty(K, dtype=np.float64)

    for k in range(K):
        # Initialize state pieces for this stream
        grid_list = List.empty_list(nb.int64)
        sum_pre_list = List.empty_list(np.zeros(v, dtype=np.float64))
        S = np.zeros(v, dtype=np.float64)
        maxx = 0.0
        maxpos = -1
        alarm = False

        # Process stream
        for t in range(N):
            x_new = X[k, t, :]
            grid_list, sum_pre_list, S, maxx, maxpos, alarm = update_numba(
                x_new,
                p,
                t + 1,  # time starts at 1 in your update
                grid_list,
                sum_pre_list,
                S,
                maxx,
                maxpos,
                alarm,
                h,
                f,
                penalty,
                penalty_constant,
            )

        max_values[k] = maxx

    return max_values


def mc_max_statistics_python(
    X,
    p,
    h,
    f,
    penalty,
    penalty_constant=0.0,
    auxiliary_data=None,
):
    """
    Pure-Python MC loop, given pre-generated data X.

    Parameters
    ----------
    X : np.ndarray, shape (K, N, p)
        Pre-generated null data.
    p : int
        Dimension of data.
    h, f, penalty : callables
    penalty_constant : float, optional
    auxiliary_data : any, optional

    Returns
    -------
    max_values : np.ndarray, shape (K,)
    """
    K, N, _ = X.shape
    max_values = np.empty(K, dtype=np.float64)

    for k in range(K):
        # Init state for this run (Python path because we call this only when use_numba=False)
        state = init_state(
            p=p,
            h=h,
            f=f,
            penalty=penalty,
            penalty_constant=penalty_constant,
            auxiliary_data=auxiliary_data,
        )

        for t in range(N):
            x_new = X[k, t, :]
            update_func(x_new, state)  # will go through update_python

        max_values[k] = state["maxx"]

    return max_values


def mc_max_statistics(
    N,
    K,
    p,
    h,
    f,
    penalty,
    null_dist,
    null_dist_args=(),
    null_dist_kwargs={},
    seed=42,
    penalty_constant=0.0,
    auxiliary_data=None,
):
    """
    Wrapper to compute MC max statistics, choosing Python or Numba path.

    Parameters
    ----------
    N : int
        Length of each data stream.
    K : int
        Number of Monte Carlo samples.
    p : int
        Dimension of the data x_t.
    h, f, penalty : callables
        As in your online algorithm.
    null_dist : callable
        Random generator, called as:
            null_dist(*null_dist_args, size=(K, N, p), **null_dist_kwargs)
    penalty_constant : float, optional
    null_dist_args : tuple, optional
    null_dist_kwargs : dict, optional
    auxiliary_data : any, optional

    Returns
    -------
    max_values : np.ndarray, shape (K,)
        Max statistic per MC sample.
    """
    if null_dist_kwargs is None:
        null_dist_kwargs = {}
    if null_dist_args is None:
        null_dist_args = ()

    # Probe h to determine v (shape of h(x))
    dummy_y = np.zeros(p, dtype=np.float64)
    h_y = h(dummy_y)
    v = h_y.shape

    use_numba = is_numba_func(h) and is_numba_func(f) and is_numba_func(penalty)

    # Generate all null data once
    X = draw_samples(
        K,
        N,
        p,
        seed,
        null_dist,
        pre_change_args=null_dist_args,
        pre_change_kwargs=null_dist_kwargs,
    )

    if not use_numba:
        # Pure Python path: pass X to the Python driver
        return mc_max_statistics_python(
            X=X,
            p=p,
            h=h,
            f=f,
            penalty=penalty,
            penalty_constant=penalty_constant,
            auxiliary_data=auxiliary_data,
        )
    else:
        # Numba path: pass X to the Numba driver (no Python calls in the inner loop)
        return mc_max_statistics_numba_driver(
            X=X,
            p=p,
            v=v,
            penalty_constant=penalty_constant,
            h=h,
            f=f,
            penalty=penalty,
        )
