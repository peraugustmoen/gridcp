import numpy as np
import numba as nb
from numba.typed import List
import math
from numba.core.registry import CPUDispatcher
from .utils import v2_numba, LOG2


@nb.njit(fastmath=False, cache=True)
def get_grid(t):
    """
    Compute the grid Gt for a given time step t. Non-recursive version.

    Parameters
    ----------
    t : int

    Returns
    -------
    np.ndarray (int64)
        The grid Gt as an integer NumPy array.
    """
    # Handle edge case t <= 1 (consistent with the commented R behavior for t==1)
    if t <= 1:
        out = np.empty(1, dtype=np.int64)
        out[0] = 1
        return out

    # Compute upper bounds using base-2 logs via math.log(...)
    # upper1 <- floor(log((t - 1) / 3, base = 2)) + 1
    # upper2 <- floor(log(t - 1, base = 2)) - 1
    upper1 = math.floor(math.log((t - 1) / 3.0) / LOG2) + 1
    upper2 = math.floor(math.log(t - 1) / LOG2) - 1

    # Preallocate output array
    length = 1 + (upper1 if upper1 > 0 else 0) + (upper2 if upper2 > 0 else 0)
    Gt = np.empty(length, dtype=np.int64)

    # Gt[1] <- 1  (R is 1-based; Python is 0-based)
    Gt[0] = 1
    counter = 1

    if upper1 >= 1:
        for j in range(1, upper1 + 1):
            # gL <- 2^j + (t - 1) %% 2^(j - 1)
            gL = (1 << j) + ((t - 1) % (1 << (j - 1)))
            Gt[counter] = gL
            counter += 1

            if j <= upper2:
                # Gt[counter] <- gL + 2^(j - 1)
                Gt[counter] = gL + (1 << (j - 1))
                counter += 1

    return Gt


@nb.njit(fastmath=False, cache=True)
def update_grid_numba(old_grid, t):
    """
    Update the grid old_grid in place to reflect the addition of a new data point at time t.
    Parameters
    ----------
    old_grid : List[int]
        The current grid to be updated.
    t : int
        The current (i.e., new) time step.
    """
    removed = -1
    removed_g = -1

    if t == 2 or t == 3:
        removed_g = old_grid[0]
        old_grid.pop(0)
        removed = 0
    elif t > 3:
        j = v2_numba(t - 1) + 1
        if j > 0:
            ind = 2 * j

            if ind < len(old_grid):
                removed = len(old_grid) - ind - 1
                removed_g = old_grid[removed]
                old_grid.pop(removed)

    old_grid.append(-t)

    return removed, removed_g


def is_numba_func(f):
    return isinstance(f, CPUDispatcher)


@nb.njit
def init_state_numba(v):
    grid_list = List.empty_list(nb.int64)
    sum_pre_list = List.empty_list(np.zeros(v, dtype=np.float64))
    return grid_list, sum_pre_list


def init_state(p, h, f, penalty, penalty_constant, auxiliary_data=None):
    # Determine v by probing h on a dummy input
    dummy_y = np.zeros(p, dtype=np.float64)
    h_y = h(dummy_y)
    v = h_y.shape

    ### now warm up the Numba functions to ensure they are compiled
    # if is_numba_func(f):
    #    # Call with dummy data to trigger compilation
    #    dummy_sum_pre_j = np.zeros(v, dtype=np.float64)
    #    dummy_sum_post_j = np.zeros(v, dtype=np.float64)
    #    dummy_g = 1
    #    dummy_t = 2
    #    f(dummy_sum_pre_j, dummy_sum_post_j, dummy_g, dummy_t)
    # if is_numba_func(penalty):
    #   dummy_g = 1
    #    dummy_t = 2
    #    dummy_p = p
    #    penalty(dummy_g, dummy_t, dummy_p)

    use_numba = is_numba_func(h) and is_numba_func(f) and is_numba_func(penalty)

    state = {
        "t": 0,
        "num_samples_processed_total": 0,
        "p": p,
        "v": v,
        "use_numba": use_numba,
        "h": h,
        "f": f,
        "penalty": penalty,
        "penalty_constant": penalty_constant,
        "auxiliary_data": auxiliary_data,
    }

    if use_numba:
        # Create Numba state
        grid_list, sum_pre_list = init_state_numba(v)
        state["grid_list"] = grid_list
        state["sum_pre_list"] = sum_pre_list
        state["sum"] = np.zeros(v, dtype=np.float64)  # Initialize sum as a NumPy array
        state["alarm"] = False
        state["maxx"] = 0.0
        state["maxpos"] = -1

    else:
        # Create Python/NumPy equivalents (lists/arrays)
        state["t"] = 0
        state["grid_list"] = List.empty_list(nb.int64)
        state["sum_pre_list"] = []
        state["sum"] = np.zeros(v, dtype=np.float64)  # Initialize sum as a NumPy array
        state["alarm"] = False
        state["maxx"] = 0.0
        state["maxpos"] = -1

    return state


@nb.njit(fastmath=False, cache=True)
def update_data_grid_numba(x_new, old_sums, old_S, removed, h):
    if removed >= 0:
        old_sums.pop(removed)
    old_sums.append(old_S)
    tmp = h(x_new)
    S = old_S + tmp
    return S


def update_data_grid(x_new, old_sums, old_S, removed, h):
    if removed >= 0:
        old_sums.pop(removed)
    old_sums.append(old_S)
    tmp = h(x_new)
    S = old_S + tmp
    return S


@nb.njit(fastmath=False, cache=True)
def update_numba(
    x_new,
    p,
    t,
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
):
    removed, removed_g = update_grid_numba(grid_list, t)
    S_new = update_data_grid_numba(x_new, sum_pre_list, S, removed, h)

    if t > 1:
        for j in range(len(grid_list) - 1, -1, -1):
            g = grid_list[j] + t + 1
            val = f(sum_pre_list[j], S_new - sum_pre_list[j], g, t)
            pen = penalty(g, t, p)
            cc = val / pen
            if cc > maxx:
                maxx = cc
                maxpos = grid_list[j] + t + 1
            if cc > penalty_constant:
                alarm = True

    return grid_list, sum_pre_list, S_new, maxx, maxpos, alarm


def update_python(x_new, state):
    removed, removed_g = update_grid_numba(state["grid_list"], state["t"])
    state["sum"] = update_data_grid(
        x_new, state["sum_pre_list"], state["sum"], removed, state["h"]
    )

    if state["t"] > 1:
        for j in range(len(state["grid_list"]) - 1, -1, -1):
            g = state["grid_list"][j] + state["t"] + 1
            val = state["f"](
                state["sum_pre_list"][j],
                state["sum"] - state["sum_pre_list"][j],
                g,
                state["t"],
            )
            pen = state["penalty"](g, state["t"], state["p"])
            cc = val / pen
            if cc > state["maxx"]:
                state["maxx"] = cc
                state["maxpos"] = state["grid_list"][j] + state["t"] + 1
            if cc > state["penalty_constant"]:
                state["alarm"] = True

    return state


def update_func(x_new, state):
    state["t"] = state["t"] + 1
    if state["use_numba"]:
        grid_list, sum_pre_list, S, maxx, maxpos, alarm = update_numba(
            np.atleast_1d(x_new),
            state["p"],
            state["t"],
            state["grid_list"],
            state["sum_pre_list"],
            state["sum"],
            state["maxx"],
            state["maxpos"],
            state["alarm"],
            state["h"],
            state["f"],
            state["penalty"],
            state["penalty_constant"],
        )
        state["grid_list"] = grid_list
        state["sum_pre_list"] = sum_pre_list
        state["sum"] = S
        state["maxx"] = maxx
        state["maxpos"] = maxpos
        state["alarm"] = alarm
    else:
        state = update_python(x_new, state)
