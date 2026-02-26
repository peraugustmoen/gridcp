import numpy as np
import numba as nb
from numba.typed import List
import math
from numba.core.registry import CPUDispatcher
from .utils import v2_numba, LOG2


@nb.njit(fastmath=False, cache=True)
def get_grid(t):
    """
    Construct the grid $G^{(t)}$ for time step $t$ (non-recursive).

    Parameters
    ----------
    t : int
        Current time step (1-based in the original R code, treated consistently here).

    Returns
    -------
    np.ndarray
        One-dimensional `int64` NumPy array containing the grid $G^{(t)}$. When
        $t \\le 1$, this returns an array with a single element `1`.

    Notes
    -----
    This function is not used by the main detection algorithm.
    It is provided for testing and debugging purposes, to verify that the grid
    is being updated correctly by `update_grid_numba`.
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
    Update the grid $G{^{(t)}}$ from $G^{(t-1)}$ according to the rules of the
    compressed grid representation.

    This function mutates the existing grid `old_grid` in-place, removing at most
    one element and appending a new negative placeholder $-t$ that encodes a
    new candidate start point in the compressed grid representation. The returned
    array is $G^{(t)}-t-1$ after the update.

    Parameters
    ----------
    old_grid : numba.typed.List[int64]
        Current grid representation (Numba typed list) to be updated.
    t : int
        Current time step (after adding a new observation).

    Returns
    -------
    removed_index : int
        Index of the removed element in `old_grid` *before* removal, or `-1`
        if no element was removed.
    removed_g : int
        Value of the removed grid element, or `-1` if no element was removed.

    Notes
    -----
    - For $t \\in \\{2, 3\\}$, the first element of the grid is always removed.
    - For $t > 3$, a removal index is computed using $v2\_numba(t-1)$ (the exponent
      of the largest power of 2 dividing $t-1$).
    - After any removal, the value $-t$ is appended to `old_grid`.
    - The updated array will contain the elements of $G^{(t)}-t-1$,
      ordered from smallest to largest.

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
    """
    Check whether a given callable is a Numba-compiled dispatcher.

    Parameters
    ----------
    f : Any
        Object to test, typically a function or callable.

    Returns
    -------
    bool
        `True` if `f` is a `numba.core.registry.CPUDispatcher` (i.e., a Numba
        compiled function), `False` otherwise.
    """
    return isinstance(f, CPUDispatcher)


@nb.njit
def init_state_numba(v):
    """
    Initialize Numba-typed containers for the detector state.

    This sets up the typed lists needed by Numba for the grid locations and
    prefix sums when running the detector in fully compiled mode.

    Parameters
    ----------
    v : int or tuple[int]
        Dimension of the transformed data $h(y_i)$. Often a scalar dimension
        (e.g. `v == 1`) or the shape of a 1D feature vector.

    Returns
    -------
    grid_list : numba.typed.List[int64]
        Empty Numba typed list to hold grid locations.
    sum_pre_list : numba.typed.List[np.ndarray]
        Empty Numba typed list of NumPy arrays, each of shape compatible
        with `np.zeros(v, dtype=np.float64)`, used to store prefix sums.
    """
    grid_list = List.empty_list(nb.int64)
    sum_pre_list = List.empty_list(np.zeros(v, dtype=np.float64))
    return grid_list, sum_pre_list


def init_state(p, h, f, penalty, penalty_constant, auxiliary_data=None):
    """
    Initialize the online changepoint detector state dictionary.

    This sets up all configuration parameters (feature map, test
    statistic, penalty, penalty constant) and allocates the internal data
    structures, either in Numba mode or pure Python/NumPy mode.

    Parameters
    ----------
    p : int
        Dimension of data to be processed.
    h : callable
        Feature map applied to each observation. Must accept a 1D NumPy array
        and return a NumPy array of fixed shape.
    f : callable
        Test statistic function. In Numba mode, it must be a Numba-compiled
        function with signature roughly:
        `f(sum_pre, sum_post, location, t) -> float`.
    penalty : callable
        Penalty function for segment length. In Numba mode, must be a
        Numba-compiled function with signature:
        `penalty(location, t, p) -> float`.
    penalty_constant : float
        Threshold on the normalized statistic `val / pen` for triggering an
        alarm in the detector.
    auxiliary_data : Any, optional
        Additional user-defined data or configuration, stored in the state
        and passed through unchanged.

    Returns
    -------
    state : dict
        A dictionary holding the detector state. Important keys include:

        - `"t"` : int
          Current time index (number of processed observations).
        - `"p"` : int
          Dimension of the data.
        - `"v"` : tuple or int
          Shape/dimension of `h(y)`.
        - `"use_numba"` : bool
          Flag indicating whether Numba-compiled functions will be used.
        - `"h"`, `"f"`, `"penalty"` : callables
          Stored user-specified functions.
        - `"penalty_constant"` : float
          Leading constant for the penalty threshold.
        - `"auxiliary_data"` : Any
          Additional stored data.
        - `"grid_list"` : numba.typed.List[int64]
          Grid locations (Numba typed list in both modes).
        - `"sum_pre_list"` : list or numba.typed.List[np.ndarray]
          Prefix sums for each grid point.
        - `"sum"` : np.ndarray
          Running sum of transformed observations.
        - `"alarm"` : bool
          Flag indicating whether an alarm is currently active.
        - `"maxx"` : float
          Maximum value of the normalized statistic so far.
        - `"maxpos"` : int
          Time index at which `"maxx"` occurred.

    Notes
    -----
    Whether Numba mode is used is determined by checking that `h`, `f`, and
    `penalty` are all Numba-compiled functions (`CPUDispatcher`).
    """
    dummy_y = np.zeros(p, dtype=np.float64)
    h_y = h(dummy_y)
    v = h_y.shape

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


@nb.njit(fastmath=False, cache=False)
def update_data_grid_numba(x_new, old_sums, old_S, removed, h):
    """
    Update running sums for all candidate segments (Numba version).

    This function maintains:

    - `old_sums`: prefix sums for each grid location in the grid list.
    - `old_S`: the overall cumulative sum of $h(x_t)$ up to time $t-1$.

    It removes the prefix sum corresponding to a removed grid point (if any),
    appends the current cumulative sum as the new prefix sum, then updates the
    overall sum with the new transformed observation.

    Parameters
    ----------
    x_new : np.ndarray
        New observation at time $t$, typically a 1D array (already transformed
        to at least 1D before calling).
    old_sums : numba.typed.List[np.ndarray]
        Prefix sums for each grid point (Numba typed list of arrays).
    old_S : np.ndarray
        Previous cumulative sum $S_{t-1}$ of $h(x_i)$.
    removed : int
        Index of the removed grid location (as returned by `update_grid_numba`),
        or `-1` if no location was removed.
    h : callable
        Numba-compiled feature map to apply to `x_new`.

    Returns
    -------
    S : np.ndarray
        Updated cumulative sum $S_t = S_{t-1} + h(x_t)$.
    """
    if removed >= 0:
        old_sums.pop(removed)
    old_sums.append(old_S)
    tmp = h(x_new)
    S = old_S + tmp
    return S


def update_data_grid(x_new, old_sums, old_S, removed, h):
    """
    Update running sums for all candidate segments (pure Python/NumPy version).

    This is the non-Numba analogue of `update_data_grid_numba`. It maintains
    and updates prefix sums for each grid location and the global cumulative sum.

    Parameters
    ----------
    x_new : array_like
        New observation at time $t$; converted to an array inside the feature map.
    old_sums : list of np.ndarray
        Prefix sums for each grid point (Python list of arrays).
    old_S : np.ndarray
        Previous cumulative sum $S_{t-1}$ of $h(x_i)$.
    removed : int
        Index of the removed grid location (as returned by `update_grid_numba`),
        or `-1` if no location was removed.
    h : callable
        Feature map to apply to `x_new`.

    Returns
    -------
    S : np.ndarray
        Updated cumulative sum $S_t = S_{t-1} + h(x_t)$.
    """
    if removed >= 0:
        old_sums.pop(removed)
    old_sums.append(old_S)
    tmp = h(x_new)
    S = old_S + tmp
    return S


@nb.njit(fastmath=False, cache=False)
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
    """
    Perform a single online update step in Numba-compiled mode.

    This function:

    1. Updates the grid representation using `update_grid_numba`.
    2. Updates prefix sums and the global cumulative sum via
       `update_data_grid_numba`.
    3. Evaluates the test statistic `f` and penalty `penalty` over all current
       grid locations to:

       - Update the maximum normalized statistic `maxx` and its location
         `maxpos`.
       - Set the `alarm` flag to `True` if any normalized statistic exceeds
         `penalty_constant`.

    Parameters
    ----------
    x_new : np.ndarray
        New observation at time $t$, passed as a 1D NumPy array.
    p : int
        Dimension of the data (used for penalty calculation).
    t : int
        Current time index (after incrementing).
    grid_list : numba.typed.List[int64]
        Numba-typed list storing the grid representation (negative offsets)
        at time $t-1$
    sum_pre_list : numba.typed.List[np.ndarray]
        Numba-typed list of prefix sums, one per grid location,
        at time $t-1$.
    S : np.ndarray
        Cumulative sum of transformed observations up to time $t-1$.
    maxx : float
        Current maximum normalized test statistic.
    maxpos : int
        Time index where `maxx` was attained.
    alarm : bool
        Current alarm flag.
    h : callable
        Numba-compiled feature map applied in `update_data_grid_numba`.
    f : callable
        Numba-compiled test statistic function `f(sum_pre, sum_post, location, t)`.
    penalty : callable
        Numba-compiled penalty function `penalty(location, t, p)`.
    penalty_constant : float
        Threshold for deciding when to raise an alarm.

    Returns
    -------
    grid_list : numba.typed.List[int64]
        Updated grid list after adding the new observation.
    sum_pre_list : numba.typed.List[np.ndarray]
        Updated prefix sums for each grid location.
    S_new : np.ndarray
        Updated cumulative sum after processing `x_new`.
    maxx : float
        Updated maximum normalized test statistic.
    maxpos : int
        Updated time index at which `maxx` is attained.
    alarm : bool
        Updated alarm flag; `True` if an alarm is raised in this step.

    Notes
    -----
    For each grid index `j`, the each corresponding element $g$ is
    computed as:

    - $g = \\text{grid\\_list}[j] + t + 1$.

    If the ratio `f(...) / penalty(...)`
    exceeds `penalty_constant`, an alarm is triggered.
    """

    removed, removed_g = update_grid_numba(grid_list, t)
    S_new = update_data_grid_numba(x_new, sum_pre_list, S, removed, h)

    if t > 1:
        for j in range(len(grid_list) - 1, -1, -1):
            g = grid_list[j] + t + 1
            val = f(sum_pre_list[j], S_new - sum_pre_list[j], t - g, t)
            pen = penalty(t - g, t, p)
            cc = val / pen
            if cc > maxx:
                maxx = cc
                maxpos = grid_list[j] + t + 1
            if cc > penalty_constant:
                alarm = True

    return grid_list, sum_pre_list, S_new, maxx, maxpos, alarm


def update_python(x_new, state):
    """
    Perform a single online update step in pure Python/NumPy mode.

    This is the non-Numba analogue of `update_numba`. It updates the grid,
    prefix sums, cumulative sum, and detection statistics in-place within
    the `state` dictionary.

    Parameters
    ----------
    x_new : float or array_like
        New observation at the current time step. Typically a scalar or 1D array.
    state : dict
        Detector state dictionary as produced by `init_state`. Must contain
        keys `"t"`, `"grid_list"`, `"sum_pre_list"`, `"sum"`, `"h"`, `"f"`,
        `"penalty"`, `"p"`, `"maxx"`, `"maxpos"`, `"penalty_constant"`,
        and `"alarm"`.

    Returns
    -------
    state : dict
        The same state dictionary, updated in-place and also returned for
        convenience.

    Notes
    -----
    The logic is identical to `update_numba`, but uses Python/NumPy and the
    `update_data_grid` helper instead of Numba-compiled code.
    """

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
                state["t"] - g,
                state["t"],
            )
            pen = state["penalty"](state["t"] - g, state["t"], state["p"])
            cc = val / pen
            if cc > state["maxx"]:
                state["maxx"] = cc
                state["maxpos"] = state["grid_list"][j] + state["t"] + 1
            if cc > state["penalty_constant"]:
                state["alarm"] = True

    return state


def update_func(x_new, state):
    """
    High-level update entry point for the online changepoint detector
    to call update in either Numba or pure Python mode.

    This increments the time index, chooses between the Numba or Python
    implementation based on the `"use_numba"` flag in `state`, and updates
    the detector statistics accordingly.

    Parameters
    ----------
    x_new : float or array_like
        New observation at the current time step. If Numba mode is used, this
        will be converted to at least 1D via `np.atleast_1d` before passing
        into the compiled update.
    state : dict
        Detector state dictionary, as returned by `init_state` and potentially
        modified by previous calls to `update_func`.

    Returns
    -------
    None
        The `state` dictionary is updated in-place. Relevant fields such as
        `"t"`, `"sum"`, `"maxx"`, `"maxpos"`, and `"alarm"` are updated.

    Notes
    -----
    - When `state["use_numba"]` is `True`, `update_numba` is called with
      Numba-compiled functions.
    - Otherwise, `update_python` is called.
    """
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
