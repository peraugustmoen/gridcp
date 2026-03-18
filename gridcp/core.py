"""Core functions for the online changepoint detector."""

import numpy as np
import numba as nb
from numba.typed import List
import math
from numba.core.registry import CPUDispatcher
from .utils import v2_numba, LOG2


@nb.njit(fastmath=False, cache=True)
def get_grid(t):
    r"""
    Construct the grid $G^{(t)}$ for time step $t$ (non-recursive).

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


@nb.njit(fastmath=False, cache=True)
def update_grid_numba(old_grid, t):
    r"""
    Update the grid $G^{(t)}$ from $G^{(t-1)}$ in shifted representation.

    Mutates `old_grid` in-place by removing at most one element and appending
    the negative placeholder $-t$, which encodes a new candidate grid point.

    Parameters
    ----------
    old_grid : numba.typed.List[int64]
        Current grid representation (Numba typed list of ``int64``) to be
        updated in-place.
    t : int
        Current time step (after adding a new observation). Must satisfy
        $t \ge 1$.

    Returns
    -------
    removed_index : int
        Index of the removed element in `old_grid` *before* removal, or ``-1``
        if no element was removed.
    removed_g : int
        Value of the removed grid element, or ``-1`` if no element was removed.

    Raises
    ------
    ValueError
        If ``t < 1``.

    Notes
    -----
    - For $t \in \{2, 3\}$, the first element of the grid is always removed.
    - For $t > 3$, a removal index is computed using ``v2_numba(t-1)``
      (the exponent of the largest power of 2 dividing $t-1$).
    - After any removal, the value $-t$ is appended to `old_grid`.
    - The updated list will contain the elements of $G^{(t)} - t - 1$,
      ordered from smallest to largest.
    """
    if t < 1:
        raise ValueError("t must be >= 1")

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


def is_numba_func(f: object) -> bool:
    """
    Check whether a given object is a Numba-compiled function.

    Parameters
    ----------
    f : object
        Object to test.

    Returns
    -------
    bool
        ``True`` if `f` is an instance of ``numba.core.registry.CPUDispatcher``
        (i.e., decorated with ``@nb.njit`` or ``@nb.jit``), ``False`` otherwise.
    """
    return isinstance(f, CPUDispatcher)


@nb.njit
def init_state_numba(v):
    """
    Initialize Numba-typed containers for the detector state.

    Creates the empty typed lists needed by Numba for grid locations and
    cumulative sums when running the detector in compiled mode.

    Parameters
    ----------
    v : tuple of int
        Shape of the transformed data $h(y_i)$, as returned by
        ``h(y).shape`` (e.g., ``(3,)`` for a 3-dimensional feature map).

    Returns
    -------
    grid_list : numba.typed.List[int64]
        Empty Numba typed list to hold grid locations.
    sum_pre_list : numba.typed.List[np.ndarray]
        Empty Numba typed list of ``float64`` arrays of shape `v`,
        used to store cumulative sums of $h(y_i)$.
    """
    grid_list = List.empty_list(nb.int64)
    sum_pre_list = List.empty_list(np.zeros(v, dtype=np.float64))
    return grid_list, sum_pre_list


def init_state(
    p: int,
    h: callable,
    f: callable,
    penalty: callable,
    penalty_constant: float,
    auxiliary_data: object = None,
) -> dict:
    r"""
    Initialize the online changepoint detector state dictionary.

    Sets up all configuration parameters and allocates the internal data
    structures, either in Numba mode or pure Python/NumPy mode.

    Parameters
    ----------
    p : int
        Dimension of the observed data. Must be a positive integer.
    h : callable
        Feature map applied to each observation. Must accept a 1D NumPy
        array of length `p` and return a NumPy array of fixed shape.
    f : callable
        Test statistic function with signature
        ``f(sum_pre, sum_post, changepoint_loc, t) -> float``.
    penalty : callable
        Penalty function with signature ``penalty(changepoint_loc, t, p) -> float``.
    penalty_constant : float
        Threshold on the normalized statistic ``f(...) / penalty(...)``
        for triggering an alarm. Must be non-negative.
    auxiliary_data : object, optional
        Additional user-defined data, stored in the state and passed
        through unchanged.

    Returns
    -------
    state : dict
        Dictionary holding the detector state with the following keys:

        - ``"t"`` (int): Current time index (number of processed observations).
        - ``"p"`` (int): Dimension of the data.
        - ``"v"`` (tuple of int): Shape of ``h(y)``.
        - ``"use_numba"`` (bool): Whether Numba-compiled functions are used.
        - ``"h"``, ``"f"``, ``"penalty"`` (callable): User-specified functions.
        - ``"penalty_constant"`` (float): Alarm threshold.
        - ``"auxiliary_data"`` (object): Additional stored data.
        - ``"grid_list"`` (numba.typed.List[int64]): Grid.
        - ``"sum_pre_list"`` (list or numba.typed.List[np.ndarray]):
          Cumulative sums of $h(y_i)$ for each grid point.
        - ``"sum"`` (np.ndarray): Running totalcumulative sum of $h(y_i)$.
        - ``"alarm"`` (bool): Whether an alarm is currently active.
        - ``"maxx"`` (float): Maximum value of the normalized statistic so far.
        - ``"maxpos"`` (int): Time index at which ``"maxx"`` occurred.

    Raises
    ------
    TypeError
        If `p` is not an integer, or if `h`, `f`, or `penalty` are not
        callable, or if `penalty_constant` is not a real number.
    ValueError
        If `p < 1` or `penalty_constant < 0`.
    ValueError
        If `h`, `f`, `penalty` are a mix of Numba-compiled and regular
        Python functions (all three must be the same kind).

    Notes
    -----
    Numba mode is only used when `h`, `f`, and `penalty` are all Numba-compiled
    functions (``CPUDispatcher``). Otherwise, the state is initialized for pure Python.
    """
    if not isinstance(p, (int, np.integer)):
        raise TypeError(f"p must be an integer, got {type(p).__name__}")
    if p < 1:
        raise ValueError(f"p must be a positive integer, got {p}")
    if not callable(h):
        raise TypeError(f"h must be callable, got {type(h).__name__}")
    if not callable(f):
        raise TypeError(f"f must be callable, got {type(f).__name__}")
    if not callable(penalty):
        raise TypeError(f"penalty must be callable, got {type(penalty).__name__}")
    if not isinstance(penalty_constant, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"penalty_constant must be a number, got {type(penalty_constant).__name__}"
        )
    if penalty_constant < 0:
        raise ValueError(
            f"penalty_constant must be non-negative, got {penalty_constant}"
        )

    numba_flags = [is_numba_func(h), is_numba_func(f), is_numba_func(penalty)]
    use_numba = all(numba_flags)

    dummy_y = np.zeros(p, dtype=np.float64)
    h_y = h(dummy_y)
    v = h_y.shape

    state = {
        "t": 0,
        "num_samples_processed_total": 0,
        "p": p,
        "v": v,
        "use_numba": use_numba,
        "h": h,
        "f": f,
        "penalty": penalty,
        "penalty_constant": float(penalty_constant),
        "auxiliary_data": auxiliary_data,
        "alarm": False,
        "maxx": 0.0,
        "maxpos": -1,
        "sum": np.zeros(v, dtype=np.float64),
    }

    if use_numba:
        grid_list, sum_pre_list = init_state_numba(v)
        state["grid_list"] = grid_list
        state["sum_pre_list"] = sum_pre_list
    else:
        state["grid_list"] = List.empty_list(nb.int64)
        state["sum_pre_list"] = []

    return state


@nb.njit(fastmath=False, cache=False)
def update_data_grid_numba(x_new, old_sums, old_S, removed, h):
    r"""
    Update cumulative sums corresponding to the grid points.

    Maintains `old_sums` (one cumulative sum of $h(y_i)$ per grid point)
    and the overall cumulative sum `old_S`. Removes the entry corresponding
    to a removed grid point (if any), appends the current cumulative sum as
    a new entry, and updates the overall sum with the new observation.

    Parameters
    ----------
    x_new : np.ndarray
        New observation at time $t$ (1D array, already converted via
        ``np.atleast_1d``).
    old_sums : numba.typed.List[np.ndarray]
        Cumulative sums of $h(y_i)$ for each grid point (Numba typed list
        of arrays), mutated in-place.
    old_S : np.ndarray
        Overall cumulative sum $S_{t-1} = \sum_{i=1}^{t-1} h(y_i)$.
    removed : int
        Index of the removed grid point (as returned by
        `update_grid_numba`), or ``-1`` if no point was removed.
    h : callable
        Numba-compiled feature map applied to `x_new`.

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
    r"""
    Update cumulative sums corresponding to the grid points.

    Pure Python/NumPy analogue of `update_data_grid_numba`. Maintains
    `old_sums` (one cumulative sum of $h(y_i)$ per grid point) and the
    overall cumulative sum `old_S`. Removes the entry corresponding to a
    removed grid point (if any), appends the current cumulative sum as a
    new entry, and updates the overall sum with the new observation.

    Parameters
    ----------
    x_new : np.ndarray
        New observation at time $t$ (1D array).
    old_sums : list of np.ndarray
        Cumulative sums of $h(y_i)$ for each grid point (Python list of
        arrays), mutated in-place.
    old_S : np.ndarray
        Overall cumulative sum $S_{t-1} = \sum_{i=1}^{t-1} h(y_i)$.
    removed : int
        Index of the removed grid point (as returned by
        `update_grid_numba`), or ``-1`` if no point was removed.
    h : callable
        Feature map applied to `x_new`.

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
    r"""
    Perform a single online update step in Numba-compiled mode.

    1. Updates the grid via `update_grid_numba`.
    2. Updates cumulative sums via `update_data_grid_numba`.
    3. Evaluates ``f`` and ``penalty`` over all current grid points to
       update `maxx`/`maxpos` and set `alarm` if the normalized statistic
       exceeds `penalty_constant`.

    Parameters
    ----------
    x_new : np.ndarray
        New observation at time $t$ (1D array).
    p : int
        Dimension of the observed data.
    t : int
        Current time index (already incremented).
    grid_list : numba.typed.List[int64]
        Grid representation (negative offsets) at time $t-1$, mutated
        in-place.
    sum_pre_list : numba.typed.List[np.ndarray]
        Cumulative sums of $h(y_i)$ for each grid point at time $t-1$,
        mutated in-place.
    S : np.ndarray
        Overall cumulative sum $S_{t-1} = \sum_{i=1}^{t-1} h(y_i)$.
    maxx : float
        Current maximum normalized test statistic.
    maxpos : int
        Time index where `maxx` was attained.
    alarm : bool
        Current alarm flag.
    h : callable
        Numba-compiled feature map.
    f : callable
        Numba-compiled test statistic function
        ``f(sum_pre, sum_post, changepoint_location, t) -> float``.
    penalty : callable
        Numba-compiled penalty function ``penalty(changepoint_location, t, p) -> float``.
    penalty_constant : float
        Threshold for triggering an alarm.

    Returns
    -------
    grid_list : numba.typed.List[int64]
        Updated grid list.
    sum_pre_list : numba.typed.List[np.ndarray]
        Updated cumulative sums for each grid point.
    S_new : np.ndarray
        Updated cumulative sum $S_t = S_{t-1} + h(x_t)$.
    maxx : float
        Updated maximum normalized test statistic.
    maxpos : int
        Updated time index at which `maxx` is attained.
    alarm : bool
        Updated alarm flag; ``True`` if an alarm is raised in this step.

    Notes
    -----
    For each grid index $j$, the corresponding grid element is
    $g = \text{grid\_list}[j] + t + 1$. A grid element $g$ corresponds to a changepoint
    location changepoint_location = $t-g$. An alarm is triggered when
    ``f(...) / penalty(...)`` exceeds ``penalty_constant``.
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


def update_python(x_new: "float | np.ndarray", state: dict) -> dict:
    """
    Perform a single online update step in pure Python/NumPy mode.

    Non-Numba analogue of `update_numba`. Updates the grid, cumulative
    sums, and detection statistics in-place within `state`.

    Parameters
    ----------
    x_new : float or np.ndarray
        New observation at the current time step.
    state : dict
        Detector state dictionary as produced by `init_state`, mutated
        in-place.

    Returns
    -------
    state : dict
        The same `state` dictionary (returned for convenience).
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


def update_func(x_new: "float | np.ndarray", state: dict) -> None:
    """
    High-level update entry point for the online changepoint detector update.

    Increments the time index and dispatches to either the Numba-compiled
    or pure Python update based on ``state["use_numba"]``.

    Parameters
    ----------
    x_new : float or np.ndarray
        New observation at the current time step. Scalars are converted
        to 1D arrays via ``np.atleast_1d`` before the Numba path.
    state : dict
        Detector state dictionary as returned by `init_state`, updated
        in-place.

    Returns
    -------
    None
        The `state` dictionary is mutated in-place. Updated fields include
        ``"t"``, ``"sum"``, ``"maxx"``, ``"maxpos"``, and ``"alarm"``.
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
