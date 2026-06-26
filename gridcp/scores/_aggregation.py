"""Shared aggregation and penalty helpers for the built-in chi-squared scores.

This module factors out the three pieces of work that every per-feature
chi-squared score shares:

* :func:`chi2_max_bound` -- the single penalty divisor for all chi-squared
  scores (the Laurent-Massart 2000 tail bound with the leading constants
  absorbed into the detector threshold).
* :func:`aggregate_features` -- a numba reduction of a ``(G, p)`` per-feature
  statistic over the feature axis (max / sum / none).  It performs **no**
  centering and **no** penalty.
* :func:`aggregation_dims` -- a tiny pure-Python lookup returning the ordered
  ``(M, df)`` pairs per output column for a given aggregation.  Each score reads
  the ``df`` from here to do its **own** centering (subtraction happens inside
  the score, never here) and the ``(M, df)`` to call :func:`chi2_max_bound`.
"""

import numba as nb
import numpy as np

# Integer codes for the aggregation mode, passed into the numba reduction.
_MODE_MAX = 0
_MODE_SUM = 1
_MODE_MAX_SUM = 2
_MODE_NONE = 3

# Allowed ``aggregation`` keyword values mapped to their integer mode code.
# ``None`` and the string ``"None"`` are treated identically.
_AGGREGATION_MODE_CODES: dict[object, int] = {
    "max": _MODE_MAX,
    "sum": _MODE_SUM,
    "max-sum": _MODE_MAX_SUM,
    None: _MODE_NONE,
    "None": _MODE_NONE,
}


@nb.njit(cache=True)
def chi2_max_bound(n_terms: int, df: int, n_samples: int) -> float:
    """Return the chi-squared max-penalty divisor ``log(tM) + sqrt(df log(tM))``.

    This is the Laurent-Massart (2000) chi-squared tail bound with the leading
    constants absorbed into the detector threshold.  It is the only penalty
    divisor used by the built-in chi-squared scores.

    Default threshold guidance
    --------------------------
    Because the penalty is kept *bare* (constants absorbed into the threshold),
    a detector threshold of ``c ≈ 2`` corresponds to the Laurent-Massart
    constant and gives a per-test false-alarm bound of ``1/(t*M)``: ``Pr((X - df) / chi2_max_bound >= 2) <= e^{-log(tM)} = 1/(tM)``.
    This is a *per-test* bound, not a uniform union bound over the grid/run; a
    uniform guarantee is obtained by calibrating the threshold.

    Parameters
    ----------
    n_terms : int
        Number ``M`` of maxed terms (the union-bound multiplicity).
    df : int
        Degrees of freedom of each chi-squared term.
    n_samples : int
        Current sample count ``t``.

    Returns
    -------
    float
        ``log(t * M) + sqrt(df * log(t * M))``.
    """
    logg = np.log(n_samples * n_terms)
    return logg + np.sqrt(df * logg)


@nb.njit(cache=True)
def aggregate_features(
    per_feature_stats: np.ndarray,
    mode_code: int,
) -> np.ndarray:
    """Reduce a ``(G, p)`` per-feature statistic over the feature axis.

    The reduction iterates features in ascending order to stay numerically
    faithful to the legacy fused kernels.  It performs no centering and no
    penalty.

    Parameters
    ----------
    per_feature_stats : np.ndarray
        Per-feature statistics, shape ``(G, p)``.
    mode_code : int
        ``0`` max, ``1`` sum, ``2`` max-sum, ``3`` none.

    Returns
    -------
    np.ndarray
        Columns array of shape ``(G, n_cols)`` where ``n_cols`` is ``1`` for
        max/sum, ``2`` for max-sum, and ``p`` for none.
    """
    n_candidates = per_feature_stats.shape[0]
    n_features = per_feature_stats.shape[1]

    if mode_code == _MODE_MAX:
        out = np.empty((n_candidates, 1), dtype=np.float64)
        for i in range(n_candidates):
            row_max = per_feature_stats[i, 0]
            for j in range(1, n_features):
                v = per_feature_stats[i, j]
                if v > row_max:
                    row_max = v
            out[i, 0] = row_max
        return out

    if mode_code == _MODE_SUM:
        out = np.empty((n_candidates, 1), dtype=np.float64)
        for i in range(n_candidates):
            total = 0.0
            for j in range(n_features):
                total += per_feature_stats[i, j]
            out[i, 0] = total
        return out

    if mode_code == _MODE_MAX_SUM:
        out = np.empty((n_candidates, 2), dtype=np.float64)
        for i in range(n_candidates):
            row_max = per_feature_stats[i, 0]
            total = per_feature_stats[i, 0]
            for j in range(1, n_features):
                v = per_feature_stats[i, j]
                total += v
                if v > row_max:
                    row_max = v
            out[i, 0] = row_max
            out[i, 1] = total
        return out

    # _MODE_NONE: one column per feature, in feature order.
    out = np.empty((n_candidates, n_features), dtype=np.float64)
    for i in range(n_candidates):
        for j in range(n_features):
            out[i, j] = per_feature_stats[i, j]
    return out


def aggregation_mode_code(aggregation: object) -> int:
    """Validate ``aggregation`` and return its integer ``mode_code``.

    Parameters
    ----------
    aggregation : {"max", "sum", "max-sum", None, "None"}
        The aggregation keyword value.  ``None`` and ``"None"`` are identical.

    Returns
    -------
    int
        The integer mode code consumed by :func:`aggregate_features`.

    Raises
    ------
    ValueError
        If ``aggregation`` is not one of the allowed values.
    """
    try:
        return _AGGREGATION_MODE_CODES[aggregation]
    except (KeyError, TypeError):
        raise ValueError(
            f"Invalid aggregation {aggregation!r}; expected one of "
            "'max', 'sum', 'max-sum', None, 'None'."
        ) from None


def aggregation_dims(aggregation: object, p: int, d: int) -> list[tuple[int, int]]:
    """Return the ordered per-column ``(M, df)`` pairs for an aggregation.

    This is a pure-Python lookup.  It returns metadata only: the score reads ``df`` to center (subtracting it inside the score) and ``(M, df)`` to call :func:`chi2_max_bound`.

    Parameters
    ----------
    aggregation : {"max", "sum", "max-sum", None, "None"}
        The aggregation keyword value.
    p : int
        Number of features.
    d : int
        Per-coordinate degrees of freedom of the underlying statistic.

    Returns
    -------
    list[tuple[int, int]]
        ``[(M, df), ...]`` -- one entry per output column.
    """
    code = aggregation_mode_code(aggregation)
    if code == _MODE_MAX:
        return [(p, d)]
    if code == _MODE_SUM:
        return [(1, p * d)]
    if code == _MODE_MAX_SUM:
        return [(p, d), (1, p * d)]
    # _MODE_NONE: one (1, d) column per feature.
    return [(1, d)] * p
