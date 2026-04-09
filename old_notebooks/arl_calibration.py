"""ARL-based calibration utilities.

This module provides ARL (Average Run Length) calibration helpers that mirror
the FA-based functions in ``gridcp.calibration``:

- ``calibrate_threshold_arl``: compute an empirical threshold for a score model
  given a target ARL γ.
- ``calibrate_detector_threshold_arl``: convenience wrapper for detector objects.

The calibration works by simulating *n_paths* null streams of length γ, collecting
the maximum score from each, and returning the (1/e)-quantile.  This yields a
detector whose expected alarm time under the null is approximately γ.

For the false-alarm approach, use ``gridcp.calibration.calibrate_threshold``.
``gridcp.calibration.with_calibrated_threshold`` is ARL-agnostic and can be
used with ARL thresholds directly.

Why the (1/e)-quantile?
-----------------------
Under the null, alarm times are approximately exponentially distributed.  For
an Exp(1/γ) random variable, P(alarm ≤ γ) = 1 − 1/e.  Setting the threshold
at the (1/e)-quantile of the null max-score distribution therefore ensures that
the expected alarm time is approximately γ.

Important
---------
ARL calibration is only valid when the test statistic is *stationary* under the
null, i.e., its distribution does not grow with time.  Scores with adaptive
penalties (e.g., the default ``MeanCUSUM`` penalty ``log(t) + sqrt(log(t))``)
are non-stationary and will give incorrect ARL results.  Use a constant penalty
or a score whose null distribution is stable over time.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

import warnings

from gridcp.detector import GridDetector
from gridcp.calibration import mc_max_scores
from gridcp.typing import PenaltyType, ScoreModel

RNGInput = np.random.Generator | int | None


def calibrate_threshold_arl(
    score: ScoreModel,
    *,
    target_arl: int,
    n_paths: int,
    pre_sampler: Callable[..., Any],
    rng: RNGInput = None,
    pre_args: tuple[Any, ...] = (),
    pre_kwargs: Mapping[str, Any] | None = None,
    parallel: bool = True,
    n_jobs: int | None = None,
    strict_equivalence: bool = False,
) -> float | np.ndarray:
    """Estimate a score threshold via Average Run Length (ARL) calibration.

    Simulates *n_paths* null streams of length *target_arl*, collects the
    maximum score from each, and returns the (1/e)-quantile.  This yields a
    detector whose expected alarm time under the null is approximately
    *target_arl*.

    Parameters
    ----------
    score : Any
        Score model compatible with ``GridDetector``.
    target_arl : int
        Desired average run length under the null.
        Also used as the calibration stream length.
    n_paths : int
        Number of Monte Carlo paths.
    pre_sampler : callable
        Null sampler.
    rng : numpy.random.Generator | int | None, optional
        Randomness control passed to Monte Carlo simulation. Must be one of:
        ``numpy.random.Generator``, ``int`` seed, or ``None``.
        ``None`` uses a fixed default seed for deterministic behavior.
    pre_args, pre_kwargs : optional
        Additional arguments passed to ``pre_sampler``.
    parallel : bool, default True
        Whether to run Monte Carlo paths in parallel.
    n_jobs : int or None, optional
        Number of parallel workers.  ``None`` uses automatic core detection.
    strict_equivalence : bool, default False
        If ``True``, sample generation is serial (only evaluation is
        parallel), guaranteeing identical output to ``parallel=False``.

    Returns
    -------
    float or np.ndarray
        The ARL threshold.  For scalar scores (K=1), a float equal to the
        (1/e)-quantile of null max scores.  For multivariate scores (K>1),
        a 1-D array of shape ``(K,)`` with *combined* thresholds that
        control the *joint* ARL: each individual threshold is first derived
        from a separate (1/e)-quantile per test, then scaled by a common
        factor ``c`` found from a second MC pass that standardises scores
        across tests and ensures the overall ARL equals *target_arl*.

    Notes
    -----
    The (1/e)-quantile arises because, under the exponential approximation
    for alarm times, P(max_score > threshold) = 1 − 1/e ≈ 0.632 corresponds
    to E[alarm time] = γ.

    This calibration is only meaningful when the score statistic is stationary
    under the null.  See the module docstring for details.

    To match a false alarm probability α over horizon N, set:

        target_arl = int(np.ceil(-N / np.log(1 - alpha)))

    Examples
    --------
    >>> import numpy as np
    >>> from gridcp.scores import MeanCUSUM, PenaltyType
    >>> threshold = calibrate_threshold_arl(
    ...     MeanCUSUM(n_features=1, penalty=PenaltyType.CONSTANT),
    ...     target_arl=2000,
    ...     n_paths=2000,
    ...     pre_sampler=np.random.default_rng(0).standard_normal,
    ...     rng=42,
    ... )
    """
    if target_arl < 1:
        raise ValueError("target_arl must be a positive integer.")

    penalty = getattr(score, "penalty", None)
    if penalty is not None and penalty != PenaltyType.CONSTANT:
        warnings.warn(
            "ARL calibration requires a stationary score. "
            "The supplied score uses a time-dependent penalty, which will "
            "produce incorrect ARL results. Set penalty=PenaltyType.CONSTANT.",
            UserWarning,
            stacklevel=2,
        )

    detector = GridDetector(score=score, threshold=1.0)

    max_scores = mc_max_scores(
        detector=detector,
        n_paths=n_paths,
        stream_len=target_arl,
        pre_sampler=pre_sampler,
        rng=rng,
        pre_args=pre_args,
        pre_kwargs=pre_kwargs,
        parallel=parallel,
        n_jobs=n_jobs,
        strict_equivalence=strict_equivalence,
    )

    if max_scores.ndim == 1:
        # K = 1: single threshold, no standardization needed.
        return float(np.quantile(max_scores, 1.0 / np.e))

    # K > 1 — two-step procedure following the OCD paper.
    # Step 1: per-test (1/e)-quantiles from the first MC pass.
    n_tests = max_scores.shape[1]
    individual_thresholds = np.array(
        [float(np.quantile(max_scores[:, k], 1.0 / np.e)) for k in range(n_tests)],
        dtype=np.float64,
    )

    # Step 2: reuse the first-pass results.  Since τ_k > 0, dividing by
    # τ_k commutes with the time-max already stored in max_scores, so no
    # second simulation is needed.
    standardized = max_scores / individual_thresholds  # (n_paths, K)
    combined_max = np.max(standardized, axis=1)        # (n_paths,)
    c = float(np.quantile(combined_max, 1.0 / np.e))
    return c * individual_thresholds


def calibrate_detector_threshold_arl(
    detector: GridDetector,
    *,
    target_arl: int,
    n_paths: int,
    pre_sampler: Callable[..., Any],
    rng: RNGInput = None,
    pre_args: tuple[Any, ...] = (),
    pre_kwargs: Mapping[str, Any] | None = None,
    parallel: bool = True,
    n_jobs: int | None = None,
    strict_equivalence: bool = False,
) -> float | np.ndarray:
    """Estimate ARL threshold for an existing detector.

    This is a convenience wrapper around :func:`calibrate_threshold_arl`.
    """
    return calibrate_threshold_arl(
        score=detector.score,
        target_arl=target_arl,
        n_paths=n_paths,
        pre_sampler=pre_sampler,
        rng=rng,
        pre_args=pre_args,
        pre_kwargs=pre_kwargs,
        parallel=parallel,
        n_jobs=n_jobs,
        strict_equivalence=strict_equivalence,
    )
