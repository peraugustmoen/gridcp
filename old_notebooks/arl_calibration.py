"""ARL-based calibration utilities.

This module provides ARL (Average Run Length) calibration helpers that mirror
the FA-based functions in ``gridcp.calibration``:

- ``calibrate_threshold_arl``: compute an empirical threshold for a score model
  given a target ARL γ.
- ``calibrate_detector_threshold_arl``: convenience wrapper for detector objects.
- ``calibrate_threshold_arl_from_samples``: threshold from pre-generated streams.
- ``calibrate_threshold_arl_from_data``: threshold via block bootstrap from training data.
- ``calibrate_detector_threshold_arl_from_data``: convenience wrapper for detector objects.

The calibration works by simulating *n_paths* null streams of length γ, collecting
the maximum score from each, and returning the (1/e)-quantile.  This yields a
detector whose expected alarm time under the null is approximately γ.

For the false-alarm approach, use ``gridcp.calibration.calibrate_threshold_false_alarm``.
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

import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike

from gridcp.calibration import (
    _block_bootstrap_samples,
    _mc_max_scores_chunk_from_samples,
    _normalize_rng,
    _path_index_chunks,
    _resolve_n_jobs,
    mc_max_scores,
)
from gridcp.detector import GridDetector
from gridcp.typing import PenaltyType, ScoreModel

RNGInput = np.random.Generator | int | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_arl_threshold_from_max_scores(
    max_scores: np.ndarray,
) -> float | np.ndarray:
    """Compute ARL threshold(s) from an array of per-path max scores.

    Parameters
    ----------
    max_scores : np.ndarray
        Shape ``(n_paths,)`` for scalar scores, ``(n_paths, K)`` for
        multivariate scores.

    Returns
    -------
    float | np.ndarray
        Scalar threshold or 1-D array of shape ``(K,)``.
    """
    if max_scores.ndim == 1:
        return float(np.quantile(max_scores, 1.0 / np.e))

    # K > 1 — two-step procedure.
    # Step 1: per-test (1/e)-quantiles.
    n_tests = max_scores.shape[1]
    individual_thresholds = np.array(
        [float(np.quantile(max_scores[:, k], 1.0 / np.e)) for k in range(n_tests)],
        dtype=np.float64,
    )

    # Step 2: scale so the joint ARL equals target_arl.
    standardized = max_scores / individual_thresholds  # (n_paths, K)
    combined_max = np.max(standardized, axis=1)  # (n_paths,)
    c = float(np.quantile(combined_max, 1.0 / np.e))
    return c * individual_thresholds


def _warn_if_non_constant_penalty(score: ScoreModel) -> None:
    penalty = getattr(score, "penalty", None)
    if penalty is not None and penalty != PenaltyType.CONSTANT:
        warnings.warn(
            "ARL calibration requires a stationary score. "
            "The supplied score uses a time-dependent penalty, which will "
            "produce incorrect ARL results. Set penalty=PenaltyType.CONSTANT.",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# MC-based ARL calibration
# ---------------------------------------------------------------------------


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
        factor ``c`` computed from the same standardized null max-score
        sample across tests to ensure the overall ARL equals *target_arl*.

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

    _warn_if_non_constant_penalty(score)

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

    return _compute_arl_threshold_from_max_scores(max_scores)


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


# ---------------------------------------------------------------------------
# Data-driven (bootstrap) ARL calibration
# ---------------------------------------------------------------------------


def calibrate_threshold_arl_from_samples(
    score: ScoreModel,
    samples: ArrayLike,
    *,
    parallel: bool = True,
    n_jobs: int | None = None,
) -> float | np.ndarray:
    """Estimate an ARL threshold from pre-generated sample paths.

    The caller supplies an array of observation streams and the function runs
    each stream through the detector to build the null distribution of maximum
    scores, then returns the ``(1/e)``-quantile as the threshold.

    Parameters
    ----------
    score : ScoreModel
        Score model compatible with ``GridDetector``.
    samples : ArrayLike
        Pre-generated observation streams with shape
        ``(n_paths, stream_len, n_features)``.  The stream length is used as
        the ARL target implicitly — it should equal the desired ``target_arl``.
    parallel : bool, optional
        If ``True`` (default), evaluate sample paths across multiple processes.
    n_jobs : int or None, optional
        Number of worker processes.  ``None`` uses all available CPU cores.

    Returns
    -------
    float | np.ndarray
        Estimated ARL threshold.  Scalar for single-test scores, shape
        ``(K,)`` for multivariate scores.

    Notes
    -----
    ARL calibration is only valid when the score is stationary under the null.
    Scores with time-dependent penalties (anything other than
    ``PenaltyType.CONSTANT``) will trigger a ``UserWarning``.
    """
    _warn_if_non_constant_penalty(score)

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 3:
        raise ValueError(
            "samples must be a 3-D array of shape "
            f"(n_paths, stream_len, n_features); got ndim={samples.ndim}."
        )

    n_features_expected = score.n_features
    if samples.shape[2] != n_features_expected:
        raise ValueError(
            f"samples last dimension ({samples.shape[2]}) does not match "
            f"score.n_features ({n_features_expected})."
        )

    n_paths = samples.shape[0]
    detector = GridDetector(score=score, threshold=1.0)

    n_workers = _resolve_n_jobs(n_jobs=n_jobs, n_paths=n_paths) if parallel else 1
    if n_workers <= 1:
        max_scores = _mc_max_scores_chunk_from_samples(detector, samples)
    else:
        chunks = _path_index_chunks(n_paths=n_paths, n_jobs=n_workers)
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(
                        _mc_max_scores_chunk_from_samples,
                        detector,
                        samples[start:end],
                    )
                    for start, end in chunks
                ]
                results = [fut.result() for fut in futures]
            max_scores = np.concatenate(results, axis=0)
        except (BrokenProcessPool, OSError, pickle.PicklingError) as exc:
            warnings.warn(
                "Parallel evaluation failed; falling back to serial execution. "
                f"Original error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            max_scores = _mc_max_scores_chunk_from_samples(detector, samples)

    return _compute_arl_threshold_from_max_scores(max_scores)


def calibrate_threshold_arl_from_data(
    score: ScoreModel,
    training_data: ArrayLike,
    *,
    target_arl: int,
    n_paths: int = 1000,
    block_length: int | None = None,
    rng: RNGInput = None,
    parallel: bool = True,
    n_jobs: int | None = None,
) -> float | np.ndarray:
    """Estimate an ARL threshold by circular block-bootstrapping training data.

    When the null distribution is unknown but a representative change-free
    training set is available, this function calibrates the threshold by:

    1.  Resampling ``n_paths`` streams of length ``target_arl`` from
        ``training_data`` using the circular block bootstrap (Politis &
        Romano, 1992).
    2.  Running each stream through the detector and recording the path-wise
        maximum score.
    3.  Setting the threshold at the empirical ``(1/e)``-quantile.

    Parameters
    ----------
    score : ScoreModel
        Score model compatible with ``GridDetector``.
    training_data : ArrayLike
        Change-free training observations, shape ``(N,)`` for univariate or
        ``(N, n_features)`` for multivariate data.
    target_arl : int
        Desired average run length under the null.  Also used as the
        bootstrap stream length.
    n_paths : int, optional
        Number of bootstrap replications (default 1000).
    block_length : int or None, optional
        Block length for the circular block bootstrap.  ``None`` selects
        ``max(1, int(N ** (1/3)))`` automatically.
    rng : numpy.random.Generator | int | None, optional
        Randomness control.
    parallel : bool, optional
        If ``True`` (default), evaluate sample paths across multiple processes.
    n_jobs : int or None, optional
        Number of worker processes.  ``None`` uses all available CPU cores.

    Returns
    -------
    float | np.ndarray
        Estimated ARL threshold.  Scalar for single-test scores, shape
        ``(K,)`` for multivariate scores.

    Notes
    -----
    ARL calibration is only valid when the score is stationary under the null.
    Scores with time-dependent penalties will trigger a ``UserWarning``.
    """
    _warn_if_non_constant_penalty(score)

    if target_arl < 1:
        raise ValueError("target_arl must be a positive integer.")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1.")

    training_data = np.asarray(training_data, dtype=np.float64)
    if training_data.ndim == 1:
        training_data = training_data.reshape(-1, 1)
    if training_data.ndim != 2:
        raise ValueError(
            f"training_data must be 1-D or 2-D; got ndim={training_data.ndim}."
        )

    n_obs, n_feat = training_data.shape
    n_features_expected = score.n_features
    if n_feat != n_features_expected:
        raise ValueError(
            f"training_data has {n_feat} features but "
            f"score.n_features={n_features_expected}."
        )
    if n_obs < 2:
        raise ValueError("training_data must have at least 2 observations.")

    if block_length is None:
        block_length = max(1, int(n_obs ** (1.0 / 3.0)))
    if block_length < 1:
        raise ValueError("block_length must be >= 1.")

    if n_obs < target_arl:
        warnings.warn(
            f"training_data has {n_obs} observations but target_arl={target_arl}. "
            "The circular block bootstrap will wrap around, but calibration may "
            "be unreliable with so few training observations.",
            UserWarning,
            stacklevel=2,
        )

    local_rng = _normalize_rng(rng)
    samples = _block_bootstrap_samples(
        data=training_data,
        n_paths=n_paths,
        stream_len=target_arl,
        block_length=block_length,
        rng=local_rng,
    )

    return calibrate_threshold_arl_from_samples(
        score=score,
        samples=samples,
        parallel=parallel,
        n_jobs=n_jobs,
    )


def calibrate_detector_threshold_arl_from_data(
    detector: GridDetector,
    training_data: ArrayLike,
    *,
    target_arl: int,
    n_paths: int = 1000,
    block_length: int | None = None,
    rng: RNGInput = None,
    parallel: bool = True,
    n_jobs: int | None = None,
) -> float | np.ndarray:
    """Estimate ARL threshold for an existing detector using training data.

    Convenience wrapper around :func:`calibrate_threshold_arl_from_data`.
    """
    return calibrate_threshold_arl_from_data(
        score=detector.score,
        training_data=training_data,
        target_arl=target_arl,
        n_paths=n_paths,
        block_length=block_length,
        rng=rng,
        parallel=parallel,
        n_jobs=n_jobs,
    )
