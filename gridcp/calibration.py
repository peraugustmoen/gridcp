"""Calibration utilities for the new gridcp API.

This module provides calibration helpers for the new API.

- ``draw_samples``: simulate Monte Carlo streams under a pre-change distribution,
  optionally with a post-change regime.
- ``mc_max_scores``: run Monte Carlo simulations through a detector and collect
  the path-wise maximum detection score.
- ``calibrate_threshold``: compute an empirical threshold for a score model.
- ``calibrate_detector_threshold``: convenience wrapper for detector objects.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping
import inspect

import numpy as np

from gridcp.detector import GridDetector

ChangepointSpec = int | Callable[[np.random.Generator, int, int], int] | None
RNGInput = np.random.Generator | int | None

# Deterministic default used when rng=None.
DEFAULT_MC_SEED = 0


def _normalize_rng(rng: RNGInput) -> np.random.Generator:
    """Return a NumPy Generator from supported rng inputs.

    Supported inputs are:
    - ``np.random.Generator``: used as-is.
    - ``int``: used as seed for ``np.random.default_rng``.
    - ``None``: uses a fixed default seed for deterministic behavior.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng(DEFAULT_MC_SEED)
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    raise TypeError("rng must be a numpy.random.Generator, int seed, or None.")


def _call_sampler(
    sampler: Callable[..., Any],
    rng: np.random.Generator,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Any:
    """Call a sampler with optional ``rng`` support.

    If the sampler accepts an ``rng`` keyword argument (or ``**kwargs``), it is
    passed through. Otherwise the sampler is called without ``rng``.
    """
    try:
        sig = inspect.signature(sampler)
    except (TypeError, ValueError):
        sig = None

    if sig is not None:
        params = sig.parameters
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if "rng" in params or accepts_kwargs:
            return sampler(*args, rng=rng, **kwargs)

    return sampler(*args, **kwargs)


def _normalise_observation(x: Any, n_features: int) -> np.ndarray:
    """Convert a sampled observation to ``(n_features,)`` float array."""
    x_arr = np.asarray(x, dtype=np.float64)

    if x_arr.ndim == 0:
        return np.full(n_features, float(x_arr), dtype=np.float64)

    flat = x_arr.reshape(-1)
    if flat.size != n_features:
        raise ValueError(
            "Sampler output has wrong size "
            f"{flat.size}; expected scalar or size {n_features}."
        )
    return flat


def _resolve_changepoint(
    changepoint: ChangepointSpec,
    rng: np.random.Generator,
    n_samples: int,
    path_index: int,
) -> int:
    """Resolve changepoint to an integer in ``[1, n_samples]``.

    The changepoint is interpreted as the first 1-based index generated from the
    post-change sampler. ``n_samples`` means there is no post-change observation
    in a stream of length ``n_samples``.
    """
    if changepoint is None:
        return n_samples

    if callable(changepoint):
        cp = int(changepoint(rng, n_samples, path_index))
    else:
        cp = int(changepoint)

    if cp < 1 or cp > n_samples:
        raise ValueError(f"changepoint must be in [1, {n_samples}], got {cp}.")
    return cp


def draw_samples(
    n_paths: int,
    stream_len: int,
    n_features: int,
    pre_sampler: Callable[..., Any],
    *,
    rng: RNGInput = None,
    pre_args: tuple[Any, ...] = (),
    pre_kwargs: Mapping[str, Any] | None = None,
    post_sampler: Callable[..., Any] | None = None,
    post_args: tuple[Any, ...] = (),
    post_kwargs: Mapping[str, Any] | None = None,
    changepoint: ChangepointSpec = None,
) -> np.ndarray:
    """Draw Monte Carlo sample paths.

    Parameters
    ----------
    n_paths : int
        Number of independent paths.
    stream_len : int
        Number of observations per path.
    n_features : int
        Observation dimension.
    pre_sampler : callable
        Baseline sampler callable.
    rng : numpy.random.Generator | int | None, optional
        Randomness control.
        - ``Generator``: used as-is.
        - ``int``: used as seed to create a generator.
        - ``None``: uses a fixed default seed, so repeated calls with the same
          inputs are reproducible.
    pre_args, pre_kwargs : optional
        Additional arguments for ``pre_sampler``.
    post_sampler : callable, optional
        Post-change sampler. If ``None``, all observations are pre-change.
    post_args, post_kwargs : optional
        Additional arguments for ``post_sampler``.
    changepoint : int | callable | None, optional
        First 1-based post-change index in each stream.
        If callable, called as ``changepoint(rng, stream_len, path_idx)``.

    Returns
    -------
    np.ndarray
        Simulated array with shape ``(n_paths, stream_len, n_features)``.
    """
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1.")
    if stream_len < 1:
        raise ValueError("stream_len must be >= 1.")
    if n_features < 1:
        raise ValueError("n_features must be >= 1.")

    if pre_kwargs is None:
        pre_kwargs = {}
    if post_kwargs is None:
        post_kwargs = {}

    if changepoint is not None and post_sampler is None:
        raise ValueError("post_sampler must be provided when changepoint is set.")

    local_rng = _normalize_rng(rng)
    out = np.empty((n_paths, stream_len, n_features), dtype=np.float64)

    for path_idx in range(n_paths):
        cp = _resolve_changepoint(changepoint, local_rng, stream_len, path_idx)
        for t in range(1, stream_len + 1):
            if post_sampler is not None and t >= cp:
                raw = _call_sampler(post_sampler, local_rng, post_args, post_kwargs)
            else:
                raw = _call_sampler(pre_sampler, local_rng, pre_args, pre_kwargs)
            out[path_idx, t - 1, :] = _normalise_observation(raw, n_features)

    return out


def mc_max_scores(
    detector: GridDetector,
    n_paths: int,
    stream_len: int,
    pre_sampler: Callable[..., Any],
    *,
    rng: RNGInput = None,
    pre_args: tuple[Any, ...] = (),
    pre_kwargs: Mapping[str, Any] | None = None,
    post_sampler: Callable[..., Any] | None = None,
    post_args: tuple[Any, ...] = (),
    post_kwargs: Mapping[str, Any] | None = None,
    changepoint: ChangepointSpec = None,
    n_features: int | None = None,
) -> np.ndarray:
    """Run Monte Carlo paths and return the maximum score for each path.

    Reproducibility follows the ``rng`` argument:
    - ``Generator``: continues from its current state.
    - ``int``: deterministic run from that seed.
    - ``None``: deterministic run from an internal fixed default seed.
    """
    if n_features is None:
        score_n_features = getattr(detector.score, "n_features", None)
        if score_n_features is None:
            raise ValueError(
                "n_features was not provided and could not be inferred from detector.score."
            )
        n_features = int(score_n_features)

    X = draw_samples(
        n_paths=n_paths,
        stream_len=stream_len,
        n_features=n_features,
        pre_sampler=pre_sampler,
        rng=rng,
        pre_args=pre_args,
        pre_kwargs=pre_kwargs,
        post_sampler=post_sampler,
        post_args=post_args,
        post_kwargs=post_kwargs,
        changepoint=changepoint,
    )

    max_scores = np.empty(n_paths, dtype=np.float64)
    for path_idx in range(n_paths):
        state = detector.init_state()
        max_score = 0.0
        for t in range(stream_len):
            state, output = detector.update(state, X[path_idx, t, :])
            temp = float(output["max_score"])
            if temp > max_score:
                max_score = temp
        max_scores[path_idx] = max_score

    return max_scores


def mc_alarm_times(
    detector: GridDetector,
    n_paths: int,
    stream_len: int,
    pre_sampler: Callable[..., Any],
    *,
    rng: RNGInput = None,
    pre_args: tuple[Any, ...] = (),
    pre_kwargs: Mapping[str, Any] | None = None,
    post_sampler: Callable[..., Any] | None = None,
    post_args: tuple[Any, ...] = (),
    post_kwargs: Mapping[str, Any] | None = None,
    changepoint: ChangepointSpec = None,
    n_features: int | None = None,
) -> np.ndarray:
    """Run Monte Carlo paths and return first alarm time for each path.

    Alarm times are 1-based sample indices. For paths with no alarm by
    ``stream_len``, the returned value is ``stream_len + 1``.

    Reproducibility follows the ``rng`` argument:
    - ``Generator``: continues from its current state.
    - ``int``: deterministic run from that seed.
    - ``None``: deterministic run from an internal fixed default seed.
    """
    if n_features is None:
        score_n_features = getattr(detector.score, "n_features", None)
        if score_n_features is None:
            raise ValueError(
                "n_features was not provided and could not be inferred from detector.score."
            )
        n_features = int(score_n_features)

    X = draw_samples(
        n_paths=n_paths,
        stream_len=stream_len,
        n_features=n_features,
        pre_sampler=pre_sampler,
        rng=rng,
        pre_args=pre_args,
        pre_kwargs=pre_kwargs,
        post_sampler=post_sampler,
        post_args=post_args,
        post_kwargs=post_kwargs,
        changepoint=changepoint,
    )

    alarm_times = np.full(n_paths, stream_len + 1, dtype=np.int64)
    for path_idx in range(n_paths):
        state = detector.init_state()
        for t in range(stream_len):
            state, output = detector.update(state, X[path_idx, t, :])
            if bool(output["alarm"]):
                alarm_times[path_idx] = t + 1
                break

    return alarm_times


def calibrate_threshold(
    score: Any,
    *,
    alpha: float,
    n_paths: int,
    stream_len: int,
    pre_sampler: Callable[..., Any],
    rng: RNGInput = None,
    pre_args: tuple[Any, ...] = (),
    pre_kwargs: Mapping[str, Any] | None = None,
    n_features: int | None = None,
) -> float:
    """Estimate a score threshold from Monte Carlo max scores under the null.

    Returns the empirical ``(1 - alpha)`` quantile of max scores.

    Parameters
    ----------
    score : Any
        Score model compatible with ``GridDetector``.
    alpha : float
        Target false alarm level in ``(0, 1)``.
    n_paths : int
        Number of Monte Carlo paths.
    stream_len : int
        Number of samples per path.
    pre_sampler : callable
        Null sampler.
    rng : numpy.random.Generator | int | None, optional
        Randomness control passed to Monte Carlo simulation.
        ``None`` uses a fixed default seed for deterministic behavior.
    pre_args, pre_kwargs : optional
        Additional arguments passed to ``pre_sampler``.
    n_features : int, optional
        Number of features per sample. If omitted, this is inferred from
        ``score.n_features`` when available. For custom score objects that do
        not expose ``n_features``, this argument must be provided explicitly.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    inferred_n_features = n_features
    if inferred_n_features is None:
        score_n_features = getattr(score, "n_features", None)
        if score_n_features is None:
            raise ValueError(
                "n_features was not provided and could not be inferred from score.n_features."
            )
        inferred_n_features = int(score_n_features)

    detector = GridDetector(score=score, threshold=1.0)

    max_scores = mc_max_scores(
        detector=detector,
        n_paths=n_paths,
        stream_len=stream_len,
        pre_sampler=pre_sampler,
        rng=rng,
        pre_args=pre_args,
        pre_kwargs=pre_kwargs,
        n_features=inferred_n_features,
    )
    return float(np.quantile(max_scores, 1.0 - alpha))


def calibrate_detector_threshold(
    detector: GridDetector,
    *,
    alpha: float,
    n_paths: int,
    stream_len: int,
    pre_sampler: Callable[..., Any],
    rng: RNGInput = None,
    pre_args: tuple[Any, ...] = (),
    pre_kwargs: Mapping[str, Any] | None = None,
    n_features: int | None = None,
) -> float:
    """Estimate threshold for an existing detector.

    This is a convenience wrapper around :func:`calibrate_threshold`.
    """
    return calibrate_threshold(
        score=detector.score,
        alpha=alpha,
        n_paths=n_paths,
        stream_len=stream_len,
        pre_sampler=pre_sampler,
        rng=rng,
        pre_args=pre_args,
        pre_kwargs=pre_kwargs,
        n_features=n_features,
    )


def with_calibrated_threshold(
    detector: GridDetector,
    threshold: float,
) -> GridDetector:
    """Return a copy of ``detector`` with updated threshold."""
    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    return replace(detector, threshold=float(threshold))
