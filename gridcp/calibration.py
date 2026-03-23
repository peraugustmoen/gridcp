"""Calibration utilities for the new gridcp API.

This module provides calibration helpers for the new API.

- ``draw_samples``: simulate Monte Carlo streams under a pre-change distribution,
  optionally with a post-change regime.
- ``mc_max_scores``: run Monte Carlo simulations through a detector and collect
  the path-wise maximum detection score.
- ``calibrate_threshold``: compute an empirical threshold for a score model.
- ``calibrate_detector_threshold``: convenience wrapper for detector objects.

Observation-shape convention
----------------------------
The Monte Carlo helpers in this module are vector-oriented: each sampled
observation is converted to a 1D ``float64`` vector of length ``n_features``.
Scalar sampler outputs are broadcast to length ``n_features``. Non-scalar
outputs are flattened with ``reshape(-1)`` and must have total size
``n_features``.

Randomness and changepoint inputs
---------------------------------
``rng`` must be one of:
- ``numpy.random.Generator``
- an integer seed
- ``None`` (deterministic internal default seed)

``changepoint`` must be one of:
- ``None`` (all samples are pre-change)
- an ``int`` in ``[0, stream_len]`` representing the first post-change index
    (0-based):
    - changepoint=0: all observations are post-change
    - changepoint=k (0 < k < stream_len): observations ``[0, k)`` are pre-change,
        observations ``[k, stream_len)`` are post-change
    - changepoint=stream_len: all observations are pre-change
- a callable ``f(rng, stream_len, path_index) -> int`` returning a value in
    ``[0, stream_len]``
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
from typing import Any, Callable, Mapping
import inspect
import os
import warnings

import numpy as np

from gridcp.detector import GridDetector

# Changepoint callables receive ``(rng, stream_len, path_index)`` and must
# return an integer in ``[0, stream_len]`` representing the first post-change
# index (0-based).
ChangepointSpec = int | Callable[[np.random.Generator, int, int], int] | None
# Public RNG input contract for Monte Carlo helpers.
RNGInput = np.random.Generator | int | None

# Deterministic default used when rng=None.
DEFAULT_MC_SEED = 0

# Worker globals initialized by ``_init_mc_worker``.
_WORKER_DETECTOR: GridDetector | None = None
_WORKER_PRE_SAMPLER: Callable[..., Any] | None = None
_WORKER_PRE_ARGS: tuple[Any, ...] | None = None
_WORKER_PRE_KWARGS: Mapping[str, Any] | None = None
_WORKER_POST_SAMPLER: Callable[..., Any] | None = None
_WORKER_POST_ARGS: tuple[Any, ...] | None = None
_WORKER_POST_KWARGS: Mapping[str, Any] | None = None
_WORKER_CHANGEPOINT: ChangepointSpec = None
_WORKER_PRE_CALL: Callable[[np.random.Generator], Any] | None = None
_WORKER_POST_CALL: Callable[[np.random.Generator], Any] | None = None


def _sampler_accepts_rng_kw_uncached(sampler: Callable[..., Any]) -> bool:
    """Return whether sampler accepts ``rng`` kwarg or ``**kwargs``."""
    try:
        sig = inspect.signature(sampler)
    except (TypeError, ValueError):
        return False

    params = sig.parameters
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    return "rng" in params or accepts_kwargs


@lru_cache(maxsize=1024)
def _sampler_accepts_rng_kw_cached(sampler: Callable[..., Any]) -> bool:
    """Cache sampler introspection for hashable callables."""
    return _sampler_accepts_rng_kw_uncached(sampler)


def _sampler_accepts_rng_kw(sampler: Callable[..., Any]) -> bool:
    """Return whether sampler accepts ``rng``, using cache when possible."""
    try:
        return _sampler_accepts_rng_kw_cached(sampler)
    except TypeError:
        return _sampler_accepts_rng_kw_uncached(sampler)


def _make_sampler_caller(
    sampler: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Callable[[np.random.Generator], Any]:
    """Build a fast callable ``f(rng)`` for a sampler."""
    if _sampler_accepts_rng_kw(sampler):
        return lambda local_rng: sampler(*args, rng=local_rng, **kwargs)
    return lambda local_rng: sampler(*args, **kwargs)


def _get_cloudpickle() -> Any:
    """Import and return cloudpickle.

    Parallel process mode relies on cloudpickle so notebook-defined callables
    can be shipped to worker processes.
    """
    try:
        import cloudpickle
    except ImportError as exc:
        raise ImportError(
            "parallel process execution requires cloudpickle. "
            "Install it with `pip install cloudpickle` or install gridcp "
            "with updated dependencies."
        ) from exc
    return cloudpickle


def _serialise_for_worker(obj: Any) -> bytes:
    """Serialize an object for process workers using cloudpickle."""
    cloudpickle = _get_cloudpickle()
    return cloudpickle.dumps(obj)


def _deserialise_from_worker(blob: bytes) -> Any:
    """Deserialize an object in workers using cloudpickle."""
    cloudpickle = _get_cloudpickle()
    return cloudpickle.loads(blob)


def _init_mc_worker(
    detector_blob: bytes,
    pre_sampler_blob: bytes,
    pre_args_blob: bytes,
    pre_kwargs_blob: bytes,
    post_sampler_blob: bytes,
    post_args_blob: bytes,
    post_kwargs_blob: bytes,
    changepoint_blob: bytes,
    n_features: int,
) -> None:
    """Initialize process-local context for Monte Carlo workers."""
    global _WORKER_DETECTOR
    global _WORKER_PRE_SAMPLER
    global _WORKER_PRE_ARGS
    global _WORKER_PRE_KWARGS
    global _WORKER_POST_SAMPLER
    global _WORKER_POST_ARGS
    global _WORKER_POST_KWARGS
    global _WORKER_CHANGEPOINT
    global _WORKER_PRE_CALL
    global _WORKER_POST_CALL

    _WORKER_DETECTOR = _deserialise_from_worker(detector_blob)
    _WORKER_PRE_SAMPLER = _deserialise_from_worker(pre_sampler_blob)
    _WORKER_PRE_ARGS = _deserialise_from_worker(pre_args_blob)
    _WORKER_PRE_KWARGS = _deserialise_from_worker(pre_kwargs_blob)
    _WORKER_POST_SAMPLER = _deserialise_from_worker(post_sampler_blob)
    _WORKER_POST_ARGS = _deserialise_from_worker(post_args_blob)
    _WORKER_POST_KWARGS = _deserialise_from_worker(post_kwargs_blob)
    _WORKER_CHANGEPOINT = _deserialise_from_worker(changepoint_blob)

    _WORKER_PRE_CALL = _make_sampler_caller(
        _WORKER_PRE_SAMPLER,
        _WORKER_PRE_ARGS,
        _WORKER_PRE_KWARGS,
    )
    if _WORKER_POST_SAMPLER is not None:
        _WORKER_POST_CALL = _make_sampler_caller(
            _WORKER_POST_SAMPLER,
            _WORKER_POST_ARGS or (),
            _WORKER_POST_KWARGS or {},
        )
    else:
        _WORKER_POST_CALL = None

    # Warm up a tiny detector path per worker to reduce first-call JIT latency.
    try:
        state = _WORKER_DETECTOR.init_state()
        x = np.zeros(int(n_features), dtype=np.float64)
        state, _ = _WORKER_DETECTOR.update(state, x)
        _WORKER_DETECTOR.update(state, x)
    except Exception:
        # Warmup is opportunistic; simulation remains valid without it.
        pass


def _resolve_n_jobs(n_jobs: int | None, n_paths: int) -> int:
    """Return an effective worker count in ``[1, n_paths]``."""
    if n_jobs is None:
        resolved = os.cpu_count() or 1
    else:
        resolved = int(n_jobs)
        if resolved < 1:
            raise ValueError("n_jobs must be >= 1 when provided.")
    return max(1, min(resolved, n_paths))


def _path_index_chunks(n_paths: int, n_jobs: int) -> list[tuple[int, int]]:
    """Partition path indices into contiguous chunks."""
    if n_jobs <= 1:
        return [(0, n_paths)]

    chunk_size = (n_paths + n_jobs - 1) // n_jobs
    chunks: list[tuple[int, int]] = []
    for start in range(0, n_paths, chunk_size):
        end = min(n_paths, start + chunk_size)
        chunks.append((start, end))
    return chunks


def _derive_base_seed(rng: RNGInput) -> int:
    """Derive a deterministic base seed from supported rng inputs.

    ``rng`` must be a ``numpy.random.Generator``, an integer seed, or ``None``.
    """
    local_rng = _normalize_rng(rng)
    return int(local_rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))


def _spawn_chunk_seeds(base_seed: int, n_chunks: int) -> list[int]:
    """Spawn one deterministic seed per worker chunk."""
    seed_seq = np.random.SeedSequence(base_seed)
    children = seed_seq.spawn(n_chunks)
    return [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children]


def _mc_worker_chunk(
    task: str,
    start: int,
    end: int,
    chunk_seed: int,
    stream_len: int,
    n_features: int,
) -> tuple[int, np.ndarray]:
    """Run a chunk of MC paths in a process worker."""
    if _WORKER_DETECTOR is None:
        raise RuntimeError("Monte Carlo worker was not initialized.")
    if _WORKER_PRE_SAMPLER is None or _WORKER_PRE_ARGS is None:
        raise RuntimeError("Monte Carlo worker sampler context is missing.")
    if _WORKER_PRE_KWARGS is None:
        raise RuntimeError("Monte Carlo worker sampler kwargs are missing.")
    if _WORKER_PRE_CALL is None:
        raise RuntimeError("Monte Carlo worker sampler callables are missing.")

    n_local_paths = end - start
    if task == "max":
        out = np.empty(n_local_paths, dtype=np.float64)
    elif task == "alarm":
        out = np.empty(n_local_paths, dtype=np.int64)
    else:
        raise ValueError(f"Unknown task kind: {task}.")

    # One RNG per worker chunk to minimize generator construction overhead.
    local_rng = np.random.default_rng(chunk_seed)

    pre_mode: str | None = None
    post_mode: str | None = None
    scalar_buffer = np.empty(n_features, dtype=np.float64)

    for local_idx in range(n_local_paths):
        path_idx = start + local_idx
        cp = _resolve_changepoint(_WORKER_CHANGEPOINT, local_rng, stream_len, path_idx)

        state = _WORKER_DETECTOR.init_state()
        if task == "max":
            max_score = 0.0

        # Note: t represents the SAMPLE SIZE (1-indexed; current observation count).
        # This ranges from 1 to stream_len, not 0 to stream_len-1.
        # When storing in arrays, use (t - 1) for 0-indexed array access.
        for t in range(1, stream_len + 1):
            if _WORKER_POST_CALL is not None and t > cp:
                raw = _WORKER_POST_CALL(local_rng)
                if post_mode is None:
                    post_mode = _infer_observation_mode(raw, n_features)
                x = _normalise_observation_with_mode(
                    raw,
                    post_mode,
                    n_features,
                    scalar_buffer,
                )
            else:
                raw = _WORKER_PRE_CALL(local_rng)
                if pre_mode is None:
                    pre_mode = _infer_observation_mode(raw, n_features)
                x = _normalise_observation_with_mode(
                    raw,
                    pre_mode,
                    n_features,
                    scalar_buffer,
                )
            state, output = _WORKER_DETECTOR.update(state, x)

            if task == "max":
                temp = float(output["max_score"])
                if temp > max_score:
                    max_score = temp
            elif bool(output["alarm"]):
                out[local_idx] = t - 1
                break
        else:
            if task == "alarm":
                out[local_idx] = stream_len

        if task == "max":
            out[local_idx] = max_score

    return start, out


def _mc_max_scores_chunk_from_samples(
    detector: GridDetector,
    X_chunk: np.ndarray,
) -> np.ndarray:
    """Compute max scores for a pre-generated sample chunk."""
    n_local = X_chunk.shape[0]
    stream_len = X_chunk.shape[1]
    out = np.empty(n_local, dtype=np.float64)

    for i in range(n_local):
        state = detector.init_state()
        max_score = 0.0
        for t in range(stream_len):
            state, output = detector.update(state, X_chunk[i, t, :])
            temp = float(output["max_score"])
            if temp > max_score:
                max_score = temp
        out[i] = max_score

    return out


def _mc_alarm_times_chunk_from_samples(
    detector: GridDetector,
    X_chunk: np.ndarray,
) -> np.ndarray:
    """Compute alarm times for a pre-generated sample chunk."""
    n_local = X_chunk.shape[0]
    stream_len = X_chunk.shape[1]
    out = np.full(n_local, stream_len, dtype=np.int64)

    for i in range(n_local):
        state = detector.init_state()
        for t in range(stream_len):
            state, output = detector.update(state, X_chunk[i, t, :])
            if bool(output["alarm"]):
                out[i] = t
                break

    return out


def _normalize_rng(rng: RNGInput) -> np.random.Generator:
    """Return a NumPy Generator from supported rng inputs.

    Supported inputs are exactly:
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


def _infer_observation_mode(x: Any, n_features: int) -> str:
    """Infer whether sampler output is scalar-like or vector-like."""
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 0:
        return "scalar"

    flat = x_arr.reshape(-1)
    if flat.size != n_features:
        raise ValueError(
            "Sampler output has wrong size "
            f"{flat.size}; expected scalar or size {n_features}."
        )
    return "vector"


def _normalise_observation_with_mode(
    x: Any,
    mode: str,
    n_features: int,
    scalar_buffer: np.ndarray,
) -> np.ndarray:
    """Normalize observation using a previously inferred output mode.

    This hot-path helper intentionally trusts the inferred mode from an early
    sample and avoids repeated shape validation per draw.
    """
    if mode == "scalar":
        scalar_buffer.fill(float(x))
        return scalar_buffer

    # Fast path: already a 1D float64 NumPy vector.
    if isinstance(x, np.ndarray) and x.dtype == np.float64 and x.ndim == 1:
        return x

    # Match documented behavior for vector-like observations.
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _validate_sampler_preflight(
    sampler: Callable[..., Any],
    *,
    sampler_name: str,
    n_features: int,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> None:
    """Validate that one sampler call returns a value compatible with n_features."""
    call = _make_sampler_caller(sampler, args, kwargs)
    probe_rng = np.random.default_rng(DEFAULT_MC_SEED)
    sample = call(probe_rng)

    try:
        sample_arr = np.asarray(sample, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{sampler_name} must return a scalar or an array-like value that "
            f"can be converted to float64; got {type(sample).__name__}."
        ) from exc

    if sample_arr.ndim == 0:
        if n_features == 1:
            return
        raise ValueError(
            f"{sampler_name} returned a scalar, but n_features={n_features}; "
            f"expected array-like output with total size {n_features}."
        )

    sample_flat = sample_arr.reshape(-1)
    if sample_flat.size != n_features:
        raise ValueError(
            f"{sampler_name} must return scalar (only when n_features=1) or "
            f"array-like output with total size {n_features}; "
            f"got size {sample_flat.size}."
        )


def _resolve_changepoint(
    changepoint: ChangepointSpec,
    rng: np.random.Generator,
    n_samples: int,
    path_index: int,
) -> int:
    """Resolve changepoint to an integer in ``[0, n_samples]``.

    The changepoint is the first post-change index (0-based).
    For example, changepoint=k means observations ``[0, k)`` are pre-change,
    and observations ``[k, n_samples)`` are post-change.
    - If changepoint=0, all observations are post-change.
    - If changepoint=n_samples, all observations are pre-change.

        ``changepoint`` may be:
        - ``None`` (treated as n_samples, all pre-change)
        - an integer in ``[0, n_samples]``
        - a callable ``f(rng, n_samples, path_index) -> int`` returning
            ``[0, n_samples]``
    """
    if changepoint is None:
        return n_samples

    if callable(changepoint):
        cp = int(changepoint(rng, n_samples, path_index))
    else:
        cp = int(changepoint)

    if cp < 0 or cp > n_samples:
        raise ValueError(f"changepoint must be in [0, {n_samples}], got {cp}.")
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
        Observation dimension (length of the per-time-step 1D observation
        vector used internally).
    pre_sampler : callable
                Baseline sampler callable.
                The returned sample is normalized as follows:
                - scalar -> broadcast to shape ``(n_features,)``
                - non-scalar -> flattened to 1D and required to have size
                    ``n_features``
                In all cases, values are converted to ``float64``.
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
        First post-change index in each stream (0-based). If changepoint=k,
        observations ``[0, k)`` use pre-change sampler and
        observations ``[k, stream_len)`` use post-change sampler.
        Allowed range: [0, stream_len], where 0 means all post-change and
        stream_len means all pre-change.
        If callable, called as ``changepoint(rng, stream_len, path_idx)`` and
        must return an integer in ``[0, stream_len]``.

    Returns
    -------
    np.ndarray
        Simulated array with shape ``(n_paths, stream_len, n_features)``.
        This reflects the vector-oriented internal representation.
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
    pre_call = _make_sampler_caller(pre_sampler, pre_args, pre_kwargs)
    post_call: Callable[[np.random.Generator], Any] | None = None
    if post_sampler is not None:
        post_call = _make_sampler_caller(post_sampler, post_args, post_kwargs)
    out = np.empty((n_paths, stream_len, n_features), dtype=np.float64)

    pre_mode: str | None = None
    post_mode: str | None = None
    scalar_buffer = np.empty(n_features, dtype=np.float64)

    for path_idx in range(n_paths):
        cp = _resolve_changepoint(changepoint, local_rng, stream_len, path_idx)
        # Note: t represents the SAMPLE SIZE (1-indexed; current observation count).
        # This ranges from 1 to stream_len, not 0 to stream_len-1.
        # When storing in arrays, use (t - 1) for 0-indexed array access.
        for t in range(1, stream_len + 1):
            if post_call is not None and t > cp:
                raw = post_call(local_rng)
                if post_mode is None:
                    post_mode = _infer_observation_mode(raw, n_features)
                mode = post_mode
            else:
                raw = pre_call(local_rng)
                if pre_mode is None:
                    pre_mode = _infer_observation_mode(raw, n_features)
                mode = pre_mode

            if mode == "scalar":
                out[path_idx, t - 1, :] = float(raw)
            else:
                out[path_idx, t - 1, :] = _normalise_observation_with_mode(
                    raw,
                    mode,
                    n_features,
                    scalar_buffer,
                )

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
    parallel: bool = True,
    n_jobs: int | None = None,
    strict_equivalence: bool = False,
) -> np.ndarray:
    """Run Monte Carlo paths and return the maximum score for each path.

    Reproducibility follows the ``rng`` argument:
    - ``Generator``: continues from its current state.
    - ``int``: deterministic run from that seed.
    - ``None``: deterministic run from an internal fixed default seed.

    Parallel behavior
    -----------------
    ``parallel=True`` uses process-based execution and serializes samplers with
    cloudpickle so notebook-defined callables can run in worker processes.
    ``n_jobs=None`` uses automatic core detection.

    If ``strict_equivalence=True``, sample generation is run serially and only
    detector evaluation is parallelized, which guarantees identical output to
    ``parallel=False``.

        Input requirements
        ------------------
        ``rng`` must be one of ``numpy.random.Generator``, ``int``, or ``None``.

        ``changepoint`` must be one of:
        - ``None`` (all pre-change)
                - ``int`` in ``[0, stream_len]`` (first post-change index, 0-based);
                    ``0`` = all post-change, ``stream_len`` = all pre-change.
        - callable ``f(rng, stream_len, path_index) -> int`` returning
            ``[0, stream_len]``
    """
    if n_features is None:
        score_n_features = getattr(detector.score, "n_features", None)
        if score_n_features is None:
            raise ValueError(
                "n_features was not provided and could not be inferred "
                "from detector.score."
            )
        n_features = int(score_n_features)

    max_scores = np.empty(n_paths, dtype=np.float64)

    if pre_kwargs is None:
        pre_kwargs = {}
    if post_kwargs is None:
        post_kwargs = {}

    _validate_sampler_preflight(
        pre_sampler,
        sampler_name="pre_sampler",
        n_features=n_features,
        args=pre_args,
        kwargs=pre_kwargs,
    )
    if post_sampler is not None:
        _validate_sampler_preflight(
            post_sampler,
            sampler_name="post_sampler",
            n_features=n_features,
            args=post_args,
            kwargs=post_kwargs,
        )

    if strict_equivalence:
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

        n_workers = _resolve_n_jobs(n_jobs=n_jobs, n_paths=n_paths) if parallel else 1
        if n_workers == 1:
            return _mc_max_scores_chunk_from_samples(detector, X)

        chunks = _path_index_chunks(n_paths=n_paths, n_jobs=n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _mc_max_scores_chunk_from_samples,
                    detector,
                    X[start:end, :, :],
                )
                for (start, end) in chunks
            ]

            for (start, end), fut in zip(chunks, futures):
                max_scores[start:end] = fut.result()
        return max_scores

    if not parallel:
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

    n_workers = _resolve_n_jobs(n_jobs=n_jobs, n_paths=n_paths)
    if n_workers == 1:
        return mc_max_scores(
            detector=detector,
            n_paths=n_paths,
            stream_len=stream_len,
            pre_sampler=pre_sampler,
            rng=rng,
            pre_args=pre_args,
            pre_kwargs=pre_kwargs,
            post_sampler=post_sampler,
            post_args=post_args,
            post_kwargs=post_kwargs,
            changepoint=changepoint,
            n_features=n_features,
            parallel=False,
            n_jobs=1,
            strict_equivalence=False,
        )

    base_seed = _derive_base_seed(rng)
    chunks = _path_index_chunks(n_paths=n_paths, n_jobs=n_workers)
    chunk_seeds = _spawn_chunk_seeds(base_seed, n_chunks=len(chunks))

    initargs = (
        _serialise_for_worker(detector),
        _serialise_for_worker(pre_sampler),
        _serialise_for_worker(pre_args),
        _serialise_for_worker(pre_kwargs),
        _serialise_for_worker(post_sampler),
        _serialise_for_worker(post_args),
        _serialise_for_worker(post_kwargs),
        _serialise_for_worker(changepoint),
        int(n_features),
    )

    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_mc_worker,
            initargs=initargs,
        ) as executor:
            futures = [
                executor.submit(
                    _mc_worker_chunk,
                    "max",
                    start,
                    end,
                    chunk_seeds[idx],
                    stream_len,
                    n_features,
                )
                for idx, (start, end) in enumerate(chunks)
            ]

            for fut in as_completed(futures):
                start, values = fut.result()
                max_scores[start : start + values.size] = values
    except Exception as exc:
        warnings.warn(
            "Parallel Monte Carlo failed; falling back to serial execution. "
            f"Original error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return mc_max_scores(
            detector=detector,
            n_paths=n_paths,
            stream_len=stream_len,
            pre_sampler=pre_sampler,
            rng=rng,
            pre_args=pre_args,
            pre_kwargs=pre_kwargs,
            post_sampler=post_sampler,
            post_args=post_args,
            post_kwargs=post_kwargs,
            changepoint=changepoint,
            n_features=n_features,
            parallel=False,
            n_jobs=1,
            strict_equivalence=False,
        )

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
    parallel: bool = True,
    n_jobs: int | None = None,
    strict_equivalence: bool = False,
) -> np.ndarray:
    """Run Monte Carlo paths and return first alarm index for each path.

    Alarm indices are 0-based. For paths with no alarm by the end of the stream,
    the returned value is ``stream_len`` (first index past the last sample).

    Reproducibility follows the ``rng`` argument:
    - ``Generator``: continues from its current state.
    - ``int``: deterministic run from that seed.
    - ``None``: deterministic run from an internal fixed default seed.

    Parallel behavior
    -----------------
    ``parallel=True`` uses process-based execution and serializes samplers with
    cloudpickle so notebook-defined callables can run in worker processes.

    If ``strict_equivalence=True``, sample generation is run serially and only
    detector evaluation is parallelized, which guarantees identical output to
    ``parallel=False``.

        Input requirements
        ------------------
        ``rng`` must be one of ``numpy.random.Generator``, ``int``, or ``None``.

        ``changepoint`` must be one of:
        - ``None`` (all pre-change)
                - ``int`` in ``[0, stream_len]`` (first post-change index, 0-based);
                    ``0`` = all post-change, ``stream_len`` = all pre-change.
        - callable ``f(rng, stream_len, path_index) -> int`` returning
            ``[0, stream_len]``
    """
    if n_features is None:
        score_n_features = getattr(detector.score, "n_features", None)
        if score_n_features is None:
            raise ValueError(
                "n_features was not provided and could not be inferred "
                "from detector.score."
            )
        n_features = int(score_n_features)

    alarm_times = np.full(n_paths, stream_len, dtype=np.int64)

    if pre_kwargs is None:
        pre_kwargs = {}
    if post_kwargs is None:
        post_kwargs = {}

    _validate_sampler_preflight(
        pre_sampler,
        sampler_name="pre_sampler",
        n_features=n_features,
        args=pre_args,
        kwargs=pre_kwargs,
    )
    if post_sampler is not None:
        _validate_sampler_preflight(
            post_sampler,
            sampler_name="post_sampler",
            n_features=n_features,
            args=post_args,
            kwargs=post_kwargs,
        )

    if strict_equivalence:
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

        n_workers = _resolve_n_jobs(n_jobs=n_jobs, n_paths=n_paths) if parallel else 1
        if n_workers == 1:
            return _mc_alarm_times_chunk_from_samples(detector, X)

        chunks = _path_index_chunks(n_paths=n_paths, n_jobs=n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _mc_alarm_times_chunk_from_samples,
                    detector,
                    X[start:end, :, :],
                )
                for (start, end) in chunks
            ]

            for (start, end), fut in zip(chunks, futures):
                alarm_times[start:end] = fut.result()
        return alarm_times

    if not parallel:
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

        for path_idx in range(n_paths):
            state = detector.init_state()
            for t in range(stream_len):
                state, output = detector.update(state, X[path_idx, t, :])
                if bool(output["alarm"]):
                    alarm_times[path_idx] = t
                    break
        return alarm_times

    n_workers = _resolve_n_jobs(n_jobs=n_jobs, n_paths=n_paths)
    if n_workers == 1:
        return mc_alarm_times(
            detector=detector,
            n_paths=n_paths,
            stream_len=stream_len,
            pre_sampler=pre_sampler,
            rng=rng,
            pre_args=pre_args,
            pre_kwargs=pre_kwargs,
            post_sampler=post_sampler,
            post_args=post_args,
            post_kwargs=post_kwargs,
            changepoint=changepoint,
            n_features=n_features,
            parallel=False,
            n_jobs=1,
            strict_equivalence=False,
        )

    base_seed = _derive_base_seed(rng)
    chunks = _path_index_chunks(n_paths=n_paths, n_jobs=n_workers)
    chunk_seeds = _spawn_chunk_seeds(base_seed, n_chunks=len(chunks))

    initargs = (
        _serialise_for_worker(detector),
        _serialise_for_worker(pre_sampler),
        _serialise_for_worker(pre_args),
        _serialise_for_worker(pre_kwargs),
        _serialise_for_worker(post_sampler),
        _serialise_for_worker(post_args),
        _serialise_for_worker(post_kwargs),
        _serialise_for_worker(changepoint),
        int(n_features),
    )

    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_mc_worker,
            initargs=initargs,
        ) as executor:
            futures = [
                executor.submit(
                    _mc_worker_chunk,
                    "alarm",
                    start,
                    end,
                    chunk_seeds[idx],
                    stream_len,
                    n_features,
                )
                for idx, (start, end) in enumerate(chunks)
            ]

            for fut in as_completed(futures):
                start, values = fut.result()
                alarm_times[start : start + values.size] = values
    except Exception as exc:
        warnings.warn(
            "Parallel Monte Carlo failed; falling back to serial execution. "
            f"Original error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return mc_alarm_times(
            detector=detector,
            n_paths=n_paths,
            stream_len=stream_len,
            pre_sampler=pre_sampler,
            rng=rng,
            pre_args=pre_args,
            pre_kwargs=pre_kwargs,
            post_sampler=post_sampler,
            post_args=post_args,
            post_kwargs=post_kwargs,
            changepoint=changepoint,
            n_features=n_features,
            parallel=False,
            n_jobs=1,
            strict_equivalence=False,
        )

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
    parallel: bool = True,
    n_jobs: int | None = None,
    strict_equivalence: bool = False,
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
        Randomness control passed to Monte Carlo simulation. Must be one of:
        ``numpy.random.Generator``, ``int`` seed, or ``None``.
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
                "n_features was not provided and could not be inferred "
                "from score.n_features."
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
        parallel=parallel,
        n_jobs=n_jobs,
        strict_equivalence=strict_equivalence,
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
    parallel: bool = True,
    n_jobs: int | None = None,
    strict_equivalence: bool = False,
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
        parallel=parallel,
        n_jobs=n_jobs,
        strict_equivalence=strict_equivalence,
    )


def with_calibrated_threshold(
    detector: GridDetector,
    threshold: float,
) -> GridDetector:
    """Return a copy of ``detector`` with updated threshold."""
    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    return replace(detector, threshold=float(threshold))
