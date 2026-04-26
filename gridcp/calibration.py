"""Calibration utilities for the new gridcp API.

This module provides calibration helpers for the new API.

- ``draw_samples``: simulate Monte Carlo streams under a pre-change distribution,
  optionally with a post-change regime.
- ``mc_max_scores``: run Monte Carlo simulations through a detector and collect
  the path-wise maximum detection score.
- ``mc_alarm_times``: return the first alarm index per Monte Carlo path.
- ``calibrate_threshold_false_alarm``: compute an empirical threshold for a score model.
- ``calibrate_detector_threshold_false_alarm``: convenience wrapper for detector objects.

Observation-shape convention
----------------------------
The Monte Carlo helpers in this module are vector-oriented: each sampled
observation is converted to a 1D ``float64`` vector of length ``n_features``.
Scalar sampler outputs are broadcast to length ``n_features``. Non-scalar
outputs are flattened with ``reshape(-1)`` and must have total size
``n_features``.

Sampler contract
----------------
All user-provided samplers (``pre_sampler``/``post_sampler``) must accept an
``rng`` argument. Supported forms are:

- ``sampler(rng, /, *args, **kwargs)`` — positional-only ``rng``; called as
  ``sampler(rng, *args, **kwargs)``.
- ``sampler(rng, *args, **kwargs)`` — ``rng`` as the first
  POSITIONAL_OR_KEYWORD parameter; also called positionally as
  ``sampler(rng, *args, **kwargs)``.  This is the most common pattern.
- ``sampler(*args, rng, **kwargs)`` — ``rng`` is keyword-only or not the first
  positional parameter; called as ``sampler(*args, rng=rng, **kwargs)``.

The rule is simple: if ``rng`` is the first positional parameter (whether
declared positional-only or not), it is passed positionally and
``pre_args``/``post_args`` fill the remaining positions in order.  Otherwise
it is passed by keyword.

This ensures all randomness is anchored in the calibrated simulation RNG and
therefore reproducible in both serial and parallel execution.

Do not pass methods bound to fixed generators (for example
``np.random.default_rng(10).normal``), since those ignore the simulation RNG.

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
    ``[0, stream_len]``. The callable must accept these three arguments in
    that order.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from functools import lru_cache
import pickle
from typing import Any, Callable, Mapping
import inspect
import os
import warnings

import numpy as np
from numpy.typing import ArrayLike

from gridcp.detector import GridDetector
from gridcp.typing import ScoreModel

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


def _sampler_rng_mode_uncached(sampler: Callable[..., Any]) -> str:
    """Return how ``rng`` must be passed to sampler.

    Returns one of:

    - ``"positional"``: ``rng`` is POSITIONAL_ONLY, **or** it is
      POSITIONAL_OR_KEYWORD and is the first positional parameter.  In both
      cases the caller uses ``sampler(local_rng, *args, **kwargs)``.
    - ``"keyword"``: ``rng`` is POSITIONAL_OR_KEYWORD but not the first
      positional parameter, or it is KEYWORD_ONLY.  The caller uses
      ``sampler(*args, rng=local_rng, **kwargs)``.
    - ``"kwargs"``: sampler has no explicit ``rng`` parameter but accepts
      ``**kwargs``; ``rng`` is injected by keyword.
    """
    try:
        sig = inspect.signature(sampler)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "sampler must be an inspectable callable that accepts an ``rng`` "
            "argument. Use a wrapper function like "
            "`def sampler(rng, ...): ...`."
        ) from exc

    params = sig.parameters
    if "rng" in params:
        kind = params["rng"].kind
        if kind == inspect.Parameter.POSITIONAL_ONLY:
            return "positional"
        if kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            # If rng is the first positional parameter, passing it positionally
            # is unambiguous: no *args can collide with it.
            positional_params = [
                p
                for p in params.values()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            if positional_params and positional_params[0].name == "rng":
                return "positional"
        return "keyword"

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return "kwargs"

    raise TypeError(
        "sampler must accept an ``rng`` argument. "
        "Use a wrapper function like `def sampler(rng, ...): ...` and pass "
        "that function."
    )


@lru_cache(maxsize=1024)
def _sampler_rng_mode_cached(sampler: Callable[..., Any]) -> str:
    """Cache sampler rng passing mode for hashable callables."""
    return _sampler_rng_mode_uncached(sampler)


def _sampler_rng_mode(sampler: Callable[..., Any]) -> str:
    """Return sampler rng passing mode, using cache when possible."""
    try:
        return _sampler_rng_mode_cached(sampler)
    except TypeError:
        return _sampler_rng_mode_uncached(sampler)


def _validate_sampler_kwargs_no_rng(
    kwargs: Mapping[str, Any],
    *,
    sampler_name: str,
) -> None:
    """Validate user kwargs do not override the internally managed rng."""
    if "rng" in kwargs:
        raise ValueError(
            f"{sampler_name} kwargs must not contain 'rng'. "
            "The simulation engine provides rng internally."
        )


def _make_sampler_caller(
    sampler: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    sampler_name: str = "sampler",
) -> Callable[[np.random.Generator], Any]:
    """Build a fast callable ``f(rng)`` for a sampler."""
    _validate_sampler_kwargs_no_rng(kwargs, sampler_name=sampler_name)
    mode = _sampler_rng_mode(sampler)

    if mode == "positional":
        return lambda local_rng: sampler(local_rng, *args, **kwargs)
    if mode in {"keyword", "kwargs"}:
        return lambda local_rng: sampler(*args, rng=local_rng, **kwargs)

    raise RuntimeError(
        f"Unexpected sampler rng passing mode for {sampler_name}: {mode}."
    )


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


def _serialize_for_worker(obj: Any) -> bytes:
    """Serialize an object for process workers using cloudpickle."""
    cloudpickle = _get_cloudpickle()
    return cloudpickle.dumps(obj)


def _deserialize_from_worker(blob: bytes) -> Any:
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

    _WORKER_DETECTOR = _deserialize_from_worker(detector_blob)
    _WORKER_PRE_SAMPLER = _deserialize_from_worker(pre_sampler_blob)
    _WORKER_PRE_ARGS = _deserialize_from_worker(pre_args_blob)
    _WORKER_PRE_KWARGS = _deserialize_from_worker(pre_kwargs_blob)
    _WORKER_POST_SAMPLER = _deserialize_from_worker(post_sampler_blob)
    _WORKER_POST_ARGS = _deserialize_from_worker(post_args_blob)
    _WORKER_POST_KWARGS = _deserialize_from_worker(post_kwargs_blob)
    _WORKER_CHANGEPOINT = _deserialize_from_worker(changepoint_blob)

    # Type narrowing: these are guaranteed non-None after deserialization above.
    assert _WORKER_PRE_SAMPLER is not None
    assert _WORKER_PRE_ARGS is not None
    assert _WORKER_PRE_KWARGS is not None
    _WORKER_PRE_CALL = _make_sampler_caller(
        _WORKER_PRE_SAMPLER,
        _WORKER_PRE_ARGS,
        _WORKER_PRE_KWARGS,
        sampler_name="pre_sampler",
    )
    if _WORKER_POST_SAMPLER is not None:
        _WORKER_POST_CALL = _make_sampler_caller(
            _WORKER_POST_SAMPLER,
            _WORKER_POST_ARGS or (),
            _WORKER_POST_KWARGS or {},
            sampler_name="post_sampler",
        )
    else:
        _WORKER_POST_CALL = None

    # Warm up a tiny detector path per worker to reduce first-call JIT latency.
    try:
        assert _WORKER_DETECTOR is not None
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
    declared_k = _WORKER_DETECTOR.score.n_tests
    out: np.ndarray
    if task == "alarm":
        out = np.empty(n_local_paths, dtype=np.int64)
    else:
        out = np.empty((n_local_paths, declared_k), dtype=np.float64)

    # One RNG per worker chunk to minimize generator construction overhead.
    local_rng = np.random.default_rng(chunk_seed)

    pre_mode: str | None = None
    post_mode: str | None = None
    scalar_buffer = np.empty(n_features, dtype=np.float64)

    for local_idx in range(n_local_paths):
        path_idx = start + local_idx
        cp = _resolve_changepoint(_WORKER_CHANGEPOINT, local_rng, stream_len, path_idx)

        state = _WORKER_DETECTOR.init_state()
        max_score = np.zeros(declared_k, dtype=np.float64)

        # Note: t represents the SAMPLE SIZE (1-indexed; current observation count).
        # This ranges from 1 to stream_len, not 0 to stream_len-1.
        # When storing in arrays, use (t - 1) for 0-indexed array access.
        for t in range(1, stream_len + 1):
            if _WORKER_POST_CALL is not None and t > cp:
                raw = _WORKER_POST_CALL(local_rng)
                if post_mode is None:
                    post_mode = _infer_observation_mode(raw, n_features)
                x = _normalize_observation_with_mode(
                    raw,
                    post_mode,
                    n_features,
                    scalar_buffer,
                )
            else:
                raw = _WORKER_PRE_CALL(local_rng)
                if pre_mode is None:
                    pre_mode = _infer_observation_mode(raw, n_features)
                x = _normalize_observation_with_mode(
                    raw,
                    pre_mode,
                    n_features,
                    scalar_buffer,
                )
            state, output = _WORKER_DETECTOR.update(state, x)

            if task == "max":
                temp = np.asarray(output["max_score"], dtype=np.float64)
                np.maximum(max_score, temp, out=max_score)
            elif bool(output["alarm"]):
                out[local_idx] = t - 1
                break
        else:
            if task == "alarm":
                out[local_idx] = stream_len

        if task == "max":
            out[local_idx, :] = max_score

    return start, out


def _mc_max_scores_chunk_from_samples(
    detector: GridDetector,
    X_chunk: np.ndarray,
) -> np.ndarray:
    """Compute max scores for a pre-generated sample chunk."""
    n_local = X_chunk.shape[0]
    stream_len = X_chunk.shape[1]
    declared_k = detector.score.n_tests
    out = np.empty((n_local, declared_k), dtype=np.float64)

    for i in range(n_local):
        state = detector.init_state()
        max_score = np.zeros(declared_k, dtype=np.float64)
        for t in range(stream_len):
            state, output = detector.update(state, X_chunk[i, t, :])
            temp = np.asarray(output["max_score"], dtype=np.float64)
            np.maximum(max_score, temp, out=max_score)
        out[i, :] = max_score

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


def _warn_broadcast_size_one(*, n_features: int) -> None:
    """Warn when a size-1 observation is broadcast to multivariate shape."""
    warnings.warn(
        "Sampler output with size 1 is being broadcast to "
        f"n_features={n_features}. This often indicates a sampler bug.",
        UserWarning,
        stacklevel=3,
    )


def _normalize_observation_with_mode(
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
        if x.size == n_features:
            return x
        if x.size == 1 and n_features != 1:
            _warn_broadcast_size_one(n_features=n_features)
            scalar_buffer.fill(float(x[0]))
            return scalar_buffer
        raise ValueError(
            "Sampler output has wrong size "
            f"{x.size}; expected scalar or size {n_features}."
        )

    # Match documented behavior for vector-like observations.
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if x_arr.size == n_features:
        return x_arr
    if x_arr.size == 1 and n_features != 1:
        _warn_broadcast_size_one(n_features=n_features)
        scalar_buffer.fill(float(x_arr[0]))
        return scalar_buffer
    raise ValueError(
        "Sampler output has wrong size "
        f"{x_arr.size}; expected scalar or size {n_features}."
    )


def _validate_post_sampler_for_changepoint(
    *,
    changepoint: ChangepointSpec,
    post_sampler: Callable[..., Any] | None,
) -> None:
    """Require post_sampler whenever changepoint is specified."""
    if changepoint is not None and post_sampler is None:
        raise ValueError("post_sampler must be provided when changepoint is set.")


def _validate_sampler_preflight(
    sampler: Callable[..., Any],
    *,
    sampler_name: str,
    n_features: int,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> None:
    """Validate that one sampler call returns a value compatible with n_features.

    Scalars are accepted for any ``n_features`` and are broadcast at runtime.
    """
    call = _make_sampler_caller(
        sampler,
        args,
        kwargs,
        sampler_name=sampler_name,
    )
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
        if n_features != 1:
            warnings.warn(
                f"{sampler_name} returned a scalar and will be broadcast to "
                f"n_features={n_features}. This often indicates a sampler bug.",
                UserWarning,
                stacklevel=3,
            )
        return

    sample_flat = sample_arr.reshape(-1)
    if sample_flat.size != n_features:
        raise ValueError(
            f"{sampler_name} must return a scalar (broadcast to n_features) or "
            f"array-like output with total size {n_features}; "
            f"got size {sample_flat.size}."
        )


def _validate_changepoint_preflight(
    changepoint: ChangepointSpec,
    *,
    stream_len: int,
) -> None:
    """Validate changepoint type, callable signature, and value range."""
    if changepoint is None:
        return

    if callable(changepoint):
        try:
            sig = inspect.signature(changepoint)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "changepoint must be an inspectable callable with signature "
                "`def changepoint(rng, stream_len, path_index): ...`."
            ) from exc

        probe_rng = np.random.default_rng(DEFAULT_MC_SEED)
        try:
            sig.bind(probe_rng, stream_len, 0)
        except TypeError as exc:
            raise TypeError(
                "changepoint callable must accept arguments "
                "(rng, stream_len, path_index)."
            ) from exc

        cp_raw = changepoint(probe_rng, stream_len, 0)
    else:
        cp_raw = changepoint

    try:
        cp = int(cp_raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "changepoint must be None, an integer, or a callable returning an integer."
        ) from exc

    if cp < 0 or cp > stream_len:
        raise ValueError(f"changepoint must be in [0, {stream_len}], got {cp}.")


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
                Baseline sampler callable. Must accept ``rng`` as an argument,
                i.e. use ``def pre_sampler(rng, ...): ...``.
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
        Post-change sampler. Must also accept ``rng`` as an argument.
        If ``None``, all observations are pre-change.
    post_args, post_kwargs : optional
        Additional arguments for ``post_sampler``.
    changepoint : int | callable | None, optional
        First post-change index in each stream (0-based). If changepoint=k,
        observations ``[0, k)`` use pre-change sampler and
        observations ``[k, stream_len)`` use post-change sampler.
        Allowed range: [0, stream_len], where 0 means all post-change and
        stream_len means all pre-change.
        If callable, called as ``changepoint(rng, stream_len, path_idx)`` and
        must return an integer in ``[0, stream_len]``. Callable signature is
        validated before simulation starts.

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

    _validate_post_sampler_for_changepoint(
        changepoint=changepoint,
        post_sampler=post_sampler,
    )

    _validate_changepoint_preflight(changepoint, stream_len=stream_len)

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

    local_rng = _normalize_rng(rng)
    pre_call = _make_sampler_caller(
        pre_sampler,
        pre_args,
        pre_kwargs,
        sampler_name="pre_sampler",
    )
    post_call: Callable[[np.random.Generator], Any] | None = None
    if post_sampler is not None:
        post_call = _make_sampler_caller(
            post_sampler,
            post_args,
            post_kwargs,
            sampler_name="post_sampler",
        )
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
                out[path_idx, t - 1, :] = _normalize_observation_with_mode(
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

    Sampler contract
    ----------------
    ``pre_sampler`` and ``post_sampler`` (when provided) must accept ``rng``
    as an argument, i.e. ``sampler(rng, *args, **kwargs)``.

    Sampler output normalization
    ----------------------------
    Per draw, sampler output is normalized exactly as in :func:`draw_samples`:
    scalar outputs are broadcast to length ``n_features`` and non-scalar
    outputs are flattened and required to have total size ``n_features``.

    If ``changepoint`` is provided, ``post_sampler`` must also be provided.

        Input requirements
        ------------------
        ``rng`` must be one of ``numpy.random.Generator``, ``int``, or ``None``.

        ``changepoint`` must be one of:
        - ``None`` (all pre-change)
                - ``int`` in ``[0, stream_len]`` (first post-change index, 0-based);
                    ``0`` = all post-change, ``stream_len`` = all pre-change.
        - callable ``f(rng, stream_len, path_index) -> int`` returning
            ``[0, stream_len]``. The callable must accept these three
            arguments in that order.
    """
    n_features: int = detector.score.n_features

    # Output array allocated lazily; shape is always (n_paths, K), including K=1.
    max_scores: np.ndarray | None = None

    if pre_kwargs is None:
        pre_kwargs = {}
    if post_kwargs is None:
        post_kwargs = {}

    _validate_post_sampler_for_changepoint(
        changepoint=changepoint,
        post_sampler=post_sampler,
    )

    _validate_changepoint_preflight(changepoint, stream_len=stream_len)

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

            results = [fut.result() for (_, _), fut in zip(chunks, futures)]
            max_scores = np.empty((n_paths, detector.score.n_tests), dtype=np.float64)
            for (start, end), res in zip(chunks, results):
                max_scores[start:end] = res
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

        return _mc_max_scores_chunk_from_samples(detector, X)

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
            parallel=False,
            n_jobs=1,
            strict_equivalence=False,
        )

    base_seed = _derive_base_seed(rng)
    chunks = _path_index_chunks(n_paths=n_paths, n_jobs=n_workers)
    chunk_seeds = _spawn_chunk_seeds(base_seed, n_chunks=len(chunks))
    max_scores = np.empty((n_paths, detector.score.n_tests), dtype=np.float64)

    initargs = (
        _serialize_for_worker(detector),
        _serialize_for_worker(pre_sampler),
        _serialize_for_worker(pre_args),
        _serialize_for_worker(pre_kwargs),
        _serialize_for_worker(post_sampler),
        _serialize_for_worker(post_args),
        _serialize_for_worker(post_kwargs),
        _serialize_for_worker(changepoint),
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
                n_vals = values.shape[0]
                max_scores[start : start + n_vals] = values
    except (BrokenProcessPool, OSError, pickle.PicklingError) as exc:
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
    parallel: bool = True,
    n_jobs: int | None = None,
    strict_equivalence: bool = False,
) -> np.ndarray:
    """Run Monte Carlo paths and return first alarm index for each path.

    Alarm indices are 0-based. For paths with no alarm by the end of the stream,
    the returned value is ``stream_len`` (first index past the last sample).

    This function is intentionally first-alarm only: each path contributes one
    value, the first time the detector alarms. It does not summarize performance
    for multiple changepoints within the same path after the first detection.
    For multi-changepoint evaluation, run a custom simulation loop that defines
    post-alarm behavior explicitly (for example reset/refractory policies and
    per-segment detection-delay metrics).

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

    Sampler contract
    ----------------
    ``pre_sampler`` and ``post_sampler`` (when provided) must accept ``rng``
    as an argument, i.e. ``sampler(rng, *args, **kwargs)``.

    Sampler output normalization
    ----------------------------
    Per draw, sampler output is normalized exactly as in :func:`draw_samples`:
    scalar outputs are broadcast to length ``n_features`` and non-scalar
    outputs are flattened and required to have total size ``n_features``.

    If ``changepoint`` is provided, ``post_sampler`` must also be provided.

        Input requirements
        ------------------
        ``rng`` must be one of ``numpy.random.Generator``, ``int``, or ``None``.

        ``changepoint`` must be one of:
        - ``None`` (all pre-change)
                - ``int`` in ``[0, stream_len]`` (first post-change index, 0-based);
                    ``0`` = all post-change, ``stream_len`` = all pre-change.
        - callable ``f(rng, stream_len, path_index) -> int`` returning
            ``[0, stream_len]``. The callable must accept these three
            arguments in that order.
    """
    n_features: int = detector.score.n_features

    alarm_times = np.full(n_paths, stream_len, dtype=np.int64)

    if pre_kwargs is None:
        pre_kwargs = {}
    if post_kwargs is None:
        post_kwargs = {}

    _validate_post_sampler_for_changepoint(
        changepoint=changepoint,
        post_sampler=post_sampler,
    )

    _validate_changepoint_preflight(changepoint, stream_len=stream_len)

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
            parallel=False,
            n_jobs=1,
            strict_equivalence=False,
        )

    base_seed = _derive_base_seed(rng)
    chunks = _path_index_chunks(n_paths=n_paths, n_jobs=n_workers)
    chunk_seeds = _spawn_chunk_seeds(base_seed, n_chunks=len(chunks))

    initargs = (
        _serialize_for_worker(detector),
        _serialize_for_worker(pre_sampler),
        _serialize_for_worker(pre_args),
        _serialize_for_worker(pre_kwargs),
        _serialize_for_worker(post_sampler),
        _serialize_for_worker(post_args),
        _serialize_for_worker(post_kwargs),
        _serialize_for_worker(changepoint),
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
    except (BrokenProcessPool, OSError, pickle.PicklingError) as exc:
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
            parallel=False,
            n_jobs=1,
            strict_equivalence=False,
        )

    return alarm_times


def calibrate_threshold_false_alarm(
    score: ScoreModel,
    *,
    false_alarm_probability: float,
    n_paths: int,
    stream_len: int,
    pre_sampler: Callable[..., Any],
    rng: RNGInput = None,
    pre_args: tuple[Any, ...] = (),
    pre_kwargs: Mapping[str, Any] | None = None,
    parallel: bool = True,
    n_jobs: int | None = None,
    strict_equivalence: bool = False,
    apply_bonferroni: bool = True,
) -> np.ndarray:
    """Estimate a score threshold from Monte Carlo max scores under the null.

    Applies a Bonferroni correction by
    default: each per-test threshold is set at the ``(1 - alpha/K)`` quantile,
    and the result is a 1-D array of shape ``(K,)`` (including ``K=1``).
    This can be disabled by
    setting ``apply_bonferroni=False``.

    Parameters
    ----------
    score : Any
        Score model compatible with ``GridDetector``.
    false_alarm_probability : float
        Target false alarm probability in ``(0, 1)``.
    n_paths : int
        Number of Monte Carlo paths.
    stream_len : int
        Number of samples per path.
    pre_sampler : callable
        Null sampler with signature ``pre_sampler(rng, *args, **kwargs)``.
        Returned values follow :func:`draw_samples` normalization:
        scalar outputs are broadcast to ``(n_features,)`` and non-scalar
        outputs are flattened and must have total size ``n_features``.
    rng : numpy.random.Generator | int | None, optional
        Randomness control passed to Monte Carlo simulation. Must be one of:
        ``numpy.random.Generator``, ``int`` seed, or ``None``.
        ``None`` uses a fixed default seed for deterministic behavior.
    pre_args, pre_kwargs : optional
        Additional arguments passed to ``pre_sampler``.
    apply_bonferroni : bool, optional
        For multivariate scores, whether to apply Bonferroni correction
        (default=True). If False, each component uses the same quantile
        level ``(1 - alpha)``.
    """
    if not (0.0 < false_alarm_probability < 1.0):
        raise ValueError("false_alarm_probability must be in (0, 1).")

    # Threshold calibration must use a non-resetting detector so Monte Carlo
    # paths estimate the intended long-run false-alarm behavior.
    detector = GridDetector(
        score=score,
        threshold=1.0,
    )

    max_scores = mc_max_scores(
        detector=detector,
        n_paths=n_paths,
        stream_len=stream_len,
        pre_sampler=pre_sampler,
        rng=rng,
        pre_args=pre_args,
        pre_kwargs=pre_kwargs,
        parallel=parallel,
        n_jobs=n_jobs,
        strict_equivalence=strict_equivalence,
    )

    return _compute_false_alarm_threshold_from_max_scores(
        max_scores, false_alarm_probability, apply_bonferroni
    )


def calibrate_detector_threshold_false_alarm(
    detector: GridDetector,
    *,
    false_alarm_probability: float,
    n_paths: int,
    stream_len: int,
    pre_sampler: Callable[..., Any],
    rng: RNGInput = None,
    pre_args: tuple[Any, ...] = (),
    pre_kwargs: Mapping[str, Any] | None = None,
    parallel: bool = True,
    n_jobs: int | None = None,
    strict_equivalence: bool = False,
    apply_bonferroni: bool = True,
) -> np.ndarray:
    """Estimate threshold for an existing detector.

    This is a convenience wrapper around :func:`calibrate_threshold_false_alarm`.
    In particular, ``pre_sampler`` must satisfy the same sampler contract:
    ``pre_sampler(rng, *args, **kwargs)`` with output normalization identical
    to :func:`draw_samples`.

    Notes
    -----
    Calibration always uses an internal non-resetting detector constructed from
    ``detector.score`` and therefore never depends on external reset policy.
    """
    return calibrate_threshold_false_alarm(
        score=detector.score,
        false_alarm_probability=false_alarm_probability,
        n_paths=n_paths,
        stream_len=stream_len,
        pre_sampler=pre_sampler,
        rng=rng,
        pre_args=pre_args,
        pre_kwargs=pre_kwargs,
        parallel=parallel,
        n_jobs=n_jobs,
        strict_equivalence=strict_equivalence,
        apply_bonferroni=apply_bonferroni,
    )


def with_calibrated_threshold(
    detector: GridDetector,
    threshold: float | np.ndarray,
) -> GridDetector:
    """Return a copy of ``detector`` with updated threshold.

    The threshold is normalised to a 1-D ``float64`` array by
    ``GridDetector.__post_init__``.
    """
    return replace(detector, threshold=threshold)


# ---------------------------------------------------------------------------
# Data-driven (bootstrap) calibration
# ---------------------------------------------------------------------------


def _block_bootstrap_samples(
    data: np.ndarray,
    n_paths: int,
    stream_len: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate circular block-bootstrap resampled streams.

    Uses the circular block bootstrap (Politis & Romano, 1992): blocks of
    length ``block_length`` are drawn with uniformly random start indices and
    wrapped around the end of the training data, ensuring every observation
    appears in the same number of blocks.

    Parameters
    ----------
    data : np.ndarray, shape ``(N, n_features)``
        Training data (assumed change-free and stationary).
    n_paths : int
        Number of bootstrap paths to generate.
    stream_len : int
        Length of each resampled stream.
    block_length : int
        Block length for the circular block bootstrap.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray, shape ``(n_paths, stream_len, n_features)``
        Resampled observation streams.
    """
    n_obs = data.shape[0]
    n_features = data.shape[1]
    out = np.empty((n_paths, stream_len, n_features), dtype=np.float64)

    for path in range(n_paths):
        t = 0
        while t < stream_len:
            start = int(rng.integers(0, n_obs))
            length = min(block_length, stream_len - t)
            indices = np.arange(start, start + length) % n_obs
            out[path, t : t + length, :] = data[indices]
            t += length

    return out


def _compute_false_alarm_threshold_from_max_scores(
    max_scores: np.ndarray,
    false_alarm_probability: float,
    apply_bonferroni: bool,
) -> np.ndarray:
    """Compute threshold(s) from an array of per-path max scores.

    Parameters
    ----------
    max_scores : np.ndarray
        Shape ``(n_paths, K)``, including ``K=1``.
    false_alarm_probability : float
        Target false alarm probability in ``(0, 1)``.
    apply_bonferroni : bool
        For multivariate scores, whether to apply Bonferroni correction.

    Returns
    -------
    np.ndarray
        Threshold vector of shape ``(K,)``.
    """
    if max_scores.ndim != 2:
        raise ValueError(
            f"max_scores must have shape (n_paths, K); got shape {max_scores.shape}."
        )

    n_tests = max_scores.shape[1]
    if apply_bonferroni:
        alpha_corrected = false_alarm_probability / n_tests
    else:
        alpha_corrected = false_alarm_probability
    quantile_level = 1.0 - alpha_corrected
    return np.array(
        [float(np.quantile(max_scores[:, k], quantile_level)) for k in range(n_tests)],
        dtype=np.float64,
    )


def _compute_arl_threshold_from_max_scores(
    max_scores: np.ndarray,
    max_scores_step2: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ARL threshold(s) from an array of per-path max scores.

    Parameters
    ----------
    max_scores : np.ndarray
        Shape ``(n_paths, K)``, including ``K=1``. Always used for step 1
        (per-test quantiles).
    max_scores_step2 : np.ndarray or None, optional
        If provided (multivariate only), used for step 2 (joint scaling) in
        place of ``max_scores``.  Ignored for scalar scores.

    Returns
    -------
    np.ndarray
        Threshold vector of shape ``(K,)``.
    """
    if max_scores.ndim != 2:
        raise ValueError(
            f"max_scores must have shape (n_paths, K); got shape {max_scores.shape}."
        )

    # K > 1 — two-step procedure.
    # Step 1: per-test (1/e)-quantiles.
    n_tests = max_scores.shape[1]
    if n_tests == 1:
        return np.array([float(np.quantile(max_scores[:, 0], 1.0 / np.e))])

    individual_thresholds = np.array(
        [float(np.quantile(max_scores[:, k], 1.0 / np.e)) for k in range(n_tests)],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(individual_thresholds)) or np.any(
        individual_thresholds <= 0
    ):
        raise ValueError(
            "Multivariate ARL calibration requires strictly positive finite "
            "per-test thresholds; got non-finite or non-positive values in "
            "`individual_thresholds`. This can happen when score maxima are "
            "often zero or the calibration sample is too small."
        )

    # Step 2: scale to satisfy the joint 1/e-quantile criterion.
    # Use max_scores_step2 if provided (independent simulation), else reuse max_scores.
    scores_for_scaling = (
        max_scores_step2 if max_scores_step2 is not None else max_scores
    )
    standardized = scores_for_scaling / individual_thresholds  # (n_paths, K)
    combined_max = np.max(standardized, axis=1)  # (n_paths,)
    c = float(np.quantile(combined_max, 1.0 / np.e))
    return c * individual_thresholds


def _warn_if_non_constant_penalty(score: ScoreModel) -> None:
    enable_penalty = getattr(score, "enable_penalty", None)
    if enable_penalty:
        warnings.warn(
            "ARL calibration requires a stationary score. "
            "The supplied score uses a time-dependent penalty, which will "
            "produce incorrect ARL results. Set enable_penalty=False.",
            UserWarning,
            stacklevel=3,
        )


def _eval_max_scores(
    score: ScoreModel,
    samples: np.ndarray,
    *,
    parallel: bool,
    n_jobs: int | None,
) -> np.ndarray:
    """Run pre-generated samples through the detector and return per-path max scores.

    Parameters
    ----------
    score : ScoreModel
        Score model compatible with ``GridDetector``.
    samples : np.ndarray
        Shape ``(n_paths, stream_len, n_features)``.  Assumed already validated.
    parallel : bool
        Whether to evaluate paths in parallel.
    n_jobs : int or None
        Number of worker processes.

    Returns
    -------
    np.ndarray
        Shape ``(n_paths, K)``, including ``K=1``.
    """
    n_paths = samples.shape[0]
    detector = GridDetector(score=score, threshold=1.0)

    n_workers = _resolve_n_jobs(n_jobs=n_jobs, n_paths=n_paths) if parallel else 1
    if n_workers <= 1:
        return _mc_max_scores_chunk_from_samples(detector, samples)

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
        return np.concatenate(results, axis=0)
    except (BrokenProcessPool, OSError, pickle.PicklingError) as exc:
        warnings.warn(
            "Parallel evaluation failed; falling back to serial execution. "
            f"Original error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _mc_max_scores_chunk_from_samples(detector, samples)


def calibrate_threshold_false_alarm_from_samples(
    score: ScoreModel,
    samples: ArrayLike,
    *,
    false_alarm_probability: float,
    apply_bonferroni: bool = True,
    parallel: bool = True,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Estimate a threshold from pre-generated sample paths.

    This is the most flexible data-driven calibration entry point.  The caller
    supplies an array of observation streams (e.g. generated by any bootstrap
    method, including external packages such as ``tsbootstrap``) and the
    function runs each stream through the detector to build the null
    distribution of maximum scores.

    Parameters
    ----------
    score : ScoreModel
        Score model compatible with ``GridDetector``.
    samples : ArrayLike
        Pre-generated observation streams with shape
        ``(n_paths, stream_len, n_features)``.  Lists, tuples, and other
        array-like inputs are accepted and converted to ``float64``.
    false_alarm_probability : float
        Target false alarm probability in ``(0, 1)``.
    apply_bonferroni : bool, optional
        For multivariate scores (K tests), apply Bonferroni correction so
        each per-test threshold uses the ``(1 - alpha/K)`` quantile
        (default ``True``).
    parallel : bool, optional
        If ``True`` (default), evaluate sample paths across multiple
        processes.
    n_jobs : int or None, optional
        Number of worker processes.  ``None`` (default) uses all available
        CPU cores.

    Returns
    -------
    np.ndarray
        Estimated threshold vector of shape ``(K,)`` (including ``K=1``).

    Examples
    --------
    Using an external bootstrap package (e.g. ``tsbootstrap``)::

        from tsbootstrap import CircularBlockBootstrap
        import numpy as np

        boot = CircularBlockBootstrap(n_bootstraps=1000, block_length=10)
        samples = np.stack([
            s[:stream_len] for s in boot.bootstrap(training_data)
        ])
        threshold = calibrate_threshold_false_alarm_from_samples(
            score=my_score,
            samples=samples,
            false_alarm_probability=0.05,
        )
    """
    if not (0.0 < false_alarm_probability < 1.0):
        raise ValueError("false_alarm_probability must be in (0, 1).")

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
    # Threshold calibration must use a non-resetting detector so Monte Carlo
    # paths estimate the intended long-run false-alarm behavior.
    detector = GridDetector(
        score=score,
        threshold=1.0,
    )

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

    return _compute_false_alarm_threshold_from_max_scores(
        max_scores, false_alarm_probability, apply_bonferroni
    )


def calibrate_threshold_false_alarm_from_data(
    score: ScoreModel,
    training_data: ArrayLike,
    *,
    false_alarm_probability: float,
    stream_len: int,
    n_paths: int = 1000,
    block_length: int | None = None,
    rng: RNGInput = None,
    apply_bonferroni: bool = True,
    parallel: bool = True,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Estimate a threshold by circular block-bootstrapping training data.

    When the null distribution is unknown but a representative change-free
    training set is available, this function calibrates the threshold by:

    1.  Resampling ``n_paths`` streams of length ``stream_len`` from
        ``training_data`` using the circular block bootstrap (Politis &
        Romano, 1992).
    2.  Running each stream through the detector and recording the path-wise
        maximum score.
    3.  Setting the threshold at the empirical
        ``(1 - false_alarm_probability)`` quantile.

    Block length
    ------------
    ``block_length`` controls the trade-off between preserving temporal
    dependence and bootstrap variability:

    - ``block_length=1``: i.i.d. bootstrap (appropriate when observations
      are independent).
    - ``block_length > 1``: preserves short-range autocorrelation.
    - ``block_length=None`` (default): automatic selection via
      ``max(1, int(N ** (1/3)))``, following the asymptotic-optimal rate
      for block bootstraps under weak dependence (Hall, Horowitz & Jing,
      1995; Lahiri, 2003).

    Parameters
    ----------
    score : ScoreModel
        Score model compatible with ``GridDetector``.
    training_data : ArrayLike
        Change-free training observations, shape ``(N,)`` for univariate or
        ``(N, n_features)`` for multivariate data.  Lists, tuples, and other
        array-like inputs are accepted and converted to ``float64``.
    false_alarm_probability : float
        Target false alarm probability in ``(0, 1)``.
    stream_len : int
        Length of each bootstrapped stream fed to the detector.
    n_paths : int, optional
        Number of bootstrap replications (default 1000).
    block_length : int or None, optional
        Block length for the circular block bootstrap.  ``None`` selects
        ``max(1, int(N ** (1/3)))`` automatically.
    rng : numpy.random.Generator | int | None, optional
        Randomness control.
    apply_bonferroni : bool, optional
        For multivariate scores (K tests), apply Bonferroni correction
        (default ``True``).
    parallel : bool, optional
        If ``True`` (default), evaluate sample paths across multiple
        processes.
    n_jobs : int or None, optional
        Number of worker processes.  ``None`` (default) uses all available
        CPU cores.

    Returns
    -------
    np.ndarray
        Estimated threshold vector of shape ``(K,)`` (including ``K=1``).
    """
    if not (0.0 < false_alarm_probability < 1.0):
        raise ValueError("false_alarm_probability must be in (0, 1).")
    if stream_len < 1:
        raise ValueError("stream_len must be >= 1.")
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

    if n_obs < stream_len:
        warnings.warn(
            f"training_data has {n_obs} observations but stream_len={stream_len}. "
            "The circular block bootstrap will wrap around, but calibration may "
            "be unreliable with so few training observations.",
            UserWarning,
            stacklevel=2,
        )

    local_rng = _normalize_rng(rng)
    samples = _block_bootstrap_samples(
        data=training_data,
        n_paths=n_paths,
        stream_len=stream_len,
        block_length=block_length,
        rng=local_rng,
    )

    # Threshold calibration must use a non-resetting detector so Monte Carlo
    # paths estimate the intended long-run false-alarm behavior.
    detector = GridDetector(
        score=score,
        threshold=1.0,
    )

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

    return _compute_false_alarm_threshold_from_max_scores(
        max_scores, false_alarm_probability, apply_bonferroni
    )


def calibrate_detector_threshold_false_alarm_from_data(
    detector: GridDetector,
    training_data: ArrayLike,
    *,
    false_alarm_probability: float,
    stream_len: int,
    n_paths: int = 1000,
    block_length: int | None = None,
    rng: RNGInput = None,
    apply_bonferroni: bool = True,
    parallel: bool = True,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Estimate threshold for an existing detector using training data.

    Convenience wrapper around :func:`calibrate_threshold_false_alarm_from_data`.
    """
    return calibrate_threshold_false_alarm_from_data(
        score=detector.score,
        training_data=training_data,
        false_alarm_probability=false_alarm_probability,
        stream_len=stream_len,
        n_paths=n_paths,
        block_length=block_length,
        rng=rng,
        apply_bonferroni=apply_bonferroni,
        parallel=parallel,
        n_jobs=n_jobs,
    )


# ---------------------------------------------------------------------------
# ARL calibration
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
    resimulate_combined_threshold: bool = False,
) -> np.ndarray:
    """Estimate a score threshold via Average Run Length (ARL) calibration.

    Simulates *n_paths* null streams of length *target_arl*, collects the
    maximum score from each, and returns the (1/e)-quantile.  This yields a
    detector whose expected alarm time under the null is approximately
    *target_arl*.

    Parameters
    ----------
    score : ScoreModel
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
    resimulate_combined_threshold : bool, default False
        For multivariate scores (K>1) only.  If ``True``, runs a second
        independent Monte Carlo simulation for step 2 of the calibration
        procedure (the joint scaling of per-test thresholds).  The second
        simulation continues from the same ``rng`` (whose state has been
        advanced by the first simulation), so step-2 samples are independent
        of step-1 samples.  This removes the sample-reuse bias between the
        two estimation steps at the cost of twice the computation.  Has no
        effect for ``K=1``.

    Returns
    -------
    np.ndarray
        ARL threshold vector of shape ``(K,)`` (including ``K=1``), with
        combined thresholds that control the joint ARL.

    Notes
    -----
    The (1/e)-quantile arises because, under the exponential approximation
    for alarm times, P(max_score > threshold) = 1 − 1/e ≈ 0.632 corresponds
    to E[alarm time] = γ.

    This calibration is only meaningful when the score statistic is stationary
    under the null.  Scores with time-dependent penalties will trigger a
    ``UserWarning``.

    To match a false alarm probability α over horizon N, set:

        target_arl = int(np.ceil(-N / np.log(1 - alpha)))
    """
    if target_arl < 1:
        raise ValueError("target_arl must be a positive integer.")

    _warn_if_non_constant_penalty(score)

    detector = GridDetector(score=score, threshold=1.0)
    resolved_rng = _normalize_rng(rng)

    max_scores = mc_max_scores(
        detector=detector,
        n_paths=n_paths,
        stream_len=target_arl,
        pre_sampler=pre_sampler,
        rng=resolved_rng,
        pre_args=pre_args,
        pre_kwargs=pre_kwargs,
        parallel=parallel,
        n_jobs=n_jobs,
        strict_equivalence=strict_equivalence,
    )

    max_scores_step2 = None
    if resimulate_combined_threshold and max_scores.shape[1] > 1:
        max_scores_step2 = mc_max_scores(
            detector=detector,
            n_paths=n_paths,
            stream_len=target_arl,
            pre_sampler=pre_sampler,
            rng=resolved_rng,
            pre_args=pre_args,
            pre_kwargs=pre_kwargs,
            parallel=parallel,
            n_jobs=n_jobs,
            strict_equivalence=strict_equivalence,
        )

    return _compute_arl_threshold_from_max_scores(max_scores, max_scores_step2)


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
    resimulate_combined_threshold: bool = False,
) -> np.ndarray:
    """Estimate ARL threshold for an existing detector.

    Convenience wrapper around :func:`calibrate_threshold_arl`.
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
        resimulate_combined_threshold=resimulate_combined_threshold,
    )


def calibrate_threshold_arl_from_samples(
    score: ScoreModel,
    samples: ArrayLike,
    *,
    parallel: bool = True,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Estimate an ARL threshold from pre-generated sample paths.

    The caller supplies an array of observation streams and the function runs
    each stream through the detector to build the null distribution of maximum
    scores, then returns the ``(1/e)``-quantile as the threshold.

    Both steps of the multivariate two-step procedure reuse the same
    ``max_scores`` derived from ``samples``.  If independent step-2 samples
    are needed, use :func:`calibrate_threshold_arl_from_data` with
    ``resimulate_combined_threshold=True``, or call this function twice and
    pass results to :func:`_compute_arl_threshold_from_max_scores` directly.

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
    np.ndarray
        Estimated ARL threshold vector of shape ``(K,)``.

    Notes
    -----
    ARL calibration is only valid when the score is stationary under the null.
    Scores with time-dependent penalties (anything other than
    ``enable_penalty=False``) will trigger a ``UserWarning``.
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

    max_scores = _eval_max_scores(score, samples, parallel=parallel, n_jobs=n_jobs)
    return _compute_arl_threshold_from_max_scores(max_scores)


def calibrate_detector_threshold_arl_from_samples(
    detector: GridDetector,
    samples: ArrayLike,
    *,
    parallel: bool = True,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Estimate ARL threshold for an existing detector from pre-generated samples.

    Convenience wrapper around :func:`calibrate_threshold_arl_from_samples`.
    """
    return calibrate_threshold_arl_from_samples(
        score=detector.score,
        samples=samples,
        parallel=parallel,
        n_jobs=n_jobs,
    )


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
    resimulate_combined_threshold: bool = False,
) -> np.ndarray:
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
    resimulate_combined_threshold : bool, default False
        For multivariate scores (K>1) only.  If ``True``, generates a second
        independent bootstrap sample for step 2 of the calibration procedure
        (the joint scaling of per-test thresholds).  The second bootstrap
        continues from the same ``rng`` (advanced by the first draw), so
        step-2 samples are independent of step-1 samples.  This removes the
        sample-reuse bias between the two estimation steps at the cost of
        twice the bootstrap computation.  Has no effect for ``K=1``.

    Returns
    -------
    np.ndarray
        Estimated ARL threshold vector of shape ``(K,)``.

    Notes
    -----
    ARL calibration is only valid when the score is stationary under the null.
    Scores with time-dependent penalties will trigger a ``UserWarning``.
    """
    _warn_if_non_constant_penalty(score)

    if target_arl < 2:
        raise ValueError(
            "target_arl must be an integer >= 2 for ARL calibration from data."
        )
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

    max_scores = _eval_max_scores(score, samples, parallel=parallel, n_jobs=n_jobs)

    max_scores_step2 = None
    if resimulate_combined_threshold and max_scores.shape[1] > 1:
        samples_step2 = _block_bootstrap_samples(
            data=training_data,
            n_paths=n_paths,
            stream_len=target_arl,
            block_length=block_length,
            rng=local_rng,  # advanced by the first bootstrap draw
        )
        max_scores_step2 = _eval_max_scores(
            score, samples_step2, parallel=parallel, n_jobs=n_jobs
        )

    return _compute_arl_threshold_from_max_scores(max_scores, max_scores_step2)


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
    resimulate_combined_threshold: bool = False,
) -> np.ndarray:
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
        resimulate_combined_threshold=resimulate_combined_threshold,
    )
