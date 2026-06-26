# Contributing to gridcp

Thanks for your interest in improving `gridcp`. This guide covers the development
setup, the conventions the codebase follows, and the contract for adding a new
detector (score).

## Development setup

From the repository root:

```bash
pip install -e .[dev]
```

Or, using `uv`:

```bash
uv pip install -e .[dev]
```

### Linting, formatting, and tests

Run formatting and linting before each commit by installing the pre-commit hooks once:

```bash
pre-commit install
```

To also run the test suite before each push:

```bash
pre-commit install --hook-type pre-push
```

Useful commands:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=gridcp --cov-report=term-missing

# Lint and check formatting
ruff check gridcp tests
ruff format --check gridcp tests

# Run all hooks manually
pre-commit run --all-files
```

Docstrings follow the NumPy convention; the line length is 88. American English
spelling is used throughout.

## Coding conventions

- **Intervals** are always left-closed, right-open: `[a, b)`. This matches Python's
  slicing and indexing conventions and avoids a whole class of off-by-one bugs.
- **Indices** are 0-indexed by default unless explicitly stated otherwise.
- A **changepoint** is the *first post-change index (0-based)* — the first index of the
  new distribution's segment, not the last pre-change index used in some literature.
  For a changepoint `cp`:
  - `data[0:cp]` is the pre-change segment; `data[cp:]` is the post-change segment.
  - `data[cp[i]:cp[i+1]]` is the i-th segment under standard slicing.
  - Detection delay = `alarm_time - cp` (delay 0 means the alarm fires exactly at the
    first post-change observation).
- A **leading underscore** (`_`) in a file, class, or function name marks a *private*
  implementation detail. It is not part of the public API and may change without
  warning; it should not be relied upon by users of the package.

## Architecture

The package has two APIs: the **new API** (active) and `gridcp/old_api/` (preserved,
not maintained, excluded from linting).

- **`gridcp/typing.py`** — the `ScoreModel` protocol that all test statistics
  implement, and the `DetectorOutput` TypedDict.
- **`gridcp/detector.py`** — `GridDetector`, the meta-detector that works with any
  `ScoreModel`. It maintains a logarithmic grid of O(log n) candidate changepoint
  positions.
- **`gridcp/scores/`** — concrete score implementations (`CUSUM`, `GaussianMean`,
  `GaussianVariance`, and so on). Built-in exponential families are accessible via
  `ExponentialFamilyGLR.from_family(name)`.
- **`gridcp/calibration.py`** — Monte Carlo helpers for threshold calibration, with
  parallel execution via `n_jobs`.
- **`gridcp/utils.py`** — internal utilities.

Terminology: in the code, a *score* is a test statistic. `GridDetector` owns a score and
an alarm threshold; you do not need to subclass anything — any object that structurally
satisfies the `ScoreModel` protocol works (duck typing).

## The `ScoreModel` contract

A compliant score implements three methods:

```python
def init_state(self) -> TScoreState: ...
def update(self, state: TScoreState, x: ArrayLike) -> TScoreState: ...
def compute_penalized_scores(
    self, state: TScoreState, grid_states: list[TScoreState]
) -> np.ndarray: ...
```

and exposes two properties: `n_features` (observation dimension) and `n_scores` (the
number of penalized scores returned per candidate).

Key rules:

- **State is an immutable snapshot.** `update(...)` must return a *new* state and must
  not mutate the input in place — `GridDetector` stores historical state snapshots for
  active candidates. See `CUSUM`/`CUSUMState` in
  [`gridcp/scores/_cusum.py`](gridcp/scores/_cusum.py) for a reference implementation.
- **Time-dependent penalties read from the state.** `GridDetector` calls
  `compute_penalized_scores(state, grid_states)`; any time-dependent penalty scaling
  (for example, `log(t)`) must be derived from `state`, which therefore needs to carry
  the relevant time information (typically an `n_samples` counter updated in `update`).
- **Return shape is `(G, n_scores)`.** `G = len(grid_states)` is the number of active
  candidates. Single-score models must return `(G, 1)`. If the returned width does not
  match `n_scores`, the detector raises `ValueError` immediately.

The intended pattern inside `compute_penalized_scores`:

- compute centered or raw statistics from `state` and `grid_states`;
- use `state.n_samples` for any time-dependent penalty divisor.

## Threshold semantics in `GridDetector`

- Threshold values must be strictly positive.
- `ScoreModel.n_scores` is the authoritative value for the score output dimension.
- A vector threshold must have length `n_scores`. A mismatch is caught at
  **construction time**, not deferred to the first `update()` call.
- `GridDetector.threshold` is always stored as a 1-D `float64` array of shape
  `(n_scores,)`. Scalar inputs are broadcast once at construction time.
- When penalized scores are available, `DetectorOutput.max_score` and
  `max_split_point` are vectors of shape `(n_scores,)` (including `(1,)` for
  single-score models).
- Each `max_split_point` entry is the first post-change index (0-based)
  `n1 = state.grid[argmax]`. For valid scored candidates (`n_samples >= 2`), `n1` is in
  `{1, ..., n_samples - 1}`: `data[0:n1]` is pre-change and `data[n1:]` is post-change.
- For `n_samples < 2`, no candidate score exists yet, so `max_score` and
  `max_split_point` are zero vectors of shape `(n_scores,)`.

## Reset semantics

`GridDetector` uses a single time scale, `n_samples`: local time since the most recent
reset, returned in every `update(...)` output. Resetting is external to the detector:

```python
from gridcp import reset_detector_state

state = reset_detector_state(detector)
```

This clears the running score state, the grid and all candidate snapshots, and sets
local `n_samples` back to 0.

**Note:** any time-dependent penalty that uses `state.n_samples` (for example, `log(t)`)
also restarts from this post-reset local time. This differs from a "continuous penalty
time" interpretation where the penalty clock keeps increasing across resets. Long-run
false-alarm guarantees that assume a globally increasing time index do not automatically
carry over across multiple resets.

## Calibration conventions

- `calibrate_threshold_*` functions take a *score* as their first argument;
  `calibrate_detector_threshold_*` functions take an already-constructed `GridDetector`.
  Both sets are equivalent — use whichever matches the objects you already have.
- Calibration threshold APIs return 1-D NumPy arrays of shape `(n_scores,)`. For
  single-score models this is shape `(1,)`.
- `mc_alarm_times(detector, ...)` returns the first alarm time per path.
- Indexing: the internal loop variable `t` is the current sample size (1-indexed,
  `t = 1, ..., stream_len`); returned alarm times are 0-indexed array indices.
- `rng` accepts a `numpy.random.Generator`, an integer seed, or `None`:
  - `Generator`: in parallel runs, workers are seeded from the generator's original
    `SeedSequence`;
  - `int`: deterministic run from that seed;
  - `None`: deterministic run from a fixed internal default seed.
- Sampler signatures: either `sampler(rng, /, *args, **kwargs)` (positional-only `rng`)
  or `sampler(*args, rng, **kwargs)` (keyword-capable `rng`).
- Sampler outputs are vector-oriented with per-step shape `(n_features,)`. Scalars are
  broadcast to length `n_features`; non-scalar outputs are flattened to 1-D and must
  have size `n_features`.
- `n_features` is inferred from `score.n_features` when present; pass it explicitly for
  custom scores that do not define it.
- If `changepoint` is provided, `post_sampler` must also be provided.

## Adding a new score

1. Add a file in `gridcp/scores/`, for example `_my_score.py`, containing two classes:
   - `MyScore` — the score implementation, satisfying the `ScoreModel` protocol.
   - `MyScoreState` — the running statistics used to compute penalized scores. Treat it
     as an immutable snapshot (see the contract above). `CUSUM`/`CUSUMState` are a good
     template.
2. Export both from `gridcp/scores/__init__.py` so the score can be imported from
   `gridcp.scores` and used with `GridDetector`.
3. Add or adjust tests in `tests/` *before* writing the functionality — tests should
   reflect the desired behavior, not be retrofitted to the implementation.
4. Run `ruff` and `pytest` (or `pre-commit run --all-files`) before opening a pull
   request.
