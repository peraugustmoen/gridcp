# Contributing to gridcp

Thanks for your interest in improving `gridcp`. This guide covers the development setup,
the conventions the codebase follows, and the contract for adding a new score model.

## Development setup

From the repository root:

```bash
pip install -e .[dev]        # or: uv pip install -e .[dev]
```

Install the pre-commit hooks so formatting and linting run on every commit:

```bash
pre-commit install
pre-commit install --hook-type pre-push   # also run the tests before each push
```

Common commands:

```bash
pytest tests/                                    # run the tests
pytest --cov=gridcp --cov-report=term-missing    # with coverage
ruff check gridcp tests                          # lint
ruff format gridcp tests                         # format
pre-commit run --all-files                       # run every hook at once
```

Docstrings use the NumPy convention, the line length is 88, and spelling is American
English throughout.

## Coding conventions

- **Intervals** are left-closed, right-open, `[a, b)`. This matches Python slicing and
  avoids a class of off-by-one bugs.
- **Indices** are 0-based unless stated otherwise.
- A **changepoint** is the first index of the post-change segment (0-based) — not the
  last pre-change index used by some authors. So for a changepoint `cp`, `data[0:cp]` is
  pre-change and `data[cp:]` is post-change, and the detection delay is `alarm_time - cp`
  (delay 0 means the alarm fires on the first post-change observation).
- A **leading underscore** marks a private name. It may change without notice, so don't
  rely on it.

## Architecture

- **`gridcp/detector.py`** — `GridDetector` plus its `DetectorState`. The detector holds
  the logarithmic-grid logic (O(log n) candidate changepoints) and works with any score.
- **`gridcp/typing.py`** — the `ScoreModel` protocol that every score implements, and the
  `DetectorOutput` TypedDict.
- **`gridcp/scores/`** — the built-in scores (`CUSUM`, `GaussianMean`, and so on).
- **`gridcp/calibration.py`** — Monte Carlo threshold calibration, with optional parallel
  execution via `n_jobs`.
- **`gridcp/utils.py`** — internal helpers.

In the code, a *score* is a test statistic. There is no base class to subclass: any object
that structurally satisfies the `ScoreModel` protocol works.

The `GridDetector` is immutable (a frozen dataclass) and stores only configuration — the
score and the threshold. All evolving per-stream state — the running score state, the grid
of candidate split points, the per-candidate snapshots, and the local `n_samples` — lives
in a separate `DetectorState` that you thread through the detector: `init_state()` returns
a fresh one, and `update(state, x)` returns a new `(state, output)` pair without mutating
the detector or the state you passed in. Because the detector holds no state, one detector
can run many independent streams, and a `DetectorState` can be copied or pickled.

## The `ScoreModel` contract

A score implements three methods:

```python
def init_state(self) -> TScoreState: ...
def update(self, state: TScoreState, x: ArrayLike) -> TScoreState: ...
def compute_penalized_scores(
    self, state: TScoreState, grid_states: list[TScoreState]
) -> np.ndarray: ...
```

and two properties: `n_features` (observation dimension) and `n_scores` (the number of
penalized scores returned per candidate).

The rules:

- **State is an immutable snapshot.** `update` returns a *new* state and must not mutate
  its input. The `DetectorState` holds a past score-state snapshot for every active grid
  candidate (`previous_score_states`), so mutating one in place would corrupt the grid.
  See `CUSUM`/`CUSUMState` in [`gridcp/scores/_cusum.py`](gridcp/scores/_cusum.py) for a
  worked example.
- **Time-dependent penalties come from the state.** `GridDetector` calls
  `compute_penalized_scores(state, grid_states)`, so any penalty scaling (for example
  `log(t)`) has to be derived from `state`. The state therefore needs to carry the time,
  typically an `n_samples` counter bumped in `update`.
- **Return shape is `(G, n_scores)`,** where `G = len(grid_states)` is the number of
  active candidates. A single-score model returns `(G, 1)`. A width that disagrees with
  `n_scores` raises `ValueError`.

## Thresholds

- Threshold values must be strictly positive.
- A scalar threshold is broadcast to all scores; a vector threshold must have length
  `n_scores`. The check happens at construction, not on the first `update`. Internally
  the threshold is stored as a `float64` array of shape `(n_scores,)`.
- In each update's output, `max_score` and `max_split_point` are vectors of shape
  `(n_scores,)`. Each `max_split_point` is the winning split `n1 = state.grid[argmax]`, a
  0-based index with `data[0:n1]` pre-change and `data[n1:]` post-change. Before two
  observations have been seen there is no candidate yet, so both are zero vectors.

## Resetting

Resetting is external to the detector: get a fresh state from `init_state`.

```python
state = detector.init_state()
```

The fresh state starts with an empty score state, an empty grid, no candidate snapshots,
and local `n_samples` at 0. A time-dependent penalty based on `n_samples` (for example
`log(t)`) therefore restarts from 0, so long-run false-alarm guarantees that assume a
globally increasing clock do not automatically carry across resets.

## Calibration

- `calibrate_threshold_*` takes a *score*; `calibrate_detector_threshold_*` takes an
  already-built `GridDetector`. The two are equivalent — use whichever matches the objects
  you already have. Both return a threshold as a 1-D array of shape `(n_scores,)`.
- `mc_alarm_times(detector, ...)` returns the first alarm time per path.
- A sampler must accept `rng` (positionally or by keyword) and return one step of shape
  `(n_features,)`; a scalar is broadcast, and `n_features` is taken from `score.n_features`
  when available. If you pass a `changepoint`, you must also pass a `post_sampler`.
- `rng` accepts a `numpy.random.Generator`, an integer seed, or `None` (a fixed default
  seed). In parallel runs, a passed `Generator` seeds the workers from its original
  `SeedSequence`.

## Adding a new score

1. Add `gridcp/scores/_my_score.py` with two classes: `MyScore` (implements `ScoreModel`)
   and `MyScoreState` (its running statistics, treated as an immutable snapshot).
   `CUSUM`/`CUSUMState` are a good template.
2. Export both from `gridcp/scores/__init__.py`.
3. Write tests in `tests/` first, describing the behavior you want rather than
   retrofitting them to the implementation.
4. Run `pre-commit run --all-files` before opening a pull request.
