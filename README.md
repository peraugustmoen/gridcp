# G-CHAD
General CHAD implementation 

## Quick start

```python
import numpy as np
from gridcp.detector import GridDetector
from gridcp.scores import MeanCUSUM

# Create a detector for univariate mean changes
detector = GridDetector(score=MeanCUSUM(n_features=1), threshold=5.0)
state = detector.init_state()

# Feed observations one at a time
rng = np.random.default_rng(0)
for t in range(200):
    x = rng.normal(0.0, 1.0) if t < 100 else rng.normal(3.0, 1.0)
    state, output = detector.update(state, [x])
    if output["alarm"]:
        print(f"Change detected at observation {t}!")
        break
```

## Installation

### User installation
```python
pip install gridcp
```

### Development installation
From the package root, run:
```python
pip install -e .[dev]
```

Or if using uv as the package manager:
```python
uv pip install -e .[dev]
```

## Developer guide
### Coding conventions

- Intervals are always left-closed, right-open: [a, b). This is the standard python
    convention for intervals and slicing. Following this drastically reduce strange bugs
    and indexing errors.
- Indices are 0-indexed by default unless explicitly stated otherwise.
- A "changepoint" is the **first post-change index (0-based)**: the first index
    of the new distribution's segment, not the last pre-change index as in some
    literature. Specifically, for a changepoint `cp`:

    * `data[0:cp]` is the pre-change segment; `data[cp:]` is the post-change segment.
    * Follows Python standard slicing convention, such that `data[cp[i]:cp[i+1]]` is the i-th segment.
    * Detection delay = `alarm_time - cp` (delay=0 means alarm fires exactly at the first post-change observation).
- A leading underscore "_" in a file name, class name of function name indicates that it
  is a "private" implementation detail, and not part of the public API. This is a common
  python convention, and is used to indicate that the implementation may change without
  warning, and should not be used directly by users of the package.

### Automated linting and formatting

Run this once after the development install to auto-run linting/formatting before each commit:

```bash
pre-commit install
```

Run this once to also execute tests before each push:

```bash
pre-commit install --hook-type pre-push
```

Run on all files manually:

```bash
pre-commit run --all-files
```

### About the new API

- "Score" is the term for a "test statistic" in the code.
- The main object is `gridcp.detector.GridDetector`, which is a "meta-detector" that can be used with any score that follows the `ScoreModel` protocol.
- `gridcp.typing.ScoreModel` defines the "protocol" or interface for a score to be 
    compatible with `gridcp.detector.GridDetector`.
- `gridcp.scores.MeanCUSUM` is an example of a score that follows the `ScoreModel` protocol, and can be used with `GridDetector`.
- `notebooks.new_api_test_martin.ipynb` is a notebook that demonstrates how to use the GridDetector with the MeanCUSUM score.

### Calibration notes

- `gridcp.calibration.calibrate_threshold_false_alarm(score, ...)` uses a score-first API.
- Calibration threshold APIs return 1-D NumPy arrays of shape `(K,)`.
    For single-score models, this is shape `(1,)`.
- `gridcp.calibration.mc_alarm_times(detector, ...)` returns the first alarm time per path.
- Indexing convention in calibration internals:
    - Loop variable `t` denotes current sample size, so it is 1-indexed (`t = 1, ..., stream_len`).
    - Returned alarm times are 0-indexed array indices (Python convention).
- For calibration/MC helpers, `rng` accepts `numpy.random.Generator`, an integer seed, or `None`.
- Reproducibility policy:
    - `rng=<Generator>`: uses that generator's current state.
    - `rng=<int>`: deterministic run from that seed.
    - `rng=None`: deterministic run from a fixed internal default seed.
- Sampler signature contract in calibration helpers:
    - Supported: `sampler(rng, /, *args, **kwargs)` (positional-only `rng`).
    - Supported: `sampler(*args, rng, **kwargs)` (keyword-capable `rng`).

- Sampler output convention in `gridcp.calibration`:
    - Monte Carlo helpers are vector-oriented with per-step shape `(n_features,)`.
    - Scalar outputs are broadcast to length `n_features`.
    - Non-scalar outputs are flattened to 1D and must have size `n_features`.
- `n_features` is inferred from `score.n_features` when present.
- For custom scores that do not define `n_features`, pass `n_features` explicitly.
- If `changepoint` is provided, `post_sampler` must also be provided.

### Threshold shape behavior in `GridDetector`

- Threshold values must be strictly positive.
- `ScoreModel.n_scores` declares the number of scores `K`. This is the authoritative
    value for the score output dimension.
- `ScoreModel.compute_penalized_scores` must return shape `(G, K)` where `K == n_scores`.
    Single-score models must return `(G, 1)`.
- A vector threshold must have length `K == n_scores`. **Mismatch is caught at
    construction time**, not deferred to the first `update()` call.
- `GridDetector.threshold` is always stored as a 1-D `float64` NumPy array of
    shape `(K,)`. Scalar inputs are broadcast once at construction time to a
    length-`K` vector.
- When penalized scores are available, `DetectorOutput.max_score` and
    `DetectorOutput.max_split_point` are vectors of shape `(K,)` (including `(1,)`
    for single-score models).
- Each `max_split_point` entry is the first post-change index (0-based)
    `n1 = state.grid[argmax]`. For valid scored candidates (`n_samples >= 2`),
    `n1` is in `{1, ..., n_samples-1}`: `data[0:n1]` is pre-change and
    `data[n1:]` is post-change.
- For `n_samples < 2`, no candidate score exists yet. `max_score` and
    `max_split_point` are zero vectors of shape `(K,)`.
- At runtime, if `compute_penalized_scores` returns an output width that does not
    match `n_scores`, the detector raises `ValueError` immediately.


### Reset semantics in `GridDetector`

`GridDetector` uses one time scale:

- `n_samples`: local time since the most recent reset.

Each call to `update(...)` returns `n_samples` in the output dictionary.

Resetting is external to `GridDetector` and handled with `reset_detector_state`:

```python
from gridcp import reset_detector_state

state = reset_detector_state(state, detector)
```

- Clears the running score state
- Clears the grid and all candidate score snapshots
- Sets local `n_samples` back to 0

**Note:** Any time-dependent penalty that uses `state.n_samples` (for example, `log(t)` or
`sqrt(log(t)) + log(t)`) also restarts from this post-reset local time. This differs from
a "continuous penalty time" interpretation where the penalty clock keeps increasing across
resets. As a result, long-run false-alarm guarantees or intuitions that assume a globally
increasing time index do not automatically carry over across multiple resets.

#### Custom score contract

Custom score models must implement:

```python
def compute_penalized_scores(
    self,
    state,
    grid_states,
) -> np.ndarray:
    ...
```

The intended pattern is:

- compute centered or raw statistics from `state` and `grid_states`
- use `state.n_samples` for any time-dependent penalty divisor

### Adding a new score/test statistic

- Add a new file in `gridcp/scores/` for your score, e.g. `_my_score.py`.
- This file needs two classes:
    * `MyScore`: The actual score implementation, which needs to follow the `ScoreModel` protocol.
    * `MyScoreState`: Holds running statistics used to compute penalized scores. See `MeanCUSUMState` and `MeanCUSUM` for an example.
- `MyScoreState` must be treated as immutable snapshots. `update(...)` must return a
    new state and must not mutate the input state in place, because `GridDetector`
    stores historical state snapshots for active candidates.
- Add the new score and state to `gridcp/scores/__init__.py`. Now the score can be imported from `gridcp.scores` and used with `GridDetector`.

