# G-CHAD
General CHAD implementation 

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
- A "changepoint" is the first index of a segment, not last as in the literature. Reasons:

    * Semantically the most correct: A change has only occured after an observation from a new distribution has been observed, not before.
    * Follows python standard slicing convention, such that `data[cp[i]:cp[i+1]]` is the i-th segment.
- A leading underscore "_" in a file name, class name of function name indicates that it
  is a "private" implementation detail, and not part of the public API. This is a common
  python convention, and is used to indicate that the implementation may change without
  warning, and should not be used directly by users of the package.

### About the new API

- "Score" is the term for a "test statistic" in the code.
- The main object is `gridcp.new_api.GridDetector`, which is a "meta-detector" that can be used with any score that follows the `ScoreModel` protocol.
- `gridcp.new_api.typing.ScoreModel` defines the "protocol" or interface for a score to be 
   compatible with `gridcp.new_api.GridDetector`.
- `gridcp.new_api.scores.MeanCUSUM` is an example of a score that follows the `ScoreModel` protocol, and can be used with `GridDetector`.
- `notebooks.new_api_test_martin.ipynb` is a notebook that demonstrates how to use the GridDetector with the MeanCUSUM score.

### Calibration notes

- `gridcp.calibration.calibrate_threshold(score, ...)` uses a score-first API.
- `gridcp.calibration.mc_alarm_times(detector, ...)` returns the first alarm time per path.
- For calibration/MC helpers, `rng` accepts `numpy.random.Generator`, an integer seed, or `None`.
- Reproducibility policy:
    - `rng=<Generator>`: uses that generator's current state.
    - `rng=<int>`: deterministic run from that seed.
    - `rng=None`: deterministic run from a fixed internal default seed.
- Sampler output convention in `gridcp.calibration`:
    - Monte Carlo helpers are vector-oriented with per-step shape `(n_features,)`.
    - Scalar outputs are broadcast to length `n_features`.
    - Non-scalar outputs are flattened to 1D and must have size `n_features`.
- `n_features` is inferred from `score.n_features` when present.
- For custom scores that do not define `n_features`, pass `n_features` explicitly.

### Adding a new score/test statistic

- Add a new file in `gridcp/scores/` for your score, e.g. `_my_score.py`.
- This files needs two classes:
    * `MyScore`: The actual score implementation, which needs to follow the `ScoreModel` protocol.
    * `MyScoreState`: Holds running statistics used to compute penalised scores. See `MeanCUSUMState` and `MeanCUSUM` for an example.
- Add the new score and state to `gridcp/new_api/scores/__init__.py`. Now the score can be imported from `gridcp.new_api.scores` and used with `GridDetector`.

