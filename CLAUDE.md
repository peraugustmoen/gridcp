# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

G-CHAD is a Python package (`gridcp`) for online grid-based changepoint detection in data streams. It is being developed for a software paper.

## Commands

```bash
# Install for development
pip install -e .[dev]

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_univariate_mean_change.py

# Run with coverage
pytest --cov=gridcp --cov-report=term-missing

# Lint and format checks
ruff check gridcp tests
ruff format --check gridcp tests

# Auto-fix formatting
pre-commit run --all-files
```

## Architecture

The package has two APIs — the **new API** (active) and `gridcp/old_api/` (preserved, not maintained, excluded from linting).

### New API structure

**`gridcp/typing.py`** — `ScoreModel` protocol: the interface all test statistics must implement.
- `init_state()` → fresh state
- `update(state, x)` → incorporate new observation
- `compute_penalised_scores(state, grid_states)` → test statistics with penalties

**`gridcp/detector.py`** — `GridDetector`: the meta-detector that works with any `ScoreModel`.
- Maintains a logarithmic grid of O(log n) candidate changepoint positions
- `update(state, x)` returns `(new_state, output_dict)` where output includes `alarm`, `max_score`, `max_score_index`, `num_samples`

**`gridcp/scores/`** — Concrete score implementations:
- `MeanCUSUM` — CUSUM for mean changes (known variance)
- `MeanCUSUMUnknownVariance` — LR test for mean changes (unknown variance)
- `ExponentialFamilyGLR` — GLR test for canonical exponential families (user supplies sufficient statistic, log-partition, derivatives). Supports both Numba-compiled and plain NumPy callables; automatically detects which and builds JIT or pure-Python kernels accordingly. Penalty controlled via `penalty: PenaltyType` (stored as `self.penalty`); computed by `_get_penalty(n_samples)` method.

**`gridcp/calibration.py`** — Monte Carlo helpers for threshold calibration:
- `draw_samples`, `mc_max_scores`, `mc_alarm_times`
- `calibrate_threshold`, `calibrate_detector_threshold`, `with_calibrated_threshold`
- Supports parallel execution via `n_jobs` parameter

**`gridcp/utils.py`** — Internal utilities: `v2(r)` (grid pruning), `fastlog`, `get_changeloc_grid`/`get_G_grid` (debugging)

### Public API

```python
from gridcp import GridDetector, calibrate_detector_threshold, with_calibrated_threshold, ...
from gridcp.scores import MeanCUSUM, MeanCUSUMUnknownVariance, ExponentialFamilyGLR
```

## Coding conventions

- **Intervals:** always left-closed, right-open `[a, b)` — matches Python slicing
- **Indices:** 0-indexed by default
- **Changepoint definition:** first index of the new segment (not last), so `data[cp[i]:cp[i+1]]` is the i-th segment
- **Private API:** leading underscore `_` in filename/class/function = internal detail, may change without warning

## Calibration conventions

- Internal loop variable `t` is 1-indexed (sample size); returned alarm times are 0-indexed
- `rng=None` → deterministic (fixed internal seed); `rng=int` → deterministic from that seed; `rng=Generator` → uses generator's current state
- Sampler outputs per step have shape `(n_features,)`; scalars are broadcast
- `n_features` inferred from `score.n_features` when present; pass explicitly for custom scores

## Adding a new score

1. Create `gridcp/scores/_my_score.py` with `MyScore` (implements `ScoreModel`) and `MyScoreState`
2. Export both from `gridcp/scores/__init__.py`

## Upcoming work

- **Exponential family tests:** Fix the test file for exponential family. Last time changes were made to it, it could no longer be committed due to the formatting check. Find issue, and fix it.
- **Built-in exponential families for `ExponentialFamilyGLR`:** Add a set of pre-defined family specifications (e.g. Gaussian mean, Poisson, Bernoulli, Exponential, Gaussian variance) so users can instantiate `ExponentialFamilyGLR` by name rather than supplying all callables manually. Design TBD — likely a factory function or class method such as `ExponentialFamilyGLR.from_family("gaussian_mean", n_features=1)` or a separate `families` submodule.
- **ARL-based calibration:** Add a Monte Carlo calibration method to `gridcp/calibration.py` where the user specifies a target Average Run Length (ARL) instead of a false alarm rate, based on the ocd algorithm. The existing calibration uses false alarm rate as the control parameter.

## Configuration notes

- `ruff` excludes `gridcp/old_api/` from linting
- Docstring style: numpy
- Line length: 88
- Tests directory: `tests/` (old_api_tests excluded)
- CI tests Python 3.10 and 3.13
