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
- `update(state, x)` returns `(new_state, output_dict)` where output includes `alarm`, `max_score`, `max_score_index`, `n_samples`

**`gridcp/scores/`** — Concrete score implementations:
- `MeanCUSUM` — CUSUM for mean changes (known variance)
- `MeanCUSUMUnknownVariance` — LR test for mean changes (unknown variance)
- `Variance` — LR test for variance changes (known zero mean)
- `MeanOrVariance` — combined mean/variance LR test (univariate)
- `MultivariateMeanIdentityCov` — multivariate mean LR test (known identity covariance)
- `MultivariateMeanUnknownCov` — multivariate mean LR test (unknown covariance)
- `MultivariateMeanOrCovariance` — combined multivariate mean/covariance LR test
- `RegressionDirect` / `RegressionMcScan` — regression changepoint scores
- `ExponentialFamilyGLR` — GLR test for canonical exponential families (user supplies sufficient statistic, log-partition, derivatives). Supports both Numba-compiled and plain NumPy callables; automatically detects which and builds JIT or pure-Python kernels accordingly. Penalty controlled via `penalty: PenaltyType` (stored as `self.penalty`); computed by `_get_penalty(n_samples)` method. Built-in families available via `ExponentialFamilyGLR.from_family(name)`.

**`gridcp/calibration.py`** — Monte Carlo helpers for threshold calibration:
- `draw_samples`, `mc_max_scores`, `mc_alarm_times`
- `calibrate_threshold_false_alarm`, `calibrate_detector_threshold_false_alarm`, `with_calibrated_threshold`
- Supports parallel execution via `n_jobs` parameter

**`gridcp/utils.py`** — Internal utilities: `v2(r)` (grid pruning), `fastlog`, `get_changeloc_grid`/`get_G_grid` (debugging)

### Public API

```python
from gridcp import GridDetector, calibrate_detector_threshold_false_alarm, with_calibrated_threshold, ...
from gridcp.scores import (
    MeanCUSUM, MeanCUSUMUnknownVariance, Variance, MeanOrVariance,
    MultivariateMeanIdentityCov, MultivariateMeanUnknownCov, MultivariateMeanOrCovariance,
    RegressionDirect, RegressionMcScan, ExponentialFamilyGLR,
)
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

- **Built-in exponential families — open question:** Whether to add a family for multivariate Gaussian with *both* mean and covariance unknown, to compare against the built-in `MultivariateMeanUnknownCov` score.
- **Promote ARL calibration to `gridcp/calibration.py`:** `calibrate_threshold_arl` and `calibrate_detector_threshold_arl` are currently prototyped in `old_notebooks/arl_calibration.py`. They should be moved into the main package (`gridcp/calibration.py`) and exported from `gridcp/__init__.py`, following the same pattern as `calibrate_threshold_false_alarm` / `calibrate_detector_threshold_false_alarm`.

## Recent work

- **`ExponentialFamilyGLR` `min_seg` fix:** Default `min_seg = v + 1` was too large for high-dimensional families (e.g. `gaussian_mean` with p=1000 → `min_seg=1001`, blocking all candidates on short streams). Fixed by adding family-specific defaults to `_families.py`: `gaussian_mean` → 2, scalar families → 2, `gaussian_mean_variance` → 3, `gaussian_covariance` → p+1. `from_family` reads these via `spec.get("min_seg")`.
- **Built-in exponential families:** `gridcp/scores/_families.py` implements `gaussian_mean` (scalar and multivariate), `gaussian_variance`, `gaussian_mean_variance`, `gaussian_covariance`, `poisson`, `exponential`, `bernoulli`. All accessible via `ExponentialFamilyGLR.from_family(name)`.
- **Simulation metrics in `sandbox_Espen.ipynb`:** `run_scenario` now reports FA% (alarms before changepoint), ms/str, and ms/obs.
- **ARL calibration prototype:** `old_notebooks/arl_calibration.py` implements `calibrate_threshold_arl` and `calibrate_detector_threshold_arl`. API mirrors `calibrate_threshold_false_alarm`/`calibrate_detector_threshold_false_alarm` — same parameters, just replacing `false_alarm_probability + stream_len` with `target_arl`. Uses `mc_max_scores` with `stream_len=target_arl` and returns the (1/e)-quantile. Requires `PenaltyType.CONSTANT` on the score (all built-in scores support this via their `penalty` field). Tested in `old_notebooks/sandbox_Espen.ipynb` (cells 64–67): exponential assumption check (QQ-plot + histogram) and 3-scenario detection delay sweep (Gaussian mean, Gaussian mean+var, Poisson).
- **`_solve` simplification:** Removed the explicit 3×3 Cramer's rule case from `_solve` in `_exponential_family_glr.py` — only the 2×2 case is kept; n≥3 falls through to `np.linalg.solve`. No measurable performance impact given O(log n) solver calls per step.
- **`ExponentialFamilyGLR` documentation pass:** Module docstring expanded with a pipeline overview (GLR formula, construction, per-step computation). Class docstring trimmed to match the style of other score classes (removed ScoreModel protocol mention and internal implementation details).

## Configuration notes

- `ruff` excludes `gridcp/old_api/` from linting
- Docstring style: numpy
- Line length: 88
- Tests directory: `tests/` (old_api_tests excluded)
- CI tests Python 3.10 and 3.13
