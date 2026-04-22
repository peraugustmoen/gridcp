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

**`gridcp/typing.py`** — `ScoreModel` protocol: the interface all test statistics must implement.

**`gridcp/detector.py`** — `GridDetector`: the meta-detector that works with any `ScoreModel`. Maintains a logarithmic grid of O(log n) candidate changepoint positions.

**`gridcp/scores/`** — Concrete score implementations: `MeanCUSUM`, `MeanCUSUMUnknownVariance`, `Variance`, `MeanOrVariance`, `MultivariateMeanIdentityCov`, `MultivariateMeanUnknownCov`, `MultivariateMeanOrCovariance`, `RegressionDirect`, `RegressionMcScan`, `ExponentialFamilyGLR`. Built-in exponential families accessible via `ExponentialFamilyGLR.from_family(name)`.

**`gridcp/calibration.py`** — Monte Carlo helpers for threshold calibration. Supports parallel execution via `n_jobs`.

**`gridcp/utils.py`** — Internal utilities.

## Adding a new score

1. Create `gridcp/scores/_my_score.py` with `MyScore` (implements `ScoreModel`) and `MyScoreState`
2. Export both from `gridcp/scores/__init__.py`

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

## Configuration notes

- `ruff` excludes `gridcp/old_api/` from linting
- Docstring style: numpy
- Line length: 88
- Tests directory: `tests/` (old_api_tests excluded)
- CI tests Python 3.10 and 3.13

## Coding notes

- Always run ruff linting after editing code so that pre-commit hooks pass
- Always write or adjust tests before writing new/changed functionality — tests should reflect desired behavior, not be retrofitted to match what the code does
- Spelling should always be according to American English
