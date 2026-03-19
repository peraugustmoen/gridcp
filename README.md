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

### About the new API

- "Score" is the term for a "test statistic" in the code.
- The main object is `gridcp.new_api.GridDetector`, which is a "meta-detector" that can be used with any score that follows the `GridScore` protocol.
- `gridcp.new_api.typing.GridScore` defines the "protocol" or interface for a score to be 
   compatible with `gridcp.new_api.GridDetector`.
- `gridcp.new_api.scores.MeanCUSUM` is an example of a score that follows the `GridScore` protocol, and can be used with `GridDetector`.
- `notebooks.new_api_test_martin.ipynb` is a notebook that demonstrates how to use the GridDetector with the MeanCUSUM score.

### Adding a new score/test statistic

- Add a new file in `gridcp/new_api/scores/` for your score, e.g. `_my_score.py`.
- This files needs two classes:
    * `MyScore`: The actual score implementation, which needs to follow the `GridScore` protocol.
    * `MyScoreState`: Holds running statistics used to compute penalised scores. See `MeanCUSUMState` and `MeanCUSUM` for an example.
- Add the new score and state to `gridcp/new_api/scores/__init__.py`. Now the score can be imported from `gridcp.new_api.scores` and used with `GridDetector`.

