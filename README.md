# G-CHAD
General CHAD implementation 

## Installation
```python
pip install gridcp
```

## Development installation
From the package root, run:
```python
pip install -e .[dev]
```

Or if using uv as the package manager:
```python
uv pip install -e .[dev]
```

## Coding conventions

- Intervals are always left-closed, right-open: [a, b). This is the standard python
    convention for intervals and slicing. Following this drastically reduce strange bugs
    and indexing errors.
- A "changepoint" is the first index of a segment, not last as in the literature. Reasons:

    * Semantically the most correct: A change has only occured after an observation from a new distribution has been observed, not before.
    * Follows python standard slicing convention, such that `data[cp[i]:cp[i+1]]` is the i-th segment.