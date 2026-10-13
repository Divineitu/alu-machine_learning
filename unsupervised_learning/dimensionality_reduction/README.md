# Dimensionality Reduction

Principal component analysis (PCA) implemented with NumPy via singular value
decomposition.

## Files

- `0-pca.py` — `pca(X, var=0.95)`: returns the weights matrix `W` of shape
  `(d, nd)` that keeps `var` fraction of the variance of the zero-mean data `X`.
- `1-pca.py` — `pca(X, ndim)`: centers `X` and returns its projection `T` of
  shape `(n, ndim)` onto the first `ndim` principal components.
