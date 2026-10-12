# Clustering

K-means and Gaussian Mixture Models built from scratch with NumPy, model
selection with the elbow method and the Bayesian Information Criterion, and
the same ideas done with scikit-learn and SciPy (K-means, GMM and
agglomerative clustering with Ward linkage).

## Files

- `0-initialize.py` — `initialize(X, k)`: draws the starting K-means
  centroids from a uniform distribution bounded by the min and max of each
  dimension.
- `1-kmeans.py` — `kmeans(X, k, iterations=1000)`: assigns every point to
  its nearest centroid, moves each centroid to the mean of its points, and
  stops early when the centroids stop moving. An empty cluster gets a new
  random centroid.
- `2-variance.py` — `variance(X, C)`: total intra-cluster variance, the sum
  of squared distances from each point to its closest centroid.
- `3-optimum.py` — `optimum_k(X, kmin=1, kmax=None, iterations=1000)`: runs
  K-means for every k in the range and returns how much the variance drops
  compared to `kmin` (the elbow method).
- `4-initialize.py` — `initialize(X, k)`: GMM starting values: equal priors,
  means from K-means, identity covariances.
- `5-pdf.py` — `pdf(X, m, S)`: multivariate Gaussian density for every
  point, vectorized, with a floor of `1e-300`.
- `6-expectation.py` — `expectation(X, pi, m, S)`: E-step, giving the
  posterior probability of each cluster for each point plus the total log
  likelihood.
- `7-maximization.py` — `maximization(X, g)`: M-step, giving new priors,
  means and covariances weighted by the posteriors.
- `8-EM.py` — `expectation_maximization(X, k, iterations=1000, tol=1e-5,
  verbose=False)`: alternates E and M steps until the log likelihood changes
  by no more than `tol`.
- `9-BIC.py` — `BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5,
  verbose=False)`: fits a GMM for each k and keeps the one with the lowest
  `BIC = p * ln(n) - 2 * l`, where
  `p = (k - 1) + k * d + k * d * (d + 1) / 2` free parameters.
- `10-kmeans.py` — `kmeans(X, k)` with `sklearn.cluster.KMeans`.
- `11-gmm.py` — `gmm(X, k)` with `sklearn.mixture.GaussianMixture`.
- `12-agglomerative.py` — `agglomerative(X, dist)`: Ward linkage with
  `scipy.cluster.hierarchy`, cut at cophenetic distance `dist`, and a
  dendrogram coloured by cluster.

## Requirements

- Python 3
- NumPy
- scikit-learn (tasks 10 and 11)
- SciPy and Matplotlib (task 12)
