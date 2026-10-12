#!/usr/bin/env python3
"""Performs K-means clustering on a dataset."""

import numpy as np


def kmeans(X, k, iterations=1000):
    """Run K-means on X.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, the number of clusters
        iterations: positive integer, the maximum number of iterations

    Returns:
        C, clss, or None, None on failure
        C: numpy.ndarray of shape (k, d) with the centroid means
        clss: numpy.ndarray of shape (n,) with the cluster index of
            each data point
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    low = X.min(axis=0)
    high = X.max(axis=0)
    C = np.random.uniform(low, high, size=(k, X.shape[1]))

    for _ in range(iterations):
        prev = C.copy()
        dist = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(dist, axis=1)
        for j in range(k):
            members = X[clss == j]
            if members.shape[0] == 0:
                C[j] = np.random.uniform(low, high)
            else:
                C[j] = members.mean(axis=0)
        if np.array_equal(prev, C):
            return C, clss

    dist = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(dist, axis=1)
    return C, clss
