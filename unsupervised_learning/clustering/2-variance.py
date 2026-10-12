#!/usr/bin/env python3
"""Calculates the total intra-cluster variance for a dataset."""

import numpy as np


def variance(X, C):
    """Compute the total intra-cluster variance.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        C: numpy.ndarray of shape (k, d) with the centroid means

    Returns:
        var, the total variance, or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    sq_dist = np.sum((X[:, np.newaxis] - C) ** 2, axis=2)
    return np.sum(np.min(sq_dist, axis=1))
