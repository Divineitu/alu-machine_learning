#!/usr/bin/env python3
"""Calculates the maximization step in the EM algorithm for a GMM."""

import numpy as np


def maximization(X, g):
    """Update the priors, means and covariances from the posteriors.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        g: numpy.ndarray of shape (k, n) with the posterior probabilities
            for each data point in each cluster

    Returns:
        pi, m, S, or None, None, None on failure
        pi: numpy.ndarray of shape (k,) with the updated priors
        m: numpy.ndarray of shape (k, d) with the updated means
        S: numpy.ndarray of shape (k, d, d) with the updated covariances
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None
    n, d = X.shape
    if g.shape[1] != n:
        return None, None, None
    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    k = g.shape[0]
    nk = np.sum(g, axis=1)
    pi = nk / n
    m = np.matmul(g, X) / nk[:, np.newaxis]
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = np.matmul(g[i] * diff.T, diff) / nk[i]
    return pi, m, S
