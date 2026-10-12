#!/usr/bin/env python3
"""Finds the best number of clusters for a GMM using the BIC."""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Pick the number of GMM clusters with the lowest BIC.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        kmin: positive integer, the minimum number of clusters (inclusive)
        kmax: positive integer, the maximum number of clusters (inclusive)
        iterations: positive integer, the maximum EM iterations
        tol: non-negative float, the EM tolerance
        verbose: boolean, whether EM should print its progress

    Returns:
        best_k, best_result, l, b, or None, None, None, None on failure
        best_k: the best value for k based on its BIC
        best_result: tuple (pi, m, S) for the best number of clusters
        l: numpy.ndarray of shape (kmax - kmin + 1) with the log
            likelihood for each cluster size tested
        b: numpy.ndarray of shape (kmax - kmin + 1) with the BIC value
            for each cluster size tested
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    n, d = X.shape
    if kmax is None:
        kmax = n
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax <= kmin:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    results = []
    log_l = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)
    for i, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, _, log_l[i] = expectation_maximization(
            X, k, iterations, tol, verbose)
        results.append((pi, m, S))
        p = (k - 1) + k * d + k * d * (d + 1) / 2
        b[i] = p * np.log(n) - 2 * log_l[i]
    best = np.argmin(b)
    return kmin + best, results[best], log_l, b
