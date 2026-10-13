#!/usr/bin/env python3
"""Steady state probabilities of a regular markov chain."""

import numpy as np


def regular(P):
    """Return the steady state probabilities of shape (1, n), or None."""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n = P.shape[0]
    if P.shape != (n, n) or np.any(P < 0):
        return None
    if not np.allclose(np.sum(P, axis=1), 1):
        return None
    Pk = P
    for _ in range((n - 1) ** 2 + 1):
        if np.all(Pk > 0):
            break
        Pk = np.matmul(Pk, P)
    else:
        return None
    A = np.vstack((P.T - np.eye(n), np.ones((1, n))))
    b = np.zeros(n + 1)
    b[-1] = 1
    return np.linalg.lstsq(A, b, rcond=None)[0].reshape(1, n)
