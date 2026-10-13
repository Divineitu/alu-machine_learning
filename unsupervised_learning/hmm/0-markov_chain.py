#!/usr/bin/env python3
"""Probability of a markov chain being in each state after t iterations."""

import numpy as np


def markov_chain(P, s, t=1):
    """Return the state probabilities of shape (1, n) after t iterations."""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n = P.shape[0]
    if P.shape != (n, n):
        return None
    if not isinstance(s, np.ndarray) or s.shape != (1, n):
        return None
    if not isinstance(t, int) or t < 0:
        return None
    return np.matmul(s, np.linalg.matrix_power(P, t))
