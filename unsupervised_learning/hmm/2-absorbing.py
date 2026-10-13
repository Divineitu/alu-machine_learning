#!/usr/bin/env python3
"""Determines if a markov chain is absorbing."""

import numpy as np


def absorbing(P):
    """Return True if the markov chain P is absorbing, else False."""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    n = P.shape[0]
    if P.shape != (n, n):
        return False
    reach = np.diag(P) == 1
    if not np.any(reach):
        return False
    while True:
        new = reach | np.any(P[:, reach] > 0, axis=1)
        if np.array_equal(new, reach):
            return bool(np.all(reach))
        reach = new
