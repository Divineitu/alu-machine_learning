#!/usr/bin/env python3
"""Backward algorithm for a hidden markov model."""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Return the likelihood P and backward path probabilities B (N, T)."""
    try:
        T = Observation.shape[0]
        N = Emission.shape[0]
        if Transition.shape != (N, N) or Initial.shape != (N, 1):
            return None, None
        B = np.zeros((N, T))
        B[:, -1] = 1
        for t in range(T - 2, -1, -1):
            B[:, t] = np.sum(Transition * B[:, t + 1] *
                             Emission[:, Observation[t + 1]], axis=1)
        P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
        return P, B
    except Exception:
        return None, None
