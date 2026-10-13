#!/usr/bin/env python3
"""Forward algorithm for a hidden markov model."""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Return the likelihood P and forward path probabilities F (N, T)."""
    try:
        T = Observation.shape[0]
        N = Emission.shape[0]
        if Transition.shape != (N, N) or Initial.shape != (N, 1):
            return None, None
        F = np.zeros((N, T))
        F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
        for t in range(1, T):
            F[:, t] = np.sum(F[:, t - 1, np.newaxis] * Transition *
                             Emission[:, Observation[t]], axis=0)
        return np.sum(F[:, -1]), F
    except Exception:
        return None, None
