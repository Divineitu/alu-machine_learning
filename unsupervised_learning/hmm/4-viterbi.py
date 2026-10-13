#!/usr/bin/env python3
"""Viterbi algorithm for a hidden markov model."""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Return the most likely hidden state path and its probability."""
    try:
        T = Observation.shape[0]
        N = Emission.shape[0]
        if Transition.shape != (N, N) or Initial.shape != (N, 1):
            return None, None
        V = np.zeros((N, T))
        back = np.zeros((N, T), dtype=int)
        V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
        for t in range(1, T):
            probs = V[:, t - 1, np.newaxis] * Transition
            back[:, t] = np.argmax(probs, axis=0)
            V[:, t] = np.max(probs, axis=0) * Emission[:, Observation[t]]
        path = [int(np.argmax(V[:, -1]))]
        for t in range(T - 1, 0, -1):
            path.insert(0, int(back[path[0], t]))
        return path, np.max(V[:, -1])
    except Exception:
        return None, None
