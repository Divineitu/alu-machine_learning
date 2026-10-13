#!/usr/bin/env python3
"""Baum-Welch algorithm for a hidden markov model."""

import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Return the converged Transition and Emission matrices."""
    try:
        T = Observations.shape[0]
        M, N = Emission.shape
        if Transition.shape != (M, M) or Initial.shape != (M, 1):
            return None, None
        A = Transition.copy()
        E = Emission.copy()
        for _ in range(iterations):
            F = np.zeros((M, T))
            B = np.zeros((M, T))
            F[:, 0] = Initial[:, 0] * E[:, Observations[0]]
            for t in range(1, T):
                F[:, t] = np.matmul(F[:, t - 1], A) * E[:, Observations[t]]
            B[:, -1] = 1
            for t in range(T - 2, -1, -1):
                B[:, t] = np.matmul(A, E[:, Observations[t + 1]] *
                                    B[:, t + 1])
            P = np.sum(F[:, -1])
            xi = (F[:, np.newaxis, :-1] * A[:, :, np.newaxis] *
                  (E[:, Observations[1:]] * B[:, 1:])[np.newaxis]) / P
            gamma = F * B / P
            A = np.sum(xi, axis=2) / np.sum(gamma[:, :-1],
                                            axis=1)[:, np.newaxis]
            for k in range(N):
                E[:, k] = np.sum(gamma[:, Observations == k], axis=1)
            E = E / np.sum(gamma, axis=1)[:, np.newaxis]
        return A, E
    except Exception:
        return None, None
