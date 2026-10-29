#!/usr/bin/env python3
"""Epsilon-greedy action selection."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Pick the next action with epsilon-greedy.

    Args:
        Q: numpy.ndarray containing the Q-table
        state: the current state
        epsilon: the probability of exploring

    Returns:
        the next action index
    """
    p = np.random.uniform(0, 1)
    if p > epsilon:
        return np.argmax(Q[state, :])
    return np.random.randint(0, Q.shape[1])
