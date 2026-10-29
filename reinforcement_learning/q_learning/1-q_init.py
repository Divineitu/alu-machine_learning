#!/usr/bin/env python3
"""Initializes the Q-table."""

import numpy as np


def q_init(env):
    """Initialize the Q-table for an environment.

    Args:
        env: the FrozenLakeEnv instance

    Returns:
        numpy.ndarray of zeros with shape (states, actions)
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
