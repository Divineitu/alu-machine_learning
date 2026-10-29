#!/usr/bin/env python3
"""Plays an episode with a trained Q-table."""

import numpy as np


def play(env, Q, max_steps=100):
    """Have the trained agent play an episode.

    Args:
        env: the FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        max_steps: maximum number of steps in the episode

    Returns:
        the total rewards for the episode
    """
    state = env.reset()
    env.render()
    total_rewards = 0
    for _ in range(max_steps):
        action = np.argmax(Q[state, :])
        state, reward, done, _ = env.step(action)
        env.render()
        total_rewards += reward
        if done:
            break
    return total_rewards
