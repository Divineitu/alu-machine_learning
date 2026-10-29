#!/usr/bin/env python3
"""Loads the FrozenLake environment from OpenAI's gym."""

import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load the pre-made FrozenLakeEnv environment.

    Args:
        desc: None or a list of lists describing a custom map
        map_name: None or a string naming a pre-made map
        is_slippery: boolean, whether the ice is slippery

    Returns:
        the environment
    """
    return gym.make('FrozenLake-v0', desc=desc, map_name=map_name,
                    is_slippery=is_slippery)
