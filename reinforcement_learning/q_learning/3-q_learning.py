#!/usr/bin/env python3
"""Q-learning training loop."""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Perform Q-learning.

    Args:
        env: the FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: the learning rate
        gamma: the discount rate
        epsilon: the initial threshold for epsilon greedy
        min_epsilon: the minimum value epsilon should decay to
        epsilon_decay: the decay rate for epsilon between episodes

    Returns:
        Q, total_rewards
        Q is the updated Q-table
        total_rewards is a list of the rewards per episode
    """
    max_epsilon = epsilon
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        rewards = 0
        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, _ = env.step(action)
            if done and reward == 0:
                reward = -1
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
            state = new_state
            rewards += reward
            if done:
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * episode)
        total_rewards.append(rewards)
    return Q, total_rewards
