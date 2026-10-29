# Q-learning

Tabular Q-learning on OpenAI Gym's FrozenLake environment: loading the
environment, building a Q-table, choosing actions with epsilon-greedy,
training with the Bellman equation, and playing an episode with the learned
policy.

## Files

- `0-load_env.py` — `load_frozen_lake(desc=None, map_name=None,
  is_slippery=False)`: loads a `FrozenLakeEnv` from a custom map, a pre-made
  map, or a random 8x8 map when neither is given.
- `1-q_init.py` — `q_init(env)`: returns a Q-table of zeros with one row per
  state and one column per action.
- `2-epsilon_greedy.py` — `epsilon_greedy(Q, state, epsilon)`: explores with
  a random action with probability `epsilon`, otherwise exploits the best
  known action.
- `3-q_learning.py` — `train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05)`: runs
  Q-learning, giving a reward of -1 for falling in a hole and decaying
  epsilon exponentially between episodes. Returns the Q-table and the reward
  of each episode.
- `4-play.py` — `play(env, Q, max_steps=100)`: plays one episode always
  exploiting the Q-table, rendering each board state, and returns the total
  reward.

## Requirements

- Ubuntu 16.04, Python 3.5, numpy 1.15, gym 0.7
- pycodestyle 2.4
