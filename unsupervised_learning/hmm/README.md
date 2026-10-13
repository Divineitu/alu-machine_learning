# Hidden Markov Models

Markov chains and hidden markov models implemented with NumPy.

## Files

- `0-markov_chain.py` — `markov_chain(P, s, t=1)`: probability of being in each
  state after `t` iterations.
- `1-regular.py` — `regular(P)`: steady state probabilities of a regular
  markov chain, or `None` if it is not regular.
- `2-absorbing.py` — `absorbing(P)`: whether a markov chain is absorbing.
- `3-forward.py` — `forward(Observation, Emission, Transition, Initial)`:
  forward algorithm, returns the likelihood and forward path probabilities.
- `4-viterbi.py` — `viterbi(Observation, Emission, Transition, Initial)`: most
  likely sequence of hidden states and its probability.
- `5-backward.py` — `backward(Observation, Emission, Transition, Initial)`:
  backward algorithm, returns the likelihood and backward path probabilities.
- `6-baum_welch.py` — `baum_welch(Observations, Transition, Emission, Initial,
  iterations=1000)`: estimates the transition and emission probabilities with
  expectation-maximization.
