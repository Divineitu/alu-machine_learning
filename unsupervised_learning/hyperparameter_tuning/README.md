# Hyperparameter Tuning

A from-scratch implementation of a 1D Gaussian process and Bayesian
optimization with the Expected Improvement acquisition function, followed by
a practical hyperparameter tuning script that uses GPyOpt to optimize a
Keras MNIST classifier.

## Files

- `0-gp.py` — `GaussianProcess` class: stores samples and builds the RBF
  kernel matrix.
- `1-gp.py` — adds `predict` for the posterior mean and variance at new
  points.
- `2-gp.py` — adds `update` to append a new observation and refresh the
  kernel.
- `3-bayes_opt.py` — `BayesianOptimization` class initialized over a fixed
  grid of acquisition samples.
- `4-bayes_opt.py` — adds the `acquisition` method using Expected
  Improvement.
- `5-bayes_opt.py` — adds `optimize`, which runs the EI loop until either
  the iteration budget is exhausted or a duplicate point is proposed.
- `6-bayes_opt.py` — applied script: tunes five hyperparameters of a Keras
  MNIST classifier (learning rate, hidden units, dropout, L2 weight, batch
  size) with GPyOpt; saves a checkpoint per configuration, plots the
  convergence to `convergence.png`, and writes a summary to
  `bayes_opt.txt`.

## Requirements

- Python 3
- NumPy
- SciPy (for `scipy.stats.norm`)
- TensorFlow / Keras (task 6 only)
- GPyOpt (task 6 only)
- Matplotlib (task 6 only)
