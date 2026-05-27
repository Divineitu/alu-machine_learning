# Optimization

A walkthrough of the standard set of neural network optimization techniques,
implemented in NumPy and TensorFlow 1.x: feature normalization, mini-batch
gradient descent, momentum, RMSProp, Adam, inverse time decay for the
learning rate, and batch normalization. The final task ties them all
together into a single train-and-save routine.

## Files

- `0-norm_constants.py` — mean and standard deviation per feature.
- `1-normalize.py` — standardize a matrix with precomputed constants.
- `2-shuffle_data.py` — shuffle two arrays with the same permutation.
- `3-mini_batch.py` — load a saved model and train it with mini-batches.
- `4-moving_average.py` — bias-corrected weighted moving average.
- `5-momentum.py` — manual gradient descent with momentum.
- `6-momentum.py` — momentum optimization op in TensorFlow.
- `7-RMSProp.py` — manual RMSProp update.
- `8-RMSProp.py` — RMSProp optimization op in TensorFlow.
- `9-Adam.py` — manual Adam update with bias correction.
- `10-Adam.py` — Adam optimization op in TensorFlow.
- `11-learning_rate_decay.py` — stepwise inverse time decay (NumPy).
- `12-learning_rate_decay.py` — stepwise inverse time decay op (TensorFlow).
- `13-batch_norm.py` — batch normalization in NumPy.
- `14-batch_norm.py` — batch normalization layer in TensorFlow.
- `15-model.py` — full model with Adam, mini-batch training, learning rate
  decay, and batch normalization.

## Requirements

- Python 3.5
- NumPy 1.15
- TensorFlow 1.12

## Data and checkpoints

Tasks 6, 8, 10, and 12 expect a `graph.ckpt` (`.data`, `.index`, `.meta`)
checkpoint and the MNIST dataset (`MNIST.npz`) in a sibling `data`
directory. These artifacts are not committed to the repository.
