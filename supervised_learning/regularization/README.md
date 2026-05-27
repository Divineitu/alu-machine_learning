# Regularization

NumPy and TensorFlow 1.x implementations of the standard regularization
techniques used to fight overfitting: L2 weight regularization, Dropout, and
Early Stopping.

## Files

- `0-l2_reg_cost.py` — adds the L2 penalty to a precomputed cost (NumPy).
- `1-l2_reg_gradient_descent.py` — one pass of gradient descent that
  updates weights and biases with the L2 penalty term included.
- `2-l2_reg_cost.py` — adds the registered regularization losses to the
  cost in a TensorFlow graph.
- `3-l2_reg_create_layer.py` — dense layer with an L2 kernel regularizer
  and He initialization.
- `4-dropout_forward_prop.py` — forward propagation with inverted dropout;
  returns activations and dropout masks per layer.
- `5-dropout_gradient_descent.py` — backward pass that respects the dropout
  masks from the forward pass.
- `6-dropout_create_layer.py` — dense layer with dropout in TensorFlow.
- `7-early_stopping.py` — early stopping decision based on a threshold and
  patience count.

## Requirements

- Python 3.5
- NumPy 1.15
- TensorFlow 1.12

## Data

The NumPy-based examples load `MNIST.npz` from a sibling `data` directory.
The dataset is not committed to this repository.
