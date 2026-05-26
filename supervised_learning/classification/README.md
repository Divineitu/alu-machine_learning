# Classification

Implementations of binary and multiclass classifiers built from scratch with
NumPy: a single neuron, a one-hidden-layer neural network, and a deep neural
network. The deep network supports sigmoid and tanh hidden activations with a
softmax output for multiclass classification, and can be saved to and loaded
from disk via pickle.

## Files

- `0-neuron.py` through `7-neuron.py` — `Neuron` class: constructor and public
  attributes, then private attributes with getters, forward propagation, cost,
  evaluation, gradient descent, and training (with optional verbose logging
  and a cost plot).
- `8-neural_network.py` through `15-neural_network.py` — `NeuralNetwork` class
  with one hidden layer: constructor, private attributes with getters, forward
  propagation, cost, evaluation, gradient descent, and training (with optional
  verbose logging and a cost plot).
- `16-deep_neural_network.py` through `23-deep_neural_network.py` —
  `DeepNeuralNetwork` class for binary classification with He initialization,
  forward propagation, cost, evaluation, gradient descent, and training.
- `24-one_hot_encode.py` — converts a label vector into a one-hot matrix.
- `25-one_hot_decode.py` — converts a one-hot matrix back into a label vector.
- `26-deep_neural_network.py` — adds `save` and `load` for persistence.
- `27-deep_neural_network.py` — extends the deep network to multiclass
  classification with softmax output and categorical cross-entropy cost.
- `28-deep_neural_network.py` — adds the `activation` argument (`sig` or
  `tanh`) to control the hidden layer activation function.

## Requirements

- Python 3
- NumPy
- Matplotlib (used by training methods that plot the cost)

## Data

The example main files load datasets (`Binary_Train.npz`, `Binary_Dev.npz`,
`MNIST.npz`) from a sibling `data` directory. The datasets themselves are not
part of this repository.
