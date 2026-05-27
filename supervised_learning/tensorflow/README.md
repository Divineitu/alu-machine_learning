# Tensorflow

A small set of building blocks for training a feedforward neural network with
TensorFlow 1.x. The functions cover placeholders, layers, forward
propagation, loss, accuracy, the training operation, and a full
train/evaluate loop that saves the model to disk and restores it for
inference.

## Files

- `0-create_placeholders.py` — placeholders `x` and `y` for input data and
  one-hot labels.
- `1-create_layer.py` — a dense layer with He initialization and a
  configurable activation function.
- `2-forward_prop.py` — builds the forward propagation graph for a
  multi-layer network.
- `3-calculate_accuracy.py` — accuracy tensor computed from one-hot labels
  and predictions.
- `4-calculate_loss.py` — softmax cross-entropy loss tensor.
- `5-create_train_op.py` — gradient descent training operation.
- `6-train.py` — builds, trains, and saves the classifier; logs training and
  validation cost and accuracy at the 0th, every 100th, and the final
  iteration.
- `7-evaluate.py` — restores a saved model and returns the prediction,
  accuracy, and loss on the provided data.

## Requirements

- Python 3
- TensorFlow 1.x (the code uses `tf.placeholder`,
  `tf.contrib.layers.variance_scaling_initializer`, `tf.layers.Dense`,
  `tf.losses.softmax_cross_entropy`, `tf.train.GradientDescentOptimizer`,
  and `tf.train.Saver`)
- NumPy (used by the main scripts to load data)

## Data

The example main scripts load `MNIST.npz` from a sibling `data` directory.
The dataset itself is not part of this repository.
