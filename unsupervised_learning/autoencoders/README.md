# Autoencoders

Four autoencoder variants built with `tensorflow.keras`: a vanilla
fully-connected autoencoder, a sparse autoencoder with L1 regularization on
the latent layer, a convolutional autoencoder for image data, and a
variational autoencoder that learns a probabilistic latent space.

## Files

- `0-vanilla.py` — fully-connected autoencoder with a configurable list of
  hidden layers, ReLU activations, and a sigmoid output.
- `1-sparse.py` — same architecture as the vanilla autoencoder with an L1
  activity regularizer on the latent representation.
- `2-convolutional.py` — convolutional encoder/decoder for image inputs;
  encoder uses Conv2D + MaxPooling, decoder uses Conv2D + UpSampling with a
  valid-padding convolution at the second-to-last step.
- `3-variational.py` — variational autoencoder; the encoder outputs the
  sampled latent, the mean, and the log variance, and the model is trained
  on the sum of reconstruction loss and KL divergence.

## Requirements

- Python 3 (3.5+)
- TensorFlow 1.12 (`tensorflow.keras`)
- NumPy 1.15

## Data

The provided main scripts use the MNIST dataset loaded via
`tensorflow.keras.datasets.mnist`.
