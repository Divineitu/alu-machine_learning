#!/usr/bin/env python3
"""Builds a variational autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Return the encoder, decoder, and full variational autoencoder."""
    K = keras.backend

    inputs = keras.Input(shape=(input_dims,))
    h = inputs
    for nodes in hidden_layers:
        h = keras.layers.Dense(nodes, activation='relu')(h)
    z_mean = keras.layers.Dense(latent_dims, activation=None)(h)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(h)

    def sampling(args):
        """Sample z from the latent distribution via the reparam trick."""
        m, lv = args
        batch = K.shape(m)[0]
        dim = K.int_shape(m)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return m + K.exp(0.5 * lv) * epsilon

    z = keras.layers.Lambda(
        sampling, output_shape=(latent_dims,)
    )([z_mean, z_log_var])
    encoder = keras.Model(inputs, [z, z_mean, z_log_var])

    decoder_input = keras.Input(shape=(latent_dims,))
    d = decoder_input
    for nodes in reversed(hidden_layers):
        d = keras.layers.Dense(nodes, activation='relu')(d)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(d)
    decoder = keras.Model(decoder_input, decoder_output)

    outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs, outputs)

    def vae_loss(y_true, y_pred):
        """Sum of reconstruction BCE and KL divergence."""
        reconstruction_loss = keras.losses.binary_crossentropy(y_true, y_pred)
        reconstruction_loss *= input_dims
        kl_loss = (
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        )
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return K.mean(reconstruction_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)
    return encoder, decoder, auto
