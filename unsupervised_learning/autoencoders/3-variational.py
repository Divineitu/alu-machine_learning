#!/usr/bin/env python3
"""Builds a variational autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Return the encoder, decoder, and full variational autoencoder."""
    K = keras.backend

    encoder_input = keras.Input(shape=(input_dims,))
    h = encoder_input
    for nodes in hidden_layers:
        h = keras.layers.Dense(nodes, activation='relu')(h)
    mu = keras.layers.Dense(latent_dims, activation=None)(h)
    log_var = keras.layers.Dense(latent_dims, activation=None)(h)

    def sampling(args):
        """Sample z from the latent distribution via the reparam trick."""
        m, lv = args
        epsilon = K.random_normal(shape=K.shape(m))
        return m + K.exp(lv / 2) * epsilon

    z = keras.layers.Lambda(sampling)([mu, log_var])
    encoder = keras.Model(encoder_input, [z, mu, log_var])

    decoder_input = keras.Input(shape=(latent_dims,))
    d = decoder_input
    for nodes in reversed(hidden_layers):
        d = keras.layers.Dense(nodes, activation='relu')(d)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(d)
    decoder = keras.Model(decoder_input, decoder_output)

    z_out, mu_out, log_var_out = encoder(encoder_input)
    auto_output = decoder(z_out)
    auto = keras.Model(encoder_input, auto_output)

    def vae_loss(y_true, y_pred):
        """Sum of reconstruction BCE and KL divergence."""
        recon = keras.losses.binary_crossentropy(y_true, y_pred) * input_dims
        kl = -0.5 * K.sum(
            1 + log_var_out - K.square(mu_out) - K.exp(log_var_out), axis=-1
        )
        return recon + kl

    auto.compile(optimizer='adam', loss=vae_loss)
    return encoder, decoder, auto
