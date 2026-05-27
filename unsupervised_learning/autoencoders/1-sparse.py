#!/usr/bin/env python3
"""Builds a sparse autoencoder with L1 regularization on the latent layer."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Return the encoder, decoder, and full sparse autoencoder models."""
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent = keras.layers.Dense(
        latent_dims, activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)
    encoder = keras.Model(encoder_input, latent)

    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, decoder_output)

    auto_input = keras.Input(shape=(input_dims,))
    auto = keras.Model(auto_input, decoder(encoder(auto_input)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
