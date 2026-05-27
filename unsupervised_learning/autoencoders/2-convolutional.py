#!/usr/bin/env python3
"""Builds a convolutional autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Return the encoder, decoder, and full convolutional autoencoder."""
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input
    for f in filters:
        x = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.Model(encoder_input, x)

    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input
    reversed_filters = list(reversed(filters))
    for f in reversed_filters[:-1]:
        x = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu'
        )(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(
        reversed_filters[-1], (3, 3), padding='valid', activation='relu'
    )(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    decoder_output = keras.layers.Conv2D(
        input_dims[-1], (3, 3), padding='same', activation='sigmoid'
    )(x)
    decoder = keras.Model(decoder_input, decoder_output)

    auto_input = keras.Input(shape=input_dims)
    auto = keras.Model(auto_input, decoder(encoder(auto_input)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
