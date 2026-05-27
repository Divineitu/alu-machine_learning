#!/usr/bin/env python3
"""Creates a layer of a neural network using dropout."""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Return the output of a dense layer with dropout applied."""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    drop = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(
        units=n, activation=activation,
        kernel_initializer=init, kernel_regularizer=drop
    )
    return layer(prev)
