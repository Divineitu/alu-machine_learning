#!/usr/bin/env python3
"""Creates a dense layer with He initialization."""

import tensorflow as tf


def create_layer(prev, n, activation):
    """Return the tensor output of a dense layer with n nodes."""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init, name='layer')
    return layer(prev)
