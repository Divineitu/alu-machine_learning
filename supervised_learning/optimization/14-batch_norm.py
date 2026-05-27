#!/usr/bin/env python3
"""Creates a batch normalization layer in tensorflow."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Return the activated output of a dense + batch norm layer."""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=init)
    z = layer(prev)
    mean, variance = tf.nn.moments(z, axes=[0])
    gamma = tf.Variable(tf.ones((n,)), name='gamma')
    beta = tf.Variable(tf.zeros((n,)), name='beta')
    z_norm = tf.nn.batch_normalization(z, mean, variance, beta, gamma, 1e-8)
    if activation is None:
        return z_norm
    return activation(z_norm)
