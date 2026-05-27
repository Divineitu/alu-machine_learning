#!/usr/bin/env python3
"""Creates the Adam training operation in tensorflow."""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Return the Adam optimization operation."""
    optimizer = tf.train.AdamOptimizer(
        alpha, beta1=beta1, beta2=beta2, epsilon=epsilon
    )
    return optimizer.minimize(loss)
