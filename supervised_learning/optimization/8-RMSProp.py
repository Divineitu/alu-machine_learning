#!/usr/bin/env python3
"""Creates the RMSProp training operation in tensorflow."""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Return the RMSProp optimization operation."""
    optimizer = tf.train.RMSPropOptimizer(
        alpha, decay=beta2, epsilon=epsilon
    )
    return optimizer.minimize(loss)
