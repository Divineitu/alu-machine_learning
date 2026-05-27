#!/usr/bin/env python3
"""Calculates the accuracy of a network's prediction."""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Return the decimal accuracy of a prediction as a tensor."""
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))
