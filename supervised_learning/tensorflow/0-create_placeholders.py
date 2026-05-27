#!/usr/bin/env python3
"""Creates the placeholders for a neural network's inputs and labels."""

import tensorflow as tf


def create_placeholders(nx, classes):
    """Return placeholders x and y for the network's input data and labels."""
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
