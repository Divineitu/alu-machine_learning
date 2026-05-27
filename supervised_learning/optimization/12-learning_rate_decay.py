#!/usr/bin/env python3
"""Creates a stepwise inverse time decay learning rate op in tensorflow."""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Return the stepwise inverse time decay learning rate operation."""
    return tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True
    )
