#!/usr/bin/env python3
"""Updates the learning rate using stepwise inverse time decay."""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Return the updated learning rate after stepwise inverse time decay."""
    return alpha / (1 + decay_rate * (global_step // decay_step))
