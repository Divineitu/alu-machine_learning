#!/usr/bin/env python3
"""Calculates the weighted moving average of a data set."""


def moving_average(data, beta):
    """Return the bias-corrected weighted moving averages of data."""
    averages = []
    v = 0
    for i, value in enumerate(data, 1):
        v = beta * v + (1 - beta) * value
        averages.append(v / (1 - beta ** i))
    return averages
