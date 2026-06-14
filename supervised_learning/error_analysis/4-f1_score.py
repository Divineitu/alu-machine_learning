#!/usr/bin/env python3
"""Calculates the F1 score for each class."""

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Return the per-class F1 score from a confusion matrix."""
    p = precision(confusion)
    r = sensitivity(confusion)
    return 2 * p * r / (p + r)
