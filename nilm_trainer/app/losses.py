"""Shared losses returning their value and gradient with respect to predictions."""

from functools import partial

import numpy as np


def binary_crossentropy(target, probability):
    # Match the Keras 2 probability loss, including clipping and the log offset.
    epsilon = np.float32(1e-7)
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    positive = clipped + epsilon
    negative = 1.0 - clipped + epsilon
    value = -np.mean(target * np.log(positive) + (1.0 - target) * np.log(negative))
    gradient = ((1.0 - target) / negative - target / positive) / target.size
    gradient *= (probability >= epsilon) & (probability <= 1.0 - epsilon)
    return float(value), gradient


def mean_squared_error(target, prediction):
    residual = prediction - target
    return float(np.mean(residual**2)), (2.0 / target.size) * residual


def huber(target, prediction, *, delta):
    residual = prediction - target
    quadratic = np.minimum(np.abs(residual), delta)
    value = np.mean(0.5 * quadratic**2 + delta * (np.abs(residual) - quadratic))
    return float(value), np.clip(residual, -delta, delta) / target.size


def resolve_losses(profile):
    losses = {}
    for output, config in profile.items():
        name = config["name"]
        if name == "huber":
            delta = float(config["delta"])
            if not np.isfinite(delta) or delta <= 0:
                raise ValueError("Huber delta must be finite and positive.")
            losses[output] = partial(huber, delta=delta)
        elif name == "mse":
            losses[output] = mean_squared_error
        elif name == "binary_crossentropy":
            losses[output] = binary_crossentropy
        else:
            raise ValueError(f"Unknown training loss: {name}")
    return losses
