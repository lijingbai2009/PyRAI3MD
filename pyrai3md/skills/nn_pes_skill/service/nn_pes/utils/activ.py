"""
======================================================================
This module provides the smooth activation functions of NNsMD and the
resolver that turns a serialized hyperparameter activation into a
callable.

leaky_softplus and shifted_softplus keep the network differentiable
everywhere, which a plain ReLU does not. get_activation resolves the
three forms a hyperparameter file may hold: a live callable, a stock
name, or the legacy {'class_name', 'config'} envelope. The last one is
resolved here on purpose: the Keras deserialization machinery stopped
passing a config to registered plain functions in Keras 2.11, so relying
on it silently drops parameters such as alpha.

Sep 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import math

import keras
import keras.activations
import keras.ops as ops


def leaky_softplus(alpha=0.3):
    """
    Leaky softplus activation function similar to leakyRELU but smooth.

    Args:
        alpha (float, optional): Leaking slope. The default is 0.3.

    Returns:
        func: lambda function of x.

    """
    return lambda x: ops.softplus(x) * (1 - alpha) + alpha * x


def shifted_softplus(x):
    """
    Softplus function shifted downwards so that f(0) is close to zero.

    Args:
        x (tf.tensor): Activation input.

    Returns:
        tf.tensor: Activation.

    """
    return ops.softplus(x) - math.log(2.0)


# Names that may appear as class_name in a serialized activation.
ACTIVATIONS = {
    'leaky_softplus': leaky_softplus,
    'shifted_softplus': shifted_softplus,
}

# A factory only becomes an activation once its config has been applied;
# a plain activation function is usable as it stands. Separating the two
# keeps an empty config from returning the factory itself.
FACTORIES = {'leaky_softplus'}


def get_activation(spec):
    """
    Resolve an activation specification into a callable.

    Args:
        spec: None, a callable, a stock activation name, or a dict
              {'class_name': name, 'config': keywords}.

    Returns:
        func: Activation function to hand to a Dense layer.

    Raises:
        ValueError: If a class_name is neither known nor a stock
                    activation.

    """
    if spec is None:
        return keras.activations.get(None)
    if callable(spec):
        return spec
    if isinstance(spec, str):
        if spec in FACTORIES:
            return ACTIVATIONS[spec]()
        return keras.activations.get(spec)

    if isinstance(spec, dict):
        name = spec.get('class_name', spec.get('name'))
        config = spec.get('config', {}) or {}
        if name is None:
            raise ValueError(f"Activation dict without class_name: {spec}")
        if name in FACTORIES:
            return ACTIVATIONS[name](**config)
        if name in ACTIVATIONS:
            if config:
                raise ValueError(f"Activation '{name}' does not accept config {config}")
            return ACTIVATIONS[name]
        if not config:
            return keras.activations.get(name)
        # A stock activation carrying a config cannot be honoured by the
        # Keras resolver either, so refuse instead of dropping it.
        raise ValueError(f"Activation '{name}' does not accept config {config}")

    raise TypeError(f"Cannot interpret activation specification {spec!r}")
