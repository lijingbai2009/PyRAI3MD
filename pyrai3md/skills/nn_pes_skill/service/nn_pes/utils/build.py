"""
======================================================================
This module provides the build helpers shared by the NNsMD models.

Keras 3 materializes the weights of a subclassed model when its layers
are first called, not when build is invoked with a shape, so a model that
only called self.build() ends up without any trainable weight until real
data arrives. That is too late for the callers, which read and write the
normalization weights right after construction. trace_model performs one
forward pass over a throwaway geometry for exactly that purpose.

Sep 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np


def trace_input(atoms):
    """
    Build a throwaway geometry that keeps the feature block well defined.

    Args:
        atoms (int): Number of atoms.

    Returns:
        np.array: Coordinates of shape (1, atoms, 3) with mutually
                  distinct atoms.

    """
    coords = np.arange(atoms * 3, dtype=np.float32).reshape(1, atoms, 3)
    coords = coords / float(atoms * 3) + 0.25
    return coords


def trace_model(model, atoms):
    """
    Force the weights of a subclassed model into existence.

    Args:
        model: Keras model whose call accepts coordinates.
        atoms (int): Number of atoms of the model.

    Returns:
        int: Number of weight tensors after the trace.

    """
    model(trace_input(atoms), training=False)
    return len(model.weights)
