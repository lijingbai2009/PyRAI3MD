"""
======================================================================
This module provides the gradient propagation layers of NNsMD.

Energy and coupling networks predict derivatives with respect to the
internal feature representation, so the chain rule back to cartesian
coordinates is applied inside the graph by contracting those derivatives
with the derivatives of the features themselves. The layers here perform
that contraction and, for the empty case, emit a correctly shaped zero
gradient.

Sep 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import tensorflow as tf

import keras
from keras import ops


def batch_preserving_contract(a, b, axis_a, axis_b):
    """
    Contract two tensors over one axis each while keeping the batch axis.

    The legacy call sites used a Keras 2 batch dot here. Its semantics are
    "the leading axis of both operands is the same batch, sum over the
    given pair of axes, and concatenate whatever is left", which a plain
    tensordot does not reproduce because it keeps both batch axes. The
    einsum subscripts are generated from the ranks so every configured
    axis pair behaves the same way.

    Args:
        a: First operand, batch axis first.
        b: Second operand, batch axis first.
        axis_a (int): Axis of `a` to contract.
        axis_b (int): Axis of `b` to contract.

    Returns:
        tf.tensor: Contraction with the batch axis kept exactly once.

    """
    letters = "abcdefghijklmnopqrstuvwxyz"
    rank_a, rank_b = len(a.shape), len(b.shape)
    axis_a, axis_b = axis_a % rank_a, axis_b % rank_b
    batch, summed = letters[0], letters[1]
    free = iter(letters[2:])

    def labels(rank, contracted):
        """Assign one subscript letter per axis of an operand."""
        out = []
        for index in range(rank):
            if index == 0:
                out.append(batch)
            elif index == contracted:
                out.append(summed)
            else:
                out.append(next(free))
        return out

    labels_a = labels(rank_a, axis_a)
    labels_b = labels(rank_b, axis_b)
    kept = [batch]
    kept += [label for index, label in enumerate(labels_a) if index not in (0, axis_a)]
    kept += [label for index, label in enumerate(labels_b) if index not in (0, axis_b)]
    subscripts = "%s,%s->%s" % ("".join(labels_a), "".join(labels_b), "".join(kept))
    return tf.einsum(subscripts, a, b)


class EmptyGradient(keras.layers.Layer):
    """
    Layer to generate empty gradient output.
    """

    def __init__(self, mult_states=1, atoms=1, **kwargs):
        """
        Initialize empty gradient layer.

        Args:
            mult_states (int): Number of output states.
            atoms (int): Number of atoms.
            **kwargs
        """
        super(EmptyGradient, self).__init__(**kwargs)
        self.mult_states = mult_states
        self.atoms = atoms
        self.out_shape = tf.constant([self.mult_states, self.atoms, 3], dtype=tf.int32)

    def call(self, inputs, **kwargs):
        """
        Generate an empty gradient placeholder.

        Args:
            inputs (tf.tensor): Energy tensor.
            **kwargs:

        Returns:
            out (tf.tensor): Zeros of shape (batch, states, atoms, 3).

        """
        pot = inputs
        batch_shape = ops.expand_dims(ops.shape(pot)[0], axis=0)
        out_shape = tf.concat((batch_shape, tf.cast(self.out_shape, dtype=batch_shape.dtype)), axis=0)
        out = tf.zeros(out_shape)
        return out

    def get_config(self):
        """Update config for layer."""
        config = super(EmptyGradient, self).get_config()
        config.update({"mult_states": self.mult_states, 'atoms': self.atoms})
        return config


class PropagateEnergyGradient(keras.layers.Layer):
    """
    Layer to propagate the gradients with precomputed layers.
    """

    def __init__(self, mult_states=1, **kwargs):
        """
        Initialize layer.

        Args:
            mult_states (int): Number of states
            **kwargs
        """
        super(PropagateEnergyGradient, self).__init__(**kwargs)
        self.mult_states = mult_states

    def call(self, inputs, **kwargs):
        """
        Propagate gradients.

        Args:
            inputs: [grads, grads2]
            - grads (tf.tensor): Gradient for NN of shape (batch, states, features)
            - grads2 (tf.tensor): Gradients of static features. (batch, features, atoms, 3)
            **kwargs:

        Returns:
            out (tf.tensor): Gradients with respect to coordinates.

        """
        grads, grads2 = inputs
        out = tf.einsum("bsf,bfij->bsij", grads, grads2)
        return out

    def get_config(self):
        """Update config for layer."""
        config = super(PropagateEnergyGradient, self).get_config()
        config.update({"mult_states": self.mult_states})
        return config


class PropagateNACGradient(keras.layers.Layer):
    """
    Propagate partial gradients for virtual NAC potentials.
    """

    def __init__(self, mult_states=1, atoms=1, **kwargs):
        """
        Initialize layer.

        Args:
            mult_states (int): number of states
            atoms (int): number of atoms
            **kwargs:
        """
        super(PropagateNACGradient, self).__init__(**kwargs)
        self.mult_states = mult_states
        self.atoms = atoms

    def call(self, inputs, **kwargs):
        """
        Propagate gradients for virtual NACs.

        Args:
            inputs: [grads, grads2]
            - grads (tf.tensor): Gradient for NN of shape (batch, states, atoms, features)
            - grads2 (tf.tensor): Gradients of static features. (batch, features, atoms, 3)
            **kwargs:

        Returns:
            out (tf.tensor): Gradients with respect to coordinates.

        """
        grads, grads2 = inputs
        # The virtual NAC of atom i only depends on the feature derivative
        # of atom i, which is the diagonal over the two atom axes.
        out = tf.einsum("bsaf,bfaj->bsaj", grads, grads2)
        return out

    def get_config(self):
        """Update config for layer."""
        config = super(PropagateNACGradient, self).get_config()
        config.update({"mult_states": self.mult_states, 'atoms': self.atoms})
        return config


class PropagateNACGradient2(keras.layers.Layer):
    """
    Layer to propagate direct gradient predictions for NACs.
    """

    def __init__(self, axis=(2, 1), **kwargs):
        """
        Initialize layer

        Args:
            axis (tuple): Which axis the contraction is done on. Default is (2,1)
            **kwargs:
        """
        super(PropagateNACGradient2, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        """
        Contract partial gradients over the configured axes.

        Args:
            inputs: [grads, grads2]
            - grads (tf.tensor): Gradient for NN.
            - grads2 (tf.tensor): Gradients of static features.
            **kwargs:

        Returns:
            out (tf.tensor): Contracted gradient.

        """
        grads, grads2 = inputs
        out = batch_preserving_contract(grads, grads2, self.axis[0], self.axis[1])
        return out

    def get_config(self):
        """Update config for layer."""
        config = super(PropagateNACGradient2, self).get_config()
        config.update({"axis": self.axis})
        return config
