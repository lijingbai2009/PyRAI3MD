"""
======================================================================
This module provides the loss, metric and history helpers of NNsMD.

Training a potential surface mixes targets of very different magnitude,
so the metrics report an error in physical units through a scale that is
installed after the scaler has seen the data, and the non adiabatic
couplings need a loss that is blind to the arbitrary sign of the
wavefunction.

Sep 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import tensorflow as tf

import keras
from keras import ops

from .phase import reduced_phase_map


class ScaledMeanAbsoluteError(keras.metrics.MeanAbsoluteError):
    """
    Mean absolute error reported in physical units.

    The training data is standardized, so an error in scaled space has no
    direct meaning. This metric multiplies both arguments by a scale that
    is set from the scaler once it has been fitted, which keeps the
    reported number comparable with the raw energies or gradients.
    """

    def __init__(self, scaling_shape=(), name='mean_absolute_error', **kwargs):
        """
        Initialize the metric.

        Args:
            scaling_shape (tuple): Broadcast shape of the scale.
            name (str): Name fixing the history key of this metric.
            **kwargs
        """
        super(ScaledMeanAbsoluteError, self).__init__(name=name, **kwargs)
        self.scale = self.add_weight(name='scale_mae',
                                     shape=scaling_shape,
                                     initializer=keras.initializers.Ones(),
                                     dtype=keras.backend.floatx())
        self.scaling_shape = scaling_shape

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Accumulate the scaled absolute error.

        Args:
            y_true: Reference values.
            y_pred: Predicted values.
            sample_weight: Optional per-sample weights.

        Returns:
            None.
        """
        scaled_true = self.scale * y_true
        scaled_pred = self.scale * y_pred
        return super(ScaledMeanAbsoluteError, self).update_state(scaled_true, scaled_pred,
                                                                 sample_weight=sample_weight)

    def get_config(self):
        """Return the serializable config of the metric."""
        config = super(ScaledMeanAbsoluteError, self).get_config()
        config.update({"scaling_shape": self.scaling_shape})
        return config

    def set_scale(self, scale):
        """
        Install the physical scale of the error.

        Args:
            scale: Broadcastable array of scale factors.

        Returns:
            None.
        """
        self.scale.assign(scale)


class ZeroEmptyLoss(keras.losses.Loss):
    """
    Constant zero loss used to mute an output that is not trained.
    """

    def __init__(self, **kwargs):
        """
        Initialize the placeholder loss.
        """
        super(ZeroEmptyLoss, self).__init__(**kwargs)
        self.zero_empty = tf.constant(0.0)

    def call(self, y_true, y_pred):
        """
        Return a per-sample zero so the reduction stays well defined.

        Args:
            y_true: Reference values, only used for the batch shape.
            y_pred: Predicted values, only used for the batch shape.

        Returns:
            tf.tensor: Zeros of shape (batch,).
        """
        return ops.zeros(ops.shape(y_pred)[:1], dtype=keras.backend.floatx()) + self.zero_empty

    def get_config(self):
        """Return the serializable config of the loss."""
        return super(ZeroEmptyLoss, self).get_config()


def get_lr_metric(optimizer):
    """
    Obtain the learning rate as a metric.

    Args:
        optimizer: Optimizer used for training.

    Returns:
        func: Metric function reporting the current learning rate.

    """

    def lr(y_true, y_pred):
        return optimizer.learning_rate

    return lr


def r2_metric(y_true, y_pred):
    """
    Compute the coefficient of determination.

    Args:
        y_true (tf.tensor): True y-values.
        y_pred (tf.tensor): Predicted y-values.

    Returns:
        tf.tensor: r2 metric.

    """
    ss_res = ops.sum(ops.square(y_true - y_pred))
    ss_tot = ops.sum(ops.square(y_true - ops.mean(y_true)))
    return 1 - ss_res / (ss_tot + keras.backend.epsilon())


def merge_hist(hist1, hist2):
    """
    Merge two history dicts obtained from consecutive fits.

    Args:
        hist1 (dict): History dict of the first fit.
        hist2 (dict): History dict of the second fit.

    Returns:
        outhist (dict): Concatenated history.

    """
    outhist = {}
    for x, y in hist1.items():
        outhist.update({x: hist1[x] + hist2[x]})
    return outhist


class NACphaselessLoss(keras.losses.Loss):
    """
    Squared error between couplings that ignores the arbitrary state signs.

    Non adiabatic couplings change sign together with the phase of the
    electronic states they connect, which is arbitrary per quantum
    chemistry calculation. One state phase has to hold across every pair
    that state appears in, so the candidates are the legal phase assignments
    of the whole state ladder, not independent per-channel signs.

    The candidates are built once from `number_state` and the declared
    pairs (see `phase.reduced_phase_map`) and stored as a table of sign
    choices. Each training step therefore computes only the two per-pair
    errors and selects between them, which keeps a partial or disconnected
    pair set on the same path as a complete one.
    """

    def __init__(self, name='phaseless_loss', number_state=2, pairs=None,
                 **kwargs):
        """
        Initialize the loss and its phase table.

        Args:
            name (str): Loss name.
            number_state (int): Number of electronic states.
            pairs (list): State pairs in the order of the coupling channel
                axis. Directions may be written either way; the sign of a
                pair does not depend on it.
            **kwargs

        Raises:
            ValueError: If no pair was declared, or a pair is malformed,
                repeated or outside the state ladder.
        """
        super(NACphaselessLoss, self).__init__(name=name, **kwargs)
        self.number_state = int(number_state)
        declared = [[int(pair[0]), int(pair[1])] for pair in (pairs or ())]
        self.pairs = declared
        table = reduced_phase_map(
            self.number_state,
            tuple((pair[0], pair[1]) for pair in declared))
        self.phase_map = tf.constant(table, dtype=tf.int32)

    def call(self, y_true, y_pred):
        """
        Return the smallest squared error over the legal phase assignments.

        Args:
            y_true (tf.tensor): Reference couplings, (batch, pairs, ...).
            y_pred (tf.tensor): Predicted couplings, same shape.

        Returns:
            tf.tensor: One loss value per sample.
        """
        y_true = tf.cast(y_true, y_pred.dtype)
        layout = tf.shape(y_pred)
        flat_pred = tf.reshape(y_pred, tf.stack([layout[0], layout[1], -1]))
        flat_true = tf.reshape(y_true, tf.stack([layout[0], layout[1], -1]))

        # The two sign choices of one pair, reduced over the vector axis.
        plus = tf.reduce_mean(tf.square(flat_pred - flat_true), axis=-1)
        minus = tf.reduce_mean(tf.square(flat_pred + flat_true), axis=-1)
        pair_error = tf.stack([plus, minus], axis=-1)

        # table is (candidate, pair); gather wants the pair axis in front of
        # the candidate axis and one copy per sample.
        table = tf.transpose(self.phase_map)
        index = tf.broadcast_to(
            table[None],
            tf.stack([layout[0], tf.shape(table)[0], tf.shape(table)[1]]))
        selected = tf.gather(pair_error, index, axis=2, batch_dims=2)
        candidates = tf.reduce_mean(selected, axis=1)
        return tf.reduce_min(candidates, axis=1)

    def get_config(self):
        """Return the config dictionary of the loss instance."""
        config = super(NACphaselessLoss, self).get_config()
        config.update({'number_state': self.number_state,
                       'pairs': self.pairs})
        return config
