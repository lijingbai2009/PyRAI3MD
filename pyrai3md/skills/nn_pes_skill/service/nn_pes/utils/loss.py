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

import numpy as np

import tensorflow as tf

import keras
from keras import ops


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
    Squared error between couplings that ignores state signs.

    Non adiabatic couplings change sign together with the phase of the
    electronic states they connect, which is arbitrary per quantum
    chemistry calculation. The loss minimizes over all sign combinations
    instead of comparing the raw arrays.
    """

    def __init__(self, name='phaseless_loss', number_state=2, shape_nac=(1, 1), **kwargs):
        """
        Initialize the loss and enumerate the sign combinations.

        Args:
            name (str): Loss name.
            number_state (int): Number of electronic states.
            shape_nac (tuple): Shape of one coupling without batch.
            **kwargs
        """
        super(NACphaselessLoss, self).__init__(name=name, **kwargs)
        self.number_state = number_state
        self.shape_nac = shape_nac

        phase_stat = np.array([[1], [-1]])
        for _ in range(number_state - 1):
            temp_len = len(phase_stat)
            temp = np.expand_dims(np.array([1] * temp_len + [-1] * temp_len), axis=-1)
            phase_stat = np.concatenate([phase_stat, phase_stat], axis=0)
            phase_stat = np.concatenate([phase_stat, temp], axis=-1)

        phase_stat_mat = np.expand_dims(phase_stat, axis=1) * np.expand_dims(phase_stat, axis=-1)
        idxs = np.triu_indices(number_state, k=1)
        phase_end = np.unique(phase_stat_mat[..., idxs[0], idxs[1]], axis=0)

        # Expand for broadcasting: combination, batch and the coupling axes
        phase_end = np.expand_dims(phase_end, axis=1)
        for _ in range(len(shape_nac)):
            phase_end = np.expand_dims(phase_end, axis=-1)

        self.phase_combo = tf.constant(phase_end, dtype=keras.backend.floatx())
        self.reduce_mean = list(range(1, len(shape_nac) + 3))

    def call(self, y_true, y_pred):
        """
        Return the smallest squared error over all sign combinations.

        Args:
            y_true (tf.tensor): Reference couplings.
            y_pred (tf.tensor): Predicted couplings.

        Returns:
            tf.tensor: Loss value.

        """
        phase_combo = tf.cast(self.phase_combo, y_pred.dtype)
        se = ops.square(ops.expand_dims(y_true, axis=0) - phase_combo * ops.expand_dims(y_pred, axis=0))
        mse = ops.mean(se, axis=self.reduce_mean)
        out = ops.min(mse, axis=0)
        return out

    def get_config(self):
        """Return the config dictionary of the loss instance."""
        config = super(NACphaselessLoss, self).get_config()
        config.update({'number_state': self.number_state,
                       'shape_nac': self.shape_nac})
        return config
