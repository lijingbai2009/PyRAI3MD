"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module evaluates a trained ensemble on a test set and reports how
far it was wrong.

The numbers are computed here rather than in the framework for one
reason: an error is a difference, and a difference needs the axis the
labels were written on. A training file carries Hartree energies and
Hartree/Bohr derivatives, the network is fed Angstrom, and the reported
metric is eV per Angstrom; only this skill knows all three, and the
scaler that turns a stored fit error into a physical one lives in the
model folder. The framework receives the same record shape it receives
for a training run and does with it what it does with any other: print
it, judge whether it came back, rank scans by it.

What is measured is a mean absolute error, a root mean square error, the
single worst component, and the worst component of each frame. The first
three say how the surface behaved over the set; the last says at which
geometry a user should not have trusted it, which is the question a test
set is run to answer. Nothing here is called relative or percent: an
energy difference can be exactly zero, and a floor invented to keep the
division alive would then be reported as though it had been measured.

The ensemble's member-to-member spread is reported beside the errors and
is not one of them: it is how much the members disagree, and a set of
members that agrees on a wrong surface says nothing about accuracy.

"""

import os
import time

import numpy as np

from ..schema.reference import (BOHR_TO_ANGSTROM, HARTREE_TO_EV,
                                METRIC_ENERGY_UNIT, METRIC_GRADIENT_UNIT,
                                METRIC_SOC_UNIT)
from . import trainer_data as data_io
from .calculator import evaluate

# What one reported property needs to become comparable: the factor that
# puts the *served* value on the axis the labels were written on, the
# factor that puts a label into the printed unit, and that unit.
#
# The two factors are kept apart on purpose. A served gradient or
# coupling is Hartree per Bohr (the boundary multiplies the container's
# per-Angstrom derivative by 0.529177 to hand that over), while a label
# of the same quantity is Hartree per Angstrom, because `match_labels`
# put it there for the network. Subtracting one from the other is not a
# difference of a quantity, it is a difference of two quantities; the
# first factor is what makes them one. Energy and spin-orbit values are
# the same on both sides, so their first factor is unity.
_CHANNELS = {
    'energy': ('energy', 1.0, HARTREE_TO_EV, METRIC_ENERGY_UNIT),
    'gradient': ('grad', 1.0 / BOHR_TO_ANGSTROM, HARTREE_TO_EV,
                 METRIC_GRADIENT_UNIT),
    'nac': ('nac', 1.0 / BOHR_TO_ANGSTROM, HARTREE_TO_EV,
            METRIC_GRADIENT_UNIT),
    'soc': ('soc', 1.0, 1.0, METRIC_SOC_UNIT),
}

# Statistics of one channel. Each is measured over every component of
# every frame of the test set, in the printed unit.
_STATISTICS = ('mae', 'rmse', 'max', 'spread_max')

_RULE = '-' * 79


def load_test_set(config):
    """Read the test set a prediction is asked to evaluate.

    The file is read by the same rules as a training file, because a
    label is a label: the units are checked against the skill's
    convention and the channel counts against the declared ones.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration, carrying `pred_data`.

    Returns
    -------
    dict
        Loaded dataset and its labels expressed on the network's axis.

    Raises
    ------
    ValueError
        If `pred_data` names nothing, or the file holds no geometry.
    """
    path = config.calculation.pred_data
    if not path:
        raise ValueError(
            "jobtype 'prediction' asks how far the trained surface is from "
            "the reference values, but pred_data is empty: name the test "
            "file to evaluate")
    dataset = data_io.load(path)
    data_io.check_channels(config, dataset)
    arrays = data_io.match_labels(dataset)
    if arrays['xyz'].shape[0] < 1:
        raise ValueError("test file %s carries no geometry" % path)
    return {'dataset': dataset, 'labels': arrays}


def predict_set(config, geometries, container):
    """Serve every frame of a set with the profile's channels.

    One geometry at a time through the serving path a dynamics step
    uses: the channel selection, the state axis and the unit factors are
    the ones that answer a step, and a test set that reported them
    differently would not be measuring the surface that runs.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration.
    geometries : numpy.ndarray
        (nframe, natom, 3) coordinates in Angstrom.
    container : object
        Opened container with the profile's ensembles loaded.

    Returns
    -------
    dict
        Channel name to (nframe, ...) predictions in the units the
        backend reports, and the same shapes for the ensemble spread.

    Raises
    ------
    ValueError
        If a channel comes back with a shape the labels cannot be
        compared against.
    """
    served = {}
    spread = {}
    for index in range(geometries.shape[0]):
        results, uncertainty = evaluate(config, geometries[index], container)
        for channel, values in results.items():
            # The comparison channels (`alternative_*`), the member ids and
            # the model count are not properties the test file labels:
            # only the profile's served channels have a reference column.
            if channel not in _CHANNELS:
                continue
            served.setdefault(channel, []).append(np.asarray(values))
            spread.setdefault(channel, []).append(
                np.asarray(uncertainty.get(channel, np.zeros_like(values))))
    return ({channel: np.asarray(items) for channel, items in served.items()},
            {channel: np.asarray(items) for channel, items in spread.items()})


def _label_for(channel, labels):
    """Take one channel's label array, shaped like its predictions.

    Parameters
    ----------
    channel : str
        Reported property name.
    labels : dict
        Labels from `load_test_set`, on the network's axis.

    Returns
    -------
    numpy.ndarray or None
        The label array, or None when the file carries no such channel.
    """
    key = _CHANNELS[channel][0]
    array = labels.get(key)
    if array is None or array.size == 0:
        return None
    if channel == 'soc' and array.ndim == 1:
        return array.reshape(array.shape[0], 1)
    return array


def channel_statistics(difference, spread):
    """Reduce one channel's differences to the reported statistics.

    Parameters
    ----------
    difference : numpy.ndarray
        Label minus prediction, in the printed unit, over every frame.
    spread : numpy.ndarray
        Ensemble spread of the same shape and unit.

    Returns
    -------
    dict
        The scalar statistics, plus 'frame_max': the worst component of
        each frame, kept as a list because a set of frames is judged
        frame by frame.
    """
    absolute = np.abs(np.asarray(difference, dtype=np.float64))
    per_frame = (absolute.reshape(absolute.shape[0], -1).max(axis=1)
                 if absolute.size else np.empty(0))
    spread_abs = np.abs(np.asarray(spread, dtype=np.float64))
    return {
        'mae': float(absolute.mean()) if absolute.size else 0.0,
        'rmse': float(np.sqrt(np.mean(absolute ** 2))) if absolute.size else 0.0,
        'max': float(absolute.max()) if absolute.size else 0.0,
        'spread_max': (float(spread_abs.max()) if spread_abs.size else 0.0),
        'frame_max': [float(value) for value in per_frame],
    }


def products_dir(config, calcdir):
    """Directory the prediction products are written into.

    The calculation directory is what `keep_tmp: 0` clears when the run
    ends, and a test-set evaluation exists to leave its raw comparison
    behind: a product the user asked for does not belong under a switch
    that governs TensorFlow scratch. The files therefore go beside the
    calculation directory, in the run's own directory, where the skill's
    `{project}_{profile}.log` and the framework's run log also live.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration, for the run directory.
    calcdir : str
        Calculation directory, used when no run directory was set.

    Returns
    -------
    str
        Existing directory for the prediction files.
    """
    root = config.environment.scratch_path or calcdir
    path = os.path.abspath(os.path.expanduser(root))
    os.makedirs(path, exist_ok=True)
    return path


def run(config, container, calcdir):
    """Evaluate the trained profile on the test set and report the error.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration, carrying `pred_data`.
    container : object
        Container with the profile's ensembles already loaded.
    calcdir : str
        Calculation directory; the products land beside it, in the run
        directory (see `products_dir`).

    Returns
    -------
    dict
        Result record: per-channel statistics and units, the frame
        count, the wall clock, the raw file paths and the summary path.

    Raises
    ------
    ValueError
        If a served channel has no labels to compare against, or a
        label array cannot be lined up with the predictions.
    """
    started = time.perf_counter()
    outdir = products_dir(config, calcdir)
    loaded = load_test_set(config)
    labels = loaded['labels']
    geometries = labels['xyz']

    served, spread = predict_set(config, geometries, container)

    errors = {}
    units = {}
    raw_files = {}
    frame_rows = {}
    for channel, predicted in sorted(served.items()):
        truth = _label_for(channel, labels)
        if truth is None:
            raise ValueError(
                "the profile serves %s but the test file %s carries no "
                "labels for it; an error needs a reference value"
                % (channel, config.calculation.pred_data))
        _key, to_label, to_metric, unit = _CHANNELS[channel]
        try:
            # One axis before one difference: the labels and the served
            # values are put on the same derivative axis, and only then
            # subtracted. Everything the record and the files carry is
            # afterwards in the single printed unit, so a reader cannot
            # line up a Hartree column against an eV one.
            reference = np.asarray(truth, dtype=np.float64) * to_metric
            served_metric = (np.asarray(predicted, dtype=np.float64)
                             * to_label * to_metric)
            spread_metric = (np.asarray(spread.get(channel, 0.0),
                                        dtype=np.float64)
                             * to_label * to_metric)
            difference = reference - served_metric
        except ValueError as error:
            raise ValueError(
                "test file %s cannot be compared against the served %s "
                "channel: labels of %s against predictions of %s (%s)"
                % (config.calculation.pred_data, channel,
                   tuple(np.shape(truth)), tuple(np.shape(predicted)),
                   error)) from error
        errors[channel] = channel_statistics(difference, spread_metric)
        units[channel] = unit
        frame_rows[channel] = errors[channel].pop('frame_max')
        raw_files[channel] = _write_raw(
            outdir, config, channel, reference, served_metric, difference,
            spread_metric)

    wall = time.perf_counter() - started
    frames = int(geometries.shape[0])
    record = {
        'predict_errors': errors,
        'predict_metric_units': units,
        'predict_frames': frames,
        'predict_wall_seconds': wall,
        'predict_raw_files': raw_files,
    }
    record['predict_summary'] = _write_summary(outdir, config, frame_rows)
    return record


def _stem(config):
    """File stem of this skill's prediction products."""
    return '%s_%s' % (config.calculation.project or 'NNPES',
                      config.calculation.profile)


def _write_raw(outdir, config, channel, truth, predicted, difference,
               spread):
    """Write one channel's raw comparison, legacy column order.

    One row per frame: the reference values, the predictions, the
    absolute deviation of each component and the ensemble spread of each.
    All four blocks carry the printed unit of the channel, which is the
    only way a reader can subtract one column from another and meet the
    third. The flat dump is what a user plots or re-sorts; the statistics
    in the record are derived from exactly these numbers.

    Parameters
    ----------
    outdir : str
        Directory to write into.
    config : SkillConfig
        Validated configuration, for the file stem.
    channel : str
        Property name.
    truth : numpy.ndarray
        Reference values of the whole set, in the printed unit.
    predicted : numpy.ndarray
        Predictions of the whole set, in the printed unit.
    difference : numpy.ndarray
        Reference minus prediction, in the printed unit.
    spread : numpy.ndarray
        Ensemble spread, in the printed unit.

    Returns
    -------
    str
        Path of the written file.
    """
    path = os.path.join(outdir, '%s.%s.pred.txt' % (_stem(config), channel))
    frames = np.asarray(difference).shape[0]
    reference = np.asarray(truth, dtype=np.float64).reshape(frames, -1)
    served = np.asarray(predicted, dtype=np.float64).reshape(frames, -1)
    deviations = np.abs(np.asarray(difference, dtype=np.float64)
                        ).reshape(frames, -1)
    widths = np.abs(np.asarray(spread, dtype=np.float64)
                    ).reshape(frames, -1) if np.asarray(spread).size else (
        np.zeros_like(deviations))
    if not reference.shape[1] == served.shape[1] == deviations.shape[1]:
        raise ValueError(
            "the %s channel cannot be written as a comparison: %d reference "
            "columns against %d prediction columns for the same frames"
            % (channel, reference.shape[1], served.shape[1]))
    block = np.concatenate((reference, served, deviations, widths), axis=1)
    header = ('# %d columns per frame: reference | prediction | '
              '|deviation| | ensemble spread   %s in %s'
              % (reference.shape[1], channel, _CHANNELS[channel][3]))
    np.savetxt(path, block, header=header, comments='', fmt='%.10e')
    return path


def _write_summary(outdir, config, frame_rows):
    """Write the per-frame worst-component table.

    This is the legacy `max_abs_dev.txt`: the largest deviation of each
    frame of each channel, with the header the legacy file never had.

    Parameters
    ----------
    outdir : str
        Directory to write into.
    config : SkillConfig
        Validated configuration, for the file stem.
    frame_rows : dict
        Channel to its per-frame maximum deviation.

    Returns
    -------
    str
        Path of the written file.
    """
    path = os.path.join(outdir, '%s.frames.txt' % _stem(config))
    channels = sorted(frame_rows)
    lines = ['  %5s%s' % ('frame', ''.join('%18s' % ('max |dev| %s' % channel)
                                          for channel in channels))]
    count = max((len(values) for values in frame_rows.values()), default=0)
    for index in range(count):
        row = '  %5d' % (index + 1)
        for channel in channels:
            values = frame_rows[channel]
            row += '%18s' % ('%.8f' % values[index]
                             if index < len(values) else '--')
        lines.append(row)
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    return path


def format_prediction(config, record):
    """Render one test-set evaluation as the log sections of a result.

    The grammar of `trainer.format_training`: a titled section between
    two rules, then the numbers with their units. A channel the profile
    did not serve prints no line at all, because a channel that was not
    computed and a channel computed as zero are different statements.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration, for the counts the report heads with.
    record : dict
        The record returned by run().

    Returns
    -------
    str
        The sections, ending in a newline.
    """
    system = config.system
    errors = record.get('predict_errors') or {}
    units = record.get('predict_metric_units') or {}
    frames = int(record.get('predict_frames') or 0)
    try:
        wall = float(record.get('predict_wall_seconds') or 0.0)
    except (TypeError, ValueError):
        wall = 0.0

    natom = int(system.natom or 0)
    lines = ['  &nn prediction error analysis', _RULE,
             '  Number of atoms:  %-10dNumber of state:  %d'
             % (natom, int(system.nstate or 0)),
             '  Test frames:      %-10dWall clock:       %.1f s' % (frames,
                                                                    wall)]
    raw = record.get('predict_raw_files') or {}
    for channel in sorted(raw):
        lines.append('  Raw comparison:   %-10s%s' % (channel, raw[channel]))
    summary = record.get('predict_summary')
    if summary:
        lines.append('  Frame table:      %s' % summary)
    lines.append(_RULE)

    for statistic, title in (('mae', 'mean absolute error'),
                             ('rmse', 'root mean square error'),
                             ('max', 'maximum absolute error'),
                             ('spread_max', 'ensemble spread')):
        header = '  &nn prediction %s' % title
        lines.extend([_RULE, header, _RULE])
        row = '  %-20s' % ''
        unit_row = '  %-20s' % ''
        value_row = '  %-20s' % 'value'
        for channel in sorted(errors):
            row += '%16s' % channel
            unit_row += '%16s' % units.get(channel, '')
            value_row += '%16.8f' % float(errors[channel][statistic])
        lines.extend([row, unit_row, value_row])
    lines.append(_RULE)
    return '\n'.join(lines) + '\n'
