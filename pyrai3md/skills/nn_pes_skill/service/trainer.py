"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the training side of the nn_pes skill.

The order of operations is the one the container has always required:
write the hyperparameters and the data, start one subprocess per
ensemble member, wait for all of them, then reload and read back the
fit error. Reversed, a reload would resurrect the weights of a previous
run and a finished fit could not be told from a crashed one.

A continuation reverses one step of that order on purpose: the folder is
loaded before anything is written over it, because `create` plus `save`
put freshly initialized weights where the trained ones were and the fit
subprocess then loads what it just replaced. See `targets_with_data`.

A training file is read as Hartree energies and Hartree/Bohr derivatives,
and the labels are put on the coordinate axis the network is fed before
fitting (`service/trainer_data.py:match_labels`). Spin-orbit labels are
values in cm-1 and are not converted at all: they are not derivatives.
Nothing here touches a prediction.

"""

import os
import time

import numpy as np

from ..schema.hyper import build_member_hyper
from ..schema.method import (CHANNEL_COUNT, PROFILE_ENSEMBLES, ROLE_ENERGY,
                             ROLE_NAC, ROLE_SECTIONS, ROLE_SOC)
from ..schema.reference import LABEL_SOC_UNIT
from . import trainer_data as data_io
from .runner import get_train_arm
from .runner.train_runner import failed_runs, run_training
from .train_record import CHANNEL_TABLE, metric_units

# Section rule of the run log this skill's own log imitates.
_RULE = '-' * 79

# Report columns and folder order, spelled as the framework's run log
# spells them (utilities/print_results.py).
_TRAINING_CHANNELS = ('energy', 'gradient', 'nac', 'soc')
_TRAINING_ROLES = (ROLE_ENERGY, ROLE_NAC, ROLE_SOC)


def targets_for(config):
    """List the ensembles the training profile must fit.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration.

    Returns
    -------
    list of tuple
        (model folder, model type) pairs.
    """
    return list(PROFILE_ENSEMBLES[config.calculation.profile])


def channel_count(config, folder, model_type):
    """Return how wide an ensemble's output axis has to be.

    The role decides it, and one model family answers two roles: the energy
    family spans the electronic states in the `energy_grad` folder and the
    declared spin-orbit pairs in the `soc` folder, because both are one
    scalar per row.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration.
    folder : str
        Ensemble folder name.
    model_type : str
        Container model type.

    Returns
    -------
    int
        Number of output channels.

    Raises
    ------
    ValueError
        If the role asks for a count the config does not carry.
    """
    key = CHANNEL_COUNT[model_type][folder]
    return int(getattr(config.system, key))


def ensemble_hyper(config, folder, model_type, natom, indices):
    """Expand one role section into the hyperparameters of an ensemble.

    The shared fields the data and the declaration decide -- atom count,
    output width, the geometric feature tables -- are written first, then
    the caller's role section is laid over them once per ensemble member. A
    keyword written without ';' serves every member; ';' groups are read in
    member order, and `mult_nn` decides how many of them are used.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration.
    folder : str
        Ensemble folder name ('energy_grad', 'nac' or 'soc').
    model_type : str
        Container model type.
    natom : int
        Number of atoms in the training geometries.
    indices : dict
        Geometric feature index lists.

    Returns
    -------
    list of dict
        One hyperparameter dictionary per ensemble member.
    """
    calculation = config.calculation
    section = ROLE_SECTIONS[folder]

    from .nn_pes.nn_pes_src.selection import (
        get_default_hyperparameters_by_modeltype)
    base = get_default_hyperparameters_by_modeltype(model_type)
    base['general']['model_type'] = model_type
    base['model']['atoms'] = int(natom)
    base['model']['states'] = channel_count(config, folder, model_type)
    # The coupling networks are the only ones with a separate channel
    # count, and an energy or gradient network rejects the keyword outright.
    if 'nnac' in base['model']:
        base['model']['nnac'] = int(config.system.nnac)
    base['model']['invd_index'] = indices['invd']
    base['model']['angle_index'] = indices['angle']
    base['model']['dihed_index'] = indices['dihed']
    if folder == ROLE_SOC:
        # A spin-orbit value is a scalar per pair in cm-1 and has no
        # gradient channel, which is what turns the energy network's
        # derivative head off.
        base['model']['energy_only'] = True
        base['plots']['unit_energy'] = LABEL_SOC_UNIT
        base['plots']['unit_gradient'] = ''

    return build_member_hyper(section, base, calculation.mult_nn,
                              dict(getattr(calculation, section) or {}))


def wants_continuation(config, member_hyper):
    """Whether this member's request continues an existing fit.

    Two spellings mean the same thing to the fit subprocess: the category
    it reads is named by `train_mode`, and inside that category
    `initialize_weights: false` tells it to load the folder's weights
    instead of making new ones. The category is what is consulted, never
    the other one: asking for 'retraining' while leaving the training
    category's switch untouched would be read as a restart.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration, for `train_mode`.
    member_hyper : dict
        One member's hyperparameter dictionary.

    Returns
    -------
    bool
        True when the fit must start from the weights on disk.
    """
    category = member_hyper.get(config.calculation.train_mode) or {}
    return not bool(category.get('initialize_weights', True))


def resumable(folder_root, names, members):
    """Whether every target folder already holds a trained ensemble.

    A continuation needs all of one member's three products, for every
    member: the hyper that rebuilds the architecture, the weights to
    continue from, and the scaler the fit reads before it re-fits. A
    folder carrying two of the three is not resumable, and the reason
    names what is missing rather than reporting a bare False.

    Parameters
    ----------
    folder_root : str
        Directory that holds one subfolder per ensemble.
    names : list of str
        Ensemble folder names.
    members : int
        Ensemble size.

    Returns
    -------
    tuple(bool, str)
        (resumable, reason). The reason is empty when every folder is
        complete.
    """
    for name in names:
        path = os.path.join(folder_root, name)
        if not os.path.isdir(path):
            return False, ("%s does not exist yet" % path)
        for index in range(members):
            for suffix in ('hyper_v%i.json' % index,
                           'weights_v%i.weights.h5' % index,
                           'scaler_v%i.json' % index):
                if not os.path.isfile(os.path.join(path, suffix)):
                    return False, "%s holds no %s" % (path, suffix)
    return True, ''


def model_shape_changes(container, folder, member_hyper):
    """Name the architecture keys a continuation cannot apply.

    The container updates a hyperparameter category by category and never
    the 'model' one, because the network was built from it: a continuation
    that asks for a different depth or width would otherwise run with the
    architecture on disk while reporting the numbers the input asked for.

    Parameters
    ----------
    container : object
        Container with the folder loaded.
    folder : str
        Ensemble folder name.
    member_hyper : list of dict
        The requested hyperparameters, one entry per member.

    Returns
    -------
    list of str
        Names of the model keys that differ; empty when the request
        keeps the architecture.
    """
    stored = (getattr(container, '_models_hyper', None) or {}).get(folder) or []
    changed = []
    for index, requested in enumerate(member_hyper):
        if index >= len(stored):
            break
        on_disk = (stored[index] or {}).get('model') or {}
        wanted = requested.get('model') or {}
        for key in sorted(set(on_disk) & set(wanted)):
            if on_disk[key] != wanted[key]:
                changed.append('%s_v%i: %s (%r asked, %r on disk)'
                               % (folder, index, key, wanted[key],
                                  on_disk[key]))
    return changed


def targets_with_data(config, container, folder_root):
    """Write hyperparameters and data, and name the model folders.

    The order of the container calls is what makes a continuation real.
    `create` builds freshly initialized networks and `save` writes their
    weights unconditionally, so a folder that already holds a trained
    surface must be loaded before anything is saved over it: create, save,
    then ask for a continuation would hand the fit subprocess the random
    weights that replaced the trained ones.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration.
    container : object
        Opened container rooted at folder_root.
    folder_root : str
        Directory that holds one subfolder per ensemble.

    Returns
    -------
    list of tuple
        (folder, model type, model folder path) for the launcher.
    """
    dataset = (data_io.from_payload(config.calculation.train_dataset,
                                     source='<inline>')
               if config.calculation.train_dataset
               else data_io.load(config.calculation.train_data))
    arrays = data_io.match_labels(dataset)
    indices = data_io.feature_indices(arrays['xyz'].shape[1])
    data_io.check_channels(config, dataset)

    members = int(config.calculation.mult_nn)
    targets = targets_for(config)
    names = [folder for folder, _type in targets]
    can_resume, reason = resumable(folder_root, names, members)

    prepared = []
    for folder, model_type in targets:
        hyper = ensemble_hyper(config, folder, model_type,
                               natom=arrays['xyz'].shape[1],
                               indices=indices)
        path = os.path.join(folder_root, folder)
        continues = any(wants_continuation(config, member) for member in hyper)

        if continues:
            if not can_resume:
                raise FileNotFoundError(
                    "train_mode '%s' with initialize_weights false asks to "
                    "continue the ensemble in %s, but %s. Point modeldir at "
                    "the trained folder, or drop the switch to fit a new "
                    "one." % (config.calculation.train_mode, folder_root,
                              reason))
            container.load(folder)
            blocked = model_shape_changes(container, folder, hyper)
            if blocked:
                raise ValueError(
                    "model parameters are not updated by a continuation: "
                    "%s. The folder's architecture is the one that can "
                    "accept these weights; train a new modeldir to change "
                    "it." % '; '.join(blocked))
            container.update({folder: hyper})
            container.save(folder)
        elif can_resume:
            mode = config.calculation.train_mode
            raise FileExistsError(
                "%s already holds a trained ensemble, and this request fits "
                "a new one: train_mode '%s' reads initialize_weights from its "
                "'%s' category and that is true, so the fit would start from "
                "fresh weights over a trained surface. Write "
                "initialize_weights: false in the %s section to continue this "
                "folder, or point modeldir somewhere else to start over."
                % (folder_root, mode, mode, ROLE_SECTIONS[folder]))
        else:
            container.create({folder: hyper})
            os.makedirs(path, exist_ok=True)
            container.save(folder)

        data_io.write_to_folder(container, folder, path, folder,
                               model_type, arrays)
        prepared.append((folder, model_type, path))
    return prepared, arrays


def physical_errors(container, folder, model_type, members):
    """Put a member's validation error back into units a reader can use.

    The fit stores its error in the scaled units of its own normalizer,
    where a number means nothing: 0.02 is neither a milli-Hartree nor a
    milli-eV until the spread the data was divided by is put back. That
    spread is read from the scaler the folder was fitted with, not from a
    re-derived guess, and the result is printed in the units the container
    declares for its own plots, eV and eV/Angstrom, because those are what
    a chemist compares against. A spin-orbit folder is the exception: its
    labels are cm-1 and stay cm-1.

    Parameters
    ----------
    container : object
        Container with the ensemble loaded.
    folder : str
        Ensemble folder name, which decides the channel it reports.
    model_type : str
        Container model type.
    members : list
        Per-member stored error in the fit's scaled units.

    Returns
    -------
    dict
        Channel name to one value per member, in metric units.
    """
    # channel, scaler attribute, factor from the fit's unit to the metric
    table = CHANNEL_TABLE[folder][model_type]
    scalers = (getattr(container, '_models_scaler', None) or {}).get(folder)

    reported = {}
    for position, (channel, spread, factor) in enumerate(table):
        values = []
        for index, stored in enumerate(members):
            entries = np.atleast_1d(np.asarray(stored, dtype=np.float64))
            value = (float(entries[position]) if entries.size > position
                     else float(entries))
            scale = 1.0
            if scalers and index < len(scalers):
                width = getattr(scalers[index], spread, None)
                if width is not None:
                    scale = float(np.mean(np.abs(np.asarray(width))))
            values.append(value * scale * factor)
        reported[channel] = values
    return reported


def prediction_spread(config, container, arrays):
    """Largest ensemble spread on the training frames, per channel.

    This is a post-fit diagnostic of the surface, not a fit error: every
    training geometry is served by every member and the largest
    member-to-member standard deviation is kept. Units follow the served
    properties: Hartree, Hartree/Bohr for the derivatives and cm-1 for
    spin-orbit coupling.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration.
    container : object
        Loaded container of the iteration's ensembles.
    arrays : dict
        Prepared training arrays; 'xyz' is (nframe, natom, 3).

    Returns
    -------
    dict
        {'ndata': int, channel: max spread} for the channels the profile
        produces.
    """
    from .calculator import evaluate

    xyz = np.asarray(arrays['xyz'], dtype=np.float64)
    maxima = {}
    for frame in xyz:
        _results, spread = evaluate(config, frame, container)
        for channel in ('energy', 'gradient', 'nac', 'soc'):
            value = spread.get(channel)
            if value is None:
                continue
            entries = np.asarray(value, dtype=np.float64)
            if not entries.size:
                continue
            current = float(np.max(np.abs(entries)))
            maxima[channel] = max(maxima.get(channel, 0.0), current)
    report = {'ndata': int(len(xyz))}
    report.update(maxima)
    return report


def summarize(path, records, errors, wall):
    """Write the human-readable summary of one training run.

    Parameters
    ----------
    path : str
        File path of the summary.
    records : list of dict
        Launcher records (name, index, return code).
    errors : dict
        Per-ensemble metric errors, as physical_errors() reports them.
    wall : float
        Wall-clock seconds the run took.

    Returns
    -------
    str
        The summary text.
    """
    lines = ['nn_pes training summary', '=' * 24,
             'wall clock: %.1f s' % wall, '']
    for record in records:
        lines.append('%s_v%i  exit %s  log %s'
                     % (record['name'], record['index'],
                        record['returncode'], record['log']))
    lines.append('')
    for name, channels in sorted(errors.items()):
        for channel, values in sorted(channels.items()):
            unit = metric_units([channel])[channel]
            for index, value in enumerate(values):
                lines.append('%s_v%i  validation MAE (%s) %.6f %s'
                             % (name, index, channel, float(value), unit))
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    return '\n'.join(lines)



def format_training(config, record):
    """Render one training run as the log sections of a result.

    The same grammar the property steps use in this skill's log: a titled
    section between two rules, then the numbers with their units. The
    table takes the layout the framework's run log uses for the same
    record, so the two renderings of one training run read alike. A
    channel a folder does not produce prints '--' instead of 0.0, because
    a channel that was never trained and a channel trained to a perfect
    zero are different statements.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration, for the counts the report heads with.
    record : dict
        The record returned by run().

    Returns
    -------
    str
        The two sections, ending in a newline.
    """
    system = config.system
    errors = record.get('train_errors') or {}
    units = record.get('train_metric_units') or {}
    trained = set(errors)
    nnac = len(system.nac_pair_map or []) if 'nac' in trained else 0
    nsoc = len(system.soc_pair_map or []) if 'soc' in trained else 0

    header = ['  &nn training', _RULE,
              '  Number of atoms:  %-10dNumber of state:  %d'
              % (int(system.natom or 0), int(system.nstate or 0)),
              '  Number of NAC:    %-10dNumber of SOC:    %d'
              % (nnac, nsoc),
              '  Training arm:     %-10sWall clock:       %.1f s'
              % (str(record.get('train_arm', '')),
                 float(record.get('train_wall_seconds') or 0.0))]
    data = str(record.get('train_data') or '')
    if data:
        header.append('  Training data:    %s' % data)
    modeldir = str(record.get('train_modeldir') or '')
    if modeldir:
        header.append('  Model folder:     %s' % modeldir)
    fitlogs = record.get('train_fitlogs')
    for path in (fitlogs if fitlogs is not None else []):
        header.append('  Fit log:          %s' % path)
    header.append(_RULE)

    titles = '  %-20s' % ''
    unit_line = '  %-20s' % ''
    for channel in _TRAINING_CHANNELS:
        titles += '%14s' % channel
        unit_line += '%14s' % str(units.get(channel, ''))

    rows = [titles, unit_line]
    folders = [folder for folder in _TRAINING_ROLES if errors.get(folder)]
    folders += sorted(set(errors) - set(folders))
    for folder in folders:
        channels = errors.get(folder) or {}
        count = max((len(np.atleast_1d(np.asarray(values)))
                     for values in channels.values() if values), default=0)
        for index in range(count):
            row = '  %-20s' % ('%s_v%d' % (folder, index))
            for channel in _TRAINING_CHANNELS:
                values = channels.get(channel)
                entries = (np.atleast_1d(np.asarray(values, dtype=np.float64))
                           if values is not None else None)
                if entries is None or index >= entries.size:
                    row += '%14s' % '--'
                else:
                    row += '%14.8f' % float(entries[index])
            rows.append(row)

    body = header + ['', '  &nn validation mean absolute error', _RULE]
    body += rows + [_RULE]
    return '\n'.join(body) + '\n' + _format_prediction_std(
        record.get('train_prediction_std'))


def _format_prediction_std(report):
    """Render the post-fit ensemble spread on the training frames."""
    if not isinstance(report, dict) or not report:
        return ''
    channels = ('energy', 'gradient', 'nac', 'soc')
    units = ('Hartree', 'Hartree/Bohr', 'Hartree/Bohr', 'cm-1')
    rows = ['  %-8s' % 'ndata'
            + ''.join('%14s' % channel for channel in channels),
            '  %-8s' % ''
            + ''.join('%14s' % unit for unit in units),
            '  %-8d' % int(report.get('ndata') or 0)
            + ''.join('%14s' % ('--' if report.get(channel) is None
                                else '%.6f' % float(report[channel]))
                      for channel in channels)]
    return '\n' + '\n'.join(['  &nn prediction max std', _RULE]
                            + rows + [_RULE]) + '\n'


def run(config, container, calcdir):
    """Train the profile's ensembles and reload them.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration.
    container : object
        Container rooted at the model directory.
    calcdir : str
        Calculation directory holding the summary.

    Returns
    -------
    dict
        Result record: per-ensemble fit errors, wall time, arm used, the
        training source, the model folder and every member's fit log.
        The scratch summary path travels for callers that read the
        record directly; the run log reports the fit logs instead.
    """
    folder_root = os.path.abspath(os.path.expanduser(config.calculation.modeldir))
    started = time.perf_counter()

    targets, _arrays = prepared_or_raise(config, container, folder_root)

    arm = get_train_arm(config)
    records = run_training(config, targets, arm=arm)
    failures = failed_runs(records)
    if failures:
        detail = ', '.join('%s_v%i exit %s (see %s)'
                           % (item['name'], item['index'], item['returncode'],
                              item['log']) for item in failures)
        raise RuntimeError(
            "Training did not finish for: %s. The model folder holds "
            "partial output and must not be used for serving." % detail)

    names = [(name, model_type)
             for name, model_type, _folder in targets]
    container.load([name for name, _type in names])
    stored = {name: container._read_fit_error(name) for name, _type in names}
    errors = {folder: physical_errors(container, folder, model_type,
                                      stored[folder])
              for folder, model_type in names}
    units = metric_units(sorted({channel
                                 for channels in errors.values()
                                 for channel in channels}))
    wall = time.perf_counter() - started
    prediction_std = (prediction_spread(config, container, _arrays)
                      if int(config.calculation.predict_std) else None)

    summary_path = os.path.join(
        calcdir, '%s_train.summary.txt' % (config.calculation.project or 'NNPES'))
    summarize(summary_path, records, errors, wall)
    print("Info: trained %s with arm '%s' in %.1f s" % (
        [name for name, _type in names], arm, wall))

    source = str(config.calculation.train_data or '')
    if not source and config.calculation.train_dataset:
        source = '<inline>'
    return {
        'train_errors': errors,
        'train_metric_units': units,
        'train_wall_seconds': wall,
        'train_arm': arm,
        'train_data': source,
        'train_modeldir': folder_root,
        'train_fitlogs': [record['log'] for record in records],
        'train_prediction_std': prediction_std,
        'train_summary': summary_path,
    }


def prepared_or_raise(config, container, folder_root):
    """Prepare ensembles, with the training source error made explicit.

    The training source is checked here rather than around the whole
    preparation, because a FileNotFoundError raised later belongs to the
    model folder (a continuation that has no weights to continue from) and
    would otherwise be reported as a problem with the data file. An inline
    dataset needs no existence check; the skill schema already guarantees
    that exactly one of the two sources is set.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration.
    container : object
        Opened container.
    folder_root : str
        Directory that holds one subfolder per ensemble.

    Returns
    -------
    tuple
        What `targets_with_data` returns.

    Raises
    ------
    FileNotFoundError
        If `train_data` names no readable file in file mode.
    """
    if config.calculation.train_dataset:
        return targets_with_data(config, container, folder_root)
    path = config.calculation.train_data
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            "train_data could not be read: %s" % (path or '(empty)'))
    return targets_with_data(config, container, folder_root)

