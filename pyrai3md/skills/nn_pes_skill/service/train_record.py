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

This module reads a trained model folder straight off the disk.

The container loads a folder through TensorFlow, which is the right
thing to do when a surface has to serve a step. It is the wrong thing to
do when all a reader wants is the numbers a finished fit wrote: a grid
scan re-reads hundreds of folders to rank them, a qualification run must
be able to say whether a folder is complete on a machine that has no
TensorFlow at all, and neither job should import Keras to answer it.

So the products the training writes last are read here as what they are:
the per-member fit error in a .npy, the architecture in a hyper .json,
and the spread the fit divided the data by in a scaler .json. The table
that turns a stored error into a physical one is shared with
`service/trainer.py`, because a folder read back and a folder reported
at the end of its own fit must say the same thing in the same units.

"""

import json
import os

import numpy as np

from ..schema.method import ROLE_ENERGY, ROLE_NAC, ROLE_SOC
from ..schema.reference import (HARTREE_TO_EV, METRIC_ENERGY_UNIT,
                                METRIC_GRADIENT_UNIT, METRIC_SOC_UNIT)

# Which channels one model type reports, in the order the fit stores them,
# with the scaler attribute that carries the spread each was divided by
# and the factor from the fit's unit to the printed one.
CHANNEL_TABLE = {
    ROLE_ENERGY: {
        'mlp_e': (('energy', 'energy_std', HARTREE_TO_EV),),
        'mlp_eg': (('energy', 'energy_std', HARTREE_TO_EV),
                   ('gradient', 'gradient_std', HARTREE_TO_EV)),
        'mlp_g2': (('gradient', 'gradient_std', HARTREE_TO_EV),),
    },
    ROLE_NAC: {
        'mlp_nac': (('nac', 'nac_std', HARTREE_TO_EV),),
        'mlp_nac2': (('nac', 'nac_std', HARTREE_TO_EV),),
    },
    ROLE_SOC: {
        'mlp_e': (('soc', 'energy_std', 1.0),),
    },
}

# The unit each printed channel is measured in.
CHANNEL_UNITS = {'energy': METRIC_ENERGY_UNIT,
                 'gradient': METRIC_GRADIENT_UNIT,
                 'nac': METRIC_GRADIENT_UNIT,
                 'soc': METRIC_SOC_UNIT}

# One file per ensemble member, named as the container names them. The
# fit error is the last product a member writes, which is why its absence
# is what marks a folder unfinished.
_MEMBER_PRODUCTS = ('hyper_v%d.json', 'weights_v%d.weights.h5',
                    'scaler_v%d.json', 'fiterr_valid_v%d.npy')


def metric_units(channels):
    """Return the unit each reported channel is printed in.

    Parameters
    ----------
    channels : iterable of str
        Channel names.

    Returns
    -------
    dict
        Channel name to its unit.
    """
    return {channel: CHANNEL_UNITS[channel] for channel in channels}


def missing_products(path, members):
    """List the member products a folder does not carry.

    Parameters
    ----------
    path : str
        Ensemble folder path.
    members : int
        Ensemble size.

    Returns
    -------
    list of str
        Names of the absent files; empty when the folder is complete.
    """
    missing = []
    for index in range(members):
        for pattern in _MEMBER_PRODUCTS:
            name = pattern % index
            if not os.path.isfile(os.path.join(path, name)):
                missing.append(name)
    return missing


def _model_type(path, member):
    """Read the container model type one member was fitted as.

    Parameters
    ----------
    path : str
        Ensemble folder path.
    member : int
        Ensemble member index.

    Returns
    -------
    str or None
        Model type, or None when the hyper file cannot be read.
    """
    try:
        with open(os.path.join(path, 'hyper_v%d.json' % member)) as handle:
            hyper = json.load(handle)
    except (OSError, ValueError):
        return None
    return (hyper.get('general') or {}).get('model_type')


def _spread(path, member, attribute):
    """Read the spread one member's normalizer divided its data by.

    Parameters
    ----------
    path : str
        Ensemble folder path.
    member : int
        Ensemble member index.
    attribute : str
        Scaler attribute name, which is its key in the JSON.

    Returns
    -------
    float or None
        The mean absolute spread, or None when the file or the key is
        absent; a stored error without a spread stays what it is rather
        than becoming a zero.
    """
    try:
        with open(os.path.join(path, 'scaler_v%d.json' % member)) as handle:
            scaler = json.load(handle)
    except (OSError, ValueError):
        return None
    width = scaler.get(attribute)
    if width is None:
        return None
    return float(np.mean(np.abs(np.asarray(width, dtype=np.float64))))


def physical_member_errors(path, member, stored):
    """Put one member's stored fit error into printed units.

    Parameters
    ----------
    path : str
        Ensemble folder path.
    member : int
        Ensemble member index.
    stored : array-like
        The per-channel errors in the fit's scaled units, in
        CHANNEL_TABLE order.

    Returns
    -------
    dict
        Channel name to its physical error for this member.
    """
    model_type = _model_type(path, member)
    table = CHANNEL_TABLE.get(path_role(path), {}).get(model_type)
    entries = np.atleast_1d(np.asarray(stored, dtype=np.float64))
    if not table:
        return {}
    reported = {}
    for position, (channel, spread, factor) in enumerate(table):
        value = (float(entries[position]) if entries.size > position
                 else float(entries))
        width = _spread(path, member, spread)
        reported[channel] = value * (width if width is not None else 1.0) * factor
    return reported


def path_role(path):
    """Name the role an ensemble folder carries.

    The folder names are the ones the container writes ('energy_grad',
    'nac', 'soc'), and the role is what decides which channels a member
    of that folder reports.

    Parameters
    ----------
    path : str
        Ensemble folder path or folder name.

    Returns
    -------
    str or None
        The folder name when this skill writes such a folder, None
        otherwise.
    """
    name = os.path.basename(os.path.normpath(path))
    if name in (ROLE_ENERGY, ROLE_NAC, ROLE_SOC):
        return name
    return None


def read_folder(path, members):
    """Read one ensemble folder without TensorFlow.

    Parameters
    ----------
    path : str
        Ensemble folder path.
    members : int
        Ensemble size.

    Returns
    -------
    dict
        'complete' (bool), 'missing' (list of file names), and 'errors'
        (channel to one physical value per member). An incomplete folder
        reports the errors of the members that do carry a fit file, which
        is how a restart tells a finished point from a dead one.
    """
    missing = missing_products(path, members)
    errors = {}
    for index in range(members):
        fit_path = os.path.join(path, 'fiterr_valid_v%d.npy' % index)
        if not os.path.isfile(fit_path):
            continue
        try:
            stored = np.load(fit_path)
        except (OSError, ValueError):
            continue
        for channel, value in physical_member_errors(
                path, index, stored).items():
            errors.setdefault(channel, []).append(value)
    return {'complete': not missing, 'missing': missing, 'errors': errors}


def read_model_dir(modeldir, members):
    """Read every ensemble folder of one trained model directory.

    Parameters
    ----------
    modeldir : str
        Directory that holds one subfolder per ensemble.
    members : int
        Ensemble size.

    Returns
    -------
    dict or None
        'errors' (folder to channel to per-member values), 'units',
        'complete' and 'incomplete' (the folders that are missing a
        product), or None when the directory does not exist.
    """
    root = os.path.abspath(os.path.expanduser(modeldir))
    if not os.path.isdir(root):
        return None
    folders = sorted(name for name in os.listdir(root)
                     if os.path.isdir(os.path.join(root, name))
                     and path_role(name) is not None)
    if not folders:
        return {'errors': {}, 'units': {}, 'complete': False,
                'incomplete': ['(no ensemble folder)']}
    errors = {}
    incomplete = []
    for folder in folders:
        record = read_folder(os.path.join(root, folder), members)
        if not record['errors']:
            # A folder that reports no channel is not a result. The names
            # it lacks are what a reader is owed: an empty error dict would
            # print blank yet rank as a point.
            incomplete.append('%s: %s' % (
                folder, ', '.join(record['missing']) or 'no fit error'))
            continue
        errors[folder] = record['errors']
        if not record['complete']:
            incomplete.append('%s: %s' % (folder, ', '.join(record['missing'])))
    channels = {channel for channels in errors.values() for channel in channels}
    return {'errors': errors, 'units': metric_units(sorted(channels)),
            'complete': not incomplete, 'incomplete': incomplete}
