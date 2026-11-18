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

This module provides the training data interface of the nn_pes skill:
it reads the data either from a file or from an inline mapping, derives
the geometric feature indices the networks need, and hands the arrays
to the container in the layout its training subprocesses expect.

The accepted data carries the keys the original project used: 'xyz'
(geometries of [symbol, x, y, z] or [x, y, z] rows), 'energy'
(nframe, nstate), 'grad' (nframe, nstate, natom, 3), 'nac'
(nframe, nchannel, natom, 3), 'soc' (nframe, nsoc), and optionally
'transition_dipole' (nframe, npair, 3), 'transition_dipole_norm'
(nframe, npair), 'osc_strength' (nframe, npair), 'natom', 'nstate',
'nnac', 'nsoc', 'npair'. Missing counts are derived from the arrays.

The transition-dipole channels are carried through unchanged for future
training profiles; this version fits energy, gradient, NAC and SOC only.

The labels are Hartree energies and Hartree/Bohr gradients and couplings,
which is what the framework's engine backends report; a file that carries
a 'unit' block is checked against it rather than trusted.

"""

import json

import numpy as np

from ..schema.method import (CHANNEL_COUNT_MODELS, PROFILE_ENSEMBLES,
                             ROLE_SOC)
from ..schema.reference import (BOHR_TO_ANGSTROM, LABEL_ENERGY_UNIT,
                                LABEL_GRADIENT_UNIT, LABEL_SOC_UNIT)


def _optional(raw, key):
    """Read one optional label array, or None when the key is empty."""
    value = raw.get(key)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    return array if array.size else None


def _as_2d(array, name, source):
    """A scalar-per-channel array; a flat row means one channel."""
    if array is None:
        return None
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(
            "%s: %s must be (nframe, nchannel), got %r"
            % (source, name, array.shape))
    return array


def _declared(raw, key, source):
    """Read one declared count, or None when the key is absent."""
    value = raw.get(key)
    if value is None or value == '':
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError("%s declares %s=%r, which is not an integer"
                         % (source, key, value))


def _read_xyz(raw, source):
    """Read geometries and (when present) atom symbols.

    Rows are either ``[x, y, z]`` or ``[symbol, x, y, z]``. A separate
    ``symbols`` key is accepted for files that carry numeric rows.
    """
    if 'xyz' not in raw:
        raise ValueError("Training file %s carries no 'xyz' geometries"
                         % source)

    frames = []
    symbols = []
    labelled = None
    for index, frame in enumerate(raw['xyz']):
        coordinates = []
        labels = []
        for row_index, row in enumerate(frame):
            if len(row) >= 4:
                has_label = True
                label = str(row[0])
                tail = row[-3:]
            elif len(row) == 3:
                has_label = False
                label = None
                tail = row
            else:
                raise ValueError(
                    "%s frame %d row %d carries an atom row of %d entries; "
                    "expected [x, y, z] or [symbol, x, y, z]"
                    % (source, index, row_index, len(row)))
            if labelled is None:
                labelled = has_label
            elif labelled != has_label:
                raise ValueError(
                    "%s frame %d row %d mixes labelled and unlabelled atom "
                    "rows" % (source, index, row_index))
            try:
                coordinates.append([float(value) for value in tail])
            except (TypeError, ValueError):
                raise ValueError(
                    "%s frame %d carries a non-numeric coordinate: %r"
                    % (source, index, row))
            if has_label:
                labels.append(label)
        frames.append(coordinates)
        symbols.append(labels)

    xyz = np.asarray(frames, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError("%s: xyz must be (nframe, natom, 3)" % source)

    if labelled:
        return xyz, np.asarray(symbols, dtype=object)
    declared = raw.get('symbols')
    if declared is not None:
        declared = np.asarray(declared, dtype=object)
        if declared.shape != xyz.shape[:2]:
            raise ValueError(
                "%s: symbols must be (nframe, natom) = %r, got %r"
                % (source, xyz.shape[:2], declared.shape))
        return xyz, declared
    return xyz, None


def from_payload(raw, source='<inline>'):
    """Read a training dataset from a JSON-style mapping.

    This is the one reader behind both input modes: a file is parsed to
    a mapping and handed here, and an inline dataset arrives here
    directly. The skill checks the same shapes and counts either way.

    Parameters
    ----------
    raw : dict
        Parsed JSON or the inline ``train_dataset`` mapping.
    source : str
        Name used in error messages; the mapping's own 'source' wins.

    Returns
    -------
    dict
        Label arrays plus the derived 'natom', 'nframe', 'nstate',
        'nnac', 'nsoc' and 'npair' counts and the 'source' label.
    """
    if not isinstance(raw, dict):
        raise ValueError("%s does not carry a training-data mapping"
                         % source)
    source = str(raw.get('source') or source)
    xyz, symbols = _read_xyz(raw, source)
    nframe, natom = xyz.shape[0], xyz.shape[1]

    declared_atoms = _declared(raw, 'natom', source)
    if declared_atoms is not None and declared_atoms != natom:
        raise ValueError(
            "Training file declares natom=%d but carries geometries of %d "
            "atoms" % (declared_atoms, natom))

    energy = _as_2d(_optional(raw, 'energy'), 'energy', source)
    soc = _as_2d(_optional(raw, 'soc'), 'soc', source)
    grad = _optional(raw, 'grad')
    nac = _optional(raw, 'nac')
    # A single-state gradient and a single-pair coupling may omit their
    # channel axis; the legacy files do, and the meaning is unambiguous
    # because the last axis is the Cartesian one.
    if grad is not None and grad.ndim == 3 and grad.shape[2] == 3:
        grad = grad[:, None, :, :]
    if nac is not None and nac.ndim == 3 and nac.shape[2] == 3:
        nac = nac[:, None, :, :]
    transition_dipole = _optional(raw, 'transition_dipole')
    transition_dipole_norm = _as_2d(
        _optional(raw, 'transition_dipole_norm'), 'transition_dipole_norm',
        source)
    osc_strength = _as_2d(_optional(raw, 'osc_strength'), 'osc_strength',
                          source)

    for name, array in (('energy', energy), ('soc', soc),
                        ('transition_dipole_norm', transition_dipole_norm),
                        ('osc_strength', osc_strength)):
        if array is not None and array.shape[0] != nframe:
            raise ValueError(
                "%s: %s carries %d frames but xyz carries %d"
                % (source, name, array.shape[0], nframe))

    if grad is not None:
        if grad.ndim != 4 or grad.shape[3] != 3:
            raise ValueError(
                "grad must be (nframe, nstate, natom, 3), got %r"
                % (grad.shape,))
        if grad.shape[0] != nframe or grad.shape[2] != natom:
            raise ValueError(
                "%s: grad frame/atom axes do not match xyz (%r vs %d, %d)"
                % (source, grad.shape, nframe, natom))
    if nac is not None:
        if nac.ndim != 4 or nac.shape[3] != 3:
            raise ValueError(
                "nac must be (nframe, nnac, natom, 3), got %r"
                % (nac.shape,))
        if nac.shape[0] != nframe or nac.shape[2] != natom:
            raise ValueError(
                "%s: nac frame/atom axes do not match xyz (%r vs %d, %d)"
                % (source, nac.shape, nframe, natom))
    if transition_dipole is not None:
        if transition_dipole.ndim != 3 or transition_dipole.shape[2] != 3:
            raise ValueError(
                "transition_dipole must be (nframe, npair, 3), got %r"
                % (transition_dipole.shape,))
        if transition_dipole.shape[0] != nframe:
            raise ValueError(
                "%s: transition_dipole carries %d frames but xyz carries %d"
                % (source, transition_dipole.shape[0], nframe))

    derived_state = None
    if energy is not None:
        derived_state = energy.shape[1]
    elif grad is not None:
        derived_state = grad.shape[1]
    nstate = _declared(raw, 'nstate', source)
    if nstate is None:
        nstate = derived_state or 0
    elif derived_state is not None and nstate != derived_state:
        raise ValueError(
            "%s declares nstate=%d but its arrays carry %d states"
            % (source, nstate, derived_state))
    if nstate < 1:
        raise ValueError("Training file carries no electronic states")
    if energy is not None and energy.shape[1] != nstate:
        raise ValueError("energy array has %d states, file declares %d"
                         % (energy.shape[1], nstate))
    if grad is not None and grad.shape[1] != nstate:
        raise ValueError("grad array has %d states, file declares %d"
                         % (grad.shape[1], nstate))

    derived_nac = None if nac is None else nac.shape[1]
    nnac = _declared(raw, 'nnac', source)
    if nnac is None:
        nnac = derived_nac or 0
    elif derived_nac is not None and nnac != derived_nac:
        raise ValueError(
            "%s declares nnac=%d but its nac array carries %d channels"
            % (source, nnac, derived_nac))

    derived_soc = None if soc is None else soc.shape[1]
    nsoc = _declared(raw, 'nsoc', source)
    if nsoc is None:
        nsoc = derived_soc or 0
    elif derived_soc is not None and nsoc != derived_soc:
        raise ValueError(
            "%s declares nsoc=%d but its soc array carries %d channels"
            % (source, nsoc, derived_soc))

    pair_counts = [array.shape[1] for array in
                   (transition_dipole, transition_dipole_norm, osc_strength)
                   if array is not None]
    derived_pair = pair_counts[0] if pair_counts else None
    if any(count != derived_pair for count in pair_counts):
        raise ValueError(
            "%s: transition-dipole arrays disagree about npair: %r"
            % (source, pair_counts))
    npair = _declared(raw, 'npair', source)
    if npair is None:
        npair = derived_pair or 0
    elif derived_pair is not None and npair != derived_pair:
        raise ValueError(
            "%s declares npair=%d but its arrays carry %d channels"
            % (source, npair, derived_pair))
    if transition_dipole is not None and transition_dipole_norm is None:
        transition_dipole_norm = np.linalg.norm(
            transition_dipole, axis=-1)

    check_units(source, raw.get('unit'))

    return {'xyz': xyz,
            'symbols': symbols,
            'energy': energy,
            'grad': grad,
            'nac': nac,
            'soc': soc,
            'transition_dipole': transition_dipole,
            'transition_dipole_norm': transition_dipole_norm,
            'osc_strength': osc_strength,
            'natom': natom, 'nframe': nframe, 'nstate': nstate,
            'nnac': nnac, 'nsoc': nsoc, 'npair': npair,
            'source': source}


def load(path):
    """Read one training data file.

    Parameters
    ----------
    path : str
        Path of the JSON data file.

    Returns
    -------
    dict
        Label arrays plus the derived counts (see :func:`from_payload`).

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the arrays disagree about the atom count or state count.
    """
    with open(path, 'r') as handle:
        raw = json.load(handle)
    return from_payload(raw, source=path)


def check_units(source, declared):
    """Refuse a training file that says it carries another unit scheme.

    The convention is not configurable, so a self-describing file that
    disagrees is the one place a mismatch can be caught before it becomes a
    network whose gradient is the derivative of its own energy by the wrong
    factor. That failure is invisible in a single prediction and shows up as
    energy drift hundreds of steps later.

    Parameters
    ----------
    source : str
        File name or '<inline>', for the message.
    declared : dict or None
        The input's 'unit' block, if it carries one.

    Raises
    ------
    ValueError
        If a declared energy, gradient or coupling unit is not the one the
        skill reads.
    """
    if not declared:
        return
    # Each channel and the spellings of its unit that mean the same thing.
    # Anything else is a different scheme, and a different scheme is the
    # answer to a question this skill does not ask.
    accepted = {
        'energy': ('hartree', 'ha', 'eh'),
        'gradient': (LABEL_GRADIENT_UNIT, 'ha/bohr', 'eh/bohr'),
        'nac': (LABEL_GRADIENT_UNIT, 'ha/bohr', 'eh/bohr'),
        'soc': (LABEL_SOC_UNIT, 'cm_1', '1/cm'),
    }
    for channel, unit in accepted.items():
        value = declared.get(channel)
        if value is None:
            continue
        spelling = str(value).strip().lower()
        if spelling not in unit:
            raise ValueError(
                "%s declares its %s labels as %r, but this skill reads them "
                "as %s; convert the file rather than letting it train a "
                "surface whose gradient is inconsistent with its energy"
                % (source, channel, value, ' or '.join(sorted(set(unit)))))


def _as_float(value):
    """One label array as float64, with an empty array for a missing one."""
    if value is None:
        return np.asarray([], dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def match_labels(dataset):
    """Express the training labels on the coordinate axis the network sees.

    A file carries Hartree energies and Hartree/Bohr gradients, which is
    the framework's convention and the one every engine backend reports. The
    network is fed Angstrom, because that is what a dynamics step hands the
    skill, so the gradient of the surface the network learns has to be the
    derivative per Angstrom. One factor of 0.529177 does that and nothing
    else: energy and gradient then describe a single surface, which is the
    only rule a training file has to satisfy.

    There is deliberately no unit keyword here. A switch would let a caller
    declare a scheme that does not match their own file, and the failure is
    silent: the network learns a surface whose reported gradient is the
    derivative of its energy by the wrong factor, and only an energy
    conservation run notices.

    The transition-dipole channels are carried through unchanged: no
    current training profile consumes them, and applying a length factor
    to a quantity with its own unit convention would invent a reading.

    Parameters
    ----------
    dataset : dict
        Loaded training data with 'xyz', 'energy', 'grad' and 'nac'.

    Returns
    -------
    dict
        Arrays as float32, in Hartree energies and Hartree/Angstrom
        derivatives of Angstrom coordinates.
    """
    xyz = np.asarray(dataset['xyz'], dtype=np.float64)
    energy = _as_float(dataset.get('energy'))
    grad = _as_float(dataset.get('grad'))
    nac = _as_float(dataset.get('nac'))
    # cm-1 values are read as they stand: no factor of the length unit
    # applies to a quantity that is not a derivative.
    soc = _as_float(dataset.get('soc'))

    # dE/dR_angstrom = dE/dR_bohr times dR_bohr/dR_angstrom
    grad = grad / BOHR_TO_ANGSTROM
    nac = nac / BOHR_TO_ANGSTROM

    return {'xyz': xyz.astype(np.float32),
            'energy': energy.astype(np.float32),
            'grad': grad.astype(np.float32),
            'nac': nac.astype(np.float32),
            'soc': soc.astype(np.float32),
            'transition_dipole': _as_float(
                dataset.get('transition_dipole')).astype(np.float32),
            'transition_dipole_norm': _as_float(
                dataset.get('transition_dipole_norm')).astype(np.float32),
            'osc_strength': _as_float(
                dataset.get('osc_strength')).astype(np.float32)}


def _declared_channels(config, folder):
    """Return the channel count the config declares for one role."""
    return {'energy_grad': config.system.nstate,
            'nac': config.system.nnac,
            'soc': config.system.nsoc}[folder]


def check_channels(config, dataset):
    """Check the declared coupling channels against the model and the file.

    The coupling channels are the state pairs the caller declared; the
    framework passes them in its coupling section, and their count sizes
    the network. Two architectural facts are checked here rather than
    assumed:

    - the label file must carry exactly as many channels as were declared,
    - `mlp_nac` builds one output channel per electronic state, because
      its virtual-potential scheme derives the width from `states`, so a
      declared pair set of another size cannot be trained by it.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration.
    dataset : dict
        Loaded training data.

    Raises
    ------
    ValueError
        If the declared count differs from the file, or a pair set that
        `mlp_nac` cannot represent was declared.
    """
    calculation = config.calculation
    source = dataset.get('source') or calculation.train_data or '<inline>'
    for folder, model_type in PROFILE_ENSEMBLES[calculation.profile]:
        declared = int(_declared_channels(config, folder))
        if folder == ROLE_SOC:
            if dataset['nsoc'] != declared:
                raise ValueError(
                    "The spin-orbit ensemble is sized by the %d pair(s) the "
                    "config declares, but %s carries %d spin-orbit channel(s). "
                    "Declare the pairs the SOC values were calculated for."
                    % (declared, source, dataset['nsoc']))
            continue
        if model_type not in CHANNEL_COUNT_MODELS:
            continue
        if dataset['nnac'] != declared:
            raise ValueError(
                "The coupling model %s is sized by the %d state pair(s) the "
                "config declares, but %s carries %d coupling channels. "
                "Declare the pairs the data was calculated for."
                % (model_type, declared, source, dataset['nnac']))
        if model_type == 'mlp_nac' and declared != int(config.system.nstate):
            raise ValueError(
                "mlp_nac represents one coupling channel per electronic "
                "state, so it needs %d pairs declared but carries %d; use "
                "method 'mlp_nac2', which trains an arbitrary number of "
                "coupling channels." % (int(config.system.nstate), declared))


def feature_indices(natom):
    """Build the geometric feature index lists of a molecule.

    Parameters
    ----------
    natom : int
        Number of atoms.

    Returns
    -------
    dict
        'invd', 'angle' and 'dihed' index lists.
    """
    return {
        'invd': [[i, j] for i in range(natom) for j in range(i)],
        'angle': [[i, i + 1, i + 2] for i in range(natom - 2)],
        'dihed': [[i, i + 1, i + 2, i + 3] for i in range(natom - 3)],
    }


def targets_for_role(role, model_type, arrays):
    """Select the label arrays an ensemble is fitted on.

    The role, not the network family, decides what the channels mean: the
    energy family serves the energy ensemble and, unchanged in shape, the
    spin-orbit one, because both are scalars per state or per pair.

    Parameters
    ----------
    role : str
        Ensemble role ('energy_grad', 'nac' or 'soc').
    model_type : str
        Container model type.
    arrays : dict
        Normalized arrays from the loader.

    Returns
    -------
    object
        The target container the training subprocess expects.

    Raises
    ------
    ValueError
        If the role asks for labels the file does not carry.
    """
    if role == ROLE_SOC:
        if arrays['soc'].size == 0:
            raise ValueError(
                "A spin-orbit ensemble is being trained but the training "
                "file carries no 'soc' labels")
        return arrays['soc']
    if model_type == 'mlp_e':
        return arrays['energy']
    if model_type == 'mlp_g2':
        return arrays['grad']
    if model_type in ('mlp_nac', 'mlp_nac2'):
        return arrays['nac']
    return [arrays['energy'], arrays['grad']]


def write_to_folder(container, name, folder, role, model_type, arrays):
    """Write the pickled data the training subprocess reads.

    Parameters
    ----------
    container : object
        Opened container (its data writer is reused unchanged).
    name : str
        Ensemble name.
    folder : str
        Model folder of this ensemble.
    role : str
        Ensemble role this folder serves.
    model_type : str
        Container model type.
    arrays : dict
        Normalized arrays.
    """
    from .nn_pes.datasets.general import model_save_data_to_folder
    x = arrays['xyz']
    y = targets_for_role(role, model_type, arrays)
    model_save_data_to_folder(x, y, name, folder, False)