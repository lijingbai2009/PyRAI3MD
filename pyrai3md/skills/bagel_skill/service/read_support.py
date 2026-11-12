"""
======================================================================
Read-entry mechanics for the bagel skill.

A skill read entry (`skill/service/read.py::read_calculation`) rebuilds
its validated configuration, parses one finished calculation directory,
and returns one frame in the same contract for every skill. This module
holds the parts that are identical across skills: locating the output
file stem, reading the input geometry, and packing a frame with the
training-data channels. It holds no engine knowledge. This skill-local copy exists so the read entry can be distributed with its skill alone: nothing here imports another pyrai3md module.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import re

import numpy as np

# The unit block of every training channel. A skill that reports in these
# units returns it unchanged; a skill that computes in native units marks
# the result with `_native_units` and the framework converts it.
TRAINING_UNITS = {
    'energy': 'hartree',
    'gradient': 'hartree/bohr',
    'nac': 'hartree/bohr',
    'soc': 'cm-1',
}

# Environment fields that carry a skill's output file stem. The
# authoritative field is `calculation.project`; the per-skill aliases are
# kept in sync because every parser reads its own spelling.
_PROJECT_FIELDS = ('bagel_project',)


# Fields whose validation checks the engine installation or creates a
# scratch directory. A read run touches neither, so they are cleared
# before the skill's own schema pipeline runs.
_ENGINE_FIELDS = ('bagel_path', 'scratch_path')


def prepare_config(config):
    """A skill config with the engine-installation fields cleared.

    A read run must work on a machine that holds the finished artifacts
    but not the engine: the skill's schema validation would otherwise
    refuse a missing installation path and create its scratch directory.
    """
    raw = dict(config)
    for name in _ENGINE_FIELDS:
        raw[name] = ''
    return raw


def resolve_project(calcdir, project):
    """The output file stem a read entry should use.

    An explicit project wins; otherwise the calculation directory's
    basename is the legacy convention. The framework read flow discovers
    a multi-trajectory directory's title before calling a skill entry, so
    this fallback only answers a direct call.
    """
    if project:
        return str(project)
    return os.path.basename(os.path.normpath(calcdir))


def set_project(config, project):
    """Point a validated skill config at one output file stem.

    Parameters
    ----------
    config : SkillConfig
        Validated skill configuration.
    project : str
        Output file stem (empty leaves the config untouched).

    Returns
    -------
    SkillConfig
        The same config, updated in place.
    """
    if not project:
        return config
    project = str(project)
    calculation = getattr(config, 'calculation', None)
    if calculation is not None and hasattr(calculation, 'project'):
        calculation.project = project
    environment = getattr(config, 'environment', None)
    if environment is not None:
        for name in _PROJECT_FIELDS:
            if hasattr(environment, name):
                setattr(environment, name, project)
    return config


def read_geometry(calcdir, project, candidates=()):
    """Read the input geometry of one calculation directory.

    Parameters
    ----------
    calcdir : str
        Calculation directory.
    project : str
        Output file stem.
    candidates : sequence of str, optional
        File names to try, relative to `calcdir`. Defaults to
        `<project>.xyz`.

    Returns
    -------
    tuple
        `(symbols, coordinates)`, coordinates in Angstrom.

    Raises
    ------
    FileNotFoundError
        When none of the candidate files exists.
    """
    names = list(candidates) or ['%s.xyz' % project]
    tried = []
    for name in names:
        path = name if os.path.isabs(name) else os.path.join(calcdir, name)
        tried.append(path)
        if os.path.isfile(path):
            symbols, coordinates, _lattice = read_xyz_frame(path)
            return symbols, coordinates
    raise FileNotFoundError(
        "no geometry file in %s; tried: %s"
        % (calcdir, ', '.join(os.path.basename(path) for path in tried)))


def last_coordinates(value, natom):
    """The last frame of a parser geometry result, or None.

    A parser reports either one frame `(natom, 3)` or the log's frames
    `(nframe, natom, 3)`; a value of the wrong shape is not geometry and
    is ignored.
    """
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return None
    if array.ndim == 3:
        array = array[-1]
    if array.ndim != 2 or array.shape != (natom, 3):
        return None
    return array


def frame(symbols, coordinates, results, unit=None):
    """Pack one parsed calculation into the read frame contract.

    Parameters
    ----------
    symbols : sequence of str
        Element labels, in the calculation's atom order.
    coordinates : array
        `(natom, 3)` coordinates in Angstrom.
    results : dict
        Parser result mapping.
    unit : dict, optional
        Unit block; the training units when omitted.

    Returns
    -------
    dict
        Frame with symbols, coordinates, the four label channels (None
        when the parser produced none) and the unit block.
    """
    labels = [str(symbol) for symbol in symbols]
    natom = len(labels)
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.shape != (natom, 3):
        raise ValueError(
            "read geometry has shape %r but the parser reports %d atoms"
            % (coords.shape, natom))
    out = {
        'symbols': labels,
        'coordinates': coords.tolist(),
        'energy': _series(results.get('energy')),
        'gradient': _atom_channel(results.get('gradient'), natom,
                                  'gradient'),
        'nac': _atom_channel(results.get('nac'), natom, 'nac'),
        'soc': _series(results.get('soc')),
        'unit': dict(TRAINING_UNITS if unit is None else unit),
    }
    if '_native_units' in results:
        out['_native_units'] = results['_native_units']
    return out


def _series(value):
    """A one-dimensional label array, or None when it carries no value."""
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return array.tolist() if array.size else None


def _atom_channel(value, natom, name):
    """An `(nchannel, natom, 3)` label array, or None when empty."""
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return None
    if array.ndim == 2 and array.shape == (natom, 3):
        array = array[np.newaxis, :, :]
    elif array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[1:] != (natom, 3):
        raise ValueError(
            "%s has shape %r; expected (nchannel, %d, 3)"
            % (name, array.shape, natom))
    return array.tolist() if array.shape[0] else None

# --- local copy of the one XYZ frame reader -------------------------------
# `utilities/io.py::read_xyz_frame` is the framework's shared reader. The
# copy below keeps this skill's read entry distributable on its own: it
# depends only on os, re and numpy, and reads the same extended-XYZ
# Lattice="..." comment line the framework writes.
_LATTICE_KEY = re.compile(r'Lattice\s*=\s*"([^"]*)"')


def read_xyz_frame(path, atoms=None, lattice_required=False):
    """Read one XYZ frame, and the cell its comment line declares."""
    if not os.path.isfile(path):
        raise ValueError(
            'the structure file %r does not exist' % (path,))
    with open(path) as handler:
        lines = handler.read().splitlines()
    if len(lines) < 2:
        raise ValueError(
            'the structure file %r is not an XYZ frame: %d line(s)'
            % (path, len(lines)))
    try:
        count = int(lines[0].split()[0])
    except (IndexError, ValueError):
        raise ValueError(
            'the structure file %r does not start with an atom count'
            % (path,)) from None
    body = lines[2:2 + count]
    if len(body) != count:
        raise ValueError(
            'the structure file %r declares %d atoms and carries %d'
            % (path, count, len(body)))
    symbols = []
    coordinates = []
    for line in body:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(
                'the structure file %r carries a line that is not '
                '"symbol x y z": %r' % (path, line))
        symbols.append(str(fields[0]))
        coordinates.append([float(fields[1]), float(fields[2]),
                            float(fields[3])])
    if atoms is not None:
        wanted = [str(name) for name in atoms]
        if symbols != wanted:
            raise ValueError(
                'the structure file %r holds %s and this run is built on %s: a '
                'path interpolated between two different atom orders is not a '
                'path' % (path, ' '.join(symbols), ' '.join(wanted)))
    lattice = read_lattice_comment(lines[1])
    if lattice_required and lattice is None:
        raise ValueError(
            'the structure file %r declares no cell and this run relaxes the '
            'box: a variable-cell band interpolates the lattice between its two '
            'endpoints, so both have to carry the Lattice="..." comment line '
            'the framework writes' % (path,))
    return symbols, np.asarray(coordinates, dtype=np.float64), lattice


def read_lattice_comment(comment):
    """The cell one frame's comment line declares, or None."""
    match = _LATTICE_KEY.search(str(comment or ''))
    if match is None:
        return None
    try:
        values = [float(value) for value in match.group(1).split()]
    except ValueError:
        return None
    if len(values) != 9:
        return None
    return np.asarray(values, dtype=np.float64).reshape(3, 3)
