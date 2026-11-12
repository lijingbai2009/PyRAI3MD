"""
======================================================================
Read-only entry for the LAMMPS skill.

The framework's `jobtype read` calls `read_calculation` on a finished
calculation directory. It rebuilds the validated configuration, reads
the geometry the run used, parses the energy/force artifacts and
returns one frame. The parser reports native units; the frame carries
the `_native_units` marker so the framework converts it. No engine is
started and no file is written.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os

import numpy as np

from . import read_support
from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..schema.validate import validate
from .builder.input_builder import _atom_coord_start
from .parser import parse_output
from .workflow import compile_workflow


def read_calculation(config, calcdir, project=None, profile=None):
    """Extract one frame from a finished LAMMPS calculation directory.

    Parameters
    ----------
    config : dict
        Merged flat skill configuration from the framework task queue.
    calcdir : str
        Calculation directory.
    project : str, optional
        Output file stem; the directory basename when omitted.
    profile : str, optional
        Profile override (the read section's `profile`).

    Returns
    -------
    dict
        Read frame (symbols, coordinates, energy, gradient) in the
        run's native unit set.
    """
    project = read_support.resolve_project(calcdir, project)
    symbols, coordinates = _input_geometry(
        calcdir, project, profile or config.get('profile'),
        config.get('atoms') or [])

    raw = read_support.prepare_config(config)
    raw['atoms'] = symbols
    raw['coordinates'] = coordinates.tolist()
    raw['project'] = project
    raw['lammps_project'] = project
    if profile:
        raw['profile'] = str(profile)

    skill_config = validate(normalize(build_config(raw)))
    read_support.set_project(skill_config, project)

    workflow = compile_workflow(skill_config.calculation.profile)
    results = dict(parse_output(skill_config, calcdir, workflow))
    results['_native_units'] = skill_config.calculation.lammps_units

    geometry = read_support.last_coordinates(
        results.get('fin_coord'), len(symbols))
    return read_support.frame(
        symbols, geometry if geometry is not None else coordinates, results)


def _input_geometry(calcdir, project, profile, symbols):
    """The geometry a LAMMPS calculation ran on.

    A dynamics/optimization run dumps its final frame to
    `<project>_<profile>.xyz`; a single point updates the coordinates in
    `<project>.data` in place. The data file carries no element labels,
    so its rows are paired with the read input's atom list.
    """
    generic = os.path.join(calcdir, '%s.xyz' % project)
    if os.path.isfile(generic):
        return read_support.read_geometry(calcdir, project)
    if profile:
        final = os.path.join(calcdir, '%s_%s.xyz' % (project, profile))
        if os.path.isfile(final):
            return read_support.read_geometry(
                calcdir, project, candidates=['%s_%s.xyz'
                                              % (project, profile)])
    return _data_geometry(calcdir, project, symbols)


def _data_geometry(calcdir, project, symbols):
    """Coordinates from the Atoms section of a LAMMPS data file."""
    path = os.path.join(calcdir, '%s.data' % project)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "no LAMMPS geometry in %s: expected %s.xyz, %s_<profile>.xyz "
            "or %s.data" % (calcdir, project, project, project))
    if not symbols:
        raise ValueError(
            "the LAMMPS data file carries no element labels; the read "
            "input must name the atoms")
    with open(path) as handle:
        lines = handle.read().splitlines()

    header = -1
    style = ''
    for index, line in enumerate(lines):
        if line.split('#')[0].strip() == 'Atoms':
            header = index
            if '#' in line:
                style = line.split('#', 1)[1].strip().split()[0].lower()
            break
    if header < 0:
        raise ValueError("LAMMPS data file has no Atoms section: %s" % path)

    coordinates = []
    for line in lines[header + 1:]:
        if not line.strip():
            if coordinates:
                break
            continue
        tokens = line.split()
        if len(tokens) < 4:
            break
        start, _n_image = _atom_coord_start(tokens, style, line, path)
        coordinates.append([float(tokens[start + offset])
                            for offset in range(3)])
        if len(coordinates) == len(symbols):
            break
    if len(coordinates) != len(symbols):
        raise ValueError(
            "LAMMPS data file %s carries %d atoms but the read input "
            "names %d" % (path, len(coordinates), len(symbols)))
    return list(symbols), np.asarray(coordinates, dtype=np.float64)