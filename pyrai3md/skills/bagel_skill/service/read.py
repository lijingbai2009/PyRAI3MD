"""
======================================================================
Read-only entry for the BAGEL skill.

The framework's `jobtype read` calls `read_calculation` on a finished
calculation directory. It rebuilds the validated configuration, reads
the geometry from the BAGEL input deck (`<project>.json`), parses the
log labels and returns one frame. No engine is started and no file is
written.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import os

import numpy as np

from . import read_support
from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..schema.validate import validate
from .parser import parse_output
from .workflow import compile_workflow


def read_calculation(config, calcdir, project=None, profile=None):
    """Extract one frame from a finished BAGEL calculation directory.

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
        Read frame (symbols, coordinates, energy, gradient, nac).
    """
    project = read_support.resolve_project(calcdir, project)
    symbols, coordinates = _input_geometry(calcdir, project)

    raw = read_support.prepare_config(config)
    raw['atoms'] = symbols
    raw['coordinates'] = coordinates.tolist()
    raw['project'] = project
    raw['bagel_project'] = project
    if profile:
        raw['profile'] = str(profile)

    skill_config = validate(normalize(build_config(raw)))
    read_support.set_project(skill_config, project)

    workflow = compile_workflow(skill_config.calculation.profile)
    results = parse_output(skill_config, calcdir, None)

    geometry = read_support.last_coordinates(
        results.get('fin_coord'), len(symbols))
    if geometry is None:
        geometry = read_support.last_coordinates(
            results.get('coordinates'), len(symbols))
    return read_support.frame(
        symbols, geometry if geometry is not None else coordinates, results)


def _input_geometry(calcdir, project):
    """The geometry a BAGEL calculation ran on.

    BAGEL reads its molecule block from the JSON deck, where the
    coordinates are Angstrom (the deck sets `"angstrom": "true"`). A
    background point charge is written as a trailing `Q` atom and is not
    part of the molecular frame.
    """
    json_path = os.path.join(calcdir, '%s.json' % project)
    if not os.path.isfile(json_path):
        return read_support.read_geometry(calcdir, project)
    with open(json_path) as handle:
        payload = json.load(handle)
    molecule = payload.get('molecule') or {}
    rows = molecule.get('geometry') or []
    symbols = []
    coordinates = []
    for row in rows:
        if str(row.get('atom', '')) == 'Q':
            continue
        symbols.append(str(row['atom']))
        coordinates.append([float(value) for value in row['xyz']])
    if not symbols:
        raise ValueError(
            "BAGEL input %s carries no molecular geometry" % json_path)
    return symbols, np.asarray(coordinates, dtype=np.float64)