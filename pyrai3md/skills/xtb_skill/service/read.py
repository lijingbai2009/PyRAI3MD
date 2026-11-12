"""
======================================================================
Read-only entry for the xTB skill.

The framework's `jobtype read` calls `read_calculation` on a finished
calculation directory. It rebuilds the validated configuration, reads
the geometry the run used, parses the engrad labels and returns one
frame. No engine is started and no file is written.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from . import read_support
from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..schema.validate import validate
from .parser import parse_output
from .workflow import compile_workflow


def read_calculation(config, calcdir, project=None, profile=None):
    """Extract one frame from a finished xTB calculation directory.

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
        Read frame (symbols, coordinates, energy, gradient, units).
    """
    project = read_support.resolve_project(calcdir, project)
    symbols, coordinates = read_support.read_geometry(calcdir, project)

    raw = read_support.prepare_config(config)
    raw['atoms'] = symbols
    raw['coordinates'] = coordinates.tolist()
    raw['project'] = project
    raw['xtb_project'] = project
    if profile:
        raw['profile'] = str(profile)

    skill_config = validate(normalize(build_config(raw)))
    read_support.set_project(skill_config, project)

    workflow = compile_workflow(skill_config.calculation.profile)
    results = parse_output(skill_config, calcdir, workflow)
    return read_support.frame(symbols, coordinates, results)