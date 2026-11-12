"""
======================================================================
Read-only entry for the OpenMolcas skill.

The framework's `jobtype read` calls `read_calculation` on a finished
calculation directory. It rebuilds the validated configuration, reads
the geometry the run used, parses the log labels and returns one frame.
No engine is started and no orbital file is copied: `parse_log` is used
instead of `parse_output`, whose orbital management writes into the
calculation directory.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os

from . import read_support
from ..schema.adapter import build_config
from ..schema.method import has_transition_request
from ..schema.normalize import normalize
from ..schema.validate import validate
from .parser import parse_log
from .parser.output_parser import parse_fin_coord
from .workflow import compile_workflow


def read_calculation(config, calcdir, project=None, profile=None):
    """Extract one frame from a finished OpenMolcas calculation directory.

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
        Read frame (symbols, coordinates, energy, gradient, nac, soc).
    """
    project = read_support.resolve_project(calcdir, project)
    symbols, coordinates = read_support.read_geometry(calcdir, project)

    raw = read_support.prepare_config(config)
    raw['atoms'] = symbols
    raw['coordinates'] = coordinates.tolist()
    raw['project'] = project
    raw['molcas_project'] = project
    if profile:
        raw['profile'] = str(profile)

    skill_config = validate(normalize(build_config(raw)))
    read_support.set_project(skill_config, project)

    transition = has_transition_request(skill_config.calculation.jobtype)
    workflow = compile_workflow(
        skill_config.calculation.profile,
        transition=transition, config=skill_config)

    log_path = os.path.join(calcdir, '%s.log' % project)
    if not os.path.isfile(log_path):
        legacy = os.path.join(calcdir, '%s.out' % project)
        if os.path.isfile(legacy):
            log_path = legacy
    results = parse_log(log_path, skill_config, workflow=workflow)

    # An iterative profile reports its final structure from the skill's
    # own product (Opt.xyz / md.xyz); a single point has none, and the
    # geometry the labels belong to is the input frame.
    geometry = None
    try:
        value = parse_fin_coord(
            skill_config, calcdir,
            not_converged=bool(results.get('log_not_converged', False)))
        geometry = read_support.last_coordinates(value, len(symbols))
    except (RuntimeError, ValueError, OSError):
        geometry = None
    return read_support.frame(
        symbols, geometry if geometry is not None else coordinates, results)