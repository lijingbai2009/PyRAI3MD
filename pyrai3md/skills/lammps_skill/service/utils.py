"""
======================================================================
This is a standalone LAMMPS calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides shared utility functions for the LAMMPS skill,
including calculation directory construction, unit conversion
constants, and error-safe execution.

"""

import os
import traceback
import warnings

from ..schema.config import DEFAULT, Dynamics

# Unit conversion constants (skill-internal, backend independence:
# never import pyrai3md.utilities here).
# LAMMPS 'real' units: energy in kcal/mol, force in kcal/mol/Angstrom
# LAMMPS 'metal' units: energy in eV, force in eV/Angstrom
# The parser keeps energy and gradient in native units and marks the
# result with `_native_units`; the framework applies the manifest's
# conversion factors. Stress is normalized to bar and velocity to au here.
HARTREE_TO_KCAL = 627.5094740631
HARTREE_TO_EV = 27.211386245988
KCAL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
BOHR_TO_ANGSTROM = 0.529177210903

# Energy conversion factors to hartree by lammps_units style
ENERGY_TO_HARTREE = {
    'real': KCAL_TO_HARTREE,
    'metal': EV_TO_HARTREE,
}
# Force conversion factors to hartree/bohr by lammps_units style.
# gradient = -force(E/Angstrom) * E_TO_HARTREE * BOHR_TO_ANGSTROM
# real: kcal/mol/Angstrom -> hartree/bohr, factor ~ 8.4330e-4
# metal: eV/Angstrom -> hartree/bohr, factor ~ 1.9373e-2
FORCE_TO_HARTREE_PER_BOHR = {
    'real': KCAL_TO_HARTREE * BOHR_TO_ANGSTROM,
    'metal': EV_TO_HARTREE * BOHR_TO_ANGSTROM,
}

# Velocity conversion from the framework Bohr/au convention.
# 1 au velocity = Bohr / t_au, with t_au = 0.024188843265857 fs,
# so 1 au = 0.529177210903 / 0.024188843265857 Angstrom/fs.
AU_TIME_IN_FS = 0.024188843265857
AU_VELOCITY_TO_ANGSTROM_PER_FS = BOHR_TO_ANGSTROM / AU_TIME_IN_FS

# Velocity units of the LAMMPS data file by units style:
# real -> Angstrom/fs, metal -> Angstrom/ps
VELOCITY_TO_LAMMPS = {
    'real': AU_VELOCITY_TO_ANGSTROM_PER_FS,
    'metal': AU_VELOCITY_TO_ANGSTROM_PER_FS * 1000.0,
}

# Reverse velocity conversion (LAMMPS dump output -> framework
# Bohr/au), used by the parser to report the final velocities of the
# MD profiles through the fin_velocity channel.
LAMMPS_VELOCITY_TO_AU = {
    units: 1.0 / factor for units, factor in VELOCITY_TO_LAMMPS.items()
}

# Pressure unit of the LAMMPS 'compute pressure' output by units style:
# real -> atm, metal -> bar (dev/skills/STRESS_DEVELOPMENT.md section 4.3).
# The skill manifest declares units.stress: bar, so the parser
# normalizes the raw configurational pressure into bar with these
# factors; the framework then converts bar -> Hartree/Bohr^3 centrally.
ATM_TO_BAR = 1.01325
PRESSURE_TO_BAR = {
    'real': ATM_TO_BAR,
    'metal': 1.0,
}


def output_basename(project: str, profile: str) -> str:
    """Return the per-task file basename for a workflow profile.

    The generated LAMMPS input script, log and run script names all
    share the dump output basename so different tasks placed in the
    same directory never overwrite each other:
      - ff_gradient: '{project}_force' (grouped with the per-step
        result files '{title}.energy'/'.coords'/'.force'/'.stress')
      - all other profiles: '{project}_{profile}' (grouped with the
        '{project}_{profile}.xyz'/'.cond' dumps)

    Parameters
    ----------
    project : str
        LAMMPS project name (lammps_project keyword).
    profile : str
        Calculation profile name.

    Returns
    -------
    str
        Basename without extension, e.g. 'h2o_ff_npt' or 'h2o_force'.
    """
    if profile == 'ff_gradient':
        return f'{project}_force'
    return f'{project}_{profile}'


# The four per-step result files a single point writes, keyed by role.
STEP_DATA_ROLES = ('energy', 'coords', 'force', 'stress')


def step_data_files(title: str, *, gradient: bool = False,
                    stress: bool = False) -> dict:
    """Names of the per-step result files, one source for writer and parser.

    A single-point step states its geometry and every number it produced
    as its own text file under the run's stem, so a person (and the
    parser) finds the result without reading a log: ``{title}.energy``
    holds one number, ``{title}.coords`` and ``{title}.force`` are
    LAMMPS text dumps in atom-id order, ``{title}.stress`` holds the six
    pressure components. ``force`` and ``stress`` appear only when the
    profile asks for a gradient and the config asks for stress, so the
    writer's command list and the parser's expectations cannot drift
    apart on their own.

    Parameters
    ----------
    title : str
        :func:`output_basename` of the run.
    gradient : bool
        Include the force file.
    stress : bool
        Include the stress file.

    Returns
    -------
    dict
        ``{'energy', 'coords'}`` plus ``'force'``/``'stress'`` when asked.
    """
    names = {'energy': f'{title}.energy', 'coords': f'{title}.coords'}
    if gradient:
        names['force'] = f'{title}.force'
    if stress:
        names['stress'] = f'{title}.stress'
    return names


def build_calcdir(base_name='tmp_LAMMPS', execution_id='', base_path=None,
                  project=''):
    """Build the calculation directory path.

    The directory name is controlled by the caller (upper-level
    functionality) to avoid conflicts when multiple backend
    instances are created simultaneously:
      - execution_id='': falls back to the base_name (tmp_LAMMPS)
      - execution_id='tmp_LAMMPS' (single task) or 'tmp_low_mm_LAMMPS': the execution namespace is
        used directly as the directory name

    Parameters
    ----------
    base_name : str
        Fallback directory name for standalone mode.
    execution_id : str
        Execution namespace identifier assigned by the Calculator.
        Empty string means standalone mode (base_name fallback).
    project : str, optional
        Project this calculation runs under. When it is set, the engine
        token of the directory name is replaced by it and every other part
        stays: `tmp_LAMMPS` becomes `tmp_<project>`, and a framework
        namespace keeps its region, role and runtime parts. Empty (the
        default) leaves the naming above untouched.
    base_path : str, optional
        Base directory (e.g. environment scratch_path).
        Defaults to the current working directory.

    Returns
    -------
    str
        Absolute calculation directory path.
    """
    dir_name = execution_id if execution_id else base_name
    if project and project != DEFAULT['project']:
        # The schema's own default is how "no project was declared" arrives
        # here, and it must not rename anything: a framework run names its
        # directory by its execution namespace, which is already unique per
        # task, and renaming it would move the artifacts of a run whose owner
        # never asked for a project.
        if project != os.path.basename(project) or '..' in project:
            raise ValueError(
                'project %r cannot be used as a directory name' % project)
        token = (base_name[len('tmp_'):] if base_name.startswith('tmp_')
                 else base_name)
        parts = dir_name.split('_')
        # Replace the window of segments that spells this engine's token, so
        # every other part of the namespace - region, role, layer, runtime
        # index - stays where it was. The token can span segments (`NN_PES`)
        # and the framework writes it in its own case (`PySCF`), hence the
        # join and the case fold. A name that carries no such window is left
        # alone: the execution namespace distinguishes it already, and
        # inventing a name here would collapse two tasks that share a project
        # into one directory.
        token_parts = token.split('_')
        width = len(token_parts)
        found = -1
        for index in range(len(parts) - width + 1):
            candidate = '_'.join(parts[index:index + width])
            if candidate.lower() == token.lower():
                found = index
        if found >= 0:
            parts[found:found + width] = [project]
            dir_name = '_'.join(parts)
    base = base_path or os.getcwd()
    return os.path.join(base, dir_name)


# Submodel flattening order.  Later submodels override earlier ones
# on name collision; environment last so its scratch_path (file
# placement) wins over calculation.scratch_path.  The framework
# job-control and adaptive-sampling sections are skipped: they drive
# the framework only (jobtype dispatch, tri-state multitask/
# remotetask graph flags) and must not reach the skill.
FLATTEN_ORDER = ('dynamics', 'system', 'properties',
                 'calculation', 'environment')

# Whitelist of skill-declared dynamics keys.  The framework Dynamics
# section also carries skill-independent keywords (initial-condition
# method, velocity scale, dispatch flags, ...) that must never leak
# into the skill flat config through silent cross-section overrides.
_DYNAMICS_KEYS = frozenset(Dynamics.model_fields)


def flatten_extended_config(config):
    """Flatten a nested extended PyRAIConfig into a legacy flat dict.

    .. deprecated::
        Use SkillConfig attribute access instead.

    The nested config built by PyRAIState (framework submodels plus
    skill calculation/environment/properties) is flattened so that
    existing services keep consuming a flat dictionary.

    Framework-only sections (control, adaptive) are skipped, and the
    dynamics section keeps only the keys declared by the skill's own
    Dynamics schema, so skill-independent framework keywords cannot
    override skill values on name collisions.

    Parameters
    ----------
    config : BaseModel
        Nested extended PyRAIConfig instance.

    Returns
    -------
    dict
        Flat configuration dictionary.
    """
    warnings.warn(
        'flatten_extended_config is deprecated. '
        'Use SkillConfig attribute access instead.',
        DeprecationWarning,
        stacklevel=2,
    )
    nested = config.model_dump()
    flat = {}
    for sub_name in FLATTEN_ORDER:
        sub_data = nested.get(sub_name)
        if not isinstance(sub_data, dict):
            continue
        if sub_name == 'dynamics':
            sub_data = {key: value for key, value in sub_data.items()
                        if key in _DYNAMICS_KEYS}
        flat.update(sub_data)
    return flat


def safe_run(func, *args, **kwargs):
    """Execute a function without interrupting the main process.

    Parameters
    ----------
    func : callable
        Function to execute.

    Returns
    -------
    str
        'success' or traceback string on failure.
    """
    try:
        func(*args, **kwargs)
        status = 'success'
    except Exception:
        status = traceback.format_exc()

    return status


# How much of a LAMMPS log an error message quotes. The engine reports
# its own failure at the end of its output, so the tail is the part that
# says what happened.
LOG_TAIL_LINES = 20


def log_tail(lines, count: int = LOG_TAIL_LINES) -> str:
    """Quote the end of a LAMMPS log, where the engine writes its error.

    Both execution paths report a failed calculation this way, so a
    resident failure reads like the subprocess one instead of stopping at
    an exit code.

    Parameters
    ----------
    lines : iterable
        Log lines.
    count : int
        How many of the last lines to keep.

    Returns
    -------
    str
        The quoted lines, '' when there were none.
    """
    return '\n'.join(str(line).rstrip('\n')
                     for line in list(lines)[-int(count):])


def read_log_tail(path, count: int = LOG_TAIL_LINES) -> str:
    """Read a LAMMPS log and quote its end; '' when it is unreadable.

    Parameters
    ----------
    path : str
        Log file path.
    count : int
        How many of the last lines to keep.

    Returns
    -------
    str
        The quoted tail, '' when the file does not exist yet.
    """
    try:
        with open(path, 'r', errors='replace') as handle:
            return log_tail(handle, count)
    except OSError:
        return ''
