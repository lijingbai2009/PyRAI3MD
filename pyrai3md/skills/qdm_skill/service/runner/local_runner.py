"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 24, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides local execution for QDM calculations.
QDM is a pure Python/numpy service, so local execution means
directly calling the calculator / reference solver in the current
process.

"""

import os
import sys

import numpy as np

from ..models import get_model
from ..calculator import calculate_properties
from ..reference import run_reference


def get_model_instance(config):
    """Instantiate the analytic model selected by the config.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    AnalyticModel
        Model with optional user parameter overrides applied.
    """
    calc = config.calculation
    params = {
        'tully_a': calc.tully_a,
        'tully_b': calc.tully_b,
        'tully_c': calc.tully_c,
        'tully_d': calc.tully_d,
        'tully_e': calc.tully_e,
        'lz_a': calc.lz_a,
        'lz_b': calc.lz_b,
    }
    return get_model(calc.method, params=params)


def dispatch_calculation(config, calcdir):
    """Execute the profile-selected calculation.

    The four single-point profiles share the same calculator; only
    the assembled output differs. The reference profile runs the
    full split-operator propagation and writes its trajectory files
    into calcdir.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str
        Calculation directory for reference output files.

    Returns
    -------
    dict
        JSON-safe formatted results (numpy arrays converted).
    """
    calc = config.calculation
    profile = calc.profile
    model = get_model_instance(config)
    coordinates = config.system.coordinates

    if profile.endswith('_reference'):
        project = calc.project or 'QDM'
        results = run_reference(
            model, coordinates, calcdir, project,
            velocities=config.system.velocities or None,
            current_state=config.system.current_state,
            model_mass=calc.model_mass,
            wavepacket_sigma=calc.wavepacket_sigma,
            ref_ngrid=calc.ref_ngrid,
            ref_dt=calc.ref_dt,
            ref_nsteps=calc.ref_nsteps,
            ref_save_every=calc.ref_save_every,
            wavepacket_k0=calc.wavepacket_k0,
            ref_xmin=calc.ref_xmin,
            ref_xmax=calc.ref_xmax)
    else:
        with_gradient = not profile.endswith('_energy')
        with_nac = profile.endswith('_nac')
        properties = calculate_properties(
            model, coordinates,
            target_states=calc.target_states or None,
            target_nac_pair=calc.target_nac_pair,
            with_gradient=with_gradient, with_nac=with_nac)
        project = calc.project or 'QDM'
        write_single_point_files(calcdir, project, properties)
        results = to_list(properties)

    return results


def write_single_point_files(calcdir, project, properties):
    """Save the single-point property arrays as plain text files.

    The energy/gradient/nac profiles otherwise return data only
    through the results dictionary; these files keep a persistent
    copy in calcdir so the skill output can be cross-checked against
    the framework run log. Naming follows the reference family
    ({project}_population.dat, ...).

    Parameters
    ----------
    calcdir : str
        Calculation directory for the output files.
    project : str
        Project name prefix.
    properties : dict
        Property dictionary from calculate_properties (numpy
        arrays; missing entries are skipped).
    """
    os.makedirs(calcdir, exist_ok=True)

    energy = properties.get('energy')
    if energy is not None:
        np.savetxt(
            os.path.join(calcdir, '%s_energy.dat' % project),
            np.asarray(energy, dtype=float).reshape(-1),
            header='%s electronic energies (Hartree), one state per row'
                   % project)

    gradient = properties.get('gradient')
    if gradient is not None:
        array = np.asarray(gradient, dtype=float)
        np.savetxt(
            os.path.join(calcdir, '%s_gradient.dat' % project),
            array.reshape(array.shape[0], -1),
            header='%s gradients (Hartree/Bohr), one row per state, '
                   'columns = atoms x xyz' % project)

    nac = properties.get('nac')
    if nac is not None:
        array = np.asarray(nac, dtype=float)
        if array.size > 0:
            np.savetxt(
                os.path.join(calcdir, '%s_nac.dat' % project),
                array.reshape(array.shape[0], -1),
                header='%s nonadiabatic couplings (Hartree/Bohr), '
                       'one row per nac pair, columns = atoms x xyz'
                       % project)


def run_local(config, calcdir) -> dict:
    """Execute QDM calculation locally (in-process).

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str
        Calculation directory for reference output files.

    Returns
    -------
    dict
        Formatted results dictionary.
    """
    # Thread count for numpy BLAS. OpenMP reads OMP_NUM_THREADS once, at
    # the first parallel region of the process, so a per-term write cannot
    # retune what is already latched: the first term fixes it for the
    # process (#302). Keep that first value and say so when it is not the
    # one this term asked for.
    requested = str(config.environment.thread)
    effective = os.environ.setdefault('OMP_NUM_THREADS', requested)
    if effective != requested:
        print('[QDM] OMP_NUM_THREADS is already %s for this process, so '
              'thread=%s is ignored; the thread count is fixed at the '
              'first parallel region.' % (effective, requested),
              file=sys.stderr)

    return dispatch_calculation(config, calcdir)


def to_list(results: dict) -> dict:
    """Convert numpy arrays in results to lists for JSON serialization.

    Parameters
    ----------
    results : dict
        Results dictionary with potential numpy arrays.

    Returns
    -------
    dict
        Results with numpy arrays converted to lists.
    """
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results[key] = value.tolist()

    return results
