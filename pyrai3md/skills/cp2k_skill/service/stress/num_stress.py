"""
======================================================================
This is a standalone CP2K calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module drives the 'num' external strain finite-difference stress
of the CP2K skill.

It owns the whole route above the single-point API: it deforms the cell
and the coordinates, plans each perturbed point as an energy-only
profile of the same method (so a ground-state run fans out to
`dft_energy` and an excited-state run to `tddft_energy`), spreads the
points over processes or keeps them in this one, checks that every
point still describes the state the central configuration described,
and assembles the tensile-positive stress tensor in GPa.

Two facts make the route necessary rather than merely convenient:
CP2K refuses to print a stress tensor for a TDDFT run, and the built-in
NUMERICAL tensor differentiates an excited-state potential surface this
skill does not set up. Differentiating the state total energy
directly is what puts an excited-state stress within reach.

The fan-out mirrors the two dispatch axes of the framework -- the pool
is the multitask arm, and each point's own runner (local subprocess or
sbatch, chosen by execution_mode) is the remotetask arm -- but the
number of engine evaluations is planned here, inside the skill, because
it belongs to the stress method rather than to a workflow node.

"""

import multiprocessing as mp
import os
import warnings

import numpy as np

from ...schema.method import resolve_profile
from .point_runner import run_energy_point
from .strain import (assemble_stress, cell_volume, difference_component,
                     direction_matrices, perturb)

# Sign suffixes of the two sides of every central difference; kept in one
# place so a directory, a project name and a result label can never
# disagree about which side of the pair they describe.
_SIGNS = ((1, 'plus', 'p'), (-1, 'minus', 'm'))


def build_payloads(config, parent_calcdir: str) -> list:
    """Plan the perturbed single points of one strain evaluation.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration of the CENTRAL configuration: its cell,
        coordinates, method, profile, strain step and scope.
    parent_calcdir : str
        Directory of the central run. The points are placed beside it,
        never inside it, so a point's own cleanup cannot reach the
        central output and the central directory is left alone.

    Returns
    -------
    list of dict
        One payload per perturbed point, in the order
        direction_matrices() returns, each side of a pair next to its
        partner: 'label', 'calcdir', 'atoms', 'coordinates', the point's
        'slurm_template' (the central run's, if the user supplied one)
        and the plain 'config' the worker rebuilds the backend from.

    Raises
    ------
    ValueError
        If the central cell is unusable for a strain (singular).
    """
    lattice = np.asarray(config.system.lattice_vector, dtype=np.float64)
    coordinates = np.asarray(config.system.coordinates, dtype=np.float64)
    # Validated here rather than at the first difference: a singular
    # central cell has no stress definition, and the run should fail
    # before twelve engine evaluations are paid for.
    cell_volume(lattice)
    delta = float(config.calculation.stress_delta)
    scope = str(config.calculation.stress_scope)
    base_project = str(config.project)
    # Energy-only profile of the SAME method: 'dft' -> dft_energy,
    # 'tddft' -> tddft_energy. The perturbed points must not relax, and
    # they need no forces -- the difference reads energies.
    child_profile = resolve_profile(config.method, 'energy')

    parent_dir = os.path.dirname(os.path.abspath(parent_calcdir))
    sibling = os.path.basename(os.path.abspath(parent_calcdir))
    # A user-supplied submission template is written for the CENTRAL
    # project stem; the points carry their own stem, so their template
    # is copied along rather than left to be found by name.
    slurm_template = _read_slurm_template(base_project, parent_dir)

    payloads = []
    for direction, matrix in direction_matrices(scope):
        for sign, sign_name, tag in _SIGNS:
            strained_lattice, strained_positions = perturb(
                lattice, coordinates, matrix, delta, sign)
            label = f'{direction}_{sign_name}'
            project = f'{base_project}_{direction}{tag}'
            payloads.append({
                'label': label,
                'calcdir': os.path.join(parent_dir,
                                        f'{sibling}_num_{direction}{tag}'),
                'atoms': list(config.system.atoms),
                'coordinates': strained_positions.tolist(),
                'slurm_template': slurm_template,
                'config': _child_config(config, child_profile, project,
                                        strained_lattice, strained_positions),
            })
    return payloads


def _read_slurm_template(base_project: str, parent_dir: str):
    """Read the user's SLURM template for the central project stem.

    The local arm ignores it; the slurm arm needs it in each point's own
    directory under the point's stem, because
    service/runner/slurm_runner.py resolves a template by project name.
    The working directory is the primary place a user drops the file,
    with the directory holding the run as the second.

    Parameters
    ----------
    base_project : str
        Project stem of the central run.
    parent_dir : str
        Directory that holds the central run and the strain points.

    Returns
    -------
    str or None
        Template content, or None when the user supplied none (the
        skill then writes its own #SBATCH header).
    """
    for candidate in (f'{base_project}.slurm',
                      os.path.join(parent_dir, f'{base_project}.slurm')):
        if os.path.isfile(candidate):
            with open(candidate, 'r') as handle:
                return handle.read()
    return None


def _child_config(config, profile: str, project: str, lattice, positions):
    """Build the plain nested config of one perturbed point.

    The point inherits the central run's model (functional, basis,
    cutoffs, state counts, run arm) so every side of the difference
    differs from the central configuration by nothing but the strain;
    it overrides the geometry, the profile, and the four keywords that
    must not travel to a perturbed point: `stress` (no tensor is
    requested from the engine at all), its method and strain settings,
    and `multitask`/`multipool` (a point that fans out would multiply
    the pool by the pool).

    Parameters
    ----------
    config : SkillConfig
        Central configuration.
    profile : str
        Energy-only profile for this method.
    project : str
        File-name stem of this point; every CP2K output of a run is
        pinned to `{project}`, so a distinct stem is what keeps the
        points from overwriting one another's files.
    lattice : numpy.ndarray
        Strained 3x3 lattice, row vectors, Angstrom.
    positions : numpy.ndarray
        Strained coordinates, Angstrom.

    Returns
    -------
    dict
        Nested skill configuration the worker can rebuild from.
    """
    child = config.model_dump()
    system = child['system']
    system['lattice_vector'] = np.asarray(lattice).tolist()
    system['coordinates'] = np.asarray(positions).tolist()
    # `cell` is left as the central box: the periodic &CELL block is
    # built from lattice_vector alone (input_builder._build_cell_block),
    # and pbc -- not this field -- is the periodicity authority.
    system['volume'] = float(abs(np.linalg.det(np.asarray(lattice))))

    calculation = child['calculation']
    calculation['profile'] = profile
    calculation['project'] = project
    calculation['stress'] = False

    environment = child['environment']
    environment['cp2k_project'] = project
    environment['multitask'] = False
    environment['multipool'] = 1
    return child


def execute_points(payloads, workers: int) -> list:
    """Run every perturbed point, in this process or across a pool.

    Parameters
    ----------
    payloads : list of dict
        Payloads from build_payloads().
    workers : int
        Pool size. 1 keeps the points in this process; a larger value
        spreads them over that many spawned workers, which is the
        skill-side reading of the framework's multitask/multipool pair
        (min(workers, len(payloads)) processes, results returned in
        payload order).

    Returns
    -------
    list of dict
        One run_energy_point() result per payload, same order.

    Notes
    -----
    The pool lives here rather than in the framework (#3.10, reviewed and
    kept): a backend receives data (`calculate(atoms, coordinates, ...)`),
    never a callback, so a skill has no channel to ask the framework to
    fan work out. The framework's own multitask/multipool expands
    *planning tasks*; a finite-difference point is not one -- its config
    is derived from the central run (the same model, the strained cell)
    and its slurm template is restaged under the point's own stem, both
    of which the planner would have to rebuild in skill terms. Adding
    that channel would mean a new skill-to-framework contract, which the
    isolation rule and the no-new-knobs principle exclude; the payload
    is by-module-and-dict for the same reason the framework's batch
    children are (`point_runner` docstring).
    """
    if workers <= 1 or len(payloads) <= 1:
        return [run_energy_point(payload) for payload in payloads]

    # 'spawn' rather than a fork: the parent may already hold an open
    # multiplicative state (a runner, a directory handle from the
    # central run), and a forked child inheriting it would share the
    # parent's cleanup.
    context = mp.get_context('spawn')
    with context.Pool(processes=min(workers, len(payloads))) as pool:
        return pool.map(run_energy_point, payloads)


def _state_guard(config, outcomes, central_excitation):
    """Check that every point still describes the central states.

    Each side of a difference must be the same state as the central
    configuration's, or the quotient is the derivative of a jump, not a
    stress. This is a cheap identity check on the excitation energies,
    not character tracking: it catches a state that fell out of the
    computed roots and a root that moved past its neighbour, and it
    refuses a central spectrum with degenerate roots -- where no
    energy-only criterion can decide which state is which, and NTO or
    transition-density overlap tracking (out of scope here) would be
    needed.

    Parameters
    ----------
    config : SkillConfig
        Central configuration.
    outcomes : list of dict
        Point results from execute_points().
    central_excitation : numpy.ndarray or None
        Excitation energies (eV) of the central run, state order.

    Raises
    ------
    RuntimeError
        On a missing or non-finite energy, a dropped root, a root that
        moved more than half the smallest central gap, or a degenerate
        central spectrum.
    """
    nstate = int(config.system.nstate)
    for outcome in outcomes:
        energies = np.asarray(outcome['energy'], dtype=np.float64)
        if energies.shape != (nstate,):
            raise RuntimeError(
                f"strain point '{outcome['label']}' returned "
                f"{energies.shape[0]} energies for nstate={nstate}.")
        if not np.all(np.isfinite(energies)):
            raise RuntimeError(
                f"strain point '{outcome['label']}' returned non-finite "
                f"energies {energies.tolist()}.")

    if str(config.method).lower() != 'tddft':
        return

    central = np.asarray([] if central_excitation is None
                              else central_excitation, dtype=np.float64)
    if central.shape != (nstate,):
        raise RuntimeError(
            "num stress of a TDDFT run needs the central excitation "
            f"spectrum, got {central.shape[0]} entries for nstate={nstate}.")

    # Gaps between consecutive central roots; the S0 row carries 0.
    roots = central[1:]
    if roots.size == 0:
        return
    gaps = np.diff(np.sort(np.concatenate(([0.0], roots))))
    positive = gaps[gaps > 0.0]
    if positive.size != gaps.size:
        raise RuntimeError(
            "num stress cannot track states through a degeneracy: the "
            f"central spectrum is degenerate (gaps {gaps.tolist()} eV). "
            "Raise tddft_nstates or resolve the crossing before "
            "differentiating; energy-only state matching (this route) "
            "cannot follow a state through a conical intersection.")
    tolerance = 0.5 * float(positive.min())

    for outcome in outcomes:
        point = np.asarray([] if outcome.get('excitation') is None
                                 else outcome['excitation'],
                           dtype=np.float64)
        if point.shape != central.shape:
            raise RuntimeError(
                f"strain point '{outcome['label']}' returned "
                f"{point.shape[0]} excitation rows where the central run "
                f"returned {central.shape[0]}; a dropped root would make "
                "the difference meaningless.")
        for index in range(1, nstate):
            if central[index] != 0.0 and point[index] == 0.0:
                raise RuntimeError(
                    f"strain point '{outcome['label']}' did not report "
                    f"state {index}, which the central run did; the "
                    "energy of that side is a zero fill, not a result.")
        moved = np.abs(point - central)
        worst = int(np.argmax(moved))
        if moved[worst] > tolerance:
            raise RuntimeError(
                f"strain point '{outcome['label']}' moved state {worst} "
                f"by {moved[worst]:.6f} eV, over half the smallest "
                f"central gap ({tolerance:.6f} eV): the two sides of the "
                "difference are not the same state.")


def _warn_oversubscription(config, workers: int, points: int):
    """Warn when the strain pool cannot fit the machine.

    Oversubscription is allowed, never refused (the machine this run
    lands on is not the one that parsed the input, and on the slurm arm
    each point is a separate job), so this only says what the numbers
    imply. The framework warns the same way for the inner MPI/OpenMP
    request in schema/validate.py::validate_environment.
    """
    if workers <= 1:
        return
    per_point = max(1, int(config.nproc)) * max(1, int(config.thread))
    capacity = os.cpu_count() or 1
    requested = workers * per_point
    if requested <= capacity:
        return
    warnings.warn(
        "cp2k_skill num stress: %d strain points in a pool of %d "
        "workers x %d ranks x %d threads = %d threads against %d "
        "detected CPUs. Lower multipool, nproc or thread."
        % (points, workers, int(config.nproc), int(config.thread),
           requested, capacity))


def compute_num_stress(config, parent_calcdir: str, central_results: dict):
    """Evaluate the strain finite-difference stress tensor.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration of the central run (stress on, route
        'num', periodic cell).
    parent_calcdir : str
        Directory of the central run, whose energies are reused as the
        reference of the state guard.
    central_results : dict
        Parsed results of the central run ('energy', 'excitation').

    Returns
    -------
    numpy.ndarray
        (nstate, 3, 3) tensile-positive stress tensor in GPa.

    Raises
    ------
    RuntimeError
        If any perturbed point fails, returns a non-finite energy, or
        no longer describes the state it is differentiated against.
    """
    payloads = build_payloads(config, parent_calcdir)
    # Tri-state read: multitask undeclared (None) or multipool unset both
    # mean serial here, the same aggregation the dispatch layer applies.
    workers = (max(1, int(config.environment.multipool or 1))
               if config.environment.multitask else 1)
    _warn_oversubscription(config, workers, len(payloads))
    outcomes = execute_points(payloads, workers)

    _state_guard(config, outcomes, central_results.get('excitation'))

    lattice = np.asarray(config.system.lattice_vector, dtype=np.float64)
    volume0 = cell_volume(lattice)
    delta = float(config.calculation.stress_delta)
    scope = str(config.calculation.stress_scope)

    by_label = {outcome['label']: outcome for outcome in outcomes}
    components = []
    for direction, _matrix in direction_matrices(scope):
        plus = np.asarray(by_label[f'{direction}_plus']['energy'],
                          dtype=np.float64)
        minus = np.asarray(by_label[f'{direction}_minus']['energy'],
                           dtype=np.float64)
        components.append(difference_component(plus, minus, delta, volume0))

    return assemble_stress(components, scope)
