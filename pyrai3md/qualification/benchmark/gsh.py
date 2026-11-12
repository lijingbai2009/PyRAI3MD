"""
======================================================================
GSH benchmark task module (flat benchmark layout).

One module per benchmark task type: gsh.py serves the unified
gsh.yaml case definition shipped next to it. The Tully model
(tully1/2/3) is selected dynamically through the production.method
key of the case, so calibrating another Tully model only changes
the method of the pmd_env benchmark entries -- no new YAML file
per model (a dedicated gsh_<method>.yaml would take precedence
for one specific method).

The GSH (Zhu-Nakamura) calibration reuses the fssh calibration
infrastructure wherever the semantics are algorithm-agnostic:

- load_case / apply_overrides: the case YAML layout and the
  pmd_env override merging are identical (REFERENCE_OVERRIDE_KEYS
  routing included);
- build_reference_input: the exact split-operator quantum
  reference depends on the Tully model and the wavepacket setup
  only, never on the hopping algorithm under calibration;
- evaluate_run: delegates to the shared evaluate_unified
  final-state metric set (the final trajectory / state / reference
  populations plus the final-state differences); every case
  supplies the deterministic ZN kernel replay as the reported
  internal hopping probability (the lz family additionally compares
  against the analytic Landau-Zener probability).

GSH-specific parts live here: the production md keyword forwarding
table spans the GSH-relevant md schema keywords only (sfhp, the
gap thresholds, the velocity adjustment policy and maxh), the
production input builder writes the gsh md workflow YAML and
forwards the model parameter overrides (lz_a/lz_b) into the qdm
section when present. The
fssh-only keywords (nactype, substep, rk4, integrate, deco and
the fssh_* kernel selectors) are deliberately absent -- the
unrecognized-key check fails loudly when they are requested
instead of forwarding them silently.

GSH consumes the all-state gradients of the Tully models: the md
schema default activestate 0 requests every state's gradient on
every step, so results['gradient'] always carries the
(nstate, natom, 3) rows the Zhu-Nakamura internal-conversion
kernel needs (a present-only 'activestate' forward keeps the
escape hatch for explicit overrides).

Sep 2 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os

import numpy as np
import yaml

from ...skills.qdm_skill.service.models import get_model

from .fssh import PRODUCTION_QDM_KEYWORDS
from ...utilities.units import EV_TO_HARTREE
from ...workflow.integrator.gsh import internal_conversion_prob
from . import lz_data
from .fssh import (
    AMU_TO_AU,
    BOHR_TO_ANGSTROM,
    LZ_METHODS,
    PRODUCTION_STRUCTURAL_KEYS,
    _find_case_yaml,
    apply_overrides,
    build_reference_input,
    collect_populations,
    ensemble_average,
    evaluate_unified,
    load_case,
    parse_reference_population,
)


# Case production keywords forwarded into the md section of the
# GSH production input: case keyword -> (md keyword, type
# converter, builder default). The table spans the GSH-relevant
# surface hopping block of the md schema (schema/config.py); a
# None default means "present-only" forwarding (an absent case key
# keeps the md-schema default so the schema stays the single
# source); the callable sfhp default is written even when the case
# does not set the key (the benchmark defines these semantics
# itself: sfhp selects the algorithm under calibration).
PRODUCTION_MD_KEYWORDS = {
    'sfhp': ('sfhp', str, 'gsh'),
    'gap': ('gap', float, None),
    'gapsoc': ('gapsoc', float, None),
    'adjust': ('adjust', int, None),
    'reflect': ('reflect', int, None),
    'maxh': ('maxh', int, None),
    # Gradient scope (0: all-state gradients, the GSH requirement;
    # the schema default is already 0, so the shipped case never
    # sets this key and it stays present-only).
    'activestate': ('activestate', int, None),
}


def build_production_input(case, run_dir):
    """Write the gsh md input YAML for the production trajectory.

    The input starts from the same fixed coordinate x0 and velocity
    v0 as the reference (initcond 0: use the state geometry and
    velocities, replicated once), so both sides share identical
    initial conditions by construction. The file lives in the
    benchmark run directory as production.yaml; the trajectories
    themselves are isolated by the md dispatch layer in
    '{title}_{idx}' directories next to the launch point.

    Parameters
    ----------
    case : dict
        Parsed case definition.
    run_dir : str
        Benchmark run directory holding the generated
        production.yaml (created here when missing).

    Returns
    -------
    str
        Path of the generated production.yaml workflow input.
    """
    case_name = str(case.get('case', 'gsh'))
    prod = case['production']
    method = str(prod.get('method', 'tully1'))
    nstep = int(prod['nstep'])
    dt = float(prod['dt'])
    ninitcond = int(prod.get('ninitcond', 1))
    # Multitask pool for the ninitcond trajectories: the md fan-out
    # dispatch reads the worker count from the multipool keyword of
    # the md input section.
    ncpus = int(prod.get('ncpus', 1))
    x0 = float(prod['x0'])
    v0 = float(prod.get('v0', 0.0))
    mass = float(prod.get('model_mass', 2000.0))
    initial_state = int(prod.get('initial_state', 0))

    os.makedirs(run_dir, exist_ok=True)

    workflow = {
        'control': {
            'title': 'benchmark_%s' % case_name,
            'jobtype': 'md',
            'theory': ['qdm'],
            'driver': 'main',
        },
        'atoms': ['X'],
        'coordinates': [[x0 * BOHR_TO_ANGSTROM, 0.0, 0.0]],
        'velocities': [[v0, 0.0, 0.0]],
        # The framework integrator reads amu masses from the system
        # section; convert the au model mass so both sides propagate
        # the identical particle (mass_au == model_mass).
        'mass': [mass / AMU_TO_AU],
        'molecule': {
            'charges': 0,
            'multiplicity': 1,
            'ci': [2],
            'spin': [0],
        },
        'md': {
            # initcond is a string keyword in the md schema ('0'
            # reuses the supplied state geometry and velocities).
            'initcond': '0',
            'ninitcond': ninitcond,
            # The dispatch pool only starts when multitask is on;
            # multipool then sizes the worker pool.
            'multitask': max(1, ncpus) > 1,
            'multipool': max(1, ncpus),
            'step': nstep,
            'size': dt,
            # md.root is externally 1-based (adapter contract);
            # initial_state keeps its 0-based generator meaning.
            'root': initial_state + 1,
        },
        # No framework-level execution switch: the retired
        # &environment keys are refused (schema/config
        # .RETIRE_ENVIRONMENT_KEYS). The qdm skill section below
        # declares its own execution_mode, which is the live source.
        'environment': {},
        'qdm': {
            'method': method,
            'model_mass': mass,
            'execution_mode': 'local',
        },
    }

    # Forward the case's md keywords through the explicit GSH table
    # (sfhp carries its builder default; the rest are present-only,
    # e.g. the shipped case pins adjust 1 and reflect 1).
    for key, (md_key, convert, default) in PRODUCTION_MD_KEYWORDS.items():
        if key in prod:
            workflow['md'][md_key] = convert(prod[key])
        elif default is not None:
            workflow['md'][md_key] = convert(default)

    # Model parameter overrides land in the qdm section (present-
    # only; the reference builder applies the same forwarding).
    for key, convert in PRODUCTION_QDM_KEYWORDS.items():
        if key in prod:
            workflow['qdm'][key] = convert(prod[key])

    # Silent keyword drops are the defect class this guard closes:
    # every production key must be consumed by one of the two
    # builders, otherwise the merged case YAML would promise a
    # parameter the execution input never carries (fssh-only keys
    # like deco/rk4/nactype fail here instead of being forwarded).
    consumed = (set(PRODUCTION_MD_KEYWORDS)
                | set(PRODUCTION_QDM_KEYWORDS)
                | set(PRODUCTION_STRUCTURAL_KEYS))
    unrecognized = sorted(set(prod) - consumed)
    if unrecognized:
        raise ValueError(
            'unrecognized production keywords: %s; extend '
            'PRODUCTION_MD_KEYWORDS or remove them from the case '
            'definition / pmd_env overrides'
            % ', '.join(unrecognized))

    yaml_path = os.path.join(run_dir, 'production.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(workflow, f, sort_keys=False)
    return yaml_path


# ------------------------------------------------------------------
# lz family calibration evaluation (lz1/lz2/lz3)
# ------------------------------------------------------------------
#
# The lz cases calibrate the GSH kernel against the single-passage
# Landau-Zener physics: the unified final-state metric set carries
# both comparison targets -- the deterministic ZN kernel replay
# (internal_hopping_probability, re-evaluated step by step on the
# recorded trajectory snapshots) and the analytic Landau-Zener
# probability (analytical_hopping_probability) -- next to the
# exact wavepacket reference. The steep linear slopes make the
# quantum packet re-cross the interaction region on a period
# shorter than the observation window (multi-passage oscillations
# the single-passage hopping model is not designed to reproduce),
# so the analytic comparison targets anchor the verdict.
#
# The tully cases carry no analytic hopping probability, but the
# deterministic ZN kernel replay of the recorded trajectories still
# supplies the reported internal hopping probability (the adiabatic
# gradients are numerically differentiated from the recorded state
# energies, the Tully models carry no closed-form adiabatic
# gradient interface); the verdict rests on the wavepacket
# reference difference alone.


def _parse_lz_trajectory(traj_dir, title):
    """Read the recorded coordinates and energies of one trajectory.

    Parameters
    ----------
    traj_dir : str
        Trajectory working directory holding the md output files.
    title : str
        Case title prefix of the md output file names.

    Returns
    -------
    tuple
        (x_bohr, ekin, state_energies): per-step coordinate (Bohr),
        kinetic energy (Hartree) and the two adiabatic state
        energies (Hartree), all aligned step by step.
    """
    x_bohr = lz_data.read_coordinates(traj_dir, title)
    ekin, state_energies = lz_data.read_energies(traj_dir, title)
    return x_bohr, ekin, state_energies


def _lz_kernel_probability(run_dir, prod):
    """Deterministic replay of the ZN kernel gate probability.

    Every trajectory of an lz case shares identical initial
    conditions (initcond 0), so the first recorded trajectory
    carries the deterministic pre-hop record. The production
    internal_conversion_prob kernel is re-evaluated step by step on
    the recorded coordinates and energies with the exact analytic
    adiabatic gradients of the selected lz model (the model class
    owns the closed form); the first non-zero probability is the
    gate value the hop draw samples from.

    Parameters
    ----------
    run_dir : str
        Benchmark run directory holding production/.
    prod : dict
        Production section of the lz case definition.

    Returns
    -------
    float
        The gate-step ZN hopping probability (0.0 when the replay
        never passes the gate).
    """
    prod_dir = os.path.join(run_dir, 'production')
    title = 'benchmark_%s' % str(prod.get('_case_name', 'gsh_lz1'))
    traj_dirs = sorted(
        d for d in os.listdir(prod_dir)
        if os.path.isdir(os.path.join(prod_dir, d))
        and d.startswith(title + '_'))
    if not traj_dirs:
        return 0.0
    x_bohr, ekin, energies = _parse_lz_trajectory(
        os.path.join(prod_dir, traj_dirs[0]), title)
    model = get_model(str(prod.get('method', 'lz1')).lower(),
                      params={'lz_a': prod.get('lz_a'),
                              'lz_b': prod.get('lz_b')})
    mass = float(prod.get('model_mass', 2000.0))
    gap_au = float(prod.get('gap', 0.5)) * EV_TO_HARTREE
    mass_au = np.array([mass])
    nstep = len(x_bohr)
    for n in range(2, nstep):
        grad_n = np.zeros((2, 1, 3))
        grad_n[0, 0, 0], grad_n[1, 0, 0] = model.adiabatic_gradients(
            x_bohr[n])
        grad_pp = np.zeros((2, 1, 3))
        grad_pp[0, 0, 0], grad_pp[1, 0, 0] = model.adiabatic_gradients(
            x_bohr[n - 2])
        prob, _ = internal_conversion_prob(
            1, 0,
            energies[n], energies[n - 1], energies[n - 2],
            grad_n, grad_pp,
            np.array([[x_bohr[n] * BOHR_TO_ANGSTROM, 0.0, 0.0]]),
            np.array([[x_bohr[n - 1] * BOHR_TO_ANGSTROM, 0.0, 0.0]]),
            np.array([[x_bohr[n - 2] * BOHR_TO_ANGSTROM, 0.0, 0.0]]),
            float(ekin[n - 1]), mass_au, gap_au)
        if prob > 0:
            return float(prob)
    return 0.0


def _numerical_adiabatic_gradients(x_bohr, energies):
    """Differentiate the recorded state energies into gradients.

    The Tully models carry no closed-form adiabatic gradient
    interface through the qdm model classes, so the replay
    differentiates the recorded state energies against the recorded
    coordinates numerically (central differences in the interior,
    one-sided at both ends).

    Parameters
    ----------
    x_bohr : numpy.ndarray
        Per-step coordinate (Bohr).
    energies : numpy.ndarray
        Per-step adiabatic state energies (nstep, nstate) in
        Hartree.

    Returns
    -------
    numpy.ndarray
        Per-step dE/dx values (nstep, nstate) in Hartree/Bohr.
    """
    nstep = len(x_bohr)
    grads = np.zeros_like(energies)
    for n in range(nstep):
        if 0 < n < nstep - 1:
            dx = x_bohr[n + 1] - x_bohr[n - 1]
            if dx == 0:
                continue
            grads[n] = (energies[n + 1] - energies[n - 1]) / dx
        else:
            m = 1 if n == 0 else nstep - 2
            dx = x_bohr[m + 1] - x_bohr[m]
            if dx == 0:
                continue
            grads[n] = (energies[m + 1] - energies[m]) / dx
    return grads


def _tully_kernel_probability(run_dir, prod):
    """Deterministic replay of the ZN kernel on a tully case.

    Mirrors the lz replay on the first recorded trajectory, but the
    adiabatic gradients are numerically differentiated from the
    recorded state energies (the Tully models carry no closed-form
    adiabatic gradient interface); the first non-zero probability
    is the gate value the hop draw samples from (the reported
    internal hopping probability).

    Parameters
    ----------
    run_dir : str
        Benchmark run directory holding production/.
    prod : dict
        Production section of the tully case definition.

    Returns
    -------
    float
        The gate-step ZN hopping probability (0.0 when the replay
        never passes the gate).
    """
    prod_dir = os.path.join(run_dir, 'production')
    title = 'benchmark_%s' % str(prod.get('_case_name', 'gsh'))
    traj_dirs = sorted(
        d for d in os.listdir(prod_dir)
        if os.path.isdir(os.path.join(prod_dir, d))
        and d.startswith(title + '_'))
    if not traj_dirs:
        return 0.0
    x_bohr, ekin, energies = _parse_lz_trajectory(
        os.path.join(prod_dir, traj_dirs[0]), title)
    mass = float(prod.get('model_mass', 2000.0))
    gap_au = float(prod.get('gap', 0.5)) * EV_TO_HARTREE
    mass_au = np.array([mass])
    grads = _numerical_adiabatic_gradients(x_bohr, energies)
    nstep = len(x_bohr)
    for n in range(2, nstep):
        grad_n = np.zeros((2, 1, 3))
        grad_n[0, 0, 0], grad_n[1, 0, 0] = grads[n]
        grad_pp = np.zeros((2, 1, 3))
        grad_pp[0, 0, 0], grad_pp[1, 0, 0] = grads[n - 2]
        prob, _ = internal_conversion_prob(
            1, 0,
            energies[n], energies[n - 1], energies[n - 2],
            grad_n, grad_pp,
            np.array([[x_bohr[n] * BOHR_TO_ANGSTROM, 0.0, 0.0]]),
            np.array([[x_bohr[n - 1] * BOHR_TO_ANGSTROM, 0.0, 0.0]]),
            np.array([[x_bohr[n - 2] * BOHR_TO_ANGSTROM, 0.0, 0.0]]),
            float(ekin[n - 1]), mass_au, gap_au)
        if prob > 0:
            return float(prob)
    return 0.0


def evaluate_run(run_dir):
    """Evaluate one computed gsh benchmark run directory.

    Delegates to the shared evaluate_unified final-state metric
    set: the final trajectory / state / reference populations plus
    the final-state differences judged against the case tolerances.
    Every case carries the deterministic ZN kernel replay as the
    reported internal hopping probability; the lz family
    (lz1/lz2/lz3) additionally compares against the analytic
    Landau-Zener probability (see the lz calibration block above).

    Parameters
    ----------
    run_dir : str
        Benchmark run directory holding the case YAML, reference/
        and production/.

    Returns
    -------
    dict
        Report payload (case, metrics, tolerances, verdict,
        runtime_seconds, details).
    """
    try:
        case_path = _find_case_yaml(run_dir)
        with open(case_path, 'r', encoding='utf-8') as f:
            case = yaml.safe_load(f)
        prod = case['production']
        lz_case = str(prod.get('method', 'tully1')).lower() in LZ_METHODS

        ref_dir = os.path.join(run_dir, 'reference')
        prod_dir = os.path.join(run_dir, 'production')
        time_grid, ref_population = parse_reference_population(
            ref_dir, int(prod['nstep']))
        ref_pop = np.asarray(ref_population, dtype=float)
        populations = collect_populations(prod_dir, ref_pop.shape[1])
        ensemble, _active_ensemble = ensemble_average(
            populations, ref_pop.shape[0])
    except Exception as exc:  # noqa: BLE001 - report, never crash silent
        return {
            'case': os.path.basename(os.path.abspath(run_dir)),
            'metrics': {},
            'tolerances': {},
            'verdict': 'ERROR',
            'runtime_seconds': 0,
            'details': {'error': str(exc)},
        }

    # The kernel replay locates the trajectory directories through
    # the case-name title prefix injected here.
    prod['_case_name'] = str(case.get('case', 'gsh'))
    try:
        return evaluate_unified(
            run_dir, case, populations, ensemble, ref_pop, time_grid,
            internal_probability_fn=(_lz_kernel_probability
                                     if lz_case
                                     else _tully_kernel_probability))
    finally:
        del prod['_case_name']
