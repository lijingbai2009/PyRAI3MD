"""
======================================================================
LZSH benchmark task module (flat benchmark layout).

One module per benchmark task type: lzsh.py serves the unified
lzsh.yaml case definition shipped next to it. The model
(tully1/2/3) is selected dynamically through the
production.method key of the case; the dedicated lzsh_lz1/2/3.yaml
case files serve the Landau-Zener crossing family (a dedicated
<task>_<method>.yaml takes precedence for one specific method).

The LZSH (Landau-Zener surface hopping) calibration reuses the
fssh calibration infrastructure wherever the semantics are
algorithm-agnostic:

- load_case / apply_overrides: the case YAML layout and the
  pmd_env override merging are identical (REFERENCE_OVERRIDE_KEYS
  routing included);
- build_reference_input: the exact split-operator quantum
  reference depends on the model and the wavepacket setup only,
  never on the hopping algorithm under calibration;
- evaluate_run: delegates to the shared evaluate_unified
  final-state metric set (the final trajectory / state / reference
  populations plus the final-state differences); every case
  supplies the deterministic LZSH kernel replay as the reported
  internal hopping probability (the lz family additionally compares
  against the analytic Landau-Zener probability).

LZSH-specific parts live here: the production md keyword forwarding
table spans the LZSH-relevant md schema keywords only (sfhp, the
gap thresholds, the velocity adjustment policy and maxh), the
production input builder writes the lzsh md workflow YAML and
forwards the model parameter overrides (lz_a/lz_b) into the qdm
section when present. The fssh-only keywords (nactype, substep,
rk4, integrate, deco and the fssh_* kernel selectors) are
deliberately absent -- the unrecognized-key check fails loudly when
they are requested instead of forwarding them silently.

LZSH kernels consume the state energies and the timestep only: no
gradients (no activestate requirement) and no NAC vectors. The
deterministic kernel replay of the lz evaluation therefore runs on
the recorded energy series alone.

Sep 2 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os

import numpy as np
import yaml

from ...utilities.units import EV_TO_HARTREE
from ...workflow.integrator.lzsh import internal_conversion_prob
from . import lz_data
from .fssh import (
    PRODUCTION_QDM_KEYWORDS,
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
# LZSH production input: case keyword -> (md keyword, type
# converter, builder default). The table spans the LZSH-relevant
# surface hopping block of the md schema (schema/config.py); a
# None default means "present-only" forwarding (an absent case key
# keeps the md-schema default so the schema stays the single
# source); the callable sfhp default is written even when the case
# does not set the key (the benchmark defines these semantics
# itself: sfhp selects the algorithm under calibration). The
# kernels consume no gradients, so the activestate escape hatch of
# the gsh table is absent on purpose.
PRODUCTION_MD_KEYWORDS = {
    'sfhp': ('sfhp', str, 'lzsh'),
    'gap': ('gap', float, None),
    'gapsoc': ('gapsoc', float, None),
    'adjust': ('adjust', int, None),
    'reflect': ('reflect', int, None),
    'maxh': ('maxh', int, None),
}


def build_production_input(case, run_dir):
    """Write the lzsh md input YAML for the production trajectory.

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
    case_name = str(case.get('case', 'lzsh'))
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

    # Forward the case's md keywords through the explicit LZSH table
    # (sfhp carries its builder default; the rest are present-only,
    # e.g. the shipped lz cases pin the gap threshold).
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
# The lz cases calibrate the LZSH kernel against the single-passage
# Landau-Zener physics: the unified final-state metric set carries
# both comparison targets -- the deterministic LZSH kernel replay
# gate probability (internal_hopping_probability, re-evaluated on
# the recorded trajectory energies) and the analytic Landau-Zener
# probability (analytical_hopping_probability) -- next to the
# exact wavepacket reference. The steep linear slopes make the
# quantum packet re-cross the interaction region on a period
# shorter than the observation window (multi-passage oscillations
# the single-passage hopping model is not designed to reproduce),
# so the analytic comparison targets anchor the verdict.
#
# The tully cases carry no analytic hopping probability, but the
# deterministic LZSH kernel replay of the recorded trajectories
# still supplies the reported internal hopping probability (the
# maximum per-step probability of the replay window); the verdict
# rests on the wavepacket reference difference alone.


def _replay_lzsh_kernel(run_dir, prod):
    """Deterministic replay of the LZSH IC kernel on one trajectory.

    The production internal_conversion_prob kernel is re-evaluated
    step by step on the first recorded trajectory's energies and
    kinetic energies (the kernel consumes the energy series and the
    timestep only: no model gradients, no coordinates) and returns
    the per-step probability series of the replay window.

    Parameters
    ----------
    run_dir : str
        Benchmark run directory holding production/.
    prod : dict
        Production section of the case definition.

    Returns
    -------
    numpy.ndarray
        The per-step kernel probabilities of the replay window
        (empty when no trajectory directory exists).
    """
    prod_dir = os.path.join(run_dir, 'production')
    title = 'benchmark_%s' % str(prod.get('_case_name', 'lzsh_lz1'))
    traj_dirs = sorted(
        d for d in os.listdir(prod_dir)
        if os.path.isdir(os.path.join(prod_dir, d))
        and d.startswith(title + '_'))
    if not traj_dirs:
        return np.asarray([])
    ekin, energies = lz_data.read_energies(
        os.path.join(prod_dir, traj_dirs[0]), title)
    dt = float(prod['dt'])
    gap_au = float(prod.get('gap', 0.5)) * EV_TO_HARTREE
    nstep = len(ekin)
    probs = []
    for n in range(2, nstep):
        prob = internal_conversion_prob(
            1, 0,
            energies[n], energies[n - 1], energies[n - 2],
            float(ekin[n - 1]), dt, gap_au)
        probs.append(max(float(prob), 0.0))
    return np.asarray(probs)


def _gate_lzsh_kernel_probability(run_dir, prod):
    """The gate-step LZSH probability of an lz case.

    Every trajectory of an lz case shares identical initial
    conditions (initcond 0), so the first recorded trajectory
    carries the deterministic pre-hop record; the first non-zero
    probability of the replay series is the gate value the hop
    draw samples from (the internal hopping probability of the
    unified final-state metric set).
    """
    for prob in _replay_lzsh_kernel(run_dir, prod):
        if prob > 0:
            return float(prob)
    return 0.0


def _max_lzsh_kernel_probability(run_dir, prod):
    """The maximum replayed LZSH probability of a tully case.

    The tully surfaces carry no single-passage crossing of a fixed
    gate, so the reported internal hopping probability takes the
    maximum per-step probability of the replay window instead of a
    gate value (the analytic LZSH probability target the replay
    reaches at the strongest avoided-crossing snapshot).
    """
    probs = _replay_lzsh_kernel(run_dir, prod)
    if probs.size == 0:
        return 0.0
    return float(np.max(probs))


def evaluate_run(run_dir):
    """Evaluate one computed lzsh benchmark run directory.

    Delegates to the shared evaluate_unified final-state metric
    set: the final trajectory / state / reference populations plus
    the final-state differences judged against the case tolerances.
    Every case carries the deterministic LZSH kernel replay as the
    reported internal hopping probability (the lz family takes the
    first non-zero gate probability, the tully cases the maximum
    per-step probability of the replay window); the lz family
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
    prod['_case_name'] = str(case.get('case', 'lzsh'))
    try:
        return evaluate_unified(
            run_dir, case, populations, ensemble, ref_pop, time_grid,
            internal_probability_fn=(_gate_lzsh_kernel_probability
                                     if lz_case
                                     else _max_lzsh_kernel_probability))
    finally:
        del prod['_case_name']
