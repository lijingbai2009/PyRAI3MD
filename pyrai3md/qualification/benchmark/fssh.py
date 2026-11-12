"""
======================================================================
FSSH benchmark task module (flat benchmark layout).

One module per benchmark task type: fssh.py serves the unified
fssh.yaml case definition shipped next to it. The model
(tully1/2/3, lz1/2/3) is selected dynamically through the
production.method key of the case, so calibrating another model
only changes the method of the pmd_env benchmark entries -- no new
YAML file per model (a dedicated fssh_<method>.yaml would take
precedence for one specific method; the lz calibrations ship such
dedicated cases with the analytic Landau-Zener verdict). The
module owns the complete
calibration task -- input generation and result analysis -- while
the execution of the reference and production calculations is
delegated to the qualification planner's generated chain script
(both inputs live in the benchmark run directory and both run
through the real pyrai3md entry point):

1. build_reference_input: write the framework YAML for the exact
   split-operator quantum propagation (control driver=skill +
   jobtype md selects the skill's public {method}_reference
   profile through SingleShot; qdm_project 'reference' names the
   output files) from the SAME initial coordinate and velocity as
   the production trajectory. The reference is generated
   independently at benchmark time (no external or committed
   dataset);
2. build_production_input: write the fssh md workflow YAML from the
   fixed initial conditions of the case definition;
3. evaluate_run: parse the reference {project}_population.dat and
   the trajectory <title>.md.pop records written by the compute
   script, ensemble-average BOTH population column blocks of every
   trajectory -- the electronic state populations (the density
   matrix diagonal) and the 0/1 active state populations --
   separately (ninitcond = 100 in the standard benchmark), then
   report the unified final-state metric set through
   evaluate_unified: the final trajectory / state / reference
   populations plus the judged differences against the wavepacket
   reference, the internal kernel replay probability (gsh/lzsh)
   and the analytic Landau-Zener probability (lz model cases) --
   whichever comparison targets the case carries; the probability
   quantities themselves are reported without a tolerance; save the
   averaged production populations and the reference populations as
   production.pop / production.act.pop / reference.pop into the
   run directory and render their S0 population curves as
   population_s0.png (diagnostic only).

The planner chains both calculations into a single script
(cd reference && pyrai3md <run_dir>/reference.yaml &&
 cd production && pyrai3md <run_dir>/production.yaml)
so the production run can only start after the reference
succeeded, identically on local and slurm.

Evaluation is orchestrated by the qualification runner, which
calls evaluate_run in-process right after the compute task
finished; no separate evaluate subprocess is planned by the
framework. The task module itself is programmatic only: the
planner imports the input builders and the qualification runner
imports evaluate_run. The CLI surface lives in the public wrapper
tools/benchmark.py (external tools directory, next to deploy.py),
which owns the command-line entry point:

Usage:
    python3 ./tools/benchmark.py <run_dir>
        # evaluate an already-computed case directory

Verdict: PASS when every metric meets its tolerance, FAIL when a
metric misses, ERROR when a compute output is missing or broken.

Per-task settings of the pmd_env.yaml benchmark section (the fssh
subsection: a single parameter dict or a list of them, each entry
carrying method, ninitcond, nstep, dt, nactype, x0, v0, sigma,
model_mass, initial_state, ngrid, substeps, ...) are merged on top
of the case YAML by apply_overrides before any input is built, so
the reference and the production run always share the environment
parameters.

Aug 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import copy
import json
import os
import time

import numpy as np
import yaml

from .plotting import plot_population_s0, plot_title


# The md input carries Angstrom coordinates while the case definition
# works in Bohr; the conversion is the framework's one definition.
from ...utilities.units import BOHR_TO_ANGSTROM

# Kept identical to pyrai3md.utilities.units: the md input carries
# amu masses while the case definition works in atomic units.
AMU_TO_AU = 1822.8884871474306

# Landau-Zener calibration family: the lz models carry the
# analytic single-passage Landau-Zener probability as an extra
# comparison target of the unified final-state metric set (the
# steep linear slopes make the exact quantum packet re-cross the
# interaction region on a period shorter than the observation
# window, so the wavepacket comparison alone is not the acceptance
# target of the single-passage hopping model).
LZ_METHODS = ('lz1', 'lz2', 'lz3')


# Case production keywords forwarded into the md section of the
# production input: case keyword -> (md keyword, type converter,
# builder default). The table spans the surface hopping and
# propagation blocks of the md schema (schema/config.py); the
# fssh_* kernel-selection keywords are present-only forwards (the
# perf benchmark case uses them to pin the numpy/numba backend per
# function; the fssh calibration case never sets them) and the
# capability triggers (dosoc/addsoc/dodipole/radiative) are
# deliberately excluded -- the unrecognized-key check below fails
# loudly when they are requested instead of forwarding them
# silently. A None default means "present-only" forwarding: an
# absent case key keeps the md-schema default so the schema stays
# the single source; a callable default is written even when the
# case does not set the key (the benchmark defines these semantics
# itself: sfhp selects the algorithm under calibration and the
# default nactype 'ktdc' here differs from the schema default).
PRODUCTION_MD_KEYWORDS = {
    'sfhp': ('sfhp', str, 'fssh'),
    'nactype': ('nactype', lambda value: str(value).lower(), 'ktdc'),
    'phasecheck': ('phasecheck', int, None),
    'gap': ('gap', float, None),
    'gapsoc': ('gapsoc', float, None),
    'substep': ('substep', int, None),
    'rk4': ('rk4', str, None),
    'integrate': ('integrate', int, None),
    'deco': ('deco', str, None),
    'adjust': ('adjust', int, None),
    'reflect': ('reflect', int, None),
    'maxh': ('maxh', int, None),
    'thermostat': ('thermostat', str, None),
    'thermodelay': ('thermodelay', int, None),
    # FSSH kernel implementation selection and profiling switch
    # (present-only: only the perf benchmark case sets them).
    'fssh_dpdt': ('fssh_dpdt', str, None),
    'fssh_rk4': ('fssh_rk4', str, None),
    'fssh_matb': ('fssh_matb', str, None),
    'fssh_ktdc': ('fssh_ktdc', str, None),
    'fssh_decoherence': ('fssh_decoherence', str, None),
    'fssh_profile': ('fssh_profile', int, None),
}

# Production keywords consumed by the two input builders outside
# the md forwarding table (structural parameters and the wavepacket
# setup; sigma feeds the reference wavepacket only because the
# production run reuses the supplied geometry, initcond '0').
PRODUCTION_STRUCTURAL_KEYS = (
    'method', 'ninitcond', 'ncpus', 'nstep', 'dt', 'x0', 'v0',
    'sigma', 'model_mass', 'initial_state',
)

# Model parameter overrides forwarded into the qdm section of the
# production input (present-only, None keeps the model class
# defaults). The reference builder forwards the same keys, so both
# legs evaluate the identical parametrized model (the lz1/lz2/lz3
# calibrations). This is the one definition: gsh, lzsh and ktdc import
# it from here rather than restating the same two keys.
PRODUCTION_QDM_KEYWORDS = {
    'lz_a': float,
    'lz_b': float,
}


def load_case(case_path):
    """Load one fssh calibration case definition.

    Parameters
    ----------
    case_path : str
        Path of the fssh case file (the unified fssh.yaml or a
        dedicated fssh_<method>.yaml).

    Returns
    -------
    dict
        Parsed case definition.
    """
    with open(case_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# pmd_env keys routed into the reference subsection of the case
# definition; every other override lands in the production
# subsection (the 'enabled' planner switch is dropped here).
REFERENCE_OVERRIDE_KEYS = ('ngrid', 'substeps')


def apply_overrides(case, overrides):
    """Merge pmd_env.yaml benchmark parameters into a case.

    The benchmark subsection of pmd_env.yaml carries the physical
    parameters of the calibration; they take priority over the
    shipped case YAML so the same case file serves every
    environment. Keys in REFERENCE_OVERRIDE_KEYS land in the
    reference subsection, every other key (method, ninitcond,
    nstep, dt, sfhp, nactype, x0, v0, sigma, model_mass,
    initial_state, ...) lands in the production subsection; the
    'enabled' planner switch is not a case parameter and is
    dropped.

    Parameters
    ----------
    case : dict
        Parsed case definition (unchanged).
    overrides : dict
        The benchmark.<task> subsection of pmd_env.yaml.

    Returns
    -------
    dict
        A new case definition with the overrides merged in.
    """
    merged = copy.deepcopy(case)
    production = merged.setdefault('production', {})
    reference = merged.setdefault('reference', {})
    for key, value in (overrides or {}).items():
        if key == 'enabled' or value is None:
            continue
        if key in REFERENCE_OVERRIDE_KEYS:
            reference[key] = value
        else:
            production[key] = value
    return merged


def build_reference_input(case, run_dir):
    """Write the framework YAML input for the reference propagation.

    The input runs through the regular pyrai3md entry point:
    control driver=skill + jobtype md routes the calculation to
    SingleShot, and the qdm skill resolves its public
    {method}_reference profile from that pair. The qdm section
    carries the wavepacket and propagation keywords plus
    qdm_project 'reference', so the skill writes
    reference_population.dat (and friends) into its calculation
    directory. The initial coordinate x0 and velocity v0 are
    identical to the production fssh trajectory by construction.

    Parameters
    ----------
    case : dict
        Parsed case definition.
    run_dir : str
        Benchmark run directory holding the generated
        reference.yaml (created here when missing).

    Returns
    -------
    str
        Path of the generated reference.yaml input.
    """
    prod = case['production']
    ref = case['reference']

    method = str(prod.get('method', 'tully1'))
    nstep = int(prod['nstep'])
    dt = float(prod['dt'])
    substeps = max(1, int(ref.get('substeps', 1)))

    x0 = float(prod['x0'])
    v0 = float(prod.get('v0', 0.0))
    sigma = float(prod.get('sigma', 0.5))
    mass = float(prod.get('model_mass', 2000.0))
    initial_state = int(prod.get('initial_state', 0))

    os.makedirs(run_dir, exist_ok=True)

    workflow = {
        'control': {
            'title': 'benchmark_reference_%s' % method,
            'jobtype': 'md',
            'theory': ['qdm'],
            # driver != main routes to SingleShot; the qdm skill
            # then resolves {method}_reference from (md, skill).
            'driver': 'skill',
        },
        'atoms': ['X'],
        # Framework input coordinates are Angstrom; the skill
        # converts to the Bohr model coordinate internally.
        'coordinates': [[x0 * BOHR_TO_ANGSTROM, 0.0, 0.0]],
        'velocities': [[v0, 0.0, 0.0]],
        'mass': [mass / AMU_TO_AU],
        'molecule': {
            'charges': 0,
            'multiplicity': 1,
            'ci': [2],
            'spin': [0],
        },
        # No framework-level execution switch: the retired
        # &environment keys are refused (schema/config
        # .RETIRE_ENVIRONMENT_KEYS). The qdm skill section below
        # declares its own execution_mode, which is the live source.
        'environment': {},
        'qdm': {
            'method': method,
            'model_mass': mass,
            'wavepacket_sigma': sigma,
            'current_state': initial_state,
            'target_states': [0, 1],
            # Substepping lands the saved snapshots on the md grid.
            'ref_ngrid': int(ref.get('ngrid', 1024)),
            'ref_dt': dt / substeps,
            'ref_nsteps': nstep * substeps,
            'ref_save_every': substeps,
            # Output prefix: reference_population.dat and friends.
            'qdm_project': 'reference',
            'execution_mode': 'local',
            # The verdict reads this record back after the call, and the
            # qdm backend discards its calculation directory per call
            # when keep_tmp is 0 (#307). Without this the reference leg
            # finishes green and the evaluator then finds no file at all.
            'keep_tmp': 1,
        },
    }

    # Model parameter overrides (present-only): the production
    # builder forwards the same keys, so both legs evaluate the
    # identical parametrized model (e.g. the lz1 calibration).
    for key in ('lz_a', 'lz_b'):
        if key in prod:
            workflow['qdm'][key] = float(prod[key])

    input_path = os.path.join(run_dir, 'reference.yaml')
    with open(input_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(workflow, f, sort_keys=False)
    return input_path


def build_production_input(case, run_dir):
    """Write the fssh md input YAML for the production trajectory.

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
    case_name = str(case.get('case', 'fssh'))
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
            # `nactype: nac` builds the hop rate from this pair's derivative
            # coupling, so the pair has to be declared: a pair that is not
            # declared is shielded, and the adjacency default that used to
            # fill it in silently is gone (docs MULTICENTER 4.4.5).
            'coupling': [[0, 1]],
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

    # Forward the case's md keywords through the explicit table
    # (sfhp/nactype carry their builder defaults; the rest are
    # present-only, e.g. the shipped case pins deco 'off' and
    # rk4 '1').
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
    # parameter the execution input never carries.
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


def _lz_analytic_probability(prod):
    """Analytic single-passage Landau-Zener probability of the case.

    Delegates to the qdm lz model module selected by the production
    method (lz1/lz2/lz3): each model class owns its analytic
    Landau-Zener physics, the crossing-point velocity follows from
    energy conservation on its lower adiabatic surface and P_LZ is
    exp(-pi*lz_b^2 / (lz_a*v_c)) for lz1 (slope difference 2*lz_a)
    and exp(-2*pi*lz_b^2 / (lz_a*v_c)) for the half-linear lz2/lz3
    models (slope difference lz_a). Shared by the fssh and gsh lz
    calibration verdicts.

    Parameters
    ----------
    prod : dict
        Production section of the lz case definition (method,
        lz_a/lz_b, model_mass, x0, v0).

    Returns
    -------
    (float, float)
        (P_LZ, v_c): the analytic hopping probability and the
        crossing-point velocity (Bohr/au).
    """
    from ...skills.qdm_skill.service.models import get_model
    method = str(prod.get('method', 'lz1')).lower()
    model = get_model(method, params={'lz_a': prod.get('lz_a'),
                                      'lz_b': prod.get('lz_b')})
    mass = float(prod.get('model_mass', 2000.0))
    v_c = model.crossing_velocity(mass, float(prod['x0']),
                                  float(prod.get('v0', 0.0)))
    return model.landau_zener_probability(v_c), v_c


def _find_case_yaml(run_dir):
    """Locate the case YAML inside a benchmark run directory.

    The run directory also holds reference.yaml and production.yaml
    (the compute inputs); the case YAML is any other YAML file that
    carries a 'production' section.

    Parameters
    ----------
    run_dir : str
        Benchmark run directory.

    Returns
    -------
    str
        Path of the case YAML.

    Raises
    ------
    FileNotFoundError
        When no case YAML exists in the run directory.
    """
    for fname in sorted(os.listdir(run_dir)):
        if not fname.endswith(('.yaml', '.yml')):
            continue
        if fname in ('reference.yaml', 'production.yaml'):
            continue
        path = os.path.join(run_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict) and 'production' in data:
            return path
    raise FileNotFoundError(
        'no benchmark case YAML found in %s' % run_dir)


def parse_reference_population(ref_dir, nstep):
    """Collect the population record written by the reference run.

    The skill reference profile saves {project}_population.dat in
    its calculation directory (reference/tmp_QDM in the benchmark
    layout): columns [time, pop_state0, ...] with snapshot 0 at
    t = 0 (the md pop records start at step 1). The recursive scan
    keeps the parser independent of the exact directory nesting.

    Parameters
    ----------
    ref_dir : str
        Reference directory to scan.
    nstep : int
        Expected number of saved md-grid snapshots.

    Returns
    -------
    tuple(list, list)
        (time grid, per-step populations) on the md grid j * dt.

    Raises
    ------
    FileNotFoundError / RuntimeError
        When the record is missing or its length disagrees with the
        production plan.
    """
    pop_files = []
    for root, _dirs, files in os.walk(ref_dir):
        for fname in files:
            if fname.endswith('_population.dat'):
                pop_files.append(os.path.join(root, fname))
    if not pop_files:
        raise FileNotFoundError(
            'no *_population.dat reference record found under %s'
            % ref_dir)

    record = np.loadtxt(sorted(pop_files)[0])
    time_grid = record[1:, 0].tolist()
    population = record[1:, 1:].tolist()
    if len(population) != nstep:
        raise RuntimeError(
            'reference saved %d snapshots but the production plan '
            'records %d md steps' % (len(population), nstep))
    return time_grid, population


def parse_pop(path):
    """Parse one <title>.md.pop population record.

    The file starts with a header line ('time', 'Pop1,2,3...',
    'Act1,2,3...') followed by one numeric row per recorded step:
    the time column, the per-state electronic populations and the
    0/1 active state columns (legacy rows carry the populations
    only).

    Parameters
    ----------
    path : str
        Path of the pop file.

    Returns
    -------
    numpy.ndarray
        Array of shape (nsteps, ncols): the time column followed
        by the population column blocks.
    """
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            try:
                values = [float(token) for token in tokens]
            except ValueError:
                continue  # header or malformed line
            rows.append(values)
    if not rows:
        raise ValueError('no data rows found in %s' % path)
    return np.asarray(rows, dtype=float)


def _split_population_record(record, nstate):
    """Split one md.pop record into electronic and active blocks.

    The writer appends the electronic state populations followed by
    the 0/1 active state population columns (time + Pop1..N +
    Act1..N); legacy records carry the time column plus the
    electronic populations only and degrade to an absent active
    block. The state count comes from the reference record, so the
    split never relies on guessing the row width.

    Parameters
    ----------
    record : numpy.ndarray
        Full numeric pop table of shape (nsteps, ncols).
    nstate : int
        Number of electronic states (from the reference record).

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray or None)
        (electronic record, active record), each of shape
        (nsteps, 1 + nstate) with the time column kept; the active
        record is None for legacy (population-only) files.

    Raises
    ------
    RuntimeError
        When the row width matches neither the legacy nor the
        extended population format.
    """
    ncols = record.shape[1]
    if ncols == 1 + 2 * nstate:
        time_col = record[:, :1]
        population = np.hstack([time_col, record[:, 1:1 + nstate]])
        active = np.hstack([time_col, record[:, 1 + nstate:]])
        return population, active
    if ncols == 1 + nstate:
        # Legacy 1 + nstate columns: electronic populations only.
        return record, None
    raise RuntimeError(
        'md.pop rows carry %d columns but the reference has %d '
        'states (expected %d or %d columns)' % (
            ncols, nstate, 1 + nstate, 1 + 2 * nstate))


def collect_populations(prod_dir, nstate):
    """Collect the trajectory md.pop records under the prod dir.

    Walks the production directory because the md workflow isolates
    dispatched trajectories in '{title}_{idx}' working directories
    (each carrying its md.pop record and tmp_QDM calculation
    directory); a single in-place trajectory writes directly into
    prod_dir.

    Parameters
    ----------
    prod_dir : str
        Production directory to scan.
    nstate : int
        Number of electronic states (from the reference record).

    Returns
    -------
    list of tuple(numpy.ndarray, numpy.ndarray or None)
        One (electronic record, active record) pair per trajectory,
        each record of shape (nsteps, 1 + nstate), sorted by path
        for determinism; the active record is None for legacy
        population-only files.

    Raises
    ------
    FileNotFoundError
        When no md.pop record exists (production produced nothing).
    """
    pop_files = []
    for root, _dirs, files in os.walk(prod_dir):
        for fname in files:
            if fname.endswith('.md.pop'):
                pop_files.append(os.path.join(root, fname))
    if not pop_files:
        raise FileNotFoundError(
            'no *.md.pop trajectory record found under %s' % prod_dir)
    return [_split_population_record(parse_pop(path), nstate)
            for path in sorted(pop_files)]


def ensemble_average(populations, nstep_ref):
    """Ensemble-average the electronic and active population blocks.

    Parameters
    ----------
    populations : list of tuple
        Trajectory (electronic record, active record) pairs as
        returned by collect_populations; the time column is dropped
        before averaging.
    nstep_ref : int
        Expected number of recorded steps.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray or None)
        (electronic ensemble, active ensemble), each of shape
        (nsteps, nstate); the active ensemble is None when every
        trajectory record lacks the active columns (legacy data).

    Raises
    ------
    ValueError
        When a trajectory record has the wrong length.
    """
    series, active_series = [], []
    for record, active in populations:
        pop = record[:, 1:]
        if pop.shape[0] != nstep_ref:
            raise ValueError(
                'trajectory records %d steps but the reference '
                'covers %d' % (pop.shape[0], nstep_ref))
        series.append(pop)
        if active is not None:
            active_series.append(active[:, 1:])
    ensemble = np.mean(np.stack(series, axis=0), axis=0)
    if not active_series:
        return ensemble, None
    return ensemble, np.mean(np.stack(active_series, axis=0), axis=0)


def trajectory_hop_fraction(populations):
    """The final-state-1 fraction of the recorded trajectories.

    The trajectory-statistics hopping probability of an
    identical-problem ensemble: every trajectory's final 0/1 active
    state indicator is averaged, so a trajectory that ends on state
    1 counts as one hop outcome regardless of the coherent density
    it may carry along the way (the pure-state algorithms gsh/lzsh
    keep the electronic and the active block identical per
    trajectory by construction).

    Parameters
    ----------
    populations : list of tuple
        Trajectory (electronic record, active record) pairs as
        returned by collect_populations.

    Returns
    -------
    float or None
        The fraction of trajectories whose final active state is
        state 1; None when every record lacks the active columns
        (legacy population-only data).
    """
    finals = [active[-1, -1] for _record, active in populations
              if active is not None]
    if not finals:
        return None
    return float(np.mean(finals))


def _save_population(path, time_grid, population):
    """Write one population table (time column + state columns).

    Parameters
    ----------
    path : str
        Output file path (production.pop / reference.pop).
    time_grid : sequence of float
        Time values of the md grid.
    population : numpy.ndarray
        Population values of shape (nstep, nstate).
    """
    data = np.column_stack(
        [np.asarray(time_grid, dtype=float),
         np.asarray(population, dtype=float)])
    header = 'time ' + ' '.join(
        'pop_state%d' % i for i in range(data.shape[1] - 1))
    np.savetxt(path, data, fmt='%20.12f', header=header)


def evaluate_unified(run_dir, case, populations, ensemble, ref_pop,
                     time_grid, internal_probability_fn=None):
    """Unified final-state evaluation shared by every benchmark task.

    Every case reports the same five physical quantities and the
    three final-state differences, skipping whatever the case does
    not carry:

    Quantities (reported values):
      final_trajectory_population  the fraction of trajectories
          whose final active state is state 1;
      final_state_population       the final S1 population of the
          ensemble-averaged electronic block;
      final_reference_population   the final S1 probability of the
          exact wavepacket reference;
      internal_hopping_probability the deterministic kernel replay
          probability (gsh/lzsh cases only, supplied through
          internal_probability_fn); reported without a tolerance;
      analytical_hopping_probability the analytic single-passage
          Landau-Zener probability (lz model cases only).

    Judged differences (each against its case YAML tolerance):
      state_reference_difference   |state - reference|, every case;
      state_internal_difference    |state - internal|, only when
          internal_probability_fn supplies a value (gsh/lzsh);
      state_analytical_difference  |state - analytical|, lz cases.

    Parameters
    ----------
    run_dir : str
        Benchmark run directory (population tables and the plot are
        written here).
    case : dict
        Parsed case definition (production and metrics sections).
    populations : list of tuple
        Trajectory (electronic record, active record) pairs as
        returned by collect_populations.
    ensemble : numpy.ndarray
        Ensemble-averaged electronic populations (nstep, nstate).
    ref_pop : numpy.ndarray
        Reference populations (nstep, nstate).
    time_grid : sequence of float
        Time values of the md grid.
    internal_probability_fn : callable or None
        fn(run_dir, prod) -> float; the deterministic kernel replay
        of the gsh/lzsh cases. None skips the internal quantity
        and its difference (fssh/ktdc).

    Returns
    -------
    dict
        Report payload (case, metrics, tolerances, verdict,
        runtime_seconds, details).
    """
    started = time.time()
    try:
        case_name = str(case.get('case', 'benchmark'))
        prod = case['production']
        metrics_spec = case.get('metrics', {})
        tolerances = {}
        for name, spec in metrics_spec.items():
            tolerance = spec.get('tolerance')
            if tolerance is None:
                # A missing tolerance used to read as infinity, which
                # certifies every value including a wrong one.
                raise ValueError(
                    "benchmark case %r declares no tolerance for %r"
                    % (case_name, name))
            tolerances[name] = float(tolerance)

        p_traj = trajectory_hop_fraction(populations)
        p_state = float(ensemble[-1, 1])
        p_ref = float(ref_pop[-1, 1])

        metrics = {
            'final_trajectory_population': p_traj,
            'final_state_population': p_state,
            'final_reference_population': p_ref,
        }
        differences = {
            'state_reference_difference': abs(p_state - p_ref),
        }

        # Internal hopping probability: the deterministic kernel
        # replay of the gsh/lzsh cases (absent for fssh/ktdc).
        if internal_probability_fn is not None:
            p_internal = float(internal_probability_fn(run_dir, prod))
            metrics['internal_hopping_probability'] = p_internal
            differences['state_internal_difference'] = \
                abs(p_state - p_internal)

        # Analytic Landau-Zener probability: the lz model cases only.
        p_lz = v_c = None
        lz_case = str(prod.get('method', 'tully1')).lower() in LZ_METHODS
        if lz_case:
            p_lz, v_c = _lz_analytic_probability(prod)
            metrics['analytical_hopping_probability'] = p_lz
            differences['state_analytical_difference'] = \
                abs(p_state - p_lz)

        metrics.update(differences)
        verdicts = {name: value <= tolerances.get(name, np.inf)
                    for name, value in differences.items()}
        verdict = 'PASS' if all(verdicts.values()) else 'FAIL'

        # Persist the compared population tables into the run
        # directory for later inspection.
        prod_pop_path = os.path.join(run_dir, 'production.pop')
        ref_pop_path = os.path.join(run_dir, 'reference.pop')
        saved = [prod_pop_path, ref_pop_path]
        _save_population(prod_pop_path, time_grid, ensemble)
        _save_population(ref_pop_path, time_grid, ref_pop)
        active_ensemble = None
        active_series = [active[:, 1:] for _record, active in populations
                         if active is not None]
        if active_series:
            active_ensemble = np.mean(
                np.stack(active_series, axis=0), axis=0)
            prod_act_path = os.path.join(run_dir, 'production.act.pop')
            _save_population(prod_act_path, time_grid, active_ensemble)
            saved.append(prod_act_path)

        details = {
            'trajectories': len(populations),
            'nstep': int(prod['nstep']),
            'dt': float(prod['dt']),
            'per_metric': {name: bool(ok)
                           for name, ok in verdicts.items()},
            'saved': saved,
        }
        if lz_case:
            details['crossing_velocity'] = v_c
        # Render the S0 population plot of the compared tables
        # right before the report returns (diagnostic only: a
        # skipped plot never affects the verdict).
        details['population_plot'] = plot_population_s0(
            run_dir, plot_title(prod))
    except Exception as exc:  # noqa: BLE001 - report, never crash silent
        case_name = os.path.basename(os.path.abspath(run_dir))
        metrics = {}
        tolerances = {}
        verdict = 'ERROR'
        details = {'error': str(exc)}

    return {
        'case': case_name,
        'metrics': metrics,
        'tolerances': tolerances,
        'verdict': verdict,
        'runtime_seconds': round(time.time() - started, 2),
        'details': details,
    }


def evaluate_run(run_dir):
    """Evaluate one computed benchmark run directory.

    Reads the outputs written by the qualification compute script
    (reference/ and production/ next to the case YAML), ensemble-
    averages the trajectory population records and delegates to the
    shared evaluate_unified final-state metric set: the final
    trajectory / state / reference populations plus the final-state
    differences judged against the case tolerances. FSSH carries no
    kernel replay, so the internal hopping probability and its
    difference are absent on every case; the lz model cases
    additionally compare against the analytic single-passage
    Landau-Zener probability (see LZ_METHODS). A missing or broken
    compute output yields verdict ERROR instead of crashing silent.
    The qualification runner calls this in-process right after the
    compute task finished (evaluation is a qualification tool,
    never a framework calculation task).

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

    return evaluate_unified(run_dir, case, populations, ensemble,
                            ref_pop, time_grid,
                            internal_probability_fn=None)


def emit(report):
    """Print the machine-readable report on the last stdout line.

    Parameters
    ----------
    report : dict
        Report payload (case, metrics, tolerances, verdict,
        runtime_seconds, details).

    Returns
    -------
    int
        Process exit code (0 PASS, 1 FAIL, 2 ERROR).
    """
    print('BENCHMARK_JSON: %s' % json.dumps(report, sort_keys=True),
          flush=True)
    return {'PASS': 0, 'FAIL': 1}.get(report['verdict'], 2)
