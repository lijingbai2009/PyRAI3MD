"""
======================================================================
Perf benchmark task module (flat benchmark layout).

One module per benchmark task type: perf.py measures the numpy vs
numba computational efficiency of the FSSH electronic propagator on
the same Tully model case the fssh.py calibration uses. The module
never runs an exact quantum reference; it executes the identical
production ensemble twice -- once with every FSSH kernel pinned to
the NumPy reference implementation and once with every kernel
pinned to the Numba implementation -- and compares the two legs:

1. build_legs: write one framework YAML input per backend leg
   (production_numpy.yaml / production_numba.yaml in run_dir/numpy
   and run_dir/numba). Both legs share the exact physical setup of
   the case and the control gl_seed, so every trajectory draws the
   identical random number stream in both legs and the two ensemble
   averages must agree to floating-point noise;
2. evaluate_run: read the leg.timing wall times written by the
   timing wrapper (benchmark/timing.py), ensemble-average the md
   pop records of both legs, measure the backend agreement and the
   wall-time speedup, best-effort parse the fssh_profile function
   tables from the captured leg stdout and persist perf.json plus
   the averaged population tables into the run directory.

Unlike the fssh task module this module does not expose the
reference/production builder pair; it implements the build_legs
contract instead (an ordered list of leg descriptors), which the
planner chains into the single compute script. Evaluation follows
the regular qualification route: the runner calls evaluate_run
in-process right after the compute task finished.

Numba is an optional dependency: when it is not installed the
input generation raises and the planner reports a diagnostic skip
task instead of planning an unrunnable comparison.

Verdict: PASS when the two backends agree within the case
tolerance (the physics of both implementations is identical), FAIL
when they disagree, ERROR when a leg output is missing or broken.
The wall-time speedup is reported, never judged.

Sep 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import copy
import json
import os
import re
import time

import numpy as np
import yaml

from . import fssh
from .fssh import load_case, apply_overrides  # noqa: F401 - contract

# Backend legs in execution order: the numpy reference runs first
# (its warm cache-independent timing anchors the comparison) and
# the numba leg follows.
BACKENDS = ('numpy', 'numba')

# Every FSSH kernel pinned to the leg backend (design document
# section 5: function-level implementation selection).
FSSH_KERNEL_KEYS = ('fssh_dpdt', 'fssh_rk4', 'fssh_matb',
                    'fssh_ktdc', 'fssh_decoherence')

# Canonical function rows of the fssh_profile summary table (the
# 'Total' row and the implementation table are not parsed); the
# function names carry digits (dPdt, RK4, matB, kTDC).
_PROFILE_ROW = re.compile(
    r'^\s*(?P<func>[A-Za-z][A-Za-z0-9 ]*?)\s+(?P<calls>\d+)\s+'
    r'(?P<seconds>[\d.]+)\s*s\s*$')


def _require_numba():
    """Fail the input generation when Numba is missing.

    The planner turns the exception into a diagnostic skip task, so
    an environment without the optional dependency never plans an
    unrunnable backend comparison.

    Raises
    ------
    RuntimeError
        When numba is not importable.
    """
    from ...workflow.integrator.fssh_numba import NUMBA_AVAILABLE
    if not NUMBA_AVAILABLE:
        raise RuntimeError(
            'numba is not installed; the perf benchmark needs the '
            'optional numba dependency to compare the backends')


def _build_backend_input(case, run_dir, backend):
    """Write the production input of one backend leg.

    Deep-copies the case, stamps the backend into the case name
    (the control title follows it: benchmark_perf_<backend>) and
    into every fssh_* kernel keyword, switches the function-level
    profiler on and forwards the whole production section through
    the regular fssh production builder, so the keyword table, the
    silent-drop guard and the md-section structure stay identical
    to the calibration case. The control gl_seed is written
    explicitly afterwards so both legs draw the identical
    per-trajectory random stream by construction.

    Parameters
    ----------
    case : dict
        Parsed case definition.
    run_dir : str
        Benchmark run directory; the leg input is written into
        run_dir/<backend>/production.yaml.
    backend : str
        'numpy' or 'numba'.

    Returns
    -------
    str
        Path of the generated leg input.
    """
    leg_case = copy.deepcopy(case)
    leg_case['case'] = 'perf_%s' % backend
    prod = leg_case.setdefault('production', {})
    for key in FSSH_KERNEL_KEYS:
        prod[key] = backend
    prod['fssh_profile'] = 1

    leg_dir = os.path.join(run_dir, backend)
    input_path = fssh.build_production_input(leg_case, leg_dir)

    # Pin the trajectory random stream explicitly (the control
    # default is 1; writing it keeps the leg inputs self-contained).
    with open(input_path, 'r', encoding='utf-8') as handler:
        workflow = yaml.safe_load(handler)
    workflow['control']['gl_seed'] = int(prod.get('gl_seed', 1))
    with open(input_path, 'w', encoding='utf-8') as handler:
        yaml.safe_dump(workflow, handler, sort_keys=False)
    return input_path


def build_legs(case, run_dir):
    """Build the ordered backend legs of the perf benchmark.

    Parameters
    ----------
    case : dict
        Parsed case definition (the perf.yaml production section).
    run_dir : str
        Benchmark run directory (the planner pre-creates it).

    Returns
    -------
    list of dict
        One descriptor per backend leg: name, leg_dir, input_path
        and the timed command the planner chains into the compute
        script (the timing wrapper saves the wall time and captures
        the calculation stdout for the profile parsing).

    Raises
    ------
    RuntimeError
        When numba is not installed.
    """
    _require_numba()
    legs = []
    for backend in BACKENDS:
        leg_dir = os.path.join(run_dir, backend)
        os.makedirs(leg_dir, exist_ok=True)
        input_path = _build_backend_input(case, run_dir, backend)
        command = (
            'cd %s && python3 -m '
            'pyrai3md.qualification.benchmark.timing %s %s %s'
            % (leg_dir, input_path, leg_dir, backend))
        legs.append({
            'name': backend,
            'leg_dir': leg_dir,
            'input_path': input_path,
            'command': command,
        })
    return legs


def _read_leg_timing(leg_dir):
    """Read the wall time (seconds) saved by the timing wrapper.

    Raises
    ------
    FileNotFoundError
        When the leg never wrote its timing file.
    """
    path = os.path.join(leg_dir, 'leg.timing')
    with open(path, 'r', encoding='utf-8') as handler:
        return float(handler.read().strip())


def _infer_nstate(leg_dir):
    """Infer the state count from the leg's first md.pop record.

    The extended pop format carries time + nstate electronic +
    nstate active columns (nstate = (ncols - 1) / 2); legacy
    population-only records cannot disambiguate the state count
    and are rejected loudly.

    Raises
    ------
    RuntimeError
        When no pop record exists or the width is inconsistent.
    """
    pop_files = []
    for root, _dirs, files in os.walk(leg_dir):
        for fname in files:
            if fname.endswith('.md.pop'):
                pop_files.append(os.path.join(root, fname))
    if not pop_files:
        raise RuntimeError(
            'no trajectory record in %s' % leg_dir)
    record = fssh.parse_pop(sorted(pop_files)[0])
    ncols = record.shape[1]
    if ncols < 5 or ncols % 2 == 0:
        raise RuntimeError(
            'leg %s carries %d pop columns; the perf benchmark '
            'needs the extended electronic + active format'
            % (leg_dir, ncols))
    return (ncols - 1) // 2


def parse_profile(stdout_path):
    """Best-effort parse of the last fssh_profile summary table.

    The profiler prints one 'FSSH Performance' block every 100
    integrator calls per worker process; the last block of the
    captured leg stdout is the most complete accumulation of the
    last worker and serves as the function-level breakdown.

    Parameters
    ----------
    stdout_path : str
        Path of the captured leg stdout file.

    Returns
    -------
    dict or None
        {function name: (call count, total seconds)} of the last
        block, or None when the file or the table is absent.
    """
    if not os.path.isfile(stdout_path):
        return None
    with open(stdout_path, 'r', encoding='utf-8',
              errors='ignore') as handler:
        lines = handler.readlines()

    starts = [i for i, line in enumerate(lines)
              if line.strip() == 'FSSH Performance']
    if not starts:
        return None

    table = {}
    for line in lines[starts[-1] + 1:]:
        text = line.rstrip('\n')
        if not text.strip() or text.startswith('Implementation'):
            break
        match = _PROFILE_ROW.match(text)
        if not match or match.group('func').strip() in (
                'Function', 'Total'):
            continue
        table[match.group('func').strip()] = (
            int(match.group('calls')), float(match.group('seconds')))
    return table or None


def _leg_results(leg_dir, nstep):
    """Ensemble-average the trajectory populations of one leg.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray or None, int)
        (electronic ensemble, active ensemble, trajectory count).
    """
    nstate = _infer_nstate(leg_dir)
    populations = fssh.collect_populations(leg_dir, nstate)
    ensemble, active = fssh.ensemble_average(populations, nstep)
    return ensemble, active, len(populations)


def evaluate_run(run_dir):
    """Evaluate one computed perf benchmark run directory.

    Reads the outputs of the two backend legs (run_dir/numpy and
    run_dir/numba), compares the wall times and the ensemble
    averages, checks the backend agreement metric against the case
    tolerance and saves perf.json plus the averaged population
    tables into the run directory. A missing or broken leg output
    yields verdict ERROR instead of crashing silent. The
    qualification runner calls this in-process right after the
    compute task finished.

    Parameters
    ----------
    run_dir : str
        Benchmark run directory holding the case YAML and the two
        backend leg directories.

    Returns
    -------
    dict
        Report payload (case, metrics, tolerances, verdict,
        runtime_seconds, details).
    """
    started = time.time()
    try:
        case_path = fssh._find_case_yaml(run_dir)
        with open(case_path, 'r', encoding='utf-8') as handler:
            case = yaml.safe_load(handler)
        case_name = str(case.get('case', 'perf'))
        prod = case['production']
        nstep = int(prod['nstep'])
        dt = float(prod['dt'])
        metrics_spec = case.get('metrics', {})
        tolerances = {name: float(spec.get('tolerance', np.inf))
                      for name, spec in metrics_spec.items()}

        leg_dirs = {backend: os.path.join(run_dir, backend)
                    for backend in BACKENDS}
        walls = {backend: _read_leg_timing(leg_dirs[backend])
                 for backend in BACKENDS}
        ensembles = {}
        actives = {}
        counts = {}
        for backend in BACKENDS:
            ensemble, active, count = _leg_results(
                leg_dirs[backend], nstep)
            ensembles[backend] = ensemble
            actives[backend] = active
            counts[backend] = count

        # Backend agreement: both legs run the identical physics on
        # the identical random stream, so the ensemble averages
        # must agree to floating-point noise.
        max_dev = float(np.max(np.abs(
            ensembles['numpy'] - ensembles['numba'])))
        if actives['numpy'] is not None \
                and actives['numba'] is not None:
            max_dev = max(max_dev, float(np.max(np.abs(
                actives['numpy'] - actives['numba']))))

        metrics = {
            'numpy_wall': walls['numpy'],
            'numba_wall': walls['numba'],
            'speedup_wall':
                walls['numpy'] / walls['numba']
                if walls['numba'] > 0 else float('inf'),
            'backend_population_max_dev': max_dev,
        }

        # Function-level breakdown from the captured profiler
        # tables (best effort: an unparseable capture only drops
        # the detail entries, never the verdict).
        profiles = {}
        for backend in BACKENDS:
            stdout_path = os.path.join(
                leg_dirs[backend], '%s.stdout' % backend)
            profiles[backend] = parse_profile(stdout_path)
        for func in sorted(set((profiles['numpy'] or {}))
                           & set((profiles['numba'] or {}))):
            numpy_seconds = profiles['numpy'][func][1]
            numba_seconds = profiles['numba'][func][1]
            metrics['profile_%s' % func] = numba_seconds
            metrics['speedup_%s' % func] = (
                numpy_seconds / numba_seconds
                if numba_seconds > 0 else float('inf'))

        verdicts = {}
        for name, value in metrics.items():
            tolerance = float(metrics_spec.get(name, {}).get(
                'tolerance', np.inf))
            verdicts[name] = value <= tolerance
        verdict = 'PASS' if all(verdicts.values()) else 'FAIL'

        # Persist the compared ensemble averages and the full
        # report into the run directory for later inspection.
        time_grid = [step * dt for step in range(1, nstep + 1)]
        saved = []
        for backend in BACKENDS:
            pop_path = os.path.join(run_dir, '%s.pop' % backend)
            fssh._save_population(
                pop_path, time_grid, ensembles[backend])
            saved.append(pop_path)
        report_path = os.path.join(run_dir, 'perf.json')
        saved.append(report_path)

        details = {
            'trajectories': counts,
            'nstep': nstep,
            'dt': dt,
            'walls': walls,
            'profiles': {backend: {
                func: {'calls': calls, 'seconds': seconds}
                for func, (calls, seconds)
                in (profiles[backend] or {}).items()}
                for backend in BACKENDS},
            'per_metric': {name: bool(ok)
                           for name, ok in verdicts.items()},
            'saved': saved,
        }
    except Exception as exc:  # noqa: BLE001 - report, never crash silent
        case_name = os.path.basename(os.path.abspath(run_dir))
        metrics = {}
        tolerances = {}
        verdict = 'ERROR'
        details = {'error': str(exc)}
        report_path = None

    report = {
        'case': case_name,
        'metrics': metrics,
        'tolerances': tolerances,
        'verdict': verdict,
        'runtime_seconds': round(time.time() - started, 2),
        'details': details,
    }
    if report_path is not None:
        try:
            with open(report_path, 'w', encoding='utf-8') as handler:
                json.dump(report, handler, indent=2, sort_keys=True,
                          default=float)
        except OSError:
            pass
    return report
