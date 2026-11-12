"""
======================================================================
This module plans the benchmark calibration tests (the third pillar
of the qualification system, executed after the main framework,
example and skill tests).

Benchmark layout is flat (same level of directory management as
certification and verification): one single-script module per
benchmark task type, named after the task (<task>.py, e.g.
fssh.py), paired with its calibration case YAMLs: the unified
<task>.yaml (one case definition serving every method of the task,
e.g. fssh.yaml covers the tully1/2/3 models through the
dynamically injected method) plus optional dedicated
<task>_<method>.yaml files for model-specific cases. The task
module owns the complete calibration task (input generation,
independent reference handling and analysis); adding a new method
of an existing task usually only changes the pmd_env benchmark
entries, no new YAML file.

Discovery pairs every <task>.py with its <task>.yaml and
<task>_*.yaml files. Each case is isolated in
work_dir/benchmark/<case_key>/ and expands into
ONE compute task, skill-test style (generated input files + one
execution script):

1. Compute task: the reference input and the production input
   (both framework YAMLs, both executed by pyrai3md) live in the
   run directory as reference.yaml / production.yaml and are
   generated at plan time through the task module's builders. The
   two runs are chained into ONE script with '&&' so production
   can only start after the reference succeeded:

       cd <ref_dir>  && pyrai3md <run_dir>/reference.yaml && \
       cd <prod_dir> && pyrai3md <run_dir>/production.yaml

   The script is a local `.sh`; in a slurm round the runner places
   this case in one job, so the planner writes no per-case
   `.sbatch`.

   Task modules that need a different leg structure (the perf
   module runs one leg per FSSH backend instead of the
   reference/production pair) implement the build_legs contract
   instead of the two builder functions: build_legs(case, run_dir)
   returns an ordered list of leg descriptors (name, leg_dir,
   input_path, command) and the planner chains the ready-made leg
   commands with '&&' into the same single compute script.

2. Evaluation is a qualification tool, not a framework task: right
   after the compute task finishes, the runner calls the task
   module's evaluate_run(run_dir) in-process to parse the
   reference and trajectory populations, compute the metrics,
   persist production.pop / reference.pop into the run directory
   and report the verdict. No evaluate subprocess is planned.

Per-task settings live in the 'benchmark' section of pmd_env.yaml
as one subsection per task module (keyed by the module stem, e.g.
fssh). The subsection is either a single parameter dict (one
calibration case) or a list of parameter dicts (several cases of
the same task calibrated in one pass, e.g. the fssh tully1/2/3
models). Every entry selects the calibration case through its
'method' key: a dedicated <task>_<method>.yaml wins over the
unified <task>.yaml, which serves every method of the task through
the dynamically injected method (fssh + tully2 -> the unified
fssh.yaml with production.method set to tully2). The remaining
fields carry the physical parameters (nstep, dt, x0, v0, sigma,
model_mass, initial_state, ninitcond, ngrid, substeps, ...) that
override the case YAML at plan time; the merged case YAML is
written into the run directory so the evaluation tool sees the
identical settings.
An optional 'enabled' flag shares the coercion semantics of the
example switches. A missing benchmark section keeps every shipped
case enabled with the case YAML defaults (backward compatible);
when the section exists only the configured entries are planned.

The compute task carries a task-level timeout (3600 s by default):
the reference + production chain runs real dynamics and needs a
much larger budget than the 300 s functional-test guard.

Reference direction: planner -> qualification.manager (shared
script writers and flag coercion), planner -> benchmark task
modules (input builders and case loading).

Aug 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import importlib
import os
import shutil

import yaml

from ..manager import (_coerce_flag, _get_shell_commands, _merge_exports,
                       _resolve_chain, _runall_mode, _write_shell_script)


# Modules that are infrastructure, not benchmark task modules.
_NON_TASK_MODULES = ('planner', '__init__', 'timing', 'plotting')

# Task-level subprocess timeout for the compute task (seconds). The
# reference + production chain runs real dynamics and needs a far
# larger budget than the 300 s functional-test guard used by the
# skill/example phases.
DEFAULT_TIMEOUT = 3600


def discover_cases(pkg_dir):
    """Pair every task module with its calibration case YAMLs.

    Parameters
    ----------
    pkg_dir : str
        Directory of the benchmark package.

    Returns
    -------
    list of tuple(str, str, str)
        (task_name, case_key, case_yaml_path), sorted by case_key.
        task_name is the module stem ('fssh'), case_key is the YAML
        stem ('fssh' for the unified case, 'fssh_tully2' for a
        dedicated one).
    """
    task_modules = sorted(
        fname[:-3] for fname in os.listdir(pkg_dir)
        if fname.endswith('.py')
        and fname[:-3] not in _NON_TASK_MODULES
    )
    cases = []
    for task_name in task_modules:
        for fname in sorted(os.listdir(pkg_dir)):
            if not fname.endswith('.yaml'):
                continue
            stem = fname[:-5]
            # Pair the unified <task>.yaml and every dedicated
            # <task>_*.yaml with the task module.
            if stem == task_name or stem.startswith(task_name + '_'):
                cases.append((task_name, stem,
                              os.path.join(pkg_dir, fname)))
    return cases


def _resolve_case_yaml(pkg_dir, task_name, method):
    """Resolve the case YAML for one calibration method.

    A dedicated <task>_<method>.yaml (a model-specific case
    definition) wins over the unified <task>.yaml (the
    method-agnostic defaults the method is injected into at plan
    time).

    Parameters
    ----------
    pkg_dir : str
        Directory of the benchmark package.
    task_name : str
        Task module stem (e.g. 'fssh').
    method : str
        Calibration method key (e.g. 'tully1').

    Returns
    -------
    str
        Path of the resolved case YAML, or an empty string when
        neither a dedicated nor the unified file is shipped.
    """
    dedicated = os.path.join(pkg_dir, '%s_%s.yaml' % (task_name, method))
    if os.path.isfile(dedicated):
        return dedicated
    unified = os.path.join(pkg_dir, '%s.yaml' % task_name)
    if os.path.isfile(unified):
        return unified
    return ''


def _configured_cases(pkg_dir, flags):
    """Resolve the cases selected by the pmd_env benchmark section.

    One subsection per task module, either a single parameter dict
    (one calibration case) or a list of parameter dicts (several
    cases of the same task calibrated in one pass, e.g. fssh with
    tully1/tully2/tully3). Every entry selects the case through its
    'method' key (a dedicated <task>_<method>.yaml wins over the
    unified <task>.yaml, whose production.method the entry
    overrides) and forwards every remaining field to the task
    module as a parameter override. Entries that cannot be
    resolved become diagnostic skip descriptors instead of silent
    drops; a duplicated method only plans its first occurrence
    (the run directory is keyed by the case).

    Parameters
    ----------
    pkg_dir : str
        Directory of the benchmark package.
    flags : dict
        The benchmark section of pmd_env.yaml (task name -> dict
        or list of dicts).

    Returns
    -------
    tuple(list, list)
        (planned, skips): planned entries are (task_name, case_key,
        case_yaml_path, overrides) tuples; skips are ready-made skip
        task descriptors.
    """
    planned = []
    skips = []
    for task_name in sorted(
            fname[:-3] for fname in os.listdir(pkg_dir)
            if fname.endswith('.py')
            and fname[:-3] not in _NON_TASK_MODULES):
        section = flags.get(task_name)
        if section is None:
            # Section present but this task is not configured: only
            # explicitly configured tasks are planned.
            continue
        entries = [section] if isinstance(section, dict) else section
        if not isinstance(entries, list):
            skips.append(_make_skip_task(
                task_name,
                'benchmark.%s must be a parameter mapping or a list '
                'of parameter mappings, got %r (the per-case boolean '
                'switch format is no longer supported)'
                % (task_name, section)))
            continue
        seen = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                skips.append(_make_skip_task(
                    '%s_%d' % (task_name, index),
                    'benchmark.%s[%d] must be a parameter mapping, '
                    'got %r' % (task_name, index, entry)))
                continue
            method = entry.get('method')
            if not method:
                skips.append(_make_skip_task(
                    '%s_%d' % (task_name, index),
                    "benchmark.%s[%d] does not set the 'method' key"
                    % (task_name, index)))
                continue
            case_key = '%s_%s' % (task_name, method)
            if case_key in seen:
                skips.append(_make_skip_task(
                    case_key,
                    'benchmark.%s configures method %r more than '
                    'once; only the first entry is planned (the run '
                    'directory is keyed by the method)'
                    % (task_name, method)))
                continue
            seen.add(case_key)
            case_yaml = _resolve_case_yaml(pkg_dir, task_name, method)
            if not case_yaml:
                skips.append(_make_skip_task(
                    case_key,
                    'no case YAML shipped for method %r (neither '
                    '%s_%s.yaml nor the unified %s.yaml)'
                    % (method, task_name, method, task_name)))
                continue
            planned.append((task_name, case_key, case_yaml, dict(entry)))
    return planned, skips


def _make_skip_task(case_key, skip_reason):
    """Create a diagnostic skip task for a case that cannot run.

    Parameters
    ----------
    case_key : str
        Case YAML stem (e.g. 'fssh_tully1').
    skip_reason : str
        Human-readable explanation for the test report.

    Returns
    -------
    dict
        A skip task descriptor.
    """
    return {
        'name': f'benchmark_{case_key}',
        'skill': 'benchmark',
        'mode': 'benchmark',
        'skip': True,
        'skip_reason': skip_reason,
    }


def plan_benchmark_tests(config, work_dir):
    """Plan benchmark calibration tasks from the flat case layout.

    Every discovered case expands into a single compute task
    (reference and production chained in one local .sh / slurm
    .sbatch script) working inside the isolated
    work_dir/benchmark/<case_key>/ directory; the task descriptor
    carries the run directory and the task module name so the
    runner can call the module's evaluate_run in-process after the
    compute finished. The script carries the global
    environment.shell and export variables exactly like the
    example and skill wrappers, so module loads and environment
    setup apply to the compute subprocesses as well.

    Parameters
    ----------
    config : dict
        Parsed pmd_env.yaml configuration.
    work_dir : str
        Working directory for generated files and runs.

    Returns
    -------
    list of dict
        List of benchmark task descriptors (may be empty when no
        task module / case YAML is shipped or every case is
        disabled).
    """
    tasks = []
    pkg_dir = os.path.dirname(__file__)
    flags = config.get('benchmark', {}) or {}
    export_vars = _merge_exports(config)
    shell_commands = _get_shell_commands(config)

    # Top-level `_runall` master switch, shared with the example and skill
    # planners: true forces every configured benchmark.<task>[].enabled switch
    # on so all shipped calibrations run, false closes them all (each case
    # still planned as a skipped task naming the switch), and 'condition' -
    # what an absent key means - leaves each case's own flag to answer. The
    # benchmark section stays the sole source of which cases are planned.
    master = _runall_mode(config)

    # Case selection: the benchmark section (one subsection per
    # task, a single parameter dict or a list of them; each entry's
    # 'method' picks the case YAML and the remaining fields
    # override the case parameters) drives the planning when
    # present; a missing section keeps every shipped case enabled
    # with the case YAML defaults (backward compatible).
    if flags:
        cases, skips = _configured_cases(pkg_dir, flags)
        tasks.extend(skips)
    else:
        cases = [(task, key, path, {})
                 for task, key, path in discover_cases(pkg_dir)]

    for task_name, case_key, case_yaml, overrides in cases:
        task_name_full = f'benchmark_{case_key}'

        # Per-task switch in pmd_env.yaml (benchmark.<task>.enabled); a missing
        # key keeps the case enabled. The master is the only thing above it, so
        # a `true` master runs the case whatever this flag says, a `false`
        # master closes it, and `condition` reads the flag.
        enabled_flag = True
        if overrides:
            enabled_flag = overrides.get('enabled', True)
        state, closer = _resolve_chain([
            (master, '_runall'),
            (_coerce_flag(enabled_flag,
                          f'benchmark.{task_name}.enabled', True, 'case'),
             f'benchmark.{task_name}.enabled'),
        ])
        if state is False:
            tasks.append(_make_skip_task(
                case_key, 'disabled by %s' % closer))
            continue

        # Isolate the run in work_dir/benchmark/<case_key>/ so the
        # compute subprocesses work in a clean directory and never
        # pollute the installed package. The pmd_env overrides are
        # merged into the case at plan time and the merged YAML is
        # written into the run directory, so the evaluation tool
        # (which re-loads this file) sees the identical settings.
        run_dir = os.path.join(work_dir, 'benchmark', case_key)
        os.makedirs(run_dir, exist_ok=True)

        # The task module supplies the input builders and the case
        # loader; a broken module surfaces as a diagnostic skip.
        try:
            module = importlib.import_module(
                f'.{task_name}', 'pyrai3md.qualification.benchmark')
        except Exception as exc:  # noqa: BLE001 - surface as diagnostic
            tasks.append(_make_skip_task(
                case_key, f'task module import failed: {exc}'))
            continue

        try:
            case = module.load_case(case_yaml)
            if overrides and hasattr(module, 'apply_overrides'):
                case = module.apply_overrides(case, overrides)
            # The run directory always carries the case YAML under
            # the case key (fssh_tully2.yaml even when the unified
            # fssh.yaml supplied the defaults) so every run is
            # self-describing regardless of the source file.
            run_case_yaml = os.path.join(
                run_dir, '%s.yaml' % case_key)
            if overrides:
                with open(run_case_yaml, 'w',
                          encoding='utf-8') as handler:
                    yaml.safe_dump(case, handler, sort_keys=False)
            else:
                shutil.copy(case_yaml, run_case_yaml)

            if hasattr(module, 'build_legs'):
                # Custom leg contract (perf: one leg per backend):
                # the module supplies ready-made leg commands and
                # the planner only chains them into the script.
                legs = module.build_legs(case, run_dir)
                if not legs:
                    raise ValueError(
                        'build_legs returned no legs')
                compute_cmd = ' && '.join(
                    leg['command'] for leg in legs)
                first_input = legs[0]['input_path']
            else:
                ref_dir = os.path.join(run_dir, 'reference')
                prod_dir = os.path.join(run_dir, 'production')
                # The compute chain cds into both leg directories
                # before launching pyrai3md; pre-create them at
                # plan time so the cd cannot fail on a fresh run
                # directory.
                os.makedirs(ref_dir, exist_ok=True)
                os.makedirs(prod_dir, exist_ok=True)
                ref_input = module.build_reference_input(
                    case, run_dir)
                prod_input = module.build_production_input(
                    case, run_dir)
                # === Compute task: reference && production in ONE
                # script. The '&&' chain guarantees the production
                # run only starts after the reference succeeded.
                # Both legs go through the real pyrai3md entry point
                # on framework YAML inputs generated into the run
                # directory.
                compute_cmd = (f'cd {ref_dir} && pyrai3md '
                               f'{ref_input} && cd {prod_dir} && '
                               f'pyrai3md {prod_input}')
                first_input = ref_input
        except Exception as exc:  # noqa: BLE001 - surface as diagnostic
            tasks.append(_make_skip_task(
                case_key, f'input generation failed: {exc}'))
            continue

        sh_compute = _write_shell_script(
            run_dir, task_name_full, compute_cmd,
            export_vars, shell_commands, timeout=DEFAULT_TIMEOUT,
        )
        task_compute = {
            'name': task_name_full,
            'skill': 'benchmark',
            'mode': 'benchmark',
            'input_file': first_input,
            'script': sh_compute,
            'use_slurm': False,
            'timeout': DEFAULT_TIMEOUT,
            # One job per calibration in a slurm round: a benchmark reads
            # its own wall clock, so no two calibrations share a job.
            'job_group': 'benchmark__%s' % case_key,
            # The runner calls the task module's evaluate_run on
            # this directory in-process after the compute task
            # finished.
            'run_dir': run_dir,
            'benchmark_task': task_name,
        }
        tasks.append(task_compute)

    return tasks
