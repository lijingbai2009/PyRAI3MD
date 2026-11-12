"""
======================================================================
This module owns the grouped Slurm execution of one qualification
round.

A round is a plan of independent test cases. When
`environment.scheduler` is 'slurm', the runner hands that plan to this
module instead of walking it in one local process: the plan is grouped
into jobs (one job per example carrier and per skill carrier leg, with
a case chain kept together inside its leg), each job is written as a
spec plus an sbatch script, the jobs are submitted once, polled, and
their per-task result rows are read back and handed to the runner in
plan order. The orchestrator stays the `pyrai3md test` process; only
the test tasks move to the cluster.

The grouping key is `job_group`, which the planners put on every task
descriptor (main unit, integration, one per example carrier, one per
skill carrier leg, one per skill runtime phase, one per benchmark
case). Grouping here is therefore a plan reading, not a second set of
rules that could drift from the planners.

`environment.scheduler` is the main framework's round mode only. A
skill's own `execution_mode: slurm` still makes that skill submit its
own sbatch from inside a job; the two switches are independent, and
the nested submission is the skill feature under test (it is subject
to the cluster's policy on compute-node submission, which is a site
setting this module does not try to override).

Sep 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import glob
import json
import os
import shlex
import signal
import subprocess
import sys
import time

import yaml

from .manager import (load_config, _get_shell_commands, _merge_exports,
                      _write_slurm_script)
from ..utilities.serialization import json_safe, write_json


# The buckets of a plan, in the order the report prints them. The index
# this module gives a task is the position of that task in this
# traversal, and it is the name of its result fragment
# (`jobs/<gid>/rows/<index>.json`); the runner walks the same order.
BUCKET_ORDER = ('main_tests', 'example_tests', 'skill_tests',
                'benchmark_tests')

# The benchmark bucket has two submission policies: the physics
# calibrations compare populations and energies and may share the
# cluster, while the perf calibration compares wall clocks and must not
# compete with anything. The round still runs them after every other
# category, never beside it.
BENCHMARK_BUCKET = 'benchmark_tests'
TIMING_BENCHMARK = 'perf'

# Slurm states that mean the job will never produce another row. The
# list is the one the framework dispatch layer waits on, repeated here
# because qualification must not import the runtime -- the dispatch
# module is the framework's, the test plan is not.
TERMINAL_STATES = frozenset((
    'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
    'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED',
    'SPECIAL_EXIT',
))

# How many jobs a round keeps submitted and unfinished before it stops
# submitting more. `environment.max_in_flight` sets the cap; 0 (the
# default) submits the whole round at once and lets the scheduler queue
# it. It lives with `planning_workers`, not with the `slurm:` directives:
# that section is rendered into #SBATCH and refuses keys that map to no
# directive.
DEFAULT_MAX_IN_FLIGHT = 0
POLL_INTERVAL = 30

# How many poll rounds a job id may be missing from both squeue and
# sacct before it is treated as ended. Accounting can lag a finishing
# job; the fragment on disk is consulted first, so this only decides
# the dead-job case. Ten rounds (5 min at the default interval) leaves
# room for a controller that is slow to show a freshly submitted job.
UNKNOWN_GRACE = 10

# After this many consecutive sbatch failures, stop submitting: a bad
# partition or account rejects every job, and one error per group says
# more than hundreds of identical ones.
SUBMIT_FAIL_STREAK = 3

SPEC_SCHEMA = 1


def scheduler_mode(config):
    """The round execution mode named by `environment.scheduler`.

    Parameters
    ----------
    config : dict
        Parsed pmd_env.yaml configuration.

    Returns
    -------
    str
        'slurm' when the round is submitted as grouped jobs, 'local'
        otherwise (including the empty and 'local' spellings).
    """
    value = str((config or {}).get('environment', {})
                .get('scheduler', '') or '')
    return 'slurm' if value.strip().lower() == 'slurm' else 'local'


def in_flight_limit(config):
    """The round's unfinished-job cap named by `environment.max_in_flight`.

    Parameters
    ----------
    config : dict
        Parsed pmd_env.yaml configuration.

    Returns
    -------
    int
        The cap; 0 (also the default when the key is absent) means no
        cap: every group is submitted at once.

    Raises
    ------
    ValueError
        When the value is not an integer or is negative.
    """
    value = ((config or {}).get('environment') or {}).get(
        'max_in_flight', DEFAULT_MAX_IN_FLIGHT)
    try:
        limit = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            'environment.max_in_flight must be a non-negative integer, '
            'got %r' % (value,))
    if limit < 0:
        raise ValueError(
            'environment.max_in_flight must be a non-negative integer, '
            'got %r' % (value,))
    return limit


def iter_plan(plan):
    """Yield every planned task with its round-wide index.

    The index is the position in the bucket-major plan traversal. It
    names the task's row fragment and it is the order the runner
    prints; skips get one too, because the report accounts for them in
    place.

    Parameters
    ----------
    plan : dict
        Test plan from manager.generate_plan().

    Yields
    ------
    tuple of (int, str, dict)
        (index, bucket, task).
    """
    index = 0
    for bucket in BUCKET_ORDER:
        for task in plan.get(bucket, []) or []:
            yield index, bucket, task
            index += 1


def build_groups(plan):
    """Group the runnable tasks of a plan into independent jobs.

    A task names its group with `job_group`; a descriptor that does
    not (a hand-built plan in a test) falls back to its own name, so it
    is a job of one. Tasks the switches closed are not jobs: the runner
    resolves those rows locally.

    Parameters
    ----------
    plan : dict
        Test plan from manager.generate_plan().

    Returns
    -------
    list of dict
        One entry per group, in first-appearance order, with keys
        `gid`, `short`, `bucket`, `indices`, `tasks` and, once
        submitted, `job_id`/`state`.
    """
    groups = []
    by_key = {}
    for index, bucket, task in iter_plan(plan):
        if task.get('skip'):
            continue
        key = str(task.get('job_group') or task.get('name') or index)
        group = by_key.get(key)
        if group is None:
            group = {
                'gid': key,
                'short': 'g%03d' % len(groups),
                'bucket': bucket,
                'indices': [],
                'tasks': [],
                'job_id': '',
                'state': '',
            }
            by_key[key] = group
            groups.append(group)
        group['indices'].append(index)
        group['tasks'].append((index, task))
    return groups


def _group_job_name(slurm_config, short):
    """The per-job name, derived from the round's configured name."""
    base = str(slurm_config.get('job_name') or 'pyrai3md')
    return '%s-%s' % (base, short)


def _job_command(spec_path, work_dir, config):
    """The commands a job runs, after the directive and export block.

    The job starts on a compute node with neither the login node's
    environment nor its working directory, so it re-establishes both:
    the conda environment the deployment installed into (best effort --
    a node without conda still runs the configured interpreter), the
    synced source tree on `PYTHONPATH`, and the round's work directory
    as cwd.
    """
    environment = (config or {}).get('environment', {}) or {}
    python = str(environment.get('python') or 'python3')
    conda_env = str(environment.get('conda') or '')
    project_root = os.path.dirname(work_dir)
    lines = []
    if conda_env:
        lines.append(
            '( source "$(conda info --base 2>/dev/null)/etc/profile.d/'
            'conda.sh" 2>/dev/null && conda activate %s ) || '
            'echo "[job] conda activate %s skipped" >&2'
            % (conda_env, conda_env))
    lines.append('export PYTHONPATH=%s${PYTHONPATH:+:$PYTHONPATH}'
                 % shlex.quote(project_root))
    lines.append('cd %s' % shlex.quote(work_dir))
    lines.append('%s -u -m pyrai3md.qualification.jobs %s'
                 % (python, shlex.quote(spec_path)))
    return '\n'.join(lines)


def prepare(plan, config, work_dir):
    """Write the round's job specs and sbatch scripts.

    Parameters
    ----------
    plan : dict
        Test plan from manager.generate_plan().
    config : dict
        Parsed pmd_env.yaml configuration.
    work_dir : str
        The round's work directory (the specs live under jobs/).

    Returns
    -------
    dict
        `groups`, `jobs_dir`, `config_path` and `ledger_path`.
    """
    groups = build_groups(plan)
    jobs_dir = os.path.join(work_dir, 'jobs')
    os.makedirs(jobs_dir, exist_ok=True)

    config_path = os.path.join(jobs_dir, 'config.yaml')
    with open(config_path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    slurm_config = dict((config or {}).get('slurm', {}) or {})
    export_vars = _merge_exports(config)
    shell_commands = _get_shell_commands(config)
    configured_time = str(slurm_config.get('time') or '')

    for group in groups:
        group_dir = os.path.join(jobs_dir, group['gid'])
        os.makedirs(os.path.join(group_dir, 'rows'), exist_ok=True)
        spec = {
            'schema': SPEC_SCHEMA,
            'gid': group['gid'],
            'bucket': group['bucket'],
            'work_dir': work_dir,
            'config': config_path,
            'tasks': [
                {'index': index, 'task': json_safe(task)}
                for index, task in group['tasks']
            ],
        }
        spec_path = os.path.join(group_dir, 'job.json')
        write_json(spec_path, spec)

        overrides = dict(slurm_config)
        overrides['job_name'] = _group_job_name(slurm_config,
                                                group['short'])
        overrides['output'] = os.path.join(
            jobs_dir, '%s.%%j.out' % group['short'])
        overrides['error'] = os.path.join(
            jobs_dir, '%s.%%j.err' % group['short'])
        group['script'] = _write_slurm_script(
            jobs_dir, group['short'],
            _job_command(spec_path, work_dir, config),
            overrides, export_vars, shell_commands)
        group['spec'] = spec_path

    multi = [group for group in groups if len(group['tasks']) > 1]
    if multi:
        largest = max(multi, key=lambda item: len(item['tasks']))
        print('  %d job group(s) hold more than one task under --time=%s; '
              'raise slurm.time if a group needs more (largest: %s, %d tasks)'
              % (len(multi), configured_time or '(cluster default)',
                 largest['gid'], len(largest['tasks'])), flush=True)

    ledger_path = os.path.join(jobs_dir, 'jobs.json')
    _write_ledger(ledger_path, work_dir, groups)
    return {
        'groups': groups,
        'jobs_dir': jobs_dir,
        'config_path': config_path,
        'ledger_path': ledger_path,
    }


def _write_ledger(path, work_dir, groups):
    """Record what this round submitted, for `deploy --force` to undo."""
    payload = {
        'schema': SPEC_SCHEMA,
        'work_dir': work_dir,
        'groups': [
            {
                'gid': group['gid'],
                'short': group['short'],
                'bucket': group['bucket'],
                'job_id': group.get('job_id', ''),
                'state': group.get('state', ''),
                'script': group.get('script', ''),
                'indices': group['indices'],
            }
            for group in groups
        ],
    }
    write_json(path, payload)


def _group_rows_dir(work_dir, group):
    return os.path.join(work_dir, 'jobs', group['gid'], 'rows')


def _read_row(work_dir, group, index):
    path = os.path.join(_group_rows_dir(work_dir, group), '%d.json' % index)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _error_row(task, group, reason):
    """The row a task gets when its job ended without one."""
    return {
        'name': task.get('name', ''),
        'mode': task.get('mode', ''),
        'theory': task.get('theory', ''),
        'input_file': task.get('input_file', ''),
        'status': 'error',
        'details': reason,
        'job_group': group['gid'],
        'job_id': group.get('job_id', ''),
    }


def _error_tail(jobs_dir, group, job_id):
    """The last lines of the job's stderr file, when it exists."""
    if not job_id:
        candidates = sorted(glob.glob(os.path.join(
            jobs_dir, '%s.*.err' % group['short'])))
    else:
        candidates = [os.path.join(
            jobs_dir, '%s.%s.err' % (group['short'], job_id))]
    for path in candidates:
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as handle:
                lines = handle.read().splitlines()[-20:]
            if lines:
                return '\n'.join(lines)
        except OSError:
            continue
    return ''


def collect(work_dir, group, jobs_dir=None, state=''):
    """Read a finished group's rows, filling the gaps with errors.

    A job killed at its time limit keeps the rows its finished tasks
    already wrote; only the tail is missing, and that tail becomes one
    error row per task naming the job, its state and the stderr tail.
    """
    jobs_dir = jobs_dir or os.path.join(work_dir, 'jobs')
    rows = []
    missing = []
    for index, task in group['tasks']:
        row = _read_row(work_dir, group, index)
        if row is None:
            missing.append((index, task))
            continue
        row.setdefault('job_group', group['gid'])
        row.setdefault('job_id', group.get('job_id', ''))
        rows.append((index, row))
    if missing:
        reason = ('job %s (id %s) ended %s without a result'
                  % (group['gid'], group.get('job_id') or '?',
                     state or 'UNKNOWN'))
        tail = _error_tail(jobs_dir, group, group.get('job_id', ''))
        if tail:
            reason = '%s; job stderr tail:\n%s' % (reason, tail)
        for index, task in missing:
            rows.append((index, _error_row(task, group, reason)))
    rows.sort(key=lambda item: item[0])
    return rows


def run_spec(spec_path):
    """Execute one job's tasks locally on the node it was submitted to.

    The plan is not regenerated here: the spec carries the descriptors
    the orchestrator planned, and regenerating them would re-plan a
    different round (a new work directory, host-dependent switches).
    Only the shared configuration is reloaded, from the snapshot the
    orchestrator wrote, so every job of a round reads one config.

    Each row is written as soon as its task finishes, which is what
    lets a job that hits its time limit keep the rows it completed.
    """
    from .runner import _execute_task

    with open(spec_path, 'r', encoding='utf-8') as handle:
        spec = json.load(handle)
    config = load_config(spec['config'])
    work_dir = spec['work_dir']
    group = {'gid': spec['gid'], 'job_id': os.environ.get('SLURM_JOB_ID', '')}
    plan = {'work_dir': work_dir, 'slurm': config.get('slurm', {})}

    rows_dir = os.path.join(work_dir, 'jobs', spec['gid'], 'rows')
    os.makedirs(rows_dir, exist_ok=True)
    failed = 0
    for entry in spec.get('tasks', []):
        task = dict(entry['task'])
        # The job is the allocation: a leg that carried its own sbatch
        # script from the old per-task path would submit a job inside a
        # job. The skill's own execution_mode is not touched here -- that
        # is the separate feature this round may be exercising.
        task.pop('sbatch_script', None)
        task['use_slurm'] = False
        result = _execute_task(task, plan, config)
        result.setdefault('job_group', spec['gid'])
        result.setdefault('job_id', group['job_id'])
        if result.get('status') in ('fail', 'error'):
            failed += 1
        write_json(os.path.join(rows_dir, '%d.json' % entry['index']),
                   json_safe(result))
        print('[jobs] %d %s -> %s'
              % (entry['index'], result.get('name', ''),
                 str(result.get('status', '')).upper()), flush=True)
    write_json(os.path.join(work_dir, 'jobs', spec['gid'], 'fragment.json'),
               {'rows': len(spec.get('tasks', [])), 'failed': failed})
    return 0


def main(argv=None):
    """Job entry point: one spec path."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print('usage: python -m pyrai3md.qualification.jobs <job.json>',
              file=sys.stderr)
        return 2
    return run_spec(argv[0])


def _normalize_state(text):
    """Slurm's state word, without its explanation suffix."""
    value = str(text or '').strip()
    if not value:
        return ''
    state = value.split()[0].upper()
    if state.startswith('CANCELLED'):
        return 'CANCELLED'
    return state


def _chunks(values, size=100):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _submit(script_path):
    """Submit one script; return `(job_id, error_text)`."""
    try:
        proc = subprocess.run(
            ['sbatch', '--parsable', script_path],
            capture_output=True, text=True)
    except OSError as exc:
        return '', str(exc)
    if proc.returncode != 0:
        return '', (proc.stderr or proc.stdout).strip()
    job_id = proc.stdout.strip().split(';')[0].strip()
    if not job_id:
        return '', 'sbatch printed no job id'
    return job_id, ''


def _query_states(job_ids):
    """The scheduler's state for each id it still knows.

    `squeue` is the live view; an id that left it is resolved through
    `sacct`, and an id accounting also does not know is read from
    `scontrol`. An id that answers nowhere is simply absent from the
    returned map, and the caller's grace decides what that means.
    """
    states = {}
    ids = [job_id for job_id in job_ids if job_id]
    for chunk in _chunks(ids):
        try:
            proc = subprocess.run(
                ['squeue', '-h', '-o', '%i %T', '-j', ','.join(chunk)],
                capture_output=True, text=True)
        except OSError:
            break
        if proc.returncode != 0:
            continue
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                states[parts[0]] = _normalize_state(parts[1])

    missing = [job_id for job_id in ids if job_id not in states]
    for chunk in _chunks(missing):
        try:
            proc = subprocess.run(
                ['sacct', '-j', ','.join(chunk), '-n', '-P',
                 '-o', 'JobID,State'],
                capture_output=True, text=True)
        except OSError:
            proc = None
        if proc is not None and proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if '|' not in line:
                    continue
                raw_id, state = line.split('|', 1)
                base = raw_id.split('.')[0].strip()
                if not base:
                    continue
                state = _normalize_state(state)
                # The .batch row can close before the base row does, and
                # a terminal state from either row is the answer; a live
                # state only stands until one arrives.
                if base not in states or (
                        state in TERMINAL_STATES
                        and states[base] not in TERMINAL_STATES):
                    states[base] = state

    still_missing = [job_id for job_id in ids if job_id not in states]
    for job_id in still_missing:
        try:
            proc = subprocess.run(
                ['scontrol', 'show', 'job', job_id],
                capture_output=True, text=True)
        except OSError:
            continue
        if proc.returncode != 0:
            continue
        for token in proc.stdout.split():
            if token.startswith('JobState='):
                states[job_id] = _normalize_state(token.split('=', 1)[1])
                break
    return states


def _scancel(job_ids):
    ids = [job_id for job_id in job_ids if job_id]
    if not ids:
        return
    try:
        subprocess.run(['scancel'] + ids, capture_output=True, text=True)
    except OSError:
        pass


def _is_timing_benchmark(group):
    """Whether this benchmark group measures wall clock (the perf case).

    `perf` compares backends by elapsed time, so it runs alone after the
    physics calibrations; fssh/ktdc/gsh/lzsh compare populations and
    energies and can share the cluster.
    """
    for _index, task in group.get('tasks', []):
        if str(task.get('benchmark_task') or '') == TIMING_BENCHMARK:
            return True
    return str(group.get('gid', '')).endswith('__%s' % TIMING_BENCHMARK)


def execute(plan, config, on_row, on_pulse=None):
    """Submit the plan's groups, poll them, and hand back every row.

    Parameters
    ----------
    plan : dict
        Test plan from manager.generate_plan().
    config : dict
        Parsed pmd_env.yaml configuration.
    on_row : callable
        Called as `on_row(index, row)` for every completed task.
    on_pulse : callable, optional
        Called once per polling round with a one-line status string.

    Returns
    -------
    dict
        The `prepare` result, with each group's final state.
    """
    work_dir = plan['work_dir']
    prepared = prepare(plan, config, work_dir)
    groups = prepared['groups']
    jobs_dir = prepared['jobs_dir']
    ledger_path = prepared['ledger_path']
    if not groups:
        return prepared

    delivered = set()

    def _deliver(group, state):
        for index, row in collect(work_dir, group, jobs_dir, state):
            if index in delivered:
                continue
            delivered.add(index)
            on_row(index, row)

    def _stamp(group, state):
        if group.get('state') == state:
            return
        group['state'] = state
        _write_ledger(ledger_path, work_dir, groups)

    def _terminal(group):
        return (group.get('state') in TERMINAL_STATES
                or group.get('state') in ('SUBMIT_FAILED', 'UNKNOWN'))

    def _pulse():
        if on_pulse is None:
            return
        done = sum(1 for group in groups if _terminal(group))
        running = sum(1 for group in groups
                      if group.get('state') in ('SUBMITTED', 'RUNNING',
                                                'PENDING', 'COMPLETING'))
        pending = len(groups) - done - running
        last = ''
        for group in groups:
            if group.get('state'):
                last = '%s %s' % (group['gid'], group['state'])
        on_pulse('done %d/%d running %d pending %d | rows %d%s'
                 % (done, len(groups), running, pending, len(delivered),
                    (' | last: %s' % last) if last else ''))

    wave_one = [group for group in groups
                if group['bucket'] != BENCHMARK_BUCKET]
    wave_benchmarks = [group for group in groups
                       if group['bucket'] == BENCHMARK_BUCKET]
    bench_parallel = [group for group in wave_benchmarks
                      if not _is_timing_benchmark(group)]
    bench_serial = [group for group in wave_benchmarks
                    if _is_timing_benchmark(group)]

    inflight = {}
    unknown = {}
    interrupted = {'value': False}
    limit = in_flight_limit(config)

    def _run_wave(queue, serial=False):
        queue = list(queue)
        fail_streak = 0
        try:
            while queue or inflight:
                capacity = 1 if serial else (
                    limit or len(queue) + len(inflight))
                submitted = False
                while queue and len(inflight) < capacity:
                    group = queue.pop(0)
                    job_id, error = _submit(group['script'])
                    if not job_id:
                        fail_streak += 1
                        _stamp(group, 'SUBMIT_FAILED')
                        for index, row in collect(
                                work_dir, group, jobs_dir,
                                'SUBMIT_FAILED: %s' % error):
                            if index not in delivered:
                                delivered.add(index)
                                on_row(index, row)
                        if fail_streak >= SUBMIT_FAIL_STREAK:
                            for rest in queue:
                                _stamp(rest, 'SUBMIT_FAILED')
                                for index, row in collect(
                                        work_dir, rest, jobs_dir,
                                        'not submitted after %d consecutive '
                                        'sbatch failures: %s'
                                        % (fail_streak, error)):
                                    if index not in delivered:
                                        delivered.add(index)
                                        on_row(index, row)
                            queue = []
                            break
                        continue
                    fail_streak = 0
                    group['job_id'] = job_id
                    group['state'] = 'SUBMITTED'
                    inflight[job_id] = group
                    unknown.pop(job_id, None)
                    submitted = True
                # One ledger write per fill, not per job: a full round
                # submits hundreds of scripts and rewriting the whole
                # ledger for each of them turned the submit loop into
                # JSON serialization.
                if submitted:
                    _write_ledger(ledger_path, work_dir, groups)

                if not queue and not inflight:
                    break

                states = _query_states(list(inflight))
                for job_id, group in list(inflight.items()):
                    state = states.get(job_id, '')
                    if state and state not in TERMINAL_STATES:
                        _stamp(group, state)
                        unknown.pop(job_id, None)
                        continue
                    fragment = os.path.join(
                        jobs_dir, group['gid'], 'fragment.json')
                    if state in TERMINAL_STATES or os.path.isfile(fragment):
                        _deliver(group, state or 'COMPLETED')
                        _stamp(group, state or 'COMPLETED')
                        del inflight[job_id]
                        unknown.pop(job_id, None)
                        continue
                    unknown[job_id] = unknown.get(job_id, 0) + 1
                    if unknown[job_id] >= UNKNOWN_GRACE:
                        _deliver(group, 'UNKNOWN')
                        _stamp(group, 'UNKNOWN')
                        del inflight[job_id]
                        unknown.pop(job_id, None)

                _pulse()
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            _scancel(list(inflight) + [g.get('job_id') for g in groups])
            for job_id, group in inflight.items():
                _deliver(group, 'CANCELLED')
                _stamp(group, 'CANCELLED')
            inflight.clear()
            for group in groups:
                if not _terminal(group):
                    _deliver(group, 'CANCELLED')
                    _stamp(group, 'CANCELLED')
            interrupted['value'] = True
            return

    def _on_signal(signum, frame):
        raise KeyboardInterrupt

    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[signum] = signal.signal(signum, _on_signal)
        except (ValueError, OSError):
            pass
    try:
        for wave, serial in ((wave_one, False),
                             (bench_parallel, False),
                             (bench_serial, True)):
            if interrupted['value']:
                break
            _run_wave(wave, serial)
    finally:
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                pass

    # Every runnable task is delivered by the waves above; the loop
    # below is the guard for the paths that skip a collect (a wave cut
    # short by a signal, a group whose row directory was removed).
    for group in groups:
        for index, row in collect(work_dir, group, jobs_dir,
                                  group.get('state', '')):
            if index not in delivered:
                delivered.add(index)
                on_row(index, row)
    _write_ledger(ledger_path, work_dir, groups)
    prepared['interrupted'] = interrupted['value']
    return prepared


if __name__ == '__main__':
    sys.exit(main())
