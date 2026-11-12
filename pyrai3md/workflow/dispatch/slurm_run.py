"""
======================================================================
This module provides slurm_run, the single-task Slurm execution
function for the dispatch subsystem.

slurm_run executes one batch task as an independent PyRAI3MD child
runtime submitted to a Slurm cluster:

1. Create the child runtime directory
   '<cwd>/runtime/{skill_name}_runtime_{idx}/'.
2. Serialize the child PyRAIState into
   '{skill_name}_runtime_{idx}.rt' inside that directory.
3. Generate a slurm.sh script: '#SBATCH' directives from the slurm
   section of the environment config, 'export' lines for the
   environment variables, then 'python3 -m pyrai3md {name}.rt'.
4. Submit with sbatch, poll sacct until terminal state, and
   collect the result.json written by the child runtime.

The function is a module-level tool used by the Scheduler's
multiprocessing pool (picklable by reference).  Concurrency is
managed entirely by the Scheduler; this function handles one task.

Aug 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import os
import subprocess
import time

from ...data.runtime.serializer import save_state
from ...data.runtime.result import Result
from ...kernel.sbatch import render_sbatch_directives
from .runtime import (
    RuntimeDirectoryManager, flatten_export_variables, package_source_root,
    resolve_skill_name,
)


# sacct states that terminate the polling loop
TERMINAL_STATES = {
    'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL',
    'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE',
}

# Default seconds between sacct polls
_DEFAULT_POLL_INTERVAL = 30.0
# A scheduler that stops answering must not hang the parent forever. The
# wall-clock budget catches a job the accounting system never settles;
# the unknown streak catches a record that never appears at all.
_POLL_TIMEOUT_S = float(
    os.environ.get('PYRAI3MD_SLURM_POLL_TIMEOUT', 86400.0))
_MAX_UNKNOWN_POLLS = int(
    os.environ.get('PYRAI3MD_SLURM_MAX_UNKNOWN', 120))


def slurm_run(unit):
    """Execute one task via Slurm sbatch submission.

    Creates the child runtime directory, serializes the state,
    generates slurm.sh, submits via sbatch, polls sacct until
    terminal state, and collects result.json.

    Parameters
    ----------
    unit : dict
        Unit descriptor with 'state' (full PyRAIState).

    Returns
    -------
    Result
        Child runtime result.

    Raises
    ------
    RuntimeError
        If submission fails or the runtime ends without result.json.
    """
    state = unit['state']

    # Set up runtime directory manager; a batch-stamped identity
    # (allocated by the Dispatcher when the batch carries more than
    # one unit) is consumed as is — including the stamped naming
    # policy (root/name_format), so the slurm layout matches the
    # local one — while a lone remote task keeps self-allocating the
    # next free index with the default layout.
    skill_name = resolve_skill_name(state)
    manager = RuntimeDirectoryManager(
        skill_name=skill_name,
        root=unit.get('runtime_root'),
        name_format=unit.get('runtime_name_format'))
    stamped_idx = unit.get('runtime_idx')
    if stamped_idx:
        idx = int(str(stamped_idx).rsplit('_', 1)[1])
    else:
        idx = manager.next_index()
    runtime_dir = manager.create(idx)
    runtime_name = unit.get('runtime_name') or manager.runtime_name(idx)
    runtime_file = manager.runtime_file(idx)

    # Serialize the child state
    save_state(state, runtime_file,
               runtime_idx=manager.runtime_idx(idx))

    # Generate slurm.sh
    script = _build_script(state, manager, idx)
    script_path = os.path.join(runtime_dir, 'slurm.sh')
    with open(script_path, 'w') as handler:
        handler.write(script)

    # Submit via sbatch
    try:
        output = subprocess.run(
            ['sbatch', 'slurm.sh'], cwd=runtime_dir,
            capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise RuntimeError(
            "sbatch submission failed for %s: %s" % (
                runtime_name, error))
    tokens = output.stdout.split()
    if not tokens:
        raise RuntimeError(
            "sbatch returned no job id for %s" % runtime_name)
    job_id = tokens[-1]

    # Poll sacct until terminal state
    poll_interval = _DEFAULT_POLL_INTERVAL
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    unknown_streak = 0
    while True:
        job_state = _query_slurm_state(job_id)
        if job_state in TERMINAL_STATES:
            break
        if job_state == 'UNKNOWN':
            unknown_streak += 1
            if unknown_streak >= _MAX_UNKNOWN_POLLS:
                raise RuntimeError(
                    "Slurm job %s for %s stayed UNKNOWN for %d polls; "
                    "sacct never reported a state."
                    % (job_id, runtime_name, unknown_streak))
        else:
            unknown_streak = 0
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Slurm job %s for %s did not reach a terminal state within "
                "%.0f s; last state %s."
                % (job_id, runtime_name, _POLL_TIMEOUT_S, job_state))
        time.sleep(poll_interval)

    # Collect result
    result_file = manager.result_file(idx)
    if not os.path.isfile(result_file):
        raise RuntimeError(
            "Dispatched runtime %s did not produce result.json: %s" % (
                runtime_name, _child_failure(runtime_dir)))
    try:
        with open(result_file, 'r') as handler:
            data = json.load(handler)
    except ValueError as error:
        # A result document that exists but does not parse means the child
        # died on the way out. Say so, and quote what the child said, rather
        # than handing up a JSON position that names the parent's reader
        # instead of the child's failure.
        raise RuntimeError(
            "Dispatched runtime %s wrote an unreadable result.json (%s): %s"
            % (runtime_name, error, _child_failure(runtime_dir)))
    return Result.from_dict(data)


def _child_failure(runtime_dir, limit=800):
    """The tail of the child's Slurm error file, for the parent's message.

    The parent's only view of a dispatched runtime is what the runtime left
    in its directory, and the engine's own words -- the traceback that killed
    it -- are in slurm-<job>.err. Without them a failed child is reported as
    a missing or malformed result, which is a statement about the parent's
    reading, not about what happened.

    Parameters
    ----------
    runtime_dir : str
        Directory holding the submitted runtime's files.
    limit : int
        Maximum characters of the tail to carry into the message.

    Returns
    -------
    str
        The quoted tail, or a pointer to the directory when there is none.
    """
    candidates = sorted(
        (os.path.join(runtime_dir, name)
         for name in os.listdir(runtime_dir)
         if name.startswith('slurm-') and name.endswith('.err'))
        if os.path.isdir(runtime_dir) else [],
        key=os.path.getmtime,
    )
    if not candidates:
        return 'no slurm-*.err under %s' % runtime_dir
    try:
        with open(candidates[-1], 'r', errors='replace') as handler:
            text = handler.read().strip()
    except OSError as error:
        return 'cannot read %s (%s)' % (candidates[-1], error)
    return '%s: %s' % (os.path.basename(candidates[-1]), text[-limit:])


def _build_script(state, manager, idx, entry='.rt'):
    """Generate the slurm.sh content for one runtime.

    Parameters
    ----------
    entry : str
        Runtime entry suffix: '.rt' for a whole child runtime, '.rtt' for
        one remote term worker. The SBATCH directives, the user shell
        section and the environment exports are identical for both.

    Structure:
    1. '#SBATCH' directives from the environment slurm section.
    2. 'export' lines for the environment variables.
    3. The runtime launch command.

    Parameters
    ----------
    state : PyRAIState
        Child state carrying the environment snapshot.
    manager : RuntimeDirectoryManager
        Runtime directory manager.
    idx : int
        Runtime index.

    Returns
    -------
    str
        Complete submission script.
    """
    env_model = getattr(getattr(state, 'pmd', None),
                        'environment', None)

    lines = ['#!/bin/bash']

    # SBATCH directives
    if env_model is not None:
        lines.extend(render_sbatch_directives(
            dict(env_model.slurm or {})))

        # User shell commands (module loads, library paths). The
        # environment shell is the framework dispatch layer's own
        # section (skills carry their own), and a Slurm job starts in a
        # fresh login shell: the parent's loaded modules are not there,
        # so the commands that made the parent able to run the engine
        # must be replayed in the child script.
        shell = str(env_model.shell or '')
        if shell:
            lines.append('')
            lines.append(shell.strip())

    # Environment variable exports.  The child runtime must
    # import the same pyrai3md as the parent process, so the
    # package source root is prepended to PYTHONPATH first.
    existing_path = os.environ.get('PYTHONPATH', '')
    python_path = package_source_root() + (
        os.pathsep + existing_path if existing_path else '')
    export_lines = ['export PYTHONPATH=%s' % python_path]

    for key, value in flatten_export_variables(state).items():
        export_lines.append('export %s=%s' % (key, value))

    lines.append('')
    lines.extend(export_lines)

    runtime_name = manager.runtime_name(idx)
    lines.append('')
    lines.append('python3 -m pyrai3md %s%s' % (runtime_name, entry))
    lines.append('')
    return '\n'.join(lines)


def _query_slurm_state(job_id):
    """Query sacct for job state.

    Parameters
    ----------
    job_id : str
        Slurm job id.

    Returns
    -------
    str
        State name (upper case, suffixes stripped), or 'UNKNOWN'
        when sacct has no record yet.
    """
    try:
        output = subprocess.run(
            ['sacct', '-j', str(job_id), '-o', 'State', '-n', '-P'],
            capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'UNKNOWN'
    for line in output.stdout.splitlines():
        state = line.strip().split()[0].upper() if line.strip() else ''
        if state:
            return state
    return 'UNKNOWN'
