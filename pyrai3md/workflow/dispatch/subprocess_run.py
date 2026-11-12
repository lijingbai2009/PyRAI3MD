"""
======================================================================
This module provides subprocess_run, the single-task local subprocess
execution function for the dispatch subsystem.

subprocess_run executes one batch task as an independent PyRAI3MD
child runtime in a local subprocess:

1. Create the child runtime directory
   '<cwd>/runtime/{skill_name}_runtime_{idx}/'.
2. Serialize the child PyRAIState into
   '{skill_name}_runtime_{idx}.rt' inside that directory.
3. Launch 'python3 -m pyrai3md {name}.rt' with the directory as
   working directory; the child runtime reloads the PyRAIState via
   the RuntimeLoader and runs the standard execution path, writing
   its result.json back into the directory.

The function is a module-level tool used by the Scheduler's
multiprocessing pool (picklable by reference).  Concurrency is
managed entirely by the Scheduler; this function handles one task.

Entry point: `subprocess_run(unit)`. No production path calls it today:
the dispatcher runs local terms in-process, and a remote child is written
and launched by `slurm_run` (same `.rt` file, submitted through sbatch).
It stays because it is the one place a full child runtime is launched
locally and read back inside a test -- the end-to-end smoke test in
`qualification/verification/test_subprocess_smoke.py` -- and that check
covers the `pyrai3md <name>.rt` entry a user can also run by hand.

Aug 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import os
import subprocess
import sys

from ...data.runtime.serializer import save_state
from ...data.runtime.result import Result
from .runtime import (
    RuntimeDirectoryManager, package_source_root, resolve_skill_name,
)


def subprocess_run(unit):
    """Execute one task as an independent local subprocess.

    Creates the child runtime directory, serializes the state,
    launches 'python3 -m pyrai3md {name}.rt', waits for completion,
    and collects result.json.

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
        If the child runtime exits without producing result.json.
    """
    state = unit['state']

    # Set up runtime directory manager
    skill_name = resolve_skill_name(state)
    manager = RuntimeDirectoryManager(skill_name=skill_name)
    idx = manager.next_index()
    runtime_dir = manager.create(idx)
    runtime_name = manager.runtime_name(idx)
    runtime_file = manager.runtime_file(idx)

    # Serialize the child state
    save_state(state, runtime_file,
               runtime_idx=manager.runtime_idx(idx))

    # Build the subprocess command
    python = sys.executable or 'python3'
    rt_basename = os.path.basename(runtime_file)
    command = [python, '-m', 'pyrai3md', rt_basename]

    # Set up environment: inherit parent env, prepend PYTHONPATH
    # so the child finds the same pyrai3md package
    environment = dict(os.environ)
    existing_path = environment.get('PYTHONPATH', '')
    environment['PYTHONPATH'] = (
        package_source_root()
        + (os.pathsep + existing_path if existing_path else '')
    )

    # Launch subprocess and wait for completion
    log_path = os.path.join(runtime_dir, 'runtime.log')
    with open(log_path, 'w') as log_file:
        proc = subprocess.run(
            command, cwd=runtime_dir, env=environment,
            stdout=log_file, stderr=subprocess.STDOUT)

    # Collect result
    result_file = manager.result_file(idx)
    if not os.path.isfile(result_file):
        raise RuntimeError(
            "Dispatched runtime %s did not produce result.json "
            "(see %s)" % (runtime_name, runtime_dir))
    with open(result_file, 'r') as handler:
        data = json.load(handler)
    return Result.from_dict(data)
