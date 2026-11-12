"""
======================================================================
This module provides the remote per-term worker (docs R3, r3e): the
single backend-term Slurm execution path used when a backend-terms
descriptor declares remotetask=True.

A term worker executes exactly ONE backend term of one evaluation and
returns the unified TaskOutcome; it never runs MD, never sums the
composite, never commits the parent state and never expands the full
task queue. This is the term-level analogue of slurm_run (which ships a
whole child PyRAIState runtime); here the payload is a self-contained
term descriptor rebuilt from scratch on the remote side.

Session policy (first version): task-cache rebuild. Every remote term
cold-starts: the payload carries the task descriptor (api + an
independent config copy), the frozen step geometry / electronic
request and the execution identity, so the worker rebuilds the backend
through resolve_api and runs one calculate(). The wavefunction cache
rebuild cost is explicit and accepted; a reusable worker session is a
later optimization (deletion gate: when remote-term-count x
cold-start cost becomes a measured bottleneck).

Layout (relative to the parent runtime working directory):

    runtime/
      {term_skill}_runtime_{idx}/          per-term runtime directory
        {term_skill}_runtime_{idx}.rtt     pickled term payload
        term_outcome.json                  unified TaskOutcome
        slurm.sh                           submission script

The per-term skill name encodes center_index / task_slot /
evaluation_id so concurrent terms of one evaluation never contend for
the same runtime index or directory.

Entry: 'python3 -m pyrai3md {name}.rtt' -> pmd.run() -> run_term_worker
(a dedicated branch, NOT _execute_runtime_file, which would run the
whole calculator.run + composite sum + set_result).

The outcome contract is built inline: the worker side carries the
extracted task_id/evaluation_id of the remote term, not a work-item
dict, so terms.make_outcome (same shape, item-driven) does not apply
here. resolve_api is imported lazily inside the worker-side rebuild
only.

Sep 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import os
import pickle
import subprocess
import time

import numpy as np

from ...skills.query import resolve_api
from ...utilities.serialization import json_safe, write_json
from .terms import backend_call_kwargs
from .runtime import RuntimeDirectoryManager, resolve_skill_name
from .slurm_run import (
    TERMINAL_STATES, _DEFAULT_POLL_INTERVAL, _build_script, _query_slurm_state,
)


# Term worker payload / entry contract
TERM_WORKER_SUFFIX = '.rtt'
TERM_OUTCOME_FILE = 'term_outcome.json'
# 1.1: the geometry section carries the frozen periodic-cell snapshot
# ('cell') consumed by backend.calculate(cell=...). Dispatcher and
# workers must run the same payload version. The dict travels opaquely,
# so growing it does not move the version: 'image' joined it (the counts
# of cell lengths the coordinate wrap made an atom cross) and a worker
# that never sees the key simply hands its backend no crossings, which is
# the same reading as a pre-image run.
TERM_PAYLOAD_VERSION = '1.1'

# Result keys whose JSON list form must be restored to ndarray on the
# parent side so the composite gradient scatter-add sees arrays.
_TERM_ARRAY_KEYS = (
    'gradient', 'nac', 'soc', 'dipole', 'transition_dipole',
    'hessian', 'coupling', 'nacme', 'socme',
)


def _term_skill_name(base_skill, identity):
    """Encode the per-term identity into the runtime directory name.

    Concurrent terms of one evaluation differ by center_index /
    task_slot / evaluation_id; folding them into the skill name keeps
    every term in its own runtime directory and index space.

    Parameters
    ----------
    base_skill : str
        Resolved skill name of the term.
    identity : dict
        Execution identity (center_index / task_slot / evaluation_id).

    Returns
    -------
    str
        Per-term skill name for RuntimeDirectoryManager.
    """
    identity = identity or {}
    parts = [str(base_skill or 'pyrai3md')]
    center = identity.get('center_index')
    slot = identity.get('task_slot')
    evaluation_id = identity.get('evaluation_id')
    if center is not None:
        parts.append('c%s' % center)
    if slot is not None:
        parts.append('s%s' % slot)
    if evaluation_id is not None:
        parts.append('e%s' % evaluation_id)
    return '_'.join(parts)


def build_term_payload(work_item, task, runtime_idx=None):
    """Build the self-contained payload of one remote backend term.

    The payload carries everything the worker needs to rebuild the
    backend and run one calculate(): the task descriptor (api + an
    independent config copy), the frozen step geometry, the electronic
    request and the execution identity. It never carries the live
    backend instance or the parent PyRAIState.

    The remote term group flags are frozen to False (one remote
    boundary): a task that still declares remotetask=True means an
    undeclared nested remote submission and is refused here, before
    any directory is created or job submitted.

    Parameters
    ----------
    work_item : dict
        Unified work item (terms.make_work_item).
    task : dict
        Task descriptor of the term (api / config / static identity).
    runtime_idx : str or None
        Runtime suffix appended to the backend execution id.

    Returns
    -------
    dict
        Picklable term payload.

    Raises
    ------
    RuntimeError
        If the task declares remotetask=True (nested remote refused).
    """
    from copy import deepcopy

    if task.get('remotetask'):
        raise RuntimeError(
            "Remote term worker refuses a nested remote submission: "
            "task '%s' still declares remotetask=True. The remote term "
            "group flags must be frozen to False at one boundary."
            % task.get('task_id', '?'))

    static_id = task.get('static_id') or task.get('task_id') or (
        "%s_%s" % (task.get('role', 'task'), task.get('skill_name', '')))
    execution_id = ("%s_%s" % (static_id, runtime_idx)
                    if runtime_idx else str(static_id))

    coords = np.asarray(work_item.get('local_coords'), dtype=np.float64)
    return {
        'schema_version': TERM_PAYLOAD_VERSION,
        'task': {
            'api': task.get('api'),
            'skill_name': task.get('skill_name'),
            'capabilities': list(task.get('capabilities') or []),
            'config': deepcopy(task.get('config')),
            'static_id': static_id,
            'task_id': task.get('task_id', ''),
            'region': task.get('region', 'all'),
            'region_indices': list(task.get('region_indices') or []),
            # Frozen: one remote boundary, no nested remote group.
            'remotetask': False,
            'multitask': False,
        },
        'geometry': {
            'local_atoms': list(work_item.get('local_atoms') or []),
            'local_coords': coords.copy(),
            # Frozen periodic-cell snapshot (plain dict or None); the
            # worker-side calculate() consumes it so the remote term
            # evaluates with the same cell as the parent step.
            'cell': work_item.get('cell'),
        },
        'request': {
            'target_states': work_item.get('target_states'),
            'nac_pairs': work_item.get('nac_pairs'),
            'soc_pairs': work_item.get('soc_pairs'),
            'point_charges': work_item.get('point_charges'),
            'external_potential': work_item.get('external_potential'),
            'qmmm_inputs': work_item.get('qmmm_inputs'),
        },
        'identity': dict(work_item.get('identity') or {}),
        'audit_key': {
            'backend_name': work_item.get('backend_name', ''),
            'runner': 'term_worker',
            'execution_mode': work_item.get('execution_mode', ''),
            'profile_semantics': work_item.get('profile_semantics', ''),
        },
        'execution_id': execution_id,
    }


def write_term_payload(payload, path):
    """Pickle a term payload to disk.

    Parameters
    ----------
    payload : dict
        Term payload from build_term_payload().
    path : str
        Destination '.rtt' file path.
    """
    with open(path, 'wb') as handler:
        pickle.dump(payload, handler)


def read_term_payload(path):
    """Load a pickled term payload.

    Parameters
    ----------
    path : str
        Source '.rtt' file path.

    Returns
    -------
    dict
        The term payload.
    """
    with open(path, 'rb') as handler:
        return pickle.load(handler)


def _rebuild_backend(task, execution_id):
    """Rebuild the backend of one term on the worker side.

    `resolve_api` lives in `skills/query.py`, so the dispatch package
    does not import the workflow package at module load.

    Parameters
    ----------
    task : dict
        Frozen task descriptor (api + config).
    execution_id : str
        Backend execution namespace.

    Returns
    -------
    object
        A fresh backend instance owning an independent config copy.
    """
    from copy import deepcopy

    backend_cls = resolve_api(task.get('api'))
    config = task.get('config')
    if config is not None:
        config = deepcopy(config)
    try:
        return backend_cls(config=config, execution_id=execution_id)
    except TypeError:
        # Backends without execution namespace support (plain mocks).
        return backend_cls(config)


def _make_term_outcome(task_id, evaluation_id, raw, success, error=None):
    """Build the unified outcome dict (matches terms.make_outcome).

    The worker side holds only the extracted identity of the remote
    term, so this takes the ids directly instead of a work-item dict;
    the returned shape is the single outcome contract shared by serial,
    thread and remote execution.

    Parameters
    ----------
    task_id : str
        Task identity routing the outcome to its archive slot.
    evaluation_id : object
        Evaluation identity guarding against stale/duplicate results.
    raw : dict or None
        Backend result payload ({} on failure).
    success : bool
        Whether the term succeeded.
    error : str or None
        Diagnostic message on failure.

    Returns
    -------
    dict
        Unified outcome {task_id, evaluation_id, raw, success, error}.
    """
    return {
        'task_id': task_id,
        'evaluation_id': evaluation_id,
        'raw': raw if raw is not None else {},
        'success': bool(success),
        'error': error,
    }


def execute_term_locally(payload):
    """Execute exactly one backend term from a payload (worker side).

    Recursion guard: a term worker never dispatches a remote term
    group. The payload task flags are frozen to False; a payload that
    still declares remotetask=True is refused here.

    Parameters
    ----------
    payload : dict
        Term payload from build_term_payload().

    Returns
    -------
    dict
        Unified outcome of the single term.

    Raises
    ------
    RuntimeError
        If the payload task declares remotetask=True (nested remote).
    """
    task = payload.get('task') or {}
    if task.get('remotetask'):
        raise RuntimeError(
            "Remote term worker refuses a nested remote submission: "
            "payload task '%s' declares remotetask=True."
            % task.get('task_id', '?'))

    identity = payload.get('identity') or {}
    task_id = task.get('task_id', '')
    evaluation_id = identity.get('evaluation_id')
    geometry = payload.get('geometry') or {}
    request = payload.get('request') or {}

    try:
        backend = _rebuild_backend(task, payload.get('execution_id'))
        kwargs = backend_call_kwargs(
            backend,
            point_charges=request.get('point_charges'),
            target_states=request.get('target_states'),
            cell=geometry.get('cell'),
            external_potential=request.get('external_potential'),
            qmmm_inputs=request.get('qmmm_inputs'),
            nac_pairs=request.get('nac_pairs'),
            soc_pairs=request.get('soc_pairs'),
        )
        raw = backend.calculate(
            geometry.get('local_atoms'),
            geometry.get('local_coords'), **kwargs)
        return _make_term_outcome(task_id, evaluation_id, raw, True)
    except Exception as exc:  # captured into the unified outcome
        return _make_term_outcome(task_id, evaluation_id, {}, False,
                                  str(exc))


def _json_safe(data):
    """Recursively convert numpy objects into JSON-serializable form.

    Kept as a name here because the dispatch tests reach for it; the walk
    itself is shared with the command-line result writer, which had its own
    copy that stopped at the top level of a dictionary. That copy is what
    crashed a dispatched md child part-way through its result document.

    Parameters
    ----------
    data : object
        Arbitrary result structure.

    Returns
    -------
    object
        JSON-safe copy.
    """
    return json_safe(data)


def _restore_arrays(raw):
    """Restore known array-typed result keys from JSON lists to ndarray.

    The composite gradient scatter-add and coupling projections expect
    ndarrays; the JSON round-trip turns them into nested lists, so the
    parent side rebuilds them here.

    Parameters
    ----------
    raw : dict
        Result payload read back from term_outcome.json.

    Returns
    -------
    dict
        Payload with array-typed keys restored to ndarray.
    """
    if not isinstance(raw, dict):
        return raw
    restored = {}
    for key, value in raw.items():
        if key in _TERM_ARRAY_KEYS and isinstance(value, list):
            restored[key] = np.asarray(value, dtype=np.float64)
        else:
            restored[key] = value
    return restored


def run_term_worker(payload_path, outcome_path=None):
    """Remote term worker entry: execute one term and write its outcome.

    Invoked by 'python3 -m pyrai3md {name}.rtt' through pmd.run(). It
    reads the pickled payload, executes exactly one backend term and
    writes the unified TaskOutcome to term_outcome.json in the current
    directory (the per-term runtime directory) for collection by the
    parent.

    Parameters
    ----------
    payload_path : str
        Path to the '.rtt' payload file.
    outcome_path : str or None
        Destination of the outcome JSON; defaults to
        '<cwd>/term_outcome.json'.

    Returns
    -------
    dict
        The unified outcome written to disk.
    """
    payload = read_term_payload(payload_path)
    outcome = execute_term_locally(payload)
    if outcome_path is None:
        outcome_path = os.path.join(os.getcwd(), TERM_OUTCOME_FILE)
    write_json(outcome_path, _json_safe(outcome))
    return outcome


def _build_term_script(state, manager, idx):
    """Generate the slurm.sh content for one remote term.

    Reuses the SBATCH directives and environment export skeleton of
    slurm_run._build_script; the launch command targets the '.rtt'
    term-worker entry instead of the '.rt' full-runtime entry.

    Parameters
    ----------
    state : PyRAIState
        Parent state carrying the environment snapshot.
    manager : RuntimeDirectoryManager
        Per-term runtime directory manager.
    idx : int
        Runtime index.

    Returns
    -------
    str
        Complete submission script.
    """
    return _build_script(state, manager, idx, entry=TERM_WORKER_SUFFIX)


def slurm_term_run(work_item, task, state, descriptor=None):
    """Execute one backend term via Slurm and collect its TaskOutcome.

    Creates the per-term runtime directory, writes the pickled payload
    and slurm.sh, submits via sbatch, polls sacct until terminal state
    and reads back term_outcome.json. A missing outcome file or a
    non-completed job is reported as a failed outcome (success=False)
    so the caller's collection protocol raises with the diagnostics.

    Parameters
    ----------
    work_item : dict
        Unified work item of the term.
    task : dict
        Task descriptor of the term.
    state : PyRAIState
        Parent state (environment snapshot for the script).
    descriptor : dict or None
        Backend-terms execution descriptor (optional runtime root /
        name format overrides).

    Returns
    -------
    dict
        Unified outcome of the single remote term.

    Raises
    ------
    RuntimeError
        If the remote configuration is missing or sbatch submission
        fails (before any polling).
    """
    descriptor = descriptor or {}
    identity = work_item.get('identity') or {}
    task_id = work_item.get('task_id', '')
    evaluation_id = identity.get('evaluation_id')

    env_model = getattr(getattr(state, 'pmd', None), 'environment', None)
    if env_model is None:
        raise RuntimeError(
            "Remote term execution requires an environment snapshot "
            "(state.pmd.environment) for the Slurm directives and "
            "exports; none is present for task '%s'." % task_id)

    base_skill = task.get('skill_name') or resolve_skill_name(state)
    term_skill = _term_skill_name(base_skill, identity)
    manager = RuntimeDirectoryManager(
        skill_name=term_skill,
        root=descriptor.get('runtime_root'),
        name_format=descriptor.get('runtime_name_format'))
    idx = manager.next_index()
    runtime_dir = manager.create(idx)
    runtime_name = manager.runtime_name(idx)
    runtime_idx = manager.runtime_idx(idx)

    payload = build_term_payload(work_item, task, runtime_idx)
    payload_path = os.path.join(
        runtime_dir, runtime_name + TERM_WORKER_SUFFIX)
    write_term_payload(payload, payload_path)

    script = _build_term_script(state, manager, idx)
    script_path = os.path.join(runtime_dir, 'slurm.sh')
    with open(script_path, 'w') as handler:
        handler.write(script)

    try:
        output = subprocess.run(
            ['sbatch', 'slurm.sh'], cwd=runtime_dir,
            capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise RuntimeError(
            "sbatch submission failed for remote term %s: %s"
            % (runtime_name, error))
    tokens = output.stdout.split()
    if not tokens:
        raise RuntimeError(
            "sbatch returned no job id for remote term %s" % runtime_name)
    job_id = tokens[-1]

    while True:
        job_state = _query_slurm_state(job_id)
        if job_state in TERMINAL_STATES:
            break
        time.sleep(_DEFAULT_POLL_INTERVAL)

    outcome_file = os.path.join(runtime_dir, TERM_OUTCOME_FILE)
    if not os.path.isfile(outcome_file):
        return _make_term_outcome(
            task_id, evaluation_id, {}, False,
            "remote term %s ended (%s) without %s (check slurm-*.out "
            "in %s)" % (runtime_name, job_state, TERM_OUTCOME_FILE,
                        runtime_dir))
    with open(outcome_file, 'r') as handler:
        data = json.load(handler)
    data['raw'] = _restore_arrays(data.get('raw') or {})
    return data


def execute_remote_terms(work_items, tasks, state, descriptor=None):
    """Submit every term of one evaluation remotely and collect outcomes.

    Sequential submission mirroring the serial fail-fast policy: the
    first failing term stops the group (no further cluster jobs are
    submitted). The outcomes stay index-aligned with the work items so
    the caller's routing/summation order is unchanged. The failure
    policy (raising with the collected diagnostics) is applied by the
    caller through terms.apply_failure_policy.

    Parameters
    ----------
    work_items : list of dict
        Unified work items of the evaluation.
    tasks : list of dict
        Task descriptors parallel to work_items.
    state : PyRAIState
        Parent state.
    descriptor : dict or None
        Backend-terms execution descriptor.

    Returns
    -------
    list of dict
        Index-aligned outcomes; shorter than the input only when a term
        failed (fail-fast), in which case the caller raises.
    """
    outcomes = []
    for work_item, task in zip(work_items, tasks):
        outcome = slurm_term_run(work_item, task, state, descriptor)
        outcomes.append(outcome)
        if not outcome.get('success', False):
            break
    return outcomes
