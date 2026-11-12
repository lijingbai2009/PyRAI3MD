"""
======================================================================
Term executor of the dispatch layer.

Provides the unified serial and thread execution of the backend terms
(the per-task single-point calculations of one backend role) with
controlled submission and a single failure protocol.

This is the term axis, orthogonal to the runtime axis of Dispatcher: a
Dispatcher unit is a materialized child PyRAIState that can be deep-copied,
pickled into a worker process and given its own runtime directory, while a
term carries a live backend instance and the geometry of one region, so it
can only run in-process (serial or thread) or as a dedicated remote term
worker (dispatch/term_worker.py). The two axes are composed, never
substituted: the base node keeps multitask=False while its terms may still
run in a thread pool.

Unified contracts (dev/MULTICENTER_DEVELOPMENT.md, R3):
- work_item: one backend term to execute. It carries the
  identity/routing fields (task_id, task_slot, center_index,
  evaluation_id, identity), the execution payload (backend,
  local_atoms, local_coords, target_states, nac_pairs, soc_pairs) and
  the thread-safety contract (backend_name, runner, execution_mode,
  profile_semantics) it is asked about. Build one with
  make_work_item().
- outcome: one term result {task_id, evaluation_id, raw, success,
  error}. Serial and thread modes return the same shape and keep the
  work-item order (index-aligned), so an out-of-order thread
  completion never changes the archive position or the summation
  order (the caller routes by task_id).

Design rules:
- The executor does not hold PyRAIState; it receives injected work
  items and returns outcomes.
- All file paths use absolute paths (no chdir).
- Backend instances and configs are fully independent per task.
- Default is serial (max_workers=1). Thread mode asks each backend,
  through the skill's own check_thread_safe(), whether the contract it
  is handed may run concurrently; the skill refuses by name when it
  cannot, and the executor never degrades to serial on its behalf. A
  backend that defines no check_thread_safe() runs concurrently.
- Controlled submission: the first failing term stops the evaluation.
  Serial short-circuits the remaining terms; thread cancels the
  unstarted futures, drains the running ones and reports every
  outcome. The executor returns outcomes and never raises on a term
  failure; raising is the caller's failure policy
  (execute_with_failure_handling).

Sep 07 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from ...kernel.logger import get_logger

logger = get_logger('pyrai3md.dispatch')


def audit_key(item):
    """Extract the thread-safety contract of one work item.

    The tuple is what the backend is asked about through
    check_thread_safe() and what the remote term worker records
    alongside its term; no framework-side table consults it any more.
    Kept as a public export of the term contract.

    Parameters
    ----------
    item : dict
        Work item carrying backend_name / runner / execution_mode /
        profile_semantics.

    Returns
    -------
    tuple
        (backend_name, runner, execution_mode, profile_semantics).
    """
    return (item.get('backend_name', ''), item.get('runner', ''),
            item.get('execution_mode', ''),
            item.get('profile_semantics', ''))


def check_thread_safe(item):
    """Ask one work item's backend whether it may share a thread pool.

    Thread safety is a property of the skill's own code: only the skill
    knows which process globals (a working directory, an environment
    variable, an interpreter's device settings) its calculation touches.
    The framework therefore does not decide it; it hands the contract
    to the backend and propagates whatever the backend refuses.

    A backend that defines no ``check_thread_safe`` runs concurrently:
    that is the design default, since every skill offers both its local
    and its slurm mode and opts out by name when one of them cannot be
    reentered. The skill raises, so the message states the skill's own
    reason; the executor never degrades to serial on its behalf.

    Parameters
    ----------
    item : dict
        Work item; its 'backend' is the skill's backend instance.

    Raises
    ------
    Exception
        Whatever the backend raises to refuse the contract.
    """
    backend = item.get('backend')
    check = getattr(backend, 'check_thread_safe', None)
    if check is None:
        return
    check(item.get('runner', ''),
          item.get('execution_mode', ''),
          item.get('profile_semantics', ''))


def make_work_item(task_id, backend, local_atoms, local_coords,
                   task_slot=None, center_index=None, evaluation_id=None,
                   target_states=None, nac_pairs=None, soc_pairs=None,
                   point_charges=None, external_potential=None, cell=None,
                   qmmm_inputs=None,
                   identity=None, backend_name='', runner='in_process',
                   execution_mode='', profile_semantics=''):
    """Build one unified backend-term work item.

    This is the sole constructor of the work_item contract consumed by
    TaskExecutor.execute(); the calculator (r3d) and the remote term
    worker (r3e) both stage their terms through it so serial, thread
    and remote execution share one payload shape.

    Parameters
    ----------
    task_id : str
        Task identity (routes the outcome back to its archive slot).
    backend : object
        Independent backend instance owning this term.
    local_atoms : list of str
        Materialised local region atom symbols.
    local_coords : ndarray
        Materialised local region coordinates.
    task_slot : int or None
        Persistent per-task slot (backend/config isolation identity).
    center_index : int or None
        Center partition index (None for the full-system baseline).
    evaluation_id : object or None
        Evaluation identity (guards against stale/duplicate outcomes).
    target_states : list of int or None
        Per-step target states handed to the backend.
    nac_pairs : list or None
        Non-adiabatic coupling pairs requested by this term.
    soc_pairs : list or None
        Spin-orbit coupling pairs requested by this term.
    point_charges : ndarray or None
        Electrostatic embedding field [q, x, y, z] (Angstrom) handed to
        the backend for a high-level task embedded in its center's mid
        shell; None for every other term (no embedding).
    external_potential : ndarray or None
        Per-atom [V, Fx, Fy, Fz] external potential/field (atomic units)
        handed to a backend that supports QMMM potential embedding.
    qmmm_inputs : dict or None
        Per-step QM/MM payload for the LAMMPS ``qmmm_energy`` task
        (``q_qm``, and the native ``v_mm``/``e_mm`` from the field task).
        None for every other task.
    cell : dict or None
        Frozen periodic-cell snapshot of this evaluation
        ({'lattice_vector', 'cell', 'pbc', 'volume'}, plain
        list/float types only); None when the system carries no cell.
        The authoritative per-evaluation cell channel: the backend
        consumes it before writing its input, so a variable-cell step
        never lags one step behind the state sync.
    identity : dict or None
        Execution identity (trajectory / center / task_slot /
        evaluation) used for runtime directory naming.
    backend_name : str
        Contract field: backend identifier.
    runner : str
        Contract field: execution runner ('in_process' default).
    execution_mode : str
        Contract field: execution mode.
    profile_semantics : str
        Contract field: resolved profile semantics.

    Returns
    -------
    dict
        Unified work item.
    """
    return {
        'task_id': task_id,
        'task_slot': task_slot,
        'center_index': center_index,
        'evaluation_id': evaluation_id,
        'backend': backend,
        'local_atoms': local_atoms,
        'local_coords': local_coords,
        'target_states': target_states,
        'nac_pairs': nac_pairs,
        'soc_pairs': soc_pairs,
        'point_charges': point_charges,
        'external_potential': external_potential,
        'qmmm_inputs': qmmm_inputs,
        'cell': cell,
        'identity': dict(identity or {}),
        'backend_name': backend_name,
        'runner': runner,
        'execution_mode': execution_mode,
        'profile_semantics': profile_semantics,
    }


def make_outcome(item, raw, success, error=None):
    """Build one unified outcome of a work item.

    Parameters
    ----------
    item : dict
        Work item (supplies task_id / evaluation_id).
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
        'task_id': item.get('task_id', ''),
        'evaluation_id': item.get('evaluation_id'),
        'raw': raw if raw is not None else {},
        'success': bool(success),
        'error': error,
    }


class TaskExecutor:
    """Serial/thread executor of backend terms.

    Does not hold PyRAIState; receives injected work items and returns
    one index-aligned outcome per item (the same shape for both modes).

    Parameters
    ----------
    mode : str
        Execution mode: 'serial' (default) or 'thread'.
    max_workers : int
        Maximum number of thread workers (default 1, only used
        when mode='thread').
    """

    def __init__(self, mode='serial', max_workers=1):
        if mode not in ('serial', 'thread'):
            raise ValueError(
                "Unknown executor mode '%s'; expected 'serial' "
                "or 'thread'." % mode)
        if mode == 'thread' and max_workers < 1:
            raise ValueError(
                "max_workers must be >= 1, got %d." % max_workers)
        self.mode = mode
        self.max_workers = max_workers if mode == 'thread' else 1

    def execute(self, work_items):
        """Execute work items and return one outcome per item.

        The outcomes keep the work-item order (index-aligned), so an
        out-of-order thread completion never changes the archive
        position or the summation order. The executor never raises on
        a term failure: it reports success=False + error and lets the
        caller apply the failure policy.

        Parameters
        ----------
        work_items : list of dict
            Unified work items (see make_work_item / the module
            docstring).

        Returns
        -------
        list of dict
            One outcome per work item, index-aligned: {task_id,
            evaluation_id, raw, success, error}. Serial short-circuits
            after the first failure, so the list may be shorter than
            the input; thread always returns one outcome per item.

        Raises
        ------
        RuntimeError
            In thread mode, when a work item's backend execution
            contract has not passed the thread-safety audit (raised
            before any term runs; never degrades to serial).
        """
        work_items = list(work_items or [])
        if self.mode == 'serial':
            return self._execute_serial(work_items)
        return self._execute_thread(work_items)

    def _execute_serial(self, work_items):
        """Execute terms in order, short-circuiting the first failure.

        Serial execution reproduces the historical fail-fast behavior:
        the terms queued after the first failure are not run.
        """
        outcomes = []
        for item in work_items:
            outcome = _run_one(item)
            outcomes.append(outcome)
            if not outcome['success']:
                # Fail fast: skip the remaining terms; the caller
                # collects the diagnostics and raises.
                break
        return outcomes

    def _execute_thread(self, work_items):
        """Execute terms in a thread pool with controlled submission.

        Every item's backend is asked, before the pool starts, whether
        it may run concurrently (the skill decides and refuses; an
        unwilling backend fails here, never silently degrading to
        serial). On the first failing term the unstarted futures are
        cancelled and the running ones are drained; every outcome
        (completed, failed or cancelled) is reported index-aligned.
        """
        for item in work_items:
            check_thread_safe(item)

        outcomes = [None] * len(work_items)
        failure_seen = False
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_idx = {pool.submit(_run_one, item): idx
                             for idx, item in enumerate(work_items)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                if future.cancelled():
                    outcomes[idx] = make_outcome(
                        work_items[idx], {}, False,
                        'cancelled: an earlier backend term failed')
                    continue
                outcome = future.result()
                outcomes[idx] = outcome
                if not outcome['success'] and not failure_seen:
                    failure_seen = True
                    # Controlled submission: cancel the terms that have
                    # not started yet; the running ones drain when the
                    # with-block exits.
                    for pending in future_to_idx:
                        if not pending.done():
                            pending.cancel()
        # The with-block drained the running futures; fill any slot
        # left unset (defensive) with a cancelled outcome so the result
        # stays index-aligned with the input.
        for idx, outcome in enumerate(outcomes):
            if outcome is None:
                outcomes[idx] = make_outcome(
                    work_items[idx], {}, False,
                    'cancelled: an earlier backend term failed')
        return outcomes

    def __repr__(self):
        return "TaskExecutor(mode=%r, max_workers=%d)" % (
            self.mode, self.max_workers)


def backend_call_kwargs(backend, *, point_charges=None, target_states=None,
                        cell=None, external_potential=None,
                        qmmm_inputs=None, nac_pairs=None, soc_pairs=None):
    """Build the keyword arguments offered to one backend term.

    A channel is offered only when the item carries it and the backend
    declares it consumes it. The local executor and the remote term
    worker share this gate, so the two paths cannot ask for different
    physics.
    """
    kwargs = {
        'point_charges': point_charges,
        'target_states': target_states,
        'cell': cell,
    }
    if external_potential is not None and getattr(
            backend, 'supports_external_potential', False):
        # The framework offers the field; a skill that does not consume
        # it ignores it. Refusing here would couple the framework to one
        # skill's present capability set.
        kwargs['external_potential'] = external_potential
    if qmmm_inputs is not None:
        kwargs['qmmm_inputs'] = qmmm_inputs
    if nac_pairs and getattr(backend, 'supports_nac_request', False):
        # Only the backends that read a per-step pair get one. A skill
        # whose own input names its couplings from its own config needs
        # nothing here, and refusing would turn an ordinary pair map into
        # a failure.
        kwargs['nac_pairs'] = nac_pairs
    if soc_pairs and getattr(backend, 'supports_soc_request', False):
        # Same gate on the spin-orbit channel.
        kwargs['soc_pairs'] = soc_pairs
    return kwargs


def _run_one(item):
    """Execute one work item into a unified outcome.

    The backend exception is captured into the outcome (success=False
    + error) so the executor controls the submission instead of
    letting one term abort the pool indiscriminately.

    Parameters
    ----------
    item : dict
        Unified work item.

    Returns
    -------
    dict
        Unified outcome {task_id, evaluation_id, raw, success, error}.
    """
    try:
        backend = item['backend']
        kwargs = backend_call_kwargs(
            backend,
            point_charges=item.get('point_charges'),
            target_states=item.get('target_states'),
            cell=item.get('cell'),
            external_potential=item.get('external_potential'),
            qmmm_inputs=item.get('qmmm_inputs'),
            nac_pairs=item.get('nac_pairs'),
            soc_pairs=item.get('soc_pairs'),
        )
        raw = backend.calculate(
            item['local_atoms'], item['local_coords'], **kwargs)
        return make_outcome(item, raw, True)
    except Exception as exc:  # captured into the unified outcome
        return make_outcome(item, {}, False, str(exc))


def execute_with_failure_handling(work_items, executor):
    """Run the terms and apply the evaluation failure policy.

    This is the calculator-side failure semantics: any term with
    success=False fails the whole evaluation. The executor already
    stopped the remaining work (serial short-circuit / thread cancel +
    drain); here every completed term diagnostic (task_id,
    evaluation_id, error) is collected and reported before raising,
    preserving the external fail-fast contract.

    Parameters
    ----------
    work_items : list of dict
        Unified work items to execute.
    executor : TaskExecutor
        Executor instance (serial or thread).

    Returns
    -------
    list of dict
        Outcomes (same format as TaskExecutor.execute()) when every
        term succeeded.

    Raises
    ------
    RuntimeError
        If any term failed; the message carries every collected
        diagnostic (task_id / evaluation_id / error).
    """
    outcomes = executor.execute(work_items)
    return apply_failure_policy(outcomes)


def apply_failure_policy(outcomes):
    """Apply the evaluation failure policy to a collected outcome set.

    Any term with success=False fails the whole evaluation: every
    completed term diagnostic (task_id, evaluation_id, error) is
    collected and reported before raising, preserving the external
    fail-fast contract. Shared by the in-process executor path and the
    remote per-term path (r3e) so both raise the identical message.

    Parameters
    ----------
    outcomes : list of dict
        Collected outcomes (serial / thread / remote).

    Returns
    -------
    list of dict
        The same outcomes when every term succeeded.

    Raises
    ------
    RuntimeError
        If any term failed; the message carries every collected
        diagnostic (task_id / evaluation_id / error).
    """
    failures = [o for o in outcomes if not o.get('success', True)]
    if failures:
        diagnostics = '; '.join(
            "task '%s' (evaluation %r): %s" % (
                f.get('task_id', '?'), f.get('evaluation_id'),
                f.get('error', ''))
            for f in failures)
        raise RuntimeError(
            "Composite evaluation failed (%d term(s)): %s"
            % (len(failures), diagnostics))
    return outcomes
