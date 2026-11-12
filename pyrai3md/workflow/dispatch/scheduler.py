"""
======================================================================
This module provides the Scheduler, which converts the multitask
boolean into an execution wrapper around the runner function
returned by the Launcher.

    execute = Scheduler(state, executor_spec)(multitask)(runner)

- multitask=False: returns a sequential wrapper that calls the
  runner for each unit one by one.
- multitask=True: returns a parallel wrapper executing the units
  through an explicit spawn multiprocessing pool.  Both local
  (runner='local') and remote (runner=slurm_run) execution go
  through the same pool; the dispatch_worker resolves the actual
  execution entry.

The pool worker count comes from the multipool keyword of the
dispatching workflow (passed in at construction); the pool passes
full unit dicts (with complete state) so remote runners can
serialize the state for child runtimes.

The Scheduler never imports the workflow package: multiprocessing
workers resolve the local execution entry through the injected
import spec (executor_spec).

What the run asks for is sized against the CPUs it can occupy
(`validate_resources`), read from each skill's own resource declaration
rather than from a table of engine keywords here; a shortfall is a
warning, never a resize (#306).

Aug 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module schedules jobs, not parameters. The ramp of a constraint force constant over
the accepted steps of a run lives in `workflow/constraints/schedule.py`, which shares the
word and nothing else (D-13).
"""

import importlib
import multiprocessing as mp
import os

from ...kernel.logger import get_logger


_DECLARATIONS = None


def resource_declarations():
    """Return the resource keywords each skill declares.

    The declaration lives in the skill's manifest.yaml `resources`
    block and is surfaced by its query module, so the framework asks
    each skill how to read its own thread and rank counts instead of
    keeping a table of engine keywords here (#306). A skill that
    declares nothing contributes no factor.

    Returns
    -------
    dict
        {skill_name: {'threads': {'keyword': str, 'default': int},
        'ranks': {...}}}; the 'ranks' entry is absent for skills
        without an MPI arm.
    """
    global _DECLARATIONS
    if _DECLARATIONS is None:
        from ...skills.query import discover_all
        _DECLARATIONS = {
            meta.get('skill_name'): meta.get('resources') or {}
            for meta in discover_all().values()}
    return _DECLARATIONS


def dispatch_worker(payload):
    """Pool worker: dispatch one unit through its runner.

    For local execution (runner='local'), resolves the executor
    from the executor_spec carried in the unit dict (avoids
    pickling the local_executor closure).

    For remote execution, calls the module-level runner function
    (slurm_run) directly; it is picklable by reference.

    Parameters
    ----------
    payload : tuple
        (runner, unit) where runner is 'local' or a callable and
        unit is the unit descriptor dict.

    Returns
    -------
    object
        Outcome of the unit execution.
    """
    runner, unit = payload
    if runner == 'local':
        executor_spec = unit['executor_spec']
        execute = getattr(importlib.import_module(executor_spec[0]),
                          executor_spec[1])
        return execute(unit['state'],
                       runtime_idx=unit.get('runtime_idx'))
    return runner(unit)


def invoke_local(executor, unit):
    """Call a local executor for one unit.

    The runtime identity is forwarded only when the unit carries
    one, so legacy single-argument executors stay compatible for
    identity-less (single-unit) dispatch.

    Parameters
    ----------
    executor : callable
        Local execution entry (state[, runtime_idx]).
    unit : dict
        Unit descriptor with 'state' and an optional 'runtime_idx'.

    Returns
    -------
    object
        Outcome of the unit execution.
    """
    runtime_idx = unit.get('runtime_idx')
    if runtime_idx is not None:
        return executor(unit['state'], runtime_idx=runtime_idx)
    return executor(unit['state'])


class Scheduler:
    """Convert the multitask flag into an execution wrapper.

    The Scheduler manages the multiprocessing.Pool for parallel
    execution.  Local execution ('local' runner) and remote
    runners (slurm_run) are both parallelized through the pool.

    Parameters
    ----------
    state : PyRAIState or None
        Parent state (environment snapshot providing the thread
        sanity-check data).
    executor_spec : tuple or None
        Importable spec ('module', 'function') of the execution
        entry, used by pool workers for remote execution.
    local_executor : callable or None
        Direct in-process execution entry.  Used by the sequential
        wrapper for local runner to avoid recursive dispatch.
    multipool : int or None
        Worker pool size declared by the dispatching workflow
        (its multipool input keyword); None keeps serial execution.
    """

    def __init__(self, state=None, executor_spec=None,
                 local_executor=None, multipool=None):
        self.state = state
        self.executor_spec = executor_spec or (
            'pyrai3md.workflow', 'execute')
        self.local_executor = local_executor
        try:
            self.multipool = max(1, int(multipool or 1))
        except (TypeError, ValueError):
            self.multipool = 1
        self.logger = get_logger('pyrai3md.dispatch')

    def __call__(self, multitask):
        """Return the scheduler wrapper for the given multitask flag.

        Parameters
        ----------
        multitask : bool
            False: sequential wrapper (calls runner per unit).
            True: multiprocessing pool wrapper.

        Returns
        -------
        callable
            Function receiving the runner and returning the
            composed execution function run(units).
        """
        if not multitask:
            return self._wrap_sequential
        return self._wrap_parallel

    def _wrap_sequential(self, runner):
        """Wrap a runner for sequential execution.

        Parameters
        ----------
        runner : str or callable
            'local' for in-process execution, or slurm_run.

        Returns
        -------
        callable
            Function executing a unit list sequentially.
        """
        local_executor = self.local_executor
        scheduler = self

        def run_all(units):
            # Checked on the sequential path too: one unit at a time is
            # still a resource request, and the machine does not care
            # which wrapper executed it (#306).
            scheduler.validate_resources(runner)
            if runner == 'local':
                # Local sequential: call local_executor directly
                # to avoid recursive re-entry into dispatch()
                # through executor_spec -> workflow.execute().
                return [invoke_local(local_executor, unit)
                        for unit in units]
            return [runner(unit) for unit in units]

        return run_all

    def _wrap_parallel(self, runner):
        """Wrap a runner for parallel execution.

        Both local and remote runners go through the multiprocessing
        pool; the dispatch_worker resolves the actual execution entry.

        Parameters
        ----------
        runner : str or callable
            'local' for in-process execution, or slurm_run.

        Returns
        -------
        callable
            Function executing a unit list.
        """
        scheduler = self

        def run_all(units):
            units = list(units)
            # The pool never runs more processes than units, so the
            # smaller of the two is what the machine actually holds.
            scheduler.validate_resources(
                runner, min(scheduler._worker_count(), len(units)))
            return scheduler._run_pool(units, runner)

        return run_all

    def _run_pool(self, units, runner):
        """Execute units through an explicit spawn worker pool.

        Full unit dicts (with complete state) are passed to the
        pool workers.  The dispatch_worker resolves the execution
        entry based on the runner type.

        Parameters
        ----------
        units : list of dict
            Unit descriptors with the child states.
        runner : str or callable
            'local' or a module-level runner function.

        Returns
        -------
        list
            One outcome per unit, in unit order.
        """
        payloads = [(runner, unit) for unit in units]
        workers = self._worker_count()
        if workers <= 1 or len(payloads) <= 1:
            return [dispatch_worker(p) for p in payloads]
        # Explicit spawn context: workers never inherit parent
        # process state (deterministic across platforms).
        context = mp.get_context('spawn')
        with context.Pool(
                processes=min(workers, len(payloads))) as pool:
            return pool.map(dispatch_worker, payloads)

    def _worker_count(self):
        """Return the worker pool size declared at construction.

        The dispatching workflow passes its multipool input keyword
        through the Dispatcher; the value is clamped to >= 1.

        Returns
        -------
        int
            multipool value (>= 1, default 1).
        """
        return max(1, self.multipool)

    def _term_workers(self):
        """How many backend terms one process runs at once.

        The term axis is orthogonal to this Scheduler's pool: a child
        process may still hold a thread pool of its own when the input
        asks ``composite executor thread``. Counting it is what makes
        the oversubscription warning describe the machine rather than
        one axis of the dispatch (#303). Serial (the default) contributes
        no factor.

        Returns
        -------
        int
            ``max_workers`` when the composite requests thread mode,
            otherwise 1.
        """
        composite = getattr(getattr(self.state, 'pmd', None),
                            'composite', None)
        if composite is None or getattr(composite, 'executor',
                                         'serial') != 'thread':
            return 1
        try:
            return max(1, int(getattr(composite, 'max_workers', 1) or 1))
        except (TypeError, ValueError):
            return 1

    def _unit_demand(self):
        """CPUs one execution unit asks for, read from the manifests.

        Each skill declares in its manifest which keyword carries its
        thread count and which its MPI rank count, with the value to
        assume when an input omits them (`resource_declarations`). A
        rank is an MPI process and each rank holds `thread` OpenMP
        threads, so a hybrid skill demands ranks x threads and a
        thread-only skill, which declares no ranks, demands its threads.

        The values are looked up in the task queue because that is where
        the effective per-skill config lives. Both spellings are asked:
        the skill section may carry the plain keyword or the framework
        spelling `<skill>_thread`/`<skill>_nproc`, and the skill itself
        applies the larger of the two -- this check only estimates the
        demand before any skill config is built (the framework does not
        run the skill schema pipeline), so it takes the same maximum and
        cannot decide anything the skill does not decide.

        The maximum over the queued skills is used: the warning asks
        about the machine, and any unit may end up running the
        hungriest skill.

        Returns
        -------
        tuple
            (CPUs per unit (>= 1), skill name that asked for them);
            (1, '') when nothing is queued or nothing is declared.
        """
        declarations = resource_declarations()
        best_demand, best_skill = 1, ''
        state = getattr(self, 'state', None)
        iter_tasks = getattr(state, 'iter_tasks', None)
        if iter_tasks is None:
            return best_demand, best_skill
        for _identity, task in iter_tasks():
            if not isinstance(task, dict):
                continue
            config = task.get('config')
            # A materialized task carries the task_config dict; a raw
            # one may still carry the framework Config model, which has
            # no skill section to read.
            skill_config = config.get('skill_config') \
                if isinstance(config, dict) else None
            if not isinstance(skill_config, dict):
                continue
            skill_name = task.get('skill_name') or ''
            declaration = declarations.get(skill_name) or {}
            prefix = skill_name[:-len('_skill')] \
                if skill_name.endswith('_skill') else skill_name
            demand = 1
            for kind in ('threads', 'ranks'):
                entry = declaration.get(kind) or {}
                keyword = entry.get('keyword')
                if not keyword:
                    continue
                candidates = [skill_config.get(keyword),
                              skill_config.get('%s_%s' % (prefix, keyword))]
                candidates.append(entry.get('default'))
                values = []
                for candidate in candidates:
                    try:
                        values.append(max(1, int(candidate)))
                    except (TypeError, ValueError):
                        continue
                if values:
                    demand *= max(values)
            if demand > best_demand:
                best_demand, best_skill = demand, skill_name
        return best_demand, best_skill

    def _available_cpus(self, runner):
        """CPUs this run may occupy.

        A remote run lands on its Slurm allocation, not on the machine
        that submitted it, so the allocation from the environment
        snapshot (`slurm.ntasks` x `slurm.cpus_per_task`) is the
        reference there. A local run has this machine's cores.

        Parameters
        ----------
        runner : str or callable or None
            'local' or None for in-process execution, a runner function
            for submitted execution.

        Returns
        -------
        int
            CPU count (>= 1).
        """
        if runner is not None and runner != 'local':
            environment = getattr(getattr(self.state, 'pmd', None),
                                  'environment', None)
            slurm = getattr(environment, 'slurm', None) or {}
            try:
                ntasks = int(slurm.get('ntasks') or 0)
                per_task = int(slurm.get('cpus_per_task') or 0)
            except (TypeError, ValueError):
                ntasks = per_task = 0
            if ntasks and per_task:
                return max(1, ntasks * per_task)
        return max(1, os.cpu_count() or 1)

    def validate_resources(self, runner=None, units=1):
        """Warn when the request exceeds the CPUs this run can occupy.

        Demand is `units x CPUs per unit`. Units is the concurrency the
        caller actually starts: the pool size on the parallel path
        (capped by how many units are in flight), 1 on the sequential
        path. The term axis multiplies it inside every process, because
        a child may run its backends in a thread pool (`_term_workers`).
        CPUs per unit come from the skill manifests (`_unit_demand`),
        never from a framework field: the ``environment.num_thread``
        this check used to read was unreachable from the input, so it
        could not fire (#303), and the run-wide floor built on that
        field was cancelled when the resource keywords moved to
        `<skill>_thread` / `<skill>_nproc` (#306).

        Oversubscription is allowed and only logged. The machine that
        runs the job is not the one that parsed the input, engines are
        oversubscribed on purpose, and the framework never resizes a
        request on the user's behalf.
        """
        demand, skill_name = self._unit_demand()
        try:
            started = max(1, int(units or 1))
        except (TypeError, ValueError):
            started = 1
        units = started * self._term_workers()
        total = units * demand
        available = self._available_cpus(runner)
        if total > available:
            self.logger.warning(
                "Parallel request %d unit(s) x %d CPU(s) per unit%s = %d "
                "exceeds the %d CPUs available to this run "
                "(oversubscribed)." % (
                    units, demand,
                    " of %s" % skill_name if skill_name else '',
                    total, available))

    def __repr__(self):
        return "Scheduler(workers=%d)" % self._worker_count()
