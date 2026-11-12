"""
======================================================================
This module provides the Dispatcher base class, the unified execution
base of the PyRAI3MD workflow layer.

Every workflow and the calculator inherit from Dispatcher. The
execution mode is fully determined by two orthogonal booleans:

    multitask  : create copies of the current subject (workflow or
                 calculator) and broadcast parameters to them
    remotetask : execute tasks remotely through Slurm submission

Dispatcher.dispatch() is the unified execution tool method:
- remotetask selects the Launcher (local vs Slurm).
- multitask creates copies of the current subject, broadcasts
  parameters via broadcast(), and executes copies in parallel.
- dispatch() strictly follows the planning execution graph flags;
  it does not make additional conditional judgments.

Subclasses call self.dispatch() from their own run() methods,
unifying the execution logic of workflow and calculator.

Universal child task rules:
- Rule 1: every dispatched unit state is labeled as a worker.
- Rule 2: a worker state never carries remotetask=True.
- Rule 3: calculator nodes never carry remotetask=True (remote
  single-point execution is controlled by the skill itself).

Aug 13 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import copy

import numpy as np

from ...data.runtime.state import RuntimeState, validate_backend_role
from ...kernel.logger import get_logger


def task_system_updates(task):
    """Derive pmd.system updates for one batch task descriptor.

    Parameters
    ----------
    task : object
        Batch task descriptor produced by a workflow's
        generate_batch().

    Returns
    -------
    dict
        Field updates for state.update_system(); empty dict means
        the task uses the current geometry as is.
    """
    if isinstance(task, dict):
        updates = {}
        coordinates = task.get('coordinates')
        if coordinates is not None:
            updates['coordinates'] = coordinates
        velocities = task.get('velocities')
        if velocities is not None:
            updates['velocities'] = velocities
        image = task.get('image')
        if image is not None:
            # The crossing ledger of the geometry being handed over. A batch
            # whose task wrapped its coordinates without carrying the count of
            # the same step leaves the child with a stale ledger, and a new
            # position written on a stale one relocates the atom by a whole box
            # (the same rule `Optimizer._apply_step` follows when it rolls a
            # rejected step back).
            updates['image'] = [list(row) for row in image]
        lattice = task.get('lattice')
        if lattice is not None:
            # A task that strains the box carries the strained lattice, and the
            # cell parameters travel with it: the two are spellings of one cell
            # and a backend that reads the six while the framework wrote the nine
            # evaluates a different box than the one the task named.
            from ...utilities.lattice import cell_volume, matrix_to_cell6

            rows = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
            six = matrix_to_cell6(rows)
            updates['lattice_vector'] = rows.tolist()
            updates['cell'] = [float(value) for value in six]
            updates['volume'] = cell_volume(rows)
        return updates
    return {}


def clear_profile(config):
    """Clear the planning-pinned profile of one task config.

    An empty profile lets skill backends in standalone mode re-derive
    the calculation profile from the (new) jobtype.

    Parameters
    ----------
    config : BaseModel or dict
        Task configuration to update.
    """
    calculation = getattr(config, 'calculation', None)
    if calculation is not None and hasattr(calculation, 'profile'):
        calculation.profile = ''
    if isinstance(config, dict):
        if 'profile' in config:
            config['profile'] = ''
        calculation_dict = config.get('calculation')
        if (isinstance(calculation_dict, dict)
                and 'profile' in calculation_dict):
            calculation_dict['profile'] = ''


def apply_child_jobtype(state, child_jobtype):
    """Apply child_jobtype to the task configs of a state.

    Child runtimes execute the basic child_jobtype calculation
    regardless of the parent jobtype, so every task config jobtype is
    aligned and the pinned profile is cleared.

    Parameters
    ----------
    state : PyRAIState
        State whose task configs are updated.
    child_jobtype : str
        Basic jobtype executed by the child runtime.
    """
    if state is None or not child_jobtype:
        return
    for _, task in state.iter_tasks():
        config = task.get('config')
        if config is None:
            continue
        control = getattr(config, 'control', None)
        if control is not None:
            control.jobtype = child_jobtype
            clear_profile(config)
            continue
        if isinstance(config, dict):
            control_dict = config.get('control')
            if isinstance(control_dict, dict):
                control_dict['jobtype'] = child_jobtype
            clear_profile(config)


def worker_execution_graph():
    """Build the frozen execution graph stamped onto worker states.

    A worker executes one local calculator node only (universal rules
    2 and 3: workers and calculators never set remotetask). The node
    carries the serial/local backend_terms descriptor so the frozen
    worker graph keeps the same schema as a root graph (the internal
    thread policy is never cleared: a worker simply never fans out).

    Returns
    -------
    list of dict
        Frozen worker execution graph.
    """
    from .planner import default_backend_execution_descriptor
    return [{
        'node': 'calculator',
        'name': 'calculator',
        'multitask': False,
        'remotetask': False,
        'multipool': 1,
        'backend_execution': default_backend_execution_descriptor(),
    }]


class Dispatcher:
    """Unified execution base for all workflows and the calculator.

    The execution mode is declared through two orthogonal booleans
    (class attributes provide the defaults, instances may override):

    - multitask : fan the units out to a multiprocessing worker pool.
    - remotetask: materialize the units as remote child runtimes
      (subprocess or slurm launch tools).

    Parameters
    ----------
    state : PyRAIState or None
        Parent state used for unit materialization, runtime
        directory naming and environment configuration.
    multitask : bool or None
        Instance override; None keeps the class default.
    remotetask : bool or None
        Instance override; None keeps the class default.
    local_executor : callable or None
        Injected in-process execution entry (workflow.execute); the
        dispatch package never imports the workflow package.
    executor_spec : tuple or None
        Importable spec ('module', 'function') of the execution
        entry, used by multiprocessing workers.
    multipool : int or None
        Worker pool size of this subject's multitask fan-out (the
        multipool keyword of its workflow input section); None keeps
        the class default.
    """

    # Class-level execution defaults (overridden by graph keywords)
    multitask = False
    remotetask = False
    multipool = 1

    def __init__(self, state=None, multitask=None, remotetask=None,
                 local_executor=None, executor_spec=None,
                 multipool=None):
        self.state = state
        if multitask is not None:
            self.multitask = bool(multitask)
        if remotetask is not None:
            self.remotetask = bool(remotetask)
        if multipool is not None:
            self.multipool = max(1, int(multipool))
        self.local_executor = local_executor
        self.executor_spec = executor_spec or (
            'pyrai3md.workflow', 'execute')
        # Init-time log: only multitask/remotetask (workflow has no
        # execution_mode/skill info; Calculator supplements those).
        self.logger = get_logger('pyrai3md.dispatch')
        self.logger.info(
            "Dispatcher initialized: multitask=%s, remotetask=%s, "
            "multipool=%d",
            self.multitask, self.remotetask, self.multipool)

    # ---- Unit materialization (batch fan-out preparation) ----

    def materialize_unit(self, task, index, child_jobtype=None,
                         active_role=None):
        """Build one independent child runtime unit from a batch task.

        Creates an independent child PyRAIState (deep copy) running
        the basic child_jobtype at the batch task geometry, stamps it
        as a worker with a parent-frozen execution graph (universal
        rule 1) so the child never re-plans or dispatches.

        With an explicit active_role the child is a single-role
        runtime: the planned task configs are kept untouched (a
        reference task must keep the capabilities and profile planning
        gave it), the inactive role's queue is dropped so its backend
        is never constructed for a calculation that will not run it,
        and the role travels on RuntimeState into execute(). Without
        the argument the historical theory behaviour is unchanged.

        Parameters
        ----------
        task : dict
            Batch task descriptor from generate_batch(), or an
            already materialized unit (carries 'state').
        index : int
            Batch task index.
        child_jobtype : str or None
            Basic jobtype the child executes. Required when
            active_role is given: it is what makes the child resolve a
            single-shot calculation instead of re-entering the parent
            workflow.
        active_role : str or None
            Backend role the child executes ('theory' or 'reference');
            None keeps 'theory' and the historical materialization.

        Returns
        -------
        dict
            Unit descriptor with keys: state, task_index.
        """
        if active_role is not None:
            active_role = validate_backend_role(active_role)
            if not child_jobtype:
                raise ValueError(
                    "An explicit active_role requires child_jobtype: the "
                    "child's control jobtype is what resolves a "
                    "single-shot calculation instead of the parent "
                    "workflow.")

        if isinstance(task, dict) and 'state' in task:
            if active_role is not None:
                child_state = task['state']
                if getattr(child_state, 'runtime', None) is None:
                    child_state.runtime = RuntimeState(
                        active_role=active_role)
                else:
                    child_state.runtime.active_role = active_role
            return task

        child = copy.deepcopy(self.state)
        if getattr(child, 'pmd', None) is not None and child_jobtype:
            child.pmd.control.jobtype = child_jobtype
            child.pmd.results = {}
        updates = task_system_updates(task)
        if updates:
            child.update_system(updates)
        if active_role is None:
            child.runtime = RuntimeState()
            apply_child_jobtype(child, child_jobtype)
        else:
            child.runtime = RuntimeState(active_role=active_role)
            self._prune_inactive_role(child, active_role)

        # Universal rule 1: dispatched jobs are labeled as workers;
        # the child execution graph is fixed by the parent and the
        # planning process is skipped (resolve_execution_graph=False).
        child.resolve_execution_graph = False
        child.execution_graph = worker_execution_graph()
        child.runtime.worker = True
        return {'state': child, 'task_index': index}

    @staticmethod
    def _prune_inactive_role(state, active_role):
        """Drop the task queue of every role but the one being run.

        A single-role child never switches roles, so keeping the other
        queue would make Calculator.initialize() construct a backend
        that is never evaluated (the NN potential loads its ensemble
        in __init__, which is too expensive to pay per reference
        sample).

        Parameters
        ----------
        state : PyRAIState
            Materialized child state (the parent is a deep copy away).
        active_role : str
            Role the child will execute.
        """
        if active_role == 'reference':
            state.theory = []
        else:
            state.reference = []

    def materialize_units(self, tasks, child_jobtype=None,
                          active_role=None):
        """Materialize a batch of task descriptors into units.

        Parameters
        ----------
        tasks : list
            Batch task descriptors from generate_batch().
        child_jobtype : str or None
            Basic jobtype executed by every child runtime.
        active_role : str or None
            Backend role every child executes; None keeps the
            historical theory materialization.

        Returns
        -------
        list of dict
            One unit descriptor per task, in task order.
        """
        return [self.materialize_unit(task, index, child_jobtype,
                                      active_role=active_role)
                for index, task in enumerate(tasks)]

    # ---- Unified execution pipeline ----

    def dispatch(self, units=None, child_jobtype=None, active_role=None):
        """Execute through the unified dual-axis pipeline.

        Strictly follows the planning execution graph:
        - remotetask: selects the runner through the Launcher
          ('local' for in-process, slurm_run for Slurm cluster
          submission).
        - multitask: the Scheduler wraps the runner in a
          multiprocessing.Pool for parallel execution.

        Subclasses call self.dispatch() from their own run() methods;
        dispatch() is a tool method, not a public entry point.

        Parameters
        ----------
        units : list or None
            Unit descriptors ('state' + 'task_index') or raw batch
            task descriptors; None builds one unit from the parent
            state.
        child_jobtype : str or None
            Basic jobtype applied when materializing raw tasks.
        active_role : str or None
            Backend role every materialized child executes ('theory'
            or 'reference'); None keeps 'theory'. The role rides on
            the child state's RuntimeState, because the multitask
            branch below rebuilds its execution units from a
            whitelist of keys and would drop a field carried on the
            unit dict.

        Returns
        -------
        list
            One outcome per unit, in unit order.
        """
        from .launcher import Launcher
        from .scheduler import Scheduler

        if units is None:
            units = [{'state': self.state, 'task_index': 0}]
        units = [
            self.materialize_unit(unit, index, child_jobtype,
                                  active_role=active_role)
            for index, unit in enumerate(units)
        ]
        if active_role is not None:
            self.logger.info(
                "Dispatch: %d unit(s) on the '%s' backend role.",
                len(units), active_role)

        # Inject executor spec into every unit so the Scheduler's
        # pool worker can resolve the local execution entry without
        # pickling the local_executor closure.
        for unit in units:
            unit['executor_spec'] = self.executor_spec

        # Runtime identity planning: a batch with more than one unit
        # gets one runtime index and one sub working directory per
        # unit, regardless of the execution location (local vs
        # remote). Single-unit batches keep today's behavior (no
        # identity; a remote single task self-allocates in slurm_run).
        if len(units) > 1:
            self.allocate_runtime_identities(units)

        if self.multitask and len(units) > 1:
            # Create copies of the current subject
            copies = [self._create_copy() for _ in units]
            # Broadcast parameters from main subject to copies
            self.broadcast(copies, units)
            # Freeze copies: no recursive fan-out
            for c in copies:
                c.multitask = False
                c.remotetask = False
            # Execute copies through Launcher/Scheduler
            exec_units = [
                {'state': c.state, 'task_index': i,
                 '_copy': c, 'executor_spec': self.executor_spec,
                 'runtime_idx': unit.get('runtime_idx'),
                 'runtime_name': unit.get('runtime_name'),
                 'runtime_root': unit.get('runtime_root'),
                 'runtime_name_format': unit.get('runtime_name_format'),
                 'workdir': unit.get('workdir')}
                for i, (c, unit) in enumerate(zip(copies, units))
            ]
            runner = Launcher()(self.remotetask)
            scheduler = Scheduler(
                state=self.state,
                executor_spec=self.executor_spec,
                local_executor=self.local_executor,
                multipool=self.multipool,
            )(self.multitask)
            return scheduler(runner)(exec_units)

        # Single-subject execution
        runner = Launcher()(self.remotetask)
        scheduler = Scheduler(
            state=self.state,
            executor_spec=self.executor_spec,
            local_executor=self.local_executor,
            multipool=self.multipool,
        )(self.multitask)
        return scheduler(runner)(units)

    # ---- Runtime identity allocation (batch fan-out) ----

    def runtime_root(self):
        """Root directory for the batch runtime directories.

        Returns
        -------
        str or None
            Runtime root; None keeps the manager default
            ('<cwd>/runtime').
        """
        return None

    def runtime_name_format(self):
        """Directory name template for the batch runtime directories.

        Returns
        -------
        str or None
            Template with '{skill_name}'/'{idx}' placeholders; None
            keeps the manager default '{skill_name}_runtime_{idx}'.
        """
        return None

    def runtime_isolated_workdir(self):
        """Whether each runtime directory fully isolates its unit.

        In isolated mode the unit's calculation directories are
        placed inside its runtime directory and the execution ids
        drop the runtime suffix (the directory is already unique).

        Returns
        -------
        bool
            False by default (shared runtime root layout).
        """
        return False

    def allocate_runtime_identities(self, units):
        """Allocate a runtime index and sub workdir for every unit.

        The indices come from RuntimeDirectoryManager.allocate_indices
        (existing directories are reused in ascending order, so a
        restarted parent maps its batch back onto the previous
        runtimes). Each unit is stamped with its 'runtime_{idx}'
        identity, its full runtime name/root/name-format (consumed
        verbatim by both the local executor and slurm_run, so the
        two dispatch modes share one layout) and its workdir; the
        workdir is also written into the unit state runtime so the
        executed workflow writes its output files into the sub
        directory. Isolated-workdir subjects additionally stamp the
        isolation marks into the unit state (scratch_path injection
        and the execution-id suffix suppression).

        Parameters
        ----------
        units : list of dict
            Materialized unit descriptors (updated in place).
        """
        from .runtime import RuntimeDirectoryManager, resolve_skill_name

        manager = RuntimeDirectoryManager(
            skill_name=resolve_skill_name(self.state),
            root=self.runtime_root(),
            name_format=self.runtime_name_format())
        isolated = bool(self.runtime_isolated_workdir())
        indices = manager.allocate_indices(len(units))
        for unit, idx in zip(units, indices):
            workdir = manager.create(idx)
            unit['runtime_idx'] = manager.runtime_idx(idx)
            unit['runtime_name'] = manager.runtime_name(idx)
            unit['runtime_root'] = manager.root
            unit['runtime_name_format'] = manager.name_format
            unit['workdir'] = workdir
            unit_state = unit.get('state')
            runtime = getattr(unit_state, 'runtime', None)
            if runtime is not None:
                runtime.workdir = workdir
            if isolated:
                self._stamp_isolated_workdir(unit_state, workdir)
        self.logger.info(
            "Dispatcher allocated %d runtime identities: %s",
            len(units),
            ', '.join(unit['runtime_idx'] for unit in units))

    def _stamp_isolated_workdir(self, unit_state, workdir):
        """Stamp the isolation marks onto one unit state.

        Isolated workdir mode keeps every artifact of the unit inside
        its own runtime directory: the task config scratch_path is
        pointed at the directory (backend calculation directories
        land inside it) and the runtime marks itself isolated so
        compose_execution_id() drops the '_runtime_{idx}' suffix
        (each directory is already unique). Both marks travel with
        the serialized state into subprocess/slurm children.

        Parameters
        ----------
        unit_state : PyRAIState or None
            Materialized child state of the unit.
        workdir : str
            Runtime directory allocated for the unit.
        """
        if unit_state is None:
            return
        runtime = getattr(unit_state, 'runtime', None)
        if runtime is not None:
            runtime.isolated_workdir = True
        try:
            tasks = list(unit_state.iter_tasks())
        except (AttributeError, TypeError) as error:
            # Without the queue nothing can be stamped, so the unit's
            # products would land outside workdir; say so instead of
            # letting the leak appear later as a missing file.
            self.logger.warning(
                'Isolated workdir stamping skipped: the unit state gave no '
                'task queue (%s), so scratch_path was not pointed at %s',
                error, workdir)
            return
        for _, task in tasks:
            if not isinstance(task, dict):
                continue
            config = task.get('config')
            if config is None:
                continue
            if isinstance(config, dict):
                skill_config = config.get('skill_config')
                if isinstance(skill_config, dict):
                    skill_config['scratch_path'] = workdir
                calculation = config.get('calculation')
                if isinstance(calculation, dict):
                    calculation['scratch_path'] = workdir
            else:
                calculation = getattr(config, 'calculation', None)
                if (calculation is not None
                        and hasattr(calculation, 'scratch_path')):
                    try:
                        calculation.scratch_path = workdir
                    except (TypeError, ValueError):
                        pass

    # ---- Multitask copy and broadcast ----

    def _create_copy(self):
        """Create a deep copy of the current subject for multitask.

        Returns
        -------
        Dispatcher
            Deep copy of self (workflow or calculator copy).
        """
        return copy.deepcopy(self)

    def broadcast(self, copies, units):
        """Broadcast parameters from the main subject to copies.

        Default implementation assigns each copy's state from the
        corresponding unit. Subclasses override for custom parameter
        passing (e.g. distribute N structures from the main object
        into N copies' PyRAIState).

        Parameters
        ----------
        copies : list of Dispatcher
            Deep copies of the current subject.
        units : list of dict
            Unit descriptors with 'state' and 'task_index'.
        """
        for c, unit in zip(copies, units):
            if isinstance(unit, dict) and 'state' in unit:
                c.state = unit['state']

    def __repr__(self):
        return "Dispatcher(multitask=%s, remotetask=%s)" % (
            self.multitask, self.remotetask)
