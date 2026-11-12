"""
======================================================================
This module provides the Calculator class for single-request
computations within the workflow layer. It reads the task queue from
PyRAIState.theory/reference, instantiates the backends at runtime,
executes the active role task queue through its backends,
synchronizes task configurations, and returns a structured Result.

Calculator inherits Dispatcher: a single-point computation is a unit
list of length 1, fully unified with batch execution. run() calls
self.dispatch() (unified execution logic shared with Workflow);
universal rule 3 keeps the calculator always local (remotetask=False),
remote single-point execution is controlled by the skill itself
through its own execution_mode.

Calculator is the sole runtime state manager and the sole owner of
the PyRAIState: workflow and other modules communicate through
get_state() and update() only, never mutating the state directly.
Backends are runtime objects owned by the Calculator (created in
initialize()); PyRAIState never stores backend instances.

Backend role grouping:
- initialize() builds the backends following iter_tasks() and labels
  every backend with its category (theory/reference).
- The active role (default 'theory') selects which queue run()
  executes; switch_role() may switch to 'reference', but both roles
  are never executed simultaneously.
- Composite aggregation and projection match the active role.

Calculator lifecycle:
    initialize() -> run() -> sync()

Aug 07 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import importlib
import glob
from copy import deepcopy

import os
import time

import numpy as np
import yaml

from ..data.geometry import continuous_coordinates
from ..data.runtime.result import Result, task_config_results
from ..data.runtime.state import (BACKEND_ROLES, RuntimeState,
                                  validate_backend_role)
from ..data.charges import (
    ChargeStore,
    embedding_region_by_center as compute_embedding_regions)
from ..kernel.logger import get_logger
from ..utilities.timer import PhaseProfiler
from ..planning.task import resolve_primary_task, resolve_system_task
from ..utilities.lattice import (LATTICE_EPS, cell6_to_matrix,
                                cell_volume, matrix_to_cell6)
from ..utilities.units import (GPA_TO_HARTREE_PER_BOHR3,
                               BAR_TO_HARTREE_PER_BOHR3,
                               ATM_TO_HARTREE_PER_BOHR3)
from .multiscale.charges import (capture_outer_reference, damp_charges,
                                 embedding_for_task,
                                 region_charge_sources, update_from_outcomes)
from .constraints import runtime as constraint_runtime
from .records import analysis_bias
from .records.conversion import (NATIVE_UNITS_MARKER, apply_result_units,
                                 convert_native_energy,
                                 convert_native_gradient)
from .multiscale.composite import (aggregate_results,
                                   TaskOutcome, validate_plan,
                                   assemble_results,
                                   collect_center_electronic,
                                   invalidate_task_result)
from .dispatch.dispatcher import Dispatcher
from .dispatch.term_worker import execute_remote_terms
from .multiscale.regions import (materialize_region, project_gradient,
                                 slice_field)
from .dispatch.terms import (TaskExecutor, apply_failure_policy,
                             execute_with_failure_handling,
                             make_work_item)


# A QM/MM ESPF fit that sums within this of the declared charge is taken as
# already neutral: the log prints four decimals, so a residual this small is
# inside the resolution the charges were read at and rescaling would only
# move rounding noise.
_CHARGE_BALANCE_TOLERANCE = 1e-4

# Runtime system fields synchronized from state.pmd.system to every
# task configuration by Calculator.sync(). Only structural and
# electronic state information is synchronized; calculation, method
# and control sections are never touched.
SYNC_SYSTEM_FIELDS = (
    'atoms', 'mass', 'coordinates', 'velocities',
    'charges', 'multiplicity', 'natom', 'nstate',
    'ci', 'spin', 'coupling', 'state_label', 'state_spin',
    'current_state',
    # Periodic cell: synced so a variable-cell NPT step propagates the
    # evolved cell/lattice to the backend (dev/methods/
    # ENSEMBLE_DEVELOPMENT.md section 4). The extended-ensemble DOF
    # (thermostat_dof / barostat_strain / barostat_momentum) are
    # deliberately NOT synced.
    'cell', 'lattice_vector', 'pbc', 'volume',
)


# Electronic-state scalars are PER-CENTER for a multi-center run: the
# global pmd.system carries nstate=1 and no per-center state axis
# (docs 8), so syncing them would overwrite the planning-derived
# per-center task configs. The per-step active state travels through
# target_states (the calculate() parameter channel), so a C>1 sync()
# pushes geometry/charge only and leaves these to the task configs.
_ELECTRONIC_STATE_FIELDS = frozenset({
    'nstate', 'ci', 'spin', 'coupling', 'state_label', 'state_spin',
    'current_state',
})


def _task_target_states(target_states, task):
    """Per-task target states from the workflow's per-step request.

    A flat list (C=1) applies to every task unchanged (bit-for-bit the
    legacy single-center routing); a per-center dict (C>1) selects the
    task's own center entry, and a baseline task with no center_index
    keeps None (the backend's initialization value).

    Parameters
    ----------
    target_states : list of int or dict or None
        Per-step target states (flat for C=1, {center_index: list} for
        C>1).
    task : dict
        Task descriptor carrying center_index.

    Returns
    -------
    list of int or None
        The target states this task's backend receives.
    """
    if isinstance(target_states, dict):
        center_index = task.get('center_index')
        if center_index is None:
            return None
        return target_states.get(int(center_index))
    return target_states


_ESFP_FALLBACK_WARNED = set()


def _warn_esfp_fallback(task_id):
    """Say once that the field came from the file, not the result block."""
    if task_id in _ESFP_FALLBACK_WARNED:
        return
    _ESFP_FALLBACK_WARNED.add(task_id)
    from ..kernel.logger import get_logger
    get_logger(__name__).warning(
        "QMMM field task '%s' returned no v_mm_native/e_mm_native columns; "
        "reading its .esfp file instead. A field task that returns the "
        "columns needs no file.", task_id)


def _field_rows_in_atomic_units(rows, field_block, skill_name):
    """Convert one field block's native ``[V, Ex, Ey, Ez]`` rows to a.u.

    The block states its unit set under ``units`` (the name the backend
    used, e.g. ``real``), and the skill's manifest maps that set to the
    unit names the framework can convert. The ``.esfp`` file the same task
    writes is this conversion applied inside the skill; doing it here keeps
    the two channels bit-for-bit interchangeable and keeps the framework
    from importing skill code. A block with no unit set is refused rather
    than read as atomic units: the factor is 627 on V and 1186 on E for
    the LAMMPS ``real`` set, and a silent factor of one is a wrong run.
    """
    from .records.conversion import resolve_result_units
    from ..skills.query import get_declaration

    declared = (get_declaration(skill_name or '', 'qmmm')
                .get('results') or {})
    unit_set = str((field_block.get('units') or {}).get('energy')
                   or declared.get('default_units') or '')
    if not unit_set:
        raise ValueError(
            "QMMM field block carries V/E columns but names no unit set "
            "and its skill declares no default; the framework cannot "
            "convert them to atomic units without one.")
    energy_factor, gradient_factor = resolve_result_units(
        skill_name, unit_set)
    v = np.asarray([row[0] for row in rows], dtype=np.float64) \
        * energy_factor
    e = np.asarray([row[1:] for row in rows], dtype=np.float64) \
        * gradient_factor
    return [[float(v[i])] + [float(value) for value in e[i]]
            for i in range(len(rows))]


def _read_esfp_rows(path):
    """Read one ``.esfp`` handoff into ``(atom_ids or None, rows)``.

    The current layout carries the atom id first (``id V Ex Ey Ez``, atomic
    units) so a row binds to an atom rather than to a position: under
    several centers each probe writes only its own center's rows, and the
    reader has to know which atoms those are. A file without the id column
    still reads, and its rows come back unbound (``None`` ids) so the caller
    keeps the positional rule it always had -- which is only sound for the
    single-center deck that wrote every QM atom in ascending order.

    Parameters
    ----------
    path : str
        Path to the ``.esfp`` file.

    Returns
    -------
    tuple(list of int or None, list of list of float)
        Ids (1-based LAMMPS atom ids) and the ``[V, Ex, Ey, Ez]`` rows in
        file order.

    Raises
    ------
    ValueError
        On a row that is neither four nor five numeric columns, or a row
        whose id is not an integer.
    """
    ids = []
    rows = []
    with open(path, 'r') as handle:
        for line in handle:
            content = line.split('#')[0].strip()
            if not content:
                continue
            tokens = content.split()
            if len(tokens) not in (4, 5):
                raise ValueError(
                    "QMMM .esfp row must be 'V Ex Ey Ez' or "
                    "'id V Ex Ey Ez': %r" % (line,))
            values = [float(value) for value in tokens[-4:]]
            rows.append(values)
            if len(tokens) == 5:
                atom_id = int(tokens[0])
                ids.append(atom_id)
    return (ids or None), rows


class Calculator(Dispatcher):
    """High-level calculator executing the PyRAIState task queue.

    Inherits Dispatcher: the execution mode is declared through the
    multitask/remotetask booleans and resolved by the unified
    self.dispatch() pipeline. Universal rule 3 keeps the calculator
    always local (remotetask=False).

    PyRAIState ownership: the Calculator is the sole owner of the
    PyRAIState. External modules read it through get_state() and
    modify it through update(); direct attribute mutation is not
    part of the interface.

    Backend lifetime: backends are runtime objects owned by this
    Calculator. initialize() instantiates one backend per task with
    its execution namespace (execution_id = task static_id plus the
    optional workflow runtime suffix), grouped by category
    (theory/reference).

    Parameters
    ----------
    state : PyRAIState or None
        Runtime state (primary API). Tasks are obtained from
        state.iter_tasks().
    runtime_idx : str or None
        Runtime execution suffix provided by the Workflow layer
        (e.g. 'replica_3', 'traj_8'); None for single-shot runs.
    dummy_backend : object or None
        Dummy backend instance for testing; replaces the task
        backends (or serves a single-shot call when the state has
        no task queue).
    multitask : bool or None
        Execution flag override (graph keywords); None keeps the
        class default.
    remotetask : bool or None
        Must stay False (universal rule 3); True raises.
    base_flags : dict or None
        Per-backend-role base node flags of the frozen execution
        graph ({role: (multitask, remotetask, multipool)}); carried
        so switch_role() applies the new role's node flags.
    backend_execution : dict or None
        Per-backend-role backend-terms execution descriptor of the
        frozen graph ({role: descriptor}); carries the term-level
        executor ('serial'/'thread') and max_workers consumed by
        run_in_process(). None keeps serial in-process execution.
    local_executor : callable or None
        Injected in-process execution entry (optional).
    executor_spec : tuple or None
        Importable spec of the execution entry for multiprocessing
        workers.
    initial_role : str or None
        Backend role this calculator starts on; None keeps 'theory'.
        A dispatched child runtime reads it from
        RuntimeState.active_role, so a reference calculation can use
        the same fan-out as a theory one. The role's base-node flags
        are selected by execute(), not applied here.
    """

    # Class-level execution defaults: the calculator node is always
    # local and sequential unless the skill keyword sections declare
    # multitask (universal rule 3 forbids remotetask=True).
    multitask = False
    remotetask = False

    def __init__(self, state=None, runtime_idx=None, dummy_backend=None,
                 multitask=None, remotetask=None, multipool=None,
                 base_flags=None, backend_execution=None,
                 local_executor=None, executor_spec=None,
                 initial_role=None):
        if remotetask:
            raise RuntimeError(
                "Universal rule 3: the calculator never sets "
                "remotetask=True; remote single-point execution is "
                "controlled by the skill itself.")
        self._state = state
        # Per-backend-role base node flags of the frozen execution
        # graph; switch_role() applies the new role's triple.
        self.base_flags = dict(base_flags or {})
        # Per-backend-role backend-terms execution descriptor of the
        # frozen graph (r3b); run_in_process() reads the active role's
        # executor/max_workers. Empty keeps serial in-process execution.
        self.backend_execution = dict(backend_execution or {})
        # Monotonic per-evaluation identity: every run_in_process()
        # call is one evaluation. The id guards the outcome routing
        # against stale/duplicate results and, with the trajectory and
        # task_slot, names the per-term runtime directories.
        self._evaluation_seq = 0
        # Per-step target states handed to backend.calculate(); set by
        # run() and consumed by run_in_process() through this
        # attribute because the local_executor closure is created
        # once. None keeps each backend's initialization value.
        self._step_target_states = None
        super().__init__(state=state, multitask=multitask,
                         remotetask=remotetask, multipool=multipool,
                         local_executor=local_executor,
                         executor_spec=executor_spec)
        self.runtime_idx = runtime_idx
        self.dummy_backend = dummy_backend
        # Backends grouped by category (theory/reference); built by
        # initialize() following the iter_tasks() order.
        self.backends = {role: [] for role in BACKEND_ROLES}
        self.execution_ids = {role: [] for role in BACKEND_ROLES}
        self._active_role = validate_backend_role(
            initial_role if initial_role is not None else 'theory')
        # A non-default role with an empty queue would fall through to
        # _run_single and fail on a missing backend; name the real cause.
        if self._active_role != 'theory' and not self._active_tasks():
            raise ValueError(
                "The '%s' backend role was requested but the state carries "
                "no '%s' tasks; a role needs its own task queue."
                % (self._active_role, self._active_role))
        self.logger = get_logger('pyrai3md.calculator')
        # The charge iteration of the last QMMM evaluation, reported with
        # the assembled result and cleared by every evaluation that does
        # not run the QMMM scheme.
        self._qmmm_iteration = None

    # ---- PyRAIState ownership interface ----

    @property
    def state(self):
        """PyRAIState owned by the Calculator (read access)."""
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    def get_state(self):
        """Return the PyRAIState owned by the Calculator.

        Returns
        -------
        PyRAIState or None
            The runtime state (sole communication object).
        """
        return self._state

    def update(self, updates):
        """Apply updates to the owned PyRAIState.

        The only supported way for workflows and other modules to
        modify the state. Recognized keys:
        - 'system': dict routed to state.update_system().
        - 'results': dict routed to state.set_result().
        - any other key: attribute assignment on the state.

        Parameters
        ----------
        updates : dict
            State updates to apply.

        Returns
        -------
        PyRAIState or None
            The updated state.
        """
        if self._state is None or not updates:
            return self._state
        for key, value in updates.items():
            if key == 'system' and isinstance(value, dict):
                self._state.update_system(value)
            elif key == 'results':
                self._state.set_result(value)
            else:
                setattr(self._state, key, value)
        return self._state

    # ---- Backend role grouping ----

    @property
    def active_role(self):
        """Currently active backend role ('theory' or 'reference')."""
        return self._active_role

    def switch_role(self, role):
        """Switch the active backend role.

        run() executes only the backends of the active role; the two
        roles are never executed simultaneously. The dispatch flags
        follow the role: the base node triple of the new role (from
        the frozen execution graph) is applied when available.

        Parameters
        ----------
        role : str
            'theory' or 'reference'.

        Returns
        -------
        Calculator
            self, for call chaining.

        Raises
        ------
        ValueError
            For unknown roles.
        """
        role = validate_backend_role(role)
        self._active_role = role
        flags = self.base_flags.get(role)
        if flags is not None:
            # Universal rule 3 keeps the base node local: only
            # multitask/multipool are applied.
            self.multitask = bool(flags[0])
            self.multipool = max(1, int(flags[2]))
        return self

    def _active_tasks(self):
        """Return the task descriptors of the active role.

        Returns
        -------
        list of dict
            Tasks of the active role in queue order.
        """
        if self._state is None:
            return []
        return [task for category, task in self._state.iter_tasks()
                if category == self._active_role]

    def close(self):
        """Release every backend that owns a process or a temporary tree.

        The unit that built this Calculator owns its backends: an
        adaptive trajectory child ends with the loop, and a resident
        LAMMPS service or a skill calculation directory must not outlive
        it. `execute()` calls this when the unit it dispatched returns,
        which is what keeps a batch of units run one after another in a
        single process from accumulating the engines of every unit before
        it. A backend without a close() has nothing to release. Cleanup
        failures are logged, never allowed to turn a completed trajectory
        into a failure.
        """
        for role, backends in self.backends.items():
            for backend in backends:
                release = getattr(backend, 'close', None)
                if not callable(release):
                    continue
                try:
                    release()
                except Exception as error:       # noqa: BLE001
                    self.logger.warning(
                        "Backend cleanup failed for %s (%s): %s",
                        role, type(backend).__name__, error)
            self.backends[role] = []

    def compose_execution_id(self, task):
        """Compose the execution namespace for one task.

        Combines the static planning identifier (task static_id, a
        semantic 'tmp_[{region}_{role}[_task{N}]_]{SKILL}' directory
        name) with the runtime suffix provided by the Workflow layer:

            execution_id = static_id                 (single-shot)
            execution_id = static_id_{runtime_idx}   (runtime suffix)

        Isolated-workdir runtimes (each batch unit owns its working
        directory) keep the plain static_id: the directory is
        already unique, so the suffix would only duplicate the
        trajectory index.

        Backend working directories are derived exclusively from
        this identifier.

        Parameters
        ----------
        task : dict
            Task descriptor with static identifiers.

        Returns
        -------
        str
            Execution identifier.
        """
        static_id = task.get('static_id') or task.get('task_id') or (
            "%s_%s" % (task.get('role', 'task'), task.get('skill_name', ''))
        )
        runtime = getattr(self._state, 'runtime', None)
        isolated = bool(getattr(runtime, 'isolated_workdir', False))
        if self.runtime_idx and not isolated:
            return "%s_%s" % (static_id, self.runtime_idx)
        return static_id

    def initialize(self):
        """Runtime initialization stage.

        Responsible for:
        1. Assigning the execution namespace (execution_id) to every
           task in the queue.
        2. Instantiating one backend per task through task['api']
           with its task config and execution_id, grouped by the
           task category (theory/reference).
        3. Initializing runtime objects (RuntimeState when missing).
        4. Initializing logging under the execution namespace.

        Calling initialize() again clears and rebuilds all backends
        (idempotent restart support).

        Returns
        -------
        Calculator
            self, for call chaining.
        """
        # Idempotent rebuild
        self.backends = {role: [] for role in BACKEND_ROLES}
        self.execution_ids = {role: [] for role in BACKEND_ROLES}

        if self._state is not None:
            for category, task in self._state.iter_tasks():
                execution_id = self.compose_execution_id(task)
                self.execution_ids[category].append(execution_id)

                backend_api_path = task.get('api')
                if self.dummy_backend is not None:
                    self.backends[category].append(self.dummy_backend)
                    continue
                if backend_api_path is None:
                    self.backends[category].append(None)
                    continue

                # Resolve backend class from import path
                from ..skills.query import resolve_api
                backend_cls = resolve_api(backend_api_path)

                # Deep copy the task config so each backend owns an
                # independent working copy; the state task keeps the
                # canonical reference.
                task_config = task.get('config')
                if task_config is not None:
                    task_config = deepcopy(task_config)

                try:
                    backend = backend_cls(
                        config=task_config,
                        execution_id=execution_id,
                    )
                except TypeError:
                    # Backends without execution namespace support
                    # (e.g. plain mock backends in tests)
                    backend = backend_cls(task_config)
                self.backends[category].append(backend)

            # Initialize runtime objects
            if self._state.runtime is None:
                self._state.runtime = RuntimeState()

            # Namespace-scoped logger (active role first)
            ordered_ids = (self.execution_ids[self._active_role]
                           + self.execution_ids[
                               'reference' if self._active_role == 'theory'
                               else 'theory'])
            if ordered_ids:
                self.logger = get_logger(
                    "pyrai3md.calculator.%s" % ordered_ids[0],
                )

            # Calculator-level dispatch log: supplement skill name and
            # execution_mode (only the calculator holds this info;
            # Dispatcher.__init__ already logged multitask/remotetask).
            active_tasks = self._active_tasks()
            if active_tasks:
                first_task = active_tasks[0]
                self.logger.info(
                    "Calculator dispatch: skill=%s, execution_mode=%s",
                    first_task.get('skill_name', 'unknown'),
                    self._task_execution_mode(first_task))

        return self

    def _snapshot_cell(self):
        """Freeze the current periodic state as a plain-dict snapshot.

        The per-evaluation cell channel (dev/methods/
        PBC_DEVELOPMENT.md section 5.3): the work item carries the cell
        frozen at the same moment as the geometry snapshot, so a
        variable-cell integrator step always evaluates with its own
        deformed cell instead of the previous step's synced config, and
        with the image counts that say which copy of the cell each atom
        sits in. Plain list/float/int types only (pickle/JSON safe for
        the thread pool and the remote term worker).

        Returns
        -------
        dict or None
            {'lattice_vector', 'cell', 'pbc', 'volume', 'image'}, or None
            when the system carries no cell (non-periodic default).
            'image' is the per-atom count of cell lengths the coordinate
            wrap made cross; an engine that tracks periodic images needs
            it beside the coordinates it is handed, because wrapped
            positions alone do not say which image an atom sits in. A run
            that predates the channel reports an empty list, which every
            consumer reads as "no crossings".
        """
        system = getattr(getattr(self._state, 'pmd', None), 'system', None)
        if system is None:
            return None
        pbc = [int(p) for p in (getattr(system, 'pbc', None) or [])]
        lattice = getattr(system, 'lattice_vector', None) or []
        lattice = [[float(x) for x in row] for row in lattice]
        has_lattice = any(x for row in lattice for x in row)
        if not any(pbc) and not has_lattice:
            return None
        image = [[int(x) for x in row]
                 for row in (getattr(system, 'image', None) or [])]
        return {
            'lattice_vector': lattice,
            'cell': [float(x) for x in (getattr(system, 'cell', None) or [])],
            'pbc': pbc,
            'volume': float(getattr(system, 'volume', 0.0) or 0.0),
            'image': image,
        }

    def _finalize_constraints(self, data):
        """Add every declared additive potential to one result mapping.

        Both execution paths call this at the same point in their contract -
        after the provider's numbers are assembled, before they are written
        back - so a bias cannot be applied on one path and missed on the other.
        The adaptive-private bias runs first and keeps the unbiased channels;
        the constraint call is a short circuit when the run declares no
        constraint action.
        """
        data = analysis_bias.finalize(self._state, data)
        return constraint_runtime.finalize(self._state, data,
                                           logger=self.logger)

    def _apply_soc_bias(self, data):
        """Add the declared constant bias to one result's SOC couplings.

        `&md addsoc` shifts every reported spin-orbit coupling by a constant
        (the archive keyword of the same name did the same to the SOC matrix).
        It is applied here rather than in the hopping integrator because the
        biased matrix is what the step must record: a hop decision, a history
        frame and a resumed run all read `results['soc']`, and a bias applied
        inside the integrator would leave the record describing couplings
        nothing propagated with.

        The bias is in the unit the backend reports, cm^-1 (the same currency
        as the `&adaptive` SOC thresholds, maxsoc 50.0 / minsoc 20.0), and it
        is applied only to a run that asked for spin-orbit couplings through
        `dosoc`: a bias on an array no one requested would invent a coupling
        the input never declared.
        """
        dynamics = getattr(getattr(self._state, 'pmd', None), 'dynamics', None)
        bias = float(getattr(dynamics, 'addsoc', 0.0) or 0.0)
        if bias == 0.0 or int(getattr(dynamics, 'dosoc', 0) or 0) == 0:
            return data
        soc = data.get('soc')
        if soc is None:
            return data
        values = np.asarray(soc, dtype=np.float64)
        if values.size == 0:
            return data
        data['soc'] = (values + bias).tolist()
        return data

    def _run_single(self, target_states=None):
        """Single-shot execution without a task queue.

        Used for dummy-backend testing when the state carries no
        task queue. Runs one backend call over the full structure.

        Parameters
        ----------
        target_states : list of int or None
            Per-step target states forwarded to the backend.

        Returns
        -------
        Result
            Computed result.
        """
        backend = self.dummy_backend
        active_backends = self.backends.get(self._active_role, [])
        if backend is None and active_backends:
            backend = active_backends[0]
        atoms = list(self._state.atoms)
        coordinates = np.asarray(self._state.coordinates, dtype=np.float64)
        result_dict = backend.calculate(atoms, coordinates,
                                        target_states=target_states,
                                        cell=self._snapshot_cell())
        result_dict = self._finalize_constraints(result_dict)
        result_dict = self._apply_soc_bias(result_dict)
        tasks = self._active_tasks()
        result = adapt_skill_result(
            result_dict, tasks[0].get('skill_name') if tasks else None)
        if self._state is not None:
            self._state.set_result(result.to_dict())
        return result

    def run(self, target_states=None):
        """Execute the active role task queue once via self.dispatch().

        Calls self.dispatch() with a single-unit list: a single-point
        computation is a unit list of length 1, fully unified with
        batch execution. The in-process execution body (run_in_process)
        runs the active role backends only; the two roles are never
        executed simultaneously.

        Parameters
        ----------
        target_states : list of int or None
            Per-step target states chosen by the workflow (e.g. the
            NAMD workflow after a surface hop). Forwarded to every
            backend.calculate() of this run; None keeps each
            backend's initialization value.

        Returns
        -------
        Result
            Computed result (raw or composite-aggregated).
        """
        # Stage the per-step target states for run_in_process(); the
        # local_executor closure is created once, so the value travels
        # through an instance attribute.
        self._step_target_states = target_states
        if self.local_executor is None:
            calculator = self

            def _in_process_executor(unit_state, runtime_idx=None):
                del unit_state, runtime_idx  # calculator owns the live state
                return calculator.run_in_process()

            self.local_executor = _in_process_executor
        outcomes = self.dispatch(
            units=[{'state': self._state, 'task_index': 0}])
        return outcomes[0]

    def _active_descriptor(self):
        """Return the backend-terms execution descriptor of the role.

        Returns
        -------
        dict or None
            The descriptor stamped on the active role's base node by
            the planner (r3b); None when the Calculator was built
            without a frozen graph (direct/test construction), which
            keeps serial in-process execution.
        """
        return self.backend_execution.get(self._active_role)

    def _resolve_executor(self):
        """Build the TaskExecutor of the active role's backend terms.

        The term-level executor is orthogonal to the base node
        multitask (the outer Dispatcher fan-out): an MD base node keeps
        multitask=False while its terms may still run in a thread pool.
        Serial is the default, so the historical single-threaded
        behavior is unchanged unless the descriptor requests 'thread'.

        Returns
        -------
        TaskExecutor
            Serial or thread executor of the backend terms.
        """
        descriptor = self._active_descriptor()
        if not descriptor:
            return TaskExecutor(mode='serial')
        mode = descriptor.get('executor', 'serial')
        max_workers = int(descriptor.get('max_workers', 1) or 1)
        return TaskExecutor(mode=mode, max_workers=max_workers)

    def _next_evaluation_id(self):
        """Advance and return this evaluation's identity.

        Every run_in_process() call is one evaluation of the active
        role task queue. The monotonic id guards the outcome routing
        against stale/duplicate results and, with the trajectory and
        task_slot, names the per-term runtime directories.

        Returns
        -------
        int
            Strictly increasing evaluation identity.
        """
        self._evaluation_seq += 1
        return self._evaluation_seq

    def _task_execution_mode(self, task):
        """Read the skill execution_mode of one task's config.

        Mirrors the source initialize() logs and the backend itself
        reads (``skill_config['execution_mode']``, the field
        skills/<skill>/service/runner selects its runner from), so the
        audit key derives from the same value the backend acts on. The
        framework's dispatch decision is written here by
        planning.assembly.build_task_queue when the run goes remote.

        Parameters
        ----------
        task : dict
            Task descriptor.

        Returns
        -------
        str
            Execution mode ('local' default).
        """
        config = task.get('config')
        if not isinstance(config, dict):
            return 'local'
        skill_config = config.get('skill_config') or {}
        return skill_config.get('execution_mode', 'local')

    def _term_audit_key(self, task):
        """Derive the thread-safety contract of one backend term.

        The tuple is the contract the backend is asked about before a
        thread pool starts: (backend_name, runner, execution_mode,
        profile_semantics). It is derived deterministically from the
        task so a skill author writes check_thread_safe() against the
        same combination the calculator stages:

        - backend_name: the skill name (planning-stable identity);
        - runner: 'in_process' (the calculator runs terms locally; the
          remote per-term worker of r3e uses 'term_worker');
        - execution_mode: skill_config['execution_mode'] (the value the
          backend itself selects its runner from);
        - profile_semantics: the sorted capability set of the task
          (what the term computes), joined by '+'.

        Parameters
        ----------
        task : dict
            Task descriptor.

        Returns
        -------
        tuple(str, str, str, str)
            (backend_name, runner, execution_mode, profile_semantics).
        """
        backend_name = task.get('skill_name') or ''
        capabilities = task.get('capabilities') or []
        profile_semantics = '+'.join(
            sorted(str(c) for c in capabilities)) or 'energy'
        return (backend_name, 'in_process',
                self._task_execution_mode(task), profile_semantics)

    def _build_work_items(self, tasks, active_backends, snapshot_atoms,
                          snapshot_coords, target_states, evaluation_id,
                          charge_store=None, embedding_regions=None,
                          snapshot_cell=None, qmmm_inputs=None,
                          payloads_by_task=None):
        """Stage one unified work item per task of the active role.

        Materialises each task's local region from the frozen snapshot
        and clears its staging (config.results) so the upcoming backend
        call is the sole data source. The backend instances are the
        persistent per-task_slot objects built by initialize(): the
        positional alignment between _active_tasks() and
        self.backends[role] (both follow iter_tasks() order) keeps one
        backend per task_slot across steps, so a term reuses its own
        wavefunction cache. No live state is read here beyond the
        already-frozen snapshot.

        Parameters
        ----------
        tasks : list of dict
            Active role task descriptors (queue order).
        active_backends : list
            Persistent backend instances (parallel to tasks).
        snapshot_atoms : list of str
            Frozen global atom symbols.
        snapshot_coords : ndarray
            Frozen global coordinates (read-only).
        target_states : list of int or dict or None
            Per-step target states: a flat list forwarded to every
            backend (C=1), or a per-center map {center_index: list}
            routed to each task by its own center (C>1).
        evaluation_id : int
            This evaluation's identity.
        charge_store : ChargeStore or None
            Bound charge store; when given, every high-level provider is
            embedded in its own center's midlevel. None (the default)
            delivers no embedding, leaving a plain run unchanged.
        embedding_regions : dict or None
            {center_index: [midlevel global indices]} used to build the
            embedding field of each high-level provider.
        snapshot_cell : dict or None
            Frozen periodic-cell snapshot of this evaluation (see
            _snapshot_cell). Only whole-system tasks inherit it: a
            sliced region is a local cluster model and does not
            silently inherit the global periodic cell
            (dev/methods/PBC_DEVELOPMENT.md section 5.3).
        qmmm_inputs : dict or None
            Per-step payload for one LAMMPS ``qmmm_energy`` work item
            (``q_qm`` and the native ``v_mm``/``e_mm``); None for every
            other task.
        payloads_by_task : dict or None
            ``{task_id: {'external_potential': rows, 'qmmm_inputs': dict}}``
            for a batch whose tasks each carry their own payload. Several
            centers share one batch, and a broadcast payload would hand
            every center the same field and every classical task the same
            charge set; a per-task entry overrides the broadcast value.

        Returns
        -------
        list of dict
            Unified work items (see terms.make_work_item).
        """
        work_items = []
        # A periodic run cuts its regions out of the continuous view rebuilt
        # from this snapshot's own crossing counts, so a center whose atoms
        # straddle a box face arrives as one connected cluster instead of as
        # two fragments 9 Angstrom apart. The cell of that instant is used,
        # which is what makes a variable-cell run follow: the region is
        # re-derived every evaluation rather than transported once.
        periodic_view = None
        if snapshot_cell and snapshot_cell.get('image'):
            periodic_view = continuous_coordinates(snapshot_coords,
                                                   snapshot_cell)
        for index, task in enumerate(tasks):
            region_indices = list(task.get('region_indices', []))
            region = task.get('region', 'all')
            # Per-task payload overrides: several centers share one batch,
            # so each one's field rows and QM/MM payload have to travel with
            # its own item instead of being broadcast over the batch.
            override = (payloads_by_task or {}).get(
                task.get('task_id')) or {}
            task_qmmm_inputs = override.get('qmmm_inputs', qmmm_inputs)
            task_field_rows = override.get('external_potential')
            # Materialise local geometry from the frozen snapshot
            # (fresh writable copies; the snapshot stays read-only).
            local_atoms, local_coords = materialize_region(
                region_indices if region != 'all' else [],
                snapshot_atoms,
                snapshot_coords if periodic_view is None else periodic_view)
            # Clear stale results so the new backend call is the sole
            # data source.
            invalidate_task_result(task)

            backend = (self.dummy_backend
                       if self.dummy_backend is not None
                       else active_backends[index])
            request = task.get('property_request_spec') or {}
            backend_name, runner, exec_mode, profile_semantics = \
                self._term_audit_key(task)
            # Electrostatic embedding: only a high-level provider is
            # embedded, and only by its own center's midlevel. A plain
            # run passes charge_store=None, so point_charges stays None
            # and the backend call is unchanged. The field is read in the
            # same frame the region geometry was cut from: charges on the
            # main image next to an unwrapped cluster would put every
            # solvent molecule on the far side of the box.
            point_charges = None
            if charge_store is not None:
                point_charges = embedding_for_task(
                    charge_store, task,
                    snapshot_coords if periodic_view is None
                    else periodic_view,
                    embedding_regions)
            if point_charges is None and task.get('external_charges'):
                # Arbitrary external charge rows (standalone background
                # field); they are never sliced or tied to atom indices.
                point_charges = [list(row)
                                 for row in task['external_charges']]
            task_external_potential = None
            n_global = len(snapshot_atoms)
            if task_field_rows is not None:
                # Already narrowed to this task's center by the caller that
                # read its own probe handoff: the length is the region's.
                if len(task_field_rows) != len(local_atoms):
                    raise ValueError(
                        "QMMM field rows number %d but task '%s' has %d "
                        "local atoms." % (len(task_field_rows),
                                          task.get('task_id', '?'),
                                          len(local_atoms)))
                task_external_potential = task_field_rows
            if task_external_potential is None and task.get('external_potential'):
                # A standalone background field: one row per QM atom of this
                # task, read from '{title}.charge' by the queue builder. It
                # replaces the point-charge channel rather than joining it -
                # one file declares one kind - so a task that somehow carried
                # both is refused instead of picking one silently.
                if point_charges is not None:
                    raise ValueError(
                        "task '%s' carries both external charges and an "
                        "external field; a background embedding is one or "
                        "the other." % task.get('task_id', '?'))
                rows = [list(row) for row in task['external_potential']]
                if len(rows) not in (len(local_atoms), n_global):
                    raise ValueError(
                        "external field has %d rows but task '%s' has %d "
                        "local atoms (global %d)."
                        % (len(rows), task.get('task_id', '?'),
                           len(local_atoms), n_global))
                if len(rows) != len(local_atoms):
                    rows = [rows[i] for i in region_indices]
                task_external_potential = rows
            # Region-sliced tasks are local cluster models: they never
            # inherit the global periodic cell implicitly.
            task_cell = (snapshot_cell
                         if region == 'all' or not region_indices
                         else None)
            work_items.append(make_work_item(
                task_id=task.get('task_id', ''),
                backend=backend,
                local_atoms=local_atoms,
                local_coords=local_coords,
                task_slot=task.get('task_slot'),
                center_index=task.get('center_index'),
                evaluation_id=evaluation_id,
                target_states=_task_target_states(target_states, task),
                nac_pairs=request.get('nac_pairs'),
                soc_pairs=request.get('soc_pairs'),
                point_charges=point_charges,
                external_potential=task_external_potential,
                qmmm_inputs=task_qmmm_inputs,
                cell=task_cell,
                identity={
                    'trajectory': self.runtime_idx,
                    'task_slot': task.get('task_slot'),
                    'center_index': task.get('center_index'),
                    'evaluation_id': evaluation_id,
                },
                backend_name=backend_name,
                runner=runner,
                execution_mode=exec_mode,
                profile_semantics=profile_semantics,
            ))
        return work_items

    def _validate_outcomes(self, work_items, outcomes, evaluation_id):
        """Validate the outcome set against the staged work items.

        Enforces the routing invariants of dev/MULTICENTER_DEVELOPMENT.md
        (R3): one outcome per work item (completeness), no task_id
        appearing twice (no duplicate) and every outcome carrying the
        current evaluation_id (no stale). An out-of-order thread
        completion never trips these checks because the executor
        returns index-aligned outcomes.

        Parameters
        ----------
        work_items : list of dict
            Staged work items.
        outcomes : list of dict
            Executor outcomes (all successful when reached).
        evaluation_id : int
            This evaluation's identity.

        Raises
        ------
        RuntimeError
            On a count mismatch, a misaligned/duplicate task_id, or a
            stale evaluation_id.
        """
        if len(outcomes) != len(work_items):
            raise RuntimeError(
                "Evaluation %r produced %d outcome(s) for %d work "
                "item(s); the executor must return one index-aligned "
                "outcome per term."
                % (evaluation_id, len(outcomes), len(work_items)))
        seen = set()
        for item, outcome in zip(work_items, outcomes):
            task_id = outcome.get('task_id')
            if task_id != item.get('task_id'):
                raise RuntimeError(
                    "Outcome task_id %r does not match work item %r "
                    "(evaluation %r); routing is misaligned."
                    % (task_id, item.get('task_id'), evaluation_id))
            if outcome.get('evaluation_id') != evaluation_id:
                raise RuntimeError(
                    "Stale outcome for task '%s': evaluation %r != "
                    "current %r."
                    % (task_id, outcome.get('evaluation_id'),
                       evaluation_id))
            if task_id in seen:
                raise RuntimeError(
                    "Duplicate outcome for task '%s' in evaluation %r."
                    % (task_id, evaluation_id))
            seen.add(task_id)

    def _route_outcomes(self, tasks, outcomes, is_multi):
        """Route outcomes into TaskOutcomes in the main thread.

        Applies the fin_coord channel and stores each result into its
        task config AFTER execution, in task order, so worker threads
        never write the live state or the parent geometry. The
        summation order is fixed by the task queue order (the outcomes
        are index-aligned), never by the completion order.

        Parameters
        ----------
        tasks : list of dict
            Active role task descriptors (queue order).
        outcomes : list of dict
            Successful executor outcomes (index-aligned with tasks).
        is_multi : bool
            Whether the evaluation has more than one task.

        Returns
        -------
        list of TaskOutcome
            Per-task outcomes for the strict aggregation.

        Raises
        ------
        ValueError
            When a multi-task composite term returns fin_coord (sub
            calculations must not update the global geometry).
        """
        routed = []
        for task, outcome in zip(tasks, outcomes):
            result_dict = outcome.get('raw') or {}
            # A skill may return its numbers in native units with a manifest
            # declaration of those units (LAMMPS does). Apply it here, before
            # any composite aggregation or task-config write-back, so a
            # mixed-skill sum never adds kcal/mol to Hartree. The conversion
            # removes its own marker, so a second pass is a no-op.
            result_dict = apply_result_units(
                result_dict, task.get('skill_name'))
            outcome['raw'] = result_dict
            # Multi-task composite: reject the fin_* channels (the
            # framework optimizer/MD updates the global state).
            if is_multi and result_dict:
                for channel in ('fin_coord', 'fin_velocity', 'fin_cell'):
                    if result_dict.get(channel):
                        raise ValueError(
                            "Task '%s' returned %s in a multi-task "
                            "composite; sub-calculations must not update "
                            "the global geometry."
                            % (task.get('task_id'), channel))
            # Single-task: apply the reported final state if present (the
            # main thread is the sole writer of the live state). Geometry,
            # then velocities, then the cell: the kinetic energy and the
            # volume each describe the snapshot they are derived from.
            if not is_multi:
                self._apply_skill_geometry(task, result_dict)
                self._apply_skill_velocities(task, result_dict)
                self._apply_skill_cell(task, result_dict)
            # Store result in task config (backward compatibility).
            store_task_result(task, result_dict)
            routed.append(TaskOutcome(
                task_id=task.get('task_id', ''),
                contribution=task.get('contribution', 1),
                raw=result_dict,
                region_indices=list(task.get('region_indices', [])),
                region=task.get('region', 'all'),
            ))
        return routed

    def _execute_terms(self, work_items, tasks, remote, descriptor):
        """Execute one batch of work items, remote per-term or local.

        Shared by the single-batch and the two-phase paths so both apply
        the same failure policy and return the same outcome shape.

        Parameters
        ----------
        work_items : list of dict
            Staged work items of this batch.
        tasks : list of dict
            Task descriptors index-aligned with work_items.
        remote : bool
            Whether the active descriptor routes terms to the remote
            per-term worker.
        descriptor : dict or None
            Active backend-terms execution descriptor.

        Returns
        -------
        list of dict
            Index-aligned outcomes of this batch.
        """
        if remote:
            return apply_failure_policy(execute_remote_terms(
                work_items, tasks, self._state, descriptor))
        executor = self._resolve_executor()
        return execute_with_failure_handling(work_items, executor)

    def _embedding_regions_by_center(self):
        """Per-center midlevel, the ONIOM electrostatic embedding source.

        ``highlevel`` and ``midlevel`` are disjoint, so a center's
        embedding region is exactly its midlevel.

        Returns
        -------
        dict
            {center_index: [midlevel global indices]}; empty when no
            partition plan is bound.
        """
        plan = getattr(self._state, 'partition_plan', None)
        if plan is None:
            return {}
        if plan.n_layer == 2:
            control = getattr(getattr(self._state, 'pmd', None), 'control', None)
            multiscale = str(getattr(control, 'multiscale_type', '') or '').lower()
            if multiscale == 'qmmm':
                return {}
            # ONIOM2 outer-charge embedding: the high-level model is
            # embedded in all atoms outside its own region.
            system = getattr(getattr(self._state, 'pmd', None), 'system', None)
            natom = int(getattr(system, 'natom', 0)
                        or len(getattr(system, 'coordinates', []) or [])
                        or len(getattr(system, 'atoms', []) or []))
            regions = {}
            for c in range(plan.n_center):
                high = set(int(i) for i in plan.center(c).high_indices)
                regions[c] = [index for index in range(natom)
                              if index not in high]
            return regions
        mid_centers = [list(plan.center(c).mid_indices)
                       for c in range(plan.n_center)]
        return compute_embedding_regions(mid_centers)

    def _run_two_phase(self, tasks, active_backends, snapshot_atoms,
                       snapshot_coords, target_states, evaluation_id,
                       charge_store, embedding_regions, is_multi, remote,
                       descriptor, snapshot_cell=None):
        """Execute an embedded ONIOM3 evaluation in two phases.

        Phase 1 runs every non-provider term -- the full-system baseline
        L(real), the mid-level inter source M(I_c) and all the low/mid
        corrections -- with no embedding; its outcomes update the
        mid-shell charges. Phase 2 then runs the high-level providers
        H(c), each embedded in its own center's freshly updated mid
        shell, and updates the highlevel charges. Both phases share one
        evaluation_id; the routed outcomes are merged back into the
        original task order so the assembly and the index-aligned
        gradient scatter stay unchanged.

        Returns
        -------
        list of TaskOutcome
            Routed outcomes index-aligned with tasks.
        """
        # The ONIOM2 outer reference is the field every centre is
        # embedded in; keep the rows this evaluation used so the log
        # can print the environment (the persistent store keeps
        # highlevel rows provider-owned, #450).
        self._oniom_outer_charges = None
        phase1_pairs = [(i, t) for i, t in enumerate(tasks)
                        if not t.get('provides_electronic')]
        phase2_pairs = [(i, t) for i, t in enumerate(tasks)
                        if t.get('provides_electronic')]
        routed = [None] * len(tasks)
        # Which rows the high-level provider owns is read off the whole
        # queue, not off the phase being scattered: phase 1 holds no
        # provider, so asked of phase 1 alone the answer is "none", and
        # theory[1] would write over the high-level rows.
        charge_sources = region_charge_sources(tasks)
        state_of = lambda task: self._charge_embedding_state(
            task, target_states)
        # ONIOM2 has no mid-level source: the full-system low-level row from
        # phase one is the environment for every center (docs ONIOM section 7).
        # It is kept apart from the persistent store so another center's
        # high-level atoms can carry low-level charges without writing them
        # under the high-level name (#450).
        outer_field = None

        for pairs, embed in ((phase1_pairs, False), (phase2_pairs, True)):
            orig_indices = [i for i, _ in pairs]
            phase_tasks = [t for _, t in pairs]
            phase_backends = [active_backends[i] for i in orig_indices]
            store = None
            if embed:
                store = outer_field if outer_field is not None else charge_store
            _t = time.perf_counter()
            work_items = self._build_work_items(
                phase_tasks, phase_backends, snapshot_atoms,
                snapshot_coords, target_states, evaluation_id,
                charge_store=store, embedding_regions=embedding_regions,
                snapshot_cell=snapshot_cell)
            self._profile('build_items', _t)
            _t = time.perf_counter()
            outcomes = self._execute_terms(
                work_items, phase_tasks, remote, descriptor)
            self._validate_outcomes(work_items, outcomes, evaluation_id)
            phase_routed = self._route_outcomes(
                phase_tasks, outcomes, is_multi)
            update_from_outcomes(
                charge_store, phase_tasks, phase_routed,
                sources=charge_sources, state=state_of)
            if not embed and not charge_sources[1]:
                values = capture_outer_reference(
                    phase_tasks, phase_routed, state=state_of)
                natom = len(snapshot_atoms)
                if values is not None and len(values) == natom:
                    outer_field = ChargeStore(natom)
                    outer_field.update(range(natom), values)
                    self._oniom_outer_charges = outer_field.point_charges(
                        snapshot_coords).tolist()
                elif values is not None:
                    self.logger.warning(
                        "ONIOM2 embedding: the outer charge row has %d rows "
                        "for %d atoms; the high-level centres run without "
                        "environment charges.", len(values), natom)
            for orig_i, routed_item in zip(orig_indices, phase_routed):
                routed[orig_i] = routed_item
        return routed

    @staticmethod
    def _rows_payload(centers, rows_by_center):
        """The per-task payloads that hand each center its V/E rows."""
        return {task.get('task_id'): {
                    'external_potential': rows_by_center[
                        task.get('center_index')]}
                for task in centers if rows_by_center}

    def _probe_field_pass(self, field_tasks, field_indexes, active_backends,
                          snapshot_atoms, snapshot_coords, target_states,
                          evaluation_id, snapshot_cell, remote, descriptor,
                          source, qm_tasks):
        """One A/B pass of every center's field probe.

        The probe is a dependency of the evaluation, not a term of it: it
        reads the current charges and writes none of the evaluation's
        state, so it is a method of its own rather than twenty-argument
        code inside `_run_qmmm`.

        Parameters
        ----------
        field_tasks : list of dict
            The plan's field tasks, one per center, in center order.
        field_indexes : list of int
            Queue positions of those tasks (their backend instances).
        active_backends : list
            Backends in queue order.
        snapshot_atoms, snapshot_coords :
            The frozen geometry of this evaluation.
        target_states : list or None
            Requested states, forwarded to the work items.
        evaluation_id : object
            Identity of this evaluation, forwarded for validation.
        snapshot_cell : dict or None
            The frozen cell.
        remote : bool
            Whether terms run on remote workers.
        descriptor : dict or None
            The active backend-terms descriptor.
        source : sequence of float
            Charges the probe embeds, one per QM atom in union order.
        qm_tasks : list of dict
            QM tasks paired with the field tasks by queue position.

        Returns
        -------
        tuple
            (outcomes, blocks, rows_by_center): the field outcomes, their
            `qmmm_field` blocks, and each center's V/E rows.
        """
        payloads = {
            task.get('task_id'): {
                'qmmm_inputs': {'q_qm': list(source)}}
            for task in field_tasks}
        items = self._build_work_items(
            field_tasks,
            [active_backends[index] for index in field_indexes],
            snapshot_atoms, snapshot_coords, target_states,
            evaluation_id, snapshot_cell=snapshot_cell,
            payloads_by_task=payloads)
        outcomes = self._execute_terms(
            items, field_tasks, remote, descriptor)
        self._validate_outcomes(items, outcomes, evaluation_id)
        rows_by_center = {}
        blocks = []
        for outcome, task, backend, qm_task in zip(
                outcomes, field_tasks,
                [active_backends[index] for index in field_indexes],
                qm_tasks):
            self._convert_outcome(outcome, task)
            block = (outcome.get('raw') or {}).get('qmmm_field')
            if not isinstance(block, dict) or not block:
                raise ValueError(
                    "QMMM field task '%s' returned no qmmm_field block."
                    % task.get('task_id'))
            # QMMM theory[0] embeds through the field V/E; the charge
            # representation is ONIOM-only. A probe is paired with the QM
            # task of the same center by queue position: the plan expands
            # both lists in center order, and the handoff names the atoms
            # its rows belong to.
            rows_by_center[qm_task.get('center_index')] = (
                self._qmmm_external_potential(
                    backend, qm_task, block, task.get('skill_name')))
            blocks.append(block)
        return outcomes, blocks, rows_by_center

    def _run_charge_rounds(self, charge_tasks, charge_indexes,
                           active_backends, snapshot_atoms, snapshot_coords,
                           target_states, evaluation_id, snapshot_cell,
                           remote, descriptor, energy_task, union,
                           field_tasks, field_indexes, qm_tasks,
                           source_charges, rows_by_center, charges_by_atom,
                           field_outcomes, field_blocks):
        """Iterate the per-center 0-order charges to their fixed point.

        A round solves every center's electrons in the field the previous
        round's charges produced and asks for nothing else; the accepted
        charges are the field source of the property round and the
        classical surface's charge set, so the pass is probed once more
        before anything is solved with a gradient.

        Returns
        -------
        dict
            The evaluation's updated rows_by_center, field_outcomes,
            field_blocks, charge_outcomes, charges_by_atom,
            source_charges and the iteration record.
        """
        system = getattr(getattr(self._state, 'pmd', None), 'system', None)

        # --- the charge rounds: field in, 0-order charges out ----------
        # A round solves every center's electrons in the field the
        # previous round's charges produced and asks for nothing else.
        # All centers read one snapshot, so the order centers were
        # declared in cannot change the answer (docs R11).
        iterate = bool(getattr(system, 'sc_charge', False))
        maxiter = int(getattr(system, 'sc_charge_maxiter', 1) or 1)
        tolerance = float(getattr(system, 'sc_charge_threshold', 0.0)
                          or 0.0)
        # `previous` is the last *raw* fit and is what the residual is
        # measured on; `carried` is what the next round's field is built
        # from, which is the same number unless the run damps the update.
        # Keeping the two apart is the whole point: a tolerance read on the
        # damped move would shrink by the damping factor and call every
        # iteration converged.
        damping = float(getattr(system, 'sc_charge_damping', 1.0) or 1.0)
        previous = {task.get('center_index'):
                    [0.0] * len(task.get('region_indices') or [])
                    for task in charge_tasks}
        carried = {center: list(values)
                   for center, values in previous.items()}
        rounds = maxiter if iterate else 1
        residuals = []
        # The stopping test is one number over the whole system (the
        # largest center move decides), but the reader needs to know
        # which center was still moving when the round was accepted.
        center_residuals = {
            task.get('center_index'): [] for task in charge_tasks}
        converged = False
        charge_outcomes = []
        for _round in range(rounds):
            items = self._build_work_items(
                charge_tasks,
                [active_backends[index] for index in charge_indexes],
                snapshot_atoms, snapshot_coords, target_states,
                evaluation_id, snapshot_cell=snapshot_cell,
                payloads_by_task=self._rows_payload(
                    charge_tasks, rows_by_center))
            outcomes = self._execute_terms(
                items, charge_tasks, remote, descriptor)
            self._validate_outcomes(items, outcomes, evaluation_id)
            residual = 0.0
            for outcome, task in zip(outcomes, charge_tasks):
                self._convert_outcome(outcome, task)
                center = task.get('center_index')
                fitted = self._qmmm_charges(
                    task, outcome.get('raw') or {}, energy_task,
                    state=self._charge_embedding_state(task,
                                                       target_states),
                    require_readback=True)
                before = previous.get(center) or []
                center_residual = 0.0
                if len(before) == len(fitted):
                    center_residual = max(
                        [0.0]
                        + [abs(float(new) - float(old))
                           for new, old in zip(fitted, before)])
                    residual = max(residual, center_residual)
                center_residuals.setdefault(center, []).append(
                    float(center_residual))
                previous[center] = list(fitted)
                mixed = damp_charges(carried.get(center) or [], fitted,
                                     damping)
                carried[center] = list(mixed)
                for index, value in zip(
                        [int(i) for i in (task.get('region_indices')
                                          or [])], mixed):
                    charges_by_atom[index] = float(value)
            residuals.append(float(residual))
            charge_outcomes = list(outcomes)
            source_charges = [charges_by_atom[index]
                              for index in union]
            converged = bool(residual <= tolerance)
            if converged or not iterate:
                break
        # The accepted charges are the field source of the property
        # round and the classical surface's charge set, so the last thing
        # the loop produced is probed once more before anything is
        # solved with a gradient.
        if field_tasks:
            field_outcomes, field_blocks, rows_by_center = (
                self._probe_field_pass(
                    field_tasks, field_indexes, active_backends,
                    snapshot_atoms, snapshot_coords, target_states,
                    evaluation_id, snapshot_cell, remote, descriptor,
                    source_charges, qm_tasks))
        iteration = {
            'sc_charge': iterate,
            'rounds_run': len(residuals),
            'maxiter': rounds,
            'tolerance': tolerance,
            'residuals': residuals,
            'centers': [{'center_index': center,
                         'residuals': values}
                        for center, values in
                        sorted(center_residuals.items(),
                               key=lambda item: (
                                   item[0] is None, item[0]))],
            'converged': converged,
            # A budget of one round is a declaration, not a stall: saying
            # `maxiter` here would read like the fixed point escaped it.
            'stopped_by': ('tolerance' if converged
                           else ('single round' if (not iterate
                                                   or rounds == 1)
                                else 'maxiter')),
        }
        if damping != 1.0:
            iteration['damping'] = damping

        return {
            'rows_by_center': rows_by_center,
            'field_outcomes': field_outcomes,
            'field_blocks': field_blocks,
            'charge_outcomes': charge_outcomes,
            'charges_by_atom': charges_by_atom,
            'source_charges': source_charges,
            'iteration': iteration,
        }

    def _run_property_round(self, qm_tasks, qm_indexes, active_backends,
                            snapshot_atoms, snapshot_coords, target_states,
                            evaluation_id, snapshot_cell, remote, descriptor,
                            rows_by_center, energy_task, union, charge_tasks,
                            charges_by_atom, source_charges):
        """Solve every QM center with gradients, in the accepted field.

        The V/E each center is handed is the one the charge iteration
        accepted, and the log reports exactly those rows. Which charges
        the classical stage then sees depends on the plan: with charge
        rounds the accepted set is already the answer; with no charge task
        this round's own fitted charges are.

        Returns
        -------
        dict
            {'outcomes', 'charges_by_atom', 'source_charges'}.
        """
        # --- the property round: every QM center, with gradients -----------
        payloads = self._rows_payload(qm_tasks, rows_by_center)
        qm_items = self._build_work_items(
            qm_tasks, [active_backends[index] for index in qm_indexes],
            snapshot_atoms, snapshot_coords, target_states, evaluation_id,
            snapshot_cell=snapshot_cell, payloads_by_task=payloads)
        qm_outcomes = self._execute_terms(
            qm_items, qm_tasks, remote, descriptor)
        self._validate_outcomes(qm_items, qm_outcomes, evaluation_id)
        for outcome, task in zip(qm_outcomes, qm_tasks):
            self._convert_outcome(outcome, task)
            rows = (payloads.get(task.get('task_id')) or {}).get(
                'external_potential')
            if rows is not None:
                # The log reports the V/E actually handed to this center.
                outcome['raw']['qmmm_external_potential'] = [
                    list(row) for row in rows]

        # Each center's own fitted charges stay with it in its task config.
        # With several centers the classical stage is fed the charges the
        # final field was built from, not this round's output: the two
        # differ by the residual the iteration stopped on, and mixing them
        # would make the energy and the forces answer two different
        # electronic structures. With one center the field holds no QM
        # charge at all, so this round's charges are the only answer.
        if not charge_tasks:
            charges_by_atom = {}
            for outcome, task in zip(qm_outcomes, qm_tasks):
                q_center = self._qmmm_charges(
                    task, outcome.get('raw') or {}, energy_task,
                    state=self._charge_embedding_state(task, target_states))
                for index, value in zip(
                        [int(i) for i in (task.get('region_indices') or [])],
                        q_center):
                    charges_by_atom[index] = float(value)
            source_charges = [charges_by_atom[index] for index in union]

        return {'outcomes': qm_outcomes,
                'charges_by_atom': charges_by_atom,
                'source_charges': source_charges}

    def _run_classical_stage(self, energy_task, energy_indexes,
                             active_backends, snapshot_atoms,
                             snapshot_coords, target_states, evaluation_id,
                             snapshot_cell, remote, descriptor, union, q_qm,
                             embedding, field_blocks):
        """Evaluate the classical surface with the QM charges.

        The V/E the surface is reduced by are the total field each center
        was embedded in (MM plus the other centers' current charges),
        assembled over the whole QM set in atom order: one subtraction per
        QM row, no loop over centers. A probe that left a row missing, or
        named an atom outside this evaluation's QM set, is refused here
        rather than filled.

        Returns
        -------
        list
            The energy task's outcomes (one).
        """
        # --- Stage 3: the full force field with q_qm ----------------------
        qmmm_inputs = {'q_qm': list(q_qm)}
        if embedding and field_blocks:
            # The V/E the classical surface is reduced by are the total
            # field each center was embedded in (MM plus the other centers'
            # current charges), assembled over the whole QM set in atom
            # order: one subtraction per QM row, no loop over centers.
            position = {index: slot for slot, index in enumerate(union)}
            v_total = np.full(len(union), np.nan, dtype=np.float64)
            e_total = np.full((len(union), 3), np.nan, dtype=np.float64)
            for block in field_blocks:
                ids = [int(value) for value in
                       (block.get('qm_atom_ids') or [])]
                v_mm = np.asarray(block.get('v_mm_native'),
                                  dtype=np.float64)
                e_mm = np.asarray(block.get('e_mm_native'),
                                  dtype=np.float64)
                if (v_mm.shape != (len(ids),)
                        or e_mm.shape != (len(ids), 3)):
                    raise ValueError(
                        "QMMM probe returned V/E with shapes %s/%s for %d "
                        "atoms." % (v_mm.shape, e_mm.shape, len(ids)))
                if (not np.all(np.isfinite(v_mm))
                        or not np.all(np.isfinite(e_mm))):
                    raise ValueError(
                        "QMMM probe returned a non-finite V/E.")
                for atom_id, value, row in zip(ids, v_mm, e_mm):
                    slot = position.get(atom_id - 1)
                    if slot is None:
                        raise ValueError(
                            "QMMM probe returned rows for atom id %d, which "
                            "is outside this evaluation's QM set."
                            % atom_id)
                    v_total[slot] = value
                    e_total[slot] = row
            if (not np.all(np.isfinite(v_total))
                    or not np.all(np.isfinite(e_total))):
                raise ValueError(
                    "QMMM centers cover %d atoms but the probes left rows "
                    "missing: every QM atom must be probed once."
                    % len(union))
            qmmm_inputs['v_mm'] = v_total.tolist()
            qmmm_inputs['e_mm'] = e_total.tolist()
        energy_items = self._build_work_items(
            [energy_task], [active_backends[energy_indexes[0]]],
            snapshot_atoms,
            snapshot_coords, target_states, evaluation_id,
            snapshot_cell=snapshot_cell, qmmm_inputs=qmmm_inputs)
        energy_outcomes = self._execute_terms(
            energy_items, [energy_task], remote, descriptor)
        self._validate_outcomes(
            energy_items, energy_outcomes, evaluation_id)
        self._convert_outcome(energy_outcomes[0], energy_task)
        self._qmmm_atomic_block(energy_outcomes[0], energy_task)

        return energy_outcomes

    def _run_qmmm(self, tasks, active_backends, snapshot_atoms,
                  snapshot_coords, target_states, evaluation_id,
                  is_multi, remote, descriptor, snapshot_cell=None):
        """Execute one QM/MM evaluation on the plan's stages.

        The embedding decision selects the stages:

        * embedding=true: ``qmmm_field -> QM -> qmmm_energy``. The field
          task returns the MM potential/field (and, when read_charge=true,
          the MM atom charges); the QM task embeds in the potential/field
          or in the point charges; the energy task subtracts the 0-order
          QM-MM electrostatics the QM calculation already contains.
        * embedding=false: ``QM -> qmmm_energy``. The QM calculation is
          isolated and the energy task keeps the QM-MM electrostatics in
          the classical term (``E_classic = U_main - U_gas``).

        The stages share one evaluation_id and are routed once in queue
        order. Standard LAMMPS numbers are converted to Hartree /
        Hartree-per-Bohr as each stage returns; the QM/MM auxiliary
        blocks stay native for the assembler.

        Parameters
        ----------
        tasks : list of dict
            QMMM task descriptors in queue order (field?, QM, energy).
        active_backends : list
            Backend instances parallel to tasks.
        snapshot_atoms, snapshot_coords, target_states, evaluation_id
            This evaluation's frozen inputs.
        is_multi : bool
            Whether the queue has more than one task.
        remote : bool
            Whether the role routes terms to the remote worker.
        descriptor : dict or None
            Active backend-terms execution descriptor.
        snapshot_cell : dict or None
            Frozen periodic-cell snapshot of this evaluation.

        Returns
        -------
        list of TaskOutcome
            Routed outcomes index-aligned with tasks.

        Raises
        ------
        ValueError
            On a missing/invalid field or energy block, a partition
            mismatch, or a missing q_qm.
        """
        by_role = {'field': [], 'qm': [], 'energy': [], 'qm_charge': []}
        for index, task in enumerate(tasks):
            role = task.get('qmmm_role')
            if role not in by_role:
                raise ValueError(
                    "QMMM queue holds field, qm, qm_charge and energy duties "
                    "only; got roles %s."
                    % [t.get('qmmm_role') for t in tasks])
            by_role[role].append(index)
        field_indexes = by_role['field']
        qm_indexes = by_role['qm']
        energy_indexes = by_role['energy']
        if not qm_indexes:
            raise ValueError(
                "QMMM queue needs at least one QM task; got %s."
                % [t.get('qmmm_role') for t in tasks])
        if len(energy_indexes) != 1:
            # One classical evaluation of the whole system per geometry:
            # the force field is a single surface, and splitting it per
            # center would add its MM-MM terms once per center.
            raise ValueError(
                "QMMM queue needs exactly one energy task; got %d."
                % len(energy_indexes))
        system = getattr(getattr(self._state, 'pmd', None), 'system', None)
        embedding = bool(getattr(system, 'embedding', False))
        # The one input question the charge duties answer for. Read here
        # because a queue that asks it without the rounds to answer it is a
        # plan defect, and a queue that carries rounds nobody asked for is
        # one too.
        iterate_charges = bool(getattr(system, 'sc_charge', False))
        if embedding and len(field_indexes) != len(qm_indexes):
            raise ValueError(
                "QMMM with embedding needs one probe per QM center; got %d "
                "field and %d QM tasks." % (len(field_indexes),
                                            len(qm_indexes)))
        if not embedding and field_indexes:
            raise ValueError(
                "QMMM without embedding must hold no field task; got %d."
                % len(field_indexes))
        if remote:
            # The field -> QM handoff is a file (or an in-memory charge
            # list) produced by the field task on this host. Per-term
            # remote execution sends each term to a separate worker, so
            # the QM stage would look for data on another host. A
            # whole-run base-node dispatch is fine.
            raise RuntimeError(
                "QMMM v1 does not support per-term remote execution "
                "(remotetask on the backend terms): the field handoff is "
                "local to the evaluation host. Run the QMMM evaluation on "
                "one host; the base-node dispatch may still be remote.")

        # The queue shape is checked here because the stage order and the
        # one-energy-per-geometry rule drive this evaluation. The assembler
        # checks the same contract again before stitching: this is the
        # execution half, that is the assembly half, and neither is a
        # duplicated parser.
        energy_task = tasks[energy_indexes[0]]
        field_tasks = [tasks[index] for index in field_indexes]
        qm_tasks = [tasks[index] for index in qm_indexes]
        # The whole QM set in global atom order: the classical charge array,
        # the V/E arrays the correction terms are reduced by, and the
        # center slicing all address atoms through this order.
        qm_sets = [[int(index) for index in (task.get('region_indices') or [])]
                   for task in qm_tasks]
        union = sorted({index for group in qm_sets for index in group})
        if len(union) != sum(len(group) for group in qm_sets):
            # Two centers claiming one atom would give that atom two 0-order
            # charges; planning refuses the overlap at input, and the charge
            # array written last would silently win here.
            raise ValueError(
                "QMMM centers overlap: %d QM atoms across %d centers, the "
                "union holds %d. Centers must be disjoint."
                % (sum(len(group) for group in qm_sets), len(qm_sets),
                   len(union)))

        # The environment charges the probes carry. Round 1 of the charge
        # iteration sees no center polarized by another, so they start at
        # zero; a later round replaces them with the previous round's fitted
        # values, which is where the inter-center polarization enters.
        source_charges = [0.0] * len(union)
        charges_by_atom = {}
        field_outcomes = []
        field_blocks = []
        rows_by_center = {}
        charge_outcomes = []
        iteration = None

        charge_tasks = [task for task in tasks
                        if task.get('qmmm_role') == 'qm_charge']
        charge_indexes = [index for index, task in enumerate(tasks)
                          if task.get('qmmm_role') == 'qm_charge']

        if field_tasks:
            field_outcomes, field_blocks, rows_by_center = (
                self._probe_field_pass(
                    field_tasks, field_indexes, active_backends,
                    snapshot_atoms, snapshot_coords, target_states,
                    evaluation_id, snapshot_cell, remote, descriptor,
                    source_charges, qm_tasks))

        if charge_tasks:
            results = self._run_charge_rounds(
                charge_tasks, charge_indexes, active_backends,
                snapshot_atoms, snapshot_coords, target_states,
                evaluation_id, snapshot_cell, remote, descriptor,
                energy_task, union,
                field_tasks, field_indexes, qm_tasks,
                source_charges, rows_by_center, charges_by_atom,
                field_outcomes, field_blocks)
            rows_by_center = results['rows_by_center']
            field_outcomes = results['field_outcomes']
            field_blocks = results['field_blocks']
            charge_outcomes = results['charge_outcomes']
            charges_by_atom = results['charges_by_atom']
            source_charges = results['source_charges']
            iteration = results['iteration']
        elif field_tasks and len(qm_tasks) > 1 and iterate_charges:
            raise ValueError(
                "multi-center QMMM with sc_charge needs a charge round per "
                "center; the plan expanded %d QM tasks and no charge task."
                % len(qm_tasks))

        # --- the property round: every QM center, with gradients -----------
        property_round = self._run_property_round(
            qm_tasks, qm_indexes, active_backends, snapshot_atoms,
            snapshot_coords, target_states, evaluation_id, snapshot_cell,
            remote, descriptor, rows_by_center, energy_task, union,
            charge_tasks, charges_by_atom, source_charges)
        qm_outcomes = property_round['outcomes']
        charges_by_atom = property_round['charges_by_atom']
        source_charges = property_round['source_charges']

        store = getattr(self._state, 'charge_store', None)
        if store is not None:
            store.update(union, source_charges)
            source_charges = [float(store.atom_charges[index])
                              for index in union]
        q_qm = list(source_charges)

        # --- Stage 3: the full force field with q_qm ----------------------
        energy_outcomes = self._run_classical_stage(
            energy_task, energy_indexes, active_backends, snapshot_atoms,
            snapshot_coords, target_states, evaluation_id, snapshot_cell,
            remote, descriptor, union, q_qm, embedding, field_blocks)

        by_task_id = {}
        for outcome in (list(field_outcomes) + list(charge_outcomes)
                        + list(qm_outcomes) + list(energy_outcomes)):
            by_task_id[outcome.get('task_id')] = outcome
        # Every duty of the queue reaches the assembler, the charge rounds
        # included: their outcome is the last round's, it contributes no
        # energy and no gradient, and the assembly's own contribution
        # pattern ([0,0,..,1,1]) is what keeps it accounted for.
        ordered = [by_task_id[task.get('task_id')] for task in tasks
                   if task.get('task_id') in by_task_id]
        # The charge iteration belongs to the evaluation, not to any one
        # task: which round stopped it and by how much is the number a
        # reader needs beside the energy to know what was converged.
        self._qmmm_iteration = iteration
        return self._route_outcomes(tasks, ordered, is_multi)

    def _convert_outcome(self, outcome, task):
        """Apply the skill's declared native→framework unit conversion."""
        raw = outcome.get('raw') or {}
        outcome['raw'] = apply_result_units(
            raw, task.get('skill_name'))
        return outcome['raw']

    def _qmmm_external_potential(self, field_backend, qm_task,
                                 field_block, field_skill=None):
        """Read the Molcas ESPF rows from the field task's result block.

        The exact spelling, ``id V Ex Ey Ez`` in atomic units, is shared
        with the ``.esfp`` file the field task also writes: that file is
        the older channel and carries the converted columns, so this one
        converts the native block columns with the same declared factors.
        Reading them raw is a silent factor of 627 on V and 1186 on E (the
        ``real`` unit set computes in kcal/mol), which embeds the QM
        calculation in a field that is not the one the classical side
        reduces; the first run that did so exploded its QM rows to
        hundreds of Hartree/Bohr.

        The file remains the fallback when the block carries no columns,
        and the partition is checked against the QM task by **atom id**
        before any number reaches the QM backend: a center handed another
        center's field would converge on the wrong environment silently.

        Parameters
        ----------
        field_backend : object
            Initialized LAMMPS backend of the field task.
        qm_task : dict
            QM task descriptor (region order).
        field_block : dict
            The field result's ``qmmm_field`` block.
        field_skill : str, optional
            Skill package name of the field task, used to look up its
            declared native-unit factors. Defaults to the QM task's
            skill, which is what every single-skill build declares.

        Returns
        -------
        list of list of float
            (nQ, 4) rows in Hartree / Hartree-per-Bohr (atomic units).

        Raises
        ------
        ValueError
            On a missing `.esfp`, a partition mismatch, or a non-finite
            row.
        """
        qm_indices = [int(index) for index in
                      (qm_task.get('region_indices') or [])]
        if not qm_indices:
            raise ValueError(
                "QMMM QM task '%s' has an empty region."
                % qm_task.get('task_id'))
        block_ids = {int(index) for index in
                     (field_block.get('qm_atom_ids') or [])}
        needed = {index + 1 for index in qm_indices}
        if block_ids != needed:
            # By id, not by position: with several centers a probe's rows
            # name the atoms they were sampled at, and an equal-length list
            # in another order would be a different field.
            raise ValueError(
                "QMMM field partition mismatch: the probe labels QM ids %s "
                "but task '%s' needs %s."
                % (sorted(block_ids), qm_task.get('task_id'),
                   sorted(needed)))

        # The field task hands V/E back in its own result block; that
        # block is the channel. The `.esfp` file beside it is the older
        # spelling and is read only when the block carries no columns,
        # once per process, because a missing block column is a producer
        # fact worth seeing rather than a silent switch of source.
        from .multiscale.composite import native_field_rows
        from ..skills.query import get_declaration

        rows = native_field_rows(
            field_block,
            get_declaration(qm_task.get('skill_name') or '', 'qmmm')
            .get('results') or {})
        if rows is not None:
            rows = _field_rows_in_atomic_units(
                rows, field_block,
                field_skill or qm_task.get('skill_name') or '')
            by_id = dict(zip(sorted(block_ids), rows))
            return [by_id[index + 1] for index in qm_indices]
        _warn_esfp_fallback(qm_task.get('task_id'))

        calcdir = getattr(field_backend, 'calcdir', None)
        paths = (sorted(glob.glob(os.path.join(calcdir, '*.esfp')))
                 if calcdir else [])
        if not paths:
            raise RuntimeError(
                "QMMM field task '%s' left no .esfp in %r; the field task "
                "must keep its calcdir (keep_tmp=1) for the QM handoff."
                % (qm_task.get('task_id'), calcdir))
        if len(paths) > 1:
            # Several handoffs in one directory cannot be told apart by
            # name, and picking one by sort order would silently embed this
            # center in another region's field.
            raise RuntimeError(
                "QMMM field task '%s' left %d .esfp files in %r: %s; one "
                "probe writes one handoff."
                % (qm_task.get('task_id'), len(paths), calcdir,
                   [os.path.basename(path) for path in paths]))
        ids, rows = _read_esfp_rows(paths[0])
        if ids is None:
            # A handoff without an id column binds its rows by position,
            # which is only meaningful for the one-center deck that wrote
            # every QM atom in ascending order.
            if len(rows) != len(qm_indices):
                raise ValueError(
                    "QMMM .esfp has %d rows for %d QM atoms."
                    % (len(rows), len(qm_indices)))
            ordered = rows
        else:
            by_id = dict(zip(ids, rows))
            missing = [index + 1 for index in qm_indices
                       if index + 1 not in by_id]
            extra = sorted(set(ids) - set(index + 1 for index in qm_indices))
            if missing or extra:
                raise ValueError(
                    "QMMM .esfp describes atoms %s but task '%s' needs %s "
                    "(missing %s, unexpected %s)."
                    % (sorted(ids), qm_task.get('task_id'),
                       sorted(index + 1 for index in qm_indices),
                       missing, extra))
            ordered = [by_id[index + 1] for index in qm_indices]
        values = np.asarray(ordered, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("QMMM .esfp contains a non-finite value.")
        return ordered

    def _charge_embedding_state(self, task, target_states):
        """The state whose charges this evaluation embeds with.

        Both multiscale paths that turn a per-state `computed_charges`
        payload into embedding charges - the QM/MM stage and the ONIOM
        charge scatter - ask this one question. The backend returns one row
        per state, so picking the row is a framework decision: a dynamics or
        single-state step uses the state it is running on, a crossing search
        names its pair and embeds with the first one, and `target_states`
        carries both. Falls back to the state the system says is current.
        """
        states = _task_target_states(target_states, task)
        for value in (states or []):
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        system = getattr(getattr(self._state, 'pmd', None), 'system', None)
        return int(getattr(system, 'current_state', 0) or 0)

    def _qmmm_charges(self, qm_task, qm_raw, energy_task=None, state=0,
                      require_readback=False):
        """Resolve the QM 0-order charges for the energy task.

        A QM backend that returns ``computed_charges`` wins; otherwise
        the stored whole-system array's highlevel slice
        (``point_charges[highlevel]``) is the fallback, and then the
        framework system ``molecule.q_qm`` seed. The length must match
        the QM region.

        Parameters
        ----------
        qm_task : dict
            QM task descriptor.
        qm_raw : dict
            QM backend result.
        energy_task : dict or None
            Energy task descriptor carrying the static q_qm fallback.
        state : int
            0-based index of the state whose row of a per-state
            ``computed_charges`` payload this evaluation embeds with.
        require_readback : bool
            Set by a charge round of a multi-center run: the value it
            returns becomes the field the other centers are solved in, so a
            stored or pinned charge set standing in for it would iterate on
            a number nothing computed. With it, only a backend readback is
            accepted and its absence is an error.

        Returns
        -------
        list of float
            (nQ,) charges in QM region order.

        Raises
        ------
        ValueError
            When no source provides a valid charge vector.
        """
        qm_indices = [int(index) for index in
                      (qm_task.get('region_indices') or [])]
        payload = qm_raw.get('computed_charges')
        values = []
        from_readback = False
        if payload is not None:
            array = np.asarray(payload, dtype=np.float64)
            if array.ndim == 2 and array.shape[0] > 0:
                if 0 <= state < array.shape[0]:
                    array = array[state]
                else:
                    # The engine printed fewer states than this evaluation
                    # asks about. Row 0 is not a stand-in for state `state`:
                    # embedding one surface's charges under another surface's
                    # name is the mistake this channel existed to fix, so the
                    # readback is refused by name and the fallback chain runs.
                    self.logger.warning(
                        "QM charges: state %d requested, backend returned "
                        "%d state(s); using the stored charges instead",
                        state, array.shape[0])
                    array = array[0:0]
            if array.ndim == 1 and array.size:
                values = [float(value) for value in array]
                from_readback = bool(values)
        if not values:
            store = getattr(self._state, 'charge_store', None)
            if store is not None and getattr(store, 'has_values', False):
                values = [float(store.atom_charges[index])
                          for index in qm_indices]
        if not values and energy_task is not None:
            config = energy_task.get('config') or {}
            skill_config = (config.get('skill_config')
                            if isinstance(config, dict) else {}) or {}
            values = [float(value) for value in
                      (skill_config.get('q_qm') or [])]
        if not values:
            system = getattr(getattr(self._state, 'pmd', None),
                             'system', None)
            values = [float(value) for value in
                      (getattr(system, 'q_qm', []) or [])]
        if require_readback and not from_readback:
            raise ValueError(
                "QMMM charge round of task '%s' got no computed_charges "
                "readback for state %d. A charge round feeds the other "
                "centers' field, so the stored or pinned charge set this "
                "call would fall back to cannot stand in for it: the "
                "iteration would converge on a number nothing computed."
                % (qm_task.get('task_id'), state))
        if not values:
            raise ValueError(
                "QMMM energy task needs q_qm: provide molecule.q_qm (the QM "
                "0-order charges in highlevel order) or have the QM backend "
                "return computed_charges.")
        if len(values) != len(qm_indices):
            raise ValueError(
                "QMMM q_qm has %d charges for %d QM atoms."
                % (len(values), len(qm_indices)))
        if not all(np.isfinite(value) for value in values):
            raise ValueError("QMMM q_qm contains a non-finite value.")
        if from_readback:
            values = self._balance_qmmm_charges(values, qm_indices, qm_task)
        return values

    def _balance_qmmm_charges(self, values, qm_indices, task=None):
        """Spread the ESPF fit's charge residual over the QM atoms.

        The fitted 0-order multipoles are a projection of the electrostatic
        potential onto a shell grid, not a constrained sum: on the deployed
        build a neutral water's three charges came back summing to 1.2e-3,
        and no denser grid removes it (4 shells at 1 Å give 1.2e-3, at
        0.25 Å 1.5e-3, 8 shells 5e-4). The classical stage, however, has to
        sit in a neutral cell - `qmmm_energy` refuses otherwise - so the
        deficit is removed by averaging it over the region, which is the
        adjustment that leaves the shape of the fit untouched and makes the
        vector admissible. The amount removed is logged, and the unadjusted
        values stay in the backend result for anyone comparing the two.

        Only the readback path is adjusted. A `molecule.q_qm` the input
        spells out is already whatever number its author chose to hand over,
        and silently rescaling it would rewrite a declared input. Under
        several centers the target is the plan's `CenterSpec.charge` for the
        center that produced this fit; a whole-system `charges` list is not
        consulted, because which entry belongs to which region is the plan's
        statement, not this one's.
        """
        system = getattr(getattr(self._state, 'pmd', None), 'system', None)
        target = getattr(system, 'charges', None)
        center = (task or {}).get('center_index')
        plan = getattr(self._state, 'partition_plan', None)
        if not isinstance(target, int) and center is not None and plan \
                and 0 <= center < plan.n_center:
            target = plan.center(center).charge
        if not isinstance(target, int):
            self.logger.info(
                "QMMM q_qm not balanced: molecule.charges is %r, which does "
                "not state one target charge for this region.",
                type(target).__name__)
            return values
        deficit = sum(values) - float(target)
        if abs(deficit) <= _CHARGE_BALANCE_TOLERANCE or not values:
            return values
        share = deficit / len(values)
        self.logger.info(
            "QMMM ESPF charges carry a %.6f residual against the declared "
            "%.6f; %.6f removed from each of %d atoms (unadjusted: %s).",
            deficit, float(target), share, len(values),
            ', '.join(f'{value:.4f}' for value in values))
        return [value - share for value in values]

    def _qmmm_atomic_block(self, outcome, task):
        """Normalize the native QM/MM block into Hartree / Hartree-Bohr.

        ``E_classic``, ``U_main``, ``U_gas`` and ``qmmm_el`` are scaled by
        the unit set the skill declared, and so are the QM-row auxiliary
        gradients ``qmmm_el_gradient`` and ``u_gas_gradient``: the skill
        already wrote them as dU/dR, so this step is a scale and nothing
        else. The native block is kept beside the atomic one for
        diagnostics.
        """
        raw = outcome.get('raw') or {}
        block = raw.get('qmmm')
        if not isinstance(block, dict) or not block:
            raise ValueError(
                "QMMM energy task '%s' returned no qmmm block."
                % task.get('task_id'))
        # The block's key names and its default unit set are the skill's
        # declaration (manifest `qmmm.results`, #3.4): the framework reads
        # a logical name through it, so renaming a written key is a change
        # in the skill and not a silently dropped field here.
        from ..skills.query import get_declaration

        declared = (get_declaration(task.get('skill_name') or '', 'qmmm')
                    .get('results') or {})
        default_units = str(declared.get('default_units') or '')
        units = str((block.get('units') or {}).get('energy')
                    or default_units)
        skill_name = task.get('skill_name')

        def _key(logical, fallback):
            return str(declared.get(logical) or fallback)

        def _read(key, convert):
            if key not in block:
                raise ValueError(
                    "QMMM energy task '%s' qmmm block lacks %r."
                    % (task.get('task_id'), key))
            return convert(block[key], units, skill_name)

        raw['qmmm_atomic'] = {
            'e_classic': _read(_key('e_classic', 'e_classic_native'),
                               convert_native_energy),
            'u_main': _read(_key('u_main', 'u_main_native'),
                            convert_native_energy),
            'u_gas': _read(_key('u_gas', 'u_gas_native'),
                           convert_native_energy),
            'qmmm_el': _read(_key('qmmm_el', 'qmmm_el_native'),
                             convert_native_energy),
            'qmmm_el_gradient': _read(
                _key('qmmm_el_gradient', 'qmmm_el_gradient_native'),
                convert_native_gradient),
            'u_gas_gradient': _read(
                _key('u_gas_gradient', 'u_gas_gradient_native'),
                convert_native_gradient),
            'native_unit_set': units,
        }
        # Under several centers the in-cell removal is a sum of per-center
        # terms; the total alone cannot say which center's pairs left the
        # classical surface, and that is the number the report is asked for.
        centers_native = block.get(_key('u_gas_centers',
                                        'u_gas_centers_native'))
        if centers_native:
            raw['qmmm_atomic']['u_gas_centers'] = [
                convert_native_energy(value, units, skill_name)
                for value in centers_native]
        outcome['raw'] = raw
        return raw['qmmm_atomic']


    def _freeze_evaluation(self, tasks, target_states):
        """Freeze this step's geometry and derive the evaluation's shape.

        Everything an evaluation reads must belong to one instant: the
        coordinates and the cell are copied and made read-only here, and
        the shape derived from the plan (how many centers, which state
        axis, whether a crossing pair was requested) is computed from that
        same frozen plan. The plan is validated here, once, before any
        backend is called.

        Parameters
        ----------
        tasks : list of dict
            Active role's tasks for this evaluation.
        target_states : dict or list or None
            Per-step requested states, as staged by run().

        Returns
        -------
        dict
            atoms, coords, cell, natom, is_multi, is_multi_center,
            nstate, active_configuration, pair_axis.
        """
        _t = time.perf_counter()
        # --- Freeze geometry snapshot (read-only for thread isolation) ---
        snapshot_atoms = list(self._state.atoms)
        snapshot_coords = np.array(
            self._state.coordinates, dtype=np.float64, copy=True)
        snapshot_coords.setflags(write=False)
        # Freeze the cell snapshot at the same moment (PBC stage B):
        # every term of this evaluation sees the cell that belongs to
        # these coordinates, including the deformation a variable-cell
        # integrator applied in this very step.
        snapshot_cell = self._snapshot_cell()
        self._profile('snapshot', _t)
        _t = time.perf_counter()
        natom = len(snapshot_atoms)
        is_multi = len(tasks) > 1

        # Multi-center is decided by the number of DISTINCT centers,
        # never by center_index being non-null: a single-center run
        # now binds center 0 too (docs 3.1).
        distinct_centers = {
            t.get('center_index') for t in tasks
            if t.get('center_index') is not None}
        is_multi_center = len(distinct_centers) > 1

        # C>1 has no global nstate axis (docs 8): state.nstate would
        # raise CenterPlanError, so the assembly request carries nstate
        # 1 and the active configuration drives per-center selection.
        active_configuration = self._state.get_active_configuration()
        if is_multi_center:
            nstate = 1
        else:
            nstate = max(int(getattr(self._state, 'nstate', 1)), 1)

        # A crossing search names a pair per center. When every center gets
        # exactly two target states, the assembly builds the two composite
        # surfaces state 0 = (first of each pair) and state 1 = (second), so
        # `meci`/`mecp` run their single-PES machinery on the sum of the
        # per-center gaps. One center keeps the historical flat axis.
        pair_axis = None
        if is_multi_center:
            n_center = int(getattr(self._state, 'n_center', 1) or 1)
            if isinstance(target_states, dict):
                picks = [target_states.get(c) for c in range(n_center)]
            elif target_states:
                picks = [list(target_states)] * n_center
            else:
                picks = []
            if picks and all(
                    isinstance(entry, (list, tuple)) and len(entry) == 2
                    for entry in picks):
                pair_axis = [[int(entry[0]), int(entry[1])]
                             for entry in picks]

        # Validate the plan once before execution
        validate_plan(tasks, natom)
        self._profile('plan', _t)

        return {
            'atoms': snapshot_atoms,
            'coords': snapshot_coords,
            'cell': snapshot_cell,
            'natom': natom,
            'is_multi': is_multi,
            'is_multi_center': is_multi_center,
            'nstate': nstate,
            'active_configuration': active_configuration,
            'pair_axis': pair_axis,
        }

    def _execute_evaluation(self, tasks, active_backends, snapshot_atoms,
                            snapshot_coords, target_states, evaluation_id,
                            is_multi, snapshot_cell, charge_store,
                            embedding, read_charge, embedding_regions):
        """Run this evaluation's terms on the branch its plan asks for.

        Three branches share one contract: every term returns a unified
        outcome, is validated against the frozen work items, and is routed
        in the main thread. QM/MM runs its staged plan, a bound charge
        store without a static field runs two phases, and anything else is
        one batch. The executor is the role's (serial by default; remote
        per-term workers or a thread pool when the descriptor says so).

        Returns
        -------
        dict
            Routed outcomes keyed as `_route_outcomes` returns them.
        """
        # --- Execute through the role's executor (serial default) ---
        # A backend-terms descriptor with remotetask=True routes every
        # term to the remote per-term worker (r3e); the outcomes keep
        # the same unified shape and index alignment, so validation,
        # routing and assembly below are identical. The failure policy
        # is shared (apply_failure_policy) with the in-process path.
        descriptor = self._active_descriptor()
        remote = bool(descriptor and descriptor.get('remotetask'))
        qmmm = any(t.get('composite') == 'qmmm' for t in tasks)
        # Two phases only when a bound store must see the mid level before
        # the high level; a static read_charge field embeds in one batch.
        two_phase = (charge_store is not None and embedding
                     and not read_charge)

        if qmmm:
            _t = time.perf_counter()
            routed = self._run_qmmm(
                tasks, active_backends, snapshot_atoms, snapshot_coords,
                target_states, evaluation_id, is_multi, remote,
                descriptor, snapshot_cell=snapshot_cell)
            self._profile('execute_terms', _t)
        elif two_phase:
            _t = time.perf_counter()
            routed = self._run_two_phase(
                tasks, active_backends, snapshot_atoms, snapshot_coords,
                target_states, evaluation_id, charge_store,
                embedding_regions, is_multi, remote, descriptor,
                snapshot_cell=snapshot_cell)
            self._profile('execute_terms', _t)
        else:
            _t = time.perf_counter()
            embed_store = charge_store if embedding else None
            work_items = self._build_work_items(
                tasks, active_backends, snapshot_atoms, snapshot_coords,
                target_states, evaluation_id,
                charge_store=embed_store,
                embedding_regions=embedding_regions,
                snapshot_cell=snapshot_cell)
            outcomes = self._execute_terms(
                work_items, tasks, remote, descriptor)
            self._profile('execute_terms', _t)
            _t = time.perf_counter()
            # --- Validate completeness / no duplicate / no stale ---
            self._validate_outcomes(work_items, outcomes, evaluation_id)
            # --- Route in the main thread (no live-state writes) ---
            routed = self._route_outcomes(tasks, outcomes, is_multi)
            self._profile('route', _t)
            # A bound store that is not a static field still records the
            # computed charges for output/logging (no embedding here).
            if charge_store is not None and not read_charge:
                update_from_outcomes(
                    charge_store, tasks, routed,
                    state=lambda task: self._charge_embedding_state(
                        task, target_states))

        return {'routed': routed, 'qmmm': qmmm}

    def run_in_process(self):
        """In-process execution body of the active role task queue.

        Unified materialization/execution/routing (docs R3):
        1. Freeze the geometry snapshot at the start of this step and
           make it read-only (thread isolation).
        2. Stage one work item per task_slot from the frozen snapshot,
           clearing each task's stale results.
        3. Execute every term through the role's TaskExecutor (serial
           by default; thread only when the descriptor requests it and
           the exact backend contract passed the thread-safety audit).
           The failure policy raises with the collected diagnostics of
           every completed term, preserving the external fail-fast
           contract.
        4. Validate the outcome set (complete / no duplicate / no
           stale) by task_id + evaluation_id.
        5. Route in the main thread (fin_coord, config store) and
           assemble with per-task gradient scatter-add in the fixed
           task order, then commit once to pmd.results.

        A single task goes through the same execution and validation
        flow. Multi-task composite rejects fin_coord (the framework
        optimizer/MD updates the global geometry, not individual
        sub-calculations).

        Returns
        -------
        Result
            Computed result (raw or composite-aggregated).
        """
        tasks = self._active_tasks()
        # One evaluation owns one charge iteration report: a scheme that
        # does not run it must not reprint the previous step's residuals.
        self._qmmm_iteration = None
        # The ONIOM2 outer reference is per-evaluation too; a step that
        # does not capture one must not report the previous step's field.
        self._oniom_outer_charges = None
        # Consume the per-step target states staged by run(); reset
        # them so a direct run_in_process() call starts clean.
        target_states = self._step_target_states
        self._step_target_states = None
        if not tasks:
            return self._run_single(target_states)

        # Lazily initialize when run is called directly
        active_backends = self.backends.get(self._active_role, [])
        if not active_backends and self.dummy_backend is None:
            self.initialize()
            active_backends = self.backends.get(self._active_role, [])

        frozen = self._freeze_evaluation(tasks, target_states)
        snapshot_atoms = frozen['atoms']
        snapshot_coords = frozen['coords']
        snapshot_cell = frozen['cell']
        is_multi = frozen['is_multi']

        _t = time.perf_counter()
        # --- Stage the unified work items of this evaluation ---
        evaluation_id = self._next_evaluation_id()

        # Charge management context (docs charge mgmt): a bound store
        # turns on electrostatic embedding. Two-phase execution applies
        # only when the midlevel must be computed before the high level
        # (embedding and not a static read_charge field); read_charge
        # embeds the high level with the static file charges in a single
        # batch. A run without a store delivers point_charges=None; the
        # whole-system array is never handed to a skill.
        charge_store = getattr(self._state, 'charge_store', None)
        system = getattr(self._state.pmd, 'system', None)
        embedding = bool(getattr(system, 'embedding', False))
        read_charge = bool(getattr(system, 'read_charge', False))
        # Which side owns the 0-order field gradient term F_el = q_qm * E
        # when a QM/MM result is stitched (composite.assemble_qmmm). True is
        # the ESPF analytic-gradient contract; a fixed-potential QM provider
        # declares False so the classical side keeps the term.
        qm_contains_field_gradient = bool(getattr(
            system, 'qm_contains_field_gradient', True))
        two_phase = (charge_store is not None and embedding
                     and not read_charge)
        embedding_regions = self._embedding_regions_by_center()
        self._profile('work_items', _t)

        # --- Execute through the role's executor (serial default) ---
        execution = self._execute_evaluation(
            tasks, active_backends, snapshot_atoms, snapshot_coords,
            target_states, evaluation_id, is_multi, snapshot_cell,
            charge_store, embedding, read_charge, embedding_regions)
        routed = execution['routed']
        qmmm = execution['qmmm']

        # --- Assemble results with strict aggregation ---
        data, system_task = self._assemble_evaluation(
            tasks, routed, qmmm, evaluation_id, frozen,
            qm_contains_field_gradient)

        # --- Write back once, then commit the evaluation ---
        return self._commit_evaluation(data, snapshot_coords, system_task)

    def _assemble_evaluation(self, tasks, routed, qmmm, evaluation_id,
                             frozen, qm_contains_field_gradient):
        """Build the aggregation request and aggregate this evaluation.

        The request names the providers the composite assembler binds
        first: the electronic primary is the high-level provider H(c0),
        not ``tasks[0]`` (the real-first order puts the L(real)
        baseline first), and the structural-track provider supplies
        stress/virial independently of it.

        Returns
        -------
        tuple
            ``(data, system_task)``; ``system_task`` is None when the
            queue has no whole-system term, and the write-back needs it
            to convert the stress unit.
        """
        _t = time.perf_counter()
        need_gradient = any(
            'gradient' in (t.get('capabilities') or [])
            for t in tasks)
        # The composite primary is the electronic provider H(c0), not
        # tasks[0]: the real-first order (docs 6.1/8) puts the L(real)
        # baseline first, but NAC/SOC and active-state properties come
        # from the high-level provider. resolve_primary_task falls back
        # to tasks[0] for legacy queues without the center fields.
        primary_task = resolve_primary_task(tasks)
        # Structural-track provider (docs 4A): the whole-system term
        # supplying stress/virial, independent of the electronic primary.
        system_task = resolve_system_task(tasks)

        request = {
            'scheme': 'qmmm' if qmmm else 'oniom',
            'natom': frozen['natom'],
            'nstate': frozen['nstate'],
            'need_gradient': need_gradient,
            'qm_contains_field_gradient': qm_contains_field_gradient,
            'multi_center': frozen['is_multi_center'],
            'primary_task_id': primary_task.get('task_id', ''),
            'system_task_id': (system_task.get('task_id', '')
                               if system_task is not None else ''),
            'active_configuration': frozen['active_configuration'],
            'pair_axis': frozen['pair_axis'],
        }

        if frozen['is_multi']:
            # Use new strict aggregation with gradient scatter-add
            data = assemble_results(tasks, routed, request)
            if frozen['is_multi_center']:
                # C>1: collect each center's electronic properties
                # as-is (docs 8), stage the active surface and commit
                # atomically so a run persists electronic[c]. R6 moves
                # this commit into the MD loop (the sole writeback
                # point) and leaves the calculator staging only.
                per_center = collect_center_electronic(
                    tasks, routed, request)
                self._state.stage_evaluation({
                    'electronic': per_center,
                    'active_surface': {
                        'active_configuration':
                            tuple(data.get('active_configuration', ())),
                        'active_energy': data.get('active_energy'),
                        'active_gradient': data.get('active_gradient'),
                        'evaluation_id': evaluation_id,
                    },
                })
                self._state.commit_step()
        else:
            # Single-task: the state's own queue is the aggregation
            # input (the old path first copied the primary task's
            # results back into its own config through an identity
            # projector, which changed nothing).
            data = aggregate_results(
                self._state, category=self._active_role)
        self._profile('assemble', _t)

        # A standalone background embedding is input, not a computed
        # property: carry it into the result so the log can print what the
        # calculation was actually embedded in. One task carries one kind -
        # the file declares it - and a result that somehow held both is
        # reported as the conflict it is rather than printed as two blocks.
        _external = next((task.get('external_charges') for task in tasks
                          if task.get('external_charges')), None)
        _external_field = next((task.get('external_potential')
                                for task in tasks
                                if task.get('external_potential')), None)
        if _external and _external_field:
            raise ValueError(
                "a background embedding is either external charges or an "
                "external field; both were assembled for this step.")
        if _external:
            data['external_charges'] = [list(row) for row in _external]
        if _external_field:
            data['external_potential'] = [list(row) for row in _external_field]

        # The charge iteration belongs to the evaluation rather than to a
        # task, so no assembler can carry it: the round count is a
        # trajectory variable, and a reader comparing two frames cannot
        # tell sc_charge=false from a converged-on-first-round true run
        # without it.
        if self._qmmm_iteration is not None:
            data['qmmm_charge_iteration'] = self._qmmm_iteration
        return data, system_task

    def _commit_evaluation(self, data, snapshot_coords, system_task):
        """Write one assembled evaluation back to the state and commit it.

        The order is the contract: charges and unit conversions settle
        the numbers, the constraint/SOC phase adds every declared
        potential once at this geometry, and only then is the result
        committed to ``pmd.results``.
        """
        _t = time.perf_counter()
        # Charge management: sync the managed-region point charges into
        # pmd.system before the result write-back so the snapshot and
        # history frame carry them for the verbose log and the MD charge
        # files. A run without a store leaves point_charges untouched.
        self._sync_point_charges(snapshot_coords)
        # The ONIOM2 environment is not the store's highlevel row: carry
        # the captured full-system low-level rows so the log can show what
        # each centre was embedded in (#462 gap).
        if self._oniom_outer_charges is not None:
            data['oniom_outer_charges'] = self._oniom_outer_charges

        # Framework-central stress unit conversion: the whole-system
        # stress arrives in the provider skill's native unit (declared
        # by its manifest units.stress) and is converted once here into
        # Hartree/Bohr^3, the unit active_stress() / the NPT barostats
        # consume. A run without stress leaves data unchanged.
        convert_stress_to_atomic(data, system_task, logger=self.logger)
        self._profile('convert', _t)
        _t = time.perf_counter()

        # Constraint finalizer: every additive potential the run declares is
        # applied to the result of THIS geometry, once, and the flat
        # `constraint.<kind>.<id>` breakdown goes in beside it. A run without a
        # constraint block short-circuits inside and changes no number, which is
        # what makes the section default closed rather than merely unused. The
        # SOC bias rides in the same phase: it is the other number a step must
        # record as evaluated, not as integrated.
        self._finalize_constraints(data)
        self._apply_soc_bias(data)
        self._profile('constraint', _t)
        _t = time.perf_counter()

        result = adapt_skill_result(data)
        self._profile('set_result', _t)
        self._state.set_result(result.to_dict())
        return result

    def _profile(self, name, started):
        """Record one phase of this evaluation into the step profile.

        ``Calculator.run()`` is where a step's force evaluation lives, and
        it was the one block of the MD loop with no breakdown of its own:
        knowing that a step spends milliseconds here without knowing
        whether it is the plan, the work items, the backend call or the
        result write-back buys nothing. The rows land in the run's phase
        table as a nested breakdown, under the step row that contains
        them, so the two are never added together.

        Parameters
        ----------
        name : str
            Phase label.
        started : float
            ``time.perf_counter()`` value taken when the phase began.
        """
        profiler = getattr(self, '_profiler', None)
        if profiler is None:
            profiler = self._profiler = PhaseProfiler()
        profiler.record(name, time.perf_counter() - started)

    def profile_lines(self, steps):
        """Render this calculator's nested breakdown.

        Parameters
        ----------
        steps : int
            Steps measured by the caller, used as the divisor so the
            nested rows are in the same ms-per-step units as the outer
            table.

        Returns
        -------
        list of str
            The rendered rows, empty when nothing was ever timed.
        """
        profiler = getattr(self, '_profiler', None)
        return profiler.render(steps) if profiler is not None else []

    def _sync_point_charges(self, coords):
        """Write the whole-system point charges into pmd.system.

        Runs only when charge management is active (a ChargeStore is
        bound). The derived [q, x, y, z] rows (row i = atom i) at the
        current geometry replace pmd.system.point_charges so the
        snapshot, the verbose log and the MD charge files all read one
        source. A run without a store leaves point_charges exactly as
        the input set it, so its output is byte-for-byte unchanged.

        Parameters
        ----------
        coords : ndarray
            Frozen global coordinates (natom, 3) of this evaluation.
        """
        charge_store = getattr(self._state, 'charge_store', None)
        if charge_store is None:
            return
        system = getattr(self._state.pmd, 'system', None)
        if system is None:
            return
        system.point_charges = charge_store.point_charges(
            np.asarray(coords, dtype=float)).tolist()

    def _apply_skill_geometry(self, task, result_dict):
        """Consume the fin_coord channel of a skill result.

        Skills performing internal iteration (geometry optimization,
        molecular dynamics, ...) report their final structure through
        the 'fin_coord' key of the returned result dict (natom, 3)
        Angstrom. The framework applies it to state.pmd.system so the
        authoritative geometry stays in sync with the skill-internal
        iteration; subsequent workflow logging and output files then
        reflect the correct structure.

        The key is a transport channel, not a physical property: it is
        popped from the result dict (and from the already stored task
        config results) after consumption so it never leaks into
        state.pmd.results.

        Parameters
        ----------
        task : dict
            Task descriptor whose backend produced the result.
        result_dict : dict or None
            Result dictionary returned by backend.calculate().
        """
        if not result_dict or self._state is None:
            return
        fin_coord = result_dict.get('fin_coord')

        # Pop the transport channel key from the return dict and from
        # the task config results written by the backend itself.
        result_dict.pop('fin_coord', None)
        stored = task_config_results(task)
        if isinstance(stored, dict):
            stored.pop('fin_coord', None)

        # Missing key or empty value: no internal iteration happened.
        if not fin_coord:
            return

        # Fail-fast shape validation: architecturally a mismatch is
        # impossible (framework-driven multiscale optimization keeps
        # the skill at single-point/gradient level; skill-internal
        # multiscale optimization always iterates the full system),
        # so a mismatch means a skill contract violation.
        natom = len(self._state.atoms)
        if len(fin_coord) != natom:
            raise ValueError(
                "fin_coord length %d does not match system natom %d; "
                "skill contract violation." % (len(fin_coord), natom))

        current = np.asarray(self._state.coordinates, dtype=np.float64)
        updated = np.asarray(fin_coord, dtype=np.float64)
        if current.shape == updated.shape and np.allclose(
                current, updated, atol=1e-8):
            return

        self._state.update_system({'coordinates': updated.tolist()})
        self.logger.info(
            "Skill-reported final structure (fin_coord) applied to "
            "system coordinates.")

    def _apply_skill_velocities(self, task, result_dict):
        """Consume the fin_velocity channel of a skill result.

        The mirror of ``_apply_skill_geometry`` for the other half of a
        propagated state: a skill whose engine carries the dynamics
        (``ff_nve``/``ff_nvt``/``ff_npt``) reports the final velocities
        through the 'fin_velocity' key, (natom, 3) Bohr/au. Without this
        the channel has no consumer at all, so a standalone dynamics step
        would log zero velocities and zero kinetic energy while the engine
        ran at 86 K, and the next caller of ``state.kinetic_energy()``
        would inherit a state that denies the run ever happened. The
        kinetic energy is recomputed here for the same reason: the
        framework MD loop derives it step by step and never runs on this
        path.

        The key is a transport channel, not a physical property: it is
        popped from the result dict (and from the already stored task
        config results) after consumption so it never leaks into
        state.pmd.results.

        Parameters
        ----------
        task : dict
            Task descriptor whose backend produced the result.
        result_dict : dict or None
            Result dictionary returned by backend.calculate().
        """
        if not result_dict or self._state is None:
            return
        fin_velocity = result_dict.get('fin_velocity')

        # Pop the transport channel key from the return dict and from
        # the task config results written by the backend itself.
        result_dict.pop('fin_velocity', None)
        stored = task_config_results(task)
        if isinstance(stored, dict):
            stored.pop('fin_velocity', None)

        # Missing key or empty value: no engine-propagated dynamics.
        if not fin_velocity:
            return

        natom = len(self._state.atoms)
        updated = np.asarray(fin_velocity, dtype=np.float64)
        if updated.ndim != 2 or updated.shape[0] != natom \
                or updated.shape[1] != 3:
            raise ValueError(
                "fin_velocity shape %s does not match the expected "
                "(%d, 3) system velocities; skill contract violation."
                % (updated.shape, natom))

        current = np.asarray(self._state.velocities, dtype=np.float64)
        if current.shape == updated.shape and np.allclose(
                current, updated, atol=1e-8):
            return

        self._state.update_system({'velocities': updated.tolist()})
        # Velocities mean nothing to the log until the energy that
        # follows from them is on record as well.
        self._state.kinetic_energy()
        self.logger.info(
            "Skill-reported final velocities (fin_velocity) applied to "
            "system velocities; kinetic energy recomputed.")

    def _apply_skill_cell(self, task, result_dict):
        """Consume the fin_cell channel of a skill result.

        The third of the three engine-owned quantities, and the one a
        stale copy of hurts most: a variable-cell run whose coordinates
        came back but whose cell did not leaves the state holding a
        structure that no longer fits the box it declares, and every
        later single point or QM evaluation is computed on that mismatch
        (the framework's own barostat writes all three fields together,
        which is why this path does the same rather than touching
        ``cell`` alone).

        A skill returns this key only when its engine moved the cell, so
        an empty or missing value is the normal case, not an error. A
        value returned for a system the framework declared non-periodic
        is a contradiction it must not paper over: the engine did run on
        a periodic box, so the input was wrong, and silently accepting
        the cell would leave ``pbc`` claiming otherwise (see the
        ``pbc`` authority in the LAMMPS skill's builder).

        The key is a transport channel, not a physical property: it is
        popped from the result dict (and from the already stored task
        config results) after consumption so it never leaks into
        state.pmd.results.

        Two shapes are accepted because that is what the engines hold: the
        six cell parameters, or the nine flattened lattice components, which
        is the pair ``schema.normalize`` already accepts for ``cell`` itself.
        Either way the state ends up holding both, so a builder that prefers
        ``lattice_vector`` cannot read a box the report did not update.

        Parameters
        ----------
        task : dict
            Task descriptor whose backend produced the result.
        result_dict : dict or None
            Result dictionary returned by backend.calculate().
        """
        if not result_dict or self._state is None:
            return
        fin_cell = result_dict.get('fin_cell')

        # Pop the transport channel key from the return dict and from
        # the task config results written by the backend itself.
        result_dict.pop('fin_cell', None)
        stored = task_config_results(task)
        if isinstance(stored, dict):
            stored.pop('fin_cell', None)

        # Missing key or empty value: the engine kept the cell it was given.
        if not fin_cell:
            return

        numbers = [float(value) for value in fin_cell]
        if len(numbers) == 9:
            lattice = np.asarray(numbers, dtype=np.float64).reshape(3, 3)
            cell = [float(value) for value in matrix_to_cell6(lattice)]
        elif len(numbers) == 6:
            cell = numbers
            lattice = cell6_to_matrix(cell)
        else:
            raise ValueError(
                "fin_cell %r is neither the six cell parameters nor the "
                "nine lattice components; skill contract violation."
                % (fin_cell,))

        lengths, angles = cell[:3], cell[3:]
        if any(value <= 0.0 for value in lengths):
            raise ValueError(
                "fin_cell has a non-positive cell length %r; skill "
                "contract violation." % (lengths,))
        if any(not 0.0 < value < 180.0 for value in angles):
            raise ValueError(
                "fin_cell has a cell angle outside (0, 180) degrees %r; "
                "skill contract violation." % (angles,))
        if abs(float(np.linalg.det(lattice))) < LATTICE_EPS:
            raise ValueError(
                "fin_cell %r describes a degenerate cell (zero volume); "
                "skill contract violation." % (cell,))

        system = self._state.pmd.system
        pbc = [int(flag) for flag in (getattr(system, 'pbc', None) or [])]
        if not any(pbc):
            raise ValueError(
                "A skill returned fin_cell %r for a system whose pbc is "
                "%r: the engine ran on a periodic cell the framework never "
                "declared. Declare pbc and cell in the input (the run's "
                "current cell is %r) instead of letting the engine keep a "
                "cell of its own."
                % (cell, getattr(system, 'pbc', None),
                   getattr(system, 'cell', None)))

        current = [float(value) for value in
                   (getattr(system, 'cell', None) or [])]
        if len(current) == 6 and np.allclose(
                np.asarray(current), np.asarray(cell),
                rtol=1e-8, atol=1e-8):
            return

        # Cell, lattice and volume travel together: the builders prefer
        # lattice_vector over cell, so a half-updated state would keep
        # propagating the old box.
        self._state.update_system({
            'cell': cell,
            'lattice_vector': lattice.tolist(),
            'volume': float(cell_volume(lattice)),
        })
        self.logger.info(
            "Skill-reported final cell (fin_cell) applied to system; "
            "lattice_vector and volume re-derived from it.")

    def sync(self):
        """Synchronize the runtime system state to every task config.

        Pushes the runtime system information of state.pmd.system
        (atoms, coordinates, velocities, charges, multiplicity and
        electronic state fields) into each task configuration so
        iterative workflows keep the task configs aligned with the
        framework state. Never modifies the calculation, method, or
        control sections.

        Per-atom fields (atoms, mass, coordinates, velocities) are
        sliced by the task region_indices so sub-region tasks receive
        only their local data.  Scalar fields (charges, multiplicity,
        nstate, current_state, ...) are synced as-is.  A task's
        region_charge/region_multiplicity (the planning-injected
        per-center region electronic structure) is never overwritten by
        the global sync.

        Calculator and Workflow intentionally share the same
        framework PyRAIConfig object (state.pmd) during runtime;
        sync() is the explicit state transition that propagates the
        modified system state to the task queue.

        Timing note (dev/methods/PBC_DEVELOPMENT.md section 5.3): the
        main MD loop calls sync() AFTER the step's force evaluation, so
        the cell fields synced here (SYNC_SYSTEM_FIELDS carries
        cell/lattice_vector/pbc/volume) refresh the task configs for
        the NEXT step / checkpoint -- they are NOT the channel that
        gives the current evaluation its deformed cell. That per-step
        authoritative channel is the frozen cell snapshot captured in
        run_in_process() (_snapshot_cell) and delivered to
        backend.calculate(cell=...); the two are complementary, not
        redundant.
        """
        if self._state is None or self._state.pmd is None:
            return
        source = self._state.pmd.system.model_dump()

        # A multi-center run carries per-center electronic scalars in
        # each task config (planning-derived); the global pmd.system
        # has nstate=1 and no per-center state axis (docs 8), so
        # syncing those scalars would corrupt them. C>1 syncs the
        # geometry/charge fields only; the per-step active state
        # travels through target_states (the calculate() parameter
        # channel). C=1 syncs every field (bit-for-bit legacy).
        fields = SYNC_SYSTEM_FIELDS
        if self._state.n_center > 1:
            fields = tuple(field for field in SYNC_SYSTEM_FIELDS
                           if field not in _ELECTRONIC_STATE_FIELDS)

        # Per-atom fields that must be sliced by region_indices
        _PER_ATOM_FIELDS = frozenset({
            'atoms', 'mass', 'coordinates', 'velocities',
        })
        # A region task is a molecular subsystem: the global box fields
        # describe the whole system, never this task's 6-atom slice. Its
        # static config already carries the region natom/cell/pbc.
        _BOX_FIELDS = frozenset({
            'natom', 'cell', 'lattice_vector', 'pbc', 'volume',
        })

        for _, task in self._state.iter_tasks():
            config = task.get('config')
            if config is None:
                continue

            indices = list(task.get('region_indices', []))
            region = task.get('region', 'all')
            use_slice = (region != 'all' and bool(indices))

            if isinstance(config, dict):
                skill = config.get('skill_config')
                if isinstance(skill, dict):
                    for field in fields:
                        if field not in skill:
                            continue
                        if use_slice and field in _BOX_FIELDS:
                            continue
                        # A task's region charge/multiplicity (planning
                        # -injected per center) is never overwritten by
                        # the global sync: for C>1 the global
                        # charges/multiplicity are per-center lists and
                        # would corrupt the region scalar.
                        if field == 'charges' and \
                                task.get('region_charge') is not None:
                            continue
                        if field == 'multiplicity' and \
                                task.get('region_multiplicity') is not None:
                            continue
                        if use_slice and field in _PER_ATOM_FIELDS:
                            skill[field] = slice_field(
                                source[field], indices)
                        else:
                            skill[field] = source[field]
                continue

            system = getattr(config, 'system', None)
            if system is None:
                continue
            declared = type(system).model_fields
            for field in fields:
                if field not in declared:
                    continue
                if use_slice and field in _BOX_FIELDS:
                    continue
                # region charge/multiplicity is never overwritten
                if field == 'charges' and \
                        task.get('region_charge') is not None:
                    continue
                if field == 'multiplicity' and \
                        task.get('region_multiplicity') is not None:
                    continue
                if use_slice and field in _PER_ATOM_FIELDS:
                    setattr(system, field,
                            slice_field(source[field], indices))
                else:
                    setattr(system, field, source[field])

    def __repr__(self):
        if self.dummy_backend is not None:
            backend_name = type(self.dummy_backend).__name__
        else:
            active_backends = self.backends.get(self._active_role, [])
            if active_backends and active_backends[0] is not None:
                backend_name = type(active_backends[0]).__name__
            else:
                backend_name = 'None'
        return "Calculator(backend=%s, role=%s)" % (
            backend_name, self._active_role)


def resolve_calculator(state=None, runtime_idx=None, dummy_backend=None,
                       multitask=None, remotetask=None,
                       local_executor=None, **kwargs):
    """Create a Calculator.

    Parameters
    ----------
    state : PyRAIState or None
        Runtime state (primary API).
    runtime_idx : str or None
        Runtime execution suffix provided by the Workflow layer.
    dummy_backend : object or None
        Dummy backend instance for testing; replaces task backends.
    multitask : bool or None
        Execution flag override (graph keywords).
    remotetask : bool or None
        Must stay False (universal rule 3).
    local_executor : callable or None
        Injected in-process execution entry (optional).
    **kwargs
        Additional keyword arguments passed to Calculator.

    Returns
    -------
    Calculator
        A configured Calculator instance.
    """
    return Calculator(state=state, runtime_idx=runtime_idx,
                      dummy_backend=dummy_backend,
                      multitask=multitask, remotetask=remotetask,
                      local_executor=local_executor, **kwargs)


# Stress unit -> Hartree/Bohr^3 conversion factors, keyed by the
# manifest 'units.stress' declaration of each skill. The whole-system
# stress is emitted by the backend in its native program unit; the
# framework converts it once, centrally, into atomic units.
_STRESS_UNIT_FACTORS = {
    'hartree/bohr^3': 1.0,
    'hartree/bohr3': 1.0,
    'au/bohr^3': 1.0,
    'gpa': GPA_TO_HARTREE_PER_BOHR3,
    'bar': BAR_TO_HARTREE_PER_BOHR3,
    'atm': ATM_TO_HARTREE_PER_BOHR3,
}

# One resolved factor per skill, kept after the first manifest read.
# A skill declares its units in package data, which cannot change while
# a run is computing, and the measured MD step table priced one manifest
# read per step at 2.42 ms - 28% of an 8.4 ms step, and more than the
# engine itself. The unknown-unit warning stays OUTSIDE this cache on
# purpose: a misdeclared manifest is a defect, and it must keep reporting
# itself exactly as often as it did before anyone started counting.
_STRESS_UNIT_FACTOR_CACHE = {}


def resolve_stress_unit_factor(skill_name, logger=None):
    """Resolve the stress -> Hartree/Bohr^3 factor from a skill manifest.

    Mirrors optimize.py::_resolve_skill_units for the whole-system
    stress tensor: reads the 'units.stress' entry of the skill's
    manifest.yaml and returns the multiplicative factor converting the
    skill's native stress unit into atomic units. A missing manifest or
    unit defaults to 1.0 (the skill already emits Hartree/Bohr^3).

    Parameters
    ----------
    skill_name : str
        Skill package directory name (e.g. 'cp2k_skill').
    logger : Logger, optional
        Logger for the unknown-unit warning.

    Returns
    -------
    float
        Multiplicative conversion factor to Hartree/Bohr^3.
    """
    if not skill_name:
        return 1.0
    cached = _STRESS_UNIT_FACTOR_CACHE.get(skill_name)
    if cached is not None:
        return cached
    skills_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'skills', skill_name)
    manifest_path = os.path.join(skills_dir, 'manifest.yaml')
    if not os.path.exists(manifest_path):
        _STRESS_UNIT_FACTOR_CACHE[skill_name] = 1.0
        return 1.0
    try:
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as error:
        # A read that failed may succeed next step, so nothing is kept;
        # report it instead of silently treating the unit as atomic.
        if logger is not None:
            logger.warning(
                "Could not read the stress unit from manifest of skill "
                "'%s' (%s); assuming Hartree/Bohr^3.",
                skill_name, error)
        return 1.0
    unit = str((manifest.get('units', {}) or {})
               .get('stress', '')).lower().strip()
    if not unit:
        _STRESS_UNIT_FACTOR_CACHE[skill_name] = 1.0
        return 1.0
    factor = _STRESS_UNIT_FACTORS.get(unit)
    if factor is None:
        if logger is not None:
            logger.warning(
                "Unknown stress unit '%s' in manifest of skill '%s', "
                "assuming Hartree/Bohr^3.", unit, skill_name)
        return 1.0
    _STRESS_UNIT_FACTOR_CACHE[skill_name] = factor
    return factor


def convert_stress_to_atomic(data, system_task, logger=None):
    """Convert data['stress_tensor'] to Hartree/Bohr^3 in place.

    The whole-system stress comes from the structural-track provider
    (resolve_system_task); its skill manifest declares the native stress
    unit. Applied after aggregation and before set_result so
    active_stress() and the NPT barostats read atomic units.

    Parameters
    ----------
    data : dict
        Assembled result dictionary (may carry 'stress_tensor').
    system_task : dict or None
        Whole-system structural provider task descriptor.
    logger : Logger, optional
        Logger for the unknown-unit warning.

    Returns
    -------
    dict
        The same dictionary with 'stress_tensor' converted (unchanged
        when no stress is present).
    """
    stress = data.get('stress_tensor')
    if stress is None:
        return data
    skill_name = (system_task or {}).get('skill_name')
    factor = resolve_stress_unit_factor(skill_name, logger=logger)
    if factor != 1.0:
        data['stress_tensor'] = (
            np.asarray(stress, dtype=np.float64) * factor)
    return data


def adapt_skill_result(skill_dict, skill_name=None):
    """Convert a skill backend result dict to a Result object.

    A skill may emit its numbers in native units and mark the result with
    ``_native_units``; the framework scales them here, before the Result is
    built, using the units that skill's manifest declares. The framework
    never imports a skill module to do it, and never decides by inspection
    which label holds the gradient: one label, ``gradient`` = dU/dR, for
    every skill.

    Parameters
    ----------
    skill_dict : dict
        Result dictionary from backend.calculate().
    skill_name : str or None
        Skill the result came from. Required when the result still carries
        its native-unit marker; an assembled composite result is already in
        framework units and carries no marker.

    Returns
    -------
    Result
        Result object with all computed properties.
    """
    if not isinstance(skill_dict, dict):
        raise TypeError("Expected dict from backend.calculate(), got %s" % type(skill_dict).__name__)
    if NATIVE_UNITS_MARKER in skill_dict and not skill_name:
        raise ValueError(
            "a result still carrying the native-unit marker %r reached the "
            "framework without a skill name, so its units cannot be "
            "resolved." % skill_dict[NATIVE_UNITS_MARKER])
    if skill_name:
        skill_dict = apply_result_units(skill_dict, skill_name)
    return Result(data=skill_dict)



def store_task_result(task, result_dict):
    """Store a task result into the task config if not already there.

    Backends write results into their config themselves; this is the
    fallback for backends that only return results (e.g. mock
    backends in tests), keeping aggregation source uniform.

    Parameters
    ----------
    task : dict
        Task descriptor.
    result_dict : dict
        Result dictionary from backend.calculate().
    """
    if not result_dict:
        return
    config = task.get('config')
    if config is None or task_config_results(task):
        return
    write_task_results(task, result_dict)


def write_task_results(task, result_dict):
    """Write a result dictionary into the task config (overwrite).

    Parameters
    ----------
    task : dict
        Task descriptor.
    result_dict : dict
        Result dictionary to store.
    """
    if result_dict is None:
        return
    config = task.get('config')
    if config is None:
        return
    if isinstance(config, dict):
        config['results'] = dict(result_dict)
    elif 'results' in type(config).model_fields:
        config.results = dict(result_dict)
