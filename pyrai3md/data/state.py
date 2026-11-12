"""
======================================================================
PyRAIState: the run state container.

Task queues, history frames, per-center electronic carriers, the
constraint-plan cache and the read-only property surface live here.
Turning a Config into those tasks is planning/assembly.py; creating the
container and its charge store is data/build.py. This split keeps the
state module free of planning/workflow imports.

Nov 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import warnings
from copy import deepcopy

import numpy as np

from ..schema.config import Optimize, Hessian
from .center import CenterPlanError
from .runtime.result import Result
from ..utilities.serialization import json_safe
from ..utilities.units import amu_to_au


HISTORY_FRAME_VERSION = '1.0'


def _active_gradient(gradient, current_state):
    """Extract the gradient of the active electronic state.

    Shared by PyRAIState.active_gradient() (the C=1 degenerate path
    that assembles the active-combination gradient from the
    state-resolved result) and the history-frame write. A 3-D
    (nstate, natom, 3) array selects the active state's row; a 2-D
    (natom, 3) array is already a single-state gradient and passes
    through; anything else yields None.

    Parameters
    ----------
    gradient : array-like or None
        State-resolved gradient (nstate, natom, 3) or a single-state
        gradient (natom, 3).
    current_state : int
        0-based index of the active electronic state.

    Returns
    -------
    np.ndarray or None
        Active-state gradient (natom, 3), or None when the input is
        missing or has an unsupported shape.
    """
    if gradient is None:
        return None
    grad = np.asarray(gradient, dtype=np.float64)
    if grad.ndim == 3:
        if not 0 <= int(current_state) < grad.shape[0]:
            return None
        return grad[int(current_state)]
    if grad.ndim == 2:
        return grad
    return None


def _active_stress(stress, current_state):
    """Extract the stress tensor of the active electronic state.

    Mirror of _active_gradient for the (nstate, 3, 3) stress tensor: a
    3-D array selects the active state's 3x3 block; a 2-D (3, 3) array
    is already a single-state stress and passes through; anything else
    yields None. The generic (nstate, 3, 3) format keeps room for
    future excited-state stress.

    Parameters
    ----------
    stress : array-like or None
        State-resolved stress tensor (nstate, 3, 3) or a single-state
        stress (3, 3).
    current_state : int
        0-based index of the active electronic state.

    Returns
    -------
    np.ndarray or None
        Active-state stress tensor (3, 3), or None when the input is
        missing or has an unsupported shape.
    """
    if stress is None:
        return None
    arr = np.asarray(stress, dtype=np.float64)
    if arr.ndim == 3:
        if not 0 <= int(current_state) < arr.shape[0]:
            return None
        return arr[int(current_state)]
    if arr.ndim == 2:
        return arr
    return None


class HistoryFrame(dict):
    """One structured MD history frame (docs 3.2/10.2; refactor R4d).

    A frame IS a dict, so the legacy hot-loop consumers keep reading
    ``snapshot['system']`` / ``snapshot['results']`` unchanged: the
    pmd-dump keys are a versioned adaptation layer that R5 migrates to
    the structured fields below (deletion gate: R5). On top of the
    legacy keys a frame carries the explicit new-frame protocol:

    - ``frame_version`` / ``step`` / ``time`` / ``evaluation_id`` /
      ``plan_fingerprint``: frame identity, shared by every center.
    - ``nuclear_sample``: the PRE-hop sample {R, V_pre, K_pre,
      active_config_pre}.
    - ``centers``: per-center static spec dicts in declared order.
    - ``electronic_outcome``: {center_index: per-center arrays}.
    - ``continuation``: the POST-hop {V_post, active_config_post,
      active_gradient_post, next_step}, or None while the frame is a
      warmup frame (the hop has not been recorded yet).
    - ``warmup``: True when the continuation is not yet available.

    The pre-hop sample and the post-hop continuation stay separate so a
    single frame records both sides of a surface hop. All centers of a
    frame share step/time and the whole frame is trimmed as one unit.
    """

    @property
    def frame_version(self):
        """Layout version of this frame (HISTORY_FRAME_VERSION)."""
        return self.get('frame_version')

    @property
    def step(self):
        """MD step index shared by every center of this frame."""
        return self.get('step')

    @property
    def time(self):
        """Simulation time (au) shared by every center of this frame."""
        return self.get('time')

    @property
    def evaluation_id(self):
        """Calculator evaluation id that produced this frame."""
        return self.get('evaluation_id')

    @property
    def plan_fingerprint(self):
        """Center-plan fingerprint in force when the frame was taken."""
        return self.get('plan_fingerprint')

    @property
    def nuclear_sample(self):
        """PRE-hop nuclear sample {R, V_pre, K_pre, active_config_pre}."""
        return self.get('nuclear_sample') or {}

    @property
    def centers(self):
        """Per-center static spec dicts in declared order."""
        return self.get('centers') or []

    @property
    def electronic_outcome(self):
        """{center_index: per-center electronic arrays} of this frame."""
        return self.get('electronic_outcome') or {}

    @property
    def continuation(self):
        """POST-hop continuation dict, or None for a warmup frame."""
        return self.get('continuation')

    @property
    def warmup(self):
        """True when the post-hop continuation is not yet available."""
        return bool(self.get('warmup', True))


class PyRAIState:
    """Global state manager organized as a Task Queue.

    Manages tasks (the minimal running unit: one independent backend
    invocation each) organized by category (theory/reference). Every
    task carries its role, execution index, contribution coefficient,
    region slice metadata, skill metadata and an independent
    PyRAIConfig. The queue order is the Calculator execution order.

    The pmd attribute holds the framework PyRAIConfig instance and is
    the SOLE runtime data source. jobtype, structure and result
    information are exposed through read-only properties that pass
    the pmd data through without rebuilding view objects. External
    code never mutates these values directly: all modifications go
    through update_system/set_result, which write into pmd. The
    Workflow, Calculator and future integration code intentionally
    share this mutable PyRAIConfig object during runtime.

    Structure:
        state.theory = [{'role': 'qm', 'task': 0, 'task_id': 'qm',
                         'layer_id': 0,
                         'static_id': 'tmp_PYSCF',
                         'contribution': 1,
                         'region': 'all', 'region_indices': [],
                         'region_charge': None,
                         'region_multiplicity': None,
                         'skill_name': 'pyscf_skill',
                         'api': 'pyrai3md.skills.pyscf_skill.backend.PySCFBackend',
                         'capabilities': [...],
                         'config': PyRAIConfig}]
        state.reference = [same structure]

    Backend instances are NOT stored in the state; they are runtime
    objects owned by the Calculator (Calculator.initialize()).

    Attributes
    ----------
    runtime : RuntimeState or None
        Runtime progress tracking.
    theory : list of dict
        Theory task queue.
    reference : list of dict
        Reference task queue.
    pmd : PyRAIConfig or None
        Framework PyRAIConfig instance: the sole runtime data source
        (control/system/dynamics/adaptive parameters and composite
        aggregated results), kept separate from skill configs.
    execution_graph : list of dict or None
        Frozen execution graph built by the ExecutionGraphBuilder
        (root runtime) or stamped by the parent Dispatcher (worker
        runtimes). Travelled with the serialized state.
    resolve_execution_graph : bool
        Whether this runtime must build its own execution graph. Root
        runtimes resolve it (True); worker runtimes created by the
        Dispatcher skip planning (False) and only verify the
        parent-frozen graph.

    History snapshots of completed timesteps are kept in a bounded
    list (_history, chronological order) and are managed exclusively
    through the official history API (update_history / get_history /
    get_history_step / clear_history). External code never touches
    _history directly.
    """

    def __init__(self, runtime=None):
        self.runtime = runtime
        self.theory = []       # task queue (list of task dicts)
        self.reference = []    # task queue (list of task dicts)
        self.pmd = None        # PyRAIConfig instance (sole data source)
        self.execution_graph = None
        self.resolve_execution_graph = True
        self._history = []     # bounded snapshot list (internal only)
        # ---- Constraint subsystem carriers ----
        # The plan is derived from pmd.system.constraints and cached, because
        # building it normalizes indices and units and a run must not do that
        # twice against two answers. The runtime state carries the bound
        # targets, the anchors and the deposited hills, and it travels through
        # the serializer rather than being re-captured from the live geometry.
        self._constraint_plan = None
        self.constraint_state = None
        # ---- Unified center data carriers (docs R4, r4a) ----
        # The canonical partition plan (data.center.PartitionPlan) is
        # attached by planning and persisted; None until bound (a mock
        # config in tests, or before build_state runs).
        self.partition_plan = None
        # Per-center mutable electronic data: a list (one entry per
        # center) of dicts holding independent arrays (energy /
        # gradient / nac / soc / density). Never padded into a fake
        # sum(S_c) global state axis.
        self.electronic = []
        # The active-surface record: active_configuration (per-center
        # local states), active_energy, active_gradient, evaluation_id.
        self.active_surface = None
        # The continuation handed to the next step (R/V/s/F + next_step).
        self.continuation = None
        # Transient staging slot for one pending evaluation (never
        # persisted; commit_step() clears it).
        self._staged_evaluation = None
        # Per-atom partial-charge store for ONIOM electrostatic
        # embedding (data.charges.ChargeStore). None unless charge
        # management is active (a three-layer ONIOM run with embedding
        # or read_charge); built by build_state, persisted on restart.
        self.charge_store = None
        # Adaptive method alignment: the planning-derived label source and
        # auxiliary channels. It is informational (run log and adaptive
        # summary), never a second source of the task plan.
        self.method_alignment = None

    # ---- Read-only pass-through properties (derived from pmd) ----

    @property
    def jobtype(self):
        """Global job type for workflow resolution (read-only)."""
        if self.pmd is None:
            return ''
        return self.pmd.control.jobtype

    @property
    def driver(self):
        """Execution driver from pmd.control (read-only).

        'main' lets the framework resolve the workflow from jobtype;
        'skill' or a skill name hands the jobtype to the skill for
        internal execution (framework single shot).
        """
        if self.pmd is None:
            return 'main'
        return getattr(self.pmd.control, 'driver', 'main')

    @property
    def atoms(self):
        """Element symbols from pmd.system (read-only)."""
        return self.pmd.system.atoms if self.pmd is not None else []

    @property
    def coordinates(self):
        """Cartesian coordinates from pmd.system (read-only)."""
        return self.pmd.system.coordinates if self.pmd is not None else []

    @property
    def velocities(self):
        """Cartesian velocities from pmd.system (read-only)."""
        return self.pmd.system.velocities if self.pmd is not None else []

    @property
    def mass(self):
        """Atomic masses in amu from pmd.system (read-only)."""
        return self.pmd.system.mass if self.pmd is not None else []

    @property
    def mass_au(self):
        """Atomic masses in atomic units (read-only).

        Central mass-unit access point: converts pmd.system.mass
        (amu) through utilities.units. The integrators (verlet,
        fssh) and kinetic_energy() must read masses here instead of
        converting them locally.

        Returns
        -------
        np.ndarray or None
            Mass array of shape (natom,) in atomic units, or None
            when pmd is absent, the masses are missing, their count
            differs from system.natom or any value is non-positive.
        """
        if self.pmd is None:
            return None
        mass = np.asarray(self.mass, dtype=np.float64)
        natom = int(self.pmd.system.natom)
        if mass.size != natom or mass.size == 0 or np.any(mass <= 0.0):
            return None
        return amu_to_au(mass)

    @property
    def timestep(self):
        """MD timestep in atomic units from pmd.dynamics (read-only).

        Passes through pmd.dynamics.size; consumed by the nuclear
        integrators when the workflow does not provide an explicit
        timestep.
        """
        if self.pmd is None:
            return 20.67
        dynamics = getattr(self.pmd, 'dynamics', None)
        return getattr(dynamics, 'size', 20.67) if dynamics else 20.67

    @property
    def charges(self):
        """Total charge from pmd.system (read-only)."""
        return self.pmd.system.charges if self.pmd is not None else 0

    @property
    def point_charges(self):
        """Whole-system point charges [q, x, y, z] (read-only).

        When charge management is active (a ChargeStore is bound), the
        per-atom partial charges of every atom are derived from the
        store at the CURRENT geometry, in global atom order (row i is
        atom i). Otherwise this falls back to the static
        pmd.system.point_charges supplied by the input, so a run
        without charge management is unchanged.

        This is the per-atom partial-charge channel and is distinct
        from the scalar state.charges (the system total charge) and
        from the arbitrary external_charges file.
        """
        if self.charge_store is not None:
            coords = np.asarray(
                self.coordinates, dtype=float).reshape(-1, 3)
            return self.charge_store.point_charges(coords)
        if self.pmd is not None:
            return self.pmd.system.point_charges
        return []

    @property
    def external_potential(self):
        """Per-atom [V, Fx, Fy, Fz] external potential (read-only)."""
        if self.pmd is not None:
            return self.pmd.system.external_potential
        return []

    @property
    def multiplicity(self):
        """Spin multiplicity from pmd.system (read-only)."""
        return self.pmd.system.multiplicity if self.pmd is not None else 1

    @property
    def nstate(self):
        """Number of electronic states from pmd.system (read-only).

        Single-center projection: for C=1 (or before a partition plan
        is bound) this is the global state count. For C>1 there is no
        public global state axis (docs 3.1 / r4a); consumers must read
        the per-center electronic_spec through center(c).nstate, so
        this property raises instead of fabricating sum(S_c).
        """
        if self._plan_n_center() > 1:
            raise CenterPlanError(
                "PyRAIState.nstate is undefined for a multi-center run "
                "(C=%d); read the per-center state count through "
                "center(c).nstate (a center index is required)."
                % self._plan_n_center())
        return self.pmd.system.nstate if self.pmd is not None else 1

    @property
    def current_state(self):
        """Currently active electronic state from pmd.system (read-only).

        0-based index consistent with the state numbering used in
        the input (state_i) and the first axis of state-resolved
        result arrays such as gradient (ns, na, 3).

        Single-center projection: for C>1 the active state is per
        center (docs 3.1 / r4a); read it through
        get_active_configuration()[c] instead of this global scalar.
        """
        if self._plan_n_center() > 1:
            raise CenterPlanError(
                "PyRAIState.current_state is undefined for a "
                "multi-center run (C=%d); read the per-center active "
                "state through get_active_configuration()[c] (a center "
                "index is required)." % self._plan_n_center())
        if self.pmd is None:
            return 0
        return getattr(self.pmd.system, 'current_state', 0)

    @property
    def root(self):
        """Initial electronic state from pmd.dynamics (read-only).

        Starting state of a dynamics run; workflows copy it into
        system.current_state at initialization so the authoritative
        active-state record begins from the input root.
        """
        if self.pmd is None:
            return 0
        return getattr(self.pmd.dynamics, 'root', 0)

    @property
    def activestate(self):
        """Gradient scope flag from pmd.dynamics (read-only).

        0 requests gradients of all electronic states; a nonzero
        value restricts the gradient evaluation to the current
        (active) state only. Workflows translate it into the
        target_states list passed to the calculator.
        """
        if self.pmd is None:
            return 0
        return getattr(self.pmd.dynamics, 'activestate', 0)

    @property
    def optimize_method(self):
        """Optimization method name from pmd.optimize (read-only)."""
        if self.pmd is None:
            return 'minima'
        opt = getattr(self.pmd, 'optimize', None)
        return getattr(opt, 'opt_method', 'minima') if opt else 'minima'

    @property
    def optimize_maxit(self):
        """Max optimization iterations from pmd.optimize (read-only)."""
        if self.pmd is None:
            return 100
        opt = getattr(self.pmd, 'optimize', None)
        return getattr(opt, 'opt_maxit', 100) if opt else 100

    @property
    def optimize_params(self):
        """Full optimize config dict from pmd.optimize (read-only).

        Values come from the `Optimize` model, so a field added there is
        carried here without a second whitelist to keep in sync.

        Returns
        -------
        dict
            The `Optimize` submodel's fields.
        """

        opt = (getattr(self.pmd, 'optimize', None)
               if self.pmd is not None else None)
        params = (opt or Optimize()).model_dump()
        # `opt_state_j` names the SECOND surface of a pair, so leaving it out
        # means there is no second surface: it resolves to the state the run
        # already follows. A literal 0 there would be read as "the user asked
        # for state 0", which is a different claim.
        if params.get('opt_state_j') is None:
            params['opt_state_j'] = params.get('opt_state_i', 0)
        return params

    @property
    def hessian_params(self):
        """The `&hessian` section as a flat dict, or the model defaults.

        Values come from the `Hessian` model, so a field added there is
        carried here without a second whitelist to keep in sync.
        """

        section = (getattr(self.pmd, 'hessian', None)
                   if self.pmd is not None else None)
        return (section or Hessian()).model_dump()

    @property
    def result(self):
        """Latest calculation result from pmd.results (read-only).

        Returns
        -------
        Result or None
            Result view of pmd.results, or None when empty.
        """
        if self.pmd is None or not self.pmd.results:
            return None
        return Result(data=dict(self.pmd.results))

    # ---- Unified center data interface (docs R4, r4a) ----

    def bind_partition_plan(self, plan):
        """Attach the canonical partition plan and size the center data.

        Planning builds the plan once (data.center.build_center_plan)
        and binds it here so it is persisted instead of discarded.
        Binding sizes the per-center electronic carrier to C entries
        (C=1 included) and seeds the active surface from each center's
        initial_root, so get_active_configuration() is valid before the
        first evaluation.

        Parameters
        ----------
        plan : PartitionPlan or None
            Canonical center plan (None keeps the unbound C=1 default).

        Returns
        -------
        PyRAIState
            self, for call chaining.
        """
        self.partition_plan = plan
        if plan is None:
            return self
        self.electronic = [self._empty_center_electronic()
                           for _ in range(plan.n_center)]
        self.active_surface = {
            'active_configuration': tuple(
                plan.center(c).electronic_spec.initial_root
                for c in range(plan.n_center)),
            'active_energy': None,
            'active_gradient': None,
            'evaluation_id': None,
        }
        return self

    def _empty_center_electronic(self):
        """One center's empty electronic carrier (independent arrays)."""
        return {'energy': None, 'gradient': None, 'nac': None,
                'soc': None, 'density': None}

    def _plan_n_center(self):
        """Center count of the bound plan (1 when unbound)."""
        plan = self.partition_plan
        return plan.n_center if plan is not None else 1

    @property
    def n_center(self):
        """Number of centers C (1 for a plain single-center run)."""
        return self._plan_n_center()

    def center(self, c):
        """Bounded read interface to one CenterSpec (docs 3.3).

        Parameters
        ----------
        c : int
            Center index (contiguous, 0-based).

        Returns
        -------
        CenterSpec
            The center's static geometry + electronic definition.
        """
        plan = self.partition_plan
        if plan is None:
            raise CenterPlanError(
                "center(%r) requires a bound partition plan; none is "
                "attached (a plain run binds C=1 through planning)."
                % (c,))
        return plan.center(c)

    def center_electronic(self, c):
        """Return the mutable electronic carrier of one center.

        Parameters
        ----------
        c : int
            Center index (0-based).

        Returns
        -------
        dict
            The center's electronic arrays (energy/gradient/nac/soc/
            density); values are None until an evaluation fills them.
        """
        if not isinstance(c, int) or isinstance(c, bool):
            raise CenterPlanError(
                "center electronic index must be an int, got %r." % (c,))
        if c < 0 or c >= len(self.electronic):
            raise CenterPlanError(
                "center electronic index %d out of range [0, %d)."
                % (c, len(self.electronic)))
        return self.electronic[c]

    def set_center_electronic(self, c, **arrays):
        """Store per-center electronic arrays (energy/gradient/nac/...).

        Independent arrays per center; no padding and no fabricated
        global state axis (docs r4a). Only the keys given are written.

        Parameters
        ----------
        c : int
            Center index (0-based).
        **arrays
            Array-typed electronic data keyed by property name.

        Returns
        -------
        dict
            The center's electronic carrier after the update.
        """
        carrier = self.center_electronic(c)
        for key, value in arrays.items():
            carrier[key] = value
        return carrier

    def get_active_configuration(self):
        """Per-center active local states as a tuple (docs r4a).

        C=1 returns (s0,); C>1 returns one local state per center.
        Before any evaluation is staged, the seeded initial roots are
        returned; for an unbound C=1 state the global current_state is
        the fallback.

        Returns
        -------
        tuple of int
            Active local state per center.
        """
        surface = self.active_surface
        if surface and surface.get('active_configuration') is not None:
            return tuple(surface['active_configuration'])
        if self._plan_n_center() == 1:
            return (getattr(self.pmd.system, 'current_state', 0)
                    if self.pmd is not None else 0,)
        raise CenterPlanError(
            "get_active_configuration() needs a staged active surface "
            "for a multi-center run (C=%d)." % self._plan_n_center())

    def get_active_energy(self):
        """Active-combination energy from the active surface (or None)."""
        surface = self.active_surface or {}
        return surface.get('active_energy')

    def get_active_gradient(self):
        """Active-combination global gradient (natom, 3), or None."""
        surface = self.active_surface or {}
        gradient = surface.get('active_gradient')
        if gradient is None:
            return None
        return np.asarray(gradient, dtype=np.float64)

    def active_gradient(self):
        """Assembled active-combination global gradient (natom, 3).

        The single source the nuclear integrator reads (docs R6, r6a):
        verlet consumes this integrated gradient and never re-derives
        it from a recorded state index, so the multi-center assembled
        gradient and the single-center active-state gradient flow
        through one accessor.

        - C>1: the committed active_surface gradient (calculator.run
          stages and commits it every step); None when no gradient was
          committed, never falling back to the global current_state
          (which raises for a multi-center run).
        - C=1 degenerate path: assembled from the state-resolved
          result gradient at the current active state, exactly the row
          the legacy integrator selected, so the single-center
          trajectory stays bit-for-bit identical.

        Returns
        -------
        np.ndarray or None
            Active-combination gradient (natom, 3), or None when no
            gradient is available.
        """
        surface = self.active_surface or {}
        gradient = surface.get('active_gradient')
        if gradient is not None:
            return np.asarray(gradient, dtype=np.float64)
        if self._plan_n_center() > 1:
            return None
        result = self.result
        if result is None:
            return None
        return _active_gradient(result.get('gradient'), self.current_state)

    def active_stress(self):
        """Active-state whole-system stress tensor (3, 3), or None.

        Structural-property counterpart of active_gradient(): reads the
        whole-system stress written back by the calculator through the
        structural-property track (resolve_system_task; dev/methods/
        ENSEMBLE_DEVELOPMENT.md section 4A), so it is authoritative for both
        C=1 and C>1. The full (nstate, 3, 3) record lives in
        pmd.results['stress_tensor']; this projects the active state's
        3x3 block (mirroring _active_gradient), the quantity the NPT
        barostats consume as the virial.

        For C>1 there is no global current_state (it raises), so the
        single whole-system provider's reference state (index 0) is
        projected instead.

        The cp2k / xtb / lammps backends populate stress_tensor when the
        planning layer derives the 'stress' capability (md+ensemble=npt
        or optimize+optimize_cell); the calculator converts it to
        Hartree/Bohr^3 before write-back. When no backend supplied a
        stress this returns None and the barostat must treat it as
        unavailable rather than as a physical zero.

        Returns
        -------
        np.ndarray or None
            Active-state stress tensor (3, 3), or None when no stress
            is available.
        """
        result = self.result
        if result is None:
            return None
        stress = result.get('stress_tensor')
        if self._plan_n_center() > 1:
            return _active_stress(stress, 0)
        return _active_stress(stress, self.current_state)

    def state_stress(self, index):
        """The whole-system stress of one named electronic state, or None.

        `active_stress()` answers "what is the stress of the state this run is
        on", which is what a barostat and a single-surface relaxation both want.
        A search whose objective is built out of two surfaces needs the other
        question: the stress of *that* state, named. Reading the active state's
        tensor twice would step the box on one surface while stepping the atoms
        on the average of two, and the two halves of one generalized vector would
        belong to different objectives.

        A single-state tensor is therefore refused here rather than reused: a
        (3, 3) reading carries no state axis, so it can say what the box is
        under and nothing about which surface put it there. A backend that means
        to serve a crossing search answers with (nstate, 3, 3) - the shape the
        properties schema declares and the numerical-strain route in the CP2K
        skill already produces.

        Parameters
        ----------
        index : int
            0-based state index, in the same order the run's `target_states`
            names.

        Returns
        -------
        numpy.ndarray or None
            (3, 3) in Hartree per Bohr^3, tensile-positive, or None when this
            evaluation carries no state-resolved tensor or no such state.
        """
        result = self.result
        if result is None:
            return None
        stress = result.get('stress_tensor')
        if stress is None:
            return None
        array = np.asarray(stress, dtype=np.float64)
        if array.ndim != 3:
            return None
        if not 0 <= int(index) < array.shape[0]:
            return None
        return array[int(index)]

    def stage_evaluation(self, evaluation):
        """Hold one evaluation pending commit (docs r4a; R6 writeback).

        The evaluation carries the per-center electronic outcome, the
        post-hop active surface and the continuation for the next step:
        {'electronic': {c: {...arrays}}, 'active_surface': {...},
        'continuation': {...}}. Staging never mutates the committed
        center data; commit_step() applies it atomically, so a failure
        before the commit leaves the previous state intact.

        Parameters
        ----------
        evaluation : dict
            Staged evaluation payload.

        Returns
        -------
        dict
            The staged payload.
        """
        self._staged_evaluation = dict(evaluation or {})
        return self._staged_evaluation

    def commit_step(self, outcome=None):
        """Atomically write the staged evaluation into the center data.

        This is the data-model commit; R6 makes it the sole writeback
        point of the MD loop. Applies electronic / active_surface /
        continuation from the staged evaluation in one step. ``outcome``
        is accepted for the R6 caller contract and, when given, is
        staged first.

        Parameters
        ----------
        outcome : dict or None
            Optional evaluation payload to stage before committing.

        Returns
        -------
        PyRAIState
            self, for call chaining.
        """
        if outcome is not None:
            self.stage_evaluation(outcome)
        staged = getattr(self, '_staged_evaluation', None)
        if not staged:
            return self
        electronic = staged.get('electronic')
        if electronic:
            for c, arrays in electronic.items():
                self.set_center_electronic(int(c), **(arrays or {}))
        surface = staged.get('active_surface')
        if surface is not None:
            self.active_surface = dict(surface)
        continuation = staged.get('continuation')
        if continuation is not None:
            self.continuation = dict(continuation)
        self._staged_evaluation = None
        return self

    # ---- History management (bounded snapshot list) ----

    @property
    def history_limit(self):
        """Max number of snapshots kept (read-only).

        Controlled by the input keyword ``history`` carried by
        pmd.dynamics; falls back to 4 while pmd is not initialized.
        PyRAIState only knows the bound, never its physical meaning.
        """
        if self.pmd is None:
            return 4
        dynamics = getattr(self.pmd, 'dynamics', None)
        limit = getattr(dynamics, 'history', 4) if dynamics else 4
        try:
            return max(int(limit), 1)
        except (TypeError, ValueError):
            return 4

    @property
    def history_size(self):
        """Number of snapshots currently stored (read-only)."""
        return len(self._history)

    def _build_history_frame(self):
        """Build one structured HistoryFrame from the current state.

        Captures the PRE-hop nuclear sample and the committed
        per-center electronic outcome; the POST-hop continuation comes
        from self.continuation once the workflow has staged it (R6
        commit_step), otherwise the frame is an explicit warmup frame
        with a null continuation. The legacy pmd-dump keys are carried
        so the existing hot-loop consumers keep reading
        snapshot['system'] / snapshot['results'] until R5 migrates them
        (docs 3.2/10.2, refactor R4d).

        Returns
        -------
        HistoryFrame
            The structured frame (a dict subclass).
        """
        frame = HistoryFrame(deepcopy(self.pmd.model_dump()))
        runtime = self.runtime
        plan = self.partition_plan

        # active_config_pre: the committed active configuration. A
        # multi-center run with no staged surface has no pre-hop
        # configuration to record, so it stays None instead of raising
        # inside the history write (which must never break the loop).
        try:
            active_config_pre = tuple(self.get_active_configuration())
        except CenterPlanError:
            active_config_pre = None

        continuation = getattr(self, 'continuation', None)
        frame['frame_version'] = HISTORY_FRAME_VERSION
        frame['step'] = getattr(runtime, 'current_step', None)
        frame['time'] = getattr(runtime, 'simulation_time', None)
        frame['evaluation_id'] = (self.active_surface or {}).get(
            'evaluation_id')
        frame['plan_fingerprint'] = (
            plan.plan_fingerprint if plan is not None else None)
        # active_gradient: the assembled active-combination gradient
        # the NEXT step's verlet reads (docs R6, r6a). Captured here so
        # the integrator never re-derives it from a recorded state
        # index; for C=1 it equals gradient[current_state] at this
        # snapshot, bit-for-bit the legacy selection.
        active_gradient = self.active_gradient()
        active_stress = self.active_stress()
        frame['nuclear_sample'] = {
            'R': np.array(self.coordinates, dtype=np.float64),
            'V_pre': np.array(self.velocities, dtype=np.float64),
            'K_pre': float(getattr(runtime, 'kinetic_energy', 0.0) or 0.0),
            'active_config_pre': active_config_pre,
            'active_gradient': (np.array(active_gradient, dtype=np.float64)
                                if active_gradient is not None else None),
            # active_stress: the active-state whole-system stress (3, 3)
            # the NEXT step's barostat momentum half-kick reads
            # (dev/methods/ENSEMBLE_DEVELOPMENT.md section 4). None when no
            # backend populated stress_tensor for this step.
            'active_stress': (np.array(active_stress, dtype=np.float64)
                              if active_stress is not None else None),
        }
        frame['centers'] = (
            [c.to_dict() for c in plan.centers]
            if plan is not None else [])
        # runtime: a light projection (not RuntimeState.to_dict()) --
        # the frame consumers (RunLogger.log_step ->
        # format_result_block, MdOutputWriter) only read the step
        # kinetic energy and the per-step ensemble diagnostics, while
        # a full to_dict() would deepcopy the ever-growing
        # step_results list into every retained frame.
        frame['runtime'] = {
            'kinetic_energy': float(
                getattr(runtime, 'kinetic_energy', 0.0) or 0.0),
            'ensemble': dict(getattr(runtime, 'ensemble', None) or {}),
        }
        frame['electronic_outcome'] = {
            c: deepcopy(self.electronic[c])
            for c in range(len(self.electronic))}
        frame['continuation'] = (
            deepcopy(continuation) if continuation else None)
        frame['warmup'] = continuation is None
        return frame

    def update_history(self):
        """Append one structured HistoryFrame to the history list.

        Builds the frame (see _build_history_frame) and appends it to
        the end of the list, so _history stays in chronological order
        (index 0 oldest, -1 newest). When the list exceeds
        history_limit, only the last N frames are kept: a frame is
        trimmed as one unit and every center in it shares step/time.
        Called by the workflow once a timestep is completed; the
        current state itself never enters the history.

        Raises
        ------
        ValueError
            If pmd is not initialized.
        """
        if self.pmd is None:
            raise ValueError("PyRAIState.pmd is not initialized.")
        self._history.append(self._build_history_frame())
        limit = self.history_limit
        if len(self._history) > limit:
            self._history = self._history[-limit:]

    def kinetic_energy(self):
        """Compute, store and return the nuclear kinetic energy (Hartree).

        KE = 0.5 * sum(mass[au] * velocity^2) with the atomic masses
        converted from amu to atomic units. The value is stored in
        runtime.kinetic_energy; 0.0 is stored and returned when the
        mass or velocity data is unavailable.

        Returns
        -------
        float
            Nuclear kinetic energy in Hartree.
        """
        ke = 0.0
        mass_au = self.mass_au
        velocities = np.asarray(self.velocities, dtype=float)
        if (mass_au is not None and velocities.size > 0
                and mass_au.shape[0] == velocities.shape[0]):
            ke = float(0.5 * np.sum(mass_au[:, None]
                                    * velocities ** 2))
        if self.runtime is not None:
            self.runtime.kinetic_energy = ke
        return ke

    def record_step_result(self):
        """Record runtime metadata of the current step (official API).

        Property data (energy, gradient, ...) is kept in pmd; runtime
        records execution metadata only (no duplication). Workflows
        call this through their checkpoint routine instead of writing
        to state.runtime directly.
        """
        if self.runtime is not None:
            self.runtime.record_step_result()

    def get_history(self, n=None):
        """Return copies of the stored snapshots (chronological order).

        Parameters
        ----------
        n : int, optional
            Number of most recent snapshots to return. None returns
            everything stored. Requesting more than available returns
            what exists (no fabricated data); n <= 0 returns [].

        Returns
        -------
        list of dict
            Deep copies of the snapshots; mutating them never affects
            the internal history.
        """
        if n is None:
            snapshots = self._history
        elif n > 0:
            snapshots = self._history[-n:]
        else:
            snapshots = []
        return deepcopy(snapshots)

    def get_history_step(self, index=-1, copy=True):
        """Return one snapshot by relative index.

        Parameters
        ----------
        index : int, optional
            List index into the chronological snapshot list (-1 = the
            state right before the current one, -2 = two steps back,
            ...; non-negative indices count from the oldest).
        copy : bool, optional
            Deep-copy the snapshot (default, safe). False returns a
            read-only reference into the internal snapshot list for
            hot-loop readers (e.g. the FSSH integrator); the caller
            must not mutate the returned dict or anything it holds.

        Returns
        -------
        dict or None
            The snapshot (deep copy or read-only reference), or None
            when the requested step does not exist (workflows decide
            how to handle missing history).
        """
        try:
            snapshot = self._history[index]
        except IndexError:
            return None
        return deepcopy(snapshot) if copy else snapshot

    def clear_history(self):
        """Remove all stored snapshots."""
        self._history = []

    # ---- Update interface (the sole legal modification path) ----

    def _update_section(self, section, updates):
        """Merge changed values into a pmd submodel with validation.

        Generic implementation: merges 'current values + updates' and
        rebuilds the submodel through Pydantic validation, so any
        field declared by the schema (including fields added in the
        future) can be updated without changing this interface.
        Fields not present in updates stay unchanged.

        Parameters
        ----------
        section : str
            PyRAIConfig submodel name (e.g. 'system').
        updates : dict
            Changed field values.

        Raises
        ------
        ValueError
            If pmd is not initialized.
        """
        if self.pmd is None:
            raise ValueError("PyRAIState.pmd is not initialized.")
        if not updates:
            return
        submodel = getattr(self.pmd, section)
        data = submodel.model_dump()
        data.update(updates)
        setattr(self.pmd, section, type(submodel).model_validate(data))

    def update_system(self, updates):
        """Update arbitrary system fields through pmd.

        Accepts a dict of changed values (coordinates, velocities,
        electronic state fields, or any future system field) and
        merges them into pmd.system. Not specialized to geometry so
        future integrators can pass any combination.

        Parameters
        ----------
        updates : dict
            Changed field values for pmd.system.
        """
        self._update_section('system', updates)

    def update_dynamics(self, updates):
        """Update arbitrary dynamics fields through pmd.

        Symmetric counterpart of update_system() for the pmd.dynamics
        submodel. Electronic dynamics algorithms write back the
        step-updated dynamics bookkeeping (e.g. the state population
        recorded by surface hopping) through this official path.

        Parameters
        ----------
        updates : dict
            Changed field values for pmd.dynamics.
        """
        self._update_section('dynamics', updates)

    def constraint_plan(self):
        """The constraint plan of this run, built once and then cached.

        An absent or empty `&system_constraint` block gives the empty plan,
        which is the single condition every consumer tests before it builds a
        controller, registers a finalizer or touches a history. The plan is
        rebuilt from the same function the input pipeline ran its refusal check
        against, so the two cannot disagree about what was declared.
        """
        if self._constraint_plan is None:
            from pyrai3md.data.constraint import ConstraintPlan
            from pyrai3md.schema.normalize import build_plan

            block = None
            if self.pmd is not None:
                block = getattr(self.pmd.system, 'constraints', None)
            if block and self.driver != 'main':
                # Ownership, not a preference: on any other driver the skill
                # receives the whole jobtype and integrates the positions, so a
                # framework projection would fight a constraint engine that
                # this run cannot even see. The block is dropped, and loudly -
                # a silently inert declaration is how a run ends up with the
                # geometry its input did not ask for.
                warnings.warn(
                    'control.driver=%r hands the jobtype to the skill, so the '
                    'framework constraint block is inert this run: declare the '
                    'constraints on the skill side instead '
                    '(D-27 ownership rule).' % self.driver)
                block = None
            if not block:
                self._constraint_plan = ConstraintPlan.empty()
            else:
                system = self.pmd.system
                mapping = (block.model_dump()
                           if hasattr(block, 'model_dump') else dict(block))
                # The run's own step count is the one piece of its length a
                # constraint can inherit: it closes a ramp whose `stop` the
                # input left open. A single-point job has none, and a ramp that
                # needs one is refused by the schema rather than guessed.
                dynamics = getattr(self.pmd, 'dynamics', None)
                steps = int(getattr(dynamics, 'step', 0) or 0) or None
                self._constraint_plan = build_plan(
                    mapping, atoms=list(system.atoms or []),
                    natom=int(system.natom or 0),
                    masses=list(system.mass or []) or None,
                    pbc=list(system.pbc or []) or None,
                    coordinates=list(system.coordinates or []) or None,
                    lattice=list(system.lattice_vector or []) or None,
                    md_step=steps)
        return self._constraint_plan

    def set_result(self, result_dict):
        """Write calculation results into pmd.results.

        Used by the Calculator for result write-back. Numpy arrays are
        converted to lists at any depth (utilities.serialization
        json_safe, the one converter): `set_result` used to convert
        only the top level, which quietly left the component arrays of
        a nested result as ndarrays -- they survive a local store,
        then fail or, worse, change shape in a subprocess payload.

        Parameters
        ----------
        result_dict : dict
            Result dictionary to store.

        Raises
        ------
        ValueError
            If pmd is not initialized.
        """
        if self.pmd is None:
            raise ValueError("PyRAIState.pmd is not initialized.")
        data = {}
        for key, value in (result_dict or {}).items():
            data[key] = json_safe(value)
        self.pmd.results = data

    def add_task(self, category, task):
        """Append a task to a category queue.

        Parameters
        ----------
        category : str
            'theory' or 'reference'.
        task : dict
            Task descriptor with keys: role, task, task_id, layer_id,
            static_id, contribution, region, region_indices,
            region_charge, skill_name, api, capabilities, config.
        """
        target = self.theory if category == 'theory' else self.reference
        target.append(task)

    def iter_tasks(self):
        """Iterate over all tasks in execution order.

        Yields
        ------
        tuple(str, dict)
            (category, task) pairs, theory queue first.
        """
        for task in self.theory:
            yield 'theory', task
        for task in self.reference:
            yield 'reference', task

    def __getitem__(self, key):
        """Dict-like access: state['theory'][0]['api']"""
        return getattr(self, key)

    def items(self):
        """Iterate over (category, task_queue) pairs."""
        yield 'theory', self.theory
        yield 'reference', self.reference

    def to_dict(self):
        """Serialize the state into a complete dictionary.

        Saves the runtime progress, task queue metadata (role, task,
        task_id, layer_id, static_id, contribution, region, skill
        name, capabilities, dispatch flags) and configs, the pmd
        runtime data source, and the history snapshot list. Backend
        instances are runtime objects owned by the Calculator and
        excluded; they are rebuilt through Calculator.initialize()
        on restart.

        Returns
        -------
        dict
            Serializable state dictionary.
        """
        data = {
            'jobtype': self.jobtype,
            'runtime': self.runtime.to_dict() if self.runtime is not None else None,
            'theory': [],
            'reference': [],
            'pmd': {},
            'execution_graph': self.execution_graph,
            'resolve_execution_graph': self.resolve_execution_graph,
            'history': deepcopy(self._history),
            'history_limit': self.history_limit,
            # Unified center data model (docs R4, r4a/r4e): the static
            # partition plan as a plain dict, the mutable per-center
            # electronic carriers, the committed active surface and the
            # staged continuation. Deep-copied so the dump is isolated
            # from the live state.
            'partition_plan': (self.partition_plan.to_dict()
                               if self.partition_plan is not None
                               else None),
            'electronic': deepcopy(self.electronic),
            'active_surface': deepcopy(self.active_surface),
            'continuation': deepcopy(self.continuation),
            # Per-atom charge store for ONIOM electrostatic embedding
            # (None unless charge management is active).
            'charge_store': (self.charge_store.to_dict()
                             if self.charge_store is not None else None),
        }
        from .task import dump_task
        for category in ('theory', 'reference'):
            for task in getattr(self, category):
                data[category].append(dump_task(task))
        if self.pmd is not None:
            # A failed dump must not silently drop the run's configuration
            # from the checkpoint; let the serialization error surface.
            data['pmd'] = self.pmd.model_dump()
        return data

    def __repr__(self):
        parts = []
        for category in ('theory', 'reference'):
            queue = getattr(self, category)
            if queue:
                labels = ["%s:task%d" % (t.get('role'), t.get('task', 0))
                          for t in queue]
                parts.append("%s=[%s]" % (category, ", ".join(labels)))
        return "PyRAIState(%s)" % ", ".join(parts) if parts else "PyRAIState()"
