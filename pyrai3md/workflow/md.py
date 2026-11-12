"""
======================================================================
This module provides the AIMD (Ab Initio Molecular Dynamics) workflow.
Each step propagates the coordinates to the current structure,
requests energy, gradient, NAC, and SOC in a single calculator call,
completes the velocities, records the timestep and checkpoints.

The iterative MD loop is owned by this concrete module; execution is
forwarded to the inherited Dispatcher pipeline, so the workflow never
inspects how the calculator is executed. The surface hopping
algorithms (FSSH/GSH) live in the integrator package and update the
authoritative active state themselves.

Multi-trajectory planning: run() resolves the initial conditions
(&md initcond/ninitcond); ninitcond > 1 fans the trajectories out
through self.dispatch() (one child runtime per trajectory, each with
its own runtime identity and sub working directory), while
ninitcond == 1 runs the single trajectory in place. Both paths
return the universal TrajectorySet: the per-trajectory outcomes
arranged by the planning-order idx, the single trajectory being the
trajectory-count-1 case. Every step additionally receives the NN
prediction deviations (err_* keys) from the backend result and
keeps them in pmd.results, so they travel with the history
snapshots for adaptive sampling.

Aug 01 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import time
from contextlib import nullcontext

import numpy as np

from .base import Workflow
from .records.deviation import (
    fetch_deviations, judge_deviations, merge_deviations)
from .records.analysis import gap_pairs
from .records.population import population_bad, state_population
from .initcond import (adopt_initial_conditions, normalize_initcond,
                       resolve_initial_conditions)
from .records.output import MdOutputWriter
from .records.diagnostics import (check_recorded_step,
                                  summary as diagnostics_summary)
from .records.trajectory import TrajectorySet
from ..utilities import tracker
from ..utilities.timer import PhaseProfiler
from .constraints import runtime as constraint_runtime
from ..data.geometry import apply_pbc, continuous_coordinates
from .integrator import (
    verlet_i, verlet_ii,
    phase_correction, apply_constrain,
    velocity_initialize, velocity_correction, apply_ensemble,
    verify_recorded, surface_hopping,
)


def resolve_target_states(activestate, current_state, nstate):
    """Derive the target_states list for a dynamics step.

    The framework owns the active-state bookkeeping (current_state,
    0-based); the skill backend only receives the list of states to
    evaluate and never learns which one is active.

    Parameters
    ----------
    activestate : int
        Gradient scope flag from &md: 0 requests gradients of all
        states; nonzero restricts them to the current state.
    current_state : int
        Currently active electronic state (0-based).
    nstate : int
        Number of electronic states.

    Returns
    -------
    list
        [0, ..., nstate-1] when activestate is 0, otherwise
        [current_state]. Never None: dynamics always needs gradients.

    Raises
    ------
    ValueError
        If current_state is outside [0, nstate).
    """
    if not 0 <= current_state < nstate:
        raise ValueError(
            "current_state=%d is out of range for nstate=%d"
            % (current_state, nstate))
    if activestate == 0:
        return list(range(nstate))
    return [current_state]


def resolve_target_states_centered(state):
    """Per-step target states, center-aware (docs R6, r6b).

    C=1 returns exactly the flat list resolve_target_states produces
    (bit-for-bit the legacy single-center request). C>1 has no global
    state axis (state.nstate / state.current_state raise), so it
    returns a per-center map {center_index: [states]} derived from
    each center's active local state and state count; the calculator
    routes each task's own center entry to that task.

    Parameters
    ----------
    state : PyRAIState
        State carrying the activestate flag and, for C>1, the bound
        partition plan and the active configuration.

    Returns
    -------
    list of int or dict
        Flat target-state list (C=1) or {center_index: list} (C>1).
    """
    activestate = state.activestate
    if state.n_center <= 1:
        return resolve_target_states(
            activestate, state.current_state, state.nstate)
    configuration = state.get_active_configuration()
    return {
        c: resolve_target_states(activestate, configuration[c],
                                 state.center(c).nstate)
        for c in range(state.n_center)
    }


def _trajectory_index(name):
    """Parse the fan-out trajectory index from a runtime name.

    Dispatched batch units run inside '{title}_{idx}' directories;
    the trailing integer suffix is the planning-order trajectory
    index. Names without a numeric suffix (single in-place runs)
    fall back to index 0.

    Parameters
    ----------
    name : str
        Runtime directory name or workdir path.

    Returns
    -------
    int
        The trajectory index (0 when unresolvable).
    """
    base = os.path.basename(str(name or '').rstrip('/'))
    tail = base.rsplit('_', 1)[-1]
    if tail.isdigit():
        return int(tail)
    return 0


def _seed_trajectory(state):
    """Seed the random number generator of one trajectory.

    The control-level gl_seed keyword drives reproducibility: every
    trajectory seeds the global NumPy generator with gl_seed plus its
    fan-out index, so identical inputs reproduce identical hopping
    sequences regardless of worker scheduling. The resolved seed is
    mirrored into runtime.random_seed for checkpoint inspection.

    A segment that resumes from `<title>.rt` does not reseed. The seed says
    which trajectory this is; the hop decisions of a step are draws from
    wherever that trajectory's stream has reached, so replaying the seed at
    the seam would make a run split into two processes part company with the
    same run in one process from the first hop after it. The stream position
    travels instead (`RuntimeState.random_state`, stamped with the runtime
    file), and the seed keeps its second job as the record of where the
    trajectory came from. A resumed state whose file predates that field
    falls back to seeding, which is what such a file has always done.

    Parameters
    ----------
    state : PyRAIState
        State with the pmd control section (gl_seed) and, for
        dispatched trajectories, the runtime workdir carrying the
        '{title}_{idx}' identity.
    """
    runtime = getattr(state, 'runtime', None)
    started = int(getattr(runtime, 'current_step', 0) or 0) if runtime \
        else 0
    carried = getattr(runtime, 'random_state', None) if runtime else None
    if started > 0 and carried is not None:
        np.random.set_state(carried)
        return
    gl_seed = int(getattr(getattr(state, 'pmd', None), 'gl_seed', 1)
                  or 1)
    workdir = getattr(runtime, 'workdir', '') if runtime is not None else ''
    seed = gl_seed + _trajectory_index(workdir)
    np.random.seed(seed)
    if runtime is not None:
        runtime.random_seed = seed


def _ground_state_reached(state):
    """Whether every electronic track of this run sits on its ground state.

    A single center is read from the framework's active-state ledger; a
    multi-center run has no global state axis, so the reading is over each
    center's active local state. The delay is judged on the whole run rather
    than per center because one Nose-Hoover chain is shared: switching it on
    for one center and off for another would give the same extended
    coordinates two different half steps.
    """
    if state.n_center <= 1:
        return int(state.current_state) == 0
    return all(int(local) == 0
               for local in state.get_active_configuration())


def _record_dissipate_landing(state, step):
    """Stamp the step this trajectory first arrived on the ground state.

    Written once and never moved back: a hop up does not close the thermostat
    gate again, and a restarted segment keeps the delay its predecessor had
    already run down. The cost when the run does not declare `dissipate` is
    one integer comparison.

    The number stamped is the loop's absolute step, not the runtime counter:
    the counter is advanced after the checkpoint of a step, so at the point a
    hop has just been decided it still reads one less than the step being
    integrated. Stamping the counter would write 0 on the very first step,
    which is the value that means 'has not landed yet'.
    """
    dynamics = getattr(getattr(state, 'pmd', None), 'dynamics', None)
    if dynamics is None or int(getattr(dynamics, 'dissipate', 0) or 0) != 1:
        return
    runtime = getattr(state, 'runtime', None)
    if runtime is None or int(getattr(runtime, 'dissipate_step', 0) or 0) > 0:
        return
    if _ground_state_reached(state):
        runtime.dissipate_step = int(step)


def _nested_profile(calculator, steps):
    """The calculator's own phase rows, when it keeps any.

    A step's force evaluation is one row of the loop table, but that row
    is where the interesting questions are: the plan, the work items, the
    backend call and the result write-back each cost something different,
    and only the backend call goes down to the engine. The component
    renders its own rows and the loop appends them under a header, so a
    reader never adds a nested row to the step row that contains it. A
    calculator without a breakdown (a stub, or a run that never reached
    one) contributes nothing.

    Parameters
    ----------
    calculator : Calculator or None
        The workflow's calculator.
    steps : int
        Steps measured, the divisor of the nested rows.

    Returns
    -------
    list of str
        Nested rows, empty when there are none.
    """
    render = getattr(calculator, 'profile_lines', None)
    return list(render(steps)) if callable(render) else []


def _center_regions(state):
    """Per-center region atom indices for the output writer (docs R6).

    Returns one high-index list per center when an environment
    partition exists (n_layer >= 2), else None so the writer keeps the
    byte-identical plain single-center output (no per-center files).

    Parameters
    ----------
    state : PyRAIState
        State carrying the bound partition plan.

    Returns
    -------
    list of list of int or None
        Per-center 0-based region atom indices, or None when there is
        no environment partition.
    """
    plan = getattr(state, 'partition_plan', None)
    if plan is None or getattr(plan, 'n_layer', 1) < 2:
        return None
    return [list(plan.center(c).high_indices)
            for c in range(plan.n_center)]


class AIMD(Workflow):
    """Ab initio molecular dynamics workflow.

    Each MD step:
    1. Propagate coordinates to the current structure (velocity-Verlet
       through the integrator package; internally skipped when no
       previous step is recorded)
    2. Call calculator for E + G + NAC + SOC in one request
    3. Complete velocities and apply the independent operations
    4. Record the timestep into the history
    5. Surface hopping (updates the active state itself) and sync;
       the checkpoint then writes the step and watches the frame it
       recorded

    Parameters
    ----------
    calculator : Calculator or None
        Calculator for E+G+NAC+SOC computations.
    maxiter : int
        Maximum MD steps.
    dt : float
        Time step in atomic units.
    sfhp : str
        Surface hopping method: 'nosh', 'gsh', or 'fssh'.
    initial_conditions : list or None
        Starting points owned by the caller, one
        {'coordinates', 'velocities'} mapping per trajectory. Only read
        when adopt_initial_conditions is True; supplied without the
        switch it raises instead of being ignored.
    adopt_initial_conditions : bool
        When True the supplied batch is the only source of starting
        points and the initcond keyword / <title>.init.xyz are not read.
        When False (default) the run resolves them exactly as before.
        The batch length is authoritative and overrides ninitcond.
    **kwargs
        Dispatcher parameters (state, multitask, remotetask,
        local_executor, executor_spec).
    """

    name = 'md'
    workflow_kind = 'iterative'
    multitask = False
    remotetask = False

    @classmethod
    def state_kwargs(cls, state):
        """Read the propagation parameters from the ``&md`` section.

        Without this wiring the loop would silently fall back to its
        class defaults (100 steps, 20.67 au, nosh).
        """
        dynamics = getattr(getattr(state, 'pmd', None), 'dynamics', None)
        if dynamics is None:
            return {}
        # The md chain takes its detection thresholds from the &md block;
        # None means the channel is not judged. An adaptive md child has
        # already had these overwritten from &adaptive's max* by the parent.
        thresholds = {
            'err_energy': getattr(dynamics, 'md_maxenergy', None),
            'err_grad': getattr(dynamics, 'md_maxgrad', None),
            'err_nac': getattr(dynamics, 'md_maxnac', None),
            'err_soc': getattr(dynamics, 'md_maxsoc', None),
        }
        thresholds = {key: float(value) for key, value in thresholds.items()
                      if value is not None}
        runtime = getattr(state, 'runtime', None)
        # The tracked geometry is a per-run constant read off the same state.
        # `tracker.plan` is the reader the schema validator calls too, so a
        # programmatic caller gets the same refusal a file does, before the
        # first step. The population tolerance is the md chain's own
        # `md_maxpop`; None turns that guard off.
        system = getattr(state.pmd, 'system', None)
        track = tracker.plan(system)
        tolerance = getattr(dynamics, 'md_maxpop', None)
        tolerance = None if tolerance is None else float(tolerance)
        return {
            'maxiter': int(dynamics.step),
            'dt': float(dynamics.size),
            'sfhp': str(dynamics.sfhp),
            'deviation_thresholds': thresholds,
            'track': track,
            'population_tolerance': tolerance,
            'trace': getattr(runtime, 'analysis_trace', None),
            'bias': getattr(runtime, 'analysis_bias', None),
        }

    def __init__(self, calculator=None, maxiter=100, dt=20.67, sfhp='nosh',
                 initial_conditions=None, adopt_initial_conditions=False,
                 deviation_thresholds=None, trace=None, bias=None,
                 track=None, population_tolerance=None,
                 **kwargs):
        super().__init__(name='md', calculator=calculator, maxiter=maxiter,
                         **kwargs)
        self.dt = float(dt)
        self.sfhp = str(sfhp)
        # Detection thresholds of this run, keyed by the err_* names. Empty
        # means no channel is judged: a QM md run carries none, a plain NN
        # md run gets them from &md's md_max*, and an adaptive child has
        # them overwritten from &adaptive by the parent workflow.
        self.deviation_thresholds = dict(deviation_thresholds or {})
        # Geometry tracking declaration (utilities/tracker.py::plan) and the
        # population tolerance. None means the corresponding guard is off: no
        # declaration, or `check_population 0`.
        self.track = dict(track) if track else None
        self.population_tolerance = (None if population_tolerance is None
                                     else float(population_tolerance))
        # Adaptive-private analysis inputs, stamped on RuntimeState by the
        # adaptive workflow. None keeps every existing path untouched.
        self.trace = trace
        self.bias = bias
        if initial_conditions is not None and not adopt_initial_conditions:
            raise ValueError(
                "initial_conditions was supplied with "
                "adopt_initial_conditions=False; enable the switch or "
                "leave the payload out, the md workflow does not silently "
                "choose between them.")
        self.initial_conditions = initial_conditions
        self.adopt_initial_conditions = bool(adopt_initial_conditions)
        self.output = None
        self._total_steps = 0

    # ---- Runtime directory policy (trajectory fan-out) ----

    def runtime_root(self):
        """Trajectory directories live next to the md run files.

        The parent md runtime writes its log and trajectory outputs
        into the current working directory; every child trajectory
        gets its '{title}_{idx}' directory right beside them instead
        of the shared '<cwd>/runtime' root.

        Returns
        -------
        str
            Parent process working directory.
        """
        return os.getcwd()

    def runtime_name_format(self):
        """Trajectory directory template '{title}_{idx}'.

        The title comes from the md control section (the input file
        title), so the trajectory directories read as
        '<title>_1' ... '<title>_{ninitcond}'.

        Returns
        -------
        str
            Name template for the trajectory directories.
        """
        title = ''
        state = (self.calculator.get_state()
                 if self.calculator is not None else self.state)
        control = getattr(getattr(state, 'pmd', None), 'control', None)
        if control is not None:
            title = str(getattr(control, 'title', '') or '')
        return '%s_{idx}' % (title or 'trajectory')

    def runtime_isolated_workdir(self):
        """Each trajectory directory fully isolates its unit.

        The backend calculation directory (tmp_QDM and friends) and
        the trajectory outputs all land inside '{title}_{idx}/';
        the execution id drops the runtime suffix (the directory is
        already unique).

        Returns
        -------
        bool
            Always True for the md trajectory fan-out.
        """
        return True

    def run(self):
        """Plan and execute the AIMD simulation.

        Worker runtimes (dispatched child trajectories) run their
        single trajectory in place: the initial coordinates/velocities
        and the sub working directory were already injected during
        unit materialization, so no initial-condition resolution or
        re-dispatch happens (recursion guard).

        Root runtimes resolve the initial conditions (&md initcond/
        ninitcond): ninitcond == 1 updates the state in place and
        runs one trajectory; ninitcond > 1 fans the trajectories out
        through self.dispatch(units=batch, child_jobtype='md').

        Returns
        -------
        PyRAIState or TrajectorySet
            The final state for a dispatched worker trajectory
            (wrapped by the parent), or the universal TrajectorySet
            (one record per trajectory, planning-order idx) for root
            runtimes; a single-trajectory run is the
            trajectory-count-1 case.
        """
        state = self.calculator.get_state()
        runtime = getattr(state, 'runtime', None)

        # Worker guard: a dispatched child trajectory carries its
        # injected starting point and sub working directory; run it
        # in place without resolving initial conditions again.
        if runtime is not None and runtime.worker:
            return self._run_trajectory(state)

        if self.adopt_initial_conditions:
            # The caller owns the starting points; the batch length is
            # authoritative and the initcond keyword is not consulted.
            batch = adopt_initial_conditions(
                self.initial_conditions, len(state.atoms))
            self.logger.info(
                "Initial conditions: injected payload (%d trajectory/ies).",
                len(batch))
            adopted = True
        else:
            batch = resolve_initial_conditions(state)
            dynamics = getattr(getattr(state, 'pmd', None), 'dynamics', None)
            self.logger.info(
                "Initial conditions: initcond=%s, ninitcond=%s "
                "(%d trajectory/ies).",
                getattr(dynamics, 'initcond', None),
                getattr(dynamics, 'ninitcond', None), len(batch))
            adopted = False

        if len(batch) == 1:
            # Only an explicit replacement (read/load or an injected
            # payload) updates the starting point; 0/none replicates
            # the state values (no needed update). An injected payload
            # does not even read the initcond keyword.
            if adopted:
                state.update_system(batch[0])
            elif normalize_initcond(state.pmd.dynamics.initcond) in (
                    'read', 'load'):
                state.update_system(batch[0])
            # Single trajectory = trajectory-count-1 case of the
            # universal multi-trajectory return structure.
            return TrajectorySet([(0, self._run_trajectory(state))])

        # Multi-trajectory fan-out: each unit becomes an independent
        # child runtime running child_jobtype 'md' (the worker guard
        # above stops the recursion inside the children).
        if self.local_executor is None:
            self.local_executor = self._execute_child
        outcomes = self.dispatch(units=batch, child_jobtype='md')
        return self.collect(state, outcomes)

    def _execute_child(self, child_state, runtime_idx=None):
        """Local execution entry for one child trajectory.

        Re-enters the unified execute() pipeline with the child
        state; the worker execution graph frozen by the Dispatcher
        guarantees the child never re-plans or re-dispatches.

        Parameters
        ----------
        child_state : PyRAIState
            Materialized child state (worker, injected starting
            point and sub working directory).
        runtime_idx : str or None
            Runtime identity suffix for the backend directories.

        Returns
        -------
        PyRAIState
            Final state of the child trajectory.
        """
        from . import execute
        return execute(child_state, runtime_idx=runtime_idx)

    def collect(self, state, outcomes):
        """Collect the trajectory outcomes of a batch run.

        No aggregation in this pass: the per-trajectory output files
        are already isolated by the sub working directories, so the
        outcomes are wrapped as is into the universal TrajectorySet.
        The dispatch pipeline (pool.map / sequential wrapper)
        returns the outcomes in unit order, which is the planning
        order, so the enumerate position is the trajectory idx.

        Parameters
        ----------
        state : PyRAIState
            Parent state of the batch run.
        outcomes : list
            One outcome per trajectory (final state or Result).

        Returns
        -------
        TrajectorySet
            The trajectory outcomes arranged by the planning idx.
        """
        del state
        return TrajectorySet(
            (idx, outcome) for idx, outcome in enumerate(outcomes))

    def _run_trajectory(self, state):
        """Execute one AIMD trajectory (self-owned iterative loop).

        Parameters
        ----------
        state : PyRAIState
            State with the trajectory starting point; when it carries
            a sub working directory (dispatched batch unit) the
            trajectory output files and the run log are written
            there.

        Returns
        -------
        PyRAIState
            The state with final trajectory and populations.
        """
        # Two different numbers were being carried in one place, which is what
        # made a resumed run disagree with a continuous one. `maxiter` is what
        # this process was asked to integrate - on a restart, the steps to add
        # - while the checkpoint says where the run stands. The step number the
        # record carries is absolute from here on (D-19, D-24), so the frame
        # written at step 7 says 7 whichever process wrote it, and the frame
        # cadence is decided on that number instead of on a per-process count.
        started = int(getattr(state.runtime, 'current_step', 0) or 0)
        total_steps = started + self.maxiter
        if state.runtime is not None:
            state.runtime.total_steps = total_steps
            # A segment that ends on its planned step carries the normal code
            # 0; the guard overwrites it when it ends the trajectory early.
            state.runtime.stop_code = 0
        self._total_steps = total_steps

        # Reproducible hopping sequences: seed the generator from
        # gl_seed plus the trajectory index before any random draw
        # (the surface hopping integrator consumes the stream).
        _seed_trajectory(state)

        # Initialize the surface hopping algorithm once (config
        # parsing, kernel selection and JIT warm-up before the loop;
        # 'nosh' returns a zero-cost pass-through).
        surfacehop = surface_hopping(self.sfhp, state)

        # Trajectory output writer (legacy-format md.*/sh.* files);
        # a sub working directory isolates the output of a dispatched
        # batch unit (the run log follows the same directory).
        logger = getattr(state, 'logger', None)
        dynamics = state.pmd.dynamics
        workdir = getattr(state.runtime, 'workdir', '') \
            if state.runtime is not None else ''
        if workdir and logger is not None and hasattr(logger, 'log_dir'):
            logger.log_dir = workdir
            title = getattr(logger, 'title', state.pmd.control.title)
            logger.log_file = os.path.join(workdir, '%s.log' % title)
        self.output = MdOutputWriter(
            title=state.pmd.control.title,
            log_dir=workdir or getattr(logger, 'log_dir', None),
            direct=dynamics.direct,
            buffer=dynamics.buffer,
            dt=self.dt,
            total_steps=total_steps,
            center_regions=_center_regions(state),
            # Charge management (ONIOM embedding / read_charge) emits the
            # md.charges/sh.charges files; a plain run writes none.
            charge_output=getattr(state, 'charge_store', None) is not None)
        # Wipe the leftovers of a previous run so the append-mode records
        # never stack across repeated executions - but only for a run that
        # starts at step 0. A state restored from `<title>.rt` carries the
        # accepted step, and wiping there would delete the frames the first
        # segment already wrote, so the resumed product would not be the
        # continuous trajectory the checkpoint exists to produce.
        resumed = int(getattr(state.runtime, 'current_step', 0) or 0) > 0
        if not resumed:
            self.output.reset()
            self.output.write_headers()

        # Per-phase wall clock of the loop below. The resident engine has
        # reported its own phase table for a while; this is the other
        # half, without which the difference between the engine's
        # milliseconds and the run's milliseconds has to be guessed at.
        # The table is a test instrument: `perf_profile` defaults to 0,
        # and when it is off no phase timer runs and no file is written.
        profile_enabled = bool(int(getattr(dynamics, 'perf_profile', 0) or 0))
        profiler = PhaseProfiler() if profile_enabled else None
        self._profiler = profiler
        profile_dir = workdir or os.getcwd()
        loop_started = time.perf_counter()

        # The constraint reference is captured here, before any atom has
        # moved: a bond length bound inside the loop would be bound to whatever
        # the first step drifted to. An empty plan returns immediately, so an
        # unconstrained run pays nothing (dev/methods/
        # CONSTRAINT_DEVELOPMENT.md section 8).
        constraint_runtime.bind(state)

        # The initial-velocity control is a one-time declaration, and only for
        # a process that starts at step 0: a resumed segment carries the kinetic
        # energy its predecessor already settled into, and scaling that again
        # would make the split run state something the continuous one never
        # did. It runs after the bind because a frozen component has no kinetic
        # energy to count, and the mask that says which are frozen is the plan.
        if started == 0:
            state = velocity_initialize(state)

        # Absolute step numbers: the first one this process integrates is
        # `started + 1`, which is also the number its record will carry.
        for step in range(started + 1, total_steps + 1):
            # 0. Extended-ensemble pre half step (dev/methods/
            #    ENSEMBLE_DEVELOPMENT.md section 2): the Nose-Hoover chain
            #    half step and the barostat momentum half kick, the
            #    latter reusing the previous frame's stress. A no-op
            #    for NVE (and for NVT/NPT with no extended-cell
            #    barostat at this stage), so the split Verlet below
            #    reduces to the classic merged form.
            with self._phase('ensemble_pre'):
                state = apply_ensemble(state, stage='pre')

            # 1. Propagate coordinates to the current structure;
            #    internally a no-op when no previous step is
            #    recorded (first step keeps the initial structure).
            with self._phase('verlet_i'):
                state = verlet_i(state, dt=self.dt)
            with self._phase('apply_pbc'):
                state = apply_pbc(state)

            # 2. Evaluate the properties at the current structure
            #    (single calculator call writes results to pmd).
            #    target_states is derived on every step from the
            #    activestate flag and the current (post-hop) active
            #    configuration; the backend only sees the state list
            #    to compute (one list per center for a multi-center
            #    run, routed by the calculator).
            with self._phase('target_states'):
                target_states = resolve_target_states_centered(state)
            with self._phase('calculator_run'):
                try:
                    result = self.calculator.run(target_states=target_states)
                except Exception as error:
                    # The legacy stop=2: the calculation that this step needs
                    # never produced a result. It stays an exception (a run
                    # without a property is a failure, not a completed
                    # trajectory), but the code is stamped on the runtime and
                    # the run-log tail names it so the reason survives the
                    # traceback.
                    state.runtime.stop_code = 4
                    state.runtime.mark_failed(
                        'calculation failed at step %d: %s: %s'
                        % (step, type(error).__name__, error))
                    raise

            # 2b. Receive the NN prediction deviations from the
            #     backend result (tolerant fetch: absent keys are
            #     filled with 0.0); the framework never computes
            #     them. The merged keys travel with the history
            #     snapshots for adaptive sampling.
            with self._phase('merge_deviations'):
                merge_deviations(state.pmd.results,
                                 fetch_deviations(result))

            # 3. Complete the current-step velocity; internally a
            #    no-op on the first step (no previous gradient).
            with self._phase('verlet_ii'):
                state = verlet_ii(state, dt=self.dt)
            with self._phase('phase_correction'):
                state = phase_correction(state)
            with self._phase('apply_constrain'):
                state = apply_constrain(state)
            # The constraint history accepts the step here, after the
            # projection has fixed the geometry and before the next force is
            # asked for: a hill belongs to an accepted step, and a replayed
            # step must not deposit a second one (dev/methods/
            # CONSTRAINT_DEVELOPMENT.md section 8). A plan without a history
            # action returns immediately.
            with self._phase('constraint_commit'):
                constraint_runtime.commit(state)
            with self._phase('velocity_correction'):
                state = velocity_correction(state, step)
            # Extended-ensemble post half step: the Nose-Hoover chain
            #    half step and the barostat momentum half kick with the
            #    freshly evaluated stress (and CSVR / Berendsen, which
            #    act once per step here). A no-op for NVE.
            with self._phase('ensemble_post'):
                state = apply_ensemble(state, stage='post')
            # The barostat rescale is not a rigid motion: it scales the
            # coordinates with the cell, which takes the geometry the
            # projection just produced off the constraint manifold by one
            # coupling step's worth - a held bond by dt/tau of its length and
            # a frozen atom by the same fraction of its distance from the
            # origin. Re-asking the geometry question after the rescale is what
            # makes the recorded step the one the constraints declare. The
            # entry touches the velocities as it always has (the frozen
            # components and the bond tangents), and for a state that the first
            # call already projected that part is a no-op; what changes is that
            # the positions are solved against the cell this step ends with.
            # A run with no plan gets the same object back, so an unconstrained
            # step pays nothing. See dev/methods/CONSTRAINT_DEVELOPMENT.md
            # section 26.
            with self._phase('constrain_after_cell'):
                state = apply_constrain(state)

            # The step's kinetic energy is sampled once, here, from the
            # velocities this step ends with - after the centre-of-mass
            # correction, the ensemble post half step and the second
            # constraint projection, each of which writes new velocities.
            # Everything that reports the step reads this one value: the run
            # log's `&kinetic energy`, the frame record's Ekin column, the
            # history frame, and the surface-hopping acceptance test that
            # runs below. Sampling earlier left the log describing a state
            # its own printed velocities could not reproduce.
            with self._phase('kinetic_energy'):
                state.kinetic_energy()

            # The tracked geometric parameters are read on the geometry this
            # step ends with - after the second projection and the barostat
            # rescale - and before the frame is built, so the reading travels
            # with the step it describes. The stop decision is taken here but
            # acted on after the checkpoint below, which is what keeps the
            # offending frame in the trajectory (the legacy GeomTracker wrote
            # its frame and then ended the run). PBC runs measure the
            # continuous coordinates, because a pair that straddles a face
            # would otherwise read as one cell length.
            with self._phase('geometry_track'):
                track_rows = tracker.evaluate(
                    self.track,
                    continuous_coordinates(state.coordinates,
                                           state.pmd.system))
                if track_rows is not None:
                    state.pmd.results['geometry_track'] = track_rows
            track_hit = bool(self.track and tracker.exceeded(track_rows))

            # 4. Save the fully computed step (R, V, G and the
            #    pre-hop active state) to the history.
            with self._phase('update_history'):
                state.update_history()
            if self.trace is not None:
                self._record_analysis_trace(state)

            # 5. Surface hopping updates the active state itself; sync
            #    pushes the updated state to the task configs. The
            #    recorded step is watched at the checkpoint below, not
            #    here: a check that fires before the counter advances
            #    reports a step number that no record carries.
            #
            #    sync() runs AFTER this step's force evaluation, so it
            #    is NOT the channel that feeds a variable-cell backend
            #    the deformed cell of the current step. That is the job
            #    of the per-evaluation cell snapshot frozen inside
            #    Calculator.run() (calculator._snapshot_cell -> the
            #    work item -> backend.calculate(cell=...)), which is
            #    captured at the same moment as the geometry and thus
            #    always matches this step's propagated cell. sync()
            #    only refreshes the task configs for the NEXT step's
            #    backend rebuild / checkpoint (dev/methods/
            #    PBC_DEVELOPMENT.md section 5.3).
            with self._phase('surfacehop'):
                state = surfacehop(state)
                # Stamped on the step the hop decision has just been made, so
                # the dissipate delay counts from the state the trajectory
                # actually ends this step on, not the one it started on.
                _record_dissipate_landing(state, step)
            # A population outside the declared tolerance ends the trajectory,
            # as legacy did. The reading is this step's electronic propagation
            # and hop decision, which is the density the integrators persist in
            # system.state_density; a run without one (nosh) has nothing to
            # judge. The stop itself is acted on after the checkpoint below.
            with self._phase('population_check'):
                population_reason = self._population_violation(state)
            with self._phase('sync'):
                self.calculator.sync()

            # 6. Nothing is written here any more. The trajectory record, the
            #    serialization file and the runtime metadata are written at
            #    phase 7 below, after the step counter has advanced, so all
            #    three describe the step that has just finished. Their phases
            #    are timed inside checkpoint(), which keeps the rows of the
            #    phase table from overlapping and their total equal to the step.

            # 7. Advance the step counter before anything is written, so the
            #    checkpoint, the trajectory record and the run log all describe
            #    the step that has just finished. A checkpoint taken before this
            #    line carried one fewer, and the resumed process then re-used a
            #    step number the previous process had already recorded - the
            #    duplicated time label and the extra frames (D-19, D-24).
            state.runtime.advance_step()
            state.runtime.simulation_time = (step - 1) * self.dt
            # Resident guards: the md_max* thresholds and the two physical
            # bounds terminate the trajectory together. The offending frame is
            # checkpointed as the segment's final frame before the loop leaves
            # it. No threshold (a QM run, or a NN run that declared none)
            # leaves the deviation channels unjudged; a backend reporting no
            # deviations fills them with 0.0, so such a run never trips that
            # guard. The geometry reading and the population reading were
            # taken earlier in the step (before the hop and after it), so this
            # is only where the three decisions are assembled.
            stop_keys = self._stop_deviations(state)
            reasons = []
            if track_hit and self.track and int(self.track.get('stop', 0)) == 1:
                reasons.append(tracker.stop_reason(track_rows))
            if population_reason:
                reasons.append(population_reason)
            if stop_keys:
                reasons.append('%s over the max threshold'
                               % ', '.join(stop_keys))
            reasons = [reason for reason in reasons if reason]
            # The checkpoint override in tests/subclasses keeps its
            # one-argument signature; the guards' final-frame request
            # travels on the instance instead of the call.
            self._stop_final = bool(reasons)
            self.checkpoint(state)
            self._stop_final = False
            if reasons:
                # The legacy early-stop vocabulary, extended with a normal
                # code: 1 ill population, 2 dissociation (a tracked geometry
                # bound), 3 prediction error above the md_max* thresholds. A
                # step that violates more than one reports the first of that
                # order, while the sentence below still names every reason.
                code = 0
                if population_reason:
                    code = 1
                elif track_hit and self.track \
                        and int(self.track.get('stop', 0)) == 1:
                    code = 2
                elif stop_keys:
                    code = 3
                state.runtime.stop_code = code
                message = ("MD stopped at step %d: %s"
                           % (step, '; '.join(reasons)))
                logger = getattr(state, 'logger', None)
                if logger is not None:
                    logger.write('\n' + message + '\n')
                else:
                    self.logger.info(message)
                break

        # The table is written once, at the end of the trajectory: it is the
        # run's overall phase cost, not a per-step trace. The divisor is
        # the steps this process actually ran, not the planned length: an
        # early-stopped trajectory must not report its phases spread over
        # steps it never evaluated. A run with `perf_profile: 0` has no
        # profiler and writes nothing.
        executed = max(int(getattr(state.runtime, 'current_step', 0) or 0)
                       - started, 0)
        if profiler is not None and executed > 0:
            profiler.mark_wall(time.perf_counter() - loop_started)
            profiler.refresh(profile_dir, executed,
                             extra=_nested_profile(self.calculator,
                                                   executed))
        self._profiler = None

        # One closing line per monitor that fired. A warn-only run
        # (control.debug false) otherwise leaves its findings scattered
        # through the step blocks, where "nothing was reported" and
        # "nothing was watched" look the same to a reader.
        diagnostics_summary(state)

        # The trajectory owns its resident/diagnostic backends: release
        # them before the worker process can be reused for the next unit.
        release = getattr(self.calculator, 'close', None)
        if callable(release):
            release()

        state.runtime.mark_completed()
        return state

    def checkpoint(self, state, final=False):
        """Write everything that names the step that just finished.

        Saves in order:
        1. the trajectory output files (md.* per step under the
           direct/buffer control, sh.* on hopping events) and the
           run-log step block, both written from the latest history
           snapshot (state.get_history_step(-1)); the live pmd
           config is NOT used because surface hopping mutates
           current_state after the step was recorded while the
           computed properties stay unchanged;
        2. the PyRAIState serialization file;
        3. the runtime metadata record via the official
           state.record_step_result() interface.

        Parameters
        ----------
        state : PyRAIState
            State with the fully computed and recorded timestep.
        final : bool
            Treat this step as the segment's final step even when it is
            not the planned last one (the uncertainty guard's frame).

        Returns
        -------
        PyRAIState
            The checkpointed state.
        """
        # 1. Write the trajectory output files from the recorded
        #    snapshot; a decisive gate result also controls the
        #    run-log step block (None keeps the previous behavior,
        #    e.g. when the writer or the snapshot is unavailable in
        #    unit tests and worker contexts).
        with self._phase('snapshot_lookup'):
            # The writer and the run-log formatter read the frame and
            # never write it, so it travels by reference; copying it here
            # is one whole frame dump per step for nothing.
            snapshot = state.get_history_step(-1, copy=False)
        # The loop advanced the counter before calling this, so the number the
        # record carries is the step that just finished - absolute, whether this
        # is the first process of a run or the third.
        step = (state.runtime.current_step
                if state.runtime is not None else 0)
        final = (bool(final) or bool(getattr(self, '_stop_final', False))
                 or (bool(self._total_steps)
                     and step >= self._total_steps))
        # A step that leaves a checkpoint also leaves a trajectory frame.
        # Without that, the last step of every segment forces a frame the
        # continuous run never writes, and the split product has one frame per
        # segment more than the single-process one. With it, the frame set is a
        # function of the absolute step number and the checkpoint cadence
        # alone, which is the only information a resumed process has (D-24).
        with self._phase('output_record'):
            gate = (self.output.record(
                step, snapshot, state,
                final=final or self._runtime_file_due(state, step, final))
                if self.output is not None else None)
            # A step the run writes to is a step a reader can hold accountable,
            # so these four questions are asked of the frame on exactly the
            # cadence the trajectory record keeps - never on a step the run
            # skips, and never off-budget, because this row of the phase table
            # is where their time lands.
            if gate is True and isinstance(snapshot, dict):
                check_recorded_step(state, snapshot)
                verify_recorded(state, snapshot)
        logger = getattr(state, 'logger', None)
        with self._phase('log_step'):
            if gate is not False and logger is not None and snapshot is not None:
                logger.log_step(step, snapshot,
                                thresholds=self.deviation_thresholds)

        # 2. The PyRAIState serialization file, when the run asks for
        #    checkpoints. `md.checkpoint` names the cadence in steps and 0 - the
        #    default - writes nothing, so no existing run gains a file or a byte
        #    of I/O. The last step always writes: a run that ends between two
        #    cadence points would otherwise leave a file describing an older
        #    geometry than its own trajectory. The constraint plan and history
        #    travel inside that file, which is what lets a resumed run keep the
        #    targets the first structure bound instead of re-capturing them from
        #    the resumed one.
        self._write_runtime_file(state, step, final)
        with self._phase('record_step_result'):
            state.record_step_result()
        return state

    def _runtime_file_due(self, state, step, final):
        """Whether this step is one the run writes a `<title>.rt` at.

        Parameters
        ----------
        state : PyRAIState
            State carrying the md dynamics section.
        step : int
            Absolute step number.
        final : bool
            Whether this step ends the run.

        Returns
        -------
        bool
            True when the serialization file is written at this step.
        """
        dynamics = getattr(getattr(state, 'pmd', None), 'dynamics', None)
        every = int(getattr(dynamics, 'checkpoint', 0) or 0)
        if every < 1:
            return False
        return bool(final) or step % every == 0

    def _write_runtime_file(self, state, step, final):
        """Write `<title>.rt` on the cadence the run declares.

        Returns
        -------
        str or None
            The path written, or None when the run asks for no checkpoints.
        """
        if not self._runtime_file_due(state, step, final):
            return None
        from ..data.runtime.serializer import save_state

        runtime = getattr(state, 'runtime', None)
        # The stream position is stamped on the way out, at the same moment
        # the geometry and the history are written, so the file describes one
        # step and not a step plus whatever draws happened after it.
        if runtime is not None:
            runtime.random_state = np.random.get_state()
        workdir = getattr(runtime, 'workdir', '') or os.getcwd()
        control = getattr(getattr(state, 'pmd', None), 'control', None)
        title = str(getattr(control, 'title', '') or 'pyrai3md')
        path = os.path.join(workdir, '%s.rt' % title)
        with self._phase('runtime_file'):
            save_state(state, path,
                       runtime_idx=getattr(runtime, 'runtime_idx', None))
        return path

    def _phase(self, name):
        """Time one checkpoint phase, when a step profile is running.

        ``checkpoint()`` is also called directly by unit tests and worker
        contexts that never enter the step loop, so the profiler may be
        absent and the block must cost nothing rather than raise.

        Parameters
        ----------
        name : str
            Phase label of the step table.

        Returns
        -------
        context manager
            The profiler's timer, or a no-op when unattached.
        """
        profiler = getattr(self, '_profiler', None)
        return profiler.measure(name) if profiler is not None else nullcontext()

    def _record_analysis_trace(self, state):
        """Append one scalar trace row for the adaptive analysis layer.

        The trace rides RuntimeState.analysis_trace, so it costs nothing
        when no analysis switch asked for it. A row carries only the
        quantities the requested spec names; the unbiased energies are the
        ones the prediction produced, so the gap analysis never sees a
        biased potential even when an exploration mode is on.
        """
        runtime = getattr(state, 'runtime', None)
        payload = (getattr(runtime, 'analysis_trace', None)
                   if runtime is not None else None)
        if not isinstance(payload, dict):
            payload = self.trace
        if not isinstance(payload, dict):
            return
        spec = payload.get('spec') or {}
        pmd = getattr(state, 'pmd', None)
        results = getattr(pmd, 'results', None) or {}
        system = getattr(pmd, 'system', None)
        row = {
            'step': int(getattr(runtime, 'current_step', 0) or 0),
            'time': float(getattr(runtime, 'simulation_time', 0.0) or 0.0),
        }
        # The analysis reads the unbiased energies; the dynamics (and the
        # energy-drift bookkeeping) reads the biased ones when a potential
        # mode is on.
        energies = results.get('unbiased_energy')
        if energies is None:
            energies = results.get('energy')
        if energies is not None:
            row['energy'] = np.asarray(
                energies, dtype=np.float64).reshape(-1).tolist()
        scope = spec.get('gap')
        if scope and row.get('energy'):
            pairs = gap_pairs(len(row['energy']), scope)
            if pairs:
                lowest = min(
                    (row['energy'][upper] - row['energy'][lower],
                     lower, upper)
                    for lower, upper in pairs)
                row['gap'] = {
                    'min': float(lowest[0]),
                    'pair': [int(lowest[1]), int(lowest[2])],
                    'step': row['step'],
                }
        if spec.get('force'):
            gradient = results.get('gradient')
            if gradient is not None:
                row['max_force'] = float(np.max(np.abs(
                    np.asarray(gradient, dtype=np.float64))))
        if spec.get('distance'):
            coordinates = getattr(system, 'coordinates', None)
            if coordinates is not None:
                positions = np.asarray(coordinates, dtype=np.float64)
                if (positions.ndim == 2 and positions.shape[1] == 3
                        and positions.shape[0] <= 200):
                    left, right = np.triu_indices(positions.shape[0], 1)
                    distances = np.linalg.norm(
                        positions[left] - positions[right], axis=1)
                    if distances.size:
                        row['min_distance'] = float(np.min(distances))
                elif positions.ndim == 2:
                    row['min_distance_status'] = (
                        'not_applicable: the trace skips the O(natom^2) '
                        'distance above 200 atoms')
        if spec.get('population'):
            raw = getattr(system, 'state_density', None)
            if raw:
                array = np.asarray(raw, dtype=np.float64)
                if array.ndim == 3 and array.shape[-1] == 2:
                    array = array[..., 0]
                if array.ndim == 2 and array.shape[0] == array.shape[1]:
                    row['population'] = [float(value)
                                         for value in np.diag(array)]
        if spec.get('energy_drift') and row.get('energy'):
            active = int(getattr(system, 'current_state', 0) or 0)
            if 0 <= active < len(row['energy']):
                dynamics = results.get('energy')
                if dynamics is None:
                    dynamics = energies
                dynamics = np.asarray(
                    dynamics, dtype=np.float64).reshape(-1)
                total = float(dynamics[active]) + float(
                    getattr(runtime, 'kinetic_energy', 0.0) or 0.0)
                reference = payload.get('reference_total')
                if reference is None:
                    payload['reference_total'] = total
                    reference = total
                row['energy_drift'] = total - float(reference)
        payload.setdefault('rows', []).append(row)

    def _stop_deviations(self, state):
        """Deviation keys over their threshold of this run.

        The resident uncertainty guard of the trajectory loop. The
        judgment rule is legacy _chkerror: error > threshold. Empty
        thresholds (a QM md run, or a NN run that declared none) leave
        every channel unjudged; an absent deviation is filled with 0.0
        by merge_deviations(), so a backend that reports none never
        trips the guard.
        """
        if not self.deviation_thresholds:
            return []
        judged = judge_deviations(
            state.pmd.results, self.deviation_thresholds)
        return [key for key, (_error, _threshold, exceeded)
                in judged.items() if exceeded]

    def _population_violation(self, state):
        """The first population outside the tolerance, or None.

        The reading is the real diagonal of the density matrix the step's
        electronic propagation and hop decision left behind, which is what the
        integrators persist in `system.state_density`. A run without a density
        (nosh, or an electronic method that never filled one) has nothing to
        judge and passes; `population_bad` treats a non-finite value as bad,
        so a NaN density stops the run instead of propagating silently.

        Parameters
        ----------
        state : PyRAIState
            State after this step's surface hopping.

        Returns
        -------
        str or None
            The run-log sentence naming the offending state and value, or
            None when the guard is off or every population is in range.
        """
        if self.population_tolerance is None:
            return None
        system = getattr(getattr(state, 'pmd', None), 'system', None)
        populations = state_population(
            {'state_density': getattr(system, 'state_density', None)})
        if populations is None:
            return None
        tolerance = float(self.population_tolerance)
        if not population_bad(populations, tolerance):
            return None
        for number, value in enumerate(populations, start=1):
            if not np.isfinite(value):
                return ('state %d population is not finite (%r)'
                        % (number, float(value)))
            if value < -tolerance or value > 1.0 + tolerance:
                return ('state %d population %.6f is outside [%+.4f, %.4f]'
                        % (number, float(value), -tolerance, 1.0 + tolerance))
        return None

    def __repr__(self):
        return "AIMD(maxiter=%d, sfhp='%s')" % (self.maxiter, self.sfhp)
