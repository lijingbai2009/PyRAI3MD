"""
======================================================================
This module provides RuntimeState, tracking the progress and status
of a PyRAI3MD calculation. It supports checkpoint and restart by
recording the current step, simulation time, and other runtime
metadata. The active electronic state is NOT tracked here; it lives
in pmd.system.current_state (sole authoritative source).

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import copy

# Backend roles a calculation can be dispatched for. Kept beside the
# runtime record because a role now travels on RuntimeState; the
# Calculator and the Dispatcher both import this single vocabulary.
BACKEND_ROLES = ('theory', 'reference')


def validate_backend_role(role):
    """Return a backend role after checking it against BACKEND_ROLES.

    Parameters
    ----------
    role : str
        Requested backend role.

    Returns
    -------
    str
        The same role.

    Raises
    ------
    ValueError
        If the role is not one of BACKEND_ROLES.
    """
    role = str(role)
    if role not in BACKEND_ROLES:
        raise ValueError(
            "Unknown backend role '%s'; expected one of %s."
            % (role, BACKEND_ROLES))
    return role


class RuntimeState:
    """Runtime calculation status for checkpoint and restart.

    Tracks the current progress of a calculation so it can be
    saved to disk and resumed later.

    Parameters
    ----------
    status : str
        Calculation status: 'initialized', 'running', 'completed', 'failed'.
    current_step : int
        Current MD or optimization step.
    total_steps : int
        Total number of steps planned.
    simulation_time : float
        Accumulated simulation time in femtoseconds.
    iteration : int
        General iteration counter.
    random_seed : int
        Random seed for reproducibility.
    checkpoint_file : str
        Path to the last checkpoint file.
    wall_time : float
        Total wall clock time in seconds.
    error_message : str
        Error description if status is 'failed'.
    worker : bool
        Whether this runtime is a dispatched worker (universal rule
        1). Root runtimes are not workers (False); child runtimes
        created by the Dispatcher are stamped as workers (True) and
        never re-plan or dispatch: their execution graph is frozen
        by the parent into state.execution_graph.
    kinetic_energy : float
        Nuclear kinetic energy of the last recorded step in Hartree.
        Computed and stored by PyRAIState.kinetic_energy().
    workdir : str
        Sub working directory allocated to a dispatched batch unit
        (created by the Dispatcher when the batch carries more than
        one unit); empty string for single-task runtimes.
    isolated_workdir : bool
        Whether the allocated workdir fully isolates this unit: the
        backend calculation directories are placed inside the workdir
        and compose_execution_id() drops the '_runtime_{idx}' suffix
        (the directory is already unique). Stamped by the Dispatcher
        for isolated-workdir subjects; travels with the serialized
        state into subprocess/slurm children.
    ensemble : dict
        Transient per-step ensemble diagnostics collected by
        workflow.integrator.ensemble.apply_ensemble (thermostat /
        barostat intermediates: scale factors, masses, pressure
        tensors, ...). Reset at the pre stage of every step and
        consumed by the trajectory writer (md.ensemble) and the
        run-log '&ensemble' section; empty for nve runs.
    diagnostics : dict
        Run-health findings collected by workflow.records.diagnostics,
        keyed by check id: {'count', 'first_step', 'last_step', 'worst',
        'aborted'}. Unlike 'ensemble' this is cumulative over the run and
        is never reset, because its whole purpose is to say how often a
        check fired and when it first did. It survives to_dict/from_dict
        and deep-copies, so a dispatched child's findings reach the parent
        record instead of dying with the child's state.
    dissipate_step : int
        The absolute MD step at which this trajectory first arrived on the
        electronic ground state, 0 while it has not. `&md dissipate` keeps the
        thermostat off until the run has been on the ground state for
        `thermodelay` steps, and it counts from this number; a value already
        set is never moved back, so a hop up does not switch the thermostat
        off again and a restarted segment does not restart the delay.
    random_state : tuple or None
        The NumPy global generator's state, as `np.random.get_state()`
        returns it. `AIMD` stamps it on the steps it writes a runtime file,
        and a resumed trajectory continues its surface-hopping random stream
        from it instead of replaying the seed. None means no record: a run
        that starts afresh, and every runtime file written before the field
        existed, both seed from `control.gl_seed`.
    active_role : str
        Backend role this runtime executes ('theory' or 'reference'),
        stamped by the Dispatcher when a batch unit is materialized with
        an explicit role. It is what lets a reference calculation use
        the same child-runtime fan-out as a theory one. Roots and every
        runtime file written before the field existed read as 'theory'.
    stop_code : int
        Why the trajectory loop ended, in the legacy vocabulary extended
        with a normal-completion code: 0 ran to its planned end, 1 state
        population ill (outside the tolerance or non-finite), 2
        dissociation (a tracked geometry bound was crossed), 3 prediction
        error above the `md_max*` thresholds, 4 the calculation failed
        (legacy stop=2; the exception still propagates, the code is what
        names it). Stamped by `AIMD`, read by the adaptive screen to decide
        whether the stopping structure goes to the reference calculation.
    """

    # Class-level defaults: a `.rt` written before these fields existed is
    # unpickled without running __init__, so instances must not depend on an
    # instance attribute being present.
    active_role = 'theory'
    stop_code = 0
    # Adaptive-private analysis payloads. None means the feature is off and
    # the field is omitted from to_dict()/result.json/.rt entirely.
    analysis_trace = None
    analysis_bias = None

    def __init__(self, status='initialized', current_step=0, total_steps=0,
                 simulation_time=0.0,
                 iteration=0, random_seed=0, checkpoint_file='',
                 wall_time=0.0, error_message='', worker=False,
                 kinetic_energy=0.0, workdir='', isolated_workdir=False,
                 dissipate_step=0, random_state=None,
                 active_role='theory', analysis_trace=None,
                 analysis_bias=None, stop_code=0):
        self.status = status
        self.current_step = int(current_step)
        self.total_steps = int(total_steps)
        self.simulation_time = float(simulation_time)
        self.iteration = int(iteration)
        self.random_seed = int(random_seed)
        self.checkpoint_file = str(checkpoint_file)
        self.wall_time = float(wall_time)
        self.error_message = str(error_message)
        self.worker = bool(worker)
        self.kinetic_energy = float(kinetic_energy)
        self.workdir = str(workdir)
        self.isolated_workdir = bool(isolated_workdir)
        # The absolute step at which this trajectory first arrived on the
        # ground state, or 0 while it has not. `&md dissipate` counts its
        # `thermodelay` from here, and the value is a step number rather than
        # a countdown so that a restarted segment keeps the delay its
        # predecessor had already run down.
        self.dissipate_step = int(dissipate_step)
        # The NumPy global generator's state, carried so a resumed segment
        # continues the surface-hopping random stream instead of replaying it
        # from the seed. Stamped by `AIMD._write_runtime_file` on the steps
        # the run writes `<title>.rt`; read back by `_seed_trajectory` only for
        # a state that already has steps behind it. None means "no record",
        # which is what every runtime file written before this field existed
        # carries, and such a run seeds from `control.gl_seed` as before.
        self.random_state = random_state
        # Backend role of this runtime ('theory' or 'reference'). The
        # value is recorded rather than validated here; the Calculator
        # is the owner of the role vocabulary.
        self.active_role = str(active_role or 'theory')
        # Why the md loop ended (0 planned end, 1 ill population, 2
        # dissociation, 3 prediction error). The adaptive screen reads it to
        # decide whether the stopping structure is collected.
        self.stop_code = int(stop_code)
        # Adaptive-private trace/bias payloads; None keeps the serialized
        # document byte-identical to a run that never asked for them.
        self.analysis_trace = analysis_trace
        self.analysis_bias = analysis_bias

        # Per-step calculation result tracking
        self.step_results = []

        # Per-step ensemble diagnostics (see the class docstring)
        self.ensemble = {}

        # Cumulative run-health findings (see the class docstring)
        self.diagnostics = {}

    def advance_step(self):
        """Advance the step counter by one.

        Counting only: no physical time is accumulated here. The
        simulation time is derived by the output functions from the
        step counter and the timestep.
        """
        self.current_step += 1
        self.iteration += 1
        if self.status == 'initialized':
            self.status = 'running'

    def mark_completed(self):
        """Mark the calculation as completed."""
        self.status = 'completed'

    def mark_failed(self, message=''):
        """Mark the calculation as failed.

        Parameters
        ----------
        message : str
            Error description.
        """
        self.status = 'failed'
        self.error_message = message

    def to_dict(self):
        """Serialize to a dictionary.

        Returns
        -------
        dict
            Dictionary representation.
        """
        return {
            'status': self.status,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'simulation_time': self.simulation_time,
            'iteration': self.iteration,
            'random_seed': self.random_seed,
            'checkpoint_file': self.checkpoint_file,
            'wall_time': self.wall_time,
            'error_message': self.error_message,
            'worker': self.worker,
            'kinetic_energy': self.kinetic_energy,
            'workdir': self.workdir,
            'isolated_workdir': self.isolated_workdir,
            'dissipate_step': self.dissipate_step,
            'random_state': self.random_state,
            'active_role': self.active_role,
            'stop_code': self.stop_code,
            'step_results': list(self.step_results),
            'ensemble': dict(self.ensemble),
            'diagnostics': copy.deepcopy(self.diagnostics),
            **({'analysis_trace': copy.deepcopy(self.analysis_trace)}
               if self.analysis_trace is not None else {}),
            **({'analysis_bias': copy.deepcopy(self.analysis_bias)}
               if self.analysis_bias is not None else {}),
        }

    @classmethod
    def from_dict(cls, data):
        """Construct a RuntimeState from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with keys matching constructor arguments.

        Returns
        -------
        RuntimeState
            New instance.
        """
        # Legacy runtimes may carry the pre-refactor 'dispatchable'
        # field (root runtimes were dispatchable=True): map it onto
        # the worker label so old serialized states stay readable.
        if 'worker' in data:
            worker = bool(data.get('worker', False))
        else:
            worker = not bool(data.get('dispatchable', True))
        state = cls(
            status=data.get('status', 'initialized'),
            current_step=data.get('current_step', 0),
            total_steps=data.get('total_steps', 0),
            simulation_time=data.get('simulation_time', 0.0),
            # Legacy runtimes may still carry 'current_electronic_state';
            # it is ignored (pmd.system.current_state is authoritative).
            iteration=data.get('iteration', 0),
            random_seed=data.get('random_seed', 0),
            checkpoint_file=data.get('checkpoint_file', ''),
            wall_time=data.get('wall_time', 0.0),
            error_message=data.get('error_message', ''),
            worker=worker,
            kinetic_energy=data.get('kinetic_energy', 0.0),
            workdir=data.get('workdir', ''),
            isolated_workdir=data.get('isolated_workdir', False),
            dissipate_step=data.get('dissipate_step', 0),
            random_state=data.get('random_state'),
            active_role=data.get('active_role') or 'theory',
            analysis_trace=data.get('analysis_trace'),
            analysis_bias=data.get('analysis_bias'),
            stop_code=data.get('stop_code', 0),
        )
        state.step_results = list(data.get('step_results', []))
        ensemble = data.get('ensemble', {})
        state.ensemble = dict(ensemble) if isinstance(ensemble, dict) else {}
        diagnostics = data.get('diagnostics', {})
        state.diagnostics = (copy.deepcopy(diagnostics)
                             if isinstance(diagnostics, dict) else {})
        return state

    def copy(self):
        """Create a deep copy.

        Returns
        -------
        RuntimeState
            Independent copy.
        """
        new = RuntimeState(
            status=self.status,
            current_step=self.current_step,
            total_steps=self.total_steps,
            simulation_time=self.simulation_time,
            iteration=self.iteration,
            random_seed=self.random_seed,
            checkpoint_file=self.checkpoint_file,
            wall_time=self.wall_time,
            error_message=self.error_message,
            worker=self.worker,
            kinetic_energy=self.kinetic_energy,
            workdir=self.workdir,
            isolated_workdir=self.isolated_workdir,
            dissipate_step=self.dissipate_step,
            random_state=self.random_state,
            active_role=self.active_role,
            analysis_trace=copy.deepcopy(self.analysis_trace),
            analysis_bias=copy.deepcopy(self.analysis_bias),
        )
        new.step_results = list(self.step_results)
        new.ensemble = dict(self.ensemble)
        new.diagnostics = copy.deepcopy(self.diagnostics)
        return new

    def __repr__(self):
        return "RuntimeState(status='%s', step=%d/%d)" % (
            self.status, self.current_step, self.total_steps)

    def record_step_result(self):
        """Record the runtime metadata of the current step.

        Runtime tracks execution metadata only (the step counter);
        property data (energy, gradient, ...) is kept in pmd and
        never duplicated here. Called through the official
        PyRAIState.record_step_result() interface.
        """
        self.step_results.append({'step': self.current_step})

    def get_step_result(self, step):
        """Get the calculation result summary for a specific step.

        Parameters
        ----------
        step : int
            Step index.

        Returns
        -------
        dict or None
            Result summary for the step, or None if not found.
        """
        for record in self.step_results:
            if record.get('step') == step:
                return record
        return None
