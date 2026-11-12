"""
======================================================================
This module provides the Optimize workflow for geometry optimization.

This workflow drives the iterative loop and the optimizer
determines the next geometry:

    for step:
        calculator.run()
        state = calculator.get_state()
        state = optimizer.run(state)
        calculator.sync()
        converged, report = optimizer.convergence(state)
        if converged:
            break

Skill-internal optimization (control.driver != 'main') never
reaches this workflow: the planning layer keeps jobtype 'optimize'
on the skill and resolve_workflow returns the SingleShot workflow
instead.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os

import numpy as np
import yaml

from .constraints import runtime as constraint_runtime
from ..data.geometry import apply_pbc
from ..kernel.logger import get_logger
from ..schema.config import PATH_METHODS
from .base import Workflow
from .multiscale.composite import _select_state_energy
from .optimizer import get_optimizer
from .records import lattice_comment
from ..utilities.units import (
    ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM,
)
from ..utilities.opt_formatter import (
    STATUS_CELL_COLUMNS,
    status_header,
    status_line,
)
from ..utilities.print_results import assert_writable

# A variable-cell run is refused where the missing half actually bites, not for
# the name of the search. Two places answer that: a saddle whose curvature comes
# from a backend (`hess_driver`, checked in `Optimize.__init__`) has no lattice
# block to follow, and a crossing whose objective is the average of two surfaces
# needs a stress per surface (`optimizer/meci.py::_cell_stress`, at the
# evaluation). A crossing on a backend that answers with (nstate, 3, 3) therefore
# runs, and one that answers with a single whole-system tensor is told which
# shape would serve it rather than being turned away for a limit its own backend
# never had (dev/methods/CELL_OPT_DEVELOPMENT.md section 4.3).

# opt_status.txt takes its columns and widths from `utilities/opt_formatter.py`,
# and the reader and record type come from `utilities/print_results.py`, so a
# judge cannot take one writer's rows for the other's table.
# `_STATUS_CELL_REPORT_KEYS` maps the cell columns onto
# the keys of the optimizer's in-memory convergence report: the pairing is what
# lets a variable-cell row be appended to the same table instead of written to
# a second one a reader would have to keep in step.
_STATUS_CELL_REPORT_KEYS = ('Cell Gradient RMS', 'Cell Gradient Max',
                            'Cell Displacement RMS', 'Cell Displacement Max',
                            'Cell Volume')
if len(_STATUS_CELL_REPORT_KEYS) != len(STATUS_CELL_COLUMNS):
    raise RuntimeError(
        'opt_status: the cell report keys and the status columns have '
        'different lengths; a row would be misaligned')

# The strict conditions whose lattice derivative has been measured against a box
# whose energy is known. A condition outside this set is refused by name when the
# box is a variable, rather than carried on the strength of a formula that holds
# for it too but has not been priced.
#
# A bond was priced first (#341: with the term the box stops at the spring's rest
# length, without it 0.227 Bohr short). An angle and a torsion were refused on the
# reason that their unfolding could change branch where a length's cannot, and that
# reason is now measured away
# (`qualification/verification/test_constraint_cell_coupling.py`
# ::TestAnAngleAndATorsionHaveTheSameCellForce): the contraction equals a finite
# strain of the condition itself on every component a cell step can move (worst
# entry 1.13e-11 rad per Bohr for the angle, 2.73e-12 for the torsion, against
# derivatives of order 1e-2), and under the ride convention the image a condition
# spans is a function of the fractional coordinates alone, which a lattice step
# holds. What this set still guards is the NEXT condition kind: adding one to
# `constraints/projection.condition_row_for` without pricing it this way is
# refused, not shipped.
CELL_MEASURED_CONDITIONS = frozenset(('bond', 'angle', 'dihedral'))


def _asks_a_backend_for_an_excited_matrix(state, params):
    """Whether the surface being stepped is one no backend differentiates.

    `opt_state_i` names the surface this search is stepped on: anything above
    zero is an excited search, and a ground-state search on a multi-state model is
    not - the matrix requested there is still the one the backend computes.
    """
    try:
        state_i = int(params.get('opt_state_i', 0) or 0)
    except (TypeError, ValueError):
        return True
    if state_i > 0:
        return True
    system = getattr(getattr(state, 'pmd', None), 'system', None)
    return False


def _unmeasured_cell_conditions(state):
    """The declared strict conditions a variable-cell run has not priced.

    The kind is read the way the controller reads it, so a plan and this answer
    cannot disagree about what was declared: a `shake` family is a bond length,
    anything else names its own condition kind.
    """
    plan = state.constraint_plan()
    if plan is None or plan.is_empty:
        return []
    kinds = set()
    for spec in plan.actions:
        if spec.kind != 'constraint':
            continue
        kinds.add('bond' if spec.family == 'shake'
                  else str(spec.parameters.get('cv_kind', spec.family)))
    return sorted(kinds - CELL_MEASURED_CONDITIONS)


def resolve_skill_units(calculator, logger=None):
    """The unit factors of the run's active skill, from its manifest.

    Reads the `units` section of the active skill's `manifest.yaml` and turns it
    into the three factors an evaluation needs before the framework's own units
    mean anything: coordinates to Bohr, a displacement back to Angstrom, and the
    skill's gradient to Hartree per Bohr. A run with no calculator, no state or
    no skill keeps the Angstrom/atomic defaults, which is what a caller that
    never reaches a backend gets.

    One implementation, used by the Optimize loop and by the Path workflow: the
    path family is asked for as an `optimize.method` and evaluates through the
    same skills, so an answer that depended on which of the two loops asked would
    be a defect in the units rather than in the search.

    Returns
    -------
    tuple
        `(coord_to_bohr, disp_to_angstrom, gradient_factor)`.
    """
    logger = logger or get_logger('pyrai3md.workflow')
    coord_to_bohr = ANGSTROM_TO_BOHR
    disp_to_angstrom = BOHR_TO_ANGSTROM
    gradient_factor = 1.0

    if calculator is None:
        return coord_to_bohr, disp_to_angstrom, gradient_factor
    state = calculator.get_state()
    if state is None:
        return coord_to_bohr, disp_to_angstrom, gradient_factor

    # Find the first skill name from the task queue
    skill_name = None
    for _, task in state.iter_tasks():
        skill_name = task.get('skill_name')
        if skill_name:
            break
    if not skill_name:
        return coord_to_bohr, disp_to_angstrom, gradient_factor

    # Locate and read the manifest.yaml
    skills_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'skills', skill_name)
    manifest_path = os.path.join(skills_dir, 'manifest.yaml')
    if not os.path.exists(manifest_path):
        logger.warning(
            "Manifest not found for skill '%s', using default unit factors.",
            skill_name)
        return coord_to_bohr, disp_to_angstrom, gradient_factor
    try:
        with open(manifest_path, 'r') as handle:
            manifest = yaml.safe_load(handle)
    except Exception as error:
        logger.warning(
            "Failed to read manifest for '%s': %s", skill_name, error)
        return coord_to_bohr, disp_to_angstrom, gradient_factor

    units = manifest.get('units', {})
    if not units:
        return coord_to_bohr, disp_to_angstrom, gradient_factor

    # Energy unit -> Hartree conversion
    energy_unit = units.get('energy', 'hartree').lower()
    if energy_unit not in ('hartree', 'au', 'hart'):
        logger.warning(
            "Unknown energy unit '%s' in manifest, assuming Hartree.",
            energy_unit)

    # Gradient unit -> Hartree/Bohr conversion
    gradient_unit = units.get('gradient', 'hartree/bohr').lower()
    if gradient_unit in ('hartree/bohr', 'au/bohr', 'hart/bohr'):
        gradient_factor = 1.0
    elif gradient_unit in ('hartree/angstrom', 'au/angstrom'):
        # Hartree/Angstrom -> Hartree/Bohr:
        # dE/dx_bohr = dE/dx_angstrom * (dx_angstrom/dx_bohr)
        #            = dE/dx_angstrom * BOHR_TO_ANGSTROM
        gradient_factor = BOHR_TO_ANGSTROM
    else:
        logger.warning(
            "Unknown gradient unit '%s' in manifest, assuming Hartree/Bohr.",
            gradient_unit)

    logger.info(
        "Skill '%s' units: energy=%s, gradient=%s (gradient_factor=%.6f)",
        skill_name, energy_unit, gradient_unit, gradient_factor)
    return coord_to_bohr, disp_to_angstrom, gradient_factor


class Optimize(Workflow):
    """Geometry optimization workflow.

    Iteratively calls the calculator for energy+gradient evaluation
    and the optimizer for geometry update until convergence. The
    optimizer is created from workflow/optimizer/ based on the
    optimization type (default 'minima').

    Parameters
    ----------
    calculator : Calculator or None
        Calculator for energy+gradient computations.
    maxiter : int
        Maximum optimization steps.
    energy_threshold : float
        Energy convergence threshold (absolute delta E).
    gradient_threshold : float
        Gradient RMS convergence threshold.
    displacement_threshold : float
        Displacement RMS convergence threshold.
    **kwargs
        Dispatcher parameters (state, multitask, remotetask,
        local_executor, executor_spec).
    """

    name = 'optimize'
    workflow_kind = 'iterative'

    def __init__(self, calculator=None, maxiter=100,
                 energy_threshold=1e-6, gradient_threshold=3e-4,
                 displacement_threshold=3e-4, **kwargs):
        super().__init__(name='optimize', calculator=calculator,
                         maxiter=maxiter, **kwargs)
        self.energy_threshold = float(energy_threshold)
        self.gradient_threshold = float(gradient_threshold)
        self.displacement_threshold = float(displacement_threshold)

        # Resolve skill backend unit conversion factors from
        # the skill manifest's units section.
        state = self.calculator.get_state() if self.calculator else None
        self._resolve_skill_units()

        # Create the optimizer based on the optimization type.
        # Read parameters through PyRAIState properties (not direct
        # pmd access) per the attribute access convention.
        params = state.optimize_params if state else {}

        method = params.get('opt_method', 'minima')
        if str(method or '').strip().lower() in PATH_METHODS:
            # `resolve_workflow` sends these three names to the Path workflow,
            # and this class has no single geometry to step on for them: a path
            # optimizer's `run` is refused by name. Reaching here means the
            # routing was bypassed, and saying so is cheaper than the first
            # evaluation of a loop that cannot take a step.
            raise ValueError(
                "optimize.method %r walks several structures and is driven by "
                "the Path workflow (`workflow/path.py`), not by this loop. "
                "Entry points route it there by name (`resolve_workflow`); a "
                "caller that built this workflow directly should build Path"
                % method)
        opt_maxiter = params.get('opt_maxit', maxiter)
        opt_energy = params.get('opt_energy', energy_threshold)
        opt_grms = params.get('opt_grms', gradient_threshold)
        opt_gmax = params.get('opt_gmax', 4.5e-4)
        opt_drms = params.get('opt_drms', displacement_threshold)
        opt_dmax = params.get('opt_dmax', 1.8e-3)
        # The cell half's own four thresholds. Their fallbacks are the same
        # numbers the schema's DEFAULT carries: a run that reaches an optimizer
        # without them has to be judged as the documented defaults say, not as
        # whatever an older caller passed for the atoms.
        opt_csrms = params.get('opt_csrms', 40.0)
        opt_csmax = params.get('opt_csmax', 60.0)
        opt_cerms = params.get('opt_cerms', 5.0e-5)
        opt_cemax = params.get('opt_cemax', 7.0e-5)
        # The box's own two options: the pressure the objective carries and how
        # many lattice directions one pressure value is allowed to move. A run
        # that got here without them asks for a zero-stress box in all six
        # directions, which is what it did before the keywords existed.
        opt_pressure = params.get('opt_pressure', 0.0)
        opt_cell_coupling = params.get('opt_cell_coupling', 'anisotropic')
        # The step bound. Unset is not a number: it is the search's own radius,
        # and the searches disagree on purpose (`minima`/`gdiis` start at 0.3,
        # the transition-state search at 0.05, `optimizer/ts.py:170`), so this
        # run passes the keyword only when the input named one. Passing a
        # framework default instead would have moved every TS leg eight-fold.
        opt_trust_radius = params.get('opt_trust_radius')
        use_rfo = bool(params.get('opt_use_rfo', False))
        # The box is a variable when the run says so; the same keyword is what
        # made the planning layer ask for the stress that drives it
        # (planning/resolver.py), so one keyword answers both halves.
        cell_variables = bool(params.get('optimize_cell', False))
        if cell_variables and state is not None:
            driver = str((getattr(state, 'hessian_params', None) or {}).get(
                'hess_driver') or '').strip().lower()
            if (method == 'ts' and driver and driver != 'main'
                    and _asks_a_backend_for_an_excited_matrix(state, params)):
                # The cell half of this pairing is measured now: the framework
                # strains the box twelve times and joins those rows to the
                # backend's atomic block (`workflow/hessian.py`), so the six
                # lattice directions are differenced rather than zero-filled.
                # What still has no producer is the atomic half of an excited
                # surface: the skill that answers a Hessian request answers it for
                # its ground state, and the excited profile chains carry
                # `hessian: None` (`skills/pyscf_skill/schema/method.py`). A
                # missing matrix is not a small one, so the run is stopped before
                # an evaluation is paid for.
                raise ValueError(
                    "optimize_cell with hess_driver=%r on an excited state has no "
                    "curvature producer: a backend's force constants are its "
                    "ground state's, and the six lattice rows ride on that block. "
                    "Ask the framework for the matrix (hess_driver: main), which "
                    "strains the box beside displacing the atoms, or optimize the "
                    "ground state" % driver)
            unmeasured = _unmeasured_cell_conditions(state)
            if unmeasured:
                # The lattice derivative of a held condition is the same
                # contraction whatever the condition measures - the multiplier
                # times the condition's own derivative with respect to the box,
                # which is what `constraints/projection.cell_virial` computes -
                # and bond, angle and torsion have each been measured against it
                # (see `CELL_MEASURED_CONDITIONS` for how). A kind that has not
                # been priced that way is refused rather than carried on the
                # strength of the formula, because a condition whose unfolding
                # can change branch under a box step would put a wrong number in
                # the cell force and no shape check would notice.
                raise ValueError(
                    "optimize_cell with a strict %s condition is not measured: "
                    "the cell force carries a held condition's own lattice "
                    "derivative, and that term has been priced against a finite "
                    "strain only for the kinds in %s. A run that must relax its "
                    "box holds such a condition through the fix keys instead, or "
                    "keeps the cell fixed and relaxes the atoms"
                    % (' and '.join(repr(kind) for kind in unmeasured),
                       sorted(CELL_MEASURED_CONDITIONS)))

        self.method = method

        # Target electronic state for the optimization (0-based,
        # consistent with the input state_i and the first axis of
        # the state-resolved result arrays). opt_state_j names the
        # second state of a crossing search; a run that names one
        # state asks for one state and reports no gap.
        state_i = params.get('opt_state_i', 0)
        state_j = params.get('opt_state_j', 0)
        state_i = (tuple(int(entry) for entry in state_i)
                   if isinstance(state_i, (list, tuple)) else int(state_i))
        state_j = (tuple(int(entry) for entry in state_j)
                   if isinstance(state_j, (list, tuple)) else int(state_j))
        if state:
            if state.n_center > 1:
                # A multi-center run has no global state axis. A scalar index
                # goes to every center and the narrowest ladder bounds it; a
                # per-center list names each center's own state (or pair), and
                # each entry is bounded by THAT center's ladder. The per-center
                # active roots stay where the plan put them, and
                # `current_state` is not written because such a run holds no
                # such scalar.
                per_center = (isinstance(state_i, tuple)
                              or isinstance(state_j, tuple))
                if not per_center:
                    limit = min(state.center(c).nstate
                                for c in range(state.n_center))
                    for name, index in (('opt_state_i', state_i),
                                        ('opt_state_j', state_j)):
                        if not 0 <= index < limit:
                            raise ValueError(
                                "%s=%d is out of range: a multi-center search "
                                "solves every center at the requested state, "
                                "and the narrowest ladder here holds %d "
                                "state(s)." % (name, index, limit))
                else:
                    for name, value in (('opt_state_i', state_i),
                                        ('opt_state_j', state_j)):
                        picks = ([int(entry) for entry in value]
                                 if isinstance(value, tuple)
                                 else [int(value)] * state.n_center)
                        if len(picks) != state.n_center:
                            raise ValueError(
                                "%s names %d centers; this run has %d."
                                % (name, len(picks), state.n_center))
                        for index, pick in enumerate(picks):
                            ladder = state.center(index).nstate
                            if not 0 <= pick < ladder:
                                raise ValueError(
                                    "%s=%d is out of range: center %d holds "
                                    "%d state(s)."
                                    % (name, pick, index, ladder))
            else:
                # Single-center range rule lives at the input edge
                # (schema/validate.py), so it is not repeated here.
                # Register the optimized state as the current state.
                state.update_system({'current_state': state_i})

        # A pair of states is requested per evaluation, through the
        # Calculator's per-step channel, so both gradients arrive in the
        # same call that produces the step. Single-state runs keep passing
        # None, which leaves each backend on its initialization value.
        if isinstance(state_i, tuple) or isinstance(state_j, tuple):
            i_picks = (state_i if isinstance(state_i, tuple)
                       else (state_i,) * state.n_center)
            j_picks = (state_j if isinstance(state_j, tuple)
                       else (state_j,) * state.n_center)
            pairs = {c: ([i_picks[c], j_picks[c]]
                         if i_picks[c] != j_picks[c] else [i_picks[c]])
                     for c in range(state.n_center)}
            self.target_states = pairs
        else:
            self.target_states = ([state_i, state_j]
                                  if state_j != state_i else None)

        # The optimizer steps on the composite's state axis. With a
        # per-center pair the assembly exposes exactly two surfaces - state 0
        # is every center at the first state of its own pair, state 1 the
        # second - so the search reads 0 and 1 regardless of where each
        # center's pair sits in its own ladder.
        optimizer_state_i = 0 if isinstance(state_i, tuple) else state_i
        optimizer_state_j = 1 if isinstance(state_j, tuple) else state_j

        # Create the framework optimizer driving the iterative loop.
        self.optimizer = get_optimizer(
            method,
            maxiter=opt_maxiter,
            energy_threshold=float(opt_energy),
            gradient_threshold=float(opt_grms),
            gradient_max_threshold=float(opt_gmax),
            displacement_threshold=float(opt_drms),
            displacement_max_threshold=float(opt_dmax),
            cell_stress_rms_threshold=float(opt_csrms),
            cell_stress_max_threshold=float(opt_csmax),
            cell_strain_rms_threshold=float(opt_cerms),
            cell_strain_max_threshold=float(opt_cemax),
            external_pressure=float(opt_pressure),
            cell_coupling=str(opt_cell_coupling),
            # Spread rather than a value: an unset radius contributes nothing and
            # the constructor keeps its own default, which is the contract the
            # comment at its read spells out.
            **({} if opt_trust_radius is None else
               {'trust_radius': float(opt_trust_radius)}),
            use_rfo=use_rfo,
            active_state=optimizer_state_i,
            active_state_j=optimizer_state_j,
            cell_variables=cell_variables,
            coord_to_bohr=self.coord_to_bohr,
            disp_to_angstrom=self.disp_to_angstrom,
        )

        # Where the run's records go. Construction only resolves it; the
        # files are opened by run(), because a dispatched unit is stamped
        # with its own workdir after this point and a record written at
        # construction time would land in the launch directory.
        self.output_dir = self._resolve_output_dir(state)

    def run(self):
        """Execute geometry optimization.

        Implements the optimization loop from the design document:
        iteratively computes energy+gradient, updates geometry via
        the optimizer, syncs the calculator, and checks convergence.
        Each iteration writes the current structure to opt.xyz
        (latest only) and opt_geom.xyz (all steps appended), and
        appends one line to opt_status.txt.

        Returns
        -------
        PyRAIState
            The state with optimized geometry.
        """
        state = self.calculator.get_state()
        # Resolved again at run time: the dispatcher stamps a dispatched
        # unit's own workdir onto the runtime after the workflow object was
        # built, and a record opened at construction time would stay in the
        # launch directory while the run log moved.
        self.output_dir = self._resolve_output_dir(state)
        self._init_output_files()
        # `opt_maxit` is the step bound for a relaxation, so it is the only
        # number that may lower it. It falls back to 100 in `PyRAIState` when the
        # input says nothing, which is why `self.maxiter` (a constructor default
        # that only AIMD's wiring replaces) is not taken into the bound: it capped
        # a run that asked for 120 iterations at 100 evaluations and left 101
        # frames in the trajectory, silently rewriting what the user declared.
        # `runtime.total_steps` can still bind the loop, but only on a resumed
        # trajectory, because it is written by the MD restart path and never by a
        # relaxation.
        maxit = state.optimize_maxit if state else self.maxiter
        total_steps = (state.runtime.total_steps
                       if state and state.runtime else None)
        if total_steps is None or total_steps <= 0:
            total_steps = maxit
        else:
            total_steps = min(total_steps, maxit)

        run_logger = getattr(state, 'logger', None)
        if run_logger is not None:
            run_logger.log_optimize_setup(self._setup_block(state, maxit))

        # Same reason the MD loop binds before its first step: a bond target
        # captured inside the loop would be captured after the optimizer has
        # already moved the atoms, and the run would then hold the length of its
        # own first step rather than the structure it was given.
        constraint_runtime.bind(state)

        # A search that follows a curvature asks the framework's own Hessian
        # computation for it before it steps. The batch displaces the geometry,
        # collects the gradients and restores the reference structure, so the
        # first evaluation of the loop is still the one the input described.
        if getattr(self.optimizer, 'needs_initial_hessian', False):
            self._supply_initial_hessian()
            state = self.calculator.get_state()

        converged = False
        steps = 0
        report = {}
        stop_message = 'Maximum optimization steps reached'
        for step in range(total_steps):
            # 1. Compute energy + gradient via the calculator
            result = self.calculator.run(target_states=self.target_states)
            state = self.calculator.get_state()

            # Convert gradient from skill units to Hartree/Bohr
            # so the optimizer (working in Bohr) is consistent.
            gradient = (result.get('gradient')
                        if result else None)
            if gradient is not None and self.gradient_factor != 1.0:
                gradient = np.asarray(gradient) * self.gradient_factor
                result['gradient'] = gradient

            # Record the step through the official PyRAIState
            # interface; property data (energy, gradient) stays in
            # pmd results (runtime records metadata only).
            state.record_step_result()

            # Snapshot the structure this energy and gradient belong to, before
            # the optimizer moves the atoms. The progress row written at the end
            # of this iteration reports this pair, and a step's frame has to
            # describe the same structure the row judges: the trial geometry the
            # optimizer proposes below is what the NEXT evaluation is scored on,
            # so writing it here put every frame one step ahead of its own row.
            # Measured on round tmp_20261017_210205's pressure leg, where frame N
            # evaluates to the energy recorded at row N+1 (-0.9291051 / -0.9264462
            # / -0.9289608 / -0.9290993 against the next row to seven digits), and
            # a reader comparing a converged structure to its criteria was
            # therefore comparing two different geometries.
            evaluated = (list(state.atoms),
                         np.array(state.coordinates, dtype=np.float64),
                         state.pmd.system.model_dump())

            # Read the constraint split here, at the geometry this energy and
            # gradient belong to, before the optimizer moves the atoms. What the
            # convergence criteria judge is the tangential part of this pair; the
            # part a condition absorbs is reported as its multiplier.
            self.optimizer.note_evaluation(state)

            # Write the run-log step block (energy/gradient at the
            # evaluated geometry, before the optimizer updates it).
            logger = getattr(state, 'logger', None)
            if logger is not None:
                logger.log_step(step + 1, state.pmd.model_dump())

            # 2. Compute next geometry via the optimizer
            state = self.optimizer.run(state)

            # 2a. Wrap the trial geometry into the cell it belongs to and accrue
            # the crossing counts, before anything reads the structure. The
            # search moves atoms across a face whenever a step carries them past
            # it, and a variable-cell run does that on every step because the
            # lattice entries carry the atoms with the box; the backend contract
            # is that a wrapped coordinate arrives with the image flags of the
            # same step, since a new position written on a stale flag relocates
            # the atom by a whole box. This is the same operation the dynamics
            # loop runs after its drift (`workflow/md.py`), because it is a
            # property of a periodic structure, not of an integrator.
            state = apply_pbc(state)

            # 2b. Land the trial geometry on the constraint manifold. The
            # optimizer projected the frozen directions out of its step, but a
            # bond length is a curved condition: only the position solve puts
            # |r_ij| back on its target. Doing it here means the structure the
            # next evaluation sees, the one written to opt_geom.xyz, and the one
            # the reported E/G pair describe are the same structure.
            constraint_runtime.constrain_state(state)

            # 3. Sync updated coordinates to task configs
            self.calculator.sync()

            # 4. Check convergence
            converged, report = self.optimizer.convergence(state)
            steps = step + 1

            # 5. Log optimization progress to the run log (each
            # judged entry carries its threshold and True/False).
            details = self.optimizer.method_details(report)
            run_logger = getattr(state, 'logger', None)
            if run_logger is not None:
                run_logger.log_optimize_step(
                    steps, report, converged,
                    thresholds=self._threshold_report(), details=details)

            # 6. Write optimization output files
            self._write_step_output(state, report, steps,
                                    evaluated=evaluated)

            state.runtime.advance_step()

            if converged:
                break
            # A search that has had its last steps rejected one after another
            # is spending evaluations on a model that keeps mispricing them;
            # stopping is the answer, not a fifth rejection.
            if getattr(self.optimizer, 'stalled', False):
                stop_message = ('Optimization stopped after %d rejected '
                                'steps' % self.optimizer.failure_count)
                break

        # A constrained run evaluates its final structure once more, and only
        # for the same reason the loop above moves after evaluating: the
        # optimizer proposes a step, then the freeze and the bond solve take it
        # back. Without this the last frame of opt_geom.xyz would be a structure
        # whose energy and gradient were never computed, and the pair the run
        # reports would describe a geometry the run does not have
        # (dev/methods/CONSTRAINT_DEVELOPMENT.md section 14, C1-6). An
        # unconstrained optimization is left exactly as it was: it gains no
        # evaluation, no history frame and no log block.
        if not state.constraint_plan().is_empty:
            self.calculator.sync()
            result = self.calculator.run(target_states=self.target_states)
            state = self.calculator.get_state()
            state.record_step_result()
            # Same unit conversion the loop applies, so the pair written here
            # is the pair the criteria are judged on.
            gradient = (result.get('gradient') if result else None)
            if gradient is not None and self.gradient_factor != 1.0:
                gradient = np.asarray(gradient) * self.gradient_factor
                result['gradient'] = gradient
            self.optimizer.record_final_evaluation(state)
            self.optimizer.note_evaluation(state)
            # The tail is one evaluation and one printed frame, so it carries
            # its own row: the frame count and the table row count stay equal,
            # and the numbers on the line are the ones measured on the
            # structure the run delivers rather than the previous pair's.
            converged, report = self.optimizer.convergence(state)
            self._write_step_output(state, report, steps + 1)
            run_logger = getattr(state, 'logger', None)
            if run_logger is not None:
                run_logger.log_step(steps + 1, state.pmd.model_dump())
                run_logger.log_optimize_step(
                    steps + 1, report, converged,
                    thresholds=self._threshold_report(),
                    details=self.optimizer.method_details(report))

        # Log the optimization outcome before settling the runtime
        run_logger = getattr(state, 'logger', None)
        if run_logger is not None:
            run_logger.log_optimize_summary(
                converged, steps, report.get('Energy'),
                message='' if converged else stop_message)

        # Mark completion status
        if converged:
            state.runtime.mark_completed()
        else:
            state.runtime.mark_failed(stop_message)

        return state

    # ---- Curvature and criteria ----

    def _supply_initial_hessian(self):
        """Produce the curvature the search starts on and hand it over.

        A mode-following search needs a force-constant matrix before its first
        step, and `&hessian` says who produces it: `hess_driver` 'main'
        displaces the geometry through the framework's own batch, 'skill' asks
        the backend for the matrix it computes (analytically, or with
        displacements of its own - which of the two is the skill's keyword, and
        from here both are one evaluation that answers with a matrix). The old
        ruling that only the first of these may exist was written when the
        framework's batch was the only producer this file could reach; the
        second one is `workflow/hessian.py::_requested_backend_matrix`.

        Two defaults are this caller's to set rather than the section's. The
        producer is the framework unless the input names the skill, because the
        framework's batch is the only answer that exists for every backend, and
        a search that silently inherited 'ask the backend' would fail on the
        skills that have no force-constant profile. And the analysis is off: the
        search follows a curvature, it does not read one into frequencies, and
        the vibrational analysis needs atomic masses (`_amu_masses`) that a run
        is not required to carry. The matrix arrives through the computation's
        own attribute rather than `state.result`, which carries it with a
        leading state axis while the optimizer steps on the flat (3N, 3N) form.
        """
        from .hessian import Hessian

        state = self.calculator.get_state()
        section = dict(getattr(state, 'hessian_params', None) or {})
        driver = section.get('hess_driver')
        mode = section.get('hess_mode')
        if mode is None:
            mode = ('analytic' if driver == 'skill' else 'numerical')
        curvature = Hessian(calculator=self.calculator, source=mode,
                            driver=driver, analysis=False)
        curvature.run()
        matrix = curvature.matrix
        if matrix is None:
            raise ValueError(
                "optimize.method 'ts' needs the framework Hessian computation "
                "and it returned no matrix: the displaced gradients were not "
                "available, so there is no curvature to follow")
        self.optimizer.set_initial_hessian(np.asarray(matrix, dtype=np.float64))

    def _threshold_report(self):
        """The threshold each judged entry of the progress block is read against."""
        thresholds = {
            'Delta E': self.optimizer.energy_threshold,
            'Gradient RMS': self.optimizer.gradient_threshold,
            'Gradient Max': self.optimizer.gradient_max_threshold,
            'Displacement RMS': self.optimizer.displacement_threshold,
            'Displacement Max': self.optimizer.displacement_max_threshold,
        }
        # The cell rows carry their own thresholds from here on: a run that
        # relaxed its box used to have its box judged against the atomic numbers
        # and printed without a threshold at all, so the record read as if the
        # two halves answered one question.
        if getattr(self.optimizer, 'cell_variables', False):
            thresholds['Cell Stress RMS'] = (
                self.optimizer.cell_stress_rms_threshold)
            thresholds['Cell Stress Max'] = (
                self.optimizer.cell_stress_max_threshold)
            thresholds['Cell Strain RMS'] = (
                self.optimizer.cell_strain_rms_threshold)
            thresholds['Cell Strain Max'] = (
                self.optimizer.cell_strain_max_threshold)
        # A crossing search judges the gap between its two surfaces against the
        # same keyword that bounds an energy change, and the row says so.
        gap_key = getattr(self.optimizer, 'gap_report_key', None)
        if gap_key:
            thresholds[gap_key] = self.optimizer.energy_threshold
        return thresholds

    # ---- Unit resolution from skill manifest ----

    def _resolve_skill_units(self):
        """Resolve unit conversion factors from the skill manifest.

        Sets three attributes:
            self.coord_to_bohr    : Angstrom -> Bohr factor for
                                    coordinates (default ANGSTROM_TO_BOHR).
            self.disp_to_angstrom : Bohr -> Angstrom factor for
                                    displacements (default BOHR_TO_ANGSTROM).
            self.gradient_factor  : skill gradient -> Hartree/Bohr
                                    factor (default 1.0).

        One implementation, shared with the Path workflow (which is asked for as
        an `optimize.method` and evaluates through the same skills): the reading
        lives in the module-level `resolve_skill_units` so the two workflows
        cannot disagree about the unit of one backend's gradient.
        """
        (self.coord_to_bohr, self.disp_to_angstrom,
         self.gradient_factor) = resolve_skill_units(self.calculator,
                                                     self.logger)

    # ---- Output file management ----

    def _resolve_output_dir(self, state):
        """The directory this run's records belong to.

        The MD chain already owns this rule: `pmd.py` builds the run logger
        from the environment's scratch path, `workflow/md.py` restamps it
        with the unit's own workdir, and `MdOutputWriter` is handed the same
        value. The optimization records read it off the logger instead of
        deriving a directory of their own, so a run cannot leave its table
        beside the launch script while its log sits in the result directory.
        A state with no logger, or a logger with no directory, keeps the
        current directory.
        """
        logger = getattr(state, 'logger', None) if state is not None else None
        if logger is None:
            return os.getcwd()
        workdir = (getattr(state.runtime, 'workdir', '')
                   if getattr(state, 'runtime', None) is not None else '')
        if workdir and hasattr(logger, 'log_dir'):
            logger.log_dir = workdir
            if hasattr(logger, 'log_file'):
                title = getattr(logger, 'title', '') or ''
                logger.log_file = os.path.join(workdir,
                                               '%s.log' % title)
        return getattr(logger, 'log_dir', None) or os.getcwd()

    def _output_path(self, name):
        """One process record, in the resolved output directory."""
        return os.path.join(self.output_dir, name)

    def _setup_block(self, state, maxit):
        """Run-level values for the `&optimization setup` block."""
        optimizer = self.optimizer
        cell = bool(getattr(optimizer, 'cell_variables', False))
        setup = {
            'Method': getattr(self, 'method', 'minima'),
            'Optimizer': type(optimizer).__name__,
            'Max steps': int(maxit),
            'Cell mode': 'variable' if cell else 'fixed',
        }
        if cell:
            setup['Cell coupling'] = getattr(optimizer, 'cell_coupling', None)
            pressure = getattr(optimizer, 'external_pressure', None)
            if pressure is not None:
                setup['Pressure target'] = '%.4f bar' % float(pressure)
            system = getattr(getattr(state, 'pmd', None), 'system', None)
            volume = getattr(system, 'volume', None)
            if volume:
                setup['Initial volume'] = '%.6f A^3' % float(volume)
        plan = state.constraint_plan() if state is not None else None
        if plan is not None and not plan.is_empty:
            setup['Constraint conditions'] = len(plan.actions)
        return setup

    def _init_output_files(self):
        """Initialize optimization output files.

        Clears opt_geom.xyz and writes the column header to
        opt_status.txt.  opt.xyz is overwritten each step so no
        initialization is needed for it.

        The directory is made here rather than trusted: the run logger creates
        it when it first opens a handle, and a dispatched unit's workdir is
        stamped onto the state before either write, so an optimization that
        reaches this point first would open a file in a directory nothing has
        created yet.

        An existing record is read before it is truncated. One directory has
        one owner for this file name: an engine writes its own convergence
        table under the same plain name inside a calculation directory it keeps
        (openqp does, with an eleven-token title-case header), and truncating
        it here would destroy the reading that engine's parser goes back for
        while leaving a header the framework's own rows would not match.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self._output_path('opt_geom.xyz'), 'w') as f:
            pass  # truncate
        status_path = self._output_path('opt_status.txt')
        assert_writable(status_path)
        with open(status_path, 'w') as f:
            f.write(self._status_header() + '\n')

    def _status_header(self, terms=()):
        """The column header of `opt_status.txt` for this run.

        A run that relaxes the box carries the cell readings beside the atomic
        ones; a run that does not keeps the table it has always written. A run
        whose energy is assembled from more than one layer carries one further
        column per term, named by the assembly itself.
        """
        return status_header(
            cell=getattr(self.optimizer, 'cell_variables', False),
            terms=terms,
            method=self.optimizer.status_method_columns())

    def _composite_terms(self, state):
        """The assembly's per-term columns for the row being written now.

        Returns
        -------
        tuple
            `(labels, values)`, both empty for a run that assembled nothing.
            A term's value is its energy at the state the assembly records for
            it (`composite_terms['state_indices']`; where the assembly records
            none the total keeps one global state axis and the run's driven
            state decides), multiplied by the term's own ONIOM coefficient, so
            the columns are
            the additive half of the objective: they sum back to the energy
            column, which is what makes them worth reading from disk rather
            than off the `&composite energy decomposition` block of the run
            log. The raw per-state energies stay in that block, which prints
            every state; this table prints the one the row is about.

            The one case the sum does not cover is a crossing search: its
            energy column is the mean of two surfaces (`optimizer/meci.py`
            `_objective`), so the term columns add up to the state named below
            and not to that mean. The columns are still the terms' own
            numbers, which is what they are here for.
        """
        terms = (state.pmd.results or {}).get('composite_terms')
        if not isinstance(terms, dict):
            return (), ()
        labels = [str(name) for name in (terms.get('labels') or [])]
        energies = terms.get('energies') or []
        coefficients = terms.get('contributions') or []
        indices = terms.get('state_indices')
        if not labels or not (len(labels) == len(energies)
                               == len(coefficients)):
            return (), ()
        if indices is not None and len(indices) != len(labels):
            run_logger = getattr(state, 'logger', None)
            if run_logger is not None:
                run_logger.write(
                    'WARNING: composite term columns refused -- the assembly '
                    'records %d state indices for %d terms, so a column could '
                    'not be matched to the term that wrote it'
                    % (len(indices), len(labels)))
            return (), ()
        driven = int(getattr(state.pmd.system, 'current_state', 0) or 0)
        # A recorded None means the term has no state of its own, which is the
        # single-center case: the assembled total carries the state axis and the
        # run's driven state picks the row the column is about.
        states = [int(indices[where]) if indices and indices[where] is not None
                  else driven for where in range(len(labels))]
        values = tuple(float(coefficient) * _select_state_energy(
                           row, at, label)
                       for row, coefficient, label, at
                       in zip(energies, coefficients, labels, states))
        return tuple(labels), values

    def _fit_status_header(self, labels):
        """Give the record its term columns before the first row is appended.

        Which terms a run assembles is a fact about the plan, and the plan
        answers only after the first evaluation, while the record is opened
        before the run starts. So `_open_output` writes the fixed header it has
        always written, and this widens it once - at step one, with no row yet
        in the file - when the first evaluation turns out to be a composite. A
        single-layer run never comes here, so the bytes it writes do not move.
        """
        path = self._output_path('opt_status.txt')
        with open(path, 'r', encoding='utf-8') as handler:
            lines = handler.read().splitlines()
        wanted = self._status_header(terms=labels)
        if lines[:1] == [wanted]:
            return
        if len(lines) > 1:
            raise ValueError(
                'refusing to widen the header of %s with %d row(s) already '
                'written against it: the term columns would be read back as a '
                'different table than the one that wrote them'
                % (path, len(lines) - 1))
        with open(path, 'w', encoding='utf-8') as handler:
            handler.write(wanted + chr(10))

    def _write_step_output(self, state, report, step, evaluated=None):
        """Write optimization output for the current step.

        Writes the latest structure to opt.xyz (overwrite),
        appends the structure to opt_geom.xyz, and appends
        one progress line to opt_status.txt.

        Parameters
        ----------
        state : PyRAIState
            Current state with updated coordinates.
        report : dict
            Convergence report from the optimizer.
        step : int
            Current step number (1-based).
        evaluated : tuple, optional
            ``(atoms, coordinates, system_dict)`` of the structure the reported
            energy and gradient belong to. Given by the loop, which snapshots it
            before the optimizer moves; without it the frame written beside this
            row would be the trial geometry the next evaluation is scored on, and
            the two artifacts would describe different structures one step apart.
        """
        if evaluated is not None:
            atoms, coordinates, system = evaluated
        else:
            atoms, coordinates, system = (state.atoms, state.coordinates,
                                          state.pmd.system.model_dump())
        if atoms is None or coordinates is None or not len(atoms) \
                or not len(np.asarray(coordinates)):
            return

        coords = np.asarray(coordinates, dtype=np.float64)
        natom = len(atoms)
        energy = report.get('Energy', 0.0)

        # A periodic frame carries its box in the comment line, the same
        # extended-XYZ key the MD trajectory writes; without it a molecule
        # sitting on two sides of a cell face reads as a broken structure.
        comment = 'Step %d  E = %.10f' % (step, energy)
        comment += lattice_comment(system)

        # 1. Write latest structure to opt.xyz (overwrite)
        self._write_xyz(self._output_path('opt.xyz'), atoms, coords, natom,
                        comment)

        # 2. Append structure to opt_geom.xyz
        self._write_xyz(self._output_path('opt_geom.xyz'), atoms, coords,
                        natom, comment, append=True)

        # 3. Append progress line to opt_status.txt
        shift = report.get('Delta E', 0.0)
        max_d = report.get('Displacement Max', 0.0)
        rms_d = report.get('Displacement RMS', 0.0)
        max_g = report.get('Gradient Max', 0.0)
        rms_g = report.get('Gradient RMS', 0.0)

        values = (energy, shift, max_d, rms_d, max_g, rms_g)
        if getattr(self.optimizer, 'cell_variables', False):
            # The cell readings belong to the same evaluation as the atomic ones,
            # so they are appended to the same row rather than written to a
            # second table that a reader would have to keep in step with this one.
            values += tuple(float(report.get(key, 0.0))
                            for key in _STATUS_CELL_REPORT_KEYS)
        method_columns = self.optimizer.status_method_columns()
        if method_columns:
            # Method readings sit after the cell block and before the composite
            # terms, so the fixed prefix `classify_header` recognizes stays put.
            values += tuple(float(value) for value
                            in self.optimizer.status_method_values(report))
        labels, term_values = self._composite_terms(state)
        if labels:
            # The term columns come last, after the cell ones, so a reader that
            # knows only the fixed table can still take the columns it knows.
            if step == 1:
                self._fit_status_header(labels)
            values += term_values
        line = status_line(step, values, extra=len(labels),
                           method=len(method_columns))
        with open(self._output_path('opt_status.txt'), 'a') as f:
            f.write(line + '\n')

    @staticmethod
    def _write_xyz(path, atoms, coords, natom, title, append=False):
        """Write one XYZ frame, overwriting unless ``append`` is set.

        The overwrite (`opt.xyz`/`path.xyz`, the latest structure) and
        the append (`opt_geom.xyz`, the trajectory) carry the same
        frame text; they were two functions whose only difference was
        the file mode, which is how a format change could reach one
        product and not the other.

        Parameters
        ----------
        path : str
            Output file path.
        atoms : list of str
            Element symbols.
        coords : np.ndarray
            Coordinates array of shape (natom, 3).
        natom : int
            Number of atoms.
        title : str
            Comment line.
        append : bool, optional
            Append to the file instead of replacing it.
        """
        with open(path, 'a' if append else 'w') as f:
            f.write('%d\n' % natom)
            f.write('%s\n' % title)
            for atom, (x, y, z) in zip(atoms, coords):
                f.write('%-6s %16.10f %16.10f %16.10f\n' % (
                    atom, x, y, z))

    def __repr__(self):
        return "Optimize(maxiter=%d, optimizer=%s)" % (
            self.maxiter, repr(self.optimizer))
