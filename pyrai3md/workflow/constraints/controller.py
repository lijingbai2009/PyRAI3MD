"""
======================================================================
The staged controller that composes the constraint subsystem.

SystemConstraintController is the public surface. It holds a plan, and every
entry point takes the data that entry point needs plus an EvaluationContext,
so no function here reads a module-level step counter or a hidden clock. The
controller is a composition of the small modules beside it: it owns no
algorithm of its own, which is what keeps a wall, a bias and a bond constraint
from turning into one class with one giant conditional.

The phases, in the order a workflow uses them:

    plan = ...                                  # from schema normalization
    controller = SystemConstraintController(plan)
    state = controller.bind_reference(coords, masses, cell)
    coords, projection = controller.project_positions(coords, state, context)
    potential = controller.evaluate_potentials(coords, state, context, masses)
    totals = controller.combine(base, potential, projection, context)
    velocities = controller.project_velocities(velocities, state, context,
                                               masses=masses,
                                               coords=projected)
    statistics = controller.statistics(coords, velocities, masses, context)
    state = controller.commit(coords, state, context, masses)

`bind_reference` is per trajectory and happens once: it captures the targets
and anchors that refer to the initial geometry. `evaluate_potentials` is pure.
Only `commit`, on an accepted step, changes history.

The empty-plan short circuit
---------------------------
When the plan holds no action, every entry point returns its input unchanged
and `ConstraintResult.empty()`, which carries no (natom, 3) array at all. That
is DC-1 and DC-3 in code: an unconstrained run does not allocate, does not
grow a history frame, and does not change a single number a workflow prints.

Velocities
----------
`project_velocities` zeroes the frozen components, and nothing else, unless a
constraint record declares `velocity_constraint: rattle`. Then it also runs the
velocity projection of `shake.solve_velocities` on the projected geometry, which
is the half a position-only solve leaves open (see the measured numbers in
design document section 1.3). The default is `none`, so no run takes that branch
until it asks for it.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from pyrai3md.data.constraint import ConstraintResult
from pyrai3md.data.constraint import ConstraintState
from pyrai3md.workflow.constraints import (
    bias, metadynamics, projection, shake, wall)


class ControllerError(ValueError):
    """The controller was asked something the plan cannot answer."""


# How far a base result's declared geometry may sit from the projected one
# before the pairing is refused. This is a copy of a structure through dict
# and JSON round trips, not a solver residual, so it is tight.
GEOMETRY_ATOL = 1.0e-10


class SystemConstraintController(object):
    """One controller per plan; no mutable state of its own.

    Parameters
    ----------
    plan : ConstraintPlan
        The normalized numeric plan. An empty plan is the normal case: it
        makes every entry point a pass-through.
    """

    def __init__(self, plan):
        self.plan = plan
        self._check_plan()

    # ------------------------------------------------------------- lifecycle

    def is_noop(self):
        """True when the plan declares no action, so nothing here does work."""
        return bool(self.plan.is_empty)

    def bind_reference(self, coords, masses=None, cell=None, context=None):
        """Capture the targets and anchors that mean 'the initial geometry'.

        Called once per trajectory, before the first evaluation, and never on
        a restored run: a restart reads the bound values from the state, since
        re-capturing them from a drifted structure would silently move the
        target the whole run is measured against.
        """
        array = np.asarray(coords, dtype=np.float64)
        state = ConstraintState()
        state.targets = {}
        state.anchors = {}
        for spec in self.plan.actions:
            if spec.kind == 'fix':
                state.anchors[spec.id] = array.copy()
            elif spec.kind == 'constraint' and spec.reference == 'initial':
                state.targets[spec.id] = self._bind_constraint_targets(
                    spec, array)
            elif spec.kind == 'bias' and spec.target is None:
                record = self._cv_of(spec)
                state.targets[spec.id] = float(
                    bias.variable_value(record, array))
            elif (spec.kind == 'metadynamics'
                  and spec.family == 'rmsd'
                  and spec.reference == 'initial'):
                # The block, not the whole geometry: an aligned RMSD is written
                # against the atoms the action names, and storing the full
                # (natom, 3) array here would make the frame depend on atoms
                # the variable does not look at.
                state.targets[spec.id] = array[
                    np.asarray(spec.atoms, dtype=np.int64)].copy().tolist()
        self._bind_wall_centers(array, state, masses)
        if cell is not None:
            state.diagnostics['reference_cell'] = np.asarray(
                cell, dtype=np.float64).tolist()
        state.history_version = 0
        state.last_committed_step = -1
        return state

    # ------------------------------------------------------------ projection

    def project_positions(self, coords, state, context=None, masses=None):
        """Restore frozen components and satisfy the declared bond lengths.

        Returns
        -------
        tuple
        (coordinates, diagnostics). The diagnostics carry the number of
        restored components, the SHAKE residual and iteration count, and the
        per-atom shift in Bohr. An empty plan returns the input array and an
        empty diagnostic mapping.
        """
        array = np.array(coords, dtype=np.float64, copy=True)
        diagnostics = {'projected': False, 'restored_components': 0,
                       'iterations': 0, 'max_residual': 0.0, 'converged': True,
                       'shift': None}
        if self.is_noop():
            diagnostics['shift'] = array
            return array, {'projected': False}
        mask = projection.build_fix_mask(self.plan, array.shape[0])
        values = np.asarray(masses, dtype=np.float64).reshape(-1) \
            if masses is not None else None
        if mask.any():
            anchors = self._anchor_geometry(state, array, mask)
            array, moved = projection.apply_fix_positions(array, anchors, mask)
            diagnostics['restored_components'] = int(moved)
            diagnostics['projected'] = True
        pairs, targets = self._shake_requests(state)
        if pairs.size or self.has_shape_constraints():
            if values is None:
                raise ControllerError(
                    'a bond constraint needs per-atom masses to share the '
                    'correction')
            frozen = mask if mask.any() else None
            anchor_geometry = (self._anchor_geometry(state, array, mask)
                               if mask.any() else None)
            if self.has_shape_constraints():
                conditions = self._constraint_conditions(state)
                solution = shake.solve_conditions(
                    array, conditions, values,
                    tolerance=self._shake_tolerance(),
                    max_iterations=self._shake_iterations(),
                    fix_mask=frozen, anchors=anchor_geometry)
                diagnostics['conditions'] = int(len(conditions))
                diagnostics['rank'] = int(solution['redundant']['rank'])
                diagnostics['redundant'] = [
                    int(position) + 1
                    for position in solution['redundant']['redundant']]
                diagnostics['worst'] = solution['worst']
            else:
                solution = shake.solve_positions(
                    array, pairs, targets, values,
                    tolerance=self._shake_tolerance(),
                    max_iterations=self._shake_iterations(),
                    fix_mask=frozen, anchors=anchor_geometry)
            array = solution['coordinates']
            diagnostics['iterations'] = int(solution['iterations'])
            diagnostics['max_residual'] = float(solution['max_residual'])
            diagnostics['converged'] = bool(solution['converged'])
            diagnostics['shift'] = solution['shift']
            diagnostics['projected'] = True
        return array, diagnostics

    def project_velocities(self, velocities, state=None, context=None,
                           masses=None, coords=None):
        """Zero the velocity of every frozen component.

        A constraint record that declares `velocity_constraint: rattle` also
        gets its velocities projected onto the tangent space of the same bond
        lengths, which is the half `solve_positions` deliberately leaves alone.
        The projection reads the geometry the positions were projected to, so a
        caller that has a projected structure passes it as `coords`; without one
        there is no Jacobian to build, and guessing the input geometry would
        project onto a constraint manifold the run is not on.

        With no record asking for it, this is the same three lines it was before
        the switch existed, returning the same array.
        """
        array = np.asarray(velocities, dtype=np.float64)
        if self.is_noop():
            return array
        mask = projection.build_fix_mask(self.plan, array.shape[0])
        # Two shapes of request: a bond plan projects pair rows through
        # `solve_velocities`, exactly as it always did; a plan that also
        # constrains an angle or a torsion projects the whole condition set
        # through `solve_velocity_conditions`.
        conditions = self._tangent_conditions(state)
        tangent = conditions
        if tangent is None:
            tangent = self._tangent_requests(state)
        if not mask.any() and tangent is None:
            return array
        working = (projection.zero_fixed_velocities(array, mask) if mask.any()
                   else array)
        if tangent is None:
            return working
        rows, tolerance, iterations = tangent
        if coords is None:
            raise ControllerError(
                'a rattle velocity projection needs the geometry it constrains: '
                'pass the projected coordinates')
        if masses is None:
            raise ControllerError(
                'a rattle velocity projection needs per-atom masses to share '
                'the impulse')
        geometry = np.asarray(coords, dtype=np.float64)
        weights = np.asarray(masses, dtype=np.float64).reshape(-1)
        frozen = mask if mask.any() else None
        if conditions is not None:
            solution = shake.solve_velocity_conditions(
                geometry, rows, working, weights,
                tolerance=tolerance, max_iterations=iterations,
                fix_mask=frozen)
        else:
            solution = shake.solve_velocities(
                geometry, rows, working, weights,
                tolerance=tolerance, max_iterations=iterations,
                fix_mask=frozen)
        return solution['velocities']

    # ------------------------------------------------------------- potentials

    def evaluate_potentials(self, coords, state, context, masses=None):
        """Sum every additive potential at one geometry.

        Pure with respect to history: no hill is deposited, no schedule moves.
        The returned result carries the correction only; `combine` turns it
        into
        totals.
        """
        array = np.asarray(coords, dtype=np.float64)
        natom = array.shape[0]
        if self.is_noop():
            # No coordinates either: the caller holds the geometry already,
            # and a key here would be written into every history frame and
            # every dispatch payload of every unconstrained run.
            return ConstraintResult.empty(context)
        gradient = np.zeros((natom, 3), dtype=np.float64)
        energy = 0.0
        components = {}
        values = np.asarray(masses, dtype=np.float64).reshape(-1) \
            if masses is not None else None
        for spec in self.plan.actions:
            if spec.kind == 'bias':
                item = bias.evaluate(spec, self.plan, array, state, context)
            elif spec.kind == 'wall':
                item = wall.evaluate(spec, self.plan, array, state, context,
                                     masses=values)
            elif spec.kind == 'metadynamics':
                hills = metadynamics.collect_hills(state, spec.id)
                item = metadynamics.evaluate(
                    spec, self.plan, array, hills, context,
                    reference=self._rmsd_reference(spec, state), masses=values)
            else:
                continue
            components[spec.id] = item
            energy += float(item['energy'])
            gradient += item['gradient']
        result = ConstraintResult(
            coordinates=array.copy(), energy_correction=float(energy),
            gradient_correction=gradient, components=components,
            provenance={'evaluation_id': getattr(context, 'evaluation_id', ''),
                        'step': int(getattr(context, 'step', 0)),
                        'history_version': int(
                            getattr(state, 'history_version', 0)),
                        'plan_fingerprint': self.plan.fingerprint,
                        'schedule_step': int(getattr(context, 'step', 0))})
        return result

    def combine(self, base, potential, projection_info=None, context=None):
        """Add the correction to an electronic result, once.

        Parameters
        ----------
        base : dict or None
            'energy' and 'gradient' of the geometry being reported, in Hartree
            and Hartree per Bohr. None is legitimate: a geometry-handling call
            may want the correction alone.
        potential : ConstraintResult
            Output of `evaluate_potentials` for the same geometry.
        projection_info : dict, optional
            Output of `project_positions`, used for provenance.
        context : EvaluationContext, optional

        Returns
        -------
        ConstraintResult
            A fresh result carrying base, correction and totals. Nothing is
            accumulated onto a previous total: totals are always rebuilt from
            the stored base, which is what makes a repeated call idempotent.
        """
        result = self._copy_result(potential)
        base_energy, base_gradient = self._base_of(base)
        natom = int(result.coordinates.shape[0]) if \
            result.coordinates is not None else 0
        if base_gradient is not None and natom and \
                base_gradient.shape[-2:] != (natom, 3):
            raise ControllerError(
                'base gradient shape %s does not describe the %d atoms of this '
                'evaluation' % (base_gradient.shape, natom))
        states = 1 if base_gradient is None else (
            base_gradient.shape[0] if base_gradient.ndim == 3 else 1)
        if base_gradient is not None and base_gradient.ndim == 3 and \
                base_energy is not None and \
                np.asarray(base_energy).reshape(-1).size not in (1, states):
            raise ControllerError(
                'the base energy holds %d values but its gradient holds %d '
                'state rows' % (np.asarray(base_energy).reshape(-1).size,
                                states))
        result.energy_base = base_energy
        result.gradient_base = base_gradient
        correction = result.energy_correction
        if base_energy is not None:
            result.energy_total = base_energy + correction
        else:
            result.energy_total = None if correction == 0.0 else float(correction)
        if base_gradient is not None:
            added = self._added_to_states(result.gradient_correction, states)
            if added is None:
                result.gradient_total = base_gradient.copy()
            else:
                result.gradient_total = base_gradient + added
        else:
            # A bias gradient alone is not the gradient of this run: nothing
            # electronic stands behind it, so publishing it as `gradient` would
            # let an integrator step on a force that no wavefunction produced.
            result.gradient_total = None
        if projection_info is not None:
            result.diagnostics['projection'] = {
                key: value for key, value in projection_info.items()
                if key != 'shift'}
        result.gradient_for_integrator = (
            None if result.gradient_total is None
            else result.gradient_total.copy())
        result.gradient_projected = self._project_gradient(result)
        result.provenance.setdefault(
            'evaluation_id', getattr(context, 'evaluation_id', ''))
        result.provenance['nstates'] = states
        return result

    # -------------------------------------------------------------- counting

    def hessian_block(self, coords, state, context=None, masses=None):
        """The analytic curvature the declared additive potentials produce.

        `NumericalHessian` already sees the bias force, because every displaced
        gradient passes through the finalizer, so the constrained curvature is
        in a numerical Hessian today. This method is the same number without the
        difference step: `d2V/dR2 = k (J J^T + chain e Q)` for a harmonic bias
        over a bond or an angle, exact where the finite difference is only
        O(h^2)-accurate and twice as noisy near a stiff force constant.

        Included: a harmonic bias over a bond, an angle or a torsion, a wall of
        any of the four geometries (atom, centre-of-geometry or centre-of-mass
        target), and a history potential over its declared variables.

        Deliberately absent: a fix and a strict constraint, which carry no
        potential of their own - their response is the projection, not a
        curvature term - and an aligned-RMSD history, whose optimal rotation
        moves with the geometry so its curvature is a coupled problem; that one
        is refused by name rather than dropped, because a caller that adds a
        partial block to a numerical Hessian would double count what the
        displaced gradients already carry while silently omitting what they
        do not.
        Returns
        -------
        dict
            'hessian' shaped (3 * natom, 3 * natom) in Hartree per Bohr squared,
            'ids' the actions summed into it, and 'size'.
        """
        array = np.asarray(coords, dtype=np.float64)
        natom = array.shape[0]
        block = np.zeros((natom * 3, natom * 3), dtype=np.float64)
        values = np.asarray(masses, dtype=np.float64).reshape(-1) \
            if masses is not None else None
        ids = []
        for spec in self.plan.actions:
            if spec.kind in ('fix', 'constraint'):
                continue
            if spec.kind == 'wall':
                if spec.parameters.get('target_kind') == 'com' \
                        and values is None:
                    raise ControllerError(
                        "the analytic Hessian of wall '%s' targets the center of "
                        'mass and the state carries no masses' % spec.id)
                block += wall.hessian(spec, self.plan, array, state, context,
                                       masses=values)['hessian']
                ids.append(spec.id)
                continue
            if spec.kind == 'metadynamics':
                block += metadynamics.hessian(
                    spec, self.plan, array,
                    metadynamics.collect_hills(state, spec.id), context,
                    reference=self._rmsd_reference(spec, state),
                    masses=values)['hessian']
                ids.append(spec.id)
                continue
            if spec.kind != 'bias':
                raise ControllerError(
                    "the analytic Hessian does not know the action kind %r"
                    % spec.kind)
            item = bias.hessian(spec, self.plan, array, state, context)
            block += item['hessian']
            ids.append(spec.id)
        return {'hessian': block, 'ids': ids, 'size': natom * 3}

    def statistics(self, coords, velocities, masses, state=None, context=None):
        """Free degrees of freedom and free kinetic energy at one geometry.

        With no action declared this is the plain mobile count and the plain
        kinetic energy, which is the same number the framework reports today;
        the difference appears only when a fix or a bond constraint exists.
        """
        values = np.asarray(masses, dtype=np.float64).reshape(-1)
        if self.is_noop():
            return projection.statistics(values, velocities=velocities)
        mask = projection.build_fix_mask(self.plan, values.size)
        pairs, _ = self._shake_requests(state)
        # A plan with an angle or a torsion is counted from the full condition
        # list, which already contains the bond rows: adding the pair rows on
        # top of it would charge each bond twice.
        conditions = (self._constraint_conditions(state)
                      if self.has_shape_constraints() else None)
        return projection.statistics(
            values, velocities=velocities, mask=mask,
            pairs=None if conditions is not None else (
                pairs if pairs.size else None),
            coords=np.asarray(coords, dtype=np.float64),
            conditions=conditions)

    # ---------------------------------------------------------------- commit

    def measure_recorded(self, coords, state):
        """How far one geometry stands from what this plan declares.

        This is the question a recorded frame is asked with: every strict
        condition against the target it was bound to, and every frozen
        component against the anchor it was restored to. The units are the
        solver's own (Bohr for a length, radians for an angle or a torsion) and
        the allowance for each is the run's SHAKE tolerance times the magnitude
        of the target, because the solver's contract is a relative residual.

        It measures, it does not solve: `solve_conditions` is what a step's
        projection calls, and a check that projected the geometry again would
        be reporting on a structure the run never recorded.

        Returns
        -------
        tuple
        (ratio, deviation, note). `ratio` is the worst excess over the
        allowance, in units of that allowance, so anything above 1.0 is a
        declared condition the recorded geometry does not satisfy; `deviation`
        is that item's distance in its own unit; `note` names the variable. A
        plan that declares nothing measurable gives (0.0, 0.0, '').
        """
        array = np.asarray(coords, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            return 0.0, 0.0, ''
        tolerance = self._shake_tolerance()
        worst_ratio = 0.0
        worst_deviation = 0.0
        worst_note = ''
        conditions = self._constraint_conditions(state)
        if conditions and all(kind == 'bond' for kind, _atoms, _t in
                              conditions):
            # The common case is a bond set, and a bond set is measured with
            # the one vectorized call the solver itself uses. Asked atom by
            # atom, 40 O-H bonds of a 60-atom box cost 206 us of interpreter
            # overhead per recorded step; through `shake.distances` the same
            # 40 cost a fraction of that, and a check that is priced out of
            # the record cadence is a check that does not run.
            rows = np.asarray([atoms for _kind, atoms, _t in conditions],
                              dtype=np.int64)
            targets = np.asarray([t for _kind, _atoms, t in conditions],
                                 dtype=np.float64)
            values = shake.distances(array, rows)
            allowed = tolerance * np.maximum(np.abs(targets), 1.0)
            ratios = np.abs(values - targets) / allowed
            position = int(np.argmax(ratios))
            if float(ratios[position]) > worst_ratio:
                worst_ratio = float(ratios[position])
                worst_deviation = float(abs(values[position] - targets[position]))
                worst_note = ('the declared bond of atoms %s reads %.10g '
                              'against its target %.10g (allowed %.1e)'
                              % (conditions[position][1],
                                 float(values[position]),
                                 float(targets[position]),
                                 float(allowed[position])))
        else:
            for kind, atoms, target in conditions:
                value = float(shake.condition_value(array, kind, atoms))
                deviation = abs(value - target)
                allowed = tolerance * max(abs(target), 1.0)
                ratio = deviation / allowed if allowed > 0.0 else 0.0
                if ratio > worst_ratio:
                    worst_ratio = ratio
                    worst_deviation = deviation
                    worst_note = ('the declared %s of atoms %s reads %.10g '
                                  'against its target %.10g (allowed %.1e)'
                                  % (kind, atoms, value, target, allowed))
        mask = projection.build_fix_mask(self.plan, array.shape[0])
        if mask is not None and mask.any():
            anchors = self._anchor_geometry(state, array, mask)
            deviation = float(np.max(np.abs(array - anchors)[mask]))
            allowed = tolerance * max(
                float(np.max(np.abs(anchors[mask]))), 1.0)
            ratio = deviation / allowed if allowed > 0.0 else 0.0
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_deviation = deviation
                worst_note = ('a frozen component sits %.10g away from the '
                              'anchor it was restored to (allowed %.1e)'
                              % (deviation, allowed))
        return worst_ratio, worst_deviation, worst_note

    def commit(self, coords, state, context, masses=None):
        """Accept a step: deposit the hills it earns and record the step.

        Returns a new state. A step that earns nothing returns a copy with the
        step counter moved, so the caller can always use the result.
        """
        array = np.asarray(coords, dtype=np.float64)
        fresh = state.copy()
        fresh.last_committed_step = int(getattr(context, 'step', 0))
        if self.is_noop():
            return fresh
        values = np.asarray(masses, dtype=np.float64).reshape(-1) \
            if masses is not None else None
        for spec in self.plan.actions:
            if spec.kind != 'metadynamics':
                continue
            if not metadynamics.should_deposit(spec, self.plan, context,
                                                 fresh):
                continue
            hill = metadynamics.hill_at(spec, self.plan, array, context,
                                        state=fresh,
                                        reference=self._rmsd_reference(
                                            spec, fresh),
                                        masses=values)
            if hill is None:
                # A well-tempered bias that has flattened to nothing has no hill
                # to write. The token is still spent, so the step is not
                # re-attempted, and the count is recorded because a history
                # potential that stopped growing is a fact about the run that
                # must not be invisible.
                token = metadynamics.deposition_token(fresh, spec, context)
                fresh = fresh.mark(token)
                key = 'zero_hills:%s' % spec.id
                fresh.diagnostics[key] = int(fresh.diagnostics.get(key, 0)) + 1
                continue
            fresh = fresh.mark(hill.token).with_hill(hill)
        return fresh

    # ------------------------------------------------------------ convenience

    def apply_structure(self, coords, state=None, context=None, masses=None,
                        evaluator=None, base=None):
        """Project, optionally re-evaluate the base, then combine.

        `evaluator` is a callable taking the projected coordinates and
        returning a base mapping with 'energy' and 'gradient'. When it is
        given,
        `base` must not be, because a base that belongs to a different geometry
        cannot be repaired by adding a correction to it. When neither is given
        the caller only gets the projected geometry and the correction.
        """
        array = np.asarray(coords, dtype=np.float64)
        bound = state
        if bound is None:
            bound = self.bind_reference(array, masses=masses)
        projected, info = self.project_positions(array, bound, context,
                                                 masses=masses)
        if evaluator is not None:
            if base is not None:
                raise ControllerError(
                    "apply_structure got both an evaluator and a base result; "
                    "the base must describe the projected geometry, so give "
                    "one or the other")
            base = evaluator(projected)
        self._check_base_geometry(base, projected)
        potential = self.evaluate_potentials(projected, bound, context,
                                             masses=masses)
        result = self.combine(base, potential, info, context)
        result.coordinates = projected
        result.delta_coordinates = projected - array
        return result

    # --------------------------------------------------------------- helpers

    def _check_base_geometry(self, base, projected):
        """Refuse to pair a base result with a geometry it does not describe.

        A base may carry the coordinates it was computed at; the design
        document makes that check mandatory, because adding a correction for
        R* to an energy evaluated at R and labelling it with R* is the single
        most expensive mistake this subsystem can make. A base that names no
        geometry is left alone: the framework finalizer is the caller that
        guarantees the pairing by construction, and it does not copy
        coordinates into its result.
        """
        if not isinstance(base, dict):
            return
        declared = base.get('geometry')
        if declared is None:
            return
        array = np.asarray(declared, dtype=np.float64)
        if array.shape != np.asarray(projected).shape \
                or not np.allclose(array, projected, atol=GEOMETRY_ATOL,
                                   rtol=0.0):
            raise ControllerError(
                'the base result was computed at a different geometry than '
                'the projected one; re-evaluate it before combining')

    def _check_plan(self):
        """Refuse a plan whose action cannot be carried out.

        Reference integrity (cv, group and schedule names) is already enforced
        when the plan is built, so what is left to check here is the shape of
        an action: a bond constraint with no pairs is a plan the solver would
        otherwise silently do nothing about.
        """
        for spec in self.plan.actions:
            if spec.kind == 'constraint' and spec.family == 'shake' \
                    and spec.pairs is None:
                raise ControllerError(
                    "constraint '%s' carries no atom pairs" % spec.id)
            if spec.kind == 'constraint' and spec.family != 'shake' \
                    and not spec.parameters.get('rows'):
                raise ControllerError(
                    "constraint '%s' of kind %s carries no atom rows"
                    % (spec.id, spec.family))
            if spec.kind == 'metadynamics' and spec.family == 'gaussian' \
                    and not spec.parameters.get('cv_names'):
                raise ControllerError(
                    "metadynamics '%s' binds no cv" % spec.id)
            if spec.kind == 'metadynamics' and spec.family == 'rmsd':
                members = 0 if spec.atoms is None else len(spec.atoms)
                if members < 2:
                    raise ControllerError(
                        "metadynamics '%s' aligns %d atoms; a deviation "
                        "against a reference needs at least two"
                        % (spec.id, members))

    def _rmsd_reference(self, spec, state):
        """The geometry an aligned RMSD measures against, or None.

        An explicitly declared reference lives on the plan; one bound at the
        start of the trajectory lives on the state. Reading it from the live
        coordinates instead would make the variable identically zero.
        """
        if spec.kind != 'metadynamics' or spec.family != 'rmsd':
            return None
        if spec.reference == 'explicit':
            return np.asarray(spec.target, dtype=np.float64)
        stored = getattr(state, 'targets', {}).get(spec.id)
        if stored is None:
            raise ControllerError(
                "metadynamics '%s' binds its reference geometry at the start "
                "of the trajectory and nothing bound it" % spec.id)
        return np.asarray(stored, dtype=np.float64)

    def _cv_of(self, spec):
        if spec.cv is None or spec.cv not in self.plan.cvs:
            raise ControllerError(
                "action '%s' has no resolvable cv" % spec.id)
        return self.plan.cvs[spec.cv]

    def _tangent_requests(self, state):
        """The pairs that want their velocities projected, and how tightly.

        Returns None when no declared constraint asks for `rattle`, which is the
        case every run today is in: the switch defaults to `none`, so the default
        path never reaches `shake.solve_velocities`. Tolerance and iteration cap
        are read from the records that opted in, with the same convention as the
        position solve.

        `state` is taken for symmetry with the other request builders; the
        velocity condition needs no bound target, only the geometry.
        """
        pairs = []
        tolerances = []
        iterations = []
        for spec in self.plan.actions:
            if spec.kind != 'constraint':
                continue
            if spec.diagnostics.get('velocity_constraint', 'none') != 'rattle':
                continue
            if spec.family != 'shake':
                # An angle or torsion record asks for the multi-condition
                # projection, which is built over the whole condition set in
                # `_tangent_conditions`; a pair row is not the right shape there.
                continue
            rows = np.asarray(spec.pairs, dtype=np.int64).reshape(-1, 2)
            if not rows.size:
                continue
            pairs.append(rows)
            tolerances.append(float(spec.diagnostics.get('tolerance', 1.0e-8)))
            iterations.append(int(spec.diagnostics.get('max_iterations', 200)))
        if not pairs:
            return None
        return np.vstack(pairs), max(tolerances), max(iterations)

    def _tangent_conditions(self, state):
        """The full condition set, when any non-bond record wants RATTLE.

        A plan that constrains both a length and an angle is one manifold: the
        bond correction moves the angle's atoms and the angle correction moves the
        bond's atoms, so projecting only the bond velocities of such a plan would
        report a RATTLE run while the angle kept absorbing energy. Once any record
        in it asks for the velocity half, the whole set is projected.

        Returns
        -------
        tuple or None
            (conditions, tolerance, max_iterations), or None when no non-bond
            record declared `velocity_constraint: rattle`.
        """
        wanted = [spec for spec in self.plan.actions
                if spec.kind == 'constraint'
                and spec.family != 'shake'
                and spec.diagnostics.get('velocity_constraint',
                                          'none') == 'rattle']
        if not wanted:
            return None
        conditions = self._constraint_conditions(state)
        tolerances = [float(spec.diagnostics.get('tolerance', 1.0e-8))
                    for spec in self.plan.actions
                    if spec.kind == 'constraint']
        iterations = [int(spec.diagnostics.get('max_iterations', 200))
                    for spec in self.plan.actions
                    if spec.kind == 'constraint']
        return (conditions, max(tolerances), max(iterations))

    def _shake_requests(self, state):
        """Effective pairs and their targets, in Bohr.

        Only bond specs come through here: an angle or a torsion is a condition
        of three or four atoms and has no pair row, and it is gathered by
        `_constraint_conditions` with the rest.
        """
        pairs = []
        targets = []
        for spec in self.plan.actions:
            if spec.kind != 'constraint' or spec.family != 'shake':
                continue
            rows = np.asarray(spec.pairs, dtype=np.int64).reshape(-1, 2)
            if spec.reference == 'initial':
                stored = getattr(state, 'targets', {}).get(spec.id)
                if stored is None:
                    raise ControllerError(
                        "constraint '%s' binds its target at the reference "
                        "geometry and nothing bound it" % spec.id)
                values = np.asarray(stored, dtype=np.float64).reshape(-1)
            else:
                values = np.full(rows.shape[0], float(spec.target))
            pairs.append(rows)
            targets.append(values)
        if not pairs:
            return np.zeros((0, 2), dtype=np.int64), np.zeros(0)
        return np.vstack(pairs), np.concatenate(targets)

    def has_shape_constraints(self):
        """True when the plan constrains an angle or a torsion, not only lengths."""
        return any(spec.kind == 'constraint' and spec.family != 'shake'
                   for spec in self.plan.actions)

    def condition_records(self, state):
        """Every strict condition, as declared, with the id it was written under.

        One record per atom row: an object line that holds four bond lengths is
        four conditions, and a run that reports "the constraint is satisfied"
        has to say which one it means. Bond rows are included beside angle and
        torsion rows because a plan that holds both is one manifold: the two
        sweeps share the atoms, and a report that left the lengths out would
        describe half of it.
        """
        out = []
        for spec in self.plan.actions:
            if spec.kind != 'constraint':
                continue
            kind = 'bond' if spec.family == 'shake' else str(
                spec.parameters.get('cv_kind', spec.family))
            if spec.family == 'shake':
                rows = np.asarray(spec.pairs, dtype=np.int64).reshape(-1, 2)
            else:
                rows = np.asarray(spec.parameters.get('rows') or [],
                                  dtype=np.int64).reshape(
                                      -1, shake.CONDITION_WIDTH[kind])
            if spec.reference == 'initial':
                stored = getattr(state, 'targets', {}).get(spec.id)
                if stored is None:
                    raise ControllerError(
                        "constraint '%s' binds its target at the reference "
                        "geometry and nothing bound it" % spec.id)
                values = np.asarray(stored, dtype=np.float64).reshape(-1)
            else:
                values = np.full(rows.shape[0], float(spec.target))
            for position, row in enumerate(rows):
                out.append({'id': str(spec.id), 'kind': kind,
                            'atoms': [int(atom) for atom in row],
                            'target': float(values[position])})
        return out

    def _constraint_conditions(self, state):
        """Every strict condition as (kind, atoms, target), in plan order.

        The shape the solver takes; `condition_records` is where the same read
        keeps the ids for a report. Bond rows are included beside length-only
        plans because two separate sweeps would move the same shared atom twice
        per pass, and the pair length an angle solve reads would then be the one
        the angle just disturbed.
        """
        return [(item['kind'], item['atoms'], item['target'])
                for item in self.condition_records(state)]

    def _bind_constraint_targets(self, spec, array):
        """The values a strict constraint binds to at the reference geometry.

        A bond set is measured with the one call it has always used. An angle or
        a torsion is read through `shake.condition_value`, and a reference
        geometry that is collinear or exactly eclipsed fails here by name: the
        target of a constraint that cannot be differentiated at the structure it
        was declared on is not a number this subsystem can invent.
        """
        if spec.family == 'shake':
            return shake.distances(array, spec.pairs).tolist()
        kind = str(spec.parameters.get('cv_kind', spec.family))
        rows = np.asarray(spec.parameters.get('rows') or [],
                          dtype=np.int64).reshape(-1,
                                                  shake.CONDITION_WIDTH[kind])
        values = []
        for row in rows:
            try:
                values.append(float(shake.condition_value(
                    array, kind, [int(atom) for atom in row])))
            except ValueError as error:
                raise ControllerError(
                    "constraint '%s' cannot bind its target over atoms %s: %s"
                    % (spec.id, [int(atom) + 1 for atom in row], error)) from error
        return values

    def _shake_tolerance(self):
        values = [float(spec.diagnostics.get('tolerance', 1.0e-8))

                  for spec in self.plan.actions if spec.kind == 'constraint']
        return max(values) if values else 1.0e-8

    def _shake_iterations(self):
        values = [int(spec.diagnostics.get('max_iterations', 200))
                  for spec in self.plan.actions if spec.kind == 'constraint']
        return max(values) if values else 200

    def _anchor_geometry(self, state, array, mask):
        """The reference geometry frozen components return to."""
        for spec in self.plan.actions:
            if spec.kind != 'fix':
                continue
            stored = getattr(state, 'anchors', {}).get(spec.id)
            if stored is not None and np.shape(stored) == np.shape(array):
                return np.asarray(stored, dtype=np.float64)
            raise ControllerError(
                "fix '%s' has no anchor geometry in this run; freezing it "
                "at the current position would silently move its reference"
                % spec.id)
        return array.copy()

    def _bind_wall_centers(self, array, state, masses):
        """Capture the center every wall is fixed to, once, at the reference.

        The rule itself is `wall.bind_center`: the declared coordinates win, then
        the atoms `wall_ref` names, then the wall's own selection. The center is
        stored per wall, keyed by the wall id, because two walls with different
        targets would otherwise overwrite one another and the second would
        confine the first wall's group.
        """
        values = np.asarray(masses, dtype=np.float64).reshape(-1) \
            if masses is not None else None
        for spec in self.plan.actions:
            if spec.kind != 'wall' or spec.parameters.get('center') is not None:
                continue
            state.anchors['center:%s' % spec.id] = wall.bind_center(
                spec, self.plan, array, masses=values)

    def _base_of(self, base):
        """Energy and gradient out of a base mapping, without inventing any.

        Shapes follow the framework's own result contract (design document
        section 9):

        - energy: a scalar, or one value per electronic state as a `(ns,)`
          array. An added potential is state independent, so it shifts every
          state by the same amount and leaves the gaps untouched.
        - gradient: `(natom, 3)` for the active-combination gradient a
          multi-center run integrates with, or `(ns, natom, 3)` for the
          state-resolved result gradient of a single-center run. Absence is
          carried by the key, not by a row: `data/state.py` returns None for a missing gradient rather than a partly-populated array, so a run that
          never computed a gradient stays without one. Adding the bias gradient
          to nothing would publish a force that no electronic structure backs.
        """
        if base is None:
            return None, None
        if not isinstance(base, dict):
            raise ControllerError(
                'the base result must be a mapping with energy and gradient')
        energy = base.get('energy')
        gradient = base.get('gradient')
        if energy is not None:
            array = np.asarray(energy, dtype=np.float64)
            energy = float(array) if array.ndim == 0 else array
        if gradient is not None:
            gradient = np.asarray(gradient, dtype=np.float64)
            if gradient.ndim not in (2, 3):
                raise ControllerError(
                    'base gradient must be (natom, 3) or (ns, natom, 3), got '
                    'shape %s' % (gradient.shape,))
        return energy, gradient

    def _added_to_states(self, array, count):
        """Broadcast one additive quantity over `count` electronic states.

        A `(natom, 3)` additive gradient joins a `(ns, natom, 3)` base by
        repetition along the state axis, which is what a state-independent
        potential means; the reverse shape combination is a declaration error
        rather than something to guess at.
        """
        if array is None:
            return None
        if count == 1:
            return array
        if array.ndim == 3 and array.shape[0] != count:
            raise ControllerError(
                'the base gradient holds %d state rows but the energy names %d'
                % (array.shape[0], count))
        return np.repeat(array[np.newaxis, ...], count, axis=0)


    def _copy_result(self, source):
        """A detached copy, so combine never mutates an evaluation result."""
        return ConstraintResult(
            coordinates=None if source.coordinates is None
            else np.array(source.coordinates, dtype=np.float64, copy=True),
            velocities=None if source.velocities is None
            else np.array(source.velocities, dtype=np.float64, copy=True),
            delta_coordinates=source.delta_coordinates,
            energy_correction=source.energy_correction,
            gradient_correction=None if source.gradient_correction is None
            else np.array(source.gradient_correction, dtype=np.float64,
                          copy=True),
            components=dict(source.components),
            diagnostics=dict(source.diagnostics),
            provenance=dict(source.provenance))

    def _project_gradient(self, result):
        """Zero the frozen components of the total gradient.

        This is the driving gradient an optimizer may step along. It is not a
        derivative of the total potential with respect to the Cartesian
        coordinates, which is why it keeps its own name. A state-resolved
        `(ns, natom, 3)` array is masked along its last two axes, so every
        state loses the same frozen components.
        """
        total = result.gradient_total
        if total is None:
            return None
        if self.is_noop():
            return total.copy()
        mask = projection.build_fix_mask(self.plan, total.shape[-2])
        projected = total.copy()
        if total.ndim == 3:
            projected[:, mask] = 0.0
        else:
            projected[mask] = 0.0
        return projected


def controller_for(plan):
    """Build a controller, or None when the plan asks for nothing."""
    if plan is None or plan.is_empty:
        return None
    return SystemConstraintController(plan)
