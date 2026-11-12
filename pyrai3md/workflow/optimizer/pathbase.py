"""
======================================================================
This module provides the base classes of the path family: the searches
that walk more than one structure.

    PathOptimizer        the joint vector (atoms, and the six relaxed
                         lattice entries of a variable-cell run), the
                         normalized mass weights, the geometry in and
                         out of the state, and the result block
    WalkerPathOptimizer  the macro/micro state machine of `mep` and `irc`:
                         one path point is one constrained minimization on
                         a sphere, and the sphere is re-centred on every
                         accepted point
    BandPathOptimizer    the images of a nudged elastic band, the FIRE
                         state that relaxes them, and the batch they are
                         evaluated in

They are not `Optimizer`s in the sense of the other searches: a path
optimizer is driven by the Path workflow (`workflow/path.py`), which owns
the calculator and hands one evaluation at a time to `observe` (walker) or
a whole round of images to `band_collect` (band). `run(state)` and
`_compute_step` are refused by name, because a step from one geometry to
the next is not what these classes compute.

What is reused rather than reimplemented: the cell half of the search.
`Optimizer._apply_step` writes a joint step with the atoms riding the box,
`_cell_gradient` is the stress conjugate of a lattice entry including the
applied pressure, `_enthalpy` is the objective under that pressure,
`_cell_direction` is the only direction an isotropic box may move, and
`apply_constraint`/`note_evaluation` are the projection and the tangential
split a declared condition needs. A path that ignored them would move
pinned atoms and judge gradients a condition already accounts for.

The weights, and the one place they are unusual: `sqrt(m_i/M)` per atomic
component, normalized so that `opt_path_radius` is a plain length in Bohr,
and 1 per lattice entry because a box has no mass. That is the same
convention `_apply_trust_radius` measures a joint step in.

Nothing here evaluates a potential.

Oct 20 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import copy

import numpy as np

from .base import Optimizer
from .pathconstraint import make_constraint
from ...data.geometry import continuous_coordinates
from ...utilities.lattice import (CELL_COMPONENT_SLOTS, cart_to_frac,
                                  frac_to_cart, normalize_pbc,
                                  wrap_fractional_with_image)

# How much higher than the previous accepted point an accepted point may be
# before the walk stops: an intrinsic reaction coordinate that starts to climb
# has entered the basin on the other side of its minimum, and the point that
# climbed is not part of the path. The floor is in Hartree (the enthalpy's unit
# too) and is there for the rounding of a run that has flattened out.
_UPHILL_FLOOR = 1.0e-9

# The FIRE integrator's constants (Bitzek et al., Phys. Rev. Lett. 97, 170201
# (2006)). `neb_step` seeds the timestep; the rest are the algorithm itself and
# are not keywords. `dt_max` is expressed against that seed.
_FIRE_ALPHA_START = 0.1
_FIRE_N_MIN = 5
_FIRE_F_INC = 1.1
_FIRE_F_DEC = 0.5
_FIRE_F_ALPHA = 0.99
_FIRE_DT_MAX_FACTOR = 10.0

# The largest single-image displacement one FIRE step may take, in Bohr. A
# climbing image whose parallel force is still large would otherwise be flung
# into a geometry the backend cannot evaluate.
_BAND_MAX_MOVE = 0.2


class PathOptimizer(Optimizer):
    """Shared geometry of the path family: the joint vector and the weights.

    The joint vector is `(3N,)` atomic coordinates in Bohr, followed by the six
    relaxed lattice entries when `cell_variables` is set - the same spelling and
    the same order `optimizer/base.py::_extend_to_cell` builds for a
    variable-cell search, so the two step on one vector and the curvature
    batch's `(3N + 6)^2` matrix is usable by a path as it is.
    """

    # A registered name here says the algorithm exists; `is_path` says which
    # driver may use it. The single-structure loop refuses a path by this flag
    # rather than by falling into a `_compute_step` that cannot answer.
    is_path = True
    path_kind = 'walker'
    is_implemented = True
    # `irc` overrides this: it needs the curvature of its transition state
    # before its first step, and says so where the input is checked.
    needs_initial_hessian = False
    uses_state_pair = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The factor that turns the skill's gradient into Hartree per Bohr. The
        # Path workflow resolves it from the skill manifest and sets it here,
        # the way the Optimize loop applies it to every evaluation before the
        # optimizer sees one - and for the band it is the same factor, because a
        # child runtime hands its gradient back in the skill's own unit too.
        self.gradient_factor = 1.0
        # Why the walk stopped when it did. Reported in the run log and written
        # to the runtime's failure message when the path did not reach its end.
        self.stop_reason = ''
        # The joint vector's two halves, and the weights and mobility of it.
        self._atomic_size = 0
        self._base_lattice = None
        self._weights = None
        # The atomic block of `_weights` on its own. A seed that displaces the
        # atoms only - the IRC's, whose mode has no lattice components
        # (`irc.py::seed`) - needs the 3N weights, while `_weights` is the
        # joint 3N+6 one as soon as the box is a variable.
        self._atomic_weights = None
        self._mobile = None

    # ---- Geometry ----

    def _mass_weights(self, state):
        """`sqrt(m_i / M)` per atomic component, normalized.

        Normalized rather than the textbook `sqrt(m_i)` so that the arc length
        of a path point is a length in Bohr: the same unit `opt_trust_radius`
        and the cell half of a joint step are written in, with no amu factor
        appearing in the input. The weighting is then a pure shape factor.
        """
        system = getattr(getattr(state, 'pmd', None), 'system', None)
        masses = np.asarray(getattr(system, 'mass', None) or [],
                            dtype=np.float64).reshape(-1)
        coordinates = np.asarray(state.coordinates, dtype=np.float64)
        natom = coordinates.reshape(-1, 3).shape[0]
        if masses.size == 0:
            raise ValueError(
                'a path is walked on mass-weighted coordinates and this system '
                'carries no atomic masses')
        if masses.size != natom:
            raise ValueError(
                'the mass-weighted coordinates of a path need one mass per '
                'atom: this system carries %d masses for %d atoms'
                % (masses.size, natom))
        total = float(np.sum(masses))
        if not np.all(np.isfinite(masses)) or total <= 0.0:
            raise ValueError(
                'the masses of a path must be finite and positive, and this '
                'system carries %r' % (masses.tolist(),))
        return np.repeat(np.sqrt(masses / total), 3)

    def _joint_weights(self, state):
        """The weights of the whole joint vector, the box half included."""
        atomic = self._mass_weights(state)
        if not self.cell_variables:
            return atomic
        return np.concatenate((atomic, np.ones(len(CELL_COMPONENT_SLOTS))))

    def _mobile_mask(self, state):
        """Boolean mask of the joint components this run is allowed to move.

        The atomic half comes from the constraint plan's fix mask, which is the
        same reading `note_evaluation` takes, so a walk cannot move a component
        the convergence criteria are already excluding. The six lattice entries
        are always mobile: a run that froze one of them did not ask for a
        variable box.
        """
        from ..constraints import projection

        plan = state.constraint_plan() if hasattr(state, 'constraint_plan') \
            else None
        if plan is None or plan.is_empty:
            return None
        natom = int(np.asarray(state.coordinates, dtype=np.float64)
                    .reshape(-1, 3).shape[0])
        frozen = projection.build_fix_mask(plan, natom)
        if not frozen.any():
            return None
        atomic = ~np.asarray(frozen, dtype=bool).reshape(-1)
        if not self.cell_variables:
            return atomic
        return np.concatenate((atomic, np.ones(len(CELL_COMPONENT_SLOTS),
                                              dtype=bool)))

    @staticmethod
    def _continuous_atoms(state):
        """The atomic geometry in the frame this walk keeps, in Angstrom.

        A periodic state stores main-image coordinates plus a crossing ledger,
        and that ledger can already be populated **before the first step**: the
        input pipeline folds a molecule written across a cell face, so a water
        whose atoms were listed on both sides of the boundary is stored folded
        with a count that puts it back together. The walk's own frame is the
        continuous one - `_adopt` reads every evaluation back through the same
        ledger - so a seed built from the stored coordinates sits one box away
        from the geometry the first evaluation adopts. Measured on
        `path_mep_pbc`, whose 60-atom box carries five folded atoms: the start
        row read `radius 4.948` and `max_d 8.399` (the box edge), and the first
        accepted point sat 4.86 Bohr from the start instead of one
        `path_radius`. Reading the continuous view here puts the seed in the
        frame the walk is walked in; `apply_trial` still writes a *relative*
        step onto the stored geometry, so the backend keeps receiving the main
        image.

        No ledger (a non-periodic run, or one whose perception refused) returns
        the stored coordinates, which is the same geometry.
        """
        from ...data.geometry import continuous_coordinates

        coordinates = np.asarray(state.coordinates, dtype=np.float64)
        system = getattr(getattr(state, 'pmd', None), 'system', None)
        continuous = continuous_coordinates(coordinates, system)
        return coordinates if continuous is None else continuous

    def _joint_of(self, state):
        """`(joint vector in Bohr, the lattice in Bohr or None)`.

        The atomic block is the continuous reading (`_continuous_atoms`), which
        is the frame `_adopt` returns evaluations in.
        """
        coordinates = (self._continuous_atoms(state).reshape(-1)
                       * self.coord_to_bohr)
        lattice = self._cell_lattice(state)
        if lattice is None:
            return coordinates, None
        slots = lattice.reshape(-1)[list(CELL_COMPONENT_SLOTS)]
        return np.concatenate((coordinates, slots)), lattice

    def _lattice_of(self, joint):
        """The 3x3 lattice one joint vector describes, in Bohr."""
        if self._base_lattice is None:
            return None
        flat = np.asarray(self._base_lattice, dtype=np.float64).reshape(-1).copy()
        flat[list(CELL_COMPONENT_SLOTS)] = joint[self._atomic_size:]
        return flat.reshape(3, 3)

    def _joint_gradient(self, state, gradient, coordinates):
        """The joint gradient: the atomic half, plus the stress conjugate.

        A variable-cell run steps on the box as well, and the force conjugate to
        a lattice entry is not the atomic gradient: it is the stress read through
        `utilities/lattice.stress_to_cell_force`, with the applied pressure in it
        (`optimizer/base.py::_cell_gradient`). One surface, one tensor - a layout
        whose objective is built from two surfaces is refused by that method, and
        a path is one surface.
        """
        _coordinates, extended, _lattice = self._extend_to_cell(
            state, coordinates, gradient)
        return extended

    def _geometry_of(self, joint):
        """`(coordinates Angstrom, lattice Angstrom or None)` of a joint vector."""
        atomic = (np.asarray(joint, dtype=np.float64)[:self._atomic_size]
                  .reshape(-1, 3) * self.disp_to_angstrom)
        if not self.cell_variables:
            return atomic, None
        return atomic, self._lattice_of(joint) * self.disp_to_angstrom

    # ---- The contract the Path workflow drives ----

    def begin(self, state):
        """Read the geometry, the masses and the box, and seed the path."""
        raise NotImplementedError

    def finished(self):
        """Whether the path has walked as far as it will."""
        raise NotImplementedError

    def report(self):
        """One row for `path_status.txt`."""
        raise NotImplementedError

    def path_points(self):
        """The accepted path points, one dict each, for `path_profile.txt`."""
        raise NotImplementedError

    def results(self):
        """The `path_*` block this run writes into the state's results."""
        raise NotImplementedError

    def _compute_step(self, gradient):
        """Refused: a path optimizer is driven by the Path workflow.

        A step from one geometry to the next is what `Optimize.run` asks an
        optimizer for, and a walker's next geometry is not that: it is a probe
        inside one path point, decided from the radius the run is holding and
        from an evaluation the workflow has already made.
        """
        raise NotImplementedError(
            '%s is a path optimizer: it is driven by the Path workflow, not by '
            'Optimize.run(), because mep, irc and neb walk several structures'
            % type(self).__name__)

    def run(self, state):
        raise NotImplementedError(
            '%s is a path optimizer: it is driven by the Path workflow, not by '
            'Optimize.run(), because mep, irc and neb walk several structures'
            % type(self).__name__)

    def __repr__(self):
        return '%s(iteration=%d)' % (type(self).__name__, self.iteration)


class WalkerPathOptimizer(PathOptimizer):
    """The macro/micro state machine of `mep` and `irc`.

    One path point is one constrained minimization on the sphere of radius
    `path_radius` centered on the last accepted point, under one of the two
    constraint models (`pathconstraint`). The workflow evaluates the geometry
    `next_trial()` hands out, calls `note_evaluation` for the tangential split a
    declared condition needs, and calls `observe` with the energy and gradient
    that came back; `observe` either takes the next probe or accepts the point
    and re-centres the sphere.

    The walk ends when the sphere can advance no further (the objective stops
    decreasing, the point budget is reached) or when the probe budget of one
    point is spent without the constrained minimum being found, which is said in
    `stop_reason` rather than left as a path that quietly stopped descending.

    Parameters
    ----------
    path_radius : float
        The arc length of one accepted path point, in Bohr, on the normalized
        mass-weighted coordinates.
    inner_maxit : int
        How many probes one path point may spend before the walk gives up on
        finding its constrained minimum.
    inner_tol : float
        The RMS of the model's convergence vector below which the constrained
        minimum counts as found.
    constraint_model : str
        'sphere' or 'relu'.
    """

    path_kind = 'walker'

    def __init__(self, path_radius=0.1, inner_maxit=20, inner_tol=3.0e-4,
                 constraint_model='sphere', **kwargs):
        super().__init__(**kwargs)
        self.path_radius = float(path_radius)
        self.inner_maxit = int(inner_maxit)
        self.inner_tol = float(inner_tol)
        self.constraint_model_name = str(constraint_model).strip().lower()

        self.phase = 'idle'
        self.macro = 0
        self.micro = 0
        self.evaluations = 0
        # The absolute joint vector to evaluate next, and the one the state
        # currently holds. Both are in this walker's own continuous frame: a
        # wrapped state holds the same structure and the step is applied as a
        # difference, so a wrap never moves the sphere.
        self.trial = None
        self.state_frame = None
        self.reference = None
        self.points = []
        self.rows = []
        self.last_row = {}
        self._model = None
        self._curvature = None
        self._seed_pending = True
        self._last_energy = None

    # ---- Seeding ----

    def seed(self, state, joint, lattice):
        """The joint vector the walk starts on.

        `mep` starts on the geometry the run was given. `irc` overrides this and
        starts one radius away along the imaginary mode of its transition state,
        which is the one thing that tells the two walks apart at the start.
        """
        return np.asarray(joint, dtype=np.float64).reshape(-1).copy()

    def begin(self, state):
        """Read the geometry, the masses and the box, and seed the walk."""
        joint, lattice = self._joint_of(state)
        self._atomic_size = int(np.asarray(state.coordinates, dtype=np.float64)
                                .reshape(-1, 3).size)
        self._base_lattice = lattice
        self._weights = self._joint_weights(state)
        self._atomic_weights = self._mass_weights(state)
        self._mobile = self._mobile_mask(state)
        seed = self.seed(state, joint, lattice)
        if seed.size != self._weights.size:
            raise ValueError(
                'a path is walked on the joint vector of this system and the '
                'seed does not match it: %d components against %d'
                % (seed.size, self._weights.size))
        self.reference = seed
        self.trial = seed.copy()
        self.state_frame = seed.copy()
        self._model = make_constraint(
            self.constraint_model_name, self.path_radius, self._weights,
            self.inner_tol, mobile=self._mobile, curvature=self._curvature)
        self._model.begin(self.reference)
        self.points = []
        self.rows = []
        self.phase = 'probe'

    # ---- Driving ----

    def _curve(self, joint):
        """The model's own reading of the radius, or the plain norm."""
        return self._model.radius_of(joint)

    def next_trial(self):
        """The joint geometry to evaluate next, or None when the walk is over.

        Returns
        -------
        tuple or None
            `(coordinates Angstrom, lattice Angstrom or None)`, or None when no
            further probe will be taken.
        """
        if self.phase != 'probe':
            return None
        return self._geometry_of(self.trial)

    def apply_trial(self, state):
        """Write the current probe into the state.

        The step is the difference between the probe and the geometry the state
        already holds, so a state whose coordinates were wrapped by `apply_pbc`
        receives the same relative move as an unwrapped one - the sphere lives in
        this optimizer's own frame and is never re-read from the state.

        The atomic half is projected through `apply_constraint` first (a frozen
        component has no displaceable direction) and the box half through
        `_cell_direction` (an isotropic run may only scale the box), and the
        step is applied by `_apply_step`, which rides the atoms with the box.
        """
        displacement = self.trial - self.state_frame
        atomic = self.apply_constraint(state, displacement[:self._atomic_size])
        if not self.cell_variables:
            step = atomic
        else:
            cell = displacement[self._atomic_size:]
            lattice = self._lattice_of(self.state_frame)
            direction = self._cell_direction(lattice)
            if direction is not None and cell.size:
                cell = np.dot(cell, direction) * direction
            step = np.concatenate((atomic, cell))
        coordinates = np.asarray(state.coordinates, dtype=np.float64)
        self._apply_step(state, coordinates.reshape(-1) * self.coord_to_bohr,
                         coordinates, self._lattice_of(self.state_frame), step)
        self.state_frame = self.trial.copy()

    def intake(self, state):
        """The objective and the joint gradient of the evaluation in the state.

        One reading, two conversions: the skill's gradient is brought to Hartree
        per Bohr with the factor the Path workflow resolved from the manifest,
        its atomic head is replaced by the tangential part a declared condition
        leaves the walk to act on (`note_evaluation` has to have been called, and
        the workflow calls it at this evaluation), and the lattice half is the
        stress conjugate of the box - so the walk, its criteria and its reported
        rows all read one vector.

        Returns
        -------
        tuple
            `(objective, joint gradient)`; the objective is the enthalpy when the
            run holds a pressure, which is what `_objective` already resolved.
        """
        energy, gradient = self._objective(state, state.result)
        if energy is None or gradient is None:
            raise ValueError(
                'a path point is found from an energy and a gradient and this '
                'evaluation returned %s'
                % ('no gradient' if energy is not None else 'no energy'))
        gradient = (np.asarray(gradient, dtype=np.float64).reshape(-1)
                    * self.gradient_factor)
        gradient = self._tangent_of(gradient)
        coordinates = (np.asarray(state.coordinates, dtype=np.float64)
                       .reshape(-1) * self.coord_to_bohr)
        return float(energy), self._joint_gradient(state, gradient, coordinates)

    def observe(self, state):
        """Take one finished evaluation of the probe `apply_trial` wrote.

        Parameters
        ----------
        state : PyRAIState
            The state carrying the energy and gradient of the geometry that was
            just evaluated. `note_evaluation` must have been called on it first:
            what the walk is allowed to act on is the tangential part of this
            gradient, and a condition's own direction is not part of it.
        """
        energy, gradient = self.intake(state)
        joint = self._adopt(state)
        radius = self._curve(joint)
        inspection = self._model.inspect(joint, energy, gradient)
        settled = bool(self._model.settled(inspection))
        self.evaluations += 1
        self.last_row = self._row(energy, radius, inspection, settled, gradient)
        self.rows.append(self.last_row)
        self._last_energy = energy

        if self._seed_pending:
            # The first evaluation is the geometry the walk was given, and it is
            # the first point of the profile whether or not it is also the first
            # constrained minimum.
            self._seed_pending = False
            self.points.append(self._point(joint, energy, radius, inspection,
                                           settled=True, role=self.seed_role))
            if settled:
                self.stop_reason = (
                    'the starting geometry is already a stationary point of this '
                    'surface, so there is no path to walk from it')
                self.phase = 'done'
                return
            self.trial = self._model.next_trial(inspection)
            self.micro += 1
            return

        if settled and inspection['radius'] <= self.inner_tol:
            # The centre of a sphere is the point the walk last accepted, so a
            # probe that settles on it says that point is stationary on the
            # surface: the walk has arrived, and re-accepting its own centre
            # would add a zero-length point to the profile for every remaining
            # iteration.
            self.stop_reason = (
                'the walk reached a stationary point of this surface at point %d'
                % self.macro)
            self.phase = 'done'
            return
        if settled:
            self._accept(joint, energy, radius, inspection, settled=True)
            return
        if self.micro + 1 >= self.inner_maxit:
            # The point was not found and the budget for finding it is spent.
            # What the walk has is kept and it *carries on*: a difficult point
            # is a difficult point, not the end of the path (the reference
            # implementation's micro-iteration takes the same branch - it stops
            # refining this image and advances the sphere). The point's own
            # `settled` flag records that it is not a constrained minimum, which
            # is what `path_profile.txt` prints, and the run is only reported
            # unfinished by the rules that end a walk: the point budget, an
            # uphill step, or a stationary surface.
            self._accept(joint, energy, radius, inspection, settled=False)
            return
        self.trial = self._model.next_trial(inspection)
        self.micro += 1

    def _adopt(self, state):
        """The joint geometry the backend actually evaluated, in this frame.

        The workflow writes this walker's probe into the state, then wraps the
        structure into its cell (`apply_pbc`) and lands a declared condition on
        its manifold (`constraint_runtime.constrain_state`); both move the
        coordinates. Trusting `state_frame` after that puts the sphere, the arc
        length and the reported energies on a geometry no force was computed for
        - measured as a walk whose held bond is off by the solve's own correction
        and whose PBC frames differ from the walker's by a whole box length.

        The stored geometry is a main-image reading plus a crossing ledger, so
        the continuous one is the derived reading (`data.geometry.continuous_
        coordinates`), which is exactly the frame this walker keeps: a wrap then
        leaves the sphere where it was and a condition's correction is adopted
        rather than ignored.
        """
        coordinates = np.asarray(state.coordinates, dtype=np.float64)
        system = getattr(getattr(state, 'pmd', None), 'system', None)
        if not np.any(self._pbc_of(state)):
            joint, lattice = self._joint_of(state)
        else:
            continuous = continuous_coordinates(coordinates, system)
            if continuous is None:
                joint, lattice = self._joint_of(state)
            else:
                lattice = self._cell_lattice(state)
                if lattice is None:
                    joint = continuous.reshape(-1) * self.coord_to_bohr
                else:
                    slots = lattice.reshape(-1)[list(CELL_COMPONENT_SLOTS)]
                    joint = np.concatenate(
                        (continuous.reshape(-1) * self.coord_to_bohr, slots))
        self.state_frame = joint.copy()
        return joint

    @staticmethod
    def _pbc_of(state):
        """The system's periodicity flags, as the framework reads them."""
        system = getattr(getattr(state, 'pmd', None), 'system', None)
        return normalize_pbc(getattr(system, 'pbc', None) if system else None)

    def _accept(self, joint, energy, radius, inspection, settled):
        """Close one path point and re-centre the sphere on it.

        Returns the reason the walk stops here when the point budget is spent,
        and the empty string otherwise: a caller that has its own reason to
        report says the two together rather than one instead of the other.
        """
        previous = self.points[-1]['energy'] if self.points else None
        self.macro += 1
        if previous is not None and energy > previous + _UPHILL_FLOOR:
            # The walk has climbed: this point is on the other side of a
            # minimum, and the profile's last point is the minimum it found.
            self.stop_reason = (
                'the path turned uphill at point %d (%.10f against %.10f); the '
                'minimum is the point before it'
                % (self.macro, energy, previous))
            self.phase = 'done'
            return ''
        self.points.append(self._point(joint, energy, radius, inspection,
                                       settled=settled, role='path'))
        self.reference = joint.copy()
        self._model.begin(self.reference)
        self.trial = joint.copy()
        self.micro = 0
        if self.macro >= self.maxiter:
            # The framework's own phrase for a run that stopped at its step
            # bound, so a truncated path is reported the way a truncated
            # relaxation is (the qualification runner keys on it) and a reader
            # does not have to learn a second wording for one outcome.
            self.stop_reason = (
                'Maximum optimization steps reached: the walk kept its %d path '
                'points' % self.maxiter)
            self.phase = 'done'
            return self.stop_reason
        return ''

    # ---- Rows and points ----

    @property
    def seed_role(self):
        """What the first point of this walk is called in the profile."""
        return 'start'

    def _row(self, energy, radius, inspection, settled, gradient):
        """One row of `path_status.txt`, keyed by column name.

        The displacement columns are the probe's own offset from the centre of
        the sphere it is being found in, in Angstrom on the atomic half: that is
        the step this point has taken so far, which is what a reader of a
        convergence table is looking for. The gradient columns are the model's
        own convergence vector, judged over the components the run may move - the
        tangential part for the sphere, which is what the criteria of a
        constrained run are read on, and the augmented gradient for the ReLU
        model, which is what its minimizer steps on.

        A variable-cell probe carries the cell half of the same three readings
        under the names `utilities/opt_formatter.py` gives them, so one number is
        spelled one way in both records.
        """
        perpendicular = np.asarray(inspection['perpendicular'], dtype=np.float64)
        offset = (self.state_frame - self.reference)[
            :self._atomic_size] * self.disp_to_angstrom
        mobile = (np.ones(offset.shape, dtype=bool) if self._mobile is None
                  else self._mobile[:offset.size])
        moving = offset[mobile]
        row = {
            'step': self.evaluations,
            'energy': energy,
            'shift': (0.0 if self._last_energy is None
                      else energy - self._last_energy),
            'max_d': float(np.max(np.abs(moving))) if moving.size else 0.0,
            'rms_d': (float(np.sqrt(np.sum(moving ** 2) / moving.size))
                      if moving.size else 0.0),
            'max_g': float(np.max(np.abs(perpendicular)))
            if perpendicular.size else 0.0,
            'rms_g': self._model.rms(perpendicular),
            'radius': radius,
            'residual': float(inspection['residual']),
            'settled': settled,
            'point': self.macro,
        }
        if self.cell_variables:
            tail = np.asarray(gradient, dtype=np.float64)[self._atomic_size:]
            step = (self.state_frame - self.reference)[self._atomic_size:]
            row.update({
                'cell_gmax': float(np.max(np.abs(tail))) if tail.size else 0.0,
                'cell_grms': (float(np.sqrt(np.mean(tail ** 2)))
                              if tail.size else 0.0),
                'cell_dmax': (float(np.max(np.abs(step))) if step.size else 0.0),
                'cell_drms': (float(np.sqrt(np.mean(step ** 2)))
                              if step.size else 0.0),
                'volume': self._volume(self.state_frame),
            })
        return row

    def _volume(self, joint):
        """The box one joint vector stands for, in cubic Angstrom."""
        lattice = self._lattice_of(joint)
        if lattice is None:
            return 0.0
        return float(abs(np.linalg.det(lattice)) * self.disp_to_angstrom ** 3)

    def _point(self, joint, energy, radius, inspection, settled, role):
        """One accepted path point of `path_profile.txt`."""
        perpendicular = np.asarray(inspection['perpendicular'], dtype=np.float64)
        arc = self.points[-1]['arc'] if self.points else 0.0
        if self.points:
            # The mass-weighted distance, which is the length this walk advances
            # by: `path_radius` is written in those coordinates, so the plain
            # Cartesian distance between two accepted points is a different
            # number (measured on the shipped HCN walk, 0.71 Angstrom of raw
            # displacement is the 0.15 Bohr radius the run asked for, because
            # the hydrogen carries the smaller weight).
            arc += float(np.linalg.norm(
                self._weights * (joint - self.points[-1]['joint'])))
        point = {
            'index': len(self.points),
            'joint': np.asarray(joint, dtype=np.float64).copy(),
            'energy': float(energy),
            'arc': float(arc),
            'radius': float(radius),
            'perp_rms': self._model.rms(perpendicular),
            'perp_max': (float(np.max(np.abs(perpendicular)))
                         if perpendicular.size else 0.0),
            'role': role,
            'settled': bool(settled),
        }
        if self.cell_variables:
            point['volume'] = self._volume(joint)
        return point

    def method_details(self, report):
        """The accepted-point and probe readings the run log adds.

        The values come from the last accepted point, so a rejected probe
        still says which point the walk is standing on. Missing numbers are
        None and the logger prints `Not measured`, never a zero.
        """
        details = {}
        points = self.path_points()
        if points:
            point = points[-1]
            previous = (float(points[-2].get('arc', 0.0))
                        if len(points) > 1 else 0.0)
            details['Arc'] = float(point.get('arc', 0.0))
            details['Length'] = float(point.get('arc', 0.0)) - previous
            details['Perp RMS'] = float(point.get('perp_rms', 0.0))
            details['Perp Max'] = float(point.get('perp_max', 0.0))
            details['Role'] = str(point.get('role', 'point'))
        direction = getattr(self, 'direction', None)
        if direction:
            details['Direction'] = str(direction)
        details['Radius'] = report.get('radius')
        details['Residual'] = report.get('residual')
        details['Settled'] = bool(report.get('settled', False))
        return details

    def finished(self):
        return self.phase == 'done'

    def report(self):
        return dict(self.last_row)

    def path_points(self):
        return [dict(point) for point in self.points]

    def results(self):
        return {
            'path_points': [
                {key: value for key, value in point.items() if key != 'joint'}
                for point in self.points],
            'path_energies': [point['energy'] for point in self.points],
            'path_arc': [point['arc'] for point in self.points],
            'path_stop_reason': self.stop_reason,
        }


class BandPathOptimizer(PathOptimizer):
    """The images of a nudged elastic band and the FIRE state that relaxes them.

    The band is a chain of joint vectors in this optimizer's own continuous
    frame; the workflow builds it (`workflow/path.py::build_band`) and hands it
    over with `set_band`, because placing the images needs the endpoint files,
    the atom list and the interpolation scheme, which are the workflow's to read.
    Each round dispatches every image as one batch task, and `band_collect`
    reads the energies and gradients back, computes the band forces (in the
    subclass) and takes one FIRE step.

    The endpoints are re-evaluated with the band every round - the profile's
    barrier is then measured on the current endpoints rather than on a pair read
    once - and they are never moved.

    Parameters
    ----------
    neb_spring : float
        The spring constant, in Hartree per Bohr^2, so that it times a segment
        length in Bohr is the same force the true force is written in.
    neb_climb : bool
        Whether the highest image becomes a climbing image once the band is
        sufficiently relaxed.
    neb_climb_threshold : float
        The perpendicular force max that arms the climbing image.
    neb_step : float
        The FIRE integrator's initial timestep, in atomic units.
    """

    path_kind = 'band'

    def __init__(self, neb_spring=0.01, neb_climb=False,
                 neb_climb_threshold=1.0e-2, neb_step=0.1, **kwargs):
        super().__init__(**kwargs)
        self.spring = float(neb_spring)
        self.climb = bool(neb_climb)
        self.climb_threshold = float(neb_climb_threshold)
        self.dt0 = float(neb_step)
        self.images = None
        self.lattices = None
        self.roles = None
        self.ledgers = None
        self.energies = None
        self.gradients = None
        self.forces = None
        self.velocity = None
        self.dt = self.dt0
        self.alpha = _FIRE_ALPHA_START
        self.n_up = 0
        self.iteration = 0
        self.latched = False
        self.climbing_index = None
        self.last_row = {}
        self.rows = []
        self._pbc = (0, 0, 0)
        # Whether the run declares anything held, the scratch state the position
        # solve and the tangential split run in, and the component mask a frozen
        # atom contributes. Built by `begin`.
        self._declared = False
        self._scratch = None
        self._fix_mask = None

    # ---- Setup ----

    def begin(self, state):
        """Read the units, the periodicity, and what the run declares held.

        A band whose run declares a condition needs two things this class does
        not have on its own: a state to run the position solve and the
        tangential split in (the framework's one implementation of both is
        state-shaped), and a way to hold a frozen component from the
        interpolation rather than from the value it happens to have. Both come
        from a scratch copy of the parent, bound to the constraint runtime here
        so its anchors are the system's own geometry - the same binding the
        single-structure loop takes before its first step.
        """
        if self.images is None:
            raise ValueError(
                'a nudged elastic band needs its images: the Path workflow '
                'builds them and hands them over with set_band')
        self._atomic_size = int(np.asarray(state.coordinates, dtype=np.float64)
                                .reshape(-1, 3).size)
        self._pbc = normalize_pbc(
            getattr(getattr(state.pmd, 'system', None), 'pbc', None))
        plan = (state.constraint_plan()
                if hasattr(state, 'constraint_plan') else None)
        self._declared = bool(plan is not None and not plan.is_empty)
        self._scratch = None
        self._fix_mask = None
        if self._declared:
            from ..constraints import runtime as constraint_runtime
            from ..constraints import projection
            self._scratch = copy.deepcopy(state)
            self._scratch.runtime = type(state.runtime)()
            constraint_runtime.bind(self._scratch)
            self._fix_mask = projection.build_fix_mask(
                self._scratch.constraint_plan(), self._atomic_size // 3)
            if not self._fix_mask.any():
                self._fix_mask = None

    def set_band(self, images, lattices=None, roles=None):
        """Take the band the workflow built.

        Parameters
        ----------
        images : array-like
            (N, 3N) joint vectors in Bohr; index 0 is the reactant and index -1
            the product.
        lattices : array-like or None
            (N, 3, 3) lattices in Bohr for a variable-cell band; None keeps the
            system's own box for every image.
        roles : list of str or None
            What each image is in the profile ('reactant', 'image', 'ts',
            'product').
        """
        band = np.asarray(images, dtype=np.float64)
        if band.ndim != 2 or band.shape[0] < 3:
            raise ValueError(
                'a nudged elastic band carries at least two endpoints and one '
                'interior image, and this band has shape %s' % (band.shape,))
        self.images = band.copy()
        count = band.shape[0]
        self.lattices = (None if lattices is None
                         else np.asarray(lattices, dtype=np.float64).copy())
        if self.lattices is not None and self.lattices.shape != (count, 3, 3):
            raise ValueError(
                'a variable-cell band carries one lattice per image: %d images '
                'against lattices of shape %s'
                % (count, self.lattices.shape))
        self.roles = (list(roles) if roles is not None
                      else ['reactant'] + ['image'] * (count - 2) + ['product'])
        # The joint vector's two halves, from the band's own width: the workflow
        # hands the band over before `begin` has a state to read the atom count
        # from, and a ledger sized by a zero would be no ledger at all.
        cells = len(CELL_COMPONENT_SLOTS) if self.cell_variables else 0
        natom = (band.shape[1] - cells) // 3
        if 3 * natom + cells != band.shape[1]:
            raise ValueError(
                'a band is a joint vector of (3N + 6) or 3N components and this '
                'one is %d wide with cell_variables=%s'
                % (band.shape[1], self.cell_variables))
        self._atomic_size = 3 * natom
        self.ledgers = [np.zeros((natom, 3), dtype=np.int64)
                        for _ in range(count)]
        self.energies = np.full(count, np.nan)
        self.gradients = [None] * count
        self.forces = np.zeros_like(band)
        self.velocity = np.zeros_like(band)

    @property
    def count(self):
        return 0 if self.images is None else int(self.images.shape[0])

    @property
    def interior(self):
        return list(range(1, self.count - 1))

    # ---- The batch ----

    def band_batch(self, state):
        """The task of every image, wrapped and carrying its ledger.

        Each task is one independent evaluation of one image, which is what
        `Dispatcher.dispatch(units=batch)` fans out: an image that crossed a
        cell face arrives wrapped, with the crossing count of the same step
        beside it (`utilities/lattice.wrap_fractional_with_image`), because a
        new position written on a stale ledger relocates the atom by a whole box.
        """
        # The declared conditions are landed on every image before anything is
        # handed out, so what the children evaluate is a structure the run is
        # allowed to have: an interpolated image satisfies no condition, and a
        # band that let one through would be relaxed against a manifold it is
        # not on (dev/methods/PATH_DEVELOPMENT.md, the constraints section).
        self._land_images()
        tasks = []
        for index in range(self.count):
            coordinates, image = self._wrapped_image(index)
            task = {'coordinates': coordinates.tolist(), 'image': image}
            if self.lattices is not None:
                task['lattice'] = (
                    self.lattice_of(index) * self.disp_to_angstrom).tolist()
            tasks.append(task)
        return tasks

    def _land_images(self):
        """Put the walkable images on the manifold the run declares, in place.

        The two endpoints are left exactly where the run was given them: they are
        the band's boundary conditions, not free images, and a condition they
        violate is the input's own inconsistency - which a reader sees as the
        nonzero residual on the profile's first or last row rather than as a
        product endpoint that quietly moved. Every interior image is landed, and
        the landing runs both before a round is dispatched and after it has
        stepped, so what the children evaluate and what the records print are
        both structures the run is allowed to have.
        """
        if not self._declared:
            return
        from ..constraints import runtime as constraint_runtime

        for index in self.interior:
            self._write_scratch(index)
            constraint_runtime.constrain_state(self._scratch)
            self._read_scratch(index)

    def _write_scratch(self, index):
        """Load one image's geometry and cell into the scratch state."""
        from ...utilities.lattice import (cell_volume, matrix_to_cell6)

        coordinates = (self.images[index][:self._atomic_size].reshape(-1, 3)
                       * self.disp_to_angstrom)
        updates = {'coordinates': coordinates.tolist()}
        if self.lattices is not None:
            lattice = self.lattice_of(index) * self.disp_to_angstrom
            updates['lattice_vector'] = lattice.tolist()
            updates['cell'] = [float(value) for value in
                               matrix_to_cell6(lattice)]
            updates['volume'] = cell_volume(lattice)
        self._scratch.update_system(updates)

    def _read_scratch(self, index):
        """Take the solved geometry back out of the scratch state."""
        coordinates = (np.asarray(self._scratch.coordinates, dtype=np.float64)
                       .reshape(-1) * self.coord_to_bohr)
        self.images[index][:self._atomic_size] = coordinates

    def _constrained_force(self, index, gradient):
        """The image's force with the declared conditions taken out of it.

        The split is the framework's own reading (`constraints/runtime
        .constraint_split`): frozen components come back zeroed and every
        condition's normal direction is removed, so the band is relaxed on the
        part of the gradient the conditions cannot absorb - the same quantity a
        constrained single-structure search is judged on. A variable-cell band
        also takes the conditions' own lattice term (`cell_virial`), which the
        engine's stress does not carry because it is the potential's.
        """
        if not self._declared:
            return gradient
        from ..constraints import runtime as constraint_runtime

        self._write_scratch(index)
        split = constraint_runtime.constraint_split(
            self._scratch, gradient[:self._atomic_size])
        if split is None:
            return gradient
        joint = np.asarray(gradient, dtype=np.float64).copy()
        joint[:self._atomic_size] = np.asarray(split['tangent'],
                                               dtype=np.float64).reshape(-1)
        virial = split.get('cell_virial')
        if self.cell_variables and virial is not None:
            joint[self._atomic_size:] -= (
                np.asarray(virial, dtype=np.float64).reshape(-1)[
                    list(CELL_COMPONENT_SLOTS)])
        return joint

    def lattice_of(self, index):
        """The box one image carries, in Bohr.

        The joint vector is the authority, not `self.lattices`: that array is the
        snapshot the band was built with, and a variable-cell band moves the six
        lattice components of every image on every step. Reading the snapshot
        instead is what made a settled-looking band report the box it started
        with, to the cubic Angstrom, for every row of its record: measured on
        `path_cell_cineb`, the volume column read [1000.0, 1000.0, 1000.0,
        1000.0] while the cell force it was stepping on fell from 0.159 to 0.128.
        A fixed-cell band has no cell components and gets the snapshot back,
        which is the same box by construction.
        """
        if self.lattices is None:
            return None
        if not self.cell_variables:
            return self.lattices[index]
        flat = np.asarray(self.lattices[index],
                          dtype=np.float64).reshape(-1).copy()
        flat[list(CELL_COMPONENT_SLOTS)] = self.images[index][
            self._atomic_size:]
        return flat.reshape(3, 3)

    def _wrapped_image(self, index):
        """`(coordinates Angstrom, ledger)` of one image, wrapped into its box."""
        joint = self.images[index]
        coordinates = (joint[:self._atomic_size].reshape(-1, 3)
                       * self.disp_to_angstrom)
        if self.lattices is None or not any(self._pbc):
            return coordinates, self.ledgers[index].tolist()
        lattice = self.lattice_of(index) * self.disp_to_angstrom
        fractional = cart_to_frac(joint[:self._atomic_size].reshape(-1, 3)
                                 * self.disp_to_angstrom, lattice)
        wrapped, crossed = wrap_fractional_with_image(fractional, self._pbc)
        self.ledgers[index] = (self.ledgers[index] + crossed
                               if self.ledgers[index].shape == crossed.shape
                               else crossed)
        return frac_to_cart(wrapped, lattice), self.ledgers[index].tolist()

    # ---- Reading the round back ----

    def band_collect(self, state, results):
        """Read one round of image evaluations and take one FIRE step.

        Parameters
        ----------
        state : PyRAIState
            The parent state (its geometry is not moved by a band).
        results : list
            One outcome per task, in `band_batch` order.
        """
        if results is None or len(results) != self.count:
            raise ValueError(
                'the band asked for %d evaluations and %d came back; a band '
                'cannot be relaxed on a partial round'
                % (self.count, 0 if results is None else len(results)))
        for index, outcome in enumerate(results):
            energy, gradient = self._image_evaluation(outcome, index)
            self.energies[index] = energy
            self.gradients[index] = self._constrained_force(index, gradient)
        self.iteration += 1
        reading = self._band_reading()
        self.last_row = dict(reading)
        self.last_row['step'] = self.iteration
        self.rows.append(dict(self.last_row))
        if reading['converged']:
            self.stop_reason = ''
            return
        self._fire_step(reading)
        # A step moves the images off the manifold the round was relaxed on, so
        # they are landed again before anything writes or dispatches them: the
        # frames a round prints are then the geometries the next round will
        # evaluate, which is the same rule the walker's records follow.
        self._land_images()

    def _image_evaluation(self, outcome, index):
        """`(objective, joint gradient)` of one image's outcome, in Hartree."""
        from ..hessian import _active_gradient, _outcome_result

        result = _outcome_result(outcome)
        if result is None:
            raise ValueError(
                'band image %d returned no result, so this round has no force '
                'to relax it with' % index)
        energies = result.get('energy')
        if energies is None:
            raise ValueError(
                'band image %d returned no energy' % index)
        value = np.asarray(energies, dtype=np.float64).reshape(-1)
        energy = float(value[min(self.active_state, value.size - 1)])
        gradient = _active_gradient(result)
        if gradient is None:
            raise ValueError(
                'band image %d returned no gradient, and a nudged elastic band '
                'is relaxed on forces' % index)
        joint = np.asarray(gradient, dtype=np.float64).reshape(-1)
        joint = joint * self.gradient_factor
        joint = self._joint_gradient_of_result(result, index, joint)
        if self.lattices is not None:
            # The objective of a run under pressure is the enthalpy, and the
            # lattice half of the force is the stress conjugate of the box this
            # image holds - one surface, one tensor.
            energy = energy + self.pressure_au * float(
                abs(np.linalg.det(self.lattice_of(index))))
        return energy, joint

    def _joint_gradient_of_result(self, result, index, gradient):
        """Join the atomic gradient to the image's own cell force.

        `cell_variables` is the one authority on the vector's width, and it has
        to be: a fixed-cell periodic band carries a cell per image (the metric
        and the wrap need it) but its vector is atomic-only, so keying this on
        `lattices` joined six lattice components to a 3N tangent and every
        periodic band died on `shapes (3N + 6,) and (3N,) not aligned`.
        """
        if not self.cell_variables:
            return gradient
        from ..hessian import _active_stress
        from ...utilities.lattice import stress_to_cell_force

        stress = _active_stress(result)
        if stress is None:
            raise ValueError(
                'optimize_cell relaxes the box of every band image and image %d '
                'returned no stress tensor; there is no lattice derivative to '
                'push the box with' % index)
        tensor = np.asarray(stress, dtype=np.float64)
        if self.pressure_au:
            tensor = tensor + self.pressure_au * np.eye(3)
        conjugate = stress_to_cell_force(tensor, self.lattice_of(index))
        return np.concatenate(
            (gradient, conjugate.reshape(-1)[list(CELL_COMPONENT_SLOTS)]))

    # ---- FIRE ----

    def _fire_step(self, reading):
        """One FIRE update of the whole band, in the joint vector.

        Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006): the velocity is mixed
        towards the force, and while the two agree the timestep grows and the
        mixing shrinks; a step after which they disagree resets the timestep and
        zeroes the velocity. A single-image displacement cap keeps a climbing
        image's still-large parallel force from flinging it into a geometry the
        backend cannot evaluate (`_BAND_MAX_MOVE`).
        """
        forces = self.forces
        interior = self.interior
        power = float(sum(np.dot(forces[index], self.velocity[index])
                          for index in interior))
        if power > 0.0:
            self.n_up += 1
            velocity_norm = float(np.sqrt(sum(
                float(np.dot(self.velocity[index], self.velocity[index]))
                for index in interior)))
            force_norm = float(np.sqrt(sum(
                float(np.dot(forces[index], forces[index]))
                for index in interior)))
            scale = (self.alpha * velocity_norm / force_norm
                     if force_norm > 0.0 else 0.0)
            for index in interior:
                self.velocity[index] = (
                    (1.0 - self.alpha) * self.velocity[index]
                    + scale * forces[index])
            if self.n_up > _FIRE_N_MIN:
                self.dt = min(self.dt * _FIRE_F_INC,
                              self.dt0 * _FIRE_DT_MAX_FACTOR)
                self.alpha *= _FIRE_F_ALPHA
        else:
            self.n_up = 0
            self.dt *= _FIRE_F_DEC
            self.alpha = _FIRE_ALPHA_START
            for index in interior:
                self.velocity[index] = np.zeros_like(self.velocity[index])
        for index in interior:
            self.velocity[index] = (self.velocity[index]
                                    + self.dt * forces[index])
            step = self.dt * self.velocity[index]
            norm = float(np.linalg.norm(step))
            if norm > _BAND_MAX_MOVE:
                step = step * (_BAND_MAX_MOVE / norm)
            self.images[index] = self.images[index] + step

    # ---- The subclass supplies the forces and the criteria ----

    def _band_reading(self):
        """Compute the band forces, the readings and whether the band settled."""
        raise NotImplementedError

    def method_details(self, report):
        """The band's barrier, supervising image and spring state."""
        details = {}
        for label, key in (('Barrier', 'barrier'),
                           ('Image', 'image'),
                           ('Climb', 'climb'),
                           ('Spring', 'spring')):
            value = report.get(key)
            if value is not None:
                details[label] = value
        return details

    def finished(self):
        return bool(self.last_row) and bool(self.last_row.get('converged'))

    def report(self):
        return dict(self.last_row)

    def path_points(self):
        """The band's images, in order, as profile rows.

        The force columns are read over the components the run may move: a
        frozen atom is not part of a band's force, and counting it would dilute
        the reading toward zero by the fraction of the system the input pinned
        (`optimizer/base.py::_rms_count`).
        """
        points = []
        arc = 0.0
        for index in range(self.count):
            if index > 0:
                arc += float(np.linalg.norm(self.images[index]
                                            - self.images[index - 1]))
            force = (np.asarray(self.forces[index], dtype=np.float64)
                     if self.forces is not None else np.zeros(0))
            moving = force
            if force.size and self._fix_mask is not None:
                moving = force[:self._atomic_size][~self._fix_mask.reshape(-1)]
            points.append({
                'index': index,
                'energy': float(self.energies[index]),
                'arc': arc,
                'radius': 0.0,
                'perp_rms': (float(np.sqrt(np.mean(moving ** 2)))
                             if moving.size else 0.0),
                'perp_max': (float(np.max(np.abs(moving)))
                             if moving.size else 0.0),
                'role': self.roles[index],
                'settled': True,
            })
        return points

    def results(self):
        energies = [float(value) for value in self.energies]
        endpoints = (energies[0], energies[-1]) if energies else (0.0, 0.0)
        return {
            'path_points': [
                {key: value for key, value in point.items()
                 if key not in ('joint',)}
                for point in self.path_points()],
            'path_energies': energies,
            'path_arc': [point['arc'] for point in self.path_points()],
            'path_barrier': (float(max(energies) - max(endpoints))
                             if energies else 0.0),
            'path_climbing_image': self.climbing_index,
            'path_stop_reason': self.stop_reason,
        }


__all__ = [
    'PathOptimizer',
    'WalkerPathOptimizer',
    'BandPathOptimizer',
]