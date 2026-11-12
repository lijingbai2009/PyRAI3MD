"""
======================================================================
This module provides the TransitionStateOptimizer for geometry
optimization towards transition states (first-order saddle points).

The step is a partitioned rational-function minimization (RFO). The
curvature model is diagonalized, and the augmented problem is solved
separately on one chosen mode and on the rest: along the chosen mode the
*upper* root of the 2x2 problem is taken, which steps against the
curvature and therefore climbs, and over the remaining modes the *lower*
root of the augmented problem is taken, which steps with the curvature and
therefore descends. That is the saddle condition, read off the model rather
than imposed on it: the point the search is walking towards is quiet in every
direction and downhill in exactly one. The construction is the one published
for saddle searches (Gonzalez and Schlegel, J. Chem. Phys. 90:2154, 1989;
Bofill, J. Chem. Phys. 95:3724, 1991), implemented here from those
descriptions.

The mode to climb is chosen once and then followed. The first step takes the
direction of lowest curvature; every later step takes the eigenvector closest
to the one it climbed last time. Without that inheritance a search would jump
between modes whenever two of them change order in the sorted list, which is
what happens on the way into a saddle, and the direction it climbs would be
different on each step.

Between curvature evaluations the model is kept alive by Bofill's mixture of
the rank-one (Murtagh-Sargent) and symmetric-Broyden updates (Bofill,
J. Comput. Chem. 15:1, 1994). A saddle search walks a surface that is convex
in every direction but one, so its steps have a negative `s . y` more often
than not; a BFGS update cannot use such a step - it would drive an eigenvalue
of the model through zero - and an update that skips those steps leaves the
search with its initial curvature for the rest of the run, which is the
failure mode this class used to have. The mixture fits the model to the
gradient difference without insisting it stay positive definite.

The curvature the search starts on comes from the framework's Hessian
computation at the geometry the run begins on: an optimizer that needs
curvature before its first step says so with `needs_initial_hessian`, and the
Optimize workflow goes and runs `workflow/hessian.py` for it. Which producer
answers that request is the input's to say (`&hessian hess_driver`), but only
the first step reads it: from the second step on the matrix is the model's
own, kept up by the updates above, so the negative-mode count a run prints is
a reading of that model, not a frequency calculation, and the row says so in
its own name.

The rejection rule a minima search uses is deliberately absent: a saddle
search must move uphill along the followed direction, and an energy rise is
the expected consequence of doing exactly that. What bounds the search is
therefore the trust radius alone, not a stall count - a run that climbs
forever is stopped by `opt_maxit`, and the radius cannot grow past the class
constant `MinimaOptimizer.TRUST_RADIUS_MAX`.

Oct 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from ..frequency import periodic_vibrational_basis, vibrational_basis
from ...utilities.units import ANGSTROM_TO_BOHR
from .minima import MinimaOptimizer

# An RFO root that coincides with a curvature makes that mode's coefficient
# unbounded; the term is dropped instead of evaluated, which is the standard
# handling of an RFO pole rather than an approximation of one.
_RFO_POLE_TOLERANCE = 1.0e-10

# Below these the step or the gradient difference carries no information to
# fit a model to, and dividing by them would be dividing by roundoff.
_UPDATE_TOLERANCE = 1.0e-14

# The rank-one term divides by `residual . s`, which is the one quantity in this
# formula that can vanish while both of its vectors stay large: a step that
# turns the gradient almost perpendicular to its own displacement would
# otherwise hand the model a curvature larger than the surface has by the ratio
# of the two norms. The test is therefore relative, the way every SR1
# implementation states it (Bofill, J. Comput. Chem. 15:1, 1994), and the
# mixture weight carries the difference over to the symmetric-Broyden form as
# the projection collapses.
_SR1_RELATIVE_TOLERANCE = 1.0e-8


class TransitionStateOptimizer(MinimaOptimizer):
    """Optimizer for transition state search (first-order saddle point).

    Steps on the machinery `MinimaOptimizer` owns - the trust radius and its
    resizing, the history, the curvature model - with three differences: the
    step is a partitioned RFO that climbs one mode while descending the rest,
    the model is updated by a rule that tolerates a negative `s . y`, and a
    rise in the energy is not a rejected step, because climbing the followed
    direction is the point of the search.

    Parameters
    ----------
    maxiter : int
        Maximum optimization iterations.
    energy_threshold : float
        Energy convergence threshold.
    gradient_threshold : float
        Gradient RMS convergence threshold.
    displacement_threshold : float
        Displacement RMS convergence threshold.
    trust_radius : float
        Initial trust radius (upper bound of the step length, Bohr).
    use_rfo : bool
        Accepted and ignored, for parity with the other optimizers the
        workflow hands the keyword to. Every step this search takes is a
        rational-function step already; the switch exists to move a *minima*
        search between its Newton and RFO forms, and here there is no other
        form to move to.
    """

    needs_initial_hessian = True
    reject_on_energy_rise = False
    # The step bounds a minima relaxation is tuned with are not a saddle
    # search's. On the minima ceiling of 0.5 Bohr, engine runs from a bridging
    # HCN/HNC guess walked an atom half a Bohr per evaluation and lifted the
    # energy 29 Hartree in 60 steps with nothing to refuse any of it, because a
    # saddle search deliberately accepts a rise (PySCF HF/sto-3g, 2026-10-12).
    # isotropic guess on a rigid-invariant analytic saddle with three
    # vibrational directions: a ceiling of 0.1 Bohr reaches the first-order
    # saddle in 17 evaluations from the true curvature and in 21 from an
    # isotropic guess, while 0.05 converged from the true curvature in 34 and
    # diverged to +2.2 Hartree from the isotropic one. The published defaults
    # for a `transition` step are the same order (geomeTRIC: trust 0.01,
    # maximum 0.03).
    TRUST_RADIUS_MAX = 0.1
    # Asked for by name, on a live-engine authentication (PySCF HF/sto-3g on
    # the HCN->HNC bridging structure, 2026-10-12, runs `re_main` and
    # `re_skill`, one scratch directory each).
    #
    # Both curvature producers were run through to the end and reach the same
    # saddle: `hess_driver` left at its default (the framework's displaced
    # gradients) converged in 32 evaluations, `hess_driver: skill` (one
    # dispatched job asking the backend for its matrix) in the same 32, and
    # their energies agree to 1e-8 Hartree at -91.5650982833. Tightening the
    # step ceiling to 0.03 moves both to 38 evaluations and the same answer. A
    # `num_freq` job on the converged geometry reports exactly one imaginary
    # frequency (-1248.4 cm-1, then 2106.5 and 3071.3), which is what the run's
    # own `Negative Modes (model)` row claims, so the model reading and the
    # engine reading now say the same thing at the point the search stops at.
    #
    # A search with part of the molecule pinned is askable too, and was the last
    # piece to fall: the frame it steps in must give back the whole-body motions
    # that the anchor *opposes*, which are ordinary deformations under a freeze
    # (`workflow/frequency.py:vibrational_basis`, mask branch). Measured on
    # `optimize_partial_ts.yaml`, a water dimer with its donor oxygen held: 75
    # evaluations to -149.8362656686 Hartree, one imaginary mode (-3064.0 cm-1)
    # on the endpoint's own frequency job, and the held atom returned where the
    # plan put it.
    #
    # Two refusals were written and cleared on the way here, and the record of
    # them is the reason the pieces above are all present: the search had no
    # rule to refuse a step its model mispredicted (now
    # `TransitionStateOptimizer._rejection_reason`), a rejection at the radius
    # floor re-proposed the geometry it had just refused forever (now
    # `MinimaOptimizer._rejection_offers_a_shorter_step`), and the followed mode
    # was carried between steps in the reduced coordinates of a basis that is
    # rebuilt at every geometry, which is not one direction at all - the mode is
    # now stored in the fixed mass-weighted frame. The last of those is what made
    # the engine runs walk uphill.
    is_implemented = True

    def __init__(self, maxiter=100, energy_threshold=1e-6,
                 gradient_threshold=3e-4, displacement_threshold=3e-4,
                 trust_radius=0.05,
                 use_rfo=False,
                 **kwargs):
        del use_rfo
        super().__init__(
            maxiter=maxiter,
            energy_threshold=energy_threshold,
            gradient_threshold=gradient_threshold,
            displacement_threshold=displacement_threshold,
            trust_radius=trust_radius,
            **kwargs,
        )
        # A predicted change smaller than the energy change this run would even
        # call a change carries no information about the step, and reading its
        # sign is reading the print's roundoff. The search's own convergence
        # scale is the only defensible currency for that question: measured on an
        # analytic saddle surface, judging the sign at 1e-14 Hartree turned
        # predictions of -3.8e-8 against actual changes of +6.7e-7 into a
        # rejection every evaluation, with the radius held at its floor and the
        # restored geometry identical each time.
        self.PREDICTION_FLOOR = max(1.0e-14, float(energy_threshold))
        # Direct curvature approximation (not its inverse): the mode to follow
        # is an eigenvector of the model itself.
        self._hessian = None
        # The vibrational subspace of the current geometry, and the per-component
        # square root of the masses that brings a Cartesian vector into the
        # mass-weighted metric that subspace is defined in. Both are taken at
        # each evaluation, because the basis belongs to the geometry.
        self._basis = None
        self._weights = None
        # How many whole-body directions the frame took out. Six for a non-linear
        # structure in vacuum, and fewer for a periodic one whose triangular cell
        # spelling cannot express one of its own rotations - a count the report
        # carries rather than leaves to be inferred from a curvature table.
        self._frame_rank = None
        # Which Cartesian components the constraint plan leaves free. None until
        # an evaluation says the geometry, and all-True whenever nothing is
        # frozen, which is what keeps an unconstrained search bit-for-bit the
        # search it was before the plan was read here.
        self._active = None
        # The direction climbed on the last step, in the coordinate basis.
        # Following it by overlap - rather than the lowest eigenvalue each
        # step - is what keeps one search climbing one mode.
        self._followed = None
        self.negative_modes = None
        self.mode_eigenvalue = None

    # ---- Curvature ----

    def set_initial_hessian(self, matrix):
        """Take the curvature the framework Hessian computation returned.

        Parameters
        ----------
        matrix : array-like
            (3N, 3N) Hessian in Hartree per Bohr^2, in the same flattened
            component order the optimizer steps on.
        """
        array = np.asarray(matrix, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(
                'an initial Hessian must be a square matrix, got shape %s'
                % (array.shape,))
        self._hessian = 0.5 * (array + array.T)

    def _update_history(self, energy, gradient_1d, coordinates_1d):
        """Record the step and learn the curvature from it.

        Unlike the minima search, this one fits the model to every step whose
        endpoints are distinct: the `s . y` product that a BFGS update would
        have to refuse is exactly the information a saddle search needs.
        """
        had_history = bool(self.history['gradients'])
        super()._update_history(energy, gradient_1d, coordinates_1d)
        if not had_history or self._hessian is None:
            return
        s = self.history['coordinates'][-1] - self.history['coordinates'][-2]
        y = self.history['gradients'][-1] - self.history['gradients'][-2]
        self._bofill_update(s, y)

    def _bofill_update(self, s, y):
        """Fit the curvature model with Bofill's rank-one/symmetric-Broyden mix.

        The two forms agree when the step is consistent with the model
        (residual along `s`), and disagree when it is not. Their weight is the
        fraction of the residual that lies along `s`, which Cauchy-Schwarz
        keeps in [0, 1]; a step the model cannot be bent to fit at all
        (`s . y - s . B s` of either sign excluded) leaves the model where it
        was rather than inventing a curvature.
        """
        ss = float(s @ s)
        if ss < _UPDATE_TOLERANCE:
            return
        active = self._active
        if active is not None and not active.all():
            # A frozen component has no force law of its own to fit: the
            # constraint supplies whatever force keeps it still. Fitting the
            # model to that difference would teach it a curvature in a direction
            # the search is not allowed to move, and the coupling terms would
            # leak into the rows the movable atoms do read.
            s = np.where(active, s, 0.0)
            y = np.where(active, y, 0.0)
        residual = y - self._hessian @ s
        rs = float(residual @ s)
        rr = float(residual @ residual)
        if rr < _UPDATE_TOLERANCE:
            # The model already predicts this gradient difference; there is
            # nothing left to learn from the step.
            return
        scale = float(np.sqrt(ss * rr))
        phi = (rs * rs) / (scale * scale)
        if abs(rs) > _SR1_RELATIVE_TOLERANCE * scale:
            rank_one = self._hessian + np.outer(residual, residual) / rs
        else:
            rank_one = self._hessian
        symmetric = (self._hessian
                     + (np.outer(residual, s) + np.outer(s, residual)) / ss
                     - rs * np.outer(s, s) / (ss * ss))
        updated = phi * rank_one + (1.0 - phi) * symmetric
        updated = 0.5 * (updated + updated.T)
        if active is not None and not active.all():
            # Only the movable block was fitted, so only the movable block is
            # written back; the rows the plan froze keep whatever the Hessian
            # computation handed over for them.
            keep = np.outer(active, active)
            updated = np.where(keep, updated, self._hessian)
        self._hessian = updated

    # ---- Algorithm implementation ----

    def note_evaluation(self, state):
        """Read the constraint split, then the subspace the curvature lives in.

        The negative-direction count has to be taken on the same subspace a
        frequency calculation takes it on. A (3N, 3N) Cartesian model also
        carries the whole-body motions, and the curvature sitting there is
        whatever the updates left, not a physical restoring force: measured on a
        search that had in fact found its saddle, the engine's own Hessian at
        that geometry had one negative direction in the vibrational block while
        the raw model showed four negatives in nine, and the even-odd tell is
        that a triatomic has three modes to have opinions about. So the step and
        the count are both read after projecting the rigid motions out, through
        the one basis `workflow/frequency.py` builds - which is what makes the
        row the run prints and the imaginary frequencies it would find the same
        number.
        """
        split = super().note_evaluation(state)
        self._set_vibrational_frame(state)
        return split

    def _set_vibrational_frame(self, state):
        """Cache the vibrational basis and mass weights of this geometry."""
        positions = np.asarray(state.coordinates, dtype=np.float64)
        if positions.size == 0:
            return
        natom = positions.reshape(-1, 3).shape[0]
        lattice = self._cell_lattice(state) if self.cell_variables else None
        cell = 0 if lattice is None else self._joint_size(positions.size) - positions.size
        if natom < 2:
            # Nothing to project: a single particle has no whole-body rotation,
            # and an instrument that hands the search three working coordinates
            # is stepped on all three. A box beside it still has its own entries,
            # and those are directions the search moves, so they stay in.
            self._basis = np.eye(positions.size + cell, dtype=np.float64)
            self._weights = np.ones(positions.size + cell, dtype=np.float64)
            self._active = np.ones(positions.size + cell, dtype=bool)
            self._frame_rank = 0
            return
        masses = getattr(getattr(state, 'pmd', None), 'system', None)
        masses = np.asarray(getattr(masses, 'mass', None) or [],
                            dtype=np.float64).reshape(-1)
        if masses.size != natom:
            raise ValueError(
                "the transition-state search projects the rigid motions out of "
                "the curvature, and that needs one mass per atom: the system "
                "carries %d masses for %d atoms" % (masses.size, natom))
        # A `fix` takes components out of the game before the rigid motions are
        # projected: the frozen ones cannot move, so they are not directions the
        # search is allowed to climb, and the whole-body motions that remain are
        # The counted-from-what-survives-on-the-movable-components mask comes
        # from `note_evaluation`, which cached the plan's freeze mask before this
        # frame is built; one source, so the step, the basis and the count cannot
        # disagree about which components are pinned.
        if lattice is None:
            basis, weights, _center, rank = vibrational_basis(
                positions.reshape(-1, 3) * ANGSTROM_TO_BOHR, masses,
                mask=self._frozen_mask)
        else:
            # A search that relaxes the box steps in the joint space, so the
            # whole-body motions it must not climb are the ones of the structure
            # *and* its cell: a rotation of the atoms alone is a deformation here,
            # and the metric has no mass to offer the six lattice entries.
            basis, weights, _center, rank = periodic_vibrational_basis(
                positions.reshape(-1, 3) * ANGSTROM_TO_BOHR, masses, lattice,
                mask=self._frozen_mask)
        if basis.shape[1] == 0:
            raise ValueError(
                "the transition-state search has no vibrational direction left "
                "to follow at %d atoms: every Cartesian degree of freedom is a "
                "rigid motion of this geometry or is frozen by the constraint "
                "plan, so the curvature it would climb moves nothing" % natom)
        self._basis = basis
        self._weights = weights
        self._frame_rank = int(rank)
        active = self._active_components()
        if active is not None and active.size < weights.size:
            # The lattice entries are never masked: a run that froze one of them
            # did not ask for a cell block, and a search that climbs it is
            # climbing a coordinate the step cannot move.
            active = np.concatenate((
                active, np.ones(weights.size - active.size, dtype=bool)))
        self._active = (np.ones(weights.size, dtype=bool) if active is None
                       else active)

    def _joint_size(self, atomic_size):
        """The size of the vector this search steps on."""
        from ...utilities.lattice import CELL_COMPONENT_SLOTS

        return atomic_size + (len(CELL_COMPONENT_SLOTS)
                              if self.cell_variables else 0)

    def _step_gradient(self, gradient):
        """Solve the step on the tangential gradient, the one the criteria read.

        The basis already ignores the frozen components, but the raw gradient
        still carries a force on them - the force the constraint supplies - and
        projecting that out of the *step* after the fact leaves the model solved
        against a vector that is not a direction of its own subspace.
        """
        return self._tangent_of(gradient)

    def _compute_step(self, gradient):
        """The partitioned RFO step, read in the vibrational subspace."""
        if self._hessian is None:
            raise ValueError(
                "the transition-state search has no curvature to follow: the "
                "framework Hessian computation returned no matrix. Run the "
                "'hessian' jobtype over this geometry, or ask for a minima "
                "search with optimize.method")
        if self._basis is None:
            raise ValueError(
                "the transition-state search was asked to step before an "
                "evaluation was noted, so it has no geometry to build its "
                "vibrational subspace from")
        gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
        # Into the mass-weighted metric, then into the complement of the rigid
        # motions: the model's curvature there is the restoring force the
        # frequency calculation will read, and nothing outside it can be climbed.
        weights = self._weights
        model = self._hessian * np.outer(weights, weights)
        reduced = self._basis.T @ model @ self._basis
        values, vectors = np.linalg.eigh(0.5 * (reduced + reduced.T))
        projected = vectors.T @ (self._basis.T @ (weights * gradient))
        mode = self._select_follow_mode(values, vectors)
        # Both readings are of the model, and the report says so: after the
        # first step neither comes from an evaluation of the surface. The count
        # is the model's negative curvatures among the vibrational directions,
        # which is the number of imaginary frequencies an evaluation there would
        # report.
        self.negative_modes = int(np.count_nonzero(values < 0.0))
        self.mode_eigenvalue = float(values[mode])
        # Stored in the mass-weighted Cartesian frame, not in the reduced
        # coordinates of this geometry's basis. The basis is rebuilt at every
        # evaluation from the geometry there, so its columns are different
        # vectors each step even when the molecule has barely moved; comparing
        # component indices across two of them compares one mode's shadow in two
        # frames, which is not a mode-following rule at all. Kept in the fixed
        # frame, the followed direction is one physical motion of the atoms and
        # the next step projects it into whatever basis the new geometry has.
        self._followed = (self._basis @ vectors[:, mode]).copy()
        step = self._partitioned_rfo_step(values, projected, vectors, mode)
        # Back out: the reduced step is a mass-weighted displacement in the
        # vibrational basis, the workflow steps in Cartesian Bohr.
        return (self._basis @ step) / weights

    def _select_follow_mode(self, values, vectors):
        """The direction to climb: the lowest curvature first, then its descendant.

        Choosing by eigenvalue order every step would hand the search a
        different direction as soon as two curvatures cross, which on the way
        into a saddle is often; the overlap test keeps one search on one mode.
        The direction stored by the last step is in the fixed mass-weighted
        frame, so it is read through the basis of *this* geometry before the
        comparison - the two bases are not the same vectors.
        """
        if self._followed is None:
            return int(np.argmin(values))
        carried = self._basis.T @ self._followed
        overlaps = np.abs(vectors.T @ carried)
        return int(np.argmax(overlaps))

    def _partitioned_rfo_step(self, values, projected, vectors, mode):
        """The step, one block per half of the saddle condition.

        Along the followed mode the 2x2 augmented problem's *upper* root is
        taken: it lies above the curvature itself, so the denominator is
        negative and the coefficient sends the step uphill along the gradient
        component of that mode. Over the rest the *lower* root of their
        augmented problem is taken: it lies below every curvature in the block,
        so each denominator is positive and each coefficient descends. A model
        with no negative curvature needs no special case here - the upper root
        climbs whether the curvature is negative or positive, which is the
        reason this replaced a Newton step that had to flip a direction to
        climb at all.
        """
        climb_curvature = float(values[mode])
        climb_gradient = float(projected[mode])
        upper_root = 0.5 * (climb_curvature
                            + np.sqrt(climb_curvature * climb_curvature
                                      + 4.0 * climb_gradient * climb_gradient))
        coefficients = np.zeros_like(values)
        denominator = climb_curvature - upper_root
        if abs(denominator) > _RFO_POLE_TOLERANCE:
            coefficients[mode] = -climb_gradient / denominator

        rest = np.array([index for index in range(len(values))
                         if index != mode], dtype=int)
        if rest.size:
            lower_root = _rfo_lower_root(values[rest], projected[rest])
            for index in rest:
                denominator = values[index] - lower_root
                if abs(denominator) > _RFO_POLE_TOLERANCE:
                    coefficients[index] = -projected[index] / denominator
        return vectors @ coefficients

    def _rejection_reason(self, actual_de):
        """Reject a step whose energy change the curvature model did not see.

        Climbing is the point of this search, so the minima rule is off and an
        energy rise on its own is not evidence of anything. What is evidence is
        a *disagreement in direction*: the model said this move would cost
        `predicted` and the surface returned the opposite sign, so the step was
        taken on a model that does not describe the ground it crossed. Accepting
        such a step and then shrinking only the next one is what made a run of
        this class crawl: the radius fell to its floor within five evaluations
        (measured: predicted -2.8e-8 against actual +1.0e-6, a ratio of -36, on a
        surface whose curvature the model still counted correctly), and a step
        capped at 1e-4 Bohr cannot walk the 1e-2 of residual gradient it still
        had to walk. Rejecting instead restores the geometry, so the smaller
        step is retaken from the ground the model was fitted on - and four such
        in a row stops the run by name rather than spending the remaining budget
        on a radius that cannot move.

        A prediction of zero says nothing either way, and is not judged; the
        floor is the same one the radius resize already reads.
        """
        predicted = self._predicted_de()
        if predicted is None or abs(predicted) < self.PREDICTION_FLOOR:
            return None
        if predicted * actual_de < 0.0:
            return ('the energy changed by %+.8f Hartree where the curvature '
                    'model predicted %+.8f' % (actual_de, predicted))
        return None

    def _predicted_de(self):
        """What the curvature model says the last step changed.

        Read straight off the Hessian the step was solved against,
        `g . s + 1/2 s . B s`, so no inversion enters the prediction that sizes
        the trust radius. Without this the search would inherit the inverse
        model's reading, which it never built.
        """
        step = self.last_displacement
        if step is None or self._hessian is None:
            return None
        if not self.history['gradients']:
            return None
        gradient = self.history['gradients'][-1]
        return (float(np.dot(gradient, step))
                + 0.5 * float(np.dot(step, self._hessian @ step)))

    def status_method_columns(self):
        """The saddle's curvature readings, plus the frame count for a box."""
        columns = list(super().status_method_columns())
        columns += ['neg_modes', 'mode_eig']
        if self.cell_variables:
            columns.append('rigid_removed')
        return tuple(columns)

    def status_method_values(self, report):
        """Values matching the saddle columns, in the same order."""
        values = list(super().status_method_values(report))
        for key in ('Negative Modes (model)', 'Mode Eigenvalue (model)'):
            value = report.get(key)
            values.append(float('nan') if value is None else float(value))
        if self.cell_variables:
            value = report.get('Rigid Directions Removed')
            values.append(float('nan') if value is None else float(value))
        return tuple(values)

    def _extra_criteria(self, report):
        """Require exactly one downhill direction, and print what was found.

        The count is labelled a model reading because after the first step it
        comes from updates of the initial curvature, not from a second Hessian
        evaluation. A run that stops at a quiet gradient with two downhill
        directions has found a higher-order saddle, not a transition state, and
        the row is what tells the two apart.
        """
        modes = self.negative_modes
        report['Negative Modes (model)'] = float(
            -1 if modes is None else modes)
        report['Mode Eigenvalue (model)'] = float(
            self.mode_eigenvalue if self.mode_eigenvalue is not None else 0.0)
        if self.cell_variables:
            # Only a box-relaxing search reports this, because only there can the
            # count be less than the six a vacuum structure always gives: the
            # triangular spelling the engines read back has no room for a rotation
            # that strains an entry above the diagonal, and a curvature count read
            # in a frame that lost a direction is a count the run should show.
            report['Rigid Directions Removed'] = float(
                -1 if self._frame_rank is None else self._frame_rank)
        return modes == 1


def _rfo_lower_root(curvatures, projected):
    """The lowest root of the augmented problem of the descended block.

    The augmented matrix is the block's diagonal curvature with the projected
    gradient bordering it; its lowest eigenvalue is the RFO root the block is
    measured against (Baker, J. Comput. Chem. 7:385, 1986). It sits below every
    curvature in the block, which is what makes each coefficient a descent.
    """
    size = curvatures.size
    augmented = np.zeros((size + 1, size + 1), dtype=np.float64)
    augmented[np.arange(size), np.arange(1, size + 1)] = curvatures
    augmented[0, 1:] = projected
    augmented[1:, 0] = projected
    return float(np.linalg.eigvalsh(augmented)[0])
