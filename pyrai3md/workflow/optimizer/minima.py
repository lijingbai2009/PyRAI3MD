"""
======================================================================
This module provides the MinimaOptimizer for geometry optimization
towards energy minima. It combines three responsibilities in a
single optimizer:

    BFGS         - Hessian inverse maintenance and update
    Step         - Newton step from the BFGS inverse Hessian
                   (default), or RFO step from the augmented
                   Hessian eigenvector method (optional)
    Trust Radius - Upper bound of the step length, resized from
                   how well the quadratic model predicted the
                   energy change of the step it just allowed

The default combination is BFGS + Trust Radius. The RFO step can be
enabled via the use_rfo option. On the first step (no history) the
optimizer falls back to steepest descent. When a step is rejected
(energy increase) the geometry is restored and a shorter step is
taken along the same direction; a run that gets several rejections in
a row stops, instead of spending the rest of its budget on a direction
the model keeps mispricing.

Aug 07 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from ...utilities.lattice import CELL_COMPONENT_SLOTS
from .base import Optimizer, usable_curvature_product

# The bounds a resized trust radius may move between, in Bohr. The floor is
# what keeps a rough surface from shrinking the step to nothing and grinding
# through the remaining budget; the ceiling sits above the initial 0.3 because
# the first step is the least informed one a run takes.
TRUST_RADIUS_MIN = 1.0e-4
TRUST_RADIUS_MAX = 0.5
# The three bands rho = actual / predicted is read against. Below the lower one
# the model was wrong enough to shrink, above the upper one it was right enough
# to grow; in between the radius stays, so a run cannot oscillate on it.
TRUST_SHRINK_RATIO = 0.25
TRUST_GROW_RATIO = 0.75
TRUST_SHRINK_FACTOR = 0.25
TRUST_GROW_FACTOR = 2.0
# How many rejections in a row a minima search is given before its workflow
# stops it. One rejection costs an evaluation to discover, so four is four
# evaluations spent on a step the local model keeps getting wrong.
MAX_CONSECUTIVE_REJECTIONS = 4
# The floor under a run's own energy criterion that a rise must clear to cost a
# rejection. It is the threshold this rule has always used, kept as a floor so a
# run that declares a tighter `opt_energy` than this does not start rejecting on
# roundoff.
REJECTION_RISE_FLOOR = 1.0e-6


class MinimaOptimizer(Optimizer):
    """Optimizer for energy minima using BFGS + Trust Radius.

    BFGS maintains the inverse Hessian approximation H through the
    standard rank-2 update. The default step is the Newton step
    -H @ g clipped to the trust radius. When use_rfo is enabled the
    step is computed from the augmented Hessian eigenvalue problem
    instead. The trust radius bounds the step length and is resized
    from the agreement between the model's predicted energy change
    and the one the evaluation returned.

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
        Whether to use the RFO step instead of the default
        BFGS Newton step.
    """

    # A search that has to go uphill somewhere says so by turning this off;
    # the crossing optimizers do, because a rising average energy is what
    # closing a gap costs.
    reject_on_energy_rise = True

    # The radius bounds are read through the instance, so that a search whose
    # notion of a long step differs from a minima relaxation can set its own.
    # These are the bounds a minima search has always used, carried over from
    # the module constants above.
    TRUST_RADIUS_MIN = TRUST_RADIUS_MIN
    TRUST_RADIUS_MAX = TRUST_RADIUS_MAX
    # Below this a step's predicted energy change is a rounding of the print
    # rather than a claim about the surface, so it is not read as evidence about
    # the step. A search whose own convergence scale is coarser than this says so
    # by setting it from `energy_threshold`; the minima relaxation keeps the
    # roundoff floor it has always used.
    PREDICTION_FLOOR = 1.0e-14

    def __init__(self, maxiter=100, energy_threshold=1e-6,
                 gradient_threshold=3e-4, displacement_threshold=3e-4,
                 trust_radius=0.3,
                 use_rfo=False,
                 **kwargs):
        super().__init__(
            maxiter=maxiter,
            energy_threshold=energy_threshold,
            gradient_threshold=gradient_threshold,
            displacement_threshold=displacement_threshold,
            trust_radius=trust_radius,
            **kwargs,
        )
        self.use_rfo = bool(use_rfo)
        # The curvature model is kept in both directions: the inverse Hessian
        # the Newton step is built from, and the direct Hessian the predicted
        # energy change and the RFO step are read on. Keeping the second is what
        # lets both of those stay at two matrix-vector products instead of an
        # n by n inversion per step, which on a 1575-atom box (n = 4731) costs
        # 2.25 s against the 29 ms the engine takes for the evaluation the model
        # is fitted to.
        self._inv_hessian = None
        self._hessian = None

    # ---- Algorithm implementation ----

    def _compute_step(self, gradient):
        """Compute the optimization step via BFGS.

        On the first call (no history) returns a steepest descent
        step. On subsequent calls:
        1. Update the inverse Hessian with BFGS.
        2. Compute the step: Newton step -H @ g by default, or the
           RFO step from the augmented Hessian when use_rfo is set.
           The step is clipped to the trust radius by the base class.

        Parameters
        ----------
        gradient : np.ndarray
            Flattened gradient (3N,).

        Returns
        -------
        np.ndarray
            Displacement vector (3N,).
        """
        n = len(gradient)

        # First step: steepest descent within trust radius.
        if len(self.history['gradients']) < 2:
            self._inv_hessian = np.eye(n)
            self._hessian = np.eye(n)
            return -gradient * self.trust_radius

        # 1. BFGS inverse Hessian update
        s = (self.history['coordinates'][-1]
             - self.history['coordinates'][-2])
        y = (self.history['gradients'][-1]
             - self.history['gradients'][-2])

        # Re-initialise after a Hessian reset (e.g. step rejection).
        if self._inv_hessian is None:
            self._inv_hessian = np.eye(n)
        if self._hessian is None:
            self._hessian = np.eye(n)

        self._bfgs_update(s, y)

        # 2. Step from the BFGS inverse Hessian
        if self.use_rfo:
            step = self._rfo_step(gradient)
        else:
            step = -self._inv_hessian @ gradient

        return step

    def _step_gradient(self, gradient):
        """Solve and fit on the tangential gradient when something is pinned.

        The two properties this relaxation depends on are the curvature product
        `s . y` and the descent direction `-H g`, and both are only meaningful on
        coordinates the run can cross. A pinned component keeps the force its
        constraint supplies, so a raw `y` carries constraint reactions and the
        rank-2 update spreads them through the inverse Hessian's movable block,
        and a raw `g` leaks the same forces into the step that `apply_constraint`
        then cuts away. With both halves tangential the update needs no extra
        shielding: `s` and `y` are zero on the pinned components, so the outer
        products vanish there and the frozen rows and columns of the model are
        never excited at all.
        """
        return self._tangent_of(gradient)

    def _bfgs_update(self, s, y):
        """Update both directions of the curvature model with the BFGS formula.

        H_new = (I - rho s y^T) H (I - rho y s^T) + rho s s^T

        where rho = 1 / (y^T s), s is the displacement and y is the
        gradient change. This form preserves symmetry and positive
        definiteness: a positive definite H and y^T s > 0 guarantee
        a positive definite H_new, so the Newton step is always a
        descent direction. The update is skipped when s . y is not
        positive, or is negligible against the lengths of the two
        vectors (`base.usable_curvature_product`).

        The direct Hessian is updated beside it by the formula that is
        its own rank-2 form,

        B_new = B - (B s)(B s)^T / (s^T B s) + rho y y^T,

        so neither direction is ever obtained by inverting the other. What
        the two are for differs: the inverse gives the Newton step, the direct
        matrix gives the predicted energy change of the step that was taken and
        the augmented eigenvalue problem of the RFO step, and reading either of
        those through an inversion costs O(n^3) per step.

        Parameters
        ----------
        s : np.ndarray
            Displacement vector (3N,) from the previous step.
        y : np.ndarray
            Gradient change (3N,) from the previous step.
        """
        sy = usable_curvature_product(s, y)
        if sy is None:
            return

        rho = 1.0 / sy
        # Both directions are updated in their rank-2 form: one matrix-vector
        # product and two outer products each. The inverse update written as the
        # product `(I - rho s y^T) H (I - rho y s^T) + rho s s^T` is the same
        # matrix, but evaluating it as written costs two dense n by n
        # multiplications, that is O(n^3) on every step: on a 1575-atom box
        # (n = 4731) seconds, against the 29 ms the engine spends on the
        # evaluation the update is fitted to.
        H = self._inv_hessian
        Hy = H @ y
        yHy = float(np.dot(y, Hy))
        ss = np.outer(s, s)
        self._inv_hessian = (H
                             - rho * (np.outer(s, Hy) + np.outer(Hy, s))
                             + (rho + rho * rho * yHy) * ss)

        # The direct Hessian carries the same information from the other side,
        # updated by its own rank-2 formula, and it is what the predicted energy
        # change and the RFO step read. A curvature product that was refused
        # above leaves it alone, so the two can only differ by roundoff - which
        # is what the guard in `qualification/verification/test_optimizer.py`
        # watches.
        B = self._hessian
        if B is not None:
            Bs = B @ s
            sBs = float(np.dot(s, Bs))
            if sBs > 0.0:
                self._hessian = (B - np.outer(Bs, Bs) / sBs
                                 + rho * np.outer(y, y))

    def _rfo_step(self, gradient):
        """Compute the RFO step from the augmented Hessian.

        Builds the augmented Hessian from the Hessian obtained by
        inverting the BFGS inverse Hessian approximation:

            [[B,    g]]
            [[g^T,  0]]

        and extracts the step from the eigenvector corresponding to
        the lowest eigenvalue. For a positive definite Hessian the
        RFO step is always shorter than the Newton-Raphson step and
        smoothly reduces towards steepest descent when the gradient
        is large.

        Parameters
        ----------
        gradient : np.ndarray
            Current flattened gradient (3N,).

        Returns
        -------
        np.ndarray
            RFO displacement vector (3N,).
        """
        n = len(gradient)
        H = (self._inv_hessian if self._inv_hessian is not None
             else np.eye(n))

        # RFO needs the Hessian B, not its inverse, and the curvature update
        # keeps both directions, so nothing here inverts one out of the other.
        # What an RFO step does still pay is the eigenvalue problem below, which
        # is the O(n^3) that step is defined by.
        B = self._hessian
        if B is None:
            try:
                B = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                # Fallback to Newton step when the inversion fails
                return -H @ gradient
        B = 0.5 * (B + B.T)

        # Build augmented Hessian (n+1 x n+1)
        augmented = np.zeros((n + 1, n + 1))
        augmented[:n, :n] = B
        augmented[:n, n] = gradient
        augmented[n, :n] = gradient

        # Solve the eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eigh(augmented)

        # The RFO step corresponds to the lowest eigenvalue
        idx = np.argmin(eigenvalues)
        eigvec = eigenvectors[:, idx]

        # Fix the eigenvector sign so the denominator is positive
        if eigvec[n] < 0:
            eigvec = -eigvec

        # Extract step: the first n components divided by the last
        if abs(eigvec[n]) < 1e-10:
            # Fallback to Newton step when the denominator is zero
            step = -H @ gradient
        else:
            step = eigvec[:n] / eigvec[n]

        return step

    # ---- Step acceptance ----

    @property
    def stalled(self):
        """True when the last steps were rejected often enough to stop.

        The workflow owns the runtime outcome; this is the reading it stops on.
        """
        return self.failure_count >= MAX_CONSECUTIVE_REJECTIONS

    def run(self, state):
        """Compute next geometry with step acceptance.

        Extends the base run() with step acceptance evaluation. The energy at
        the stepped-to geometry only becomes available on the next call, so the
        previous step is evaluated here: how much the energy really changed
        against what the quadratic model said it would change is what the trust
        radius is resized on, and when the energy increased the step is
        rejected, the coordinates restored, and a shorter step taken along the
        same direction under the radius the rejection just shrank.

        Parameters
        ----------
        state : PyRAIState
            Current state with energy/gradient.

        Returns
        -------
        PyRAIState
            State with updated coordinates.
        """
        result = state.result
        energy, _ = self._objective(state, result)

        # Evaluate the previous step now that the energy at the
        # stepped-to geometry is known (skip on the first call). The
        # resize happens whether or not the rise is a rejection: a search
        # that climbs on purpose still wants to know how well its model
        # predicted the climb.
        if energy is not None and self.history['energies']:
            actual_de = float(energy - self.history['energies'][-1])

            reason = self._rejection_reason(actual_de)
            if reason is not None and not self._rejection_offers_a_shorter_step(
                    self.last_displacement):
                # A rejection exists to replace the step with a shorter one. At
                # the radius floor there is no shorter step to take: the move it
                # would restore to and re-step onto is the move just rejected, so
                # the run repeats one geometry forever (measured on an analytic
                # saddle: the same "changed by +6.7e-7 where the model predicted
                # -3.8e-8" line printed 26 evaluations running, radius held at
                # 1e-4 Bohr, the history length frozen). Keep the step, say why,
                # and let the search spend its remaining budget on ground rather
                # than on its own bookkeeping.
                self.logger.warning(
                    "MinimaOptimizer: rejection skipped, the radius is already "
                    "at its floor of %.4f Bohr (%s); the step is kept",
                    self.TRUST_RADIUS_MIN, reason)
                reason = None

            if reason is not None:
                # Reject: the previous step is one the search will not keep. The
                # radius shrinks before the re-step so the shorter move it takes
                # is the move the new radius allows, not a halving of a step the
                # model has just been proved wrong about.
                self.failure_count += 1
                self.trust_radius = max(
                    self.trust_radius * TRUST_SHRINK_FACTOR,
                    self.TRUST_RADIUS_MIN)
                self.logger.warning(
                    "%s: step rejected (%s); radius shrunk to "
                    "%.4f Bohr, rejection %d of %d", type(self).__name__,
                    reason, self.trust_radius,
                    self.failure_count, MAX_CONSECUTIVE_REJECTIONS)
                return self._reject_and_restep(state)

            self.failure_count = 0
            self._resize_trust_radius(actual_de)

        return super().run(state)

    def _rejection_offers_a_shorter_step(self, direction=None):
        """Whether rejecting would actually allow a shorter move.

        The re-step is the rejected direction made shorter, so a rejection only
        changes anything while either of the two ways of shortening it bites: the
        shrunken radius capping the move, or, when the move is already inside the
        radius (which is what a line search hands back), the factor that cuts it
        down. Reading only the radius is what let a variable-cell run refuse the
        same geometry four times in a row: the move the line search had already
        cut to 2.5e-3 Bohr sat below every radius the shrink produced, so
        restoring and re-stepping reproduced the refused step bit for bit and the
        energy rise repeated to the last printed digit.
        """
        radius_after = self.trust_radius * TRUST_SHRINK_FACTOR
        if radius_after > self.TRUST_RADIUS_MIN:
            return True
        if direction is None:
            direction = self.last_displacement
        if direction is None:
            return False
        return (float(np.linalg.norm(direction)) * TRUST_SHRINK_FACTOR
                > self.TRUST_RADIUS_MIN)

    def _rejection_reason(self, actual_de):
        """Why the last step costs a rejection, or None to keep it.

        A relaxation has no business moving uphill, and the rise the evaluation
        reports is the evidence. What counts as a rise is bounded below by the
        run's own energy criterion: a step whose whole offence is smaller than
        the threshold the run declares itself converged under cannot be told
        apart from a step that satisfies it, and refusing it spends an
        evaluation undoing nothing while the trust radius - which the ratio rule
        in `_resize_trust_radius` shrinks on the same reading - collapses.

        Measured on a 1575-atom water box relaxing its own cell: the four rises
        that stopped the run at step 206 were 4.47e-2, 2.67e-3, 1.37e-4 and
        2.88e-6 Hartree against an `opt_energy` of 1e-5. The first three are
        real overshoots and each shrank the radius correctly (0.5 -> 0.125 ->
        0.031 -> 0.0078 Bohr); the fourth, five times below the run's own
        criterion, took the radius to 0.0020 and left the cell displacement at
        6.8e-7 Bohr with a cell gradient still at 0.049 - a search that had
        stopped moving the box it was asked to relax.

        A search that climbs on purpose overrides this and says what its own
        evidence is; the machinery above (shrink, restore, re-step, count toward
        `stalled`) is the same either way.
        """
        if not self.reject_on_energy_rise:
            return None
        if actual_de <= max(REJECTION_RISE_FLOOR, float(self.energy_threshold)):
            return None
        predicted = self._predicted_de()
        if predicted is None or predicted == 0.0:
            return 'the energy rose by %.8f Hartree' % actual_de
        return ('the energy rose by %.8f Hartree where the model predicted '
                '%.8f (rho = %+.3f)' % (actual_de, predicted,
                                        actual_de / predicted))

    def _predicted_de(self):
        """What the model behind the last step said that step would cost.

        The same quadratic the step was built from, read on the step that was
        taken: `g . s + s^T B s / 2`. It is read on the direct Hessian the
        update keeps beside the inverse, so it costs two matrix-vector products
        rather than inverting the inverse - which on a 1575-atom box (n = 4731)
        is 2.25 s per step against the 29 ms the engine spends on the evaluation
        the model is fitted to.
        """
        step = self.last_displacement
        if step is None or not self.history['gradients']:
            return None
        curvature = self._hessian
        if curvature is None:
            return None
        gradient = self.history['gradients'][-1]
        return (float(np.dot(gradient, step))
                + 0.5 * float(np.dot(step, curvature @ step)))

    def _resize_trust_radius(self, actual_de):
        """Grow, keep or shrink the radius on how the accepted step went."""
        predicted = self._predicted_de()
        if predicted is None or abs(predicted) < self.PREDICTION_FLOOR:
            return
        ratio = actual_de / predicted
        if ratio < TRUST_SHRINK_RATIO:
            self.trust_radius = max(self.trust_radius * TRUST_SHRINK_FACTOR,
                                    self.TRUST_RADIUS_MIN)
        elif ratio > TRUST_GROW_RATIO:
            self.trust_radius = min(self.trust_radius * TRUST_GROW_FACTOR,
                                    self.TRUST_RADIUS_MAX)

    def _reject_and_restep(self, state):
        """Reject the previous step and take a smaller one.

        Restores the previous geometry from the optimization history and
        re-steps along the rejected direction, capped by the radius the
        rejection shrank. The inverse Hessian is kept for the subsequent BFGS
        updates. A variable-cell run restores the box with the atoms: the
        rejected move was one vector over both halves, and putting only the
        atoms back would leave the box where a step the run has just
        disowned moved it. No extra electronic structure calculation is
        consumed.

        Parameters
        ----------
        state : PyRAIState
            State carrying the energy/gradient of the rejected
            geometry; its coordinates are restored and re-stepped.

        Returns
        -------
        PyRAIState
            State with coordinates updated from the previous
            geometry along the new smaller step.
        """
        previous = self.history['coordinates'][-1]
        direction = self.last_displacement
        if direction is None:
            return state

        # The rejected move is the vector the step was taken on: the atomic
        # components, and the lattice entries when the run relaxes the box. It
        # is read back from what was recorded rather than from the difference of
        # two geometries, because a variable-cell run's new geometry also carries
        # the ride of the atoms with the box, and a direction measured on the
        # coordinates would re-step that ride a second time.
        atomic_size = np.asarray(state.coordinates).size
        has_cell = direction.size > atomic_size
        rejected_norm = float(np.linalg.norm(direction))
        displacement = self._apply_trust_radius(direction)
        if float(np.linalg.norm(displacement)) >= rejected_norm:
            # The radius did not bite: the move the run is refusing is already
            # shorter than it (a line search hands back exactly this). Cut the
            # move itself, or the re-step writes the geometry that was just
            # refused and the run refuses it again with the same digits.
            displacement = displacement * TRUST_SHRINK_FACTOR
        # Take the immovable components out again, the way the accepted path
        # does. The direction being re-scaled was already projected, so this is
        # idempotent today; it is here so that "the step is cut down to what the
        # run may move" has exactly one place where it happens.
        atomic_part = self.apply_constraint(state,
                                            displacement[:atomic_size])
        displacement = (np.concatenate((atomic_part, displacement[atomic_size:]))
                        if has_cell else atomic_part)

        self.last_displacement = displacement.copy()
        if self.history['displacements']:
            self.history['displacements'][-1] = displacement.copy()

        # The geometry the rejected move was made from: the atomic block history
        # kept, the box whose six entries history kept beside it, and the
        # crossing counts of that structure.
        lattice = None
        base_image = None
        base_bohr, base_angstrom, step_lattice, base_image = self.step_base
        if has_cell:
            slots = list(CELL_COMPONENT_SLOTS)
            flat = step_lattice.reshape(-1).copy()
            flat[slots] = previous[previous.size - len(slots):]
            lattice = flat.reshape(3, 3)
        self._apply_step(state, base_bohr, base_angstrom, lattice,
                         displacement, image=base_image)
        self.iteration += 1

        return state
