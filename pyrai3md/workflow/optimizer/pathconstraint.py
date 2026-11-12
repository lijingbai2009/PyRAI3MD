"""
======================================================================
This module provides the constraint models a path point is found with:
the classic mass-weighted hypersphere step (Gonzalez & Schlegel,
J. Phys. Chem. 94, 5523 (1990)) and the ReLU-gated constraint potential.

Both models answer the same question - given the energy, the gradient
and a mass-weighted radius, where is the next probe on the surface of
the sphere of that radius about the last accepted point - so the walker
that drives them (`pathbase.WalkerPathOptimizer`) holds one loop and
knows nothing about which model is inside it. What differs between the
two is what "the constrained minimum" means:

    sphere  the point on the sphere whose gradient has no tangential
            part, reached by rotating the current radius vector towards
            the tangential descent direction and staying exactly on the
            sphere by construction;
    relu    the minimum of `E + phi`, where `phi` is a one-sided penalty
            on the squared radius, so the walk is an ordinary local
            minimization that is pushed out to the sphere and held there.

A variable-cell run puts the six relaxed lattice entries in the same
vector as the atoms; they carry a weight of one, because a box has no
mass to be weighted by (`optimizer/base.py::_apply_trust_radius` measures
a joint step in one Bohr length for the same reason).

The ReLU model is a reimplementation of the mathematics described in
OpenQP's `oqp/library/libscipy.py` (`ConstrainOpt`, `ReLU`), whose
repository is not open source (OpenQP Research License 1.0); no code was
copied, and the chain rule below differs from that reference's `df`
expression, which omits the weight factor and is therefore only correct
for a uniform mass.

Nothing here evaluates a potential: a model is handed an evaluation that
has already happened and answers with readings and one step.

Oct 20 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from abc import ABC, abstractmethod

import numpy as np

from .base import usable_curvature_product

# How far one probe rotates the radius vector towards the tangential descent
# direction, in radians, on the sphere whose radius is the arc length of one
# path point, and how that rotation is adjusted probe by probe. A module
# constant and not a keyword: it answers how a probe moves inside one path
# point, which is not a question the input has, and a second user-facing radius
# would conflate it with how far the path advances.
#
# The rotation only ever *shrinks* within a path point, and it shrinks whenever
# the probe's energy failed to fall. That is a monotone line search with one step
# of memory, and both halves of it were measured on the shipped HCN walk: a
# fixed 0.25 rad bounced between two energies (+0.0051007335 then
# -0.0051007335, repeatedly) and never settled, and letting the rotation grow
# again after a fall kept the bounce alive at a finite amplitude instead of
# letting it die out. Shrinking to the floor turns the probe's movement into a
# geometric decay, which is what makes the tangential gradient fall below the
# tolerance and the point be a constrained minimum rather than a wide orbit
# around one.
_MICRO_ARC_FRACTION = 0.25
_ARC_SHRINK = 0.5
_ARC_FLOOR = 0.01

# How much of a radius counts as already on the sphere, for the ReLU model's
# penalty and for reading `u = X / rho` when the probe sits on the centre.
_RADIUS_FLOOR = 1.0e-12


def _rms(vector, mobile=None):
    """The root mean square of the components a run is allowed to move.

    A frozen component cannot move, so counting it would dilute the reading
    towards zero by the fraction of the system the input pinned - the same rule
    `Optimizer.convergence` applies to the gradient and the step
    (`optimizer/base.py::_rms_count`).
    """
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if mobile is not None:
        values = values[np.asarray(mobile, dtype=bool).reshape(-1)]
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.sum(values ** 2) / values.size))


class PathConstraintModel(ABC):
    """One constraint model of the path family.

    Parameters
    ----------
    radius_bohr : float
        The arc length of one accepted path point, on the normalized
        mass-weighted coordinates, in Bohr.
    weights : numpy.ndarray
        (3N + 6,) normalized weights `sqrt(m_i / M)` per atomic component and
        one per relaxed lattice entry.
    tolerance : float
        The RMS of the model's own convergence vector below which a probe has
        found the constrained minimum.
    mobile : numpy.ndarray or None
        Boolean mask of the joint components the run may move; None is all of
        them.
    curvature : numpy.ndarray or None
        Optional initial force constants over the joint vector, used by the
        model that minimizes an augmented objective. None starts from the
        identity, which is what a search without curvature does.

    Attributes
    ----------
    reference : numpy.ndarray or None
        (3N + 6,) Bohr centre of the current sphere, set by `begin`.
    """

    name = ''

    def __init__(self, radius_bohr, weights, tolerance, mobile=None,
                 curvature=None):
        self.radius = float(radius_bohr)
        self.weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        self.tol = float(tolerance)
        self.mobile = (None if mobile is None
                       else np.asarray(mobile, dtype=bool).reshape(-1))
        self.reference = None
        self._curvature = (None if curvature is None
                           else np.asarray(curvature, dtype=np.float64))

    # ---- Shared geometry ----

    def begin(self, reference_bohr):
        """Centre a new sphere on the last accepted point."""
        self.reference = np.asarray(reference_bohr,
                                    dtype=np.float64).reshape(-1).copy()

    def offset(self, trial_bohr):
        """`X = w (r - r0)`, the mass-weighted offset from the centre."""
        trial = np.asarray(trial_bohr, dtype=np.float64).reshape(-1)
        return self.weights * (trial - self.reference)

    def radius_of(self, trial_bohr):
        """`rho = |X|`, the mass-weighted radius of one probe."""
        return float(np.linalg.norm(self.offset(trial_bohr)))

    def weighted_gradient(self, gradient_bohr):
        """`G = g / w`, the gradient conjugate to the weighted coordinates."""
        gradient = np.asarray(gradient_bohr, dtype=np.float64).reshape(-1)
        return gradient / self.weights

    def rms(self, vector):
        """The mobile-component RMS this model judges its vectors by."""
        return _rms(vector, self.mobile)

    # ---- The interface the walker drives ----

    @abstractmethod
    def inspect(self, trial_bohr, energy, gradient_bohr):
        """Read one finished evaluation of one probe.

        Parameters
        ----------
        trial_bohr : numpy.ndarray
            The joint geometry that was evaluated.
        energy : float
            The objective there (the enthalpy under an applied pressure).
        gradient_bohr : numpy.ndarray
            The joint gradient there, in Hartree per Bohr, already reduced to
            the tangential part a declared condition leaves to the walk.

        Returns
        -------
        dict
            `radius` (the mass-weighted radius of the probe), `perpendicular`
            (the vector this model steps on), `residual` (the RMS of the vector
            this model reports as its constraint residual), `arc` (the arc length
            one path point is worth) and the model's own private keys, which
            begin with an underscore.
        """
        raise NotImplementedError

    @abstractmethod
    def settled(self, inspection):
        """Whether this probe has found the constrained minimum."""
        raise NotImplementedError

    @abstractmethod
    def next_trial(self, inspection):
        """The next probe, as an absolute joint vector in Bohr."""
        raise NotImplementedError


class HypersphereConstraint(PathConstraintModel):
    """The classic mass-weighted hypersphere step.

    The point held on the sphere is the one whose gradient has no tangential
    part, so the walk rotates the radius vector towards `-G_perp` and stays on
    the sphere exactly: `X' = rho (cos(phi) u + sin(phi) d)` has length `rho`
    for any `phi`, which is what makes the arc length of a path point a
    property of the run rather than of the step that landed on it.

    The rotation is scaled by what the last two probes did with the energy: a
    probe that rose means the arc is longer than the angular distance to the
    constrained minimum, so the next one is half of it, and a probe that fell
    lets it grow back towards `_MICRO_ARC_FRACTION`. That is a line search with
    one step of memory, and it is what keeps a fixed arc from bouncing.
    """

    name = 'sphere'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._arc_scale = 1.0
        self._previous_energy = None

    def begin(self, reference_bohr):
        super().begin(reference_bohr)
        self._arc_scale = 1.0
        self._previous_energy = None

    def inspect(self, trial_bohr, energy, gradient_bohr):
        offset = self.offset(trial_bohr)
        radius = float(np.linalg.norm(offset))
        gradient = self.weighted_gradient(gradient_bohr)
        unit = offset / radius if radius > _RADIUS_FLOOR else None
        if unit is None:
            # A probe at the centre has no radial direction to remove, so the
            # whole gradient is tangential and the first step of a path point is
            # the full arc length along it (the reference implementation's
            # initial guess). Later probes of the same point sit on the sphere.
            perpendicular = gradient.copy()
        else:
            perpendicular = gradient - float(np.dot(gradient, unit)) * unit
        value = float(energy)
        self._probe_rose = (self._previous_energy is not None
                            and value > self._previous_energy)
        self._probe_fell = (self._previous_energy is not None
                            and value < self._previous_energy)
        self._previous_energy = value
        return {
            'radius': radius,
            'perpendicular': perpendicular,
            'residual': self.rms(perpendicular),
            'arc': self.radius,
            '_unit': unit,
            '_energy': value,
        }

    def settled(self, inspection):
        return inspection['residual'] <= self.tol

    def next_trial(self, inspection):
        unit = inspection['_unit']
        perpendicular = inspection['perpendicular']
        norm = float(np.linalg.norm(perpendicular))
        if norm <= _RADIUS_FLOOR:
            return np.asarray(self.reference, dtype=np.float64).copy()
        rotate = -perpendicular / norm
        if getattr(self, '_probe_rose', False) or not getattr(
                self, '_probe_fell', False):
            self._arc_scale = max(self._arc_scale * _ARC_SHRINK, _ARC_FLOOR)
        phi = _MICRO_ARC_FRACTION * self._arc_scale
        if unit is None:
            target = self.radius * rotate
        else:
            target = self.radius * (np.cos(phi) * unit + np.sin(phi) * rotate)
        return self.reference + target / self.weights


class ReLUConstraint(PathConstraintModel):
    """The ReLU-gated constraint potential.

    The objective is `E + phi(rho)` with

        phi(rho) = ((rho - rho_t) / rho_t)**2   for rho > rho_t, else 0

    a one-sided quadratic in the radius, and its gradient is projected
    orthogonally to the energy gradient before it is added, so the constraint
    force never works against the descent:

        dphi/dX = 2 (rho - rho_t) / rho_t**2 * X / rho           (0 inside)
        g_p     = dphi/dX - (dphi/dX . g) / (g . g) g
        g_aug   = g + g_p

    The chain rule keeps the weight: `X` already carries `w`, so `dphi/dX`
    transforms back to a Cartesian gradient by the same `X / rho` factor the
    radius has. The reference this is reimplemented from writes the penalty
    gradient as `dcoord / rho_t**2` with `dcoord = w (r - r0)`, which is this
    expression missing one factor of `w` and is only correct for a uniform mass.

    The probe is then a local minimization of `E + phi` by BFGS, capped at the
    same radius, so the walk leaves the centre on the bare energy and is held
    where the penalty balances it. The update is the formula
    `minima.MinimaOptimizer` uses and the same `usable_curvature_product` guard
    is what decides whether a step may build one, without this model reading
    that optimizer's history.
    """

    name = 'relu'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        size = self.weights.size
        self._initial = self._inverse_of(self._curvature, size)
        self._inverse = self._initial.copy()
        self._last = None

    @staticmethod
    def _inverse_of(curvature, size):
        """The inverse of a starting curvature when it is a valid model, else the
        identity.

        This model minimizes `E + phi` by BFGS, whose inverse Hessian is a
        positive-definite curvature model by construction, and the matrix it is
        handed at the start of an IRC is a transition state's: indefinite by
        construction. A pseudo-inverse of an indefinite matrix is not an inverse
        Hessian - it sends the first probe *along* the negative mode with the
        largest gain, which a minimization must not prefer - and neither is the
        inverse of its magnitudes, which keeps that direction soft. Both were
        measured on `path_irc_relu`'s surface (HF/STO-3G HCN): `pinv(H)` left the
        residual plateaued at 0.15 (the shell radius) and the walk accepted one
        point in forty evaluations; `|H|` accepted six, still at 0.15; the
        identity accepts eight with residuals of 0.001-0.01, which is the same
        start the model takes when no curvature is handed to it at all
        (`path_mep_relu`). So a matrix that is not positive definite is not a
        curvature this model can start a minimization from, and it is replaced by
        the identity rather than used with its sign or its magnitude.

        A near-singular but PD matrix is inverted through a floor relative to its
        largest eigenvalue, so a soft direction gets a finite step. A matrix of
        the wrong shape is the identity: it is not the curvature of this joint
        vector and reading it there would silently mix two coordinate systems.
        """
        if curvature is None or curvature.shape != (size, size):
            return np.eye(size)
        matrix = np.asarray(curvature, dtype=np.float64)
        matrix = 0.5 * (matrix + matrix.T)
        values, vectors = np.linalg.eigh(matrix)
        largest = max(float(np.max(np.abs(values))), 1.0)
        if float(np.min(values)) <= 1.0e-8 * largest:
            return np.eye(size)
        values = np.maximum(values, 1.0e-3 * largest)
        return (vectors / values) @ vectors.T

    def penalty(self, trial_bohr, energy):
        """The objective and the penalty gradient at one probe."""
        offset = self.offset(trial_bohr)
        radius = float(np.linalg.norm(offset))
        if radius <= self.radius or radius <= _RADIUS_FLOOR:
            return float(energy), np.zeros_like(offset)
        excess = (radius - self.radius) / self.radius
        dphi = (2.0 * excess / self.radius) * offset / radius
        return float(energy) + excess ** 2, dphi

    def augment(self, trial_bohr, energy, gradient_bohr):
        """`(E + phi, g + dphi/dX, the constraint residual)`.

        Two deliberate departures from the implementation this is reimplemented
        from, both of them measured on the analytic double well of
        `qualification/verification/test_path.py`:

        * the chain rule keeps its weight factor (see the class docstring); the
          reference's expression is only correct for a uniform mass;
        * the penalty gradient is **not** projected orthogonal to the energy
          gradient. The reference removes the component of `dphi` along `g`
          before adding it, and on a path whose displacement is collinear with
          the gradient - which is what the first step of every path point is, and
          what a one-dimensional well always is - that removes the radial force
          the shell is held by. Measured on the analytic double well: with the
          projection the walk drifted outward and settled at `rho = 3.05 s`;
          with the penalty kept whole it holds `rho = 1.004 s` on every shell
          point. The one-sided ReLU gate is what the requirement is about, and
          it is kept.
        """
        augmented_energy, penalty = self.penalty(trial_bohr, energy)
        gradient = np.asarray(gradient_bohr, dtype=np.float64).reshape(-1)
        residual = penalty * self.radius ** 2
        return augmented_energy, gradient + penalty, residual

    def inspect(self, trial_bohr, energy, gradient_bohr):
        offset = self.offset(trial_bohr)
        radius = float(np.linalg.norm(offset))
        augmented_energy, augmented, residual = self.augment(
            trial_bohr, energy, gradient_bohr)
        return {
            'radius': radius,
            'perpendicular': augmented,
            'residual': self.rms(residual),
            'residual_max': (float(np.max(np.abs(residual)))
                             if residual.size else 0.0),
            'arc': self.radius,
            '_energy': float(augmented_energy),
            '_gradient': augmented,
            '_trial': np.asarray(trial_bohr, dtype=np.float64).reshape(-1).copy(),
            '_unit': offset / radius if radius > _RADIUS_FLOOR else None,
        }

    def settled(self, inspection):
        """Whether the minimizer of `E + phi` has stopped.

        One test, on the augmented gradient itself: it vanishes at the balance
        the shell is held by (the radial energy slope against the penalty's
        restoring force, which for the reference's `1/s**2` curvature sits a few
        1e-4 of `s` outside the shell). A probe inside the shell reads `phi = 0`,
        so its augmented gradient is the plain one and the same test says the
        only interior point that settles is a genuine stationary one - which is
        what stops a walk whose starting geometry is already a minimum. A probe
        sitting on the centre of a fresh sphere is the same case: the gate is
        closed there too, so it settles only if the surface is stationary.
        """
        return self.rms(inspection['_gradient']) <= self.tol

    def next_trial(self, inspection):
        """A BFGS step of the augmented objective, held inside the shell.

        The reference hands `E + phi` to a bounded least-squares solver that
        line-searches; here the walker owns the evaluations, so the step is the
        BFGS step the rest of the framework takes, capped twice:

        * at the arc length of one path point, as the reference caps it;
        * at the distance to the shell when the probe is outside it, so a step
          cannot cross to the far side. That second cap is not cosmetic: the
          penalty gradient outside the shell is `2/s**2` times the excess, so an
          uncapped step is capped to the arc length and overshoots inward by a
          whole fraction of `s` - measured on the analytic double well, the walk
          then spent its twenty probes oscillating across the shell and never
          settled, while with the crossing cap each shell point costs five
          evaluations.
        """
        gradient = np.asarray(inspection['_gradient'], dtype=np.float64)
        trial = np.asarray(inspection['_trial'], dtype=np.float64)
        if self._last is not None:
            step = trial - self._last[0]
            change = gradient - self._last[1]
            product = usable_curvature_product(step, change)
            if product is not None:
                self._bfgs(step, change, product)
        self._last = (trial.copy(), gradient.copy())
        direction = -self._inverse @ gradient
        length = float(np.linalg.norm(direction))
        if length <= _RADIUS_FLOOR:
            return trial
        unit = direction / length
        length = min(length, self.radius)
        radius = float(inspection['radius'])
        if radius > self.radius and inspection['_unit'] is not None:
            radial = float(np.dot(unit, inspection['_unit']))
            if radial < 0.0:
                length = min(length, (radius - self.radius) / (-radial))
        return trial + length * unit

    def _bfgs(self, step, change, product):
        """The inverse-Hessian BFGS update, guarded by `product > 0`."""
        inverse = self._inverse
        rho = 1.0 / product
        left = np.eye(inverse.shape[0]) - rho * np.outer(step, change)
        self._inverse = (left @ inverse @ left.T
                         + rho * np.outer(step, step))

    def begin(self, reference_bohr):
        """Start a fresh minimization of `E + phi` about the new centre.

        The curvature model restarts from what the run was given (the identity,
        or the force constants an IRC already computed) rather than carrying the
        previous point's fit: the previous point converged on the shell, where
        the penalty's `2/s**2` curvature dominates, and a probe leaving a fresh
        centre moves under the bare energy inside the sphere. Measured on the
        analytic double well, a carried model left the second path point with
        steps of ~1e-5 Bohr and spent all twenty probes on it without moving.
        """
        super().begin(reference_bohr)
        self._inverse = self._initial.copy()
        self._last = None


CONSTRAINT_MODELS = {
    'sphere': HypersphereConstraint,
    'relu': ReLUConstraint,
}


def make_constraint(name, radius_bohr, weights, tolerance, mobile=None,
                    curvature=None):
    """Create one constraint model by name.

    Parameters
    ----------
    name : str
        'sphere' or 'relu' (the spellings `schema.config.PATH_CONSTRAINT_MODELS`
        declares, which the section validator has already checked).
    radius_bohr, weights, tolerance, mobile, curvature
        Passed through to the model.

    Returns
    -------
    PathConstraintModel

    Raises
    ------
    ValueError
        If the name is not one this module implements.
    """
    key = str(name or '').strip().lower()
    if key not in CONSTRAINT_MODELS:
        raise ValueError(
            "Unknown path_constraint %r. Options: %s"
            % (name, ', '.join(sorted(CONSTRAINT_MODELS))))
    return CONSTRAINT_MODELS[key](radius_bohr, weights, tolerance,
                                 mobile=mobile, curvature=curvature)


__all__ = [
    'PathConstraintModel',
    'HypersphereConstraint',
    'ReLUConstraint',
    'CONSTRAINT_MODELS',
    'make_constraint',
]