"""
======================================================================
This module provides the IRCOptimizer for intrinsic reaction coordinate
(IRC) calculations.

An intrinsic reaction coordinate is the minimum energy path that leaves a
first-order saddle along the direction the surface actually falls away in,
which is the imaginary mode of its force constants: the walk is the MEP
walk of `optimizer/mep.py` (the same constrained descent on the sphere of
`optimize.path_radius`, under the same two constraint models) seeded one
radius away from the saddle along that mode instead of on it. Without the
displacement there is no direction to leave on - a saddle's gradient is
zero - which is the whole difference between the two methods.

Where the mode comes from is `optimize.irc_mode`:

    hessian  the framework's own Hessian computation on this geometry, the
             route `&hessian hess_driver`/`hess_mode` names. The matrix also
             seeds the ReLU model's inner minimizer, so a `relu` run pays
             for the curvature once and uses it twice.
    molden   a vibration file that already holds the modes, written by
             `workflow/frequency.py::write_freq_molden` or by a backend.

Which side of the saddle is walked is `optimize.irc_direction`, and `both`
runs one walker per side from the one saddle. The two are labels for the
two ends and not physical directions: the sign of a mode read from a file
is arbitrary, and what makes them meaningful is that a symmetric surface
gives the two branches mirror energies and mirror geometries
(`qualification/verification/test_path.py`).

Oct 20 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from .mep import MEPOptimizer
from ..frequency import vibration
from ..sampling import read_molden

# How negative a frequency has to be to count as the imaginary mode. A saddle
# that has been optimized to its own criteria carries the mode at a few cm^-1,
# and the rigid-motion projection leaves numerical noise of the same order
# around zero; the floor is the reference implementation's `negative_tol`.
_IMAGINARY_FLOOR = 1.0e-8

# Which side of the saddle the two branch labels are, in the seed's sign.
_DIRECTION_SIGNS = {'forward': 1, 'backward': -1}


def imaginary_mode(frequencies, modes, tolerance=_IMAGINARY_FLOOR):
    """The most negative mode of one frequency/mode set.

    Parameters
    ----------
    frequencies : array-like
        Frequencies in cm^-1, imaginary ones negative. A result block's leading
        state axis is accepted and dropped.
    modes : array-like
        `(nmode, natom, 3)` displacement vectors, or `(1, nmode, natom, 3)` as
        the state carries them.
    tolerance : float
        How negative a frequency must be to count. Frequencies above it are
        numerical noise around the rigid-motion projection, not a saddle.

    Returns
    -------
    tuple
        `(index, frequency cm^-1, (natom, 3) vector)` of the most negative mode.

    Raises
    ------
    ValueError
        When there is no mode below the tolerance: the starting geometry is not
        a first-order saddle, and an IRC has no direction to leave it on.
    """
    values = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    array = np.asarray(modes, dtype=np.float64)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(
            'the modes of an IRC are (nmode, natom, 3) vectors and this set has '
            'shape %s' % (np.asarray(modes).shape,))
    if values.size == 0 or array.shape[0] != values.size:
        raise ValueError(
            'the modes of an IRC need one vector per frequency: %d against %d'
            % (array.shape[0], values.size))
    index = int(np.argmin(values))
    if values[index] >= -abs(float(tolerance)):
        raise ValueError(
            'IRC needs an imaginary mode and this starting point has none '
            '(lowest frequency %.3f cm^-1): it is not a first-order saddle, and '
            'a walk from it has no direction to leave on. Optimize the saddle '
            'first (optimize.method ts), or read the mode of one that is'
            % values[index])
    return index, float(values[index]), array[index]


def displace_along_mode(mode, weights, radius_bohr, sign=1):
    """The Cartesian displacement whose mass-weighted radius is `radius_bohr`.

    `mode` is an unmass-weighted Cartesian vector, and only its **direction** is
    used: the normalization divides by the weighted length, so
    `w (d v) = radius * w v / |w v|` is a unit vector times the radius whatever
    scale the vector arrived on. That matters because the readers of the
    frequency records disagree about normalization (`sampling.py`: a Molcas
    molden carries unnormalized modes, this framework's own writer carries
    `sum(m x**2) = 1`), and a seed that depended on which reader produced the
    file would put the two routes at different distances.
    """
    vector = np.asarray(mode, dtype=np.float64).reshape(-1)
    weighted = np.asarray(weights, dtype=np.float64).reshape(-1) * vector
    norm = float(np.linalg.norm(weighted))
    if norm <= 0.0:
        raise ValueError(
            'the imaginary mode of this transition state is the zero vector, so '
            'there is no direction to displace along')
    return (1 if sign >= 0 else -1) * (float(radius_bohr) / norm) * vector


class IRCOptimizer(MEPOptimizer):
    """Intrinsic reaction coordinate from a first-order saddle.

    Parameters
    ----------
    direction : str
        Which side of the saddle this walker takes: 'forward' or 'backward'.
        `optimize.irc_direction` names the request, and the Path workflow turns
        `both` into one walker per side.
    irc_mode : str
        'hessian' or 'molden' (`optimize.irc_mode`).
    irc_mode_file : str
        The vibration file, when `irc_mode` is 'molden'.
    path_radius, inner_maxit, inner_tol, constraint_model
        As for `MEPOptimizer`.

    Attributes
    ----------
    frequency : float or None
        The signed imaginary frequency this run started from, in cm^-1, once
        `begin` has resolved it.
    mode : numpy.ndarray or None
        The `(natom, 3)` imaginary mode, once `begin` has resolved it.
    imaginary_count : int
        How many modes of this starting point are imaginary. One is a
        first-order saddle; more is a higher-order one, which is reported
        rather than refused because a run that asked for the path should see
        what it started on.
    """

    # The curvature has to exist before the first step, which is what routes
    # this run through the framework's Hessian computation
    # (`workflow/path.py::_supply_initial_hessian`).
    is_implemented = True
    needs_initial_hessian = True
    path_kind = 'walker'

    def __init__(self, direction='forward', irc_mode='hessian',
                 irc_mode_file='', **kwargs):
        super().__init__(**kwargs)
        self.direction = str(direction or 'forward').strip().lower()
        if self.direction not in _DIRECTION_SIGNS:
            raise ValueError(
                "Unknown IRC branch %r. Options: %s"
                % (direction, ', '.join(sorted(_DIRECTION_SIGNS))))
        self.irc_mode = str(irc_mode or 'hessian').strip().lower()
        self.irc_mode_file = str(irc_mode_file or '')
        self.frequency = None
        self.mode = None
        self.imaginary_count = 0
        self._matrix = None

    # ---- The curvature the workflow supplies ----

    def set_curvature(self, matrix):
        """Take the transition state's force constants.

        `(3N, 3N)` from a fixed-cell run, `(3N + 6, 3N + 6)` from a
        variable-cell one - the shape the mode is read out of is the atomic
        block, and the whole matrix is handed to the ReLU model as its starting
        curvature in the joint shape it needs. That model uses it only when it
        is a positive-definite curvature model
        (`pathconstraint.py::ReLUConstraint._inverse_of`): a saddle's matrix is
        indefinite, and a minimization started from its inverse (or from the
        inverse of its magnitudes) probes along the wrong side of the surface,
        which is measured on `path_irc_relu` and recorded there.
        """
        self._curvature = np.asarray(matrix, dtype=np.float64)
        self._matrix = self._curvature
        return self._curvature

    # ---- Seeding ----

    def begin(self, state):
        """Resolve the imaginary mode, then seed one radius along it."""
        if self.mode is None:
            self._resolve_mode(state)
        super().begin(state)

    def _resolve_mode(self, state):
        """Read the modes by the route `irc_mode` names, and pick the mode."""
        if self.irc_mode == 'molden':
            self._mode_from_file(state)
        elif self.irc_mode == 'hessian':
            self._mode_from_curvature(state)
        else:
            raise ValueError(
                "Unknown irc_mode %r. Options: hessian, molden" % self.irc_mode)
        index, frequency, vector = imaginary_mode(
            self._frequencies, self._modes)
        values = np.asarray(self._frequencies, dtype=np.float64).reshape(-1)
        self.imaginary_count = int(np.count_nonzero(values < -_IMAGINARY_FLOOR))
        self.frequency = frequency
        self.mode = np.asarray(vector, dtype=np.float64).reshape(-1, 3)
        return index

    def _mode_from_file(self, state):
        """The modes of a vibration file written by this framework or a backend."""
        data = read_molden(self.irc_mode_file)
        atoms = [str(name) for name in np.asarray(data['atoms']).reshape(-1)]
        wanted = [str(name) for name in np.asarray(state.atoms).reshape(-1)]
        if atoms != wanted:
            raise ValueError(
                'the vibration file %r holds the modes of %s and this run is '
                'walking %s: a mode read against another atom order is another '
                'displacement'
                % (self.irc_mode_file, ' '.join(atoms), ' '.join(wanted)))
        self._frequencies = np.asarray(data['freqs'], dtype=np.float64)
        self._modes = np.asarray(data['vib'], dtype=np.float64)

    def _mode_from_curvature(self, state):
        """The modes of the transition state's own force constants."""
        if self._matrix is None:
            raise ValueError(
                "irc_mode 'hessian' needs the force constants of the transition "
                'state and none were computed for this run')
        coordinates = (np.asarray(state.coordinates, dtype=np.float64)
                       .reshape(-1) * self.coord_to_bohr)
        masses = np.asarray(getattr(state.pmd.system, 'mass', None) or [],
                            dtype=np.float64).reshape(-1)
        natom = coordinates.reshape(-1, 3).shape[0]
        if masses.size != natom:
            raise ValueError(
                'the imaginary mode of an IRC needs one mass per atom: this '
                'system carries %d for %d atoms' % (masses.size, natom))
        atomic = self._matrix[:3 * natom, :3 * natom]
        result = vibration(coordinates, masses, atomic)
        self._frequencies = result['frequencies']
        self._modes = result['modes']

    def seed(self, state, joint, lattice):
        """One path radius away from the saddle, along the imaginary mode.

        The displacement is atomic: an imaginary mode of a variable-cell run has
        no lattice components, and the box relaxes on its own while the path is
        walked. That is also why the weights are the atomic block and not the
        joint vector's: `WalkerPathOptimizer.begin` sets `_weights` to the joint
        (3N+6) weights as soon as the box is a variable, and this mode is 3N, so
        the joint array against it is a shape mismatch - the seed used to hand
        that pair to `displace_along_mode` and raise a broadcast error on every
        variable-cell IRC, whatever its own docstring promised.
        """
        if self.mode is None:
            raise ValueError('the imaginary mode of this IRC was never resolved')
        weights = self._atomic_weights
        if weights is None:
            raise ValueError(
                'the seed of this IRC has no atomic weights: the walker binds '
                'them in begin() before it seeds, so a seed asked for earlier '
                'has nothing to normalize the mode against')
        if np.asarray(weights).size != self.mode.size:
            raise ValueError(
                'the imaginary mode of this IRC has %d component(s) and the '
                'atomic block of this run has %d: the mode belongs to another '
                'system, or it was read as a joint vector'
                % (self.mode.size, np.asarray(weights).size))
        sign = _DIRECTION_SIGNS[self.direction]
        atomic = displace_along_mode(self.mode, weights,
                                     self.path_radius, sign=sign)
        seeded = np.asarray(joint, dtype=np.float64).reshape(-1).copy()
        seeded[:atomic.size] += atomic * self.coord_to_bohr
        return seeded

    @property
    def seed_role(self):
        """What the first point of this walk is called in the profile."""
        return 'saddle'

    def results(self):
        block = super().results()
        block['path_direction'] = self.direction
        block['path_imaginary_frequency'] = self.frequency
        block['path_imaginary_count'] = self.imaginary_count
        if self.mode is not None:
            block['path_imaginary_mode'] = self.mode.reshape(-1).tolist()
        return block


__all__ = ['IRCOptimizer', 'imaginary_mode', 'displace_along_mode']