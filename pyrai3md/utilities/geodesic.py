"""
======================================================================
This module ports the geometric interpolation of the `geodesic-interpolate`
package into the framework's utility layer: a path between two geometries
is found as a geodesic of the Riemmanian manifold whose metric comes from
a set of redundant internal distances, so the images it produces are smooth
in the coordinates that matter and stay in feasible Cartesian space.

    https://github.com/ZhuGroup/geodesic-interpolate
    MIT License, Copyright (c) 2024 Xiaolei Zhu

The license above is the one that governs this file; the algorithm is
Xiaolei Zhu et al.'s (the configuration space is viewed as a Riemannian
manifold with a Morse-scaled pairwise-distance metric, distances are the
`sum(m x**2) = 1`-free internal lengths, and the path is the curve of
least total internal length through Cartesian coordinates).

MIT License

Copyright (c) 2024 Xiaolei Zhu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

What was adapted, and why (six points, each of them a decision):

1. `atoms` arrives from the run's own system, so a pair's rest length comes
   from `state.atoms` rather than from a file. Symbols are turned into radii by
   `schema.reference.COVALENT_RADII` (Cordero single-bond radii, H-Cm; every
   H-Ar value is the ported table's own, so those bands are unchanged), with
   the reference's 1.5 Angstrom fallback for an element the table does not name;
   with no symbols the port keeps the reference's 2.0 Angstrom fallback.
2. Units stay Angstrom throughout, the unit the reference uses and the one
   the band is stored in. Nothing is converted at the boundary.
3. Periodic systems: a pair's vector is reduced to its nearest image
   (`utilities/lattice.py::min_image_displacement`), and the pair list's
   screen is a direct wrapped-distance scan rather than a K-D tree, because
   a minimum image is not a Euclidean metric a tree can index. A variable-cell
   band interpolates the cell linearly between its endpoints and reads each
   image's distances at that image's own cell; the metric smooths the nuclear
   path at those cells, and the box itself is a ramp - a stated trade, not an
   oversight.
4. The pair-cutoff radius, the metric's stiffness and the minimum coordination
   are the caller's (`optimize.neb_geodesic_*`), and the stiffness is now
   actually used: the reference computes `morse_scaler(re=self.re, alpha=1.7)`
   and ignores its own `scaler` argument, so its `--scaling` option does
   nothing.
5. Reproducibility: the reference nudges its bisection guesses from the global
   NumPy state, so two runs of the same input can differ. Every draw here goes
   through a `numpy.random.Generator` the caller seeds from the run's
   `control.gl_seed`.
6. The solver is this framework's own. The reference hands its residual and
   analytic Jacobian to `scipy.optimize.least_squares` with the `soft_l1`
   robust loss and assembles the Jacobian through `scipy.sparse`. This port
   keeps the residual, the analytic Jacobian and the robust loss (as iteratively
   reweighted least squares) and solves the normal equations densely with a
   fixed Levenberg damping: the band sizes this is used at are tens of
   unknowns, where sparsity buys nothing, and scipy is not a dependency of this
   package (see `setup.py`; the same reason the O(N^2) screens above are
   acceptable). The minimizer is reported as returned; a caller that needs the
   reference's exact solver will not get a bit-identical path.

Nothing here evaluates a potential, and nothing here writes a file.

Oct 20 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from ..schema.reference import COVALENT_RADII
from .lattice import min_image_displacement

# How many atoms a geodesic band is a reasonable tool for. The smoothing step
# solves a dense normal-equation system whose Jacobian is
# `((2l + m) nrij) x (m 3N)`; at `m` interior images of `N` atoms that is a few
# megabytes at 30 atoms and a few gigabytes at 200, which is why the reference
# switches to its one-image-at-a-time sweep and why this port refuses rather
# than tries. The applied threshold is the atom count; the one above which the
# sweep is used is the reference's own rule.
_SWEEP_ATOMS = 35
_MAX_ATOMS = 200

# The soft_l1 robust loss's scale, the reference's `f_scale` (it does not pass
# one, so `least_squares` uses 1.0). A residual of this size carries full weight
# and a much larger one is downweighted, which is what keeps one badly placed
# bisection from dragging the whole path.
_ROBUST_SCALE = 1.0

# Levenberg damping on the normal equations. The residual is smooth and the
# starting path is already close to the geodesic it smooths, so a small fixed
# damping is enough to keep a nearly singular curvature matrix from producing a
# wild step.
_DAMPING = 1.0e-3


# ---- Rigid-body placement ----

def align_geom(refgeom, geom):
    """The rotation and translation that best overlap one geometry on another.

    The Kabsch algorithm on the two centred geometries. This is how the
    reference keeps a chain of independently positioned images a coherent
    motion: each image's *internal* geometry is what the geodesic determines,
    and where in space it sits is a gauge choice that has to be made the same
    way for every image or the chain reads as a sequence of unrelated molecules.
    """
    reference = np.asarray(refgeom, dtype=np.float64).reshape(-1, 3)
    geometry = np.asarray(geom, dtype=np.float64).reshape(-1, 3)
    center = np.mean(reference, axis=0)
    centered_reference = reference - center
    centered_geometry = geometry - np.mean(geometry, axis=0)
    covariance = centered_geometry.T @ centered_reference
    left, singular, right = np.linalg.svd(covariance)
    if np.linalg.det(left) * np.linalg.det(right) < 0:
        singular[-1] = -singular[-1]
        left[:, -1] = -left[:, -1]
    rotation = left @ right
    moved = centered_geometry @ rotation + center
    rmsd = float(np.sqrt(np.mean((moved - reference) ** 2)))
    return rmsd, moved


def align_path(path, endpoints=True):
    """Chain-align a path onto its own first image, keeping the endpoints.

    Each image is rotated and translated onto the one before it, so the path
    becomes a motion rather than a set of independent placements. Two deviations
    from the reference, both because this port builds a *band*:

    * the reference moves the first image's centroid to the origin before it
      starts, which re-anchors the path in space. Here the first image stays
      exactly where the caller put it, because it is the band's reactant.
    * the reference aligns every image onto its predecessor, the last one
      included, which moves the product. Here the last image is left where it
      is, because it is the band's product.

    Parameters
    ----------
    path : array-like
        `(nimage, natom, 3)` geometries; modified in place and returned.
    endpoints : bool
        Whether the two endpoints are to stay put (the band's answer). False
        aligns every image onto its predecessor, which is the reference's own
        behaviour for a path whose ends are free.

    Returns
    -------
    tuple
        `(largest rmsd moved, the path)`.
    """
    frames = np.asarray(path, dtype=np.float64)
    last = frames.shape[0] - 1 if endpoints else frames.shape[0]
    largest = 0.0
    for index in range(1, last):
        rmsd, moved = align_geom(frames[index - 1], frames[index])
        frames[index] = moved
        largest = max(largest, rmsd)
    return largest, frames


# ---- The metric ----

def consistent_pair(first, second, lattice=None):
    """The second structure brought into the first one's periodic image.

    A structure file lists coordinates in the main image, so a molecule written
    across a cell face has some of its atoms on the far side. Two endpoints of
    one reaction can then differ by a whole lattice translation even when they
    describe the same physical pair, and interpolating the raw arrays drags
    those atoms through the cell wall: the geodesic's metric would still read
    the right internal distances (`compute_rij` reduces every pair to its
    nearest image), but its initial guess and a linear path would not. An
    interpolation needs one frame, so the second structure is moved atom by atom
    to the image nearest the first.

    The reduction is the metric's own convention (`min_image_displacement`), and
    a non-periodic pair is returned untouched. The physical structures do not
    change: a lattice translation is the same crystal.

    Parameters
    ----------
    first, second : array-like
        `(natom, 3)` endpoints in Angstrom.
    lattice : array-like or None
        `(3, 3)` cell in Angstrom the pair is compared in.

    Returns
    -------
    tuple of numpy.ndarray
        `(first, second)` as copies, the second in the first's frame.
    """
    one = np.asarray(first, dtype=np.float64).reshape(-1, 3).copy()
    other = np.asarray(second, dtype=np.float64).reshape(-1, 3).copy()
    if lattice is None or one.shape != other.shape:
        return one, other
    cell = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    return one, one + min_image_displacement(other - one, cell)


def compute_rij(geom, rij_list, lattice=None):
    """The distances of a pair list, and their Cartesian gradients.

    Parameters
    ----------
    geom : array-like
        `(natom, 3)` Cartesian geometry, Angstrom.
    rij_list : sequence of (int, int)
        The pairs to measure.
    lattice : array-like or None
        `(3, 3)` cell in Angstrom. Given, each pair's vector is reduced to its
        nearest image, which is the distance that is periodic in the cell; the
        gradient is unchanged, because a minimum image differs from a plain
        difference by a translation of one atom.

    Returns
    -------
    tuple
        `(rij, bmat)`: distances `(nrij,)` and their gradients
        `(nrij, 3 natom)`.
    """
    geometry = np.asarray(geom, dtype=np.float64).reshape(-1, 3)
    natom = geometry.shape[0]
    count = len(rij_list)
    rij = np.zeros(count, dtype=np.float64)
    bmat = np.zeros((count, 3 * natom), dtype=np.float64)
    for index, (i, j) in enumerate(rij_list):
        vector = geometry[int(i)] - geometry[int(j)]
        if lattice is not None:
            vector = min_image_displacement(vector, lattice)
        distance = float(np.linalg.norm(vector))
        if distance <= 0.0:
            # Coincident atoms have no direction, and the derivative of a
            # distance is undefined there. Zero is the same statement the
            # reference's division would produce as a nan, without the nan.
            continue
        rij[index] = distance
        gradient = vector / distance
        bmat[index, 3 * int(i):3 * int(i) + 3] = gradient
        bmat[index, 3 * int(j):3 * int(j) + 3] = -gradient
    return rij, bmat


def compute_wij(geom, rij_list, func, lattice=None):
    """The scaled distances of a pair list, and their Cartesian gradients.

    `func` is one of the scalers below: it takes the raw distances and returns
    the scaled values and the derivative of the scale with respect to the raw
    distance, so the chain rule is a row-wise multiply.
    """
    rij, bmat = compute_rij(geom, rij_list, lattice)
    wij, dwdr = func(rij)
    return np.asarray(wij, dtype=np.float64), bmat * np.asarray(
        dwdr, dtype=np.float64)[:, None]


def morse_scaler(re=1.5, alpha=1.7, beta=0.01):
    """The Morse-scaled metric of the internal coordinates.

    `w(r) = exp(alpha (1 - r/re)) + beta/re * ...`; the return value is a
    function of the distances, not of one distance, because `re` is a per-pair
    rest length. The metric is what gives a long pair a slowly growing length
    and a short one a steeply growing length, which is the reference's reading
    of an internal coordinate's stiffness.
    """
    def scaler(x):
        distance = np.asarray(x, dtype=np.float64)
        ratio = distance / re
        value = np.exp(alpha * (1.0 - ratio))
        second = beta / ratio
        derivative = -alpha / re * value - second / distance
        return value + second, derivative
    return scaler


def elu_scaler(re=2.0, alpha=2.0, beta=0.01):
    """The ELU-scaled metric, the reference's alternative to `morse_scaler`."""
    def scaler(x):
        distance = np.asarray(x, dtype=np.float64)
        value = (1.0 - distance / re) * alpha + 1.0
        derivative = np.full(distance.shape, -alpha / re)
        large = distance > re
        tail = np.exp(alpha * (1.0 - distance[large] / re))
        value[large] = tail
        derivative[large] = -alpha / re * tail
        second = beta * re / distance
        return value + second, derivative - second / distance
    return scaler


# ---- The pair list ----

def _pairs_within(geom, threshold, lattice=None):
    """Every pair of atoms closer than `threshold`, as sorted index tuples."""
    geometry = np.asarray(geom, dtype=np.float64).reshape(-1, 3)
    natom = geometry.shape[0]
    pairs = set()
    for i in range(natom):
        offsets = geometry - geometry[i]
        if lattice is not None:
            offsets = min_image_displacement(offsets, lattice)
        distances = np.linalg.norm(offsets, axis=1)
        for j in np.nonzero(distances < threshold)[0]:
            if int(j) == i:
                continue
            pairs.add((i, int(j)) if i < int(j) else (int(j), i))
    return pairs


def _nearest(geom, index, count, lattice=None):
    """The indices of the `count` atoms nearest to one atom, itself excluded."""
    geometry = np.asarray(geom, dtype=np.float64).reshape(-1, 3)
    offsets = geometry - geometry[int(index)]
    if lattice is not None:
        offsets = min_image_displacement(offsets, lattice)
    order = np.argsort(np.linalg.norm(offsets, axis=1))
    return [int(other) for other in order if int(other) != int(index)][:count]


def get_bond_list(geom, atoms=None, threshold=3.0, min_neighbors=4,
                  snapshots=30, bond_threshold=1.8, enforce=(),
                  lattice=None, rng=None):
    """The pairs of atoms the metric is built on, and their rest lengths.

    Every pair closer than `threshold` in any of the sampled images is kept,
    together with the pairs of atoms within three or fewer bonds of each other
    (so a bond that only appears along the path is still measured), and then
    every atom is topped up to `min_neighbors` by its nearest neighbours. The
    reference's screen exists because a pair that only comes into contact in the
    middle of a path would otherwise be measured for the first time there.

    Parameters
    ----------
    geom : array-like
        `(nimage, natom, 3)` or `(natom, 3)` geometries, Angstrom.
    atoms : sequence of str or None
        Element symbols, for the per-pair rest length. None uses 2.0 Angstrom.
    threshold, bond_threshold, min_neighbors, snapshots, enforce, lattice
        The reference's parameters; `enforce` is a set of pairs that must be
        included (the bisection point adds the pairs it discovered).
    rng : numpy.random.Generator or None
        Which images are sampled when the path is longer than `snapshots`. None
        draws a fresh unseeded generator, which the reference does too.

    Returns
    -------
    tuple
        `(pairs, re)`: a sorted list of index tuples and the rest length of each.
    """
    geometry = np.asarray(geom, dtype=np.float64)
    if geometry.ndim < 3:
        geometry = geometry.reshape(1, -1, 3)
    images_count, natom = geometry.shape[0], geometry.shape[1]
    if natom < 2:
        return [], np.zeros(0)
    generator = rng if isinstance(rng, np.random.Generator) \
        else np.random.default_rng(rng)
    min_neighbors = min(int(min_neighbors), natom - 1)
    sampled = min(int(snapshots), images_count)
    images = [0, images_count - 1]
    if sampled > 2:
        images.extend(int(index) for index in generator.choice(
            range(1, sampled - 1), sampled - 2, replace=False))

    pairs = set(enforce)
    for image in images:
        pairs.update(_pairs_within(geometry[image], threshold, lattice))
        bonded = _pairs_within(geometry[image], bond_threshold, lattice)
        neighbors = {index: {index} for index in range(natom)}
        for first, second in bonded:
            neighbors[first].add(second)
            neighbors[second].add(first)
        for first, second in bonded:
            for near_first in neighbors[first]:
                for near_second in neighbors[second]:
                    if near_first != near_second:
                        pairs.add(tuple(sorted((near_first, near_second))))
    pair_list = sorted(pairs)

    count = np.zeros(natom, dtype=int)
    for first, second in pair_list:
        count[first] += 1
        count[second] += 1
    last = geometry[-1]
    for index, total in enumerate(count):
        if total >= min_neighbors:
            continue
        for other in _nearest(last, index, min_neighbors, lattice):
            pair = (index, other) if index < other else (other, index)
            if pair in pairs:
                continue
            pairs.add(pair)
            pair_list.append(pair)
            count[other] += 1
            count[index] += 1
    pair_list = sorted(set(pair_list))

    if atoms is None:
        rest = np.full(len(pair_list), 2.0)
    else:
        radius = np.array([COVALENT_RADII.get(str(atom).capitalize(), 1.5)
                           for atom in atoms])
        rest = np.array([radius[first] + radius[second]
                         for first, second in pair_list])
    return pair_list, rest


# ---- The solver ----

def _robust_loss(residuals, scale=_ROBUST_SCALE):
    """The `soft_l1` cost of a residual vector, up to a constant factor.

    `sum(2 (sqrt(1 + (r/scale)**2) - 1))`, which is the loss the IRLS weights in
    the solver are the stationarity condition of. Only comparisons between two
    residual vectors are made with it, so the constant the reference's cost
    carries is not reproduced.
    """
    ratio = np.asarray(residuals, dtype=np.float64) / scale
    return float(np.sum(2.0 * (np.sqrt(1.0 + ratio ** 2) - 1.0)))


def _least_squares(residual, jacobian, x0, ftol=1.0e-3, gtol=1.0e-3,
                   max_evaluations=50, scale=_ROBUST_SCALE,
                   damping=_DAMPING):
    """Minimize `||residual(x)||` with a Levenberg-damped Gauss-Newton.

    The robust `soft_l1` loss of the reference is applied as iteratively
    reweighted least squares: a residual of size `scale` carries full weight and
    a larger one is downweighted by `1 / sqrt(1 + (r / scale)**2)`, which is the
    weight that loss's minimizer solves with.

    The damping is what makes this a least-squares solver rather than one
    Newton step: a step whose cost rises is refused and the damping grown, a
    step that lowers it is taken and the damping shrunk. Measured on the
    HCN -> HNC pair, a fixed 1e-3 damping (whose value is negligible against
    this curvature matrix) let the very first step of the smoothing throw the
    images thousands of Angstrom away, while the adapted one descends
    monotonically.

    Parameters
    ----------
    residual, jacobian : callable
        `residual(x) -> (n,)` and `jacobian(x) -> (n, m)`. The Jacobian is
        analytic, so a refused step costs one evaluation and an accepted one
        costs one: `max_evaluations` counts the reference's `max_nfev`.
    ftol : float
        Stop when the step's largest component is this small against `x`'s.
        The reference's `ftol` is on the cost's change; a step criterion is the
        same statement about the same iterate and needs no second evaluation.
    gtol : float
        Stop when the weighted gradient's largest component is this small.

    Returns
    -------
    numpy.ndarray
        The minimizer. The caller evaluates the residual there if it wants it.
    """
    x = np.asarray(x0, dtype=np.float64).reshape(-1).copy()
    residuals = np.asarray(residual(x), dtype=np.float64).reshape(-1)
    matrix = np.asarray(jacobian(x), dtype=np.float64)
    cost = _robust_loss(residuals, scale)
    damping = max(float(damping), 1.0e-12)
    evaluations = 1
    steps = max(int(max_evaluations), 1)
    while evaluations < steps:
        weight = 1.0 / np.sqrt(1.0 + (residuals / scale) ** 2)
        weighted_j = weight[:, None] * matrix
        normal = weighted_j.T @ weighted_j
        gradient = weighted_j.T @ (weight * residuals)
        if gradient.size == 0:
            break
        if float(np.max(np.abs(gradient))) <= gtol:
            break
        for _ in range(8):
            if evaluations >= steps:
                break
            try:
                step = np.linalg.solve(
                    normal + damping * np.eye(normal.shape[0]), -gradient)
            except np.linalg.LinAlgError:
                step = -gradient / damping
            if not np.all(np.isfinite(step)):
                step = -gradient * 0.1
            trial = x + step
            trial_residuals = np.asarray(residual(trial),
                                         dtype=np.float64).reshape(-1)
            evaluations += 1
            trial_cost = _robust_loss(trial_residuals, scale)
            if trial_cost < cost:
                x, residuals, cost = trial, trial_residuals, trial_cost
                matrix = np.asarray(jacobian(trial), dtype=np.float64)
                damping = max(damping * 0.3, 1.0e-12)
                reference = max(1.0, float(np.max(np.abs(x))))
                if float(np.max(np.abs(step))) <= ftol * reference:
                    evaluations = steps
                break
            damping = min(damping * 10.0, 1.0e12)
    return x


# ---- The path ----

class Geodesic:
    """The geodesic smoother of one path.

    Parameters
    ----------
    atoms : sequence of str
        Element symbols, for the pair rest lengths.
    path : array-like
        `(nimage, natom, 3)` geometries in Angstrom, endpoints included. The
        images from `start` to `end` of `smooth`/`sweep` are the unknowns; the
        endpoints are never moved, which is what makes this usable as a band's
        initial guess rather than as a curve the caller has to re-anchor.
    scaler : float
        The Morse metric's `alpha` (`optimize.neb_geodesic_scaler`).
    threshold : float
        The pair-cutoff radius (`optimize.neb_geodesic_threshold`).
    min_neighbors : int
        The minimum coordination (`optimize.neb_geodesic_min_neighbors`).
    friction : float
        How strongly a step is pulled back towards the path it started from,
        which is what keeps a bisection guess from wandering off.
    lattices : array-like or None
        `(nimage, 3, 3)` cells in Angstrom, for a periodic or variable-cell
        band; None measures plain distances.
    rng : numpy.random.Generator or None
        Seed of the pair-list sampler.

    Attributes
    ----------
    path : numpy.ndarray
        The current path, `(nimage, natom, 3)`.
    length : float
        The internal length of the segment last measured.
    """

    def __init__(self, atoms, path, scaler=1.7, threshold=3.0, min_neighbors=4,
                 friction=1.0e-2, lattices=None, rng=None):
        band = np.asarray(path, dtype=np.float64)
        if band.ndim != 3:
            raise ValueError(
                'the path to be interpolated must be (nimage, natom, 3) and this '
                'one has shape %s' % (band.shape,))
        self.path = band.copy()
        self.nimages, self.natoms, _ = self.path.shape
        if self.nimages < 3:
            raise ValueError(
                'a geodesic needs at least one interior image to move, and this '
                'path has %d images' % self.nimages)
        self.lattices = (None if lattices is None
                         else np.asarray(lattices, dtype=np.float64).copy())
        if self.lattices is not None and self.lattices.shape != (self.nimages,
                                                                3, 3):
            raise ValueError(
                'a periodic geodesic carries one cell per image: %d images '
                'against cells of shape %s'
                % (self.nimages, self.lattices.shape))
        self.rij_list, self.re = get_bond_list(
            self.path, atoms, threshold=threshold, min_neighbors=min_neighbors,
            lattice=self._lattice(0), rng=rng)
        self.scaler = morse_scaler(re=self.re, alpha=scaler)
        self.nrij = len(self.rij_list)
        self.friction = float(friction)
        self.neval = 0
        self.w = [None] * self.nimages
        self.dwdR = [None] * self.nimages
        self.X_mid = [None] * (self.nimages - 1)
        self.w_mid = [None] * (self.nimages - 1)
        self.dwdR_mid = [None] * (self.nimages - 1)
        self.disps = None
        self.grad = None
        self.segment = None
        self.length = 0.0
        self.optimality = 0.0
        self._target_x0 = None
        self._target_friction = None

    def _lattice(self, index):
        """The cell one image is measured at, or None."""
        if self.lattices is None:
            return None
        return self.lattices[index]

    def _resolve_segment(self, start, end):
        """Resolve and check the range of images this call optimizes."""
        if end < 0:
            end += self.nimages
        if start < 1 or end > self.nimages - 1 or start >= end:
            raise ValueError(
                'the optimized segment must satisfy 1 <= start < end <= '
                'nimages - 1, and this run asked for %s..%s of %d images'
                % (start, end, self.nimages))
        return int(start), int(end)

    # ---- Internal coordinates ----

    def update_intc(self):
        """Fill in the internal coordinates and derivatives that are unknown."""
        for index, (geometry, values) in enumerate(zip(self.path, self.w)):
            if values is None:
                self.w[index], self.dwdR[index] = compute_wij(
                    geometry, self.rij_list, self.scaler,
                    self._lattice(index))
        for index, (first, second, values) in enumerate(
                zip(self.path, self.path[1:], self.w_mid)):
            if values is None:
                self.X_mid[index] = middle = (first + second) * 0.5
                self.w_mid[index], self.dwdR_mid[index] = compute_wij(
                    middle, self.rij_list, self.scaler, self._lattice(index))

    def update_geometry(self, flat, start, end):
        """Move a segment of the path, and forget what described it."""
        moved = np.asarray(flat, dtype=np.float64).reshape(
            self.path[start:end].shape)
        if np.array_equal(moved, self.path[start:end]):
            return False
        self.path[start:end] = moved
        for index in range(start, end):
            self.w_mid[index] = self.w[index] = None
        self.w_mid[start - 1] = None
        return True

    # ---- The target function ----

    def compute_disps(self, start=1, end=-1, dx=None, friction=1.0e-3):
        """The displacement vector of every segment, and the total length."""
        start, end = self._resolve_segment(start, end)
        self.update_intc()
        left = [middle - first for first, middle in
                zip(self.w[start - 1:end], self.w_mid[start - 1:end])]
        right = [last - middle for last, middle in
                 zip(self.w[start:end + 1], self.w_mid[start - 1:end])]
        self.length = (float(np.sum(np.linalg.norm(left, axis=1)))
                       + float(np.sum(np.linalg.norm(right, axis=1))))
        if dx is None:
            translation = np.zeros(self.path[start:end].size)
        else:
            translation = friction * dx
        self.disps = np.concatenate(left + right + [translation])

    def compute_disp_grad(self, start, end, friction=1.0e-3):
        """The derivative of the displacement vector, `(2l nrij + m 3N, m 3N)`.

        The reference assembles this as a sparse block matrix; the block layout
        is kept exactly, only the blocks are dense. `l` is the number of images
        the segment spans and `m` the number it moves, and the two halves of the
        matrix are of different heights on purpose: the segment displacements
        contribute `nrij` rows each and the friction term contributes one row per
        unknown, which is what makes the friction a regularization of the step
        rather than another measured distance.
        """
        start, end = self._resolve_segment(start, end)
        span = end - start + 1
        moved = end - start
        natom3 = 3 * self.natoms
        rows = 2 * span * self.nrij + moved * natom3
        self.grad = np.zeros((rows, moved * natom3), dtype=np.float64)

        def put(row, column, block):
            self.grad[row * self.nrij:(row + 1) * self.nrij,
                      column * natom3:(column + 1) * natom3] = block

        for column, image in enumerate(range(start, end)):
            first_half = self.dwdR_mid[image - 1] * 0.5
            second_half = self.dwdR_mid[image] * 0.5
            put(column + 1, column, second_half - self.dwdR[image])
            put(column, column, first_half)
            put(span + column + 1, column, -second_half)
            put(span + column, column, self.dwdR[image] - first_half)
            if friction:
                top = 2 * span * self.nrij + column * natom3
                self.grad[top:top + natom3,
                          column * natom3:(column + 1) * natom3] = (
                    friction * np.eye(natom3))

    def compute_target_func(self, x=None, start=1, end=-1, x0=None,
                            friction=1.0e-3):
        """Recompute the displacement vector for one geometry of the unknowns."""
        start, end = self._resolve_segment(start, end)
        current = None if x0 is None else np.asarray(x0).ravel()
        unchanged = x is not None and not self.update_geometry(x, start, end)
        same_origin = ((current is None and self._target_x0 is None)
                       or (current is not None and self._target_x0 is not None
                           and np.array_equal(current, self._target_x0)))
        if (unchanged and self.segment == (start, end)
                and friction == self._target_friction and same_origin):
            return
        self.segment = (start, end)
        self._target_friction = friction
        self._target_x0 = None if current is None else current.copy()
        dx = (np.zeros(self.path[start:end].size) if current is None
              else self.path[start:end].ravel() - current)
        self.compute_disps(start, end, dx=dx, friction=friction)
        self.compute_disp_grad(start, end, friction=friction)
        if self.grad.size:
            self.optimality = float(np.max(np.abs(self.grad.T @ self.disps)))
        else:
            self.optimality = 0.0
        self.neval += 1

    def target_func(self, x, **kwargs):
        """The displacement vector, cached against a repeated geometry."""
        self.compute_target_func(x, **kwargs)
        return self.disps

    def target_deriv(self, x, **kwargs):
        """The Jacobian of the displacement vector."""
        self.compute_target_func(x, **kwargs)
        return self.grad

    # ---- The two minimizers ----

    def smooth(self, tol=1.0e-3, max_iter=50, start=1, end=-1,
               friction=None, xref=None):
        """Move every image of the segment at once onto the geodesic.

        This is the efficient route and the one to use for a small band. It is
        the global minimization of the internal length; the reference uses its
        one-image sweep for larger systems because the Jacobian stops fitting in
        memory, which is why `interpolate` chooses between the two by size.
        """
        start, end = self._resolve_segment(start, end)
        flat = np.asarray(self.path[start:end]).ravel()
        if xref is None:
            xref = flat
        self.disps = self.grad = self.segment = None
        if friction is None:
            friction = self.friction
        settings = dict(start=start, end=end, x0=xref, friction=friction)
        self.compute_target_func(**settings)
        if self.optimality > tol:
            solved = _least_squares(
                lambda value: self.target_func(value, **settings),
                lambda value: self.target_deriv(value, **settings),
                flat, ftol=tol, gtol=tol, max_evaluations=max_iter)
            self.update_geometry(solved, start, end)
        return self.path

    def sweep(self, tol=1.0e-3, max_iter=50, micro_iter=20, start=1, end=-1):
        """Move one image at a time, sweeping the optimization across the path.

        Slower per image but its Jacobian is three image-blocks tall however
        long the path is, which is what makes it the route for a large system.
        The tolerance tightens as the sweep converges, and the direction of the
        sweep alternates so a bias from the sweeping order does not accumulate.
        """
        start, end = self._resolve_segment(start, end)
        self.neval = 0
        images = list(range(start, end))
        current = tol * 10.0
        self.compute_disps()
        for iteration in range(max_iter):
            largest = 0.0
            previous = self.path.copy()
            for index in images[:-1]:
                middle = (self.path[index - 1] + self.path[index + 1]) * 0.5
                self.smooth(current,
                            max_iter=min(micro_iter, iteration + 6),
                            start=index, end=index + 1,
                            friction=self.friction if iteration else 0.1,
                            xref=middle)
                largest = max(largest, self.optimality)
            self.compute_disps()
            if largest < tol:
                break
            current = max(tol * 0.5, largest * 0.2)
            images = list(reversed(images))
        return self.path


# ---- Interpolation ----

def mid_point(atoms, geom1, geom2, tol=1.0e-2, nudge=0.01, threshold=4.0,
              lattice=None, rng=None):
    """The geometry that bisects two endpoints in internal coordinates.

    The bisection point is the geometry whose scaled distances are the average
    of the two endpoints', found by least squares from either endpoint as the
    starting guess (the Cartesian average is deliberately not used as a guess:
    it tears molecules apart). A pair that comes into contact only at the
    bisection is added to the coordinate set and the search repeated, which is
    what keeps a new contact from being measured for the first time there.
    """
    first = np.asarray(geom1, dtype=np.float64).reshape(-1, 3)
    second = np.asarray(geom2, dtype=np.float64).reshape(-1, 3)
    generator = rng if isinstance(rng, np.random.Generator) \
        else np.random.default_rng(rng)
    extra = set()
    geometries = [first, second]
    while True:
        pairs, rest = get_bond_list(geometries, threshold=threshold + 1.0,
                                    enforce=extra, lattice=lattice, rng=generator)
        scaler = morse_scaler(alpha=0.7, re=rest)
        target, _ = compute_wij(first, pairs, scaler, lattice)
        target = (target + compute_wij(second, pairs, scaler, lattice)[0]) * 0.5
        friction = 0.1 / np.sqrt(first.shape[0])
        best_distance, best_geometry = np.inf, None
        for coefficient in (0.02, 0.98):
            guess = (first * coefficient
                     + (1.0 - coefficient) * second).ravel()
            guess = guess + nudge * generator.random(guess.shape)

            def residual(value, pairs=pairs, scaler=scaler, target=target,
                         guess=guess):
                scaled, _ = compute_wij(value.reshape(-1, 3), pairs, scaler,
                                        lattice)
                return np.concatenate((scaled - target,
                                       (value - guess) * friction))

            def jacobian(value, pairs=pairs, scaler=scaler, guess=guess):
                _scaled, derivative = compute_wij(
                    value.reshape(-1, 3), pairs, scaler, lattice)
                return np.vstack((derivative,
                                  np.identity(value.size) * friction))

            solved = _least_squares(residual, jacobian, guess, ftol=tol,
                                    gtol=tol, max_evaluations=50)
            middle = solved.reshape(-1, 3)
            grown = geometries + [middle]
            new_pairs, _rest = get_bond_list(grown, threshold=threshold,
                                             min_neighbors=0, lattice=lattice,
                                             rng=generator)
            discovered = set(new_pairs) - set(pairs)
            if discovered:
                # A screened pair came into contact at the bisection, so the
                # coordinate set was too small: add it and search again.
                geometries = grown
                extra |= discovered
                break
            smoother = Geodesic(atoms, [first, middle, second], 0.7,
                                threshold=threshold, friction=1.0,
                                lattices=(None if lattice is None
                                          else [lattice, lattice, lattice]),
                                rng=generator)
            smoother.compute_disps()
            width = max(float(np.sqrt(np.mean((geometry - smoother.path[1])
                                              ** 2)))
                        for geometry in (first, second))
            distance = width + smoother.length
            if distance < best_distance:
                best_distance, best_geometry = distance, smoother.path[1]
        else:
            break
    return best_geometry


def redistribute(atoms, geoms, nimages, tol=1.0e-2, lattice=None, rng=None,
                 align=True):
    """Add or remove images until the path carries `nimages` of them.

    Too few: bisect the segment with the largest Cartesian RMSD. Too many:
    remove the image whose removal merges the shortest pair of segments. The
    chain is re-aligned after every change, so the RMSDs the choice is made on
    measure the motion rather than the placement.

    `align` is False for a periodic band: a rigid rotation of an image inside a
    fixed cell is a different structure unless the cell is rotationally
    symmetric, so a periodic path keeps every image where its bisection put it
    (and the bisection is anchored on the two endpoints' own frame through the
    restraint in `mid_point`).
    """
    path = [np.asarray(geometry, dtype=np.float64).reshape(-1, 3)
            for geometry in geoms]
    if align:
        align_path(path)
    generator = rng if isinstance(rng, np.random.Generator) \
        else np.random.default_rng(rng)
    while len(path) < nimages:
        gaps = [float(np.sqrt(np.mean((first - second) ** 2)))
                for first, second in zip(path[1:], path)]
        widest = int(np.argmax(gaps))
        insertion = mid_point(atoms, path[widest], path[widest + 1], tol,
                             threshold=4.0, lattice=lattice, rng=generator)
        path.insert(widest + 1, insertion)
        if align:
            align_path(path)
    while len(path) > nimages:
        gaps = [float(np.sqrt(np.mean((first - second) ** 2)))
                for first, second in zip(path[2:], path)]
        closest = int(np.argmin(gaps))
        del path[closest + 1]
        if align:
            align_path(path)
    return path


def interpolate(atoms, endpoints, nimages, lattice=None, scaler=1.7,
                threshold=3.0, min_neighbors=4, friction=1.0e-2,
                tol=2.0e-3, max_iter=15, rng=None):
    """`nimages` interior geometries between two endpoints, on the geodesic.

    Parameters
    ----------
    atoms : sequence of str
        Element symbols.
    endpoints : sequence
        Two `(natom, 3)` geometries in Angstrom: the reflections this path is
        anchored on, kept exactly as given (nothing here re-anchors or
        realigns them, which is what lets a band hold them fixed).
    nimages : int
        How many interior geometries to produce.
    lattice : array-like or None
        `(3, 3)` cell in Angstrom for a periodic system, or the two endpoints'
        cells for a variable-cell one. A variable-cell band interpolates the
        cell linearly between its endpoints; the metric then reads each image's
        distances at that image's own cell, and the box itself is a ramp.
    scaler, threshold, min_neighbors
        The metric: the Morse stiffness, the pair cutoff, the minimum
        coordination.
    tol, max_iter
        The smoother's stopping tolerance and evaluation budget.
    rng : numpy.random.Generator or None
        Seed of the bisection nudges, so one band is reproducible.

    Returns
    -------
    numpy.ndarray
        `(nimages, natom, 3)` interiors, in Angstrom.

    Raises
    ------
    ValueError
        When fewer than one interior image is asked for, or when the system is
        larger than this port's dense solver is the right tool for.
    """
    count = int(nimages)
    if count < 1:
        raise ValueError(
            'a geodesic interpolation produces interior images and this run '
            'asked for %d of them' % count)
    if len(endpoints) < 2:
        raise ValueError('a geodesic interpolation needs two endpoints')
    symbols = list(atoms)
    if len(symbols) > _MAX_ATOMS:
        raise ValueError(
            'a geodesic band with %d atoms needs a dense normal-equation system '
            'of ((2l + m) nrij) x (m 3N); this port solves that densely and '
            'stops being the right tool past %d atoms. Ask for '
            'neb_interpolate linear, which needs no solve'
            % (len(symbols), _MAX_ATOMS))
    first = np.asarray(endpoints[0], dtype=np.float64).reshape(-1, 3)
    second = np.asarray(endpoints[-1], dtype=np.float64).reshape(-1, 3)
    generator = rng if isinstance(rng, np.random.Generator) \
        else np.random.default_rng(rng)
    # A single cell is one the whole path is measured in; a pair of cells is a
    # variable-cell band, whose images each read their distances at their own
    # cell along the ramp below.
    ladder = None
    single = None
    if lattice is not None:
        given = np.asarray(lattice, dtype=np.float64)
        if given.ndim == 3:
            ladder = given
        elif given.shape == (3, 3):
            single = given
        else:
            raise ValueError(
                'a periodic interpolation takes one (3, 3) cell for the whole '
                'path or one per endpoint, and this run gave shape %s'
                % (given.shape,))
    # One frame before one interpolation: a product written across a face
    # is the same structure in another image, and the metric alone would
    # not stop the initial guess from tearing it through the wall.
    first, second = consistent_pair(
        first, second,
        single if single is not None else (ladder[0] if ladder is not None
                                           else None))
    chain = redistribute(symbols, [first, second], count + 2, tol=tol * 5.0,
                         lattice=single, rng=generator,
                         align=(ladder is None and single is None))
    frames = None
    if ladder is not None or single is not None:
        pair = (ladder if ladder is not None
                else np.asarray([single, single], dtype=np.float64))
        frames = [pair[0] * (1.0 - step) + pair[-1] * step
                  for step in np.linspace(0.0, 1.0, len(chain))]
    smoother = Geodesic(symbols, chain, scaler=scaler, threshold=threshold,
                        min_neighbors=min_neighbors, friction=friction,
                        lattices=frames, rng=generator)
    if len(symbols) > _SWEEP_ATOMS:
        smoother.sweep(tol=tol, max_iter=max_iter)
    else:
        smoother.smooth(tol=tol, max_iter=max_iter)
    # No alignment after the smoothing, deliberately. This metric measures a
    # segment between the *Cartesian midpoint* of two adjacent images, so where
    # an image sits is part of the length and not a gauge choice: re-aligning the
    # smoothed path moves it off the optimum the smoother just found. Measured on
    # the HCN -> HNC pair, the final alignment took the internal length from
    # 2.3920 back up to 9.2144, while the chain was already coherent because
    # `redistribute` aligned it before the smoothing ran.
    return smoother.path[1:-1]


__all__ = [
    'compute_rij',
    'compute_wij',
    'morse_scaler',
    'elu_scaler',
    'get_bond_list',
    'Geodesic',
    'mid_point',
    'redistribute',
    'interpolate',
]