"""
======================================================================
Collective variables and their coordinate Jacobians.

A collective variable maps an atomic geometry to one scalar. This module
provides the geometric ones the constraint families need (bond distance,
bond angle, dihedral angle and the periodic difference) together with the exact
derivative dq/dR, so a bias or a wall can chain its derivative back to the
Cartesian coordinates without finite differences. `evaluate_on_images` is the
same cores read on unfolded coordinates. It also
provides the cell variables (axis length, axis-pair angle, volume and the six
Green-St. Venant strain components) with dq/dB, the derivative with respect to
the lattice.

A cell variable has no atomic derivative at all: its value does not depend on
where the atoms sit, only on the lattice. That is a trap on the gradient path,
because a workflow that scatters a CV derivative onto atoms gets an all-zero
atom gradient and can conclude there is no force. The cell derivative is the
whole of it, and the caller must route it to the barostat or to the cell degrees
of freedom, not drop it.

Conventions
-----------
- The numeric core works in atomic units: coordinates in Bohr, lengths in
  Bohr, angles in radians, energies in Hartree. Values here carry the length
  unit of the coordinates passed in, and Jacobians carry its inverse, so the
  caller stays in one unit system end to end. Angles are radians, as in
  utilities.geometric.
- A Jacobian is returned as a compact array covering only the involved
  atoms, in the order named by the variable, not as a dense (natom, 3). The
  callers scatter it with `scatter_jacobian`. A cell variable's Jacobian is
  instead a dense (3, 3), one entry per lattice number, because the cell has no
  atom axis to scatter onto: an atomic gradient of a cell-only variable is
  identically zero, and the driver lives entirely in dq/dB.
- Cells follow the repository's row convention (utilities/lattice.py): the rows
  of the matrix are the cell vectors, `R = S @ B` for fractional `S`, so the
  deformation gradient that carries a reference cell to the current one is
  `T = cell @ inv(reference)`. Writing that product the other way round is the
  classic way to get a shear sign wrong, so it is formed once, in
  `_strain_transform`.
- Degenerate geometry raises CVSingularError instead of producing nan. A nan
  in a constraint gradient is worse than a failure: it flows into SHAKE
  multipliers, thermostat scaling and an optimizer step, and the run keeps
  going with a geometry that is quietly wrong. The shared
  `utilities.geometric` cores raise the same error, so the CV front doors wrap
  those cores rather than carrying a second formula.

The analytic forms for the bond, angle and dihedral come from the shared cores
in `utilities/geometric.py`; the compact front doors keep the caller's atom
order and unit system, and `evaluate_on_images` uses the dense `BND`/`AGL`/`DHD`
adapters over unfolded coordinates.

The module imports `numpy` and `.geometric` only; it does not import schema,
data, workflow or skills.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

# The bond, angle and torsion arithmetic lives in `utilities/geometric.py`, in
# one implementation that the whole package shares (D-01 closed at the merge:
# this module used to carry its own torsion because the shared one's gradient
# was wrong on three of four atoms; the shared one is now correct, so the copy
# is gone rather than kept in parallel). These names stay here as aliases
# because callers and tests read them from this module.
from pyrai3md.utilities.geometric import (  # noqa: F401  (re-exported names)
    COS_TOLERANCE,
    CVSingularError,
    SINGULAR_LENGTH,
    SINGULAR_SINE,
    AGL,
    BND,
    DHD,
    _acos,
    _as_coords,
    angle_core,
    bond_core,
    torsion_core,
)


def distance(coords, i, j, grad=False):
    """Bond distance between two atoms, in the length unit of the input.

    Returns
    -------
    float or tuple
        The distance, or (distance, jacobian) where the jacobian has shape
        (2, 3) in the atom order (i, j).
    """
    value, rows = bond_core(_as_coords(coords), int(i), int(j))
    return value if not grad else (value, rows)


def angle(coords, i, j, k, grad=False):
    """Angle i-j-k at the central atom j, in radians.

    Returns
    -------
    float or tuple
        The angle, or (angle, jacobian) with the jacobian shaped (3, 3) in
        the atom order (i, j, k).
    """
    value, rows = angle_core(_as_coords(coords), int(i), int(j), int(k))
    return value if not grad else (value, rows)


def dihedral(coords, i, j, k, m, grad=False):
    """Signed dihedral i-j-k-m in radians, in (-pi, pi].

    The value is the IUPAC torsion: 0 for cis, pi for trans, with the sign
    following the right-handed rotation about the j-k bond. It comes from
    atan2 of the two scalar invariants of the defining vectors, so no acos
    domain breach is possible and a strict trans or cis geometry is a number
    rather than a nan.

    Returns
    -------
    float or tuple
        The dihedral, or (dihedral, jacobian) with the jacobian shaped
        (4, 3) in the atom order (i, j, k, m).
    """
    value, rows = torsion_core(_as_coords(coords), int(i), int(j), int(k),
                               int(m))
    return value if not grad else (value, rows)


def cell_length(cell, axis, grad=False):
    """Length of one cell vector, in the length unit of the cell.

    Parameters
    ----------
    cell : array-like
        (3, 3) lattice with the cell vectors as rows, the repository's
        convention in utilities/lattice.py.
    axis : int
        0, 1 or 2 for a, b or c.
    grad : bool
        Also return dq/dcell as a (3, 3) array whose [i, j] entry is the
        derivative with respect to `cell[i, j]`.

    Returns
    -------
    float or tuple
    """
    matrix = _as_cell(cell)
    vector = matrix[axis]
    value = float(np.linalg.norm(vector))
    if value < SINGULAR_LENGTH:
        raise CVSingularError(
            'cell axis %d has zero length (%.3e); its direction, and so the '
            'derivative of its length, is undefined' % (axis, value))
    if not grad:
        return value
    jacobian = np.zeros((3, 3), dtype=np.float64)
    jacobian[axis] = vector / value
    return value, jacobian


def cell_angle(cell, first, second, grad=False):
    """Angle between two cell vectors, in radians.

    Returns
    -------
    float or tuple
        The angle, or (angle, dq/dcell) shaped (3, 3).
    """
    matrix = _as_cell(cell)
    u = matrix[first]
    v = matrix[second]
    lu = float(np.linalg.norm(u))
    lv = float(np.linalg.norm(v))
    if lu < SINGULAR_LENGTH or lv < SINGULAR_LENGTH:
        raise CVSingularError(
            'cell angle %d-%d has a zero-length vector (%.3e, %.3e)'
            % (first, second, lu, lv))
    cosine = float(np.dot(u, v)) / lu / lv
    value = _acos(cosine, 'cell angle %d-%d' % (first, second))
    if not grad:
        return value
    sine = float(np.sin(value))
    if abs(sine) < SINGULAR_SINE:
        raise CVSingularError(
            'cell angle %d-%d is 0 or pi (%.6f rad); the two vectors are '
            'parallel and the angle has no derivative'
            % (first, second, value))
    jacobian = np.zeros((3, 3), dtype=np.float64)
    # d cos / du = v/(lu lv) - cos u / lu^2, and the angle is -acos of that.
    du = -1.0 / sine * (v / lu / lv - cosine * u / (lu * lu))
    dv = -1.0 / sine * (u / lu / lv - cosine * v / (lv * lv))
    jacobian[first] = du
    jacobian[second] = dv
    return value, jacobian


def cell_volume(cell, grad=False):
    """Cell volume as the absolute determinant, in the cube of the length unit.

    Returns
    -------
    float or tuple
        The volume, or (volume, dq/dcell). The derivative is the signed
        determinant times the transpose of the inverse, which keeps the same
        sign as the determinant itself: a left-handed cell would already be a
        refusal, so the volume is the plain determinant here.
    """
    matrix = _as_cell(cell)
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1.0e-12:
        raise CVSingularError(
            'the cell is singular (det %.3e); it has no volume and no inverse'
            % determinant)
    value = abs(determinant)
    if not grad:
        return value
    jacobian = determinant * np.linalg.inv(matrix).T
    return value, jacobian


def strain_components(cell, reference):
    """The six independent Green-St. Venant strain components of one cell.

    With `T = cell @ inv(reference)` the deformation gradient (rows are cell
    vectors, so the product reads in the repository's convention), the strain
    is `E = 1/2 (T^T T - I)`. That is the tensor whose energy conjugate is the
    second Piola-Kirchhoff stress, and it is exactly the right choice for the
    variable of a restraint, because at the reference cell it vanishes to first
    order and its derivative there is the symmetric small strain the barostat
    already integrates. It is NOT the barostat's own log-deformation at finite
    strain: the two agree through first order and differ beyond, and a run that
    restrains strain under NPT must say which one it means.

    The returned order is Voigt: xx, yy, zz, yz, xz, xy.
    """
    matrix = _as_cell(cell)
    base = _as_cell(reference)
    transform = _strain_transform(matrix, base)
    tensor = 0.5 * (transform.T @ transform - np.eye(3))
    return np.array([tensor[0, 0], tensor[1, 1], tensor[2, 2],
                     2.0 * tensor[1, 2], 2.0 * tensor[0, 2],
                     2.0 * tensor[0, 1]])


def strain(cell, component, reference=None, grad=False):
    """One strain component of a cell, and its derivative with respect to it.

    Parameters
    ----------
    cell : array-like
        The current (3, 3) lattice, rows are cell vectors.
    component : int or tuple
        A Voigt index 0..5, or an (k, l) pair naming the tensor component
        directly.
    reference : array-like, optional
        The undeformed cell. Without one the cell is measured against itself,
        which makes the value zero and is only useful for checking the
        derivative at the reference.
    grad : bool
        Also return dq/dcell as a (3, 3) array.

    Returns
    -------
    float or tuple
    """
    matrix = _as_cell(cell)
    base = matrix if reference is None else _as_cell(reference)
    transform = _strain_transform(matrix, base)
    first, second = _strain_pair(component)
    value = 0.5 * (float(np.dot(transform[:, first], transform[:, second]))
                   - (1.0 if first == second else 0.0))
    if first != second:
        # Voigt's factor of two is a bookkeeping convention, not a different
        # number: the engineering shear is twice the tensor component.
        value *= 2.0
    if not grad:
        return value
    inverse = np.linalg.inv(base)
    jacobian = 0.5 * (np.outer(transform[:, first], inverse[:, second])
                      + np.outer(transform[:, second], inverse[:, first]))
    if first != second:
        jacobian *= 2.0
    return value, jacobian


def _as_cell(cell):
    """A float64 (3, 3) lattice, checked, with a non-singular shape."""
    array = np.asarray(cell, dtype=np.float64)
    if array.shape != (3, 3):
        raise CVSingularError(
            'a cell must be (3, 3), got shape %s' % (array.shape,))
    if not np.all(np.isfinite(array)):
        raise CVSingularError('the cell holds a non-finite number')
    return array


def _strain_transform(matrix, base):
    """The deformation gradient `T` that takes the reference cell to `matrix`.

    Rows are cell vectors, so `cell = T @ reference` and `T = cell @
    inv(reference)`: the same row-vector reading utilities/lattice.py uses for
    `S = R @ inv(B)`. Transposing this relation is the most common way to get
    the sign of a shear wrong, so it is written once, here.
    """
    if abs(float(np.linalg.det(base))) < 1.0e-12:
        raise CVSingularError(
            'the reference cell is singular; no deformation is measured '
            'against it')
    return matrix @ np.linalg.inv(base)


def _strain_pair(component):
    """Voigt index or (k, l) pair as the two tensor indices."""
    if isinstance(component, (int, np.integer)):
        index = int(component)
        if not 0 <= index <= 5:
            raise CVSingularError(
                'Voigt strain index %d is outside 0..5' % index)
        return _VOIGT[index]
    first, second = (int(value) for value in component)
    if not all(0 <= value <= 2 for value in (first, second)):
        raise CVSingularError(
            'strain component (%d, %d) is outside the tensor' % (first, second))
    return (first, second)


_VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def periodic_difference(value, target):
    """Shortest signed difference between two periodic values, in radians.

    Both arguments are angles; the result lies in (-pi, pi]. This is the
    form the harmonic bias needs on a dihedral: subtracting and then folding
    by hand, as legacy PyRAI2MD did with '360 - abs(dr)', loses the direction
    of the crossing near +-180.
    """
    delta = float(value) - float(target)
    return float(np.arctan2(np.sin(delta), np.cos(delta)))


def scatter_jacobian(jacobian, indices, natom):
    """Expand a compact Jacobian into a dense (natom, 3) array.

    Parameters
    ----------
    jacobian : array-like
        Shape (k, 3), one row per atom named in `indices`.
    indices : array-like
        The k atom indices, in the same order.
    natom : int
        Total atom count.

    Returns
    -------
    numpy.ndarray
        float64 (natom, 3) with zeros outside the involved atoms.
    """
    rows = np.asarray(jacobian, dtype=np.float64)
    members = np.asarray(indices, dtype=np.int64).reshape(-1)
    if rows.shape != (members.size, 3):
        raise CVSingularError(
            'jacobian shape %s does not match %d indices'
            % (rows.shape, members.size))
    dense = np.zeros((int(natom), 3), dtype=np.float64)
    np.add.at(dense, members, rows)
    return dense


def zero_gradient(natom):
    """A dense (natom, 3) gradient of zeros, float64."""
    return np.zeros((int(natom), 3), dtype=np.float64)

def evaluate_on_images(expanded, var, kind, grad=False):
    """Evaluate one CV on unfolded coordinates, reusing the single core.

    This is `BND`, `AGL` and `DHD` reached by name, on an array that came out of
    `bonds.unfold_components`; no geometry math is duplicated here. The gradient
    it returns is shaped (natom, 3) against the unfolded coordinates, which by
    the write-back identity of unfolding is the same array against the stored
    coordinates.

    Parameters
    ----------
    expanded : numpy.ndarray
        Unfolded coordinates, shape (natom, 3).
    var : sequence
        Atom indices the CV reads, 2 for a bond, 3 for an angle, 4 for a
        dihedral.
    kind : str
        'bond', 'angle' or 'dihedral'.

    Returns
    -------
    float or tuple
        The value, or (value, gradient) when `grad` is set.

    Raises
    ------
    ValueError
        Unknown CV kind, or an index count that does not match the kind.
    CVSingularError
        The unfolded geometry is degenerate, as the underlying core reports it.
    """
    cores = {'bond': 2, 'angle': 3, 'dihedral': 4}
    if kind not in cores:
        raise ValueError(
            "unknown CV kind %r; this layer evaluates %s"
            % (kind, ', '.join(sorted(cores))))
    indices = [int(atom) for atom in var]
    if len(indices) != cores[kind]:
        raise ValueError(
            "a %s CV takes %d atom indices, got %d"
            % (kind, cores[kind], len(indices)))
    if kind == 'bond':
        return BND(expanded, indices, grad=grad)
    if kind == 'angle':
        return AGL(expanded, indices, grad=grad)
    return DHD(expanded, indices, grad=grad)

