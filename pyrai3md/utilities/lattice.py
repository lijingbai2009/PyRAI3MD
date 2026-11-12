"""
======================================================================
This module provides the shared periodic-cell math used across the
schema layer (input normalization and validation) and the workflow
layer (integrator operations).

Pure functions only: this module never stores cell state. The runtime
authority for the cell is the PyRAIState system fields
(lattice_vector / cell / pbc / volume, defined by the schema System
model); cross-process transfer uses the calculate(cell=...) snapshot
channel. See dev/methods/PBC_DEVELOPMENT.md sections 2 and 5.

Conventions
-----------
- Row-vector layout: the lattice matrix B has the cell vectors as
  rows; Cartesian coordinates R and fractional coordinates S satisfy
  R = S @ B and S = R @ B^-1.
- Lengths in Angstrom, angles in degrees, volume in Angstrom^3.
- Standard orientation: a along x, b in the xy plane.
- pbc is normalized to exactly three 0/1 integers.

The module must stay a leaf: no imports from schema, workflow or data
(skills must not import it either; framework-skill isolation).

Sep 25 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

# Lengths (Angstrom) and |det| (Angstrom^3) below this tolerance are
# treated as zero: a degenerate cell axis or a singular matrix.
LATTICE_EPS = 1e-8

# The lattice components a variable-cell run relaxes: the six entries of the
# row-vector lattice on and below the diagonal, in row order. The three above it
# stay as they were, because that triangular spelling is what the engines read
# back (a LAMMPS data file and a cp2k cell card both ask for it), and a full 3x3
# step walks the box out of that spelling within one iteration. Nothing is lost
# by holding them: those directions shear the atoms against the box, and the 3N
# atomic components of the same vector already answer that question.
CELL_COMPONENT_SLOTS = (0, 3, 4, 6, 7, 8)


def cell6_to_matrix(cell6):
    """Build the (3, 3) lattice matrix from the 6 classic cell parameters.

    Inverse of matrix_to_cell6 (ported from the legacy
    Molecule.pbc_helper.compute_cell): the input is
    [a, b, c, alpha, beta, gamma] with the lengths in Angstrom and the
    angles in degrees, and the returned rows are the cell vectors
    (Angstrom) in the standard orientation (a along x, b in the xy
    plane).

    Parameters
    ----------
    cell6 : array-like
        The 6 cell parameters [a, b, c, alpha, beta, gamma].

    Returns
    -------
    np.ndarray
        Lattice matrix, shape (3, 3), rows are cell vectors (Angstrom).
    """
    params = np.asarray(cell6, dtype=np.float64).reshape(-1)[:6]
    a, b, c = params[0], params[1], params[2]
    alpha = np.radians(params[3])
    beta = np.radians(params[4])
    gamma = np.radians(params[5])
    cos_a, cos_b, cos_g = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sin_g = np.sin(gamma)
    x = np.array([1.0, 0.0, 0.0]) * a
    y = np.array([cos_g, sin_g, 0.0]) * b
    cz = cos_a - cos_b * cos_g
    sz2 = (1.0 + 2.0 * cos_a * cos_b * cos_g
           - cos_a ** 2 - cos_b ** 2 - cos_g ** 2)
    sz = np.sqrt(max(sz2, 0.0))
    z = np.array([cos_b, cz / sin_g, sz / sin_g]) * c
    return np.array([x, y, z])


def parse_cell(value):
    """Read one cell spelling into the (3, 3) row-vector matrix.

    The three accepted spellings are the six classic parameters
    ``[a, b, c, alpha, beta, gamma]`` (lengths Angstrom, angles degrees),
    nine row-major components, and a nested 3x3 matrix; a classic string
    is split on whitespace or commas. The adapter's `molecule.cell`
    keyword and validate's periodicity gate both call this, so they
    cannot disagree about what one spelling means.

    Parameters
    ----------
    value : str, list, tuple
        The cell as written.

    Returns
    -------
    list of list of float
        Three rows of three components (Angstrom).
    """
    if isinstance(value, str):
        try:
            value = [float(t) for t in value.replace(',', ' ').split()]
        except ValueError:
            raise ValueError(
                "cell expects numeric parameters, got %r" % value)
    elif isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            matrix = [[float(x) for x in row] for row in value]
            if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                raise ValueError(
                    "cell as a nested matrix must have shape (3, 3), "
                    "got %r" % (value,))
            return matrix
        try:
            value = [float(x) for x in value]
        except (TypeError, ValueError):
            raise ValueError(
                "cell expects numeric parameters, got %r" % (value,))
    else:
        raise ValueError(
            "cell expects 6 parameters, 9 components or a 3x3 matrix, "
            "got %r" % (value,))

    if len(value) == 6:
        return cell6_to_matrix(value).tolist()
    if len(value) == 9:
        return [value[0:3], value[3:6], value[6:9]]
    raise ValueError(
        "cell expects 6 parameters, 9 components or a 3x3 matrix, "
        "got %d values: %r" % (len(value), value))


def matrix_to_cell6(lattice):
    """Convert a (3, 3) lattice matrix to the 6 cell parameters.

    The lattice rows are the cell vectors (Angstrom); the returned
    parameters are [a, b, c, alpha, beta, gamma] with the angles in
    degrees, matching the classic cell keyword layout.

    Parameters
    ----------
    lattice : array-like
        Lattice matrix, shape (3, 3), rows are cell vectors (Angstrom).

    Returns
    -------
    list of float
        [a, b, c, alpha, beta, gamma].
    """
    lattice = np.asarray(lattice, dtype=np.float64)
    a_vec, b_vec, c_vec = lattice[0], lattice[1], lattice[2]
    a = float(np.linalg.norm(a_vec))
    b = float(np.linalg.norm(b_vec))
    c = float(np.linalg.norm(c_vec))

    def _angle(u, v):
        denom = np.linalg.norm(u) * np.linalg.norm(v)
        if denom == 0.0:
            return 90.0
        cosang = np.clip(np.dot(u, v) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))

    alpha = _angle(b_vec, c_vec)
    beta = _angle(a_vec, c_vec)
    gamma = _angle(a_vec, b_vec)
    return [a, b, c, alpha, beta, gamma]


def cell_volume(lattice):
    """Return the unit-cell volume |det(B)| in Angstrom^3.

    Parameters
    ----------
    lattice : array-like
        Lattice matrix, shape (3, 3).

    Returns
    -------
    float
        Cell volume; 0.0 when the matrix is all zero.
    """
    lattice = np.asarray(lattice, dtype=np.float64)
    if lattice.shape != (3, 3) or not np.any(lattice):
        return 0.0
    return float(abs(np.linalg.det(lattice)))


def stress_to_cell_force(stress, lattice):
    """The lattice derivative a stress tensor is the conjugate of.

    With the row-vector lattice (`R = S B`) and a strain that acts as
    `B -> (I + eps) B`, solving `dE = V Tr(sigma eps)` for the lattice
    derivative gives

        dE/dB = V sigma^T B^-T

    which for a cubic row of length h reduces to the checkable `V sigma / h`:
    stretching one axis by dh changes the volume by h^2 dh, so the force on that
    entry is the pressure conjugate times h^2 and not times h^3.

    This is the derivative at *fixed fractional* coordinates, which is the path a
    cell step takes when the atoms ride the box. A step that holds the atoms in
    Cartesian space instead differentiates along another path, and the two differ
    by the nuclear gradient contracted with the fractional coordinates - measured
    on an equilibrated water box as factors between -9.5 and +36.9 with three sign
    flips, against 0.93-1.07 for this one
    (`tools/cell_stress_fd.py`, `dev/methods/CELL_OPT_DEVELOPMENT.md` section 5.3).

    Parameters
    ----------
    stress : array-like
        (3, 3) tensile-positive stress, in energy per unit volume.
    lattice : array-like
        (3, 3) row-vector lattice, in the length unit the volume is wanted in.

    Returns
    -------
    numpy.ndarray
        (3, 3) `dE/dB`, in energy per unit length of a lattice entry.
    """
    tensor = np.asarray(stress, dtype=np.float64)
    rows = np.asarray(lattice, dtype=np.float64)
    if tensor.shape != (3, 3):
        raise ValueError('a stress tensor must be (3, 3), got %s'
                         % (tensor.shape,))
    if rows.shape != (3, 3):
        raise ValueError('a lattice must be (3, 3), got %s' % (rows.shape,))
    volume = float(abs(np.linalg.det(rows)))
    return volume * tensor.T @ np.linalg.inv(rows).T


def cell_force_to_stress(cell_force, lattice):
    """The stress tensor a lattice derivative is the conjugate of.

    Inverse of `stress_to_cell_force`. Two spellings are accepted, because the
    two places that need the reading hold different things:

    - a full (3, 3) `dE/dB`, closed as `sigma = B (dE/dB)^T / V`, which
      multiplies out to the identity on any matrix that came from a stress, so
      no symmetry has to be assumed and no system has to be solved;
    - the six relaxed components in `CELL_COMPONENT_SLOTS` order, which is all
      a variable-cell optimizer keeps. There the three entries the search does
      not move are missing, and they are not zero: one shear of the stress puts
      force on both `B[1][0]` and `B[0][1]`. The six independent components of
      a symmetric tensor are therefore recovered by solving the 6 by 6 map the
      slots define. On an orthorhombic box the two spellings agree, and they
      part ways as soon as the box is tilted.

    This is what turns a cell convergence threshold written in one unit into
    the other: a threshold on `dE/dB` (Hartree per Bohr) means a different
    pressure on boxes of different size, and the two readings have to say the
    same thing about the same box. Measured on the eighth round's water box,
    `9.807e-4` Hartree per Bohr on its x row is 130.5 bar, and the same reading
    on a box twice as long is four times softer.

    Parameters
    ----------
    cell_force : array-like
        (3, 3) `dE/dB`, or the six relaxed components in
        `CELL_COMPONENT_SLOTS` order, in energy per unit lattice length.
    lattice : array-like
        (3, 3) row-vector lattice, in the length unit the stress is wanted in.

    Returns
    -------
    numpy.ndarray
        (3, 3) tensile-positive stress, in energy per unit volume.

    Raises
    ------
    ValueError
        When the shapes are wrong, or when the lattice has no volume to divide
        by.
    """
    tensor = np.asarray(cell_force, dtype=np.float64)
    rows = np.asarray(lattice, dtype=np.float64)
    if rows.shape != (3, 3):
        raise ValueError('a lattice must be (3, 3), got %s' % (rows.shape,))
    determinant = float(np.linalg.det(rows))
    if abs(determinant) <= LATTICE_EPS:
        raise ValueError('a lattice with no volume cannot carry a stress: '
                         'det(B) = %.3e' % determinant)
    if tensor.shape == (3, 3):
        return rows @ tensor.T / abs(determinant)
    flat = tensor.reshape(-1)
    if flat.size != len(CELL_COMPONENT_SLOTS):
        raise ValueError(
            'a lattice derivative is (3, 3) or the %d relaxed components, got '
            'shape %s' % (len(CELL_COMPONENT_SLOTS), tensor.shape))
    slots = list(CELL_COMPONENT_SLOTS)
    # Column k of the map is what the slots read when the stress is the k-th
    # independent component of a symmetric tensor alone. The slots are exactly
    # the lower triangle of a 3x3 in row order, so they index both the relaxed
    # lattice entries and the independent stress components.
    map6 = np.empty((6, 6), dtype=np.float64)
    for index, slot in enumerate(slots):
        basis = np.zeros((3, 3), dtype=np.float64)
        basis[divmod(slot, 3)] = 1.0
        basis[(slot % 3), (slot // 3)] = 1.0
        map6[:, index] = stress_to_cell_force(basis, rows).reshape(-1)[slots]
    solution = np.linalg.solve(map6, flat)
    stress = np.zeros((3, 3), dtype=np.float64)
    for value, slot in zip(solution, slots):
        stress[divmod(slot, 3)] = value
        stress[(slot % 3), (slot // 3)] = value
    return stress


def normalize_pbc(value):
    """Normalize a pbc input to exactly three 0/1 integers.

    Accepted forms: None or empty (non-periodic, [0, 0, 0]); a single
    0/1 scalar (broadcast to the three axes); a length-3 sequence of
    0/1 values (int, float or bool). Anything else (wrong length,
    values outside {0, 1}, non-numeric) raises ValueError: the pbc
    flags are the single normalized authority for periodicity, so
    ambiguity is rejected instead of guessed.

    Parameters
    ----------
    value : None, int, or sequence
        Raw pbc input.

    Returns
    -------
    list of int
        Three 0/1 integers.
    """
    if value is None:
        return [0, 0, 0]
    if isinstance(value, (bool, np.bool_)):
        value = int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        items = [value] * 3
    else:
        items = list(value)
        if len(items) == 0:
            return [0, 0, 0]
    if len(items) != 3:
        raise ValueError(
            "pbc must have exactly 3 elements (one per axis), got %d: %r"
            % (len(items), value))
    normalized = []
    for item in items:
        if isinstance(item, (bool, np.bool_)):
            item = int(item)
        if isinstance(item, (int, np.integer)):
            flag = int(item)
        elif isinstance(item, (float, np.floating)):
            if item not in (0.0, 1.0):
                raise ValueError(
                    "pbc values must be 0 or 1, got %r in %r" % (item, value))
            flag = int(item)
        else:
            raise ValueError(
                "pbc values must be 0 or 1, got non-numeric %r in %r"
                % (item, value))
        if flag not in (0, 1):
            raise ValueError(
                "pbc values must be 0 or 1, got %r in %r" % (item, value))
        normalized.append(flag)
    return normalized


def boundary_condition(pbc):
    """Derive the boundary type from pbc flags.

    Takes over the semantics of the deleted data.structure Cell class.

    Parameters
    ----------
    pbc : None, int, or sequence
        Raw or normalized pbc flags.

    Returns
    -------
    str
        'cluster' (no periodic axis), 'periodic' (all three axes) or
        'surface' (mixed, 1 or 2 axes).
    """
    flags = normalize_pbc(pbc)
    n = sum(flags)
    if n == 0:
        return 'cluster'
    if n == 3:
        return 'periodic'
    return 'surface'


def validate_lattice(lattice, pbc=None, eps=LATTICE_EPS):
    """Validate a lattice matrix and return its volume.

    Structural checks: shape (3, 3) and all-finite entries. When pbc
    is given (any periodic axis), additionally every periodic axis
    vector must have a length above eps and the matrix must be
    non-singular (|det| > eps). A non-periodic validation only rejects
    malformed matrices that claim to be cells (all-zero passes as
    "no cell defined" and returns 0.0).

    Parameters
    ----------
    lattice : array-like or None
        Lattice matrix, shape (3, 3).
    pbc : None, int, or sequence
        Periodicity flags; None or all-zero skips the periodic checks.
    eps : float
        Tolerance for zero lengths and the determinant.

    Returns
    -------
    float
        The cell volume |det(B)| in Angstrom^3 (0.0 when no cell).

    Raises
    ------
    ValueError
        On a malformed, non-finite, degenerate or singular cell.
    """
    flags = normalize_pbc(pbc)
    if lattice is None:
        lattice = []
    arr = np.asarray(lattice, dtype=np.float64)
    if arr.size == 0:
        if any(flags):
            raise ValueError(
                "PBC is enabled (pbc=%s) but no lattice matrix is defined"
                % flags)
        return 0.0
    if arr.shape != (3, 3):
        raise ValueError(
            "lattice_vector must have shape (3, 3), got %s" % (arr.shape,))
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "lattice_vector contains non-finite entries: %r"
            % arr.tolist())
    if not any(flags):
        return cell_volume(arr)
    for axis in range(3):
        if flags[axis]:
            length = float(np.linalg.norm(arr[axis]))
            if length <= eps:
                raise ValueError(
                    "PBC is enabled on axis %d but the cell vector is "
                    "degenerate (length %.3e Angstrom)" % (axis, length))
    volume = float(abs(np.linalg.det(arr)))
    if volume <= eps:
        raise ValueError(
            "PBC is enabled but the lattice matrix is singular "
            "(|det| = %.3e Angstrom^3)" % volume)
    return volume


def cart_to_frac(coordinates, lattice):
    """Convert Cartesian coordinates to fractional coordinates.

    Parameters
    ----------
    coordinates : array-like
        Cartesian coordinates, shape (natom, 3), Angstrom.
    lattice : array-like
        Lattice matrix, shape (3, 3), rows are cell vectors.

    Returns
    -------
    np.ndarray
        Fractional coordinates S = R @ B^-1, shape (natom, 3).

    Raises
    ------
    ValueError
        When the lattice matrix is singular.
    """
    lattice = np.asarray(lattice, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    try:
        inv_lattice = np.linalg.inv(lattice)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "singular lattice matrix, cannot convert to fractional "
            "coordinates: %r" % lattice.tolist()) from exc
    return coordinates @ inv_lattice


def frac_to_cart(fractional, lattice):
    """Convert fractional coordinates to Cartesian coordinates.

    Parameters
    ----------
    fractional : array-like
        Fractional coordinates, shape (natom, 3).
    lattice : array-like
        Lattice matrix, shape (3, 3), rows are cell vectors.

    Returns
    -------
    np.ndarray
        Cartesian coordinates R = S @ B, shape (natom, 3), Angstrom.
    """
    lattice = np.asarray(lattice, dtype=np.float64)
    fractional = np.asarray(fractional, dtype=np.float64)
    return fractional @ lattice


def wrap_fractional_with_image(fractional, pbc):
    """Reduce fractional coordinates modulo 1 and report the crossings.

    The wrap half of PBC together with the bookkeeping the wrapped value
    throws away: for every periodic axis the integer count of cell
    lengths the atom crossed is returned alongside the wrapped
    coordinate, so the continuous position can be rebuilt exactly with
    unwrap_coordinates. Non-periodic axes keep their fractional value and
    report 0 crossings.

    The convention matches the image flags an explicit periodic engine
    stores per atom: ``fractional = wrapped + image`` on a periodic axis,
    which is also the identity the upper-edge snap is required to
    preserve (an atom sitting a hair below the cell origin wraps to 0.0
    and is credited with the crossing it made to get there).

    Parameters
    ----------
    fractional : array-like
        Fractional coordinates, shape (natom, 3).
    pbc : sequence
        Normalized or raw pbc flags (three 0/1 values).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ``(wrapped, image)``: wrapped fractional coordinates in
        [0, 1) on the periodic axes, and the integer (natom, 3) crossing
        count. Both are new arrays; the input is not modified.
    """
    flags = normalize_pbc(pbc)
    frac = np.array(fractional, dtype=np.float64, copy=True)
    image = np.zeros(frac.shape, dtype=np.int64)
    axis = np.asarray(flags, dtype=bool)
    if not np.any(axis):
        return frac, image

    crossings = np.floor(frac[:, axis]).astype(np.int64)
    wrapped = frac[:, axis] - crossings
    # numpy's modulo can return exactly 1.0 for a tiny negative input
    # (e.g. -1e-17 % 1.0 == 1.0), which violates the half-open [0, 1)
    # convention: a coordinate infinitesimally below the cell origin
    # would land on the upper edge instead of 0.0, making a boundary
    # atom flip between lo and hi every step. Snap the upper edge back
    # down (same periodic image) and credit the extra crossing, so that
    # fractional = wrapped + image survives the snap.
    at_upper_edge = wrapped >= 1.0
    if np.any(at_upper_edge):
        wrapped = wrapped - at_upper_edge.astype(np.float64)
        crossings = crossings + at_upper_edge.astype(np.int64)
    frac[:, axis] = wrapped
    image[:, axis] = crossings
    return frac, image


def wrap_fractional(fractional, pbc):
    """Reduce fractional coordinates modulo 1 on the periodic axes.

    Non-periodic axes are left untouched. Wrapping and minimum-image
    displacement are different operations; this is the wrap half only,
    and it discards the crossing count. Callers that need to rebuild the
    continuous coordinates must use wrap_fractional_with_image.

    Parameters
    ----------
    fractional : array-like
        Fractional coordinates, shape (natom, 3).
    pbc : sequence
        Normalized or raw pbc flags (three 0/1 values).

    Returns
    -------
    np.ndarray
        Wrapped fractional coordinates (a copy; the input is not
        modified).
    """
    return wrap_fractional_with_image(fractional, pbc)[0]


def unwrap_coordinates(coordinates, image, lattice):
    """Rebuild continuous Cartesian coordinates from the main image.

    Inverse of the wrap: the stored (wrapped) coordinates plus the per
    atom crossing count, taken back through the cell. The result is the
    trajectory of the cell the atom started in, which is what a
    cross-frame displacement needs; it is not what a backend that insists
    on in-cell coordinates should receive.

    Parameters
    ----------
    coordinates : array-like
        Wrapped Cartesian coordinates, shape (natom, 3), Angstrom.
    image : array-like or None
        Integer crossing count, shape (natom, 3). None or an empty array
        is read as "no crossings", which returns the input unchanged.
    lattice : array-like
        Lattice matrix, shape (3, 3), rows are cell vectors.

    Returns
    -------
    np.ndarray
        Continuous Cartesian coordinates, shape (natom, 3), Angstrom.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if image is None or len(image) == 0:
        return coordinates.copy()
    image = np.asarray(image, dtype=np.int64)
    if image.size == 0:
        return coordinates.copy()
    lattice = np.asarray(lattice, dtype=np.float64)
    return frac_to_cart(cart_to_frac(coordinates, lattice) + image, lattice)


def continuous_view(coordinates, image, lattice):
    """The un-wrapped position of every atom, in the run's own length unit.

    A periodic run stores main-image coordinates plus the integer count of
    cell lengths each atom has crossed (`System.image`). The unwrapped view is
    a derived reading and not a second truth: it is what lets a cross-frame
    difference, a restart, and a region cut-out agree on the same continuous
    path. Guards, in order: no image, no lattice, a lattice that is not a
    3x3 cell, or an empty count array all return the input as a copy, so a
    non-periodic run and a configuration whose bond perception refused
    (utilities/bonds hands back no counts) both take the identity path.
    """
    array = np.asarray(coordinates, dtype=np.float64)
    rows = [[int(value) for value in row] for row in (image or [])]
    cell = np.asarray(lattice if lattice is not None else [],
                      dtype=np.float64)
    if not rows or cell.shape != (3, 3):
        return array.copy()
    return unwrap_coordinates(array, rows, cell)


def half_box_bound(lattice_vector):
    """Return the distance a periodic bond must stay below (Angstrom).

    Half of the shortest cell extent, where the extent along one axis is
    the interplanar spacing rather than the length of its vector: a
    tilted cell's vectors are longer than the gap between the planes
    they span, and what LAMMPS' own consistency check measures is that
    gap. For an orthorhombic cell the two coincide. The result carries
    the length unit of the input, not always Angstrom.
    """
    lattice = np.asarray(lattice_vector, dtype=np.float64).reshape(3, 3)
    volume = abs(float(np.linalg.det(lattice)))
    if volume <= 0.0:
        return 0.0
    extents = []
    for axis in range(3):
        first = lattice[(axis + 1) % 3]
        second = lattice[(axis + 2) % 3]
        area = float(np.linalg.norm(np.cross(first, second)))
        if area <= 0.0:
            return 0.0
        extents.append(volume / area)
    return 0.5 * min(extents)


# The candidate images a nearest-image search probes first: the 27 cells
# sharing a face, edge or corner with the main image, i.e. +/-1 along each
# fractional axis around the component-wise reduction. The zero shift comes
# first so a tie keeps the component-wise vector.
_MIN_IMAGE_SHELL = np.array(
    [(0, 0, 0)] + [(dx, dy, dz)
                   for dx in (-1, 0, 1)
                   for dy in (-1, 0, 1)
                   for dz in (-1, 0, 1)
                   if (dx, dy, dz) != (0, 0, 0)],
    dtype=np.int64)

# How far the search may widen around the component-wise value. The
# certificate below asks for more only on a cell that is not size-reduced,
# where the honest answer is that this function cannot name the nearest
# image cheaply, not a longer silent scan.
_MIN_IMAGE_MAX_SHELL = 3


def _image_shifts(radius):
    """Integer probe shifts within +/- radius of the component-wise value."""
    span = np.arange(-radius, radius + 1, dtype=np.int64)
    return np.array([(dx, dy, dz)
                     for dx in span
                     for dy in span
                     for dz in span], dtype=np.int64)


def _masked_shifts(shifts, flags):
    """The shift set with non-periodic axes pinned to zero, zero first."""
    masked = np.asarray(shifts, dtype=np.int64) * flags[None, :]
    ordered = []
    seen = set()
    zero = (0, 0, 0)
    if zero in {tuple(int(value) for value in row) for row in masked}:
        ordered.append(zero)
        seen.add(zero)
    for row in masked:
        key = tuple(int(value) for value in row)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return np.array(ordered, dtype=np.int64)


def min_image_displacement(displacements, lattice, pbc=(1, 1, 1)):
    """Map Cartesian difference vectors into the nearest periodic image.

    For displacements (not positions): a vector between two atoms is
    reduced to the shortest equivalent one under the cell. This is what
    makes bond detection and bond analysis work on the main-image
    coordinates the framework stores, where a molecule may straddle a
    cell face.

    The reduction searches the candidate images around the component-wise
    value and keeps the shortest, rather than rounding each fractional
    component on its own. Component-wise rounding is only the answer for an
    orthogonal cell: in a tilted one the three axes are not independent, and
    a shift that lengthens one component can shorten the vector overall. A
    60 degree hexagonal cell of 10 Angstrom sides, which is anything but
    pathological, is mis-reduced by over 2 Angstrom by the component-wise
    rule (tests/test_lattice.py pins that margin, and the capability rule of
    dev/methods/CONSTRAINT_DEVELOPMENT.md section 5.2 is what consumes it).

    The first pass covers the 27 neighbours of the component-wise value,
    which is exhaustive for a size-reduced cell. The result then certifies
    itself: a shorter image would have to lie inside the ellipsoid the best
    length spans, whose extent along a fractional axis is that length times
    the norm of the matching column of the inverse lattice. When the
    certificate shows the shell was not exhaustive the search widens to the
    box it asks for and is exact over that box.

    Parameters
    ----------
    displacements : array-like
        Difference vectors in Cartesian form, shape (..., 3), Angstrom.
    lattice : array-like
        Lattice matrix, shape (3, 3), rows are cell vectors.
    pbc : sequence of int, optional
        Three 0/1 flags, one per fractional axis. Omitted means all axes
        periodic. A non-periodic axis is never shifted, so the result
        keeps the input component on that axis. None is refused: omitting
        the argument is the all-periodic request, while
        `normalize_pbc(None)` is the separate non-periodic contract.

    Returns
    -------
    np.ndarray
        Nearest-image displacement vectors, same shape as the input.

    Raises
    ------
    ValueError
        The lattice is singular, so no fractional representation exists, or
        the certificate demands a wider search than _MIN_IMAGE_MAX_SHELL,
        which means the cell is so non-reduced that its vectors are much
        longer than the gaps between the planes they span.
    """
    displacements = np.asarray(displacements, dtype=np.float64)
    lattice = np.asarray(lattice, dtype=np.float64)
    if pbc is None:
        raise ValueError(
            'min_image_displacement takes explicit pbc flags; omit the '
            'argument for an all-periodic reduction')
    flags = np.asarray(normalize_pbc(pbc), dtype=np.int64)
    flat = displacements.reshape(-1, 3)
    try:
        inv_lattice = np.linalg.inv(lattice)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "singular lattice matrix, cannot reduce a displacement: %r"
            % lattice.tolist()) from exc
    # One subtraction per component, written the way the component-wise
    # reduction always wrote it, so a cell where that rule was already
    # minimal returns bit-for-bit the numbers it returned before the search.
    # A non-periodic axis keeps its raw fractional component and is masked
    # out of the shift set below, so its Cartesian component never moves.
    deviation = flat @ inv_lattice
    deviation = deviation - np.round(deviation) * flags
    shifts = _masked_shifts(_MIN_IMAGE_SHELL, flags)
    shell = 1
    while True:
        cart = (deviation[:, None, :] - shifts[None, :, :]) @ lattice
        lengths = np.sqrt(np.einsum('ijk,ijk->ij', cart, cart))
        winner = np.argmin(lengths, axis=1)
        rows = np.arange(cart.shape[0])
        best = cart[rows, winner]
        shortest = lengths[rows, winner]
        # Extent of the ellipsoid of vectors shorter than `shortest` along
        # each fractional axis. An image inside it sits within that plus the
        # half a unit the component-wise value already leaves it, so the
        # radius below is the one a search must cover to be exhaustive. A
        # non-periodic axis cannot be shifted, so it contributes no reach.
        reach = shortest[:, None] * np.linalg.norm(inv_lattice, axis=0)
        reach = reach * flags[None, :]
        needed = int(np.ceil(0.5 + float(reach.max()))) if reach.size else 1
        if needed <= shell:
            return best.reshape(displacements.shape)
        if needed > _MIN_IMAGE_MAX_SHELL:
            raise ValueError(
                "the nearest image is not certifiable within a search of +/-%d "
                "cells: a candidate shorter than %.3f reaches %.2f fractional "
                "units along an axis, which means this lattice is not "
                "size-reduced (its vectors are far longer than the spacing "
                "between the planes they span): %r"
                % (_MIN_IMAGE_MAX_SHELL, float(shortest.max()),
                   float(reach.max()), lattice.tolist()))
        shell = needed
        shifts = _masked_shifts(_image_shifts(shell), flags)
