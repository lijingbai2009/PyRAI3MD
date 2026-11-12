"""
======================================================================
This module provides the vibrational and thermodynamic analysis of a
force-constant matrix: normal modes, frequencies, and the harmonic
thermochemical corrections built from them.

It is a pure-function service and owns no workflow. `workflow/hessian.py`
hands it a matrix, whether that matrix arrived analytically from a backend
or was assembled here from displaced gradients, and both routes therefore
report the same numbers from the same arithmetic. The Frequency workflow this
used to be was the other half of that duplication: it re-asked a Hessian and
then diagonalized a placeholder, so the two files between them computed
nothing.

Units and conventions, stated once because every one of them has been a bug
somewhere else:

  coordinates   Bohr
  masses        amu (g/mol), the unit `system.mass` is written in
  hessian       Hartree / Bohr^2
  frequencies   cm^-1, negative for an imaginary mode
  modes         Bohr, unmass-weighted, normalized so sum(m * x^2) = 1,
                which is the convention `workflow.sampling` reads frequency
                records in and the one `norm_mode` is declared with
  reduced mass  amu
  rotation      GHz, the unit the result model declares
  constants

# A mode is found in the mass-weighted space, where the kinetic energy is a
# plain squared norm, and divided back out by the square root of each atom's
# mass on the way out. Dividing once, on the way out, is what makes the record
# readable by the sampler without a second scaling step.

Oct 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

# A rigid-motion column this shallow in singular value is not a direction the
# structure resists: for a linear molecule the rotation about its own axis is
# exactly zero, and for a fragment of one it is zero to within the rounding of
# the input coordinates. Treating such a column as a constraint would remove a
# vibration that exists.
RIGID_TOLERANCE = 1.0e-8


def rigid_block(coords, masses):
    """The candidate translation and rotation vectors, mass-weighted.

    Six columns are handed over rather than five or three because the count is
    not known here: `svd` of this block reports the rank, and the rank is what
    says how many degrees of freedom the structure really carries as a whole. A
    linear molecule drops its own axial rotation to zero and comes back with
    rank 5, a single atom with rank 3, and a non-linear one with rank 6. The
    axis frame is not special: any fixed orthonormal set spans the same space,
    and the SVD orthogonalizes it, so the lab axes are used and no principal
    axis calculation is needed to get here.

    Parameters
    ----------
    coords : numpy.ndarray
        (natom, 3) positions in Bohr.
    masses : numpy.ndarray
        (natom,) masses in amu, positive.

    Returns
    -------
    tuple
        (block, center_of_mass). `block` is (3 * natom, 6): the first three
        columns are the translations, the last three the rotations about the
        center of mass, each weighted by the square root of the atom's mass.
    """
    positions = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    if values.size != positions.shape[0]:
        raise ValueError(
            'rigid block needs one mass per atom: %d atoms, %d masses'
            % (positions.shape[0], values.size))
    if np.any(values <= 0.0):
        raise ValueError('rigid block needs positive masses')
    weights = np.sqrt(values)
    center = np.sum(positions * values.reshape(-1, 1), axis=0) / float(
        np.sum(values))

    block = np.zeros((positions.size, 6), dtype=np.float64)
    for axis in range(3):
        rows = block[:, axis]
        rows.reshape(-1, 3)[:, axis] = weights
    for axis in range(3):
        direction = np.zeros(3)
        direction[axis] = 1.0
        lever = np.cross(direction, positions - center)
        block[:, 3 + axis] = (weights.reshape(-1, 1) * lever).reshape(-1)
    return block, center


def vibrational_basis(coords, masses, tolerance=RIGID_TOLERANCE, mask=None):
    """The orthonormal complement of the rigid motions, in mass-weighted space.

    `vibration` uses this to read frequencies; a curvature-following optimizer
    needs the same subspace to count downhill directions in, because the raw
    (3N, 3N) Cartesian model carries the rigid motions as directions whose
    curvature is whatever the updates left there. One construction, so the two
    counts cannot drift apart.

    Parameters
    ----------
    coords : numpy.ndarray
        (natom, 3) positions in Bohr.
    masses : numpy.ndarray
        (natom,) masses in amu, positive.
    mask : numpy.ndarray or None
        (natom, 3) boolean, True where a coordinate cannot move. The complement
        is then taken *within* the movable components, and the directions taken
        out of it are the whole-body motions the frozen set *absorbs*: a rigid
        motion that leaves every frozen atom in place still costs nothing, while
        one the anchors oppose has the curvature of an ordinary deformation and
        stays in. Columns of the returned basis are zero on the masked
        components, so a caller can use them without scattering.

    Returns
    -------
    tuple
        (basis, weights, center, rank). `basis` is (3N, nmode) with orthonormal
        columns spanning the vibrational subspace, `weights` is the (3N,) square
        root of each atom's mass repeated per Cartesian component, which is what
        carries a Cartesian gradient or Hessian into the same metric, and
        `center` is the center of mass the rotation vectors were built about,
        while `rank` is how many directions were taken out as costless (six for
        a non-linear molecule with nothing frozen, five for a linear one, three
        for a single atom, and as many as the frozen set absorbs of the
        remainder once a `fix` is in play).
    """
    positions = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    weights = np.sqrt(values)
    rigid, center = rigid_block(positions, values)
    weighted = np.repeat(weights, 3)
    if mask is None:
        left, singular, _right = np.linalg.svd(rigid, full_matrices=True)
        top = float(singular[0]) if singular.size else 0.0
        rank = (int(np.count_nonzero(singular > tolerance * top))
                if top > 0.0 else 0)
        basis = left[:, rank:] if rank < left.shape[1] else np.zeros(
            (left.shape[0], 0))
        return basis, weighted, center, rank

    active = np.flatnonzero(~np.asarray(mask, dtype=bool).reshape(-1))
    frozen = np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1))
    if active.size == weighted.size:
        # Nothing is frozen: the whole block is in play and every rigid motion
        # is a null direction of it.
        nulls = np.eye(rigid.shape[1], dtype=np.float64)
    else:
        # A whole-body motion that leaves every frozen atom where it stands is
        # still a direction the surface offers no restoring force against, and
        # must go. One the anchors *oppose* is not a free motion any more: it is
        # an ordinary deformation with an ordinary curvature, and a search that
        # projects it away cannot reach the stationary point the freeze still
        # has. Measured on the six-atom saddle with one atom pinned: taking the
        # three rotations the anchor absorbs leaves the search a subspace it can
        # converge in (73 evaluations, `max_g 1.9e-4` at the saddle); also
        # removing the three it opposes holds `max_g` at 1.73e-2 for 400
        # evaluations, because the force along those directions is real and
        # nothing else in the search can answer it.
        _left, singular, rowh = np.linalg.svd(rigid[frozen, :],
                                              full_matrices=True)
        top = float(singular[0]) if singular.size else 0.0
        kept_rank = (int(np.count_nonzero(singular > tolerance * top))
                     if top > 0.0 else 0)
        nulls = rowh[kept_rank:].T
    # The combinations the anchors absorb, written on the components that move.
    # They are orthonormal already: the rigid block is, and the frozen rows of
    # these combinations are zero by construction.
    absorbed = rigid[np.ix_(active, np.arange(rigid.shape[1]))] @ nulls
    left, singular, _right = np.linalg.svd(absorbed, full_matrices=True)
    top = float(singular[0]) if singular.size else 0.0
    rank = (int(np.count_nonzero(singular > tolerance * top))
            if top > 0.0 else 0)
    reduced = left[:, rank:] if rank < left.shape[1] else np.zeros(
        (left.shape[0], 0))
    basis = np.zeros((weighted.size, reduced.shape[1]), dtype=np.float64)
    basis[active, :] = reduced
    return basis, weighted, center, rank


def periodic_rigid_block(coords, masses, lattice):
    """The candidate whole-body motions of a structure and its box.

    Six columns, as `rigid_block` hands over, but each is a vector of the joint
    (3N + 6) space a variable-cell search steps in: the first three are the
    translations, which move the atoms and leave the lattice alone; the last
    three are the rotations, which move the atoms *and* the box, because a
    rotation of the atoms alone is not a symmetry of a periodic structure - it
    changes the fractional coordinates, which is a deformation with a restoring
    force, not a motion the surface offers nothing against. The lattice half of a
    rotation about axis a is the row-vector `e_a x h_k` of each cell vector.

    A general rotation also strains the three entries above the diagonal, and the
    run does not move those: the engines read back a triangular box, so the six
    entries on and below the diagonal are the whole coordinate set
    (`optimizer/base.py::CELL_COMPONENT_SLOTS`). The candidates are therefore
    written into those six and the rest is dropped, which can cost the block a
    rank. That is reported rather than hidden - a caller reads the rank the SVD
    finds, and a box whose triangular spelling cannot express one of its own
    rotations is a box whose curvature count has one direction fewer.

    Parameters
    ----------
    coords : numpy.ndarray
        (natom, 3) positions in Bohr.
    masses : numpy.ndarray
        (natom,) masses in amu, positive.
    lattice : numpy.ndarray
        (3, 3) row-vector lattice in Bohr.

    Returns
    -------
    tuple
        (block, center_of_mass). `block` is (3N + 6, 6): the first three columns
        the translations, the last three the rotations, in the metric the joint
        space uses (`sqrt(m)` on the atomic components, one on the lattice
        entries, which are themselves lengths in Bohr).
    """
    positions = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    rows = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    atomic, center = rigid_block(positions, values)
    from ..utilities.lattice import CELL_COMPONENT_SLOTS

    block = np.zeros((positions.size + len(CELL_COMPONENT_SLOTS), 6),
                     dtype=np.float64)
    block[:positions.size, :] = atomic
    slots = list(CELL_COMPONENT_SLOTS)
    for axis in range(3):
        direction = np.zeros(3)
        direction[axis] = 1.0
        # dh_k = Omega x h_k, written into the six entries the run moves.
        strained = np.cross(np.broadcast_to(direction, (3, 3)), rows)
        block[positions.size:, 3 + axis] = strained.reshape(-1)[slots]
    return block, center


def periodic_vibrational_basis(coords, masses, lattice,
                               tolerance=RIGID_TOLERANCE, mask=None):
    """The orthonormal complement of the whole-body motions, box included.

    The joint reading of `vibrational_basis` for a search whose step carries the
    six lattice entries beside the 3N atomic ones. The two are separate functions
    rather than one with an optional argument, because the metric differs: a
    lattice entry has no mass, so the joint space is weighted by `sqrt(m)` on the
    atomic components and by one on the cell's, and a frequency read off a matrix
    in that metric is a curvature and not something convertible to cm^-1. A
    caller that wants frequencies calls `vibrational_basis`; a curvature search
    that relaxes a box calls this one.

    Parameters
    ----------
    coords : numpy.ndarray
        (natom, 3) positions in Bohr.
    masses : numpy.ndarray
        (natom,) masses in amu, positive.
    lattice : numpy.ndarray or None
        (3, 3) row-vector lattice in Bohr. None gives the atomic answer: the
        basis, the weights and the rank are then element-for-element what
        `vibrational_basis` returns, which is the criterion that keeps the two
        from drifting apart.
    tolerance : float
        Relative singular-value cut on the rigid block, shared with the atomic
        reading.
    mask : numpy.ndarray or None
        (natom, 3) boolean, True where a coordinate cannot move, with the meaning
        `vibrational_basis` gives it: the directions taken out are the
        whole-body motions the frozen set absorbs. The six lattice entries are
        never masked - a run that froze one did not ask for a cell block.

    Returns
    -------
    tuple
        (basis, weights, center, rank). `basis` is (3N + 6, nmode) with
        orthonormal columns, `weights` is (3N + 6,) - `sqrt(m)` per atomic
        component and one per lattice entry - `center` is the center of mass the
        rotations were built about, and `rank` is how many whole-body directions
        were taken out.
    """
    positions = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    atomic = 3 * positions.shape[0]
    if lattice is None:
        basis, weights, center, rank = vibrational_basis(
            positions, values, tolerance=tolerance, mask=mask)
        return basis, weights, center, rank

    from ..utilities.lattice import CELL_COMPONENT_SLOTS

    cell = len(CELL_COMPONENT_SLOTS)
    rigid, center = periodic_rigid_block(positions, values, lattice)
    weighted = np.concatenate((np.repeat(np.sqrt(values), 3), np.ones(cell)))
    if mask is None:
        frozen = np.zeros(atomic, dtype=bool)
    else:
        frozen = np.asarray(mask, dtype=bool).reshape(-1)
    # The joint mask: the atomic components the plan froze, and nothing among the
    # lattice entries.
    joint_frozen = np.concatenate((frozen, np.zeros(cell, dtype=bool)))
    active = np.flatnonzero(~joint_frozen)
    if active.size == weighted.size:
        nulls = np.eye(rigid.shape[1], dtype=np.float64)
    else:
        _left, singular, rowh = np.linalg.svd(rigid[joint_frozen, :],
                                              full_matrices=True)
        top = float(singular[0]) if singular.size else 0.0
        kept = (int(np.count_nonzero(singular > tolerance * top))
                if top > 0.0 else 0)
        nulls = rowh[kept:].T
    absorbed = rigid[np.ix_(active, np.arange(rigid.shape[1]))] @ nulls
    left, singular, _right = np.linalg.svd(absorbed, full_matrices=True)
    top = float(singular[0]) if singular.size else 0.0
    rank = (int(np.count_nonzero(singular > tolerance * top))
            if top > 0.0 else 0)
    reduced = left[:, rank:] if rank < left.shape[1] else np.zeros(
        (left.shape[0], 0))
    basis = np.zeros((weighted.size, reduced.shape[1]), dtype=np.float64)
    basis[active, :] = reduced
    return basis, weighted, center, rank


def vibration(coord, mass, hessian, mobile=None, tolerance=RIGID_TOLERANCE):
    """Diagonalize one Hessian into frequencies and normal modes.

    Parameters
    ----------
    coord : array-like
        (natom, 3) equilibrium positions in Bohr.
    mass : array-like
        (natom,) masses in amu.
    hessian : array-like
        (3 natom, 3 natom) force constants in Hartree per Bohr^2. A partial
        Hessian is accepted as it is written: the rows and columns of the atoms
        outside `mobile` must be the zeros the caller left there, and they take
        no part in the arithmetic below.
    mobile : array-like, optional
        0-based indices of the atoms whose block is the vibration problem, the
        selection a local Hessian declares. None is every atom. The rigid
        motions removed are those of *this* set, about this set's center of
        mass: projecting the whole molecule's translations and rotations out of
        a fragment's block would leave the fragment's own rigid drift in the
        result as if it were a vibration.
    tolerance : float
        Relative singular-value cut on the rigid block.

    Returns
    -------
    dict
        ``frequencies`` cm^-1 with the imaginary modes signed negative,
        ``modes`` of shape (nmode, natom, 3) in the sampling convention above,
        ``reduced_mass`` amu, ``inertia`` the three principal moments in
        amu Bohr^2 of the mobile set, ``rotation_constants`` GHz, ``linear``
        whether that set is collinear, ``mobile`` the 0-based indices used and
        ``rigid_removed`` the rank the projection took out.
    """
    positions = np.asarray(coord, dtype=np.float64).reshape(-1, 3)
    values = np.asarray(mass, dtype=np.float64).reshape(-1)
    matrix = np.asarray(hessian, dtype=np.float64)
    natom = positions.shape[0]
    if values.size != natom:
        raise ValueError(
            'vibration needs one mass per atom: %d atoms, %d masses'
            % (natom, values.size))
    size = 3 * natom
    if matrix.shape != (size, size):
        raise ValueError(
            'vibration needs the (3N, 3N) Hessian, got %s for %d atoms'
            % (matrix.shape, natom))

    if mobile is None:
        keep = np.arange(natom, dtype=np.int64)
    else:
        keep = np.asarray(mobile, dtype=np.int64).reshape(-1)
        if keep.size == 0:
            raise ValueError('vibration was given an empty mobile selection')
        if np.any(keep < 0) or np.any(keep >= natom):
            raise ValueError(
                'mobile selection names an atom outside the structure')
        keep = np.unique(keep)
    components = (keep.reshape(-1, 1) * 3
                  + np.arange(3, dtype=np.int64).reshape(1, -1)).reshape(-1)

    mobile_mass = values[keep]
    block = matrix[np.ix_(components, components)]
    block = 0.5 * (block + block.T)
    # One rigid-motion construction with the curvature follower: the basis is
    # the complement of the six (or five, or three) whole-body directions in the
    # mass-weighted metric, and `weighted` is what brings the Cartesian block
    # into that metric.
    basis, weighted, center, rank = vibrational_basis(
        positions[keep], mobile_mass, tolerance=tolerance)
    scale = np.outer(weighted, weighted)
    mass_weighted = block / scale
    nmode = basis.shape[1]
    if nmode == 0:
        empty = np.zeros(0)
        return {
            'frequencies': empty, 'modes': np.zeros((0, natom, 3)),
            'reduced_mass': empty, 'inertia': np.zeros(3),
            'rotation_constants': np.zeros(3), 'linear': True,
            'mobile': keep.tolist(), 'rigid_removed': rank,
        }

    # The projected Hessian lives in the orthonormal complement of the rigid
    # block, so its eigenvectors are already normalized in the mass-weighted
    # metric and the frequencies follow from the eigenvalues directly.
    projected = basis.T @ mass_weighted @ basis
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (projected + projected.T))
    signed = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
    frequencies = signed * _vibration_frequency_factor()

    # Out of the mass-weighted space and back into Bohr, once: the columns of
    # `mass_weighted_modes` satisfy sum(|L|^2) = 1, and dividing by sqrt(m)
    # turns that into sum(m * x^2) = 1, the normalization the sampler assumes.
    mass_weighted_modes = (basis @ eigenvectors).T
    mobile_modes = mass_weighted_modes / weighted.reshape(1, -1)
    modes = np.zeros((nmode, natom, 3), dtype=np.float64)
    modes[:, keep, :] = mobile_modes.reshape(nmode, keep.size, 3)
    reduced = 1.0 / np.sum(np.abs(mobile_modes) ** 2, axis=1)

    inertia, linear = _principal_moments(positions[keep], mobile_mass, center)
    rotation_constants = _rotation_constants(inertia)
    return {
        'frequencies': frequencies,
        'modes': modes,
        'reduced_mass': reduced,
        'inertia': inertia,
        'rotation_constants': rotation_constants,
        'linear': linear,
        'mobile': keep.tolist(),
        'rigid_removed': rank,
    }


def _vibration_frequency_factor():
    """sqrt(Hartree / (amu Bohr^2)) in cm^-1, from the unit table."""
    from pyrai3md.utilities.units import VIBRATION_TO_CM
    return VIBRATION_TO_CM


def _principal_moments(coords, masses, center):
    """Principal moments of inertia (amu Bohr^2) and the collinear flag.

    The collinear test is a statement about the structure, not a number the
    caller declares: the smallest moment is zero for a line of atoms, and that
    is the same fact the rigid block already reported as its rank.
    """
    inertia = np.zeros((3, 3), dtype=np.float64)
    shifted = np.asarray(coords, dtype=np.float64) - center
    for position, mass in zip(shifted, masses):
        inertia += mass * (float(np.dot(position, position)) * np.eye(3)
                           - np.outer(position, position))
    values = np.linalg.eigvalsh(0.5 * (inertia + inertia.T))
    values = np.sort(values)
    top = float(abs(values[-1])) if values.size else 0.0
    linear = bool(top > 0.0 and abs(values[0]) <= RIGID_TOLERANCE * top)
    return values, linear


def thermo_analysis(energy, masses, frequencies, rotation_constants,
                    temperature=298.15, pressure=1.0, multiplicity=1,
                    scale_factor=1.0, cutoff=100.0, damping_alpha=4.0,
                    symmetry=1, linear=False):
    """Harmonic thermochemical corrections, per molecule, in Hartree.

    Every term is the textbook rigid-rotor harmonic-oscillator sum written in
    SI and converted to Hartree once at the end, which is what lets the numbers
    be checked against an analytical backend key by key: on the B3LYP/6-31g*
    water used by the verification, the zero point energy agrees with PySCF's to
    1.3e-9 relative and `heat_capacity_cp - heat_capacity_cv` is exactly the gas
    constant. Two deliberate differences from that backend are recorded rather
    than hidden: the symmetry number here is whatever the caller declared
    (default 1, so a C2v water comes out 2.1952e-6 Hartree/K = k ln 2 hotter in
    entropy than PySCF's auto-detected sigma=2, and the run log says so), and
    the vibrational entropy may be damped (below) where the backend is plain.

    The standard state is the declared pressure, 1 bar by default, through the
    translational partition function's kT/P factor. The legacy OQP analysis
    reached the same place through a hardcoded 1000 angstrom^3 box; that is not
    reproduced here because a box that size is a concentration nobody else in
    this package uses.

    Parameters
    ----------
    energy : float
        Electronic energy in Hartree.
    masses : array-like
        Atomic masses in amu; only the total enters (translation).
    frequencies : array-like
        cm^-1, signed, with the imaginary modes negative.
    rotation_constants : array-like
        GHz, from `vibration`; a linear rotor has one entry at zero.
    temperature : float
        K.
    pressure : float
        bar, the standard state of the translational and total enthalpy terms.
    multiplicity : float
        Spin multiplicity, the electronic entropy's R ln(g). Zero or one gives
        zero electronic entropy: the legacy default of 0 took the logarithm of
        zero and returned -inf, which then propagated into the free energy.
    scale_factor : float
        Harmonic frequency scaling factor applied before everything else.
    cutoff : float
        Frequency below which the vibrational entropy starts blending into the
        free-rotor form, in cm^-1. Zero turns the blending off and leaves the
        plain harmonic oscillator.
    damping_alpha : float
        Exponent of that blend.
    symmetry : int
        Rotational symmetry number sigma.
    linear : bool
        Collinear structure, which costs the rotor one degree of freedom.

    Returns
    -------
    dict
        The synthetic properties under the names the result model declares
        (``zero_point_energy``, ``internal_energy``, ``enthalpy``, ``entropy``,
        ``free_energy``, ``heat_capacity_cv``, ``heat_capacity_cp``, all Hartree
        or Hartree/K) and a ``terms`` dict of the pieces they were built from,
        so a printed correction can be added up rather than trusted.
    """
    from pyrai3md.utilities.units import (
        AMU_TO_KG, BAR_TO_PA, BOLTZMANN_CONSTANT, J_TO_HARTREE,
        PLANCK_CONSTANT, SPEED_OF_LIGHT_CM)

    kB = BOLTZMANN_CONSTANT                       # J/K
    T = float(temperature)
    if T <= 0.0:
        raise ValueError('thermo needs a positive temperature, got %g' % T)
    kT = kB * T
    pa = float(pressure) * BAR_TO_PA
    if pa <= 0.0:
        raise ValueError('thermo needs a positive pressure, got %g bar'
                         % pressure)

    wavenumbers = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    wavenumbers = wavenumbers * float(scale_factor)
    # An imaginary mode is not a bound vibration and contributes nothing to the
    # sums; the sign that marks it is kept in `frequencies` for the reader.
    real = wavenumbers[wavenumbers > 0.0]
    # theta = h c nu / kB, the vibrational temperature of each mode, in K.
    theta = (PLANCK_CONSTANT * SPEED_OF_LIGHT_CM * real / kB)
    reduced = theta / T
    with np.errstate(over='ignore', divide='ignore'):
        excit = np.where(reduced > 0.0, 1.0 / np.expm1(reduced), 0.0)

    zpe = 0.5 * kB * float(np.sum(theta))                       # J
    u_trans = 1.5 * kT
    u_rot = (1.0 if linear else 1.5) * kT
    u_vib = kT * float(np.sum(reduced * excit))
    # Hartree in, Hartree out: the electronic term is already a Hartree, so it
    # joins the sums after the SI pieces are converted, not before.
    electronic = float(energy)
    u_total_hartree = ((zpe + u_trans + u_rot + u_vib) * J_TO_HARTREE
                       + electronic)
    pv = kT * J_TO_HARTREE
    enthalpy = u_total_hartree + pv

    mult = float(multiplicity)
    s_el = kB * (np.log(mult) if mult > 1.0 else 0.0)
    total_mass = float(np.sum(np.asarray(masses, dtype=np.float64)))
    from pyrai3md.utilities.units import AMU_TO_KG
    q_trans = ((2.0 * np.pi * total_mass * AMU_TO_KG * kT
                / PLANCK_CONSTANT ** 2) ** 1.5 * kT / pa)
    s_trans = kB * (2.5 + np.log(q_trans))
    constants = np.sort(
        np.asarray(rotation_constants, dtype=np.float64).reshape(-1))[::-1]
    live = constants[constants > 0.0]
    if live.size == 0:
        s_rot = 0.0
    elif linear:
        q_rot = kT / (float(symmetry) * PLANCK_CONSTANT * live[0] * 1.0e9)
        s_rot = kB * (1.0 + np.log(q_rot))
    else:
        product = float(np.prod(live)) * 1.0e27
        q_rot = ((kT / PLANCK_CONSTANT) ** 1.5 * np.pi ** 0.5
                 / (float(symmetry) * product ** 0.5))
        s_rot = kB * (1.5 + np.log(q_rot))

    with np.errstate(over='ignore', divide='ignore'):
        harmonic_terms = kB * (
            reduced * excit - np.log1p(-np.exp(-reduced)))
    if cutoff > 0.0 and real.size:
        # Grimme's quasi-RRHO blend: below `cutoff` a vibration is nearer a free
        # internal rotation than an oscillator, and the harmonic formula's log
        # diverges there. The effective inertia of a mode of frequency nu is
        # h/(8 pi^2 nu), capped by the mean rotational inertia of the molecule,
        # in Hz throughout (the legacy analysis reached here through cm^-1
        # without the speed of light, which is a factor 3e10 off; this is the
        # same formula with the units it is written in).
        damp = 1.0 / (1.0 + (float(cutoff) / real) ** float(damping_alpha))
        inertia_mean = (PLANCK_CONSTANT
                        / (8.0 * np.pi ** 2 * float(np.mean(live)) * 1.0e9)
                        if live.size else 0.0)
        inertia_mode = PLANCK_CONSTANT / (
            8.0 * np.pi ** 2 * real * SPEED_OF_LIGHT_CM)
        blended = (inertia_mean * inertia_mode
                   / (inertia_mean + inertia_mode) if inertia_mean > 0.0
                   else inertia_mode)
        s_free = kB * (0.5 + 0.5 * np.log(
            8.0 * np.pi ** 3 * blended * kT / PLANCK_CONSTANT ** 2))
        s_vib = float(np.sum(damp * harmonic_terms + (1.0 - damp) * s_free))
    else:
        s_vib = float(np.sum(harmonic_terms))
    entropy_hartree_per_k = (s_el + s_trans + s_rot + s_vib) * J_TO_HARTREE
    free_energy = enthalpy - T * entropy_hartree_per_k

    with np.errstate(over='ignore', divide='ignore'):
        term = np.where(reduced > 0.0,
                        reduced ** 2 * np.exp(-reduced) / np.square(
                            -np.expm1(-reduced)), 0.0)
    # Translation 3/2 R, rotation R (linear) or 3/2 R, vibration the Einstein
    # sum above: written term by term because the printed terms have to add back
    # to what is reported.
    cv = ((1.5 + (1.0 if linear else 1.5)) * kB
          + kB * float(np.sum(term)))
    cp = cv + kB

    return {
        'zero_point_energy': zpe * J_TO_HARTREE,
        'internal_energy': u_total_hartree,
        'enthalpy': enthalpy,
        'entropy': entropy_hartree_per_k,
        'free_energy': free_energy,
        'heat_capacity_cv': cv * J_TO_HARTREE,
        'heat_capacity_cp': cp * J_TO_HARTREE,
        'temperature': T,
        'pressure': float(pressure),
        'symmetry_number': int(symmetry),
        'linear': bool(linear),
        'modes_kept': int(real.size),
        'terms': {
            'electronic': electronic,
            'zero_point': zpe * J_TO_HARTREE,
            'translation_energy': u_trans * J_TO_HARTREE,
            'rotation_energy': u_rot * J_TO_HARTREE,
            'vibration_energy': u_vib * J_TO_HARTREE,
            'pV': pv,
            'electronic_entropy': s_el * J_TO_HARTREE,
            'translation_entropy': s_trans * J_TO_HARTREE,
            'rotation_entropy': s_rot * J_TO_HARTREE,
            'vibration_entropy': s_vib * J_TO_HARTREE,
            'vibrational_temperatures': theta.tolist(),
        },
    }


def format_freq_molden(atoms, coords, frequencies, modes):
    """A Molcas-layout frequency record, the shape `sampling` reads.

    The three arrays are written as given, in the convention `vibration`
    returns: coordinates Bohr, frequencies cm^-1 with an imaginary mode signed
    negative, modes unmass-weighted with sum(m * x^2) = 1. That is the contract
    of `workflow.sampling.read_molden`, which uses the vectors as it finds them,
    so no scaling is applied here and none should be.

    Parameters
    ----------
    atoms : list of str
        Element symbols, length natom.
    coords : array-like
        (natom, 3) equilibrium positions in Bohr.
    frequencies : array-like
        (nmode,) cm^-1.
    modes : array-like
        (nmode, natom, 3) Cartesian displacement vectors.

    Returns
    -------
    str
        The record text.
    """
    natom = len(atoms)
    nmode = len(frequencies)
    frequency = '\n'.join('%10.2f' % float(value) for value in frequencies)
    coord = ''.join('%-5s %24.8f %24.8f %24.8f\n'
                    % (symbol, float(x), float(y), float(z))
                    for symbol, (x, y, z) in zip(atoms,
                                                 np.asarray(coords,
                                                            dtype=float)))
    vibs = ''
    for index in range(nmode):
        rows = '\n'.join(' '.join('%24.8f' % float(value) for value in row)
                         for row in modes[index])
        vibs += '  vibration  %5s\n%s\n' % (index + 1, rows)
    return """[MOLDEN FORMAT]
[N_FREQ]
%s
[NATOM]
%s
[FREQ]
%s
[FR-COORD]
%s
[FR-NORM-COORD]
%s

""" % (nmode, natom, frequency, coord, vibs)


def write_freq_molden(path, atoms, coords, frequencies, modes):
    """Write one frequency record and return the path it went to.

    The name matters more than it looks: `sampling.load_frequency_data` finds
    the record as `<title>.freq.molden`, and a record written under any other
    name is not an error the run can see - the sampler simply reads nothing, or
    reads a stale file left by the run before it.

    Parameters
    ----------
    path : str
        Destination file, directory included.
    atoms, coords, frequencies, modes
        See `format_freq_molden`.
    """
    text = format_freq_molden(atoms, coords, frequencies, modes)
    with open(path, 'w') as handle:
        handle.write(text)
    return path


def _rotation_constants(inertia_amu_bohr2):
    """Rotational constants in GHz, from principal moments in amu Bohr^2.

    `h / (8 pi^2 I)` with I in kg m^2 is already an inverse second, so the step
    into GHz is the SI prefix and nothing else; the legacy analysis ended at
    cm^-1 and needs the speed of light there, which is the factor this function
    deliberately does not apply (measured: applying it leaves every constant
    thirty times too small). GHz is what the result model declares.

    Three numbers come back for every structure, a linear one included: the
    moment a collinear set does not resist yields a constant of zero rather than
    the infinity the formula would divide out to, which keeps both the declared
    shape and the thermodynamic sums finite.
    """
    from pyrai3md.utilities.units import (
        AMU_TO_KG, BOHR_TO_METER, PLANCK_CONSTANT)
    quantum = PLANCK_CONSTANT / (8.0 * np.pi ** 2)
    moments = np.maximum(
        np.asarray(inertia_amu_bohr2, dtype=np.float64) * AMU_TO_KG
        * BOHR_TO_METER ** 2, 0.0)
    with np.errstate(divide='ignore'):
        hertz = np.where(moments > 0.0, quantum / moments, 0.0)
    return np.sort(hertz / 1.0e9)[::-1]
