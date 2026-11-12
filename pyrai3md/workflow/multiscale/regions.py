"""
======================================================================
Pure geometric region operations for PyRAI3MD composite workflows.

Provides atom selection, gradient/NAC scatter-add, region validation
and the per-center electronic projection helpers (energy / NAC / SOC /
dipole) used by the composite aggregation layer and the Calculator.

Design rules:
- This module never imports Calculator, composite or state.
- All functions are pure: no side effects, no state mutation.
- Coordinates follow the existing Angstrom convention.
- Projection helpers keep per-center data in the center's LOCAL frame
  (energy/dipole per local state, SOC per local pair); only the atom
  axis is scattered to the global system (gradient, NAC). The coupling
  direction rules (NAC antisymmetry, SOC Hermitian conjugate) are
  applied from an explicit per-pair sign so the stored pair order stays
  authoritative (docs 5.2).

Sep 07 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from ...utilities.lattice import continuous_view


def validate_region(region_indices, natom):
    """Validate that region indices are legal.

    Parameters
    ----------
    region_indices : list of int
        Atom indices (0-based) selecting a sub-region.
    natom : int
        Total number of atoms in the global system.

    Raises
    ------
    ValueError
        If any index is out of range [0, natom) or duplicates exist.
    """
    if not region_indices:
        return
    for idx in region_indices:
        if idx < 0 or idx >= natom:
            raise ValueError(
                "Region index %d out of range [0, %d)." % (idx, natom))
    if len(region_indices) != len(set(region_indices)):
        raise ValueError(
            "Duplicate atom indices in region: %s" % list(region_indices))


def materialize_region(region_indices, atoms, coordinates, image=None,
                       lattice=None):
    """Slice local atoms and coordinates from the global structure.

    When region_indices is empty or None the full system is returned
    (as copies, not views).

    A periodic run passes `image` and `lattice`, and the slice is then cut
    out of the continuous view rather than out of the main image. That is
    not a cosmetic choice: a center whose atoms straddle a box face is a
    broken cluster in the wrapped picture - measured on the shipped water
    box, an O-H bond of a molecule torn by the face reads 9.24 Angstrom
    raw and 0.965 Angstrom across the boundary - and a molecular backend
    asked to evaluate the first number is being handed two fragments
    pretending to be one molecule.

    The translation is per-atom and constant during one evaluation, so the
    result needs no gradient transform: `project_gradient` scatters local
    row j back to global atom index i unchanged, because a shift by an
    integer lattice vector has the identity as its Jacobian.

    Parameters
    ----------
    region_indices : list of int or None
        0-based atom indices for the sub-region.  Empty/None means
        the full system.
    atoms : list of str
        Global element symbols.
    coordinates : array-like
        Global coordinates (natom, 3), in the run's unit (Angstrom).
    image : array-like, optional
        Global per-atom crossing counts, (natom, 3). Given together with
        `lattice`, the region is cut from the continuous view.
    lattice : array-like, optional
        The cell of the same instant, (3, 3), same length unit as the
        coordinates.

    Returns
    -------
    tuple(list, ndarray)
        (local_atoms, local_coordinates) where local_coordinates is
        a fresh float64 ndarray copy.
    """
    atoms = list(atoms)
    coordinates = np.asarray(coordinates, dtype=np.float64)

    if not region_indices:
        return atoms, coordinates.copy()

    if image is not None or lattice is not None:
        coordinates = continuous_view(coordinates, image, lattice)

    indices = list(region_indices)
    local_atoms = [atoms[i] for i in indices]
    local_coordinates = coordinates[indices].copy()
    return local_atoms, local_coordinates


def project_gradient(local_gradient, region_indices, natom):
    """Scatter-add a local gradient into a full-system gradient.

    The global gradient has shape (nstate, natom, 3) and is
    initialised to zero.  For each local atom j mapping to global
    atom i:  G[:, i, :] += g[:, j, :].

    Parameters
    ----------
    local_gradient : array-like
        Local gradient.  Accepted shapes:
        - (natom_local, 3)       -- single-state, promoted to (1, n, 3)
        - (nstate, natom_local, 3)
    region_indices : list of int
        0-based mapping from local index j to global index i.
        Empty means the full system (region_indices = range(natom)).
    natom : int
        Number of global atoms.

    Returns
    -------
    ndarray
        Full-system gradient of shape (nstate, natom, 3), freshly
        allocated.

    Raises
    ------
    ValueError
        If the local gradient atom count does not match the index
        count, or the shape is unrecognised.
    """
    grad = np.asarray(local_gradient, dtype=np.float64)

    # Promote (natom_local, 3) -> (1, natom_local, 3)
    if grad.ndim == 2:
        grad = grad[np.newaxis, :, :]

    if grad.ndim != 3 or grad.shape[2] != 3:
        raise ValueError(
            "Local gradient must have shape (nstate, natom_local, 3) "
            "or (natom_local, 3), got %s." % (grad.shape,))

    nstate, nlocal, _ = grad.shape

    if not region_indices:
        # Full system: indices are identity
        if nlocal != natom:
            raise ValueError(
                "Full-region gradient has %d atoms but system has %d."
                % (nlocal, natom))
        return grad.copy()

    indices = list(region_indices)
    if nlocal != len(indices):
        raise ValueError(
            "Local gradient has %d atoms but region has %d indices."
            % (nlocal, len(indices)))

    global_grad = np.zeros((nstate, natom, 3), dtype=np.float64)
    for j, i in enumerate(indices):
        global_grad[:, i, :] += grad[:, j, :]
    return global_grad


def slice_field(field_value, region_indices):
    """Slice a per-atom field by region indices.

    Parameters
    ----------
    field_value : list or ndarray
        Per-atom data (natom, ...) or flat list of length natom.
    region_indices : list of int
        0-based atom indices.

    Returns
    -------
    list or ndarray
        Sliced data for the sub-region.
    """
    if not region_indices:
        return field_value
    if isinstance(field_value, np.ndarray):
        return field_value[region_indices].copy()
    if isinstance(field_value, list):
        return [field_value[i] for i in region_indices]
    return field_value


# ----------------------------------------------------------------------
# Per-center electronic projection (docs 3.2, 8; refactor phase R4c)
#
# Each center's high-level provider returns electronic data in the
# center's LOCAL frame. These pure helpers normalize that data and,
# where an atom axis exists, scatter it to the global system. They are
# the projection counterparts of project_gradient: energy/SOC/dipole
# carry no atom axis (validated local copies), NAC does (scatter + the
# antisymmetry sign rule). None of them sum across centers -- that is
# composite's ONIOM job -- and none write to any state.
# ----------------------------------------------------------------------

def project_energy(local_energy):
    """Normalize one center's local-state energy array to (S_c,).

    Energy carries no atom or pair axis, so its projection is a
    validated 1D copy kept in the center's LOCAL state framing (docs
    3.2). It is never scatter-added across atoms nor summed across
    centers here; the ONIOM U(R;s) combination is composite's job and
    per-center storage keeps this array as-is.

    Parameters
    ----------
    local_energy : array-like
        Center energy, a scalar or an (S_c,) array.

    Returns
    -------
    ndarray
        1D float64 array of shape (S_c,), freshly allocated.

    Raises
    ------
    ValueError
        If the energy is empty.
    """
    arr = np.asarray(local_energy, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Center energy must have at least one state.")
    return arr.copy()


def project_nac(local_nac, region_indices, natom, pair_signs=None):
    """Scatter one center's NAC to the global frame with the sign rule.

    The non-adiabatic coupling is a per-atom vector field over the
    center's coupling pairs, local shape (P_nac, A_c, 3). Projection
    scatter-adds the atom axis to the full system (P_nac, natom, 3),
    exactly like project_gradient (the pair axis plays the role of the
    leading row axis), then applies the NAC antisymmetry rule (docs
    5.2): a pair held in reversed direction (b, a) relative to the
    canonical (a, b), a < b, satisfies D[b, a] = -D[a, b], so passing
    -1 for that pair negates its rows and expresses every pair in the
    canonical order. Atoms outside the region get zero coupling.

    Parameters
    ----------
    local_nac : array-like
        Local NAC, (P_nac, A_c, 3) or (A_c, 3) for a single pair.
    region_indices : list of int
        0-based local->global atom map; empty means the full system.
    natom : int
        Number of global atoms.
    pair_signs : sequence of {+1, -1} or None
        Per-pair direction sign; -1 negates that pair (a reversed
        direction). None keeps every pair as given (all +1).

    Returns
    -------
    ndarray
        Global NAC of shape (P_nac, natom, 3), freshly allocated.

    Raises
    ------
    ValueError
        If the shape is unrecognised, the atom count mismatches the
        region, or pair_signs disagrees with the pair count.
    """
    nac = np.asarray(local_nac, dtype=np.float64)
    # Promote a single pair (A_c, 3) -> (1, A_c, 3).
    if nac.ndim == 2:
        nac = nac[np.newaxis, :, :]
    if nac.ndim != 3 or nac.shape[2] != 3:
        raise ValueError(
            "Local NAC must have shape (P_nac, A_c, 3) or (A_c, 3), "
            "got %s." % (nac.shape,))
    if pair_signs is not None:
        signs = np.asarray(pair_signs, dtype=np.float64).reshape(-1)
        if signs.size != nac.shape[0]:
            raise ValueError(
                "pair_signs has %d entries but NAC has %d pairs."
                % (signs.size, nac.shape[0]))
        nac = nac * signs[:, np.newaxis, np.newaxis]
    # The pair axis is the leading row axis, so the gradient scatter
    # (nstate, nlocal, 3) -> (nstate, natom, 3) applies unchanged.
    return project_gradient(nac, region_indices, natom)


def project_soc(local_soc, pair_signs=None):
    """Normalize one center's SOC scalars with the Hermitian rule.

    Spin-orbit coupling is a scalar per coupling pair in the backend's
    adapted pair layout, shape (P_soc,) (docs 3.2); it carries no atom
    axis, so projection is a validated 1D copy kept in the center's
    LOCAL pair framing. The Hermitian rule (docs 5.2) is
    SOC[b, a] = conj(SOC[a, b]); for the real-valued first version the
    conjugate is the identity, so a reversed pair keeps its magnitude
    -- unlike NAC there is NO automatic sign flip. pair_signs is an
    explicit hook so a reversed-pair phase can be applied when a future
    complex/antisymmetric SOC backend needs it (docs 5.2: the rule is
    recorded, the stored pair order stays authoritative).

    Parameters
    ----------
    local_soc : array-like
        Local SOC magnitudes, a scalar or a (P_soc,) array.
    pair_signs : sequence of {+1, -1} or None
        Optional per-pair phase for a reversed direction; None keeps
        every pair as given.

    Returns
    -------
    ndarray
        1D float64 array of shape (P_soc,), freshly allocated.

    Raises
    ------
    ValueError
        If pair_signs disagrees with the pair count.
    """
    soc = np.asarray(local_soc, dtype=np.float64).reshape(-1)
    if pair_signs is not None:
        signs = np.asarray(pair_signs, dtype=np.float64).reshape(-1)
        if signs.size != soc.size:
            raise ValueError(
                "pair_signs has %d entries but SOC has %d pairs."
                % (signs.size, soc.size))
        soc = soc * signs
    return soc.copy()


def project_dipole(local_dipole):
    """Normalize one center's dipole to a per-row (N, 3) array.

    The dipole is a 3-vector per local state, shape (S_c, 3); the
    transition dipole is a 3-vector per transition and shares the same
    (N, 3) layout (config Properties). Neither carries an atom axis, so
    projection is a validated copy kept in the center's LOCAL framing;
    dipoles are never scatter-added across atoms nor summed across
    centers (docs 8: per-center properties stay per center).

    Parameters
    ----------
    local_dipole : array-like
        Dipole, (3,) for a single row or (N, 3).

    Returns
    -------
    ndarray
        float64 array of shape (N, 3), freshly allocated.

    Raises
    ------
    ValueError
        If the last axis is not a 3-vector or the shape is
        unrecognised.
    """
    dip = np.asarray(local_dipole, dtype=np.float64)
    # Promote a single 3-vector (3,) -> (1, 3).
    if dip.ndim == 1:
        dip = dip[np.newaxis, :]
    if dip.ndim != 2 or dip.shape[1] != 3:
        raise ValueError(
            "Local dipole must have shape (N, 3) or (3,), got %s."
            % (dip.shape,))
    return dip.copy()
