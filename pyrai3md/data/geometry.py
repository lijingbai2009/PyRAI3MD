"""
======================================================================
Structural operations every workflow shares: the periodic wrap and the
crossing ledger, the rule that decides which spelling of the cell is
authoritative, the continuous view rebuilt from that ledger, and the
Kabsch scalar.

These functions read and write a PyRAIState - or the plain dict a
recorded frame carries - and hold no arithmetic of their own. The
arithmetic lives in ``utilities/lattice.py`` and ``utilities/rmsd.py``,
which are leaf modules that must not import this layer (the rule stated
in ``utilities/lattice.py``). What lives here is the state-level API, so
a dynamics loop and a geometry search ask the same question in the same
words: `apply_pbc` was written for the timestep and is what a
variable-cell relaxation needs between its own steps, and a cell that
only one of the two wrapped is a cell the two disagree about.

Ownership of `System.image`. The field carries the crossing counts of the
coordinates stored beside it: `wrapped + image @ cell` is the continuous
position, which is what a cross-frame difference, a restart and a region
cut-out all read (`utilities/lattice.py::unwrap_coordinates`). An engine that
receives a wrapped position beside a count belonging to an earlier geometry
relocates that atom by a whole box, so the field has exactly one writer,
`record_wrapped_geometry`, and what that writer guarantees is the pairing: the
coordinates and their counts leave in one commit. The counts it takes are
absolute for those coordinates, and the two callers that reach it know them in
two different ways - a wrap adds its own crossings to the ledger the state
carries, a constraint fold already reports absolutes - which is documented on
the writer, together with the measurement that settles why the two may not be
added to each other.

Layer rule. This module must not import ``workflow`` at module scope:
``workflow/constraints/state_adapter.py`` imports ``data.constraint`` at
module scope, so the reverse edge would be a cycle. A function here that
one day needs constraint information imports it inside the function body,
the way ``data/state.py`` already does.

Oct 13 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from ..utilities import rmsd as _rmsd
from ..utilities.lattice import (cart_to_frac, cell6_to_matrix,
                                 continuous_view, frac_to_cart, normalize_pbc,
                                 wrap_fractional_with_image)

__all__ = ['apply_pbc', 'continuous_coordinates', 'kabsch', 'lattice_matrix',
           'record_wrapped_geometry']


def _field(source, name):
    """Read one field from a system model or from a recorded frame's dict.

    The two spellings of "the system" are the live Pydantic model and the
    dict a history frame carries, and every consumer of the functions below
    holds one or the other. Reading through this helper is what keeps a
    structural rule from being written twice, once per spelling.
    """
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def lattice_matrix(source):
    """The (3, 3) row-vector lattice in Angstrom, or None.

    One rule for which spelling of the cell is authoritative: the
    `lattice_vector` rows when they hold a non-zero (3, 3), else the six
    classic parameters through `cell6_to_matrix`, else None. A deformation
    aware barostat keeps the two in sync, so either is current; what must
    not happen is two callers preferring them in opposite orders, which is
    what four copies of this rule used to risk.

    Parameters
    ----------
    source : object or dict
        A system model, or the dict a recorded frame carries, exposing
        `lattice_vector` and `cell`.

    Returns
    -------
    numpy.ndarray or None
        Lattice rows in Angstrom, or None when the source declares no cell.
        Callers that cannot proceed without one raise their own refusal,
        because what a missing cell means is a question of the caller's
        contract and not of this rule.
    """
    rows = np.asarray(_field(source, 'lattice_vector') or [],
                      dtype=np.float64)
    if rows.shape == (3, 3) and np.any(rows):
        return rows
    cell6 = np.asarray(_field(source, 'cell') or [],
                       dtype=np.float64).reshape(-1)
    if cell6.size == 6 and np.any(cell6):
        return cell6_to_matrix(cell6)
    return None


def record_wrapped_geometry(state, coordinates, image=None, also=None):
    """Commit a wrapped geometry together with the crossing counts it implies.

    The single writer of `System.image`. What makes it single is the pairing,
    not the arithmetic: the coordinates and the counts that describe them go out
    in one `update_system` call, because an engine that receives a wrapped
    position beside a count belonging to an earlier geometry relocates that atom
    by a whole box, and LAMMPS answers that with `Bond atom missing in image
    check` (src/domain.cpp) instead of a number.

    `image` is the **absolute** count for these coordinates: the integer cell
    offset such that `wrapped + image @ cell` is the continuous position the
    caller folded. It is not an increment, and this function does not add
    anything to the ledger the state carries. Two callers reach it with two
    different ways of knowing that absolute count, and both are right:

    - `apply_pbc` wraps the stored coordinates, so the counts its wrap reports
      are relative to the frame the ledger already describes, and it adds them
      to that ledger before handing the sum over;
    - the constraint solve folds a geometry it unfolded along the bond graph, so
      `images.fold_back` reports counts that are already absolute for the frame
      it returns (`workflow/constraints/images.py`, whose docstring warns that
      adding the expansion's own offsets on top charges every crossed face
      twice).

    Adding the ledger to the second caller's counts was tried and is wrong by
    measurement: the ledger is per atom (each atom accrues the faces *it*
    crossed) while a fold's counts are uniform per connected component, so the
    sum shifts the atoms of one molecule against each other and a bond then
    spans more than one cell. A periodic LAMMPS run with a strict bond constraint
    died on it at evaluation 197 of 600. What the two conventions do not share is
    the ledger's absolute level, which is the open item recorded in
    `dev/methods/PBC_DEVELOPMENT.md`, not something a write may average.

    Parameters
    ----------
    state : PyRAIState
        State whose geometry is being replaced.
    coordinates : array-like
        Wrapped coordinates in Angstrom, shape (natom, 3).
    image : array-like, optional
        Absolute integer (natom, 3) counts for `coordinates`. None leaves the
        ledger untouched, which is what a non-periodic commit passes: inventing
        a zero count there would turn an empty ledger into a populated one and
        make a non-periodic system look periodic to every reader that asks
        whether the field holds anything.
    also : dict, optional
        Further system fields that belong to this same commit, such as a
        velocity projection of the geometry being written. They land in one
        `update_system` call rather than a second one behind it.

    Returns
    -------
    PyRAIState
        The state, updated.
    """
    updates = {'coordinates': np.asarray(coordinates,
                                        dtype=np.float64).tolist()}
    if image is not None:
        updates['image'] = [[int(value) for value in row]
                            for row in np.asarray(image, dtype=np.int64)]
    if also:
        updates.update(also)
    state.update_system(updates)
    return state


def apply_pbc(state):
    """Wrap the coordinates into the periodic cell.

    Periodic boundary coordinate wrapping (ported from the legacy
    Molecule.pbc_helper.apply_pbc): the cell is built from
    lattice_vector (3x3) or, failing that, from the 6 classic cell
    parameters; the Cartesian coordinates are turned into fractional
    coordinates s = R B^-1, reduced modulo 1 on every periodic axis
    (pbc != 0) and turned back into Cartesian coordinates, so the
    structure sent to the backend always sits inside the (possibly
    deformed) cell. It runs before the calculator so electronic
    structure properties are evaluated on wrapped coordinates. Not part
    of the Verlet integration, and not part of any one workflow: a
    geometry search whose step carries the atoms with the box crosses a
    face on every iteration and needs the same wrap a timestep does.

    A no-op when the state is incomplete or no axis is periodic. When
    at least one axis is periodic, an invalid cell (missing, all-zero,
    non-finite or singular) raises ValueError instead of silently
    skipping the wrap: coordinates that escape an intended cell would
    otherwise propagate undetected into the backend and the history
    (dev/methods/PBC_DEVELOPMENT.md sections 3.4 and 5.1). The crossing
    count each atom accrues is committed alongside the wrapped
    coordinates as the system's image field, which is what lets the
    continuous coordinates be rebuilt (dev/methods/PBC_DEVELOPMENT.md
    stage C).

    Parameters
    ----------
    state : PyRAIState
        State with propagated coordinates.

    Returns
    -------
    PyRAIState
        Wrapped state (unchanged when PBC does not apply).

    Raises
    ------
    ValueError
        When PBC is enabled but the cell is missing or invalid.
    """
    if state is None or getattr(state, 'pmd', None) is None:
        return state
    system = getattr(state.pmd, 'system', None)
    if system is None:
        return state

    flags = normalize_pbc(getattr(system, 'pbc', None))
    if not any(flags):
        return state

    coordinates = np.asarray(state.coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.size == 0:
        return state

    # Build the cell (Angstrom) through the one rule that decides which
    # spelling is authoritative. A deformation-aware barostat keeps both in
    # sync (verlet._scale_cell), so either representation is current.
    lattice = lattice_matrix(system)
    if lattice is None:
        raise ValueError(
            "PBC is enabled (pbc=%s) but no cell is defined: both "
            "lattice_vector and the 6 cell parameters are missing "
            "or zero" % flags)

    if not np.all(np.isfinite(lattice)):
        raise ValueError(
            "PBC is enabled (pbc=%s) but the cell contains non-finite "
            "entries: %r" % (flags, lattice.tolist()))
    try:
        frac = cart_to_frac(coordinates, lattice)
    except ValueError as exc:
        raise ValueError(
            "PBC is enabled (pbc=%s) but the cell is singular and "
            "coordinates cannot be wrapped" % flags) from exc

    wrapped_frac, crossed = wrap_fractional_with_image(frac, flags)
    wrapped = frac_to_cart(wrapped_frac, lattice)
    # The count accrues in fractional units, so it stays meaningful when
    # the cell changes shape; rebuilding the continuous position under a
    # deformed cell is the affine-transport question PBC stage D owns,
    # not something this accumulation may silently decide.
    #
    # The wrap reports the crossings of *this* wrap, against the frame the
    # ledger already describes, so the absolute count for the coordinates being
    # committed is the sum of the two. A ledger that is missing, short or
    # malformed restarts from this step rather than failing: a checkpoint written
    # before the image channel existed, or a run whose atom list changed, simply
    # has no crossings to add.
    previous = _field(system, 'image')
    try:
        base = np.asarray(previous, dtype=np.int64)
    except (TypeError, ValueError):
        base = np.empty((0, 0), dtype=np.int64)
    absolute = crossed if base.shape != crossed.shape else base + crossed
    return record_wrapped_geometry(state, wrapped, absolute)


def continuous_coordinates(coordinates, source):
    """The unwrapped reading of a stored geometry, from its own ledger.

    A periodic run stores main-image coordinates plus the crossing count,
    so the continuous path is a derived reading and not a second truth:
    it is what lets a cross-frame difference, a restart and a region
    cut-out agree on where an atom has been. Reading it takes three fields
    off whichever spelling of the system the caller holds, and doing that
    in each caller is what let the guards drift apart.

    Parameters
    ----------
    coordinates : array-like or None
        Wrapped coordinates, shape (natom, 3), in the caller's length unit.
        None is returned as None, which is what a frame without a geometry
        hands over.
    source : object or dict
        A system model, or the dict a recorded frame or an evaluation
        snapshot carries, exposing `pbc`, `lattice_vector` / `cell` and
        `image`.

    Returns
    -------
    numpy.ndarray or None
        Continuous coordinates, or the input when the run is not periodic,
        declares no cell, or carries no ledger yet - a frame that predates
        the image channel cannot be unwrapped and is returned as it is
        rather than being reported as a violation nobody declared.
    """
    if coordinates is None:
        return None
    array = np.asarray(coordinates, dtype=np.float64)
    if not any(int(flag) for flag in (_field(source, 'pbc') or [])):
        return array
    lattice = lattice_matrix(source)
    if lattice is None:
        return array
    return continuous_view(array, _field(source, 'image'), lattice)


def kabsch(p, q, masses=None):
    """RMSD between two coordinate sets after the optimal superposition.

    Kabsch algorithm (the legacy Dynamics.reset_velocity helper, corrected):
    both coordinate sets first have their centre removed - the mass-weighted
    centre when `masses` is given, the plain centroid when it is not - the
    optimal rotation is then taken from the SVD of the *weighted*
    cross-covariance with the determinant sign fix that keeps a reflected
    solution out, and the deviation of the superposed blocks is returned as a
    scalar. The rotation itself is computed but not returned, and no gradient
    is returned either; the previous docstring described the rotation step this
    function did not perform (it rotated without centring, so the number it
    reported was not the RMSD after the optimal rotation). That over-claim is
    defect D-10 of dev/methods/CONSTRAINT_DEVELOPMENT.md section 18 and is
    closed here. Kept for the second-phase RMSD collective variable
    (dev/methods/ENSEMBLE_DEVELOPMENT.md section 4); the velocity correction
    of the dynamics loop does not use it, because a periodic system keeps its
    rotation.

    Single implementation. The alignment arithmetic lives in
    ``pyrai3md/utilities/rmsd.py`` and nowhere else: this function is a scalar
    wrapper over ``rmsd.rmsd_squared`` (its square root), and the uncentred,
    unweighted SVD that used to be duplicated beside it is deleted. The
    deduplication can only run in that direction, because ``utilities`` is a
    leaf layer that must not import ``data`` or ``workflow`` (the rule stated
    in ``utilities/rmsd.py`` and ``utilities/lattice.py``, and the import
    discipline the shadow build was checked with, section 17), while this
    module imports from ``utilities`` - so the higher layer consumes the
    lower one, never the reverse. The RMSD history potential of
    ``workflow/constraints/metadynamics.py`` reads the same implementation
    through ``rmsd.rmsd_squared``, which is why there is no second RMSD path.

    Degeneracy guard. A non-unique optimal rotation leaves the *value* single-
    valued (it is an expression in the singular values alone) but not the
    per-atom gradient, so the refusal lives with the gradient: the shared
    implementation raises ``rmsd.RmsdError`` when a free rotation direction
    still carries reference spread and a gradient was requested. This wrapper
    asks for no gradient, so it cannot bypass that guard and cannot hand a
    silent derivative to a caller.

    Parameters
    ----------
    p : np.ndarray
        Mobile coordinates, shape (natom, 3).
    q : np.ndarray
        Reference coordinates, shape (natom, 3), same atom order.
    masses : np.ndarray, optional
        Per-atom positive masses, shape (natom,). Given, both centres are
        mass-weighted and the deviation is a mass-weighted RMSD; omitted
        (the default, which is what every existing caller passes), the plain
        centroid and the uniform mean over atoms are used, as before.

    Returns
    -------
    float
        The RMSD after the optimal superposition, in the length unit of the
        inputs (the core uses Bohr). Only the scalar: no rotation, no
        gradient.

    Raises
    ------
    rmsd.RmsdError
        A block that is not (natom, 3) or holds a non-finite number, two
        blocks of different atom count, fewer than two atoms (one atom has no
        orientation, so its aligned deviation is a translation and not a
        variable), or masses that are not one positive value per atom.
    """
    weights = None if masses is None else np.asarray(masses, dtype=np.float64)
    return float(np.sqrt(_rmsd.rmsd_squared(p, q, weights=weights)))
