"""
======================================================================
This module provides the velocity-Verlet integrator of the NAMD
workflow, written in the reversible Trotter/Liouville split form so the
extended-ensemble operators (thermostat / barostat, integrator/
ensemble.py) can wrap around the force kicks at the mathematically
correct positions (dev/methods/ENSEMBLE_DEVELOPMENT.md sections 2 and 5):

    verlet_i   (Vn, Gn)       -> Rn+1, Vn+dt/2   half kick + drift
    verlet_ii  (Vn+dt/2, Gn+1)-> Vn+1            half kick

One MD step therefore reads

    ensemble_pre -> verlet_i -> [pbc, force eval] -> verlet_ii
                 -> ensemble_post

with a single force evaluation per step: verlet_i consumes the previous
step's assembled active-combination gradient Gn (read from the latest
history frame) and verlet_ii consumes the current step's gradient Gn+1.

    V(t+dt/2) = V(t)     - 0.5 Gn/m   dt        (half kick)
    R(t+dt)   = R(t)     + V(t+dt/2) dt         (drift, cell-aware)
    V(t+dt)   = V(t+dt/2)- 0.5 Gn+1/m dt        (half kick)

Merging the two half kicks of consecutive steps reproduces the classic
PyRAI2MD two-stage form exactly, so the NVE trajectory is unchanged;
the split only exposes the half-step velocity where the ensemble
operators attach.

The drift is cell-aware: when a barostat supplies a cell velocity
v_eps (state.pmd.system.barostat_velocity, a 3x3 matrix), the
coordinates and the cell are scaled by the matrix exponential
exp(v_eps dt) (the combined iL_eps1 + iL_1 MTTK drift, dropping only
the O((v_eps dt)^2) sinh correction), and the cell/lattice/volume are
updated in place. With no barostat v_eps is zero and the drift is the
plain R += V dt.

The integrators only perform the time integration of nuclear
coordinates and velocities. Freeze, constraints, PBC, gradient descent
and external potentials are NOT part of the integrator; they are
independent workflow operations. The thermostat/barostat are NOT part
of the integrator either: they only enter the drift through v_eps.

Unit convention (framework-wide internal units, converted at the
state boundary with utilities.units constants, no magic numbers):
    coordinates  Angstrom      (pmd.system)
    velocities   Bohr / au
    gradients    Hartree / Bohr
    masses       amu (atomic-unit masses via state.mass_au)
    timestep     au
    cell/lattice Angstrom

All reads go through the PyRAIState read-only properties and all
writes go through state.update_system() (the sole legal
modification path). The previous-step data comes from the official
history API (state.get_history_step(-1)) instead of a dedicated
grad1 storage; each frame carries the assembled active-combination
gradient of its step (nuclear_sample.active_gradient), so the
integrator reads one integrated gradient and never re-derives it
from a recorded state index (docs R6, r6a). Both integrators skip
the first timestep internally when no previous step is recorded, so
the workflow loop needs no branching.

Sep 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import math

import numpy as np

from ...utilities.units import BOHR_TO_ANGSTROM
from ...utilities.lattice import cell6_to_matrix, matrix_to_cell6
from ..records.diagnostics import (
    CELL_GATE, CELL_IMPOSSIBLE, CELL_NOTE, MAX_CUMULATIVE_LOG_STRAIN,
    MAX_STEP_AXIS_RATIO, MAX_STEP_FRAME_ROTATION_RAD,
    WARN_CUMULATIVE_LOG_STRAIN, CellDivergenceError, check_finite, note,
    report as diagnostics_report)

try:
    from scipy.linalg import expm as _scipy_expm
except Exception:  # pragma: no cover - scipy is expected to be present
    _scipy_expm = None


def _matrix_expm(matrix):
    """Matrix exponential exp(M) of a small (3, 3) matrix.

    Uses scipy.linalg.expm when available and falls back to a
    scaling-and-squaring Taylor series otherwise, so the cell-aware
    drift never hard-depends on scipy at import time.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix, shape (n, n).

    Returns
    -------
    np.ndarray
        exp(matrix), shape (n, n).
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if _scipy_expm is not None:
        return np.asarray(_scipy_expm(matrix), dtype=np.float64)
    n = matrix.shape[0]
    norm = float(np.max(np.sum(np.abs(matrix), axis=1))) if n else 0.0
    squarings = 0
    if norm > 0.5:
        squarings = int(np.ceil(np.log2(norm))) + 1
        matrix = matrix / (2.0 ** squarings)
    result = np.eye(n)
    term = np.eye(n)
    for k in range(1, 25):
        term = term @ matrix / k
        result = result + term
    for _ in range(squarings):
        result = result @ result
    return result


def _frame_active_gradient(snapshot):
    """Read the assembled active-combination gradient of a frame.

    The history frame carries the previous step's integrated active
    gradient (nuclear_sample.active_gradient), captured when the step
    was recorded, so the integrator reads one assembled gradient and
    never re-derives it from a recorded state index (docs R6, r6a).

    Parameters
    ----------
    snapshot : dict or None
        History frame produced by state.update_history().

    Returns
    -------
    np.ndarray or None
        The frame's active-combination gradient (natom, 3), or None
        when the frame carries no gradient.
    """
    if not snapshot:
        return None
    sample = snapshot.get('nuclear_sample') or {}
    gradient = sample.get('active_gradient')
    if gradient is None:
        return None
    return np.asarray(gradient, dtype=np.float64)


def _cell_velocity(state):
    """Read the barostat cell velocity v_eps (3, 3) or None.

    Returns None when no barostat is driving the cell this step (an
    empty or all-zero barostat_velocity), so the drift stays the plain
    R += V dt of NVE/NVT.

    Parameters
    ----------
    state : PyRAIState
        State possibly carrying pmd.system.barostat_velocity.

    Returns
    -------
    np.ndarray or None
        The (3, 3) cell velocity, or None when inert.
    """
    system = getattr(getattr(state, 'pmd', None), 'system', None)
    raw = getattr(system, 'barostat_velocity', None) if system else None
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape != (3, 3) or not np.any(arr):
        return None
    return arr


def compute_cell(cell6):
    """Build the (3, 3) lattice matrix from the 6 classic cell parameters.

    Thin forward to utilities.lattice.cell6_to_matrix (the shared
    cell-math module); kept under its historical name because the
    ensemble/integrator code and the verification tests import it from
    here. Input [a, b, c, alpha, beta, gamma], lengths in Angstrom and
    angles in degrees; the returned rows are the cell vectors in the
    standard orientation (a along x, b in the xy plane).

    Parameters
    ----------
    cell6 : array-like
        The 6 cell parameters [a, b, c, alpha, beta, gamma].

    Returns
    -------
    np.ndarray
        Lattice matrix, shape (3, 3), rows are cell vectors (Angstrom).
    """
    return cell6_to_matrix(cell6)


def _lattice_to_cell6(lattice):
    """Convert a (3, 3) lattice matrix to the 6 cell parameters.

    Thin forward to utilities.lattice.matrix_to_cell6 (the shared
    cell-math module). The lattice rows are the cell vectors
    (Angstrom); the returned parameters are
    [a, b, c, alpha, beta, gamma] with the angles in degrees.

    Parameters
    ----------
    lattice : array-like
        Lattice matrix, shape (3, 3), rows are cell vectors (Angstrom).

    Returns
    -------
    list of float
        [a, b, c, alpha, beta, gamma].
    """
    return matrix_to_cell6(lattice)


def verlet_i(state, dt=None):
    """Velocity-Verlet half kick + (cell-aware) drift: (Vn, Gn) -> Rn+1.

    Applies the first velocity half kick with the previous step's
    assembled active-combination gradient, then drifts the coordinates
    one full timestep (scaling coordinates and cell by exp(v_eps dt)
    when a barostat is active):

        V(t+dt/2) = V(t) - 0.5 G(t)/m dt
        R(t+dt)   = R(t) exp(v_eps dt) + (V(t+dt/2) dt) exp(v_eps dt/2)

    The first timestep carries no history snapshot and is skipped
    internally (the initial structure is kept for the initialization
    calculation); the workflow loop needs no branching. Missing
    velocities are treated as zero; the state is returned unchanged
    when the gradient or the masses are unavailable.

    Parameters
    ----------
    state : PyRAIState
        State carrying coordinates (Angstrom), velocities (Bohr/au),
        masses (amu), the optional cell/lattice and the history
        snapshots (previous gradient).
    dt : float or None
        Timestep in atomic units; None reads it through the
        state.timestep property (pmd.dynamics.size).

    Returns
    -------
    PyRAIState
        State with updated coordinates (Angstrom) and the half-kicked
        velocities (Bohr/au); the cell/lattice/volume are updated when
        a barostat drives the cell.
    """
    if state is None or state.pmd is None:
        return state
    timestep = state.timestep if dt is None else float(dt)

    # Read-only access: the frame is the integrator's input, never its
    # output, and the deep copy of a 60-atom frame costs more than the
    # arithmetic it feeds (dev/methods/MD_DEVELOPMENT.md, phase table).
    previous = state.get_history_step(-1, copy=False)
    if previous is None:
        # First timestep: no previous step recorded yet.
        return state
    gradient = _frame_active_gradient(previous)

    coordinates = np.asarray(state.coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.size == 0:
        return state
    if gradient is None or gradient.shape != coordinates.shape:
        return state

    mass_au = state.mass_au
    if mass_au is None:
        return state

    velocities = np.asarray(state.velocities, dtype=np.float64)
    if velocities.size == 0:
        velocities = np.zeros_like(coordinates)
    if velocities.shape != coordinates.shape:
        return state

    inv_mass = 1.0 / mass_au[:, None]

    # 1. First velocity half kick with the previous gradient G(t).
    half_velocities = velocities - 0.5 * gradient * inv_mass * timestep

    # 2. Cell-aware drift (atomic units: displacement in Bohr,
    #    converted back to Angstrom at the state boundary).
    updates = {}
    cell_velocity = _cell_velocity(state)
    displacement = half_velocities * timestep * BOHR_TO_ANGSTROM
    if cell_velocity is None:
        new_coordinates = coordinates + displacement
    else:
        scale_full = _matrix_expm(cell_velocity * timestep)
        scale_half = _matrix_expm(cell_velocity * timestep * 0.5)
        new_coordinates = coordinates @ scale_full + displacement @ scale_half
        # The cell commit may re-triangularize the deformed lattice and
        # rotate the frame; it then carries the rotated coordinates /
        # velocities, which take precedence over the unrotated arrays.
        updates.update(
            _scale_cell(state, scale_full,
                        coordinates=new_coordinates,
                        velocities=half_velocities))

    if 'coordinates' not in updates:
        updates['coordinates'] = new_coordinates.tolist()
    if 'velocities' not in updates:
        updates['velocities'] = half_velocities.tolist()
    # A drift that produced a non-finite lattice or strain is arithmetic,
    # not a judgement: it stops the run in every mode, and stops it here,
    # with the step and the drive named, instead of two fields later in a
    # schema validator that only knows the victim's field name.
    check_finite(state, updates)
    state.update_system(updates)
    return state


# The cell's own extended degrees of freedom, in the order they enter a
# commit: strain (the accumulated log-deformation), its momentum and the
# velocity derived from it. All three are (3, 3) tensors OF THE
# SIMULATION FRAME, so a re-triangularization rotates them; a tensor
# left in the frame the cell no longer uses makes the run depend on the
# gauge (see _scale_cell).
_ROTATED_CELL_DOF = ('barostat_strain', 'barostat_momentum',
                     'barostat_velocity')

# The bounds a cell commit is measured against, the errors it raises and
# the policy of who has to hear about them all live in
# pyrai3md/workflow/records/diagnostics.py: a commit no lattice can hold stops the
# run for everyone, while a bound that says "this box has left its own
# start" is a judgement, and judgements interrupt only a run that was
# started as a test (control.debug). Re-exported here because the
# integrator is where a reader of this file looks for them, and because
# workflow.integrator.__init__ re-exports CellDivergenceError from this
# module.
def _frame_rotation_angle(rotation):
    """Angle of an orthogonal matrix, arccos((tr - 1) / 2)."""
    trace = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(trace))


def _divergence_verdict(state, old_lattice, new_lattice, rotation):
    """Measure one cell commit against the bounds, in two families.

    Returns
    -------
    tuple of (list, list, float or None)
        (impossible reasons, judgement reasons, worst accumulated strain).
        The split is what makes `control.debug` mean something: the first
        list is arithmetic (a lattice that cannot be committed), the
        second is an opinion about how far the box has travelled.
    """
    impossible = []
    judgement = []
    det = float(np.linalg.det(new_lattice))
    if not np.isfinite(det) or det <= 0.0:
        impossible.append('determinant %.6e (the box collapsed)' % det)
    if rotation is not None:
        angle = _frame_rotation_angle(rotation)
        if angle > MAX_STEP_FRAME_ROTATION_RAD:
            impossible.append('frame turned %.4f rad in one step' % angle)
    # The accumulated strain IS the run-start-relative measure: epsilon is
    # defined against the cell the run began with (Tr(epsilon) = ln(V/V0)),
    # it is carried through the frame transport, and it does not care how
    # much history survived. The first cut of this bound compared against
    # the oldest retained history frame, which the 07:13 run proved useless
    # -- with a trimmed history the reference sits a few steps back, so a
    # box creeping 5e57-fold over two thousand steps never looked like 2x
    # to it. Reading epsilon is both cheaper and correct.
    worst = None
    system = getattr(state.pmd, 'system', None)
    strain = getattr(system, 'barostat_strain', None) if system else None
    if strain:
        epsilon = np.asarray(strain, dtype=np.float64)
        if epsilon.shape == (3, 3):
            # Every component, not just the diagonal: in the triangular
            # gauge a shape collapse shows up in an off-diagonal epsilon
            # first, and a bound the shear components walk past unnoticed
            # is not a bound.
            labels = (('xx', 'xy', 'xz'),
                      ('yx', 'yy', 'yz'),
                      ('zx', 'zy', 'zz'))
            measured = [(float(np.trace(epsilon)),
                         'volume is e^%.2f = %.1f of the box the run '
                         'started with')]
            for row in range(3):
                for col in range(3):
                    measured.append((
                        float(epsilon[row, col]),
                        'strain %s is e^%.2f = %.1f of the box the run '
                        'started with' % (labels[row][col], epsilon[row, col],
                                          np.exp(abs(epsilon[row, col])))))
            for value, text in measured:
                if abs(value) > MAX_CUMULATIVE_LOG_STRAIN:
                    # The volume text is a template: it is the only one of
                    # the measurements whose sentence does not name a
                    # component, so it gets its numbers here.
                    if text.startswith('volume'):
                        text = text % (value, np.exp(abs(value)))
                    judgement.append(text)
                if worst is None or abs(value) > abs(worst):
                    worst = value
    old_axes = np.linalg.norm(old_lattice, axis=1)
    new_axes = np.linalg.norm(new_lattice, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = new_axes / old_axes
    for axis, ratio in enumerate(ratios):
        if not np.isfinite(ratio):
            continue
        if ratio > MAX_STEP_AXIS_RATIO or ratio < 1.0 / MAX_STEP_AXIS_RATIO:
            impossible.append('axis %d changed by x%.3f in one step'
                              % (axis + 1, ratio))
    return impossible, judgement, worst


def _divergence_report(state, old_lattice, new_lattice, rotation):
    """Stop or report a cell commit that left the bounds; else return.

    Parameters
    ----------
    state : PyRAIState
        State carrying the runtime diagnostics of the current step.
    old_lattice : np.ndarray
        Lattice matrix before this commit (Angstrom).
    new_lattice : np.ndarray
        Lattice matrix the commit is about to write.
    rotation : np.ndarray or None
        The orthogonal factor the re-triangularization applied.

    Raises
    ------
    CellDivergenceError
        When the commit is arithmetically impossible, or when it is only
        past the accumulated-strain bound and this run was started with
        control.debug. A warn-only run keeps the same message in its log.
    """
    impossible, judgement, worst = _divergence_verdict(
        state, old_lattice, new_lattice, rotation)
    if not impossible and not judgement:
        # Below the bound that means anything to the run, but not below
        # the bound worth saying once: a box that has left the thermal
        # band is on its way somewhere, and a reader of a warn-only run
        # should learn that early instead of reconstructing it from
        # md.ensemble. note() speaks the first time and then counts.
        if worst is not None and abs(worst) > WARN_CUMULATIVE_LOG_STRAIN:
            note(state, CELL_NOTE,
                 'cell strain is e^%.2f = %.2f of the box the run started '
                 'with (the bound is e^%.2f = %.2f)'
                 % (worst, np.exp(abs(worst)), MAX_CUMULATIVE_LOG_STRAIN,
                    np.exp(MAX_CUMULATIVE_LOG_STRAIN)), worst=worst)
        return
    record = getattr(getattr(state, 'runtime', None), 'ensemble', None)
    record = record if isinstance(record, dict) else {}

    def pressure(key):
        value = record.get(key)
        if value is None:
            return 'n/a'
        array = np.asarray(value, dtype=np.float64)
        return '%.3e' % float(np.max(np.abs(array)))

    strain = getattr(getattr(state.pmd, 'system', None),
                     'barostat_strain', None)
    norm = ('%.3e' % float(np.linalg.norm(np.asarray(strain,
                                                    dtype=np.float64)))
            if strain is not None and len(strain) else '0')
    reasons = impossible + judgement
    message = ('cell diverged at step %d: %s; volume %.4f -> %.4f A^3, '
               '|barostat_strain| %s, this step read |p_kin| %s Ha/Bohr^3, '
               '|p_vir| %s, |p_int| %s (see md.ensemble for the series)'
               % (getattr(getattr(state, 'runtime', None), 'current_step', -1),
                  '; '.join(reasons), abs(float(np.linalg.det(old_lattice))),
                  abs(float(np.linalg.det(new_lattice))), norm,
                  pressure('p_kin'), pressure('p_vir'), pressure('p_int')))
    # The two families get two different check ids, so a log shows which
    # bound was crossed without re-reading the sentence.
    check = CELL_IMPOSSIBLE if impossible else CELL_GATE
    if diagnostics_report(state, check, message, fatal=bool(impossible),
                          worst=worst):
        raise CellDivergenceError(message)


def _retriangularize(lattice):
    """Lower-triangular representative of the same cell metric.

    An anisotropic deformation B @ S of a lower-triangular cell
    produces upper-triangular components, breaking the framework
    convention (a along x, b in the xy plane) that every consumer
    relies on (cell6 round-trips, backend data files, PBC wrapping).
    The Cholesky factor L of the Gram matrix G = B B^T is lower
    triangular with L L^T = G, i.e. the SAME cell shape, and
    O = B^-1 L is orthogonal (B @ O = L): rotating the whole system
    by O restores the convention as a pure frame change -- fractional
    coordinates, volume and every interparticle distance are exactly
    preserved.

    Parameters
    ----------
    lattice : np.ndarray
        Deformed (3, 3) lattice matrix (rows are cell vectors).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray or None)
        (lower-triangular lattice, rotation O) or (lattice, None) when
        the lattice is already lower triangular within tolerance or
        the factorization fails (a degenerate cell is left to the
        System validators).
    """
    B = np.asarray(lattice, dtype=np.float64)
    upper = max(abs(B[0, 1]), abs(B[0, 2]), abs(B[1, 2]))
    scale = max(abs(B[0, 0]), abs(B[1, 1]), abs(B[2, 2]), 1e-30)
    if upper <= 1e-12 * scale:
        return B, None
    try:
        L = np.linalg.cholesky(B @ B.T)
        O = np.linalg.solve(B, L)
    except np.linalg.LinAlgError:
        return B, None
    return L, O


def _scale_cell(state, scale_full, coordinates=None, velocities=None):
    """Scale the periodic cell by the drift matrix and refresh volume.

    Preserves fractional coordinates under a homogeneous deformation:
    B_new = B @ S, volume = |det(B_new)|, and the classic 6-parameter
    cell is recomputed from the deformed lattice so every cell
    representation stays consistent for the backend.

    When the deformation leaves upper-triangular components (the
    anisotropic variable-cell case), the lattice is re-triangularized
    (_retriangularize) and the orthogonal frame rotation is applied to
    the provided ALREADY-DEFORMED coordinates/velocities, so the
    committed state satisfies the framework lower-triangular
    convention with the physical configuration unchanged.

    The cell's own extended degrees of freedom (barostat strain,
    momentum and velocity) are tensors of the same frame and travel with
    it, as a second-rank tensor under the row-vector convention:
    T' = O^T T O. They must, because the increment they receive next is
    computed in the new frame: the barostat kick is assembled from the
    kinetic pressure of the ROTATED velocities and a stress evaluated
    AFTER this commit, and the strain accumulates the cell velocity that
    drives the NEXT drift. Leaving them behind makes the run depend on
    the gauge -- the per-step rotation is small (O(shear increment),
    measured 5.7e-5 rad) but it does not cancel, and a 400-step aniso-
    tropic run turns the representative frame by more than a degree.

    Tensors of OTHER systems need no transport here, and that is the
    distinction worth keeping: the previous frame's gradient and stress
    were both captured after that frame's own commit, and nothing
    rotates the frame between then and the half kicks that read them, so
    they are already in the current basis.

    Parameters
    ----------
    state : PyRAIState
        State carrying the current lattice_vector / cell.
    scale_full : np.ndarray
        The (3, 3) cell deformation matrix exp(v_eps dt).
    coordinates : np.ndarray or None
        Deformed coordinates of this commit (rotated with the frame
        when a re-triangularization happens).
    velocities : np.ndarray or None
        Velocities of this commit (rotated with the frame).

    Returns
    -------
    dict
        System updates for lattice_vector, cell and volume (empty when
        no lattice is defined), plus rotated coordinates/velocities and
        rotated barostat tensors when a frame rotation was applied.
    """
    system = getattr(state.pmd, 'system', None)
    lattice = getattr(system, 'lattice_vector', None) if system else None
    lattice = np.asarray(lattice, dtype=np.float64) if lattice is not None \
        else None
    if lattice is None or lattice.shape != (3, 3) or not np.any(lattice):
        return {}
    new_lattice, rotation = _retriangularize(lattice @ scale_full)
    _divergence_report(state, lattice, new_lattice, rotation)
    volume = float(abs(np.linalg.det(new_lattice)))
    updates = {
        'lattice_vector': new_lattice.tolist(),
        'cell': _lattice_to_cell6(new_lattice),
        'volume': volume,
    }
    if rotation is not None:
        if coordinates is not None:
            updates['coordinates'] = (
                np.asarray(coordinates, dtype=np.float64)
                @ rotation).tolist()
        if velocities is not None:
            updates['velocities'] = (
                np.asarray(velocities, dtype=np.float64)
                @ rotation).tolist()
        for field in _ROTATED_CELL_DOF:
            carried = getattr(system, field, None) if system else None
            if carried is None:
                continue
            tensor = np.asarray(carried, dtype=np.float64)
            if tensor.shape != (3, 3):
                continue
            updates[field] = (rotation.T @ tensor @ rotation).tolist()
    return updates


def verlet_ii(state, dt=None):
    """Velocity-Verlet second half kick: (Vn+dt/2, Gn+1) -> Vn+1.

    Completes the velocity update with the current step's assembled
    active-combination gradient (state.active_gradient()):

        V(t+dt) = V(t+dt/2) - 0.5 G(t+dt)/m dt

    Combined with the first half kick of verlet_i this reproduces the
    classic two-stage velocity-Verlet; the previous step's gradient was
    already consumed by verlet_i (and captured at its own active
    combination, which may differ after a surface hop).

    The first timestep carries no history snapshot and is skipped
    internally (the initial velocity is used as-is); the workflow loop
    needs no branching.

    Parameters
    ----------
    state : PyRAIState
        State carrying the half-kicked velocities (Bohr/au), masses
        (amu), the latest result (current gradient), the history
        snapshots and the current electronic state.
    dt : float or None
        Timestep in atomic units; None reads it through the
        state.timestep property (pmd.dynamics.size).

    Returns
    -------
    PyRAIState
        State with updated velocities (Bohr/au). Coordinates are
        untouched.
    """
    if state is None or state.pmd is None:
        return state
    timestep = state.timestep if dt is None else float(dt)

    current_gradient = state.active_gradient()
    if current_gradient is None:
        return state

    previous = state.get_history_step(-1, copy=False)
    if _frame_active_gradient(previous) is None:
        # First timestep, or a frame carrying no assembled gradient:
        # the matching verlet_i half kick did not run, so skip to keep
        # the two half kicks of the step paired.
        return state

    velocities = np.asarray(state.velocities, dtype=np.float64)
    if (velocities.size == 0
            or velocities.shape != current_gradient.shape):
        return state

    mass_au = state.mass_au
    if mass_au is None:
        return state

    inv_mass = 1.0 / mass_au[:, None]
    new_velocities = velocities - 0.5 * current_gradient * inv_mass * timestep

    state.update_system({'velocities': new_velocities.tolist()})
    return state
