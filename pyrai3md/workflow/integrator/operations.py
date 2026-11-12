"""
======================================================================
This module provides the independent physics operations inserted
into the NAMD timestep data flow (dev/methods/MD_DEVELOPMENT.md):

    phase_correction      NAC/SOC temporal phase consistency
    apply_constrain       constraint entry (geometric constraints,
                          frozen atoms)
    velocity_initialize   initial-velocity control, once per trajectory
    velocity_correction   global translation (and, on request,
                          rotation) removal
    apply_ensemble        ensemble control (thermostat/barostat)
    verify_recorded       the constraint question of a recorded frame

The periodic wrap is not on this list any more. `apply_pbc` belongs to no
one timestep: a geometry search whose step carries the atoms with the box
crosses a cell face on every iteration and needs the same wrap, so it and
the crossing ledger it writes live in `data/geometry.py`, beside the rule
that decides which spelling of the cell is authoritative.

The kinetic energy is computed and stored by the official
PyRAIState.kinetic_energy() method.

None of them belongs to the Verlet time integration itself.
`apply_constrain` delegates to the constraint runtime under
workflow/constraints/, which owns the hard geometric constraints for every
profile the framework integrates, and it adds no energy: an additive
potential belongs to the evaluation the Calculator just finished, which is
why `verify_recorded` measures a recorded frame rather than correcting one.
Every implementation must read and write the state exclusively through
the official PyRAIState get/update interfaces (read-only properties,
state.result, state.get_history_step and state.update_system /
state.set_result), never through direct pmd attribute mutation.

apply_ensemble delegates to integrator/ensemble.py (the thermostat /
barostat operators split around the force kicks); it takes a stage
argument ('pre' / 'post') so the main loop can wrap the extended
ensemble around verlet_i / verlet_ii.

Aug 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np
from pyrai3md.utilities import geometric

from ...data.geometry import continuous_coordinates, lattice_matrix
from ...utilities.lattice import normalize_pbc
from ...utilities.units import ANGSTROM_TO_BOHR, KELVIN_TO_HARTREE
from ..constraints import controller as _constraint_controller
from ..constraints import projection as _projection
from ..constraints import runtime as constraint_runtime
from ..records.diagnostics import (CONSTRAINT_GEOMETRY, RecordedStepError,
                                   report as diagnostics_report)
from .ensemble import apply_ensemble as _apply_ensemble_impl

# Inner products at or below this magnitude carry no dynamics:
# the coupling vectors themselves are numerical noise there (an
# analytic coupling crossing zero, or an ab initio gap closure),
# so the eigenvector phase is arbitrary and its sign must not be
# followed. Far below any physical NAC in Hartree/Bohr or 1/Bohr.
PHASE_OVERLAP_EPS = 1e-24

# An axis whose moment of inertia is at or below this magnitude carries
# no rotation to remove (a linear rotor has two such axes, and one atom
# has all three), so its component of the angular velocity is dropped
# rather than divided out. In mass x Bohr^2, four orders below the
# smallest real molecule.
ZERO_INERTIA_BOHR2 = 1e-8


def phase_correction(state):
    """Correct the NAC/SOC phase against the previous timestep.

    Restores the temporal phase consistency of the nonadiabatic
    coupling vectors (ported from the legacy PyRAI2MD
    trajectory.phase_correction): ab initio eigenvector phases may
    flip between consecutive steps, and every pair whose inner
    product with the previous step is negative gets its sign
    flipped back. Runs after the calculator and before the history
    snapshot (the md loop order), so both the recorded step and the
    surface hopping integrator see the corrected vectors. Not part
    of the Verlet integration.

    The correction is gated by the &md phasecheck keyword (1 = on,
    the default) and needs the previous step in the history; the
    first step and steps without NAC records pass through
    unchanged. Overlaps whose magnitude is physically negligible
    (a coupling crossing zero, e.g. at a symmetry point, leaves
    only noise in the vectors) keep the pair untouched instead of
    following the arbitrary phase.

    Parameters
    ----------
    state : PyRAIState
        State with the latest electronic properties.

    Returns
    -------
    PyRAIState
        Phase-corrected state (the NAC record of pmd.results is
        rewritten through state.set_result when any pair flipped).
    """
    if state is None or getattr(state, 'pmd', None) is None:
        return state
    dynamics = getattr(state.pmd, 'dynamics', None)
    try:
        phasecheck = int(getattr(dynamics, 'phasecheck', 1))
    except (TypeError, ValueError):
        phasecheck = 1
    if phasecheck != 1:
        return state

    result = state.result
    if result is None:
        return state
    nac = result.get('nac')
    if nac is None:
        return state

    # The previous step's NAC lives in the results section of the
    # last history snapshot (the history dumps the full pmd state).
    prev_snap = state.get_history_step(-1, copy=False)
    prev_results = (prev_snap or {}).get('results') or {}
    nac_prev = prev_results.get('nac')
    if nac_prev is None:
        return state

    nac = np.asarray(nac, dtype=np.float64)
    nac_prev = np.asarray(nac_prev, dtype=np.float64)
    if nac.ndim < 2 or nac.shape != nac_prev.shape:
        return state

    # Per-pair sign from the overlap with the previous step. When
    # the coupling magnitude is physically negligible the phase is
    # arbitrary (the vectors are rounding noise, possibly exactly
    # antiparallel), so such pairs keep their current sign.
    overlap = np.sum(nac_prev * nac, axis=tuple(range(1, nac.ndim)))
    sign = np.sign(overlap)
    insignificant = np.abs(overlap) <= PHASE_OVERLAP_EPS
    sign = np.where(insignificant | (sign == 0.0), 1.0, sign)
    if np.all(sign == 1.0):
        return state

    corrected = sign.reshape((-1,) + (1,) * (nac.ndim - 1)) * nac
    updated = result.to_dict()
    updated['nac'] = corrected
    state.set_result(updated)
    return state


def apply_constrain(state):
    """Apply constraints to the geometry and the velocities.

    The one entry through which the timestep asks the constraint question:
    frozen components return to their anchor, strict bond, angle and torsion
    conditions are solved against what is left, and the velocities lose the
    frozen directions (plus the bond tangents for a record that asks for
    RATTLE). It consolidates the legacy `apply_potential()` and `freeze_atom()`
    calls, and only the geometry half of that pair: an additive potential is
    not applied here. Not part of the Verlet integration.

    Parameters
    ----------
    state : PyRAIState
        State after the electronic structure calculation.

    Returns
    -------
    PyRAIState
        The state, with the geometry the constraints declare. A run with no
        constraint block gets the same object back having done no work, which is
        the default-closed contract of dev/methods/CONSTRAINT_DEVELOPMENT.md
        section 1.1.
    """
    # Additive energies and gradients are NOT applied here: they belong to the
    # evaluation the Calculator just finished (section 7), and adding them twice
    # from two places is how a run reports a force nothing produced. What this
    # entry owns is the geometry: frozen components return to their anchor, bond
    # constraints are solved against what is left, and the velocities lose the
    # frozen components - plus, for a record declaring
    # `velocity_constraint: rattle`, the bond tangent projection.
    return constraint_runtime.constrain_state(state)


def _frozen_mask(state, natom):
    """(natom, 3) mask of frozen Cartesian components, or None.

    The initial-velocity control must not put motion into a component the
    run declared frozen, and must not count it when it rescales toward a
    kinetic energy. A run with no constraint block gets None, which every
    caller reads as "nothing is held".
    """
    plan = state.constraint_plan()
    if plan is None or plan.is_empty:
        return None
    return _projection.build_fix_mask(plan, natom)


def _is_periodic(state):
    """Whether this run is periodic on any axis.

    The one reading of `&system pbc` the whole framework shares
    (utilities/lattice.normalize_pbc), so the rotation half of a velocity
    reset and the wrap cannot disagree about what box the run lives in.
    """
    system = getattr(getattr(state, 'pmd', None), 'system', None)
    return any(normalize_pbc(getattr(system, 'pbc', None)))


def _rigid_reset_due(dynamics, step):
    """Whether this step is one the run removes the rigid rotation at.

    `resetstep == 0` means the first step only; a positive cadence fires on
    the absolute step number, so a segment that resumes mid-run removes the
    rotation at the same steps the continuous run would have.
    """
    if int(getattr(dynamics, 'reset', 0) or 0) != 1:
        return False
    cadence = int(getattr(dynamics, 'resetstep', 0) or 0)
    current = int(step or 1)
    if cadence == 0:
        return current <= 1
    return current % cadence == 0


def _maxwell_boltzmann_velocities(state, dynamics, mask):
    """Draw one Cartesian velocity per atom from the run's temperature.

    Each component is a Gaussian of width sqrt(k_B T / m) in atomic units,
    which is the classical Maxwell-Boltzmann distribution of the three
    translational degrees of freedom of one atom: its mean kinetic energy is
    k_B T / 2 per component, so a drawn ensemble reads the declared
    temperature through the same formula the thermostat reports it with.

    This is not the vibrational sampling of `initcond: wigner/boltzmann`,
    which needs the normal modes of a frequency record: `randvelo` is the
    temperature-only draw, the one that works when no frequency file exists
    (a periodic force-field box, for instance).

    The draw consumes the global NumPy stream, so `control.gl_seed` fixes it
    and a resumed run replays it from the serialized generator state.
    """
    masses = np.asarray(state.mass_au, dtype=np.float64)
    if masses.size == 0:
        raise ValueError('randvelo needs per-atom masses and this run has none')
    temperature = float(getattr(dynamics, 'temperature', 0.0) or 0.0)
    if temperature <= 0.0:
        raise ValueError(
            'randvelo draws its velocities from &md temperature, which is %s; '
            'a non-positive temperature has no distribution to sample'
            % temperature)
    kinetic_target = temperature * KELVIN_TO_HARTREE
    sigma = np.sqrt(kinetic_target / masses)
    drawn = np.random.normal(size=(masses.shape[0], 3)) * sigma[:, None]
    if mask is not None:
        drawn[mask] = 0.0
    return drawn


def velocity_initialize(state):
    """Apply the declared initial-velocity control to a trajectory once.

    Three spellings ask for one thing, a value of the first step's kinetic
    energy, and they are refused if a run declares more than one
    (schema/validate.py::validate_dynamics):

    excess  raise the kinetic energy the state carries by this many Hartree
    scale   multiply the kinetic energy the state carries by this factor
    target  bring the kinetic energy to this value in Hartree

    Each is one multiplicative factor on the velocity vector, because
    kinetic energy is quadratic in velocity: K' = f^2 K. The measured
    quantity is `state.kinetic_energy()`, the same value the record's Ekin
    column and the run log print, so a run cannot state a kinetic energy in
    one place and read it from another.

    `randvelo` replaces the velocities with a Maxwell-Boltzmann draw at the
    run's temperature first; a scaling keyword on top of it is a separate
    declaration and applies to what was drawn.

    The whole routine runs once per trajectory, and only for a process that
    starts at step 0: a restarted segment must not rescale the kinetic energy
    its predecessor already settled into (the same absolute-step discipline
    that keeps the frames of a segmented run equal to a continuous one).
    After the draw or a rescale the center-of-mass drift is removed, so a
    drawn ensemble does not fly as a whole.

    Parameters
    ----------
    state : PyRAIState
        State carrying the initial velocities and the &md section.

    Returns
    -------
    PyRAIState
        The state with its initial velocities settled. Unchanged, and at no
        cost, when the run declares none of these keywords.

    Raises
    ------
    ValueError
        When `excess` or `target` is asked of a state whose kinetic energy is
        zero (there is no factor to scale), or when `randvelo` is asked
        without masses or with a non-positive temperature.
    """
    if state is None or getattr(state, 'pmd', None) is None:
        return state
    dynamics = getattr(state.pmd, 'dynamics', None)
    if dynamics is None:
        return state
    randvelo = int(getattr(dynamics, 'randvelo', 0) or 0)
    excess = float(getattr(dynamics, 'excess', 0.0) or 0.0)
    scale = float(getattr(dynamics, 'scale', 1.0) or 1.0)
    target = float(getattr(dynamics, 'target', 0.0) or 0.0)
    if not randvelo and excess == 0.0 and scale == 1.0 and target == 0.0:
        return state

    velocities = np.asarray(state.velocities, dtype=np.float64)
    carried = velocities.ndim == 2 and velocities.size > 0
    mask = _frozen_mask(state, velocities.shape[0] if carried
                        else len(state.atoms))

    if randvelo:
        # The draw replaces the velocities rather than scaling them, so it is
        # the one spelling that works on an input carrying none. That is also
        # what makes it the temperature-only start: a run with no velocities
        # block, no frequency record and no ensemble file can still begin at a
        # declared temperature.
        velocities = _maxwell_boltzmann_velocities(state, dynamics, mask)
        state.update_system({'velocities': velocities.tolist()})
        carried = True
    elif not carried:
        raise ValueError(
            'excess/scale/target act on the velocities the state carries and '
            'this run has none. Write a velocities block, draw them with '
            'randvelo, or sample them with initcond: wigner/boltzmann')

    factor = 1.0
    if excess:
        kinetic = float(state.kinetic_energy())
        if kinetic <= 0.0:
            raise ValueError(
                'excess adds to a kinetic energy this run does not carry: '
                'K0 = %r Ha. Write initial velocities, or draw them with '
                'randvelo' % kinetic)
        factor = ((kinetic + excess) / kinetic) ** 0.5
    elif scale != 1.0:
        factor = scale ** 0.5
    elif target:
        kinetic = float(state.kinetic_energy())
        if kinetic <= 0.0:
            raise ValueError(
                'target rescales a kinetic energy this run does not carry: '
                'K0 = %r Ha. Write initial velocities, or draw them with '
                'randvelo' % kinetic)
        factor = (target / kinetic) ** 0.5

    if factor != 1.0:
        velocities = velocities * factor
        if mask is not None:
            velocities[mask] = 0.0
        state.update_system({'velocities': velocities.tolist()})
    if randvelo or factor != 1.0:
        state = velocity_correction(state)
        if mask is not None:
            # Removing the drift subtracts one vector from every atom, and
            # that hands motion back to the components the input froze. The
            # constraint subsystem zeroes them whenever velocities are
            # produced (projection.zero_fixed_velocities), so the initial
            # condition is held to the same rule as any later step.
            held = _projection.zero_fixed_velocities(
                np.asarray(state.velocities, dtype=np.float64), mask)
            state.update_system({'velocities': held.tolist()})
    return state


def _rigid_rotation_velocity(coords_bohr, velocities, masses, mobile):
    """The rigid-rotation part of a velocity array, in the lab frame.

    Ported from PyRAI2MD's `Dynamics/reset_velocity.py` (Jingbai Li,
    Sep 5 2021). The inertia tensor is diagonalized directly: the legacy
    code took `numpy.linalg.eig` and then a Kabsch comparison to pick each
    axis' sign, and the sign choice cannot matter here - the angular
    velocity and the lever arm both flip with the axis, so their cross
    product does not. A zero-inertia axis (a linear rotor has two) has no
    rotation about it, so its component of omega is dropped instead of
    divided out.

    Parameters
    ----------
    coords_bohr : np.ndarray
        (natom, 3) coordinates relative to the center of mass, Bohr.
    velocities : np.ndarray
        (natom, 3) velocities, Bohr atomic units.
    masses : np.ndarray
        (natom,) masses, atomic units.
    mobile : np.ndarray
        (natom, 3) boolean mask of components that may carry motion.

    Returns
    -------
    np.ndarray
        (natom, 3) the velocity of the rigid rotation, ready to subtract.
    """
    tensor = np.einsum('n,ni,nj->ij', masses, coords_bohr, coords_bohr)
    trace = np.einsum('n,n->', masses, np.sum(coords_bohr ** 2, axis=1))
    tensor = trace * np.eye(3) - tensor
    inertia, axes = np.linalg.eigh(tensor)
    # Components of the position and velocity in the principal frame.
    local = coords_bohr @ axes
    local_v = velocities @ axes
    # L = sum m r x v in the principal frame, where the tensor is diagonal,
    # so omega is one division per axis. An axis whose moment of inertia
    # sits at or below the guard carries no rotation to remove.
    angular_momentum = np.einsum('n,ni->i', masses,
                                 np.cross(local, local_v))
    rotating = inertia > ZERO_INERTIA_BOHR2
    omega = np.where(rotating,
                     angular_momentum / np.where(rotating, inertia, 1.0),
                     0.0)
    rotation_local = np.cross(omega, local)
    rotation = rotation_local @ axes.T
    rotation[~mobile] = 0.0
    return rotation


def velocity_correction(state, step=None):
    """Remove the global center-of-mass translation from the velocities.

    Ported from the translation half of the legacy
    Dynamics.reset_velocity.reset_velo (the "flying ice cube"
    suppression): the mass-weighted center-of-mass velocity is
    subtracted from every atom and the result is rescaled to conserve
    the kinetic energy, so the temperature is untouched.

    The rigid rotation is the second half of the legacy reset and it is
    opt-in: `&md reset 1` asks for it, `&md resetstep` says how often
    (0 means the first step only), and the count is on absolute step
    numbers so a segmented run removes it at the same steps a continuous
    one does. A periodic system never takes this half, whatever `reset`
    says: the cell-internal flow of a box is its own angular momentum and
    removing it would rewrite the momentum balance the barostat reads,
    which is why the translation half above is the only one that runs
    when `any(pbc)`. Not part of the Verlet integration.

    Frozen-atom velocities are re-zeroed by apply_constrain (the
    constraint runtime), not here; the rotation correction is applied
    only to the components the run lets move, so a reset cannot hand
    motion back to an atom the input froze.

    Parameters
    ----------
    state : PyRAIState
        State with completed velocities.
    step : int or None
        Absolute step number this correction belongs to. None is read as
        the first step, which is what the initial-velocity control sees.

    Returns
    -------
    PyRAIState
        State with the center-of-mass drift removed, and with the rigid
        rotation removed too when the run asks for it (unchanged when the
        mass/velocity data is unavailable).
    """
    if state is None or getattr(state, 'pmd', None) is None:
        return state
    velocities = np.asarray(state.velocities, dtype=np.float64)
    if velocities.ndim == 2 and velocities.shape[0] < 2:
        # One atom has no internal degree of freedom, so subtracting the
        # mass-weighted center takes away all of its motion and the
        # kinetic rescale has nothing left to scale. A single-particle
        # run is that particle's whole dynamics - it is how the Tully and
        # Landau-Zener pseudo-atom of the surface-hopping benchmarks is
        # written - so the drift removal is skipped rather than applied.
        return state
    mass_au = state.mass_au
    if (mass_au is None or velocities.size == 0            or velocities.ndim != 2
            or mass_au.shape[0] != velocities.shape[0]):
        return state
    total_mass = float(np.sum(mass_au))
    if total_mass <= 0.0:
        return state

    kinetic_before = 0.5 * float(np.sum(mass_au[:, None] * velocities ** 2))
    # D-04: one mass-weighted center implementation, shared with the walls.
    # Reading it through `geometric.group_center` over every atom gives the same
    # number the inline sum gave, because the helper multiplies by the raw
    # masses and divides once, in the same row order.
    vcom, _members, _weights = geometric.group_center(
        velocities, np.arange(velocities.shape[0]), masses=mass_au)
    corrected = velocities - vcom
    kinetic_after = 0.5 * float(np.sum(mass_au[:, None] * corrected ** 2))
    if kinetic_after > 0.0 and kinetic_before > 0.0:
        corrected = corrected * np.sqrt(kinetic_before / kinetic_after)

    dynamics = getattr(state.pmd, 'dynamics', None)
    if (_rigid_reset_due(dynamics, step) and not _is_periodic(state)
            and velocities.shape[0] > 2):
        # Two atoms define an axis with no rotation about anything but
        # itself, so the rotation half starts at three atoms.
        coords = np.asarray(state.coordinates, dtype=np.float64)
        com, _members, _weights = geometric.group_center(
            coords, np.arange(coords.shape[0]), masses=mass_au)
        mobile = np.ones(velocities.shape, dtype=bool)
        mask = _frozen_mask(state, velocities.shape[0])
        if mask is not None:
            mobile = ~mask
        rotation = _rigid_rotation_velocity(
            (coords - com) * ANGSTROM_TO_BOHR, corrected, mass_au, mobile)
        candidate = corrected - rotation
        kinetic_rotated = 0.5 * float(
            np.sum(mass_au[:, None] * candidate ** 2))
        if (kinetic_rotated > 0.0 and kinetic_before > 0.0
                and np.any(rotation)) :
            corrected = candidate * np.sqrt(kinetic_before / kinetic_rotated)

    state.update_system({'velocities': corrected.tolist()})
    return state


def apply_ensemble(state, stage='post'):
    """Apply the ensemble control (thermostat / barostat) to one stage.

    Thin delegation to integrator/ensemble.py (docs sections 2 and 6):
    the NVE / NVT / NPT operators (Nose-Hoover chain, CSVR, Berendsen,
    Parrinello-Rahman, MTTK) live there and are split around the force
    kicks. The main loop calls this twice per step, stage='pre' before
    verlet_i and stage='post' after velocity_correction. Not part of
    the Verlet integration.

    Parameters
    ----------
    state : PyRAIState
        State with completed velocities for this stage.
    stage : str
        'pre' (before verlet_i) or 'post' (after velocity_correction).

    Returns
    -------
    PyRAIState
        The ensemble-controlled state.
    """
    return _apply_ensemble_impl(state, stage=stage)


def verify_recorded(state, frame):
    """Ask the geometry a run recorded the question its projection answered.

    The step is projected onto its constraints twice (`md.py` phases
    `apply_constrain` and `constrain_after_cell`), so by construction the
    geometry that enters the history frame satisfies the plan. Nothing here
    re-solves it: the frame is measured against the same targets, which is what
    catches the case a projection cannot see - a write-back that overwrote the
    projected geometry afterwards, or a cell rescale that moved a frozen atom
    and was not followed by the second projection.

    A periodic frame is recorded in the main image, so the question has to be
    put to the continuous view rebuilt from its own crossing counts. A bond
    that straddles a box face reads 9.2 Angstrom wrapped and 0.97 unfolded;
    reporting the first number as a violated constraint would be describing the
    reader's choice of picture, not the run's geometry.

    The verdict travels through `workflow/records/diagnostics.py`, which owns
    who has to stop; this operation changes nothing about the dynamics. The
    other two questions a recorded frame answers about itself (its kinetic
    energy and its crossing counts) are pure readings and live there, as
    `diagnostics.check_recorded_step`.

    Parameters
    ----------
    state : PyRAIState
        State that recorded the frame.
    frame : dict
        One history frame, read as written: its `system` section carries the
        Angstrom coordinates, the cell, the periodicity and the crossing
        counts of that step.
    """
    plan = state.constraint_plan() if hasattr(state, 'constraint_plan') \
        else None
    if plan is None or plan.is_empty:
        return
    bound = getattr(state, 'constraint_state', None)
    if bound is None:
        # Nothing has been bound, so there is no target to measure against.
        # The plan's `reference: initial` values are captured before the first
        # step, and a call earlier than that has no answer to give.
        return
    system = frame.get('system') if isinstance(frame, dict) else None
    if not isinstance(system, dict) or not system.get('coordinates'):
        return
    array = np.asarray(system['coordinates'], dtype=np.float64)
    if any(int(flag) for flag in (system.get('pbc') or [])):
        if not system.get('image'):
            # Without crossing counts the continuous view cannot be rebuilt,
            # and a cross-face bond would read as a violation nobody
            # declared. The counts themselves are watched by the
            # `image_integrity` sentinel.
            return
        if lattice_matrix(system) is None:
            # A periodic frame with no cell to unfold against is a frame this
            # check cannot answer, not a frame that violates anything.
            return
        array = continuous_coordinates(array, system)
    array = array * ANGSTROM_TO_BOHR
    controller = _constraint_controller.SystemConstraintController(plan)
    ratio, deviation, note = controller.measure_recorded(array, bound)
    if ratio <= 1.0 or not note:
        return
    message = ('step %s records a geometry that does not satisfy what the run '
               'declared: %s' % (frame.get('step'), note))
    if diagnostics_report(state, CONSTRAINT_GEOMETRY, message,
                          worst=deviation):
        raise RecordedStepError(message)
