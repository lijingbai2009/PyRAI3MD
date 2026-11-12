"""
======================================================================
This module tests the stability and frame consistency of the anisotropic
variable-cell barostats (Parrinello-Rahman, MTTK) by driving the real
split step loop with an analytic stress, with no engine involved.

It exists because a qualification run produced eight NPT failures that
were one physical divergence in two death modes: the resident LAMMPS
session died at some mid-run step, or the lattice reached NaN and the
System validator raised. Four further NPT examples passed while their
boxes deformed by tens of percent, and no judgement noticed. Reading
those trajectories showed the failing cases share one signature: the
cell accumulates an actual rotation (its axes turn by degrees, measured
from the per-frame Lattice= records) while the framework
re-triangularizes the lattice every step.

So the tests here separate what is geometry from what is physics:

- the volume-restoring response of each barostat at equilibrium, and
  the period of that response (which pins the barostat mass formula);
- whether an anisotropic barostat run accumulates frame rotation at all;
- whether a change of description -- the same physical cell and the same
  atoms written in a rotated Cartesian frame -- changes the run. It must
  not: the lattice is re-triangularized into the same representative
  either way, so the two runs have to stay identical step by step;
- whether the stored particle velocities are peculiar (thermal) or
  lab-frame ones, which decides whether a streaming-velocity correction
  is owed at all.

Everything is deterministic: the stress is a closed-form function of the
cell, the velocities are seeded once with no random number drawn, and
the initial state is exactly at the target pressure, so a failure names
a mechanism rather than a sample.

Sep 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import math
import unittest

import numpy as np

from pyrai3md.utilities.units import (
    AMU_TO_AU, ANGSTROM_TO_BOHR, FS_TO_AU_TIME, KELVIN_TO_HARTREE,
    bar_to_au_pressure)
from pyrai3md.data.geometry import apply_pbc
from pyrai3md.workflow.integrator import (
    CellDivergenceError, apply_ensemble, verlet_i, verlet_ii,
    velocity_correction)
from pyrai3md.workflow.integrator import ensemble as ensemble_mod
from pyrai3md.workflow.integrator.ensemble import (
    _barostat_mass, _volume_bohr3 as _ensemble_volume_bohr3)
from pyrai3md.workflow.integrator.verlet import (
    _matrix_expm, _retriangularize, _scale_cell)
from pyrai3md.qualification.verification.test_integrator import (
    CUBIC_VOLUME, _kinetic_temperature, _make_ensemble_state, _spring_grad)

# A six-atom box: enough atoms that the kinetic pressure tensor is not
# dominated by one particle, few enough that a few thousand steps cost
# milliseconds. The masses are equal so the pattern below is isotropic.
NATOM = 6
MASS_AMU = 39.948
DT = FS_TO_AU_TIME                  # 1 fs, the timestep the examples use
TEMPERATURE = 300.0

# An out-of-equilibrium SHAPE at the right volume: b2 tilted by 3 Angstrom
# along x keeps det(B) = 1000, so the box starts with zero volume drive
# and a large shear drive. That shear is what makes the cell drift
# non-triangular, so _retriangularize has something to rotate on every
# single step of the run.
TILTED_LATTICE = [[10.0, 0.0, 0.0], [3.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
BETA_BAR = 4.5e-5                   # bar^-1, the examples' compressibility
# The volume spring is set to 1/beta on purpose: that is the stiffness
# the barostat mass W = (V/beta)(tau_p/2pi)^2 was derived against, so the
# cell period comes out of the run as a check of the mass formula rather
# than of a fitted constant.
K_VOL_BAR = 1.0 / BETA_BAR
# Deviatoric stiffness (bar per unit of the dimensionless shape
# deviation). At the 3 Angstrom tilt above it yields an off-diagonal
# stress of about 150 bar, the order of magnitude the diverging
# trajectories showed.
K_DEV_BAR = 500.0
# Relative step for the central difference that turns `cell_potential` into a
# force: 1e-5 Bohr on a ~16 Bohr lattice, which differentiates a quadratic to
# ~1e-9 relative.
FD_STEP = 1.0e-5
VOLUME0 = CUBIC_VOLUME


def seeded_coordinates(natom):
    """Deterministic in-box positions (Angstrom) for a free-particle box.

    The atoms feel no force in this harness, so their positions matter
    only as the configuration the cell contains; a fixed stride puts
    every atom inside the 10 Angstrom box with no two coinciding.
    """
    return [[(index * 3.7) % 10.0, (index * 5.3) % 10.0,
             (index * 7.1) % 10.0] for index in range(natom)]


def thermal_velocities(natom=NATOM):
    """Seeded velocities (Bohr/au) with zero COM and an isotropic P_kin.

    Each atom gets one axial direction with an alternating sign, so the
    mass-weighted outer product sum is proportional to the identity and
    the initial kinetic pressure carries no deviatoric part. All the
    drive in the run then comes from the cell shape. The magnitude is set
    by the target temperature under the framework's own degree-of-
    freedom count, and natom is a parameter because MTTK's cell force is
    scaled by alpha = 1 + 3/N_df, which a six-atom box makes large and a
    sixty-atom box (the examples) makes small.
    """
    mass_au = MASS_AMU * AMU_TO_AU
    ndof = 3 * natom - ensemble_mod.N_COM_DOF
    ke_target = 0.5 * ndof * TEMPERATURE * KELVIN_TO_HARTREE
    speed = math.sqrt(2.0 * ke_target / (natom * mass_au))
    rows = []
    for index in range(natom):
        row = [0.0, 0.0, 0.0]
        row[index % 3] = speed if (index // 3) % 2 == 0 else -speed
        rows.append(row)
    return rows


def make_npt_state(barostat='parrinello-rahman', barostat_mode='anisotropic',
                   thermostat='nose-hoover', lattice=None, volume=None,
                   temperature=TEMPERATURE, pressure=1.0, velos=None,
                   coords=None, natom=NATOM):
    """Build a periodic NPT state with the seeded thermal sample."""
    lattice = TILTED_LATTICE if lattice is None else [list(r) for r in lattice]
    volume = CUBIC_VOLUME if volume is None else float(volume)
    coords = seeded_coordinates(natom) if coords is None \
        else [list(r) for r in coords]
    return _make_ensemble_state(
        mass=[MASS_AMU] * natom,
        coords=coords,
        velos=thermal_velocities(natom) if velos is None else velos,
        size=DT,
        ensemble='npt', thermostat=thermostat, thermostat_chain=3,
        barostat=barostat, barostat_mode=barostat_mode,
        temperature=temperature, pressure=pressure,
        thermostat_tau=100.0, barostat_tau=1000.0,
        compressibility=BETA_BAR,
        pbc=[1, 1, 1], lattice_vector=lattice, volume=volume)


def make_nvt_state(**kwargs):
    """The same box with no barostat in play (the baseline, T01)."""
    state = make_npt_state(barostat='off', **kwargs)
    state.pmd.dynamics.ensemble = 'nvt'
    return state


def _shape_deviation(lattice):
    """Traceless part of the cell metric, normalized to unit determinant.

    B^T B (the metric of the cell in its own fractional basis) is the
    quantity a change of Cartesian frame carries as a tensor: for a
    rotation Q, B -> B @ Q gives B^T B -> Q^T (B^T B) Q. Dividing by the
    determinant removes the volume, so what is left is pure shape, and
    subtracting the trace makes it deviatoric. A cell whose vectors are
    orthogonal and equal-length reads exactly zero here.
    """
    b = np.asarray(lattice, dtype=np.float64)
    metric = b.T @ b
    det = float(np.linalg.det(metric))
    if det <= 0.0 or not np.isfinite(det):
        return np.zeros((3, 3), dtype=np.float64)
    shape = metric / det ** (1.0 / 3.0)
    dev = shape - np.eye(3)
    return dev - (np.trace(dev) / 3.0) * np.eye(3)


def surrogate_stress(state, p_iso_ref, k_dev_bar=K_DEV_BAR):
    """Tensile-positive whole-system stress (Ha/Bohr^3) of the surrogate.

    The law is deliberately simple and frame-covariant: the box pushes
    back on volume change and on shape change, and nothing else. The
    isotropic part is referenced to the initial kinetic pressure minus
    the external pressure, so the state starts at equilibrium pressure
    and every later drive is the cell's own deformation. Like a real
    backend the value depends on the configuration only -- never on the
    current velocities -- so any coupling that appears between the
    temperature and the cell belongs to the framework.

    Parameters
    ----------
    state : PyRAIState
        State carrying the current cell.
    p_iso_ref : float
        Initial isotropic kinetic pressure (Ha/Bohr^3).
    k_dev_bar : float
        Deviatoric stiffness in bar; 0 gives a purely isotropic spring.

    Returns
    -------
    np.ndarray
        The (3, 3) stress block.
    """
    p_ext = bar_to_au_pressure(float(state.pmd.dynamics.pressure))
    volume = float(state.pmd.system.volume)
    isotropic = (p_iso_ref - p_ext
                 + bar_to_au_pressure(K_VOL_BAR)
                 * (volume - CUBIC_VOLUME) / CUBIC_VOLUME)
    deviatoric = bar_to_au_pressure(k_dev_bar) * _shape_deviation(
        state.pmd.system.lattice_vector)
    return isotropic * np.eye(3) - deviatoric


def cell_potential(lattice_bohr, offset_au, k_dev_au, k_vol_au, volume0_bohr3):
    """The scalar energy the conservative drive is derived from (Hartree).

        U = offset*V + 1/2 k_dev V |offdiag(B^T B / V^(2/3))|^2
                  + 1/2 k_vol (V - V0)^2 / V0

    The middle term is a shape spring on a dimensionless metric (det = 1), the
    last is the volume spring, and the first is the constant isotropic bias that
    puts the box at equilibrium pressure to begin with. Every term is a plain
    function of the lattice, which is what makes the resulting force
    conservative -- the point of this law existing at all (see
    `surrogate_stress` for the contrast).
    """
    g = lattice_bohr.T @ lattice_bohr
    volume = float(abs(np.linalg.det(lattice_bohr)))
    if volume <= 0.0:
        return float('inf')
    shape = g / volume ** (2.0 / 3.0)
    off = shape.copy()
    np.fill_diagonal(off, 0.0)
    return (offset_au * volume
            + 0.5 * k_dev_au * volume * float(np.sum(off ** 2))
            + 0.5 * k_vol_au * (volume - volume0_bohr3) ** 2 / volume0_bohr3)


def conservative_stress(state, p_iso_ref, k_dev_bar=K_DEV_BAR):
    """sigma = sym(B^T dU/dB)/V, the exact gradient of `cell_potential`.

    dU/dB is taken by central difference in Bohr, so no hand-derived chain
    rule can be mis-transcribed and the drive is conservative by
    construction. This is the law every STABILITY claim in this module uses:
    asking whether a barostat is stable against a stress field that is not
    the gradient of anything is not a question with an answer -- an earlier
    version of these tests drove the shape channel with
    `surrogate_stress`'s deviatoric part, which has curl, and reported the
    resulting growth as a framework instability (CHANGELOG #250 retracts
    that). Geometric tests, where the point is the basis a tensor is
    written in rather than the trajectory, keep using the simpler law.

    Parameters
    ----------
    state : PyRAIState
        State carrying the current cell.
    p_iso_ref : float
        Initial isotropic kinetic pressure (Ha/Bohr^3), so the box starts at
        equilibrium pressure.
    k_dev_bar : float
        Deviatoric stiffness in bar; 0 gives a purely isotropic spring.

    Returns
    -------
    np.ndarray
        The (3, 3) stress block in Hartree/Bohr^3.
    """
    b = (np.asarray(state.pmd.system.lattice_vector, dtype=np.float64)
         * ANGSTROM_TO_BOHR)
    k_dev = bar_to_au_pressure(k_dev_bar)
    k_vol = bar_to_au_pressure(K_VOL_BAR)
    volume0 = VOLUME0 * ANGSTROM_TO_BOHR ** 3
    offset = (p_iso_ref
              - bar_to_au_pressure(float(state.pmd.dynamics.pressure)))
    gradient = np.zeros((3, 3), dtype=np.float64)
    for row in range(3):
        for column in range(3):
            plus = b.copy()
            minus = b.copy()
            plus[row, column] += FD_STEP
            minus[row, column] -= FD_STEP
            gradient[row, column] = (
                cell_potential(plus, offset, k_dev, k_vol, volume0)
                - cell_potential(minus, offset, k_dev, k_vol, volume0)) \
                / (2.0 * FD_STEP)
    volume = float(abs(np.linalg.det(b)))
    sigma = (b.T @ gradient) / volume
    return 0.5 * (sigma + sigma.T)


def kinetic_pressure(state):
    """P_kin = sum(m v v)/V (Ha/Bohr^3), the barostat's own definition."""
    volume_bohr3 = ensemble_mod._volume_bohr3(state)
    return ensemble_mod._kinetic_pressure_tensor(state, volume_bohr3)


def run_npt_loop(state, nsteps, k_dev_bar=K_DEV_BAR, record_every=1,
                 law='surrogate', energy=False):
    """Run the framework's split step loop against the analytic stress.

    The order is the one AIMD._run_trajectory uses, including the
    history write, so the barostat reads a previous-frame stress exactly
    where the real loop does.

    Returns
    -------
    dict
        Per-recorded-frame 'volume', 'temperature', 'lattice' and
        'rotation' series (rotation in radians per step), plus
        'p_iso_ref' for reporting.
    """
    p_kin0 = kinetic_pressure(state)
    p_iso_ref = float(np.trace(p_kin0) / 3.0)
    natom = int(state.pmd.system.natom)

    def evaluate(st):
        gradient = np.zeros((natom, 3), dtype=np.float64)
        stress = (conservative_stress(st, p_iso_ref, k_dev_bar)
                  if law == 'potential'
                  else surrogate_stress(st, p_iso_ref, k_dev_bar))
        st.set_result({'gradient': gradient.tolist(),
                       'stress_tensor': stress[None].tolist()})

    evaluate(state)
    state.update_history()

    series = {'volume': [], 'temperature': [], 'lattice': [], 'rotation': []}
    if energy:
        series['energy'] = []
    previous_lattice = np.asarray(state.pmd.system.lattice_vector,
                                  dtype=np.float64)
    dt = float(state.timestep)
    for step in range(nsteps):
        apply_ensemble(state, stage='pre')
        verlet_i(state, dt=dt)
        apply_pbc(state)
        evaluate(state)
        verlet_ii(state, dt=dt)
        state.kinetic_energy()
        velocity_correction(state)
        apply_ensemble(state, stage='post')
        state.update_history()

        if step % record_every == 0 or step == nsteps - 1:
            lattice = np.asarray(state.pmd.system.lattice_vector,
                                 dtype=np.float64)
            series['volume'].append(float(state.pmd.system.volume))
            series['temperature'].append(
                _kinetic_temperature(state, state.mass_au,
                                     ensemble_mod._dof(state)))
            series['lattice'].append(lattice.copy())
            series['rotation'].append(_increment_rotation(previous_lattice,
                                                          lattice))
            if energy:
                # Measured HERE, on this frame's own momenta: rebuilding a
                # state from a recorded lattice and the final velocities
                # measures nothing, which is how an energy "drift" of 0.45%
                # was once read off a scheme that conserves to 4e-9.
                series['energy'].append(_cell_energy(state))
            previous_lattice = lattice
    series['p_iso_ref'] = p_iso_ref
    return series


def _increment_rotation(lattice_before, lattice_after):
    """Angle of the orthogonal part of the deformation between two frames.

    B = lattice_after @ inv(lattice_before) is the affine map the step
    applied to the cell; its polar orthogonal factor Q is the frame
    rotation, and arccos((tr(Q) - 1)/2) is the angle the box turned by.
    """
    before = np.asarray(lattice_before, dtype=np.float64)
    after = np.asarray(lattice_after, dtype=np.float64)
    try:
        deform = after @ np.linalg.inv(before)
    except np.linalg.LinAlgError:
        return float('nan')
    u, _, vt = np.linalg.svd(deform)
    q = u @ vt
    trace = float(np.clip((np.trace(q) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(trace))


def _rotation_about_z(angle):
    """The (3, 3) rotation matrix about z (row-vector convention)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64)














def _negate_momenta(state):
    """Flip the particle and cell momenta: the reverse leg of a run."""
    velocity = -np.asarray(state.velocities, dtype=np.float64)
    momentum = -np.asarray(state.pmd.system.barostat_momentum
                           or np.zeros((3, 3)), dtype=np.float64)
    state.update_system({'velocities': velocity.tolist(),
                         'barostat_momentum': momentum.tolist()})


def _unwrapped(state):
    """Positions outside the main image, and the momenta.

    Coordinates are stored wrapped, so a reversible run legitimately
    differs by a whole box vector when an atom changes image; the image
    counts are exactly the bookkeeping that says which box it is in, and
    comparing without them would turn a correct result into a 10 Angstrom
    apparent failure.
    """
    coordinates = np.asarray(state.coordinates, dtype=np.float64)
    lattice = np.asarray(state.pmd.system.lattice_vector, dtype=np.float64)
    image = np.asarray(state.pmd.system.image or np.zeros_like(coordinates),
                       dtype=np.float64)
    return (coordinates + image @ lattice,
            np.asarray(state.velocities, dtype=np.float64))


def _cell_energy(state):
    """E = KE_peculiar + sum(p_eps^2)/(2W) + U(B), in Hartree."""
    volume_bohr3 = _ensemble_volume_bohr3(state)
    kt = float(state.pmd.dynamics.temperature) * KELVIN_TO_HARTREE
    mass = _barostat_mass(state, kt, volume_bohr3)
    momentum = np.asarray(state.pmd.system.barostat_momentum
                          or np.zeros((3, 3)), dtype=np.float64)
    lattice_bohr = (np.asarray(state.pmd.system.lattice_vector,
                               dtype=np.float64) * ANGSTROM_TO_BOHR)
    offset = (kinetic_pressure(state)[0, 0]
              - bar_to_au_pressure(float(state.pmd.dynamics.pressure)))
    return (float(state.runtime.kinetic_energy or 0.0)
            + 0.5 * float(np.sum(momentum ** 2)) / mass
            + cell_potential(lattice_bohr, offset,
                             bar_to_au_pressure(K_DEV_BAR),
                             bar_to_au_pressure(K_VOL_BAR),
                             VOLUME0 * ANGSTROM_TO_BOHR ** 3))



def _first_oscillation_period(series):
    """Steps between the first two extrema of a series, or None."""
    values = np.asarray(series, dtype=np.float64)
    slope = np.diff(values)
    turning = [index + 1 for index in range(len(slope) - 1)
               if slope[index] * slope[index + 1] < 0.0
               and 0 < index + 1 < len(values) - 1]
    if len(turning) < 3:
        return None
    return 2.0 * float(turning[1] - turning[0])






if __name__ == '__main__':
    unittest.main()
