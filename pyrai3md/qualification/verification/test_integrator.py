"""
======================================================================
This module tests the NAMD workflow integrator package:

Tests cover:
- verlet_i coordinate propagation (history-driven, first-step skip,
  uniform motion, constant force, unit conversion, graceful
  degradation, assembled active-gradient reading, half kick,
  cell-aware drift exp(v_eps dt) scaling of coordinates and cell)
- verlet_ii velocity propagation (second half kick, first-step
  skip, assembled active-gradient reading)
- NVE split-Verlet equivalence to the classic merged form and total
  energy conservation
- apply_pbc periodic wrapping (cubic / triclinic / partial axis)
- velocity_correction center-of-mass removal and kinetic-energy
  conservation
- NVT Nose-Hoover chain (extended-energy conservation, temperature
  relaxation) and CSVR canonical kinetic-energy sampling
- NPT Berendsen / Parrinello-Rahman / MTTK full-tensor cell coupling
  with inline mock stress, the missing-stress refusal, and multi-step
  Berendsen volume equilibration / compressed-cell relaxation
- stress projection (_active_stress / active_stress) in the generic
  (nstate, 3, 3) format, including after a surface hop
- extended-ensemble DOF checkpoint/restart round-trip
- Active-state gradient extraction (state-resolved and single-state)
- Timestep fallback to pmd.dynamics.size
- Remaining placeholder operations return the state unchanged
- phase_correction NAC temporal phase consistency (sign flip from
  the previous-step overlap, phasecheck gating, first-step and
  no-NAC pass-through)
- surface_hopping factory (nosh/gsh/fssh initialization, unknown
  method)
- state.kinetic_energy() computation and storage

Aug 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md.data.state import PyRAIState
from pyrai3md.data.runtime.state import RuntimeState
from pyrai3md.schema.config import PyRAIConfig
from pyrai3md.utilities.units import (
    AMU_TO_AU, BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR, KELVIN_TO_HARTREE,
    FS_TO_AU_TIME, bar_to_au_pressure, compressibility_bar_to_au)
from pyrai3md.utilities.lattice import unwrap_coordinates
from pyrai3md.data.geometry import apply_pbc, kabsch
from pyrai3md.workflow.integrator import (
    verlet_i, verlet_ii,
    phase_correction, apply_constrain,
    velocity_correction, apply_ensemble,
    surface_hopping,
)
from pyrai3md.workflow.integrator.verlet import compute_cell
from pyrai3md.workflow.integrator import ensemble as ensemble_mod
from pyrai3md.data.state import _active_gradient, _active_stress

# Test system constants
MASS = [12.0, 1.0]                 # amu
COORDS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]   # Angstrom
VELOS = [[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]]  # Bohr/au
GRAD = [[0.02, 0.0, 0.0], [0.0, -0.04, 0.0]]  # Hartree/Bohr
DT = 2.0                           # au


def _make_state(gradient=None, velocities=VELOS, mass=MASS,
                coordinates=COORDS, size=None, current_state=0):
    """Create a minimal PyRAIState for integrator tests."""
    config = PyRAIConfig()
    config.system.atoms = ['C', 'H']
    config.system.coordinates = [list(row) for row in coordinates]
    config.system.velocities = [list(row) for row in velocities]
    config.system.mass = list(mass)
    config.system.natom = len(config.system.atoms)
    config.system.nstate = 2
    if size is not None:
        config.dynamics.size = float(size)

    state = PyRAIState(runtime=RuntimeState(total_steps=10))
    state.pmd = config
    if current_state:
        state.update_system({'current_state': current_state})
    if gradient is not None:
        state.set_result({'gradient': np.asarray(gradient).tolist()})
    return state


def _seed_history(state, gradient, current_state=None):
    """Record a previous timestep into the history.

    Sets the (optional) active state and the previous gradient,
    snapshots the step through the official history API, then
    restores the current result for the next step.
    """
    saved_state = state.current_state
    saved_result = state.result.to_dict() if state.result else {}
    if current_state is not None:
        state.update_system({'current_state': current_state})
    state.set_result({'gradient': np.asarray(gradient).tolist()}
                     if gradient is not None else {})
    state.update_history()
    state.update_system({'current_state': saved_state})
    state.set_result(saved_result)  # restore (may be empty)


class TestActiveGradient(unittest.TestCase):
    """Test active-state gradient extraction."""

    def test_state_resolved_selects_active_row(self):
        grad = np.stack([np.zeros((2, 3)), np.ones((2, 3))])
        out = _active_gradient(grad, 1)
        np.testing.assert_allclose(out, np.ones((2, 3)))

    def test_single_state_passes_through(self):
        grad = np.ones((2, 3))
        out = _active_gradient(grad, 0)
        np.testing.assert_allclose(out, grad)

    def test_missing_returns_none(self):
        self.assertIsNone(_active_gradient(None, 0))

    def test_out_of_range_returns_none(self):
        grad = np.stack([np.zeros((2, 3)), np.ones((2, 3))])
        self.assertIsNone(_active_gradient(grad, 5))


class TestVerletI(unittest.TestCase):
    """Test the history-driven verlet_i coordinate propagation."""

    def test_first_step_without_history_skips(self):
        state = _make_state(gradient=GRAD)
        verlet_i(state, dt=DT)
        np.testing.assert_allclose(state.coordinates, COORDS)

    def test_uniform_motion_zero_gradient(self):
        state = _make_state()
        _seed_history(state, np.zeros((2, 3)))
        verlet_i(state, dt=DT)
        expected = (np.asarray(COORDS)
                    + np.asarray(VELOS) * DT * BOHR_TO_ANGSTROM)
        np.testing.assert_allclose(state.coordinates, expected)

    def test_constant_force_matches_analytic(self):
        state = _make_state()
        _seed_history(state, GRAD)
        verlet_i(state, dt=DT)
        mass_au = np.asarray(MASS)[:, None] * AMU_TO_AU
        disp = (np.asarray(VELOS) * DT
                - 0.5 * np.asarray(GRAD) / mass_au * DT ** 2)
        expected = np.asarray(COORDS) + disp * BOHR_TO_ANGSTROM
        np.testing.assert_allclose(state.coordinates, expected)

    def test_velocities_half_kicked(self):
        # Split form (docs section 5): verlet_i now writes the half
        # kicked velocity V(t+dt/2) = V(t) - 0.5 G(t)/m dt, exposing
        # the half-step velocity where the ensemble operators attach.
        state = _make_state()
        _seed_history(state, GRAD)
        verlet_i(state, dt=DT)
        mass_au = np.asarray(MASS)[:, None] * AMU_TO_AU
        expected = (np.asarray(VELOS)
                    - 0.5 * np.asarray(GRAD) / mass_au * DT)
        np.testing.assert_allclose(state.velocities, expected)

    def test_missing_gradient_in_history_returns_unchanged(self):
        state = _make_state()
        _seed_history(state, None)
        verlet_i(state, dt=DT)
        np.testing.assert_allclose(state.coordinates, COORDS)

    def test_missing_velocities_treated_as_zero(self):
        state = _make_state(velocities=[])
        _seed_history(state, GRAD)
        verlet_i(state, dt=DT)
        mass_au = np.asarray(MASS)[:, None] * AMU_TO_AU
        disp = -0.5 * np.asarray(GRAD) / mass_au * DT ** 2
        expected = np.asarray(COORDS) + disp * BOHR_TO_ANGSTROM
        np.testing.assert_allclose(state.coordinates, expected)

    def test_missing_mass_returns_unchanged(self):
        state = _make_state(mass=[])
        _seed_history(state, GRAD)
        verlet_i(state, dt=DT)
        np.testing.assert_allclose(state.coordinates, COORDS)

    def test_timestep_falls_back_to_dynamics_size(self):
        state = _make_state(size=3.0)
        _seed_history(state, np.zeros((2, 3)))
        verlet_i(state)  # dt=None reads pmd.dynamics.size
        expected = (np.asarray(COORDS)
                    + np.asarray(VELOS) * 3.0 * BOHR_TO_ANGSTROM)
        np.testing.assert_allclose(state.coordinates, expected)

    def test_gradient_uses_recorded_active_state_after_hop(self):
        # The previous step was computed on state 1, but a hop moved
        # the active state to 0 afterwards: the frame carries the
        # assembled gradient of row 1 (captured at update_history
        # time), so verlet_i integrates row 1, not the current row 0.
        grad = np.stack([np.zeros((2, 3)), np.asarray(GRAD)])
        state = _make_state(current_state=0)
        _seed_history(state, grad, current_state=1)
        state.update_system({'current_state': 0})

        verlet_i(state, dt=DT)

        mass_au = np.asarray(MASS)[:, None] * AMU_TO_AU
        disp = (np.asarray(VELOS) * DT
                - 0.5 * np.asarray(GRAD) / mass_au * DT ** 2)
        expected = np.asarray(COORDS) + disp * BOHR_TO_ANGSTROM
        np.testing.assert_allclose(state.coordinates, expected)

    def test_frame_without_active_gradient_returns_unchanged(self):
        # r6a: verlet reads the frame's assembled active gradient, not
        # a recorded state index; a frame carrying none degrades
        # gracefully (the current_state fail-fast is retired).
        state = _make_state()
        state._history.append({'system': {}, 'results': {}})
        verlet_i(state, dt=DT)
        np.testing.assert_allclose(state.coordinates, COORDS)


class TestVerletII(unittest.TestCase):
    """Test verlet_ii velocity propagation."""

    def test_first_step_without_history_skips(self):
        state = _make_state(gradient=GRAD)
        verlet_ii(state, dt=DT)
        np.testing.assert_allclose(state.velocities, VELOS)

    def test_second_half_kick_uses_current_gradient(self):
        # Split form (docs section 5): verlet_ii applies ONLY the second
        # half kick with the current gradient G(t+dt); the previous
        # gradient was already consumed by verlet_i. The previous frame
        # only gates the first-step skip.
        prev_grad = [[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]]
        state = _make_state(gradient=GRAD)
        _seed_history(state, prev_grad)
        state.set_result({'gradient': np.asarray(GRAD).tolist()})

        verlet_ii(state, dt=DT)

        mass_au = np.asarray(MASS)[:, None] * AMU_TO_AU
        expected = (np.asarray(VELOS)
                    - 0.5 * np.asarray(GRAD) / mass_au * DT)
        np.testing.assert_allclose(state.velocities, expected)

    def test_coordinates_untouched(self):
        state = _make_state(gradient=GRAD)
        _seed_history(state, GRAD)
        state.set_result({'gradient': np.asarray(GRAD).tolist()})
        verlet_ii(state, dt=DT)
        np.testing.assert_allclose(state.coordinates, COORDS)

    def test_missing_current_gradient_skips(self):
        state = _make_state(gradient=None)
        _seed_history(state, GRAD)
        verlet_ii(state, dt=DT)
        np.testing.assert_allclose(state.velocities, VELOS)

    def test_current_gradient_uses_active_state_after_hop(self):
        # A hop moved the active state from 1 to 0 after the previous
        # step: verlet_ii applies the second half kick with the CURRENT
        # active-state gradient row (0), while the previous frame (row 1)
        # only gates the first-step skip.
        prev_grad = np.stack([np.zeros((2, 3)),
                              np.asarray([[0.01, 0.0, 0.0],
                                          [0.0, 0.02, 0.0]])])
        curr_grad = np.stack([np.asarray(GRAD), np.zeros((2, 3))])
        state = _make_state(current_state=0)
        _seed_history(state, prev_grad, current_state=1)
        state.update_system({'current_state': 0})
        state.set_result({'gradient': curr_grad.tolist()})

        verlet_ii(state, dt=DT)

        mass_au = np.asarray(MASS)[:, None] * AMU_TO_AU
        expected = (np.asarray(VELOS)
                    - 0.5 * curr_grad[0] / mass_au * DT)
        np.testing.assert_allclose(state.velocities, expected)

    def test_frame_without_active_gradient_returns_unchanged(self):
        # r6a: a frame carrying no assembled active gradient degrades
        # gracefully (the current_state fail-fast is retired).
        state = _make_state(gradient=GRAD)
        state._history.append({'system': {}, 'results': {}})
        verlet_ii(state, dt=DT)
        np.testing.assert_allclose(state.velocities, VELOS)


class TestOperations(unittest.TestCase):
    """Test the independent workflow operations."""

    def test_default_closed_operations_return_same_state(self):
        # Every one of these is a no-op on a state that declares nothing:
        # pbc off, no constraint block, ensemble nve, no reset. The
        # default-closed contract is what lets the loop call all of them
        # on every step at no cost.
        state = _make_state(gradient=GRAD)
        for operation in (apply_pbc, apply_constrain,
                          velocity_correction, apply_ensemble):
            self.assertIs(operation(state), state)

    def test_default_closed_operations_do_not_modify_state(self):
        # apply_constrain delegates to a constraint runtime that short
        # circuits on an empty plan, and phase_correction is a no-op
        # without a NAC record; the implemented paths of both are covered
        # by their dedicated suites.
        state = _make_state(gradient=GRAD)
        apply_constrain(phase_correction(state))
        np.testing.assert_allclose(state.coordinates, COORDS)
        np.testing.assert_allclose(state.velocities, VELOS)

    def test_apply_pbc_noop_without_periodic_axis(self):
        # The default system has pbc=[0, 0, 0]: wrapping is inert.
        state = _make_state(gradient=GRAD)
        apply_pbc(state)
        np.testing.assert_allclose(state.coordinates, COORDS)

    def test_apply_ensemble_noop_for_nve(self):
        # The default ensemble is 'nve': both stages leave the state
        # untouched, so the split Verlet reduces to the merged form.
        state = _make_state(gradient=GRAD)
        apply_ensemble(state, stage='pre')
        apply_ensemble(state, stage='post')
        np.testing.assert_allclose(state.velocities, VELOS)
        np.testing.assert_allclose(state.coordinates, COORDS)


def _seed_history_result(state, results):
    """Record one previous step carrying the given result keys.

    Snapshots the step through the official history API, then
    restores the current result for the next step.
    """
    saved = state.result.to_dict() if state.result else {}
    state.set_result(results)
    state.update_history()
    state.set_result(saved)


class TestPhaseCorrection(unittest.TestCase):
    """Test the NAC temporal phase consistency correction."""

    # Two coupling pairs over two atoms; pair 0 flips sign between
    # the steps, pair 1 keeps its phase.
    PREV_NAC = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                         [[0.5, 0.5, 0.0], [0.0, 0.0, 0.5]]])
    CURR_NAC = np.array([[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                         [[0.5, 0.5, 0.0], [0.0, 0.0, 0.5]]])

    def _state_with_nac(self, nac):
        state = _make_state(gradient=GRAD)
        state.set_result({'energy': [0.0, 0.05],
                          'nac': np.asarray(nac).tolist()})
        return state

    def test_flips_pairs_with_negative_overlap(self):
        state = self._state_with_nac(self.CURR_NAC)
        _seed_history_result(state, {'nac': self.PREV_NAC.tolist()})
        # Restore the current step's result after the snapshot.
        state.set_result({'energy': [0.0, 0.05],
                          'nac': self.CURR_NAC.tolist()})

        phase_correction(state)

        corrected = np.asarray(state.result['nac'])
        np.testing.assert_allclose(corrected, self.PREV_NAC)
        # The other result keys survive the rewrite.
        np.testing.assert_allclose(state.result['energy'],
                                   [0.0, 0.05])

    def test_consistent_phase_keeps_the_result(self):
        state = self._state_with_nac(self.PREV_NAC)
        _seed_history_result(state, {'nac': self.PREV_NAC.tolist()})
        state.set_result({'nac': self.PREV_NAC.tolist()})
        self.assertIs(phase_correction(state), state)

    def test_phasecheck_off_disables_the_correction(self):
        state = self._state_with_nac(self.CURR_NAC)
        state.pmd.dynamics.phasecheck = 0
        _seed_history_result(state, {'nac': self.PREV_NAC.tolist()})
        state.set_result({'nac': self.CURR_NAC.tolist()})
        self.assertIs(phase_correction(state), state)
        np.testing.assert_allclose(state.result['nac'],
                                   self.CURR_NAC)

    def test_first_step_without_history_skips(self):
        state = self._state_with_nac(self.CURR_NAC)
        self.assertIs(phase_correction(state), state)

    def test_missing_previous_nac_skips(self):
        state = self._state_with_nac(self.CURR_NAC)
        _seed_history_result(state, {'energy': [0.0, 0.05]})
        state.set_result({'nac': self.CURR_NAC.tolist()})
        self.assertIs(phase_correction(state), state)

    def test_zero_overlap_keeps_the_pair(self):
        curr = np.array([[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                         [[0.5, 0.5, 0.0], [0.0, 0.0, 0.5]]])
        prev = np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                         [[0.5, 0.5, 0.0], [0.0, 0.0, 0.5]]])
        state = self._state_with_nac(curr)
        _seed_history_result(state, {'nac': prev.tolist()})
        state.set_result({'nac': curr.tolist()})
        self.assertIs(phase_correction(state), state)

    def test_negligible_coupling_keeps_its_phase(self):
        # A coupling crossing zero leaves noise-level vectors that
        # can be exactly antiparallel; their arbitrary phase must
        # not be flipped (the overlap magnitude carries no
        # dynamics).
        curr = np.array([[[1.4e-17, 0.0, 0.0], [0.0, 0.0, 0.0]],
                         [[0.5, 0.5, 0.0], [0.0, 0.0, 0.5]]])
        prev = np.array([[[-4.2e-18, 0.0, 0.0], [0.0, 0.0, 0.0]],
                         [[0.5, 0.5, 0.0], [0.0, 0.0, 0.5]]])
        state = self._state_with_nac(curr)
        _seed_history_result(state, {'nac': prev.tolist()})
        state.set_result({'nac': curr.tolist()})
        self.assertIs(phase_correction(state), state)


class TestSurfaceHopping(unittest.TestCase):
    """Test the surface hopping factory (one-time initialization)."""

    def test_nosh_returns_pass_through(self):
        state = _make_state(gradient=GRAD)
        surfacehop = surface_hopping('nosh', state)
        self.assertIs(surfacehop(state), state)

    def test_gsh_placeholder_returns_pass_through(self):
        state = _make_state(gradient=GRAD)
        surfacehop = surface_hopping('gsh', state)
        self.assertIs(surfacehop(state), state)

    def test_fssh_returns_callable_integrator(self):
        from pyrai3md.workflow.integrator.fssh import FsshIntegrator
        from pyrai3md.workflow.integrator.electronic import (
            ElectronicStepper)
        state = _make_state(gradient=GRAD)
        surfacehop = surface_hopping('fssh', state)
        self.assertIsInstance(surfacehop, ElectronicStepper)
        self.assertIsInstance(
            surfacehop.binding(0).algorithm, FsshIntegrator)

    def test_method_is_case_insensitive(self):
        state = _make_state(gradient=GRAD)
        surfacehop = surface_hopping('NOSH', state)
        self.assertIs(surfacehop(state), state)

    def test_unknown_method_raises(self):
        state = _make_state(gradient=GRAD)
        with self.assertRaises(ValueError):
            surface_hopping('unknown', state)


class TestKineticEnergyState(unittest.TestCase):
    """Test the official state.kinetic_energy() method."""

    def test_computes_and_stores_value(self):
        state = _make_state(gradient=None)
        expected = float(0.5 * np.sum(
            np.asarray(MASS)[:, None] * AMU_TO_AU
            * np.asarray(VELOS) ** 2))
        self.assertAlmostEqual(state.kinetic_energy(), expected)
        self.assertAlmostEqual(state.runtime.kinetic_energy, expected)

    def test_missing_data_stores_zero(self):
        state = _make_state(gradient=None, velocities=[])
        self.assertEqual(state.kinetic_energy(), 0.0)
        self.assertEqual(state.runtime.kinetic_energy, 0.0)


class TestStateProperties(unittest.TestCase):
    """Test the new PyRAIState read-only properties."""

    def test_mass_property(self):
        state = _make_state(gradient=None)
        self.assertEqual(state.mass, MASS)

    def test_timestep_property(self):
        state = _make_state(gradient=None, size=7.5)
        self.assertAlmostEqual(state.timestep, 7.5)

    def test_timestep_default(self):
        state = _make_state(gradient=None)
        self.assertAlmostEqual(state.timestep, 20.67)

    def test_kinetic_energy_roundtrip(self):
        state = _make_state(gradient=None)
        state.kinetic_energy()
        restored = RuntimeState.from_dict(state.runtime.to_dict())
        self.assertAlmostEqual(restored.kinetic_energy,
                               state.runtime.kinetic_energy)


# ---------------------------------------------------------------------------
# Ensemble / PBC / NVE suites (dev/methods/ENSEMBLE_DEVELOPMENT.md section 10)
# ---------------------------------------------------------------------------

SPRING_K = 0.002            # Hartree / Bohr^2 (a soft, well-resolved spring)
CUBIC_LATTICE = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
CUBIC_VOLUME = 1000.0       # Angstrom^3
_SYSTEM_FIELDS = {
    'pbc', 'lattice_vector', 'cell', 'volume', 'current_state', 'freeze',
    'image',
    'thermostat_dof', 'barostat_strain', 'barostat_momentum',
    'barostat_velocity', 'barostat_thermostat_dof'}


def _spring_grad(coords):
    """Harmonic restoring gradient (Hartree/Bohr) toward the origin."""
    r = np.asarray(coords, dtype=np.float64) * ANGSTROM_TO_BOHR
    return SPRING_K * r


def _spring_pe(coords):
    """Harmonic potential energy (Hartree) conjugate to _spring_grad."""
    r = np.asarray(coords, dtype=np.float64) * ANGSTROM_TO_BOHR
    return 0.5 * SPRING_K * float(np.sum(r ** 2))


def _make_ensemble_state(mass=MASS, coords=COORDS, velos=VELOS, size=20.0,
                         **fields):
    """Build a PyRAIState carrying dynamics/system fields for ensembles.

    Keyword arguments are routed to pmd.dynamics (ensemble, thermostat,
    barostat, temperature, pressure, coupling constants, ...) or to
    pmd.system (pbc, cell, lattice_vector, volume, extended DOF, ...).
    """
    config = PyRAIConfig()
    mass = list(mass)
    natom = len(mass)
    config.system.atoms = (['C', 'H'] if natom == 2
                           else ['X'] * natom)[:natom]
    config.system.coordinates = [list(r) for r in coords]
    config.system.velocities = [list(r) for r in velos]
    config.system.mass = mass
    config.system.natom = natom
    config.system.nstate = 2
    config.dynamics.size = float(size)
    updates = {}
    for key, value in fields.items():
        if key in _SYSTEM_FIELDS:
            updates[key] = value
        else:
            setattr(config.dynamics, key, value)
    state = PyRAIState(runtime=RuntimeState(total_steps=100000))
    state.pmd = config
    if updates:
        state.update_system(updates)
    return state


def _propagate(state, grad_fn, nsteps, dt, pre=None, post=None, record=None):
    """Run nsteps of the split main loop and return the recorded list.

    Mirrors md._run_trajectory: seed the initial force into the history,
    then per step ensemble_pre -> verlet_i -> force eval -> verlet_ii ->
    kinetic_energy -> ensemble_post -> update_history. A single force
    evaluation per step (verlet_i reuses the previous frame gradient).
    """
    state.set_result({'gradient': grad_fn(state.coordinates).tolist()})
    state.update_history()
    recorded = []
    for _ in range(nsteps):
        if pre is not None:
            pre(state)
        verlet_i(state, dt=dt)
        state.set_result({'gradient': grad_fn(state.coordinates).tolist()})
        verlet_ii(state, dt=dt)
        state.kinetic_energy()
        if post is not None:
            post(state)
        state.update_history()
        if record is not None:
            recorded.append(record(state))
    return recorded




class TestVerletCellAwareDrift(unittest.TestCase):
    """verlet_i cell-aware drift: exp(v_eps dt) scaling (docs section 5).

    The barostat-driven branch of verlet_i scales the coordinates and
    the periodic cell by the matrix exponential exp(v_eps dt) (the
    combined iL_eps1 + iL_1 MTTK drift). These tests pin that branch
    deterministically without depending on how expm is computed: a
    diagonal v_eps gives the exact elementwise scaling, and a full
    (non-symmetric, shear-carrying) tensor v_eps preserves the
    fractional coordinates while scaling the volume by the independent
    identity det(exp(M)) = exp(tr(M)).
    """

    def test_diagonal_cell_velocity_scales_cell_and_coords(self):
        dt = 2.0
        g = np.array([0.001, 0.002, -0.001])
        v_eps = np.diag(g)
        lattice = np.asarray(CUBIC_LATTICE, dtype=np.float64)
        state = _make_ensemble_state(
            coords=COORDS, velos=VELOS,
            lattice_vector=lattice.tolist(), volume=CUBIC_VOLUME,
            barostat_velocity=v_eps.tolist())
        _seed_history(state, GRAD)

        verlet_i(state, dt=dt)

        scale_full = np.diag(np.exp(g * dt))
        scale_half = np.diag(np.exp(g * dt * 0.5))
        mass_au = np.asarray(MASS) * AMU_TO_AU
        half_v = (np.asarray(VELOS)
                  - 0.5 * np.asarray(GRAD) / mass_au[:, None] * dt)
        disp = half_v * dt * BOHR_TO_ANGSTROM
        expected_coords = np.asarray(COORDS) @ scale_full + disp @ scale_half
        expected_lattice = lattice @ scale_full
        expected_volume = abs(float(np.linalg.det(expected_lattice)))
        np.testing.assert_allclose(
            np.asarray(state.coordinates), expected_coords,
            rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(state.pmd.system.lattice_vector), expected_lattice,
            rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(
            float(state.pmd.system.volume), expected_volume, places=9)
        # The classic 6-parameter cell is refreshed for backend sync.
        cell6 = np.asarray(state.pmd.system.cell, dtype=np.float64)
        np.testing.assert_allclose(cell6[:3], 10.0 * np.exp(g * dt), rtol=1e-9)
        np.testing.assert_allclose(cell6[3:], [90.0, 90.0, 90.0], atol=1e-6)

    def test_full_tensor_drift_preserves_fractional_coordinates(self):
        dt = 1.5
        # A non-symmetric (anisotropic, shear-carrying) cell velocity.
        v_eps = np.array([[0.0005, 0.0003, 0.0],
                          [0.0002, -0.0004, 0.0001],
                          [0.0, 0.0006, 0.0003]])
        lattice = compute_cell([9.0, 10.0, 11.0, 82.0, 96.0, 104.0])
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 0.5, 1.5]])
        volume_before = abs(float(np.linalg.det(lattice)))
        state = _make_ensemble_state(
            coords=coords.tolist(), velos=np.zeros((2, 3)).tolist(),
            lattice_vector=lattice.tolist(), volume=volume_before,
            barostat_velocity=v_eps.tolist())
        _seed_history(state, np.zeros((2, 3)))

        frac_before = coords @ np.linalg.inv(lattice)

        verlet_i(state, dt=dt)

        new_lattice = np.asarray(
            state.pmd.system.lattice_vector, dtype=np.float64)
        new_coords = np.asarray(state.coordinates, dtype=np.float64)
        # Homogeneous deformation preserves fractional coordinates.
        frac_after = new_coords @ np.linalg.inv(new_lattice)
        np.testing.assert_allclose(frac_after, frac_before, rtol=1e-9, atol=1e-9)
        # det(exp(M)) = exp(tr(M)) gives an expm-free volume check.
        expected_volume = volume_before * float(np.exp(np.trace(v_eps) * dt))
        self.assertAlmostEqual(
            float(state.pmd.system.volume), expected_volume, places=9)
        np.testing.assert_allclose(
            float(state.pmd.system.volume),
            abs(float(np.linalg.det(new_lattice))), rtol=1e-12)

    def test_zero_cell_velocity_leaves_cell_unchanged(self):
        dt = 2.0
        lattice = np.asarray(CUBIC_LATTICE, dtype=np.float64)
        state = _make_ensemble_state(
            coords=COORDS, velos=VELOS,
            lattice_vector=lattice.tolist(), volume=CUBIC_VOLUME,
            barostat_velocity=np.zeros((3, 3)).tolist())
        _seed_history(state, GRAD)

        verlet_i(state, dt=dt)

        # Plain R += V(t+dt/2) dt drift; the cell is untouched.
        mass_au = np.asarray(MASS) * AMU_TO_AU
        half_v = (np.asarray(VELOS)
                  - 0.5 * np.asarray(GRAD) / mass_au[:, None] * dt)
        expected_coords = np.asarray(COORDS) + half_v * dt * BOHR_TO_ANGSTROM
        np.testing.assert_allclose(
            np.asarray(state.coordinates), expected_coords,
            rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(state.pmd.system.lattice_vector), lattice, atol=1e-12)
        self.assertAlmostEqual(float(state.pmd.system.volume), CUBIC_VOLUME,
                               places=9)


class TestApplyPbcWrapping(unittest.TestCase):
    """Periodic boundary coordinate wrapping (docs section 7)."""

    def test_wraps_into_cubic_cell(self):
        state = _make_ensemble_state(
            coords=[[12.0, -3.0, 25.0], [1.0, 0.0, 0.0]],
            pbc=[1, 1, 1], lattice_vector=CUBIC_LATTICE, volume=CUBIC_VOLUME)
        apply_pbc(state)
        coords = np.asarray(state.coordinates, dtype=np.float64)
        np.testing.assert_allclose(coords[0], [2.0, 7.0, 5.0])
        np.testing.assert_allclose(coords[1], [1.0, 0.0, 0.0])

    def test_image_records_the_crossings_of_the_wrap(self):
        state = _make_ensemble_state(
            coords=[[12.0, -3.0, 25.0], [1.0, 0.0, 0.0]],
            pbc=[1, 1, 1], lattice_vector=CUBIC_LATTICE, volume=CUBIC_VOLUME)
        apply_pbc(state)
        # 12 -> 2 crossed +1, -3 -> 7 crossed -1, 25 -> 5 crossed +2; the
        # atom already inside the cell crossed nothing.
        self.assertEqual([[int(v) for v in row]
                          for row in state.pmd.system.image],
                         [[1, -1, 2], [0, 0, 0]])

    def test_image_accumulates_across_two_crossings(self):
        state = _make_ensemble_state(
            coords=[[12.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            pbc=[1, 1, 1], lattice_vector=CUBIC_LATTICE, volume=CUBIC_VOLUME)
        apply_pbc(state)
        first = [[int(v) for v in row] for row in state.pmd.system.image]
        self.assertEqual(first, [[1, 0, 0], [0, 0, 0]])
        # Drift one cell further out and wrap again: the tally is a sum,
        # because the wrapped coordinate alone cannot say how far the
        # atom has travelled.
        drifted = np.asarray(state.coordinates, dtype=np.float64)
        drifted[0][0] += 20.0
        state.update_system({'coordinates': drifted.tolist()})
        apply_pbc(state)
        self.assertEqual([[int(v) for v in row]
                          for row in state.pmd.system.image],
                         [[3, 0, 0], [0, 0, 0]])

    def test_image_counts_only_periodic_axes(self):
        state = _make_ensemble_state(
            coords=[[12.0, -3.0, 25.0]],
            pbc=[1, 0, 0], lattice_vector=CUBIC_LATTICE, volume=CUBIC_VOLUME)
        apply_pbc(state)
        np.testing.assert_allclose(
            np.asarray(state.coordinates, dtype=np.float64)[0],
            [2.0, -3.0, 25.0])
        self.assertEqual([[int(v) for v in row]
                          for row in state.pmd.system.image], [[1, 0, 0]])

    def test_image_stays_empty_when_non_periodic(self):
        state = _make_ensemble_state(
            coords=[[12.0, -3.0, 25.0]], pbc=[0, 0, 0],
            lattice_vector=CUBIC_LATTICE, volume=CUBIC_VOLUME)
        apply_pbc(state)
        self.assertEqual(list(state.pmd.system.image), [])

    def test_malformed_previous_image_restarts_the_tally(self):
        # A checkpoint written before the image channel existed carries no
        # usable count; the wrap must still work and credit this step.
        state = _make_ensemble_state(
            coords=[[12.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            pbc=[1, 1, 1], lattice_vector=CUBIC_LATTICE, volume=CUBIC_VOLUME,
            image=[[7, 0, 0]])
        apply_pbc(state)
        self.assertEqual([[int(v) for v in row]
                          for row in state.pmd.system.image],
                         [[1, 0, 0], [0, 0, 0]])

    def test_a_seeded_base_continues_under_the_wrap(self):
        # The shape check above protects a tally that cannot be trusted;
        # a well-formed seed is the opposite case: the crossings the
        # input already implies are the base this step adds to, and
        # dropping them would move the atoms back across the box.
        state = _make_ensemble_state(
            coords=[[12.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            pbc=[1, 1, 1], lattice_vector=CUBIC_LATTICE, volume=CUBIC_VOLUME,
            image=[[2, 0, 0], [0, 0, 0]])
        apply_pbc(state)
        self.assertEqual([[int(v) for v in row]
                          for row in state.pmd.system.image],
                         [[3, 0, 0], [0, 0, 0]])

    def test_wrapped_coordinate_plus_image_rebuilds_the_path(self):
        coords = np.asarray([[12.0, -3.0, 25.0], [1.0, 4.0, -0.5]])
        state = _make_ensemble_state(
            coords=coords.tolist(), pbc=[1, 1, 1],
            lattice_vector=CUBIC_LATTICE, volume=CUBIC_VOLUME)
        apply_pbc(state)
        rebuilt = unwrap_coordinates(
            np.asarray(state.coordinates),
            np.asarray(state.pmd.system.image),
            np.asarray(CUBIC_LATTICE, dtype=np.float64))
        # A cubic lattice of 10 puts the wrapped coordinates back on the
        # path the integrator actually took, box crossings included.
        np.testing.assert_allclose(rebuilt, coords, rtol=1e-12, atol=1e-9)

    def test_partial_axis_only_wraps_periodic(self):
        state = _make_ensemble_state(
            coords=[[12.0, -3.0, 25.0], [0.0, 0.0, 0.0]],
            pbc=[1, 0, 0], lattice_vector=CUBIC_LATTICE, volume=CUBIC_VOLUME)
        apply_pbc(state)
        coords = np.asarray(state.coordinates, dtype=np.float64)
        np.testing.assert_allclose(coords[0], [2.0, -3.0, 25.0])

    def test_triclinic_fractional_in_unit_cell(self):
        lattice = compute_cell([8.0, 9.0, 10.0, 80.0, 95.0, 100.0])
        frac_in = np.array([1.3, -0.4, 2.1])
        cart = frac_in @ lattice
        state = _make_ensemble_state(
            coords=[cart.tolist(), [0.0, 0.0, 0.0]],
            pbc=[1, 1, 1], lattice_vector=lattice.tolist(),
            volume=float(abs(np.linalg.det(lattice))))
        apply_pbc(state)
        wrapped = np.asarray(state.coordinates, dtype=np.float64)[0]
        frac_out = wrapped @ np.linalg.inv(lattice)
        np.testing.assert_allclose(frac_out, frac_in % 1.0, atol=1e-9)

    def test_builds_cell_from_six_parameters(self):
        state = _make_ensemble_state(
            coords=[[12.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            pbc=[1, 1, 1], cell=[10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
            lattice_vector=[[0.0, 0.0, 0.0]] * 3, volume=CUBIC_VOLUME)
        apply_pbc(state)
        coords = np.asarray(state.coordinates, dtype=np.float64)
        np.testing.assert_allclose(coords[0], [2.0, 0.0, 0.0])

    def test_raises_without_cell(self):
        state = _make_ensemble_state(
            coords=[[12.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            pbc=[1, 1, 1], lattice_vector=[[0.0, 0.0, 0.0]] * 3,
            cell=[0.0] * 6)
        with self.assertRaises(ValueError):
            apply_pbc(state)

    def test_raises_on_singular_cell(self):
        state = _make_ensemble_state(
            coords=[[12.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            pbc=[1, 1, 1],
            lattice_vector=[[10.0, 0.0, 0.0],
                            [0.0, 10.0, 0.0],
                            [5.0, 0.0, 0.0]],
            cell=[0.0] * 6)
        with self.assertRaises(ValueError):
            apply_pbc(state)


class TestVelocityCorrectionOp(unittest.TestCase):
    """Center-of-mass drift removal (docs section 7)."""

    def test_removes_com_and_conserves_kinetic_energy(self):
        state = _make_ensemble_state()
        mass_au = np.asarray(MASS) * AMU_TO_AU
        v0 = np.asarray(VELOS, dtype=np.float64)
        ke0 = 0.5 * float(np.sum(mass_au[:, None] * v0 ** 2))
        velocity_correction(state)
        v = np.asarray(state.velocities, dtype=np.float64)
        vcom = np.sum(mass_au[:, None] * v, axis=0) / np.sum(mass_au)
        np.testing.assert_allclose(vcom, [0.0, 0.0, 0.0], atol=1e-12)
        ke = 0.5 * float(np.sum(mass_au[:, None] * v ** 2))
        np.testing.assert_allclose(ke, ke0, rtol=1e-12)

    def test_single_particle_keeps_its_velocity(self):
        # The Tully and Landau-Zener benchmark particle is one pseudo-atom.
        # Subtracting the center of mass of a one-atom system removes all
        # of its motion, and the kinetic rescale that follows has nothing
        # left to scale, so the wavepacket sits still, never reaches the
        # crossing and no hop is ever attempted. Drift removal does not
        # apply to a state with no internal degree of freedom.
        state = _make_ensemble_state(mass=[1.0], coords=[[0.0, 0.0, 0.0]],
                                     velos=[[0.005, 0.0, 0.0]])
        velocity_correction(state)
        np.testing.assert_allclose(
            np.asarray(state.velocities, dtype=np.float64),
            [[0.005, 0.0, 0.0]], atol=1e-15)

    def test_kabsch_identical_is_zero(self):
        p = np.asarray(COORDS, dtype=np.float64)
        self.assertAlmostEqual(kabsch(p, p.copy()), 0.0, places=10)


def _ensemble_pre(state):
    """apply_ensemble at the pre stage (before verlet_i)."""
    return apply_ensemble(state, stage='pre')


def _ensemble_post(state):
    """apply_ensemble at the post stage (after velocity_correction)."""
    return apply_ensemble(state, stage='post')


def _kinetic_temperature(state, mass_au, ndof):
    """Instantaneous kinetic temperature (K) of the current velocities."""
    v = np.asarray(state.velocities, dtype=np.float64)
    ke = 0.5 * float(np.sum(mass_au[:, None] * v ** 2))
    return 2.0 * ke / (ndof * KELVIN_TO_HARTREE)


def _nhc_extended_energy(state, mass_au):
    """Nose-Hoover-chain extended Hamiltonian H_ext (Hartree).

    H_ext = KE + PE + sum_k p_k^2/(2 Q_k) + N_df kT xi_1
            + kT sum_{k>=2} xi_k, the quantity the reversible NHC
    integrator conserves (docs section 6).
    """
    dyn = state.pmd.dynamics
    kt = float(dyn.temperature) * KELVIN_TO_HARTREE
    ndof = ensemble_mod._dof(state)
    chain = max(int(dyn.thermostat_chain), 1)
    Q = ensemble_mod._thermostat_masses(state, ndof, kt, chain)
    v = np.asarray(state.velocities, dtype=np.float64)
    ke = 0.5 * float(np.sum(mass_au[:, None] * v ** 2))
    pe = _spring_pe(state.coordinates)
    dof = np.asarray(state.pmd.system.thermostat_dof,
                     dtype=np.float64).reshape(-1)
    if dof.size != 2 * chain:
        dof = np.zeros(2 * chain)
    xi, p = dof[:chain], dof[chain:]
    h = ke + pe + float(np.sum(p ** 2 / (2.0 * Q)))
    h += ndof * kt * xi[0] + kt * float(np.sum(xi[1:]))
    return h








def _berendsen_mock_stress(state, k_spring, target):
    """Volume-restoring mock stress (Hartree/Bohr^3) for the barostat.

    stress = -P_ext I + K (V - target)/target I, so P_int = P_kin -
    stress gives P_ext I - P_int = K (V - target)/target I - P_kin: a
    smooth restoring drive of the cell toward the target volume.
    Deterministic; stands in for a real backend virial so the barostat
    integration is exercised without a QC/FF stress evaluation.
    """
    volume = float(state.pmd.system.volume)
    p_ext = bar_to_au_pressure(float(state.pmd.dynamics.pressure))
    return -p_ext * np.eye(3) + k_spring * (volume - target) / target * np.eye(3)


def _run_npt_berendsen(state, nsteps, dt, target, k_spring):
    """Run nsteps of the split NPT loop; return the per-step volume.

    Mirrors md._run_trajectory: ensemble_pre -> verlet_i -> apply_pbc ->
    force/stress eval -> verlet_ii -> kinetic_energy ->
    velocity_correction -> ensemble_post -> update_history, with a
    single force/stress evaluation per step.
    """
    def _evaluate(st):
        st.set_result({
            'gradient': _spring_grad(st.coordinates).tolist(),
            'stress_tensor':
                _berendsen_mock_stress(st, k_spring, target)[None].tolist()})

    _evaluate(state)
    state.update_history()
    volumes = []
    for _ in range(nsteps):
        apply_ensemble(state, stage='pre')
        verlet_i(state, dt=dt)
        apply_pbc(state)
        _evaluate(state)
        verlet_ii(state, dt=dt)
        state.kinetic_energy()
        velocity_correction(state)
        apply_ensemble(state, stage='post')
        state.update_history()
        volumes.append(float(state.pmd.system.volume))
    return np.asarray(volumes)




class TestStressProjection(unittest.TestCase):
    """Generic (nstate, 3, 3) stress projection (docs sections 4, 8)."""

    def test_selects_active_row(self):
        s = np.stack([np.zeros((3, 3)), np.eye(3) * 2.0])
        np.testing.assert_allclose(_active_stress(s, 1), np.eye(3) * 2.0)

    def test_single_matrix_passes_through(self):
        s = np.eye(3) * 3.0
        np.testing.assert_allclose(_active_stress(s, 0), s)

    def test_missing_returns_none(self):
        self.assertIsNone(_active_stress(None, 0))

    def test_out_of_range_returns_none(self):
        s = np.stack([np.zeros((3, 3)), np.eye(3)])
        self.assertIsNone(_active_stress(s, 5))

    def test_state_active_stress_switches_after_hop(self):
        state = _make_ensemble_state()
        stress = np.stack([np.eye(3), np.eye(3) * 5.0])
        state.set_result({'stress_tensor': stress.tolist()})
        np.testing.assert_allclose(state.active_stress(), np.eye(3))
        state.update_system({'current_state': 1})
        np.testing.assert_allclose(state.active_stress(), np.eye(3) * 5.0)

    def test_state_active_stress_none_without_stress(self):
        state = _make_ensemble_state()
        state.set_result({'gradient': np.zeros((2, 3)).tolist()})
        self.assertIsNone(state.active_stress())




class TestRetriangularization(unittest.TestCase):
    """Anisotropic variable-cell commits keep the framework
    lower-triangular lattice convention through a metric-preserving
    frame rotation: L = chol(B B^T), O = B^-1 L orthogonal, so
    fractional coordinates, volume and all distances are invariant."""

    def test_retriangularize_noop_for_lower_triangular(self):
        from pyrai3md.workflow.integrator.verlet import _retriangularize
        B = np.array([[10.0, 0.0, 0.0], [0.5, 9.0, 0.0], [0.2, 0.3, 8.0]])
        L, O = _retriangularize(B)
        self.assertIsNone(O)
        np.testing.assert_allclose(L, B)

    def test_retriangularize_preserves_metric(self):
        from pyrai3md.workflow.integrator.verlet import _retriangularize
        B = np.array([[10.0, 0.4, 0.2], [0.5, 9.0, 0.1], [0.3, 0.6, 8.0]])
        L, O = _retriangularize(B)
        self.assertLess(max(abs(L[0, 1]), abs(L[0, 2]), abs(L[1, 2])), 1e-12)
        # Same Gram matrix (same cell shape), orthogonal rotation,
        # B @ O == L.
        np.testing.assert_allclose(L @ L.T, B @ B.T, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(O @ O.T, np.eye(3), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(B @ O, L, rtol=1e-12, atol=1e-12)

    def test_scale_cell_rotates_frame_and_preserves_fractions(self):
        from pyrai3md.workflow.integrator.verlet import _scale_cell
        state = _make_ensemble_state(
            ensemble='npt', barostat='berendsen', temperature=300.0,
            pbc=[1, 1, 1], lattice_vector=CUBIC_LATTICE,
            volume=CUBIC_VOLUME)
        B = np.asarray(CUBIC_LATTICE, dtype=np.float64)
        # Full symmetric deformation (the anisotropic barostat case).
        v = np.array([[1e-3, 2e-4, 1e-4],
                      [2e-4, 1e-3, 3e-4],
                      [1e-4, 3e-4, 1e-3]])
        S = np.eye(3) + v
        coords = np.asarray(state.coordinates, dtype=np.float64)
        velos = np.asarray(state.velocities, dtype=np.float64)
        deformed = coords @ S
        updates = _scale_cell(state, S, coordinates=deformed,
                              velocities=velos)
        L = np.asarray(updates['lattice_vector'])
        self.assertLess(max(abs(L[0, 1]), abs(L[0, 2]), abs(L[1, 2])),
                        1e-12)
        # Volume is the determinant of the deformed cell (rotation is
        # metric-preserving).
        self.assertAlmostEqual(updates['volume'],
                               abs(float(np.linalg.det(B @ S))), places=10)
        # Fractional coordinates are invariant through the commit.
        frac_before = deformed @ np.linalg.inv(B @ S)
        frac_after = (np.asarray(updates['coordinates'])
                      @ np.linalg.inv(L))
        np.testing.assert_allclose(frac_after, frac_before,
                                   rtol=1e-10, atol=1e-12)
        # Velocities are purely rotated (norms preserved).
        v_new = np.asarray(updates['velocities'])
        np.testing.assert_allclose(
            np.linalg.norm(v_new, axis=1),
            np.linalg.norm(velos, axis=1), rtol=1e-12)

    def test_npt_propagation_keeps_lattice_lower_triangular(self):
        p_ext = bar_to_au_pressure(1.0)
        sigma = np.array([[1.875e-05, 7.6e-06, 5.2e-06],
                          [7.6e-06, -3.8e-06, -6.1e-06],
                          [5.2e-06, -6.1e-06, 3.05e-05]])
        dt = 20.0
        state = _make_ensemble_state(
            size=dt, ensemble='npt', thermostat='nose-hoover',
            barostat='berendsen', temperature=300.0, pressure=1.0,
            barostat_tau=100.0, compressibility=4.5e-5,
            barostat_mode='anisotropic',
            pbc=[1, 1, 1], lattice_vector=CUBIC_LATTICE,
            volume=CUBIC_VOLUME,
            velos=[[1e-4, 0.0, 0.0], [0.0, 2e-4, 0.0]])
        result = {'gradient': np.zeros((2, 3)).tolist(),
                  'stress_tensor': sigma[None].tolist()}
        state.set_result(result)
        state.update_history()
        for _ in range(5):
            apply_ensemble(state, stage='pre')
            verlet_i(state, dt=dt)
            state.set_result(result)
            verlet_ii(state, dt=dt)
            state.kinetic_energy()
            apply_ensemble(state, stage='post')
            state.update_history()
            B = np.asarray(state.pmd.system.lattice_vector)
            upper = max(abs(B[0, 1]), abs(B[0, 2]), abs(B[1, 2]))
            self.assertLess(upper, 1e-10 * abs(B[0, 0]))
            self.assertAlmostEqual(
                abs(float(np.linalg.det(B))),
                float(state.pmd.system.volume), places=8)


class TestExtendedDofSerialization(unittest.TestCase):
    """Extended-ensemble DOF persist through the checkpoint (docs 4)."""

    def _filled_config(self):
        config = PyRAIConfig()
        config.system.thermostat_dof = [0.1, 0.2, 0.3, 0.4]
        config.system.barostat_strain = np.eye(3).tolist()
        config.system.barostat_momentum = (np.eye(3) * 2.0).tolist()
        config.system.barostat_velocity = (np.eye(3) * 0.5).tolist()
        config.system.barostat_thermostat_dof = [0.5, 0.6]
        return config

    def test_pydantic_roundtrip(self):
        config = self._filled_config()
        restored = PyRAIConfig.model_validate(config.model_dump())
        self.assertEqual(restored.system.thermostat_dof,
                         config.system.thermostat_dof)
        np.testing.assert_allclose(restored.system.barostat_strain, np.eye(3))
        np.testing.assert_allclose(restored.system.barostat_momentum,
                                   np.eye(3) * 2.0)
        np.testing.assert_allclose(restored.system.barostat_velocity,
                                   np.eye(3) * 0.5)
        self.assertEqual(restored.system.barostat_thermostat_dof, [0.5, 0.6])

    def test_json_roundtrip(self):
        config = self._filled_config()
        restored = PyRAIConfig.model_validate_json(config.model_dump_json())
        self.assertEqual(restored.system.thermostat_dof,
                         [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(restored.system.barostat_strain, np.eye(3))


if __name__ == '__main__':
    unittest.main()
