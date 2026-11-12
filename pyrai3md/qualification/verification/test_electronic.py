"""
======================================================================
This module tests the multi-center electronic dispatch layer of the
NAMD workflow (dev/MULTICENTER_DEVELOPMENT.md, refactor R5; the layer
lives in workflow/integrator/electronic.py).

The dispatch layer sits between the surface_hopping factory and the
unchanged FSSH/GSH/LZSH cores. These tests lock the R5 gate:

- the factory binds an ElectronicStepper with one ElectronicBinding
  per center for C>1 (a single transparent binding for C=1), each
  binding wiring the right algorithm class onto a CenterView;
- each CenterView serves its OWN center framing: a distinct local
  state count / active state, coordinates zero-masked to the center
  region (the user's coordinate-mask rule), and gradient / NAC
  projected to the global atom axis with a zero mask outside the
  center;
- the write-back is center-local: a velocity adjustment lands only on
  the center's own atoms (so a global isotropic rescale can never
  touch another center) and an active-state change lands only on the
  center's own slot;
- end to end, a forced hop on center 0 advances center 0 and rescales
  only center 0's atoms, while center 1 (fed a zero NAC) keeps its
  active state and its velocity bit-for-bit.

Sep 08 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md.data.state import PyRAIState
from pyrai3md.data.center import build_center_plan
from pyrai3md.data.runtime.state import RuntimeState
from pyrai3md.schema.config import PyRAIConfig
from pyrai3md.workflow.integrator import surface_hopping
from pyrai3md.workflow.integrator.electronic import (
    CenterView, ElectronicBinding, ElectronicStepper,
)
from pyrai3md.workflow.integrator.fssh import FsshIntegrator
from pyrai3md.workflow.integrator.gsh import GshIntegrator
from pyrai3md.workflow.integrator.lzsh import LzshIntegrator


# A fixed 5-atom geometry with distinct non-zero coordinates so every
# zero-mask assertion probes a non-trivial component.
_COORDS_5 = [[0.10, 0.20, 0.30],
             [0.30, -0.40, 0.50],
             [-0.20, 0.10, 0.50],
             [0.60, -0.30, 0.20],
             [0.10, 0.20, -0.40]]


def _distinct_state_c2(coords=_COORDS_5):
    """A C=2 state with DISTINCT state axes: center 0 -> 4 states
    (high atoms [0, 1], active root 1); center 1 -> 3 states (high
    atoms [2, 3, 4], active root 2). Mirrors the R4f gate plan so the
    view framing is exercised on never-broadcast local axes."""
    config = PyRAIConfig()
    config.system.atoms = ['H'] * 5
    config.system.coordinates = [list(r) for r in coords]
    config.system.velocities = [[0.0, 0.0, 0.0]] * 5
    config.system.mass = [1.0] * 5
    config.system.natom = 5
    config.dynamics.history = 4

    state = PyRAIState(runtime=RuntimeState(total_steps=10))
    state.pmd = config
    plan = build_center_plan({
        'highlevel': [[0, 1], [2, 3, 4]],
        'ci': [[2, 2], [3]],
        'spin': [[0, 1], [0]],
        'root': [1, 2],
    })
    state.bind_partition_plan(plan)
    return state


class TestElectronicFactoryBinding(unittest.TestCase):
    """The surface_hopping factory dispatches C>1 through an
    ElectronicStepper with one algorithm-on-view binding per center."""

    def test_c2_binds_one_view_binding_per_center(self):
        state = _distinct_state_c2()
        for method, cls in (('fssh', FsshIntegrator),
                            ('gsh', GshIntegrator),
                            ('lzsh', LzshIntegrator)):
            stepper = surface_hopping(method, state)
            self.assertIsInstance(stepper, ElectronicStepper)
            self.assertEqual(stepper.n_center, 2)
            self.assertFalse(stepper.transparent)
            for c in range(2):
                binding = stepper.binding(c)
                self.assertIsInstance(binding, ElectronicBinding)
                self.assertEqual(binding.center_index, c)
                self.assertFalse(binding.transparent)
                self.assertIsInstance(binding.view, CenterView)
                self.assertIsInstance(binding.algorithm, cls)

    def test_c1_stays_transparent(self):
        config = PyRAIConfig()
        config.system.atoms = ['C', 'H']
        config.system.coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        config.system.velocities = [[0.0, 0.0, 0.0]] * 2
        config.system.mass = [12.0, 1.0]
        config.system.natom = 2
        config.system.nstate = 2
        state = PyRAIState(runtime=RuntimeState(total_steps=10))
        state.pmd = config

        stepper = surface_hopping('fssh', state)
        self.assertIsInstance(stepper, ElectronicStepper)
        self.assertEqual(stepper.n_center, 1)
        self.assertTrue(stepper.transparent)
        binding = stepper.binding(0)
        self.assertTrue(binding.transparent)
        # The transparent view IS the real state (bit-for-bit legacy).
        self.assertIs(binding.view, state)
        self.assertIsInstance(binding.algorithm, FsshIntegrator)


class TestCenterViewFraming(unittest.TestCase):
    """Each CenterView serves its own center's framing: distinct local
    state count / active state, zero-masked coordinates, and gradient
    / NAC projected to the global axis with a zero mask outside."""

    def setUp(self):
        self.state = _distinct_state_c2()
        self.stepper = surface_hopping('fssh', self.state)
        self.view0 = self.stepper.binding(0).view
        self.view1 = self.stepper.binding(1).view

    def test_distinct_local_state_count_and_active_state(self):
        # center 0: 4 states, active root 1; center 1: 3 states, root 2.
        self.assertEqual(self.view0.nstate, 4)
        self.assertEqual(self.view1.nstate, 3)
        self.assertEqual(self.view0.current_state, 1)
        self.assertEqual(self.view1.current_state, 2)
        # The local axes are never broadcast into a common sum(S_c).
        self.assertEqual(self.state.get_active_configuration(), (1, 2))

    def test_coordinates_zero_masked_to_center_region(self):
        original = np.asarray(_COORDS_5, dtype=np.float64)
        c0 = np.asarray(self.view0.coordinates, dtype=np.float64)
        c1 = np.asarray(self.view1.coordinates, dtype=np.float64)
        self.assertEqual(c0.shape, (5, 3))
        self.assertEqual(c1.shape, (5, 3))
        # center 0 keeps atoms [0, 1]; everything else is masked to zero.
        np.testing.assert_allclose(c0[0:2], original[0:2])
        np.testing.assert_allclose(c0[2:5], np.zeros((3, 3)))
        # center 1 keeps atoms [2, 3, 4]; everything else is masked.
        np.testing.assert_allclose(c1[2:5], original[2:5])
        np.testing.assert_allclose(c1[0:2], np.zeros((2, 3)))

    def test_gradient_and_nac_projected_with_zero_mask(self):
        # center 0 local framing: gradient (S=4, A_c=2, 3), nac (P=1, 2, 3)
        g0 = np.arange(4 * 2 * 3, dtype=np.float64).reshape(4, 2, 3) * 0.01
        n0 = np.ones((1, 2, 3), dtype=np.float64)
        self.state.set_center_electronic(0, gradient=g0.tolist(),
                                         nac=n0.tolist())
        result0 = self.view0.result
        grad = np.asarray(result0.get('gradient'), dtype=np.float64)
        nac = np.asarray(result0.get('nac'), dtype=np.float64)
        # Projected to the GLOBAL atom axis (5 atoms), zero outside [0, 1].
        self.assertEqual(grad.shape, (4, 5, 3))
        self.assertEqual(nac.shape, (1, 5, 3))
        np.testing.assert_allclose(grad[:, 0:2, :], g0)
        np.testing.assert_allclose(grad[:, 2:5, :], np.zeros((4, 3, 3)))
        np.testing.assert_allclose(nac[:, 0:2, :], n0)
        np.testing.assert_allclose(nac[:, 2:5, :], np.zeros((1, 3, 3)))

        # center 1 local framing: gradient (S=3, A_c=3, 3) on atoms [2,3,4]
        g1 = np.arange(3 * 3 * 3, dtype=np.float64).reshape(3, 3, 3) * 0.02
        self.state.set_center_electronic(1, gradient=g1.tolist())
        grad1 = np.asarray(self.view1.result.get('gradient'),
                           dtype=np.float64)
        self.assertEqual(grad1.shape, (3, 5, 3))
        np.testing.assert_allclose(grad1[:, 2:5, :], g1)
        np.testing.assert_allclose(grad1[:, 0:2, :], np.zeros((3, 2, 3)))


    def test_an_empty_nac_carrier_projects_to_no_nac(self):
        # `nactype: ktdc` derives its couplings from the curvature, and a
        # provider that reports no coupling rows leaves an empty array on
        # the carrier. That is "none", not a malformed (A_c, 3) array;
        # the projection used to raise `got (0,)` and kill
        # `multicenter_md` at its first hop (#466).
        g0 = np.zeros((1, 2, 3), dtype=np.float64)
        self.state.set_center_electronic(0, gradient=g0.tolist(), nac=[])
        result = self.view0.result
        self.assertIn('gradient', result)
        self.assertNotIn('nac', result)

    def test_a_non_empty_malformed_nac_is_still_refused(self):
        self.state.set_center_electronic(0, nac=[[1.0, 2.0]])
        with self.assertRaises(ValueError):
            _ = self.view0.result


class TestBindingWritebackLocality(unittest.TestCase):
    """The captured write-back lands on one center only: a velocity
    adjustment is restricted to the center's atoms and an active-state
    change to the center's slot."""

    def setUp(self):
        self.state = _distinct_state_c2()
        self.stepper = surface_hopping('fssh', self.state)

    def test_apply_velocity_restricted_to_center_atoms(self):
        original = np.asarray(self.state.velocities, dtype=np.float64)
        original = original.reshape(-1, 3)
        # A crafted GLOBAL velocity an isotropic rescale would produce
        # (every atom changed); only center 0's atoms may land.
        crafted = (np.arange(5 * 3, dtype=np.float64).reshape(5, 3) + 1.0)
        binding0 = self.stepper.binding(0)
        binding0._apply_velocity(self.state, crafted.tolist())

        after = np.asarray(self.state.velocities, dtype=np.float64)
        after = after.reshape(-1, 3)
        # center 0 atoms [0, 1] take the adjustment ...
        np.testing.assert_allclose(after[0:2], crafted[0:2])
        # ... every other center's atoms stay bit-for-bit untouched.
        np.testing.assert_allclose(after[2:5], original[2:5])

    def test_apply_active_restricted_to_center_slot(self):
        binding0 = self.stepper.binding(0)
        binding0._apply_active(self.state, 3)
        # center 0 advances to local state 3; center 1 keeps root 2.
        self.assertEqual(self.state.get_active_configuration(), (3, 2))

        binding1 = self.stepper.binding(1)
        binding1._apply_active(self.state, 0)
        # center 1 moves to 0; center 0 stays where it was.
        self.assertEqual(self.state.get_active_configuration(), (3, 0))


# ---- End-to-end forced-hop isolation (FSSH deterministic recipe) ----

# center 0 atoms [0, 1]: the proven weak V.NAC coupling that drives a
# forced hop; center 1 atoms [2, 3]: a spectator velocity that a global
# isotropic rescale WOULD touch if the write-back were not masked.
_FSSH_VELOS = [[0.003, 0.0, 0.0], [0.0, 0.003, 0.0],
               [0.005, 0.0, 0.0], [0.0, 0.005, 0.0]]
_FSSH_NAC0 = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]   # (P=1, A_c=2, 3)
_FSSH_NAC1 = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]   # zero -> no coupling
_FSSH_ENERGY = [0.0, 0.05]
_FSSH_WARMUP = 4


def _fssh_c2_state():
    """A C=2 state of two 2-state singlet centers wired for the FSSH
    deterministic forced-hop recipe: center 0 (atoms [0, 1]) carries a
    non-zero NAC and hops; center 1 (atoms [2, 3]) carries a zero NAC
    and must stay put. adjust = 1 makes the hop a GLOBAL isotropic
    rescale, so only the region-masked write-back keeps center 1's
    velocity intact."""
    config = PyRAIConfig()
    config.system.atoms = ['C', 'H', 'C', 'H']
    config.system.coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                 [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    config.system.velocities = [list(v) for v in _FSSH_VELOS]
    config.system.mass = [12.0, 1.0, 12.0, 1.0]
    config.system.natom = 4
    config.dynamics.history = _FSSH_WARMUP
    for key, value in (('substep', 10), ('integrate', 1),
                       ('fssh_force_hop', 1), ('deco', 'OFF'),
                       ('nactype', 'nac'), ('adjust', 1), ('reflect', 1),
                       ('maxh', 1), ('rk4', '1'),
                       ('state_population', [1.0, 0.0])):
        setattr(config.dynamics, key, value)

    state = PyRAIState(runtime=RuntimeState(total_steps=10))
    state.pmd = config
    plan = build_center_plan({
        'highlevel': [[0, 1], [2, 3]],
        'ci': [[2], [2]],
        'spin': [[0], [0]],
        'coupling': [[[0, 1]], [[0, 1]]],
        'root': [0, 0],
    })
    state.bind_partition_plan(plan)

    # Prime WARMUP frames of constant-gap history per center; the live
    # electronic carrier doubles as the current step (the view reads it
    # through center_electronic).
    for _ in range(_FSSH_WARMUP):
        state.set_center_electronic(0, energy=list(_FSSH_ENERGY),
                                    nac=_FSSH_NAC0)
        state.set_center_electronic(1, energy=list(_FSSH_ENERGY),
                                    nac=_FSSH_NAC1)
        state.update_history()
    return state


class TestMultiCenterHopIsolation(unittest.TestCase):
    """End to end: a forced hop on center 0 advances center 0 and
    rescales only its own atoms; center 1 (zero NAC) keeps its active
    state and its velocity bit-for-bit."""

    def test_center0_hop_leaves_center1_untouched(self):
        state = _fssh_c2_state()
        velo_before = np.asarray(state.velocities, dtype=np.float64)
        velo_before = velo_before.reshape(-1, 3)
        self.assertEqual(state.get_active_configuration(), (0, 0))

        stepper = surface_hopping('fssh', state)
        self.assertEqual(stepper.n_center, 2)
        np.random.seed(11)
        stepper(state)

        # center 0 hopped 0 -> 1; center 1 stayed on 0.
        self.assertEqual(state.get_active_configuration(), (1, 0))

        velo_after = np.asarray(state.velocities, dtype=np.float64)
        velo_after = velo_after.reshape(-1, 3)
        # center 0's atoms were rescaled by the hop (adjust = 1).
        self.assertFalse(np.allclose(velo_after[0:2], velo_before[0:2]))
        # center 1's atoms are bit-for-bit unchanged: the global
        # isotropic rescale never crossed the center boundary.
        np.testing.assert_array_equal(velo_after[2:4], velo_before[2:4])

    def test_center0_density_persisted_per_center(self):
        state = _fssh_c2_state()
        stepper = surface_hopping('fssh', state)
        np.random.seed(11)
        stepper(state)
        # The hop persisted center 0's coherence into its OWN carrier.
        self.assertIsNotNone(state.center_electronic(0).get('density'))


class TestCenterViewFreshness(unittest.TestCase):
    """The view serves the CURRENT step's per-center arrays and active
    state, never an init-cached reference (docs §13.3-14)."""

    def test_view_result_reflects_updated_center_data(self):
        state = _distinct_state_c2()
        stepper = surface_hopping('fssh', state)
        view0 = stepper.binding(0).view
        # One evaluation's energies, read through the view.
        state.set_center_electronic(0, energy=[0.0, 0.1])
        np.testing.assert_allclose(
            np.asarray(view0.result.get('energy')), [0.0, 0.1])
        # A new evaluation replaces the center's arrays; the view must
        # serve the NEW values (no stale initialization-time reference).
        state.set_center_electronic(0, energy=[0.2, 0.3])
        np.testing.assert_allclose(
            np.asarray(view0.result.get('energy')), [0.2, 0.3])

    def test_view_current_state_tracks_active_configuration(self):
        state = _distinct_state_c2()
        stepper = surface_hopping('fssh', state)
        view0 = stepper.binding(0).view
        view1 = stepper.binding(1).view
        # Seeded roots: center 0 -> 1, center 1 -> 2.
        self.assertEqual(view0.current_state, 1)
        self.assertEqual(view1.current_state, 2)
        # A hop on center 0 is reflected on the next read; center 1 is
        # untouched (the view reads the live active configuration).
        state.active_surface['active_configuration'] = (3, 2)
        self.assertEqual(view0.current_state, 3)
        self.assertEqual(view1.current_state, 2)


if __name__ == '__main__':
    unittest.main()
