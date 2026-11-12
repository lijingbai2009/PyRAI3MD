"""
======================================================================
Test suite for the data core state object (pyrai3md/data/state.py).

Tests cover:
- PyRAIState history API: bounded snapshot list, chronological
  order, slice-to-last-N on overflow, copy semantics, missing-step
  handling
- to_dict() completeness: runtime, history and dispatch fields
- StateSerializer round trip preserving the history list
- Electronic state governance: current_state authoritative write
  via update_system, read-only root property, serialization round
  trip, legacy runtime dict compatibility

Aug 08 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import pickle
import tempfile
import unittest

import numpy as np

from pyrai3md.schema.optimizer_spec import OPTIMIZER_DECLARATIONS


def make_state(history_limit=4):
    """Build a minimal PyRAIState with a pmd attached."""
    from pyrai3md.data.state import PyRAIState
    from pyrai3md.data.runtime.state import RuntimeState
    from pyrai3md.schema.config import PyRAIConfig

    config = PyRAIConfig()
    config.system.atoms = ['H', 'H']
    config.system.coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
    config.dynamics.history = history_limit
    state = PyRAIState(runtime=RuntimeState(total_steps=10))
    state.pmd = config
    return state


class TestPyRAIStateHistory(unittest.TestCase):
    """Test the official PyRAIState history API."""

    def test_initial_state_empty(self):
        state = make_state()
        self.assertEqual(state.history_size, 0)
        self.assertEqual(state.history_size, 0)
        self.assertEqual(state.get_history(), [])
        self.assertIsNone(state.get_history_step(-1))

    def test_update_history_appends_chronological(self):
        state = make_state()
        for step in range(3):
            state.set_result({'energy': [float(step)]})
            state.update_history()
        self.assertEqual(state.history_size, 3)
        self.assertGreater(state.history_size, 0)
        energies = [s['results']['energy'] for s in state.get_history()]
        self.assertEqual(energies, [[0.0], [1.0], [2.0]])

    def test_overflow_slices_last_n(self):
        state = make_state(history_limit=4)
        for step in range(5):
            state.set_result({'energy': [float(step)]})
            state.update_history()
        self.assertEqual(state.history_size, 4)
        energies = [s['results']['energy'] for s in state.get_history()]
        # t0 dropped, [t1, t2, t3, t4] kept in chronological order
        self.assertEqual(energies, [[1.0], [2.0], [3.0], [4.0]])
        newest = state.get_history_step(-1)
        self.assertEqual(newest['results']['energy'], [4.0])

    def test_get_history_n(self):
        state = make_state(history_limit=10)
        for step in range(5):
            state.set_result({'energy': [float(step)]})
            state.update_history()
        last_two = state.get_history(2)
        self.assertEqual(len(last_two), 2)
        self.assertEqual(last_two[0]['results']['energy'], [3.0])
        self.assertEqual(last_two[1]['results']['energy'], [4.0])
        # Requesting more than stored returns what exists (no fake data)
        self.assertEqual(len(state.get_history(10)), 5)
        self.assertEqual(state.get_history(0), [])

    def test_get_history_step_relative_index(self):
        state = make_state()
        for step in range(3):
            state.set_result({'energy': [float(step)]})
            state.update_history()
        self.assertEqual(state.get_history_step(-1)['results']['energy'], [2.0])
        self.assertEqual(state.get_history_step(-2)['results']['energy'], [1.0])
        self.assertEqual(state.get_history_step(-3)['results']['energy'], [0.0])
        # Missing steps return None; workflows decide how to handle
        self.assertIsNone(state.get_history_step(-4))
        self.assertIsNone(state.get_history_step(3))

    def test_get_history_returns_copies(self):
        state = make_state()
        state.set_result({'energy': [1.0]})
        state.update_history()
        history = state.get_history()
        history[0]['results']['energy'] = [99.0]
        step = state.get_history_step(-1)
        self.assertEqual(step['results']['energy'], [1.0])

    def test_get_history_step_read_only_view(self):
        # copy=False returns the internal snapshot reference itself
        # (hot-loop readers like the FSSH integrator avoid the
        # per-step deep copy); copy=True keeps the deep-copy default.
        state = make_state()
        state.set_result({'energy': [1.0]})
        state.update_history()
        view = state.get_history_step(-1, copy=False)
        self.assertIs(view, state._history[-1])
        copied = state.get_history_step(-1)
        self.assertIsNot(copied, state._history[-1])
        self.assertEqual(copied['results']['energy'],
                         view['results']['energy'])
        # Out-of-range views stay None.
        self.assertIsNone(state.get_history_step(-2, copy=False))

    def test_snapshot_isolated_from_later_pmd_mutation(self):
        state = make_state()
        state.set_result({'energy': [1.0]})
        state.update_history()
        # Advance the current state after the snapshot
        state.update_system({'coordinates': [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]})
        state.set_result({'energy': [2.0]})
        snapshot = state.get_history_step(-1)
        self.assertEqual(snapshot['results']['energy'], [1.0])
        self.assertEqual(snapshot['system']['coordinates'][0], [0.0, 0.0, 0.0])

    def test_update_history_without_pmd_raises(self):
        from pyrai3md.data.state import PyRAIState
        state = PyRAIState()
        with self.assertRaises(ValueError):
            state.update_history()

    def test_clear_history(self):
        state = make_state()
        state.set_result({'energy': [1.0]})
        state.update_history()
        self.assertGreater(state.history_size, 0)
        state.clear_history()
        self.assertEqual(state.history_size, 0)
        self.assertEqual(state.history_size, 0)

    def test_history_limit_from_dynamics(self):
        state = make_state(history_limit=7)
        self.assertEqual(state.history_limit, 7)

    def test_history_limit_fallback_without_pmd(self):
        from pyrai3md.data.state import PyRAIState
        state = PyRAIState()
        self.assertEqual(state.history_limit, 4)

    def test_limit_shrink_converges_on_next_update(self):
        state = make_state(history_limit=6)
        for step in range(5):
            state.set_result({'energy': [float(step)]})
            state.update_history()
        state.pmd.dynamics.history = 4
        state.set_result({'energy': [5.0]})
        state.update_history()
        self.assertEqual(state.history_size, 4)
        energies = [s['results']['energy'] for s in state.get_history()]
        self.assertEqual(energies, [[2.0], [3.0], [4.0], [5.0]])


class TestMassAu(unittest.TestCase):
    """Central mass-unit access point on the data side."""

    def _state_with_mass(self, mass, natom):
        from pyrai3md.data.state import PyRAIState
        from pyrai3md.data.runtime.state import RuntimeState
        from pyrai3md.schema.config import PyRAIConfig

        config = PyRAIConfig()
        config.system.atoms = ['C', 'H']
        config.system.coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        config.system.mass = list(mass)
        config.system.natom = natom
        state = PyRAIState(runtime=RuntimeState(total_steps=10))
        state.pmd = config
        return state

    def test_mass_au_converts_through_units(self):
        from pyrai3md.utilities.units import AMU_TO_AU
        state = self._state_with_mass([12.0, 1.0], 2)
        np.testing.assert_allclose(state.mass_au,
                                   np.array([12.0, 1.0]) * AMU_TO_AU)

    def test_mass_au_none_without_pmd(self):
        state = make_state()
        state.pmd = None
        self.assertIsNone(state.mass_au)

    def test_mass_au_none_when_missing(self):
        state = self._state_with_mass([], 2)
        self.assertIsNone(state.mass_au)

    def test_mass_au_none_on_count_mismatch(self):
        state = self._state_with_mass([12.0], 2)
        self.assertIsNone(state.mass_au)

    def test_mass_au_none_on_non_positive(self):
        state = self._state_with_mass([12.0, 0.0], 2)
        self.assertIsNone(state.mass_au)


class TestPyRAIStateToDict(unittest.TestCase):
    """Test to_dict() completeness after the history addition."""

    def test_to_dict_includes_runtime_and_history(self):
        state = make_state()
        state.set_result({'energy': [1.0]})
        state.update_history()
        data = state.to_dict()
        self.assertIsNotNone(data['runtime'])
        self.assertEqual(data['runtime']['total_steps'], 10)
        self.assertEqual(len(data['history']), 1)
        self.assertEqual(data['history'][0]['results']['energy'], [1.0])
        self.assertEqual(data['history_limit'], 4)

    def test_to_dict_history_is_copy(self):
        state = make_state()
        state.set_result({'energy': [1.0]})
        state.update_history()
        data = state.to_dict()
        data['history'][0]['results']['energy'] = [99.0]
        self.assertEqual(state.get_history_step(-1)['results']['energy'], [1.0])

    def test_to_dict_task_dispatch_fields(self):
        state = make_state()
        state.add_task('theory', {
            'role': 'qm', 'task': 0, 'task_id': 'qm',
            'layer_id': 0, 'static_id': 'qm_layer0_mock',
            'contribution': 1, 'region': 'all', 'region_indices': [],
            'region_charge': None, 'skill_name': 'mock_skill',
            'capabilities': ['energy'], 'config': None,
            'multitask': True, 'remotetask': False,
        })
        data = state.to_dict()
        entry = data['theory'][0]
        self.assertTrue(entry['multitask'])
        self.assertFalse(entry['remotetask'])


class TestHistorySerialization(unittest.TestCase):
    """Test that history travels with the serialized state."""

    def test_serializer_round_trip_preserves_history(self):
        from pyrai3md.data.runtime.serializer import save_state, load_state

        state = make_state(history_limit=4)
        for step in range(5):
            state.set_result({'energy': [float(step)]})
            state.update_history()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'state.rt')
            save_state(state, path)
            restored, _ = load_state(path)

        self.assertEqual(restored.history_size, 4)
        energies = [s['results']['energy'] for s in restored.get_history()]
        self.assertEqual(energies, [[1.0], [2.0], [3.0], [4.0]])
        self.assertEqual(restored.get_history_step(-1)['results']['energy'], [4.0])


class TestPeriodicImagePersistence(unittest.TestCase):
    """The crossing count is state, so every persistence path keeps it.

    A restart that lost the counts would resume with every atom filed
    under the wrong image, which is the same defect a warm-starting
    periodic engine hits when it is handed coordinates without them
    (dev/methods/PBC_DEVELOPMENT.md stage C).
    """

    LATTICE = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]

    def _state(self, image):
        state = make_state(history_limit=4)
        state.update_system({'pbc': [1, 1, 1],
                             'lattice_vector': self.LATTICE,
                             'volume': 1000.0,
                             'image': image})
        return state

    def test_checkpoint_round_trip_preserves_image(self):
        from pyrai3md.data.runtime.serializer import save_state, load_state

        state = self._state([[1, 0, -2], [0, 3, 0]])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'state.rt')
            save_state(state, path)
            restored, _ = load_state(path)
        self.assertEqual([[int(v) for v in row]
                          for row in restored.pmd.system.image],
                         [[1, 0, -2], [0, 3, 0]])

    def test_each_history_frame_carries_its_own_image(self):
        # The count moves between frames, so a consumer diffing two of
        # them needs the value each frame held, not the live one.
        state = self._state([[0, 0, 0], [0, 0, 0]])
        state.set_result({'energy': [0.0]})
        state.update_history()
        state.update_system({'image': [[1, 0, 0], [0, 0, 0]]})
        state.set_result({'energy': [0.1]})
        state.update_history()

        frames = state.get_history()
        self.assertEqual(frames[0]['system']['image'], [[0, 0, 0], [0, 0, 0]])
        self.assertEqual(frames[1]['system']['image'], [[1, 0, 0], [0, 0, 0]])

    def test_flattened_skill_config_keeps_image(self):
        # A standalone skill consumes the flattened form; dropping the
        # counts there is how a skill ends up evaluating the wrong image.
        from pyrai3md.data.runtime.serializer import _flatten_config

        state = self._state([[2, -1, 0], [0, 0, 1]])
        flat = _flatten_config(state.pmd.model_dump())
        self.assertEqual(flat['image'], [[2, -1, 0], [0, 0, 1]])


class TestElectronicStateGovernance(unittest.TestCase):
    """Test current_state as the sole authoritative active-state store."""

    def test_update_system_current_state_matches_property(self):
        state = make_state()
        self.assertEqual(state.current_state, 0)
        state.update_system({'current_state': 2})
        self.assertEqual(state.current_state, 2)
        self.assertEqual(state.pmd.system.current_state, 2)

    def test_root_property_passes_through_dynamics(self):
        state = make_state()
        state.pmd.dynamics.root = 1
        self.assertEqual(state.root, 1)

    def test_activestate_property_passes_through_dynamics(self):
        state = make_state()
        self.assertEqual(state.activestate, 0)
        state.pmd.dynamics.activestate = 1
        self.assertEqual(state.activestate, 1)

    def test_fallback_values_without_pmd(self):
        from pyrai3md.data.state import PyRAIState
        state = PyRAIState()
        self.assertEqual(state.current_state, 0)
        self.assertEqual(state.root, 0)
        self.assertEqual(state.activestate, 0)

    def test_serializer_round_trip_preserves_current_state(self):
        from pyrai3md.data.runtime.serializer import save_state, load_state

        state = make_state()
        state.update_system({'current_state': 3})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'state.rt')
            save_state(state, path)
            restored, _ = load_state(path)

        self.assertEqual(restored.current_state, 3)
        self.assertEqual(restored.pmd.system.current_state, 3)

    def test_legacy_runtime_dict_ignores_current_electronic_state(self):
        from pyrai3md.data.runtime.state import RuntimeState
        runtime = RuntimeState.from_dict({
            'status': 'running',
            'current_step': 5,
            'current_electronic_state': 2,
        })
        self.assertEqual(runtime.current_step, 5)
        self.assertEqual(runtime.status, 'running')
        self.assertFalse(hasattr(runtime, 'current_electronic_state'))


class TestTaskPropertyRequest(unittest.TestCase):
    """R2 per-task static property request (docs 6.3).

    _task_property_request replaces the single global target_states:
    the high-level provider H(c) requests its OWN center's local
    states and coupling pairs, while a reference correction and the
    L(real) baseline read only their fixed reference_state and never
    another center's active state.
    """

    def _plan(self):
        from pyrai3md.data.center import build_center_plan
        # C=1, ci=[2,2] spin=[0,1] -> nstate=4 (the R0 snapshot).
        return build_center_plan({'ci': [2, 2], 'spin': [0, 1],
                                  'root': 0})

    def test_provider_requests_full_center_states(self):
        from pyrai3md.planning.assembly import _task_property_request
        plan = self._plan()
        es = plan.center(0).electronic_spec
        descriptor = {'center_index': 0, 'provides_electronic': True,
                      'state_policy': 'active'}
        spec = _task_property_request(descriptor, plan)
        # The provider follows its own center: every local state plus
        # the center's coupling pair maps (never a global target).
        self.assertEqual(spec['target_states'], list(range(es.nstate)))
        self.assertEqual(spec['nac_pairs'],
                         [list(p) for p in es.nac_pair_map])
        self.assertEqual(spec['soc_pairs'],
                         [list(p) for p in es.soc_pair_map])
        self.assertIsNone(spec['evaluation_id'])

    def test_reference_correction_reads_own_state_only(self):
        from pyrai3md.planning.assembly import _task_property_request
        plan = self._plan()
        descriptor = {'center_index': 0, 'provides_electronic': False,
                      'state_policy': 'reference', 'reference_state': 0}
        spec = _task_property_request(descriptor, plan)
        # A reference term never requests the center's active states.
        self.assertEqual(spec['target_states'], [0])
        self.assertEqual(spec['nac_pairs'], [])
        self.assertEqual(spec['soc_pairs'], [])

    def test_real_baseline_unbound_no_active_request(self):
        from pyrai3md.planning.assembly import _task_property_request
        plan = self._plan()
        descriptor = {'center_index': None, 'provides_electronic': False,
                      'state_policy': 'reference', 'reference_state': 0}
        spec = _task_property_request(descriptor, plan)
        self.assertEqual(spec['target_states'], [0])
        self.assertEqual(spec['nac_pairs'], [])

    def test_no_center_plan_falls_back(self):
        from pyrai3md.planning.assembly import _task_property_request
        # A mock queue without a center plan: the provider request
        # stays empty rather than guessing states.
        descriptor = {'center_index': 0, 'provides_electronic': True,
                      'state_policy': 'active'}
        spec = _task_property_request(descriptor, None)
        self.assertIsNone(spec['target_states'])
        self.assertEqual(spec['nac_pairs'], [])
        self.assertEqual(spec['soc_pairs'], [])


class TestR4aCenterDataInterface(unittest.TestCase):
    """R4a unified center data carriers on PyRAIState (docs r4a).

    Locks in: the bound partition plan sizes the per-center electronic
    carrier and seeds the active surface; nstate / current_state are a
    single-center projection that raises for C>1 (no fabricated global
    state axis); the bounded center read interface; and stage/commit as
    the atomic data-model writeback.
    """

    def _c2_plan(self):
        from pyrai3md.data.center import build_center_plan
        # Two centers with DISTINCT state counts (no broadcast):
        # center 0 -> ci [2, 2] = 4 states, root 0;
        # center 1 -> ci [3]    = 3 states, root 2.
        return build_center_plan({
            'highlevel': [[0, 1], [2, 3, 4]],
            'ci': [[2, 2], [3]],
            'spin': [[0, 1], [0]],
            'root': [0, 2],
        })

    def test_unbound_state_is_single_center(self):
        state = make_state()
        self.assertIsNone(state.partition_plan)
        self.assertEqual(state.n_center, 1)
        # C=1 projection stays valid before any plan is bound.
        self.assertEqual(state.nstate, 1)
        self.assertEqual(state.current_state, 0)
        self.assertEqual(state.get_active_configuration(), (0,))

    def test_unbound_center_read_raises(self):
        from pyrai3md.data.center import CenterPlanError
        state = make_state()
        with self.assertRaises(CenterPlanError):
            state.center(0)

    def test_bind_sizes_electronic_and_seeds_active_surface(self):
        state = make_state()
        state.bind_partition_plan(self._c2_plan())
        self.assertEqual(state.n_center, 2)
        self.assertEqual(len(state.electronic), 2)
        # Each center's carrier starts empty (independent arrays, no
        # padding into a fake sum(S_c) axis).
        for c in range(2):
            carrier = state.center_electronic(c)
            self.assertIsNone(carrier['energy'])
            self.assertIsNone(carrier['gradient'])
        # Active surface seeded from each center's initial_root.
        self.assertEqual(state.get_active_configuration(), (0, 2))
        self.assertIsNone(state.get_active_energy())

    def test_multi_center_nstate_and_current_state_raise(self):
        from pyrai3md.data.center import CenterPlanError
        state = make_state()
        state.bind_partition_plan(self._c2_plan())
        with self.assertRaises(CenterPlanError):
            _ = state.nstate
        with self.assertRaises(CenterPlanError):
            _ = state.current_state
        # The per-center state count is read through the plan instead.
        self.assertEqual(state.center(0).nstate, 4)
        self.assertEqual(state.center(1).nstate, 3)

    def test_center_electronic_bounds_and_type(self):
        from pyrai3md.data.center import CenterPlanError
        state = make_state()
        state.bind_partition_plan(self._c2_plan())
        with self.assertRaises(CenterPlanError):
            state.center_electronic(2)
        with self.assertRaises(CenterPlanError):
            state.center_electronic(True)
        with self.assertRaises(CenterPlanError):
            state.center(5)

    def test_set_center_electronic_writes_only_given_keys(self):
        state = make_state()
        state.bind_partition_plan(self._c2_plan())
        state.set_center_electronic(0, energy=[1.0, 2.0, 3.0, 4.0])
        carrier = state.center_electronic(0)
        self.assertEqual(carrier['energy'], [1.0, 2.0, 3.0, 4.0])
        # Untouched keys stay None; other centers are independent.
        self.assertIsNone(carrier['gradient'])
        self.assertIsNone(state.center_electronic(1)['energy'])

    def test_stage_then_commit_applies_atomically(self):
        state = make_state()
        state.bind_partition_plan(self._c2_plan())
        gradient = [[0.0, 0.0, 0.1]] * 5
        state.stage_evaluation({
            'electronic': {0: {'energy': [1.0, 2.0, 3.0, 4.0]},
                           1: {'energy': [5.0, 6.0, 7.0]}},
            'active_surface': {'active_configuration': (1, 0),
                               'active_energy': -75.5,
                               'active_gradient': gradient,
                               'evaluation_id': 7},
            'continuation': {'next_step': 3},
        })
        # Staging never mutates the committed center data.
        self.assertIsNone(state.center_electronic(0)['energy'])
        state.commit_step()
        self.assertEqual(state.center_electronic(0)['energy'],
                         [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(state.center_electronic(1)['energy'],
                         [5.0, 6.0, 7.0])
        self.assertEqual(state.get_active_configuration(), (1, 0))
        self.assertEqual(state.get_active_energy(), -75.5)
        self.assertEqual(state.continuation, {'next_step': 3})
        # The staging slot is cleared by the commit.
        self.assertIsNone(state._staged_evaluation)

    def test_commit_step_stages_outcome_first(self):
        state = make_state()
        state.bind_partition_plan(self._c2_plan())
        state.commit_step(outcome={'electronic': {1: {'energy': [9.0]}}})
        self.assertEqual(state.center_electronic(1)['energy'], [9.0])
        self.assertIsNone(state._staged_evaluation)

    def test_get_active_gradient_is_float_array(self):
        state = make_state()
        state.bind_partition_plan(self._c2_plan())
        state.commit_step(outcome={
            'active_surface': {'active_configuration': (0, 0),
                               'active_gradient': [[0, 0, 1], [0, 0, 2]],
                               'active_energy': None,
                               'evaluation_id': 1}})
        gradient = state.get_active_gradient()
        self.assertEqual(gradient.dtype, np.float64)
        self.assertEqual(gradient.tolist(), [[0.0, 0.0, 1.0],
                                             [0.0, 0.0, 2.0]])

    def test_nested_ci_flat_scalars_are_neutral(self):
        """r4a deletion gate: the center0 flat projection shim is gone.

        A nested (C>1) input no longer projects center 0 onto the flat
        single-valued scalars; they stay at neutral Pydantic-valid
        defaults while the per-center plan (build_center_plan) remains
        the authoritative source.
        """
        from pyrai3md.schema.normalize import compute_derived_quantities
        data = {'ci': [[2, 2], [3]], 'spin': [[0, 1], [0]],
                'coupling': [], 'atoms': ['H', 'H', 'H', 'H', 'H'],
                'charges': 0, 'multiplicity': 1}
        data = compute_derived_quantities(data)
        self.assertEqual(data['nstate'], 1)
        self.assertEqual(data['state_spin'], [])
        self.assertEqual(data['npair'], 0)
        self.assertEqual(data['nnac'], 0)
        self.assertEqual(data['nsoc'], 0)
        self.assertEqual(data['transition_pair'], [[]])
        self.assertEqual(data['nac_pair_map'], [[]])
        self.assertEqual(data['soc_pair_map'], [[]])

    def test_framework_soc_shape_is_canonical_1d(self):
        """r4a soc unification: the framework result contract is (nsoc,)."""
        from pyrai3md.schema.config import Properties
        self.assertEqual(Properties().soc, [0.0])


class TestR4dHistoryFrame(unittest.TestCase):
    """R4d explicit structured history-frame protocol (docs 3.2/10.2).

    Locks in: update_history writes a structured HistoryFrame (a dict
    subclass) carrying the explicit new-frame identity
    (frame_version/step/time/evaluation_id/plan_fingerprint), the
    PRE-hop nuclear_sample, the per-center centers/electronic_outcome
    and the POST-hop continuation; the legacy pmd-dump keys stay so the
    hot-loop consumers keep reading snapshot['system']/['results']
    unchanged (versioned adaptation layer, deletion gate R5); the
    pre-hop sample and post-hop continuation stay separate; a frame is
    trimmed as one unit with every center sharing step/time; and
    get_history_step(copy=False) keeps returning the read-only internal
    reference the FSSH hot loop depends on.
    """

    def _c2_plan(self):
        from pyrai3md.data.center import build_center_plan
        return build_center_plan({
            'highlevel': [[0, 1], [2, 3, 4]],
            'ci': [[2, 2], [3]],
            'spin': [[0, 1], [0]],
            'root': [0, 2],
        })

    def _bound_c2_state(self):
        plan = self._c2_plan()
        state = make_state()
        state.update_system({
            'atoms': ['H'] * 5,
            'coordinates': [[0.0, 0.0, float(i)] for i in range(5)],
            'velocities': [[0.0, 0.0, 0.1]] * 5})
        state.bind_partition_plan(plan)
        return state, plan

    def test_update_history_writes_structured_frame(self):
        from pyrai3md.data.state import HistoryFrame, HISTORY_FRAME_VERSION
        state = make_state()
        state.set_result({'energy': [1.0]})
        state.update_history()
        frame = state.get_history_step(-1, copy=False)
        self.assertIsInstance(frame, HistoryFrame)
        self.assertIsInstance(frame, dict)
        self.assertEqual(frame.frame_version, HISTORY_FRAME_VERSION)
        self.assertEqual(frame['frame_version'], HISTORY_FRAME_VERSION)

    def test_legacy_pmd_dump_keys_preserved(self):
        # The hot-loop consumers (fssh/gsh/lzsh/verlet/operations) keep
        # reading snapshot['system']/['results']; the structured frame
        # is a superset, never a replacement (deletion gate R5).
        state = make_state()
        state.set_result({'energy': [3.0]})
        state.update_history()
        frame = state.get_history_step(-1)
        self.assertIn('system', frame)
        self.assertIn('results', frame)
        self.assertEqual(frame['results']['energy'], [3.0])
        self.assertEqual(frame['system']['coordinates'][0], [0.0, 0.0, 0.0])

    def test_nuclear_sample_is_pre_hop(self):
        state = make_state()
        state.update_system({'velocities': [[0.0, 0.0, 1.0],
                                            [0.0, 0.0, -1.0]]})
        state.update_history()
        sample = state.get_history_step(-1)['nuclear_sample']
        self.assertEqual(sample['R'].tolist(),
                         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        self.assertEqual(sample['V_pre'].tolist(),
                         [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        self.assertEqual(sample['K_pre'], 0.0)
        # Unbound C=1 -> active_config_pre is the single-center tuple.
        self.assertEqual(sample['active_config_pre'], (0,))

    def test_warmup_when_no_continuation(self):
        # R4 never stages a continuation (R6 commit_step does), so the
        # frame is an explicit warmup frame with a null continuation.
        state = make_state()
        state.update_history()
        frame = state.get_history_step(-1)
        self.assertIsNone(frame['continuation'])
        self.assertTrue(frame['warmup'])
        self.assertTrue(frame.warmup)

    def test_continuation_marks_post_hop_frame(self):
        # Simulate the R6 writeback: a staged continuation flips the
        # frame off warmup and records the post-hop side, kept separate
        # from the pre-hop nuclear_sample.
        state = make_state()
        state.continuation = {'V_post': [[0.0, 0.0, 0.5],
                                         [0.0, 0.0, -0.5]],
                              'active_config_post': (1,),
                              'next_step': 2}
        state.update_history()
        frame = state.get_history_step(-1)
        self.assertFalse(frame['warmup'])
        self.assertFalse(frame.warmup)
        self.assertEqual(frame['continuation']['next_step'], 2)
        self.assertEqual(frame['continuation']['active_config_post'], (1,))
        # The pre-hop sample is untouched by the post-hop continuation.
        self.assertEqual(frame['nuclear_sample']['active_config_pre'], (0,))

    def test_frame_identity_shared_by_centers(self):
        state = make_state()
        state.runtime.current_step = 4
        state.runtime.simulation_time = 0.8
        state.set_result({'energy': [1.0]})
        # evaluation_id comes from the committed active surface.
        state.active_surface = {'active_configuration': (0,),
                                'active_energy': 1.0,
                                'active_gradient': None,
                                'evaluation_id': 9}
        state.update_history()
        frame = state.get_history_step(-1)
        self.assertEqual(frame['step'], 4)
        self.assertEqual(frame['time'], 0.8)
        self.assertEqual(frame['evaluation_id'], 9)
        self.assertEqual(frame.step, 4)
        self.assertEqual(frame.time, 0.8)
        self.assertEqual(frame.evaluation_id, 9)
        # Unbound C=1 carries no plan fingerprint.
        self.assertIsNone(frame['plan_fingerprint'])
        self.assertIsNone(frame.plan_fingerprint)

    def test_multi_center_frame_records_every_center(self):
        state, plan = self._bound_c2_state()
        state.commit_step(outcome={
            'electronic': {0: {'energy': [1.0, 2.0, 3.0, 4.0]},
                           1: {'energy': [5.0, 6.0, 7.0]}},
            'active_surface': {'active_configuration': (1, 0),
                               'active_energy': -75.5,
                               'active_gradient': None,
                               'evaluation_id': 3}})
        state.update_history()
        frame = state.get_history_step(-1)
        self.assertEqual(len(frame['centers']), 2)
        self.assertEqual(frame['plan_fingerprint'], plan.plan_fingerprint)
        self.assertEqual(sorted(frame['electronic_outcome'].keys()), [0, 1])
        self.assertEqual(frame['electronic_outcome'][0]['energy'],
                         [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(frame['electronic_outcome'][1]['energy'],
                         [5.0, 6.0, 7.0])
        # The pre-hop active configuration is the committed 2-tuple and
        # both centers share the frame step/time.
        self.assertEqual(frame['nuclear_sample']['active_config_pre'], (1, 0))
        self.assertEqual(frame['centers'][0]['center_index'], 0)
        self.assertEqual(frame['centers'][1]['center_index'], 1)

    def test_electronic_outcome_isolated_from_later_mutation(self):
        state, _ = self._bound_c2_state()
        state.commit_step(outcome={'electronic': {0: {'energy': [1.0]}},
                                   'active_surface': {
                                       'active_configuration': (0, 0),
                                       'active_energy': None,
                                       'active_gradient': None,
                                       'evaluation_id': 1}})
        state.update_history()
        # Mutate the live center data after the snapshot.
        state.set_center_electronic(0, energy=[99.0])
        frame = state.get_history_step(-1)
        self.assertEqual(frame['electronic_outcome'][0]['energy'], [1.0])

    def test_frame_trimmed_as_one_unit(self):
        state = make_state(history_limit=3)
        for step in range(5):
            state.runtime.current_step = step
            state.set_result({'energy': [float(step)]})
            state.update_history()
        self.assertEqual(state.history_size, 3)
        frames = state.get_history()
        self.assertEqual([f['step'] for f in frames], [2, 3, 4])
        # Every kept frame is still a full structured frame.
        for frame in frames:
            self.assertIn('nuclear_sample', frame)
            self.assertIn('frame_version', frame)

    def test_read_only_reference_contract_preserved(self):
        from pyrai3md.data.state import HistoryFrame
        state = make_state()
        state.update_history()
        view = state.get_history_step(-1, copy=False)
        self.assertIs(view, state._history[-1])
        self.assertIsInstance(view, HistoryFrame)
        copied = state.get_history_step(-1)
        self.assertIsNot(copied, state._history[-1])
        self.assertEqual(copied['frame_version'], view['frame_version'])

    def test_frame_isolated_from_later_pmd_mutation(self):
        state = make_state()
        state.set_result({'energy': [1.0]})
        state.update_history()
        state.update_system({'coordinates': [[1.0, 1.0, 1.0],
                                             [2.0, 2.0, 2.0]]})
        frame = state.get_history_step(-1)
        self.assertEqual(frame['system']['coordinates'][0], [0.0, 0.0, 0.0])
        self.assertEqual(frame['nuclear_sample']['R'].tolist()[0],
                         [0.0, 0.0, 0.0])


class TestR4eSerializerVersioning(unittest.TestCase):
    """R4e versioned serialization + legacy migration (docs 11, r4e).

    Locks in: every payload carries a file-level version block; the
    full center data model (partition_plan / electronic /
    active_surface / continuation) survives a save-load round trip
    while the transient staging slot is never persisted; a legacy file
    (no version block, center attributes absent because unpickling
    bypasses __init__) migrates to the C=1 record; a stale file -- an
    expected-fingerprint mismatch, a history frame recorded under a
    different plan, an incompatible version, or a center-count
    disagreement -- is refused instead of silently resumed; and the
    mutable center data / history are isolated under a deep copy while
    the static plan stays equal.
    """

    def _c2_plan(self):
        from pyrai3md.data.center import build_center_plan
        return build_center_plan({
            'highlevel': [[0, 1], [2, 3, 4]],
            'ci': [[2, 2], [3]],
            'spin': [[0, 1], [0]],
            'root': [0, 2],
        })

    def _bound_c2_state(self):
        plan = self._c2_plan()
        state = make_state()
        state.update_system({
            'atoms': ['H'] * 5,
            'coordinates': [[0.0, 0.0, float(i)] for i in range(5)],
            'velocities': [[0.0, 0.0, 0.1]] * 5})
        state.bind_partition_plan(plan)
        return state, plan

    def _roundtrip(self, state):
        from pyrai3md.data.runtime.serializer import save_state, load_state
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'state.rt')
            save_state(state, path)
            with open(path, 'rb') as handler:
                payload = pickle.load(handler)
            restored, _ = load_state(path)
        return restored, payload

    def _write_legacy(self, state, tmpdir):
        """Pickle a pre-versioning payload: no version block and no
        center attributes (unpickling bypasses __init__, so a legacy
        object simply never had them); the frames are stripped of the
        r4d structured keys so they read as pre-r4d plain dumps."""
        from pyrai3md.data.runtime.serializer import _prepare_state
        prepared = _prepare_state(state)
        for attr in ('partition_plan', 'electronic', 'active_surface',
                     'continuation', '_staged_evaluation'):
            if hasattr(prepared, attr):
                delattr(prepared, attr)
        legacy_frames = []
        for frame in prepared._history:
            plain = dict(frame)
            for key in ('frame_version', 'step', 'time', 'evaluation_id',
                        'plan_fingerprint', 'nuclear_sample', 'centers',
                        'electronic_outcome', 'continuation', 'warmup'):
                plain.pop(key, None)
            legacy_frames.append(plain)
        prepared._history = legacy_frames
        path = os.path.join(tmpdir, 'legacy.rt')
        with open(path, 'wb') as handler:
            pickle.dump({'state': prepared, 'runtime_idx': None}, handler)
        return path

    def test_payload_carries_version_block(self):
        from pyrai3md.data.runtime.serializer import (
            SERIALIZER_FORMAT_VERSION, ELECTRONIC_SEMANTICS_VERSION)
        from pyrai3md.data.center import CENTER_PLAN_VERSION
        from pyrai3md.data.state import HISTORY_FRAME_VERSION
        _, payload = self._roundtrip(make_state())
        self.assertIn('version', payload)
        version = payload['version']
        self.assertEqual(version['format'], SERIALIZER_FORMAT_VERSION)
        self.assertEqual(version['electronic'],
                         ELECTRONIC_SEMANTICS_VERSION)
        # partition/history semantics come from their source of truth.
        self.assertEqual(version['partition'], CENTER_PLAN_VERSION)
        self.assertEqual(version['history'], HISTORY_FRAME_VERSION)

    def test_round_trip_preserves_center_model(self):
        state, plan = self._bound_c2_state()
        state.commit_step(outcome={
            'electronic': {0: {'energy': [1.0, 2.0, 3.0, 4.0]},
                           1: {'energy': [5.0, 6.0, 7.0]}},
            'active_surface': {'active_configuration': (1, 0),
                               'active_energy': -75.5,
                               'active_gradient': None,
                               'evaluation_id': 3},
            'continuation': {'next_step': 2}})
        state.set_result({'energy': [-75.5]})
        state.update_history()
        restored, _ = self._roundtrip(state)
        self.assertEqual(restored.n_center, 2)
        self.assertEqual(restored.partition_plan.plan_fingerprint,
                         plan.plan_fingerprint)
        self.assertEqual(restored.center_electronic(0)['energy'],
                         [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(restored.center_electronic(1)['energy'],
                         [5.0, 6.0, 7.0])
        self.assertEqual(restored.get_active_configuration(), (1, 0))
        self.assertEqual(restored.continuation, {'next_step': 2})
        # The history frame travels with its plan fingerprint.
        self.assertEqual(
            restored.get_history_step(-1)['plan_fingerprint'],
            plan.plan_fingerprint)

    def test_the_charge_store_and_each_centers_coupling_travel(self):
        """The three things only a composite evaluation produces.

        A multi-center QMMM step answers with a charge set fitted inside that
        step (one cold-start round per center), a coupling block per center, and
        an active configuration with one entry per center. A restart that lost
        any of the three resumes the trajectory on a surface the previous step
        never evaluated - and because a frame's energy and forces come out of
        one assembly, the loss reads as a wrong number rather than as an error.
        The engine-side half of this claim is the shipped pair
        `multicenter_qmmm_md_restart.yaml` against
        `multicenter_qmmm_md_restart_base.yaml`, judged frame by frame
        (dev/methods/QMMM_DEVELOPMENT.md section 9); this is the engine-free
        half, on the same three objects.
        """
        from pyrai3md.data.charges import ChargeStore

        state, _plan = self._bound_c2_state()
        store = ChargeStore(5, managed_indices=[0, 1, 2, 3, 4])
        charges = [-0.8, 0.4, 0.3, -0.2, 0.3]
        store.update([0, 1, 2, 3, 4], charges)
        state.charge_store = store
        state.commit_step(outcome={
            'electronic': {0: {'energy': [1.0, 2.0], 'nac': [[0.0, 0.1]]},
                           1: {'energy': [5.0, 6.0, 7.0],
                               'soc': [[0.0, 0.0, 0.25]]}},
            'active_surface': {'active_configuration': (1, 2),
                               'active_energy': -75.0,
                               'active_gradient': None,
                               'evaluation_id': 4},
            'continuation': None})
        state.set_result({'energy': [-75.0]})
        state.update_history()

        restored, _payload = self._roundtrip(state)
        got = restored.charge_store
        self.assertIsNotNone(got, 'the store is a state field, not a run-local')
        self.assertTrue(got.has_values,
                        'an untouched store must not read as zeros')
        self.assertEqual([float(value) for value in got.atom_charges],
                         charges)
        self.assertEqual(restored.center_electronic(0)['nac'], [[0.0, 0.1]])
        self.assertEqual(restored.center_electronic(1)['soc'],
                         [[0.0, 0.0, 0.25]])
        self.assertEqual(restored.get_active_configuration(), (1, 2))

    def test_round_trip_preserves_region_charge_spin(self):
        """CENTER_PLAN_VERSION 1.1: per-center charge/multiplicity and
        the full-system aggregate survive a save-load round trip."""
        from pyrai3md.data.center import build_center_plan
        plan = build_center_plan({
            'highlevel': [[0, 1], [2, 3, 4]],
            'ci': [[1], [1]], 'spin': [[0], [0]],
            'charges': [0, -1], 'multiplicity': [3, 2],
        })
        state = make_state()
        state.update_system({
            'atoms': ['H'] * 5,
            'coordinates': [[0.0, 0.0, float(i)] for i in range(5)],
            'velocities': [[0.0, 0.0, 0.1]] * 5})
        state.bind_partition_plan(plan)
        restored, _ = self._roundtrip(state)
        rp = restored.partition_plan
        self.assertEqual(rp.center(0).charge, 0)
        self.assertEqual(rp.center(0).multiplicity, 3)
        self.assertEqual(rp.center(1).charge, -1)
        self.assertEqual(rp.center(1).multiplicity, 2)
        self.assertEqual(rp.system_charge, -1)
        # triplet + doublet -> (2 + 1) mod 2 + 1 = 2 (doublet)
        self.assertEqual(rp.system_multiplicity, 2)

    def test_staged_evaluation_is_not_persisted(self):
        state, _ = self._bound_c2_state()
        # Stage but do NOT commit: a checkpoint is committed state only.
        state.stage_evaluation({'electronic': {0: {'energy': [9.0]}}})
        restored, _ = self._roundtrip(state)
        self.assertIsNone(restored._staged_evaluation)
        # The un-committed staging never leaked into the center data.
        self.assertIsNone(restored.center_electronic(0)['energy'])

    def test_legacy_file_migrates_to_single_center(self):
        from pyrai3md.data.runtime.serializer import load_state
        state = make_state()
        state.set_result({'energy': [-1.0]})
        state.update_history()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_legacy(state, tmpdir)
            restored, _ = load_state(path)
        # A legacy file is a single-center run migrated to the C=1
        # record: the canonical plan is rebuilt from the restored pmd.
        self.assertEqual(restored.n_center, 1)
        self.assertIsNotNone(restored.partition_plan)
        self.assertEqual(len(restored.electronic), 1)
        # active_surface stays null so the configuration falls back to
        # the saved current_state (never reseeded to root).
        self.assertIsNone(restored.active_surface)
        self.assertEqual(restored.get_active_configuration(), (0,))
        self.assertTrue(hasattr(restored, 'continuation'))
        self.assertIsNone(restored._staged_evaluation)
        # The legacy plain-dict frames stay readable by hot-loop keys.
        self.assertEqual(
            restored.get_history_step(-1)['results']['energy'], [-1.0])

    def test_expected_fingerprint_mismatch_raises(self):
        from pyrai3md.data.runtime.serializer import (
            save_state, load_state, StaleRuntimeError)
        state, _ = self._bound_c2_state()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'state.rt')
            save_state(state, path)
            with self.assertRaises(StaleRuntimeError):
                load_state(path, expected_fingerprint='deadbeefdeadbeef')

    def test_expected_fingerprint_match_loads(self):
        from pyrai3md.data.runtime.serializer import save_state, load_state
        state, plan = self._bound_c2_state()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'state.rt')
            save_state(state, path)
            restored, _ = load_state(
                path, expected_fingerprint=plan.plan_fingerprint)
        self.assertEqual(restored.n_center, 2)

    def test_history_frame_fingerprint_mismatch_raises(self):
        from pyrai3md.data.runtime.serializer import (
            save_state, load_state, StaleRuntimeError)
        state, _ = self._bound_c2_state()
        state.update_history()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'state.rt')
            save_state(state, path)
            with open(path, 'rb') as handler:
                payload = pickle.load(handler)
            payload['state']._history[-1]['plan_fingerprint'] = (
                'stale0000stale00')
            tampered = os.path.join(tmpdir, 'tampered.rt')
            with open(tampered, 'wb') as handler:
                pickle.dump(payload, handler)
            with self.assertRaises(StaleRuntimeError):
                load_state(tampered)

    def test_incompatible_version_raises(self):
        from pyrai3md.data.runtime.serializer import save_state, load_state
        state = make_state()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'state.rt')
            save_state(state, path)
            with open(path, 'rb') as handler:
                payload = pickle.load(handler)
            payload['version']['format'] = '9.9'
            bad = os.path.join(tmpdir, 'bad.rt')
            with open(bad, 'wb') as handler:
                pickle.dump(payload, handler)
            with self.assertRaises(ValueError):
                load_state(bad)

    def test_center_count_mismatch_raises(self):
        from pyrai3md.data.runtime.serializer import save_state, load_state
        state, _ = self._bound_c2_state()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'state.rt')
            save_state(state, path)
            with open(path, 'rb') as handler:
                payload = pickle.load(handler)
            payload['state'].electronic = payload['state'].electronic[:1]
            bad = os.path.join(tmpdir, 'bad.rt')
            with open(bad, 'wb') as handler:
                pickle.dump(payload, handler)
            with self.assertRaises(ValueError):
                load_state(bad)

    def test_deep_copy_isolates_mutable_center_data(self):
        from pyrai3md.data.runtime.serializer import deep_copy_state
        state, _ = self._bound_c2_state()
        state.commit_step(outcome={
            'electronic': {0: {'energy': [1.0]}},
            'active_surface': {'active_configuration': (0, 0),
                               'active_energy': None,
                               'active_gradient': None,
                               'evaluation_id': 1}})
        clone = deep_copy_state(state)
        clone.set_center_electronic(0, energy=[99.0])
        clone.update_history()
        # The mutable center data / history are isolated; the original
        # is untouched and the static plan stays equal but independent.
        self.assertEqual(state.center_electronic(0)['energy'], [1.0])
        self.assertEqual(state.history_size, 0)
        self.assertEqual(clone.history_size, 1)
        self.assertEqual(clone.partition_plan.plan_fingerprint,
                         state.partition_plan.plan_fingerprint)

    def test_to_dict_covers_center_fields(self):
        state, plan = self._bound_c2_state()
        state.commit_step(outcome={
            'electronic': {0: {'energy': [1.0, 2.0, 3.0, 4.0]}},
            'active_surface': {'active_configuration': (1, 0),
                               'active_energy': -75.5,
                               'active_gradient': None,
                               'evaluation_id': 3},
            'continuation': {'next_step': 2}})
        data = state.to_dict()
        self.assertEqual(data['partition_plan']['plan_fingerprint'],
                         plan.plan_fingerprint)
        self.assertEqual(data['partition_plan']['n_center'], 2)
        self.assertEqual(len(data['electronic']), 2)
        self.assertEqual(data['electronic'][0]['energy'],
                         [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(data['active_surface']['active_configuration'],
                         (1, 0))
        self.assertEqual(data['continuation'], {'next_step': 2})


class TestMultiCenterRegionInjection(unittest.TestCase):
    """_build_task_queue injects per-region charge/multiplicity (C>1).

    Locks docs MULTICENTER 6.2: the full-system baseline L(real)
    (center_index None) carries the plan aggregate; every center-region
    task (H(c)/L(c)) carries center c's charge/multiplicity; and the
    scalar is written into the task's skill_config so the first backend
    call already sees the region value instead of the global list.
    """

    @classmethod
    def setUpClass(cls):
        from pyrai3md.kernel.environment import Environment
        Environment().initialize()

    def _build(self, charges, multiplicity):
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.config import PyRAIConfig
        config = PyRAIConfig()
        config.control.jobtype = 'energy'
        config.control.theory = ['pyscf', 'xtb']
        config.system.atoms = ['H', 'H', 'H', 'H']
        config.system.coordinates = [[float(i), 0.0, 0.0]
                                     for i in range(4)]
        config.system.natom = 4
        config.system.highlevel = [[0, 1], [2, 3]]
        config.system.ci = [[1], [1]]
        config.system.spin = [[0], [0]]
        config.system.charges = charges
        config.system.multiplicity = multiplicity
        return build_state(config)

    def test_per_center_region_charge_multiplicity_injected(self):
        state = self._build([0, -1], [1, 2])
        tasks = [task for _, task in state.iter_tasks()]
        baseline = [t for t in tasks if t.get('center_index') is None]
        center0 = [t for t in tasks if t.get('center_index') == 0]
        center1 = [t for t in tasks if t.get('center_index') == 1]
        self.assertTrue(baseline and center0 and center1)
        # L(real): aggregate charge -1, (0 + 1) mod 2 + 1 = 2 (doublet)
        for task in baseline:
            self.assertEqual(task['region_charge'], -1)
            self.assertEqual(task['region_multiplicity'], 2)
            self.assertEqual(task['config']['skill_config']['charges'], -1)
            self.assertEqual(
                task['config']['skill_config']['multiplicity'], 2)
        for task in center0:
            self.assertEqual(task['region_charge'], 0)
            self.assertEqual(task['region_multiplicity'], 1)
            self.assertEqual(task['config']['skill_config']['charges'], 0)
            self.assertEqual(
                task['config']['skill_config']['multiplicity'], 1)
        for task in center1:
            self.assertEqual(task['region_charge'], -1)
            self.assertEqual(task['region_multiplicity'], 2)

    def test_scalar_charge_multiplicity_broadcast_to_all_regions(self):
        state = self._build(0, 1)
        tasks = [task for _, task in state.iter_tasks()]
        self.assertTrue(tasks)
        for task in tasks:
            self.assertEqual(task['region_charge'], 0)
            self.assertEqual(task['region_multiplicity'], 1)
            self.assertEqual(task['config']['skill_config']['charges'], 0)
            self.assertEqual(
                task['config']['skill_config']['multiplicity'], 1)

    def _build_multistate(self):
        """A C=2 run with two states per center and an explicit coupling
        pair, so the per-center electronic projection is observable."""
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.config import PyRAIConfig
        config = PyRAIConfig()
        config.control.jobtype = 'energy gradient'
        config.control.theory = ['pyscf', 'xtb']
        config.system.atoms = ['H'] * 6
        config.system.coordinates = [[float(i), 0.0, 0.0]
                                     for i in range(6)]
        config.system.natom = 6
        config.system.highlevel = [[0, 1, 2], [3, 4, 5]]
        config.system.ci = [[2], [2]]
        config.system.spin = [[0], [0]]
        config.system.coupling = [[[0, 1]], [[0, 1]]]
        config.system.charges = 0
        config.system.multiplicity = 1
        config.dynamics.root = [0, 1]
        return build_state(config)

    def test_per_center_electronic_projection_is_flat(self):
        """Docs MULTICENTER 6.2 line 214: every center-region task's
        skill_config carries its OWN center as a flat single-center
        system (never the global nested lists, which the flat-only
        skill schema rejects), and highlevel/midlevel are cleared
        because the runtime slices the region geometry."""
        state = self._build_multistate()
        tasks = [task for _, task in state.iter_tasks()]
        center_tasks = [t for t in tasks
                        if t.get('center_index') is not None]
        self.assertTrue(center_tasks)
        for task in center_tasks:
            skill = task['config']['skill_config']
            c = task['center_index']
            self.assertEqual(skill['ci'], [2])
            self.assertEqual(skill['spin'], [0])
            self.assertEqual(skill['coupling'], [[0, 1]])
            self.assertEqual(skill['nstate'], 2)
            self.assertEqual(skill['state_spin'], [0, 0])
            self.assertEqual(skill['nac_pair_map'], [[0, 1]])
            self.assertEqual(skill['soc_pair_map'], [])
            # current_state is the center's own initial root.
            self.assertEqual(skill['current_state'], c)
            # The sliced region is the whole QM system for this task.
            self.assertEqual(skill['highlevel'], [])
            self.assertEqual(skill['midlevel'], [])

    def test_baseline_electronic_projection_is_single_center_ground(self):
        """The full-system baseline L(real) (center_index None) carries
        a single-center ground-state electronic spec."""
        state = self._build_multistate()
        baseline = [t for _, t in state.iter_tasks()
                    if t.get('center_index') is None]
        self.assertTrue(baseline)
        for task in baseline:
            skill = task['config']['skill_config']
            self.assertEqual(skill['nstate'], 1)
            self.assertEqual(skill['ci'], [1])
            self.assertEqual(skill['spin'], [0])
            self.assertEqual(skill['coupling'], [])
            self.assertEqual(skill['state_spin'], [])
            self.assertEqual(skill['current_state'], 0)
            self.assertEqual(skill['highlevel'], [])
            self.assertEqual(skill['midlevel'], [])

    def test_region_geometry_sliced_into_skill_config(self):
        """Docs MULTICENTER 6.2 line 214: a sub-region task's
        skill_config carries the region atoms/natom, so the skill
        normalize derives natom from the region (not the full system)
        and its natom-keyed result arrays match the region gradient
        the runtime calculate() produces. L(real) keeps the full
        system. Regression: a full-system natom made the skill
        allocate (natom, 3) and fail to broadcast the (region, 3)
        gradient."""
        state = self._build_multistate()
        tasks = [task for _, task in state.iter_tasks()]
        for task in tasks:
            skill = task['config']['skill_config']
            indices = task.get('region_indices') or []
            if indices:
                self.assertEqual(len(skill['atoms']), len(indices))
                self.assertEqual(skill['natom'], len(indices))
                self.assertEqual(len(skill['coordinates']), len(indices))
            else:
                # L(real) baseline covers the whole 6-atom system.
                self.assertEqual(skill['natom'], 6)
                self.assertEqual(len(skill['atoms']), 6)


class TestStressFlagInjection(unittest.TestCase):
    """_build_task_queue injects the boolean stress flag (docs sec.8).

    Planning derives the 'stress' capability from md+ensemble=npt or
    optimize+optimize_cell. md/optimize are advanced jobtypes that
    bypass spell_skill_jobtype, so the request reaches the skill
    through task_config['skill_config']['stress']=True (orthogonal to
    the profile). A run without those switches never sets the flag
    True; it stays at the skill schema default (False for skills that
    declare 'stress', absent otherwise), both of which the backends
    read as 'not requested'.
    """

    @classmethod
    def setUpClass(cls):
        from pyrai3md.kernel.environment import Environment
        Environment().initialize()

    def _build(self, jobtype, **kwargs):
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.config import PyRAIConfig
        config = PyRAIConfig()
        config.control.jobtype = jobtype
        config.control.theory = ['xtb']
        config.system.atoms = ['H', 'H']
        config.system.coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
        config.system.natom = 2
        if jobtype == 'md':
            config.system.velocities = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            config.dynamics.ensemble = kwargs.get('ensemble', 'nve')
        elif jobtype == 'optimize':
            config.optimize.optimize_cell = kwargs.get('optimize_cell', False)
        return build_state(config)

    def _stress_flags(self, state):
        return [task['config']['skill_config'].get('stress')
                for _, task in state.iter_tasks()]

    def test_md_npt_injects_stress_flag(self):
        state = self._build('md', ensemble='npt')
        flags = self._stress_flags(state)
        self.assertTrue(flags)
        self.assertTrue(all(f is True for f in flags))

    def test_md_nve_no_stress_flag(self):
        state = self._build('md', ensemble='nve')
        flags = self._stress_flags(state)
        self.assertTrue(flags)
        self.assertTrue(all(f is not True for f in flags))

    def test_optimize_cell_injects_stress_flag(self):
        state = self._build('optimize', optimize_cell=True)
        flags = self._stress_flags(state)
        self.assertTrue(flags)
        self.assertTrue(all(f is True for f in flags))

    def test_optimize_no_cell_no_stress_flag(self):
        state = self._build('optimize', optimize_cell=False)
        flags = self._stress_flags(state)
        self.assertTrue(flags)
        self.assertTrue(all(f is not True for f in flags))

    def test_stress_flag_does_not_alter_jobtype_spelling(self):
        # The flag is orthogonal to the profile: the skill jobtype is
        # still spelled from the non-stress capabilities (stress is
        # absent from CAPABILITY_TO_TOKEN), so resolve_profile is
        # never altered by the stress request.
        state = self._build('md', ensemble='npt')
        for _, task in state.iter_tasks():
            skill = task['config']['skill_config']
            self.assertEqual(skill.get('stress'), True)
            self.assertNotIn('stress', str(skill.get('jobtype', '')))


class TestCurvatureProducerScope(unittest.TestCase):
    """`hess_driver: skill` is a question one of several backends cannot answer.

    The curvature a mode-following search asks for before its first step is one
    geometry's force constants. With more than one backend the input does not say
    whose matrix it is, and the planner's own refusal of a 'hessian' capability
    for a multi-backend run (`planning/resolver.py`) is not reached, because an
    optimization run spells its request through the dispatch channel rather than
    through the capability plan. So the same constraint is enforced here, where
    the input is still cheap to change and no geometry has been paid for.

    The guards read the pure declaration table (`needs_initial_hessian`), not a name list, so
    one of these cases drives it through an accepted optimizer with the flag
    patched on: that also keeps the guard honest if the search that carries it
    natively were ever removed from the registry.
    """

    @classmethod
    def setUpClass(cls):
        from pyrai3md.kernel.environment import Environment
        Environment().initialize()

    def _build(self, theory, hess_driver, method='minima', hess_atoms=None):
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.config import PyRAIConfig
        config = PyRAIConfig()
        config.control.jobtype = 'optimize'
        config.control.driver = 'main'
        config.control.title = 'curvature_scope'
        config.control.theory = list(theory)
        if len(theory) > 1:
            config.system.atoms = ['H', 'H', 'H', 'H']
            config.system.coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74],
                                         [2.0, 0.0, 0.0], [2.0, 0.0, 0.74]]
            config.system.natom = 4
            # Two backends over ONE center: what these cases are about is the
            # backend count, and a multi-center partition is a separate refusal
            # (TestCurvatureActiveSet covers it).
            config.system.highlevel = [[0, 1, 2, 3]]
            config.system.ci = [[1]]
            config.system.spin = [[0]]
            config.system.charges = 0
            config.system.multiplicity = 1
        else:
            config.system.atoms = ['H', 'H']
            config.system.coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
            config.system.natom = 2
            config.system.charges = 0
            config.system.multiplicity = 1
        config.optimize.opt_method = method
        if hess_driver is not None:
            config.hessian.hess_driver = hess_driver
        if hess_atoms is not None:
            config.hessian.hess_atoms = list(hess_atoms)
        return build_state(config)

    def test_a_multi_backend_run_is_refused_when_the_skill_must_answer(self):
        from unittest import mock
        from pyrai3md.workflow.optimizer.minima import MinimaOptimizer
        with mock.patch.dict(OPTIMIZER_DECLARATIONS['minima'], {'needs_initial_hessian': True}):
            with self.assertRaises(ValueError) as ctx:
                self._build(['xtb', 'pyscf'], 'skill')
        message = str(ctx.exception)
        self.assertIn('hess_driver', message)
        self.assertIn("hess_driver 'main'", message,
                      'the refusal has to name the way out')

    def test_the_framework_producer_is_not_refused_for_two_backends(self):
        # The negative control: the same two-backend input with the framework
        # as the producer builds, so the refusal above is the pairing and not
        # the backend count.
        from unittest import mock
        from pyrai3md.workflow.optimizer.minima import MinimaOptimizer
        with mock.patch.dict(OPTIMIZER_DECLARATIONS['minima'], {'needs_initial_hessian': True}):
            state = self._build(['xtb', 'pyscf'], 'main')
        self.assertTrue([task for _, task in state.iter_tasks()])

    def test_one_backend_can_be_asked_for_its_own_matrix(self):
        # The scope case kept honest: ONE backend that declares force constants
        # is a well-posed question, so neither refusal above fires. pyscf spells
        # its capability `frequency` (`skills/pyscf_skill/manifest.yaml:46`),
        # which is one of the names `MATRIX_CAPABILITIES` accepts.
        from unittest import mock
        from pyrai3md.workflow.optimizer.minima import MinimaOptimizer
        with mock.patch.dict(OPTIMIZER_DECLARATIONS['minima'], {'needs_initial_hessian': True}):
            state = self._build(['pyscf'], 'skill')
        self.assertTrue([task for _, task in state.iter_tasks()])

    def test_a_backend_without_force_constants_is_refused_by_name(self):
        # The other half of "whose matrix": xtb and cp2k declare energy, gradient
        # and stress and no curvature producer, so `hess_driver: skill` has
        # nothing behind it. Until this guard the input was accepted and the run
        # found out at dispatch, after paying for an evaluation - the same shape
        # of defect #367 recorded for `driver: skill`.
        from unittest import mock
        from pyrai3md.workflow.optimizer.minima import MinimaOptimizer
        for skill in ('xtb', 'cp2k', 'lammps'):
            with self.subTest(skill=skill):
                with mock.patch.dict(OPTIMIZER_DECLARATIONS['minima'], {'needs_initial_hessian': True}):
                    with self.assertRaises(ValueError) as ctx:
                        self._build([skill], 'skill')
                message = str(ctx.exception)
                self.assertIn('force constants', message)
                self.assertIn(skill, message,
                              'the refusal has to say which skill')
                self.assertIn("hess_driver 'main'", message,
                              'and name the way out')

    def test_the_framework_producer_is_never_refused_for_lacking_a_matrix(self):
        # The negative control on both refusals: the same three skills with the
        # framework differencing the gradients build, because the framework does
        # not need the skill to offer force constants.
        from unittest import mock
        from pyrai3md.workflow.optimizer.minima import MinimaOptimizer
        for skill in ('xtb', 'cp2k', 'lammps'):
            with self.subTest(skill=skill):
                with mock.patch.dict(OPTIMIZER_DECLARATIONS['minima'], {'needs_initial_hessian': True}):
                    state = self._build([skill], 'main')
                self.assertTrue([task for _, task in state.iter_tasks()])

    def test_a_partial_hessian_is_not_something_a_search_can_climb(self):
        # The held atoms of a partial Hessian are exactly zero rows, so the
        # direction of lowest curvature is an atom the input froze. Refused at
        # input, where dropping one keyword is still cheap.
        from unittest import mock
        from pyrai3md.workflow.optimizer.minima import MinimaOptimizer
        with mock.patch.dict(OPTIMIZER_DECLARATIONS['minima'], {'needs_initial_hessian': True}):
            with self.assertRaises(ValueError) as ctx:
                self._build(['xtb'], 'main', hess_atoms=[1])
        self.assertIn('hess_atoms', str(ctx.exception))

    def test_a_minima_search_may_still_be_given_a_partial_hessian(self):
        # The negative control: nothing asks for a curvature here, so the same
        # hess_atoms selection is an ordinary property request and stays legal.
        state = self._build(['xtb'], None, hess_atoms=[1])
        self.assertTrue([task for _, task in state.iter_tasks()])


class TestCurvatureActiveSet(unittest.TestCase):
    """The constraint plan, not a second keyword, says what a search may move.

    `fix` freezes components and `hess_atoms` names the atoms a numerical
    Hessian displaces. During a search the two must agree or the curvature model
    holds zero rows on atoms the run pinned, and the lowest curvature in it is a
    direction the input said not to climb. The plan is therefore the authority:
    the search derives the atom set from it, and an input that names a different
    one is refused instead of being overwritten.
    """

    @classmethod
    def setUpClass(cls):
        from pyrai3md.kernel.environment import Environment
        Environment().initialize()

    def _build(self, theory, fixes=None, hess_atoms=None, method='ts',
               centers=None, state_j=None):
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.config import PyRAIConfig
        config = PyRAIConfig()
        config.control.jobtype = 'optimize'
        config.control.driver = 'main'
        config.control.title = 'active_set'
        config.control.theory = list(theory)
        config.system.atoms = ['O', 'H', 'H', 'H']
        config.system.coordinates = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0],
                                     [-0.24, 0.93, 0.0], [-0.24, -0.47, 0.81]]
        config.system.natom = 4
        config.system.mass = [16.0, 1.0, 1.0, 1.0]
        config.system.charges = 0
        config.system.multiplicity = 1
        if centers is not None:
            config.system.highlevel = [list(group) for group in centers]
            config.system.ci = [[1] for _ in centers]
            config.system.spin = [[0] for _ in centers]
            config.system.charges = [0] * len(centers)
            config.system.multiplicity = [1] * len(centers)
            config.system.nstate = len(centers)
        elif len(theory) > 1:
            # Two layers of one program is a composite, and a composite names
            # its center.
            config.system.highlevel = [[0, 1, 2, 3]]
        if fixes:
            # The block is typed, not a bare dict: assigning the dict works
            # here only because the plan builder is permissive, and it makes
            # the serializer warn about the config it is asked to dump.
            from pyrai3md.schema.config import SystemConstraintConfig
            config.system.constraints = SystemConstraintConfig.model_validate(
                {'fixes': fixes})
        config.optimize.opt_method = method
        if state_j is not None:
            config.optimize.opt_state_j = int(state_j)
        if hess_atoms is not None:
            config.hessian.hess_atoms = list(hess_atoms)
        return build_state(config)

    def test_one_skill_can_hold_both_layers_of_a_composite(self):
        # The name-keyed binding dict dropped the second position, so a
        # single-center composite could not be built out of one program at all.
        from pyrai3md.kernel.maestro import Maestro
        from pyrai3md.schema.config import PyRAIConfig
        config = PyRAIConfig()
        config.control.jobtype = 'optimize'
        config.control.theory = ['pyscf', 'pyscf']
        bindings = Maestro().resolve_skills(config)['theory']
        self.assertEqual(sorted(bindings), ['pyscf_skill', 'pyscf_skill@qm2'],
                         'the second position needs a role-suffixed key, and '
                         'the first keeps the name every consumer indexes by')
        self.assertEqual([info['role'] for info in bindings.values()],
                         ['qm', 'qm2'])
        state = self._build(['pyscf', 'pyscf'], method='minima')
        tasks = [task for _, task in state.iter_tasks()]
        # ONIOM2 of one center: L(real) + H(model) - L(model).
        self.assertEqual([task['contribution'] for task in tasks],
                         [1, 1, -1],
                         [task['role'] for task in tasks])
        self.assertEqual({task['role'] for task in tasks}, {'qm', 'qm2'})

    def test_the_curvature_rules_run_once_per_plan_not_per_binding(self):
        # Both rules read the state's constraint plan and write one derived
        # atom list, so they are queue-level. The loop used to call them once
        # per descriptor; a two-layer run of one program asked the same
        # question of the same geometry twice.
        from unittest import mock
        from pyrai3md.planning import assembly as core_module
        calls = []
        with mock.patch.object(
                core_module, '_validate_curvature_freezes',
                side_effect=lambda state, method: calls.append('fixes')), \
                mock.patch.object(
                    core_module, '_derive_curvature_atoms',
                    side_effect=lambda state, method: calls.append('atoms')):
            self._build(['pyscf', 'pyscf'], method='ts')
        self.assertEqual(calls, ['fixes', 'atoms'])

    def test_a_resolved_binding_role_is_one_the_skill_declares(self):
        # `roles` in a manifest has no routing consumer: planning binds a skill
        # by position (kernel/maestro.py:_ROLE_SEQUENCE), so the declaration can
        # drift from what a run actually gets and nothing stops it. It drifted
        # the other way for PySCF, which said it could not serve `qm2` while a
        # two-layer PySCF search was being planned and converged. The gate keeps
        # the sheet honest about the machine.
        from pyrai3md.kernel.maestro import Maestro
        from pyrai3md.schema.config import PyRAIConfig
        from pyrai3md.skills.pyscf_skill.query import get_roles
        config = PyRAIConfig()
        config.control.jobtype = 'optimize'
        config.control.theory = ['pyscf', 'pyscf']
        bindings = Maestro().resolve_skills(config)['theory']
        declared = get_roles()
        for binding_key, info in bindings.items():
            role = info['role']
            self.assertIn(role, declared,
                          '%s resolved into role %r, which the manifest does '
                          'not name at all' % (binding_key, role))
            self.assertNotEqual(
                declared[role], False,
                '%s resolved into role %r while pyscf_skill/manifest.yaml '
                'still declares that role false: either the skill serves it '
                'and the sheet must say so, or the binding must be refused'
                % (binding_key, role))

    def test_the_free_atoms_of_a_fix_become_the_curvature_set(self):
        state = self._build(['pyscf'],
                            fixes=[{'id': 'f', 'atoms': [2],
                                    'direction': 'xyz'}])
        self.assertEqual(state.hessian_params['hess_atoms'], [1, 3, 4])

    def test_a_curvature_set_that_fights_the_fix_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self._build(['pyscf'],
                        fixes=[{'id': 'f', 'atoms': [2],
                                'direction': 'xyz'}],
                        hess_atoms=[1, 2, 3])
        message = str(ctx.exception)
        self.assertIn('hess_atoms', message)
        self.assertIn('constraint plan freezes', message)

    def test_a_run_without_fixes_leaves_the_section_as_it_was(self):
        # The negative control, and the byte-for-byte guarantee: no frozen atom
        # means no derivation, so `hess_atoms: []` keeps meaning every atom.
        state = self._build(['pyscf'])
        self.assertEqual(state.hessian_params['hess_atoms'], [])

    def test_a_per_component_fix_leaves_the_other_axes_free(self):
        # `fix` acts on components and `hess_atoms` on atoms: an atom frozen in
        # x only is still a movable atom, and the batch drops one row, not three.
        # Built as a minima run because a search refuses this granularity
        # outright (test_a_component_freeze_is_refused_for_a_curvature_search);
        # what is under test here is the row selection, not the policy.
        from pyrai3md.workflow.hessian import _active_rows
        state = self._build(['pyscf'], method='minima',
                            fixes=[{'id': 'f', 'atoms': [1],
                                    'direction': 'x'}])
        rows = _active_rows(state, list(range(4)))
        self.assertEqual(len(rows), 11, rows)
        self.assertNotIn((0, 0), rows)


    def test_a_component_freeze_is_refused_for_a_curvature_search(self):
        # Direction x/y/z/xy of one atom pins a laboratory axis, and the torque
        # that leaves behind has no free coordinate to be removed along: see
        # `_validate_curvature_freezes` for the measured plateau.
        with self.assertRaises(ValueError) as ctx:
            self._build(['pyscf'],
                        fixes=[{'id': 'f', 'atoms': [2],
                                'direction': 'z'}])
        message = str(ctx.exception)
        self.assertIn('component-level fix', message)
        self.assertIn('direction: xyz', message,
                      'the refusal has to name the shape that works')

    def test_a_component_freeze_still_relaxes_a_minimum(self):
        # The negative control: the refusal is about following curvature, not
        # about the constraint itself, and a minima relaxation keeps the meaning
        # it had before (a gradient on a chosen submanifold).
        state = self._build(['pyscf'],
                            fixes=[{'id': 'f', 'atoms': [2],
                                    'direction': 'z'}],
                            method='minima')
        self.assertTrue([task for _, task in state.iter_tasks()])

    def test_a_curvature_set_without_a_fix_behind_it_is_refused(self):
        # hess_atoms alone would leave zero rows in a model the search climbs.
        with self.assertRaises(ValueError) as ctx:
            self._build(['pyscf'], hess_atoms=[1, 2])
        self.assertIn('no fix', str(ctx.exception))


    def test_a_second_surface_is_refused_by_a_single_surface_search(self):
        # opt_state_j names a pair that only the crossing searches read; with a
        # single-surface method it would buy a second gradient at every step and
        # discard it.
        with self.assertRaises(ValueError) as ctx:
            self._build(['pyscf'], centers=[[0, 1, 2, 3]], state_j=1)
        self.assertIn('reads one surface', str(ctx.exception))

    def test_the_crossing_searches_still_ask_for_the_pair(self):
        # The negative control: the refusal is about the method reading one
        # surface, not about naming two.
        state = self._build(['pyscf'], centers=[[0, 1, 2, 3]], method='meci',
                            state_j=1)
        self.assertTrue([task for _, task in state.iter_tasks()])

    def test_a_multi_center_partition_searches_a_saddle_on_the_composite(self):
        # A curvature search needs one surface over the whole system's
        # coordinates, and a multi-center composite is that one surface
        # (E = sum_c E_QM(c) + E_classic). The two curvature helpers below
        # read the global constraint mask, so what a fix derives is a global
        # atom list: the active set is never split per center, and atom 4
        # frozen leaves 1-3 movable whether there is one region or two.
        # Read on the engines: two water centers under a QMMM saddle search
        # report `Negative Modes (model)` and a mode eigenvalue at every step.
        state = self._build(['pyscf', 'pyscf'],
                            centers=[[0, 1], [2, 3]],
                            fixes=[{'id': 'f', 'atoms': [4],
                                    'direction': 'xyz'}])
        self.assertEqual(state.hessian_params['hess_atoms'], [1, 2, 3])

    def test_the_same_two_centers_still_relax_a_minimum(self):
        # The other half of the same input: without a curvature model to build,
        # the search never asks for an active set at all, so `hess_atoms` keeps
        # the empty meaning it had before (every atom moves).
        state = self._build(['pyscf', 'pyscf'], centers=[[0, 1], [2, 3]],
                            method='minima')
        self.assertEqual(state.hessian_params['hess_atoms'], [])


class TestExecutionModeDerivation(unittest.TestCase):
    """_build_task_queue hands the skill the framework's execution
    decision (CHANGELOG #300).

    execution_mode is a skill keyword: skills/<skill>/service/runner
    returns run_slurm only for 'slurm'. The framework's dispatch
    decision is spelled in multitask/remotetask/multipool, which the
    skill schema never receives. When the framework is going to submit
    the calculation itself it therefore writes 'local' into the skill
    config, so exactly one component submits; when it is not going
    remote it writes nothing and the declared value (skill section or
    the pmd_env.yaml skill switch) stays in charge.
    """

    @classmethod
    def setUpClass(cls):
        from pyrai3md.kernel.environment import Environment
        Environment().initialize()

    def _build(self, jobtype='energy', system_remotetask=None,
               dynamics_remotetask=None, raw_skill_section=None):
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.config import PyRAIConfig
        config = PyRAIConfig()
        config.control.jobtype = jobtype
        config.control.theory = ['xtb']
        config.system.atoms = ['H', 'H']
        config.system.coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
        config.system.natom = 2
        if jobtype == 'md':
            config.system.velocities = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        if system_remotetask is not None:
            config.system.remotetask = system_remotetask
        if dynamics_remotetask is not None:
            config.dynamics.remotetask = dynamics_remotetask
        raw_config = ({'xtb': dict(raw_skill_section)}
                      if raw_skill_section else None)
        return build_state(config, raw_config=raw_config)

    def _modes(self, state):
        return [task['config']['skill_config'].get('execution_mode')
                for _, task in state.iter_tasks()]

    def _tasks(self, state):
        return [task for _, task in state.iter_tasks()]

    def test_local_run_keeps_the_declared_value(self):
        # No remotetask anywhere: the framework does not touch the
        # value the tester declared, which is how the skill-side
        # self-submission switch stays usable.
        state = self._build(raw_skill_section={'execution_mode': 'slurm'})
        modes = self._modes(state)
        self.assertTrue(modes)
        self.assertTrue(all(mode == 'slurm' for mode in modes), modes)

    def test_skill_section_remotetask_forces_local_and_warns(self):
        raw = {'execution_mode': 'slurm', 'remotetask': True}
        with self.assertLogs('pyrai3md.planning.assembly', level='WARNING') as log:
            modes = self._modes(self._build(raw_skill_section=raw))
        self.assertTrue(all(mode == 'local' for mode in modes), modes)
        self.assertTrue(any('remotetask is set' in line
                            for line in log.output), log.output)
        self.assertTrue(any('slurm' in line for line in log.output),
                        log.output)

    def test_molecule_scope_remotetask_forces_local(self):
        state = self._build(system_remotetask=True,
                            raw_skill_section={'execution_mode': 'slurm'})
        modes = self._modes(state)
        self.assertTrue(all(mode == 'local' for mode in modes), modes)

    def test_md_section_remotetask_forces_local(self):
        # The workflow-section axis: md.remotetask ships the whole child
        # runtime, so the skill inside it must not self-submit.
        state = self._build(jobtype='md', dynamics_remotetask=True,
                            raw_skill_section={'execution_mode': 'slurm'})
        modes = self._modes(state)
        self.assertTrue(all(mode == 'local' for mode in modes), modes)

    def test_derivation_survives_the_worker_round_trip(self):
        # A dispatched child runtime is rebuilt from the serialized
        # state, so the derived value must travel with it; otherwise the
        # skill inside the remote runtime self-submits and the same job
        # is queued twice.
        import pickle
        state = self._build(raw_skill_section={'remotetask': True,
                                               'execution_mode': 'slurm'})
        self.assertTrue(all(mode == 'local' for mode in self._modes(state)))
        restored = pickle.loads(pickle.dumps(state))
        modes = [task['config']['skill_config'].get('execution_mode')
                 for _, task in restored.iter_tasks()]
        self.assertTrue(modes)
        self.assertTrue(all(mode == 'local' for mode in modes), modes)

    def test_helper_is_a_union_over_the_dispatch_levels(self):
        from pyrai3md.planning.assembly import _dispatch_requests_remote
        from pyrai3md.schema.config import PyRAIConfig

        config = PyRAIConfig()
        self.assertFalse(_dispatch_requests_remote(config, {}))
        self.assertTrue(_dispatch_requests_remote(config,
                                                 {'remotetask': True}))
        config.system.remotetask = True
        self.assertTrue(_dispatch_requests_remote(config, {}))
        config.system.remotetask = None
        # An uninvoked section still counts: the helper deliberately
        # does not map the jobtype token to its section, because that
        # table belongs to the planner.
        config.hessian.remotetask = True
        self.assertTrue(_dispatch_requests_remote(config, {}))


if __name__ == '__main__':
    unittest.main()
