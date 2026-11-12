"""
======================================================================
This module tests the R6 multi-center functional extension of the MD
loop (dev/MULTICENTER_DEVELOPMENT.md, refactor R6): the loop keeps the
single-center execution strategy and data order, and only grows the
single-center behavior into a multi-center one.

Tests cover:
- the unified assembled active-combination gradient accessor
  (state.active_gradient): C>1 reads the committed active_surface
  gradient, C=1 degenerates to gradient[current_state] bit-for-bit,
  and a C>1 state with no committed gradient returns None instead of
  raising on the global current_state;
- verlet reading the assembled active gradient (a C=2 frame carries
  nuclear_sample.active_gradient, so the integrator never re-derives
  it from a recorded state index);
- center-aware target states (resolve_target_states_centered returns a
  flat list for C=1 and a per-center map for C>1) and the calculator
  per-task routing (_task_target_states);
- sync() leaving the per-center electronic scalars untouched for C>1
  (the global pmd.system nstate=1 must not overwrite a center's
  planning-derived nstate);
- the partitioned trajectory output matrix: global md.*/sh.* carry the
  whole system, per-center md.#.*/sh.#.* carry the region, and C=1
  with a region degrades to a single center;
- the run-log per-center state/population protocol (snapshot_centers +
  format_result_block);
- the trajectory center_history_step query.

Sep 08 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import tempfile
import unittest

import numpy as np

from pyrai3md.data.state import PyRAIState
from pyrai3md.data.center import build_center_plan
from pyrai3md.data.runtime.state import RuntimeState
from pyrai3md.schema.config import PyRAIConfig
from pyrai3md.utilities.print_results import (
    format_result_block, snapshot_centers,
)
from pyrai3md.workflow.calculator import (
    Calculator, SYNC_SYSTEM_FIELDS, _ELECTRONIC_STATE_FIELDS,
    _task_target_states,
)
from pyrai3md.workflow.integrator import verlet_i, verlet_ii
from pyrai3md.workflow.md import (
    resolve_target_states, resolve_target_states_centered,
)
from pyrai3md.workflow.records.output import MdOutputWriter
from pyrai3md.workflow.records.trajectory import TrajectoryResult

DT = 2.0
_COORDS_5 = [[0.10, 0.20, 0.30],
             [0.30, -0.40, 0.50],
             [-0.20, 0.10, 0.50],
             [0.60, -0.30, 0.20],
             [0.10, 0.20, -0.40]]


def _density_list(population):
    """A persisted (S, S, 2) [re, im] density with a given real diag."""
    n = len(population)
    real = np.zeros((n, n), dtype=np.float64)
    for index, value in enumerate(population):
        real[index, index] = float(value)
    return np.stack([real, np.zeros((n, n))], axis=-1).tolist()


def _c2_state():
    """A C=2 state: center 0 (2 states, atoms [0, 1]) and center 1
    (2 states, atoms [2, 3, 4]); active configuration (0, 0)."""
    config = PyRAIConfig()
    config.system.atoms = ['H'] * 5
    config.system.coordinates = [list(r) for r in _COORDS_5]
    config.system.velocities = [[0.001, 0.0, 0.0]] * 5
    config.system.mass = [12.0, 1.0, 12.0, 1.0, 1.0]
    config.system.natom = 5
    config.system.nstate = 1        # C>1 has no global state axis
    config.dynamics.history = 4

    state = PyRAIState(runtime=RuntimeState(total_steps=10))
    state.pmd = config
    plan = build_center_plan({
        'highlevel': [[0, 1], [2, 3, 4]],
        'ci': [[2], [2]],
        'spin': [[0], [0]],
        'coupling': [[[0, 1]], [[0, 1]]],
        'root': [0, 0],
    })
    state.bind_partition_plan(plan)
    return state


def _c1_region_state():
    """A C=1 state WITH an environment region (highlevel [0, 1] of 3
    atoms): n_layer=2, so the output is partitioned but there is a
    single center."""
    config = PyRAIConfig()
    config.system.atoms = ['H'] * 3
    config.system.coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                 [2.0, 0.0, 0.0]]
    config.system.velocities = [[0.001, 0.0, 0.0]] * 3
    config.system.mass = [12.0, 1.0, 1.0]
    config.system.natom = 3
    config.system.nstate = 2
    config.system.current_state = 0
    config.dynamics.history = 4
    config.dynamics.state_population = [1.0, 0.0]

    state = PyRAIState(runtime=RuntimeState(total_steps=10))
    state.pmd = config
    plan = build_center_plan({
        'highlevel': [[0, 1]],
        'ci': [[2]],
        'spin': [[0]],
        'coupling': [[[0, 1]]],
        'root': [0],
    })
    state.bind_partition_plan(plan)
    return state


class TestActiveGradientAccessor(unittest.TestCase):
    """state.active_gradient(): the one source verlet reads (r6a)."""

    def test_c1_selects_current_state_row(self):
        config = PyRAIConfig()
        config.system.atoms = ['C', 'H']
        config.system.coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        config.system.mass = [12.0, 1.0]
        config.system.natom = 2
        config.system.nstate = 2
        config.system.current_state = 1
        state = PyRAIState(runtime=RuntimeState(total_steps=10))
        state.pmd = config
        grad = np.stack([np.zeros((2, 3)), np.ones((2, 3))])
        state.set_result({'gradient': grad.tolist()})
        # C=1 degenerate path: gradient[current_state], bit-for-bit the
        # legacy selection.
        np.testing.assert_allclose(state.active_gradient(), np.ones((2, 3)))

    def test_c2_returns_committed_active_gradient(self):
        state = _c2_state()
        grad = np.arange(15, dtype=np.float64).reshape(5, 3) * 0.01
        state.active_surface = {
            'active_configuration': (0, 0), 'active_energy': 0.0,
            'active_gradient': grad.tolist(), 'evaluation_id': 1}
        np.testing.assert_allclose(state.active_gradient(), grad)

    def test_c2_uncommitted_returns_none_without_raising(self):
        state = _c2_state()
        # active_surface seeded with active_gradient None; the global
        # current_state would raise, so the accessor must return None.
        self.assertIsNone(state.active_gradient())


class TestVerletMultiCenter(unittest.TestCase):
    """verlet reads the assembled active gradient, never a state index
    (r6a). A C>1 run has no global current_state, so this is the only
    way the integrator works for multiple centers."""

    def test_verlet_ii_uses_committed_active_gradient(self):
        # Split form (docs section 5): verlet_ii applies ONLY the second
        # half kick with the committed active-combination gradient
        # g_curr; the previous frame gradient g_prev only gates the
        # first-step skip (it was consumed by verlet_i).
        state = _c2_state()
        g_prev = np.arange(15, dtype=np.float64).reshape(5, 3) * 0.01
        g_curr = np.arange(15, dtype=np.float64).reshape(5, 3) * 0.02
        state.active_surface = {
            'active_configuration': (0, 0), 'active_energy': 0.0,
            'active_gradient': g_prev.tolist(), 'evaluation_id': 1}
        state.update_history()          # frame carries g_prev
        state.active_surface = {
            'active_configuration': (0, 0), 'active_energy': 0.0,
            'active_gradient': g_curr.tolist(), 'evaluation_id': 2}

        velo_before = np.asarray(state.velocities, dtype=np.float64)
        velo_before = velo_before.reshape(-1, 3)
        verlet_ii(state, dt=DT)
        mass_au = np.asarray(state.mass_au, dtype=np.float64)
        expected = velo_before - 0.5 * g_curr / mass_au[:, None] * DT
        np.testing.assert_allclose(
            np.asarray(state.velocities).reshape(-1, 3), expected)

    def test_verlet_i_uses_frame_active_gradient(self):
        state = _c2_state()
        g_prev = np.arange(15, dtype=np.float64).reshape(5, 3) * 0.01
        state.active_surface = {
            'active_configuration': (0, 0), 'active_energy': 0.0,
            'active_gradient': g_prev.tolist(), 'evaluation_id': 1}
        state.update_history()

        from pyrai3md.utilities.units import BOHR_TO_ANGSTROM
        coords_before = np.asarray(state.coordinates, dtype=np.float64)
        velo_before = np.asarray(state.velocities,
                                 dtype=np.float64).reshape(-1, 3)
        verlet_i(state, dt=DT)
        mass_au = np.asarray(state.mass_au, dtype=np.float64)
        # Split form: half kick then drift with the half-kicked velocity.
        v_half = velo_before - 0.5 * g_prev / mass_au[:, None] * DT
        expected = coords_before + v_half * DT * BOHR_TO_ANGSTROM
        np.testing.assert_allclose(
            np.asarray(state.coordinates), expected)
        np.testing.assert_allclose(
            np.asarray(state.velocities).reshape(-1, 3), v_half)


class TestTargetStatesCentered(unittest.TestCase):
    """Center-aware per-step target states (r6b)."""

    def test_c1_returns_flat_list_identical_to_legacy(self):
        state = _c1_region_state()
        state.update_system({'current_state': 1})
        state.pmd.dynamics.activestate = 0
        got = resolve_target_states_centered(state)
        self.assertEqual(got, resolve_target_states(0, 1, state.nstate))
        self.assertEqual(got, [0, 1])

    def test_c2_returns_per_center_map(self):
        state = _c2_state()
        state.pmd.dynamics.activestate = 1     # active-state gradients
        state.active_surface['active_configuration'] = (1, 0)
        got = resolve_target_states_centered(state)
        self.assertIsInstance(got, dict)
        # center 0 active local state 1 -> [1]; center 1 active 0 -> [0]
        self.assertEqual(got, {0: [1], 1: [0]})

    def test_c2_activestate_zero_lists_all_center_states(self):
        state = _c2_state()
        state.pmd.dynamics.activestate = 0
        got = resolve_target_states_centered(state)
        self.assertEqual(got, {0: [0, 1], 1: [0, 1]})

    def test_task_target_states_routes_by_center(self):
        # A flat list (C=1) applies to every task unchanged.
        self.assertEqual(_task_target_states([0, 1], {'center_index': 0}),
                         [0, 1])
        # A per-center dict (C>1) selects the task's own center.
        per_center = {0: [1], 1: [0]}
        self.assertEqual(_task_target_states(per_center,
                                             {'center_index': 1}), [0])
        # A baseline task with no center keeps None.
        self.assertIsNone(_task_target_states(per_center,
                                              {'center_index': None}))
        # None passes through.
        self.assertIsNone(_task_target_states(None, {'center_index': 0}))


class TestSyncMultiCenterGuard(unittest.TestCase):
    """sync() must not overwrite the per-center electronic scalars of a
    C>1 run with the global pmd.system values (r6b)."""

    def test_electronic_state_fields_excluded_for_c2(self):
        state = _c2_state()
        task_config = PyRAIConfig()
        task_config.system.atoms = ['H'] * 5
        task_config.system.coordinates = [list(r) for r in _COORDS_5]
        task_config.system.velocities = [[0.0, 0.0, 0.0]] * 5
        task_config.system.mass = [12.0, 1.0, 12.0, 1.0, 1.0]
        task_config.system.natom = 5
        task_config.system.nstate = 2          # center 0's state count
        state.add_task('theory', {
            'role': 'qm', 'task': 0, 'task_id': 'qm_c0', 'task_slot': 0,
            'center_index': 0, 'region': 'model', 'region_indices': [0, 1],
            'config': task_config,
        })
        # The global system carries nstate=1 (no global state axis).
        self.assertEqual(state.pmd.system.nstate, 1)
        Calculator(state=state).sync()
        # The center's planning-derived nstate survives the sync.
        self.assertEqual(task_config.system.nstate, 2)
        # Geometry is still synced (sliced to the region).
        self.assertEqual(len(task_config.system.coordinates), 2)

    def test_c1_sync_still_pushes_every_field(self):
        state = _c1_region_state()
        state.update_system({'current_state': 1})
        task_config = PyRAIConfig()
        task_config.system.atoms = ['H'] * 3
        task_config.system.coordinates = [[0.0, 0.0, 0.0]] * 3
        task_config.system.velocities = [[0.0, 0.0, 0.0]] * 3
        task_config.system.mass = [12.0, 1.0, 1.0]
        task_config.system.natom = 3
        task_config.system.nstate = 2
        state.add_task('theory', {
            'role': 'qm', 'task': 0, 'task_id': 'qm', 'task_slot': 0,
            'center_index': 0, 'region': 'all', 'region_indices': [],
            'config': task_config,
        })
        Calculator(state=state).sync()
        # C=1 keeps the legacy full sync (current_state propagated).
        self.assertEqual(task_config.system.current_state, 1)
        # Sanity: the guard set is a subset of the synced fields.
        self.assertTrue(_ELECTRONIC_STATE_FIELDS.issubset(
            set(SYNC_SYSTEM_FIELDS)))


class TestOutputPartitioned(unittest.TestCase):
    """The partitioned trajectory output matrix (r6c)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log_dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _read(self, title, suffix):
        path = os.path.join(self.log_dir, '%s.%s' % (title, suffix))
        if not os.path.exists(path):
            return None
        with open(path) as handle:
            return handle.read()

    def _c2_snapshot(self):
        state = _c2_state()
        state.set_center_electronic(0, energy=[0.0, 0.05],
                                    density=_density_list([1.0, 0.0]))
        state.set_center_electronic(1, energy=[0.0, 0.08],
                                    density=_density_list([0.0, 1.0]))
        state.set_result({'energy': [0.10], 'active_energy': 0.10})
        grad = np.zeros((5, 3), dtype=np.float64)
        state.active_surface = {
            'active_configuration': (0, 0), 'active_energy': 0.10,
            'active_gradient': grad.tolist(), 'evaluation_id': 1}
        state.update_history()
        snapshot = state.get_history_step(-1)
        # A hop on center 0 after the snapshot (post-hop live state).
        state.active_surface = dict(state.active_surface,
                                    active_configuration=(1, 0))
        return state, snapshot

    def test_c2_writes_global_and_per_center_files(self):
        state, snapshot = self._c2_snapshot()
        writer = MdOutputWriter(
            title='t', log_dir=self.log_dir, direct=2000, buffer=500,
            dt=20.67, total_steps=1, center_regions=[[0, 1], [2, 3, 4]])
        self.assertTrue(writer.partitioned)
        writer.write_headers()
        gate = writer.record(1, snapshot, state, final=True)
        self.assertTrue(gate)

        # Global coordinates carry the whole system (5 atoms).
        self.assertEqual(self._read('t', 'md.xyz').splitlines()[0], '5')
        # Per-center files carry the region atoms (2 and 3).
        self.assertEqual(self._read('t', 'md.1.xyz').splitlines()[0], '2')
        self.assertEqual(self._read('t', 'md.2.xyz').splitlines()[0], '3')
        # A hop happened on center 0 -> the sh.* files exist too.
        self.assertIsNotNone(self._read('t', 'sh.xyz'))
        self.assertIsNotNone(self._read('t', 'sh.1.xyz'))
        self.assertIsNotNone(self._read('t', 'sh.2.energies'))

    def test_c2_global_energies_and_pop_columns(self):
        state, snapshot = self._c2_snapshot()
        writer = MdOutputWriter(
            title='t', log_dir=self.log_dir, total_steps=1,
            center_regions=[[0, 1], [2, 3, 4]])
        writer.write_headers()
        writer.record(1, snapshot, state, final=True)

        # md.energies data row: time | Epot Ekin Etot | 2 center active
        # states | 2 + 2 per-center per-state energies = 10 fields.
        energies = self._read('t', 'md.energies').splitlines()
        self.assertEqual(len(energies[1].split()), 10)
        # md.pop data row: time | 2 + 2 populations | 2 + 2 active = 9.
        pop = self._read('t', 'md.pop').splitlines()
        self.assertEqual(len(pop[1].split()), 9)

    def test_c1_with_region_degrades_to_single_center(self):
        state = _c1_region_state()
        state.set_result({'energy': [0.0, 0.05]})
        state.update_history()
        snapshot = state.get_history_step(-1)
        writer = MdOutputWriter(
            title='t', log_dir=self.log_dir, total_steps=1,
            center_regions=[[0, 1]])
        self.assertTrue(writer.partitioned)
        writer.write_headers()
        writer.record(1, snapshot, state, final=True)

        # Global md.xyz keeps all 3 atoms; md.1.xyz keeps the 2-atom
        # region; there is no md.2.* (a single center).
        self.assertEqual(self._read('t', 'md.xyz').splitlines()[0], '3')
        self.assertEqual(self._read('t', 'md.1.xyz').splitlines()[0], '2')
        self.assertIsNone(self._read('t', 'md.2.xyz'))

    def test_plain_run_has_no_per_center_files(self):
        # No center regions -> not partitioned -> byte-identical legacy.
        writer = MdOutputWriter(title='t', log_dir=self.log_dir,
                                total_steps=1)
        self.assertFalse(writer.partitioned)
        suffixes = writer._file_suffixes()
        self.assertNotIn('md.1.xyz', suffixes)
        self.assertIn('md.xyz', suffixes)


class TestRunLogCenters(unittest.TestCase):
    """The run-log per-center state/population protocol (r6c)."""

    def _c2_frame(self):
        state = _c2_state()
        state.set_center_electronic(0, energy=[0.0, 0.05],
                                    density=_density_list([1.0, 0.0]))
        state.set_center_electronic(1, energy=[0.0, 0.08],
                                    density=_density_list([0.0, 1.0]))
        state.set_result({'energy': [0.10], 'active_energy': 0.10})
        state.active_surface = {
            'active_configuration': (0, 0), 'active_energy': 0.10,
            'active_gradient': np.zeros((5, 3)).tolist(),
            'evaluation_id': 1}
        state.update_history()
        return state.get_history_step(-1)

    def test_snapshot_centers_none_for_single_center(self):
        state = _c1_region_state()
        state.set_result({'energy': [0.0, 0.05]})
        state.update_history()
        self.assertIsNone(snapshot_centers(state.get_history_step(-1)))

    def test_snapshot_centers_per_center_for_c2(self):
        frame = self._c2_frame()
        centers = snapshot_centers(frame)
        self.assertEqual(len(centers), 2)
        self.assertEqual(centers[0]['active'], 0)
        np.testing.assert_allclose(centers[0]['population'], [1.0, 0.0])
        np.testing.assert_allclose(centers[1]['population'], [0.0, 1.0])

    def test_format_result_block_renders_per_center(self):
        frame = self._c2_frame()
        centers = snapshot_centers(frame)
        block = format_result_block(frame.get('system'),
                                    frame.get('results'),
                                    step=1, centers=centers)
        self.assertIn('Center   1', block)
        self.assertIn('Center   2', block)

    def _charged_c2_frame(self):
        """A C=2 frame whose centers carry explicit region charge/spin
        (center 0 neutral triplet, center 1 anion doublet)."""
        config = PyRAIConfig()
        config.system.atoms = ['H'] * 5
        config.system.coordinates = [list(r) for r in _COORDS_5]
        config.system.velocities = [[0.001, 0.0, 0.0]] * 5
        config.system.mass = [12.0, 1.0, 12.0, 1.0, 1.0]
        config.system.natom = 5
        config.system.nstate = 1
        config.dynamics.history = 4
        state = PyRAIState(runtime=RuntimeState(total_steps=10))
        state.pmd = config
        plan = build_center_plan({
            'highlevel': [[0, 1], [2, 3, 4]],
            'ci': [[2], [2]], 'spin': [[0], [0]],
            'coupling': [[[0, 1]], [[0, 1]]],
            'root': [0, 0],
            'charges': [0, -1], 'multiplicity': [3, 2],
        })
        state.bind_partition_plan(plan)
        state.set_center_electronic(0, energy=[0.0, 0.05],
                                    density=_density_list([1.0, 0.0]))
        state.set_center_electronic(1, energy=[0.0, 0.08],
                                    density=_density_list([0.0, 1.0]))
        state.set_result({'energy': [0.10], 'active_energy': 0.10})
        state.active_surface = {
            'active_configuration': (0, 0), 'active_energy': 0.10,
            'active_gradient': np.zeros((5, 3)).tolist(),
            'evaluation_id': 1}
        state.update_history()
        return state.get_history_step(-1)

    def test_snapshot_centers_carries_region_charge_spin(self):
        frame = self._charged_c2_frame()
        centers = snapshot_centers(frame)
        self.assertEqual(centers[0]['charge'], 0)
        self.assertEqual(centers[0]['multiplicity'], 3)
        self.assertEqual(centers[1]['charge'], -1)
        self.assertEqual(centers[1]['multiplicity'], 2)

    def test_format_result_block_renders_region_charge_spin(self):
        frame = self._charged_c2_frame()
        centers = snapshot_centers(frame)
        block = format_result_block(frame.get('system'),
                                    frame.get('results'),
                                    step=1, centers=centers)
        self.assertIn('&region charge & spin multiplicity', block)
        self.assertIn('L(real)', block)

    def test_no_region_charge_spin_section_for_single_center(self):
        state = _c1_region_state()
        state.set_result({'energy': [0.0, 0.05]})
        state.update_history()
        frame = state.get_history_step(-1)
        centers = snapshot_centers(frame)
        self.assertIsNone(centers)
        block = format_result_block(frame.get('system'),
                                    frame.get('results'),
                                    step=1, centers=centers)
        self.assertNotIn('&region charge & spin multiplicity', block)


class TestTrajectoryCenterHistory(unittest.TestCase):
    """The trajectory center_history_step query (r6c)."""

    def test_center_history_step_exposes_per_center_record(self):
        state = _c2_state()
        state.set_center_electronic(0, energy=[0.0, 0.05])
        state.set_center_electronic(1, energy=[0.0, 0.08])
        state.set_result({'energy': [0.10], 'active_energy': 0.10})
        state.active_surface = {
            'active_configuration': (0, 0), 'active_energy': 0.10,
            'active_gradient': np.zeros((5, 3)).tolist(),
            'evaluation_id': 1}
        state.update_history()

        record = TrajectoryResult(0, state).center_history_step(0)
        self.assertIsNotNone(record)
        self.assertEqual(record['region'], [0, 1])
        self.assertEqual(record['active_config_pre'], (0, 0))
        np.testing.assert_allclose(record['electronic']['energy'],
                                   [0.0, 0.05])
        record1 = TrajectoryResult(0, state).center_history_step(1)
        self.assertEqual(record1['region'], [2, 3, 4])

    def test_center_history_step_none_without_state(self):
        self.assertIsNone(
            TrajectoryResult(0, object()).center_history_step(0))


class _AnalyticRegionBackend:
    """Analytic per-region energy/gradient for the C>1 MD loop gate.

    A harmonic surface with a per-state scale (w_s = 1 + 0.25 s), so
    every state is distinct yet exactly differentiable; nstate comes
    from the task config, so centers return distinct state axes.
    """

    def __init__(self, config=None, execution_id=''):
        self.config = config or {}
        self.execution_id = execution_id

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None):
        local = np.asarray(coordinates, dtype=np.float64)
        nstate = int(self.config.get('nstate', 1))
        base = 0.5 * float(np.sum(local * local))
        weights = np.array([1.0 + 0.25 * s for s in range(nstate)],
                           dtype=np.float64)
        return {'energy': weights * base,
                'gradient': weights[:, None, None] * local[None, :, :]}


def _analytic_task(task_id, center_index, provides, policy, ref_state,
                   region, region_indices, contribution, nstate):
    """One ONIOM task descriptor over the analytic backend."""
    return {
        'role': 'qm' if provides else 'mm',
        'task': 0, 'task_id': task_id, 'layer_id': 0,
        'static_id': task_id + '_static',
        'contribution': contribution,
        'region': region, 'region_indices': list(region_indices),
        'region_charge': None,
        'skill_name': 'mock_skill', 'api': _AnalyticRegionBackend,
        'capabilities': ['energy', 'gradient'],
        'config': {'nstate': nstate},
        'center_index': center_index,
        'provides_electronic': provides,
        'state_policy': policy,
        'reference_state': ref_state,
    }


def _c2_md_state():
    """A C=2 ONIOM state wired for a full AIMD loop (r4f task queue
    plus the MD-only velocities/title/dynamics)."""
    config = PyRAIConfig()
    config.system.atoms = ['H'] * 5
    config.system.coordinates = [list(r) for r in _COORDS_5]
    config.system.velocities = [[0.002, 0.0, 0.0]] * 5
    config.system.mass = [12.0, 1.0, 12.0, 1.0, 1.0]
    config.system.natom = 5
    config.system.charges = 0
    config.system.multiplicity = 1
    config.control.title = 'c2md'
    config.dynamics.history = 4
    config.dynamics.direct = 2000
    config.dynamics.buffer = 500

    state = PyRAIState(runtime=RuntimeState(total_steps=3))
    state.pmd = config
    plan = build_center_plan({
        'highlevel': [[0, 1], [2, 3, 4]],
        'ci': [[2, 2], [3]], 'spin': [[0, 1], [0]], 'root': [1, 2]})
    state.bind_partition_plan(plan)
    state.add_task('theory', _analytic_task(
        'mm_real', None, False, 'reference', 0, 'real', [], 1, 1))
    state.add_task('theory', _analytic_task(
        'qm_c0', 0, True, 'active', None, 'model', [0, 1], 1, 4))
    state.add_task('theory', _analytic_task(
        'mm_c0', 0, False, 'reference', 0, 'model', [0, 1], -1, 1))
    state.add_task('theory', _analytic_task(
        'qm_c1', 1, True, 'active', None, 'model', [2, 3, 4], 1, 3))
    state.add_task('theory', _analytic_task(
        'mm_c1', 1, False, 'reference', 0, 'model', [2, 3, 4], -1, 1))
    return state


class TestMultiCenterMdLoop(unittest.TestCase):
    """The full C>1 AIMD loop runs end to end and writes the
    partitioned output (r6e): verlet, target states, calculator
    commit, sync and output all work for multiple centers without
    touching the global current_state / nstate."""

    def test_c2_aimd_loop_runs_and_writes_partitioned_output(self):
        from pyrai3md.workflow.md import AIMD
        state = _c2_md_state()
        coords0 = np.asarray(state.coordinates, dtype=np.float64).copy()
        calc = Calculator(state=state).initialize()
        wf = AIMD(calculator=calc, maxiter=3, dt=1.0, sfhp='nosh')
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                wf._run_trajectory(state)
                for suffix in ('md.xyz', 'md.velo', 'md.energies',
                               'md.pop', 'md.1.xyz', 'md.2.xyz',
                               'md.1.energies', 'md.2.pop'):
                    self.assertTrue(
                        os.path.exists('c2md.%s' % suffix),
                        'missing c2md.%s' % suffix)
                # The global trajectory carries all 5 atoms; the
                # per-center files carry the 2- and 3-atom regions.
                with open('c2md.md.xyz') as handle:
                    self.assertEqual(handle.readline().strip(), '5')
                with open('c2md.md.1.xyz') as handle:
                    self.assertEqual(handle.readline().strip(), '2')
                with open('c2md.md.2.xyz') as handle:
                    self.assertEqual(handle.readline().strip(), '3')
            finally:
                os.chdir(cwd)
        # The loop advanced and the nuclei moved (verlet integrated the
        # assembled active gradient); nosh keeps the seeded (1, 2).
        self.assertEqual(state.runtime.current_step, 3)
        self.assertEqual(state.get_active_configuration(), (1, 2))
        self.assertFalse(np.allclose(
            np.asarray(state.coordinates, dtype=np.float64), coords0))

    def test_c2_aimd_loop_runs_with_fssh_electronic_stepper(self):
        # sfhp='fssh' routes C>1 through the ElectronicStepper dispatch
        # layer (one CenterView-bound FSSH integrator per center, kTDC
        # coupling from the per-center energy curvature); this is the
        # loop the multicenter_md example exercises end to end.
        from pyrai3md.workflow.md import AIMD
        state = _c2_md_state()
        coords0 = np.asarray(state.coordinates, dtype=np.float64).copy()
        calc = Calculator(state=state).initialize()
        wf = AIMD(calculator=calc, maxiter=3, dt=1.0, sfhp='fssh')
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                wf._run_trajectory(state)
                for suffix in ('md.xyz', 'md.velo', 'md.energies',
                               'md.pop', 'md.1.xyz', 'md.2.xyz',
                               'md.1.energies', 'md.2.pop'):
                    self.assertTrue(
                        os.path.exists('c2md.%s' % suffix),
                        'missing c2md.%s' % suffix)
            finally:
                os.chdir(cwd)
        self.assertEqual(state.runtime.current_step, 3)
        # Each center stays inside its own state axis (nstate 4 / 3);
        # hopping may or may not fire on the analytic surfaces.
        active = state.get_active_configuration()
        self.assertEqual(len(active), 2)
        self.assertIn(active[0], (0, 1, 2, 3))
        self.assertIn(active[1], (0, 1, 2))
        self.assertFalse(np.allclose(
            np.asarray(state.coordinates, dtype=np.float64), coords0))


if __name__ == '__main__':
    unittest.main()
