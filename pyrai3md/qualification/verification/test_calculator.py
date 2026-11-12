"""
======================================================================
Test suite for the calculator layer (pyrai3md/workflow/calculator.py)
and the composite aggregation (pyrai3md/workflow/multiscale/composite.py).

Tests cover:
- Module imports
- adapt_skill_result() conversion from mock dict to Result
- Calculator.run() with a mock backend (single-shot and task queue)
- Composite aggregation via contribution coefficients
  (COMPOSITE_PATTERNS, validate_contributions, aggregate_results)
- Composite cap-atom projection placeholder (project)
- Calculator.sync() runtime state synchronization
- PyRAIState read-only properties and update interface

Capability resolution tests are in test_property.py (planning layer).
Task queue resolution tests are in test_task.py (planning layer).

Aug 04 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Inline mock data matching the backend.calculate() return format.
MOCK_SKILL_RESULT = {
    'energy': -75.123456,
    'gradient': [[[0.01, 0.02, 0.03], [-0.01, -0.02, -0.03]]],
    'nac': None,
    'soc': None,
    'dipole': [0.1, 0.2, 0.3],
}

MOCK_SKILL_RESULT_FULL = {
    'energy': -100.5,
    'gradient': [[[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]],
    'nac': [[[0.01, 0.02, 0.03]]],
    'soc': [0.05],
    'dipole': [1.0, 2.0, 3.0],
    'populations': [0.8, 0.2],
    'hessian': [[1.0, 0.0], [0.0, 1.0]],
}


class MockBackend:
    """Mock backend following the real backend contract.

    Accepts (config, execution_id) like the skill backends and
    returns a result dict from calculate(). The gradient is
    generated with the correct shape matching the input atom count.
    """

    def __init__(self, config=None, execution_id='', result_dict=None):
        self.config = config
        self.execution_id = execution_id
        self._result = result_dict or MOCK_SKILL_RESULT
        self.call_count = 0
        self.last_atoms = None
        self.last_coords = None
        self.last_target_states = None
        self.last_cell = None

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None):
        self.call_count += 1
        self.last_atoms = atoms
        self.last_coords = coordinates
        self.last_target_states = target_states
        self.last_cell = cell
        result = dict(self._result)
        # Adapt gradient shape to input atom count
        natom = len(atoms)
        if 'gradient' in result and result['gradient'] is not None:
            grad = np.asarray(result['gradient'])
            if grad.ndim >= 2:
                nstate = grad.shape[0] if grad.ndim == 3 else 1
                result['gradient'] = np.ones(
                    (nstate, natom, 3)) * 0.01
        return result


def _make_task(role, task_idx=0, contribution=1, region='all',
               region_indices=None, energy=-1.0):
    """Create a minimal task dict with a dict config for testing."""
    task_id = role if task_idx == 0 else "%s_task%d" % (role, task_idx)
    return {
        'role': role,
        'task': task_idx,
        'task_id': task_id,
        'layer_id': 0,
        'static_id': "%s_layer0_mock" % task_id,
        'contribution': contribution,
        'region': region,
        'region_indices': list(region_indices or []),
        'region_charge': None,
        'region_multiplicity': None,
        'skill_name': 'mock_skill',
        'api': MockBackend,
        'capabilities': ['energy', 'gradient'],
        'config': {'results': {'energy': energy}},
    }


def _make_mock_state():
    """Create a minimal PyRAIState with pmd as the data source."""
    from pyrai3md.data.state import PyRAIState
    from pyrai3md.data.runtime.state import RuntimeState
    from pyrai3md.schema.config import PyRAIConfig

    config = PyRAIConfig()
    config.system.atoms = ['C', 'H']
    config.system.coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    config.system.natom = 2
    config.system.charges = 0
    config.system.multiplicity = 1
    config.system.nstate = 2

    state = PyRAIState(runtime=RuntimeState(total_steps=10))
    state.pmd = config
    return state


class TestCalculatorImports(unittest.TestCase):
    """Test that all calculator modules can be imported."""

    def test_import_calculator_class(self):
        from pyrai3md.workflow.calculator import Calculator
        self.assertTrue(callable(Calculator))

    def test_import_adapt_skill_result(self):
        from pyrai3md.workflow.calculator import adapt_skill_result
        self.assertTrue(callable(adapt_skill_result))

    def test_import_resolve_calculator(self):
        from pyrai3md.workflow.calculator import resolve_calculator
        self.assertTrue(callable(resolve_calculator))

    def test_import_composite_functions(self):
        import pyrai3md.workflow.multiscale.composite as composite
        self.assertTrue(callable(composite.aggregate_results))
        # The static contribution table, its queue check and the
        # identity projector are retired: the queue's own coefficients
        # are the single source (the table's 'qmmm' row had already
        # drifted from the shape the assembler builds).
        for gone in ('COMPOSITE_PATTERNS', 'queue_contributions',
                     'validate_contributions', 'project'):
            self.assertFalse(hasattr(composite, gone), gone)

    def test_composite_has_no_calculator_import(self):
        import inspect
        import pyrai3md.workflow.multiscale.composite as composite_module
        source = inspect.getsource(composite_module)
        self.assertNotIn('from .calculator', source)
        self.assertNotIn('from pyrai3md.workflow.calculator', source)

    def test_import_from_workflow_package(self):
        from pyrai3md.workflow import Calculator, adapt_skill_result, resolve_calculator
        self.assertTrue(callable(Calculator))
        self.assertTrue(callable(adapt_skill_result))
        self.assertTrue(callable(resolve_calculator))


class TestAdaptSkillResult(unittest.TestCase):
    """Test adapt_skill_result conversion."""

    def test_basic_energy_only(self):
        from pyrai3md.workflow.calculator import adapt_skill_result
        result = adapt_skill_result({'energy': -75.0})
        self.assertAlmostEqual(result['energy'], -75.0)
        self.assertIsNone(result.get('gradient'))

    def test_full_result(self):
        from pyrai3md.workflow.calculator import adapt_skill_result
        result = adapt_skill_result(MOCK_SKILL_RESULT_FULL)
        self.assertAlmostEqual(result['energy'], -100.5)
        self.assertIsNotNone(result['gradient'])
        self.assertIsNotNone(result['nac'])
        self.assertIsNotNone(result['soc'])
        self.assertIsNotNone(result['dipole'])
        self.assertIsNotNone(result['populations'])
        self.assertIn('hessian', result.keys())

    def test_none_values(self):
        from pyrai3md.workflow.calculator import adapt_skill_result
        result = adapt_skill_result({'energy': None, 'gradient': None})
        self.assertIsNone(result['energy'])
        self.assertIsNone(result['gradient'])

    def test_extra_keys(self):
        from pyrai3md.workflow.calculator import adapt_skill_result
        result = adapt_skill_result({'energy': -10.0, 'hessian': [1, 2, 3], 'custom': 'val'})
        self.assertIn('hessian', result.keys())
        self.assertIn('custom', result.keys())
        self.assertEqual(result['custom'], 'val')

    def test_empty_dict(self):
        from pyrai3md.workflow.calculator import adapt_skill_result
        result = adapt_skill_result({})
        self.assertEqual(len(result), 0)

    def test_non_dict_raises(self):
        from pyrai3md.workflow.calculator import adapt_skill_result
        with self.assertRaises(TypeError):
            adapt_skill_result([1, 2, 3])

    def test_dipole_values(self):
        from pyrai3md.workflow.calculator import adapt_skill_result
        result = adapt_skill_result(MOCK_SKILL_RESULT)
        np.testing.assert_array_almost_equal(result['dipole'], [0.1, 0.2, 0.3])


class TestCalculatorClass(unittest.TestCase):
    """Test Calculator with mock backend."""

    def test_construction(self):
        from pyrai3md.workflow.calculator import Calculator
        backend = MockBackend()
        calc = Calculator(dummy_backend=backend)
        self.assertIs(calc.dummy_backend, backend)

    def test_run_returns_result(self):
        from pyrai3md.workflow.calculator import Calculator
        from pyrai3md.data.runtime.result import Result
        backend = MockBackend()
        calc = Calculator(dummy_backend=backend)
        state = _make_mock_state()
        calc.state = state
        result = calc.run()
        self.assertIsInstance(result, Result)
        self.assertAlmostEqual(result['energy'], -75.123456)

    def test_run_calls_backend_once(self):
        from pyrai3md.workflow.calculator import Calculator
        backend = MockBackend()
        calc = Calculator(dummy_backend=backend)
        state = _make_mock_state()
        calc.state = state
        calc.run()
        self.assertEqual(backend.call_count, 1)

    def test_run_passes_correct_atoms(self):
        from pyrai3md.workflow.calculator import Calculator
        backend = MockBackend()
        calc = Calculator(dummy_backend=backend)
        state = _make_mock_state()
        calc.state = state
        calc.run()
        self.assertEqual(backend.last_atoms, ['C', 'H'])

    def test_run_passes_correct_coordinates(self):
        from pyrai3md.workflow.calculator import Calculator
        backend = MockBackend()
        calc = Calculator(dummy_backend=backend)
        state = _make_mock_state()
        calc.state = state
        calc.run()
        expected = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        np.testing.assert_array_almost_equal(backend.last_coords, expected)

    def test_run_passes_target_states(self):
        from pyrai3md.workflow.calculator import Calculator
        backend = MockBackend()
        calc = Calculator(dummy_backend=backend)
        state = _make_mock_state()
        calc.state = state
        calc.run(target_states=[1])
        self.assertEqual(backend.last_target_states, [1])

    def test_run_default_target_states_none(self):
        from pyrai3md.workflow.calculator import Calculator
        backend = MockBackend()
        calc = Calculator(dummy_backend=backend)
        state = _make_mock_state()
        calc.state = state
        calc.run()
        self.assertIsNone(backend.last_target_states)

    def test_single_shot_writes_pmd_results(self):
        from pyrai3md.workflow.calculator import Calculator
        backend = MockBackend()
        calc = Calculator(dummy_backend=backend)
        state = _make_mock_state()
        calc.state = state
        calc.run()
        self.assertAlmostEqual(state.pmd.results['energy'], -75.123456)
        self.assertAlmostEqual(state.result['energy'], -75.123456)

    def test_repr(self):
        from pyrai3md.workflow.calculator import Calculator
        backend = MockBackend()
        calc = Calculator(dummy_backend=backend)
        r = repr(calc)
        self.assertIn('MockBackend', r)

    def test_resolve_calculator(self):
        from pyrai3md.workflow.calculator import resolve_calculator, Calculator
        backend = MockBackend()
        calc = resolve_calculator(dummy_backend=backend)
        self.assertIsInstance(calc, Calculator)


class TestCompositeAggregation(unittest.TestCase):
    """Test composite aggregation functions over executed task queues."""

    def _make_oniom2_state(self):
        """State with an executed ONIOM2 task queue (e0 + e1 - e2)."""
        state = _make_mock_state()
        state.add_task('theory', _make_task('qm', 0, 1, 'model',
                                            [0], energy=10.0))
        state.add_task('theory', _make_task('qm2', 0, 1, 'real',
                                            [], energy=5.0))
        state.add_task('theory', _make_task('qm2', 1, -1, 'model',
                                            [0], energy=2.0))
        return state

    def test_oniom2_aggregation(self):
        from pyrai3md.workflow.multiscale.composite import aggregate_results
        state = self._make_oniom2_state()
        data = aggregate_results(state)
        # 10.0 + 5.0 - 2.0 = 13.0
        self.assertAlmostEqual(float(data['energy']), 13.0)

    def test_aggregation_returns_dict_only(self):
        from pyrai3md.workflow.multiscale.composite import aggregate_results
        state = self._make_oniom2_state()
        aggregate_results(state)
        # Pure function: never writes to pmd
        self.assertEqual(state.pmd.results, {})

    def test_oniom3_aggregation(self):
        from pyrai3md.workflow.multiscale.composite import aggregate_results
        state = _make_mock_state()
        energies = [10.0, 8.0, 2.0, 5.0, 3.0]
        contributions = [1, 1, -1, 1, -1]
        roles = ['qm', 'qm2', 'qm2', 'qm3', 'qm3']
        for i, (energy, contribution) in enumerate(zip(energies, contributions)):
            state.add_task('theory', _make_task(roles[i], i % 2, contribution,
                                                'model', [], energy=energy))
        data = aggregate_results(state)
        # 10 + 8 - 2 + 5 - 3 = 18
        self.assertAlmostEqual(float(data['energy']), 18.0)

class TestCalculatorTaskQueue(unittest.TestCase):
    """Test Calculator execution over task queues."""

    def _make_single_task_state(self, preseed=True):
        state = _make_mock_state()
        task = _make_task('qm', 0, 1, 'all', [])
        if not preseed:
            # Empty config so the backend result is stored by the
            # calculator fallback before aggregation
            task['config'] = {}
        state.add_task('theory', task)
        return state, task

    def test_single_task_runs_backend_once(self):
        from pyrai3md.workflow.calculator import Calculator
        state, task = self._make_single_task_state(preseed=False)
        calc = Calculator(state=state).initialize()
        result = calc.run()
        self.assertEqual(calc.backends['theory'][0].call_count, 1)
        self.assertAlmostEqual(result['energy'], -75.123456)

    def test_single_task_full_geometry(self):
        from pyrai3md.workflow.calculator import Calculator
        state, task = self._make_single_task_state()
        calc = Calculator(state=state).initialize()
        calc.run()
        self.assertEqual(calc.backends['theory'][0].last_atoms, ['C', 'H'])

    def test_single_task_writes_pmd_results(self):
        from pyrai3md.workflow.calculator import Calculator
        state, task = self._make_single_task_state(preseed=False)
        Calculator(state=state).initialize().run()
        self.assertAlmostEqual(state.pmd.results['energy'], -75.123456)

    def test_single_task_passes_target_states(self):
        from pyrai3md.workflow.calculator import Calculator
        state, task = self._make_single_task_state()
        calc = Calculator(state=state).initialize()
        calc.run(target_states=[0, 2])
        self.assertEqual(calc.backends['theory'][0].last_target_states,
                         [0, 2])

    def test_single_task_default_target_states_none(self):
        from pyrai3md.workflow.calculator import Calculator
        state, task = self._make_single_task_state()
        calc = Calculator(state=state).initialize()
        calc.run()
        self.assertIsNone(calc.backends['theory'][0].last_target_states)

    def test_target_states_reset_between_runs(self):
        from pyrai3md.workflow.calculator import Calculator
        state, task = self._make_single_task_state()
        calc = Calculator(state=state).initialize()
        calc.run(target_states=[1])
        calc.run()
        # The staged per-step value must not leak into the next run
        self.assertIsNone(calc.backends['theory'][0].last_target_states)

    def test_non_periodic_passes_cell_none(self):
        from pyrai3md.workflow.calculator import Calculator
        state, task = self._make_single_task_state()
        calc = Calculator(state=state).initialize()
        calc.run()
        self.assertIsNone(calc.backends['theory'][0].last_cell)

    def test_periodic_system_passes_cell_snapshot(self):
        from pyrai3md.workflow.calculator import Calculator
        state, task = self._make_single_task_state()
        lattice = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        state.update_system({'pbc': [1, 1, 1], 'lattice_vector': lattice,
                             'volume': 1000.0})
        calc = Calculator(state=state).initialize()
        calc.run()
        cell = calc.backends['theory'][0].last_cell
        self.assertIsNotNone(cell)
        self.assertEqual(cell['pbc'], [1, 1, 1])
        self.assertEqual(cell['lattice_vector'], lattice)
        self.assertAlmostEqual(cell['volume'], 1000.0)

    def test_snapshot_carries_image_with_the_cell(self):
        # A backend that keeps periodic images needs the crossing count
        # beside the coordinates: wrapped positions alone do not say which
        # copy of the cell an atom sits in, and the counts belong to the
        # same instant as the geometry, so they travel in the same frozen
        # snapshot rather than in the step-lagging sync channel.
        from pyrai3md.workflow.calculator import Calculator
        state, task = self._make_single_task_state()
        lattice = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        state.update_system({'pbc': [1, 1, 1], 'lattice_vector': lattice,
                             'volume': 1000.0,
                             'image': [[1, 0, -2], [0, 0, 0]]})
        calc = Calculator(state=state).initialize()
        calc.run()
        cell = calc.backends['theory'][0].last_cell
        self.assertEqual(cell['image'], [[1, 0, -2], [0, 0, 0]])

    def test_snapshot_image_is_empty_before_the_first_wrap(self):
        # A run that never crossed a face (or a checkpoint predating the
        # channel) reports no crossings instead of a wrong box-wide shift.
        from pyrai3md.workflow.calculator import Calculator
        state, task = self._make_single_task_state()
        lattice = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        state.update_system({'pbc': [1, 1, 1], 'lattice_vector': lattice,
                             'volume': 1000.0})
        calc = Calculator(state=state).initialize()
        calc.run()
        self.assertEqual(calc.backends['theory'][0].last_cell['image'], [])

    def test_image_is_not_a_synced_field(self):
        # sync() pushes the system block into task configs at step end; the
        # crossing count is per-evaluation state and must not take that
        # route (dev/methods/PBC_DEVELOPMENT.md section 3.1).
        from pyrai3md.workflow.calculator import SYNC_SYSTEM_FIELDS
        self.assertNotIn('image', SYNC_SYSTEM_FIELDS)
        self.assertIn('coordinates', SYNC_SYSTEM_FIELDS)

    def test_variable_cell_uses_current_step_lattice(self):
        # The priority fix (PBC docs 3.1): calculate() reads the cell
        # frozen from the LIVE state at evaluation time, so a
        # variable-cell integrator deformation applied this step
        # (verlet._scale_cell -> update_system) reaches the backend of
        # the SAME step instead of lagging one step behind via sync().
        from pyrai3md.workflow.calculator import Calculator
        state, task = self._make_single_task_state()
        lattice0 = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        state.update_system({'pbc': [1, 1, 1], 'lattice_vector': lattice0,
                             'volume': 1000.0})
        calc = Calculator(state=state).initialize()
        # The integrator deforms the cell for this step AFTER
        # initialize(); no sync() has run yet, so the task config still
        # carries lattice0. The snapshot channel must deliver lattice1.
        lattice1 = [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]]
        state.update_system({'lattice_vector': lattice1, 'volume': 1728.0})
        calc.run()
        cell = calc.backends['theory'][0].last_cell
        self.assertEqual(cell['lattice_vector'], lattice1)
        self.assertAlmostEqual(cell['volume'], 1728.0)

    def test_region_slice_geometry(self):
        from pyrai3md.workflow.calculator import Calculator
        state = _make_mock_state()
        task = _make_task('qm2', 1, -1, 'model', [1])
        state.add_task('theory', task)
        calc = Calculator(state=state).initialize()
        calc.run()
        self.assertEqual(calc.backends['theory'][0].last_atoms, ['H'])
        np.testing.assert_array_almost_equal(
            calc.backends['theory'][0].last_coords, [[1.0, 0.0, 0.0]])

    def test_geometry_updates_between_runs(self):
        from pyrai3md.workflow.calculator import Calculator
        state = _make_mock_state()
        task = _make_task('qm', 0, 1, 'all', [])
        state.add_task('theory', task)
        calc = Calculator(state=state).initialize()
        calc.run()
        # Update the live geometry through pmd and run again
        state.update_system({
            'coordinates': [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        })
        calc.run()
        np.testing.assert_array_almost_equal(
            calc.backends['theory'][0].last_coords,
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        self.assertEqual(calc.backends['theory'][0].call_count, 2)

    def test_backend_writes_task_config(self):
        from pyrai3md.workflow.calculator import Calculator
        state, task = self._make_single_task_state()
        task['config'] = {}
        Calculator(state=state).initialize().run()
        # Mock backend does not write; Calculator fallback stores it
        self.assertAlmostEqual(task['config']['results']['energy'],
                               -75.123456)

    def test_composite_queue_aggregates(self):
        from pyrai3md.workflow.calculator import Calculator
        state = _make_mock_state()
        # Use a mock backend that returns specific energies
        energies = [10.0, 5.0, 2.0]
        call_idx = [0]  # mutable counter for closure

        class EnergyMockBackend:
            def __init__(self, config=None, execution_id=''):
                self.call_count = 0
                self.last_atoms = None
                self.last_coords = None
                self.last_target_states = None

            def calculate(self, atoms, coords, point_charges=None,
                          target_states=None, cell=None):
                e = energies[min(call_idx[0], len(energies) - 1)]
                call_idx[0] += 1
                self.call_count += 1
                natom = len(atoms)
                return {
                    'energy': e,
                    'gradient': np.ones((1, natom, 3)) * 0.01,
                }

        for role, idx, contribution in [
                ('qm', 0, 1), ('qm2', 0, 1), ('qm2', 1, -1)]:
            task = _make_task(role, idx, contribution, 'all', [])
            task['api'] = EnergyMockBackend
            task['config'] = {'results': {}}
            state.add_task('theory', task)
        result = Calculator(state=state).initialize().run()
        # Backend results: e0 + e1 - e2 = 10 + 5 - 2
        energy = np.asarray(result['energy'])
        self.assertAlmostEqual(float(energy.flat[0]), 13.0)

    def test_composite_queue_writes_pmd_results(self):
        from pyrai3md.workflow.calculator import Calculator
        state = _make_mock_state()
        energies = [10.0, 5.0, 2.0]
        call_idx = [0]

        class EnergyMockBackend:
            def __init__(self, config=None, execution_id=''):
                self.call_count = 0
                self.last_atoms = None
                self.last_coords = None
                self.last_target_states = None

            def calculate(self, atoms, coords, point_charges=None,
                          target_states=None, cell=None):
                e = energies[min(call_idx[0], len(energies) - 1)]
                call_idx[0] += 1
                self.call_count += 1
                natom = len(atoms)
                return {
                    'energy': e,
                    'gradient': np.ones((1, natom, 3)) * 0.01,
                }

        for role, idx, contribution in [
                ('qm', 0, 1), ('qm2', 0, 1), ('qm2', 1, -1)]:
            task = _make_task(role, idx, contribution, 'all', [])
            task['api'] = EnergyMockBackend
            task['config'] = {'results': {}}
            state.add_task('theory', task)
        Calculator(state=state).initialize().run()
        energy = np.asarray(state.pmd.results['energy'])
        self.assertAlmostEqual(float(energy.flat[0]), 13.0)

    def test_composite_queue_executes_all_tasks(self):
        from pyrai3md.workflow.calculator import Calculator
        state = _make_mock_state()
        for role, idx, contribution in [('qm', 0, 1), ('qm2', 0, 1),
                                        ('qm2', 1, -1)]:
            task = _make_task(role, idx, contribution)
            state.add_task('theory', task)
        calc = Calculator(state=state).initialize()
        calc.run()
        for backend in calc.backends['theory']:
            self.assertEqual(backend.call_count, 1)


class TestCalculatorInitialize(unittest.TestCase):
    """Test Calculator.initialize() runtime backend ownership."""

    def _make_two_task_state(self):
        state = _make_mock_state()
        task_a = _make_task('qm', 0, 1, 'model', [0])
        task_b = _make_task('qm2', 1, -1, 'model', [0])
        state.add_task('theory', task_a)
        state.add_task('theory', task_b)
        return state, task_a, task_b

    def test_initialize_creates_backend_per_task(self):
        from pyrai3md.workflow.calculator import Calculator
        state, _, _ = self._make_two_task_state()
        calc = Calculator(state=state).initialize()
        # Backends grouped by category (theory/reference)
        all_backends = (calc.backends['theory']
                        + calc.backends['reference'])
        self.assertEqual(len(all_backends), sum(1 for _ in state.iter_tasks()))
        for backend in all_backends:
            self.assertIsInstance(backend, MockBackend)

    def test_state_tasks_carry_no_instance(self):
        state, task_a, task_b = self._make_two_task_state()
        for task in (task_a, task_b):
            self.assertNotIn('instance', task)

    def test_execution_id_matches_static_id(self):
        from pyrai3md.workflow.calculator import Calculator
        state, task_a, task_b = self._make_two_task_state()
        calc = Calculator(state=state).initialize()
        self.assertEqual(calc.execution_ids['theory'][0], task_a['static_id'])
        self.assertEqual(calc.execution_ids['theory'][1], task_b['static_id'])

    def test_execution_id_with_runtime_idx(self):
        from pyrai3md.workflow.calculator import Calculator
        state, task_a, _ = self._make_two_task_state()
        calc = Calculator(state=state, runtime_idx='replica_3').initialize()
        self.assertEqual(calc.execution_ids['theory'][0],
                         "%s_replica_3" % task_a['static_id'])

    def test_reinitialize_rebuilds_backends(self):
        from pyrai3md.workflow.calculator import Calculator
        state, _, _ = self._make_two_task_state()
        calc = Calculator(state=state).initialize()
        first_backends = list(calc.backends['theory'])
        calc.initialize()
        self.assertEqual(len(calc.backends['theory']),
                         len(first_backends))
        for old, new in zip(first_backends, calc.backends['theory']):
            self.assertIsNot(old, new)

    def test_initialize_creates_runtime_state(self):
        from pyrai3md.workflow.calculator import Calculator
        from pyrai3md.data.runtime.state import RuntimeState
        state = _make_mock_state()
        state.runtime = None
        state.add_task('theory', _make_task('qm', 0, 1, 'all', []))
        Calculator(state=state).initialize()
        self.assertIsInstance(state.runtime, RuntimeState)

    def test_run_without_explicit_initialize(self):
        from pyrai3md.workflow.calculator import Calculator
        state, _, _ = self._make_two_task_state()
        calc = Calculator(state=state)
        result = calc.run()
        # run() lazily initializes backends
        self.assertEqual(len(calc.backends['theory']), sum(1 for _ in state.iter_tasks()))
        self.assertIsNotNone(result['energy'])


class TestCalculatorSync(unittest.TestCase):
    """Test Calculator.sync() runtime state synchronization."""

    def _make_sync_state(self):
        from pyrai3md.schema.config import PyRAIConfig
        state = _make_mock_state()
        # Task with a PyRAIConfig-style config carrying a system section
        task_config = PyRAIConfig()
        task_config.system.atoms = ['C', 'H']
        task_config.system.coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        task = _make_task('qm', 0, 1, 'all', [])
        task['config'] = task_config
        state.add_task('theory', task)
        # Task with the framework dict task_config structure:
        # {'skill_config': flat_dict, 'results': {}}
        dict_task = _make_task('qm2', 0, 1, 'all', [])
        dict_task['config'] = {
            'skill_config': {
                'atoms': ['C', 'H'],
                'coordinates': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                'natom': 2,
                'current_state': 0,
            },
            'results': {},
        }
        state.add_task('theory', dict_task)
        return state, task, dict_task

    def test_sync_updates_task_configs(self):
        from pyrai3md.workflow.calculator import Calculator
        state, task, dict_task = self._make_sync_state()
        new_coords = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        state.update_system({'coordinates': new_coords})
        Calculator(state=state).sync()
        # PyRAIConfig task config
        self.assertEqual(task['config'].system.coordinates, new_coords)
        # Dict task config: pushed into the flat skill_config
        self.assertEqual(dict_task['config']['skill_config']['coordinates'],
                         new_coords)

    def test_sync_keeps_control_section(self):
        from pyrai3md.workflow.calculator import Calculator
        state, task, _ = self._make_sync_state()
        before = task['config'].control.model_dump()
        state.update_system({'coordinates': [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]})
        Calculator(state=state).sync()
        self.assertEqual(task['config'].control.model_dump(), before)

    def test_sync_without_state_is_noop(self):
        from pyrai3md.workflow.calculator import Calculator
        calc = Calculator(state=None)
        calc.sync()  # should not raise

    def test_sync_propagates_multiplicity(self):
        from pyrai3md.workflow.calculator import Calculator
        state, task, _ = self._make_sync_state()
        state.update_system({'multiplicity': 3})
        Calculator(state=state).sync()
        self.assertEqual(task['config'].system.multiplicity, 3)

    def test_sync_writes_global_charge_multiplicity_when_no_region(self):
        from pyrai3md.workflow.calculator import Calculator
        state, _, dict_task = self._make_sync_state()
        skill = dict_task['config']['skill_config']
        skill['charges'] = 0
        skill['multiplicity'] = 1
        state.update_system({'charges': 5, 'multiplicity': 3})
        Calculator(state=state).sync()
        self.assertEqual(skill['charges'], 5)
        self.assertEqual(skill['multiplicity'], 3)

    def test_sync_protects_region_charge_multiplicity_dict(self):
        # A planning-injected region scalar (per-center charge/spin) is
        # never overwritten by the global sync (docs MULTICENTER 6.3).
        from pyrai3md.workflow.calculator import Calculator
        state, _, dict_task = self._make_sync_state()
        dict_task['region_charge'] = -1
        dict_task['region_multiplicity'] = 2
        skill = dict_task['config']['skill_config']
        skill['charges'] = -1
        skill['multiplicity'] = 2
        state.update_system({'charges': 5, 'multiplicity': 3})
        Calculator(state=state).sync()
        self.assertEqual(skill['charges'], -1)
        self.assertEqual(skill['multiplicity'], 2)

    def test_sync_protects_region_charge_multiplicity_pyrai_config(self):
        from pyrai3md.workflow.calculator import Calculator
        state, task, _ = self._make_sync_state()
        task['region_charge'] = -1
        task['region_multiplicity'] = 2
        task['config'].system.charges = -1
        task['config'].system.multiplicity = 2
        state.update_system({'charges': 5, 'multiplicity': 3})
        Calculator(state=state).sync()
        self.assertEqual(task['config'].system.charges, -1)
        self.assertEqual(task['config'].system.multiplicity, 2)

    def test_sync_keeps_region_task_molecular(self):
        # A sliced task is a 6-atom (here 1-atom) molecular subsystem.
        # The global box fields must not overwrite its natom/cell/pbc,
        # or the NN high layer is re-labelled as the 1581-atom box.
        from pyrai3md.workflow.calculator import Calculator
        state, _, dict_task = self._make_sync_state()
        dict_task['region'] = 'model'
        dict_task['region_indices'] = [0]
        skill = dict_task['config']['skill_config']
        skill['natom'] = 1
        skill['pbc'] = [0, 0, 0]
        skill['cell'] = [0.0] * 6
        state.update_system({
            'coordinates': [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
            'pbc': [1, 1, 1],
            'cell': [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
        })
        Calculator(state=state).sync()
        self.assertEqual(skill['natom'], 1)
        self.assertEqual(skill['pbc'], [0, 0, 0])
        self.assertEqual(skill['cell'], [0.0] * 6)
        self.assertEqual(skill['coordinates'], [[0.0, 0.0, 0.0]])
        self.assertEqual(skill['atoms'], ['C'])

    def test_sync_propagates_current_state(self):
        from pyrai3md.workflow.calculator import Calculator
        state, task, dict_task = self._make_sync_state()
        # The authoritative active state is updated through the
        # official interface (as NAMD surface hopping does)
        state.update_system({'current_state': 1})
        Calculator(state=state).sync()
        # PyRAIConfig task config
        self.assertEqual(task['config'].system.current_state, 1)
        # Dict task config: pushed into the flat skill_config
        self.assertEqual(
            dict_task['config']['skill_config']['current_state'], 1)

    def test_sync_does_not_inject_missing_fields(self):
        from pyrai3md.workflow.calculator import Calculator
        state, _, dict_task = self._make_sync_state()
        skill = dict_task['config']['skill_config']
        Calculator(state=state).sync()
        # Framework-only fields absent from the flat skill dict must
        # never be injected into the skill configuration
        for field in ('ci', 'coupling', 'velocities', 'spin',
                      'mass', 'state_label'):
            self.assertNotIn(field, skill)


class TestFinCoordConsumption(unittest.TestCase):
    """Test the fin_coord channel: skills performing internal
    iteration report their final structure, and the Calculator
    applies it to state.pmd.system before result storage."""

    def _run_with_result(self, result_dict):
        from pyrai3md.workflow.calculator import Calculator
        state = _make_mock_state()
        task = _make_task('qm', 0, 1, 'all', [])
        state.add_task('theory', task)
        backend = MockBackend(result_dict=result_dict)
        calc = Calculator(state=state, dummy_backend=backend)
        result = calc.run()
        return state, task, result

    def test_fin_coord_updates_system_when_different(self):
        new_coords = [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]
        state, task, result = self._run_with_result(
            {'energy': [-75.0], 'fin_coord': new_coords})
        self.assertEqual(state.coordinates, new_coords)

    def test_fin_coord_identical_keeps_system(self):
        same = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        state, _, _ = self._run_with_result(
            {'energy': [-75.0], 'fin_coord': same})
        self.assertEqual(state.coordinates, same)

    def test_fin_coord_empty_is_noop(self):
        state, _, _ = self._run_with_result(
            {'energy': [-75.0], 'fin_coord': []})
        self.assertEqual(state.coordinates,
                         [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def test_fin_coord_missing_is_noop(self):
        state, _, _ = self._run_with_result({'energy': [-75.0]})
        self.assertEqual(state.coordinates,
                         [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def test_fin_coord_length_mismatch_raises(self):
        bad = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        with self.assertRaises(ValueError):
            self._run_with_result(
                {'energy': [-75.0], 'fin_coord': bad})

    def test_fin_coord_not_left_in_results(self):
        new_coords = [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]
        state, task, result = self._run_with_result(
            {'energy': [-75.0], 'fin_coord': new_coords})
        # Transport channel key must not leak into task results,
        # the aggregated result or state.pmd.results
        self.assertNotIn('fin_coord', task['config'].get('results', {}))
        self.assertNotIn('fin_coord', result)
        self.assertNotIn('fin_coord', state.pmd.results)

    def test_fin_coord_applied_before_next_run(self):
        from pyrai3md.workflow.calculator import Calculator
        state = _make_mock_state()
        task = _make_task('qm', 0, 1, 'all', [])
        state.add_task('theory', task)
        new_coords = [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]
        backend = MockBackend(result_dict={
            'energy': [-75.0], 'fin_coord': new_coords})
        calc = Calculator(state=state, dummy_backend=backend)
        calc.run()
        # Second run sees the skill-reported geometry, and the empty
        # fin_coord of a plain energy step keeps it untouched
        backend._result = {'energy': [-75.1], 'fin_coord': []}
        calc.run()
        np.testing.assert_array_almost_equal(
            backend.last_coords, new_coords)
        self.assertEqual(state.coordinates, new_coords)


class TestFinVelocityConsumption(unittest.TestCase):
    """Test the fin_velocity channel: a skill whose engine propagates the
    nuclei reports their final velocities, and the Calculator applies them
    to state.pmd.system and recomputes the kinetic energy from them."""

    def _run_with_result(self, result_dict, mass=(12.0, 1.008)):
        from pyrai3md.workflow.calculator import Calculator
        state = _make_mock_state()
        if mass is not None:
            state.pmd.system.mass = list(mass)
            state.pmd.system.natom = len(mass)
        task = _make_task('qm', 0, 1, 'all', [])
        state.add_task('theory', task)
        backend = MockBackend(result_dict=result_dict)
        calc = Calculator(state=state, dummy_backend=backend)
        result = calc.run()
        return state, task, result

    def test_fin_velocity_updates_system_when_different(self):
        new_velo = [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]]
        state, _, _ = self._run_with_result(
            {'energy': [-75.0], 'fin_velocity': new_velo})
        np.testing.assert_array_almost_equal(
            np.asarray(state.velocities, dtype=float), new_velo)

    def test_fin_velocity_empty_is_noop(self):
        state, _, _ = self._run_with_result(
            {'energy': [-75.0], 'fin_velocity': []})
        self.assertEqual(np.asarray(state.velocities, dtype=float).sum(), 0.0)

    def test_fin_velocity_missing_is_noop(self):
        state, _, _ = self._run_with_result({'energy': [-75.0]})
        self.assertEqual(np.asarray(state.velocities, dtype=float).sum(), 0.0)

    def test_fin_velocity_length_mismatch_raises(self):
        bad = [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]]
        with self.assertRaises(ValueError):
            self._run_with_result(
                {'energy': [-75.0], 'fin_velocity': bad})

    def test_fin_velocity_row_width_mismatch_raises(self):
        bad = [[0.1, 0.0], [0.0, 0.2]]
        with self.assertRaises(ValueError):
            self._run_with_result(
                {'energy': [-75.0], 'fin_velocity': bad})

    def test_fin_velocity_not_left_in_results(self):
        new_velo = [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]]
        state, task, result = self._run_with_result(
            {'energy': [-75.0], 'fin_velocity': new_velo})
        # Transport channel key must not leak into task results,
        # the aggregated result or state.pmd.results
        self.assertNotIn('fin_velocity', task['config'].get('results', {}))
        self.assertNotIn('fin_velocity', result)
        self.assertNotIn('fin_velocity', state.pmd.results)

    def test_kinetic_energy_is_recomputed(self):
        new_velo = [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]]
        state, _, _ = self._run_with_result(
            {'energy': [-75.0], 'fin_velocity': new_velo})
        mass_au = state.mass_au
        expect = float(0.5 * np.sum(mass_au[:, None]
                                    * np.asarray(new_velo) ** 2))
        self.assertGreater(state.runtime.kinetic_energy, 0.0)
        self.assertAlmostEqual(state.runtime.kinetic_energy, expect)

    def test_kinetic_energy_stays_zero_without_velocities(self):
        state, _, _ = self._run_with_result({'energy': [-75.0]})
        self.assertEqual(state.runtime.kinetic_energy, 0.0)

    def test_composite_rejects_both_fin_channels(self):
        from pyrai3md.workflow.calculator import Calculator
        state = _make_mock_state()
        calc = Calculator(state=state,
                          dummy_backend=MockBackend(result_dict={}))
        for channel in ('fin_coord', 'fin_velocity'):
            tasks = [_make_task('qm', 0, 0.5, 'all', []),
                     _make_task('qm', 1, 0.5, 'all', [])]
            outcomes = [{'raw': {'energy': [-75.0], channel: [[0.]*3]*2}},
                        {'raw': {'energy': [-74.0]}}]
            with self.assertRaises(ValueError) as caught:
                calc._route_outcomes(tasks, outcomes, True)
            self.assertIn(channel, str(caught.exception))


class TestSkillFinCellConsumption(unittest.TestCase):
    """Test the fin_cell channel: an engine that owns the cell degrees of
    freedom reports the box it ended on, and the framework applies cell,
    lattice_vector and volume together."""

    CELL = [10.0, 11.0, 12.0, 90.0, 90.0, 90.0]

    def _run_with_result(self, result_dict, pbc=(1, 1, 1), cell=None):
        from pyrai3md.workflow.calculator import Calculator
        state = _make_mock_state()
        state.pmd.system.pbc = list(pbc)
        state.pmd.system.cell = list(cell if cell is not None else self.CELL)
        task = _make_task('qm', 0, 1, 'all', [])
        state.add_task('theory', task)
        backend = MockBackend(result_dict=result_dict)
        calc = Calculator(state=state, dummy_backend=backend)
        result = calc.run()
        return state, task, result

    def test_fin_cell_updates_cell_lattice_and_volume(self):
        moved = [10.5, 10.5, 10.5, 90.0, 90.0, 90.0]
        state, _, _ = self._run_with_result(
            {'energy': [-75.0], 'fin_cell': moved})
        from pyrai3md.utilities.lattice import cell6_to_matrix, cell_volume
        system = state.pmd.system
        self.assertEqual([float(v) for v in system.cell], moved)
        np.testing.assert_allclose(
            np.asarray(system.lattice_vector, dtype=float),
            cell6_to_matrix(moved), rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(system.volume,
                              float(cell_volume(cell6_to_matrix(moved))))

    def test_fin_cell_identical_keeps_system(self):
        from pyrai3md.workflow.calculator import Calculator
        state = _make_mock_state()
        state.pmd.system.pbc = [1, 1, 1]
        state.pmd.system.cell = list(self.CELL)
        state.pmd.system.volume = 1234.5      # a value the run must not touch
        task = _make_task('qm', 0, 1, 'all', [])
        state.add_task('theory', task)
        calc = Calculator(
            state=state,
            dummy_backend=MockBackend(result_dict={
                'energy': [-75.0], 'fin_cell': list(self.CELL)}))
        calc.run()
        self.assertEqual([float(v) for v in state.pmd.system.cell],
                         self.CELL)
        self.assertEqual(state.pmd.system.volume, 1234.5)

    def test_fin_cell_empty_is_noop(self):
        state, _, _ = self._run_with_result(
            {'energy': [-75.0], 'fin_cell': []})
        self.assertEqual([float(v) for v in state.pmd.system.cell],
                         self.CELL)

    def test_fin_cell_missing_is_noop(self):
        state, _, _ = self._run_with_result({'energy': [-75.0]})
        self.assertEqual([float(v) for v in state.pmd.system.cell],
                         self.CELL)

    def test_fin_cell_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            self._run_with_result(
                {'energy': [-75.0], 'fin_cell': [10.0, 10.0, 10.0]})

    def test_fin_cell_non_positive_length_raises(self):
        with self.assertRaises(ValueError):
            self._run_with_result(
                {'energy': [-75.0],
                 'fin_cell': [0.0, 10.0, 10.0, 90.0, 90.0, 90.0]})

    def test_fin_cell_bad_angle_raises(self):
        with self.assertRaises(ValueError):
            self._run_with_result(
                {'energy': [-75.0],
                 'fin_cell': [10.0, 10.0, 10.0, 200.0, 90.0, 90.0]})

    def test_fin_cell_degenerate_cell_raises(self):
        # Angles inside (0, 180) that still flatten the cell: the volume
        # check is what catches a box no later step can be wrapped in.
        with self.assertRaises(ValueError):
            self._run_with_result(
                {'energy': [-75.0],
                 'fin_cell': [10.0, 10.0, 10.0, 90.0, 180.0 - 1e-9,
                              90.0]})

    def test_fin_cell_on_an_aperiodic_system_raises(self):
        # The engine ran on a periodic box the framework never declared;
        # accepting the cell would leave pbc contradicting the state.
        with self.assertRaises(ValueError) as caught:
            self._run_with_result(
                {'energy': [-75.0], 'fin_cell': [10.5, 10.5, 10.5,
                                                 90.0, 90.0, 90.0]},
                pbc=(0, 0, 0))
        self.assertIn('pbc', str(caught.exception))

    def test_nine_lattice_components_are_reduced_to_the_six(self):
        # Engines that hold a lattice report nine numbers; the state keeps
        # both forms, so the six must come out of the nine the way the
        # framework's own reduction says, not out of thin air.
        from pyrai3md.utilities.lattice import matrix_to_cell6

        lattice = [[11.0, 0.0, 0.0], [1.0, 9.0, 0.0], [0.5, 0.25, 10.0]]
        flat = [value for row in lattice for value in row]
        state, _, _ = self._run_with_result(
            {'energy': [-75.0], 'fin_cell': flat})
        system = state.pmd.system
        np.testing.assert_allclose(
            np.asarray(system.lattice_vector, dtype=float), lattice,
            rtol=1e-12, atol=1e-12)
        for got, want in zip([float(v) for v in system.cell],
                             matrix_to_cell6(lattice)):
            self.assertAlmostEqual(got, want, places=9)

    def test_fin_cell_of_three_or_twelve_numbers_raises(self):
        for wrong in ([10.0, 10.0, 10.0, 90.0],
                      [10.0] * 12):
            with self.assertRaises(ValueError):
                self._run_with_result(
                    {'energy': [-75.0], 'fin_cell': wrong})

    def test_fin_cell_not_left_in_results(self):
        moved = [10.5, 10.5, 10.5, 90.0, 90.0, 90.0]
        state, task, result = self._run_with_result(
            {'energy': [-75.0], 'fin_cell': moved})
        self.assertNotIn('fin_cell', task['config'].get('results', {}))
        self.assertNotIn('fin_cell', result)
        self.assertNotIn('fin_cell', state.pmd.results)

    def test_composite_rejects_fin_cell(self):
        from pyrai3md.workflow.calculator import Calculator
        state = _make_mock_state()
        state.pmd.system.pbc = [1, 1, 1]
        calc = Calculator(state=state,
                          dummy_backend=MockBackend(result_dict={}))
        tasks = [_make_task('qm', 0, 0.5, 'all', []),
                 _make_task('qm', 1, 0.5, 'all', [])]
        outcomes = [{'raw': {'energy': [-75.0],
                             'fin_cell': [10.5, 10.5, 10.5, 90.0, 90.0,
                                          90.0]}},
                    {'raw': {'energy': [-74.0]}}]
        with self.assertRaises(ValueError) as caught:
            calc._route_outcomes(tasks, outcomes, True)
        self.assertIn('fin_cell', str(caught.exception))


class TestPyRAIStateDataSource(unittest.TestCase):
    """Test PyRAIState pmd single data source behavior."""

    def test_properties_pass_through_pmd(self):
        state = _make_mock_state()
        self.assertEqual(state.jobtype, state.pmd.control.jobtype)
        self.assertEqual(state.atoms, ['C', 'H'])
        self.assertEqual(state.coordinates,
                         [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        self.assertEqual(state.charges, 0)
        self.assertEqual(state.multiplicity, 1)
        self.assertEqual(state.nstate, 2)

    def test_properties_read_only(self):
        state = _make_mock_state()
        for name in ('jobtype', 'atoms', 'coordinates', 'velocities',
                     'charges', 'multiplicity', 'nstate', 'result'):
            with self.assertRaises(AttributeError):
                setattr(state, name, None)

    def test_result_none_when_empty(self):
        state = _make_mock_state()
        self.assertIsNone(state.result)

    def test_update_system_partial_fields(self):
        state = _make_mock_state()
        state.update_system({'coordinates': [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]})
        self.assertEqual(state.coordinates, [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        # Untouched fields stay unchanged
        self.assertEqual(state.atoms, ['C', 'H'])
        self.assertEqual(state.multiplicity, 1)

    def test_update_system_multiple_fields(self):
        state = _make_mock_state()
        state.update_system({
            'coordinates': [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            'velocities': [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            'multiplicity': 3,
        })
        self.assertEqual(state.coordinates, [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        self.assertEqual(state.velocities, [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
        self.assertEqual(state.multiplicity, 3)

    def test_update_system_without_pmd_raises(self):
        from pyrai3md.data.state import PyRAIState
        state = PyRAIState()
        with self.assertRaises(ValueError):
            state.update_system({'coordinates': []})

    def test_set_result_and_result_property(self):
        state = _make_mock_state()
        state.set_result({'energy': -75.0, 'dipole': [0.1, 0.2, 0.3]})
        result = state.result
        self.assertAlmostEqual(result['energy'], -75.0)
        np.testing.assert_array_almost_equal(result['dipole'], [0.1, 0.2, 0.3])

    def test_set_result_converts_numpy(self):
        state = _make_mock_state()
        state.set_result({'gradient': np.array([[0.1, 0.2, 0.3]])})
        self.assertEqual(state.pmd.results['gradient'], [[0.1, 0.2, 0.3]])


if __name__ == '__main__':
    unittest.main()


# ---------------------------------------------------------------------------
# Region operations tests (P1)
# ---------------------------------------------------------------------------

class TestValidateRegion(unittest.TestCase):
    """Test region index validation."""

    def test_valid_indices(self):
        from pyrai3md.workflow.multiscale.regions import validate_region
        validate_region([0, 1, 2], 5)  # should not raise

    def test_empty_indices(self):
        from pyrai3md.workflow.multiscale.regions import validate_region
        validate_region([], 5)  # should not raise

    def test_out_of_range(self):
        from pyrai3md.workflow.multiscale.regions import validate_region
        with self.assertRaises(ValueError):
            validate_region([0, 5], 5)

    def test_negative_index(self):
        from pyrai3md.workflow.multiscale.regions import validate_region
        with self.assertRaises(ValueError):
            validate_region([-1], 5)

    def test_duplicate_indices(self):
        from pyrai3md.workflow.multiscale.regions import validate_region
        with self.assertRaises(ValueError):
            validate_region([0, 1, 1], 5)


class TestMaterializeRegion(unittest.TestCase):
    """Test geometry slicing from global structure."""

    def test_full_system(self):
        from pyrai3md.workflow.multiscale.regions import materialize_region
        atoms = ['C', 'H', 'O']
        coords = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        local_atoms, local_coords = materialize_region([], atoms, coords)
        self.assertEqual(local_atoms, atoms)
        np.testing.assert_array_equal(local_coords, coords)

    def test_sub_region(self):
        from pyrai3md.workflow.multiscale.regions import materialize_region
        atoms = ['C', 'H', 'O', 'N']
        coords = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        local_atoms, local_coords = materialize_region([0, 2], atoms, coords)
        self.assertEqual(local_atoms, ['C', 'O'])
        np.testing.assert_array_equal(
            local_coords, [[0, 0, 0], [0, 1, 0]])

    def test_returns_copy(self):
        from pyrai3md.workflow.multiscale.regions import materialize_region
        atoms = ['C', 'H']
        coords = [[0, 0, 0], [1, 0, 0]]
        local_atoms, local_coords = materialize_region([], atoms, coords)
        local_coords[0, 0] = 999
        self.assertAlmostEqual(coords[0][0], 0)


class TestProjectGradient(unittest.TestCase):
    """Test gradient scatter-add."""

    def test_full_region(self):
        from pyrai3md.workflow.multiscale.regions import project_gradient
        grad = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
        result = project_gradient(grad, [], 2)
        np.testing.assert_array_almost_equal(result, grad)

    def test_sub_region_scatter(self):
        from pyrai3md.workflow.multiscale.regions import project_gradient
        grad = np.array([[[0.1, 0.2, 0.3]]])  # (1, 1, 3)
        result = project_gradient(grad, [2], 4)
        self.assertEqual(result.shape, (1, 4, 3))
        np.testing.assert_array_almost_equal(
            result[0, 2], [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(
            result[0, 0], [0, 0, 0])

    def test_2d_gradient_promoted(self):
        from pyrai3md.workflow.multiscale.regions import project_gradient
        grad = np.array([[0.1, 0.2, 0.3]])  # (1, 3)
        result = project_gradient(grad, [0], 2)
        self.assertEqual(result.shape, (1, 2, 3))

    def test_atom_mismatch_raises(self):
        from pyrai3md.workflow.multiscale.regions import project_gradient
        grad = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])  # 2 atoms
        with self.assertRaises(ValueError):
            project_gradient(grad, [0], 3)  # but only 1 index


class TestR4cCenterProjection(unittest.TestCase):
    """R4c: per-center electronic projection pure functions (docs 8).

    energy/SOC/dipole carry no atom axis (validated LOCAL copies);
    NAC scatter-adds its atom axis to the global system and applies the
    antisymmetry sign for a reversed pair. SOC keeps its magnitude for a
    reversed pair (real Hermitian conjugate is the identity); the phase
    is an explicit hook, never automatic. None sum across centers.
    """

    def test_project_energy_normalizes_1d(self):
        from pyrai3md.workflow.multiscale.regions import project_energy
        self.assertEqual(project_energy(5.0).shape, (1,))
        np.testing.assert_array_almost_equal(
            project_energy([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])

    def test_project_energy_returns_copy(self):
        from pyrai3md.workflow.multiscale.regions import project_energy
        src = np.array([1.0, 2.0])
        out = project_energy(src)
        out[0] = 99.0
        self.assertAlmostEqual(src[0], 1.0)

    def test_project_energy_empty_raises(self):
        from pyrai3md.workflow.multiscale.regions import project_energy
        with self.assertRaises(ValueError):
            project_energy([])

    def test_project_nac_scatter_and_outside_zero(self):
        from pyrai3md.workflow.multiscale.regions import project_nac
        nac = np.ones((2, 2, 3))          # 2 pairs, 2 local atoms
        glob = project_nac(nac, [0, 2], 4)
        self.assertEqual(glob.shape, (2, 4, 3))
        np.testing.assert_array_almost_equal(glob[0, 0], [1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(glob[0, 2], [1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(glob[0, 1], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(glob[1, 3], [0.0, 0.0, 0.0])

    def test_project_nac_sign_flip_for_reversed_pair(self):
        from pyrai3md.workflow.multiscale.regions import project_nac
        nac = np.ones((2, 1, 3))
        glob = project_nac(nac, [0], 1, pair_signs=[-1, 1])
        # pair 0 reversed -> negated; pair 1 canonical -> unchanged
        np.testing.assert_array_almost_equal(glob[0, 0], [-1.0, -1.0, -1.0])
        np.testing.assert_array_almost_equal(glob[1, 0], [1.0, 1.0, 1.0])

    def test_project_nac_2d_promoted(self):
        from pyrai3md.workflow.multiscale.regions import project_nac
        nac = np.ones((2, 3))             # single pair, 2 local atoms
        glob = project_nac(nac, [0, 1], 2)
        self.assertEqual(glob.shape, (1, 2, 3))

    def test_project_nac_full_region(self):
        from pyrai3md.workflow.multiscale.regions import project_nac
        nac = np.ones((1, 3, 3))          # A_c == natom
        glob = project_nac(nac, [], 3)
        self.assertEqual(glob.shape, (1, 3, 3))

    def test_project_nac_sign_length_mismatch_raises(self):
        from pyrai3md.workflow.multiscale.regions import project_nac
        nac = np.ones((2, 1, 3))
        with self.assertRaises(ValueError):
            project_nac(nac, [0], 1, pair_signs=[-1])

    def test_project_nac_bad_shape_raises(self):
        from pyrai3md.workflow.multiscale.regions import project_nac
        with self.assertRaises(ValueError):
            project_nac(np.ones((2, 2, 2)), [0, 1], 2)

    def test_project_soc_normalizes_1d(self):
        from pyrai3md.workflow.multiscale.regions import project_soc
        self.assertEqual(project_soc(0.5).shape, (1,))
        np.testing.assert_array_almost_equal(
            project_soc([0.1, 0.2]), [0.1, 0.2])

    def test_project_soc_no_automatic_sign_flip(self):
        # Real Hermitian conjugate is the identity: a reversed pair
        # keeps its magnitude unless the caller passes an explicit sign.
        from pyrai3md.workflow.multiscale.regions import project_soc
        np.testing.assert_array_almost_equal(
            project_soc([0.1, 0.2]), [0.1, 0.2])
        np.testing.assert_array_almost_equal(
            project_soc([0.1, 0.2], pair_signs=[1, -1]), [0.1, -0.2])

    def test_project_soc_sign_length_mismatch_raises(self):
        from pyrai3md.workflow.multiscale.regions import project_soc
        with self.assertRaises(ValueError):
            project_soc([0.1, 0.2], pair_signs=[1])

    def test_project_dipole_promotes_single(self):
        from pyrai3md.workflow.multiscale.regions import project_dipole
        self.assertEqual(project_dipole([1.0, 2.0, 3.0]).shape, (1, 3))

    def test_project_dipole_per_state_copy(self):
        from pyrai3md.workflow.multiscale.regions import project_dipole
        src = np.zeros((3, 3))
        out = project_dipole(src)
        self.assertEqual(out.shape, (3, 3))
        out[0, 0] = 99.0
        self.assertAlmostEqual(src[0, 0], 0.0)

    def test_project_dipole_bad_shape_raises(self):
        from pyrai3md.workflow.multiscale.regions import project_dipole
        with self.assertRaises(ValueError):
            project_dipole(np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            project_dipole(np.zeros((2, 3, 3)))


# ---------------------------------------------------------------------------
# Composite strict aggregation tests (P1)
# ---------------------------------------------------------------------------

class TestValidatePlan(unittest.TestCase):
    """Test composite plan validation."""

    def test_valid_plan(self):
        from pyrai3md.workflow.multiscale.composite import validate_plan
        tasks = [
            {'task_id': 'qm', 'contribution': 1,
             'region': 'model', 'region_indices': [0, 1]},
            {'task_id': 'qm2', 'contribution': -1,
             'region': 'model', 'region_indices': [0, 1]},
        ]
        validate_plan(tasks, 3)  # should not raise

    def test_duplicate_task_id(self):
        from pyrai3md.workflow.multiscale.composite import validate_plan
        tasks = [
            {'task_id': 'qm', 'contribution': 1,
             'region': 'all', 'region_indices': []},
            {'task_id': 'qm', 'contribution': -1,
             'region': 'all', 'region_indices': []},
        ]
        with self.assertRaises(ValueError):
            validate_plan(tasks, 3)

    def test_zero_contribution(self):
        from pyrai3md.workflow.multiscale.composite import validate_plan
        tasks = [
            {'task_id': 'qm', 'contribution': 0,
             'region': 'all', 'region_indices': []},
        ]
        with self.assertRaises(ValueError):
            validate_plan(tasks, 3)


class TestNormalizeTaskResult(unittest.TestCase):
    """Test result normalization."""

    def test_valid_energy(self):
        from pyrai3md.workflow.multiscale.composite import normalize_task_result
        raw = {'energy': -75.0}
        task = {'task_id': 'qm'}
        request = {'nstate': 1, 'nlocal': 2, 'need_gradient': False}
        result = normalize_task_result(raw, task, request)
        np.testing.assert_array_almost_equal(result['energy'], [-75.0])

    def test_missing_energy(self):
        from pyrai3md.workflow.multiscale.composite import normalize_task_result
        raw = {'gradient': np.zeros((2, 3))}
        task = {'task_id': 'qm'}
        request = {'nstate': 1, 'nlocal': 2, 'need_gradient': True}
        with self.assertRaises(ValueError):
            normalize_task_result(raw, task, request)

    def test_nan_energy_rejected(self):
        from pyrai3md.workflow.multiscale.composite import normalize_task_result
        raw = {'energy': float('nan')}
        task = {'task_id': 'qm'}
        request = {'nstate': 1, 'nlocal': 2, 'need_gradient': False}
        with self.assertRaises(ValueError):
            normalize_task_result(raw, task, request)

    def test_gradient_shape_mismatch(self):
        from pyrai3md.workflow.multiscale.composite import normalize_task_result
        raw = {'energy': -1.0, 'gradient': np.zeros((3, 3))}
        task = {'task_id': 'qm'}
        request = {'nstate': 1, 'nlocal': 2, 'need_gradient': True}
        with self.assertRaises(ValueError):
            normalize_task_result(raw, task, request)

    def test_stress_single_matrix_promoted_single_state(self):
        # A (3, 3) stress is promoted to the generic (1, 3, 3) format.
        from pyrai3md.workflow.multiscale.composite import normalize_task_result
        raw = {'energy': -1.0, 'stress_tensor': np.eye(3) * 2.0}
        task = {'task_id': 'qm'}
        request = {'nstate': 1, 'nlocal': 2, 'need_gradient': False}
        normed = normalize_task_result(raw, task, request)
        self.assertEqual(np.asarray(normed['stress_tensor']).shape, (1, 3, 3))
        np.testing.assert_allclose(normed['stress_tensor'][0], np.eye(3) * 2.0)

    def test_stress_single_matrix_broadcast_to_nstate(self):
        # A single-state (3, 3) stress broadcasts across nstate so a
        # future excited-state stress fits the (nstate, 3, 3) contract.
        from pyrai3md.workflow.multiscale.composite import normalize_task_result
        raw = {'energy': [1.0, 2.0], 'stress_tensor': np.eye(3) * 3.0}
        task = {'task_id': 'qm'}
        request = {'nstate': 2, 'nlocal': 2, 'need_gradient': False}
        normed = normalize_task_result(raw, task, request)
        arr = np.asarray(normed['stress_tensor'])
        self.assertEqual(arr.shape, (2, 3, 3))
        np.testing.assert_allclose(arr[0], np.eye(3) * 3.0)
        np.testing.assert_allclose(arr[1], np.eye(3) * 3.0)

    def test_stress_state_resolved_passes_through(self):
        from pyrai3md.workflow.multiscale.composite import normalize_task_result
        stress = np.stack([np.eye(3), np.eye(3) * 5.0])
        raw = {'energy': [1.0, 2.0], 'stress_tensor': stress}
        task = {'task_id': 'qm'}
        request = {'nstate': 2, 'nlocal': 2, 'need_gradient': False}
        normed = normalize_task_result(raw, task, request)
        np.testing.assert_allclose(
            np.asarray(normed['stress_tensor']), stress)

    def test_stress_wrong_shape_rejected(self):
        from pyrai3md.workflow.multiscale.composite import normalize_task_result
        raw = {'energy': -1.0, 'stress_tensor': np.zeros((3, 4))}
        task = {'task_id': 'qm'}
        request = {'nstate': 1, 'nlocal': 2, 'need_gradient': False}
        with self.assertRaises(ValueError):
            normalize_task_result(raw, task, request)

    def test_stress_nan_rejected(self):
        from pyrai3md.workflow.multiscale.composite import normalize_task_result
        raw = {'energy': -1.0,
               'stress_tensor': np.full((1, 3, 3), float('nan'))}
        task = {'task_id': 'qm'}
        request = {'nstate': 1, 'nlocal': 2, 'need_gradient': False}
        with self.assertRaises(ValueError):
            normalize_task_result(raw, task, request)


class TestStressUnitConversion(unittest.TestCase):
    """Framework-central stress unit conversion (calculator.py).

    The whole-system stress arrives in the provider skill's native
    unit (declared by its manifest units.stress) and is converted once
    into Hartree/Bohr^3 before set_result, so active_stress() and the
    NPT barostats always read atomic units.
    """

    def test_resolve_factor_cp2k_gpa(self):
        from pyrai3md.workflow.calculator import resolve_stress_unit_factor
        from pyrai3md.utilities.units import GPA_TO_HARTREE_PER_BOHR3
        self.assertAlmostEqual(
            resolve_stress_unit_factor('cp2k_skill'),
            GPA_TO_HARTREE_PER_BOHR3)

    def test_resolve_factor_lammps_bar(self):
        from pyrai3md.workflow.calculator import resolve_stress_unit_factor
        from pyrai3md.utilities.units import BAR_TO_HARTREE_PER_BOHR3
        self.assertAlmostEqual(
            resolve_stress_unit_factor('lammps_skill'),
            BAR_TO_HARTREE_PER_BOHR3)

    def test_resolve_factor_xtb_atomic(self):
        # xTB declares hartree/bohr^3 -> identity factor.
        from pyrai3md.workflow.calculator import resolve_stress_unit_factor
        self.assertEqual(resolve_stress_unit_factor('xtb_skill'), 1.0)

    def test_resolve_factor_unknown_skill_defaults_one(self):
        from pyrai3md.workflow.calculator import resolve_stress_unit_factor
        self.assertEqual(
            resolve_stress_unit_factor('nonexistent_skill_xyz'), 1.0)

    def test_resolve_factor_empty_skill_defaults_one(self):
        from pyrai3md.workflow.calculator import resolve_stress_unit_factor
        self.assertEqual(resolve_stress_unit_factor(''), 1.0)
        self.assertEqual(resolve_stress_unit_factor(None), 1.0)

    def test_a_declared_unit_is_read_from_disk_only_once(self):
        # The MD step table priced one manifest read per step at 2.42 ms
        # of an 8.4 ms step, so the resolution is kept; what must not
        # change is the answer.
        import yaml
        from pyrai3md.workflow import calculator as calculator_module
        reads = []
        original = yaml.safe_load

        def counting_load(stream):
            reads.append(stream)
            return original(stream)

        calculator_module._STRESS_UNIT_FACTOR_CACHE.clear()
        try:
            with mock.patch.object(yaml, 'safe_load', counting_load):
                first = calculator_module.resolve_stress_unit_factor(
                    'lammps_skill')
                second = calculator_module.resolve_stress_unit_factor(
                    'lammps_skill')
            self.assertEqual(len(reads), 1)
            self.assertEqual(first, second)
        finally:
            calculator_module._STRESS_UNIT_FACTOR_CACHE.clear()

    def test_an_unknown_unit_keeps_warning_every_single_time(self):
        # The cache holds package data, not defects: a manifest that
        # declares a unit the framework cannot resolve has to keep
        # reporting itself as often as it did before the cache existed.
        import yaml
        from pyrai3md.workflow import calculator as calculator_module

        class RecordingLogger:
            def __init__(self):
                self.warnings = []

            def warning(self, message, *args):
                self.warnings.append(message % args)

        logger = RecordingLogger()
        calculator_module._STRESS_UNIT_FACTOR_CACHE.clear()
        try:
            with mock.patch.object(
                    yaml, 'safe_load',
                    lambda stream: {'units': {'stress': 'furlong_per_week'}}):
                first = calculator_module.resolve_stress_unit_factor(
                    'lammps_skill', logger=logger)
                second = calculator_module.resolve_stress_unit_factor(
                    'lammps_skill', logger=logger)
        finally:
            calculator_module._STRESS_UNIT_FACTOR_CACHE.clear()
        self.assertEqual(first, 1.0)
        self.assertEqual(second, 1.0)
        self.assertEqual(len(logger.warnings), 2)

    def test_convert_no_stress_passthrough(self):
        # A run without stress leaves the data dict untouched.
        from pyrai3md.workflow.calculator import convert_stress_to_atomic
        data = {'energy': -1.0}
        out = convert_stress_to_atomic(
            data, {'skill_name': 'cp2k_skill'})
        self.assertIs(out, data)
        self.assertNotIn('stress_tensor', out)

    def test_convert_gpa_to_atomic(self):
        # CP2K emits GPa; the factor scales it into Hartree/Bohr^3.
        from pyrai3md.workflow.calculator import convert_stress_to_atomic
        from pyrai3md.utilities.units import GPA_TO_HARTREE_PER_BOHR3
        native = np.eye(3) * 2.0
        data = {'stress_tensor': native.copy()[None]}
        convert_stress_to_atomic(data, {'skill_name': 'cp2k_skill'})
        np.testing.assert_allclose(
            np.asarray(data['stress_tensor'])[0],
            np.eye(3) * 2.0 * GPA_TO_HARTREE_PER_BOHR3)

    def test_convert_atomic_skill_noop(self):
        # xTB already emits Hartree/Bohr^3 -> values unchanged.
        from pyrai3md.workflow.calculator import convert_stress_to_atomic
        native = np.eye(3) * 3.0
        data = {'stress_tensor': native.copy()[None]}
        convert_stress_to_atomic(data, {'skill_name': 'xtb_skill'})
        np.testing.assert_allclose(
            np.asarray(data['stress_tensor'])[0], np.eye(3) * 3.0)

    def test_convert_none_system_task_noop(self):
        # No structural provider -> factor 1.0, stress preserved.
        from pyrai3md.workflow.calculator import convert_stress_to_atomic
        native = np.eye(3) * 4.0
        data = {'stress_tensor': native.copy()[None]}
        convert_stress_to_atomic(data, None)
        np.testing.assert_allclose(
            np.asarray(data['stress_tensor'])[0], np.eye(3) * 4.0)


class TestR4bCompositeSeparation(unittest.TestCase):
    """R4b: composite separates per-center saving from ONIOM math.

    Locks the non-aggregate key split (docs 8): the locally usable
    per-center keys (nac/soc/dipole/transition_dipole) pass through
    normalization and are collected as-is per center; the expensive
    on-demand undefined-global-aggregation keys (overlap/hessian) are
    refused for multi-center; and the unconditional backend byproducts
    with no consumer yet (computed_charges) are dropped silently for
    multi-center rather than refused, so an openqp/xtb ONIOM is not
    hard-failed by a key xTB always emits. C>1 builds NO energy[nstate]
    public axis (only U(R;s)/G(R;s) at the active configuration); the
    C=1 traditional full-state view is unchanged.
    """

    @staticmethod
    def _task(task_id, center_index, contribution,
              provides_electronic=False, state_policy='reference',
              reference_state=0, region='all', region_indices=None):
        return {
            'task_id': task_id,
            'center_index': center_index,
            'contribution': contribution,
            'provides_electronic': provides_electronic,
            'state_policy': state_policy,
            'reference_state': reference_state,
            'region': region,
            'region_indices': list(region_indices or []),
            'capabilities': ['energy', 'gradient'],
        }

    @staticmethod
    def _task_outcome(task, raw, success=True, error=None):
        from pyrai3md.workflow.multiscale.composite import TaskOutcome
        return TaskOutcome(
            task['task_id'], task['contribution'], raw,
            task['region_indices'], region=task['region'],
            success=success, error=error)

    def test_a_region_coupling_row_reaches_the_whole_system(self):
        # A derivative coupling is a per-nuclear-coordinate vector of the
        # electronic region that made it, so a region provider answers with
        # (npair, nlocal, 3) while the crossing search steps in whole-system
        # coordinates. Passing it through as-is is what raised
        # `shapes (18,) and (4743,) not aligned` on the first `meci` run that
        # actually received a coupling; the atoms outside the region get no
        # row, because the wavefunction the derivative acts on stops there.
        from pyrai3md.workflow.multiscale.composite import assemble_results
        real = self._task('real', None, 1, region='real')
        high = self._task('high', 0, 1, provides_electronic=True,
                          state_policy='active', region='model',
                          region_indices=[1, 2])
        tasks = [real, high]
        outcomes = [
            self._task_outcome(real, {'energy': [-100.0]}),
            self._task_outcome(high, {'energy': [-10.0],
                                      'nac': [[
                                          [0.0, 1.0, 2.0],
                                          [3.0, 4.0, 5.0]]]}),
        ]
        request = {'natom': 3, 'nstate': 1, 'need_gradient': False,
                   'multi_center': False, 'primary_task_id': 'high',
                   'system_task_id': 'real'}
        rows = np.asarray(
            assemble_results(tasks, outcomes, request)['nac'])
        self.assertEqual(rows.shape, (1, 3, 3))
        self.assertEqual(rows[0][0].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(rows[0][1].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(rows[0][2].tolist(), [3.0, 4.0, 5.0])

    def test_high_layer_deviations_survive_assembly(self):
        # The NN high layer is the electronic primary; its per-step
        # err_energy/err_grad are the adaptive sampling signal and must
        # not be replaced by the LAMMPS baseline's (absent) values.
        from pyrai3md.workflow.multiscale.composite import assemble_results
        real = self._task('real', None, 1, region='real')
        high = self._task('high', 0, 1, provides_electronic=True,
                          state_policy='active', region='model',
                          region_indices=[0, 1])
        low = self._task('low', 0, -1, region='model',
                         region_indices=[0, 1])
        tasks = [real, high, low]
        outcomes = [
            self._task_outcome(real, {'energy': [-100.0],
                                      'err_energy': 9.9}),
            self._task_outcome(high, {'energy': [-10.0],
                                      'err_energy': 0.1,
                                      'err_grad': 0.2}),
            self._task_outcome(low, {'energy': [-2.0]}),
        ]
        request = {'natom': 2, 'nstate': 1, 'need_gradient': False,
                   'multi_center': False, 'primary_task_id': 'high',
                   'system_task_id': 'real'}
        data = assemble_results(tasks, outcomes, request)
        self.assertAlmostEqual(float(np.asarray(data['energy']).flat[0]),
                               -108.0)
        self.assertAlmostEqual(float(data['err_energy']), 0.1)
        self.assertAlmostEqual(float(data['err_grad']), 0.2)

    def test_per_center_keys_pass_through_normalize(self):
        from pyrai3md.workflow.multiscale.composite import normalize_task_result
        raw = {'energy': [1.0, 2.0],
               'gradient': np.zeros((2, 3, 3)),
               'nac': np.zeros((1, 3, 3)),
               'soc': [0.1, 0.2],
               'dipole': np.zeros((2, 3)),
               'transition_dipole': np.zeros((2, 3))}
        task = self._task('h0', 0, 1, provides_electronic=True,
                          state_policy='active')
        request = {'nstate': 2, 'nlocal': 3, 'need_gradient': True,
                   'multi_center': True}
        normed = normalize_task_result(raw, task, request)
        for key in ('nac', 'soc', 'dipole', 'transition_dipole'):
            self.assertIn(key, normed)

    def test_refused_keys_still_raise_for_multi_center(self):
        from pyrai3md.workflow.multiscale.composite import normalize_task_result
        for key in ('overlap', 'hessian'):
            with self.subTest(key=key):
                raw = {'energy': [1.0], key: np.zeros((1, 1))}
                task = self._task('h0', 0, 1, provides_electronic=True,
                                  state_policy='active')
                request = {'nstate': 1, 'nlocal': 1,
                           'need_gradient': False, 'multi_center': True}
                with self.assertRaises(ValueError):
                    normalize_task_result(raw, task, request)

    def test_ignored_byproduct_dropped_for_multi_center(self):
        # computed_charges is an unconditional xTB/pyscf byproduct with
        # no ONIOM sum rule and no framework consumer yet, so a
        # multi-center run drops it silently instead of hard-failing
        # (regression: openqp/xtb ONIOM aborted with "refuses key
        # 'computed_charges' ... no ONIOM sum rule defined").
        from pyrai3md.workflow.multiscale.composite import normalize_task_result
        raw = {'energy': [1.0], 'computed_charges': np.zeros((1, 3))}
        task = self._task('qm2', None, 1)
        request = {'nstate': 1, 'nlocal': 3, 'need_gradient': False,
                   'multi_center': True}
        normed = normalize_task_result(raw, task, request)
        self.assertNotIn('computed_charges', normed)
        self.assertIn('energy', normed)

    def test_refused_keys_allowed_when_single_center(self):
        # The refusal is multi-center only; C=1 keeps legacy tolerance.
        from pyrai3md.workflow.multiscale.composite import normalize_task_result
        raw = {'energy': [1.0], 'hessian': np.zeros((1, 1))}
        task = self._task('qm', None, 1)
        request = {'nstate': 1, 'nlocal': 1, 'need_gradient': False,
                   'multi_center': False}
        normed = normalize_task_result(raw, task, request)
        self.assertIn('hessian', normed)

    def test_ignored_byproduct_passes_through_when_single_center(self):
        # C=1 keeps the legacy pass-through for computed_charges (the
        # silent drop is multi-center only).
        from pyrai3md.workflow.multiscale.composite import normalize_task_result
        raw = {'energy': [1.0], 'computed_charges': np.zeros((1, 3))}
        task = self._task('qm', None, 1)
        request = {'nstate': 1, 'nlocal': 3, 'need_gradient': False,
                   'multi_center': False}
        normed = normalize_task_result(raw, task, request)
        self.assertIn('computed_charges', normed)

    def test_collect_gathers_only_providers(self):
        from pyrai3md.workflow.multiscale.composite import collect_center_electronic
        base = self._task('base', None, 1)          # L(real) baseline
        h0 = self._task('h0', 0, 1, provides_electronic=True,
                        state_policy='active')
        l0 = self._task('l0', 0, -1)                # reference corr
        h1 = self._task('h1', 1, 1, provides_electronic=True,
                        state_policy='active')
        tasks = [base, h0, l0, h1]
        outcomes = [
            self._task_outcome(base, {'energy': [100.0]}),
            self._task_outcome(h0, {'energy': [10.0, 12.0],
                                    'nac': np.zeros((1, 3, 3)),
                                    'soc': [0.1]}),
            self._task_outcome(l0, {'energy': [3.0]}),
            self._task_outcome(h1, {'energy': [20.0, 21.0, 22.0],
                                    'dipole': np.zeros((3, 3))}),
        ]
        request = {'natom': 3, 'nstate': 1, 'need_gradient': False,
                   'multi_center': True}
        per_center = collect_center_electronic(tasks, outcomes, request)
        self.assertEqual(set(per_center), {0, 1})
        np.testing.assert_array_almost_equal(
            per_center[0]['energy'], [10.0, 12.0])
        self.assertIn('nac', per_center[0])
        self.assertIn('soc', per_center[0])
        self.assertNotIn('dipole', per_center[0])
        self.assertIn('dipole', per_center[1])
        self.assertNotIn('nac', per_center[1])

    def test_collect_skips_failed_provider(self):
        from pyrai3md.workflow.multiscale.composite import collect_center_electronic
        h0 = self._task('h0', 0, 1, provides_electronic=True,
                        state_policy='active')
        outcomes = [self._task_outcome(h0, {'energy': [1.0]},
                                       success=False, error='boom')]
        request = {'natom': 1, 'nstate': 1, 'multi_center': True}
        self.assertEqual(
            collect_center_electronic([h0], outcomes, request), {})

    def test_assemble_multi_center_active_energy(self):
        from pyrai3md.workflow.multiscale.composite import assemble_results
        base = self._task('base', None, 1)
        h0 = self._task('h0', 0, 1, provides_electronic=True,
                        state_policy='active')
        l0 = self._task('l0', 0, -1, reference_state=0)
        h1 = self._task('h1', 1, 1, provides_electronic=True,
                        state_policy='active')
        l1 = self._task('l1', 1, -1, reference_state=0)
        tasks = [base, h0, l0, h1, l1]
        outcomes = [
            self._task_outcome(base, {'energy': [100.0]}),
            self._task_outcome(h0, {'energy': [10.0, 12.0]}),
            self._task_outcome(l0, {'energy': [3.0, 3.5]}),
            self._task_outcome(h1, {'energy': [20.0, 21.0, 22.0]}),
            self._task_outcome(l1, {'energy': [5.0]}),
        ]
        request = {'natom': 1, 'nstate': 1, 'need_gradient': False,
                   'multi_center': True, 'active_configuration': (1, 2)}
        data = assemble_results(tasks, outcomes, request)
        # U = +100 (base) +12 (H0@s1) -3 (L0@ref0) +22 (H1@s2) -5 (L1)
        self.assertAlmostEqual(data['active_energy'], 126.0)
        self.assertAlmostEqual(
            float(np.asarray(data['energy']).flat[0]), 126.0)
        self.assertEqual(data['active_configuration'], (1, 2))
        # C>1 builds NO energy[nstate] axis: one total energy only
        self.assertEqual(np.asarray(data['energy']).size, 1)

    def test_assemble_multi_center_active_state_out_of_range(self):
        from pyrai3md.workflow.multiscale.composite import assemble_results
        h0 = self._task('h0', 0, 1, provides_electronic=True,
                        state_policy='active')
        outcomes = [self._task_outcome(h0, {'energy': [1.0, 2.0]})]
        request = {'natom': 1, 'nstate': 1, 'need_gradient': False,
                   'multi_center': True, 'active_configuration': (5,)}
        with self.assertRaises(ValueError):
            assemble_results([h0], outcomes, request)

    def test_assemble_multi_center_gradient_scatter(self):
        # Routes to the multi-center assembler via the request flag;
        # H/L pair on one center verifies state selection + scatter.
        from pyrai3md.workflow.multiscale.composite import assemble_results
        h0 = self._task('h0', 0, 1, provides_electronic=True,
                        state_policy='active', region='model',
                        region_indices=[0, 1])
        l0 = self._task('l0', 0, -1, reference_state=0, region='model',
                        region_indices=[2, 3])
        g_h = np.ones((2, 2, 3)) * 0.5
        g_h[1] = 1.0                       # active state 1 gradient
        g_l = np.ones((1, 2, 3)) * 0.25
        outcomes = [
            self._task_outcome(h0, {'energy': [1.0, 2.0], 'gradient': g_h}),
            self._task_outcome(l0, {'energy': [0.5], 'gradient': g_l}),
        ]
        request = {'natom': 4, 'nstate': 1, 'need_gradient': True,
                   'multi_center': True, 'active_configuration': (1,)}
        data = assemble_results([h0, l0], outcomes, request)
        grad = np.asarray(data['active_gradient'])
        self.assertEqual(grad.shape, (4, 3))
        np.testing.assert_array_almost_equal(grad[0], [1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(grad[1], [1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(
            grad[2], [-0.25, -0.25, -0.25])
        np.testing.assert_array_almost_equal(
            grad[3], [-0.25, -0.25, -0.25])
        self.assertEqual(np.asarray(data['gradient']).shape, (1, 4, 3))

    def test_single_center_full_state_view_unchanged(self):
        from pyrai3md.workflow.multiscale.composite import assemble_results
        t0 = self._task('qm', None, 1)
        t1 = self._task('qm2', None, 1)
        t2 = self._task('qm3', None, -1)
        outcomes = [
            self._task_outcome(t0, {'energy': [10.0, 10.0]}),
            self._task_outcome(t1, {'energy': [5.0, 5.0]}),
            self._task_outcome(t2, {'energy': [2.0, 2.0]}),
        ]
        request = {'natom': 1, 'nstate': 2, 'need_gradient': False,
                   'multi_center': False, 'primary_task_id': 'qm'}
        data = assemble_results([t0, t1, t2], outcomes, request)
        # C=1 keeps the traditional energy[nstate] axis: 10 + 5 - 2
        self.assertEqual(np.asarray(data['energy']).shape, (2,))
        np.testing.assert_array_almost_equal(
            np.asarray(data['energy']), [13.0, 13.0])


# ---------------------------------------------------------------------------
# Adapter semicolon parsing tests (P2)
# ---------------------------------------------------------------------------

class TestParseCenterIndices(unittest.TestCase):
    """Test semicolon-separated center index parsing."""

    def test_flat_string_single_center(self):
        from pyrai3md.schema.adapter import _parse_center_indices
        result = _parse_center_indices('1 2 3', 'highlevel', start=1)
        self.assertEqual(result, [[0, 1, 2]])

    def test_semicolon_two_centers(self):
        from pyrai3md.schema.adapter import _parse_center_indices
        result = _parse_center_indices('1-3; 11-14', 'highlevel', start=1)
        self.assertEqual(result, [[0, 1, 2], [10, 11, 12, 13]])

    def test_range_with_spaces(self):
        from pyrai3md.schema.adapter import _parse_center_indices
        result = _parse_center_indices('1-3 5; 11-14', 'highlevel', start=1)
        self.assertEqual(result, [[0, 1, 2, 4], [10, 11, 12, 13]])

    def test_flat_list_input(self):
        from pyrai3md.schema.adapter import _parse_center_indices
        result = _parse_center_indices([1, 2, 3], 'highlevel', start=1)
        self.assertEqual(result, [[0, 1, 2]])

    def test_nested_list_input(self):
        from pyrai3md.schema.adapter import _parse_center_indices
        result = _parse_center_indices([[1, 2], [3, 4]], 'highlevel', start=1)
        self.assertEqual(result, [[0, 1], [2, 3]])

    def test_empty_group_raises(self):
        from pyrai3md.schema.adapter import _parse_center_indices
        with self.assertRaises(ValueError):
            _parse_center_indices('1 2;; 3 4', 'highlevel', start=1)

    def test_trailing_semicolon_raises(self):
        from pyrai3md.schema.adapter import _parse_center_indices
        with self.assertRaises(ValueError):
            _parse_center_indices('1 2;', 'highlevel', start=1)

    def test_empty_string(self):
        from pyrai3md.schema.adapter import _parse_center_indices
        result = _parse_center_indices('', 'highlevel', start=1)
        self.assertEqual(result, [[]])


# ---------------------------------------------------------------------------
# Calculator executor wiring tests (R3 / r3d)
# ---------------------------------------------------------------------------

class _FixedEnergyBackend:
    """Thread-safe mock backend returning a fixed per-task energy.

    The energy comes from the task config, so concurrent terms never
    share mutable state (unlike a closure counter): each backend
    instance owns its own call_count and result.
    """

    def __init__(self, config=None, execution_id=''):
        self.config = config or {}
        self.execution_id = execution_id
        self.call_count = 0

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None):
        self.call_count += 1
        natom = len(atoms)
        return {
            'energy': self.config.get('energy', 0.0),
            'gradient': np.ones((1, natom, 3)) * 0.01,
        }


class _RefusingBackend(_FixedEnergyBackend):
    """Mock backend that declines a thread pool the way a skill does.

    check_thread_safe() is the skill-side hook (#301): only the skill
    knows which process globals its calculation touches, so the skill
    refuses and states its own reason while the framework only
    propagates.
    """

    def check_thread_safe(self, runner, execution_mode, profile_semantics):
        raise ValueError(
            "mock_skill holds process-wide state, so runner='%s', "
            "execution_mode='%s', profile_semantics='%s' cannot share a "
            "thread pool" % (runner, execution_mode, profile_semantics))


class TestCalculatorExecutorWiring(unittest.TestCase):
    """Test run_in_process executor wiring + thread isolation (r3d)."""

    # _make_task stamps skill_name='mock_skill' and
    # capabilities=['energy','gradient']; the calculator derives the
    # thread contract from them (the dict config has no skill_config,
    # so execution_mode is 'local'). The contract is what the backend's
    # own check_thread_safe() is handed; nothing registers it.
    AUDIT_CONTRACT = ('mock_skill', 'in_process', 'local',
                      'energy+gradient')

    def _make_composite_state(self, api=None):
        """Three-task ONIOM-style composite: 10 + 5 - 2 = 13."""
        state = _make_mock_state()
        specs = [('qm', 0, 1, 10.0), ('qm2', 0, 1, 5.0),
                 ('qm2', 1, -1, 2.0)]
        for role, idx, contribution, energy in specs:
            task = _make_task(role, idx, contribution, 'all', [])
            task['api'] = api or _FixedEnergyBackend
            task['config'] = {'results': {}, 'energy': energy}
            state.add_task('theory', task)
        return state

    def test_default_executor_is_serial(self):
        from pyrai3md.workflow.calculator import Calculator
        calc = Calculator(
            state=self._make_composite_state()).initialize()
        # No descriptor: serial in-process default (behavior unchanged).
        self.assertEqual(calc._resolve_executor().mode, 'serial')

    def test_descriptor_selects_thread_executor(self):
        from pyrai3md.workflow.calculator import Calculator
        calc = Calculator(
            state=self._make_composite_state(),
            backend_execution={'theory': {'executor': 'thread',
                                          'max_workers': 3}},
        ).initialize()
        executor = calc._resolve_executor()
        self.assertEqual(executor.mode, 'thread')
        self.assertEqual(executor.max_workers, 3)

    def test_thread_mode_matches_serial_aggregation(self):
        from pyrai3md.workflow.calculator import Calculator
        serial = Calculator(
            state=self._make_composite_state()).initialize().run()
        thread = Calculator(
            state=self._make_composite_state(),
            backend_execution={'theory': {'executor': 'thread',
                                          'max_workers': 3}},
        ).initialize().run()
        # Identical deterministic summation regardless of the executor.
        np.testing.assert_array_almost_equal(
            np.asarray(serial['energy']), np.asarray(thread['energy']))
        self.assertAlmostEqual(
            float(np.asarray(thread['energy']).flat[0]), 13.0)

    def test_thread_mode_backend_refusal_fails_before_running(self):
        from pyrai3md.workflow.calculator import Calculator
        calc = Calculator(
            state=self._make_composite_state(api=_RefusingBackend),
            backend_execution={'theory': {'executor': 'thread',
                                          'max_workers': 2}},
        ).initialize()
        with self.assertRaises(ValueError) as caught:
            calc.run()
        # The refusal is the skill's own sentence, not a framework
        # message: it names the contract it was handed.
        self.assertIn('mock_skill holds process-wide state',
                      str(caught.exception))
        self.assertIn("profile_semantics='energy+gradient'",
                      str(caught.exception))
        # The pool refused to start: no term ran, never degraded.
        for backend in calc.backends['theory']:
            self.assertEqual(backend.call_count, 0)

    def test_evaluation_id_advances_per_run(self):
        from pyrai3md.workflow.calculator import Calculator
        calc = Calculator(
            state=self._make_composite_state()).initialize()
        first = calc._next_evaluation_id()
        self.assertEqual(calc._next_evaluation_id(), first + 1)

    def test_work_items_carry_identity_and_audit_key(self):
        from pyrai3md.workflow.calculator import Calculator
        state = self._make_composite_state()
        calc = Calculator(state=state, runtime_idx='traj_2').initialize()
        coords = np.asarray(state.coordinates, dtype=np.float64)
        items = calc._build_work_items(
            calc._active_tasks(), calc.backends['theory'],
            list(state.atoms), coords, None, 7)
        self.assertEqual(len(items), 3)
        for item in items:
            self.assertEqual(item['evaluation_id'], 7)
            self.assertEqual(item['runner'], 'in_process')
            self.assertEqual(item['backend_name'], 'mock_skill')
            self.assertEqual(item['profile_semantics'], 'energy+gradient')
            self.assertEqual(item['identity']['trajectory'], 'traj_2')
            self.assertEqual(item['identity']['evaluation_id'], 7)

    def test_snapshot_read_only_terms_get_writable_copies(self):
        from pyrai3md.workflow.calculator import Calculator
        state = self._make_composite_state()
        calc = Calculator(state=state).initialize()
        coords = np.array(state.coordinates, dtype=np.float64, copy=True)
        coords.setflags(write=False)
        items = calc._build_work_items(
            calc._active_tasks(), calc.backends['theory'],
            list(state.atoms), coords, None, 1)
        # Each term owns a writable private copy; the frozen snapshot
        # stays read-only (a worker cannot corrupt the shared geometry).
        for item in items:
            self.assertTrue(item['local_coords'].flags.writeable)
        self.assertFalse(coords.flags.writeable)

    def test_validate_outcomes_rejects_stale(self):
        from pyrai3md.workflow.calculator import Calculator
        calc = Calculator(
            state=self._make_composite_state()).initialize()
        items = [{'task_id': 'qm', 'evaluation_id': 1}]
        stale = [{'task_id': 'qm', 'evaluation_id': 0, 'success': True}]
        with self.assertRaises(RuntimeError):
            calc._validate_outcomes(items, stale, 1)

    def test_validate_outcomes_rejects_duplicate(self):
        from pyrai3md.workflow.calculator import Calculator
        calc = Calculator(
            state=self._make_composite_state()).initialize()
        items = [{'task_id': 'qm', 'evaluation_id': 1},
                 {'task_id': 'qm2', 'evaluation_id': 1}]
        dup = [{'task_id': 'qm', 'evaluation_id': 1, 'success': True},
               {'task_id': 'qm', 'evaluation_id': 1, 'success': True}]
        with self.assertRaises(RuntimeError):
            calc._validate_outcomes(items, dup, 1)

    def test_validate_outcomes_rejects_missing(self):
        from pyrai3md.workflow.calculator import Calculator
        calc = Calculator(
            state=self._make_composite_state()).initialize()
        items = [{'task_id': 'qm', 'evaluation_id': 1},
                 {'task_id': 'qm2', 'evaluation_id': 1}]
        partial = [{'task_id': 'qm', 'evaluation_id': 1, 'success': True}]
        with self.assertRaises(RuntimeError):
            calc._validate_outcomes(items, partial, 1)

    def test_remotetask_descriptor_routes_remote_terms(self):
        from unittest import mock

        from pyrai3md.workflow import calculator as calc_module
        from pyrai3md.workflow.calculator import Calculator

        captured = {}

        def fake_execute_remote_terms(work_items, tasks, state,
                                      descriptor=None):
            del state
            captured['count'] = len(work_items)
            captured['descriptor'] = descriptor
            # Emulate the remote per-term outcomes (index-aligned):
            # one calculate() per term, the same raw the serial path
            # would produce, so the assembly is identical.
            outcomes = []
            for item in work_items:
                raw = item['backend'].calculate(
                    item['local_atoms'], item['local_coords'],
                    target_states=item.get('target_states'))
                outcomes.append({
                    'task_id': item['task_id'],
                    'evaluation_id': item['evaluation_id'],
                    'raw': raw, 'success': True, 'error': None})
            return outcomes

        calc = Calculator(
            state=self._make_composite_state(),
            backend_execution={'theory': {'executor': 'thread',
                                          'max_workers': 3,
                                          'remotetask': True}},
        ).initialize()
        with mock.patch.object(calc_module, 'execute_remote_terms',
                               side_effect=fake_execute_remote_terms):
            result = calc.run()
        # remotetask wins over the thread executor: every term went
        # remote, and the composite sum is unchanged (10 + 5 - 2).
        self.assertEqual(captured['count'], 3)
        self.assertTrue(captured['descriptor']['remotetask'])
        self.assertAlmostEqual(
            float(np.asarray(result['energy']).flat[0]), 13.0)

    def test_local_descriptor_does_not_route_remote(self):
        from unittest import mock

        from pyrai3md.workflow import calculator as calc_module
        from pyrai3md.workflow.calculator import Calculator

        calc = Calculator(
            state=self._make_composite_state()).initialize()
        with mock.patch.object(
                calc_module, 'execute_remote_terms',
                side_effect=AssertionError('remote path must not run')):
            result = calc.run()
        self.assertAlmostEqual(
            float(np.asarray(result['energy']).flat[0]), 13.0)


# ---------------------------------------------------------------------------
# R3 gate tests (r3f): real concurrency, out-of-order == serial, the four
# molecule-flag combinations and the orthogonal base/term execution axes.
# ---------------------------------------------------------------------------

class _BarrierBackend:
    """Dummy backend proving real thread-pool concurrency (r3f gate).

    Every term must arrive at the barrier before any proceeds, so a
    serial executor would time out (BrokenBarrierError) and fail the
    evaluation; a passing run proves the pool ran `parties` terms
    concurrently. The shared tracker records the observed peak.
    """

    def __init__(self, barrier, tracker, config=None, execution_id=''):
        self.barrier = barrier
        self.tracker = tracker
        self.config = config or {}
        self.execution_id = execution_id

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None):
        del point_charges, target_states
        with self.tracker['lock']:
            self.tracker['active'] += 1
            self.tracker['total'] += 1
            self.tracker['peak'] = max(self.tracker['peak'],
                                       self.tracker['active'])
        try:
            self.barrier.wait(timeout=5)
        finally:
            with self.tracker['lock']:
                self.tracker['active'] -= 1
        natom = len(atoms)
        return {'energy': self.config.get('energy', 0.0),
                'gradient': np.ones((1, natom, 3)) * 0.01}


class _SleepTrackerBackend:
    """Dummy backend recording the concurrency peak under a worker cap.

    Used with max_workers < n_terms: the pool's hard worker cap keeps
    the observed peak <= W regardless of timing.
    """

    def __init__(self, tracker, config=None, execution_id=''):
        self.tracker = tracker
        self.config = config or {}
        self.execution_id = execution_id

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None):
        import time
        del point_charges, target_states
        with self.tracker['lock']:
            self.tracker['active'] += 1
            self.tracker['total'] += 1
            self.tracker['peak'] = max(self.tracker['peak'],
                                       self.tracker['active'])
        try:
            time.sleep(0.05)
        finally:
            with self.tracker['lock']:
                self.tracker['active'] -= 1
        natom = len(atoms)
        return {'energy': self.config.get('energy', 0.0),
                'gradient': np.ones((1, natom, 3)) * 0.01}


class _StaggeredEnergyBackend:
    """Per-task backend with a config-driven delay (reverse completion).

    The delay forces a deterministic out-of-order completion in the
    thread pool; the executor's index-aligned outcomes must make the
    assembly identical to the serial run.
    """

    def __init__(self, config=None, execution_id=''):
        self.config = config or {}
        self.execution_id = execution_id
        self.call_count = 0

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None):
        import time
        del point_charges, target_states
        self.call_count += 1
        delay = float(self.config.get('delay', 0.0))
        if delay:
            time.sleep(delay)
        natom = len(atoms)
        return {'energy': self.config.get('energy', 0.0),
                'gradient': np.ones((1, natom, 3)) * 0.01}


class TestR3Gate(unittest.TestCase):
    """R3 acceptance gate (docs R3, r3f).

    Proves the executor wiring really enters a thread pool (concurrency
    peak > 1 and <= W), that an out-of-order completion archives/sums
    identically to serial, that the four molecule-flag combinations
    (serial/thread x local/remote) all pass, and that the base-node
    multitask axis is orthogonal to the inner term-thread axis.
    """

    def _make_composite_state(self):
        """Three-task ONIOM-style composite: 10 + 5 - 2 = 13."""
        state = _make_mock_state()
        specs = [('qm', 0, 1, 10.0), ('qm2', 0, 1, 5.0),
                 ('qm2', 1, -1, 2.0)]
        for role, idx, contribution, energy in specs:
            task = _make_task(role, idx, contribution, 'all', [])
            task['api'] = _FixedEnergyBackend
            task['config'] = {'results': {}, 'energy': energy}
            state.add_task('theory', task)
        return state

    def _make_staggered_state(self, delays):
        """Composite whose terms carry per-task completion delays."""
        state = _make_mock_state()
        specs = [('qm', 0, 1, 10.0), ('qm2', 0, 1, 5.0),
                 ('qm2', 1, -1, 2.0)]
        for (role, idx, contribution, energy), delay in zip(specs,
                                                            delays):
            task = _make_task(role, idx, contribution, 'all', [])
            task['api'] = _StaggeredEnergyBackend
            task['config'] = {'results': {}, 'energy': energy,
                              'delay': delay}
            state.add_task('theory', task)
        return state

    def test_thread_pool_enters_real_concurrency(self):
        import threading

        from pyrai3md.workflow.calculator import Calculator

        n_terms = 3
        barrier = threading.Barrier(n_terms, timeout=5)
        tracker = {'active': 0, 'peak': 0, 'total': 0,
                   'lock': threading.Lock()}
        backend = _BarrierBackend(barrier, tracker,
                                  config={'energy': 4.0})
        calc = Calculator(
            state=self._make_composite_state(),
            dummy_backend=backend,
            backend_execution={'theory': {'executor': 'thread',
                                          'max_workers': n_terms}},
        ).initialize()
        # A serial executor would deadlock at the barrier (timeout) and
        # fail the evaluation; reaching here proves all n_terms ran at
        # once. The peak is > 1 and never exceeds the worker count.
        calc.run()
        self.assertEqual(tracker['total'], n_terms)
        self.assertEqual(tracker['peak'], n_terms)
        self.assertGreater(tracker['peak'], 1)
        self.assertLessEqual(tracker['peak'], n_terms)

    def test_concurrency_peak_never_exceeds_workers(self):
        import threading

        from pyrai3md.workflow.calculator import Calculator

        tracker = {'active': 0, 'peak': 0, 'total': 0,
                   'lock': threading.Lock()}
        backend = _SleepTrackerBackend(tracker, config={'energy': 1.0})
        # 3 terms but only 2 workers: the pool cap bounds the peak.
        calc = Calculator(
            state=self._make_composite_state(),
            dummy_backend=backend,
            backend_execution={'theory': {'executor': 'thread',
                                          'max_workers': 2}},
        ).initialize()
        calc.run()
        self.assertEqual(tracker['total'], 3)
        self.assertLessEqual(tracker['peak'], 2)

    def test_invalid_worker_count_fails_before_running(self):
        from pyrai3md.workflow.dispatch.terms import TaskExecutor
        # An invalid worker count / mode fails at construction, before
        # any backend term is staged or run.
        with self.assertRaises(ValueError):
            TaskExecutor(mode='thread', max_workers=0)
        with self.assertRaises(ValueError):
            TaskExecutor(mode='bogus')

    def test_out_of_order_completion_matches_serial(self):
        from pyrai3md.workflow.calculator import Calculator

        serial = Calculator(
            state=self._make_staggered_state([0.0, 0.0, 0.0]),
        ).initialize().run()
        # Reverse completion order: term0 slowest, term2 fastest.
        thread = Calculator(
            state=self._make_staggered_state([0.09, 0.05, 0.0]),
            backend_execution={'theory': {'executor': 'thread',
                                          'max_workers': 3}},
        ).initialize().run()
        # Index-aligned outcomes make the archive position and the
        # summation order independent of the completion order.
        np.testing.assert_array_almost_equal(
            np.asarray(serial['energy']), np.asarray(thread['energy']))
        self.assertIn('gradient', thread)
        np.testing.assert_array_almost_equal(
            np.asarray(serial['gradient']), np.asarray(thread['gradient']))
        self.assertAlmostEqual(
            float(np.asarray(thread['energy']).flat[0]), 13.0)

    def test_four_flag_combinations(self):
        from unittest import mock

        from pyrai3md.workflow import calculator as calc_module
        from pyrai3md.workflow.calculator import Calculator

        def fake_remote(work_items, tasks, state, descriptor=None):
            del tasks, state, descriptor
            return [{
                'task_id': it['task_id'],
                'evaluation_id': it['evaluation_id'],
                'raw': it['backend'].calculate(
                    it['local_atoms'], it['local_coords'],
                    target_states=it.get('target_states')),
                'success': True, 'error': None} for it in work_items]

        combos = [
            ({'executor': 'serial'}, 'serial+local'),
            ({'executor': 'thread', 'max_workers': 3}, 'thread+local'),
            ({'executor': 'serial', 'remotetask': True}, 'serial+remote'),
            ({'executor': 'thread', 'max_workers': 3,
              'remotetask': True}, 'thread+remote'),
        ]
        for descriptor, label in combos:
            with self.subTest(combo=label):
                calc = Calculator(
                    state=self._make_composite_state(),
                    backend_execution={'theory': descriptor},
                ).initialize()
                with mock.patch.object(
                        calc_module, 'execute_remote_terms',
                        side_effect=fake_remote):
                    result = calc.run()
                self.assertAlmostEqual(
                    float(np.asarray(result['energy']).flat[0]), 13.0,
                    msg=label)

    def test_md_base_multitask_false_inner_thread_works(self):
        from pyrai3md.workflow import (_graph_backend_execution,
                                       _graph_base_flags)
        from pyrai3md.workflow.calculator import Calculator

        graph = [{'node': 'base', 'role': 'theory', 'multitask': False,
                  'remotetask': False, 'multipool': 1,
                  'backend_execution': {'executor': 'thread',
                                        'max_workers': 3}}]
        # Outer axis: the MD-style base node keeps multitask=False (no
        # process fan-out); inner axis: the term descriptor still
        # requests a thread pool. The two axes are orthogonal.
        self.assertEqual(_graph_base_flags(graph)['theory'][0], False)
        self.assertEqual(
            _graph_backend_execution(graph),
            {'theory': {'executor': 'thread', 'max_workers': 3}})

        calc = Calculator(
            state=self._make_composite_state(),
            backend_execution=_graph_backend_execution(graph),
        ).initialize()
        self.assertEqual(calc._resolve_executor().mode, 'thread')
        result = calc.run()
        self.assertAlmostEqual(
            float(np.asarray(result['energy']).flat[0]), 13.0)


# ---------------------------------------------------------------------------
# Task executor tests (P4)
# ---------------------------------------------------------------------------

class TestTaskExecutor(unittest.TestCase):
    """Test serial and thread task executor."""

    def test_serial_execution(self):
        from pyrai3md.workflow.dispatch.terms import TaskExecutor
        executor = TaskExecutor(mode='serial')

        class FakeBackend:
            def calculate(self, atoms, coords, point_charges=None, target_states=None, cell=None):
                return {'energy': -1.0}

        items = [
            {'backend': FakeBackend(), 'local_atoms': ['C'],
             'local_coords': np.zeros((1, 3)),
             'target_states': None, 'task_id': 't1'},
            {'backend': FakeBackend(), 'local_atoms': ['H'],
             'local_coords': np.zeros((1, 3)),
             'target_states': None, 'task_id': 't2'},
        ]
        outcomes = executor.execute(items)
        self.assertEqual(len(outcomes), 2)
        self.assertTrue(all(o['success'] for o in outcomes))

    def test_thread_backend_without_hook_runs(self):
        from pyrai3md.workflow.dispatch.terms import (
            TaskExecutor, make_work_item)
        executor = TaskExecutor(mode='thread', max_workers=2)

        class FakeBackend:
            def calculate(self, atoms, coords, point_charges=None, target_states=None, cell=None):
                return {'energy': -1.0}

        # A backend that declares nothing is taken as safe to run
        # concurrently: the skill opts out by name, the framework keeps
        # no capability table (#301).
        items = [make_work_item(
            't1', FakeBackend(), ['C'], np.zeros((1, 3)),
            backend_name='fake', runner='in_process',
            execution_mode='energy', profile_semantics='energy')]
        outcomes = executor.execute(items)
        self.assertTrue(outcomes[0]['success'])

    def test_thread_hook_receives_the_contract(self):
        from pyrai3md.workflow.dispatch.terms import (
            TaskExecutor, make_work_item)
        executor = TaskExecutor(mode='thread', max_workers=2)
        seen = []

        class RecordingBackend:
            def check_thread_safe(self, runner, execution_mode,
                                  profile_semantics):
                seen.append((runner, execution_mode, profile_semantics))

            def calculate(self, atoms, coords, point_charges=None, target_states=None, cell=None):
                return {'energy': -1.0}

        items = [make_work_item(
            't1', RecordingBackend(), ['C'], np.zeros((1, 3)),
            backend_name='recording', runner='in_process',
            execution_mode='local', profile_semantics='energy')]
        self.assertTrue(executor.execute(items)[0]['success'])
        # The skill is handed the contract it is asked about, in the
        # order the executor stages it, before the pool starts.
        self.assertEqual(seen, [('in_process', 'local', 'energy')])

    def test_thread_hook_can_scope_per_contract(self):
        from pyrai3md.workflow.dispatch.terms import (
            TaskExecutor, make_work_item)
        executor = TaskExecutor(mode='thread', max_workers=2)

        class ScopedBackend:
            """Allows one contract only, the way a skill narrows itself."""

            def check_thread_safe(self, runner, execution_mode,
                                  profile_semantics):
                if (runner, execution_mode, profile_semantics) != (
                        'in_process', 'energy', 'energy'):
                    raise ValueError(
                                "scoped backend allows energy only, got '%s'"
                                % profile_semantics)

            def calculate(self, atoms, coords, point_charges=None, target_states=None, cell=None):
                return {'energy': -1.0}

        # The same backend under a different profile is refused by its
        # own hook: contract granularity now lives in the skill.
        items = [make_work_item(
            't1', ScopedBackend(), ['C'], np.zeros((1, 3)),
            backend_name='scoped', runner='in_process',
            execution_mode='gradient',
            profile_semantics='energy_gradient')]
        with self.assertRaises(ValueError):
            executor.execute(items)

        allowed = [make_work_item(
            't1', ScopedBackend(), ['C'], np.zeros((1, 3)),
            backend_name='scoped', runner='in_process',
            execution_mode='energy', profile_semantics='energy')]
        self.assertTrue(executor.execute(allowed)[0]['success'])

    def test_outcome_carries_evaluation_id(self):
        from pyrai3md.workflow.dispatch.terms import (
            TaskExecutor, make_work_item)
        executor = TaskExecutor(mode='serial')

        class FakeBackend:
            def calculate(self, atoms, coords, point_charges=None, target_states=None, cell=None):
                return {'energy': -1.0}

        items = [make_work_item(
            't1', FakeBackend(), ['C'], np.zeros((1, 3)),
            task_slot=0, center_index=0, evaluation_id='eval-7')]
        outcome = executor.execute(items)[0]
        self.assertEqual(outcome['task_id'], 't1')
        self.assertEqual(outcome['evaluation_id'], 'eval-7')
        self.assertTrue(outcome['success'])
        self.assertIsNone(outcome['error'])

    def test_serial_short_circuits_after_first_failure(self):
        from pyrai3md.workflow.dispatch.terms import (
            TaskExecutor, make_work_item)
        executor = TaskExecutor(mode='serial')
        calls = []

        class OkBackend:
            def calculate(self, atoms, coords, point_charges=None, target_states=None, cell=None):
                calls.append('ok')
                return {'energy': -1.0}

        class FailBackend:
            def calculate(self, atoms, coords, point_charges=None, target_states=None, cell=None):
                calls.append('fail')
                raise RuntimeError("boom")

        items = [
            make_work_item('t1', FailBackend(), ['C'], np.zeros((1, 3))),
            make_work_item('t2', OkBackend(), ['H'], np.zeros((1, 3))),
        ]
        outcomes = executor.execute(items)
        # The term queued after the failure is never run.
        self.assertEqual(len(outcomes), 1)
        self.assertFalse(outcomes[0]['success'])
        self.assertEqual(calls, ['fail'])

    def test_thread_reports_every_outcome_index_aligned(self):
        from pyrai3md.workflow.dispatch.terms import (
            TaskExecutor, make_work_item)
        executor = TaskExecutor(mode='thread', max_workers=3)

        class FakeBackend:
            def __init__(self, value):
                self.value = value

            def calculate(self, atoms, coords, point_charges=None, target_states=None, cell=None):
                return {'energy': self.value}

        items = [
            make_work_item('t%d' % i, FakeBackend(float(i)), ['C'],
                           np.zeros((1, 3)), evaluation_id='e%d' % i,
                           backend_name='aligned', runner='in_process',
                           execution_mode='energy',
                           profile_semantics='energy')
            for i in range(5)
        ]
        outcomes = executor.execute(items)
        # Index-aligned regardless of the completion order.
        self.assertEqual([o['task_id'] for o in outcomes],
                         ['t0', 't1', 't2', 't3', 't4'])
        self.assertEqual([o['evaluation_id'] for o in outcomes],
                         ['e0', 'e1', 'e2', 'e3', 'e4'])
        self.assertTrue(all(o['success'] for o in outcomes))

    def test_thread_cancels_pending_after_failure(self):
        import time
        from pyrai3md.workflow.dispatch.terms import (
            TaskExecutor, make_work_item)
        # One worker: the first term fails; once a queued term starts
        # it sleeps, so the main thread cancels every remaining term.
        executor = TaskExecutor(mode='thread', max_workers=1)
        started = []

        class FailBackend:
            def calculate(self, atoms, coords, point_charges=None, target_states=None, cell=None):
                started.append('fail')
                raise RuntimeError("boom")

        class SlowBackend:
            def calculate(self, atoms, coords, point_charges=None, target_states=None, cell=None):
                started.append('slow')
                time.sleep(0.05)
                return {'energy': -1.0}

        def _item(task_id, backend):
            return make_work_item(
                task_id, backend, ['C'], np.zeros((1, 3)),
                backend_name='cancel', runner='in_process',
                execution_mode='energy', profile_semantics='energy')

        items = [_item('t0', FailBackend())]
        items += [_item('t%d' % i, SlowBackend()) for i in range(1, 6)]
        outcomes = executor.execute(items)
        self.assertEqual(len(outcomes), 6)
        self.assertFalse(outcomes[0]['success'])
        self.assertIn('boom', outcomes[0]['error'])
        # At most one queued term sneaks onto the worker before the
        # failure is observed; the rest are cancelled, never executed.
        self.assertLessEqual(started.count('slow'), 1)
        cancelled = [o for o in outcomes[1:]
                     if o['error'] and 'cancelled' in o['error']]
        self.assertGreaterEqual(len(cancelled), 4)

    def test_failure_handling_collects_diagnostics(self):
        from pyrai3md.workflow.dispatch.terms import (
            TaskExecutor, execute_with_failure_handling, make_work_item)
        executor = TaskExecutor(mode='serial')

        class FailBackend:
            def calculate(self, atoms, coords, point_charges=None, target_states=None, cell=None):
                raise RuntimeError("backend error")

        items = [make_work_item(
            't1', FailBackend(), ['C'], np.zeros((1, 3)),
            evaluation_id='eval-3')]
        with self.assertRaises(RuntimeError) as ctx:
            execute_with_failure_handling(items, executor)
        message = str(ctx.exception)
        self.assertIn('t1', message)
        self.assertIn('eval-3', message)
        self.assertIn('backend error', message)


# ---------------------------------------------------------------------------
# Serializer plan_version tests (P3)
# ---------------------------------------------------------------------------

class TestSerializerPlanVersion(unittest.TestCase):
    """Test plan_version compatibility in serializer."""

    def test_prepare_task_adds_defaults(self):
        from pyrai3md.data.runtime.serializer import _prepare_task
        task = {'task_id': 'qm', 'config': None}
        prepared = _prepare_task(task)
        self.assertIsNone(prepared['center_index'])
        self.assertIsNone(prepared['region_id'])
        self.assertEqual(prepared['plan_version'], 1)

    def test_restore_task_fills_defaults(self):
        from pyrai3md.data.runtime.serializer import _restore_task
        task = {'task_id': 'qm'}
        _restore_task(task)
        self.assertIsNone(task['center_index'])
        self.assertIsNone(task['region_id'])
        self.assertEqual(task['plan_version'], 0)

    def test_restore_task_preserves_existing(self):
        from pyrai3md.data.runtime.serializer import _restore_task
        task = {'task_id': 'qm', 'center_index': 2, 'plan_version': 1}
        _restore_task(task)
        self.assertEqual(task['center_index'], 2)
        self.assertEqual(task['plan_version'], 1)

    def test_restore_task_migrates_legacy_center_id(self):
        from pyrai3md.data.runtime.serializer import _restore_task
        # A pre-rename .rt task still carries center_id; restore moves
        # the value onto center_index and drops the legacy key
        # (docs 3.1), leaving an explicit center_index untouched.
        task = {'task_id': 'qm', 'center_id': 2, 'plan_version': 1}
        _restore_task(task)
        self.assertEqual(task['center_index'], 2)
        self.assertNotIn('center_id', task)
        self.assertEqual(task['plan_version'], 1)


# ---------------------------------------------------------------------------
# Config schema tests (P2)
# ---------------------------------------------------------------------------

class TestCompositeSubmodel(unittest.TestCase):
    """Test Composite configuration submodel."""

    def test_defaults(self):
        from pyrai3md.schema.config import Composite
        c = Composite()
        self.assertEqual(c.executor, 'serial')
        self.assertEqual(c.max_workers, 1)

    def test_thread_mode(self):
        from pyrai3md.schema.config import Composite
        c = Composite(executor='thread', max_workers=4)
        self.assertEqual(c.executor, 'thread')
        self.assertEqual(c.max_workers, 4)

    def test_in_pyrai_config(self):
        from pyrai3md.schema.config import PyRAIConfig
        config = PyRAIConfig()
        self.assertEqual(config.composite.executor, 'serial')
        self.assertEqual(config.composite.max_workers, 1)


class TestHighlevelMidlevelTypes(unittest.TestCase):
    """Test highlevel/midlevel Union type support."""

    def test_flat_list(self):
        from pyrai3md.schema.config import System
        s = System(highlevel=[0, 1, 2])
        self.assertEqual(s.highlevel, [0, 1, 2])

    def test_nested_list(self):
        from pyrai3md.schema.config import System
        s = System(highlevel=[[0, 1], [4, 5]])
        self.assertEqual(s.highlevel, [[0, 1], [4, 5]])

    def test_empty_default(self):
        from pyrai3md.schema.config import System
        s = System()
        self.assertEqual(s.highlevel, [])
        self.assertEqual(s.midlevel, [])


class _AnalyticRegionBackend:
    """Analytic per-region energy/gradient for the R4f gate.

    For a region's local coordinates r_j (j = 0..nlocal-1) and state s:
        E_s = w_s * 0.5 * sum_j |r_j|^2,   G_s,j = dE_s/dr_j = w_s * r_j
    with w_s = 1 + 0.25*s a per-state scale, so every state's surface is
    distinct (exercising per-state selection) yet exactly differentiable.
    A fixed active configuration therefore makes the assembled ONIOM
    U(R;s) and G(R;s) satisfy dU/dR == G to finite-difference order.
    nstate comes from the task config, so centers with different state
    counts return different (never broadcast) axes.
    """

    def __init__(self, config=None, execution_id=''):
        self.config = config or {}
        self.execution_id = execution_id
        self.call_count = 0
        self.last_coords = None

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None):
        self.call_count += 1
        local = np.asarray(coordinates, dtype=np.float64)
        self.last_coords = local
        nstate = int(self.config.get('nstate', 1))
        base = 0.5 * float(np.sum(local * local))
        weights = np.array([1.0 + 0.25 * s for s in range(nstate)],
                           dtype=np.float64)
        return {'energy': weights * base,
                'gradient': weights[:, None, None] * local[None, :, :]}


# A fixed 5-atom geometry with distinct non-zero coordinates so every
# finite-difference direction probes a non-trivial gradient component.
_R4F_COORDS = [[0.10, 0.20, 0.30],
               [0.30, -0.40, 0.50],
               [-0.20, 0.10, 0.50],
               [0.60, -0.30, 0.20],
               [0.10, 0.20, -0.40]]


class TestR4fMultiCenterGate(unittest.TestCase):
    """R4f gate: end-to-end C=2 calculator.run over the unified center
    data model (docs r4f).

    Locks in the integration the r4a-r4e units only exercised in
    isolation: a real Calculator.run() on a two-center ONIOM queue
    archives each center's electronic arrays in its OWN local framing
    with DISTINCT state axes (never broadcast into a fake sum(S_c)
    global axis); the assembled active-combination U(R;s)/G(R;s) obey
    the finite-difference identity dU/dR == G for a FIXED active
    configuration; and the committed center data survives a full
    serialize -> restore round trip together with history and the RNG
    seed.
    """

    def _analytic_task(self, task_id, center_index, provides, policy,
                       ref_state, region, region_indices, contribution,
                       nstate):
        return {
            'role': 'qm' if provides else 'mm',
            'task': 0,
            'task_id': task_id,
            'layer_id': 0,
            'static_id': task_id + '_static',
            'contribution': contribution,
            'region': region,
            'region_indices': list(region_indices),
            'region_charge': None,
            'skill_name': 'mock_skill',
            'api': _AnalyticRegionBackend,
            'capabilities': ['energy', 'gradient'],
            'config': {'nstate': nstate},
            'center_index': center_index,
            'provides_electronic': provides,
            'state_policy': policy,
            'reference_state': ref_state,
        }

    def _c2_state(self, coords=_R4F_COORDS):
        from pyrai3md.data.state import PyRAIState
        from pyrai3md.data.runtime.state import RuntimeState
        from pyrai3md.data.center import build_center_plan
        from pyrai3md.schema.config import PyRAIConfig

        config = PyRAIConfig()
        config.system.atoms = ['H'] * 5
        config.system.coordinates = [list(r) for r in coords]
        config.system.velocities = [[0.0, 0.0, 0.0]] * 5
        config.system.natom = 5
        config.system.charges = 0
        config.system.multiplicity = 1
        config.dynamics.history = 4

        state = PyRAIState(runtime=RuntimeState(total_steps=10))
        state.pmd = config
        # center 0 -> ci [2, 2] = 4 states, root 1; center 1 -> ci [3] =
        # 3 states, root 2. The active configuration is therefore (1, 2).
        plan = build_center_plan({
            'highlevel': [[0, 1], [2, 3, 4]],
            'ci': [[2, 2], [3]],
            'spin': [[0, 1], [0]],
            'root': [1, 2],
        })
        state.bind_partition_plan(plan)

        # Two-layer ONIOM per center plus the L(real) baseline (docs 8):
        # U = E_real + sum_c (E_H(c) - E_L(c)); the provider H(c) follows
        # the center's active state, the corrections stay at reference 0.
        state.add_task('theory', self._analytic_task(
            'mm_real', None, False, 'reference', 0, 'real', [], 1, 1))
        state.add_task('theory', self._analytic_task(
            'qm_c0', 0, True, 'active', None, 'model', [0, 1], 1, 4))
        state.add_task('theory', self._analytic_task(
            'mm_c0', 0, False, 'reference', 0, 'model', [0, 1], -1, 1))
        state.add_task('theory', self._analytic_task(
            'qm_c1', 1, True, 'active', None, 'model', [2, 3, 4], 1, 3))
        state.add_task('theory', self._analytic_task(
            'mm_c1', 1, False, 'reference', 0, 'model', [2, 3, 4], -1, 1))
        return state, plan

    def _run(self, state):
        from pyrai3md.workflow.calculator import Calculator
        Calculator(state=state).initialize().run()

    def test_c2_run_archives_distinct_state_axes(self):
        state, _ = self._c2_state()
        self._run(state)
        self.assertEqual(state.get_active_configuration(), (1, 2))
        e0 = state.center_electronic(0)
        e1 = state.center_electronic(1)
        # Distinct state counts, never broadcast into a common axis.
        self.assertEqual(np.asarray(e0['energy']).shape, (4,))
        self.assertEqual(np.asarray(e1['energy']).shape, (3,))
        # Each center's gradient stays in its OWN local framing
        # (S_c, A_c, 3): center 0 has 2 atoms, center 1 has 3.
        self.assertEqual(np.asarray(e0['gradient']).shape, (4, 2, 3))
        self.assertEqual(np.asarray(e1['gradient']).shape, (3, 3, 3))
        # The L(real) baseline (center None) is never collected as a
        # center: exactly C=2 carriers exist.
        self.assertEqual(len(state.electronic), 2)

    def test_c2_provider_energy_matches_active_state(self):
        state, _ = self._c2_state()
        self._run(state)
        coords = np.asarray(_R4F_COORDS, dtype=np.float64)
        s01 = float(np.sum(coords[0:2] ** 2))
        s234 = float(np.sum(coords[2:5] ** 2))
        # The provider H(c) returns w_s * 0.5 * sum|r|^2 for each state.
        e0 = np.asarray(state.center_electronic(0)['energy'])
        e1 = np.asarray(state.center_electronic(1)['energy'])
        for s in range(4):
            self.assertAlmostEqual(e0[s], (1.0 + 0.25 * s) * 0.5 * s01)
        for s in range(3):
            self.assertAlmostEqual(e1[s], (1.0 + 0.25 * s) * 0.5 * s234)

    def test_c2_active_energy_and_gradient_are_oniom(self):
        state, _ = self._c2_state()
        self._run(state)
        coords = np.asarray(_R4F_COORDS, dtype=np.float64)
        s_all = float(np.sum(coords ** 2))
        s01 = float(np.sum(coords[0:2] ** 2))
        s234 = float(np.sum(coords[2:5] ** 2))
        # U = 0.5*[w0*S_all + (w1-w0)*S01 + (w2-w0)*S234], active (1,2).
        expected_u = 0.5 * (1.0 * s_all + 0.25 * s01 + 0.5 * s234)
        self.assertAlmostEqual(state.get_active_energy(), expected_u)
        # G_i = (baseline + provider - correction) * R_i per region:
        # center 0 atoms -> 1.25*R_i, center 1 atoms -> 1.5*R_i.
        grad = state.get_active_gradient()
        self.assertEqual(grad.shape, (5, 3))
        np.testing.assert_array_almost_equal(grad[0:2], 1.25 * coords[0:2])
        np.testing.assert_array_almost_equal(grad[2:5], 1.5 * coords[2:5])

    def test_finite_difference_energy_matches_global_gradient(self):
        from pyrai3md.workflow.calculator import Calculator
        state, _ = self._c2_state()
        calc = Calculator(state=state).initialize()
        calc.run()
        coords0 = np.asarray(_R4F_COORDS, dtype=np.float64)
        # G at the fixed active configuration (1, 2), captured once.
        grad = state.get_active_gradient().copy()
        delta = 1e-5
        for i in range(5):
            for a in range(3):
                rp = coords0.copy(); rp[i, a] += delta
                rm = coords0.copy(); rm[i, a] -= delta
                state.update_system({'coordinates': rp.tolist()})
                calc.run()
                up = state.get_active_energy()
                state.update_system({'coordinates': rm.tolist()})
                calc.run()
                um = state.get_active_energy()
                self.assertAlmostEqual((up - um) / (2 * delta),
                                       grad[i, a], places=6)

    def test_c2_serialize_restore_round_trip(self):
        from pyrai3md.data.runtime.serializer import save_state, load_state
        state, plan = self._c2_state()
        state.runtime.random_seed = 987654321
        self._run(state)
        state.update_history()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'c2.rt')
            save_state(state, path)
            restored, _ = load_state(path)
        self.assertEqual(restored.n_center, 2)
        self.assertEqual(restored.partition_plan.plan_fingerprint,
                         plan.plan_fingerprint)
        self.assertEqual(restored.get_active_configuration(), (1, 2))
        self.assertAlmostEqual(restored.get_active_energy(),
                               state.get_active_energy())
        np.testing.assert_array_almost_equal(
            np.asarray(restored.center_electronic(0)['energy']),
            np.asarray(state.center_electronic(0)['energy']))
        np.testing.assert_array_almost_equal(
            np.asarray(restored.center_electronic(1)['energy']),
            np.asarray(state.center_electronic(1)['energy']))
        np.testing.assert_array_almost_equal(
            restored.get_active_gradient(), state.get_active_gradient())
        # The RNG seed travels with runtime; the history frame survives
        # stamped with the plan fingerprint.
        self.assertEqual(restored.runtime.random_seed, 987654321)
        self.assertEqual(restored.history_size, 1)
        self.assertEqual(
            restored.get_history_step(-1)['plan_fingerprint'],
            plan.plan_fingerprint)


class TestCalculatorPhaseProfile(unittest.TestCase):
    """Test the calculator's own breakdown of one evaluation.

    The step loop times 'calculator_run' as one row, which is where the
    question actually lives: the plan, the work items, the backend call
    and the write-back are four different costs with four different
    remedies, and only one of them goes down to the engine. These tests
    hold that nested table to its contract - the rows exist, they are in
    the same milliseconds-per-step units as the loop table they are
    appended to, and a calculator that never ran leaves nothing behind.
    """

    def _run(self, steps=3):
        from pyrai3md.workflow.calculator import Calculator
        state = _make_mock_state()
        state.add_task('theory', _make_task('qm', 0, 1, 'all', []))
        calc = Calculator(state=state).initialize()
        for _step in range(steps):
            calc.run()
        return calc

    def test_the_evaluation_phases_are_recorded(self):
        calc = self._run()
        rows = '\n'.join(calc.profile_lines(3))
        for name in ('snapshot', 'plan', 'work_items', 'execute_terms',
                     'assemble', 'convert', 'set_result'):
            self.assertIn('phase=%s ' % name, rows,
                          'missing calculator phase %s' % name)
        self.assertIn('phase=TOTAL', rows)

    def test_the_rows_are_divided_by_the_caller_step_count(self):
        # The divisor comes from the loop, not from the calculator's own
        # count, so a nested row and the step row containing it are the
        # same quantity per step and can be read side by side.
        calc = self._run(steps=3)
        rows = calc.profile_lines(6)
        self.assertIn('steps=6', rows[0])

    def test_an_unevaluated_calculator_has_no_rows(self):
        from pyrai3md.workflow.calculator import Calculator
        state = _make_mock_state()
        state.add_task('theory', _make_task('qm', 0, 1, 'all', []))
        calc = Calculator(state=state).initialize()
        self.assertEqual(calc.profile_lines(100), [])


class TestExecutionModeReadSource(unittest.TestCase):
    """The calculator reads the execution_mode the backend acts on.

    The dispatch log and the thread-safety audit key used to look for
    the value at ``task['config']['environment']``, but a task config is
    the dict ``{'skill_config': ..., 'results': ...}`` built by
    data.core._build_task_queue -- it carries no 'environment' key, so
    both reported 'local' whatever the skill had been handed. The field
    the backend actually selects its runner from is
    ``skill_config['execution_mode']`` (skills/<skill>/service/runner),
    which is also where the framework writes its own dispatch decision
    (#300).
    """

    def _calc(self):
        from pyrai3md.workflow.calculator import Calculator
        return Calculator(state=_make_mock_state())

    def _task_with(self, config):
        task = _make_task('qm', 0, 1, 'all', [])
        task['config'] = config
        return task

    def test_reads_the_skill_config_value(self):
        calc = self._calc()
        task = self._task_with(
            {'skill_config': {'execution_mode': 'slurm'}, 'results': {}})
        self.assertEqual(calc._task_execution_mode(task), 'slurm')

    def test_missing_skill_config_defaults_to_local(self):
        calc = self._calc()
        self.assertEqual(
            calc._task_execution_mode(self._task_with({'results': {}})),
            'local')

    def test_the_old_environment_source_is_no_longer_consulted(self):
        # Causal control: the pre-fix reader would have returned 'slurm'
        # for this shape, even though no backend ever reads it.
        calc = self._calc()
        task = self._task_with(
            {'environment': {'execution_mode': 'slurm'}, 'results': {}})
        self.assertEqual(calc._task_execution_mode(task), 'local')

    def test_audit_key_carries_the_skill_value(self):
        calc = self._calc()
        task = self._task_with(
            {'skill_config': {'execution_mode': 'slurm'}, 'results': {}})
        task['skill_name'] = 'mock_skill'
        task['capabilities'] = ['energy', 'gradient']
        self.assertEqual(
            calc._term_audit_key(task),
            ('mock_skill', 'in_process', 'slurm', 'energy+gradient'))
