"""
======================================================================
This module verifies the adaptive-sampling workflow: the screening and
labelling functions, the planning wiring of its stage flags, the
resident MD uncertainty guard, and the iteration orchestration (directory
contents, data flow, stop rules) with mocked children.

Aug 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from pyrai3md.data.runtime.state import RuntimeState
from pyrai3md.schema.adapter import ADAPTIVE_TYPES
from pyrai3md.schema.config import DEFAULT, Adaptive
from pyrai3md.schema.validate import validate_adaptive
from pyrai3md.training_data import TrainingData
from pyrai3md.utilities.print_results import format_adaptive_iteration
from pyrai3md.workflow.adaptive import AdaptiveSampling, _child_log_name
from pyrai3md.workflow.records.adaptive import (
    build_frame_dataset, distance_ok, energy_order_bad, energy_outlier_flags,
    frame_contract, history_frames, labels_from_outcome, outcome_stop_code,
    screen_iteration, screen_trajectory,
)
from pyrai3md.workflow.records.population import population_bad

ATOMS = ['H', 'H']
COORDINATES = [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]


class _Log:
    """Collects the text the workflow writes to the run log."""

    def __init__(self):
        self.text = ''

    def write(self, text):
        self.text += text


def _frame(step, err_energy=0.0, err_grad=0.0, coordinates=None,
           energy=(0.0, 1.0), density=None, state_spin=None):
    system = {
        'coordinates': COORDINATES if coordinates is None else coordinates,
        'nstate': len(energy),
        'state_density': density,
    }
    if state_spin is not None:
        system['state_spin'] = list(state_spin)
    return {
        'step': step,
        'system': system,
        'results': {
            'err_energy': err_energy,
            'err_grad': err_grad,
            'err_nac': 0.0,
            'err_soc': 0.0,
            'energy': list(energy),
        },
    }


def _maxima(value=0.4):
    return {'err_energy': value, 'err_grad': value,
            'err_nac': value, 'err_soc': value}


def _minima(value=0.05):
    return {'err_energy': value, 'err_grad': value,
            'err_nac': value, 'err_soc': value}


def _dataset(nframe=1, soc=False, nac=False):
    xyz = np.tile(np.asarray(COORDINATES, dtype=float), (nframe, 1, 1))
    symbols = np.asarray([ATOMS for _ in range(nframe)], dtype=object)
    energy = np.zeros((nframe, 2))
    grad = np.zeros((nframe, 2, 2, 3))
    nac_array = np.zeros((nframe, 1, 2, 3)) if nac else None
    soc_array = np.zeros((nframe, 2)) if soc else None
    return TrainingData(
        xyz=xyz, symbols=symbols, energy=energy, grad=grad,
        nac=nac_array, soc=soc_array,
        unit={'energy': 'hartree', 'gradient': 'hartree/bohr'})


class TestScreening(unittest.TestCase):

    def test_equality_does_not_trigger(self):
        frames = [_frame(1, err_energy=0.4)]
        result = screen_trajectory(
            0, frames, _maxima(0.4), _minima(), 4,
            check_population=False, check_energy_order=False)
        self.assertFalse(result['triggered'])
        self.assertEqual(result['selected'], [])

    def test_pool_requires_a_trigger(self):
        frames = [_frame(1, err_energy=0.1)]
        result = screen_trajectory(
            0, frames, _maxima(0.4), _minima(0.05), 4,
            check_population=False, check_energy_order=False)
        self.assertFalse(result['triggered'])
        self.assertEqual(result['selected'], [])

    def test_newest_first_and_traceable(self):
        frames = [_frame(8, 0.1), _frame(9, 0.5), _frame(10, 0.2)]
        result = screen_trajectory(
            7, frames, _maxima(0.4), _minima(0.05), 4,
            check_population=False, check_energy_order=False)
        self.assertTrue(result['triggered'])
        self.assertEqual([row['step'] for row in result['selected']],
                         [10, 9, 8])
        self.assertTrue(all(row['traj_id'] == 7
                            for row in result['frames']))

    def test_maxsample_caps_the_pool(self):
        frames = [_frame(8, 0.1), _frame(9, 0.5), _frame(10, 0.2)]
        result = screen_trajectory(
            0, frames, _maxima(0.4), _minima(0.05), 1,
            check_population=False, check_energy_order=False)
        self.assertEqual(len(result['selected']), 1)
        self.assertEqual(result['selected'][0]['step'], 10)

    def test_distance_filter_rejects_close_pair(self):
        close = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
        frames = [_frame(1, 0.5, coordinates=close)]
        result = screen_trajectory(
            0, frames, _maxima(0.4), _minima(0.05), 1,
            check_population=False, check_energy_order=False,
            atoms=ATOMS)
        self.assertEqual(result['selected'], [])
        self.assertEqual(result['frames'][0]['reason'], 'distance filter')
        self.assertTrue(result['frames'][0]['flags']['geometry_bad'])
        self.assertEqual(result['n_discarded'], 1)
        self.assertEqual(result['n_bad'], 1)
        self.assertFalse(distance_ok(ATOMS, close))
        self.assertTrue(distance_ok(ATOMS, COORDINATES))

    def test_discarded_separates_bad_geometry_from_the_cap(self):
        close = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
        # The bad frame is the oldest, beyond the one selected and the one
        # stopped by the cap; it still counts as Bad.
        frames = [_frame(8, 0.5, coordinates=close),
                  _frame(9, 0.5),
                  _frame(10, 0.5)]
        result = screen_trajectory(
            0, frames, _maxima(0.4), _minima(0.05), 1,
            check_population=False, check_energy_order=False,
            atoms=ATOMS)
        self.assertEqual([row['step'] for row in result['selected']], [10])
        self.assertEqual(result['n_discarded'], 2)
        self.assertEqual(result['n_bad'], 1)
        bad = [row for row in result['frames']
               if row['flags']['geometry_bad']]
        self.assertEqual([row['step'] for row in bad], [8])

    def test_population_stop_is_collected_by_code(self):
        self.assertTrue(population_bad([1.02, -0.02], 1e-2))
        self.assertTrue(population_bad([0.5, float('nan')], 1e-2))
        self.assertFalse(population_bad([0.5, 0.5], 1e-2))

        # The screen does not judge a frame's density: the same frame is
        # collected only because the md run reported the ill-population code.
        bad = np.asarray([[1.02, 0.0], [0.0, -0.02]])
        frames = [_frame(1, 0.0, density=bad.tolist())]
        result = screen_trajectory(
            0, frames, _maxima(), _minima(), 1,
            check_population=True, stop_code=0,
            check_energy_order=False)
        self.assertFalse(result['triggered'])
        self.assertEqual(result['population_status'], 'not_stopped')

        result = screen_trajectory(
            0, frames, _maxima(), _minima(), 1,
            check_population=True, stop_code=1,
            check_energy_order=False)
        self.assertTrue(result['triggered'])
        self.assertEqual(len(result['selected']), 1)
        self.assertEqual(result['population_status'], 'stopped')
        self.assertTrue(result['frames'][-1]['flags']['population_stop'])

        result = screen_trajectory(
            0, frames, _maxima(), _minima(), 1,
            check_population=False, stop_code=1,
            check_energy_order=False)
        self.assertFalse(result['triggered'])
        self.assertEqual(result['selected'], [])
        self.assertEqual(result['population_status'], 'off')

    def test_energy_order_rule_and_switch(self):
        self.assertTrue(energy_order_bad([1.0, 0.5], [0, 0], 1e-6))
        self.assertFalse(energy_order_bad([1.0, 0.5], [0, 2], 1e-6))
        self.assertFalse(energy_order_bad([0.5, 1.0], [0, 0], 1e-6))

        frames = [_frame(1, 0.0, energy=(1.0, 0.5), state_spin=(0, 0))]
        result = screen_trajectory(
            0, frames, _maxima(), _minima(), 1,
            check_population=False, check_energy_order=True)
        self.assertTrue(result['triggered'])

        result = screen_trajectory(
            0, frames, _maxima(), _minima(), 1,
            check_population=False, check_energy_order=False)
        self.assertFalse(result['triggered'])

    def test_history_window_and_iteration_screen(self):
        class State:
            def __init__(self, frames):
                self._frames = frames

            def get_history(self, n=None):
                return self._frames[-n:] if n else list(self._frames)

        frames = [_frame(1, 0.0), _frame(2, 0.0), _frame(3, 0.5)]
        result = screen_iteration([State(frames)], {
            'record_step': 2, 'maxima': _maxima(), 'minima': _minima(),
            'maxsample': 1, 'check_population': 0,
            'check_energy_order': 0, 'atoms': ATOMS, 'mdstep': 3,
        })
        rows = result['per_trajectory'][0]['frames']
        self.assertEqual([row['step'] for row in rows], [2, 3])
        self.assertTrue(result['per_trajectory'][0]['completed'])
        self.assertEqual(result['selected'][0]['traj_id'], 0)

    def test_history_frames_accepts_remote_result(self):
        payload = {'history': np.asarray([
            {'step': 1}, {'step': 2}], dtype=object)}
        self.assertEqual(len(history_frames(payload, 1)), 1)
        self.assertEqual(history_frames(
            {'adaptive_status': 'error'}, 1), [])

    def test_md_stop_code_selects_the_stopping_frame(self):
        frames = [_frame(1, 0.0), _frame(2, 0.0)]
        outcomes = [{'history': frames, 'runtime': {'stop_code': 1}}]
        settings = {
            'record_step': 4, 'maxima': _maxima(), 'minima': _minima(),
            'maxsample': 1, 'check_population': 1,
            'check_energy_order': 0, 'atoms': ATOMS, 'mdstep': 10,
        }
        result = screen_iteration(outcomes, settings)
        self.assertTrue(result['triggered'])
        self.assertEqual([row['step'] for row in result['selected']], [2])
        self.assertEqual(
            result['per_trajectory'][0]['population_status'], 'stopped')

        settings['check_population'] = 0
        result = screen_iteration(outcomes, settings)
        self.assertFalse(result['triggered'])
        self.assertEqual(result['selected'], [])
        self.assertEqual(
            result['per_trajectory'][0]['population_status'], 'off')

    def test_a_normal_stop_code_collects_nothing(self):
        frames = [_frame(1, 0.0), _frame(2, 0.0)]
        settings = {
            'record_step': 4, 'maxima': _maxima(), 'minima': _minima(),
            'maxsample': 1, 'check_population': 1,
            'check_energy_order': 0, 'atoms': ATOMS, 'mdstep': 2,
        }
        for code in (0, 4):
            result = screen_iteration(
                [{'history': frames, 'runtime': {'stop_code': code}}],
                settings)
            self.assertFalse(result['triggered'])
            self.assertEqual(result['selected'], [])

    def test_error_marker_keeps_the_calculation_failure_code(self):
        self.assertEqual(outcome_stop_code(
            {'adaptive_status': 'error', 'stop_code': 4}), 4)
        self.assertEqual(outcome_stop_code({'adaptive_status': 'error'}), 0)


class TestLabels(unittest.TestCase):

    def test_labels_and_units(self):
        dataset = _dataset(soc=True)
        contract = frame_contract(dataset)
        labels, reason = labels_from_outcome({
            'energy': [0.1, 1.1],
            'grad': np.zeros((2, 2, 3)),
            'soc': [[10.0], [20.0]],
        }, contract)
        self.assertEqual(reason, '')
        self.assertEqual(labels['energy'].shape, (2,))
        self.assertEqual(labels['grad'].shape, (2, 2, 3))
        self.assertEqual(labels['soc'].shape, (2,))
        self.assertIsNone(labels['nac'])

    def test_missing_nac_is_refused(self):
        dataset = _dataset(nac=True)
        contract = frame_contract(dataset)
        labels, reason = labels_from_outcome({
            'energy': [0.1, 1.1],
            'gradient': np.zeros((2, 2, 3)),
        }, contract)
        self.assertIsNone(labels)
        self.assertIn('NAC', reason)

    def test_channel_count_mismatch_is_refused(self):
        contract = frame_contract(_dataset())
        labels, reason = labels_from_outcome({
            'energy': [0.1],
            'gradient': np.zeros((2, 2, 3)),
        }, contract)
        self.assertIsNone(labels)
        self.assertIn('states', reason)

    def test_build_dataset_appends(self):
        dataset = _dataset()
        labels = {
            'coordinates': COORDINATES,
            'energy': np.asarray([0.1, 1.1]),
            'grad': np.zeros((2, 2, 3)),
        }
        new = build_frame_dataset([labels], ATOMS,
                                  unit=dataset.unit)
        dataset.append(new)
        self.assertEqual(dataset.nframe, 2)

    def test_appended_frames_stay_out_of_validation(self):
        # The iteration's sampled frames go to the tail. NN validation is
        # taken from the front, so no frame added by this iteration can
        # enter it.
        from pyrai3md.skills.nn_pes_skill.service.nn_pes.datasets.general \
            import split_validation_training_index

        old = 100
        dataset = _dataset(old)
        labels = {
            'coordinates': COORDINATES,
            'energy': np.asarray([0.1, 1.1]),
            'grad': np.zeros((2, 2, 3)),
        }
        new = build_frame_dataset(
            [labels, dict(labels)], ATOMS, unit=dataset.unit)
        dataset.append(new)
        self.assertEqual(dataset.nframe, old + 2)
        self.assertTrue(np.array_equal(dataset.xyz[-2:], new.xyz))
        total = dataset.nframe
        val_size = int(total * 0.1)
        for offset in range(2):
            _train, validation = split_validation_training_index(
                np.arange(total), val_size, True, offset)
            self.assertTrue(all(index < old for index in validation))


class TestInitWriter(unittest.TestCase):

    def test_init_and_init_xyz_share_the_readable_layout(self):
        from pyrai3md.workflow.initcond import read_init_xyz, write_init_xyz

        folder = tempfile.mkdtemp(prefix='pyrai3md_adaptive_init_')
        try:
            batch = [{'coordinates': COORDINATES,
                      'velocities': [[0.1, 0.0, 0.0],
                                     [-0.1, 0.0, 0.0]]}]
            init = os.path.join(folder, 'ethylene.init')
            copied = os.path.join(folder, 'ethylene.init.xyz')
            write_init_xyz(init, batch, ATOMS, comment='Init ethylene')
            write_init_xyz(copied, batch, ATOMS, comment='Init ethylene')
            with open(init) as handle:
                first = handle.read()
            with open(copied) as handle:
                second = handle.read()
            self.assertEqual(first, second)
            frames = read_init_xyz(copied)
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0][0].shape, (2, 3))
            self.assertEqual(frames[0][1].shape, (2, 3))
        finally:
            shutil.rmtree(folder, ignore_errors=True)


class TestPlanning(unittest.TestCase):

    def test_adaptive_kwargs_are_wired(self):
        
        state = SimpleNamespace(pmd=SimpleNamespace(
            adaptive=Adaptive(maxiter=4, maxsample=2, load=1,
                              record_step=7)))
        kwargs = AdaptiveSampling.state_kwargs(state)
        self.assertEqual(kwargs, {'maxiter': 4, 'maxsample': 2,
                                  'load': 1, 'record_step': 7})

    def test_stage_flags_come_from_the_adaptive_section(self):
        from pyrai3md.workflow import WORKFLOW_REGISTRY
        from pyrai3md.workflow.dispatch.planner import (
            ExecutionGraphBuilder,
        )

        adaptive = Adaptive(md_multitask=True, md_multipool=3,
                            reference_multitask=True,
                            reference_multipool=2, remote_train=1)
        pmd = SimpleNamespace(adaptive=adaptive, dynamics=SimpleNamespace(),
                              control=SimpleNamespace(title='t'),
                              system=SimpleNamespace())
        theory = [{'skill_name': 'nn_pes_skill', 'config': {},
                   'capabilities': ['energy']}]
        reference = [{'skill_name': 'pyscf_skill', 'config': {},
                      'capabilities': ['energy']}]
        state = SimpleNamespace(
            pmd=pmd, theory=theory, reference=reference,
            execution_graph=None, resolve_execution_graph=True,
            jobtype='adaptive')
        state.iter_tasks = lambda: [('theory', theory[0]),
                                    ('reference', reference[0])]
        graph = ExecutionGraphBuilder(WORKFLOW_REGISTRY).build(
            state, AdaptiveSampling)
        stages = {stage['name']: stage for stage in graph[0]['stages']}
        self.assertTrue(stages['md']['multitask'])
        self.assertEqual(stages['md']['multipool'], 3)
        self.assertTrue(stages['base.reference']['multitask'])
        self.assertEqual(stages['base.reference']['multipool'], 2)
        self.assertFalse(stages['base.reference']['remotetask'])
        self.assertFalse(stages['train']['multitask'])
        self.assertTrue(stages['train']['remotetask'])

    def test_workflow_does_not_initialize_backends(self):
        self.assertFalse(AdaptiveSampling.needs_backends)


class TestEarlyStop(unittest.TestCase):

    def test_above_max_stops(self):
        from pyrai3md.workflow.md import AIMD
        state = SimpleNamespace(pmd=SimpleNamespace(
            results={'err_grad': 0.2}))
        workflow = AIMD(deviation_thresholds={'err_grad': 0.1})
        self.assertEqual(workflow._stop_deviations(state), ['err_grad'])

    def test_equality_and_absent_channels_do_not_stop(self):
        from pyrai3md.workflow.md import AIMD
        state = SimpleNamespace(pmd=SimpleNamespace(
            results={'err_grad': 0.1}))
        workflow = AIMD(deviation_thresholds={'err_grad': 0.1})
        self.assertEqual(workflow._stop_deviations(state), [])
        state.pmd.results = {}
        self.assertEqual(workflow._stop_deviations(state), [])

    def test_no_threshold_leaves_every_channel_unjudged(self):
        from pyrai3md.workflow.md import AIMD
        state = SimpleNamespace(pmd=SimpleNamespace(
            results={'err_grad': 9.0}))
        self.assertEqual(AIMD()._stop_deviations(state), [])


class TestRegionAwareAdaptive(unittest.TestCase):
    """The trainable surface may be one ONIOM region of a larger box."""

    ATOMS8 = ['C', 'C', 'H', 'H', 'H', 'H', 'O', 'H']
    COORDS8 = [[0.0, 0.0, float(index)] for index in range(8)]
    HIGH = [0, 1, 2, 3, 4, 5]

    def _state(self):
        adaptive = Adaptive(maxenergy=0.05, minenergy=0.045,
                            maxgrad=0.10, mingrad=0.09)
        pmd = SimpleNamespace(
            adaptive=adaptive,
            dynamics=SimpleNamespace(
                step=10, history=4, initcond='0', ninitcond=1,
                format='molden', temperature=300.0),
            control=SimpleNamespace(title='t', gl_seed=1),
            system=SimpleNamespace(
                nstate=2, pbc=[True, True, True],
                cell=[10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
                coordinates=np.asarray(self.COORDS8, dtype=float),
                velocities=np.zeros((8, 3))),
            results={})
        low = {'skill_name': 'lammps_skill', 'region': 'real',
               'region_indices': [], 'contribution': 1,
               'provides_electronic': False, 'capabilities': ['energy'],
               'config': {'skill_config': {}}}
        high = {'skill_name': 'nn_pes_skill', 'region': 'model',
                'region_indices': list(self.HIGH), 'contribution': 1,
                'provides_electronic': True,
                'capabilities': ['energy', 'gradient'],
                'config': {'skill_config': {}}}
        subtract = dict(low)
        subtract.update({'region': 'model',
                         'region_indices': list(self.HIGH),
                         'contribution': -1})
        reference = {'skill_name': 'openqp_skill', 'region': 'model',
                     'region_indices': list(self.HIGH),
                     'capabilities': ['energy', 'gradient'],
                     'config': {'skill_config': {}}}
        state = SimpleNamespace(
            pmd=pmd, theory=[low, high, subtract], reference=[reference],
            atoms=list(self.ATOMS8), nstate=2,
            execution_graph=None, resolve_execution_graph=True,
            runtime=RuntimeState())
        state.iter_tasks = lambda: [('theory', low), ('theory', high),
                                    ('theory', subtract),
                                    ('reference', reference)]
        return state, high

    @staticmethod
    def _dataset():
        return TrainingData(
            xyz=np.zeros((1, 6, 3)),
            symbols=np.asarray([['C', 'C', 'H', 'H', 'H', 'H']]),
            energy=np.zeros((1, 2)), grad=np.zeros((1, 2, 6, 3)),
            unit={'energy': 'hartree', 'gradient': 'hartree/bohr'})

    def test_find_ml_task_selects_the_high_region(self):
        state, high = self._state()
        workflow = AdaptiveSampling(
            calculator=_FakeCalculator(state), maxiter=2,
            maxsample=1, load=1, record_step=10)
        slot, task, _surface = workflow._find_ml_task(state)
        self.assertIs(task, high)
        self.assertEqual(slot, 1)
        indices, atoms = workflow._ml_region(state, task)
        self.assertEqual(indices, self.HIGH)
        self.assertEqual(atoms, ['C', 'C', 'H', 'H', 'H', 'H'])

    def test_preflight_compares_the_dataset_with_the_region(self):
        state, high = self._state()
        workflow = AdaptiveSampling(
            calculator=_FakeCalculator(state), maxiter=2,
            maxsample=1, load=1, record_step=10)
        surface = {'query_module': 'x', 'targets': {},
                                                'query': _nn_query()}
        workflow._preflight(state, self._dataset(), high, surface, {})
        with self.assertRaises(ValueError):
            workflow._preflight(
                state, _dataset(2), high, surface, {})

    def test_preflight_refuses_exploration_with_metadynamics(self):
        state, high = self._state()
        state.pmd.adaptive.explore_gap = 1
        state.pmd.adaptive.gap_method = 'cv'
        state.pmd.system.constraints = SimpleNamespace(
            metadynamics=[SimpleNamespace(id='m')])
        workflow = AdaptiveSampling(
            calculator=_FakeCalculator(state), maxiter=2,
            maxsample=1, load=1, record_step=10)
        surface = {'query_module': 'x', 'targets': {},
                                                'query': _nn_query()}
        with self.assertRaisesRegex(ValueError, 'metadynamics'):
            workflow._preflight(state, self._dataset(), high, surface, {})

    def test_reference_labels_are_region_frames(self):
        state, _high = self._state()
        workflow = AdaptiveSampling(
            calculator=_FakeCalculator(state), maxiter=2,
            maxsample=1, load=1, record_step=10)
        workflow.materialize_unit = lambda *args, **kwargs: {
            'state': SimpleNamespace(iter_tasks=lambda: []),
            'task_index': 0}
        workflow._slice_child_system = lambda *args, **kwargs: None
        workflow._stamp_reference = lambda *args, **kwargs: None
        workflow._dispatch_batch = lambda *args, **kwargs: [
            {'energy': np.asarray([0.0, 1.0]),
             'gradient': np.zeros((2, 6, 3))}]
        selected = [{'traj_id': 0, 'frame_index': 0, 'step': 1,
                     'coordinates': self.COORDS8}]
        contract = {'nstate': 2, 'natom': 6, 'nframe': 0,
                    'nnac': 0, 'nsoc': 0, 'npair': 0,
                    'has_cell': False, 'has_pbc': False,
                    'has_nac': False, 'has_soc': False,
                    'has_group_id': False, 'has_permutation_id': False}
        labels, failures = workflow._reference_iteration(
            state, 2, '/tmp', selected, contract,
            {'reference_remotetask': False, 'reference_multitask': False,
             'reference_multipool': 1}, self.HIGH)
        self.assertEqual(failures, [])
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0]['coordinates'],
                         [self.COORDS8[index] for index in self.HIGH])
        self.assertEqual(labels[0]['grad'].shape, (2, 6, 3))


class TestAdaptiveExamplePlans(unittest.TestCase):
    """Example coverage: one method, ONIOM2 and QMMM, mechanical only.

    The multiscale cases use the shipped twenty-water box (the same
    geometry and cell as ``examples/md_lammps_nve.yaml``) and one water
    as the trainable region. Both composites are mechanical: the ONIOM
    case passes no embedding, and the QMMM case has no field probe. The
    reference list is its own plan; the label source is the reference
    layer whose region equals the neural-network surface.
    """

    @classmethod
    def setUpClass(cls):
        import yaml
        path = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir,
            'examples', 'md_lammps_nve.yaml')
        with open(path, encoding='utf-8') as handle:
            box = yaml.safe_load(handle)
        cls.atoms = list(box['atoms'])
        cls.coordinates = [list(row) for row in box['coordinates']]
        cls.molecule = dict(box.get('molecule') or {})
        cls.cell = list(cls.molecule['cell'])
        cls.pbc = list(cls.molecule['pbc'])

    def _state(self, theory, reference, multiscale='', molecule=None,
               atoms=None, coordinates=None):
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate

        raw = {
            'control': {
                'title': 'adaptive_example',
                'jobtype': 'adaptive',
                'theory': list(theory),
                'reference': list(reference),
            },
            'atoms': list(atoms if atoms is not None else self.atoms),
            'coordinates': [list(row) for row in
                            (coordinates if coordinates is not None
                             else self.coordinates)],
            'molecule': dict(molecule or {}),
            'adaptive': {'maxiter': 1},
            'nn': {
                'method': 'mlp_eg',
                'modeldir': './unused-model',
                'train_data': './unused-data.json',
                'mult_nn': 1,
                'target_states': [0],
            },
            'openqp': {'method': 'mrsf', 'functional': 'pbe0',
                       'basis': 'sto-3g'},
            'xtb': {'method': 'gfn2'},
            'molcas': {'method': 'caspt2', 'basis': 'ano-l-vdzp',
                       'nact_elec': 6, 'nact_orb': 5, 'inactive_orb': 2,
                       'caspt2_type': 'xms'},
            'lammps': {'method': 'ff', 'lammps_project': 'h2o',
                       'python_enabled': True},
        }
        if multiscale:
            raw['control']['multiscale_type'] = multiscale
        config = validate(normalize(build_config(raw)))
        return build_state(config, raw_config=raw)

    @staticmethod
    def _plan(state):
        plan = next(
            (task.get('label_plan') for task in state.reference
             if task.get('label_plan')), None)
        assert plan is not None
        return plan

    @classmethod
    def _molecule_box(cls, highlevel):
        return {
            'charges': 0,
            'multiplicity': 1,
            'ci': [1],
            'spin': [0],
            'highlevel': list(highlevel),
            'highref': list(highlevel),
            'pbc': list(cls.pbc),
            'cell': list(cls.cell),
        }

    def test_single_method_adaptive_plan(self):
        molecule = {'charges': 0, 'multiplicity': 1, 'ci': [2], 'spin': [0]}
        state = self._state(
            ['nn'], ['openqp'], molecule=molecule,
            atoms=['O', 'H', 'H'], coordinates=self.coordinates[:3])
        self.assertEqual(len(state.theory), 1)
        self.assertEqual(len(state.reference), 1)
        self.assertEqual(state.reference[0]['region'], 'all')
        self.assertEqual(state.reference[0]['region_indices'], [])
        plan = self._plan(state)
        self.assertEqual(plan['source'], 'layer')
        self.assertEqual(plan['source_layer'], 'high')
        self.assertEqual(plan['region_indices'], [0, 1, 2])
        self.assertFalse(plan['full_system'])
        self.assertEqual(plan['aux'], [])

    def test_oniom2_mechanical_adaptive_plan(self):
        molecule = self._molecule_box([1, 2, 3])
        state = self._state(
            ['nn', 'xtb'], ['openqp', 'xtb'], multiscale='oniom',
            molecule=molecule)
        self.assertEqual(
            [(task['region'], task['contribution'])
             for task in state.theory],
            [('real', 1), ('model', 1), ('model', -1)])
        self.assertEqual(
            [(task['region'], task['contribution'])
             for task in state.reference],
            [('real', 1), ('model', 1), ('model', -1)])
        plan = self._plan(state)
        self.assertEqual(plan['source'], 'layer')
        self.assertEqual(plan['source_layer'], 'high')
        self.assertEqual(plan['region_indices'], [0, 1, 2])
        self.assertTrue(plan['full_system'])
        self.assertEqual(plan['aux'], [])
        self.assertFalse(any(
            task.get('qmmm_role') for task in state.reference))

    def test_qmmm_mechanical_adaptive_plan(self):
        molecule = self._molecule_box([1, 2, 3])
        molecule['q_qm'] = [-0.834, 0.417, 0.417]
        state = self._state(
            ['nn', 'lammps'], ['molcas', 'lammps'], multiscale='qmmm',
            molecule=molecule)
        self.assertEqual(
            [(task['qmmm_role'], task['region'])
             for task in state.reference],
            [('qm', 'model'), ('energy', 'all')])
        self.assertFalse(any(
            task.get('qmmm_role') == 'field'
            for task in state.theory + state.reference))
        plan = self._plan(state)
        self.assertEqual(plan['source'], 'layer')
        self.assertEqual(plan['source_layer'], 'high')
        self.assertEqual(plan['region_indices'], [0, 1, 2])
        self.assertEqual(plan['aux'], ['mm_energy', 'q_qm'])
        self.assertTrue(any('without a field probe' in note
                            for note in plan['notes']))

    def test_qmmm_label_channels_project_onto_the_water(self):
        contract = {
            'natom': 3, 'nstate': 1, 'nnac': 0, 'nsoc': 0,
            'npair': 0, 'has_nac': False, 'has_soc': False,
            'has_cell': False, 'has_pbc': False,
            'has_group_id': False, 'has_permutation_id': False,
        }
        gradient = np.zeros((1, len(self.atoms), 3))
        gradient[:, :3, :] = 2.0
        channels = {
            'source': 'layer',
            'region_indices': [0, 1, 2],
            'energy': np.asarray([-1.0]),
            'gradient': gradient,
            'mm_energy': -10.0,
            'q_qm': [-0.834, 0.417, 0.417],
        }
        labels, reason = labels_from_outcome(
            {'label_channels': channels}, contract, [0, 1, 2])
        self.assertEqual(reason, '')
        self.assertEqual(labels['grad'].shape, (1, 3, 3))
        self.assertTrue(np.all(labels['grad'] == 2.0))
        self.assertEqual(labels['energy'].tolist(), [-1.0])


class _FakeCalculator:
    def __init__(self, state):
        self._state = state

    def get_state(self):
        return self._state

    def update(self, payload):
        if 'results' in payload:
            self._state.pmd.results = payload['results']


def _make_state(adaptive):
    pmd = SimpleNamespace(
        adaptive=adaptive,
        dynamics=SimpleNamespace(
            step=10, history=4, initcond='0', ninitcond=1,
            format='molden', temperature=300.0),
        control=SimpleNamespace(title='t', gl_seed=1),
        system=SimpleNamespace(
            pbc=[False, False, False],
            nstate=2,
            coordinates=np.asarray(COORDINATES, dtype=float),
            velocities=np.zeros((2, 3))),
        results={})
    theory = [{'skill_name': 'x', 'config': {'skill_config': {}},
               'capabilities': []}]
    reference = [{'skill_name': 'y', 'config': {'skill_config': {}},
                  'capabilities': []}]
    state = SimpleNamespace(
        pmd=pmd, theory=theory, reference=reference,
        execution_graph=None, resolve_execution_graph=True,
        atoms=list(ATOMS), runtime=RuntimeState())
    state.iter_tasks = lambda: [('theory', theory[0]),
                                ('reference', reference[0])]
    return state, theory


def _nn_query():
    """The real skill query module: the model folder name is its call."""
    from pyrai3md.skills.nn_pes_skill import query
    return query


class TestRoundOrchestration(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix='pyrai3md_adaptive_')
        self.previous = os.getcwd()
        os.chdir(self.folder)

    def tearDown(self):
        os.chdir(self.previous)
        shutil.rmtree(self.folder, ignore_errors=True)

    def _workflow(self, maxiter=3, keep_md=1, selection=(True, True, False)):
        adaptive = Adaptive(maxiter=maxiter, maxsample=1, record_step=10,
                            keep_md=keep_md, check_population=0,
                            check_energy_order=0,
                            maxenergy=0.4, minenergy=0.05)
        state, theory = _make_state(adaptive)
        state.logger = _Log()
        workflow = AdaptiveSampling(
            calculator=_FakeCalculator(state), maxiter=maxiter,
            maxsample=1, load=1, record_step=10)
        workflow._find_ml_task = lambda _state: (
            0, theory[0], {'query_module': 'x', 'targets': {},
                                                'query': _nn_query()})
        workflow._load_initial_dataset = lambda _task: _dataset(1)
        workflow._preflight = lambda *args, **kwargs: None
        workflow._configured_modeldir = lambda _task: os.path.join(
            self.folder, 'external-model')
        workflow._require_servable = lambda *args, **kwargs: None

        references = {'count': 0}
        events = []

        def md_iteration(_state, iteration, iter_dir, _slot, _serving,
                         _settings):
            events.append(('md', iteration))
            trajectory = os.path.join(iter_dir, 'md', 'traj_1')
            os.makedirs(trajectory, exist_ok=True)
            with open(os.path.join(trajectory, 't.md.pop'), 'w') as handle:
                handle.write('time Pop1 Pop2 Act1 Act2\n')
                handle.write('0.0 1.0 0.0 1 0\n')
                handle.write('20.67 0.8 0.2 1 0\n')
            err = 0.5 if selection[iteration - 1] else 0.1
            return [{'history': [_frame(9, 0.0), _frame(10, err)]}]

        reference_dirs = []

        def reference_iteration(_state, iteration, iter_dir, selected,
                            _contract, _settings, _ml_indices):
            events.append(('reference', iteration))
            reference_dirs.append(iter_dir)
            references['count'] += 1
            labels = []
            for row in selected:
                labels.append({
                    'coordinates': row['coordinates'],
                    'energy': np.asarray([0.0, 1.0]),
                    'grad': np.zeros((2, 2, 3)),
                })
            return labels, []

        def train_iteration(_state, _slot, _surface, data_path, modeldir,
                            _iter_dir, _settings, _ml_indices):
            try:
                iteration = int(str(modeldir).rstrip('/').split('iter-')[-1])
            except (TypeError, ValueError):
                iteration = 0
            events.append(('train', iteration))
            os.makedirs(modeldir, exist_ok=True)
            note = {'reason': '', 'result': {
                'train_errors': {'energy_grad': {
                    'energy': [0.1], 'gradient': [0.01]}},
                'train_metric_units': {'energy': 'eV',
                                       'gradient': 'eV/angstrom'},
                'train_wall_seconds': 1.0,
                'train_arm': 'local',
                'train_data': data_path,
                'train_modeldir': modeldir,
                'train_fitlogs': [os.path.join(
                    modeldir, 'energy_grad', 'fitlog_0.txt')],
            }}
            return True, modeldir, note

        workflow._md_iteration = md_iteration
        workflow._reference_iteration = reference_iteration
        workflow._train_iteration = train_iteration
        self.references = references
        self.events = events
        self.reference_dirs = reference_dirs
        return workflow, state

    def test_reference_and_training_share_the_iteration(self):
        workflow, state = self._workflow(maxiter=3)
        workflow.run()
        self.assertEqual(
            self.events,
            [('md', 1), ('reference', 2), ('train', 2), ('md', 2),
             ('reference', 3), ('train', 3), ('md', 3)])
        root = os.path.join(self.folder, 't.adaptive')
        self.assertEqual([os.path.realpath(path)
                          for path in self.reference_dirs], [
            os.path.realpath(os.path.join(root, 'iter-2')),
            os.path.realpath(os.path.join(root, 'iter-3'))])

    def test_iter_directories_and_data_flow(self):
        workflow, state = self._workflow(maxiter=3)
        workflow.run()
        root = os.path.join(self.folder, 't.adaptive')

        iter_one = os.path.join(root, 'iter-1')
        self.assertFalse(os.path.exists(os.path.join(iter_one, 'data')))
        self.assertFalse(os.path.exists(os.path.join(iter_one, 'models')))
        self.assertTrue(os.path.exists(
            os.path.join(iter_one, 'selected_iter-1.xyz')))
        self.assertTrue(os.path.exists(
            os.path.join(iter_one, 'population_iter-1.json')))
        self.assertTrue(os.path.exists(
            os.path.join(iter_one, 'status_iter-1.json')))
        with open(os.path.join(
                iter_one, 'population_iter-1.json')) as handle:
            population = json.load(handle)
        first = population['trajectories'][0]
        self.assertEqual(first['traj_id'], 1)
        self.assertEqual(first['population'],
                         [[1.0, 0.0], [0.8, 0.2]])

        iter_two = os.path.join(root, 'iter-2')
        data_two = os.path.join(iter_two, 'data_iter-2_2.json')
        self.assertTrue(os.path.exists(data_two))
        self.assertTrue(os.path.isdir(
            os.path.join(iter_two, 'NN-t-iter-2')))
        self.assertFalse(os.path.exists(os.path.join(iter_two, 'data')))
        self.assertFalse(os.path.exists(os.path.join(iter_two, 'models')))
        self.assertFalse(os.path.exists(os.path.join(iter_two, 'train')))
        with open(data_two) as handle:
            self.assertEqual(len(json.load(handle)['xyz']), 2)

        iter_three = os.path.join(root, 'iter-3')
        data_three = os.path.join(iter_three, 'data_iter-3_3.json')
        with open(data_three) as handle:
            self.assertEqual(len(json.load(handle)['xyz']), 3)

        record = state.pmd.results['adaptive']
        self.assertEqual(record['reason'], 'no sample selected')
        self.assertTrue(record['converged'])
        self.assertEqual(record['iterations'][0]['selected'], 1)
        self.assertEqual(record['iterations'][2]['selected'], 0)
        self.assertEqual(record['iterations'][1]['received'], 1)
        self.assertEqual(record['iterations'][1]['verified'], 1)
        # Iteration 3 is terminal, so its selected set never reaches reference.
        self.assertEqual(self.references['count'], 2)

    def test_each_iteration_is_reported_in_the_run_log(self):
        workflow, state = self._workflow(maxiter=2, selection=(True, False))
        workflow.run()
        text = state.logger.text
        self.assertIn('&adaptive sampling', text)
        self.assertIn('Traj(ID:', text)
        self.assertIn('Number of trajectories:', text)
        self.assertIn('Discard:', text)
        self.assertIn('(Bad', text)
        self.assertIn('Discarded:', text)
        self.assertIn('Metrics', text)
        self.assertIn('Sampling status:', text)
        self.assertIn('Iteration:                  1', text)
        self.assertIn('Iteration:                  2', text)
        self.assertIn('Received data:', text)
        self.assertIn('Verified data:', text)
        self.assertIn('&nn training', text)
        self.assertIn('Training data:', text)
        self.assertNotIn('Reference:', text)
        self.assertNotIn('New training frames:', text)
        self.assertNotIn('NOT PASSED (', text)
        # The data/model belong to the &nn section, not to this report.
        self.assertNotIn('Dataset:', text)
        self.assertNotIn('Model:', text)
        self.assertFalse(os.path.exists(os.path.join(
            self.folder, 't.adaptive', 't.adaptive_status.txt')))

    def test_reference_batch_root_is_the_iteration_reference_folder(self):
        workflow, state = self._workflow(maxiter=2)
        raw = AdaptiveSampling(
            calculator=_FakeCalculator(state), maxiter=2,
            maxsample=1, load=1, record_step=10)
        captured = {}

        def dispatch_batch(units, **kwargs):
            captured['root'] = kwargs['root']
            return []

        raw._dispatch_batch = dispatch_batch
        raw.materialize_unit = lambda *args, **kwargs: {
            'state': SimpleNamespace(iter_tasks=lambda: []),
            'task_index': 0}
        raw._slice_child_system = lambda *args, **kwargs: None
        iter_dir = os.path.join(self.folder, 't.adaptive', 'iter-2')
        raw._reference_iteration(
            state, 2, iter_dir,
            [{'coordinates': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}],
            {'nframe': 1},
            {'reference_remotetask': False,
             'reference_multitask': False, 'reference_multipool': 1},
            [0, 1])
        self.assertEqual(captured['root'],
                         os.path.join(iter_dir, 'reference'))

    def test_run_root_is_the_title_subdirectory(self):
        workflow, state = self._workflow(maxiter=1)
        self.assertEqual(
            os.path.realpath(workflow.run_root(state)),
            os.path.realpath(os.path.join(self.folder, 't.adaptive')))

    def test_training_log_takes_the_model_directory_name(self):
        training = SimpleNamespace(
            jobtype='train', iter_tasks=lambda: [(
                'theory', {'config': {'skill_config': {
                    'modeldir': '/x/NN-t-iter-2'}}})])
        self.assertEqual(_child_log_name(training), 'NN-t-iter-2')
        cloud = SimpleNamespace(
            jobtype='md', iter_tasks=lambda: [(
                'theory', {'config': {'skill_config': {
                    'modeldir': '/x/NN-t-iter-2'}}})])
        self.assertIsNone(_child_log_name(cloud))

    def test_maxiter_terminates_without_reference(self):
        workflow, state = self._workflow(maxiter=1, selection=(True,))
        workflow.run()
        record = state.pmd.results['adaptive']
        self.assertEqual(record['reason'], 'maxiter exhausted')
        self.assertFalse(record['iterations'][0]['reference_ran'])
        self.assertEqual(self.references['count'], 0)
        # Iteration 1 has no handed-over data and no fit.
        self.assertNotIn('Received data:', state.logger.text)
        self.assertNotIn('&nn training', state.logger.text)

    def test_keep_md_zero_removes_trajectory_directory(self):
        workflow, state = self._workflow(maxiter=2, keep_md=0,
                                         selection=(True, False))
        workflow.run()
        root = os.path.join(self.folder, 't.adaptive')
        for iteration in (1, 2):
            self.assertFalse(os.path.exists(
                os.path.join(root, 'iter-%d' % iteration, 'md')))


class TestTrainingDataQualityGate(unittest.TestCase):
    """The `maxmae` gate: label statistics, frame subsetting, reporting."""

    REFERENCE = np.array([[-100.0], [-99.9], [-100.1], [-100.05],
                          [-99.95], [-100.02]])

    def test_a_distant_new_energy_is_flagged(self):
        flags, detail = energy_outlier_flags(
            self.REFERENCE, np.array([[-90.0]]))
        self.assertTrue(detail['testable'])
        self.assertEqual(['mad'], detail['scale_kind'])
        self.assertEqual([True], flags.tolist())
        self.assertEqual([0], detail['flagged'])

    def test_a_frame_inside_the_band_is_not_flagged(self):
        flags, detail = energy_outlier_flags(
            self.REFERENCE, np.array([[-100.03], [-99.98]]))
        self.assertTrue(detail['testable'])
        self.assertEqual([False, False], flags.tolist())

    def test_the_worst_state_decides(self):
        reference = np.array([[-10.0, -9.0], [-10.1, -9.1],
                              [-9.9, -8.9], [-10.05, -9.05]])
        flags, detail = energy_outlier_flags(
            reference, np.array([[-10.0, -2.0]]))
        self.assertEqual([True], flags.tolist())
        self.assertEqual([0], detail['flagged'])

    def test_a_degenerate_state_falls_back_to_the_standard_deviation(self):
        reference = np.array([[-5.0], [-5.0], [-5.0], [-5.0], [-5.2]])
        flags, detail = energy_outlier_flags(
            reference, np.array([[-5.0], [-9.0]]))
        self.assertEqual(['std'], detail['scale_kind'])
        self.assertEqual([False, True], flags.tolist())

    def test_a_constant_reference_flags_any_different_energy(self):
        flags, detail = energy_outlier_flags(
            np.zeros((5, 1)), np.array([[0.0], [1e-9]]))
        self.assertEqual(['zero'], detail['scale_kind'])
        self.assertEqual([False, True], flags.tolist())
        # The unbounded z is recorded as null, not as Infinity: the
        # status file is written with allow_nan=False.
        self.assertIsNone(detail['max_abs_z'][1])

    def test_too_few_reference_frames_are_untestable(self):
        flags, detail = energy_outlier_flags(
            np.array([[-1.0], [-1.1]]), np.array([[-2.0]]))
        self.assertFalse(detail['testable'])
        self.assertEqual([False], flags.tolist())

    def test_a_non_finite_new_energy_is_untestable(self):
        flags, detail = energy_outlier_flags(
            self.REFERENCE, np.array([[np.nan]]))
        self.assertFalse(detail['testable'])
        self.assertIn('non-finite', detail['reason'])

    def test_subset_reports_the_named_frames_only(self):
        size = 5
        data = TrainingData(
            xyz=np.arange(size * 6, dtype=float).reshape(size, 2, 3),
            symbols=np.array([['C', 'H']] * size),
            energy=np.arange(size, dtype=float).reshape(size, 1) + 100.0,
            grad=np.zeros((size, 1, 2, 3)),
            group_id=np.arange(size), source='unit-test')
        part = data.subset([0, 2, 4])
        self.assertEqual(3, part.nframe)
        self.assertEqual((3, 1, 2, 3), part.grad.shape)
        self.assertEqual([0, 2, 4], part.group_id.tolist())
        self.assertEqual('unit-test', part.source)
        self.assertEqual([100.0, 102.0, 104.0], part.energy[:, 0].tolist())
        # The receiver is a candidate's origin, never a scratch pad.
        self.assertEqual(5, data.nframe)
        self.assertEqual(list(range(size)), data.group_id.tolist())

    def test_subset_refuses_repeats_and_out_of_range(self):
        data = TrainingData(xyz=np.zeros((3, 1, 3)), energy=np.zeros((3, 1)))
        for indices in ([1, 1], [0, 3], [-1]):
            with self.assertRaises(ValueError):
                data.subset(indices)

    def test_maxmae_registration_and_validation(self):
        self.assertEqual(0.0, DEFAULT['maxmae'])
        self.assertEqual(0.0, Adaptive().maxmae)
        self.assertEqual(0.1, Adaptive(maxmae=0.1).maxmae)
        self.assertEqual('f', ADAPTIVE_TYPES['maxmae'])
        validate_adaptive({})
        validate_adaptive({'maxmae': 0.1})
        for bad in (-1.0, True, 'x'):
            with self.assertRaises((ValueError, TypeError)):
                validate_adaptive({'maxmae': bad})

    def test_the_gate_block_reports_removal_and_verdict(self):
        gate = {
            'status': 'aborted', 'target': 0.1, 'mae': 0.3, 'unit': 'eV',
            'new_frames': 2, 'kept_frames': 1, 'pruned': [102],
            'attempts': [
                {'round': 0, 'modeldir': 'NN-t-iter-2', 'mae': 0.3,
                 'decision': 'start'},
                {'round': 1, 'modeldir': 'NN-t-iter-2-r1', 'mae': 0.32,
                 'pruned': 102, 'pruned_z': 5.31, 'decision': 'rejected'}],
            'reason': 'training quality gate (iteration 2): no improvement',
        }
        block = format_adaptive_iteration(
            {'iteration': 2, 'new_frames': 2, 'kept_frames': 1,
             'pruned_frames': 1, 'received': 2, 'verified': 2,
             'status': 'training quality gate aborted'},
            screen=None, gate=gate)
        self.assertIn('&nn validation gate', block)
        self.assertIn('NOT PASSED', block)
        self.assertIn('remove frame 102', block)
        self.assertIn('rejected; frame stays', block)
        self.assertIn('training quality gate aborted', block)

    def test_the_gate_block_is_absent_when_the_gate_is_off(self):
        block = format_adaptive_iteration(
            {'iteration': 2, 'new_frames': 0, 'received': 0, 'verified': 0},
            screen=None, gate={'status': 'off', 'target': 0.0, 'attempts': []})
        self.assertNotIn('&nn validation gate', block)


class TestValidationGateLoop(unittest.TestCase):
    """The trial-prune loop: acceptance, index shift, restore-on-no-gain."""

    REFERENCE = np.array([0.0, 0.1, -0.1, 0.05])

    def _gate(self, new_energies, maes, maxmae=0.1, fail_round=None):
        energies = np.concatenate([self.REFERENCE, np.asarray(new_energies)])
        size = int(energies.size)
        dataset = TrainingData(
            xyz=np.zeros((size, 1, 3)),
            energy=energies.reshape(-1, 1))
        base = int(self.REFERENCE.size)
        remaining = list(range(base, size))
        sent = []

        def train(state, ml_slot, surface, data_path, modeldir, iter_dir,
                  settings, ml_indices, name_format='train_{idx}'):
            sent.append((data_path, modeldir, name_format))
            if fail_round is not None and len(sent) == fail_round:
                return False, modeldir, {'reason': 'fit exploded'}
            mae = maes[len(sent)]
            return True, modeldir, {
                'reason': '',
                'result': {'train_errors': {
                    'energy_grad': {'energy': [mae]}}}}

        workflow = AdaptiveSampling.__new__(AdaptiveSampling)
        workflow._train_iteration = train
        workflow._save_dataset = lambda dataset, path: None
        surface = {'query': SimpleNamespace(
            model_dir_name=lambda title, key: 'NN-%s-%s' % (title, key))}
        gate = workflow._validation_gate(
            None, 2, '/tmp/iter-2', 0, surface, 't', [], dataset,
            '/tmp/data.json', '/tmp/NN-t-iter-2',
            {'train_errors': {'energy_grad': {'energy': [maes[0]]}}},
            remaining, base, {'maxmae': maxmae})
        return gate, sent

    def test_a_fit_inside_the_target_passes_without_a_retrain(self):
        gate, sent = self._gate([0.02], [0.05])
        self.assertEqual('passed', gate['status'])
        self.assertEqual([], sent)
        self.assertEqual([], gate['pruned'])

    def test_the_gate_is_off_at_zero(self):
        gate, sent = self._gate([10.0], [0.5], maxmae=0.0)
        self.assertEqual('off', gate['status'])
        self.assertEqual([], sent)

    def test_every_new_frame_flagged_stops_before_any_retrain(self):
        gate, sent = self._gate([10.0, 20.0], [0.5])
        self.assertEqual('aborted', gate['status'])
        self.assertIn('every frame added this iteration is an outlier',
                      gate['reason'])
        self.assertEqual([], sent)

    def test_removals_that_keep_improving_are_accepted(self):
        # Two of three new frames are outliers; each removal lowers the
        # MAE, and the surviving frame is the clean one (0.02).
        gate, sent = self._gate([10.0, 20.0, 0.02], [0.5, 0.4, 0.2])
        self.assertEqual('aborted', gate['status'])
        self.assertIn('no outlier among the 1 new frame(s)', gate['reason'])
        self.assertEqual([5, 4], gate['pruned'])
        self.assertEqual(5, gate['dataset'].nframe)
        self.assertAlmostEqual(0.02, float(gate['dataset'].energy[-1, 0]))
        self.assertEqual(['train_r1_{idx}', 'train_r2_{idx}'],
                         [entry[2] for entry in sent])
        self.assertTrue(sent[0][1].endswith('NN-t-iter-2-r1'))
        self.assertTrue(sent[1][1].endswith('NN-t-iter-2-r2'))

    def test_a_removal_without_gain_is_restored_and_stops(self):
        gate, _sent = self._gate([10.0, 0.02], [0.5, 0.5])
        self.assertEqual('aborted', gate['status'])
        self.assertIn('did not improve', gate['reason'])
        self.assertEqual([], gate['pruned'])
        self.assertEqual(6, gate['dataset'].nframe)
        self.assertEqual(2, gate['kept_frames'])

    def test_a_failed_retrain_is_restored_and_stops(self):
        gate, _sent = self._gate([10.0, 0.02], [0.5], fail_round=1)
        self.assertEqual('aborted', gate['status'])
        self.assertIn('retraining without frame 4 failed: fit exploded',
                      gate['reason'])
        self.assertEqual([], gate['pruned'])
        self.assertEqual(6, gate['dataset'].nframe)

    def test_a_fit_without_an_energy_mae_cannot_be_judged(self):
        gate, sent = self._gate([0.02], [None])
        self.assertEqual('aborted', gate['status'])
        self.assertIn('no energy validation MAE', gate['reason'])
        self.assertEqual([], sent)


class TestQualityGateCarrier(unittest.TestCase):
    """The shipped `maxmae` example and the fixture it narrows."""

    EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, os.pardir, 'examples')

    def _pipeline(self, path):
        from pyrai3md.pmd import parse_input
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate

        return validate(normalize(build_config(parse_input(path))))

    def test_the_example_carries_the_measured_target(self):
        path = os.path.join(self.EXAMPLES,
                            'adaptive_quality_gate_local.yaml')
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(0.0082, self._pipeline(path).adaptive.maxmae)
        with open(path) as handle:
            self.assertIn('./egs_ethylene_gate_100.json', handle.read())

    def test_the_classic_carrier_keeps_the_target(self):
        import tempfile

        from pyrai3md.qualification.certification.converter import (
            convert, yaml_to_config)

        example = os.path.join(self.EXAMPLES,
                               'adaptive_quality_gate_local.yaml')
        with tempfile.TemporaryDirectory() as folder:
            json_path = os.path.join(folder, 'case.json')
            input_path = os.path.join(folder, 'case.input')
            yaml_to_config(example, json_path)
            convert(json_path, input_path)
            with open(input_path) as handle:
                self.assertIn('maxmae', handle.read())
            self.assertEqual(0.0082, self._pipeline(input_path).adaptive.maxmae)

    def test_the_fixture_halves_the_reference_band(self):
        from pyrai3md.training_data import load

        original = load(os.path.join(self.EXAMPLES, 'egs_ethylene_100.json'))
        narrowed = load(os.path.join(self.EXAMPLES,
                                     'egs_ethylene_gate_100.json'))
        self.assertEqual(original.nframe, narrowed.nframe)
        self.assertTrue(np.allclose(original.xyz, narrowed.xyz))
        self.assertTrue(np.allclose(original.grad, narrowed.grad))

        def mad(dataset):
            median = np.median(dataset.energy, axis=0)
            return np.median(np.abs(dataset.energy - median), axis=0)

        before, after = mad(original), mad(narrowed)
        self.assertTrue(np.allclose(after, 0.5 * before, rtol=1e-12),
                        (before, after))


if __name__ == '__main__':
    unittest.main()