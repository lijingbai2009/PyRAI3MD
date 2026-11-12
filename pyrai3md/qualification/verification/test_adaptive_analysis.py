"""
Unit tests for the adaptive analysis layer (workflow/records/analysis.py).

The layer is default-closed: the switch predicate, the three-state
semantics, the gap/trace/coverage/saturation summaries and the point-pair
PES quantities are exercised here without running a trajectory.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from pyrai3md.training_data import TrainingData
from pyrai3md.utilities.units import ANGSTROM_TO_BOHR
from pyrai3md.workflow.records import analysis_bias
from pyrai3md.workflow.records.analysis import (
    MISSING, NOT_APPLICABLE, NOT_REQUESTED, OK,
    analysis_active, analysis_plan, predictor_status, saturation,
    summarize_coverage, summarize_gap, summarize_pes, summarize_trace,
    trace_rows, trace_spec,
)
from pyrai3md.data.runtime.state import RuntimeState


class _Adaptive:
    def __init__(self, **values):
        for key, value in values.items():
            setattr(self, key, value)


class TestSwitches(unittest.TestCase):

    def test_analysis_frame_carries_the_version(self):
        from pyrai3md.workflow.records.analysis import (
            ANALYSIS_VERSION, AnalysisFrame)
        frame = AnalysisFrame(iteration=1, trajectories=[], summary={})
        self.assertEqual(frame['version'], ANALYSIS_VERSION)
        self.assertEqual(frame['iteration'], 1)

    def test_all_off_is_inactive_and_has_no_trace(self):
        plan = analysis_plan(_Adaptive())
        self.assertFalse(analysis_active(plan))
        self.assertIsNone(trace_spec(plan))

    def test_a_single_switch_activates_the_layer(self):
        plan = analysis_plan(_Adaptive(analyze_gap='s0s1'))
        self.assertTrue(analysis_active(plan))
        self.assertEqual(trace_spec(plan)['gap'], 's0s1')

    def test_trace_alone_asks_for_the_scalar_rows(self):
        spec = trace_spec(analysis_plan(_Adaptive(analyze_trace=1)))
        self.assertTrue(spec['force'])
        self.assertTrue(spec['population'])
        self.assertTrue(spec['distance'])
        self.assertIsNone(spec['gap'])

    def test_exploration_activates_the_layer_without_a_trace(self):
        plan = analysis_plan(_Adaptive(explore_udd=1))
        self.assertTrue(analysis_active(plan))
        self.assertIsNone(trace_spec(plan))

    def test_predict_mode_activates_the_comparison_channel(self):
        plan = analysis_plan(_Adaptive(predict_mode=1))
        self.assertTrue(analysis_active(plan))
        self.assertEqual(plan['predict_mode'], 1)


class TestGap(unittest.TestCase):

    def _rows(self):
        return [
            {'traj_id': 0, 'step': 10, 'energy': [0.00, 0.10, 0.30]},
            {'traj_id': 0, 'step': 20, 'energy': [0.00, 0.05, 0.20]},
            {'traj_id': 0, 'step': 30, 'energy': [0.00, 0.02, 0.10]},
        ]

    def test_s0s1_reports_minimum_and_first_crossing(self):
        report = summarize_gap(self._rows(), 's0s1', 0.03)
        self.assertEqual(report['status'], OK)
        self.assertAlmostEqual(report['min_gap'], 0.02)
        self.assertEqual(report['min_pair'], [0, 1])
        self.assertEqual(report['first_below_threshold']['step'], 30)
        self.assertAlmostEqual(report['fraction_below_threshold'], 1.0 / 3)

    def test_adjacent_scope_covers_every_neighbouring_pair(self):
        report = summarize_gap(self._rows(), 'adjacent', 0.03)
        self.assertEqual(report['status'], OK)
        self.assertEqual(report['n_values'], 6)

    def test_no_frame_energies_is_missing(self):
        report = summarize_gap(
            [{'traj_id': 0, 'step': 1}, {'traj_id': 0, 'step': 2}],
            's0s1', 0.03)
        self.assertEqual(report['status'], MISSING)

    def test_one_state_is_not_applicable(self):
        rows = [{'traj_id': 0, 'step': 1, 'energy': [0.0]}]
        report = summarize_gap(rows, 's0s1', 0.03)
        self.assertEqual(report['status'], NOT_APPLICABLE)
        self.assertIn('two states', report['reason'])


class TestTraceAndPredictor(unittest.TestCase):

    def test_trace_summary_reads_the_scalars(self):
        rows = [
            {'step': 1, 'gap': {'min': 0.2, 'pair': [0, 1]},
             'max_force': 0.4, 'energy_drift': 0.001, 'population': [0.9, 0.1]},
            {'step': 2, 'gap': {'min': 0.1, 'pair': [0, 1]},
             'max_force': 0.5, 'energy_drift': -0.002, 'population': [0.8, 0.2]},
        ]
        report = summarize_trace(rows)
        self.assertEqual(report['status'], OK)
        self.assertAlmostEqual(report['min_gap'], 0.1)
        self.assertAlmostEqual(report['max_force'], 0.5)
        self.assertAlmostEqual(report['max_energy_drift'], 0.002)

    def test_predictor_zero_is_not_requested(self):
        report = predictor_status([{'alternatives': None}], 0)
        self.assertEqual(report['status'], NOT_REQUESTED)

    def test_predictor_partial_channels_is_missing(self):
        rows = [{'alternatives': {'energy': [[1.0]], 'gradient': None,
                                  'model_ids': [2]}}]
        report = predictor_status(rows, 1)
        self.assertEqual(report['status'], MISSING)
        self.assertIn('gradient', report['reason'])

    def test_predictor_available(self):
        rows = [{'alternatives': {'energy': [[1.0]], 'gradient': [[[0.0]]],
                                  'model_ids': [2]}}]
        report = predictor_status(rows, 1)
        self.assertEqual(report['status'], OK)

    def test_trace_reads_a_remote_result_runtime(self):
        from pyrai3md.data.runtime.result import Result
        result = Result(data={'runtime': {'analysis_trace': {
            'spec': {}, 'rows': [{'step': 1, 'max_force': 0.2}]}}})
        rows = trace_rows(result)
        self.assertEqual(rows, [{'step': 1, 'max_force': 0.2}])


class TestCoverageAndSaturation(unittest.TestCase):

    def _dataset(self, natom=2, nframe=3):
        xyz = np.zeros((nframe, natom, 3))
        xyz[:, 0, 0] = np.linspace(0.0, 0.2, nframe)
        xyz[:, 1, 0] = 5.0
        return TrainingData(
            xyz=xyz, energy=np.zeros((nframe, 1)),
            grad=np.zeros((nframe, 1, natom, 3)),
            symbols=np.asarray([['H'] * natom for _ in range(nframe)],
                               dtype=object))

    def test_coverage_reports_a_fraction(self):
        rows = [{'traj_id': 0, 'step': 1,
                 'coordinates': [[0.1, 0.0, 0.0], [5.0, 0.0, 0.0]]}]
        report = summarize_coverage(rows, self._dataset())
        self.assertEqual(report['status'], OK)
        self.assertEqual(report['n_candidates'], 1)
        self.assertIn('covered_fraction', report)

    def test_coverage_large_system_is_not_applicable(self):
        rows = [{'traj_id': 0, 'step': 1,
                 'coordinates': [[0.0, 0.0, 0.0]] * 300}]
        report = summarize_coverage(rows, self._dataset(natom=2))
        self.assertEqual(report['status'], NOT_APPLICABLE)
        self.assertIn('descriptor', report['reason'])

    def test_saturation_needs_a_window(self):
        report = saturation([1.0], window=3, global_tolerance=0.01)
        self.assertEqual(report['status'], NOT_APPLICABLE)

    def test_saturation_detects_stagnation(self):
        report = saturation([1.0, 0.8, 0.8001, 0.8002], window=2,
                            global_tolerance=0.01)
        self.assertEqual(report['status'], OK)
        self.assertTrue(report['saturated'])

    def test_local_tolerance_keeps_a_loose_global_stop_open(self):
        history = [1.0, 0.8, 0.8201, 0.8202]
        loose = saturation(history, window=2, global_tolerance=0.1)
        self.assertTrue(loose['saturated'])
        local = saturation(history, window=2, global_tolerance=0.1,
                           local_tolerance=0.005)
        self.assertFalse(local['saturated'])
        self.assertTrue(local['local_progress'])


class TestPesReconstruction(unittest.TestCase):

    def _dataset(self, power):
        positions = np.linspace(-0.4, 0.4, 5)
        xyz = np.zeros((5, 2, 3))
        xyz[:, 0, 0] = positions / ANGSTROM_TO_BOHR
        xyz[:, 1, 0] = 5.0
        energy = np.zeros((5, 1))
        grad = np.zeros((5, 1, 2, 3))
        for index, value in enumerate(positions):
            energy[index, 0] = value ** power / power
            grad[index, 0, 0, 0] = value ** (power - 1)
        return TrainingData(
            xyz=xyz, energy=energy, grad=grad,
            symbols=np.asarray([['H', 'H']] * 5, dtype=object))

    def _candidate(self):
        return [{'traj_id': 0, 'step': 1,
                 'coordinates': [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]}]

    def test_quadratic_surface_reconstructs_with_zero_error(self):
        report = summarize_pes(self._candidate(), self._dataset(2))
        candidate = report['candidates'][0]
        self.assertIsNotNone(candidate['reconstruction'])
        self.assertEqual(candidate['reconstruction']['rank'], 1)
        self.assertLess(candidate['reconstruction']['rF'], 1e-6)
        self.assertEqual(len(report['resolution_pairs']), 1)

    def test_a_nonlinear_surface_leaves_a_positive_residual(self):
        report = summarize_pes(self._candidate(), self._dataset(4))
        candidate = report['candidates'][0]
        self.assertIsNotNone(candidate['reconstruction'])
        self.assertGreater(candidate['reconstruction']['rF'], 1e-6)


class TestPesQuadratic(unittest.TestCase):

    def test_quadratic_surface_has_zero_residual_and_analytic_kappa(self):
        # E = 0.5*k*x_bohr^2, g = k*x_bohr, so r^E = 0 and kappa = k.
        stiffness = 2.0
        x_bohr = np.asarray([-0.2, 0.0, 0.2])
        x_angstrom = x_bohr / ANGSTROM_TO_BOHR
        xyz = np.zeros((3, 2, 3))
        xyz[:, 0, 0] = x_angstrom
        xyz[:, 1, 0] = 5.0
        energy = np.zeros((3, 1))
        grad = np.zeros((3, 1, 2, 3))
        for index, value in enumerate(x_bohr):
            energy[index, 0] = 0.5 * stiffness * value * value
            grad[index, 0, 0, 0] = stiffness * value
        dataset = TrainingData(
            xyz=xyz, energy=energy, grad=grad,
            symbols=np.asarray([['H', 'H']] * 3, dtype=object))
        rows = [{'traj_id': 0, 'step': 1,
                 'coordinates': [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]}]
        report = summarize_pes(rows, dataset)
        self.assertEqual(report['status'], OK)
        candidate = report['candidates'][0]
        self.assertAlmostEqual(candidate['curvature'], stiffness, places=8)
        self.assertAlmostEqual(candidate['energy_residual'], 0.0, places=10)


class TestTraceCollection(unittest.TestCase):

    def test_record_analysis_trace_writes_the_requested_scalars(self):
        from pyrai3md.workflow.md import AIMD
        runtime = RuntimeState(analysis_trace={
            'spec': {'gap': 's0s1', 'force': True,
                     'population': True, 'energy_drift': True,
                     'distance': True},
            'rows': []})
        runtime.current_step = 5
        runtime.simulation_time = 1.0
        runtime.kinetic_energy = 0.01
        state = SimpleNamespace(
            runtime=runtime,
            pmd=SimpleNamespace(
                results={
                    'energy': [0.0, 0.1],
                    'gradient': [[[1.0, 0.0, 0.0]],
                                 [[0.0, 0.0, 0.0]]]},
                system=SimpleNamespace(
                    current_state=0,
                    coordinates=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    state_density=[[1.0, 0.0], [0.0, 0.0]])))
        AIMD()._record_analysis_trace(state)
        rows = runtime.analysis_trace['rows']
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['gap']['min'], 0.1)
        self.assertEqual(rows[0]['gap']['pair'], [0, 1])
        self.assertAlmostEqual(rows[0]['max_force'], 1.0)
        self.assertAlmostEqual(rows[0]['min_distance'], 1.0)
        self.assertEqual(rows[0]['population'], [1.0, 0.0])
        self.assertAlmostEqual(rows[0]['energy_drift'], 0.0)

    def test_no_trace_spec_records_nothing(self):
        from pyrai3md.workflow.md import AIMD
        runtime = RuntimeState()
        state = SimpleNamespace(
            runtime=runtime,
            pmd=SimpleNamespace(results={}, system=SimpleNamespace()))
        AIMD()._record_analysis_trace(state)
        self.assertIsNone(runtime.analysis_trace)


class TestAdaptiveBiasSpec(unittest.TestCase):

    def test_both_explorations_produce_one_spec(self):
        from pyrai3md.workflow.adaptive import AdaptiveSampling
        spec = AdaptiveSampling._bias_spec({
            'explore_udd': 1, 'explore_gap': 1,
            'udd_mode': 'potential', 'udd_amplitude': 0.01,
            'udd_width': 0.02, 'gap_method': 'cv',
            'gap_pair': [1, 2], 'gap_threshold': 0.04,
            'gap_height': 0.002, 'gap_width': 0.01})
        self.assertEqual(spec['udd']['udd_mode'], 'potential')
        self.assertEqual(spec['gap']['gap_pair'], [1, 2])
        self.assertAlmostEqual(spec['gap']['gap_threshold'], 0.04)

    def test_no_exploration_is_none(self):
        from pyrai3md.workflow.adaptive import AdaptiveSampling
        self.assertIsNone(AdaptiveSampling._bias_spec({}))

    def test_udd_waits_for_its_start_iteration(self):
        from pyrai3md.workflow.adaptive import AdaptiveSampling
        settings = {'explore_udd': 1, 'udd_start_iteration': 2}
        self.assertIsNone(AdaptiveSampling._bias_spec(settings, iteration=1))
        spec = AdaptiveSampling._bias_spec(settings, iteration=2)
        self.assertIsNotNone(spec['udd'])

    def test_validation_mae_reads_the_energy_channel(self):
        from pyrai3md.workflow.adaptive import AdaptiveSampling
        training = {'train_errors': {
            'energy_grad': {'energy': [0.1, 0.3],
                            'gradient': [9.0, 9.0]}}}
        self.assertAlmostEqual(
            AdaptiveSampling._validation_mae(training), 0.2)


class TestDynamicsDataStatus(unittest.TestCase):

    @staticmethod
    def _frame(density='present', energy=True, spin=True):
        system = {'current_state': 0, 'nstate': 2}
        if density == 'present':
            system['state_density'] = [[1.0, 0.0], [0.0, 0.0]]
        if spin:
            system['state_spin'] = [0, 0]
        results = {'energy': [0.0, 0.1]} if energy else {}
        return {'step': 1, 'system': system, 'results': results}

    def _screen(self, frame, expected=True, population=True, stop_code=0):
        from pyrai3md.workflow.records.adaptive import screen_trajectory
        return screen_trajectory(
            0, [frame], {}, {}, 1, check_population=population,
            stop_code=stop_code,
            check_energy_order=True, population_expected=expected)

    def test_no_population_stop_reports_not_stopped(self):
        result = self._screen(self._frame(density=None))
        self.assertEqual(result['population_status'], 'not_stopped')

    def test_a_nosh_run_reports_not_applicable(self):
        result = self._screen(self._frame(density=None), expected=False)
        self.assertEqual(result['population_status'], 'not_applicable')

    def test_a_present_density_is_not_rejudged(self):
        result = self._screen(self._frame())
        self.assertEqual(result['population_status'], 'not_stopped')
        self.assertEqual(result['order_status'], 'ok')

    def test_a_population_stop_reports_stopped(self):
        result = self._screen(self._frame(), stop_code=1)
        self.assertEqual(result['population_status'], 'stopped')

    def test_missing_energy_is_a_missing_order_channel(self):
        result = self._screen(self._frame(energy=False))
        self.assertEqual(result['order_status'], 'missing')

    def test_status_payload_carries_the_data_checks(self):
        from pyrai3md.workflow.adaptive import AdaptiveSampling
        result = self._screen(self._frame(density=None), expected=False)
        payload = AdaptiveSampling._trajectory_payload(result)
        self.assertEqual(payload['population_status'], 'not_applicable')
        self.assertEqual(payload['order_status'], 'ok')
        self.assertFalse(payload['exploration'])


class TestExplorationFlag(unittest.TestCase):

    def test_screening_marks_exploration(self):
        from pyrai3md.workflow.records.adaptive import screen_trajectory
        result = screen_trajectory(0, [], {}, {}, 1, exploration=True)
        self.assertTrue(result['exploration'])
        self.assertFalse(screen_trajectory(1, [], {}, {}, 1)['exploration'])


class TestGapConditions(unittest.TestCase):
    def _workflow(self):
        from pyrai3md.workflow.adaptive import AdaptiveSampling
        workflow = AdaptiveSampling(maxiter=1)
        workflow._initial_conditions = [
            {'coordinates': [[0.0, 0.0, 0.0]],
             'velocities': [[1.0, 0.0, 0.0]]}]
        workflow._last_screen = {'per_trajectory': [{
            'frames': [
                {'flags': {'trigger': False},
                 'coordinates': [[1.0, 0.0, 0.0]],
                 'velocity': [[2.0, 0.0, 0.0]]},
                {'flags': {'trigger': True},
                 'coordinates': [[9.0, 0.0, 0.0]],
                 'velocity': [[9.0, 0.0, 0.0]]},
            ]}]}
        return workflow

    def test_paper_gap_uses_the_pre_trigger_frame_on_both_surfaces(self):
        workflow = self._workflow()
        conditions = workflow._gap_initial_conditions({
            'explore_gap': 1, 'gap_method': 'paper', 'gap_pair': [0, 1]})
        self.assertEqual(len(conditions), 2)
        self.assertEqual([item['current_state'] for item in conditions],
                         [0, 1])
        self.assertEqual(conditions[0]['coordinates'], [[1.0, 0.0, 0.0]])
        self.assertEqual(conditions[0]['velocities'], [[2.0, 0.0, 0.0]])

    def test_cv_method_keeps_the_configured_batch(self):
        workflow = self._workflow()
        conditions = workflow._gap_initial_conditions({
            'explore_gap': 1, 'gap_method': 'cv'})
        self.assertIs(conditions, workflow._initial_conditions)

    def test_no_pre_trigger_frame_falls_back(self):
        workflow = self._workflow()
        workflow._last_screen = {'per_trajectory': [{
            'frames': [{'flags': {'trigger': True},
                        'coordinates': [[1.0, 0.0, 0.0]],
                        'velocity': [[2.0, 0.0, 0.0]]}]}]}
        conditions = workflow._gap_initial_conditions({
            'explore_gap': 1, 'gap_method': 'paper'})
        self.assertIs(conditions, workflow._initial_conditions)


class TestAnalysisIteration(unittest.TestCase):

    def test_all_off_writes_nothing(self):
        from pyrai3md.workflow.adaptive import AdaptiveSampling
        workflow = AdaptiveSampling(maxiter=1)
        with tempfile.TemporaryDirectory() as directory:
            report, fatal, saturated = workflow._analyze_iteration(
                SimpleNamespace(), 1, directory,
                {'per_trajectory': [], 'selected': [], 'n_trajectory': 0},
                [], None, {'analysis': analysis_plan(_Adaptive())}, None)
            self.assertIsNone(report)
            self.assertIsNone(fatal)
            self.assertFalse(saturated)

    def test_trace_is_written_and_summarized(self):
        from pyrai3md.workflow.adaptive import AdaptiveSampling
        workflow = AdaptiveSampling(maxiter=1)
        plan = analysis_plan(_Adaptive(analyze_trace=1))
        outcome = SimpleNamespace(runtime=RuntimeState(analysis_trace={
            'spec': {'gap': None, 'force': True, 'population': False,
                     'energy_drift': False},
            'rows': [{'step': 1, 'gap': {'min': 0.1},
                      'max_force': 0.2}]}))
        screen = {
            'per_trajectory': [{'traj_id': 0, 'last_step': 10,
                                'frames': [{'traj_id': 0, 'step': 1}]}],
            'selected': [], 'n_trajectory': 1}
        with tempfile.TemporaryDirectory() as directory:
            report, fatal, saturated = workflow._analyze_iteration(
                SimpleNamespace(), 1, directory, screen, [outcome], None,
                {'analysis': plan, 'gap_threshold': 0.03,
                 'predict_mode': 0, 'mdstep': 10}, None)
            self.assertIsNotNone(report)
            self.assertIsNone(fatal)
            self.assertFalse(saturated)
            self.assertEqual(
                report['trajectories'][0]['trace']['status'], OK)
            self.assertTrue(os.path.isfile(
                os.path.join(directory, 'analysis_iter-1.json')))
            self.assertEqual(
                report['summary']['progress']['cumulative_new_frames'], 0)

    def test_bias_details_carry_the_paper_switch(self):
        from pyrai3md.workflow.adaptive import AdaptiveSampling
        workflow = AdaptiveSampling(maxiter=1)
        plan = analysis_plan(_Adaptive(analyze_gap='s0s1'))
        frame = {'traj_id': 0, 'step': 1, 'energy': [0.0, 0.1],
                 'bias_status': 'applied', 'gap_excess': 0.01,
                 'gap_velocity_scale': 0.8, 'gap_force_switched': True}
        screen = {'per_trajectory': [{'traj_id': 0, 'last_step': 10,
                                      'frames': [frame]}],
                  'selected': [], 'n_trajectory': 1}
        with tempfile.TemporaryDirectory() as directory:
            report, _fatal, _saturated = workflow._analyze_iteration(
                SimpleNamespace(), 1, directory, screen, [None], None,
                {'analysis': plan, 'gap_threshold': 0.03,
                 'predict_mode': 0, 'mdstep': 10}, None)
            entry = report['trajectories'][0]
            self.assertEqual(entry['gap_velocity_scale'], 0.8)
            self.assertTrue(entry['gap_force_switched'])

    def test_progress_counts_previous_rounds_and_current_labels(self):
        from pyrai3md.workflow.adaptive import AdaptiveSampling
        workflow = AdaptiveSampling(maxiter=1)
        plan = analysis_plan(_Adaptive(analyze_trace=1))
        outcome = SimpleNamespace(runtime=RuntimeState(analysis_trace={
            'spec': {}, 'rows': [{'step': 1, 'max_force': 0.2}]}))
        screen = {
            'per_trajectory': [{'traj_id': 0, 'last_step': 10,
                                'frames': [{'traj_id': 0, 'step': 1}]}],
            'selected': [], 'n_trajectory': 1}
        with tempfile.TemporaryDirectory() as directory:
            report, _fatal, _saturated = workflow._analyze_iteration(
                SimpleNamespace(), 2, directory, screen, [outcome], None,
                {'analysis': plan, 'gap_threshold': 0.03,
                 'predict_mode': 0, 'mdstep': 10},
                {'train_errors': {'energy_grad': {'energy': [0.2, 0.2]}}},
                record={'iterations': [{'new_frames': 2}]}, new_labels=1)
            progress = report['summary']['progress']
            self.assertEqual(progress['cumulative_new_frames'], 3)
            self.assertEqual(progress['rounds_since_improvement'], 0)


class TestHoppingReadsUnbiased(unittest.TestCase):

    def test_fssh_gsh_lzsh_prefer_the_unbiased_energy(self):
        from pyrai3md.workflow.integrator import fssh, gsh, lzsh
        results = {'energy': [9.0, 9.0],
                   'unbiased_energy': [0.0, 1.0]}
        for module in (fssh, gsh, lzsh):
            np.testing.assert_allclose(
                module._energy_of(results, 2), [0.0, 1.0])


class TestAnalysisFormatter(unittest.TestCase):

    def test_every_section_prints_its_numbers(self):
        from pyrai3md.utilities.print_results import format_analysis
        report = {
            'version': 1,
            'plan': {'gap': 's0s1', 'trace': 1, 'pes': 1, 'coverage': 1,
                     'predict_mode': 1, 'explore_udd': True},
            'trajectories': [{
                'traj_id': 0,
                'exploration': True,
                'population_status': 'not_applicable',
                'order_status': 'ok',
                'gap': {'status': 'ok', 'reason': '',
                        'min_gap': 0.02, 'min_pair': [0, 1],
                        'min_step': 3, 'threshold': 0.03,
                        'first_below_threshold': {'step': 3},
                        'fraction_below_threshold': 0.5},
                'trace': {'status': 'ok', 'reason': '',
                          'min_gap': 0.02, 'max_force': 0.5,
                          'max_energy_drift': 0.001,
                          'min_distance': 0.9, 'min_population': 0.8},
                'pes': {'status': 'ok', 'reason': '',
                        'candidates': [{'curvature': 0.1,
                                        'energy_residual': 0.001,
                                        'reconstruction': {'rF': 0.02}}]},
                'predictor': {'status': 'ok', 'reason': '',
                              'model_ids': [2]},
                'bias_status': 'applied',
                'udd_energy': -1e-6,
                'gap_bias_energy': 1e-3,
                'gap_velocity_scale': 0.5,
                'gap_force_switched': True,
                'udd_force_ratio': 0.1,
                'udd_state_switches': 1,
            }],
            'summary': {
                'coverage': {'status': 'ok', 'reason': '',
                             'covered_fraction': 0.5, 'n_candidates': 2,
                             'coverage_radius': 0.1,
                             'nearest_distance': {'median': 0.05}},
                'pes': {'status': 'ok', 'reason': '',
                        'candidates': [{'curvature': 0.1,
                                        'energy_residual': 0.001,
                                        'reconstruction': {'rF': 0.02}}],
                        'resolution_pairs': [[0.2, 0.02]]},
                'saturation': {'status': 'ok', 'reason': '',
                               'moving_average': 0.2, 'improvement': 0.001,
                               'local_tolerance': 0.01,
                               'global_tolerance': 0.1,
                               'saturated': True},
                'progress': {'cumulative_new_frames': 2,
                             'rounds_since_improvement': 0},
            },
        }
        text = format_analysis(report)
        for token in (
                'version', 'switches', 'predict=1', 'explore=udd',
                'population=not_applicable', 'order=ok', 'exploration=True',
                'min 0.0200 Ha pair [0, 1]', 'first below at step 3',
                'min_gap 0.0200', 'max_force 0.5', 'max_drift 0.001',
                'min_dist 0.9000', 'min_pop 0.800',
                '1 candidate(s)', 'rF 2.00e-02', 'kappa 0.100',
                '|rE|max 1.00e-03', 'model_ids [2]',
                'gap_bias_energy=0.001', 'udd_force_ratio=0.1',
                'gap_velocity_scale=0.5', 'gap_force_switched=True',
                '0.500 of 2 candidates', 'nearest median 0.05',
                'MAE 0.200000', 'local 0.01', 'global 0.1', 'SATURATED',
                '2 new labels'):
            self.assertIn(token, text)

    def test_a_not_applicable_section_prints_its_reason(self):
        from pyrai3md.utilities.print_results import format_analysis
        text = format_analysis({
            'version': 1, 'plan': {'gap': 's0s1'},
            'trajectories': [{'traj_id': 0, 'gap': {
                'status': 'not_applicable',
                'reason': 'scope needs at least two states; the frames carry 1'}}],
            'summary': {}})
        self.assertIn('scope needs at least two states', text)


class TestSamplingBlockDataChecks(unittest.TestCase):

    @staticmethod
    def _screen(*statuses):
        per_trajectory = []
        for traj_id, (population, order) in enumerate(statuses):
            per_trajectory.append({
                'traj_id': traj_id, 'last_step': 10, 'last_state': 0,
                'n_sampled': 1, 'n_selected': 0, 'n_discarded': 0,
                'n_bad': 0, 'triggered': False,
                'population_status': population, 'order_status': order,
                'max_errors': {'err_energy': 0.0, 'err_grad': 0.0},
            })
        return {
            'per_trajectory': per_trajectory,
            'n_passed': len(per_trajectory),
            'n_completed': len(per_trajectory),
            'n_sampled': len(per_trajectory), 'n_selected': 0,
            'n_discarded': 0, 'n_bad': 0, 'triggered': False,
            'max_errors': {'err_energy': 0.0, 'err_grad': 0.0},
        }

    def test_a_checks_only_run_still_reports_its_statuses(self):
        from pyrai3md.utilities.print_results import (
            format_adaptive_iteration)
        text = format_adaptive_iteration(
            {'iteration': 1}, screen=self._screen(('not_applicable', 'ok')),
            thresholds={'err_energy': 0.1, 'err_grad': 0.1})
        self.assertIn('Data checks:', text)
        self.assertIn('population=not_applicable, order=ok', text)

    def test_switched_off_checks_are_left_out(self):
        from pyrai3md.utilities.print_results import (
            format_adaptive_iteration)
        text = format_adaptive_iteration(
            {'iteration': 1}, screen=self._screen(('off', 'off')),
            thresholds={'err_energy': 0.1, 'err_grad': 0.1})
        self.assertNotIn('Data checks:', text)

    def test_mixed_statuses_collapse_to_counts(self):
        from pyrai3md.utilities.print_results import (
            format_adaptive_iteration)
        text = format_adaptive_iteration(
            {'iteration': 1},
            screen=self._screen(('missing', 'ok'), ('ok', 'ok')),
            thresholds={'err_energy': 0.1, 'err_grad': 0.1})
        self.assertIn('population=missing(1), ok(1)', text)
        self.assertIn('order=ok', text)


class TestAdaptiveExampleCheck(unittest.TestCase):

    @staticmethod
    def _write(root, *parts, payload):
        path = os.path.join(root, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as handle:
            json.dump(payload, handle)

    def _product(self, root, title, iterations, ntraj):
        self._write(root, title + '.adaptive', title + '.adaptive.json',
                    payload={'n_iterations': iterations})
        for iteration in range(1, iterations + 1):
            status = {'ntraj': ntraj}
            if iteration == 2:
                status.update({'reference_ran': True, 'trained': True,
                               'new_frames': 1, 'modeldir': 'model'})
            self._write(root, title + '.adaptive', 'iter-%d' % iteration,
                        'status_iter-%d.json' % iteration, payload=status)
        if iterations >= 2:
            self._write(root, title + '.adaptive', 'iter-2',
                        'data_iter-2_1.json', payload={})
            self._write(root, title + '.adaptive', 'iter-2', 'reference',
                        'new-samples.json', payload={})
            os.makedirs(os.path.join(root, title + '.adaptive', 'iter-2',
                                     'model'), exist_ok=True)

    def test_a_one_iteration_feature_carrier_passes(self):
        from pyrai3md.qualification.runner import _check_adaptive_products
        with tempfile.TemporaryDirectory() as root:
            yaml_path = os.path.join(root, 'feature.yaml')
            with open(yaml_path, 'w') as handle:
                handle.write('control:\n  title: feat\nmd:\n  ninitcond: 2\n'
                             'adaptive:\n  maxiter: 1\n')
            self._product(root, 'feat', 1, 2)
            result = {'status': 'pass'}
            _check_adaptive_products(result, yaml_path, base_dir=root,
                                     title='feat')
            self.assertEqual(result['status'], 'pass')

    def test_a_two_iteration_feature_carrier_passes(self):
        from pyrai3md.qualification.runner import _check_adaptive_products
        with tempfile.TemporaryDirectory() as root:
            yaml_path = os.path.join(root, 'feature.yaml')
            with open(yaml_path, 'w') as handle:
                handle.write('control:\n  title: feat\nmd:\n  ninitcond: 2\n'
                             'adaptive:\n  maxiter: 2\n')
            self._product(root, 'feat', 2, 2)
            result = {'status': 'pass'}
            _check_adaptive_products(result, yaml_path, base_dir=root,
                                     title='feat')
            self.assertEqual(result['status'], 'pass')

    def test_the_input_mode_reads_the_flat_config(self):
        from pyrai3md.qualification.runner import _check_adaptive_products
        with tempfile.TemporaryDirectory() as root:
            self._write(root, 'feature.json',
                        payload={'maxiter': 3, 'ninitcond': 4})
            self._product(root, 'feat', 3, 4)
            result = {'status': 'pass'}
            _check_adaptive_products(result, os.path.join(root, 'feature.input'),
                                     base_dir=root, title='feat')
            self.assertEqual(result['status'], 'pass')

    def test_a_wrong_iteration_count_still_fails(self):
        from pyrai3md.qualification.runner import _check_adaptive_products
        with tempfile.TemporaryDirectory() as root:
            yaml_path = os.path.join(root, 'feature.yaml')
            with open(yaml_path, 'w') as handle:
                handle.write('control:\n  title: feat\nmd:\n  ninitcond: 2\n'
                             'adaptive:\n  maxiter: 2\n')
            self._product(root, 'feat', 1, 2)
            result = {'status': 'pass'}
            _check_adaptive_products(result, yaml_path, base_dir=root,
                                     title='feat')
            self.assertEqual(result['status'], 'fail')
            self.assertIn('expected 2', result['details'])


class TestRuntimeStateSerialization(unittest.TestCase):

    def test_analysis_fields_are_omitted_when_unset(self):
        payload = RuntimeState().to_dict()
        self.assertNotIn('analysis_trace', payload)
        self.assertNotIn('analysis_bias', payload)

    def test_analysis_fields_round_trip_when_set(self):
        state = RuntimeState(analysis_trace={'spec': {}, 'rows': [1]},
                             analysis_bias={'spec': {}})
        payload = state.to_dict()
        self.assertIn('analysis_trace', payload)
        restored = RuntimeState.from_dict(payload)
        self.assertEqual(restored.analysis_trace, {'spec': {}, 'rows': [1]})
        self.assertEqual(restored.analysis_bias, {'spec': {}})


class TestPropagationBias(unittest.TestCase):

    @staticmethod
    def _state(spec, current=0):
        return SimpleNamespace(
            pmd=SimpleNamespace(system=SimpleNamespace(current_state=current)),
            runtime=RuntimeState(analysis_bias=spec))

    def test_no_spec_is_untouched(self):
        data = {'energy': np.asarray([0.0, 0.1]),
                'gradient': np.zeros((2, 1, 3))}
        before = {key: np.asarray(value).copy()
                  for key, value in data.items()}
        out = analysis_bias.finalize(self._state(None), data)
        np.testing.assert_array_equal(out['energy'], before['energy'])
        self.assertNotIn('bias_status', out)

    def test_gap_bias_preserves_the_mean_and_stops_at_the_threshold(self):
        data = {'energy': np.asarray([0.0, 0.1]),
                'gradient': np.zeros((2, 1, 3))}
        out = analysis_bias.finalize(
            self._state({'gap': {'gap_pair': [0, 1],
                                 'gap_threshold': 0.03,
                                 'gap_height': 0.001}}),
            data)
        self.assertEqual(out['bias_status'], 'applied')
        self.assertAlmostEqual(float(np.sum(out['energy'])),
                               float(np.sum(out['unbiased_energy'])))
        self.assertGreater(float(out['gap_bias_energy']), 0.0)
        self.assertAlmostEqual(float(out['energy'][1] - out['energy'][0]),
                               float(out['unbiased_energy'][1]
                                     - out['unbiased_energy'][0]
                                     + out['gap_bias_energy']))

    def test_gap_bias_is_exactly_zero_below_the_threshold(self):
        data = {'energy': np.asarray([0.0, 0.02]),
                'gradient': np.zeros((2, 1, 3))}
        out = analysis_bias.finalize(
            self._state({'gap': {'gap_pair': [0, 1],
                                 'gap_threshold': 0.03}}),
            data)
        np.testing.assert_array_equal(out['energy'], data['energy'])
        self.assertNotIn('bias_status', out)

    def test_udd_force_mode_changes_the_gradient_not_the_energy(self):
        data = {
            'energy': np.asarray([0.0]),
            'gradient': np.asarray([[[1.0, 0.0, 0.0]]]),
            'alternative_energy': np.asarray([[0.01]]),
            'alternative_gradient': np.asarray([[[[-1.0, 0.0, 0.0]]]]),
            'predict_model_id': 1,
        }
        original_gradient = np.array(data['gradient'], copy=True)
        out = analysis_bias.finalize(
            self._state({'udd': {'udd_mode': 'force',
                                 'udd_amplitude': 0.005,
                                 'udd_width': 0.1}}),
            data)
        self.assertEqual(out['bias_status'], 'applied')
        self.assertAlmostEqual(float(out['energy'][0]), 0.0)
        self.assertFalse(np.allclose(out['gradient'], original_gradient))
        self.assertIn('unbiased_gradient', out)
        self.assertGreater(float(out['udd_force_ratio']), 0.0)

    def test_udd_counts_state_switches(self):
        spec = {'udd': {'udd_mode': 'force', 'udd_amplitude': 0.005,
                        'udd_width': 0.1}}
        data = {
            'energy': np.asarray([0.0, 0.0]),
            'gradient': np.asarray([[[1.0, 0.0, 0.0]],
                                    [[1.0, 0.0, 0.0]]]),
            'alternative_energy': np.asarray([[0.01, 0.01]]),
            'alternative_gradient': np.asarray([[[[-1.0, 0.0, 0.0]],
                                                 [[-1.0, 0.0, 0.0]]]]),
            'predict_model_id': 1,
        }
        first = analysis_bias.finalize(self._state(spec, current=0), data)
        self.assertEqual(first['udd_state_switches'], 0)
        second = analysis_bias.finalize(self._state(spec, current=1), data)
        self.assertEqual(second['udd_state_switches'], 1)

    def test_udd_force_mode_accumulates_bias_work(self):
        spec = {'udd': {'udd_mode': 'force', 'udd_amplitude': 0.005,
                        'udd_width': 0.1, 'last_coordinates': [[0.0, 0.0, 0.0]],
                        'bias_work': 0.0}}
        state = SimpleNamespace(
            pmd=SimpleNamespace(system=SimpleNamespace(
                current_state=0, coordinates=[[0.1, 0.0, 0.0]])),
            runtime=RuntimeState(analysis_bias=spec))
        data = {
            'energy': np.asarray([0.0]),
            'gradient': np.asarray([[[1.0, 0.0, 0.0]]]),
            'alternative_energy': np.asarray([[0.01]]),
            'alternative_gradient': np.asarray([[[[-1.0, 0.0, 0.0]]]]),
            'predict_model_id': 1,
        }
        out = analysis_bias.finalize(state, data)
        self.assertNotEqual(float(out.get('bias_work', 0.0)), 0.0)

    def test_udd_potential_mode_moves_the_active_state_energy(self):
        data = {
            'energy': np.asarray([0.0]),
            'gradient': np.asarray([[[1.0, 0.0, 0.0]]]),
            'alternative_energy': np.asarray([[0.01]]),
            'alternative_gradient': np.asarray([[[[-1.0, 0.0, 0.0]]]]),
            'predict_model_id': 1,
        }
        out = analysis_bias.finalize(
            self._state({'udd': {'udd_mode': 'potential',
                                 'udd_amplitude': 0.005,
                                 'udd_width': 0.1}}),
            data)
        self.assertNotAlmostEqual(float(out['energy'][0]), 0.0)
        self.assertAlmostEqual(float(out['udd_energy']),
                               float(out['energy'][0]))

    def test_udd_needs_a_primary_member(self):
        data = {
            'energy': np.asarray([0.0]),
            'gradient': np.asarray([[[1.0, 0.0, 0.0]]]),
            'alternative_energy': None,
            'alternative_gradient': None,
            'predict_model_id': 0,
        }
        out = analysis_bias.finalize(
            self._state({'udd': {'udd_mode': 'force'}}), data)
        self.assertNotIn('bias_status', out)

    def test_udd_follows_the_current_state(self):
        data = {
            'energy': np.asarray([0.0, 0.0]),
            'gradient': np.asarray([[[5.0, 0.0, 0.0]],
                                    [[1.0, 0.0, 0.0]]]),
            'alternative_energy': np.asarray([[0.0, 0.01]]),
            'alternative_gradient': np.asarray([[[[-5.0, 0.0, 0.0]],
                                                 [[-1.0, 0.0, 0.0]]]]),
            'predict_model_id': 1,
        }
        out = analysis_bias.finalize(
            self._state({'udd': {'udd_mode': 'force',
                                 'udd_amplitude': 0.005,
                                 'udd_width': 0.1}}, current=1),
            data)
        self.assertEqual(out['bias_status'], 'applied')
        np.testing.assert_allclose(out['gradient'][0, 0], [5.0, 0.0, 0.0])
        self.assertFalse(np.allclose(out['gradient'][1, 0],
                                     [1.0, 0.0, 0.0]))

    def test_paper_gap_replaces_the_active_gradient(self):
        updates = {}

        def update_system(values):
            updates.update(values)

        state = SimpleNamespace(
            pmd=SimpleNamespace(system=SimpleNamespace(current_state=0)),
            runtime=RuntimeState(analysis_bias={'gap': {
                'gap_method': 'paper', 'gap_pair': [0, 1],
                'gap_threshold': 0.03}}),
            velocities=np.asarray([[1.0, 0.0, 0.0]]),
            kinetic_energy=lambda: 0.1,
            update_system=update_system)
        data = {'energy': np.asarray([0.0, 0.1]),
                'gradient': np.asarray([[[5.0, 0.0, 0.0]],
                                        [[2.0, 0.0, 0.0]]])}
        out = analysis_bias.finalize(state, data)
        self.assertTrue(out.get('gap_force_switched'))
        np.testing.assert_allclose(
            out['gradient'][0], out['unbiased_gradient'][1]
            - out['unbiased_gradient'][0])
        self.assertEqual(out.get('gap_velocity_scale', 1.0), 1.0)

    def test_paper_gap_rescales_velocities_when_excess_is_payable(self):
        updates = {}

        def update_system(values):
            updates.update(values)

        spec = {'gap': {'gap_method': 'paper', 'gap_pair': [0, 1],
                        'gap_threshold': 0.03, 'reference_total': 0.1}}
        state = SimpleNamespace(
            pmd=SimpleNamespace(system=SimpleNamespace(current_state=0)),
            runtime=RuntimeState(analysis_bias=spec),
            velocities=np.asarray([[1.0, 0.0, 0.0]]),
            kinetic_energy=lambda: 0.2,
            update_system=update_system)
        data = {'energy': np.asarray([0.0, 0.1]),
                'gradient': np.asarray([[[5.0, 0.0, 0.0]],
                                        [[2.0, 0.0, 0.0]]])}
        out = analysis_bias.finalize(state, data)
        self.assertAlmostEqual(out['gap_velocity_scale'],
                               float(np.sqrt(0.5)))
        self.assertIn('velocities', updates)


if __name__ == '__main__':
    unittest.main()
