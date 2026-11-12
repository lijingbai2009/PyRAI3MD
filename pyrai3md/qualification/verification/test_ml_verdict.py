"""
======================================================================
Test suite for the machine-learning outcome channel and the scan
workflow's registration:
- pyrai3md/workflow/records/ml.py (the reception both ML classes call)
- pyrai3md/workflow/train.py and prediction.py (the nodes those jobtypes
  resolve to) and single.py, which keeps plain completion for everything
  that is not machine learning
- pyrai3md/workflow/search.py (the class the scan token resolves to)

Tests cover:
- a training or prediction call is settled by whether it carried the
  record its jobtype promises, with the missing or empty channel named
- a non-ML jobtype keeps the plain completion of the single-shot path
  now that the ML jobtypes have their own classes
- the banner a run leaves in its own log
- 'search' resolves to the grid scan, whose child jobtype is a training,
  and 'train'/'prediction' still resolve to the single-shot path

Nothing here trains a network: the channel reads records, so the records
are written out here.

Oct 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

from pyrai3md.workflow.records.ml import (ML_JOBTYPES, ml_detail, ml_verdict,
                                          settle_ml_outcome)

_TRAIN_RECORD = {
    'train_errors': {'energy_grad': {'energy': [0.068, 0.091],
                                     'gradient': [0.011, 0.013]}},
    'train_metric_units': {'energy': 'eV', 'gradient': 'eV/angstrom'},
    'train_wall_seconds': 63.1,
    'train_arm': 'local',
    'train_data': '/data/TRAIN.json',
    'train_modeldir': '/run/NN',
    'train_fitlogs': ['/run/NN/energy_grad/fitlog_0.txt',
                      '/run/NN/energy_grad/fitlog_1.txt'],
    'train_summary': '/run/NN_train.summary.txt',
}

_PREDICT_RECORD = {
    'predict_errors': {'energy': {'mae': 0.02, 'rmse': 0.03, 'max': 0.1,
                                  'spread_max': 0.01}},
    'predict_metric_units': {'energy': 'eV'},
    'predict_frames': 24,
    'predict_wall_seconds': 2.7,
}


class _Runtime:
    """Records the outcome a settle call marked."""

    def __init__(self):
        self.marks = []

    def mark_completed(self):
        self.marks.append(('completed', None))

    def mark_failed(self, message):
        self.marks.append(('failed', message))


class _Logger:
    """Collects the banners a settle call wrote."""

    def __init__(self):
        self.banners = []

    def log_ml_summary(self, settled, detail='', message='', label=''):
        self.banners.append((settled, detail, message, label))


class _Pmd:
    def __init__(self, results):
        self.results = results


def _declared_contract():
    """The record contract the serving skill declares (its manifest)."""
    from pyrai3md.skills.nn_pes_skill import query
    return dict(query.get_metadata()['result_contract'])


class _State:
    """The attributes the channel reads, and nothing more."""

    def __init__(self, jobtype, results=None):
        self.jobtype = jobtype
        self.runtime = _Runtime()
        self.logger = _Logger()
        self.pmd = _Pmd(results or {})

    def iter_tasks(self):
        # The verdict reads the serving skill's declared contract, so
        # the stub names the skill the records below come from.
        yield 'theory', {'skill_name': 'nn_pes_skill'}


class TestVerdict(unittest.TestCase):
    """What settles a training or prediction call."""

    def test_a_populated_training_record_settles(self):
        settled, reason = ml_verdict('train', _TRAIN_RECORD,
                                  _declared_contract()['train'])
        self.assertTrue(settled)
        self.assertEqual('', reason)

    def test_an_empty_or_absent_channel_fails_by_name(self):
        cases = (
            ({}, 'no train_errors channel'),
            ({'train_errors': {}}, 'empty train_errors'),
            ({'train_errors': {'energy_grad': {}}}, 'empty train_errors'),
            ({'train_errors': {'energy_grad': {'energy': [0.1]}}},
             'no train_metric_units'),
        )
        for record, expected in cases:
            with self.subTest(record=sorted(record)):
                settled, reason = ml_verdict('train', record,
                                      _declared_contract()['train'])
                self.assertFalse(settled)
                self.assertIn(expected, reason)

    def test_a_prediction_over_no_frames_is_not_a_result(self):
        record = dict(_PREDICT_RECORD, predict_frames=0)
        settled, reason = ml_verdict('prediction', record,
                                     _declared_contract()['prediction'])
        self.assertFalse(settled)
        self.assertIn('no test frame', reason)

    def test_a_missing_frame_count_is_read_as_zero(self):
        record = {key: value for key, value in _PREDICT_RECORD.items()
                  if key != 'predict_frames'}
        self.assertFalse(ml_verdict('prediction', record,
                                     _declared_contract()['prediction'])[0])

    def test_the_leading_token_decides_the_job(self):
        # A jobtype is a token string; only the leading one names the job.
        self.assertTrue(ml_verdict('train nn', _TRAIN_RECORD,
                                            _declared_contract()['train'])[0])

    def test_result_objects_are_read_as_mappings(self):
        from pyrai3md.data.runtime.result import Result

        settled, _reason = ml_verdict('train', Result(_TRAIN_RECORD),
                                          _declared_contract()['train'])
        self.assertTrue(settled)


class TestSettle(unittest.TestCase):
    """The state a settle call leaves behind."""

    def test_a_training_run_marks_completed_and_reports(self):
        state = _State('train')
        settle_ml_outcome(state, _TRAIN_RECORD)
        self.assertEqual([('completed', None)], state.runtime.marks)
        settled, detail, message, label = state.logger.banners[0]
        self.assertTrue(settled)
        self.assertEqual('Model training', label)
        self.assertIn('1 ensembles', detail)
        self.assertIn('63.1 s', detail)
        self.assertIn('fit logs /run/NN/energy_grad', detail)
        self.assertNotIn('summary', detail)
        self.assertEqual('', message)

    def test_an_empty_training_run_fails_with_the_named_channel(self):
        state = _State('train')
        settle_ml_outcome(state, {'train_errors': {}})
        self.assertEqual('failed', state.runtime.marks[0][0])
        self.assertIn('train_errors', state.runtime.marks[0][1])
        self.assertFalse(state.logger.banners[0][0])

    def test_fitlogs_that_travelled_as_an_array_are_read(self):
        # Result converts list channels to numpy arrays; a truth test on
        # that array raises, so the detail must read it as a sequence.
        from pyrai3md.data.runtime.result import Result

        detail = ml_detail('train', Result(dict(_TRAIN_RECORD)))
        self.assertIn('fit logs /run/NN/energy_grad', detail)

    def test_a_record_only_on_the_state_is_still_read(self):
        state = _State('train', results=dict(_TRAIN_RECORD))
        settle_ml_outcome(state, None)
        self.assertEqual([('completed', None)], state.runtime.marks)

    def test_a_prediction_run_settles_on_frames_and_errors(self):
        state = _State('prediction')
        settle_ml_outcome(state, _PREDICT_RECORD)
        self.assertEqual([('completed', None)], state.runtime.marks)
        detail = state.logger.banners[0][1]
        self.assertIn('24 frames', detail)
        # A test-set record is keyed by channel, not by ensemble folder,
        # so the banner must not call its keys ensembles.
        self.assertIn('1 channels', detail)
        self.assertNotIn('ensembles', detail)

    def test_other_jobtypes_keep_the_plain_completion(self):
        # The delegation must not change what an energy, gradient or
        # skill-driven job settles to: it is the same mark as before.
        for jobtype in ('energy', 'gradient', 'optimize', 'md', 'frequency'):
            state = _State(jobtype, results={'energy': -1.0})
            settle_ml_outcome(state, {'energy': -1.0})
            with self.subTest(jobtype=jobtype):
                self.assertEqual([('completed', None)], state.runtime.marks)
                self.assertEqual([], state.logger.banners)

    def test_a_state_without_a_runtime_still_reports(self):
        state = _State('train')
        state.runtime = None
        settle_ml_outcome(state, _TRAIN_RECORD)
        self.assertEqual(1, len(state.logger.banners))

    def test_a_state_without_a_logger_still_marks(self):
        state = _State('train')
        state.logger = None
        settle_ml_outcome(state, _TRAIN_RECORD)
        self.assertEqual([('completed', None)], state.runtime.marks)

    def test_the_ml_word_list_is_the_contract(self):
        self.assertEqual(('train', 'prediction'), ML_JOBTYPES)

    def test_detail_of_an_absent_record_is_empty(self):
        self.assertEqual('', ml_detail('train', None))


class TestRegistration(unittest.TestCase):
    """Which class the scan and training tokens resolve to."""

    def test_search_resolves_to_the_grid_scan(self):
        from pyrai3md.workflow import JOBTYPE_MAP, WORKFLOW_REGISTRY
        from pyrai3md.workflow.search import GridSearch

        self.assertIs(GridSearch, JOBTYPE_MAP['search'])
        self.assertIs(GridSearch, WORKFLOW_REGISTRY['search'])
        self.assertEqual('search', GridSearch.name)
        self.assertEqual('batch', GridSearch.workflow_kind)

    def test_a_point_trains_rather_than_evaluates_a_geometry(self):
        from pyrai3md.workflow.search import GridSearch

        self.assertEqual('train', GridSearch.child_jobtype)

    def test_training_jobtypes_have_their_own_classes(self):
        # The ML jobtypes are workflow nodes of their own now, so their
        # dispatch keywords can be steered from &train / &prediction.
        from pyrai3md.workflow import JOBTYPE_MAP, WORKFLOW_REGISTRY
        from pyrai3md.workflow.prediction import Prediction
        from pyrai3md.workflow.single import SingleShot
        from pyrai3md.workflow.train import Train

        self.assertIs(Train, JOBTYPE_MAP['train'])
        self.assertIs(Prediction, JOBTYPE_MAP['prediction'])
        self.assertIs(Train, WORKFLOW_REGISTRY['train'])
        self.assertIs(Prediction, WORKFLOW_REGISTRY['prediction'])
        self.assertEqual('train', Train.name)
        self.assertEqual('prediction', Prediction.name)
        self.assertEqual('single', Train.workflow_kind)
        self.assertEqual('single', Prediction.workflow_kind)
        # A single-call class still lets the pmd tail print the result
        # block, which is what `workflow_kind` decides.
        self.assertNotEqual('iterative', Train.workflow_kind)
        # SingleShot is no longer in the jobtype map for these tokens.
        self.assertIsNot(SingleShot, JOBTYPE_MAP['train'])

    def test_a_grid_point_resolves_to_the_training_class(self):
        # The scan dispatches child_jobtype 'train'; the child re-enters
        # execute() under the main driver, so its node is Train.
        from pyrai3md.qualification.verification.test_grid_plan import _nn_task
        from pyrai3md.workflow import resolve_workflow
        from pyrai3md.workflow.search import GridSearch
        from pyrai3md.workflow.train import Train

        self.assertEqual('train', GridSearch.child_jobtype)
        state = _nn_task(mult_nn=1)
        state.pmd.control.jobtype = 'train'
        self.assertIs(Train, resolve_workflow(state))

    def test_single_shot_keeps_plain_completion_for_other_jobtypes(self):
        # Reverting the else arm to a bare mark_completed must not change
        # what a skill-driven or property job settles to.
        from pyrai3md.workflow.single import SingleShot

        shot = SingleShot()
        for jobtype in ('energy', 'gradient', 'frequency', 'scan'):
            state = _State(jobtype, results={'energy': -1.0})
            shot._finalize(state, {'energy': -1.0})
            with self.subTest(jobtype=jobtype):
                self.assertEqual([('completed', None)], state.runtime.marks)
                self.assertEqual([], state.logger.banners)

    def test_the_structural_placeholder_is_gone(self):
        # The scan replaced it; a class named Search under this token would
        # mean two different jobs in one vocabulary.
        import pyrai3md.workflow.search as module

        self.assertFalse(hasattr(module, 'Search'))
        self.assertFalse(hasattr(module, 'search_energies'))

    def test_the_scan_node_can_be_parallel(self):
        # A declaration that forces multitask False would outvote the
        # &search section and make every scan serial.
        from pyrai3md.workflow.dispatch.planner import (
            WORKFLOW_NODE_DECLARATIONS, WORKFLOW_SECTIONS)

        self.assertEqual('search', WORKFLOW_SECTIONS['search'])
        self.assertNotIn('multitask', WORKFLOW_NODE_DECLARATIONS['search'])

    def test_the_titles_name_the_jobs(self):
        from pyrai3md.kernel.logger import JOBTYPE_SECTIONS, WORKFLOW_TITLES

        self.assertIn('Grid Search', WORKFLOW_TITLES['search'])
        self.assertEqual(['search'], JOBTYPE_SECTIONS['search'])


    def test_a_batch_class_names_an_importable_child_entry(self):
        # A batch hands its units to the same execution pipeline a worker
        # process uses, so the entry that runs one unit must be reachable
        # by name: a closure over the parent workflow would drag its
        # Calculator -- and any live backend inside it -- into a payload
        # that cannot be pickled. This is the defect that made the grid
        # scan's parallel arm die, and the one the Hessian shared.
        import importlib

        from pyrai3md.workflow.hessian import NumericalHessian
        from pyrai3md.workflow.search import GridSearch

        for workflow_cls in (GridSearch, NumericalHessian):
            with self.subTest(workflow=workflow_cls.__name__):
                workflow = workflow_cls()
                module, function = workflow.executor_spec
                self.assertTrue(callable(
                    getattr(importlib.import_module(module), function)))

    def test_the_scan_and_the_point_do_not_reuse_a_bound_executor(self):
        # The copy a pooled batch makes must carry no calculator and no
        # injected executor, or the unit payload pickles the parent's
        # live backend.
        from pyrai3md.workflow.search import GridSearch

        scan = GridSearch()
        scan.local_executor = lambda state, runtime_idx=None: None
        copy = scan._create_copy()
        self.assertIsNone(copy.calculator)
        self.assertIsNone(copy.local_executor)


class TestScanSection(unittest.TestCase):
    """The plan refuses before any work, and says why."""

    def test_an_empty_grid_is_refused(self):
        from pyrai3md.workflow.grid.plan import plan_points, skill_surface

        surface = skill_surface('nn_pes_skill')
        with self.assertRaises(ValueError):
            plan_points({}, surface['searchable'], surface['query'])

    def test_a_typo_in_the_ranking_vocabulary_is_refused(self):
        from pyrai3md.schema.validate import validate_search_section

        for value in ('average', 'descend'):
            key = 'sort_stat' if value == 'average' else 'sort_order'
            with self.assertRaises(ValueError) as caught:
                validate_search_section({'search': {key: value}})
            self.assertIn('search.%s' % key, str(caught.exception))

    def test_the_declared_defaults_pass(self):
        from pyrai3md.schema.config import Search

        from pyrai3md.schema.validate import validate_search_section

        validate_search_section({'search': Search().model_dump()})

    def test_a_search_multitask_keyword_reaches_the_graph_node(self):
        # The declaration layer beats the input keyword, so a node that
        # declares multitask False could never be parallel. This is the
        # test that fails if that declaration comes back.
        from pyrai3md.qualification.verification.test_grid_plan import (
            _nn_task)
        from pyrai3md.workflow import WORKFLOW_REGISTRY
        from pyrai3md.workflow.dispatch.planner import ExecutionGraphBuilder
        from pyrai3md.workflow.search import GridSearch

        state = _nn_task()
        state.pmd.search.multitask = True
        state.pmd.search.multipool = 4
        graph = ExecutionGraphBuilder(WORKFLOW_REGISTRY).build(
            state, GridSearch)
        node = next(entry for entry in graph if entry['name'] == 'search')
        self.assertTrue(node['multitask'])
        self.assertEqual(4, node['multipool'])

        state = _nn_task()
        graph = ExecutionGraphBuilder(WORKFLOW_REGISTRY).build(
            state, GridSearch)
        node = next(entry for entry in graph if entry['name'] == 'search')
        self.assertFalse(node['multitask'])


class TestSearchGridCarriers(unittest.TestCase):
    """One grid, three carriers, and the refusals that keep them one.

    The classic carrier is the one that used to lose data: the section
    reader kept only the last value of a repeated key, so two `grid` lines
    planned a scan over one axis and nothing downstream could see the
    other line had existed. Both classic spellings (several lines, or one
    line with ';' groups) now reach the same mapping as the yaml one.
    """

    def setUp(self):
        import os
        import shutil
        import tempfile

        from pyrai3md.pmd import parse_input
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate

        self.dir = tempfile.mkdtemp(prefix='grid_carriers_')
        self.addCleanup(shutil.rmtree, self.dir, True)
        # The classic reader resolves `&molecule coord` against the working
        # directory, as it does for a user, so the case runs from its own.
        origin = os.getcwd()
        self.addCleanup(os.chdir, origin)
        os.chdir(self.dir)
        self.parse_input = parse_input
        self.pipeline = lambda raw: validate(normalize(build_config(raw)))
        self._write_xyz()

    def _write_xyz(self):
        import os

        with open(os.path.join(self.dir, 'grid.xyz'), 'w') as handle:
            handle.write('1\nenergy 0.0\nC 0.0 0.0 0.0\n')

    def _classic(self, grid_lines):
        import os

        path = os.path.join(self.dir, 'grid_case.input')
        with open(path, 'w') as handle:
            handle.write(
                '&control\n title grid_case\n jobtype search\n'
                ' theory nn\n&end\n&molecule\n coord grid.xyz\n&end\n'
                '&search\n%s&end\n&nn\n method mlp_eg\n'
                ' modeldir ./nn\n train_data ./train.json\n&end\n'
                % grid_lines)
        return self.pipeline(self.parse_input(path))

    EXPECTED = {'depth': [1, 2], 'nn_size': [8, 16]}

    def test_two_keyword_lines_and_one_semicolon_line_agree(self):
        two_lines = self._classic(' grid depth 1 2\n grid nn_size 8 16\n')
        one_line = self._classic(' grid depth 1 2; nn_size 8 16\n')
        self.assertEqual(self.EXPECTED, two_lines.search.grid)
        self.assertEqual(one_line.search.grid, two_lines.search.grid)

    def test_the_section_keeps_its_other_keywords(self):
        config = self._classic(' grid depth 1 2; nn_size 8 16\n'
                               ' retrieve 1\n sort_stat mean\n')
        self.assertEqual(1, config.search.retrieve)
        self.assertEqual('mean', config.search.sort_stat)
        self.assertEqual(self.EXPECTED, config.search.grid)

    def test_a_repeated_keyword_is_refused_not_resolved_by_order(self):
        from pyrai3md.schema.adapter import _parse_search_grid

        for value in ('depth 1 2; depth 3', ['depth 1 2', 'depth 3']):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as caught:
                    _parse_search_grid(value)
                self.assertIn("names 'depth' twice", str(caught.exception))

    def test_a_group_without_candidates_is_refused(self):
        from pyrai3md.schema.adapter import _parse_search_grid

        with self.assertRaises(ValueError) as caught:
            _parse_search_grid('depth 1 2; nn_size')
        self.assertIn('no candidate value', str(caught.exception))

    def test_a_non_numeric_candidate_reaches_the_skill_as_text(self):
        from pyrai3md.schema.adapter import _parse_search_grid

        # The reader keeps 'two' as text; the skill that declared `depth`
        # as an integer is what refuses it, when the grid is planned. The
        # framework keeps no type table of a skill's keywords.
        self.assertEqual({'depth': [1, 'two']},
                         _parse_search_grid('depth 1 two'))
        from pyrai3md.skills.nn_pes_skill import query

        with self.assertRaises(ValueError) as caught:
            query.check_searchable('depth', 'two')
        self.assertIn('expects an integer candidate', str(caught.exception))

    def test_an_unreadable_carrier_names_the_two_spellings(self):
        from pyrai3md.schema.adapter import _parse_search_grid

        with self.assertRaises(ValueError) as caught:
            _parse_search_grid(42)
        message = str(caught.exception)
        self.assertIn('mapping', message)
        self.assertIn('classic input', message)

    def test_a_fullwidth_semicolon_says_which_separator_it_meant(self):
        from pyrai3md.schema.adapter import _parse_search_grid

        with self.assertRaises(ValueError) as caught:
            _parse_search_grid('depth 1\uff1b nn_size 2')
        self.assertIn('group separator', str(caught.exception))

    def test_a_bare_candidate_wraps_into_a_one_entry_list(self):
        from pyrai3md.schema.adapter import _parse_search_grid

        self.assertEqual({'depth': [2]}, _parse_search_grid('depth 2'))
        self.assertEqual({'depth': [2]}, _parse_search_grid({'depth': 2}))


class TestScanRefusesARetrainingRequest(unittest.TestCase):
    """`train_mode: retraining` cannot describe a scan, and says so early."""

    def _context(self, mode):
        import json
        import os
        import tempfile

        from pyrai3md.qualification.verification.test_grid_plan import _nn_task
        from pyrai3md.workflow.grid.plan import build_context, skill_surface

        directory = tempfile.mkdtemp(prefix='grid_mode_')
        self.addCleanup(lambda: __import__('shutil').rmtree(directory, True))
        data = os.path.join(directory, 'train.json')
        with open(data, 'w') as handle:
            json.dump({'natom': 2, 'nstate': 1,
                       'xyz': [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
                       'energy': [[-1.0]],
                       'grad': [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]}, handle)
        state = _nn_task(train_data=data)
        state.theory[0]['config']['skill_config']['train_mode'] = mode
        return build_context(state, skill_surface('nn_pes_skill'),
                             {'grid_root': os.path.join(directory, 'grid')})

    def test_a_plain_training_scan_is_planned(self):
        self.assertEqual('mlp_eg_train', self._context('training')['profile'])

    def test_a_continuation_scan_is_refused_before_any_point(self):
        with self.assertRaises(ValueError) as caught:
            self._context('retraining')
        message = str(caught.exception)
        self.assertIn('fits every point from scratch', message)
        self.assertIn('model', message, 'the reason the two cannot combine')



if __name__ == '__main__':
    unittest.main()
