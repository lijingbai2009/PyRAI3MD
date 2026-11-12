"""
======================================================================
Test suite for the plan side of a hyperparameter grid scan:
- pyrai3md/workflow/grid/plan.py

Tests cover:
- skill_surface: the declaration read through the skill's query module,
  and the refusals when a skill declares nothing
- resolve_train_profile: a pinned profile, the unique profile of a
  method, and an ambiguous method named with its alternatives
- plan_points: the Cartesian order, the candidate rules inherited from
  the skill, and the deterministic point key
- write_point on a state materialized by the real Dispatcher: the values
  land in the role sections the trained profile fits, the member axis is
  never confused with the grid axis, and the child carries a training
  profile, its own model directory and no capability request

Nothing here needs TensorFlow: the plan reads declarations and writes
dictionaries.

Oct 09 2026
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

from pyrai3md.workflow.dispatch.dispatcher import Dispatcher
from pyrai3md.workflow.grid.plan import (build_context, first_skill_name,
                                         plan_points, point_key,
                                         skill_surface, write_point)

_SKILL = 'nn_pes_skill'


def _nn_task(profile='mlp_eg_train', method='mlp_eg', train_data='',
             mult_nn=2):
    """One static task shaped the way the task queue builds an nn task."""
    from pyrai3md.data.state import PyRAIState
    from pyrai3md.data.runtime.state import RuntimeState
    from pyrai3md.schema.config import PyRAIConfig

    config = PyRAIConfig()
    config.control.jobtype = 'search'
    config.system.atoms = ['C', 'H']
    config.system.coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    config.system.natom = 2
    config.control.title = 'grid_unit'

    state = PyRAIState(runtime=RuntimeState(total_steps=1))
    state.pmd = config
    task = {
        'role': 'qm', 'task': 0, 'task_id': _SKILL, 'layer_id': 0,
        'static_id': 'qm_layer0_grid', 'contribution': 1,
        'region': 'all', 'region_indices': [], 'region_charge': None,
        'region_multiplicity': None,
        'skill_name': _SKILL,
        'api': 'pyrai3md.skills.nn_pes_skill.backend.NNPESBackend',
        'capabilities': ['energy', 'gradient'],
        'config': {
            'skill_config': {
                'profile': profile, 'method': method,
                'modeldir': './nn_parent', 'train_data': train_data,
                'pred_data': '', 'mult_nn': mult_nn,
                'execution_mode': 'local', 'scratch_path': './scratch',
                'project': 'NNPES', 'eg': {'depth': 4}, 'nac': {}, 'soc': {},
            },
            'results': {},
        },
        'multitask': None, 'remotetask': None, 'multipool': None,
    }
    state.add_task('theory', task)
    return state


class TestSkillSurface(unittest.TestCase):
    """The declaration a scan plans against."""

    def setUp(self):
        self.surface = skill_surface(_SKILL)

    def test_surface_carries_declaration_and_targets(self):
        self.assertEqual('nn', self.surface['label'])
        self.assertTrue(self.surface['searchable'])
        self.assertIn('mlp_eg_train', self.surface['targets'])

    def test_unknown_skill_is_named(self):
        with self.assertRaises(ValueError) as caught:
            skill_surface('no_such_skill')
        self.assertIn('no_such_skill', str(caught.exception))

    def test_first_skill_name_follows_the_task_queue(self):
        self.assertEqual(_SKILL, first_skill_name(_nn_task()))


class TestTrainProfile(unittest.TestCase):
    """Which training profile one point fits -- the skill's own answer."""

    def setUp(self):
        self.query = skill_surface(_SKILL)['query']

    def _profile(self, **config):
        return self.query.scan_requirements(
            dict({'train_data': 'x.json'}, **config))['profile']

    def test_a_pinned_train_profile_is_honored(self):
        self.assertEqual('mlp_nac2_train', self._profile(
            profile='mlp_nac2_train', method='mlp_nac2'))

    def test_a_serving_profile_falls_back_to_the_method(self):
        self.assertEqual('mlp_eg_train', self._profile(
            profile='mlp_eg_gradient', method='mlp_eg'))

    def test_an_ambiguous_method_names_its_alternatives(self):
        with self.assertRaises(ValueError) as caught:
            self._profile(profile='', method='mlp_e')
        message = str(caught.exception)
        self.assertIn('mlp_e_train', message)
        self.assertIn('mlp_e_soc_train', message)

    def test_an_unknown_method_is_named(self):
        with self.assertRaises(ValueError):
            self._profile(profile='', method='nope')


class TestPlanPoints(unittest.TestCase):
    """Grid expansion and the candidate rules."""

    def setUp(self):
        surface = skill_surface(_SKILL)
        self.searchable = surface['searchable']
        self.query = surface['query']

    def test_product_order_is_keyword_sorted_and_complete(self):
        points = plan_points({'nn_size': [24, 8], 'depth': [1, 2]},
                             self.searchable, self.query)
        self.assertEqual(4, len(points))
        self.assertEqual(['depth=1~nn_size=24', 'depth=1~nn_size=8',
                          'depth=2~nn_size=24', 'depth=2~nn_size=8'],
                         [point['key'] for point in points])

    def test_keys_are_unique_and_recover_the_values(self):
        points = plan_points({'depth': [1, 2], 'dropout': [0.0, 0.005],
                              'loss_weights': [[1, 1], [1, 10]]},
                             self.searchable, self.query)
        keys = [point['key'] for point in points]
        self.assertEqual(len(keys), len(set(keys)))
        for point in points:
            self.assertEqual(point['key'], point_key(point['values']))
            pairs = dict(chunk.split('=', 1)
                         for chunk in point['key'].split('~'))
            self.assertEqual(sorted(point['values']), sorted(pairs))
            for keyword, value in point['values'].items():
                # Each token is what that one value alone would render,
                # so a folder name can be read back into its point.
                self.assertEqual(point_key({keyword: value}),
                                 '%s=%s' % (keyword, pairs[keyword]))

    def test_a_bool_candidate_prints_a_readable_token(self):
        points = plan_points({'use_dropout': [False, True]},
                             self.searchable, self.query)
        self.assertEqual(['use_dropout=false', 'use_dropout=true'],
                         [point['key'] for point in points])

    def test_an_empty_grid_refuses_rather_than_running_nothing(self):
        for grid in ({}, {'depth': []}, []):
            with self.assertRaises(ValueError):
                plan_points(grid, self.searchable, self.query)

    def test_an_undeclared_keyword_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            plan_points({'epochs': [1, 2]}, self.searchable, self.query)
        self.assertIn('epochs', str(caught.exception))

    def test_a_refused_candidate_names_the_grid_as_its_source(self):
        with self.assertRaises(ValueError) as caught:
            plan_points({'depth': [2, -1]}, self.searchable, self.query)
        self.assertIn('&search grid', str(caught.exception))

    def test_a_candidate_the_skill_cannot_use_is_refused_by_the_skill(self):
        # Text that is not a number reaches the plan as text: the skill
        # that declared the keyword's type is the one that refuses it, so
        # the framework never keeps a second copy of that table.
        with self.assertRaises(ValueError) as caught:
            plan_points({'depth': ['two']}, self.searchable, self.query)
        self.assertIn("'depth' expects an integer candidate",
                      str(caught.exception))


class TestWritePoint(unittest.TestCase):
    """What one stamped child state carries."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='grid_plan_')
        self.data = os.path.join(self.dir, 'train.json')
        with open(self.data, 'w') as handle:
            json.dump({'natom': 2, 'nstate': 1, 'xyz': [[[0.0, 0.0, 0.0],
                                                         [1.0, 0.0, 0.0]]],
                       'energy': [[-1.0]], 'grad': [[[0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0]]]},
                      handle)
        surface = skill_surface(_SKILL)
        self.surface = surface

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _context(self, state=None, grid_root=''):
        section = {'grid_root': grid_root or os.path.join(self.dir, 'grid')}
        return build_context(state or _nn_task(train_data=self.data),
                             self.surface, section)

    def _stamp(self, values, context=None, state=None):
        point = {'values': values, 'key': point_key(values)}
        state = state or _nn_task(train_data=self.data)
        unit = Dispatcher(state=state).materialize_unit({}, 0,
                                                       child_jobtype='train')
        directory = write_point(unit['state'], point,
                                context or self._context())
        return unit, directory

    def test_values_land_in_the_section_the_profile_trains(self):
        unit, _ = self._stamp({'depth': 3})
        skill = unit['state'].theory[0]['config']['skill_config']
        self.assertEqual(3, skill['eg']['depth'])
        self.assertNotIn('depth', skill['nac'],
                         "a section this profile does not train stays alone")

    def test_a_scalar_never_reaches_the_skill_as_a_member_list(self):
        unit, _ = self._stamp({'nn_size': 60}, context=self._context(
            _nn_task(mult_nn=2, train_data=self.data)))
        skill = unit['state'].theory[0]['config']['skill_config']
        self.assertEqual(60, skill['eg']['nn_size'])

    def test_a_list_candidate_is_wrapped_as_one_member_group(self):
        unit, _ = self._stamp({'loss_weights': [1, 10]})
        skill = unit['state'].theory[0]['config']['skill_config']
        self.assertEqual([[1, 10]], skill['eg']['loss_weights'])

    def test_the_child_carries_a_training_profile_and_no_capabilities(self):
        unit, directory = self._stamp({'depth': 2})
        task = unit['state'].theory[0]
        skill = task['config']['skill_config']
        self.assertEqual('mlp_eg_train', skill['profile'])
        self.assertEqual([], task['capabilities'])
        self.assertTrue(os.path.isabs(skill['modeldir']))
        self.assertTrue(os.path.isabs(skill['train_data']))
        self.assertEqual(directory, skill['modeldir'])

    def test_the_key_decides_the_directory_alone(self):
        # The same point re-derived in a different state maps onto the same
        # folder, which is what lets a restarted scan find its own work.
        first, dir_a = self._stamp({'depth': 2, 'nn_size': 8})
        second, dir_b = self._stamp({'nn_size': 8, 'depth': 2},
                                    state=_nn_task(train_data=self.data))
        self.assertEqual(dir_a, dir_b)
        self.assertIn('depth=2~nn_size=8', dir_a)

    def test_the_parent_state_is_never_touched(self):
        state = _nn_task(train_data=self.data)
        before = json.dumps(state.theory[0]['config']['skill_config'],
                            default=str)
        self._stamp({'depth': 5}, state=state)
        self.assertEqual(before,
                         json.dumps(state.theory[0]['config']['skill_config'],
                                    default=str))

    def test_a_missing_training_file_is_refused_before_any_folder(self):
        state = _nn_task(train_data='')
        with self.assertRaises(ValueError) as caught:
            self._context(state)
        self.assertIn('train_data', str(caught.exception))
        self.assertFalse(os.path.isdir(
            os.path.join(self.dir, 'grid', 'NN-grid_unit-depth=2')))

    def test_a_multi_skill_plan_is_refused(self):
        state = _nn_task(train_data=self.data)
        other = dict(state.theory[0])
        other['skill_name'] = 'other_skill'
        other['task_id'] = 'other'
        other['static_id'] = 'qm_layer1_other'
        other['config'] = {'skill_config': {'method': 'hf'}, 'results': {}}
        state.add_task('theory', other)
        with self.assertRaises(ValueError) as caught:
            self._context(state)
        self.assertIn('several skills', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
