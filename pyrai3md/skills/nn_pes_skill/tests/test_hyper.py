"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 03 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Unit tests of the hyperparameter vocabulary: how one role section turns
into one hyperparameter dictionary per ensemble member.

The member axis is the outer one in every carrier, which is what these
tests are about. They need no TensorFlow: the expansion is pure text and
dict work, and the defaults dict is written out here rather than taken
from the container.

"""

import unittest

from pyrai3md.skills.nn_pes_skill.schema.hyper import (HYPER_KEYWORDS,
                                                       build_member_hyper,
                                                       member_groups,
                                                       member_list)

# The shape the container's default dictionary has, trimmed to the keys
# these tests read. What matters is that it is the same nesting the real
# default carries, because the writer addresses it by dotted path.
DEFAULTS = {
    'model': {
        'atoms': 3,
        'states': 2,
        'nn_size': 100,
        'depth': 3,
        'activ': {'class_name': 'leaky_softplus', 'config': {'alpha': 0.03}},
        'use_dropout': False,
        'dropout': 0.005,
        'use_reg_activ': None,
        'use_reg_weight': None,
        'use_reg_bias': None,
        'invd_index': True,
        'angle_index': [],
        'dihed_index': [],
    },
    'training': {
        'epo': 3000,
        'batch_size': 64,
        'epostep': 10,
        'val_split': 0.1,
        'val_disjoint': True,
        'learning_rate': 1e-3,
        'initialize_weights': True,
        'normalization_mode': 1,
        'loss_weights': [1, 10],
        'energy_only': False,
        'auto_scaling': {'x_mean': True, 'x_std': True,
                         'energy_std': True, 'energy_mean': True},
        'step_callback': {'use': False, 'epoch_step_reduction': [500],
                          'learning_rate_step': [1e-3]},
        'linear_callback': {'use': False, 'learning_rate_start': 1e-3,
                            'learning_rate_stop': 1e-6, 'epomin': 100,
                            'epo': 1000},
        'early_callback': {'use': False, 'epomin': 5000, 'patience': 600,
                           'max_time': 600, 'delta_loss': 1e-5,
                           'loss_monitor': 'val_loss', 'factor_lr': 0.1,
                           'learning_rate_start': 1e-3,
                           'learning_rate_stop': 1e-6, 'epostep': 1},
        'exp_callback': {'use': False, 'factor_lr': 0.1, 'epomin': 100,
                         'learning_rate_start': 1e-3},
    },
    'retraining': {
        'epo': 1000,
        'batch_size': 64,
        'epostep': 10,
        'val_split': 0.1,
        'loss_weights': [1, 10],
        'initialize_weights': False,
    },
    'predict': {'batch_size_predict': 265, 'try_predict_hessian': False},
    'plots': {'unit_energy': 'eV', 'unit_gradient': 'eV/A'},
}

# The same shape with no loss weights in its retraining section, which is
# how the gradient-only family is written.
DEFAULTS_NO_RETRAIN_LOSS = {
    **DEFAULTS,
    'retraining': {key: value for key, value in DEFAULTS['retraining'].items()
                   if key != 'loss_weights'},
}


class TestMemberAxis(unittest.TestCase):

    def test_one_group_serves_every_member(self):
        built = build_member_hyper('eg', DEFAULTS, 2, {'epo': '30'})
        self.assertEqual([m['training']['epo'] for m in built], [30, 30])

    def test_semicolon_splits_members(self):
        built = build_member_hyper('eg', DEFAULTS, 2, {'epo': '30;20'})
        self.assertEqual([m['training']['epo'] for m in built], [30, 20])

    def test_a_yaml_list_is_the_same_member_axis(self):
        text = build_member_hyper('eg', DEFAULTS, 2, {'epo': '30;20'})
        typed = build_member_hyper('eg', DEFAULTS, 2, {'epo': [30, 20]})
        self.assertEqual([m['training']['epo'] for m in text],
                         [m['training']['epo'] for m in typed])

    def test_surplus_groups_are_ignored(self):
        # mult_nn decides how many members are read, so writing a third
        # group costs nothing and changes nothing.
        built = build_member_hyper('eg', DEFAULTS, 2, {'epo': '30;20;10'})
        self.assertEqual([m['training']['epo'] for m in built], [30, 20])

    def test_partial_member_coverage_is_refused(self):
        with self.assertRaises(ValueError):
            build_member_hyper('eg', DEFAULTS, 3, {'epo': '30;20'})

    def test_a_list_value_nests_one_level_deeper_per_member(self):
        # A flat list is one group: the same weights for every member.
        flat = build_member_hyper('eg', DEFAULTS, 2,
                                  {'loss_weights': '1 10'})
        self.assertEqual([m['training']['loss_weights'] for m in flat],
                         [[1, 10], [1, 10]])
        # Nested, each inner list is one member's value.
        nested = build_member_hyper('eg', DEFAULTS, 2,
                                    {'loss_weights': [[1, 1], [1, 10]]})
        self.assertEqual([m['training']['loss_weights'] for m in nested],
                         [[1, 1], [1, 10]])
        # The text spelling of the same thing uses both separators: a
        # space inside a group, ';' between members.
        text = build_member_hyper('eg', DEFAULTS, 2,
                                  {'loss_weights': '1 1;1 10'})
        self.assertEqual([m['training']['loss_weights'] for m in text],
                         [[1, 1], [1, 10]])

    def test_nn_size_stays_scalar(self):
        # nn_size is the width of every layer, so per member it is one
        # number. A list here would be a different keyword, not this one.
        built = build_member_hyper('eg', DEFAULTS, 2, {'nn_size': '60;100'})
        self.assertEqual([m['model']['nn_size'] for m in built], [60, 100])
        with self.assertRaises(ValueError):
            build_member_hyper('eg', DEFAULTS, 2, {'nn_size': [[60], [100]]})

    def test_an_index_table_keeps_its_own_row_shape(self):
        # A comma ends a row, a space separates the indices in it, and ';'
        # separates members. A row is not judged by another table's shape:
        # inverse distances are pairs, angles are triples.
        built = build_member_hyper('eg', DEFAULTS, 2,
                                   {'angle_index': '1 2 3;4 5 6'})
        self.assertEqual([m['model']['angle_index'] for m in built],
                         [[[1, 2, 3]], [[4, 5, 6]]])

    def test_a_row_of_indices_reads_the_same_in_every_carrier(self):
        text = build_member_hyper('eg', DEFAULTS, 1,
                                  {'invd_index': '1 0, 2 0, 2 1'})
        typed = build_member_hyper('eg', DEFAULTS, 1,
                                   {'invd_index': [[[1, 0], [2, 0], [2, 1]]]})
        self.assertEqual(text[0]['model']['invd_index'],
                         typed[0]['model']['invd_index'])
        self.assertEqual(text[0]['model']['invd_index'],
                         [[1, 0], [2, 0], [2, 1]])


class TestDerivedValues(unittest.TestCase):

    def test_activation_is_assembled_from_name_and_alpha(self):
        built = build_member_hyper('eg', DEFAULTS, 1,
                                   {'activ': 'tanh', 'activ_alpha': '0.1'})
        self.assertEqual(built[0]['model']['activ'],
                         {'class_name': 'tanh', 'config': {'alpha': 0.1}})

    def test_regularizer_switch_expands_with_the_factors(self):
        built = build_member_hyper('eg', DEFAULTS, 1,
                                   {'use_reg_weight': 'l1', 'reg_l1': '1e-4'})
        self.assertEqual(built[0]['model']['use_reg_weight'],
                         {'class_name': 'L1', 'config': {'l1': 1e-4}})

    def test_regularizer_none_clears_the_switch(self):
        built = build_member_hyper('eg', DEFAULTS, 1, {'use_reg_bias': 'none'})
        self.assertIsNone(built[0]['model']['use_reg_bias'])

    def test_l1_l2_uses_both_factors(self):
        built = build_member_hyper(
            'eg', DEFAULTS, 1,
            {'use_reg_activ': 'l1_l2', 'reg_l1': '1e-5', 'reg_l2': '2e-5'})
        self.assertEqual(built[0]['model']['use_reg_activ'],
                         {'class_name': 'L1_L2',
                          'config': {'l1': 1e-5, 'l2': 2e-5}})

    def test_a_regulariser_may_differ_between_members(self):
        built = build_member_hyper(
            'eg', DEFAULTS, 2,
            {'use_reg_weight': 'l1;none', 'reg_l1': '1e-4'})
        self.assertEqual(built[0]['model']['use_reg_weight']['class_name'],
                         'L1')
        self.assertIsNone(built[1]['model']['use_reg_weight'])

    def test_booleans_accept_the_classic_spellings(self):
        built = build_member_hyper(
            'eg', DEFAULTS, 2, {'use_dropout': 'yes;0', 'dropout': '0.1'})
        self.assertEqual([m['model']['use_dropout'] for m in built],
                         [True, False])
        self.assertEqual([m['model']['dropout'] for m in built], [0.1, 0.1])

    def test_callback_switches_reach_their_nested_dict(self):
        built = build_member_hyper('eg', DEFAULTS, 1,
                                   {'use_step_callback': 'true',
                                    'epoch_step_reduction': '100 200',
                                    'learning_rate_step': '1e-3 1e-5'})
        step = built[0]['training']['step_callback']
        self.assertTrue(step['use'])
        self.assertEqual(step['epoch_step_reduction'], [100, 200])
        self.assertEqual(step['learning_rate_step'], [1e-3, 1e-5])


class TestRetrainingMirror(unittest.TestCase):

    def test_a_spelled_keyword_applies_to_both_modes(self):
        # The container reads 'training' or 'retraining' depending on the
        # fit mode, so spelling epo once must not mean the other mode
        # keeps the shipped default.
        built = build_member_hyper('eg', DEFAULTS, 1, {'epo': '40'})
        self.assertEqual(built[0]['training']['epo'], 40)
        self.assertEqual(built[0]['retraining']['epo'], 40)

    def test_untouched_defaults_stay_default(self):
        built = build_member_hyper('eg', DEFAULTS, 1, {'epo': '40'})
        self.assertEqual(built[0]['retraining']['batch_size'], 64)

    def test_a_section_without_the_slot_does_not_invent_one(self):
        built = build_member_hyper('eg', DEFAULTS_NO_RETRAIN_LOSS, 1,
                                   {'loss_weights': '1 1'})
        self.assertEqual(built[0]['training']['loss_weights'], [1, 1])
        self.assertNotIn('loss_weights', built[0]['retraining'])


class TestVocabulary(unittest.TestCase):

    def test_every_keyword_names_a_known_kind(self):
        for keyword, (kind, _target) in HYPER_KEYWORDS.items():
            self.assertIn(kind, ('scalar', 'list', 'pairs', 'bool', 'reg',
                                 'text'), keyword)

    def test_unknown_keyword_is_refused_with_the_list(self):
        with self.assertRaises(ValueError) as caught:
            build_member_hyper('eg', DEFAULTS, 1, {'nonsense': 1})
        self.assertIn('unknown hyperparameter in &eg', str(caught.exception))

    def test_the_classic_names_of_the_original_project_are_kept(self):
        # Renaming these would make an old input file unreadable, which is
        # the one thing a compatibility port must not do.
        for keyword in ('depth', 'nn_size', 'activ', 'activ_alpha',
                        'loss_weights', 'use_reg_weight', 'reg_l1',
                        'use_dropout', 'dropout', 'epo', 'batch_size',
                        'epostep', 'learning_rate', 'phase_less_loss',
                        'pre_epo', 'scale_x_mean', 'scale_y_std',
                        'angle_index', 'dihed_index', 'invd_index'):
            self.assertIn(keyword, HYPER_KEYWORDS)


class TestSharedMemberList(unittest.TestCase):

    def test_device_indices_split_by_member(self):
        self.assertEqual(member_list('0;1', 2, 'gpu'), [0, 1])

    def test_one_device_broadcasts(self):
        self.assertEqual(member_list('-1', 3, 'gpu'), [-1, -1, -1])

    def test_a_typed_list_is_the_same_axis(self):
        self.assertEqual(member_list([0, 1], 2, 'gpu'), [0, 1])

    def test_member_groups_reads_the_separator_rule(self):
        self.assertEqual(member_groups('30;20', 'scalar'), ['30', '20'])
        self.assertEqual(member_groups('30', 'scalar'), ['30'])
        self.assertEqual(member_groups(30, 'scalar'), [30])


if __name__ == '__main__':
    unittest.main()
