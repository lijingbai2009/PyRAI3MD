"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Unit tests of the searchable declaration a hyperparameter scan reads.

Three things are pinned here, and none of them needs TensorFlow. Every
declared keyword must exist in the skill's own hyperparameter vocabulary,
because a manifest that drifts from it would hand a scan a name the
trainer refuses mid-fit. The kind of each keyword must come from that
vocabulary rather than from the manifest, because the kind is what tells
a scan whether a list means several members or several candidates. And
the candidate checks must refuse the shapes that cannot be trained.

"""

import unittest

from pyrai3md.skills.nn_pes_skill import query
from pyrai3md.skills.nn_pes_skill.schema.hyper import (HYPER_KEYWORDS,
                                                       REG_STYLES)
from pyrai3md.skills.nn_pes_skill.schema.method import (PROFILE_ENSEMBLES,
                                                        ROLE_SECTIONS)


class TestSearchableVocabulary(unittest.TestCase):
    """The declared surface and the vocabulary it must match."""

    def setUp(self):
        self.manifest = query.load_manifest()
        self.searchable = query.get_searchable()

    def test_every_declared_keyword_exists_in_the_vocabulary(self):
        undeclared = sorted(set(self.manifest['searchable'])
                            - set(HYPER_KEYWORDS))
        self.assertEqual([], undeclared,
                         "manifest declares names the trainer does not know")

    def test_kind_is_read_from_the_vocabulary_not_the_manifest(self):
        for keyword, entry in self.searchable.items():
            with self.subTest(keyword=keyword):
                self.assertEqual(HYPER_KEYWORDS[keyword][0], entry['kind'])
                self.assertNotIn('kind', self.manifest['searchable'][keyword],
                                 "a manifest copy of kind would drift")

    def test_manifest_type_matches_the_declared_entries(self):
        for keyword, info in self.manifest['searchable'].items():
            self.assertEqual(info['type'], self.searchable[keyword]['type'])

    def test_choice_declares_its_alternatives(self):
        for keyword, entry in self.searchable.items():
            if entry['type'] != 'choice':
                self.assertEqual([], entry['values'])
                continue
            self.assertTrue(entry['values'])
            if entry['kind'] == 'reg':
                self.assertEqual(sorted(REG_STYLES),
                                 sorted(entry['values']))

    def test_surface_is_reachable_through_get_metadata(self):
        meta = query.get_metadata()
        self.assertEqual(self.searchable, meta['searchable'])
        self.assertEqual(query.get_search_targets(), meta['search_targets'])


class TestSearchTargets(unittest.TestCase):
    """What each trainable profile tells a scan to stamp."""

    def setUp(self):
        self.targets = query.get_search_targets()
        self.manifest = query.load_manifest()

    def test_only_train_profiles_are_targets(self):
        train = sorted(name for name, info in self.manifest['profiles'].items()
                       if (info or {}).get('runtype') == 'train')
        self.assertEqual(train, sorted(self.targets))

    def test_every_target_is_declared_in_the_ensemble_map(self):
        for name in self.targets:
            self.assertIn(name, PROFILE_ENSEMBLES)

    def test_sections_follow_the_ensembles_the_profile_trains(self):
        for name, entry in self.targets.items():
            expected = [ROLE_SECTIONS[folder]
                        for folder, _kind in PROFILE_ENSEMBLES[name]]
            self.assertEqual(expected, entry['sections'], name)

    def test_channels_are_one_per_ensemble(self):
        for name, entry in self.targets.items():
            self.assertEqual(len(PROFILE_ENSEMBLES[name]), len(entry['channels']),
                             name)

    def test_method_names_a_family_the_profile_belongs_to(self):
        for name, entry in self.targets.items():
            self.assertTrue(name.startswith(entry['method']),
                            '%s is not a %s profile' % (name, entry['method']))

    def test_a_scan_jobtype_resolves_to_a_trainable_profile(self):
        # The run that plans the scan carries a training profile: pointing
        # it at a serving profile would make the parent try to load a
        # surface no point has fitted yet.
        for name, entry in self.targets.items():
            resolved = query.resolve_profile(entry['method'], 'search')
            self.assertIn(resolved, self.targets)
            self.assertEqual(entry['method'], self.targets[resolved]['method'])


class TestCheckSearchable(unittest.TestCase):
    """The candidate rules a scan inherits from the skill."""

    def test_undeclared_keyword_is_named_with_the_alternatives(self):
        with self.assertRaises(ValueError) as caught:
            query.check_searchable('not_a_keyword', 1)
        message = str(caught.exception)
        self.assertIn('not_a_keyword', message)
        self.assertIn('depth', message)

    def test_scalar_keyword_refuses_a_group(self):
        with self.assertRaises(ValueError) as caught:
            query.check_searchable('dropout', [0.01, 0.05])
        self.assertIn('member group', str(caught.exception))

    def test_list_keyword_refuses_a_scalar(self):
        with self.assertRaises(ValueError) as caught:
            query.check_searchable('loss_weights', 10)
        self.assertIn('must be a list', str(caught.exception))

    def test_int_keyword_refuses_a_fraction(self):
        with self.assertRaises(ValueError):
            query.check_searchable('depth', 2.5)

    def test_bool_keyword_refuses_a_string(self):
        with self.assertRaises(ValueError):
            query.check_searchable('use_dropout', 'yes')

    def test_bounds_of_the_vocabulary_apply_to_candidates(self):
        with self.assertRaises(ValueError):
            query.check_searchable('nn_size', 0)
        with self.assertRaises(ValueError):
            query.check_searchable('val_split', 1.5)

    def test_choice_keyword_refuses_an_undeclared_alternative(self):
        with self.assertRaises(ValueError) as caught:
            query.check_searchable('use_reg_weight', 'l3')
        self.assertIn('accepts', str(caught.exception))

    def test_admitted_candidates_come_back_unchanged(self):
        for keyword, value in (('depth', 3), ('nn_size', 60), ('dropout', 0.01),
                               ('val_split', 0.2), ('use_dropout', True),
                               ('activ', 'leaky_softplus'),
                               ('use_reg_weight', 'l1'),
                               ('loss_weights', [1, 10])):
            self.assertEqual(value, query.check_searchable(keyword, value))


class TestDiscoveryNeedsNoTensorFlow(unittest.TestCase):
    """The whole scan surface is readable without the optional stack."""

    def test_query_modules_never_import_tensorflow(self):
        import subprocess
        import sys

        code = (
            'import sys;'
            'from pyrai3md.skills.nn_pes_skill import query;'
            'query.get_searchable(); query.get_search_targets();'
            'query.check_searchable("depth", 2);'
            'print("tensorflow" in sys.modules)'
        )
        done = subprocess.run([sys.executable, '-c', code],
                              capture_output=True, text=True)
        self.assertEqual(0, done.returncode, done.stderr[-400:])
        self.assertEqual('False', done.stdout.strip().splitlines()[-1])


if __name__ == '__main__':
    unittest.main()
