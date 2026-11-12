"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Unit tests of the query interfaces: manifest metadata, the profile
table, the defaults vocabulary the framework filters unknown keys with,
and resolve_profile including token order, duplicates and the driver
guard.

These tests need no TensorFlow.

"""

import unittest

from pyrai3md.skills.nn_pes_skill import query
from pyrai3md.skills.nn_pes_skill.schema.config import FIELD_TO_SUBMODEL
from pyrai3md.skills.nn_pes_skill.schema.method import (
    PROFILE_METHOD_MAP, PROFILE_ENSEMBLES, METHODS)


class TestMetadata(unittest.TestCase):

    def test_name_is_the_reserved_ml_label(self):
        # 'nn' is the machine-learning theory label of the framework
        self.assertEqual(query.get_name(), 'nn')

    def test_metadata_fields(self):
        meta = query.get_metadata()
        self.assertEqual(meta['name'], 'nn')
        self.assertEqual(meta['skill_path'], 'backend')
        self.assertEqual(meta['backend_class'], 'NNPESBackend')
        self.assertFalse(meta['supports_pbc'])

    def test_metadata_declares_its_input_subsections(self):
        # The framework folds a declared block into the skill config and
        # renders it back out of a yaml; it can only do that if the block
        # names are visible here instead of hardcoded upstream.
        # One block per network family the skill can fit or serve: the
        # energy+gradient one, the coupling one, and the spin-orbit one.
        self.assertEqual(query.get_metadata()['subsections'],
                         ['eg', 'nac', 'soc'])
        self.assertEqual(query.get_subsections(), ['eg', 'nac', 'soc'])

    def test_declared_units_are_the_ones_the_code_uses(self):
        # The manifest is a promise to a reader; schema/reference.py is what
        # the trainer and the summary actually convert with. Drift between
        # them would make the printed unit a lie about the number next to it.
        from pyrai3md.skills.nn_pes_skill.schema import reference
        units = query.get_metadata()['units']
        self.assertEqual(units['label']['energy'],
                         reference.LABEL_ENERGY_UNIT)
        self.assertEqual(units['label']['gradient'],
                         reference.LABEL_GRADIENT_UNIT)
        self.assertEqual(units['label']['soc'], reference.LABEL_SOC_UNIT)
        self.assertEqual(units['metric']['energy'],
                         reference.METRIC_ENERGY_UNIT)
        self.assertEqual(units['metric']['gradient'],
                         reference.METRIC_GRADIENT_UNIT)
        self.assertEqual(units['metric']['soc'], reference.METRIC_SOC_UNIT)
        self.assertEqual(units['gradient'], reference.OUTPUT_GRADIENT_UNIT)

    def test_metadata_declares_tensorflow(self):
        # The planner reports a TensorFlow-free machine as skipped only
        # if the requirement is visible through the query interface.
        self.assertIn('tensorflow', query.get_metadata()['requires'])


class TestProfiles(unittest.TestCase):

    def test_every_manifest_profile_is_resolvable(self):
        declared = set(query.list_profiles())
        self.assertEqual(declared, set(PROFILE_METHOD_MAP))

    def test_every_profile_names_a_known_method(self):
        for profile, method in PROFILE_METHOD_MAP.items():
            self.assertIn(method, METHODS, profile)

    def test_every_profile_declares_ensembles(self):
        for profile in PROFILE_METHOD_MAP:
            self.assertIn(profile, PROFILE_ENSEMBLES, profile)
            self.assertTrue(len(PROFILE_ENSEMBLES[profile]) >= 1)

    def test_profile_entries_match_manifest(self):
        for name, entry in query.get_profiles().items():
            self.assertEqual(entry['method'], PROFILE_METHOD_MAP[name])

    def test_get_profile_unknown(self):
        self.assertIsNone(query.get_profile('mlp_eg_hessian'))


class TestDefaults(unittest.TestCase):
    """The defaults dict is the keyword vocabulary of the skill."""

    def test_defaults_cover_the_schema(self):
        defaults = query.get_defaults()
        self.assertTrue(len(defaults) > 0)
        # Fields the caller never sets (geometry and result channels) are
        # excluded; anything else missing would be dropped by the config
        # builder before the backend could see it.
        not_settable = {'atoms', 'coordinates', 'velocities', 'point_charges',
                        'mass', 'comment', 'skill_name', 'skill_path',
                        'energy', 'gradient', 'nac', 'soc', 'fin_coord',
                        'fin_velocity', 'state_label', 'state_spin',
                        'nac_pair_map', 'soc_pair_map', 'transition_pair',
                        'natom', 'npair', 'nsoc', 'nnac', 'nstate',
                        'current_state'}
        missing = set(FIELD_TO_SUBMODEL) - not_settable - set(defaults)
        self.assertEqual(missing, set(),
                         'fields missing from DEFAULT: %s' % sorted(missing))

    def test_defaults_have_no_unknown_keys(self):
        for key in query.get_defaults():
            self.assertIn(key, FIELD_TO_SUBMODEL,
                          'DEFAULT key %r is not a schema field' % key)


class TestResolveProfile(unittest.TestCase):

    def test_energy_gradient_order_invariant(self):
        self.assertEqual(query.resolve_profile('mlp_eg', 'gradient energy'),
                         'mlp_eg_gradient')

    def test_duplicate_tokens_collapsed(self):
        self.assertEqual(query.resolve_profile('mlp_eg', 'energy energy'),
                         'mlp_eg_energy')

    def test_md_maps_to_the_superset(self):
        self.assertEqual(query.resolve_profile('eg_nac', 'md'), 'eg_nac_nac')

    def test_train_jobtype(self):
        self.assertEqual(query.resolve_profile('mlp_eg', 'train'),
                         'mlp_eg_train')

    def test_method_without_the_capability_is_rejected(self):
        # An energy network has no gradient head; a coupling network has
        # no energy head. Both must be refused rather than resolved into
        # a profile that cannot answer.
        with self.assertRaises(ValueError):
            query.resolve_profile('mlp_e', 'energy gradient')
        with self.assertRaises(ValueError):
            query.resolve_profile('mlp_nac2', 'energy')

    def test_skill_driver_is_rejected(self):
        with self.assertRaises(ValueError):
            query.resolve_profile('mlp_eg', 'md', driver='skill')

    def test_an_optimize_request_under_a_skill_driver_is_refused(self):
        # A network potential never searches a surface, and there is no
        # optimize row here either. The refusal has to name the missing half
        # rather than leave the default serving profile to be read as a
        # finished search.
        with self.assertRaises(ValueError) as caught:
            query.resolve_profile('mlp_eg', 'optimize', driver='skill')
        message = str(caught.exception)
        self.assertIn('optimize', message)
        self.assertIn('skill', message)
        # The arm that keeps this from becoming a blanket driver refusal:
        # a property request under the same driver resolves as usual.
        self.assertEqual(query.resolve_profile('mlp_eg', 'energy gradient',
                                               driver='skill'),
                         'mlp_eg_gradient')

    def test_unknown_method_and_jobtype(self):
        with self.assertRaises(ValueError):
            query.resolve_profile('mlp_nothing', 'energy')
        with self.assertRaises(ValueError):
            query.resolve_profile('mlp_eg', 'hessian')


if __name__ == '__main__':
    unittest.main()
