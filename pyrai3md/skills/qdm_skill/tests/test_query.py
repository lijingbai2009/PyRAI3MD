"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 24, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Unit tests for the query interfaces: manifest metadata, profile
listing, defaults exposure, and the resolve_profile token mapping
(including order/duplicate invariance and the driver split on
jobtype 'md').

"""

import unittest

from pyrai3md.skills.qdm_skill import query
from pyrai3md.skills.qdm_skill.schema.method import PROFILE_METHOD_MAP


class TestManifestInterfaces(unittest.TestCase):

    def test_skill_name(self):
        self.assertEqual(query.get_name(), 'qdm')

    def test_metadata(self):
        metadata = query.get_metadata()
        self.assertEqual(metadata['name'], 'qdm')
        self.assertEqual(metadata['backend_class'], 'QDMBackend')

    def test_roles(self):
        roles = query.get_roles()
        self.assertIn('qm', roles)

    def test_capabilities(self):
        capabilities = set(query.get_capabilities())
        self.assertEqual(capabilities,
                         {'energy', 'gradient', 'nac', 'md'})

    def test_profile_listing(self):
        profiles = query.list_profiles()
        self.assertEqual(len(profiles), 24)
        self.assertEqual(profiles, sorted(PROFILE_METHOD_MAP))
        for method in ['tully1', 'tully2', 'tully3',
                       'lz1', 'lz2', 'lz3']:
            for suffix in ['_energy', '_gradient',
                           '_nac', '_reference']:
                self.assertIn(method + suffix, profiles)

    def test_get_profile_lookup(self):
        info = query.get_profile('tully1_nac')
        self.assertIsNotNone(info)
        self.assertIsNone(query.get_profile('not_a_profile'))

    def test_registry_shape(self):
        registry = query.get_registry()
        self.assertIn('qdm', registry)
        entry = registry['qdm']
        self.assertEqual(entry['backend_class'], 'QDMBackend')
        self.assertEqual(len(entry['tasks']), 24)

    def test_defaults_expose_model_parameters(self):
        defaults = query.get_defaults()
        self.assertEqual(defaults['model_mass'], 2000.0)
        self.assertEqual(defaults['method'], 'tully1')
        self.assertIsNone(defaults['tully_a'])
        self.assertIsNone(defaults['tully_e'])
        self.assertIsNone(defaults['lz_a'])
        self.assertIsNone(defaults['lz_b'])
        self.assertEqual(defaults['ref_ngrid'], 2048)


class TestResolveProfile(unittest.TestCase):

    def test_basic_jobtypes(self):
        self.assertEqual(query.resolve_profile('tully1', 'energy'),
                         'tully1_energy')
        self.assertEqual(query.resolve_profile('tully1',
                                               'energy gradient'),
                         'tully1_gradient')
        self.assertEqual(
            query.resolve_profile('tully1', 'energy gradient nac'),
            'tully1_nac')

    def test_token_order_and_duplicates_invariant(self):
        expected = 'tully1_nac'
        for jobtype in ['nac gradient energy', 'energy nac gradient',
                        'nac nac energy gradient']:
            self.assertEqual(
                query.resolve_profile('tully1', jobtype), expected)

    def test_md_driver_main(self):
        self.assertEqual(query.resolve_profile('tully1', 'md'),
                         'tully1_nac')
        self.assertEqual(query.resolve_profile('tully1', 'md',
                                               driver='main'),
                         'tully1_nac')

    def test_md_driver_skill(self):
        self.assertEqual(query.resolve_profile('tully1', 'md',
                                               driver='skill'),
                         'tully1_reference')

    def test_other_methods(self):
        self.assertEqual(query.resolve_profile('tully2',
                                               'energy gradient'),
                         'tully2_gradient')
        self.assertEqual(query.resolve_profile('tully3', 'md',
                                               driver='skill'),
                         'tully3_reference')

    def test_lz1_profiles(self):
        self.assertEqual(query.resolve_profile('lz1', 'energy'),
                         'lz1_energy')
        self.assertEqual(query.resolve_profile('lz1',
                                               'energy gradient'),
                         'lz1_gradient')
        self.assertEqual(query.resolve_profile('lz1',
                                               'energy gradient nac'),
                         'lz1_nac')
        self.assertEqual(query.resolve_profile('lz1', 'md',
                                               driver='main'),
                         'lz1_nac')
        self.assertEqual(query.resolve_profile('lz1', 'md',
                                               driver='skill'),
                         'lz1_reference')

    def test_lz2_lz3_profiles(self):
        for method in ('lz2', 'lz3'):
            self.assertEqual(query.resolve_profile(method, 'energy'),
                             method + '_energy')
            self.assertEqual(query.resolve_profile(method,
                                                   'energy gradient'),
                             method + '_gradient')
            self.assertEqual(query.resolve_profile(
                method, 'energy gradient nac'), method + '_nac')
            self.assertEqual(query.resolve_profile(method, 'md',
                                                   driver='main'),
                             method + '_nac')
            self.assertEqual(query.resolve_profile(method, 'md',
                                                   driver='skill'),
                             method + '_reference')

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            query.resolve_profile('tully9', 'energy')

    def test_unknown_jobtype_raises(self):
        with self.assertRaises(ValueError):
            query.resolve_profile('tully1', 'hessian')

    def test_an_optimize_request_under_a_skill_driver_is_refused(self):
        # These models propagate nothing themselves except 'md' (one
        # '{method}_reference' call), and there is no optimize row for them
        # at all: under the driver that scans the surface the request has no
        # profile to land on, and the refusal must say which half is missing
        # rather than leave a single point to be read as a search.
        with self.assertRaises(ValueError) as caught:
            query.resolve_profile('tully1', 'optimize', driver='skill')
        message = str(caught.exception)
        self.assertIn('optimize', message)
        self.assertIn('skill', message)
        with self.assertRaises(ValueError):
            query.resolve_profile('tully1', 'surface_hopping',
                                  driver='skill')
        # The arm that keeps the guard from being a blanket refusal of this
        # driver: 'md' is the one iteration these models own, and it stays.
        self.assertEqual(query.resolve_profile('tully1', 'md',
                                               driver='skill'),
                         'tully1_reference')


if __name__ == '__main__':
    unittest.main()
