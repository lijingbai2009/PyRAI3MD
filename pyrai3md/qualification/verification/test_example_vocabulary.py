"""
======================================================================
This module locks the example vocabulary to the examples directory.

The plan used to carry three hand lists: an exhaustive
QUALIFIED_EXAMPLES tuple (so a new YAML was planned as postponed until
a Python edit added it), a PATH_EXAMPLES tuple that restated what the
`path_` prefix already said, and per-family prefix tests scattered over
tests and tools. Membership is now read from disk, the postponement
policy is an exception map, and a family is the name prefix the switch
table declares.

The criterion is the one the list could not meet: adding a YAML to the
examples directory adds its two planned modes with no Python edit.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import shutil
import tempfile
import unittest

from pyrai3md.qualification import manager as manager_module
from pyrai3md.qualification.manager import (
    EXAMPLE_FAMILIES, POSTPONED_EXAMPLES, family_members, shipped_examples)
from pyrai3md.qualification.verification._source_paths import source_read

SHIPPED = os.path.join(manager_module.examples_directory(), 'energy.yaml')
CONFIG = {'_runall': 'condition', 'examples': {}}


class TestThePlanFollowsTheDirectory(unittest.TestCase):
    """A new YAML joins the plan without a Python edit."""

    def _plan(self, directory):
        return manager_module._plan_example_tests(
            CONFIG, os.path.join(directory, 'work'), examples_dir=directory)

    def test_a_new_example_adds_its_two_modes(self):
        with tempfile.TemporaryDirectory() as root:
            shutil.copy(SHIPPED, os.path.join(root, 'energy.yaml'))
            before = self._plan(root)
            shutil.copy(SHIPPED, os.path.join(root, 'energy_copy.yaml'))
            after = self._plan(root)
            self.assertEqual(len(after), len(before) + 2)
            self.assertEqual(
                {task['name'] for task in after} - {task['name']
                                                    for task in before},
                {'example_energy_copy_yaml', 'example_energy_copy_input'})

    def test_the_shipped_list_is_the_directory(self):
        directory = manager_module.examples_directory()
        stems = sorted(os.path.splitext(name)[0]
                       for name in os.listdir(directory)
                       if name.endswith('.yaml'))
        self.assertEqual(shipped_examples(), stems)

    def test_a_family_is_its_declared_prefix(self):
        shipped = set(shipped_examples())
        for prefix in EXAMPLE_FAMILIES.values():
            members = family_members(prefix)
            self.assertTrue(members, prefix)
            self.assertTrue(set(members) <= shipped)
            self.assertEqual(members,
                             sorted(name for name in shipped
                                    if name.startswith(prefix)))

    def test_the_postponement_policy_is_an_exception_map(self):
        self.assertEqual(POSTPONED_EXAMPLES, {})
        text = source_read('pyrai3md/qualification/manager.py')
        self.assertNotIn('QUALIFIED_EXAMPLES', text)
        self.assertNotIn('PATH_EXAMPLES', text)
        # No per-example name literal: the only names in the module are the
        # family switches and the chains.
        for stem in shipped_examples()[:40]:
            self.assertNotIn("'%s'" % stem, text)


class TestOnePrefixRule(unittest.TestCase):
    """Family membership is decided by the switch table, once."""

    def test_the_tools_read_the_family_prefix(self):
        text = source_read('tools/refresh_water_fixture.py')
        self.assertNotIn("startswith('md_lammps_')", text)
        self.assertIn('EXAMPLE_FAMILIES', text)

    def test_the_deploy_help_names_the_family_table(self):
        text = source_read('tools/deploy.py')
        self.assertNotIn('examples.constraint_md_test', text)
        self.assertIn('EXAMPLE_FAMILIES', text)


if __name__ == '__main__':
    unittest.main()