"""
======================================================================
Tests that a qualification switch means one thing at every level.

The three-valued switch (`true` / `false` / `condition`) and the
two-valued leaf were parsed in four places: `manager._coerce_switch`,
`manager._coerce_flag`, `manager._example_state` (which walked its own
chain instead of `_resolve_chain`) and the estimation fallback in
`tools/deploy.py`, which typed the token list out a fourth time and
handled a package it could not import by guessing. The keys a planner
alone reads (`test`, `tests`, `runtime_tests`) were likewise listed in
the planner while the example staging filtered only `test`, so a
`tests:` case map could be injected into a generated skill config that
the planner's own comment says must never see it.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import ast
import json
import os
import shutil
import tempfile
import unittest
import warnings
from unittest import mock

from pyrai3md.qualification import manager
from pyrai3md.qualification.certification import planner
from pyrai3md.qualification.verification._source_paths import (
    SOURCE_PARENT, source_tree)

TOKENS = ("'true', 'yes', 'on', '1'", "'false', 'no', 'off', '0'")


class TestSharedSwitchVocabulary(unittest.TestCase):

    def test_both_parsers_answer_the_same_tokens(self):
        for spelling in ('true', 'TRUE', 'Yes', 'on', '1'):
            with self.subTest(token=spelling):
                self.assertIs(manager._coerce_switch(spelling, 'x', False),
                              True)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    self.assertIs(
                        manager._coerce_flag(spelling, 'x', default=False),
                        True)
        for spelling in ('false', 'No', 'off', '0'):
            with self.subTest(token=spelling):
                self.assertIs(manager._coerce_switch(spelling, 'x', True),
                              False)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    self.assertIs(
                        manager._coerce_flag(spelling, 'x', default=True),
                        False)

    def test_a_typo_defers_the_same_way_in_both(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertIs(manager._coerce_switch('fasle', 'k', True), True)
            self.assertIs(manager._coerce_flag('fasle', 'k', True), True)
        self.assertEqual(len(caught), 2,
                         'both parsers must warn; a silent fallback is how '
                         'a typo closes or opens a whole family')

    def test_the_key_list_is_one_list_read_by_both_sides(self):
        self.assertIs(planner.QUALIFICATION_ONLY_KEYS,
                      manager.QUALIFICATION_ONLY_KEYS)
        offenders = []
        for path in source_tree('pyrai3md/**/*.py'):
            if os.path.basename(path).startswith('test_'):
                continue
            with open(path, 'r', encoding='utf-8') as handle:
                tree = ast.parse(handle.read(), filename=path)
            for node in tree.body:
                if (isinstance(node, ast.Assign)
                        and any(getattr(target, 'id', '')
                                == 'QUALIFICATION_ONLY_KEYS'
                                for target in node.targets)):
                    offenders.append(os.path.relpath(path, SOURCE_PARENT))
        self.assertEqual(offenders,
                         [os.path.join('pyrai3md', 'qualification',
                                       'manager.py')], offenders)

    def test_the_example_staging_keeps_planner_only_keys_out(self):
        directory = tempfile.mkdtemp(prefix='switch_keys_')
        self.addCleanup(shutil.rmtree, directory, True)
        config_path = os.path.join(directory, 'orca.json')
        with open(config_path, 'w', encoding='utf-8') as handle:
            json.dump({'theory': ['orca'], 'jobtype': 'sp'}, handle)
        skills = {'orca_skill': {'test': True, 'tests': {'a': False},
                                 'runtime_tests': True, 'thread': 4,
                                 'shell': 'module load orca'}}
        manager._inject_example_skill_env(config_path, skills)
        with open(config_path, 'r', encoding='utf-8') as handle:
            written = json.load(handle)
        flat = written
        for forbidden in ('test', 'tests', 'runtime_tests'):
            self.assertNotIn(forbidden, flat, forbidden)
        self.assertEqual(flat.get('thread'), 4)
        self.assertEqual(flat.get('shell'), 'module load orca')

    def test_the_token_table_is_not_retyped_anywhere(self):
        # The qualification vocabulary is a tuple, not any list of words
        # that happens to share letters: `kernel/environment.py` keeps a
        # deliberately wider off-set for the slurm `exclusive` directive
        # ('none' and the empty string among them), and that is a
        # different rule, not a copy of this one.
        def _matches(path):
            with open(path, 'r', encoding='utf-8') as handle:
                tree = ast.parse(handle.read(), filename=path)
            found = []
            for node in ast.walk(tree):
                container = node
                if (isinstance(node, ast.Call)
                        and getattr(node.func, 'id', '')
                        in ('frozenset', 'set', 'tuple', 'list')
                        and len(node.args) == 1):
                    container = node.args[0]
                if not isinstance(container, (ast.Tuple, ast.List, ast.Set)):
                    continue
                values = tuple(getitem.value for getitem in container.elts
                               if isinstance(getitem, ast.Constant)
                               and isinstance(getitem.value, str))
                if values == manager.TRUE_TOKENS or \
                        values == manager.FALSE_TOKENS:
                    found.append((os.path.relpath(path, SOURCE_PARENT),
                                  node.lineno, values))
            return found

        copies = []
        for path in source_tree('pyrai3md/**/*.py'):
            if os.path.basename(path).startswith('test_'):
                continue
            if path.endswith(os.path.join('qualification', 'manager.py')):
                continue
            copies.extend(_matches(path))
        tools = os.path.join(SOURCE_PARENT, 'tools', 'deploy.py')
        if os.path.isfile(tools):
            copies.extend(_matches(tools))
        self.assertEqual(copies, [],
                         'the qualification token table was typed out '
                         'again: %s' % copies)

    def test_the_family_chain_is_the_shared_resolver(self):
        # `_example_state` defers through `_resolve_chain`, so a family
        # that says `condition` hands the decision down instead of
        # answering with its own label.
        states = {'md_lammps_test': manager.SWITCH_CONDITION}
        self.assertEqual(
            manager._example_state(manager.SWITCH_CONDITION, states,
                                   'md_lammps_npt_nh_pr'),
            (manager.SWITCH_CONDITION, ''))
        self.assertEqual(
            manager._example_state(True, states, 'md_lammps_npt_nh_pr'),
            (True, '_runall'))


if __name__ == '__main__':
    unittest.main()
