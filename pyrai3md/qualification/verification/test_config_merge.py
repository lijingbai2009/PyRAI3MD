"""
======================================================================
Tests for the shared configuration-layer merge (utilities/config_merge.py).

Four framework modules used to carry their own copy of "merge one layer
over another, an unset value falls through": `schema/adapter.py` twice
(`_deep_update` and `_deep_merge`, both mutating their base in place),
`kernel/environment.py` once and `qualification/manager.py` once (both
returning copies). Same rule, two contracts, and no test anywhere pinned
either: the sweep below found none. One implementation now answers all
four layers, and these cases are what keeps it that way.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import ast
import copy
import os
import unittest

from pyrai3md.qualification.verification._source_paths import (
    SOURCE_PARENT, source_tree)
from pyrai3md.kernel import environment
from pyrai3md.schema import adapter
from pyrai3md.utilities.config_merge import deep_merge, is_empty_value

# The layers a merge decision is read from in production: the environment
# package + user pair, the example/config layer pair the manager loads, and
# the runtime input over the environment defaults in the adapter.
BASE = {'slurm': {'partition': 'gpu', 'time': '10:00'},
        'skills': {'orca': {'test': True, 'shell': 'module load a'}},
        '_runall': 'condition'}
OVERRIDE = {'slurm': {'time': '', 'mem': '8G'},
            'skills': {'orca': {'test': False, 'thread': 4}},
            'environment': {'scratch_path': None}}


class TestSharedMerge(unittest.TestCase):

    def test_the_rule_is_one_rule(self):
        merged = deep_merge(BASE, OVERRIDE)
        self.assertEqual(merged['slurm'], {'partition': 'gpu', 'time': '10:00',
                                          'mem': '8G'},
                         'an unset override value keeps the base answer')
        self.assertEqual(merged['skills']['orca'],
                         {'test': False, 'shell': 'module load a',
                          'thread': 4},
                         'a written leaf wins, nested keys survive')
        # A section the base does not have arrives whole, including its
        # unset keys: the fall-through is per key, and only a base that
        # already answers the key can be fallen back on. This is the rule
        # all four former copies implemented.
        self.assertEqual(merged['environment'], {'scratch_path': None})
        answered = deep_merge({'environment': {'scratch_path': '/work'}},
                              OVERRIDE)
        self.assertEqual(answered['environment'],
                         {'scratch_path': '/work'})

    def test_the_arguments_are_never_touched(self):
        base = copy.deepcopy(BASE)
        override = copy.deepcopy(OVERRIDE)
        deep_merge(base, override)
        self.assertEqual(base, BASE)
        self.assertEqual(override, OVERRIDE)

    def test_a_merged_value_is_not_shared_with_either_layer(self):
        merged = deep_merge(BASE, {})
        merged['slurm']['partition'] = 'cpu'
        self.assertEqual(BASE['slurm']['partition'], 'gpu')

    def test_the_unset_vocabulary_is_the_documented_one(self):
        for unset in (None, '', [], {}):
            self.assertTrue(is_empty_value(unset), repr(unset))
        for answer in (False, 0, True, 1, 'false', [0], {'a': 0}):
            self.assertFalse(is_empty_value(answer), repr(answer))

    def test_the_layer_helpers_use_the_shared_merge(self):
        combined = environment._combine_configs(BASE, OVERRIDE)
        self.assertEqual(combined, deep_merge(BASE, OVERRIDE))
        # The adapter's runtime layer: the environment defaults with the
        # input's answers over them.
        self.assertEqual(adapter._apply_env_defaults(BASE, OVERRIDE),
                         deep_merge(BASE, OVERRIDE))
        # and it does not reach into the layer it was handed
        self.assertEqual(BASE, {'slurm': {'partition': 'gpu',
                                          'time': '10:00'},
                               'skills': {'orca': {'test': True,
                                                   'shell': 'module load a'}},
                               '_runall': 'condition'})

    def test_no_module_defines_a_second_merge(self):
        offenders = []
        owner = os.path.abspath(os.path.join(
            SOURCE_PARENT, 'pyrai3md', 'utilities', 'config_merge.py'))
        for path in source_tree('pyrai3md/**/*.py'):
            base = os.path.basename(path)
            if base.startswith('test_') or os.path.abspath(path) == owner:
                continue
            with open(path, 'r', encoding='utf-8') as handle:
                tree = ast.parse(handle.read(), filename=path)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in (
                        '_deep_merge', '_deep_update', '_is_empty_value',
                        'deep_merge', 'is_empty_value'):
                    offenders.append('%s:%d %s'
                                     % (os.path.relpath(path, SOURCE_PARENT),
                                        node.lineno, node.name))
        self.assertEqual(offenders, [],
                         'a private copy of the merge rule came back: %s'
                         % offenders)


    def test_no_module_defines_a_second_layer_transform(self):
        # `flatten_config_dump` / `partition_to_submodels` walk the tables
        # in schema/config.py, so they live there. normalize.py and
        # validate.py each carried a verbatim copy, and the copies were
        # free to drift while the tables could not.
        offenders = []
        owner = os.path.abspath(os.path.join(
            SOURCE_PARENT, 'pyrai3md', 'schema', 'config.py'))
        for path in source_tree('pyrai3md/**/*.py'):
            if os.path.basename(path).startswith('test_') \
                    or os.path.abspath(path) == owner:
                continue
            with open(path, 'r', encoding='utf-8') as handle:
                tree = ast.parse(handle.read(), filename=path)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in (
                        '_flatten_config_dump', '_partition_to_submodels',
                        'flatten_config_dump', 'partition_to_submodels'):
                    offenders.append('%s:%d %s'
                                     % (os.path.relpath(path, SOURCE_PARENT),
                                        node.lineno, node.name))
        self.assertEqual(offenders, [],
                         'a second copy of the layer transform came '
                         'back: %s' % offenders)

if __name__ == '__main__':
    unittest.main()
