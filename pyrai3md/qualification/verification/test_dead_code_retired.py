"""
======================================================================
The retirement record of WP6: every symbol the review found with no
reachable caller, pinned absent so it cannot grow back.

Each assertion below names the item it closes and the reading that
justified the deletion (zero call sites, or a caller that only the
tests provided). The two items whose investigation ended in "keep"
carry the reason in the assertion itself: the local child-runtime
launcher (6.10, no production caller but the one end-to-end test of
the `.rt` entry) and the composite contribution sums (the two live
assemblers build different views; their merge is item 7.2, where it is
held to a bit-identical reading).

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import ast
import importlib
import unittest

from pyrai3md.qualification.verification._source_paths import (
    source_read, source_tree)


def _defines(path, name):
    """Whether one file defines the function/class `name` at module level."""
    tree = ast.parse(source_read(path))
    return any(getattr(node, 'name', None) == name
               for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.ClassDef)))


class TestRetiredSymbols(unittest.TestCase):
    """One item per assertion, in plan order."""

    def test_6_1_the_task_model_is_gone(self):
        import pyrai3md.schema as schema
        self.assertFalse(hasattr(schema, 'Task'))
        self.assertFalse(_defines('pyrai3md/schema/validate.py', 'Task'))

    def test_6_2_the_loader_alias_is_gone(self):
        loader = importlib.import_module('pyrai3md.interface.loader')
        self.assertFalse(hasattr(loader, 'from_dict'))
        self.assertTrue(hasattr(loader, 'normalize'))

    def test_6_3_the_skill_write_back_helpers_are_gone(self):
        for path in source_tree('pyrai3md/skills/*/schema/adapter.py'):
            with self.subTest(path=path.rsplit('/', 1)[-1]):
                text = open(path, encoding='utf-8').read()
                self.assertNotIn('def update_config(', text)
                self.assertNotIn('def save_json(', text)
                self.assertIn('def read_json(', text)

    def test_6_4_the_zero_call_query_helpers_are_gone(self):
        for name in ('get_steps', 'get_modules', 'get_sections'):
            for path in source_tree('pyrai3md/skills/*/query.py'):
                self.assertNotIn('def %s(' % name,
                                 open(path, encoding='utf-8').read(), path)

    def test_6_5_the_dead_section_wrappers_are_gone(self):
        for name in ('_build_molcas_section', '_build_bagel_section',
                     '_build_pyscf_section', 'convert_all'):
            with self.subTest(name):
                self.assertNotIn(name,
                                 source_read('pyrai3md/qualification/'
                                             'certification/handler.py')
                                 + source_read('pyrai3md/qualification/'
                                               'certification/converter.py'))

    def test_6_6_no_task_is_reconstructed_from_its_mode(self):
        text = source_read('pyrai3md/qualification/runner.py')
        self.assertNotIn('pyrai3md_agent {input_file}', text)
        self.assertIn('neither a script nor an sbatch script', text)

    def test_6_7_the_empty_capability_hook_is_gone(self):
        self.assertNotIn('_register_capabilities',
                         source_read('pyrai3md/kernel/maestro.py'))

    def test_6_8_the_static_pattern_table_and_placeholder_are_gone(self):
        import pyrai3md.workflow.multiscale.composite as composite
        for name in ('COMPOSITE_PATTERNS', 'queue_contributions',
                     'validate_contributions', 'project'):
            self.assertFalse(hasattr(composite, name), name)
        self.assertTrue(callable(composite.aggregate_results))
        self.assertTrue(callable(composite.assemble_results))

    def test_6_9_the_dead_channels_are_gone(self):
        text = source_read('pyrai3md/workflow/calculator.py')
        self.assertNotIn('static_point_charges', text)
        self.assertNotIn('read_base_flags',
                         source_read('pyrai3md/workflow/dispatch/planner.py'))

    def test_6_10_the_local_launcher_is_kept_and_documented(self):
        module = importlib.import_module(
            'pyrai3md.workflow.dispatch.subprocess_run')
        self.assertTrue(callable(module.subprocess_run))
        self.assertIn('No production path calls it today',
                      module.__doc__)
        self.assertIn('subprocess_run', source_read(
            'pyrai3md/qualification/verification/test_subprocess_smoke.py'))


class TestOneCapabilityVocabulary(unittest.TestCase):
    """6.11: one table pair, derived names, and a doc list that matches."""

    def test_the_round_trip_is_the_identity(self):
        from pyrai3md.schema.capabilities import (
            CAPABILITY_TO_TOKEN, TOKEN_TO_CAPABILITY)
        for token in CAPABILITY_TO_TOKEN.values():
            with self.subTest(token):
                capability = TOKEN_TO_CAPABILITY[token]
                self.assertEqual(CAPABILITY_TO_TOKEN[capability], token)

    def test_the_name_set_is_derived_from_the_two_tables(self):
        from pyrai3md.schema.capabilities import (
            CAPABILITY_NAMES, CAPABILITY_TO_TOKEN, TOKEN_TO_CAPABILITY)
        self.assertEqual(
            CAPABILITY_NAMES,
            frozenset(set(TOKEN_TO_CAPABILITY.values())
                      | set(CAPABILITY_TO_TOKEN) | {'overlap'}))
        for capability in CAPABILITY_NAMES:
            self.assertTrue(
                capability == 'overlap'
                or capability in TOKEN_TO_CAPABILITY.values()
                or capability in CAPABILITY_TO_TOKEN,
                capability)

    def test_the_planning_doc_lists_the_same_names(self):
        from pyrai3md.schema.capabilities import CAPABILITY_NAMES
        text = source_read('pyrai3md/planning/__init__.py')
        listed = set()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith('- ') and ':' in stripped:
                listed.add(stripped[2:].split(':', 1)[0].strip())
        self.assertEqual(listed, set(CAPABILITY_NAMES))

    def test_the_vocabulary_is_defined_once(self):
        holders = []
        for path in source_tree('pyrai3md/**/*.py'):
            tree = ast.parse(open(path, encoding='utf-8').read())
            targets = {target.id
                       for node in tree.body
                       if isinstance(node, ast.Assign)
                       for target in node.targets
                       if isinstance(target, ast.Name)}
            if targets & {'TOKEN_TO_CAPABILITY', 'CAPABILITY_TO_TOKEN'}:
                holders.append(path.rsplit('/', 1)[-1])
        self.assertEqual(holders, ['capabilities.py'])


if __name__ == '__main__':
    unittest.main()