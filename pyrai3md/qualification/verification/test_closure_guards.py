"""
======================================================================
This module freezes the closing checks of the 2026-11-06 audit.

The audit's verification section asks for a handful of `grep`
assertions that the same rule is defined once and that the framework
does not branch on a skill's name. Run once by hand they are gone the
next time someone edits the file, so each one lives here instead:

* no production framework file compares against a literal skill
  directory name (the skill tree and tests are excluded: a test names
  the skill it drives, that is not a branch);
* `deep_merge`, `ATOMIC_MASS` and the capability tables have exactly
  one definition each;
* the framework's Bohr constant is defined in `utilities/units.py` only.

Nov 06 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import ast
import os
import unittest

PACKAGE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

SKILL_NAMES = frozenset({
    'bagel_skill', 'cp2k_skill', 'lammps_skill', 'molcas_skill',
    'nn_pes_skill', 'openqp_skill', 'orca_skill', 'pyscf_skill',
    'qdm_skill', 'xtb_skill',
})


def _framework_files():
    """Production framework files, without the skill tree and tests."""
    for root, _, files in os.walk(PACKAGE):
        parts = root.split(os.sep)
        if 'skills' in parts or 'tests' in parts:
            continue
        for name in files:
            if name.endswith('.py') and not name.startswith('test_'):
                yield os.path.join(root, name)


def _definitions(pattern):
    """Files whose top level binds `pattern` (a plain string search)."""
    hits = []
    for path in _framework_files():
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                if line.startswith(pattern):
                    hits.append(path)
                    break
    return hits


class TestTheFrameworkDoesNotBranchOnASkillName(unittest.TestCase):
    """A literal skill name in a comparison is a branch, not a lookup."""

    def test_no_production_comparison_names_a_skill(self):
        hits = []
        for path in _framework_files():
            with open(path, encoding='utf-8') as handle:
                tree = ast.parse(handle.read())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Compare):
                    continue
                for side in [node.left] + list(node.comparators):
                    if (isinstance(side, ast.Constant)
                            and side.value in SKILL_NAMES):
                        hits.append('%s:%d %r'
                                    % (os.path.relpath(path, PACKAGE),
                                       node.lineno, side.value))
        self.assertEqual(hits, [], hits)


class TestTheSingleDefinitions(unittest.TestCase):
    """The rules the audit collapsed to one home stay at one home."""

    def test_deep_merge_has_one_definition(self):
        self.assertEqual(len(_definitions('def deep_merge(')), 1,
                         _definitions('def deep_merge('))

    def test_the_atomic_mass_table_has_one_definition(self):
        self.assertEqual(len(_definitions('ATOMIC_MASS = ')), 1,
                         _definitions('ATOMIC_MASS = '))

    def test_the_capability_tables_have_one_definition(self):
        for pattern in ('TOKEN_TO_CAPABILITY = ',
                        'CAPABILITY_TO_TOKEN = ',
                        'CAPABILITY_NAMES = '):
            hits = _definitions(pattern)
            self.assertEqual(len(hits), 1, (pattern, hits))
            self.assertTrue(hits[0].endswith('schema/capabilities.py'),
                            hits)

    def test_the_bohr_constant_is_defined_in_units_only(self):
        hits = _definitions('BOHR_TO_ANGSTROM = ')
        self.assertEqual(len(hits), 1, hits)
        self.assertTrue(hits[0].endswith('utilities/units.py'), hits)


class TestTheSkillsPrefixHasOneHome(unittest.TestCase):
    """4.6: the default package name is read, never spelled in code.

    Docstrings may show an example import path; executable strings may
    not, because each one pins the default layout a third-party build
    moves.
    """

    def test_no_production_code_spells_the_default_package(self):
        hits = []
        for path in _framework_files():
            with open(path, encoding='utf-8') as handle:
                tree = ast.parse(handle.read())
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                    body = getattr(node, 'body', ())
                    if body and isinstance(body[0], ast.Expr) \
                            and isinstance(body[0].value, ast.Constant):
                        docstrings.add(id(body[0].value))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant):
                    continue
                if not isinstance(node.value, str) or id(node) in docstrings:
                    continue
                if 'pyrai3md.skills.' in node.value:
                    hits.append('%s:%d'
                                % (os.path.relpath(path, PACKAGE),
                                   getattr(node, 'lineno', 0)))
        self.assertEqual(hits, [], hits)


class TestTheCellVerdictIsSharedOnPurpose(unittest.TestCase):
    """5.12, option 2: verify calls the judged code and says so.

    The audit cannot catch a wrong threshold because there is only one
    function and one table: moving the runner's threshold moves both
    readings at once, by construction. Identity is the control; a copy
    would pass an equality check today and drift tomorrow.
    """

    def test_verify_holds_the_runner_function_objects(self):
        import importlib.util
        from pyrai3md.qualification import runner
        spec = importlib.util.spec_from_file_location(
            'verify_under_test',
            os.path.join(os.path.dirname(PACKAGE), 'tools', 'verify.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in ('check_cell_optimization', 'row_length_ratios',
                     'optimization_budget_stop', 'read_optimize_tail'):
            self.assertIs(getattr(module, name), getattr(runner, name), name)

    def test_verify_does_not_define_the_criterion_itself(self):
        with open(os.path.join(os.path.dirname(PACKAGE), 'tools',
                               'verify.py'), encoding='utf-8') as handle:
            text = handle.read()
        for name in ('def check_cell_optimization(',
                     'def row_length_ratios(',
                     'def optimization_budget_stop('):
            self.assertNotIn(name, text, name)


if __name__ == '__main__':
    unittest.main()