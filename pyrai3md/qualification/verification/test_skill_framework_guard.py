"""
======================================================================
This module keeps the skill tree free of framework imports.

A skill ships and is tested on its own; it may import its own package
and the standard library, never `pyrai3md.qualification`, `.schema`,
`.kernel`, `.utilities`, `.planning`, `.interface`, `.data`,
`.workflow` or `.training_data`. The rule was broken 28 times in test
code by criteria that compared a skill's copy of a contract with the
framework's own values; those moved to
`qualification/verification/test_skill_framework_contracts.py` and the
two framework-driven files moved with them.

The scan is AST-based, so a comment that mentions an import does not
count and a real import cannot hide behind formatting. The second test
feeds the same detector a known violation, so an empty glob cannot pass
as a clean tree.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import ast
import unittest

from pyrai3md.qualification.verification._source_paths import (
    relative_to_source, source_tree)

FRAMEWORK_PACKAGES = (
    'pyrai3md.qualification',
    'pyrai3md.schema',
    'pyrai3md.kernel',
    'pyrai3md.utilities',
    'pyrai3md.planning',
    'pyrai3md.interface',
    'pyrai3md.data',
    'pyrai3md.workflow',
    'pyrai3md.training_data',
)


def framework_imports(source):
    """Yield (line, module) for every framework import in one source text."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif (isinstance(node, ast.ImportFrom) and node.level == 0
                and node.module):
            modules = [node.module]
        else:
            continue
        for module in modules:
            if any(module == package or module.startswith(package + '.')
                   for package in FRAMEWORK_PACKAGES):
                yield node.lineno, module


class TestNoFrameworkImportInTheSkillTree(unittest.TestCase):
    """Nothing under pyrai3md/skills imports the framework."""

    def test_every_skill_file_is_clean(self):
        offenders = []
        files = source_tree('pyrai3md/skills/**/*.py')
        self.assertTrue(files, 'the skill tree was not found')
        for path in files:
            with open(path, encoding='utf-8') as handle:
                source = handle.read()
            for line, module in framework_imports(source):
                offenders.append('%s:%d imports %s'
                                 % (relative_to_source(path), line, module))
        self.assertEqual(offenders, [])

    def test_the_detector_sees_a_violation(self):
        source = ('from pyrai3md.utilities.units import BOHR_TO_ANGSTROM\n'
                  'import pyrai3md.schema\n'
                  'from .schema import validate\n')
        found = sorted(module for _line, module in framework_imports(source))
        self.assertEqual(found, ['pyrai3md.schema',
                                 'pyrai3md.utilities.units'])


if __name__ == '__main__':
    unittest.main()