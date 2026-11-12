"""
======================================================================
Tests that the physical constants have one definition in the framework.

`utilities/units.py` states them, and until now several framework modules
carried their own rounded copies beside it:

  quantity            utilities/units        the copy
  Bohr -> Angstrom    0.52917721090299996    0.529177249        (7.199e-08)
  amu -> au mass      1822.8884871474306     1822.88852         (1.802e-08)
  cm-1 -> Hartree     4.556335252e-06        4.55633518e-06     (1.580e-08)
  K -> Hartree        3.1668115634555182e-06 3.16681156e-06     (1.091e-09)
  Angstrom -> Bohr    1.8897261246257504     1.8897259886       (7.199e-08)
  cm-1 per Kelvin     (derived) 0.6950348006 0.69503            (6.907e-06)

Relative differences of 1e-8 are invisible in a printed energy and
decisive in a seeded Wigner draw or a TurboMole coordinate written to
sixteen decimals, and the schema's copy could drift from the workflow's
without either side noticing. These cases fail if a literal comes back
into the framework, or if a module re-derives a value the constants
already give.

The skills keep their own copies by design: framework and skill are
independent, and a skill that imported `utilities.units` would be
reaching across that boundary for a number it can state itself. Only
framework modules are scanned here.

Nov 6 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import ast
import inspect
import os
import unittest

from pyrai3md.qualification.verification._source_paths import (
    SOURCE_PARENT, source_read, source_tree)
from pyrai3md.schema import reference
from pyrai3md.utilities import units
from pyrai3md.utilities import structure_formatter
from pyrai3md.workflow import sampling
from pyrai3md.workflow.optimizer import base as optimizer_base

# Modules that belong to the framework: the same rule that keeps engine
# knowledge out of them keeps a second copy of a constant out of them.
FRAMEWORK_DIRS = ('schema', 'workflow', 'data', 'kernel', 'utilities',
                  'planning', 'interface', 'qualification', 'training_data')
# utilities/units.py owns these numbers and is not an offender.
OWNER = os.path.join('pyrai3md', 'utilities', 'units.py')
# The retired spellings, compared as values rather than as prefixes: a
# rounded copy differs from the shared constant in the value, so a
# literal that equals one of these is a second source of truth.
RETIRED_VALUES = {0.529177249, 1822.88852, 4.55633518e-6, 3.16681156e-6,
                  1.8897259886, 0.69503, 0.529177210903}
# A test that compares the live route with the route it replaced states
# the old number on purpose, under a LEGACY_ name. That is a record of
# history, not a second source of truth, so the sweep skips it.
LEGACY_PREFIXES = ('LEGACY_', '_LEGACY_')


def _framework_source_files():
    found = []
    # This file is the one place the retired spellings have to be written
    # down, so it cannot be its own offender.
    here = os.path.abspath(__file__)
    for directory in FRAMEWORK_DIRS:
        root = os.path.join(SOURCE_PARENT, 'pyrai3md', directory)
        for dirpath, dirnames, names in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != '__pycache__']
            for name in sorted(names):
                if name.endswith('.py'):
                    path = os.path.join(dirpath, name)
                    if os.path.relpath(path, SOURCE_PARENT) == OWNER:
                        continue
                    if os.path.abspath(path) == here:
                        continue
                    found.append(path)
    # tools/ runs on the same numbers, from outside the package.
    tools = os.path.join(SOURCE_PARENT, 'tools')
    if os.path.isdir(tools):
        found.extend(os.path.join(tools, name) for name in
                     sorted(os.listdir(tools)) if name.endswith('.py'))
    return sorted(found)


def _legacy_assignment_lines(tree):
    """Lines where a LEGACY_-named binding holds the frozen old number."""
    lines = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            name = getattr(target, 'id', '')
            if name.startswith(LEGACY_PREFIXES):
                lines.update(range(node.lineno, (node.end_lineno
                                                 or node.lineno) + 1))
    return lines


class TestOneSetOfConstants(unittest.TestCase):

    def test_sampling_uses_the_shared_numbers(self):
        self.assertEqual(sampling.BOHR_TO_ANGSTROM, units.BOHR_TO_ANGSTROM)
        self.assertEqual(sampling.AMU_TO_AU, units.AMU_TO_AU)
        self.assertEqual(sampling.CM_TO_HARTREE, units.CM_TO_HARTREE)
        self.assertEqual(sampling.BOLTZMANN_TO_HARTREE_PER_K,
                         units.BOLTZMANN_TO_HARTREE_PER_K)
        self.assertEqual(sampling.CM_TO_KELVIN,
                         units.BOLTZMANN_TO_HARTREE_PER_K
                         / units.CM_TO_HARTREE)

    def test_the_legacy_names_are_gone_from_the_module(self):
        for retired in ('MU_TO_HARTREE', 'MA_TO_AMU', 'K_TO_AU'):
            self.assertFalse(hasattr(sampling, retired), retired)
        source = source_read('pyrai3md/workflow/sampling.py')
        # The docstring records what used to be there; the code may not
        # carry it as a live value.
        self.assertNotIn('MU_TO_HARTREE =', source)
        self.assertNotIn('MA_TO_AMU =', source)

    def test_the_mass_table_has_one_owner(self):
        # The copy in sampling carried 86 symbols of the table's 118; the
        # actinides and transactinides raised ValueError there only.
        self.assertIs(sampling.ATOMIC_MASS, reference.ATOMIC_MASS)
        self.assertEqual(len(reference.ATOMIC_MASS), 118)
        for symbol in ('Ac', 'Pa', 'U', 'Cn', 'Og'):
            self.assertAlmostEqual(sampling.atomic_mass(symbol),
                                   reference.ATOMIC_MASS[symbol])
        self.assertGreater(len(reference.ATOMIC_MASS), 86,
                           'the table the copy was cut from must still be '
                           'the wider one')

    def test_the_formatter_and_the_optimizer_read_the_same_factor(self):
        converted = structure_formatter.print_turbocoord(
            [['C', 1.0, 0.0, 0.0]], [], [])
        self.assertIn('%24.16f' % (1.0 * units.ANGSTROM_TO_BOHR), converted)
        defaults = inspect.signature(
            optimizer_base.Optimizer.__init__).parameters
        self.assertEqual(defaults['coord_to_bohr'].default,
                         units.ANGSTROM_TO_BOHR)
        self.assertEqual(defaults['disp_to_angstrom'].default,
                         units.BOHR_TO_ANGSTROM)

    def test_the_schema_re_exports_instead_of_restating(self):
        from pyrai3md import schema as schema_pkg

        for name in ('BOHR_TO_ANGSTROM', 'ANGSTROM_TO_BOHR', 'AMU_TO_AU',
                     'HARTREE_TO_EV', 'EV_TO_HARTREE', 'HARTREE_TO_KCAL',
                     'KCAL_TO_HARTREE', 'CM_TO_HARTREE', 'HARTREE_TO_CM'):
            self.assertEqual(getattr(schema_pkg, name), getattr(units, name),
                             name)
        source = source_read('pyrai3md/schema/reference.py')
        tree = ast.parse(source)
        self.assertFalse(
            any(isinstance(node, (ast.Import, ast.ImportFrom))
                for node in ast.walk(tree)),
            'msg=schema/reference.py must stay pure data with no import '
            'statement; the public re-export lives in schema/__init__.py')
        self.assertNotIn('BOHR_TO_ANGSTROM = 0.529', source)

    def test_no_framework_module_carries_a_retired_literal(self):
        offenders = []
        for path in _framework_source_files():
            with open(path, 'r', encoding='utf-8') as handle:
                tree = ast.parse(handle.read(), filename=path)
            frozen = _legacy_assignment_lines(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant):
                    continue
                if not isinstance(node.value, (int, float)) or isinstance(
                        node.value, bool):
                    continue
                if node.lineno in frozen:
                    continue
                if float(node.value) in RETIRED_VALUES:
                    offenders.append('%s:%d %r' % (
                        os.path.relpath(path, SOURCE_PARENT), node.lineno,
                        node.value))
        self.assertEqual(offenders, [],
                         'a retired copy of a shared constant came back: %s'
                         % offenders)

    def test_the_scan_covers_the_framework_tree(self):
        files = _framework_source_files()
        self.assertGreater(len(files), 100,
                           'the constant sweep must read the framework tree, '
                           'not an empty subset of it')
        self.assertTrue(any(path.endswith(os.path.join('workflow',
                                                       'sampling.py'))
                            for path in files))


if __name__ == '__main__':
    unittest.main()
