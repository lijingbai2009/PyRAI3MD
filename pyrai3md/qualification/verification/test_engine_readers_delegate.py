"""
======================================================================
The audit's engine readers, and which of them ask the skill (#3.7).

tools/verify.py reads a fetched run tree without the run's config. Two of
its engine readers can hand the work to the skill's own parser, because
that parser takes nothing but the file:

- LAMMPS: `read_energy_file` (the reader the backend itself uses),
- ORCA: `parse_engrad` (the line ORCA writes).

The other six parsers need the run's config or runtime record, which a
product tree does not carry; those readers stay local and each says so
in its docstring, naming the parser and what it would need. This module
pins both halves, including the ORCA case the old `$ENERGY` reader got
wrong (it took the print-level integer as the energy).

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import importlib.util
import os
import tempfile
import unittest

from pyrai3md.qualification.verification._source_paths import source_path

CONFIG = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    'tools', 'verify.py')


def _verify():
    spec = importlib.util.spec_from_file_location('verify_under_test', CONFIG)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheDelegatedReaders(unittest.TestCase):
    """Two readers ask the skill; the answer is the skill's."""

    def test_lammps_reads_what_the_skill_reader_reads(self):
        from pyrai3md.skills.lammps_skill.service.parser.output_parser import (
            read_energy_file)

        verify = _verify()
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'demo.energy')
            with open(path, 'w') as handle:
                handle.write('-10.5\n')
            self.assertEqual(verify._engine_lammps(root),
                             [read_energy_file(path)])

    def test_lammps_follows_a_changed_skill_reader(self):
        from pyrai3md.skills.lammps_skill.service.parser import (
            output_parser as lammps_output)

        verify = _verify()
        original = lammps_output.read_energy_file
        lammps_output.read_energy_file = lambda path, log_path=None: -42.0
        try:
            with tempfile.TemporaryDirectory() as root:
                open(os.path.join(root, 'demo.energy'), 'w').write('1.0\n')
                self.assertEqual(verify._engine_lammps(root), [-42.0])
        finally:
            lammps_output.read_energy_file = original

    def test_orca_reads_the_layout_orca_writes(self):
        verify = _verify()
        with tempfile.TemporaryDirectory() as root:
            open(os.path.join(root, 'demo.engrad'), 'w').write(
                '#\n# The current total energy in Eh\n#\n   -76.123\n')
            self.assertEqual(verify._engine_orca(root), [-76.123])

    def test_an_unmarked_engrad_is_unread_not_zero(self):
        # The skill parser zero-fills when it finds no energy line; an
        # audit must report the file as unread rather than hand a zero to
        # the comparison.
        verify = _verify()
        with tempfile.TemporaryDirectory() as root:
            open(os.path.join(root, 'demo.engrad'), 'w').write(
                '$ENERGY\n     1\n   -76.1\n')
            self.assertIsNone(verify._engine_orca(root))


class TestTheLocalReadersSayWhy(unittest.TestCase):
    """Each reader that stays local names its parser and the reason."""

    PARSERS = {
        '_engine_openqp': 'parse_export_files',
        '_engine_xtb': 'parse_output',
        '_engine_bagel': '_parse_energy',
        '_engine_cp2k': 'parse_log',
        '_engine_pyscf': 'parse_output',
        '_engine_molcas': 'parse_log',
    }

    def test_every_local_reader_documents_its_exception(self):
        import ast
        with open(source_path('tools/verify.py'), encoding='utf-8') as handle:
            text = handle.read()
        tree = ast.parse(text)
        for name, parser in self.PARSERS.items():
            with self.subTest(reader=name):
                node = next(item for item in tree.body
                            if isinstance(item, ast.FunctionDef)
                            and item.name == name)
                doc = ast.get_docstring(node) or ''
                self.assertIn('#3.7', doc)
                self.assertIn(parser, doc)


if __name__ == '__main__':
    unittest.main()