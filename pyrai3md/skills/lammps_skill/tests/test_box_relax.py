"""
======================================================================
Test suite for the engine-side box relaxation of the LAMMPS skill
(`profile: ff_optimize_cell`, `box_relax`, `box_relax_pressure`).

`minimize` has no cell degree of freedom on its own, so a box the engine is
to relax needs `fix box/relax` beside it, and that fix exists only under
`minimize` (measured on the test build: under `run` it is accepted and
the box does not move at all - CHANGELOG.md #362). Three things are pinned
here: the line the builder writes is the grammar this build accepts and it
names `vmax` out loud (its default 1e-4 held a 100 kbar target to 0.01% of an
edge), the pairing between the style and the profile is refused rather than
silently ignored, and the box the engine stopped on comes back rather than
being replaced by the one that was asked for.

The model input is built in memory; no engine is executed here.

Oct 17 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import shutil
import tempfile
import unittest

from pyrai3md.skills.lammps_skill.schema import validate as skill_validate
from pyrai3md.skills.lammps_skill.service.builder import input_builder
from pyrai3md.skills.lammps_skill.service.parser import log_parser


class _Config:
    """The keywords the optimize section reads, and nothing else."""

    lammps_project = 'h2o'
    lammps_units = 'real'
    dump_freq = 1
    min_style = 'cg'
    min_etol = 1.0e-4
    min_ftol = 1.0e-4
    min_maxiter = 10
    min_maxeval = 100
    shake_enabled = False
    box_relax = 'off'
    box_relax_pressure = 0.0


class _Workflow:
    profile = 'ff_optimize_cell'
    is_md = False
    needs_gradient = False


def _section(**overrides):
    config = _Config()
    for key, value in overrides.items():
        setattr(config, key, value)
    return input_builder._build_optimize_section(config, _Workflow())


class TestTheFixLineIsTheGrammarThisBuildAccepts(unittest.TestCase):
    """One pressure value, the style asked for, and `vmax` stated."""

    def test_an_off_style_writes_the_table_it_always_wrote(self):
        text = _section()
        self.assertNotIn('box/relax', text)
        self.assertNotIn('CELL', text)
        self.assertIn('min_style cg', text)

    def test_iso_writes_one_value_and_names_vmax(self):
        text = _section(box_relax='iso', box_relax_pressure=1000.0)
        self.assertIn('fix PYRAI_BOX all box/relax iso 986.923 vmax 0.01',
                      text)
        self.assertIn('unfix PYRAI_BOX', text)

    def test_the_pressure_travels_in_the_units_the_run_declares(self):
        bar = _section(box_relax='aniso', box_relax_pressure=1000.0)
        metal = _section(box_relax='aniso', box_relax_pressure=1000.0)
        metal_config_units = bar.split('box/relax')[1].split('\n')[0]
        self.assertIn('aniso 986.923', metal_config_units)
        config = _Config()
        config.box_relax = 'aniso'
        config.box_relax_pressure = 1000.0
        config.lammps_units = 'metal'
        direct = input_builder._build_optimize_section(config, _Workflow())
        self.assertIn('box/relax aniso 1000 ', direct)

    def test_the_answer_box_is_printed_after_the_relaxation(self):
        text = _section(box_relax='iso')
        lines = text.splitlines()
        fix = next(i for i, line in enumerate(lines) if 'box/relax' in line)
        minimize = next(i for i, line in enumerate(lines)
                        if line.startswith('minimize'))
        cell = next(i for i, line in enumerate(lines)
                    if line.startswith('print "CELL'))
        unfix = next(i for i, line in enumerate(lines)
                     if line.startswith('unfix'))
        self.assertLess(fix, minimize,
                        'the fix must be configured before the minimize that '
                        'uses it')
        self.assertLess(minimize, cell,
                        'a box printed before the relaxation is the request, '
                        'not the answer')
        self.assertLess(cell, unfix)


class TestTheBoxTheEngineStoppedOnComesBack(unittest.TestCase):
    """`parse_cell_file`, and the refusal to keep the request box."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='box_relax_')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, text, name='h2o_ff_optimize_cell.cell'):
        path = os.path.join(self.root, name)
        with open(path, 'w') as handler:
            handler.write(text)
        return path

    def test_a_read_line_gives_the_six_box_numbers(self):
        path = self._write('CELL 25.4 23.3 26.6 0.1 -0.2 0.3\n')
        self.assertEqual(log_parser.parse_cell_file(path),
                         [25.4, 23.3, 26.6, 0.1, -0.2, 0.3])

    def test_a_missing_file_is_not_a_box(self):
        self.assertIsNone(log_parser.parse_cell_file(
            os.path.join(self.root, 'nothing.cell')))

    def test_a_malformed_line_is_not_a_box(self):
        path = self._write('CELL 25.4 23.3 26.6 0.1 -0.2\n')
        self.assertIsNone(log_parser.parse_cell_file(path))
        path = self._write('CELL 25.4 23.3 26.6 0.1 -0.2 abc\n')
        self.assertIsNone(log_parser.parse_cell_file(path))


class TestTheStyleAndTheProfileHaveToAgree(unittest.TestCase):
    """`validate_internal` refuses the three pairings that read as nothing."""

    BASE = dict(profile='ff_optimize_cell', box_relax='iso',
                box_relax_pressure=1000.0, min_maxiter=10, min_maxeval=100,
                min_etol=1.0e-4, min_ftol=1.0e-4)

    def _validate(self, **overrides):
        data = dict(self.BASE)
        data.update(overrides)
        return skill_validate.validate_internal(data)

    def test_a_named_style_on_the_plain_profile_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self._validate(profile='ff_optimize')
        self.assertIn('ff_optimize_cell', str(caught.exception))

    def test_the_cell_profile_without_a_style_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self._validate(box_relax='off')
        self.assertIn('box_relax', str(caught.exception))

    def test_an_unknown_style_lists_the_three(self):
        with self.assertRaises(ValueError) as caught:
            self._validate(box_relax='triclinic')
        message = str(caught.exception)
        for style in ('off', 'iso', 'aniso'):
            self.assertIn(style, message)

    def test_the_task_of_the_cell_profile_is_still_optimize(self):
        # The suffix is a modifier, not a task: a builder that resolved it as a
        # task called 'cell' would write no section at all.
        self.assertEqual(skill_validate._profile_to_task('ff_optimize_cell'),
                         'optimize')
        self.assertEqual(skill_validate._profile_to_task('ff_npt'), 'npt')


if __name__ == '__main__':
    unittest.main()
