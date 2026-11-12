"""
======================================================================
This module locks the '[perf]' row format the framework and the
lammps skill print for the same phases.

The two sides cannot share code (a skill does not import framework
internals), so they share a format instead: the framework's
PhaseProfiler renders one row per phase, and the skill had the same
row spelled out twice inside itself -- once in its backend table and
once in the resident service -- with the refresh cadence as a third
copy. The skill now renders through `profile.py::perf_row` and reads
its cadence from `PROFILE_EVERY`, and this module lays the rows side
by side so a format change on either side fails here.

The rows are compared byte for byte, not field by field: the tables
are read next to each other by whoever is looking at a slow step, and
a difference in precision or column order is the defect.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

from pyrai3md.qualification.verification._source_paths import (
    source_read, source_tree)
from pyrai3md.skills.lammps_skill.service.persistent import profile
from pyrai3md.skills.lammps_skill.service.persistent import server
from pyrai3md.utilities.timer import PhaseProfiler

PHASES = (('box', 4, 0.002), ('evaluate', 4, 0.006))


class TestOnePerfFormat(unittest.TestCase):
    """Same numbers, same rows, on both sides."""

    @staticmethod
    def _framework_lines(steps):
        profiler = PhaseProfiler()
        for name, calls, seconds in PHASES:
            profiler.stats[name] = [calls, seconds]
        return profiler.render(steps)

    @staticmethod
    def _skill_lines(steps):
        profiler = profile.StepProfiler()
        for name, calls, seconds in PHASES:
            profiler.stats[name] = [calls, seconds]
        return profiler.rows(steps)

    @staticmethod
    def _service_lines(steps):
        engine = server.ResidentEngine(lmp=None, adapter=None)
        engine.perf_steps = steps
        for name, _calls, seconds in PHASES:
            engine.perf[name] = seconds
        return engine._perf_lines()

    def test_the_backend_table_matches_the_framework(self):
        framework = self._framework_lines(4)
        skill = self._skill_lines(4)
        for row in skill:
            self.assertIn(row, framework)
        self.assertEqual(skill[:2], framework[:2])

    def test_the_service_table_matches_the_framework(self):
        framework = self._framework_lines(4)
        service = self._service_lines(4)
        # The service prints its phases in loop order, not first-seen
        # order; the rows themselves must be identical.
        for row in service:
            self.assertIn(row, framework)

    def test_the_skill_spells_the_row_once(self):
        sweep = source_tree('pyrai3md/skills/lammps_skill/**/*.py')
        holders = [path for path in sweep
                   if '/tests/' not in path and '/test_' not in path
                   and 'ms_per_step=' in open(path, encoding='utf-8').read()]
        self.assertEqual([path.rsplit('/', 1)[-1] for path in holders],
                         ['profile.py'])

    def test_the_skill_has_one_refresh_cadence(self):
        sweep = source_tree('pyrai3md/skills/lammps_skill/**/*.py')
        holders = [path for path in sweep
                   if '/tests/' not in path and '/test_' not in path
                   and 'PROFILE_EVERY = ' in open(path, encoding='utf-8').read()]
        self.assertEqual([path.rsplit('/', 1)[-1] for path in holders],
                         ['profile.py'])
        self.assertNotIn('_PERF_EVERY = ', source_read(
            'pyrai3md/skills/lammps_skill/service/persistent/server.py'))


if __name__ == '__main__':
    unittest.main()