"""
======================================================================
Tests for where a qualification round puts its files.

A `pyrai3md test` round stages every generated deck, engine output and
result file under one timestamped `tmp_YYYYMMDD_HHMMSS` directory created
in the launch directory. These guards pin the three ways that contract can
be broken without any test failing: a planner that forgets to carry its
work directory, a runner that falls back to the package directory, and a
unit test whose relative scratch path leaves a directory inside
`pyrai3md/`.

Nov 6 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import re
import shutil
import tempfile
import unittest
from unittest import mock

from pyrai3md.qualification import manager, runner
from pyrai3md.qualification.verification._source_paths import source_tree

STAMP = re.compile(r'^tmp_\d{8}_\d{6}$')


class TempCwd(unittest.TestCase):
    """A test that runs in a throwaway directory and restores the cwd."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='artifact_routing_')
        self.addCleanup(shutil.rmtree, self.root, True)
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.root)


class TestCreateWorkDir(TempCwd):

    def test_the_round_directory_is_timestamped_under_the_launch_directory(self):
        path = manager.create_work_dir()
        self.assertTrue(os.path.isdir(path))
        self.assertEqual(os.path.realpath(os.path.dirname(path)),
                         os.path.realpath(self.root))
        self.assertRegex(os.path.basename(path), STAMP)

    def test_an_explicit_base_directory_is_honoured(self):
        base = os.path.join(self.root, 'rounds')
        os.makedirs(base)
        path = manager.create_work_dir(base)
        self.assertEqual(os.path.realpath(os.path.dirname(path)),
                         os.path.realpath(base))
        self.assertRegex(os.path.basename(path), STAMP)


class TestGeneratePlan(TempCwd):

    def _plan(self, work_dir):
        with mock.patch.object(manager, '_plan_example_tests',
                               return_value=[]), \
                mock.patch('pyrai3md.qualification.verification.planner'
                           '.plan_main_tests', return_value=[]), \
                mock.patch('pyrai3md.qualification.certification.planner'
                           '.plan_skill_tests', return_value=[]), \
                mock.patch('pyrai3md.qualification.benchmark.planner'
                           '.plan_benchmark_tests', return_value=[]):
            return manager.generate_plan({}, work_dir)

    def test_a_plan_always_carries_its_work_directory(self):
        work_dir = manager.create_work_dir(self.root)
        self.assertEqual(self._plan(work_dir)['work_dir'], work_dir)

    def test_an_omitted_work_directory_is_created_under_the_cwd(self):
        plan = self._plan(None)
        work_dir = plan['work_dir']
        self.assertTrue(os.path.isdir(work_dir))
        self.assertEqual(os.path.realpath(os.path.dirname(work_dir)),
                         os.path.realpath(self.root))
        self.assertRegex(os.path.basename(work_dir), STAMP)


class TestRunTestsRefusesMissingWorkDir(TempCwd):

    def test_a_plan_without_a_work_directory_raises_before_writing(self):
        package_dir = os.path.dirname(os.path.abspath(runner.__file__))
        with self.assertRaises(ValueError) as caught:
            runner.run_tests({}, {})
        self.assertIn('work_dir', str(caught.exception))
        for name in ('results.json', 'status.json'):
            self.assertFalse(os.path.exists(os.path.join(package_dir, name)),
                             '%s was written into the package' % name)


class TestNoRunArtifactsInSourceTree(unittest.TestCase):

    def test_no_test_run_directories_are_left_in_the_package(self):
        found = source_tree('pyrai3md/**/tests/run')
        self.assertEqual(found, [],
                         'unit tests left run directories behind: %r'
                         % [os.path.relpath(path) for path in found])

    def test_no_pytest_cache_is_left_in_the_package(self):
        found = source_tree('pyrai3md/**/.pytest_cache')
        self.assertEqual(found, [],
                         'pytest cache left in the package: %r'
                         % [os.path.relpath(path) for path in found])


if __name__ == '__main__':
    unittest.main()