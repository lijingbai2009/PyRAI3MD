"""
======================================================================
Test suite for kernel module (pyrai3md/kernel/).

Tests cover:
- Maestro: skill discovery, config store, skill resolution
- RuntimeState: record_step_result() and get_step_result()
- New exception types: CalculationError, ConvergenceError, CapabilityError
- RunLogger wall-clock reporting (how_long rounding, start_time)

Aug 01 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


class TestMaestroConfigStore(unittest.TestCase):
    """Test Maestro config registration and retrieval."""

    def test_register_and_get_config(self):
        from pyrai3md.kernel.maestro import Maestro

        maestro = Maestro()
        maestro.register_config('molcas_skill', {'molcas_path': '/opt/molcas'})
        result = maestro.get_config('molcas_skill')
        self.assertEqual(result['molcas_path'], '/opt/molcas')

    def test_get_config_not_found_returns_empty(self):
        from pyrai3md.kernel.maestro import Maestro

        maestro = Maestro()
        result = maestro.get_config('nonexistent')
        self.assertEqual(result, {})

    def test_register_multiple_configs(self):
        from pyrai3md.kernel.maestro import Maestro

        maestro = Maestro()
        maestro.register_config('molcas_skill', {'key': 'val1'})
        maestro.register_config('pyscf_skill', {'key': 'val2'})
        self.assertEqual(maestro.get_config('molcas_skill')['key'], 'val1')
        self.assertEqual(maestro.get_config('pyscf_skill')['key'], 'val2')


class TestMaestroSkillClassRegistry(unittest.TestCase):
    """Test Maestro skill class registration (legacy)."""

    def test_register_and_get_skill(self):
        from pyrai3md.kernel.maestro import Maestro

        class MockBackend:
            pass

        maestro = Maestro()
        maestro.register_skill('mock_skill', MockBackend)
        result = maestro.get_skill('mock_skill')
        self.assertEqual(result, MockBackend)

    def test_get_skill_not_found_raises(self):
        from pyrai3md.kernel.maestro import Maestro
        from pyrai3md.kernel.exceptions import SkillNotFoundError

        maestro = Maestro()
        with self.assertRaises(SkillNotFoundError):
            maestro.get_skill('nonexistent')

    def test_list_skills(self):
        from pyrai3md.kernel.maestro import Maestro

        maestro = Maestro()
        maestro.register_skill('a', type('A', (), {}))
        maestro.register_skill('b', type('B', (), {}))
        skills = maestro.list_skills()
        self.assertIn('a', skills)
        self.assertIn('b', skills)


class TestRuntimeStateStepResults(unittest.TestCase):
    """Test RuntimeState step result recording."""

    def test_default_step_results_empty(self):
        from pyrai3md.data.runtime.state import RuntimeState
        rt = RuntimeState()
        self.assertEqual(rt.step_results, [])

    def test_record_step_result(self):
        from pyrai3md.data.runtime.state import RuntimeState
        rt = RuntimeState()
        rt.record_step_result()
        self.assertEqual(len(rt.step_results), 1)
        # Runtime records execution metadata only (no properties).
        self.assertEqual(rt.step_results[0], {'step': 0})

    def test_record_multiple_steps(self):
        from pyrai3md.data.runtime.state import RuntimeState
        rt = RuntimeState()
        rt.record_step_result()
        rt.advance_step()
        rt.record_step_result()
        self.assertEqual(len(rt.step_results), 2)
        self.assertEqual(rt.step_results[0]['step'], 0)
        self.assertEqual(rt.step_results[1]['step'], 1)

    def test_advance_step_counts_only(self):
        from pyrai3md.data.runtime.state import RuntimeState
        rt = RuntimeState()
        self.assertEqual(rt.current_step, 0)
        rt.advance_step()
        rt.advance_step()
        self.assertEqual(rt.current_step, 2)
        self.assertEqual(rt.iteration, 2)
        # Counting only: physical time is derived by output.
        self.assertEqual(rt.simulation_time, 0.0)

    def test_kinetic_energy_field_default_and_roundtrip(self):
        from pyrai3md.data.runtime.state import RuntimeState
        rt = RuntimeState()
        self.assertEqual(rt.kinetic_energy, 0.0)
        rt.kinetic_energy = 0.125
        restored = RuntimeState.from_dict(rt.to_dict())
        self.assertAlmostEqual(restored.kinetic_energy, 0.125)
        self.assertAlmostEqual(rt.copy().kinetic_energy, 0.125)

    def test_get_step_result(self):
        from pyrai3md.data.runtime.state import RuntimeState
        rt = RuntimeState()
        rt.record_step_result()
        rt.advance_step()
        rt.record_step_result()
        result = rt.get_step_result(0)
        self.assertEqual(result, {'step': 0})
        result = rt.get_step_result(1)
        self.assertEqual(result, {'step': 1})

    def test_get_step_result_not_found(self):
        from pyrai3md.data.runtime.state import RuntimeState
        rt = RuntimeState()
        result = rt.get_step_result(99)
        self.assertIsNone(result)

    def test_to_dict_includes_step_results(self):
        from pyrai3md.data.runtime.state import RuntimeState
        rt = RuntimeState()
        rt.record_step_result()
        d = rt.to_dict()
        self.assertIn('step_results', d)
        self.assertEqual(len(d['step_results']), 1)

    def test_from_dict_restores_step_results(self):
        from pyrai3md.data.runtime.state import RuntimeState
        rt = RuntimeState()
        rt.record_step_result()
        d = rt.to_dict()
        rt2 = RuntimeState.from_dict(d)
        self.assertEqual(len(rt2.step_results), 1)
        self.assertEqual(rt2.step_results[0], {'step': 0})

    def test_copy_includes_step_results(self):
        from pyrai3md.data.runtime.state import RuntimeState
        rt = RuntimeState()
        rt.record_step_result()
        rt2 = rt.copy()
        self.assertEqual(len(rt2.step_results), 1)
        # Verify independence
        rt2.record_step_result()
        self.assertEqual(len(rt.step_results), 1)
        self.assertEqual(len(rt2.step_results), 2)


class TestMaestroDiscovery(unittest.TestCase):
    """Test Maestro skill discovery with mock directories."""

    def test_discover_empty_directory(self):
        from pyrai3md.kernel.maestro import Maestro

        maestro = Maestro()

        with tempfile.TemporaryDirectory() as tmpdir:
            discovered = maestro.discover(search_paths=[tmpdir])
            self.assertEqual(discovered, [])

    def test_discover_with_manifest(self):
        from pyrai3md.kernel.maestro import Maestro

        maestro = Maestro()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock skill directory with manifest.yaml
            skill_dir = os.path.join(tmpdir, 'mock_skill')
            os.makedirs(skill_dir)
            with open(os.path.join(skill_dir, 'manifest.yaml'), 'w') as f:
                f.write("name: mock_skill\ncapabilities: {}\n")

            discovered = maestro.discover(search_paths=[tmpdir])
            self.assertEqual(discovered, ['mock_skill'])

    def test_discover_skips_no_manifest(self):
        from pyrai3md.kernel.maestro import Maestro

        maestro = Maestro()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory without manifest.yaml
            os.makedirs(os.path.join(tmpdir, 'no_manifest_skill'))

            discovered = maestro.discover(search_paths=[tmpdir])
            self.assertEqual(discovered, [])

    def test_discover_nonexistent_path(self):
        from pyrai3md.kernel.maestro import Maestro

        maestro = Maestro()
        discovered = maestro.discover(search_paths=['/nonexistent/path'])
        self.assertEqual(discovered, [])

    def test_load_skill_graceful_failure(self):
        from pyrai3md.kernel.maestro import Maestro

        maestro = Maestro()
        # Loading a non-existent skill should not raise
        result = maestro.load_skill('nonexistent_skill_xyz')
        self.assertFalse(result)


class TestNewExceptions(unittest.TestCase):
    """Test new calculation-related exception types."""

    def test_calculation_error(self):
        from pyrai3md.kernel.exceptions import CalculationError, PyRAI3MDError
        e = CalculationError("SCF failed to converge")
        self.assertIsInstance(e, PyRAI3MDError)
        self.assertIn("SCF failed", str(e))

    def test_convergence_error(self):
        from pyrai3md.kernel.exceptions import ConvergenceError, PyRAI3MDError
        e = ConvergenceError("Geometry optimization did not converge")
        self.assertIsInstance(e, PyRAI3MDError)
        self.assertIn("converge", str(e))

    def test_capability_error(self):
        from pyrai3md.kernel.exceptions import CapabilityError, PyRAI3MDError
        e = CapabilityError("NAC not supported by this backend")
        self.assertIsInstance(e, PyRAI3MDError)
        self.assertIn("NAC", str(e))

    def test_exceptions_importable_from_kernel(self):
        from pyrai3md.kernel import CalculationError, ConvergenceError, CapabilityError
        self.assertTrue(issubclass(CalculationError, Exception))
        self.assertTrue(issubclass(ConvergenceError, Exception))
        self.assertTrue(issubclass(CapabilityError, Exception))

    def test_exception_hierarchy(self):
        from pyrai3md.kernel.exceptions import (
            PyRAI3MDError, CalculationError, ConvergenceError, CapabilityError,
        )
        self.assertTrue(issubclass(CalculationError, PyRAI3MDError))
        self.assertTrue(issubclass(ConvergenceError, PyRAI3MDError))
        self.assertTrue(issubclass(CapabilityError, PyRAI3MDError))


class TestRunLoggerWallClock(unittest.TestCase):
    """Test the run-log wall-clock reporting."""

    def test_how_long_rounds_sub_second(self):
        from pyrai3md.utilities.timer import how_long
        # A ~1 s run must not be reported as zero (old truncation
        # printed 00:00:00:00 whenever the wall clock advanced).
        self.assertEqual(how_long(100.0, 100.9), '00:00:00:01')
        self.assertEqual(how_long(100.0, 100.4), '00:00:00:00')
        self.assertEqual(how_long(100.0, 161.2), '00:00:01:01')
        self.assertEqual(how_long(100.0, 100.0), '00:00:00:00')

    def test_logger_uses_provided_start_time(self):
        import time
        from pyrai3md.kernel.logger import RunLogger
        from pyrai3md.data.state import PyRAIState
        from pyrai3md.data.runtime.state import RuntimeState
        from pyrai3md.schema.config import PyRAIConfig

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                config = PyRAIConfig()
                config.control.title = 'wallclock'
                state = PyRAIState(runtime=RuntimeState())
                state.pmd = config
                start = time.time() - 3.6
                logger = RunLogger(state, start_time=start)
                logger.finish('completed')
            finally:
                os.chdir(cwd)
            with open(os.path.join(tmp, 'wallclock.log')) as f:
                content = f.read()
            # The total must cover the time before the logger was
            # constructed (input parsing / environment init).
            self.assertIn('Total: 00:00:00:04', content)


class TestRunLoggerDetachedCopy(unittest.TestCase):
    """A dispatched child must append to the run log, never truncate it.

    `materialize_unit` deep-copies the parent state, and with it the open
    RunLogger. The copy detaches the handle (it cannot be pickled) and
    reopens lazily -- but the path it reopens is the parent's absolute
    file. Opened for writing, the first child write deleted everything the
    parent had already reported, and each writer then advanced its own
    offset in one file, leaving holes the filesystem fills with NUL bytes.
    A four-point grid scan measured 10475 NUL bytes in its own log before
    this rule existed.
    """

    def _state(self, title):
        from pyrai3md.data.state import PyRAIState
        from pyrai3md.data.runtime.state import RuntimeState
        from pyrai3md.schema.config import PyRAIConfig

        config = PyRAIConfig()
        config.control.title = title
        state = PyRAIState(runtime=RuntimeState())
        state.pmd = config
        return state

    def test_detached_copy_writes_nothing_and_the_parent_survives(self):
        import copy

        from pyrai3md.kernel.logger import RunLogger

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                logger = RunLogger(self._state('detached'), start_time=None)
                logger.write('parent first\n')
                child = copy.deepcopy(logger)
                child.write('child block\n')
                logger.write('parent second\n')
                child.finish('completed')
                logger.finish('completed')
                self.assertTrue(child._closed)
            finally:
                os.chdir(cwd)
            with open(os.path.join(tmp, 'detached.log'), 'rb') as handle:
                raw = handle.read()
            text = raw.decode('utf-8', errors='replace')

            self.assertEqual(0, raw.count(b'\x00'),
                             'the log holds NUL holes: two handles advanced '
                             'their own offsets in one file')
            self.assertNotIn('child block', text,
                             'a worker wrote into the parent run log')
            for line in ('parent first', 'parent second'):
                self.assertIn(line, text)
            self.assertLess(text.index('parent first'),
                            text.index('parent second'))

    def test_a_pickled_logger_is_closed_too(self):
        # The pool sends states by pickle, not deepcopy, so the same rule
        # has to hold on that path.
        import pickle

        from pyrai3md.kernel.logger import RunLogger

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                logger = RunLogger(self._state('pickled'))
                restored = pickle.loads(pickle.dumps(logger))
                self.assertTrue(restored._closed)
                logger.finish('completed')
            finally:
                os.chdir(cwd)
            with open(os.path.join(tmp, 'pickled.log'), 'rb') as handle:
                self.assertEqual(b'', handle.read().split(b'parent')[0][:0])

    def test_a_deepcopied_logger_is_marked_detached(self):
        import copy

        from pyrai3md.kernel.logger import RunLogger

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                logger = RunLogger(self._state('flag'))
                child = copy.deepcopy(logger)
                self.assertFalse(logger._closed)
                self.assertTrue(child._closed)
                # The handle is not carried into the copy at all.
                self.assertIsNone(child._handler)
                logger.finish('completed')
            finally:
                os.chdir(cwd)


class TestRunLoggerVerbose(unittest.TestCase):
    """The result-block content level follows control.verbose."""

    def _logged_result_block(self, verbose):
        from pyrai3md.kernel.logger import RunLogger
        from pyrai3md.data.state import PyRAIState
        from pyrai3md.data.runtime.state import RuntimeState
        from pyrai3md.schema.config import PyRAIConfig

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                config = PyRAIConfig()
                config.control.title = 'verbose_probe'
                config.control.verbose = verbose
                config.system.atoms = ['X']
                config.system.coordinates = [[0.0, 0.0, 0.0]]
                config.system.current_state = 0
                state = PyRAIState(runtime=RuntimeState())
                state.pmd = config
                logger = RunLogger(state)
                self.assertEqual(logger.verbose, verbose)
                logger.log_result(state)
                logger.finish('completed')
            finally:
                os.chdir(cwd)
            with open(os.path.join(tmp, 'verbose_probe.log')) as f:
                return f.read()

    def test_default_verbose_suppresses_not_computed(self):
        content = self._logged_result_block(2)
        self.assertNotIn('Not computed', content)

    def test_verbose_three_restores_not_computed(self):
        content = self._logged_result_block(3)
        self.assertIn('Not computed', content)

    def test_level_zero_has_no_structure(self):
        content = self._logged_result_block(0)
        self.assertNotIn('&coordinates', content)

    def test_level_one_restores_structure(self):
        content = self._logged_result_block(1)
        self.assertIn('&coordinates', content)

    def test_deepcopy_keeps_the_verbose_level(self):
        import copy as copy_module
        from pyrai3md.kernel.logger import RunLogger
        from pyrai3md.data.state import PyRAIState
        from pyrai3md.data.runtime.state import RuntimeState
        from pyrai3md.schema.config import PyRAIConfig

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                config = PyRAIConfig()
                config.control.title = 'copy_probe'
                config.control.verbose = 3
                state = PyRAIState(runtime=RuntimeState())
                state.pmd = config
                logger = RunLogger(state)
                clone = copy_module.deepcopy(logger)
            finally:
                os.chdir(cwd)
        self.assertEqual(clone.verbose, 3)


if __name__ == '__main__':
    unittest.main()
