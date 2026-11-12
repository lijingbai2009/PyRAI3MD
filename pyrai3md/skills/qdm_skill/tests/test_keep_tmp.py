"""
======================================================================
This is a standalone surface hopping dynamics calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 05 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

What survives a run (CHANGELOG #307).

``keep_tmp`` was declared in this skill's schema and read by nobody, so
every run left its calculation directory behind whatever the input said.
It now means what it means in the other skills: 0 removes the directory
once the results are collected, 1 keeps the ``.dat`` dumps the run wrote
into it.

Clearing per step is safe here because every writer recreates the
directory (``service/runner/local_runner.py``, ``service/reference.py``),
which the last case checks by running the same backend twice.

Run: pytest skills/qdm_skill/tests/test_keep_tmp.py -v
"""

import os
import shutil
import tempfile
import unittest

from ..backend import QDMBackend

# Tully1 at a fixed geometry: an analytic evaluation, so the gate is
# about the files and costs nothing beyond them.
_STEP = {
    'skill_name': 'qdm_skill',
    'jobtype': 'energy',
    'method': 'tully1',
    'profile': 'tully1_energy',
    'atoms': ['X'],
    'coordinates': [[-7.9376581, 0.0, 0.0]],
    'nstate': 2,
    'state_spin': [0, 0],
    'target_states': [0, 1],
    'execution_mode': 'local',
    'qdm_project': 'HOLD',
    'thread': 1,
}


class TestKeepTmp(unittest.TestCase):
    """One keyword, two outcomes, measured on real artifacts."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _backend(self, execution_id, **overrides):
        config = dict(_STEP)
        config['scratch_path'] = self.root
        config.update(overrides)
        return QDMBackend(config, execution_id=execution_id)

    def _run(self, execution_id, **overrides):
        backend = self._backend(execution_id, **overrides)
        results = backend.calculate(_STEP['atoms'], _STEP['coordinates'])
        return backend, results

    def test_keeping_leaves_the_dumps_behind(self):
        backend, results = self._run('tmp_keep_QDM', keep_tmp=1)
        self.assertTrue(os.path.isdir(backend.calcdir))
        self.assertTrue(os.path.isfile(
            os.path.join(backend.calcdir, 'HOLD_energy.dat')),
            'keep_tmp=1 has to keep the files a run is read back from')
        self.assertIn('energy', results)

    def test_clearing_removes_the_directory(self):
        backend, results = self._run('tmp_clear_QDM', keep_tmp=0)
        self.assertFalse(os.path.exists(backend.calcdir),
                         'the keyword was declared and read by nobody')
        self.assertIn('energy', results,
                      'the results are collected before the cleanup')

    def test_the_default_clears(self):
        backend, _results = self._run('tmp_default_QDM')
        self.assertEqual(backend.config.environment.keep_tmp, 0)
        self.assertFalse(os.path.exists(backend.calcdir))

    def test_a_second_step_still_writes_after_a_cleanup(self):
        """The cleanup does not strand a stepped workflow."""
        backend = self._backend('tmp_twice_QDM', keep_tmp=0)
        first = backend.calculate(_STEP['atoms'], _STEP['coordinates'])
        self.assertFalse(os.path.exists(backend.calcdir))
        second = backend.calculate(_STEP['atoms'], _STEP['coordinates'])
        self.assertEqual(first['energy'], second['energy'])
        self.assertFalse(os.path.exists(backend.calcdir))


if __name__ == '__main__':
    unittest.main()
