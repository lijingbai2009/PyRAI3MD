"""
======================================================================
This is a standalone quantum chemistry calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 05 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

What survives a run (CHANGELOG #306).

``keep_tmp`` was declared in this skill's schema and read by nobody:
every run left its calculation directory behind whatever the input said.
The keyword now means what the other skills make it mean -- 0 removes
the calculation directory once the results are in hand, 1 keeps it with
its log, checkpoint and molden files.

Removing it is safe for a stepped workflow because the local runner
recreates the directory it writes into, which the last case here checks
by running the same backend twice.

Run: pytest skills/pyscf_skill/tests/test_keep_tmp.py -v
"""

import os
import shutil
import tempfile
import unittest

from ..backend import PySCFBackend

# H2 / sto-3g: an SCF in about a second, so the gate stays cheap while
# still running the real library rather than a stub.
_H2 = {
    'skill_name': 'pyscf_skill',
    'jobtype': 'energy',
    'method': 'hf',
    'basis': 'sto3g',
    'atoms': ['H', 'H'],
    'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
    'charges': 0,
    'multiplicity': 1,
    'execution_mode': 'local',
    'thread': 1,
    'verbose': 0,
    'project': 'H2',
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
        config = dict(_H2)
        config['scratch_path'] = self.root
        config.update(overrides)
        return PySCFBackend(config, execution_id=execution_id)

    def _run(self, execution_id, **overrides):
        backend = self._backend(execution_id, **overrides)
        results = backend.calculate(_H2['atoms'], _H2['coordinates'])
        return backend, results

    def test_keeping_leaves_the_log_behind(self):
        backend, results = self._run('tmp_keep_PySCF', keep_tmp=1)
        self.assertTrue(os.path.isdir(backend.calcdir))
        self.assertTrue(os.path.isfile(
            os.path.join(backend.calcdir, 'H2.log')),
            'keep_tmp=1 has to keep the files a run is read back from')
        self.assertIn('energy', results)

    def test_clearing_removes_the_directory(self):
        backend, results = self._run('tmp_clear_PySCF', keep_tmp=0)
        self.assertFalse(os.path.exists(backend.calcdir),
                         'the keyword was declared and read by nobody')
        self.assertIn('energy', results,
                      'the results are collected before the cleanup')

    def test_the_default_clears(self):
        backend, _results = self._run('tmp_default_PySCF')
        self.assertEqual(backend.config.environment.keep_tmp, 0)
        self.assertFalse(os.path.exists(backend.calcdir))

    def test_a_second_step_still_writes_after_a_cleanup(self):
        """The cleanup does not strand a stepped workflow."""
        backend = self._backend('tmp_twice_PySCF', keep_tmp=0)
        first = backend.calculate(_H2['atoms'], _H2['coordinates'])
        self.assertFalse(os.path.exists(backend.calcdir))
        second = backend.calculate(_H2['atoms'], _H2['coordinates'])
        self.assertAlmostEqual(first['energy'], second['energy'])
        self.assertFalse(os.path.exists(backend.calcdir))


if __name__ == '__main__':
    unittest.main()
