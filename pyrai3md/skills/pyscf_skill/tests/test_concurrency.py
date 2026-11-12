"""
======================================================================
This is a standalone quantum chemistry calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 04 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Two terms of this skill in one interpreter (#302).

PySCF is the only skill that runs in-process, so it is the one whose
thread pool actually shares Python state. Three things used to be
process-wide and are now per-term or per-thread:

- the log tee, which used to replace ``sys.stdout`` for the whole
  process and now travels on the molecule through the thread binding in
  ``service/utils.current_stdout``;
- ``OMP_NUM_THREADS``, which used to be rewritten by every term even
  though OpenMP latches it at the first parallel region, and is now set
  once (first term wins, the others are told);
- the artifact root: ``log``, ``chk``, ``mo.molden`` and ``opt.xyz`` are
  named after ``config.project`` under ``config.scratch_path``, which two
  terms used to share (and ``set_checkpoint`` then fed one term's
  checkpoint to the other as its initial guess). A local run now points
  that root at its own ``calcdir``.

Run: pytest skills/pyscf_skill/tests/test_concurrency.py -v
"""

import contextlib
import io
import os
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from ..backend import PySCFBackend
from ..service.utils import current_stdout, set_stdout

# H2 / sto-3g: an SCF in about a second, so the gate stays cheap while
# still running the real library rather than a stub.
_H2 = {
    'skill_name': 'pyscf_skill',
    'jobtype': 'energy',
    'method': 'hf',
    'basis': 'sto3g',
    'functional': 'pbe0',
    'atoms': ['H', 'H'],
    'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
    'charges': 0,
    'multiplicity': 1,
    'execution_mode': 'local',
    'thread': 1,
    'verbose': 0,
    'project': 'H2',
    # These cases inspect the files a term writes, so they have to
    # survive the run: keep_tmp=0 (the default since #306) would remove
    # the calculation directory as soon as the results are collected.
    'keep_tmp': 1,
}


def _config(scratch, **overrides):
    """A flat skill config for one term, under one scratch root."""
    config = dict(_H2)
    config['scratch_path'] = scratch
    config.update(overrides)
    return config


def _backend(root, execution_id, **overrides):
    """One term's backend: the execution namespace is a caller argument.

    The identity that separates terms is the ``execution_id`` handed to
    the backend, not a keyword inside the config, so the test stages it
    exactly as the framework's task queue does.
    """
    return PySCFBackend(_config(root, **overrides),
                        execution_id=execution_id)


class TestTwoTerms(unittest.TestCase):
    """Two concurrent terms keep their own files and their own output."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.root = tempfile.mkdtemp()

    def _run(self, execution_id, **overrides):
        """Build one backend and run one energy through it."""
        backend = _backend(self.root, execution_id, **overrides)
        results = backend.calculate(_H2['atoms'], _H2['coordinates'])
        return backend, results

    def test_local_run_moves_the_artifact_root_to_calcdir(self):
        backend = _backend(self.root, 'tmp_qm_PySCF')
        self.assertEqual(os.path.realpath(backend.config.scratch_path),
                         os.path.realpath(backend.calcdir),
                         'a local term still keys its artifacts on the '
                         'shared scratch root')
        self.assertNotEqual(os.path.realpath(backend.config.scratch_path),
                            os.path.realpath(self.root))

    def test_two_terms_do_not_share_one_artifact_path(self):
        first = _backend(self.root, 'tmp_qm_PySCF')
        second = _backend(self.root, 'tmp_qm2_PySCF')
        roots = [os.path.realpath(first.calcdir),
                 os.path.realpath(second.calcdir)]
        self.assertNotEqual(roots[0], roots[1])
        # The names are identical by design (same project and profile);
        # only the directory under them differs, which is what keeps the
        # second term from reading the first one's checkpoint.
        expected = os.path.join(roots[0], 'H2.log')
        self.assertTrue(expected.startswith(roots[0]))
        self.assertFalse(expected.startswith(roots[1]))

    def test_energies_match_whether_serial_or_concurrent(self):
        """Serial then concurrent, same process, same molecule twice.

        The two terms ask for the same settings, which is the case that
        used to collide: one shared default checkpoint whose presence
        made the other reuse its orbitals as the initial guess.
        """
        serial_a, results_a = self._run('tmp_a1_PySCF')
        serial_b, results_b = self._run('tmp_b1_PySCF')

        backends = [_backend(self.root, 'tmp_a2_PySCF'),
                    _backend(self.root, 'tmp_b2_PySCF')]
        with ThreadPoolExecutor(max_workers=2) as pool:
            concurrent = list(pool.map(
                lambda backend: backend.calculate(_H2['atoms'],
                                                  _H2['coordinates']),
                backends))

        reference = float(results_a['energy'][0])
        for tag, results in (('serial b', results_b),
                             ('concurrent 1', concurrent[0]),
                             ('concurrent 2', concurrent[1])):
            self.assertAlmostEqual(float(results['energy'][0]), reference,
                                   places=10, msg=tag)

    def test_each_term_logs_into_its_own_file(self):
        backends = [_backend(self.root, 'tmp_c1_PySCF', verbose=4),
                    _backend(self.root, 'tmp_c2_PySCF', verbose=4)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda backend: backend.calculate(
                _H2['atoms'], _H2['coordinates']), backends))

        for index, backend in enumerate(backends):
            path = os.path.join(backend.calcdir, 'H2.log')
            self.assertTrue(os.path.exists(path),
                            '%s wrote no log' % path)
            with open(path, encoding='utf-8') as handle:
                text = handle.read()
            self.assertIn('converged SCF energy', text)
            # The trace of one term never lands in the other's file.
            # PySCF prints its own chkfile path, so the sibling's
            # directory naming itself here is the observable symptom.
            sibling = os.path.realpath(
                backends[1 - index].calcdir)
            self.assertNotIn(sibling, text)

    def test_process_stdout_is_never_replaced(self):
        original = sys.stdout
        backend = _backend(self.root, 'tmp_d1_PySCF', verbose=4)
        backend.calculate(_H2['atoms'], _H2['coordinates'])
        self.assertIs(sys.stdout, original)


class TestOutputSinkIsThreadLocal(unittest.TestCase):
    """The tee travels with the thread, not with the process."""

    def test_binding_is_visible_only_to_its_own_thread(self):
        first = io.StringIO()
        seen = {}
        inside = threading.Event()
        finish = threading.Event()

        def bind_in_thread():
            set_stdout(first)
            inside.set()
            finish.wait(10)
            seen['worker'] = current_stdout()
            set_stdout(None)

        thread = threading.Thread(target=bind_in_thread)
        thread.start()
        self.assertTrue(inside.wait(10), 'the worker never bound its stream')

        # The test thread reads the sink while the worker still holds it.
        seen['main'] = current_stdout()
        finish.set()
        thread.join(timeout=15)

        # Under the old process-wide sys.stdout swap the main thread saw
        # the worker's tee, and a second term's binding overwrote the
        # first one's for everybody.
        self.assertIs(seen['worker'], first)
        self.assertIs(seen['main'], sys.stdout)

    def test_unbound_thread_falls_back_to_sys_stdout(self):
        set_stdout(None)
        self.assertIs(current_stdout(), sys.stdout)


class TestThreadCountIsSetOnce(unittest.TestCase):
    """OMP latches the value at the first parallel region; so do we."""

    def setUp(self):
        self._saved = os.environ.get('OMP_NUM_THREADS')

    def tearDown(self):
        if self._saved is None:
            os.environ.pop('OMP_NUM_THREADS', None)
        else:
            os.environ['OMP_NUM_THREADS'] = self._saved

    def _run_with_env(self, preexisting, requested):
        if preexisting is None:
            os.environ.pop('OMP_NUM_THREADS', None)
        else:
            os.environ['OMP_NUM_THREADS'] = str(preexisting)
        root = self._mkroot()
        errors = io.StringIO()
        backend = _backend(root, 'tmp_omp_PySCF', thread=requested)
        with contextlib.redirect_stderr(errors):
            backend.calculate(_H2['atoms'], _H2['coordinates'])
        return os.environ.get('OMP_NUM_THREADS'), errors.getvalue()

    def _mkroot(self):
        import tempfile
        path = tempfile.mkdtemp()
        self.addCleanup(lambda: _rmtree(path))
        return path

    def test_first_term_sets_the_value(self):
        effective, notes = self._run_with_env(None, 5)
        self.assertEqual(effective, '5')
        self.assertEqual(notes, '')

    def test_later_term_keeps_it_and_says_so(self):
        effective, notes = self._run_with_env(3, 7)
        self.assertEqual(effective, '3',
                         'the second term rewrote a value OpenMP has '
                         'already latched')
        self.assertIn('OMP_NUM_THREADS is already 3', notes)
        self.assertIn('thread=7 is ignored', notes)


def _rmtree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
