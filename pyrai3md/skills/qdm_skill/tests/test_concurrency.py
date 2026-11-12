"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 04 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Two QDM terms in one interpreter (#302).

QDM runs in-process on numpy, so a thread pool shares its interpreter
state. The one process-wide setting it used to touch was
``os.environ['OMP_NUM_THREADS']``, written by every term although OpenMP
reads the value once at the first parallel region; the write is now a
first-set-wins default, with the disagreement printed instead of hidden.

Run: pytest skills/qdm_skill/tests/test_concurrency.py -v
"""

import contextlib
import io
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from ..schema.config import SkillConfig
from ..service.runner.local_runner import run_local

# Tully1 at the crossing geometry, in Angstrom (0 A == 0 Bohr).
_COORDINATES = [[0.0, 0.0, 0.0]]
_ATOMS = ['X']


def _config(scratch, thread=1):
    """A minimal single-point config for one term."""
    return SkillConfig(
        system={'atoms': _ATOMS, 'coordinates': _COORDINATES, 'nstate': 2},
        calculation={'method': 'tully1', 'profile': 'tully1_energy'},
        environment={'scratch_path': scratch, 'thread': thread},
    )


def _energy(config, calcdir):
    """One term's energy through the local runner."""
    os.makedirs(calcdir, exist_ok=True)
    return run_local(config, calcdir)


class TestTwoTerms(unittest.TestCase):
    """Concurrent terms reproduce the serial numbers."""

    def test_concurrent_terms_match_serial(self):
        with tempfile.TemporaryDirectory() as root:
            config = _config(root)
            serial_first = _energy(config, os.path.join(root, 'term_1'))
            serial_second = _energy(config, os.path.join(root, 'term_2'))

            backends = [os.path.join(root, 'term_3'),
                        os.path.join(root, 'term_4')]
            with ThreadPoolExecutor(max_workers=2) as pool:
                concurrent = list(pool.map(
                    lambda calcdir: _energy(config, calcdir), backends))

            reference = serial_first['energy']
            for tag, results in (('serial second', serial_second),
                                 ('concurrent first', concurrent[0]),
                                 ('concurrent second', concurrent[1])):
                self.assertEqual(results['energy'], reference, tag)

    def test_each_term_writes_only_into_its_own_directory(self):
        """Same file names, different directories, nothing in the root.

        The names are identical by design (they follow the project and
        profile); what a term must not do is write beside its own
        calculation directory, because the shared root is where a second
        term would then overwrite it.
        """
        with tempfile.TemporaryDirectory() as root:
            config = _config(root)
            dirs = [os.path.join(root, 'a'), os.path.join(root, 'b')]
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda path: _energy(config, path), dirs))

            listed_by_dir = {}
            for path in dirs:
                names = sorted(os.listdir(path))
                self.assertTrue(names, '%s kept no output' % path)
                listed_by_dir[path] = names

            # Both terms produced the same set of names, each inside its
            # own directory...
            self.assertEqual(listed_by_dir[dirs[0]], listed_by_dir[dirs[1]])
            # ...and neither reached past its own boundary.
            stray = [name for name in sorted(os.listdir(root))
                     if name not in ('a', 'b')]
            self.assertEqual([], stray)


class TestThreadCountIsSetOnce(unittest.TestCase):
    """OMP latches the value at the first parallel region; so do we."""

    def setUp(self):
        self._saved = os.environ.get('OMP_NUM_THREADS')

    def tearDown(self):
        if self._saved is None:
            os.environ.pop('OMP_NUM_THREADS', None)
        else:
            os.environ['OMP_NUM_THREADS'] = self._saved

    def _run(self, preexisting, requested):
        if preexisting is None:
            os.environ.pop('OMP_NUM_THREADS', None)
        else:
            os.environ['OMP_NUM_THREADS'] = str(preexisting)
        notes = io.StringIO()
        with tempfile.TemporaryDirectory() as root:
            config = _config(root, thread=requested)
            with contextlib.redirect_stderr(notes):
                _energy(config, root)
        return os.environ.get('OMP_NUM_THREADS'), notes.getvalue()

    def test_first_term_sets_the_value(self):
        effective, notes = self._run(None, 5)
        self.assertEqual(effective, '5')
        self.assertEqual(notes, '')

    def test_later_term_keeps_it_and_says_so(self):
        effective, notes = self._run(3, 7)
        self.assertEqual(effective, '3',
                         'a later term rewrote a latched value')
        self.assertIn('OMP_NUM_THREADS is already 3', notes)
        self.assertIn('thread=7 is ignored', notes)


if __name__ == '__main__':
    unittest.main()
