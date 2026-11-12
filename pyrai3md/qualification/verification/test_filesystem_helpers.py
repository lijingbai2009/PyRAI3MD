"""
======================================================================
Unit tests for the filesystem helpers (pyrai3md/utilities/filesystem.py).

These three functions used to be exported with `pass` bodies, so every
caller that trusted them got a broken no-op: `ensure_dir` created
nothing, `safe_copy` copied nothing, and `working_directory` raised
`TypeError: 'NoneType' object is not an iterator` the moment anyone
entered it. The tests below are the criterion that the helpers do what
their docstrings state, and that the directory change is restored on the
failure path too.

Nov 6 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import shutil
import tempfile
import time
import unittest

from pyrai3md.utilities.filesystem import ensure_dir, safe_copy
from pyrai3md.utilities.filesystem import working_directory


class FilesystemCase(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='filesystem_unit_')
        self.addCleanup(shutil.rmtree, self.root, True)


class TestEnsureDir(FilesystemCase):

    def test_a_missing_tree_is_created_and_the_path_is_returned(self):
        target = os.path.join(self.root, 'a', 'b', 'c')
        self.assertEqual(ensure_dir(target), target)
        self.assertTrue(os.path.isdir(target))

    def test_an_existing_directory_is_not_an_error(self):
        ensure_dir(self.root)
        self.assertEqual(ensure_dir(self.root), self.root)

    def test_an_empty_path_creates_nothing(self):
        self.assertEqual(ensure_dir(''), '')


class TestSafeCopy(FilesystemCase):

    def test_the_content_and_the_metadata_travel(self):
        src = os.path.join(self.root, 'src.txt')
        with open(src, 'w') as handle:
            handle.write('payload')
        os.utime(src, (time.time() - 4000, time.time() - 4000))
        dst = os.path.join(self.root, 'dst.txt')
        self.assertEqual(safe_copy(src, dst), dst)
        with open(dst) as handle:
            self.assertEqual(handle.read(), 'payload')
        self.assertAlmostEqual(os.path.getmtime(src), os.path.getmtime(dst),
                               places=2)

    def test_a_missing_source_names_the_pair_it_failed_on(self):
        with self.assertRaises(OSError) as caught:
            safe_copy(os.path.join(self.root, 'absent'),
                      os.path.join(self.root, 'dst'))
        message = str(caught.exception)
        self.assertIn('absent', message)
        self.assertIn('dst', message)


class TestWorkingDirectory(FilesystemCase):

    def test_the_body_runs_in_the_requested_directory(self):
        before = os.getcwd()
        with working_directory(self.root) as previous:
            self.assertEqual(os.path.realpath(os.getcwd()),
                             os.path.realpath(self.root))
            self.assertEqual(previous, before)
        self.assertEqual(os.getcwd(), before)

    def test_an_exception_still_restores_the_previous_directory(self):
        before = os.getcwd()
        with self.assertRaises(RuntimeError):
            with working_directory(self.root):
                raise RuntimeError('the body failed')
        self.assertEqual(os.getcwd(), before)


if __name__ == '__main__':
    unittest.main()
