"""
======================================================================
Guard that skill discovery imports only the query module.

SKILL_APP_DEVELOPMENT.md section 4.1: startup must never import a
skill's computational modules. Each check runs in a fresh interpreter
so an earlier test's imports cannot hide a regression.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import subprocess
import sys
import unittest

from pyrai3md.skills.query import list_skills

CODE = """
import importlib
import sys
importlib.import_module('pyrai3md.skills.{skill}.query')
assert 'pyrai3md.skills.{skill}.backend' not in sys.modules, \
    'importing {skill}.query loaded {skill}.backend'
"""


class SkillDiscoveryIsLazy(unittest.TestCase):
    def test_query_import_does_not_load_a_backend(self):
        for skill in list_skills():
            with self.subTest(skill=skill):
                completed = subprocess.run(
                    [sys.executable, '-c', CODE.format(skill=skill)],
                    capture_output=True, text=True)
                self.assertEqual(
                    completed.returncode, 0,
                    '%s: %s' % (skill, completed.stderr))


if __name__ == '__main__':
    unittest.main()
