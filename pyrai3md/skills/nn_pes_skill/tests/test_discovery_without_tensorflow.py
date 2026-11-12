"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Discovery must survive a machine without the optional neural network
stack.

Framework discovery imports only each skill's query module, and
'pyrai3md/skills/query.py' wraps that import in 'except ImportError:
pass'. If this skill imported TensorFlow anywhere on that path, the skill
would vanish silently on a machine that has no TensorFlow — the worst
possible failure, because it looks like the skill was never installed.

These checks run in a child interpreter with 'tensorflow' made
unimportable, and assert both that the skill is still discovered and that
using it fails with a named reason.

"""

import subprocess
import sys
import unittest

BLOCK = r"""
import sys


class _Blocker:
    def find_module(self, name, path=None):
        return self if name == 'tensorflow' else None

    def find_spec(self, name, path=None, target=None):
        if name == 'tensorflow':
            raise ImportError('tensorflow is blocked by this test')
        return None


sys.meta_path.insert(0, _Blocker())

try:
    import tensorflow  # noqa: F401
    print('BLOCK_FAILED')
except ImportError:
    pass

from pyrai3md.skills import query
registry = query.discover_all()
print('DISCOVERED' if 'nn' in registry else 'MISSING')
print('SKILL_NAME=%s' % registry.get('nn', {}).get('skill_name'))
print('REQUIRES=%s' % registry.get('nn', {}).get('requires'))

import pyrai3md.skills.nn_pes_skill as package
print('PACKAGE_IMPORT_OK')
print('CLASS=%s' % package.NNPESBackend.__name__)

try:
    package.NNPESBackend({'profile': 'mlp_eg_gradient', 'method': 'mlp_eg',
                          'modeldir': '/tmp/none', 'atoms': ['C', 'H'],
                          'coordinates': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                          'nstate': 2, 'scratch_path': ''})
    print('NO_ERROR')
except RuntimeError as error:
    print('RUNTIME_ERROR=%s' % str(error).split(';')[0])
except Exception as error:
    print('WRONG_ERROR=%s' % type(error).__name__)
"""


class TestDiscoveryWithoutTensorFlow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.output = subprocess.run(
            [sys.executable, '-c', BLOCK], capture_output=True, text=True,
            cwd=None)

    def lines(self):
        return self.output.stdout.splitlines()

    def test_child_ran(self):
        self.assertEqual(self.output.returncode, 0,
                         msg=self.output.stderr[-800:])

    def test_blocker_was_effective(self):
        self.assertNotIn('BLOCK_FAILED', self.lines())

    def test_skill_is_discovered(self):
        self.assertIn('DISCOVERED', self.lines())
        self.assertIn('SKILL_NAME=nn_pes_skill', self.lines())

    def test_requirement_is_visible_to_the_planner(self):
        self.assertIn("REQUIRES=['tensorflow']", self.lines())

    def test_package_and_backend_class_import(self):
        self.assertIn('PACKAGE_IMPORT_OK', self.lines())
        self.assertIn('CLASS=NNPESBackend', self.lines())

    def test_use_fails_with_a_named_reason(self):
        self.assertIn('RUNTIME_ERROR=tensorflow is not installed',
                      self.lines())


if __name__ == '__main__':
    unittest.main()
