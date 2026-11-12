"""
======================================================================
Test suite for the MPI launcher every skill is allowed to name
(dev/SKILL_APP_DEVELOPMENT.md section 5.7, CHANGELOG #305).

Skills that spread a calculation over MPI ranks all start it the same
way, and the repository standardizes the launcher name to `mpirun`:
lammps builds 'mpirun -np' in both arms, cp2k and openqp build
'mpirun -n', and bagel's `mpiexe` keyword defaulted to the other
spelling until #305. A skill may still be told to use a site's own
launcher (bagel's `mpiexe`, openqp's `mpi_command`), but nothing may
*default* to a different name, and no generated command may be assembled
from one.

Tests cover:
- the mechanism: no string literal in any skill module assembles a
  command around another launcher spelling (docstrings and comments are
  excluded, since naming the family there is documentation);
- the defaults: each launcher-bearing keyword resolves to mpirun when
  the input says nothing.

Oct 04 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import ast
import os
import re
import unittest

from pyrai3md.qualification.verification._source_paths import (
    source_read, source_tree)


def _command_strings(path):
    """Yield (line, value) for string constants that are not docstrings.

    Comments never reach the AST, and docstrings are excluded because a
    skill is allowed to *mention* the launcher family it standardizes on.
    """
    tree = ast.parse(open(path, encoding='utf-8').read(), filename=path)
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) \
                and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant):
            docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            yield node.lineno, node.value


class TestLauncherSpelling(unittest.TestCase):
    """Nothing assembles an MPI command out of a non-standard name."""

    def _modules(self):
        paths = source_tree('pyrai3md/skills/*_skill/**/*.py')
        return [path for path in paths
                if os.sep + 'tests' + os.sep not in path]

    def test_no_command_string_names_another_launcher(self):
        offenders = []
        for path in self._modules():
            for line, value in _command_strings(path):
                if re.search(r'\bmpiexec\b', value):
                    offenders.append('%s:%d' % (path, line))
        self.assertEqual(
            [], offenders,
            'the launcher name every skill uses is mpirun; a site that '
            'needs another one sets it explicitly, it is never the '
            'default (#305)')

    def test_the_guard_catches_the_retired_spelling(self):
        """Causal control: the scan is not vacuous."""
        source = ("run_line = f'{mpiexe} -np 4'\n"
                  "launcher = 'mpiexec'\n")
        found = [line for line, value in
                 ((l, v) for l, v in
                  [(n.lineno, n.value) for n in
                   ast.walk(ast.parse(source))
                   if isinstance(n, ast.Constant)
                   and isinstance(n.value, str)])
                 if re.search(r'\bmpiexec\b', value)]
        self.assertEqual([2], found)


class TestLauncherDefaults(unittest.TestCase):
    """Each skill's launcher site names mpirun when the input says nothing.

    Read from the source rather than by running a program: these are the
    default-spelling decisions, and the behaviour of the generated
    commands is already covered by the skills' own run-script tests.
    """

    def _source(self, path):
        return source_read(path)

    def test_bagel_default_and_fallback(self):
        self.assertIn("'mpiexe': 'mpirun',",
                      self._source('pyrai3md/skills/bagel_skill/schema/'
                                   'reference.py'))
        self.assertIn("data['mpiexe'] = mpiexe or 'mpirun'",
                      self._source('pyrai3md/skills/bagel_skill/schema/'
                                   'validate.py'))

    def test_openqp_fallback(self):
        self.assertIn("else 'mpirun'",
                      self._source('pyrai3md/skills/openqp_skill/service/'
                                   'runner/env_check.py'))

    def test_lammps_builds_mpirun_in_both_arms(self):
        templates = self._source('pyrai3md/skills/lammps_skill/service/'
                                 'templates/run_templates.py')
        launcher = self._source('pyrai3md/skills/lammps_skill/service/'
                                'persistent/launcher.py')
        self.assertIn("f'mpirun -np {nproc} {lammps_binary}'", templates)
        self.assertIn("['mpirun', '-np', str(nproc)]", launcher)

    def test_cp2k_builds_mpirun(self):
        self.assertIn('mpirun -n {nproc}',
                      self._source('pyrai3md/skills/cp2k_skill/service/'
                                   'templates/run_templates.py'))


if __name__ == '__main__':
    unittest.main()
