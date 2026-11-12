"""
======================================================================
Test suite for the process-global state a backend term must not touch
(workflow/dispatch/terms.py, term axis).

Two terms of one skill run in threads of one interpreter. Anything the
skill moves for the whole process therefore belongs to both of them at
once: the working directory (os.chdir), and the command line the runner
resolves against it. Before #301 every external-program runner did
`os.chdir(calcdir)`, spawned its script by bare relative name and
restored the directory afterwards; a second term churning the same
process cwd made the first term's script resolve in the wrong
directory. The runners now anchor the child with cwd= and pass an
absolute script path, which changes nothing for the child and nothing
for the process.

Tests cover:
- mechanism: an AST scan of every skill runner module shows no chdir
  and one cwd= on every subprocess call;
- causal control: the same fixture resolved by relative name while
  another term has moved the cwd lands in the wrong directory, so the
  guard below is not vacuous;
- production path: xtb_skill.run_local executed from two threads
  records its own calculation directory in each of them;
- ownership: thread safety is decided by the skill
  (backend.check_thread_safe), so no framework-side capability table
  may come back.

Oct 04 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import ast
import os
import subprocess
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor


from pyrai3md.qualification.verification._source_paths import (
    relative_to_source, source_read, source_tree)

RUNNER_GLOB = 'pyrai3md/skills/*_skill/service/runner/*_runner.py'


def _call_nodes(source):
    """Yield every subprocess call node of one module source."""
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, 'attr', None) or getattr(func, 'id', None)
        if name in ('run', 'Popen', 'call', 'check_call', 'check_output',
                    'check_run'):
            yield node


def _chdir_calls(tree):
    """Return the line numbers of every chdir call in a tree."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (getattr(node.func, 'attr', None)
                    or getattr(node.func, 'id', None))
            if name == 'chdir':
                lines.append(node.lineno)
    return lines


class TestRunnerModulesKeepTheProcessCwd(unittest.TestCase):
    """No skill runner moves the process cwd; every child is anchored."""

    def _modules(self):
        paths = source_tree(RUNNER_GLOB)
        self.assertTrue(paths, 'no runner module matched %s' % RUNNER_GLOB)
        return paths

    def test_runner_modules_never_chdir(self):
        offenders = []
        for path in self._modules():
            with open(path, encoding='utf-8') as handle:
                tree = ast.parse(handle.read(), filename=path)
            for line in _chdir_calls(tree):
                offenders.append('%s:%d' % (path, line))
        self.assertEqual(
            [], offenders,
            'os.chdir moves the cwd of every other term in the process')

    def test_every_subprocess_call_anchors_its_cwd(self):
        offenders = []
        for path in self._modules():
            with open(path, encoding='utf-8') as handle:
                tree = ast.parse(handle.read(), filename=path)
            for node in _call_nodes(tree):
                keywords = {kw.arg for kw in node.keywords}
                if 'cwd' not in keywords:
                    offenders.append('%s:%d' % (path, node.lineno))
        self.assertEqual(
            [], offenders,
            'a child without cwd= inherits a cwd another term may move')


class TestProcessGlobalsStayUntouched(unittest.TestCase):
    """No skill writes a process-wide interpreter setting per term.

    #302 removed the last two: a per-term ``sys.stdout`` swap (pyscf) and
    a per-term ``os.environ['OMP_NUM_THREADS']`` write (pyscf, qdm). Both
    are read by every thread in the process, so a term cannot own them;
    the pyscf runner now binds its tee to the thread and both skills set
    the thread count only if the process has not chosen one yet.
    """

    def _modules(self):
        paths = source_tree('pyrai3md/skills/*_skill/**/*.py')
        return [path for path in paths
                if os.sep + 'tests' + os.sep not in path]

    def _offenders(self, source):
        """Lines that assign sys.stdout or an os.environ item."""
        tree = ast.parse(source)
        lines = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute) and \
                        getattr(target.value, 'id', '') == 'sys' and \
                        target.attr in ('stdout', 'stderr'):
                    lines.append((node.lineno, 'sys.%s' % target.attr))
                elif isinstance(target, ast.Subscript) and \
                        isinstance(target.value, ast.Attribute) and \
                        getattr(target.value.value, 'id', '') == 'os' and \
                        target.value.attr == 'environ':
                    lines.append((node.lineno, 'os.environ[...]'))
        return lines

    def test_no_skill_reassigns_process_streams(self):
        offenders = []
        for path in self._modules():
            with open(path, encoding='utf-8') as handle:
                source = handle.read()
            for line, kind in self._offenders(source):
                if kind.startswith('sys.'):
                    offenders.append('%s:%d %s' % (path, line, kind))
        self.assertEqual([], offenders,
                         'a process-wide stream swap is not per-term: '
                         'bind it to the thread instead')

    def test_no_skill_writes_environ_per_term(self):
        offenders = []
        for path in self._modules():
            with open(path, encoding='utf-8') as handle:
                source = handle.read()
            for line, kind in self._offenders(source):
                if kind.startswith('os.environ'):
                    offenders.append('%s:%d %s' % (path, line, kind))
        self.assertEqual([], offenders,
                         'setdefault is the first-wins form; a write '
                         'retunes what OpenMP already latched for every '
                         'other thread')

    def test_the_scan_catches_the_retired_shapes(self):
        """Causal control: the two pre-#302 forms are not invisible."""
        old = ('def run_local(config, workflow, log_file):\n'
               '    sys.stdout = _Tee(log_file, sys.stdout)\n'
               '    os.environ["OMP_NUM_THREADS"] = str(config.thread)\n')
        found = self._offenders(old)
        self.assertEqual([(2, 'sys.stdout'), (3, 'os.environ[...]')], found)


class TestAnchoringHasTeeth(unittest.TestCase):
    """The guard is not vacuous: relative resolution really does move."""

    def _make_pair(self, root):
        """Two term directories, each with a script that marks its cwd."""
        dirs = {}
        for name in ('term_a', 'term_b'):
            path = os.path.join(root, name)
            os.makedirs(path)
            with open(os.path.join(path, 'case.sh'), 'w',
                      encoding='utf-8') as handle:
                handle.write('#!/bin/bash\npwd > cwd.txt\n')
            dirs[name] = path
        return dirs

    def test_anchored_call_ignores_the_process_cwd(self):
        with tempfile.TemporaryDirectory() as root:
            dirs = self._make_pair(root)
            original = os.getcwd()
            os.chdir(dirs['term_b'])
            try:
                subprocess.run(
                    ['bash', os.path.join(dirs['term_a'], 'case.sh')],
                    capture_output=False, cwd=dirs['term_a'])
            finally:
                os.chdir(original)

            with open(os.path.join(dirs['term_a'], 'cwd.txt'),
                      encoding='utf-8') as handle:
                self.assertTrue(
                    _same_path(handle.read().strip(), dirs['term_a']),
                    'the anchored child ran in %s' % handle.read().strip())
            self.assertFalse(os.path.exists(
                os.path.join(dirs['term_b'], 'cwd.txt')))

    def test_control_relative_name_resolves_in_the_moved_cwd(self):
        """The pre-#301 shape, reproduced: the wrong term is run.

        This is the causal control for the two guards above. Building the
        same call the old runner built (bare 'case.sh', process cwd moved
        by the other term) makes term_a's marker land in term_b, which is
        exactly the failure the anchored form prevents.
        """
        with tempfile.TemporaryDirectory() as root:
            dirs = self._make_pair(root)
            original = os.getcwd()
            os.chdir(dirs['term_a'])
            try:
                os.chdir(dirs['term_b'])          # the other term moves it
                # The old runner handed bash a bare 'case.sh'. It does not
                # fail: it finds the *other* term's script and runs it,
                # which is worse than an error.
                subprocess.run(['bash', 'case.sh'], capture_output=False,
                               check=True)
            finally:
                os.chdir(original)

            # term_b's script ran instead: its marker exists, term_a's
            # does not. The relative form is not term-local.
            self.assertTrue(os.path.exists(
                os.path.join(dirs['term_b'], 'cwd.txt')))
            self.assertFalse(os.path.exists(
                os.path.join(dirs['term_a'], 'cwd.txt')))


class TestConcurrentTermsOfOneSkill(unittest.TestCase):
    """Two terms of xtb_skill in one thread pool keep their own dirs."""

    def _run_two_terms(self):
        from pyrai3md.skills.xtb_skill.service.runner import local_runner
        from pyrai3md.skills.xtb_skill.service.templates import run_templates

        root = tempfile.mkdtemp()
        self.addCleanup(lambda: _rmtree(root))
        dirs = {name: os.path.join(root, name)
                for name in ('qm', 'qm2')}
        for path in dirs.values():
            os.makedirs(path)

        class _StubConfig:
            xtb_project = 'case'

        barrier = threading.Barrier(2, timeout=10)

        def fake_script(config):
            # Both terms are inside the runner at the same time before
            # their child is spawned, so any process-wide move would be
            # observed by the other one.
            barrier.wait()
            return '#!/bin/bash\nsleep 0.2\npwd > cwd.txt\n'

        original = os.getcwd()
        with _patched(run_templates, 'build_local_script', fake_script):
            with ThreadPoolExecutor(max_workers=2) as pool:
                codes = list(pool.map(
                    lambda path: local_runner.run_local(_StubConfig(), path),
                    list(dirs.values())))

        self.assertEqual([0, 0], codes)
        self.assertEqual(os.getcwd(), original,
                         'the runner moved the cwd of the whole process')
        for name, path in dirs.items():
            marker = os.path.join(path, 'cwd.txt')
            self.assertTrue(os.path.exists(marker),
                            '%s recorded nothing' % name)
            with open(marker, encoding='utf-8') as handle:
                recorded = handle.read().strip()
            self.assertTrue(
                _same_path(recorded, path),
                '%s ran in %s instead of its own directory' % (name,
                                                               recorded))

    def test_two_terms_keep_their_own_directories(self):
        self._run_two_terms()


class TestThreadSafetyOwnership(unittest.TestCase):
    """The capability decision lives in the skill, not in the framework."""

    def test_framework_keeps_no_capability_table(self):
        # Spelled in pieces so this guard does not match itself.
        retired = ('THREAD_AUDIT' + '_REGISTRY',
                   'register_' + 'thread_safe')
        here = os.path.abspath(__file__)
        for path in source_tree('pyrai3md/**/*.py'):
            if os.path.abspath(path) == here:
                continue
            with open(path, encoding='utf-8') as handle:
                source = handle.read()
            for name in retired:
                self.assertNotIn(
                    name, source,
                    '%s still mentions the framework-side capability '
                    'registry %s' % (path, name))

    def test_the_refusal_sites_are_the_skills(self):
        owners = source_tree('pyrai3md/skills/*_skill/backend.py')
        refusals = [relative_to_source(path) for path in owners
                   if 'def check_thread_safe' in _read(path)]
        # #302 moved the two process-wide dependencies pyscf and qdm used
        # to mutate (a sys.stdout tee installed process-wide, a per-term
        # OMP_NUM_THREADS write), so both now run their terms
        # concurrently. nn_pes is the one skill left that still declines,
        # and only for the profile that retrains TensorFlow.
        self.assertEqual(['pyrai3md/skills/nn_pes_skill/backend.py'],
                         refusals)
        for path in refusals:
            self.assertRegex(
                _read(path), r'[A-Za-z_/]+\.py:\d',
                '%s refuses without naming the code that moves process '
                'state' % path)


class _patched:
    """Context manager patching one module attribute without mock."""

    def __init__(self, module, name, value):
        self._module = module
        self._name = name
        self._value = value

    def __enter__(self):
        self._old = getattr(self._module, self._name)
        setattr(self._module, self._name, self._value)
        return self._value

    def __exit__(self, *exc):
        setattr(self._module, self._name, self._old)
        return False


def _same_path(left, right):
    """Compare two paths the way the shell reports them (macOS /var)."""
    return os.path.realpath(left) == os.path.realpath(right)


def _read(path):
    return source_read(path)


def _rmtree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
