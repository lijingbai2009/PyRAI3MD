"""
The two Slurm script builders must speak one dialect.

`qualification/manager.py::build_slurm_script` wraps a whole test
invocation; `workflow/dispatch/slurm_run.py::_build_script` wraps one
child runtime (and `term_worker` one backend term). They ship different
jobs from the same `slurm:` section, so the key names, the directive
spellings and the completion rules are shared (`kernel/sbatch.py`).
Before 2026-10-20 the qualification path passed each key through as
`--<key>=<value>`, so `cpus_per_task` became `--cpus_per_task` and
`extra` became `--extra=...`; the dispatch path mapped the same keys to
the canonical hyphens. A user moving between the two paths would have
had to restate the section.

The criteria below read both builders' own output, not the mapping
constant, so deleting the shared renderer from either caller is visible.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import unittest
from types import SimpleNamespace


def _qualification_script(section):
    from pyrai3md.qualification.manager import build_slurm_script
    return build_slurm_script(
        dict(section), {'OMP_NUM_THREADS': 4}, 'pyrai3md case.input')


def _dispatch_script(section):
    # The package exports a function of the same name, so importing the
    # module through the package attribute hands back the function.
    import importlib
    slurm_run = importlib.import_module(
        'pyrai3md.workflow.dispatch.slurm_run')
    environment = SimpleNamespace(
        slurm=dict(section), shell='', environment={})
    state = SimpleNamespace(pmd=SimpleNamespace(environment=environment))
    manager = SimpleNamespace(
        runtime_name=lambda idx: 'case_runtime_%d' % idx)
    return slurm_run._build_script(state, manager, 1)


def _directives(script):
    return [line for line in script.splitlines()
            if line.startswith('#SBATCH')]


SECTION = {
    'partition': 'testq',
    'nodes': 1,
    'ntasks': 2,
    'cpus_per_task': 4,
    'mem': '5G',
    'time': '0:37:00',
    'job_name': 'dialect',
    'output': 'slurm-%j.out',
    'error': 'slurm-%j.err',
    'exclusive': False,
    'extra': '#SBATCH --comment=dialect',
}

EXPECTED = [
    '#SBATCH --partition=testq',
    '#SBATCH --nodes=1',
    '#SBATCH --ntasks=2',
    '#SBATCH --cpus-per-task=4',
    '#SBATCH --mem=5G',
    '#SBATCH --time=0:37:00',
    '#SBATCH --job-name=dialect',
    '#SBATCH --output=slurm-%j.out',
    '#SBATCH --error=slurm-%j.err',
    '#SBATCH --comment=dialect',
]


class TestTheTwoBuildersSpeakOneDialect(unittest.TestCase):

    def test_both_builders_emit_the_same_directive_block(self):
        self.assertEqual(_directives(_qualification_script(SECTION)),
                         _directives(_dispatch_script(SECTION)),
                         'the test-shipping and child-runtime scripts asked '
                         'the cluster for different things')

    def test_the_directive_spellings_are_the_hyphenated_ones(self):
        for label, script in (('qualification',
                               _qualification_script(SECTION)),
                              ('dispatch', _dispatch_script(SECTION))):
            directives = _directives(script)
            for expected in EXPECTED:
                self.assertIn(expected, directives, label)
            self.assertFalse(
                [line for line in directives
                 if '--cpus_per_task' in line
                 or '--job_name' in line
                 or '--mem_per_cpu' in line],
                '%s kept a snake_case directive the cluster does not name: %s'
                % (label, directives))

    def test_extra_is_raw_and_never_a_directive_of_its_own(self):
        for label, script in (('qualification',
                               _qualification_script(SECTION)),
                              ('dispatch', _dispatch_script(SECTION))):
            directives = _directives(script)
            self.assertIn('#SBATCH --comment=dialect', directives, label)
            self.assertFalse([line for line in directives
                              if '--extra' in line], label)

    def test_a_false_boolean_writes_nothing_in_either_script(self):
        for label, script in (('qualification',
                               _qualification_script(SECTION)),
                              ('dispatch', _dispatch_script(SECTION))):
            self.assertFalse([line for line in _directives(script)
                              if '--exclusive' in line], label)

    def test_a_true_boolean_is_a_bare_flag_in_both_scripts(self):
        section = dict(SECTION, exclusive=True)
        for label, script in (('qualification',
                               _qualification_script(section)),
                              ('dispatch', _dispatch_script(section))):
            self.assertIn('#SBATCH --exclusive', _directives(script), label)
            self.assertFalse([line for line in _directives(script)
                              if '--exclusive=' in line], label)

    def test_every_mapped_key_reaches_both_scripts(self):
        # One key per section, so the mutually exclusive allocation keys
        # (`mem` / `mem_per_cpu`) do not have to be stated together.
        from pyrai3md.kernel.sbatch import SBATCH_OPTIONS
        for key, directive in SBATCH_OPTIONS.items():
            value = True if key == 'exclusive' else '%s-value' % key
            section = {key: value}
            expected = ('#SBATCH %s' % directive if value is True
                        else '#SBATCH %s=%s' % (directive, value))
            for label, script in (('qualification',
                                   _qualification_script(section)),
                                  ('dispatch', _dispatch_script(section))):
                self.assertIn(expected, _directives(script),
                              '%s lost slurm.%s' % (label, key))

    def test_an_unknown_key_is_refused_by_name_in_both_paths(self):
        section = {'memmory': '8G'}
        for label, builder in (('qualification', _qualification_script),
                               ('dispatch', _dispatch_script)):
            with self.assertRaises(ValueError) as raised:
                builder(section)
            message = str(raised.exception)
            self.assertIn('memmory', message, label)
            self.assertIn('cpus_per_task', message,
                          '%s does not name the known keys: %s'
                          % (label, message[:160]))


if __name__ == '__main__':
    unittest.main()
