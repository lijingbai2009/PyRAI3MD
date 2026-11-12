"""
Tests for the automatic initial orbital (title.StrOrb) channel.

The interface layer detects a {title}.StrOrb file next to the input
file and stores its path under raw_config['strorb_file']; the task
queue builder injects it into the molcas skill guess_orbital field
(fill-if-empty) so the skill stages it as RasOrb.

Run: pytest pyrai3md/qualification/verification/test_strorb.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import os
import tempfile
import unittest


class TestTheDetectorFollowsTheDeclaration(unittest.TestCase):
    """The interface resolves what a skill's manifest declares."""

    def _detect(self, raw_config, tmp, input_name='run.inp',
                strorb_names=()):
        from pyrai3md.interface.parser import _detect_initial_guess

        input_file = os.path.join(tmp, input_name)
        with open(input_file, 'w') as f:
            f.write('')
        for name in strorb_names:
            with open(os.path.join(tmp, name), 'w') as f:
                f.write('ORBITALS')
        raw_config.setdefault('control', {})['theory'] = ['molcas']
        return _detect_initial_guess(raw_config, input_file)

    def test_the_declared_suffix_is_resolved_from_the_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = self._detect({'control': {'title': 'myjob'}}, tmp,
                               strorb_names=('myjob.StrOrb',))
            expected = os.path.abspath(os.path.join(tmp, 'myjob.StrOrb'))
            self.assertEqual(raw['initial_guess']['path'], expected)
            self.assertEqual(raw['initial_guess']['field'], 'guess_orbital')
            self.assertEqual(raw['initial_guess']['skill'], 'molcas_skill')

    def test_the_input_stem_is_the_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = self._detect({'control': {'title': 'myjob'}}, tmp,
                               strorb_names=('run.StrOrb',))
            self.assertTrue(
                raw['initial_guess']['path'].endswith('run.StrOrb'))

    def test_the_title_candidate_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = self._detect(
                {'control': {'title': 'myjob'}}, tmp,
                strorb_names=('myjob.StrOrb', 'run.StrOrb'))
            self.assertTrue(
                raw['initial_guess']['path'].endswith('myjob.StrOrb'))

    def test_no_file_leaves_the_config_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = self._detect({}, tmp)
            self.assertNotIn('initial_guess', raw)

    def test_an_undeclared_skill_is_never_consulted(self):
        # Without the manifest block there is no suffix to look for, so
        # the file sits there untouched -- that is the negative control
        # for the declaration being the channel.
        from pyrai3md.skills import query
        with tempfile.TemporaryDirectory() as tmp:
            import pyrai3md.interface.parser as parser
            original = query.get_declaration
            query.get_declaration = lambda label, key: {}
            try:
                raw = self._detect({'control': {'title': 'myjob'}}, tmp,
                                   strorb_names=('myjob.StrOrb',))
            finally:
                query.get_declaration = original
        self.assertNotIn('initial_guess', raw)


class TestTheQueueFillsTheDeclaredField(unittest.TestCase):
    """`_build_task_queue` fills the field the declaration names."""

    @classmethod
    def setUpClass(cls):
        from pyrai3md.kernel.environment import Environment
        Environment().initialize()

    def _build_state(self, theory, raw_config):
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.config import PyRAIConfig

        config = PyRAIConfig()
        config.control.jobtype = 'energy'
        config.control.theory = list(theory)
        config.system.atoms = ['C', 'H']
        config.system.coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        config.system.natom = 2
        return build_state(config, raw_config=raw_config)

    @staticmethod
    def _guess(skill='molcas_skill', field='guess_orbital',
               path='/tmp/run.StrOrb'):
        return {'initial_guess': {'skill': skill, 'field': field,
                                  'path': path}}

    def test_the_declared_field_is_filled(self):
        state = self._build_state(('molcas',), self._guess())
        flat = state.theory[0]['config']['skill_config']
        self.assertEqual(flat['guess_orbital'], '/tmp/run.StrOrb')

    def test_an_explicit_value_precedes(self):
        raw = self._guess()
        raw['molcas'] = {'guess_orbital': '/explicit.StrOrb'}
        state = self._build_state(('molcas',), raw)
        flat = state.theory[0]['config']['skill_config']
        self.assertEqual(flat['guess_orbital'], '/explicit.StrOrb')

    def test_no_resolution_no_fill(self):
        state = self._build_state(('molcas',), {})
        flat = state.theory[0]['config']['skill_config']
        self.assertEqual(flat.get('guess_orbital', ''), '')

    def test_a_declaration_for_another_skill_is_not_applied(self):
        # The channel names the skill it belongs to, so a Molcas guess
        # cannot land in bagel's guess_orbital (which reads .archive).
        state = self._build_state(('bagel',), self._guess())
        flat = state.theory[0]['config']['skill_config']
        self.assertNotEqual(flat.get('guess_orbital', ''),
                            '/tmp/run.StrOrb')

    def test_a_second_skill_needs_no_framework_change(self):
        # The same declaration aimed at bagel's own field: zero
        # framework code names molcas, so the fill follows it.
        state = self._build_state(
            ('bagel',), self._guess(skill='bagel_skill',
                                    field='guess_orbital',
                                    path='/tmp/run.archive'))
        flat = state.theory[0]['config']['skill_config']
        self.assertEqual(flat['guess_orbital'], '/tmp/run.archive')


if __name__ == '__main__':
    unittest.main()
