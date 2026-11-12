"""
======================================================================
The driver contract of every skill's `query.resolve_profile` (#466).

`_build_skill_flat_and_submodels` asks the skill which profile a
jobtype maps to and passes the driver, because the reverse contract
says a skill driver owns an iteration jobtype: a profile table with no
MD row must refuse one rather than answer with its single-point
fallback. Seven of the ten skills used to answer a two-argument
signature, so the call raised TypeError and the framework retried
without the driver -- the refusal could not bind there at all.

Measured after the third parameter landed everywhere (jobtype 'md',
driver 'skill'):

- refused by their own table: orca, cp2k, pyscf, bagel, lammps
  (`md` resolves to a `*_gradient` profile), nn_pes (no `_nac` profile
  for a dynamics request) and xtb (a dynamics request under a skill
  driver);
- answered with a real dynamics profile: molcas
  (`casscf_abinitio_dynamics`), openqp
  (`mrsf_nonadiabatic_dynamics`) and qdm (`tully1_reference`), which
  is what the skill runs when it owns the propagation.

The framework no longer carries a fallback: a TypeError from this call
is a defect now, not a normal path.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import importlib
import inspect
import unittest

from pyrai3md.planning.assembly import _build_skill_flat_and_submodels
from pyrai3md.qualification.verification._source_paths import source_read

# (skill package, method, behaviour under driver='skill')
CONTRACT = {
    'bagel_skill': ('casscf', 'refuse'),
    'cp2k_skill': ('dft', 'refuse'),
    'lammps_skill': ('ff', 'refuse'),
    'nn_pes_skill': ('mlp_nac', 'refuse'),
    'orca_skill': ('dft', 'refuse'),
    'pyscf_skill': ('hf', 'refuse'),
    'xtb_skill': ('gfn2', 'refuse'),
    'molcas_skill': ('casscf', 'answer'),
    'openqp_skill': ('mrsf', 'answer'),
    'qdm_skill': ('tully1', 'answer'),
}


def _query(skill):
    return importlib.import_module('pyrai3md.skills.%s.query' % skill)


class TestEverySkillSeesTheDriver(unittest.TestCase):
    """One signature, three parameters, 'main' by default."""

    def test_every_query_takes_the_driver(self):
        for skill in CONTRACT:
            with self.subTest(skill):
                parameters = inspect.signature(
                    _query(skill).resolve_profile).parameters
                self.assertEqual(list(parameters)[:3],
                                 ['method', 'jobtype', 'driver'])
                self.assertEqual(parameters['driver'].default, 'main')

    def test_the_framework_has_no_two_argument_fallback(self):
        import ast
        text = source_read('pyrai3md/planning/assembly.py')
        self.assertNotIn('_warn_two_argument_profile_signature', text)
        self.assertNotIn('_TWO_ARGUMENT_PROFILE_SIGNATURES', text)
        calls = [node for node in ast.walk(ast.parse(text))
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr == 'resolve_profile']
        self.assertEqual(len(calls), 1)
        self.assertEqual([argument.id for argument in calls[0].args],
                         ['method', 'jobtype', 'driver'])


class TestTheDynamicsRequestUnderASkillDriver(unittest.TestCase):
    """Each skill either answers 'md' or refuses it; none falls back."""

    def test_the_measured_table_holds(self):
        for skill, (method, behaviour) in CONTRACT.items():
            with self.subTest(skill):
                resolve = _query(skill).resolve_profile
                main = resolve(method, 'md', 'main')
                self.assertTrue(main)
                if behaviour == 'refuse':
                    with self.assertRaises(ValueError):
                        resolve(method, 'md', 'skill')
                else:
                    # The skill owns the propagation and answers with its
                    # own dynamics profile (pinned below by name).
                    self.assertTrue(resolve(method, 'md', 'skill'))

    def test_an_answering_skill_names_its_own_dynamics_profile(self):
        with self.subTest('molcas'):
            self.assertEqual(
                _query('molcas_skill').resolve_profile('casscf', 'md',
                                                       'skill'),
                'casscf_abinitio_dynamics')
        with self.subTest('openqp'):
            self.assertEqual(
                _query('openqp_skill').resolve_profile('mrsf', 'md',
                                                       'skill'),
                'mrsf_nonadiabatic_dynamics')
        with self.subTest('qdm'):
            self.assertEqual(
                _query('qdm_skill').resolve_profile('tully1', 'md',
                                                    'skill'),
                'tully1_reference')


class TestTheFrameworkPropagatesTheRefusal(unittest.TestCase):
    """A refused dynamics request reaches the caller unchanged."""

    def test_a_skill_driver_md_request_raises(self):
        with self.assertRaises(ValueError) as raised:
            _build_skill_flat_and_submodels(
                'orca_skill',
                {'method': 'dft', 'jobtype': 'md', 'profile': ''},
                driver='skill')
        self.assertIn('does not run the dynamics itself',
                      str(raised.exception))

    def test_the_same_request_under_main_resolves(self):
        merged = _build_skill_flat_and_submodels(
            'orca_skill',
            {'method': 'dft', 'jobtype': 'md', 'profile': ''},
            driver='main')
        self.assertEqual(merged['profile'], 'dft_gradient')


class TestTheIterationTokenSurvivesTheSkillDriver(unittest.TestCase):
    """A skill-driven iteration must reach the skill as its own token.

    `md` under driver='skill' asks the skill to own the propagation; the
    skill's (jobtype, driver) table answers `{method}_reference`. Spelling
    the capability set instead handed it `energy gradient`, so the qdm
    reference leg ran a gradient profile and the benchmark's
    `*_population.dat` was never written (#467).
    """

    def test_md_keeps_its_token_under_a_skill_driver(self):
        from types import SimpleNamespace

        from pyrai3md.planning.assembly import _spell_advanced_jobtype

        self.assertFalse(_spell_advanced_jobtype(
            SimpleNamespace(jobtype='md', driver='skill')))
        self.assertTrue(_spell_advanced_jobtype(
            SimpleNamespace(jobtype='md', driver='main')))
        self.assertFalse(_spell_advanced_jobtype(
            SimpleNamespace(jobtype='optimize', driver='skill')))
        self.assertTrue(_spell_advanced_jobtype(
            SimpleNamespace(jobtype='optimize', driver='main')))


if __name__ == '__main__':
    unittest.main()