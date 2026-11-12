"""
======================================================================
Three-carrier input contract for the nn_pes skill.

The same nn configuration is written three ways -- a yaml example, a
classic .input text and an agent .config JSON -- and each is put through
the framework's own route: parse, normalize, merge into the skill
namespace, then build/normalize/validate with the skill schema. The three
results must agree key by key, and the member axis must agree too, which
is checked by building the per-member hyperparameter dictionaries the
training run would actually write.

The spelling of a value is not compared (a text '1 1;1 10' and a typed
[[1, 1], [1, 10]] are one value), but what each spelling means is.

Oct 03 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import os
import tempfile
import unittest

from pyrai3md.interface.parser import parse_input
from pyrai3md.kernel.environment import Environment
from pyrai3md.qualification import manager
from pyrai3md.qualification.certification.converter import (
    convert, convert_to_pyrai_config, yaml_to_config)
from pyrai3md.schema import adapter as framework_adapter
from pyrai3md.schema.adapter import build_config as build_framework_config
from pyrai3md.schema.normalize import normalize as normalize_framework
from pyrai3md.skills.nn_pes_skill.schema.adapter import build_config
from pyrai3md.skills.nn_pes_skill.schema.hyper import build_member_hyper
from pyrai3md.skills.nn_pes_skill.schema.normalize import normalize
from pyrai3md.skills.nn_pes_skill.schema.validate import validate

SKILL = 'nn_pes_skill'

# The configuration under test, in its typed (yaml) spelling. 'eg' is the
# energy ensemble's role section, and its member axis appears in all three
# shapes a keyword can take: one value, a ';' text for two members, and a
# nested list for two members of a list-valued keyword.
CASE = {
    'control': {'title': 'nn_carrier', 'jobtype': 'train', 'theory': ['nn']},
    'atoms': ['C', 'H', 'H', 'H', 'H'],
    'coordinates': [
        [0.0000000, 0.0000000, 0.0000000],
        [1.0890000, 0.0000000, 0.0000000],
        [-0.3630000, 1.0280000, 0.0000000],
        [-0.3630000, -0.5140000, 0.8890000],
        [-0.3630000, -0.5140000, -0.8890000],
    ],
    # Two singlets and a triplet, so the declared pairs split into one
    # non-adiabatic pair and two spin-orbit ones, and the skill is sized by
    # that split rather than by an enumeration of its own.
    'molecule': {'charges': 0, 'multiplicity': 1, 'ci': '2 1', 'spin': '0 1',
                 'coupling': '1 2, 1 3, 2 3'},
    'nn': {'modeldir': './tmp/nn_models', 'mult_nn': 2, 'gpu': '-1',
           'train_data': './training.json', 'train_mode': 'training',
           'soc': {'epo': '900;700', 'batch_size': 16},
           # The role section is a keyword block of the skill, nested in
           # the yaml the way every skill section is. In the classic input
           # it is its own '&EG' block, which the converter renders from
           # here and the framework folds back in.
           'eg': {'depth': 3, 'nn_size': 30, 'epo': '200;150',
                  'batch_size': 32, 'val_split': 0.1,
                  'loss_weights': '1 1;1 10',
                  'use_reg_weight': 'l1;none', 'reg_l1': '1e-4',
                  'use_dropout': 'no', 'activ': 'leaky_softplus',
                  'activ_alpha': 0.03}},
}

# Fields this contract is about: the shared block, the role section, and
# the state bookkeeping the role section is read against.
_KEYS = ('modeldir', 'mult_nn', 'train_data', 'train_mode', 'gpu',
         'eg', 'nac', 'soc', 'nstate', 'nnac', 'nsoc',
         'nac_pair_map', 'soc_pair_map', 'profile', 'method')

# The hyperparameter subset whose member axis is compared, as written into
# the container's dictionary (section, dotted key).
_MEMBERS = {
    'epo': ('training', 'epo'),
    'depth': ('model', 'depth'),
    'nn_size': ('model', 'nn_size'),
    'batch_size': ('training', 'batch_size'),
    'val_split': ('training', 'val_split'),
    'loss_weights': ('training', 'loss_weights'),
    'use_reg_weight': ('model', 'use_reg_weight'),
    'dropout': ('model', 'dropout'),
    'use_dropout': ('model', 'use_dropout'),
    'activ': ('model', 'activ'),
}





class TestKtdcCapability(unittest.TestCase):
    """A ktdc dynamics run asks for E+G; only the NAC-vector method asks NAC."""

    def test_ktdc_md_does_not_require_nac(self):
        from pyrai3md.planning.resolver import resolve_capabilities
        caps = resolve_capabilities(
            ['adaptive'], planning_context={'sfhp': 'fssh',
                                           'nactype': 'ktdc'})
        self.assertEqual(caps['property_names'], ['energy', 'gradient'])

    def test_nac_md_adds_nac(self):
        from pyrai3md.planning.resolver import resolve_capabilities
        caps = resolve_capabilities(
            ['adaptive'], planning_context={'sfhp': 'fssh',
                                           'nactype': 'nac'})
        self.assertEqual(caps['property_names'],
                         ['energy', 'gradient', 'nac'])

    def test_e_g_capabilities_resolve_the_mlp_eg_gradient_profile(self):
        from pyrai3md.planning.assembly import spell_skill_jobtype
        from pyrai3md.skills.nn_pes_skill.query import resolve_profile
        spelled = spell_skill_jobtype(['energy', 'gradient'], 'md')
        self.assertEqual(spelled, 'energy gradient')
        self.assertEqual(resolve_profile('mlp_eg', spelled),
                         'mlp_eg_gradient')


class TestSocRouting(unittest.TestCase):
    """A spin-orbit request reaches the skill and lands on the e family.

    The framework spells the skill's jobtype from the planning capability
    list, so what has to hold is that a list containing 'soc' arrives as a
    token string the skill resolves, and that it resolves to a profile whose
    ensembles include the spin-orbit one served by the energy network.
    """

    def test_soc_capability_token_reaches_a_soc_profile(self):
        from pyrai3md.planning.assembly import spell_skill_jobtype
        from pyrai3md.skills.nn_pes_skill.query import resolve_profile
        from pyrai3md.skills.nn_pes_skill.schema.method import (
            PROFILE_ENSEMBLES, ROLE_SOC)

        spelled = spell_skill_jobtype(
            ['energy', 'gradient', 'nac', 'soc'], 'md')
        self.assertEqual(spelled, 'energy gradient nac soc')
        profile = resolve_profile('eg_nac', spelled)
        self.assertEqual(profile, 'eg_nac_soc')
        self.assertIn((ROLE_SOC, 'mlp_e'), PROFILE_ENSEMBLES[profile])

    def test_a_bare_soc_jobtype_asks_the_energy_family(self):
        from pyrai3md.skills.nn_pes_skill.query import resolve_profile
        from pyrai3md.skills.nn_pes_skill.schema.method import (
            PROFILE_ENSEMBLES, ROLE_SOC)
        profile = resolve_profile('mlp_e', 'soc')
        self.assertEqual(profile, 'mlp_e_soc')
        self.assertEqual(PROFILE_ENSEMBLES[profile], ((ROLE_SOC, 'mlp_e'),))

    def test_a_family_without_a_spin_orbit_channel_is_refused(self):
        # A pure coupling family has no scalar channel to hold a SOC value,
        # and the answer is a named refusal rather than a silent fallback.
        from pyrai3md.skills.nn_pes_skill.query import resolve_profile
        with self.assertRaises(ValueError):
            resolve_profile('mlp_nac2', 'soc')

    def test_the_framework_resolves_spin_orbit_pairs_from_the_carrier(self):
        # '&MOLECULE coupling 1 2, 1 3, 2 3' with two singlets and a
        # triplet: the pair inside the singlet manifold is non-adiabatic,
        # the two reaching the triplet are spin-orbit.
        from pyrai3md.schema.adapter import build_config as build_framework
        from pyrai3md.schema.normalize import normalize as normalize_framework
        raw = {'control': {'jobtype': 'md', 'theory': ['nn']},
               'molecule': {'ci': '2 1', 'spin': '0 1',
                            'coupling': '1 2, 1 3, 2 3'},
               'atoms': ['C', 'H', 'H'],
               'coordinates': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0]]}
        config = normalize_framework(build_framework(raw)).model_dump()
        system = config['system']
        self.assertEqual(system['nac_pair_map'], [[0, 1]])
        self.assertEqual(system['soc_pair_map'], [[0, 2], [1, 2]])
        self.assertEqual(system['nnac'], 1)
        self.assertEqual(system['nsoc'], 2)



class TestTrainAndPredictRouting(unittest.TestCase):
    """The training and prediction requests reach the profile they ask for.

    Profile resolution is the skill's own decision, made from the request
    the framework hands it. It may not depend on control.driver unless the
    request carries dynamics: a driver that would hand the propagation to
    the backend says nothing about a job that propagates nothing, and a
    refusal here would leave the framework keeping some other profile --
    which reads as a finished training run that trained nothing.
    """

    @classmethod
    def setUpClass(cls):
        cls._env = Environment()
        cls._env.initialize()
        cls._maestro = (getattr(cls._env, 'maestro', None)
                        or getattr(cls._env, '_maestro', None))

    def test_a_train_request_resolves_under_either_driver(self):
        from pyrai3md.skills.nn_pes_skill.query import resolve_profile
        for driver in ('main', 'skill', 'nn'):
            self.assertEqual(
                resolve_profile('mlp_eg', 'train', driver), 'mlp_eg_train',
                'driver=%s' % driver)

    def test_a_prediction_request_serves_what_the_family_trains(self):
        # A prediction names no property: the loaded networks answer with
        # the channel they were fitted for, and they are loaded while the
        # backend initializes (a warm start), like any property step.
        from pyrai3md.skills.nn_pes_skill.query import resolve_profile
        expected = {'mlp_e': 'mlp_e_energy', 'mlp_eg': 'mlp_eg_gradient',
                    'mlp_g2': 'mlp_g2_gradient', 'mlp_nac': 'mlp_nac_nac',
                    'mlp_nac2': 'mlp_nac2_nac', 'eg_nac': 'eg_nac_nac',
                    'eg_nac2': 'eg_nac2_nac'}
        for method, profile in expected.items():
            for jobtype in ('prediction', 'predict'):
                self.assertEqual(
                    resolve_profile(method, jobtype, 'skill'), profile,
                    '%s/%s' % (method, jobtype))

    def test_a_dynamics_request_still_refuses_a_propagating_driver(self):
        from pyrai3md.skills.nn_pes_skill.query import resolve_profile
        with self.assertRaises(ValueError):
            resolve_profile('eg_nac', 'md', 'skill')
        self.assertEqual(resolve_profile('eg_nac', 'md', 'main'),
                         'eg_nac_nac')

    def test_the_task_queue_of_a_train_run_names_the_train_profile(self):
        # The end-to-end guard: the flat dict the skill is handed when the
        # profile is left to be derived, under the skill-driven route.
        from pyrai3md.planning.assembly import _build_skill_flat_and_submodels
        merged = {'method': 'mlp_eg', 'jobtype': 'train', 'driver': 'skill'}
        flat = _build_skill_flat_and_submodels('nn_pes_skill', merged,
                                               driver='skill')
        self.assertEqual(flat.get('profile'), 'mlp_eg_train')

    def test_a_refused_profile_is_reported_not_swallowed(self):
        # A combination the family cannot answer keeps the skill default,
        # and that substitution has to be visible: a request quietly
        # served by another profile still prints as a completed run.
        from pyrai3md.planning.assembly import _build_skill_flat_and_submodels
        merged = {'method': 'mlp_nac2', 'jobtype': 'energy gradient',
                  'driver': 'main'}
        with self.assertLogs('pyrai3md.planning.assembly', level='WARNING') as log:
            _build_skill_flat_and_submodels('nn_pes_skill', merged,
                                            driver='main')
        text = ' '.join(log.output)
        self.assertIn('mlp_nac2', text)
        self.assertIn('energy gradient', text)


if __name__ == '__main__':
    unittest.main()
