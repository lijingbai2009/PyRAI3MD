"""
======================================================================
Cross-side contracts that used to live as framework imports inside
skill tests.

A skill may not import the framework, and the rule is enforced by a
source guard over the skill tree (test_skill_framework_guard.py), so a
criterion that compares the skill's copy of a contract with the
framework's own values has to live on the framework side. Each class
below is one such contract, moved with its body unchanged:

- the unit constants the nn_pes skill carries (utilities.units);
- the capability names its manifest declares (planning.resolver);
- the training-report grammar it prints (utilities.print_results);
- the cp2k handoff through the framework merge (_merge_one_skill_config).

The skill side of each pair keeps its own tests; these only read the
skill through its public modules, which a framework test may do.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import os
import tempfile
import unittest

from pyrai3md.skills.cp2k_skill.schema.adapter import build_config as cp2k_build
from pyrai3md.skills.cp2k_skill.schema.normalize import normalize as cp2k_normalize
from pyrai3md.skills.cp2k_skill.schema.validate import validate as cp2k_validate
from pyrai3md.skills.nn_pes_skill import query as nn_query
from pyrai3md.skills.nn_pes_skill.schema import reference as nn_reference
from pyrai3md.skills.nn_pes_skill.service.calculator import GRADIENT_TO_AU


class TestTheSkillUnitCopy(unittest.TestCase):
    """The skill's copy of the unit constants is the framework's."""

    def test_constants_match_the_framework(self):
        # A skill carries its own copy of the numbers; this test is
        # what keeps the copy honest.
        from pyrai3md.utilities import units
        self.assertEqual(nn_reference.BOHR_TO_ANGSTROM,
                         units.BOHR_TO_ANGSTROM)
        self.assertAlmostEqual(nn_reference.ANGSTROM_TO_BOHR,
                               units.ANGSTROM_TO_BOHR, places=15)
        self.assertAlmostEqual(GRADIENT_TO_AU,
                               units.BOHR_TO_ANGSTROM, places=15)


class TestTheDeclaredCapabilities(unittest.TestCase):
    """Every capability the skill declares is a framework name."""

    def test_capabilities_are_framework_names(self):
        from pyrai3md.planning.resolver import CAPABILITY_NAMES
        for capability in nn_query.get_capabilities():
            self.assertIn(capability, CAPABILITY_NAMES)


TRAINING_RECORD = {
    'train_errors': {'energy_grad': {'energy': [0.1, 0.2],
                                     'gradient': [0.01, 0.02]}},
    'train_metric_units': {'energy': 'eV', 'gradient': 'eV/angstrom'},
    'train_wall_seconds': 12.5,
    'train_arm': 'local',
    'train_data': '/data/TRAIN.json',
    'train_modeldir': '/run/NN-TEST',
    'train_fitlogs': ['/run/NN-TEST/energy_grad/fitlog_0.txt',
                      '/run/NN-TEST/energy_grad/fitlog_1.txt'],
    'train_prediction_std': {'ndata': 4, 'energy': 0.001,
                             'gradient': 0.002, 'nac': None, 'soc': None},
    'train_summary': '/run/TEST_train.summary.txt',
}


def training_config():
    """A configuration carrying the counts the report heads with."""
    from types import SimpleNamespace
    return SimpleNamespace(system=SimpleNamespace(
        natom=4, nstate=3, nac_pair_map=[[0, 1]], soc_pair_map=[]))


class TestTrainingReportAgreesWithTheRunLog(unittest.TestCase):
    """The skill's rendering and the framework's report say the same thing.

    The two live on opposite sides of the skill boundary and cannot share
    code, so the section titles and the column names are the contract; a
    rename on one side has to be caught here.
    """

    def test_titles_and_channels_match(self):
        from pyrai3md.skills.nn_pes_skill.service.trainer import (
            _TRAINING_CHANNELS, _TRAINING_ROLES)
        from pyrai3md.utilities.print_results import (
            TRAINING_CHANNELS, _TRAINING_ROLES as FW_ROLES)
        self.assertEqual(_TRAINING_CHANNELS, TRAINING_CHANNELS)
        self.assertEqual(_TRAINING_ROLES, FW_ROLES)

    def test_the_same_record_renders_the_same_numbers(self):
        from pyrai3md.skills.nn_pes_skill.service.trainer import (
            format_training)
        from pyrai3md.utilities.print_results import (format_ml_report,
                                                      format_result_block)
        record = TRAINING_RECORD
        skill_text = format_training(training_config(), record)
        system = {'natom': 4, 'nstate': 3, 'nac_pair_map': [[0, 1]],
                  'soc_pair_map': []}
        framework_text = format_result_block(
            system, dict(record), verbose=2,
            record_report=format_ml_report(dict(record), system))
        for number in ('0.10000000', '0.20000000', '0.01000000',
                       '0.02000000'):
            self.assertIn(number, skill_text)
            self.assertIn(number, framework_text)
        self.assertIn('&nn validation mean absolute error', skill_text)
        self.assertIn('&nn validation mean absolute error', framework_text)


class TestFrameworkHandoff(unittest.TestCase):
    """The real framework merge -> real skill schema, end to end."""

    def test_undeclared_keywords_from_the_framework_reach_the_schema(self):
        # `_merge_one_skill_config` hands the skill a flat dict; when the
        # user did not write the dispatch keywords they are absent from
        # it, and the schema then reads them as undeclared rather than
        # rejecting the run. The three keys are declared in the skill
        # schema but deliberately absent from reference.DEFAULT, so this
        # is the path that decides whether a framework-driven CP2K run
        # starts at all.
        from pyrai3md.kernel.maestro import Maestro
        from pyrai3md.schema.adapter import _merge_one_skill_config
        from pyrai3md.schema.config import Config

        from pyrai3md.skills.cp2k_skill.backend import (
            Cp2kBackend)

        raw = {
            'control': {'jobtype': 'energy', 'driver': 'main'},
            'cp2k': {'method': 'dft', 'profile': 'dft_energy',
                     'execution_mode': 'local'},
        }
        merged = _merge_one_skill_config(Maestro(), 'cp2k_skill',
                                         config=Config(**raw), raw_config=raw)
        self.assertNotIn('multitask', merged)

        config = cp2k_validate(cp2k_normalize(cp2k_build(dict(merged))))
        self.assertIsNone(config.environment.multitask)
        self.assertIsNone(config.environment.multipool)

        # The pipeline completed: the only refusal left is the missing
        # CP2K installation, which is the backend's own next step.
        with self.assertRaisesRegex(RuntimeError, 'cp2k_path'):
            Cp2kBackend({'skill_config': merged})

    def test_declared_pool_survives_the_same_handoff(self):
        from pyrai3md.kernel.maestro import Maestro
        from pyrai3md.schema.adapter import _merge_one_skill_config
        from pyrai3md.schema.config import Config

        raw = {
            'control': {'jobtype': 'energy', 'driver': 'main'},
            'cp2k': {'method': 'dft', 'profile': 'dft_energy',
                     'execution_mode': 'local', 'multitask': True,
                     'multipool': 6},
        }
        merged = _merge_one_skill_config(Maestro(), 'cp2k_skill',
                                         config=Config(**raw), raw_config=raw)
        config = cp2k_validate(cp2k_normalize(cp2k_build(dict(merged))))
        self.assertIs(config.environment.multitask, True)
        self.assertEqual(config.environment.multipool, 6)


class TestTargetStatesReachTheSkillSection(unittest.TestCase):
    """A skill field routes through its defaults table, nothing else.

    Round tmp_20261110_093321 ran `tddft_gradient` on three carriers: the
    skill entry wrote OpenQP `grad=2,3` (from the case's
    `target_states: [1, 2]`) while the framework agent and classic input
    wrote `grad=1`, because the converter only forwards skill keywords
    registered in `query.get_defaults()`. The run passed on every leg --
    each leg was judged by its own exit and closing status -- so this is
    the check that has to catch it.
    """

    FLAT = {
        'title': 'tddft_gradient_test',
        'jobtype': 'energy gradient',
        'theory': ['openqp'],
        'skill_name': 'openqp_skill',
        'skill_path': 'backend',
        'driver': 'skill',
        'atoms': ['C', 'C', 'H', 'H', 'H', 'H'],
        'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 1.33],
                        [0.0, 0.92, -0.57], [0.0, -0.92, -0.57],
                        [0.92, 0.0, 1.9], [-0.92, 0.0, 1.9]],
        'charges': 0,
        'multiplicity': 1,
        'nstate': 3,
        'state_spin': [0, 0, 0],
        'method': 'tddft',
        'basis': 'sto-3g',
        'functional': 'bhhlyp',
        'td_type': 'tda',
        'profile': 'tddft_gradient',
        'execution_mode': 'local',
        'target_states': [1, 2],
    }

    def test_openqp_declares_the_keyword(self):
        from pyrai3md.skills.openqp_skill import query
        self.assertEqual(query.get_defaults()['target_states'], [0])

    def test_the_converter_carries_the_request(self):
        from pyrai3md.qualification.certification.converter import (
            convert_to_pyrai_config)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'tddft_gradient.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(self.FLAT, handle)
            sectioned = convert_to_pyrai_config(path)
        self.assertEqual(sectioned['openqp_skill']['target_states'], [1, 2])


if __name__ == '__main__':
    unittest.main()
