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

try:
    import pytest
except ImportError:  # unittest-only hosts do not need pytest
    pytest = None

if pytest is not None:
    pytestmark = pytest.mark.integration

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


class TestNnPesThreeCarrierInput(unittest.TestCase):
    """yaml, classic input and agent config reach the skill identically."""

    @classmethod
    def setUpClass(cls):
        cls._env = Environment()
        cls._env.initialize()
        cls._maestro = (getattr(cls._env, 'maestro', None)
                        or getattr(cls._env, '_maestro', None))
        cls._pmd_env = manager.load_config()

    def _config_for_mode(self, label, tmpdir):
        """Build one carrier from CASE and return its skill config."""
        yaml_path = os.path.join(tmpdir, 'nn_carrier.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as handle:
            import yaml
            handle.write(yaml.safe_dump(CASE))

        flat_path = os.path.join(tmpdir, 'nn_carrier.json')
        yaml_to_config(yaml_path, flat_path)
        manager._inject_example_skill_env(
            flat_path, self._pmd_env.get('skills', {}))
        with open(flat_path, encoding='utf-8') as handle:
            flat = json.load(handle)

        if label == 'yaml':
            raw = parse_input(yaml_path)
        elif label == 'input':
            input_path = os.path.join(tmpdir, 'nn_carrier.input')
            convert(flat_path, input_path)
            raw = parse_input(input_path)
        elif label == 'agent':
            # The agent carrier is read as the JSON it is, the same way
            # pyrai3md_agent consumes it; parse_input would hand a .config
            # file to the classic text reader.
            agent_path = os.path.join(tmpdir, 'nn_carrier.config')
            with open(agent_path, 'w', encoding='utf-8') as handle:
                json.dump(convert_to_pyrai_config(flat_path), handle)
            with open(agent_path, encoding='utf-8') as handle:
                raw = json.load(handle)
        else:
            raise AssertionError('unknown mode %r' % label)

        framework = normalize_framework(build_framework_config(raw))
        merged = framework_adapter._merge_one_skill_config(
            self._maestro, SKILL, config=framework, raw_config=raw)
        self.assertTrue(merged, '%s mode produced no skill config' % label)
        return validate(normalize(build_config(merged))), raw

    def _member_table(self, config, role='eg'):
        """Expand a role section into per-member values, TF-free."""
        from pyrai3md.skills.nn_pes_skill.schema.hyper import (
            HYPER_KEYWORDS, broadcast, member_groups, parse_value)
        table = {}
        section = config.calculation.eg if role == 'eg' else config.calculation.soc
        for keyword, raw in section.items():
            kind = HYPER_KEYWORDS[keyword][0]
            groups = [parse_value(group, kind)
                      for group in member_groups(raw, kind)]
            table[keyword] = broadcast(groups, config.mult_nn, keyword)
        return table

    def test_the_role_section_arrives_in_every_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for label in ('yaml', 'input', 'agent'):
                config, _raw = self._config_for_mode(label, tmpdir)
                self.assertTrue(config.calculation.eg,
                                '%s mode lost the &EG keywords' % label)

    def test_a_typed_nested_list_is_refused_not_dropped(self):
        # The classic text spells one value per line, so a nested list has
        # no faithful spelling. Flattening it, or leaving it out, would
        # hand the two entry points different configurations while both
        # reported a pass, so the renderer refuses it.
        with tempfile.TemporaryDirectory() as tmpdir:
            case = json.loads(json.dumps(CASE))
            case['nn']['eg']['loss_weights'] = [[1, 1], [1, 10]]
            yaml_path = os.path.join(tmpdir, 'nested.yaml')
            import yaml
            with open(yaml_path, 'w', encoding='utf-8') as handle:
                handle.write(yaml.safe_dump(case))
            flat_path = os.path.join(tmpdir, 'nested.json')
            yaml_to_config(yaml_path, flat_path)
            with self.assertRaises(ValueError) as caught:
                convert(flat_path, os.path.join(tmpdir, 'nested.input'))
            self.assertIn("keyword 'loss_weights' carries a nested list",
                          str(caught.exception))
            self.assertFalse(os.path.exists(os.path.join(tmpdir,
                                                         'nested.input')))

    def test_shared_block_and_state_fields_agree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            readings = {label: self._config_for_mode(label, tmpdir)[0]
                        for label in ('yaml', 'input', 'agent')}
            for key in _KEYS:
                values = {label: getattr(config.calculation, key)
                          if hasattr(config.calculation, key)
                          else getattr(config.system, key, None)
                          for label, config in readings.items()}
                if key in ('eg', 'nac', 'soc'):
                    # A role block is compared after parsing, because the
                    # classic carrier spells every value as text and a typed
                    # list has no one-line spelling; the meaning, not the
                    # quotation marks, is what has to agree.
                    continue
                with self.subTest(key=key):
                    self.assertEqual(values['input'], values['yaml'],
                                     '%s differs between yaml and input' % key)
                    self.assertEqual(values['agent'], values['yaml'],
                                     '%s differs between yaml and agent' % key)

    def test_member_axis_means_the_same_in_every_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tables = {label: self._member_table(self._config_for_mode(
                label, tmpdir)[0], role='eg')
                for label in ('yaml', 'input', 'agent')}
            self.assertEqual(sorted(tables['input']), sorted(tables['yaml']))
            for keyword in sorted(tables['yaml']):
                with self.subTest(keyword=keyword):
                    for label in ('input', 'agent'):
                        self.assertEqual(tables[label][keyword],
                                         tables['yaml'][keyword],
                                         '%s: %s member groups differ in %s '
                                         'mode' % (keyword,
                                                   tables['yaml'][keyword],
                                                   label))

    def test_spin_orbit_members_read_the_same_axis(self):
        # The same rule on the third role block, and the state split the
        # framework resolved from the mixed-spin coupling section: one
        # non-adiabatic pair, two spin-orbit pairs.
        with tempfile.TemporaryDirectory() as tmpdir:
            tables = {}
            for label in ('yaml', 'input', 'agent'):
                config = self._config_for_mode(label, tmpdir)[0]
                tables[label] = self._member_table(config, role='soc')
                self.assertEqual(config.system.nnac, 1)
                self.assertEqual(config.system.nsoc, 2)
                self.assertEqual(config.system.soc_pair_map, [[0, 2], [1, 2]])
            self.assertEqual(tables['input'], tables['yaml'])
            self.assertEqual(tables['agent'], tables['yaml'])
            self.assertEqual(tables['yaml']['epo'], [900, 700])
            self.assertEqual(tables['yaml']['batch_size'], [16, 16])

    def test_two_members_were_actually_written(self):
        # The whole point of the ';' axis: the two members must differ
        # where the input said so, and agree where it did not.
        with tempfile.TemporaryDirectory() as tmpdir:
            table = self._member_table(
                self._config_for_mode('input', tmpdir)[0], role='eg')
            self.assertEqual(table['epo'], [200, 150])
            self.assertEqual(table['loss_weights'], [[1, 1], [1, 10]])
            self.assertEqual(table['depth'], [3, 3])
            self.assertEqual(table['batch_size'], [32, 32])

    def test_built_hyper_dicts_agree_across_modes(self):
        # The endpoint a training run reads: one hyper dictionary per
        # member, expanded from the defaults of the model type. The
        # defaults come from the container, so this needs TensorFlow;
        # where it is absent the member table above still holds.
        try:
            from pyrai3md.skills.nn_pes_skill.service.nn_pes.nn_pes_src \
                .selection import get_default_hyperparameters_by_modeltype
        except Exception:
            self.skipTest('tensorflow is not installed; the hyper build '
                          'needs the container defaults')
        with tempfile.TemporaryDirectory() as tmpdir:
            built = {}
            for label in ('yaml', 'input', 'agent'):
                config = self._config_for_mode(label, tmpdir)[0]
                base = get_default_hyperparameters_by_modeltype('mlp_eg')
                base['model']['atoms'] = 5
                base['model']['states'] = config.system.nstate
                built[label] = build_member_hyper(
                    'eg', base, config.mult_nn, config.calculation.eg)
            def read(hyper, section, dotted):
                node = hyper[section]
                for part in dotted.split('.'):
                    node = node[part]
                return node

            for index in range(2):
                for keyword, (section, dotted) in sorted(_MEMBERS.items()):
                    expected = read(built['yaml'][index], section, dotted)
                    with self.subTest(member=index, keyword=keyword):
                        for label in ('input', 'agent'):
                            self.assertEqual(
                                read(built[label][index], section, dotted),
                                expected,
                                '%s: member %d reads a different value in %s '
                                'mode' % (keyword, index, label))









if __name__ == '__main__':
    unittest.main()
