"""
======================================================================
Test suite for the certification handler (the former skill_input):
skill test file discovery and the generic classic-input skill
section builder that replaced the per-skill dev_tool builders.

Tests cover:
- handler: tests-directory discovery of a shipped skill package and
  the empty result for an unknown skill
- handler: YAML case discovery (sorted .yaml) and runtime test
  discovery (test_*.py, planned ahead of the YAML task tests)
- handler: the generic _build_skill_section renders any skill from
  its identity alone (header, path/project, method, execution keys)
  and the compat wrappers dispatch onto it

The slurm/execution-key injection semantics are covered separately
in test_qualification_slurm.py.

Aug 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import io
import json
import os
import tempfile
import unittest

from pyrai3md.qualification.certification.handler import (
    _build_method_section, _build_skill_section, _skill_tests_dir,
    list_skill_runtime_tests, list_skill_test_yamls,
)


class TestSkillTestsDirectory(unittest.TestCase):
    """tests-directory discovery of skill packages."""

    def test_shipped_skill_tests_dir_is_found(self):
        tests_dir = _skill_tests_dir('qdm_skill')
        self.assertIsNotNone(tests_dir)
        self.assertTrue(os.path.isdir(tests_dir))
        self.assertTrue(tests_dir.endswith(
            os.path.join('skills', 'qdm_skill', 'tests')))

    def test_unknown_skill_returns_none(self):
        self.assertIsNone(_skill_tests_dir('no_such_skill'))


class TestSkillTestFileDiscovery(unittest.TestCase):
    """YAML case and runtime test script discovery."""

    def test_yaml_discovery_returns_sorted_existing_files(self):
        yamls = list_skill_test_yamls('qdm_skill')
        self.assertTrue(yamls)
        self.assertEqual(yamls, sorted(yamls))
        for path in yamls:
            self.assertTrue(path.endswith('.yaml'))
            self.assertTrue(os.path.isfile(path))
        # The shipped tully model cases are all discovered.
        names = {os.path.basename(path) for path in yamls}
        self.assertIn('tully1_energy.yaml', names)
        self.assertIn('tully1_reference.yaml', names)

    def test_runtime_test_discovery_returns_test_scripts(self):
        scripts = list_skill_runtime_tests('qdm_skill')
        self.assertTrue(scripts)
        for path in scripts:
            base = os.path.basename(path)
            self.assertTrue(base.startswith('test_'))
            self.assertTrue(base.endswith('.py'))
            self.assertTrue(os.path.isfile(path))
        names = {os.path.basename(path) for path in scripts}
        self.assertIn('test_calculator.py', names)

    def test_discovery_of_unknown_skill_is_empty(self):
        self.assertEqual(list_skill_test_yamls('no_such_skill'), [])
        self.assertEqual(list_skill_runtime_tests('no_such_skill'), [])


class TestSkillSectionBuilder(unittest.TestCase):
    """Generic classic-input skill section (the dev_tool successor)."""

    def _config(self, **extra):
        config = {
            'title': 'mock_job',
            'method': 'tully1',
            'qdm_path': '/opt/qdm',
            'qdm_project': 'mock_project',
        }
        config.update(extra)
        return config

    def test_generic_builder_renders_section_from_identity(self):
        section = _build_skill_section(self._config(), 'qdm_skill')
        lines = section.splitlines()
        self.assertEqual(lines[0], '&QDM')
        body = '\n'.join(lines)
        # path / project / method lines from the config identity.
        self.assertIn('qdm', body)
        self.assertIn('qdm_project', body)
        self.assertIn('mock_project', body)
        self.assertIn('method', body)
        self.assertIn('tully1', body)

    def test_execution_keys_are_forwarded_into_the_section(self):
        config = self._config(
            execution_mode='slurm', slurm_partition='gpu')
        body = _build_skill_section(config, 'qdm_skill')
        self.assertIn('execution_mode', body)
        self.assertIn('slurm', body)
        self.assertIn('slurm_partition', body)
        self.assertIn('gpu', body)


class TestSkillFixtureStaging(unittest.TestCase):
    """The data a case names by file name has to reach the case directory.

    A skill case runs with the run directory as its working directory, so a
    shipped fixture is invisible to it unless the planner lays it down there.
    The example flow has staged launch-directory files all along; this is the
    same duty on the skill-case side, and the nn cases depend on it for their
    training files.
    """

    def test_shipped_data_reaches_the_run_directory(self):
        import tempfile
        from pyrai3md.qualification.certification.planner import (
            stage_skill_fixtures)
        with tempfile.TemporaryDirectory() as run_dir:
            staged = stage_skill_fixtures('nn_pes_skill', run_dir)
            self.assertIn('egs_small.json', staged)
            self.assertIn('soc_small.json', staged)
            for name in staged:
                path = os.path.join(run_dir, name)
                self.assertTrue(os.path.isfile(path), name)
                self.assertGreater(os.path.getsize(path), 0, name)

    def test_a_skill_without_data_is_a_no_op(self):
        import tempfile
        from pyrai3md.qualification.certification.planner import (
            stage_skill_fixtures)
        with tempfile.TemporaryDirectory() as run_dir:
            self.assertEqual(stage_skill_fixtures('qdm_skill', run_dir), [])
            self.assertEqual(os.listdir(run_dir), [])
            self.assertEqual(
                stage_skill_fixtures('no_such_skill', run_dir), [])


class TestNnCaseOrdering(unittest.TestCase):
    """Each nn serving case must follow the case that trained its folder.

    The cases of one skill share a run directory and the plan runs them in
    name order, so a stem that sorts wrongly leaves a serving case pointing
    at a folder nothing created. That dependency is carried entirely by the
    names, which is why it is asserted here rather than left to reading.
    """

    # Serving case, and the case that has to have fitted the folder it reads.
    PAIRS = (('01_mlp_eg_gradient', '00_mlp_eg_train'),
             ('03_mlp_nac2_nac', '02_mlp_nac2_train'),
             ('05_mlp_e_soc', '04_mlp_e_soc_train'),
             ('06_mlp_eg_predict', '00_mlp_eg_train'))

    def test_every_trained_ensemble_is_served_afterwards(self):
        stems = {os.path.splitext(os.path.basename(path))[0]
                 for path in list_skill_test_yamls('nn_pes_skill')}
        for served, trained in self.PAIRS:
            self.assertIn(served, stems)
            self.assertIn(trained, stems)
            self.assertLess(sorted(list(stems)).index(trained),
                            sorted(list(stems)).index(served),
                            '%s runs before the %s run that trains it'
                            % (served, trained))

    def test_the_model_directories_the_pairs_agree_on(self):
        """A serving case reads the folder its own training case wrote."""
        import yaml
        tests_dir = os.path.dirname(list_skill_test_yamls('nn_pes_skill')[0])
        for served, trained in self.PAIRS:
            with io.open(os.path.join(tests_dir, trained + '.yaml'),
                         encoding='utf-8') as handle:
                train_input = yaml.safe_load(handle)['input']
            with io.open(os.path.join(tests_dir, served + '.yaml'),
                         encoding='utf-8') as handle:
                serve_input = yaml.safe_load(handle)['input']
            self.assertEqual(serve_input['modeldir'], train_input['modeldir'])
            self.assertEqual(serve_input['nstate'], train_input['nstate'])
            # A serving case names its request either by profile or by
            # jobtype; a case that does both contradicts itself.
            if 'profile' in serve_input:
                self.assertNotEqual(serve_input['profile'],
                                    train_input['profile'])
                self.assertNotIn('jobtype', serve_input)

    def test_a_case_named_by_jobtype_resolves_the_serving_profile(self):
        """The request a case writes has to reach the profile it means.

        A case that names no profile is the only one that exercises the
        framework-to-skill profile resolution, which is exactly where a
        train request used to be answered with a serving profile. The case
        is loaded the way the certification loads it: converted to the
        classic input, parsed back, then handed to the merge the run uses.
        """
        import tempfile
        from pyrai3md.planning.assembly import _build_skill_flat_and_submodels
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.kernel.environment import Environment
        from pyrai3md.schema import adapter as framework_adapter
        from pyrai3md.schema.adapter import build_config as build_framework
        from pyrai3md.schema.normalize import normalize as normalize_fw
        from pyrai3md.qualification.certification.converter import convert

        from pyrai3md.qualification.certification.converter import (
            convert_to_pyrai_config, parse_test_yaml)

        tests_dir = os.path.dirname(list_skill_test_yamls('nn_pes_skill')[0])
        case_path = os.path.join(tests_dir, '06_mlp_eg_predict.yaml')
        flat = parse_test_yaml(case_path, 'nn_pes_skill')
        with tempfile.TemporaryDirectory() as tmpdir:
            flat_path = os.path.join(tmpdir, 'nn_predict.json')
            with open(flat_path, 'w', encoding='utf-8') as handle:
                json.dump(flat, handle)
            # The agent carrier: the framework-shaped config pyrai3md_agent
            # reads. The classic .input text needs a same-basename .xyz the
            # planner stages, which this assertion does not exercise.
            config_path = os.path.join(tmpdir, 'nn_predict.config')
            convert_to_pyrai_config(flat_path, config_path,
                                    skill_name='nn_pes_skill')
            with open(config_path, encoding='utf-8') as handle:
                raw = json.load(handle)

        self.assertEqual(raw['control']['jobtype'], 'prediction')
        env = Environment()
        env.initialize()
        maestro = (getattr(env, 'maestro', None)
                   or getattr(env, '_maestro', None))
        merged = framework_adapter._merge_one_skill_config(
            maestro, 'nn_pes_skill',
            config=normalize_fw(build_framework(raw)), raw_config=raw) or {}
        flat = _build_skill_flat_and_submodels('nn_pes_skill', merged,
                                               driver='main')
        # The family's serving profile, and not the training profile the
        # folder was written with.
        self.assertEqual(flat.get('profile'), 'mlp_eg_gradient')


class TestInlineChargeStaging(unittest.TestCase):
    """A test YAML may spell its background charges inline.

    `generate_test_configs` turns `external_charges: [[q, x, y, z], ...]` into
    the file the runtime switch selects - '{title}.charge' in the layout the
    framework's own charge reader uses - and leaves `read_charge: true` in
    every generated document. Two carriers are then checked with it: the
    classic text and the agent `.config`, both of which must render the
    framework field as a bool, because `System.embedding` and
    `System.read_charge` are bools and the skill's word for the decision
    belongs to the skill's own section.
    """

    def _yaml(self, directory):
        path = os.path.join(directory, 'espf_case.yaml')
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(
                'version: 1.0\n'
                'name: espf_case_test\n'
                'input:\n'
                '  atoms: [C, C]\n'
                '  coordinates: [[0.0, 0.0, 0.0], [0.0, 0.0, 1.34]]\n'
                '  charges: 0\n'
                '  multiplicity: 1\n'
                '  nstate: 1\n'
                '  state_spin: [0]\n'
                '  jobtype: energy\n'
                '  profile: casscf_properties\n'
                '  method: casscf\n'
                '  basis: ano-s-mb\n'
                '  nact_elec: 2\n'
                '  nact_orb: 2\n'
                '  inactive_orb: 7\n'
                '  embedding: charge\n'
                '  external_charges:\n'
                '  - [1.0, 0.0, 0.0, 5.0]\n'
                '  - [-1.0, 0.0, 0.0, -5.0]\n')
        return path

    def test_rows_become_the_titled_charge_file(self):
        from pyrai3md.qualification.certification.converter import parse_test_yaml
        from pyrai3md.qualification.certification.converter import (
            _stage_charge_sidecar)
        with tempfile.TemporaryDirectory() as tmp:
            config = parse_test_yaml(self._yaml(tmp), 'molcas_skill')
            _stage_charge_sidecar(config, tmp)
            written = os.path.join(tmp, 'espf_case_test.charge')
            self.assertTrue(os.path.exists(written))
            with open(written, encoding='utf-8') as handle:
                lines = handle.read().splitlines()
        # A declaration first, then the rows in that declaration's order.
        self.assertEqual(lines[0], 'point angstrom external')
        self.assertEqual([row.split() for row in lines[2:]],
                         [['1.0000000000', '0.0000000000', '0.0000000000',
                           '5.0000000000'],
                          ['-1.0000000000', '0.0000000000', '0.0000000000',
                           '-5.0000000000']])
        self.assertNotIn('external_charges', config)
        self.assertIs(config['read_charge'], True)

    def test_a_path_value_is_left_alone(self):
        from pyrai3md.qualification.certification.converter import (
            _stage_charge_sidecar)
        config = {'title': 'demo', 'external_charges': 'somewhere.charge'}
        with tempfile.TemporaryDirectory() as tmp:
            _stage_charge_sidecar(config, tmp)
        self.assertEqual(config['external_charges'], 'somewhere.charge')
        self.assertNotIn('read_charge', config)

    def test_a_field_declares_itself_in_the_same_file(self):
        from pyrai3md.qualification.certification.converter import (
            _stage_charge_sidecar)
        config = {'title': 'demo',
                  'external_potential': [[0.01, 0.0, 0.0, 0.001],
                                         [0.02, 0.0, 0.0, -0.001]]}
        with tempfile.TemporaryDirectory() as tmp:
            _stage_charge_sidecar(config, tmp)
            written = os.path.join(tmp, 'demo.charge')
            with open(written, encoding='utf-8') as handle:
                lines = handle.read().splitlines()
        self.assertEqual(lines[0], 'field au')
        self.assertEqual(len(lines), 4)
        self.assertNotIn('external_potential', config)
        self.assertIs(config['read_charge'], True)

    def test_charge_and_field_together_are_refused(self):
        # One file carries one embedding; two of them is a question with two
        # answers and the converter says so instead of picking one.
        from pyrai3md.qualification.certification.converter import (
            _stage_charge_sidecar)
        config = {'title': 'demo',
                  'external_charges': [[1.0, 0.0, 0.0, 5.0]],
                  'external_potential': [[0.01, 0.0, 0.0, 0.001]]}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _stage_charge_sidecar(config, tmp)

    def test_a_short_row_is_refused(self):
        from pyrai3md.qualification.certification.converter import (
            _stage_charge_sidecar)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _stage_charge_sidecar({'title': 'demo',
                                       'external_charges': [[1.0, 0.0, 0.0]]},
                                      tmp)

    def test_carriers_render_the_framework_switch_as_a_bool(self):
        # What `System` accepts is a bool; a string in &MOLECULE or in the
        # `.config` molecule section fails the model, which is exactly what the
        # deployment's ESPF legs would have hit.
        from pyrai3md.qualification.certification.converter import (
            _stage_charge_sidecar, convert, convert_to_pyrai_config,
            parse_test_yaml)
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.schema.adapter import build_config
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = self._yaml(tmp)
            config = parse_test_yaml(yaml_path, 'molcas_skill')
            _stage_charge_sidecar(config, tmp)
            json_path = os.path.join(tmp, 'espf_case.json')
            with open(json_path, 'w', encoding='utf-8') as handle:
                json.dump(config, handle)
            input_path = os.path.join(tmp, 'espf_case.input')
            convert(json_path, input_path)
            with open(input_path, encoding='utf-8') as handle:
                molecule = handle.read().split('&MOLECULE', 1)[1]
            self.assertIn('embedding     true\n', molecule)
            self.assertIn('read_charge   true\n', molecule)

            agent_path = os.path.join(tmp, 'espf_case.config')
            convert_to_pyrai_config(json_path, agent_path)
            with open(agent_path, encoding='utf-8') as handle:
                sectioned = json.load(handle)
            self.assertIs(sectioned['molecule']['embedding'], True)
            self.assertIs(sectioned['molecule']['read_charge'], True)
            # The skill's own section keeps the skill's word for the decision.
            self.assertEqual(sectioned['molcas_skill']['embedding'], 'charge')
            # Both documents build the framework model, and read_charge stands.
            self.assertTrue(build_config(
                parse_input(input_path)).system.read_charge)
            self.assertTrue(build_config(sectioned).system.read_charge)


if __name__ == '__main__':
    unittest.main()
