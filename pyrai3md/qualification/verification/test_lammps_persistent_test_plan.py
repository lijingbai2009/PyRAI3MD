"""
======================================================================
Qualification-plan tests for the resident LAMMPS test case.

ff_gradient_persistent.yaml is the resident twin of ff_gradient.yaml:
the same water box, the same expected shapes, one extra keyword. A twin
is only worth having if that keyword survives every hand the planner
makes, since the case reaches the backend through three artifacts (the
flat .json a skill test writes, the classic .input text the input mode
reads, the agent .config the agent mode reads) and any one of them can
drop a key the schema has not registered. These tests pin all three,
plus the read-back: a classic input carries 'true' as a *string*, so a
resident case that arrives as the text 'true' and lands on the
subprocess path would pass every assertion here while testing nothing.

These tests do NOT require a LAMMPS installation.

Run: python -m unittest \
    pyrai3md.qualification.verification.test_lammps_persistent_test_plan

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import glob
import json
import os
import shutil
import tempfile
import unittest

import yaml

import pyrai3md.skills.lammps_skill

from pyrai3md.qualification.manager import examples_directory
from pyrai3md.skills.lammps_skill.schema.validate import (
    is_persistent_enabled)

TESTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(pyrai3md.skills.lammps_skill.__file__)),
    'tests')
EXAMPLES_DIR = examples_directory()
RESIDENT_CASE = 'ff_gradient_persistent.yaml'
COLD_CASE = 'ff_gradient.yaml'
SKILL_NAME = 'lammps_skill'


def _case_inputs(name):
    """Return the 'input' section of one shipped test case."""
    with open(os.path.join(TESTS_DIR, name), 'r', encoding='utf-8') as f:
        case = yaml.safe_load(f)
    return case['input'], case


def _flat_config(name):
    """Parse one shipped case through the planner's own entry point."""
    from pyrai3md.qualification.certification.converter import parse_test_yaml
    return parse_test_yaml(os.path.join(TESTS_DIR, name), SKILL_NAME)


def _write_flat_json(config, out_dir, name=RESIDENT_CASE):
    """Write a flat config where generate_test_configs would write it."""
    path = os.path.join(str(out_dir),
                        os.path.splitext(name)[0] + '.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f)
    return path


def _keyword(text, key, section='LAMMPS'):
    """Return one keyword's rendered value, or None when not emitted.

    A classic section aligns its values in a column, so matching
    'debug true' as a substring would test the padding rather than the
    keyword.
    """
    body = text.split(f'&{section}')[1] if f'&{section}' in text else ''
    for line in body.splitlines():
        tokens = line.split()
        if tokens and tokens[0] == key:
            return ' '.join(tokens[1:])
    return None


def _example_paths():
    """Every shipped framework-driven LAMMPS MD example."""
    return sorted(glob.glob(os.path.join(EXAMPLES_DIR, 'md_lammps_*.yaml')))


class _TempDirTest(unittest.TestCase):
    """Base class that gives each test its own scratch directory."""

    def setUp(self):
        self.tmp_path = tempfile.mkdtemp(prefix='lammps_persistent_')
        self.addCleanup(shutil.rmtree, self.tmp_path, True)


class TestCasePairing(unittest.TestCase):
    """The resident case must differ from the cold one by its switches."""

    def test_the_only_added_inputs_are_the_resident_switches(self):
        resident, _ = _case_inputs(RESIDENT_CASE)
        cold, _ = _case_inputs(COLD_CASE)
        added = {key for key in resident
                 if cold.get(key, object()) != resident[key]}
        assert added == {'python_enabled', 'debug'}
        assert resident['python_enabled'] is True
        # debug=true is why this case ships in the plan at all: the run
        # leaves the same log and dumps the subprocess case writes, so a
        # difference between the two paths is visible as a file
        # difference instead of an argument.
        assert resident['debug'] is True

    def test_expectations_and_name_follow_the_case(self):
        resident_input, resident = _case_inputs(RESIDENT_CASE)
        cold_input, cold = _case_inputs(COLD_CASE)
        assert resident['expected'] == cold['expected']
        assert resident['name'] == 'ff_gradient_persistent_test'
        assert resident['name'] != cold['name']

    def test_the_cold_twin_asks_for_nothing_resident(self):
        cold, _ = _case_inputs(COLD_CASE)
        assert 'python_enabled' not in cold
        assert 'debug' not in cold
        assert is_persistent_enabled(cold) is False


class TestPlannerCarriesTheKeyword(_TempDirTest):
    """Every artifact the planner emits keeps the switch."""

    def test_flat_json_keeps_a_native_bool(self):
        config = _flat_config(RESIDENT_CASE)
        assert config['python_enabled'] is True
        assert config['debug'] is True
        assert config['profile'] == 'ff_gradient'
        assert config['execution_mode'] == 'local'
        assert config['skill_name'] == SKILL_NAME
        # The project name follows the case name, so the twin does not
        # collide with the cold case over one shared topology file.
        assert config['lammps_project'] == 'ff_gradient_persistent_test'

    def test_classic_input_emits_the_keyword(self):
        from pyrai3md.qualification.certification.converter import convert
        path = _write_flat_json(_flat_config(RESIDENT_CASE), self.tmp_path)
        text = convert(path)
        assert _keyword(text, 'python_enabled') == 'true'
        assert _keyword(text, 'debug') == 'true'
        assert _keyword(text, 'execution_mode') == 'local'

    def test_agent_config_emits_the_keyword(self):
        from pyrai3md.qualification.certification.converter import (
            convert_to_pyrai_config)
        path = _write_flat_json(_flat_config(RESIDENT_CASE), self.tmp_path)
        sectioned = convert_to_pyrai_config(path)
        out = os.path.join(str(self.tmp_path), 'ff_gradient_persistent.config')
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(sectioned, f)
        with open(out, 'r', encoding='utf-8') as f:
            reloaded = json.load(f)
        # json round trip is what the agent mode actually reads, so the
        # value must still be the bool True rather than a string.
        assert reloaded['lammps_skill']['python_enabled'] is True
        assert reloaded['lammps_skill']['debug'] is True

    def test_the_cold_case_emits_nothing_resident(self):
        from pyrai3md.qualification.certification.converter import convert
        path = _write_flat_json(_flat_config(COLD_CASE), self.tmp_path,
                                name=COLD_CASE)
        text = convert(path)
        assert _keyword(text, 'python_enabled') is None
        assert _keyword(text, 'debug') is None

    def test_no_machine_local_path_is_frozen(self):
        """python_path stays out of the shipped case.

        The service interpreter is a property of the machine that runs
        the test, not of the case: validate_persistent fills it with
        sys.executable when empty, so pinning a path here would make the
        case unrunnable everywhere else.
        """
        from pyrai3md.qualification.certification.converter import (
            convert, convert_to_pyrai_config)
        config = _flat_config(RESIDENT_CASE)
        path = _write_flat_json(config, self.tmp_path)
        assert 'python_path' not in config
        assert 'python_path' not in convert(path)
        assert 'python_path' not in convert_to_pyrai_config(path)[
            SKILL_NAME]


class TestGenerateTestConfigs(_TempDirTest):
    """The planner's own entry point stages the case like its twin."""

    def test_the_case_is_discovered_with_its_topology(self):
        from pyrai3md.qualification.certification.converter import (
            generate_test_configs)
        out = os.path.join(str(self.tmp_path), 'lammps_skill')
        configs = generate_test_configs(SKILL_NAME, out)
        assert 'ff_gradient_persistent' in configs
        with open(configs['ff_gradient_persistent'], 'r',
                  encoding='utf-8') as f:
            config = json.load(f)
        assert config['python_enabled'] is True
        # The shared h2o fixtures are staged under the project name, so
        # the resident case finds its topology the same way the cold one
        # does and the two never contend for one launch directory.
        for suffix in ('.data', '.in.init', '.in.settings'):
            assert os.path.isfile(
                out + os.sep + 'ff_gradient_persistent_test' + suffix)


class TestReadBackRoutesToTheService(unittest.TestCase):
    """The written keyword is the keyword the backend acts on."""

    def test_classic_text_value_routes_resident(self):
        # A classic input hands every keyword over as a string.
        from pyrai3md.skills.lammps_skill.schema.adapter import (
            build_config)
        config = build_config({'profile': 'ff_gradient',
                               'execution_mode': 'local',
                               'python_enabled': 'true'})
        assert config.environment.python_enabled is True
        assert is_persistent_enabled(config) is True

    def test_agent_bool_value_routes_resident(self):
        from pyrai3md.skills.lammps_skill.schema.adapter import (
            build_config)
        config = build_config({'profile': 'ff_gradient',
                               'execution_mode': 'local',
                               'python_enabled': True})
        assert is_persistent_enabled(config) is True

    def test_absent_switch_stays_cold(self):
        from pyrai3md.skills.lammps_skill.schema.adapter import (
            build_config)
        assert build_config({'profile': 'ff_gradient'}).environment \
            .python_enabled is False
        assert is_persistent_enabled({'profile': 'ff_gradient'}) is False


class _StubMaestro:
    """Duck-typed Maestro exposing only what the merge looks up."""

    def __init__(self, runtime=None):
        self._runtime = runtime or {}

    def get_config(self, skill_name):
        return self._runtime.get(skill_name)

    def get_skill_info(self, skill_name):
        return {'name': 'lammps'}


class TestFrameworkForwardsTheSwitch(unittest.TestCase):
    """A user must be able to reach the switch from any input route.

    The framework owns no field called debug, so what has to hold is that
    no layer of the merge drops it: the skill section of a classic input,
    the skills.lammps_skill block of pmd_env.yaml or config.json, and the
    package-level Maestro defaults each have to deliver it, and the value
    still has to be a bool once the skill schema has read it.
    """

    def _merge(self, raw_config, runtime=None):
        from pyrai3md.schema.adapter import _merge_one_skill_config
        from pyrai3md.schema.config import PyRAIConfig

        maestro = _StubMaestro(runtime={'lammps_skill': runtime or {}})
        return _merge_one_skill_config(
            maestro, SKILL_NAME, config=PyRAIConfig(),
            raw_config=raw_config)

    def test_classic_input_section_forwards_it_as_text(self):
        merged = self._merge({'lammps': {'python_enabled': 'true',
                                         'debug': 'true'}})
        assert merged['debug'] == 'true'
        from pyrai3md.skills.lammps_skill.schema.adapter import (
            build_config)
        config = build_config(merged)
        assert config.environment.debug is True

    def test_pmd_env_skill_section_forwards_the_bool(self):
        merged = self._merge({'skills': {SKILL_NAME: {'debug': True,
                                                      'python_enabled':
                                                          True}}})
        assert merged['debug'] is True
        from pyrai3md.skills.lammps_skill.schema.adapter import (
            build_config)
        assert build_config(merged).environment.debug is True

    def test_package_defaults_reach_the_skill(self):
        merged = self._merge({}, runtime={'debug': True,
                                          'python_enabled': True})
        assert merged['debug'] is True

    def test_nothing_set_leaves_the_default_off(self):
        from pyrai3md.skills.lammps_skill.schema.adapter import (
            build_config)
        # No input section at all: the merge reports there is nothing to
        # hand over, and the skill's own default has to be the off side.
        assert self._merge({}) in (None, {})
        assert build_config({'profile': 'ff_gradient'}).environment.debug \
            is False


class TestOfficialExamplesRouteResident(_TempDirTest):
    """The official MD examples ask for the resident service by name.

    A framework-driven MD step is one ff_gradient call per step, so the
    per-step LAMMPS cold start is what these runs actually spend their
    time on. The examples declare the switch in their own lammps section,
    which is a longer route than a skill test case takes: the yaml is
    first flattened to .json, then rendered as a classic .input, and the
    input mode re-reads that text. Either hand can drop a key the
    framework owns no field for, and the two modes would then certify a
    resident and a subprocess execution of the same physics while both
    reporting the same trajectory.
    """

    def _merged(self, raw_config):
        from pyrai3md.schema.adapter import _merge_one_skill_config
        from pyrai3md.schema.config import PyRAIConfig

        return _merge_one_skill_config(
            _StubMaestro(), SKILL_NAME, config=PyRAIConfig(),
            raw_config=raw_config) or {}

    def _resident_config(self, raw_config):
        """Merge, resolve the profile and validate, as the backend sees it.

        The framework does not run the skill schema pipeline, so only the
        skill's own build_config can say whether a resident request on
        this profile is accepted or rejected.
        """
        from pyrai3md.planning.assembly import _build_skill_flat_and_submodels
        from pyrai3md.skills.lammps_skill.schema.adapter import (
            build_config)

        merged = self._merged(raw_config)
        merged['jobtype'] = 'gradient'
        flat = _build_skill_flat_and_submodels(
            'lammps_skill', merged, driver='main')
        return flat, build_config(flat)

    def test_the_examples_are_the_ones_the_suite_ships(self):
        assert len(_example_paths()) == 12

    def test_every_example_declares_the_switch_and_a_local_engine(self):
        for path in _example_paths():
            with open(path, 'r', encoding='utf-8') as handle:
                case = yaml.safe_load(handle)
            stem = os.path.basename(path)
            # 'local' is a precondition, not a detail: the switch is inert
            # under slurm, where every step is its own batch job.
            assert case['lammps'].get('python_enabled') is True, stem
            # Where that precondition is actually stated: execution_mode
            # is a skill keyword (this class' own docstring says the
            # examples declare the switch in their lammps section), and
            # the framework refuses it in its own &environment section
            # (#300). Nothing in the example may ask for remote dispatch,
            # because that would hand the step off-process and end the
            # resident service.
            environment = case.get('environment') or {}
            for retired in ('execution_mode', 'workflow_execution',
                            'workflow_workers'):
                assert retired not in environment, (stem, retired)
            for section in ('lammps', 'md', 'molecule', 'environment'):
                block = case.get(section) or {}
                assert not block.get('remotetask'), (stem, section)

    def test_both_example_modes_route_to_the_service(self):
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.qualification.certification.converter import (
            convert, yaml_to_config)

        for path in _example_paths():
            stem = os.path.splitext(os.path.basename(path))[0]
            json_path = os.path.join(str(self.tmp_path), stem + '.json')
            yaml_to_config(path, json_path)
            input_path = os.path.join(str(self.tmp_path), stem + '.input')
            convert(json_path, input_path)
            modes = {'yaml': parse_input(path),
                     'input': parse_input(input_path)}
            for label, raw in modes.items():
                flat, config = self._resident_config(raw)
                # Profile 'ff_gradient' is what a framework MD step asks
                # for, and it is one of the two profiles the resident path
                # implements; build_config would raise otherwise.
                assert flat['profile'] == 'ff_gradient', (stem, label)
                assert config.environment.python_enabled is True, (stem,
                                                                   label)
                assert is_persistent_enabled(flat) is True, (stem, label)
