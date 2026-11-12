"""
======================================================================
Test suite for the certification converter's multi-center example
support: the classic-input / agent-config rendering of the
center-dimensioned molecule/md keywords and of a multi-theory
composite (multi-center ONIOM) skill sections.

Tests cover:
- yaml_to_config preserves the center strings (highlevel/ci/spin/
  coupling/root) and one parameter block per theory skill
  (skill_sections), with the primary skill following control.theory
- convert renders the classic input with the ';'-separated center
  keywords, one method section per skill (&OPENQP + &XTB) and an &MD
  section carrying the per-center root even for the non-advanced
  'gradient' jobtype
- convert_to_pyrai_config emits the same center strings into the
  molecule/md sections and one agent section per skill
- the flat example .json is read back by the task queue, which takes each
  bound skill's method from its own block in skill_sections rather than
  from the primary skill's flattened top level, and resolves a profile
  that matches the jobtype the task actually asks for
- a run that binds ONE skill at TWO positions (`theory: [pyscf, pyscf]`, the
  multiscale carriers) counts as a multi-task run to the converter: its block
  is kept, its flat config reads back at all, and the rendering puts the
  System fields where every already-shipped two-backend example puts them
- single-center regression: a plain one-skill example keeps the
  historical output (no skill_sections, one method section, no &MD)

These are pure converter-output assertions: no external program
(openqp/xtb) is executed, so the tests run locally without the
multi-center runtime environment. The flat-config class is the one
exception in kind, not in degree: it builds a task queue from the
converter's own product, which reads each skill's query module and still
calls no engine.

Sep 9 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import glob
import json
import os
import tempfile
import unittest

from pyrai3md.qualification.certification.converter import (
    convert, convert_to_pyrai_config, yaml_to_config,
    _is_multicenter, _render_center_string, _build_molecule_pairs,
)

# Package-relative examples directory (pyrai3md/qualification/
# verification -> pyrai3md/examples)
_EXAMPLES_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, 'examples'))
_MULTICENTER_YAML = os.path.join(_EXAMPLES_DIR,
                                 'multicenter_gradient.yaml')
_GRADIENT_YAML = os.path.join(_EXAMPLES_DIR, 'gradient.yaml')


class TestMulticenterYamlToConfig(unittest.TestCase):
    """yaml_to_config preserves center fields and both skill blocks."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, 'mc.json')
        self.config = yaml_to_config(_MULTICENTER_YAML, self.config_path)

    def test_center_strings_preserved(self):
        self.assertEqual(self.config['highlevel'], '1-6; 7-12')
        self.assertEqual(self.config['ci'], '3; 3')
        self.assertEqual(self.config['spin'], '0; 0')
        self.assertEqual(self.config['coupling'],
                         '1 2, 2 3, 1 3; 1 2, 2 3, 1 3')
        self.assertEqual(self.config['root'], '1; 1')

    def test_theory_and_primary_skill(self):
        self.assertEqual(self.config['theory'], ['openqp', 'xtb'])
        # Primary skill follows control.theory order (high level first)
        self.assertEqual(self.config['skill_name'], 'openqp_skill')
        # The primary skill keywords stay flat for the single-skill path
        self.assertEqual(self.config['method'], 'mrsf')

    def test_skill_sections_preserved_separately(self):
        sections = self.config['skill_sections']
        self.assertEqual(list(sections.keys()), ['openqp', 'xtb'])
        self.assertEqual(sections['openqp']['method'], 'mrsf')
        self.assertEqual(sections['openqp']['basis'], 'sto-3g')
        # xtb keeps its own method (no collision with openqp)
        self.assertEqual(sections['xtb']['method'], 'gfn2')


class TestMulticenterClassicInput(unittest.TestCase):
    """convert renders the classic multi-center input text."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, 'mc.json')
        yaml_to_config(_MULTICENTER_YAML, self.config_path)
        self.input_path = os.path.join(self.tmpdir, 'mc.input')
        self.text = convert(self.config_path, self.input_path)

    def test_molecule_center_keywords(self):
        self.assertIn('highlevel', self.text)
        self.assertIn('1-6; 7-12', self.text)
        self.assertIn('3; 3', self.text)
        self.assertIn('1 2, 2 3, 1 3; 1 2, 2 3, 1 3', self.text)

    def test_md_root_section_present(self):
        self.assertIn('&MD', self.text)
        self.assertIn('1; 1', self.text)

    def test_both_skill_sections_rendered(self):
        self.assertIn('&OPENQP', self.text)
        self.assertIn('&XTB', self.text)
        # Each skill keeps its own method line
        openqp_block = self.text.split('&OPENQP')[1].split('&')[0]
        xtb_block = self.text.split('&XTB')[1].split('&')[0]
        self.assertIn('mrsf', openqp_block)
        self.assertIn('gfn2', xtb_block)

    def test_control_theory_lists_both_skills(self):
        control_block = self.text.split('&CONTROL')[1].split('&')[0]
        self.assertIn('openqp xtb', control_block)

    def test_a_per_center_state_pair_stays_a_list_on_both_sides(self):
        # The classic carrier splits a value on whitespace, so a Python list
        # would arrive as the literal "[0, 0]" that no type code can parse -
        # which is how the first multi-center crossing and excited-state
        # examples failed on the input leg only (#460). Written as "0 0" and
        # read back through the 'il' type code, the pair survives; a missing
        # second surface stays missing.
        import os as _os
        import tempfile as _tempfile

        from pyrai3md.kernel.environment import Environment
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate

        examples = _os.path.abspath(_os.path.join(
            _os.path.dirname(__file__), _os.pardir, _os.pardir, 'examples'))
        for name, state_i, state_j in (
                ('multicenter_qmmm_mecp', [0, 0], [1, 1]),
                ('multicenter_qmmm_ex_opt', [1, 1], None)):
            work = _tempfile.mkdtemp()
            self.addCleanup(__import__('shutil').rmtree, work, True)
            json_path = _os.path.join(work, 'case.json')
            yaml_to_config(_os.path.join(examples, '%s.yaml' % name),
                           json_path)
            text = convert(json_path, _os.path.join(work, 'case'),
                           skill_name='examples')
            block = text.split('&OPTIMIZE')[1].split('&')[0]
            self.assertIn('state_i', block)
            self.assertNotIn('[', block, name)
            case_input = _os.path.join(work, 'case.input')
            with open(case_input, 'w') as handle:
                handle.write(text)
            Environment().initialize()
            config = validate(normalize(build_config(
                parse_input(case_input))))
            self.assertEqual(config.optimize.opt_state_i, state_i, name)
            self.assertEqual(config.optimize.opt_state_j, state_j, name)

    def test_xyz_file_written(self):
        xyz_path = os.path.join(self.tmpdir, 'mc.xyz')
        self.assertTrue(os.path.isfile(xyz_path))
        with open(xyz_path, 'r', encoding='utf-8') as f:
            first = f.readline().strip()
        self.assertEqual(first, '60')


class TestMulticenterAgentConfig(unittest.TestCase):
    """convert_to_pyrai_config emits center strings and both skills."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, 'mc.json')
        yaml_to_config(_MULTICENTER_YAML, self.config_path)
        self.agent_path = os.path.join(self.tmpdir, 'mc.config')
        self.sectioned = convert_to_pyrai_config(self.config_path,
                                                 self.agent_path)

    def test_molecule_section_center_strings(self):
        molecule = self.sectioned['molecule']
        self.assertEqual(molecule['highlevel'], '1-6; 7-12')
        self.assertEqual(molecule['ci'], '3; 3')
        self.assertEqual(molecule['spin'], '0; 0')
        self.assertEqual(molecule['coupling'],
                         '1 2, 2 3, 1 3; 1 2, 2 3, 1 3')

    def test_md_section_root(self):
        self.assertIn('md', self.sectioned)
        self.assertEqual(self.sectioned['md']['root'], '1; 1')

    def test_both_skill_sections_present(self):
        self.assertIn('openqp_skill', self.sectioned)
        self.assertIn('xtb_skill', self.sectioned)
        self.assertEqual(self.sectioned['openqp_skill']['method'], 'mrsf')
        self.assertEqual(self.sectioned['xtb_skill']['method'], 'gfn2')

    def test_config_file_written(self):
        self.assertTrue(os.path.isfile(self.agent_path))
        with open(self.agent_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        self.assertIn('openqp_skill', loaded)
        self.assertIn('xtb_skill', loaded)


def _flat_task_table(raw):
    """Per-binding (method, profile, jobtype) the task queue is handed.

    Built through the same loader chain a run uses (parse_input ->
    build_config -> build_state), so the readings below are what a
    backend is constructed from, not what the file spells.
    """
    from pyrai3md.schema.adapter import build_config
    from pyrai3md.data.build import build_state
    state = build_state(build_config(raw), raw_config=raw)
    table = {}
    for task in state.theory + state.reference:
        skill = (task.get('config') or {}).get('skill_config') or {}
        table[task['skill_binding_id']] = (skill.get('method'),
                                           skill.get('profile'),
                                           skill.get('jobtype'))
    return table





class TestOneSkillBoundTwice(unittest.TestCase):
    """`theory: [pyscf, pyscf]` is a multi-task run, not a single-skill job.

    The converter used to decide `skill_sections` by counting DISTINCT section
    spellings, so a run that binds one skill at two positions wrote no per-skill
    block at all. The consequence was not cosmetic: that flat config carried no
    nested value, `schema/adapter.py` refused it as pure-flat input
    (`Flat config input is not supported (keys: atoms, charges, coordinates,
    driver, highlevel, jobtype, method, multiplicity)`, measured on
    `optimize_ts_multiscale.yaml`), and the agent dialect - the one the
    converter's own docstring says the entry point must read identically - was
    unavailable for the multiscale carriers.
    """

    _MULTI = os.path.join(_EXAMPLES_DIR, 'optimize_ts_multiscale.yaml')
    _TWO_BACKEND = os.path.join(_EXAMPLES_DIR,
                                'optimize_cell_meci_lammps.yaml')
    _SINGLE = os.path.join(_EXAMPLES_DIR, 'gradient.yaml')

    def _render(self, yaml_path):
        tmpdir = tempfile.mkdtemp()
        config_path = os.path.join(tmpdir, os.path.basename(yaml_path)
                                   .replace('.yaml', '.json'))
        config = yaml_to_config(yaml_path, config_path)
        text = convert(config_path, config_path.replace('.json', '.input'))
        return config, text

    def test_the_two_bindings_keep_one_block(self):
        config, _text = self._render(self._MULTI)
        # Two theory entries, one spelling: the key is written because the run
        # binds more than one task, and it holds the one block that exists.
        self.assertEqual(config['theory'], ['pyscf', 'pyscf'])
        self.assertEqual(sorted(config['skill_sections']), ['pyscf'])
        self.assertEqual(config['skill_sections']['pyscf']['method'], 'hf')
        # The primary stays flat too, so the single-skill builder path is
        # still readable from the same file.
        self.assertEqual(config['method'], 'hf')

    def test_the_flat_config_builds_the_yaml_modes_queue(self):
        from pyrai3md.interface.parser import parse_input
        config, _text = self._render(self._MULTI)
        self.assertEqual(
            _flat_task_table(config),
            _flat_task_table(parse_input(self._MULTI)),
            'the flat mode does not hand the queue what the yaml mode hands it')

    def test_the_system_fields_stay_where_the_two_backend_cases_put_them(self):
        # Keeping the block makes this example render through the sectioned
        # path, and that path reads calculation keywords from the block: the
        # two lines the example used to print twice leave `&PYSCF`. This is
        # the shape every already-shipped two-backend example has (asserted
        # against it below), and &MOLECULE still carries both fields.
        for path in (self._MULTI, self._TWO_BACKEND):
            _config, text = self._render(path)
            self.assertEqual(text.count('&PYSCF'), 1, path)
            pyscf_block = text.split('&PYSCF')[1].split('&')[0]
            molecule = text.split('&MOLECULE')[1].split('&')[0]
            self.assertNotIn('charges', pyscf_block, path)
            self.assertNotIn('multiplicity', pyscf_block, path)
            self.assertIn('charges', molecule, path)
            self.assertIn('multiplicity', molecule, path)

    def test_a_single_skill_example_still_writes_no_block(self):
        # The boundary is entries bound, not sections seen: an example with
        # one theory entry keeps the historical byte-for-byte config, block
        # and System lines inside its own section included.
        config, text = self._render(self._SINGLE)
        self.assertNotIn('skill_sections', config)
        pyscf_block = text.split('&PYSCF')[1].split('&')[0]
        self.assertIn('charges', pyscf_block)


class TestMulticenterChargeSpinRendering(unittest.TestCase):
    """charges/multiplicity join the center axis in the converter."""

    def test_scalar_charges_alone_is_not_multicenter(self):
        self.assertFalse(_is_multicenter({'charges': 0, 'multiplicity': 1}))
        # ... but a scalar alongside a center field still renders in the
        # multi-center branch (broadcast to every center downstream).
        self.assertTrue(_is_multicenter(
            {'highlevel': '1-6; 7-12', 'charges': 0}))

    def test_list_and_string_charges_are_multicenter(self):
        self.assertTrue(_is_multicenter({'charges': [0, -1]}))
        self.assertTrue(_is_multicenter({'multiplicity': [1, 2]}))
        self.assertTrue(_is_multicenter({'charges': '0; -1'}))

    def test_render_center_string(self):
        self.assertEqual(_render_center_string([0, -1]), '0; -1')
        self.assertEqual(_render_center_string([1, 2]), '1; 2')
        self.assertEqual(_render_center_string('0; -1'), '0; -1')
        self.assertEqual(_render_center_string(0), '0')

    def test_molecule_pairs_render_charges_as_center_string(self):
        header, pairs = _build_molecule_pairs({
            'highlevel': [[0, 1], [2, 3]], 'ci': [[1], [1]],
            'charges': [0, -1], 'multiplicity': [1, 2],
        })
        rendered = dict(pairs)
        self.assertEqual(header, 'molecule')
        self.assertEqual(rendered['charges'], '0; -1')
        self.assertEqual(rendered['multiplicity'], '1; 2')

    def test_example_scalar_charges_broadcast_in_classic_input(self):
        # multicenter_gradient.yaml keeps scalar charges: 0 /
        # multiplicity: 1; they render verbatim (single value) and the
        # framework broadcasts them to both centers.
        tmpdir = tempfile.mkdtemp()
        config_path = os.path.join(tmpdir, 'mc.json')
        yaml_to_config(_MULTICENTER_YAML, config_path)
        text = convert(config_path, os.path.join(tmpdir, 'mc.input'))
        molecule_block = text.split('&MOLECULE')[1].split('&')[0]
        self.assertIn('charges', molecule_block)
        self.assertIn('multiplicity', molecule_block)


class TestSingleCenterRegression(unittest.TestCase):
    """A plain one-skill example keeps the historical rendering."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, 'grad.json')
        self.config = yaml_to_config(_GRADIENT_YAML, self.config_path)
        self.text = convert(self.config_path,
                            os.path.join(self.tmpdir, 'grad.input'))
        self.sectioned = convert_to_pyrai_config(self.config_path)

    def test_no_skill_sections_for_single_skill(self):
        # A single theory skill needs no per-skill preservation
        self.assertNotIn('skill_sections', self.config)
        self.assertEqual(self.config['skill_name'], 'pyscf_skill')

    def test_single_method_section_and_no_md(self):
        self.assertIn('&PYSCF', self.text)
        self.assertNotIn('&MD', self.text)
        self.assertNotIn('&OPENQP', self.text)

    def test_sectioned_dict_single_skill(self):
        self.assertIn('pyscf_skill', self.sectioned)
        self.assertNotIn('md', self.sectioned)


if __name__ == '__main__':
    unittest.main()
