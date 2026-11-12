"""
======================================================================
Test suite for the traditional CLI entry point (pmd.py), the
interface.parser module, and the schema pipeline.

Tests cover:
- Module imports
- Parser: text-format and JSON-format input parsing
- Adapter: raw dict to Config conversion with type conversion
- Normalize: derived quantity computation
- Validate: consistency checks
- resolve_skill: skill discovery logic

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import os
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Inline mock input matching the classic mixaimd format.
# Covers all sections and values needed by the test suite.
MOCK_INPUT_TEXT = """\
&control
title atod
jobtype md
qm nn
verbose 2

&molcas
project molcas_proj
state_average 2 2

&molecule
ci 2 2
spin 0 1
natom 4

&md
temp 300
step 2
root 2
sfhp gsh
size 10.0
multipool 4

&nn
model nn_model

&eg
method xtb

&eg2
method orca

&soc
method soc_method

&soc2
method soc_method2
"""


class TestImports(unittest.TestCase):
    """Test that all required modules can be imported."""

    def test_import_interface(self):
        from pyrai3md.interface import parser, loader
        self.assertTrue(hasattr(parser, 'parse_input'))
        self.assertTrue(hasattr(loader, 'load_config'))

    def test_import_schema(self):
        import pyrai3md.schema as schema
        from pyrai3md.schema.config import Config, Control, System, Dynamics
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate
        self.assertTrue(callable(build_config))
        # The retired task model: the planner's task descriptor is a
        # dict, and this class was never instantiated. Pinned so it
        # cannot grow back.
        self.assertFalse(hasattr(schema, 'Task'))

    def test_import_pmd(self):
        from pyrai3md.pmd import main, run
        self.assertTrue(callable(main))

    def test_import_agent(self):
        from pyrai3md.agent import PyRAI3Agent
        self.assertTrue(hasattr(PyRAI3Agent, 'from_file'))
        self.assertTrue(hasattr(PyRAI3Agent, 'from_dict'))


class TestParserTextFormat(unittest.TestCase):
    """Test parser with text-format input files."""

    @classmethod
    def setUpClass(cls):
        from pyrai3md.interface.parser import _parse_text
        cls.raw = _parse_text(MOCK_INPUT_TEXT.splitlines(keepends=True))

    def test_sections_present(self):
        expected = ['control', 'molcas', 'molecule', 'md', 'nn', 'eg', 'eg2', 'soc', 'soc2']
        for section in expected:
            self.assertIn(section, self.raw)

    def test_control_values(self):
        ctrl = self.raw['control']
        self.assertEqual(ctrl['title'], 'atod')
        self.assertEqual(ctrl['jobtype'], 'md')
        self.assertEqual(ctrl['qm'], 'nn')

    def test_molecule_values(self):
        mol = self.raw['molecule']
        self.assertEqual(mol['ci'], '2 2')
        self.assertEqual(mol['spin'], '0 1')

    def test_values_are_raw_strings(self):
        self.assertIsInstance(self.raw['molecule']['ci'], str)
        self.assertIsInstance(self.raw['md']['step'], str)


class TestParserInlineText(unittest.TestCase):
    """Test parser with inline text input."""

    def _parse_text(self, text):
        from pyrai3md.interface.parser import _parse_text
        return _parse_text(text.splitlines(keepends=True))

    def test_basic_section(self):
        result = self._parse_text("&CONTROL\ntitle test_job\njobtype sp\n")
        self.assertEqual(result['control']['title'], 'test_job')

    def test_comments_skipped(self):
        result = self._parse_text("# comment\n&CONTROL\ntitle test\n")
        self.assertEqual(result['control']['title'], 'test')
        self.assertEqual(len(result['control']), 1)

    def test_empty_input(self):
        self.assertEqual(self._parse_text(""), {})


class TestAdapter(unittest.TestCase):
    """Test the adapter build_config function."""

    def test_sectioned_input(self):
        from pyrai3md.interface.parser import _parse_text
        from pyrai3md.schema.adapter import build_config

        raw = _parse_text(MOCK_INPUT_TEXT.splitlines(keepends=True))
        config = build_config(raw)
        self.assertEqual(config.title, 'atod')
        self.assertEqual(config.jobtype, 'md')
        self.assertEqual(config.theory, ['nn'])
        self.assertEqual(config.ci, [2, 2])
        self.assertEqual(config.spin, [0, 1])
        self.assertEqual(config.sfhp, 'gsh')
        self.assertEqual(config.step, 2)
        # MOCK_INPUT writes the external 1-based 'root 2' (S1); the
        # adapter decrements it once to the internal 0-based index.
        self.assertEqual(config.root, 1)

    def test_type_conversion(self):
        from pyrai3md.schema.adapter import build_config

        raw = {'control': {'gl_seed': '4', 'verbose': '2'},
               'md': {'temp': '300', 'step': '100',
                      'multipool': '4'}}
        config = build_config(raw)
        self.assertEqual(config.gl_seed, 4)
        self.assertEqual(config.verbose, 2)
        self.assertEqual(config.temperature, 300.0)
        self.assertEqual(config.step, 100)
        # multipool lives in several workflow sections; the &md value
        # lands in the nested dynamics submodel (the flat accessor is
        # ambiguous across sections).
        self.assertEqual(config.dynamics.multipool, 4)

    def test_coupling_conversion(self):
        from pyrai3md.schema.adapter import build_config

        raw = {'molecule': {'coupling': '1 3, 1 4, 2 3, 2 4'}}
        config = build_config(raw)
        # Input is 1-based per documentation; adapter converts to 0-based
        self.assertEqual(config.coupling, [[0, 2], [0, 3], [1, 2], [1, 3]])

    def test_flat_input_rejected(self):
        from pyrai3md.schema.adapter import build_config

        raw = {'title': 'flat_test', 'jobtype': 'md', 'step': 50}
        with self.assertRaises(TypeError):
            build_config(raw)

    def test_dotted_flat_input(self):
        from pyrai3md.schema.adapter import build_config

        # Dotted flat format (agent medium): single-level dict with
        # 'section.keyword' keys. Per-section dispatch keywords must
        # coexist without overwriting each other.
        raw = {'control.title': 'dotted_test',
               'control.jobtype': 'adaptive',
               'md.step': 100,
               'md.multipool': 4,
               'adaptive.multipool': 2,
               'skill_name': 'molcas_skill'}
        config = build_config(raw)
        self.assertEqual(config.title, 'dotted_test')
        self.assertEqual(config.jobtype, 'adaptive')
        self.assertEqual(config.step, 100)
        self.assertEqual(config.dynamics.multipool, 4)
        self.assertEqual(config.adaptive.multipool, 2)

    def test_dotted_flat_conflict_rejected(self):
        from pyrai3md.schema.adapter import build_config

        raw = {'md': {'step': 100}, 'md.multipool': 4}
        with self.assertRaises(ValueError):
            build_config(raw)

    def test_unknown_fields_ignored(self):
        from pyrai3md.schema.adapter import build_config

        raw = {'control': {'title': 'test', 'unknown_key': 'value'}}
        config = build_config(raw)
        self.assertEqual(config.title, 'test')


class TestNormalize(unittest.TestCase):
    """Test the normalize function."""

    def test_natom_computation(self):
        from pyrai3md.schema.config import Config
        from pyrai3md.schema.normalize import normalize

        c = Config(atoms=['C', 'H', 'H', 'H', 'H'])
        c2 = normalize(c)
        self.assertEqual(c2.natom, 5)

    def test_electron_computation(self):
        from pyrai3md.schema.config import Config
        from pyrai3md.schema.normalize import normalize

        c = Config(atoms=['C', 'H', 'H', 'H', 'H'], charges=0, multiplicity=1)
        c2 = normalize(c)
        self.assertEqual(c2.total_electron, 10)  # C=6, H*4=4
        self.assertEqual(c2.nalpha, 5)
        self.assertEqual(c2.nbeta, 5)

    def test_nstate_from_ci(self):
        from pyrai3md.schema.config import Config
        from pyrai3md.schema.normalize import normalize

        # R1 unified-center contract: ci and spin must have the same
        # length (one spin value per ci state block). A mismatch is a
        # hard error instead of the old silent zip truncation, so the
        # explicit spin block is required to reach nstate=4.
        c = Config(ci=[2, 2], spin=[0, 0])
        c2 = normalize(c)
        self.assertEqual(c2.nstate, 4)

    def test_pair_counts(self):
        from pyrai3md.schema.config import Config
        from pyrai3md.schema.normalize import normalize

        # 0-based, as this layer always receives them (the adapter does the
        # 1-based decrement; a pair naming state 4 of a 4-state run is out
        # of range and now raises rather than being filed under NAC).
        c = Config(ci=[2, 2], spin=[0, 1],
                   coupling=[[0, 1], [2, 3], [0, 2], [1, 3]])
        c2 = normalize(c)
        self.assertEqual(c2.npair, 4)
        self.assertEqual(c2.nnac, 2)
        self.assertEqual(c2.nsoc, 2)

    def test_nothing_is_invented_for_an_undeclared_coupling(self):
        # Adjacent pairs used to be filled in here so run logs could label
        # NAC blocks; that read back as couplings the run never asked for.
        from pyrai3md.schema.config import Config
        from pyrai3md.schema.normalize import normalize

        c = Config(ci=[3], spin=[0])
        c2 = normalize(c)
        self.assertEqual(c2.nstate, 3)
        self.assertEqual(c2.coupling, [])
        self.assertEqual(c2.nnac, 0)

    def test_declared_pairs_fill_the_authority(self):
        # The nested input format's extra entry: naming the pair maps alone
        # is a complete declaration, and `coupling` is filled in from it.
        from pyrai3md.schema.config import Config
        from pyrai3md.schema.normalize import normalize

        c = Config(ci=[3], spin=[0], nac_pair_map=[[0, 1]])
        c2 = normalize(c)
        self.assertEqual(c2.coupling, [[0, 1]])
        self.assertEqual(c2.nac_pair_map, [[0, 1]])

    def test_coupling_explicit_not_overridden(self):
        from pyrai3md.schema.config import Config
        from pyrai3md.schema.normalize import normalize

        c = Config(ci=[2, 2], spin=[0, 1], coupling=[[0, 3]])
        c2 = normalize(c)
        self.assertEqual(c2.coupling, [[0, 3]])
        self.assertEqual(c2.soc_pair_map, [[0, 3]])

    def test_history_below_minimum_normalized_to_4(self):
        from pyrai3md.schema.config import Config
        from pyrai3md.schema.normalize import normalize

        c = Config(dynamics={'history': 2})
        c2 = normalize(c)
        self.assertEqual(c2.dynamics.history, 4)

    def test_history_above_minimum_kept(self):
        from pyrai3md.schema.config import Config
        from pyrai3md.schema.normalize import normalize

        c = Config(dynamics={'history': 10})
        c2 = normalize(c)
        self.assertEqual(c2.dynamics.history, 10)

    def test_history_default_normalized(self):
        from pyrai3md.schema.config import Config
        from pyrai3md.schema.normalize import normalize

        c2 = normalize(Config())
        self.assertEqual(c2.dynamics.history, 4)

    def test_classic_input_history_keyword_routed_to_dynamics(self):
        from pyrai3md.interface.parser import _parse_text
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize

        raw = _parse_text([
            '&control\n', 'title hist\n', 'jobtype md\n', 'qm nn\n',
            '&md\n', 'step 10\n', 'history 8\n',
        ])
        config = normalize(build_config(raw))
        self.assertEqual(config.dynamics.history, 8)


class TestValidate(unittest.TestCase):
    """Test the validate function."""

    def test_valid_config(self):
        from pyrai3md.schema.config import Config
        from pyrai3md.schema.validate import validate

        c = Config(jobtype='energy', title='test')
        c2 = validate(c)
        self.assertEqual(c2.jobtype, 'energy')

    def test_invalid_jobtype(self):
        from pyrai3md.schema.config import Config
        from pyrai3md.schema.validate import validate

        c = Config(jobtype='invalid_type')
        with self.assertRaises(ValueError):
            validate(c)

    def test_invalid_temperature(self):
        from pyrai3md.schema.config import Config
        from pyrai3md.schema.validate import validate

        c = Config(temperature=-100.0)
        with self.assertRaises(ValueError):
            validate(c)

    def test_invalid_multiplicity(self):
        from pyrai3md.schema.config import Config
        from pyrai3md.schema.validate import validate

        c = Config(atoms=['H'], multiplicity=0)
        with self.assertRaises(ValueError):
            validate(c)


class TestValidateEnsemble(unittest.TestCase):
    """Test the NVE/NVT/NPT ensemble combination rules (docs section 3)."""

    def _check(self, **overrides):
        from pyrai3md.schema.validate import validate_ensemble
        data = {'ensemble': 'nve', 'thermostat': 'off', 'barostat': 'off',
                'barostat_mode': 'anisotropic', 'thermostat_tau': 100.0,
                'barostat_tau': 1000.0, 'compressibility': 4.5e-5}
        data.update(overrides)
        return validate_ensemble(data)

    def test_nve_requires_thermostat_off(self):
        self._check(ensemble='nve', thermostat='off')
        with self.assertRaises(ValueError):
            self._check(ensemble='nve', thermostat='csvr')

    def test_nvt_requires_a_thermostat(self):
        with self.assertRaises(ValueError):
            self._check(ensemble='nvt', thermostat='off')
        self._check(ensemble='nvt', thermostat='csvr')

    def test_npt_requires_barostat_and_thermostat(self):
        with self.assertRaises(ValueError):
            self._check(ensemble='npt', thermostat='nose-hoover',
                        barostat='off')
        with self.assertRaises(ValueError):
            self._check(ensemble='npt', thermostat='off',
                        barostat='berendsen')
        self._check(ensemble='npt', thermostat='nose-hoover',
                    barostat='berendsen')

    def test_mttk_only_with_nose_hoover(self):
        self._check(ensemble='npt', thermostat='nose-hoover-chain',
                    barostat='mttk')
        with self.assertRaises(ValueError):
            self._check(ensemble='npt', thermostat='csvr', barostat='mttk')

    def test_aliases_normalized(self):
        # nh -> nose-hoover, csnr -> csvr, pr -> parrinello-rahman.
        self._check(ensemble='nvt', thermostat='nh')
        self._check(ensemble='nvt', thermostat='csnr')
        self._check(ensemble='npt', thermostat='nh', barostat='pr')

    def test_unknown_tokens_rejected(self):
        for key, value in (('ensemble', 'nps'), ('thermostat', 'langevin'),
                           ('barostat', 'andersen'),
                           ('barostat_mode', 'semi')):
            with self.subTest(key=key):
                overrides = {'ensemble': 'nvt', 'thermostat': 'nh'}
                overrides[key] = value
                with self.assertRaises(ValueError):
                    self._check(**overrides)

    def test_nonpositive_coupling_rejected(self):
        with self.assertRaises(ValueError):
            self._check(ensemble='nvt', thermostat='nh', thermostat_tau=0.0)
        with self.assertRaises(ValueError):
            self._check(ensemble='nvt', thermostat='nh',
                        compressibility=-1.0)


class TestResolveSkill(unittest.TestCase):
    """Test the Maestro.resolve_skill function."""

    def test_resolve_explicit_skill_name(self):
        """When skill_name is set explicitly, resolve_skill returns it."""
        from pyrai3md.schema.config import Config
        from pyrai3md.kernel.maestro import Maestro

        config = Config(jobtype='energy', skill_name='molcas_skill')
        maestro = Maestro()
        maestro.populate()
        result = maestro.resolve_skill(config)
        self.assertEqual(result, 'molcas_skill')

    def test_resolve_from_theory_label(self):
        """When skill_name is empty, resolve_skill resolves from theory label via Maestro."""
        from pyrai3md.schema.config import Config
        from pyrai3md.kernel.maestro import Maestro

        config = Config(jobtype='energy', theory=['molcas'])
        maestro = Maestro()
        maestro.populate()
        result = maestro.resolve_skill(config)
        self.assertEqual(result, 'molcas_skill')

    def test_resolve_unknown_label_returns_empty(self):
        """When theory label is unknown, resolve_skill returns empty string."""
        from pyrai3md.schema.config import Config
        from pyrai3md.kernel.maestro import Maestro

        config = Config(jobtype='energy', theory=['unknown_method'])
        maestro = Maestro()
        maestro.populate()
        result = maestro.resolve_skill(config)
        self.assertEqual(result, '')


class TestAdaptive(unittest.TestCase):
    """Test Adaptive model and validate_adaptive logic."""

    def test_adaptive_import(self):
        from pyrai3md.schema import Adaptive
        self.assertTrue(hasattr(Adaptive, 'model_fields'))
        self.assertIn('maxenergy', Adaptive.model_fields)
        self.assertIn('maxgrad', Adaptive.model_fields)
        self.assertIn('maxsoc', Adaptive.model_fields)

    def test_adaptive_defaults(self):
        from pyrai3md.schema import Adaptive
        a = Adaptive()
        self.assertEqual(a.maxenergy, 0.05)
        self.assertEqual(a.minenergy, 0.02)
        self.assertEqual(a.maxgrad, 0.15)
        self.assertEqual(a.maxsoc, 50.0)
        self.assertEqual(a.maxiter, 1)
        self.assertEqual(a.refine, 0)
        self.assertEqual(a.record_step, 10)
        self.assertEqual(a.keep_md, 1)
        self.assertEqual(a.check_population, 1)
        self.assertEqual(a.check_energy_order, 1)
        self.assertAlmostEqual(a.gap_tolerance, 1e-6)
        self.assertIsNone(a.md_multipool)
        self.assertIsNone(a.reference_multipool)
        self.assertEqual(a.sample_energy, 1)
        self.assertEqual(a.sample_grad, 1)
        self.assertEqual(a.sample_nac, 0)
        self.assertEqual(a.sample_soc, 0)

    def test_adaptive_pop_tolerance_is_retired(self):
        from pyrai3md.schema.adapter import build_config

        with self.assertRaises(ValueError) as caught:
            build_config({'control': {'title': 't'},
                          'adaptive': {'pop_tolerance': 0.02}})
        self.assertIn('md_maxpop', str(caught.exception))

    def test_sample_switches_reconcile_with_md(self):
        import warnings
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize

        raw = {'control': {'jobtype': 'adaptive'},
               'adaptive': {'sample_nac': 1, 'sample_soc': 0},
               'dynamics': {'nactype': 'ktdc', 'dosoc': 1}}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            config = normalize(build_config(raw))
        self.assertEqual(config.dynamics.nactype, 'nac')
        self.assertEqual(config.adaptive.sample_nac, 1)
        self.assertEqual(config.adaptive.sample_soc, 1)
        self.assertTrue(any('sample_nac' in str(item.message)
                            for item in caught))

    def test_sample_energy_and_grad_are_required(self):
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize

        raw = {'control': {'jobtype': 'adaptive'},
               'adaptive': {'sample_energy': 0}}
        with self.assertRaises(ValueError):
            normalize(build_config(raw))

    def test_adaptive_section_parsing(self):
        from pyrai3md.interface.parser import _parse_text
        from pyrai3md.schema.adapter import build_config

        text = """\
&control
title adaptive_test
jobtype adaptive

&adaptive
maxenergy 0.1
minenergy 0.03
maxgrad 0.2
maxiter 5
refine 1
refine_num 8
record_step 12
keep_md 0
check_population 0
check_energy_order 0
gap_tolerance 0.001
md_multitask 1
md_multipool 3
reference_multitask 1
reference_multipool 2
"""
        raw = _parse_text(text.splitlines(keepends=True))
        self.assertIn('adaptive', raw)
        self.assertEqual(raw['adaptive']['maxenergy'], '0.1')

        config = build_config(raw)
        self.assertEqual(config.maxenergy, 0.1)
        self.assertEqual(config.minenergy, 0.03)
        self.assertEqual(config.maxgrad, 0.2)
        self.assertEqual(config.maxiter, 5)
        self.assertEqual(config.refine, 1)
        self.assertEqual(config.refine_num, 8)
        self.assertEqual(config.record_step, 12)
        self.assertEqual(config.keep_md, 0)
        self.assertEqual(config.check_population, 0)
        self.assertEqual(config.check_energy_order, 0)
        self.assertAlmostEqual(config.gap_tolerance, 0.001)
        self.assertTrue(config.md_multitask)
        self.assertEqual(config.md_multipool, 3)
        self.assertTrue(config.reference_multitask)
        self.assertEqual(config.reference_multipool, 2)

    def test_adaptive_type_conversion(self):
        from pyrai3md.schema.adapter import build_config

        raw = {'adaptive': {'maxenergy': '0.08', 'maxiter': '10', 'fwdenergy': '2'}}
        config = build_config(raw)
        self.assertIsInstance(config.maxenergy, float)
        self.assertEqual(config.maxenergy, 0.08)
        self.assertIsInstance(config.maxiter, int)
        self.assertEqual(config.maxiter, 10)
        self.assertIsInstance(config.fwdenergy, int)
        self.assertEqual(config.fwdenergy, 2)

    def test_adaptive_accessible_via_config(self):
        """Adaptive parameters are accessible through merged Config model."""
        from pyrai3md.schema.config import Config

        c = Config(maxenergy=0.2, minenergy=0.05, maxiter=3)
        self.assertEqual(c.maxenergy, 0.2)
        self.assertEqual(c.minenergy, 0.05)
        self.assertEqual(c.maxiter, 3)

    def test_validate_adaptive_threshold_consistency(self):
        from pyrai3md.schema.config import Config
        from pyrai3md.schema.validate import validate

        # Valid: max >= min
        c = Config(maxenergy=0.1, minenergy=0.05)
        c2 = validate(c)
        self.assertEqual(c2.maxenergy, 0.1)

    def test_validate_adaptive_invalid_threshold(self):
        from pyrai3md.schema.config import Config
        from pyrai3md.schema.validate import validate

        # Invalid: maxenergy < minenergy
        c = Config(maxenergy=0.01, minenergy=0.1)
        with self.assertRaises(ValueError):
            validate(c)

    def test_validate_adaptive_invalid_grad_threshold(self):
        from pyrai3md.schema.config import Config
        from pyrai3md.schema.validate import validate

        # Invalid: maxgrad < mingrad
        c = Config(maxgrad=0.01, mingrad=0.2)
        with self.assertRaises(ValueError):
            validate(c)

    def test_validate_adaptive_negative_refine(self):
        from pyrai3md.schema.config import Config
        from pyrai3md.schema.validate import validate

        # Invalid: refine < 0
        c = Config(refine=-1)
        with self.assertRaises(ValueError):
            validate(c)


class TestFullPipeline(unittest.TestCase):
    """Test the full pipeline: parse -> build -> normalize -> validate."""

    def test_classic_input_pipeline(self):
        from pyrai3md.interface.parser import _parse_text
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate

        raw = _parse_text(MOCK_INPUT_TEXT.splitlines(keepends=True))
        config = build_config(raw)
        config = normalize(config)
        config = validate(config)

        self.assertEqual(config.title, 'atod')
        self.assertEqual(config.jobtype, 'md')
        self.assertEqual(config.nstate, 4)
        self.assertEqual(config.sfhp, 'gsh')


class TestResultDocument(unittest.TestCase):
    """The result file a dispatched runtime hands back to its parent.

    Two shapes this round found missing: an array nested below the top level
    (an md record is a list of per-step dictionaries), which crashed the
    Slurm child on the way out of its own dump; and a dump that fails, whose
    half-written file the parent then reads as a corrupt document instead of
    a dead child.
    """

    def test_nested_arrays_reach_the_document(self):
        import numpy as np

        from pyrai3md import pmd as pmd_module

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                pmd_module._write_result_json({
                    'pop': [{'step': 1, 'values': np.array([1.0, 0.0])},
                            {'step': 2, 'values': np.array([0.5, 0.5])}],
                    'gradient': np.zeros((2, 3)),
                }, None)
                with open('result.json') as handle:
                    data = json.load(handle)
            finally:
                os.chdir(cwd)
        self.assertEqual(data['pop'][1]['values'], [0.5, 0.5])
        self.assertEqual(data['gradient'], [[0.0] * 3, [0.0] * 3])

    def test_a_failed_dump_leaves_no_document_behind(self):
        # A set has no JSON form, so the dump fails partway: exactly the
        # state that left a 30230-byte truncated result.json on the
        # test environment for the parent to misread.
        from pyrai3md.utilities.serialization import write_json

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'result.json')
            with self.assertRaises(TypeError):
                write_json(path, {'pop': {'not', 'serializable'}})
            self.assertFalse(os.path.exists(path))
            self.assertEqual(os.listdir(tmp), [],
                             'a crashed dump must not leave a partial file')


if __name__ == '__main__':
    unittest.main()
