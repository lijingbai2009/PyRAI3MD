"""
======================================================================
Test suite for the `optimize_cell` keyword across the three carriers
(qualification/certification/converter.py and interface/parser.py).

A variable-cell run is decided by one boolean, and the same boolean has to
arrive through three spellings of one input: the YAML a user edits, the agent
JSON the generator writes from it, and the classic input the text carrier
runs. It has been lost before - prefixing the keyword invented
`opt_optimize_cell`, a name no model owns, and a run that asked to relax its
box relaxed only its atoms while every other report stayed green
(dev/methods/OPTIMIZE_DEVELOPMENT.md section 11). The shipped examples now
carry a cell run each, so this walks them instead of asserting a spelling in
isolation: whatever the example directory declares must be what the optimizer
is handed, on both generated carriers.

Oct 15 2026
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

import glob
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import yaml

from pyrai3md.interface.parser import parse_input
from pyrai3md.qualification.certification.converter import (
    convert as convert_to_classic_input, yaml_to_config,
)
from pyrai3md.schema.adapter import OPTIMIZE_PASSTHROUGH_KEYS


def _examples_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, os.pardir, os.pardir, 'examples')


def _cell_examples():
    """Every shipped example that asks the box to move."""
    found = []
    for path in sorted(glob.glob(os.path.join(_examples_root(), '*.yaml'))):
        with open(path) as handle:
            try:
                spec = yaml.safe_load(handle)
            except yaml.YAMLError:
                continue
        optimize = (spec or {}).get('optimize') or {}
        if optimize.get('optimize_cell'):
            found.append(path)
    return found


class TestTheCellSwitchIsDeclaredSomewhere(unittest.TestCase):
    """A guard against this file quietly becoming vacuous."""

    def test_the_example_directory_carries_a_cell_run(self):
        # If nobody ships a variable-cell example, the parity assertions below
        # would pass on an empty walk - which is precisely how the keyword was
        # lost the first time: no run exercised it, so no carrier was read.
        self.assertTrue(_cell_examples(),
                        'no shipped example declares optimize_cell')


class TestTheSwitchSurvivesBothGeneratedCarriers(unittest.TestCase):
    """YAML -> agent JSON -> classic input -> reader, on each shipped case."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='cell_carrier_')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _config_of(self, example):
        config_path = os.path.join(self.root, os.path.basename(example) + '.json')
        yaml_to_config(example, config_path)
        import json
        with open(config_path) as handle:
            return json.load(handle)

    def test_the_agent_json_names_the_keyword_as_the_schema_does(self):
        for example in _cell_examples():
            config = self._config_of(example)
            name = os.path.basename(example)
            self.assertIs(config.get('optimize_cell'), True,
                          msg='%s: the JSON carrier lost the switch' % name)
            self.assertNotIn('opt_optimize_cell', config,
                             msg='%s: the generator prefixed a keyword that '
                                 'already names its own section' % name)

    def test_the_classic_input_carries_the_line(self):
        for example in _cell_examples():
            config_path = os.path.join(self.root,
                                       os.path.basename(example) + '.json')
            yaml_to_config(example, config_path)
            input_path = os.path.join(
                self.root, os.path.basename(example).split('.')[0] + '.input')
            convert_to_classic_input(config_path, input_path)
            with open(input_path) as handle:
                text = handle.read()
            block = text.split('&OPTIMIZE', 1)
            self.assertGreater(len(block), 1, os.path.basename(example))
            lines = block[1].split('&END', 1)[0]
            self.assertIn('optimize_cell', lines,
                          msg='%s: &OPTIMIZE carries no cell switch, so the '
                              'text carrier would relax a molecule while the '
                              'YAML carrier relaxes a box'
                              % os.path.basename(example))

    def test_the_reader_hands_the_optimizer_the_same_boolean(self):
        for example in _cell_examples():
            config_path = os.path.join(self.root,
                                       os.path.basename(example) + '.json')
            yaml_to_config(example, config_path)
            input_path = os.path.join(
                self.root, os.path.basename(example).split('.')[0] + '.input')
            convert_to_classic_input(config_path, input_path)
            parsed = parse_input(input_path)
            params = parsed.get('optimize') or parsed
            value = (params.get('optimize_cell')
                     if isinstance(params, dict) else None)
            self.assertTrue(value,
                            msg='%s: written and read back, the switch is off'
                                % os.path.basename(example))

    def test_the_prefixed_spelling_is_not_what_serves_the_run(self):
        # The negative control that keeps the three assertions above from being
        # tautological: take the classic input that was just generated, change
        # only the spelling of the keyword, and show the run no longer relaxes
        # its box. If a reader accepted both spellings, the parity tests above
        # would prove nothing - and the invented one is exactly what the
        # generator used to write (dev/methods/OPTIMIZE_DEVELOPMENT.md
        # section 11).
        examples = _cell_examples()
        example = examples[0]
        config_path = os.path.join(self.root, 'control_case.json')
        yaml_to_config(example, config_path)
        input_path = os.path.join(self.root, 'control_case.input')
        convert_to_classic_input(config_path, input_path)
        with open(input_path) as handle:
            text = handle.read()
        self.assertIn('optimize_cell', text)

        # The reader takes coordinates from a same-basename .xyz beside the
        # input, so the variant goes in its own directory with that file copied
        # along: the only difference between the two parsed texts is the
        # spelling of one keyword.
        stem = os.path.splitext(os.path.basename(input_path))[0]
        sibling = os.path.splitext(input_path)[0] + '.xyz'
        variant_dir = os.path.join(self.root, 'invented')
        os.makedirs(variant_dir, exist_ok=True)
        if os.path.exists(sibling):
            shutil.copy2(sibling, os.path.join(variant_dir, stem + '.xyz'))
        renamed = os.path.join(variant_dir, stem + '.input')
        with open(renamed, 'w') as handle:
            handle.write(text.replace('optimize_cell True',
                                      'opt_optimize_cell True'))

        def cell_of(path):
            parsed = parse_input(path)
            params = parsed.get('optimize') or parsed
            return bool(isinstance(params, dict) and params.get('optimize_cell'))

        self.assertTrue(cell_of(input_path),
                        'the real file does not enable the cell either: the '
                        'assertions above would be reading a value that never '
                        'reaches the optimizer')
        self.assertFalse(cell_of(renamed),
                         'the invented opt_optimize_cell reached the optimizer; '
                         'then the unprefixed name is not the only spelling that '
                         'works and the parity tests prove nothing')

    def test_the_passthrough_set_is_the_single_source(self):
        # The set the runtime flattener uses is the one the generator consults:
        # two lists would drift, and the drift is the defect this file is about.
        self.assertIn('optimize_cell', OPTIMIZE_PASSTHROUGH_KEYS)


class TestTheOptimizerSelectsAndTheSystemCounts(unittest.TestCase):
    """How many states a run solves is not a key of the optimizer section.

    Round `tmp_20261015_211219` ran `examples/optimize_cell_cp2k.yaml` with
    `nstate: 3` written under `optimize:` and both of its legs answered at the
    engine-side guard: `TDDFPT profile 'tddft_gradient' requires at least one
    excited state (tddft_nstates=0)`. The optimizer section never reaches the
    skill: `Calculator.sync()` propagates the manifold from the system
    (`SYNC_SYSTEM_FIELDS`, workflow/calculator.py:129-135), and the cp2k schema
    derives `tddft_nstates` from what that hands it. The division the ruling
    draws is therefore carried here - the system declares how many states exist,
    `optimize` names the one it drives.
    """

    @staticmethod
    def _chain_of(raw):
        """Walk the real load chain; return (system manifold, selected state).

        The selected state is read from the validated config, not from the
        example's text: a key no model owns (`current_state`, which an earlier
        draft of `optimize_cell_cp2k.yaml` wrote) is dropped in silence, so
        reading the file would certify a selection that never reaches the
        optimizer.
        """
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate
        config = validate(normalize(build_config(raw)))
        state = build_state(config, raw_config=raw)
        return state.pmd.system, config.optimize.opt_state_i

    def _cell_examples_with_a_selected_state(self):
        found = []
        for path in _cell_examples():
            with open(path) as handle:
                spec = yaml.safe_load(handle)
            _system, selected = self._chain_of(spec)
            if selected:
                found.append((path, spec, selected))
        return found

    def test_a_selected_state_has_a_manifold_to_be_selected_from(self):
        cases = self._cell_examples_with_a_selected_state()
        self.assertTrue(cases,
                        'no shipped cell example selects an excited state, so '
                        'this file would be checking an empty set')
        for path, spec, selected in cases:
            system = self._chain_of(spec)[0]
            self.assertGreaterEqual(
                system.nstate, selected + 1,
                msg='%s: the optimizer drives state %d of a system '
                    'that solves %d state(s)'
                    % (os.path.basename(path), selected, system.nstate))

    def test_the_count_the_skill_is_handed_is_the_one_it_derives_from(self):
        # The propagation the round was missing, read at the far end: the
        # system's count is what a sync would hand cp2k, and the skill's own
        # normalise turns that into the excited-state roots it asks TDDFPT for.
        from pyrai3md.skills.cp2k_skill.schema.adapter import build_config
        from pyrai3md.skills.cp2k_skill.schema.normalize import normalize
        cases = [(path, spec, selected) for path, spec, selected
                 in self._cell_examples_with_a_selected_state()
                 if 'cp2k' in str((spec.get('control') or {})
                                  .get('theory'))]
        self.assertTrue(cases,
                        'no shipped cell example runs on cp2k, so the derive '
                        'below would have nothing to read')
        path, spec, _selected = cases[0]
        system = self._chain_of(spec)[0]
        raw = {
            'atoms': spec['atoms'],
            'coordinates': spec['coordinates'],
            'charges': spec['molecule']['charges'],
            'multiplicity': spec['molecule']['multiplicity'],
            'nstate': system.nstate,
            'state_spin': list(system.state_spin),
            'pbc': spec['molecule']['pbc'],
            'cell': spec['molecule']['cell'],
            'method': spec['cp2k']['method'],
            'profile': spec['cp2k']['profile'],
            'functional': spec['cp2k']['functional'],
            'cutoff': spec['cp2k']['cutoff'],
            'project': 'cell_state_parity',
        }
        work = normalize(build_config(raw))
        self.assertGreater(
            work.tddft_nstates, 0,
            msg='the sync hands the skill nstate=%d but its own derive still '
                'asks for %d excited roots' % (system.nstate,
                                               work.tddft_nstates))

    def test_the_optimizer_section_alone_is_refused_at_load(self):
        # The negative control, in the round's own spelling: move the manifold
        # out of the system section and leave it under `optimize:`, exactly as
        # shipped when tmp_20261015_211219 failed. Nothing then refused the
        # input - the framework read a one-state system, and the skill's
        # derive landed on zero excited roots, which is the message the engine
        # host returned at the first evaluation rather than at load time.
        # Since 2026-10-20 the load chain refuses the same spelling by name
        # (`schema/validate.py::validate_centers` judges the selected state
        # against the system manifold), so this control now pins the earlier
        # refusal; the skill-side half below still shows why the framework has
        # to be the one to give it.
        from pyrai3md.skills.cp2k_skill.schema.adapter import build_config
        from pyrai3md.skills.cp2k_skill.schema.normalize import normalize
        cases = [(path, spec, selected) for path, spec, selected
                 in self._cell_examples_with_a_selected_state()
                 if 'cp2k' in str((spec.get('control') or {})
                                  .get('theory'))]
        self.assertTrue(cases,
                        'no shipped cell example runs on cp2k and selects a '
                        'state, so this control has nothing to spell wrong')
        path, spec, selected = cases[0]
        moved = yaml.safe_load(open(path))
        moved['molecule'].pop('ci', None)
        moved['molecule'].pop('spin', None)
        moved['optimize']['nstate'] = selected + 1
        moved['optimize']['state_spin'] = [0] * (selected + 1)
        with self.assertRaises(ValueError) as raised:
            self._chain_of(moved)
        message = str(raised.exception)
        self.assertIn('optimize.state_i=%d' % selected, message,
                      'the broken spelling was refused for another reason: %s'
                      % message[:160])
        self.assertIn('nstate=1', message,
                      'the refusal does not name the manifold it is judged '
                      'against: %s' % message[:160])

        # The reason the load-time refusal is the right place: the skill handed
        # that same one-state system still derives zero excited roots.
        raw = dict(nstate=1, state_spin=[0],
                   method=spec['cp2k']['method'],
                   profile=spec['cp2k']['profile'])
        self.assertEqual(normalize(build_config(raw)).tddft_nstates, 0,
                         'a one-state system no longer derives the zero that '
                         'the engine-side guard refuses, so the guard is not '
                         'what the round hit')





class TestTheCellThresholdsReachEveryConsumer(unittest.TestCase):
    """The four cell thresholds are read where the atomic four are read.

    A keyword added to one model and not the next is dropped on the way to the
    runtime: `optimize_params` says so in its own docstring (the key set there is
    a whitelist), and the flattener's `opt_` prefixing is a second place a name
    can be lost. These criteria walk one declared set of thresholds through the
    same three carriers the switch itself is walked through, and then compare the
    defaults the four holders carry, so a drift between two of them is a failure
    rather than a run judged by a number no input named.
    """

    VALUES = {'opt_csrms': 25.0, 'opt_csmax': 30.0,
              'opt_cerms': 1.0e-3, 'opt_cemax': 2.0e-3}

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='cell_threshold_')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _spec_file(self):
        path = os.path.join(self.root, 'cell_thresholds.yaml')
        with open(path, 'w') as handle:
            handle.write('control:\n  title: cell_thresholds\n'
                         '  jobtype: optimize\n'
                         '  theory: [pyscf]\n'
                         'atoms: [H, H]\n'
                         'coordinates:\n  - [0.0, 0.0, 0.0]\n'
                         '  - [0.0, 0.0, 0.74]\n'
                         'molecule:\n  charges: 0\n  multiplicity: 1\n'
                         'optimize:\n  optimize_cell: true\n')
            # Written bare, the way the section's own keywords are spelled: the
            # flattener is what turns `csrms` into `opt_csrms`.
            for name, value in self.VALUES.items():
                handle.write('  %s: %r\n' % (name[4:], value))
            handle.write('  maxit: 3\nsystem:\n  atoms: [H]\n'
                         '  coordinates: [[0.0, 0.0, 0.0]]\n')
        return path

    def test_the_json_carrier_names_the_prefixed_keys(self):
        import json
        config_path = os.path.join(self.root, 'cell_thresholds.json')
        yaml_to_config(self._spec_file(), config_path)
        with open(config_path) as handle:
            config = json.load(handle)
        for name, value in self.VALUES.items():
            self.assertAlmostEqual(float(config[name]), value,
                                   msg='%s did not reach the agent config' % name)

    def test_the_classic_carrier_returns_the_same_numbers(self):
        config_path = os.path.join(self.root, 'cell_thresholds.json')
        yaml_to_config(self._spec_file(), config_path)
        input_path = os.path.join(self.root, 'cell_thresholds.input')
        convert_to_classic_input(config_path, input_path)
        parsed = parse_input(input_path)
        section = (parsed.get('optimize')
                   if isinstance(parsed, dict) and parsed.get('optimize')
                   else parsed)
        for name, value in self.VALUES.items():
            got = section.get(name)
            if got is None:
                got = section.get(name[4:])
            # A keyword the writer left out is the schema default applied, which
            # is the same run: the test values above are chosen away from the
            # defaults so a lost line cannot hide behind this branch.
            if got is None:
                from pyrai3md.schema.config import DEFAULT
                got = DEFAULT[name]
            self.assertAlmostEqual(float(got), value,
                                   msg='%s did not survive the text carrier'
                                       % name)

    OPT_KEYS = ('opt_energy', 'opt_grms', 'opt_gmax', 'opt_drms', 'opt_dmax',
                'opt_csrms', 'opt_csmax', 'opt_cerms', 'opt_cemax',
                'optimize_cell')

    def _skill_configs(self, raw):
        """The per-binding skill configs one input builds, optimizer keys only."""
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate
        state = build_state(validate(normalize(build_config(raw))),
                            raw_config=raw)
        out = {}
        for task in state.theory + state.reference:
            skill = (task.get('config') or {}).get('skill_config') or {}
            out[task['skill_binding_id']] = {
                key: skill.get(key) for key in self.OPT_KEYS}
        return out

    def test_a_sectioned_carrier_hands_the_skill_the_optimizer_namespace(self):
        # A multi-skill classic input renders each engine from its OWN block,
        # and the yaml carrier delivers more than that block: the framework
        # merge hands every skill the `Optimize` submodel's keys too. Without
        # that fill the same case runs on two different skill configs -
        # measured on round `tmp_20261019_163711`'s products before the fix:
        # every binding of the six composite cell legs read the thresholds as
        # None from `.input` and as the declared number from yaml.
        # The thresholds here are deliberately away from the schema defaults,
        # so a lost line cannot hide behind "the default is the same number".
        path = os.path.join(self.root, 'sectioned_thresholds.yaml')
        with open(path, 'w') as handle:
            # Two sections, the outer one periodic-capable (a molecular
            # backend may only be a centre), and a centre region - the shape
            # every composite cell leg has, at the smallest size that builds.
            handle.write('control:\n  title: sectioned_thresholds\n'
                         '  jobtype: optimize\n  theory: [pyscf, xtb]\n'
                         '  driver: main\n'
                         'atoms: [O, H, H, H]\n'
                         'coordinates:\n  - [0.0, 0.0, 0.0]\n'
                         '  - [0.7, 0.0, 0.0]\n'
                         '  - [-0.4, 0.6, 0.0]\n'
                         '  - [0.0, 0.0, 8.0]\n'
                         'molecule:\n  charges: 0\n  multiplicity: 1\n'
                         '  highlevel: [[1, 2, 3]]\n'
                         '  pbc: [1, 1, 1]\n'
                         '  cell: [9.0, 9.0, 9.0, 90.0, 90.0, 90.0]\n'
                         'optimize:\n  method: minima\n  maxit: 3\n'
                         '  optimize_cell: true\n')
            for name, value in self.VALUES.items():
                handle.write('  %s: %r\n' % (name[4:], value))
            handle.write('pyscf:\n  method: hf\n  basis: sto3g\n'
                         '  ci: \"3\"\n  spin: \"0\"\n'
                         'xtb:\n  method: gfn_ff\n'
                         '  profile: gfn_ff_gradient\n')
        from pyrai3md.interface.parser import parse_input
        yaml_built = self._skill_configs(parse_input(path))
        config_path = os.path.join(self.root, 'sectioned_thresholds.json')
        input_path = os.path.join(self.root, 'sectioned_thresholds.input')
        yaml_to_config(path, config_path)
        convert_to_classic_input(config_path, input_path)
        previous = os.getcwd()
        os.chdir(self.root)
        try:
            classic_built = self._skill_configs(parse_input(os.path.basename(
                input_path)))
        finally:
            os.chdir(previous)
        self.assertEqual(sorted(yaml_built), sorted(classic_built),
                         'the two carriers bound different tasks')
        for binding in yaml_built:
            for key in self.OPT_KEYS:
                self.assertEqual(
                    classic_built[binding].get(key), yaml_built[binding].get(key),
                    '%s: %s reads %r from the classic carrier and %r from the '
                    'yaml one, so one input states the run twice'
                    % (binding, key, classic_built[binding].get(key),
                       yaml_built[binding].get(key)))

    def test_no_shipped_cell_leg_states_a_different_number_in_the_two_carriers(self):
        # The shipped set is the weaker claim, and the one that stays true as
        # cases are added: a key the writer omits because its value IS the
        # schema default arrives as that default, which is the same run - so
        # the classic side may hold None or the default, but never a third
        # number.
        import glob as _glob
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.schema.config import DEFAULT
        examples = os.path.abspath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'examples'))
        checked = 0
        for spec in sorted(_glob.glob(os.path.join(examples,
                                                   'optimize_cell_*.yaml'))):
            name = os.path.basename(spec)[:-len('.yaml')]
            workdir = os.path.join(self.root, name)
            os.makedirs(workdir, exist_ok=True)
            config_path = os.path.join(workdir, name + '.json')
            input_path = os.path.join(workdir, name + '.input')
            yaml_to_config(spec, config_path)
            convert_to_classic_input(config_path, input_path)
            for carrier in os.listdir(examples):
                if carrier.startswith(name) and not carrier.endswith('.yaml'):
                    import shutil
                    shutil.copy(os.path.join(examples, carrier), workdir)
            yaml_built = self._skill_configs(parse_input(spec))
            previous = os.getcwd()
            os.chdir(workdir)
            try:
                classic_built = self._skill_configs(parse_input(
                    os.path.basename(input_path)))
            finally:
                os.chdir(previous)
            for binding, values in yaml_built.items():
                for key, declared in values.items():
                    if declared is None or key not in DEFAULT:
                        continue
                    got = classic_built.get(binding, {}).get(key)
                    if got is None:
                        got = DEFAULT[key]
                    self.assertAlmostEqual(
                        float(got), float(declared),
                        msg='%s/%s: %s arrives as %r in the classic carrier '
                            'while the yaml carrier runs it as %r'
                            % (name, binding, key, got, declared))
            checked += 1
        # Measured 2026-10-20: seventeen shipped `optimize_cell_*` legs, the
        # same number the registration chain pins. A new leg that states the
        # run twice trips this count.
        self.assertEqual(checked, 17, checked)

    def test_the_four_holders_agree_on_the_defaults(self):
        from pyrai3md.data.state import PyRAIState
        from pyrai3md.schema import adapter
        from pyrai3md.schema.config import Calculation, DEFAULT, Optimize
        from types import SimpleNamespace
        for name, value in self.VALUES.items():
            self.assertIn(name, DEFAULT,
                          msg='%s was never registered as a default' % name)
            self.assertIn(name, Optimize.model_fields,
                          msg='%s is not a field of the Optimize model' % name)
            self.assertAlmostEqual(
                float(Optimize.model_fields[name].default),
                float(DEFAULT[name]))
            self.assertIn(name, Calculation.model_fields,
                          msg='%s is not a field of the Calculation model'
                              % name)
            self.assertAlmostEqual(
                float(Calculation.model_fields[name].default),
                float(DEFAULT[name]))
            self.assertEqual(adapter.OPTIMIZE_TYPES.get(name), 'f',
                             msg='%s has no type for the reader' % name)
        # The whitelist in `optimize_params`: a key missing there is read as the
        # default no matter what the input said.
        # `optimize_params` is a property, so it is read through its
        # getter on an object that has no runtime: that is the branch which
        # answers with the whitelist of fallbacks.
        params = PyRAIState.optimize_params.fget(SimpleNamespace(pmd=None))
        for name, value in self.VALUES.items():
            self.assertIn(name, params,
                          msg='%s is not in the whitelist the optimizer reads'
                              % name)
            self.assertAlmostEqual(float(params[name]), float(DEFAULT[name]),
                                   msg='%s drifts between DEFAULT and the '
                                       'fallback the runtime carries' % name)

    def test_the_optimizer_construction_keeps_the_numbers(self):
        from pyrai3md.workflow.optimizer import get_optimizer
        search = get_optimizer('minima',
                               cell_stress_rms_threshold=self.VALUES['opt_csrms'],
                               cell_stress_max_threshold=self.VALUES['opt_csmax'],
                               cell_strain_rms_threshold=self.VALUES['opt_cerms'],
                               cell_strain_max_threshold=self.VALUES['opt_cemax'],
                               cell_variables=True)
        self.assertAlmostEqual(search.cell_stress_rms_threshold,
                               self.VALUES['opt_csrms'])
        self.assertAlmostEqual(search.cell_strain_max_threshold,
                               self.VALUES['opt_cemax'])

class TestTheStepBoundReachesTheSearch(unittest.TestCase):
    """`trust_radius` is the one keyword that bounds the step, and it arrives.

    The radius is what clips a joint nuclear-plus-lattice step
    (`workflow/optimizer/base.py::_apply_trust_radius`), and the round readings
    that motivated naming it are in `examples/optimize_cell_pressure.yaml`: the
    pressurised box passed 181.5 bar of residual 0.49 A^3 short of its rest point
    and then travelled 3.95 A^3 on the next step. Unset must stay unset though:
    the searches do not agree on a starting radius on purpose - `minima` and
    `gdiis` take 0.3, the transition-state search 0.05 (`optimizer/ts.py:170`) -
    so a framework-wide default written into every run would move the TS legs
    eight-fold, and the measured probe on this host showed the resize reaching
    its ceiling (0.5) within four steps regardless of the start.

    The key set in `data/state.py::optimize_params` is a whitelist that a new
    field is lost by, so this class walks the number through every holder and
    both carriers, and checks the unset case keeps each search's own value.
    """

    RADIUS = 0.125

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='step_bound_')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _spec(self, declared=True):
        path = os.path.join(self.root, 'step_bound_%s.yaml'
                            % ('on' if declared else 'off'))
        with open(path, 'w') as handle:
            handle.write('control:\n  title: step_bound\n'
                         '  jobtype: optimize\n  theory: [xtb]\n'
                         'atoms: [O, H, H]\n'
                         'coordinates:\n  - [0.0, 0.0, 0.0]\n'
                         '  - [0.7, 0.0, 0.0]\n'
                         '  - [-0.4, 0.6, 0.0]\n'
                         'molecule:\n  charges: 0\n  multiplicity: 1\n'
                         '  pbc: [1, 1, 1]\n'
                         '  cell: [9.0, 9.0, 9.0, 90.0, 90.0, 90.0]\n'
                         'optimize:\n  method: minima\n  maxit: 3\n'
                         '  optimize_cell: true\n')
            if declared:
                handle.write('  trust_radius: %r\n' % self.RADIUS)
            handle.write('xtb:\n  method: gfn_ff\n'
                         '  profile: gfn_ff_gradient\n')
        return path

    def _state(self, raw):
        """The state one carrier builds, through the same chain a run uses."""
        from pyrai3md.data.build import build_state
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate
        if isinstance(raw, str):
            raw = parse_input(raw)
        return build_state(validate(normalize(build_config(raw))),
                           raw_config=raw)

    def _built_radius(self, state, method='minima'):
        """The radius the REAL construction site hands the search.

        Reached through `workflow/optimize.py` rather than reimplemented here:
        a criterion that copies that call's conditional cannot fail when the
        conditional is removed, which is the whole point of the check.
        """
        from pyrai3md.workflow.optimize import Optimize

        class Calculator:
            def __init__(self, state):
                self.state = state

            def get_state(self):
                return self.state

            def sync(self):
                pass

        state.pmd.optimize.opt_method = method
        return Optimize(calculator=Calculator(state), maxiter=3).optimizer.trust_radius

    def test_the_unprefixed_spelling_reaches_the_whitelisted_key(self):
        from pyrai3md.data.state import PyRAIState
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate
        from pyrai3md.data.build import build_state
        path = self._spec()
        raw = parse_input(path)
        state = build_state(validate(normalize(build_config(raw))),
                            raw_config=raw)
        # `optimize: {trust_radius: ...}` arrives prefixed, the same way `csrms`
        # becomes `opt_csrms`; an input spelling the prefixed name also works.
        self.assertAlmostEqual(float(state.optimize_params['opt_trust_radius']),
                               self.RADIUS)
        self.assertIn('opt_trust_radius', PyRAIState.optimize_params.fget(
            SimpleNamespace(pmd=None)),
            'the key is not in the whitelist the runtime reads, so a declared '
            'radius would be discarded no matter what the input said')

    def test_both_carriers_build_the_same_radius(self):
        from pyrai3md.interface.parser import parse_input
        path = self._spec()
        config_path = os.path.join(self.root, 'step_bound.json')
        input_path = os.path.join(self.root, 'step_bound.input')
        yaml_to_config(path, config_path)
        convert_to_classic_input(config_path, input_path)
        text = open(input_path).read()
        self.assertIn('opt_trust_radius', text,
                      'the classic carrier never states the step bound, which is '
                      'the omission this keyword exists to close')
        yaml_state = self._state(path)
        previous = os.getcwd()
        os.chdir(self.root)
        try:
            classic_state = self._state(os.path.basename(input_path))
        finally:
            os.chdir(previous)
        self.assertAlmostEqual(
            self._built_radius(classic_state),
            self._built_radius(yaml_state),
            msg='the two carriers start the search at different step bounds')
        self.assertAlmostEqual(self._built_radius(classic_state), self.RADIUS)

    def test_an_undeclared_run_keeps_each_searchs_own_radius(self):
        # The witness that `None` is the right default rather than a number:
        # minima and the TS search start differently, and both must survive an
        # input that says nothing.
        state = self._state(self._spec(declared=False))
        self.assertIsNone(state.optimize_params['opt_trust_radius'])
        self.assertAlmostEqual(self._built_radius(state, 'minima'), 0.3)
        self.assertAlmostEqual(self._built_radius(state, 'gdiis'), 0.3)
        self.assertAlmostEqual(self._built_radius(state, 'ts'), 0.05)
        config_path = os.path.join(self.root, 'quiet.json')
        input_path = os.path.join(self.root, 'quiet.input')
        yaml_to_config(self._spec(declared=False), config_path)
        convert_to_classic_input(config_path, input_path)
        self.assertNotIn('trust_radius', open(input_path).read(),
                         'an undeclared run gained a step-bound line, so every '
                         'shipped carrier would change')


if __name__ == '__main__':
    unittest.main()
