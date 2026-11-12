"""
======================================================================
Test suite for the `&md` keyword vocabulary: which spellings the block
answers to, which are refused by name, and what the framework does with
the ones that are alive (dev/methods/MD_DEVELOPMENT.md section 23).

The archive carried more `&md` keywords than this framework reads. Four
of them were taken over by PyRAIState itself: `restart` and `addstep`
belonged to a pickle-based continuation that `<title>.rt` plus the
absolute step counter replaces, and `record`/`record_step` belonged to a
snapshot ring buffer whose length `history` now owns. Two more describe
a decision the framework deliberately does not make: `thermo` was the
single-thermostat spelling that `ensemble` + `thermostat` + `barostat`
superseded, and `popreset` collapsed an out-of-range population to a
pure state, while this port reports the violation and propagates the
values it found (workflow/integrator/fssh.py::_diagnose_population). The
hybrid-NAMD references (`ref_energy`/`ref_grad`/`ref_nac`/`ref_soc`/
`datapath`) were the pre-adaptive way to name a reference calculation,
and `graddesc` was a zero-velocity integrator that is not provided.

Every one of them is refused at the input, naming what replaced it: a
silently dropped keyword is an input that believes something the run
never does. `Dynamics` stays lenient about unknown keys on purpose (it
absorbs the classic text, the normalized config and the agent's dotted
form), so the refusal is a before-validator on the model, which covers
all carriers at once.

Two keywords were dead and are now alive: `addsoc` biases the spin-orbit
couplings of one evaluation (applied in the Calculator's finalization, so
the biased matrix is what the step records), and the velocity-control
family is implemented in workflow/integrator/operations.py.

Oct 08 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import shutil
import tempfile
import unittest

from pyrai3md.data.build import build_state
from pyrai3md.interface.parser import _parse_text, parse_input
from pyrai3md.kernel.environment import Environment
from pyrai3md.schema import normalize, validate
from pyrai3md.schema.adapter import MD_TYPES, build_config
from pyrai3md.schema.config import (PyRAIConfig, RETIRE_DYNAMICS_KEYS,
                                    Dynamics)

_MD_INPUT = """control:
  title: t_md_keywords
  jobtype: md
  theory: [pyscf]
atoms: [H, H]
coordinates:
  - [0.0, 0.0, 0.0]
  - [0.0, 0.0, 0.74]
molecule:
  ci: [1]
  spin: [0]
md:
  step: 4
  size: 20.67
{extra}pyscf:
  method: hf
  basis: sto3g
"""

# The `&md` vocabulary the archive shipped (legacy Keywords/key_md.py).
# `gl_seed` is a control-section keyword and `temp` is the legacy
# spelling of `temperature`; both are checked where they live.
ARCHIVE_MD_KEYWORDS = (
    'initcond', 'excess', 'scale', 'target', 'graddesc', 'reset',
    'resetstep', 'ninitcond', 'method', 'format', 'randvelo', 'step',
    'size', 'root', 'activestate', 'sfhp', 'nactype', 'phasecheck',
    'popreset', 'gap', 'gapsoc', 'addsoc', 'substep', 'rk4', 'integrate',
    'deco', 'adjust', 'reflect', 'maxh', 'dosoc', 'thermo', 'thermodelay',
    'direct', 'buffer', 'record', 'record_step', 'checkpoint', 'restart',
    'addstep')


def _md_input(**keywords):
    """One &md block per case: the base input plus the keywords under test."""
    extra = ''.join('  %s: %r\n' % (key, value)
                    for key, value in keywords.items())
    return _MD_INPUT.format(extra=extra)


class _Loaded(unittest.TestCase):
    """Shared helper: load a case through the real pipeline."""

    @classmethod
    def setUpClass(cls):
        Environment().initialize()

    def _write(self, text):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        path = os.path.join(directory, 'case.yaml')
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(text)
        return path

    def _load(self, text):
        return validate(normalize(build_config(parse_input(self._write(text)))))


class TestRetiredDynamicsKeywords(_Loaded):
    """A retired `&md` spelling is an error, never a dropped field."""

    def test_every_retired_spelling_is_refused_with_its_replacement(self):
        for key in sorted(RETIRE_DYNAMICS_KEYS):
            with self.subTest(keyword=key):
                with self.assertRaises(ValueError) as caught:
                    self._load(_md_input(**{key: 1}))
                message = str(caught.exception)
                self.assertIn('`%s`' % key, message)
                self.assertIn('no longer an input keyword', message)
                self.assertIn(RETIRE_DYNAMICS_KEYS[key].split(':')[0][:20],
                              message,
                              'the refusal has to name the replacement')

    def test_the_classic_text_route_refuses_too(self):
        # The classic reader types only the keywords it still knows, so
        # an unlisted spelling reaches the model as a raw string; the
        # refusal belongs to the model, which is why one table covers
        # yaml, json, the dotted agent form and this route at once.
        text = ("&control\n jobtype md\n title t\n&end\n"
                "&system\n atoms 2\n h 0.0 0.0 0.0\n h 0.0 0.0 1.4\n&end\n"
                "&md\n step 4\n restart 1\n&end\n")
        raw = _parse_text([line + '\n' for line in text.splitlines()])
        raw['system'] = {'atoms': ['h', 'h'],
                         'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]}
        with self.assertRaises(ValueError) as caught:
            build_config(raw)
        self.assertIn('`restart`', str(caught.exception))
        self.assertIn('<title>.rt', str(caught.exception))

    def test_the_model_carries_no_retired_field(self):
        dynamics = PyRAIConfig().dynamics
        for key in RETIRE_DYNAMICS_KEYS:
            with self.subTest(keyword=key):
                self.assertFalse(hasattr(dynamics, key),
                                 'a field nobody writes is a second source '
                                 'of a decision the run already made')

    def test_the_classic_reader_types_no_retired_spelling(self):
        # MD_TYPES is the vocabulary the classic section is read with; a
        # retired name still listed there would be typed, echoed and
        # looked alive in every tool that walks the table.
        overlap = sorted(set(MD_TYPES) & set(RETIRE_DYNAMICS_KEYS))
        self.assertEqual([], overlap)

    def test_no_archive_keyword_is_silently_ignored(self):
        """Every spelling the archive carried is either read or refused."""
        live = set(Dynamics.model_fields)
        for key in ARCHIVE_MD_KEYWORDS:
            with self.subTest(keyword=key):
                self.assertTrue(key in live or key in RETIRE_DYNAMICS_KEYS,
                                '%s is accepted by the input and read by '
                                'nobody' % key)


class TestLiveDynamicsKeywords(_Loaded):
    """The keywords this round brought back to life still load."""

    def test_velocity_and_restart_family_loads(self):
        text = _md_input(randvelo=1, excess=0.001, scale=1.0, target=0.0,
                         reset=1, resetstep=2, thermodelay=3, addsoc=0.001,
                         direct=1, buffer=2, checkpoint=4, history=8)
        dynamics = self._load(text).dynamics
        self.assertEqual(1, dynamics.randvelo)
        self.assertAlmostEqual(0.001, dynamics.excess)
        self.assertEqual(1, dynamics.reset)
        self.assertEqual(2, dynamics.resetstep)
        self.assertEqual(3, dynamics.thermodelay)
        self.assertAlmostEqual(0.001, dynamics.addsoc)
        self.assertEqual(4, dynamics.checkpoint)
        self.assertEqual(8, dynamics.history)

    def test_temperature_is_still_reachable_under_its_legacy_spelling(self):
        # `temp` was the legacy keyword of the same setting and FIELD_MAP
        # routes it, so the archive input still loads and the refusal
        # table has nothing to say about it.
        dynamics = self._load(_md_input(temp=350.0)).dynamics
        self.assertAlmostEqual(350.0, dynamics.temperature)


class TestSocBias(unittest.TestCase):
    """`addsoc` shifts the couplings of one evaluation, where they are recorded."""

    class _Stub:
        """Two pair couplings to read the arithmetic off."""

        name = 'stub'

        def calculate(self, atoms, coordinates, target_states=None, cell=None):
            n = len(atoms)
            return {'energy': [0.0],
                    'gradient': [[[0.0] * 3] * n],
                    'soc': [10.0, 20.0]}

    def setUp(self):
        Environment().initialize()

    def _state(self, **md):
        config = {'control': {'title': 't', 'jobtype': 'md'},
                  'system': {'atoms': ['h', 'h'],
                             'coordinates': [[0.0, 0.0, 0.0],
                                             [0.0, 0.0, 1.4]]},
                  'molecule': {'ci': [1], 'spin': [0]},
                  'md': dict({'step': 1, 'size': 1.0}, **md)}
        return build_state(validate(normalize(build_config(config))))

    def _soc(self, **md):
        from pyrai3md.workflow.calculator import Calculator
        state = self._state(**md)
        Calculator(state=state,
                   dummy_backend=self._Stub())._run_single()
        return list(state.result['soc'])

    def test_bias_shifts_every_coupling_of_the_step(self):
        self.assertEqual([10.001, 20.001],
                         self._soc(dosoc=1, addsoc=0.001))

    def test_bias_does_not_reach_a_run_that_never_asked_for_soc(self):
        # A bias on couplings no one requested would invent a spin-orbit
        # channel the input does not declare.
        self.assertEqual([10.0, 20.0], self._soc(dosoc=0, addsoc=0.001))

    def test_no_bias_changes_nothing(self):
        self.assertEqual([10.0, 20.0], self._soc(dosoc=1, addsoc=0.0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
