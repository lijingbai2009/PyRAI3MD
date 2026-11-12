"""
======================================================================
Unit tests for the OpenQP skill NAC-type routing (nac_type/nactype).

Covers the framework-driven dynamics contract:
- nac_type 'ktdc' (and the framework spelling 'nactype', and a
  missing value) routes *_nonadiabatic_dynamics to runtype=data
  (per-step energies + gradients, no [nac] section, no file2);
- nac_type 'dcm' routes to runtype=prop with [guess]file2 and
  [properties]nac=nacme (MO overlap);
- nac_type 'nac' inside a dynamics profile stays on runtype=data and
  adds the numerical NAC vectors that path answers (the mrsf_nac
  single-point profile keeps runtype=nac);
- the parser refuses to return silent all-zero results when the
  OpenQP 'energies' export is missing (an aborted run must surface).

Sep 26 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import tempfile
import unittest

from pyrai3md.skills.openqp_skill.schema.adapter import build_config
from pyrai3md.skills.openqp_skill.schema.normalize import (
    normalize, derive_nac_type)
from pyrai3md.skills.openqp_skill.schema.method import resolve_profile
from pyrai3md.skills.openqp_skill.service.workflow.compiler import (
    compile_workflow)
from pyrai3md.skills.openqp_skill.service.builder.input_builder import (
    build_input)
from pyrai3md.skills.openqp_skill.service.parser.output_parser import (
    parse_output)

ATOMS = ['C', 'C', 'H', 'H', 'H', 'H']
COORDS = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.33], [0.0, 0.92, -0.57],
          [0.0, -0.92, -0.57], [0.92, 0.0, 1.9], [-0.92, 0.0, 1.9]]


def _config(**extra):
    raw = {
        'atoms': list(ATOMS),
        'coordinates': [list(row) for row in COORDS],
        'charges': 0,
        'multiplicity': 1,
        'nstate': 3,
        'state_spin': [0, 0, 0],
        'target_states': [0, 1, 2],
        'profile': 'mrsf_nonadiabatic_dynamics',
        'method': 'mrsf',
        'basis': 'sto-3g',
        'functional': 'bhhlyp',
        'execution_mode': 'local',
    }
    raw.update(extra)
    return normalize(build_config(raw))


class TestCompileWorkflowRouting(unittest.TestCase):
    """runtype is decided by (profile, nac_type)."""

    def test_dynamics_ktdc_routes_to_data(self):
        wf = compile_workflow('mrsf_nonadiabatic_dynamics',
                              nac_type='ktdc')
        self.assertEqual(wf.runtype, 'data')
        self.assertNotIn('nac', wf.sections)

    def test_dynamics_dcm_routes_to_prop(self):
        wf = compile_workflow('mrsf_nonadiabatic_dynamics',
                              nac_type='dcm')
        self.assertEqual(wf.runtype, 'prop')

    def test_dynamics_nac_keeps_vectors_on_the_data_path(self):
        # OpenQP 1.3 answers numerical NAC vectors inside runtype=data
        # ([properties] nac=nac plus the [nac] displacement control), so a
        # dynamics profile asking for them gets them instead of losing the
        # coupling request to the kTDC path.
        self.assertEqual(derive_nac_type(
            {'profile': 'mrsf_nonadiabatic_dynamics',
             'nac_type': 'nac'})['nac_type'], 'nac')
        wf = compile_workflow('mrsf_nonadiabatic_dynamics',
                              nac_type='nac')
        self.assertEqual(wf.runtype, 'data')
        self.assertTrue(wf.nac_vectors)
        self.assertIn('nac', wf.sections)

    def test_dynamics_missing_nac_type_defaults_to_data(self):
        self.assertEqual(derive_nac_type(
            {'profile': 'mrsf_nonadiabatic_dynamics'})['nac_type'],
            'ktdc')

    def test_mrsf_nac_single_point_keeps_runtype_nac(self):
        wf = compile_workflow('mrsf_nac', nac_type='ktdc')
        self.assertEqual(wf.runtype, 'nac')
        self.assertIn('nac', wf.sections)

    def test_mrsf_nac_dcm_switches_to_prop(self):
        wf = compile_workflow('mrsf_nac', nac_type='dcm')
        self.assertEqual(wf.runtype, 'prop')

    def test_resolve_profile_md_maps_to_dynamics(self):
        self.assertEqual(resolve_profile('mrsf', 'md'),
                         'mrsf_nonadiabatic_dynamics')


class TestNactypeAlias(unittest.TestCase):
    """The framework spelling 'nactype' feeds the skill 'nac_type'."""

    def test_nactype_alias_accepted(self):
        config = _config(nactype='ktdc')
        self.assertEqual(config.calculation.nac_type, 'ktdc')

    def test_explicit_nac_type_wins_over_alias(self):
        config = _config(nactype='ktdc', nac_type='dcm')
        self.assertEqual(config.calculation.nac_type, 'dcm')

    def test_default_is_ktdc(self):
        config = _config()
        self.assertEqual(config.calculation.nac_type, 'ktdc')


class TestInputRouting(unittest.TestCase):
    """The generated inp carries the routed runtype and guess block."""

    def test_ktdc_input_is_data_without_file2(self):
        config = _config(nactype='ktdc')
        wf = compile_workflow(config.calculation.profile,
                              nac_type=config.calculation.nac_type)
        text = build_input(config, wf)
        self.assertIn('runtype=data', text)
        self.assertNotIn('file2=', text)
        self.assertNotIn('nac=nacme', text)

    def test_dcm_is_refused_before_the_input_is_built(self):
        # `dcm` continues from the previous step's orbital json
        # ([guess]file2, which the builder always emits for it), so a single
        # calculation has nothing to continue from and the engine aborts
        # (`guess.file2: Second restart file is set but does not exist` ->
        # `mol object OQP.json does not exist`, rc=1; reproduced in isolation
        # in the test environment for the retired `tddft_nonadiabatic_dynamics`
        # case, round tmp_20261011_121148, whose earlier green came from an
        # orbital json another case had left in the shared calcdir). Ruled
        # unsupported at validation until the two-step mechanism has its own
        # test -- the refusal names both the reason and the way out.
        from pyrai3md.skills.openqp_skill.schema.validate import validate

        with self.assertRaises(ValueError) as ctx:
            validate(_config(nac_type='dcm'))
        self.assertIn("nac_type 'dcm' is not supported", str(ctx.exception))


class TestParserStrictness(unittest.TestCase):
    """A missing energies export raises instead of returning zeros."""

    @staticmethod
    def _write_log(tmp):
        # parse_output validates the run log first: an existing log
        # without failure markers means 'completed', so the energies
        # export is the next mandatory artifact.
        with open(os.path.join(tmp, 'openqp_calc.log'), 'w') as handle:
            handle.write('OpenQP finished normally\n')

    def test_missing_energies_raises(self):
        config = _config(nactype='ktdc')
        wf = compile_workflow(config.calculation.profile,
                              nac_type=config.calculation.nac_type)
        with tempfile.TemporaryDirectory() as tmp:
            self._write_log(tmp)
            with self.assertRaises(RuntimeError) as ctx:
                parse_output(config, tmp, wf)
            self.assertIn('energies', str(ctx.exception))

    def test_present_energies_parsed(self):
        config = _config(nactype='ktdc')
        wf = compile_workflow(config.calculation.profile,
                              nac_type=config.calculation.nac_type)
        with tempfile.TemporaryDirectory() as tmp:
            self._write_log(tmp)
            with open(os.path.join(tmp, 'energies'), 'w') as handle:
                # mrsf: line 0 is the internal triplet reference,
                # the physical states follow.
                handle.write('-76.5\n-76.4\n-76.3\n-76.2\n')
            # The dynamics path (runtype=data) exports grad_<i> for each target
            # state beside the energies, so this fixture writes them too; the
            # parser now refuses to answer for a requested state whose file is
            # absent rather than leaving its row at zero.
            for state in (1, 2, 3):
                with open(os.path.join(tmp, 'grad_%d' % state), 'w') as handle:
                    for index in range(len(ATOMS)):
                        handle.write('    %.10f    %.10f    %.10f\n'
                                     % (0.1 * state, 0.0, 0.0))
            results = parse_output(config, tmp, wf)
        self.assertAlmostEqual(results['energy'][0], -76.4, places=9)
        self.assertAlmostEqual(results['energy'][2], -76.2, places=9)
        self.assertAlmostEqual(float(results['gradient'][2][0][0]), 0.3,
                               places=9)

    def test_a_short_energies_export_is_refused_not_clamped(self):
        # Two lines for three declared states: the reading that took
        # min(nstate-1, available) filled what it could and left the rest at
        # zero, which satisfies the shape and describes no surface.
        config = _config(nactype='ktdc')
        wf = compile_workflow(config.calculation.profile,
                              nac_type=config.calculation.nac_type)
        with tempfile.TemporaryDirectory() as tmp:
            self._write_log(tmp)
            with open(os.path.join(tmp, 'energies'), 'w') as handle:
                handle.write('-76.5\n-76.4\n')
            with self.assertRaises(RuntimeError) as ctx:
                parse_output(config, tmp, wf)
            self.assertIn('no energy to be read into', str(ctx.exception))

    def test_a_requested_gradient_without_its_file_is_refused(self):
        config = _config(nactype='ktdc')
        wf = compile_workflow(config.calculation.profile,
                              nac_type=config.calculation.nac_type)
        with tempfile.TemporaryDirectory() as tmp:
            self._write_log(tmp)
            with open(os.path.join(tmp, 'energies'), 'w') as handle:
                handle.write('-76.5\n-76.4\n-76.3\n-76.2\n')
            for state in (1, 2):
                with open(os.path.join(tmp, 'grad_%d' % state), 'w') as handle:
                    for index in range(len(ATOMS)):
                        handle.write('    0.1    0.0    0.0\n')
            # grad_3 is absent: state 2 is one of the pair this run follows.
            with self.assertRaises(RuntimeError) as ctx:
                parse_output(config, tmp, wf)
            self.assertIn('grad_3', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
