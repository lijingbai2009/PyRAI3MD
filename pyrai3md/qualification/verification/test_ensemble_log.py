"""
======================================================================
Test suite for the run-log ensemble section
(pyrai3md/utilities/print_results.py: _format_ensemble).

Tests cover:
- Non-ensemble runs (missing dynamics / ensemble nve): 'Not computed'
  marker at verbose >= 3 and an empty string at lower verbosity
- NVT Nose-Hoover chain: header lines (ensemble/thermostat/targets),
  N_df / Ekin / T_inst, the pre/post velocity scale line and the
  chain table (Q / xi / p_eta) read from the persisted system DOF
- NVT CSVR: alpha / ratio / c lines from the runtime diagnostics
- NPT Berendsen: kinetic / virial / internal pressures in bar and the
  mu matrix rows
- NPT MTTK: W, G_eps rows, persisted strain / momentum / velocity
  rows, the cell-bath lines and the cell scale line
- Skipped-step notes (thermostat / barostat guard branches)
- format_result_block wiring: the section renders after the stress
  section only when the dynamics section is passed

Sep 25 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md.utilities.print_results import (_format_ensemble,
                                              format_result_block)
from pyrai3md.utilities.units import (au_pressure_to_bar,
                                      KELVIN_TO_HARTREE)

TITLE = '&ensemble'

NHC_DYNAMICS = {
    'ensemble': 'nvt', 'thermostat': 'nose-hoover-chain',
    'thermostat_chain': 2, 'thermostat_tau': 100.0, 'temperature': 300.0,
}
NHC_DIAG = {
    'ndof': 57, 'ke': 0.09, 'q': [11.0, 12.0],
    'scale_pre': 0.9, 'scale_post': 1.1, 'thermostat_skipped': None,
}
NHC_SYSTEM = {
    'thermostat_dof': [0.1, 0.2, 0.3, 0.4],
    'volume': 1000.0,
    'cell': [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
}

EYE = [[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]]
NEG_EYE = [[-1e-4, 0.0, 0.0], [0.0, -1e-4, 0.0], [0.0, 0.0, -1e-4]]


class TestEnsembleMissing(unittest.TestCase):
    """A non-ensemble run renders nothing (or the verbose marker)."""

    def test_no_dynamics_low_verbose(self):
        self.assertEqual(_format_ensemble(None, {}, {}, False), '')

    def test_no_dynamics_show_missing(self):
        text = _format_ensemble(None, {}, {}, True)
        self.assertIn(TITLE, text)
        self.assertIn('Not computed', text)

    def test_nve_show_missing(self):
        text = _format_ensemble({'ensemble': 'nve'}, {}, {}, True)
        self.assertIn('Not computed', text)


class TestEnsembleNhc(unittest.TestCase):
    """NVT Nose-Hoover chain section content."""

    def _text(self):
        return _format_ensemble(
            NHC_DYNAMICS, NHC_SYSTEM, {'ensemble': dict(NHC_DIAG)}, False)

    def test_header_lines(self):
        text = self._text()
        self.assertIn(TITLE, text)
        self.assertIn('nose-hoover-chain (M=2, tau 100.0 fs)', text)
        self.assertIn('%20.4f K' % 300.0, text)
        self.assertIn('57', text)

    def test_temperature_and_scale(self):
        text = self._text()
        t_inst = 2.0 * 0.09 / (57 * KELVIN_TO_HARTREE)
        self.assertIn('%20.4f K' % t_inst, text)
        self.assertIn('%20.12f' % 0.9, text)
        self.assertIn('%14.12f' % 1.1, text)

    def test_chain_table(self):
        text = self._text()
        self.assertIn('Nose-Hoover chain (atomic units):', text)
        row = '  %-4d%24.12f%24.12f%24.12f\n' % (2, 12.0, 0.2, 0.4)
        self.assertIn(row, text)

    def test_volume_and_cell(self):
        text = self._text()
        self.assertIn('%20.4f A^3' % 1000.0, text)
        self.assertIn('Cell (A, deg):', text)


class TestEnsembleCsvr(unittest.TestCase):
    """NVT CSVR section content."""

    def test_alpha_ratio_c(self):
        dynamics = {'ensemble': 'nvt', 'thermostat': 'csvr',
                    'temperature': 300.0, 'thermostat_tau': 100.0}
        diag = {'ndof': 3, 'ke': 0.001, 'csvr_alpha': 1.25,
                'csvr_ratio': 0.9, 'csvr_c': 0.5}
        text = _format_ensemble(dynamics, {}, {'ensemble': diag}, False)
        self.assertIn("CSVR alpha (K'/K):", text)
        self.assertIn('%20.12f' % 1.25, text)
        self.assertIn('%20.12f' % 0.9, text)
        self.assertNotIn('Nose-Hoover chain', text)


class TestEnsembleBerendsen(unittest.TestCase):
    """NPT Berendsen section content."""

    def test_pressures_and_mu(self):
        dynamics = {'ensemble': 'npt', 'thermostat': 'off',
                    'barostat': 'berendsen', 'temperature': 300.0,
                    'pressure': 1.0, 'barostat_tau': 1000.0,
                    'barostat_mode': 'isotropic'}
        mu = np.eye(3) * 0.999
        diag = {'p_kin': EYE, 'p_vir': NEG_EYE,
                'p_int': (np.eye(3) * 2e-4).tolist(), 'mu': mu.tolist()}
        text = _format_ensemble(dynamics, {}, {'ensemble': diag}, False)
        self.assertIn('berendsen (tau 1000.0 fs, isotropic)', text)
        expected_bar = '%20.4f bar' % au_pressure_to_bar(1e-4)
        self.assertIn(expected_bar, text)
        self.assertIn('Berendsen mu (atomic units):', text)
        self.assertIn(''.join('%20.12f' % v for v in mu[0]), text)


class TestEnsembleMttk(unittest.TestCase):
    """NPT MTTK section content."""

    def test_extended_cell_block(self):
        dynamics = {'ensemble': 'npt', 'thermostat': 'nose-hoover',
                    'thermostat_chain': 1, 'barostat': 'mttk',
                    'temperature': 300.0, 'pressure': 1.0,
                    'barostat_tau': 1000.0,
                    'barostat_mode': 'anisotropic'}
        diag = {'ndof': 57, 'w': 7.0, 'p_kin': EYE, 'p_vir': NEG_EYE,
                'g_eps': np.eye(3).tolist(), 'ke_bar': 0.25,
                'scale_bar': 0.98, 'q_bar': [9.0],
                'scale_particle_post': 1.0001}
        system = {'barostat_strain': (np.eye(3) * 0.01).tolist(),
                  'barostat_momentum': np.eye(3).tolist(),
                  'barostat_velocity': (np.eye(3) * 0.1).tolist(),
                  'barostat_thermostat_dof': [0.2, 0.3]}
        text = _format_ensemble(dynamics, system, {'ensemble': diag},
                                False)
        self.assertIn('mttk (tau 1000.0 fs, anisotropic)', text)
        self.assertIn('W (barostat mass):', text)
        self.assertIn('G_eps (atomic units):', text)
        self.assertIn('Strain eps:', text)
        self.assertIn('Cell-momentum NHC chain (atomic units):', text)
        self.assertIn('Cell scale (pre/post):', text)
        row = '  %-4d%24.12f%24.12f%24.12f\n' % (1, 9.0, 0.2, 0.3)
        self.assertIn(row, text)


class TestEnsembleSkipNotes(unittest.TestCase):
    """Guard-branch skips leave an explicit note."""

    def test_skip_note_rendered(self):
        diag = {'thermostat_skipped': 'Ekin <= 0 (cold start, no rescale)',
                'barostat_skipped_pre': 'no previous stress'}
        text = _format_ensemble(
            dict(NHC_DYNAMICS, ensemble='npt', barostat='mttk'),
            NHC_SYSTEM, {'ensemble': diag}, False)
        self.assertIn('thermostat step skipped', text)
        self.assertIn('cold start', text)
        self.assertIn('barostat pre kick skipped', text)
        self.assertIn('no previous stress', text)


class TestFormatResultBlockWiring(unittest.TestCase):
    """format_result_block renders &ensemble after the stress block."""

    def test_section_after_stress_when_dynamics_given(self):
        results = {'energy': [0.0], 'stress_tensor': [EYE]}
        block = format_result_block(
            dict(NHC_SYSTEM, atoms=['C'], coordinates=[[0.0, 0.0, 0.0]]),
            results, step=1, runtime={'ensemble': dict(NHC_DIAG)},
            verbose=2, dynamics=NHC_DYNAMICS)
        self.assertIn(TITLE, block)
        self.assertLess(block.index('&stress tensor'), block.index(TITLE))

    def test_no_section_without_dynamics(self):
        block = format_result_block({'atoms': []}, {}, step=1, verbose=2)
        self.assertNotIn(TITLE, block)


if __name__ == '__main__':
    unittest.main()
