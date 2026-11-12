"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 24, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Unit tests for the split-operator reference solver: norm
conservation (free and interacting propagation), population
normalization, dt-halving convergence, velocity -> k0 transfer,
the fixed default window, out-of-window rejection, the edge
boundary warning, the trajectory output files and the adiabatic
initial-state convention (current_state indexes the adiabatic
state, not the diabatic component).

"""

import os
import tempfile
import unittest

import numpy as np

from pyrai3md.skills.qdm_skill.service.models import get_model
from pyrai3md.skills.qdm_skill.service.reference import (
    REF_XMAX_DEFAULT, REF_XMIN_DEFAULT,
    build_grid, initial_wavepacket, run_reference)
from pyrai3md.skills.qdm_skill.service.utils import ANGSTROM_TO_BOHR

# Common compact settings keeping the tests fast (the grid keeps
# the Nyquist bound pi/dx ~ 14.6 above the packet momentum k0 = 10)
_BASE_SETTINGS = {
    'model_mass': 2000.0,
    'ref_ngrid': 512,
    'ref_xmin': None,
    'ref_xmax': None,
    'ref_dt': 10.0,
    'ref_nsteps': 40,
    'ref_save_every': 5,
    'wavepacket_sigma': 0.5,
    'wavepacket_k0': 10.0,
}


def _free_model():
    """Vanishing Hamiltonian: tully1 with A = C = 0."""
    return get_model('tully1', params={'tully_a': 0.0,
                                       'tully_c': 0.0})


class TestGridAndPacket(unittest.TestCase):

    def test_grid_spacing(self):
        grid = build_grid(8, -4.0, 4.0)
        self.assertEqual(grid.size, 8)
        self.assertAlmostEqual(grid[0], -4.0)
        self.assertAlmostEqual(grid[1] - grid[0], 1.0)

    def test_wavepacket_norm_and_state(self):
        grid = build_grid(256, -25.0, 25.0)
        dx = grid[1] - grid[0]
        psi = initial_wavepacket(grid, 0.0, 0.5, 10.0, [0.0, 1.0])
        norm = np.sum(np.abs(psi) ** 2) * dx
        self.assertAlmostEqual(norm, 1.0, places=12)
        self.assertEqual(np.sum(np.abs(psi[:, 0]) ** 2), 0.0)

    def test_wavepacket_component_weights(self):
        # A mixed weight vector distributes the envelope over both
        # diabatic components (the adiabatic-eigenvector path).
        grid = build_grid(256, -25.0, 25.0)
        dx = grid[1] - grid[0]
        w = np.array([0.6, 0.8])
        psi = initial_wavepacket(grid, 0.0, 0.5, 10.0, w)
        norm = np.sum(np.abs(psi) ** 2) * dx
        self.assertAlmostEqual(norm, 1.0, places=12)
        frac0 = np.sum(np.abs(psi[:, 0]) ** 2) * dx
        self.assertAlmostEqual(frac0, 0.36, places=10)


class TestRunReference(unittest.TestCase):

    def _run(self, model, calcdir, **overrides):
        velocities = overrides.pop('_velocities', None)
        current_state = overrides.pop('_current_state', 0)
        coordinates = overrides.pop(
            '_coordinates', [[-15.0 / ANGSTROM_TO_BOHR, 0.0, 0.0]])
        settings = dict(_BASE_SETTINGS)
        settings.update(overrides)
        return run_reference(
            model, coordinates,
            calcdir, 'T', velocities=velocities,
            current_state=current_state, **settings)

    def test_free_propagation_norm_conservation(self):
        with tempfile.TemporaryDirectory() as calcdir:
            results = self._run(_free_model(), calcdir)
        self.assertLess(abs(results['final_norm'] - 1.0), 1e-6)
        self.assertAlmostEqual(sum(results['final_population']),
                               1.0, places=6)

    def test_tully2_norm_and_population(self):
        with tempfile.TemporaryDirectory() as calcdir:
            results = self._run(get_model('tully2'), calcdir)
        self.assertLess(abs(results['final_norm'] - 1.0), 1e-6)
        self.assertAlmostEqual(sum(results['final_population']),
                               1.0, places=6)
        # All populations non-negative
        self.assertTrue(all(p >= 0.0
                            for p in results['final_population']))

    def test_output_files_written(self):
        with tempfile.TemporaryDirectory() as calcdir:
            results = self._run(_free_model(), calcdir)
            for key in ['population_file', 'norm_file',
                        'grid_file', 'density_file']:
                self.assertTrue(os.path.isfile(results[key]), key)
                self.assertTrue(results[key].startswith(calcdir))
            population = np.loadtxt(results['population_file'])
            density = np.load(results['density_file'])
        # header row + one row per snapshot (step 0 + every 5)
        self.assertEqual(population.shape, (9, 3))
        self.assertEqual(density.shape, (9, 512, 2))

    def test_dt_halving_convergence(self):
        # The final population change must shrink as dt halves
        model = get_model('tully2')
        with tempfile.TemporaryDirectory() as calcdir:
            p_full = np.array(
                self._run(model, calcdir)['final_population'])
            p_half = np.array(
                self._run(model, calcdir, ref_dt=5.0,
                          ref_nsteps=80)['final_population'])
            p_quarter = np.array(
                self._run(model, calcdir, ref_dt=2.5,
                          ref_nsteps=160)['final_population'])
        err_half = np.abs(p_full - p_half).max()
        err_quarter = np.abs(p_half - p_quarter).max()
        self.assertLess(err_quarter, err_half)

    def test_velocity_sets_k0(self):
        # k0 = model_mass * velocities[0][0] overrides wavepacket_k0
        with tempfile.TemporaryDirectory() as calcdir:
            results = self._run(
                _free_model(), calcdir, _velocities=[[0.005, 0, 0]],
                wavepacket_k0=99.0)
        self.assertAlmostEqual(results['k0'], 2000.0 * 0.005)

    def test_k0_fallback(self):
        with tempfile.TemporaryDirectory() as calcdir:
            results = self._run(_free_model(), calcdir,
                                wavepacket_k0=3.0)
        self.assertAlmostEqual(results['k0'], 3.0)

    def test_default_window_is_fixed(self):
        with tempfile.TemporaryDirectory() as calcdir:
            results = self._run(_free_model(), calcdir)
        # None boundaries fall back on the fixed default window,
        # independent of x0.
        self.assertAlmostEqual(results['x0'], -15.0, places=5)
        self.assertAlmostEqual(results['xmin'], REF_XMIN_DEFAULT,
                               places=5)
        dx = (REF_XMAX_DEFAULT - REF_XMIN_DEFAULT) / 512
        self.assertAlmostEqual(results['xmax'],
                               REF_XMAX_DEFAULT - dx, places=5)

    def test_x0_outside_window_raises(self):
        with tempfile.TemporaryDirectory() as calcdir:
            with self.assertRaises(ValueError):
                # x0 = -15 Bohr lies outside the explicit window.
                self._run(_free_model(), calcdir,
                          ref_xmin=-40.0, ref_xmax=-30.0)

    def test_edge_boundary_warning(self):
        # A packet driven into a narrow explicit window must trip
        # the periodic-boundary warning instead of silently
        # wrapping around.
        with tempfile.TemporaryDirectory() as calcdir:
            with self.assertWarns(UserWarning):
                self._run(_free_model(), calcdir,
                          ref_xmin=-20.0, ref_xmax=-12.0,
                          ref_nsteps=100)

    def test_current_state_selection(self):
        with tempfile.TemporaryDirectory() as calcdir:
            results = self._run(_free_model(), calcdir,
                                _current_state=1, ref_nsteps=1)
        self.assertEqual(results['initial_state'], 1)
        population = results['final_population']
        self.assertGreater(population[1], 0.99)

    def test_tully1_scattering_population_anchor(self):
        # Benchmark calibration anchor: tully1 from x0 = -20 Bohr
        # with momentum k0 = 10. The packet scatters at the
        # avoided crossing and the asymptotic upper-state
        # population agrees with the literature ~15% benchmark
        # (Subotnik 2010; Tully 1990 model I).
        settings = {
            'ref_ngrid': 2048,
            'ref_dt': 2.0,
            'ref_nsteps': 2500,
            'ref_save_every': 250,
        }
        with tempfile.TemporaryDirectory() as calcdir:
            results = self._run(
                get_model('tully1'), calcdir,
                _coordinates=[[-20.0 / ANGSTROM_TO_BOHR, 0.0, 0.0]],
                **settings)
        self.assertAlmostEqual(results['final_population'][1],
                               0.155, places=2)


if __name__ == '__main__':
    unittest.main()
