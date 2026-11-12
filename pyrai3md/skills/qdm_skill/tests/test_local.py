"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 25, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Unit tests for the local execution path: the single-point
profiles must persist their property arrays as plain text files
in the calculation directory so the skill output can be
cross-checked against the framework run log.

"""

import os
import tempfile
import unittest

import numpy as np

from pyrai3md.skills.qdm_skill.schema.config import SkillConfig
from pyrai3md.skills.qdm_skill.service.runner.local_runner import run_local

# Coordinates stored in Angstrom; 0.0 Angstrom = 0.0 Bohr (Tully1
# crossing geometry with the analytic NAC benchmark value
# d_01 * (E_0 - E_1) = +A*B = +0.016).
_COORDINATES = [[0.0, 0.0, 0.0]]


def _config(profile):
    """Build a minimal single-point SkillConfig for tully1."""
    return SkillConfig(
        system={
            'atoms': ['X'],
            'coordinates': _COORDINATES,
            'nstate': 2,
        },
        calculation={
            'method': 'tully1',
            'profile': profile,
        },
    )


class TestSinglePointFiles(unittest.TestCase):
    """Each single-point profile writes its data files to calcdir."""

    def _run(self, profile):
        config = _config(profile)
        with tempfile.TemporaryDirectory() as calcdir:
            results = run_local(config, calcdir)
            files = sorted(os.listdir(calcdir))
            contents = {
                name: np.atleast_2d(np.loadtxt(os.path.join(calcdir, name)))
                for name in files
            }
        return results, files, contents

    def test_energy_profile_writes_energy_only(self):
        results, files, contents = self._run('tully1_energy')
        self.assertEqual(files, ['QDM_energy.dat'])
        energy = np.asarray(results['energy'], dtype=float)
        np.testing.assert_allclose(
            contents['QDM_energy.dat'].reshape(-1),
            energy, atol=1e-12)

    def test_gradient_profile_writes_energy_and_gradient(self):
        results, files, contents = self._run('tully1_gradient')
        self.assertEqual(files, ['QDM_energy.dat', 'QDM_gradient.dat'])
        gradient = np.asarray(results['gradient'], dtype=float)
        saved = contents['QDM_gradient.dat']
        np.testing.assert_allclose(saved,
                                   gradient.reshape(gradient.shape[0], -1),
                                   atol=1e-12)

    def test_nac_profile_writes_all_three_files(self):
        results, files, contents = self._run('tully1_nac')
        self.assertEqual(files,
                         ['QDM_energy.dat', 'QDM_gradient.dat',
                          'QDM_nac.dat'])
        nac = np.asarray(results['nac'], dtype=float)
        saved = contents['QDM_nac.dat']
        np.testing.assert_allclose(saved,
                                   nac.reshape(nac.shape[0], -1),
                                   atol=1e-12)
        # Tully1 crossing benchmark: <0| dH/dx |1> = d_01 *
        # (E_1 - E_0) = -A*B (d_01 = -AB/(2C), gap = 2C)
        self.assertAlmostEqual(saved[0, 0], -0.016, places=8)

    def test_custom_project_prefix(self):
        config = _config('tully1_energy')
        config.calculation.project = 'MyRun'
        with tempfile.TemporaryDirectory() as calcdir:
            run_local(config, calcdir)
            self.assertIn('MyRun_energy.dat', os.listdir(calcdir))


if __name__ == '__main__':
    unittest.main()
