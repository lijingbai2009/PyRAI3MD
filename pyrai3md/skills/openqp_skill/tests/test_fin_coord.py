"""
======================================================================
The OpenQP skill's fin_coord channel after an internal optimization.

`control.driver != 'main'` keeps the geometry optimization inside the
skill (SingleShot). The framework learns the optimized structure only
through the `fin_coord` channel; a parser that publishes it under
`coordinates` instead leaves the run's own `opt.xyz`, `opt_geom.xyz`
and final `&coordinates` block reporting the input structure while the
skill's calc directory holds the optimized one (round
tmp_20261110_093321, `tddft_geometry_optimization`).

Nov 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import tempfile
import unittest
from types import SimpleNamespace

from pyrai3md.skills.openqp_skill.service.parser.output_parser import (
    parse_export_files, parse_output)

FINAL = [[-1.0, 0.0, -2.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]


def _write_xyz(path, comment, coords):
    with open(path, 'w') as handle:
        handle.write('%d\n%s\n' % (len(coords), comment))
        for symbol, xyz in zip(('C', 'H', 'H'), coords):
            handle.write('%s %.10f %.10f %.10f\n'
                         % (symbol, xyz[0], xyz[1], xyz[2]))


class TestOptimizedGeometryReachesTheFramework(unittest.TestCase):

    def test_fin_coord_is_the_optimized_structure(self):
        with tempfile.TemporaryDirectory() as calcdir:
            _write_xyz(os.path.join(calcdir, 'demo.xyz'), 'input',
                       [[-1.0, 0.0, -2.5], [0.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0]])
            _write_xyz(os.path.join(calcdir, 'opt.xyz'), 'Geom 2',
                       FINAL)
            with open(os.path.join(calcdir, 'energies'), 'w') as handle:
                handle.write('-77.0\n')
            with open(os.path.join(calcdir, 'opt_status.txt'), 'w') as handle:
                handle.write(' step energy rms_g\n 2 -77.1 0.01\n')
            config = SimpleNamespace(
                project='demo', nstate=1, natom=3, method='hf',
                target_states=[0], nac_type='ktdc')
            workflow = SimpleNamespace(runtype='optimize',
                                       two_manifold=False, is_soc=False)
            results = parse_export_files(config, calcdir, workflow)
        self.assertEqual(results['fin_coord'], FINAL)
        self.assertEqual(results['coordinates'], FINAL)

    def test_the_convergence_flag_uses_the_framework_key(self):
        with tempfile.TemporaryDirectory() as calcdir:
            _write_xyz(os.path.join(calcdir, 'demo.xyz'), 'input', FINAL)
            _write_xyz(os.path.join(calcdir, 'opt.xyz'), 'Geom 2', FINAL)
            with open(os.path.join(calcdir, 'energies'), 'w') as handle:
                handle.write('-77.0\n')
            with open(os.path.join(calcdir, 'demo.log'), 'w') as handle:
                handle.write('PyOQP: Geometry Optimization Has Not Converged. '
                             'Reached The Maximum Iteration\n')
            config = SimpleNamespace(
                project='demo', nstate=1, natom=3, method='hf',
                target_states=[0], nac_type='ktdc')
            workflow = SimpleNamespace(runtype='optimize',
                                       two_manifold=False, is_soc=False)
            results = parse_output(config, calcdir, workflow)
        self.assertIs(results['opt_converge'], False)
        self.assertIs(results['opt_converged'], False)


if __name__ == '__main__':
    unittest.main()
