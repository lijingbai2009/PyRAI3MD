"""
======================================================================
Tests that tools/analyze_npt_frames.py reads what the round reads.

The tool used to carry its own copies of three judged quantities: a
determinant over reshaped matrices, an SVD polar factor for the frame
rotation, and the pair of thresholds `volume_drift > 0.25` /
`rotation > 17.2 degrees`. The runner computes the first two in
`cell_frame_metrics` and decides with `BOX_TOLERANCES` / `BOX_NOTES`.
Measured on the same synthetic frames, the two determinant paths
disagreed in the thirteenth significant digit and the two polar factors
by 8.795e-15 rad, and 17.2 degrees was never the number the round notes
with (0.15 rad = 8.594 degrees). These cases pin that one function now
answers both sides, and that the threshold the tool reports is the
constant the round uses.

Nov 6 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import importlib.util
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from pyrai3md.qualification import runner

TENSOR_PARTS = ('xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz')
COLUMNS = (['time', 'T_inst(K)', 'Pint_bar', 'Pkin_bar', 'Pvir_bar',
            'scale_cell_pre']
           + ['peps_' + part for part in TENSOR_PARTS]
           + ['eps_' + part for part in TENSOR_PARTS])


def _load_tool():
    """Load tools/analyze_npt_frames.py by path (tools/ is not a package)."""
    repo_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
    path = os.path.join(repo_root, 'tools', 'analyze_npt_frames.py')
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location('npt_frames_tool', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


class _Frames(unittest.TestCase):
    """One synthetic NPT trajectory written like the products are written."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='npt_frames_unit_')
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.title = 'md_lammps_npt_case'

    def _write(self, cells, temperature=300.0, frames=None):
        xyz = os.path.join(self.dir, '%s.md.xyz' % self.title)
        with open(xyz, 'w', encoding='utf-8') as handle:
            for row in cells:
                flat = ' '.join('%.6f' % value
                                for value in np.asarray(row).reshape(-1))
                handle.write('2\nLattice="%s"\nH 0 0 0\nH 0 0 0.74\n' % flat)
        values = frames if frames is not None else len(cells)
        ensemble = os.path.join(self.dir, '%s.md.ensemble' % self.title)
        with open(ensemble, 'w', encoding='utf-8') as handle:
            handle.write(' '.join(COLUMNS) + '\n')
            for index in range(values):
                row = ([0.5 * index, temperature + 2.0 * index, 1.2, 0.9,
                        0.3, 1.0] + [0.01] * 18)
                handle.write(' '.join('%.6f' % value for value in row)
                             + '\n')
        return xyz, ensemble

    def _metrics(self, xyz, ensemble, temperature=300.0):
        rows = runner.read_md_lattice(xyz)
        columns = runner.read_md_ensemble(ensemble, tuple(COLUMNS))
        return runner.cell_frame_metrics(rows, columns, temperature)


@unittest.skipIf(TOOL is None, 'tools/ is only shipped with the source tree')
class TestOneSourceOfTruth(_Frames):

    def test_the_judged_quantities_are_bit_for_bit_the_runner_s(self):
        cells = [np.diag([10.0, 10.0, 10.0]).tolist(),
                 np.diag([10.1, 9.9, 10.0]).tolist(),
                 [[10.0, 0.5, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                 np.diag([10.0, 13.0, 10.0]).tolist()]
        xyz, ensemble = self._write(cells)
        tool = TOOL.analyze(xyz, ensemble, 300.0)
        reference = self._metrics(xyz, ensemble, 300.0)
        for key in ('volume_drift', 'cell_rotation_rad', 'temperature_drift'):
            self.assertEqual(repr(tool[key]), repr(reference[key]), key)

    def test_the_threshold_that_reports_a_row_is_the_round_s_constant(self):
        # A run in the healthy band: 2% volume, 0.05 rad of accumulated
        # turn. It passes the round's bounds and fails them only when the
        # shared constants are moved, which is what proves the tool reads
        # the constants rather than a number copied out of one round.
        cells = [np.diag([10.0, 10.0, 10.0]).tolist(),
                 [[10.0, 0.5, 0.0], [0.0, 10.2, 0.0], [0.0, 0.0, 10.0]],
                 [[10.0, 1.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]]
        xyz, ensemble = self._write(cells)
        metrics = TOOL.analyze(xyz, ensemble, 300.0)
        row = dict(metrics, rotation_deg=float(
            np.degrees(metrics['cell_rotation_rad'])))
        self.assertGreater(row['volume_drift'], 0.0)
        self.assertLess(row['volume_drift'],
                        TOOL.BOX_TOLERANCES['volume_drift'])
        self.assertLess(row['cell_rotation_rad'],
                        TOOL.BOX_NOTES['cell_rotation_rad'])
        self.assertEqual(TOOL.diverged([row]), [])
        with mock.patch.dict(TOOL.BOX_NOTES, {'cell_rotation_rad': 0.01}):
            self.assertEqual(TOOL.diverged([row]), [self.title])
        with mock.patch.dict(TOOL.BOX_TOLERANCES, {'volume_drift': 0.001}):
            self.assertEqual(TOOL.diverged([row]), [self.title])

    def test_no_local_reimplementation_of_the_products_remains(self):
        source = open(os.path.join(os.path.dirname(TOOL.__spec__.origin),
                                   'analyze_npt_frames.py'),
                      encoding='utf-8').read()
        # Read the code, not the docstring: the docstring is where the
        # retired literals are recorded as history.
        code = source.split('import argparse', 1)[1]
        for gone in ('def read_lattices', 'def read_ensemble',
                     'def frame_rotation', '17.2', 'LATTICE_RE'):
            self.assertNotIn(gone, code, gone)


if __name__ == '__main__':
    unittest.main()
