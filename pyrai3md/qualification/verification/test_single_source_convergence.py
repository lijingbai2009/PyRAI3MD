"""
======================================================================
Every convergence point of the "one layer, one source" pass, with the
comparison that shows the two places now produce the same thing.

Each class below names the copies that existed and the equality that
holds after they were merged:

- the three numpy-to-plain converters (set_result, Result.to_dict,
  utilities.serialization.json_safe) are one function;
- the training channel list is the same tuple in both report blocks;
- the lz trajectory energy reader is one function behind two cases;
- the CenterView accessors are read through one expression, and the
  pmd.system view returns the CenterView's own values;
- the two optimizer xyz writers are one function whose only parameter
  is the file mode.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import tempfile
import types
import unittest

import numpy as np

from pyrai3md.data.state import PyRAIState
from pyrai3md.data.runtime.result import Result
from pyrai3md.utilities.serialization import json_safe
from pyrai3md.workflow.integrator.electronic import CenterView
from pyrai3md.workflow.optimize import Optimize

NESTED = {'energy': np.asarray([1.0, 2.0]),
          'count': np.int64(3),
          'scalar': np.float32(0.5),
          'nested': {'gradient': np.asarray([[1.0, 0.0, 0.0]])},
          'flag': np.True_}


class TestOneNumpyConverter(unittest.TestCase):
    """Three converters became one; each caller now answers the same."""

    def test_the_result_dump_and_the_store_agree(self):
        state = types.SimpleNamespace()
        state.pmd = types.SimpleNamespace(results=None)
        PyRAIState.set_result(state, NESTED)
        self.assertEqual(state.pmd.results, json_safe(NESTED))

    def test_a_result_dump_matches_json_safe(self):
        self.assertEqual(Result(data=dict(NESTED)).to_dict(),
                         json_safe(NESTED))

    def test_a_nested_array_no_longer_survives(self):
        state = types.SimpleNamespace()
        state.pmd = types.SimpleNamespace(results=None)
        PyRAIState.set_result(state, {'active': [{'gradient': np.zeros(3)}]})
        nested = state.pmd.results['active'][0]['gradient']
        self.assertIsInstance(nested, list)
        self.assertEqual(nested, [0.0, 0.0, 0.0])


class TestOneTrainingChannelList(unittest.TestCase):
    """Both nn report blocks read the same channel tuple."""

    def test_the_prediction_block_uses_the_training_channels(self):
        from pyrai3md.utilities.print_results import (
            TRAINING_CHANNELS, _format_prediction_std)
        report = {'ndata': 4, 'energy': 1.0, 'gradient': 2.0}
        text = _format_prediction_std({'train_prediction_std': report})
        header = [line for line in text.splitlines()
                  if 'ndata' in line and 'gradient' in line][0]
        for channel in TRAINING_CHANNELS:
            self.assertIn(channel, header)


class TestOneLzEnergyReader(unittest.TestCase):
    """The gsh and lzsh cases read the same columns through one reader."""

    @staticmethod
    def _trajectory(root, title='benchmark_case'):
        with open(os.path.join(root, '%s.md.energies' % title), 'w') as f:
            f.write('time Epot Ekin Etot Epot1 Epot2\n')
            for step in range(3):
                f.write('%d %.6f %.6f 0.0 %.6f %.6f\n'
                        % (step, 1.0 + step, 0.5, 2.0, 3.0 + step))
        with open(os.path.join(root, '%s.md.xyz' % title), 'w') as f:
            for step in range(3):
                f.write('2\ntitle\n')
                f.write('H %.10f 0.0 0.0\nC %.10f 0.0 0.0\n'
                        % (step, step + 1.0))
        return title

    def test_the_two_cases_agree_on_one_file(self):
        from pyrai3md.qualification.benchmark import gsh, lz_data, lzsh
        from pyrai3md.utilities.units import BOHR_TO_ANGSTROM
        with tempfile.TemporaryDirectory() as root:
            title = self._trajectory(root)
            x_bohr, ekin, energies = gsh._parse_lz_trajectory(root, title)
            _x, ekin_shared, energies_shared = (
                lz_data.read_coordinates(root, title),
                *lz_data.read_energies(root, title))
            self.assertEqual(ekin.tolist(), [0.5, 0.5, 0.5])
            self.assertEqual(energies[:, 1].tolist(), [3.0, 4.0, 5.0])
            self.assertTrue(np.array_equal(ekin, ekin_shared))
            self.assertTrue(np.array_equal(energies, energies_shared))
            self.assertTrue(np.allclose(
                x_bohr[:2], np.asarray([0.0, 1.0]) / BOHR_TO_ANGSTROM))

    def test_only_the_shared_module_reads_the_energy_columns(self):
        from pyrai3md.qualification.verification._source_paths import (
            source_tree)
        holders = [path.rsplit('/', 1)[-1]
                   for path in source_tree(
                       'pyrai3md/qualification/benchmark/*.py')
                   if 'table[:, 4:6]' in open(path, encoding='utf-8').read()]
        self.assertEqual(holders, ['lz_data.py'])


class TestOneCenterViewExpression(unittest.TestCase):
    """The system view answers with the CenterView's own reads."""

    @staticmethod
    def _view():
        coordinates = np.asarray([[1.0, 2.0, 3.0],
                                  [4.0, 5.0, 6.0],
                                  [7.0, 8.0, 9.0]])
        state = types.SimpleNamespace(
            center=lambda index: types.SimpleNamespace(
                high_indices=[0, 2], nstate=1),
            pmd=types.SimpleNamespace(
                system=types.SimpleNamespace(natom=3)),
            get_active_configuration=lambda: [1],
            coordinates=coordinates,
            velocities=coordinates * 2.0,
        )
        return CenterView(state, 0), coordinates

    def test_the_three_accessors_match_in_both_views(self):
        view, coordinates = self._view()
        system = view.pmd.system
        masked = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0],
                          [7.0, 8.0, 9.0]])
        self.assertTrue(np.array_equal(view.coordinates, masked))
        self.assertTrue(np.array_equal(system.coordinates, view.coordinates))
        self.assertTrue(np.array_equal(view.velocities, coordinates * 2.0))
        self.assertTrue(np.array_equal(system.velocities, view.velocities))
        self.assertEqual(system.current_state, view.current_state)
        self.assertEqual(view.current_state, 1)


class TestOneXyzWriter(unittest.TestCase):
    """One writer, one frame format, two file modes."""

    ATOMS = ['H', 'C']
    COORDS = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    TITLE = 'Step 1  E = -1.0000000000'

    def _block(self):
        return ('2\n%s\nH          0.0000000000     0.0000000000     0.0000000000\n'
                'C          1.0000000000     0.0000000000     0.0000000000\n'
                % self.TITLE)

    def test_write_then_append_yields_both_frames(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'opt_geom.xyz')
            Optimize._write_xyz(path, self.ATOMS, self.COORDS, 2, self.TITLE)
            with open(path) as handle:
                self.assertEqual(handle.read(), self._block())
            Optimize._write_xyz(path, self.ATOMS, self.COORDS, 2, self.TITLE,
                                append=True)
            with open(path) as handle:
                self.assertEqual(handle.read(), self._block() * 2)

    def test_the_append_writer_is_gone(self):
        self.assertFalse(hasattr(Optimize, '_append_xyz'))


if __name__ == '__main__':
    unittest.main()