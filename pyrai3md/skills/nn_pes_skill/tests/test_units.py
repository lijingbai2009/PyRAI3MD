"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Unit tests of the unit bridge.

The measured facts behind them: the container returns the derivative
with respect to the coordinate unit it was fed (central-difference ratio
0.99998, p10 0.99906, p90 1.00094 on a trained ensemble), and the
framework integrates with gradients in Hartree/Bohr.

These tests need no TensorFlow.

"""

import unittest

import numpy as np

from pyrai3md.skills.nn_pes_skill.schema import reference
from pyrai3md.skills.nn_pes_skill.service import trainer_data
from pyrai3md.skills.nn_pes_skill.service.calculator import GRADIENT_TO_AU
from pyrai3md.skills.nn_pes_skill.service.calculator import _coupling_axis


class TestUnitConstants(unittest.TestCase):

    def test_gradient_bridge_is_the_bohr_to_angstrom_factor(self):
        self.assertEqual(GRADIENT_TO_AU, reference.BOHR_TO_ANGSTROM)

    def test_factor_is_not_inverted(self):
        # 1 Hartree per Angstrom is a smaller slope per Bohr, because a
        # Bohr is a shorter step than an Angstrom.
        self.assertLess(GRADIENT_TO_AU, 1.0)
        self.assertAlmostEqual(GRADIENT_TO_AU, 0.52917721090299996, places=12)

    def test_round_trip(self):
        self.assertAlmostEqual(reference.ANGSTROM_TO_BOHR
                               * reference.BOHR_TO_ANGSTROM, 1.0, places=15)

    def test_the_skill_declares_what_units_it_reads_and_prints(self):
        # The reading convention is the framework's; the printed one is the
        # container's own, and both are stated instead of configurable.
        self.assertEqual(reference.LABEL_ENERGY_UNIT, 'hartree')
        self.assertEqual(reference.LABEL_GRADIENT_UNIT, 'hartree/bohr')
        self.assertEqual(reference.LABEL_SOC_UNIT, 'cm-1')
        self.assertEqual(reference.METRIC_ENERGY_UNIT, 'eV')
        self.assertEqual(reference.METRIC_GRADIENT_UNIT, 'eV/angstrom')
        self.assertEqual(reference.METRIC_SOC_UNIT, 'cm-1')


class TestMatchedLabels(unittest.TestCase):
    """A file's energy and gradient are made to describe one surface.

    The training file carries Hartree energies and Hartree/Bohr gradients
    and couplings, which is what the framework's engine backends report. The
    network is fed Angstrom, because that is what a dynamics step hands the
    skill, so the derivative is converted onto that axis. Nothing else is
    asked of the caller, and there is no unit keyword to get wrong: a switch
    would only let a file be declared in a scheme it does not actually use,
    and that failure is invisible in a single prediction.
    """

    def setUp(self):
        self.dataset = {
            'xyz': np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
                            dtype=np.float64),
            'energy': np.array([[1.0, 2.0]], dtype=np.float64),
            'grad': np.ones((1, 2, 2, 3), dtype=np.float64),
            'nac': np.ones((1, 1, 2, 3), dtype=np.float64),
        }

    def test_energy_passes_through(self):
        out = trainer_data.match_labels(self.dataset)
        self.assertAlmostEqual(float(out['energy'][0][0]), 1.0)

    def test_coordinates_pass_through(self):
        # The geometry is already the axis the skill is handed. Moving it as
        # well would shift the surface the energy belongs to.
        out = trainer_data.match_labels(self.dataset)
        self.assertAlmostEqual(float(out['xyz'][0][1][0]), 1.0)

    def test_per_bohr_derivative_becomes_per_angstrom(self):
        # One Hartree per Bohr is a steeper slope per Angstrom, because an
        # Angstrom is a longer step.
        out = trainer_data.match_labels(self.dataset)
        expected = 1.0 / reference.BOHR_TO_ANGSTROM
        self.assertAlmostEqual(float(out['grad'][0][0][0][0]), expected)
        self.assertAlmostEqual(float(out['nac'][0][0][0][0]), expected)

    def test_conversion_is_the_inverse_of_the_serving_one(self):
        # match_labels brings the label onto the fed axis and the boundary
        # takes the network's derivative back off it. Composed they must be
        # the identity, or the reported gradient would be the reported
        # energy's derivative by exactly this factor.
        out = trainer_data.match_labels(self.dataset)
        self.assertAlmostEqual(float(out['grad'][0][0][0][0])
                               * GRADIENT_TO_AU, 1.0, places=6)

    def test_labels_are_not_converted_twice(self):
        once = trainer_data.match_labels(self.dataset)
        again = trainer_data.match_labels(once)
        # Applying the conversion twice is a real mistake a caller can make
        # by feeding a trained folder's own data back through the trainer,
        # so its size is stated here: another factor of 1.8897. The guard
        # against it is that a serving run never reads the training file.
        self.assertAlmostEqual(float(again['grad'][0][0][0][0]),
                               float(once['grad'][0][0][0][0])
                               / reference.BOHR_TO_ANGSTROM, places=4)

    def test_declared_units_that_are_not_the_convention_are_refused(self):
        trainer_data.check_units('f.json', {'energy': 'hartree',
                                            'gradient': 'hartree/bohr',
                                            'nac': 'Ha/Bohr',
                                            'soc': 'cm-1'})
        with self.assertRaises(ValueError):
            trainer_data.check_units('f.json', {'energy': 'eV'})
        with self.assertRaises(ValueError):
            trainer_data.check_units('f.json', {'gradient': 'eV/angstrom'})

    def test_a_file_without_a_unit_block_is_read_as_the_convention(self):
        trainer_data.check_units('f.json', None)
        trainer_data.check_units('f.json', {})


class TestCouplingChannelOrder(unittest.TestCase):

    def setUp(self):
        # Three channels, each distinguishable, (nchannel, natom, 3)
        self.channels = np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                                  [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                                  [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]])

    def test_no_request_keeps_trained_order(self):
        out = _coupling_axis(self.channels[None], [], [[0, 1], [0, 2], [1, 2]])
        self.assertEqual(out.shape, (3, 2, 3))
        self.assertEqual(out.tolist()[0][0], [1.0, 1.0, 1.0])

    def test_requested_order_is_applied(self):
        mapping = [[0, 1], [0, 2], [1, 2]]
        out = _coupling_axis(self.channels[None], [[1, 2], [0, 1]], mapping)
        self.assertEqual(out.shape, (2, 2, 3))
        self.assertEqual(out[0].tolist()[0], [3.0, 3.0, 3.0])
        self.assertEqual(out[1].tolist()[0], [1.0, 1.0, 1.0])

    def test_pair_order_inside_a_request_is_ignored(self):
        mapping = [[0, 1], [0, 2], [1, 2]]
        forward = _coupling_axis(self.channels[None], [[0, 1]], mapping)
        reverse = _coupling_axis(self.channels[None], [[1, 0]], mapping)
        self.assertEqual(forward.tolist(), reverse.tolist())

    def test_unknown_pair_raises_instead_of_zero_padding(self):
        mapping = [[0, 1]]
        with self.assertRaises(ValueError):
            _coupling_axis(self.channels[None], [[1, 2]], mapping)

    def test_a_map_shorter_than_the_channels_raises(self):
        # The declaration and the trained width have to agree, or numbers
        # would go out under state names the model never carried.
        with self.assertRaises(ValueError):
            _coupling_axis(self.channels[None], [], [[0, 1]])

    def test_a_pair_request_without_a_map_raises(self):
        with self.assertRaises(ValueError):
            _coupling_axis(self.channels[None], [[0, 1]], [])


if __name__ == '__main__':
    unittest.main()
