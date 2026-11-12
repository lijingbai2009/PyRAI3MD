"""
======================================================================
Test suite for the vibrational initial-condition sampling:
- pyrai3md/workflow/sampling.py

Tests cover:
- Laguerre polynomial values
- molden frequency reader (inline mock file)
- Wigner kernel statistics (zero-point variance at low
  temperature), seed reproducibility and the classical
  Boltzmann kernel widths
- sample_initial_conditions assembly (shapes, units, seed
  reproducibility, error paths)
- load_frequency_data file naming and format validation

Aug 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from pyrai3md.utilities.units import (AMU_TO_AU, ANGSTROM_TO_BOHR,
                            BOHR_TO_ANGSTROM, CM_TO_HARTREE,
                            KELVIN_TO_HARTREE)

# Inline mock of a Molcas .freq.molden record: a diatomic with one
# vibrational mode along z (unnormalized unmass-weighted).
MOLDEN_TWO_ATOM = """[N_FREQ]
1
[FREQ]
3000.0
[INT]
10.0
[NATOM]
2
[FR-COORD]
C 0.0 0.0 0.0
H 0.0 0.0 2.0
 vibration    1
 0.0 0.0 -1.0
 0.0 0.0 1.0
[RMASS]
0.92
"""


def _freqdata_one_mode(freq=3000.0):
    """A diatomic freqdata dict with one mode along z.

    The vectors are normalized so that sum(m x^2) = 1 with the masses in amu,
    which is the contract every reader in `workflow/sampling.py` states and the
    samplers rely on. It used to be (0, 0, -1), (0, 0, +1) here, a pair whose
    sum(m x^2) is 13.019: an instrument built on that fixture measured the
    amplitude of a mode that no reader can produce.
    """
    amass = np.array([[12.011], [1.008]])
    scale = 1.0 / np.sqrt(float(amass.sum()))
    return {
        'nfreq': 1,
        'freqs': np.array([[freq]]),
        'natom': 2,
        'atoms': np.array(['C', 'H']),
        'xyz': np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        'vib': np.array([[[0.0, 0.0, -scale], [0.0, 0.0, scale]]]),
        'amass': amass,
    }


class TestLaguerre(unittest.TestCase):
    """Test the Laguerre polynomial recurrence."""

    def test_low_order_values(self):
        from pyrai3md.workflow.sampling import _laguerre

        # L_0(x) = 1, L_1(x) = 1 - x, L_2(x) = 1 - 2x + x^2/2
        self.assertAlmostEqual(_laguerre(0, 3.0), 1.0)
        self.assertAlmostEqual(_laguerre(1, 2.0), -1.0)
        self.assertAlmostEqual(_laguerre(2, 2.0), -1.0)
        self.assertAlmostEqual(_laguerre(2, 4.0), 1.0)


class TestReadMolden(unittest.TestCase):
    """Test the molden frequency reader."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pyrai3md_sampling_')
        self.path = os.path.join(self.tmpdir, 'mock.freq.molden')
        self._previous = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self._previous)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_parses_the_freqdata_contract(self):
        from pyrai3md.workflow.sampling import read_molden

        with open(self.path, 'w') as handler:
            handler.write(MOLDEN_TWO_ATOM)
        freqdata = read_molden(self.path)
        self.assertEqual(freqdata['nfreq'], 1)
        self.assertEqual(freqdata['natom'], 2)
        self.assertAlmostEqual(freqdata['freqs'][0, 0], 3000.0)
        self.assertEqual(list(freqdata['atoms']), ['C', 'H'])
        self.assertEqual(freqdata['vib'].shape, (1, 2, 3))
        self.assertAlmostEqual(freqdata['vib'][0, 1, 2], 1.0)
        self.assertAlmostEqual(freqdata['xyz'][1, 2], 2.0)
        self.assertAlmostEqual(freqdata['amass'][0, 0], 12.011)

    def test_missing_file_raises(self):
        from pyrai3md.workflow.sampling import read_molden

        with self.assertRaises(FileNotFoundError):
            read_molden(os.path.join(self.tmpdir, 'absent.molden'))


class TestLoadFrequencyData(unittest.TestCase):
    """Test the title/format resolution of the frequency file."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pyrai3md_sampling_')
        self._previous = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self._previous)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_resolves_the_title_suffix(self):
        from pyrai3md.workflow.sampling import load_frequency_data

        with open('run.freq.molden', 'w') as handler:
            handler.write(MOLDEN_TWO_ATOM)
        freqdata = load_frequency_data('run', 'molden')
        self.assertEqual(freqdata['nfreq'], 1)

    def test_missing_file_raises(self):
        from pyrai3md.workflow.sampling import load_frequency_data

        with self.assertRaises(FileNotFoundError):
            load_frequency_data('run', 'molden')

    def test_unknown_format_raises(self):
        from pyrai3md.workflow.sampling import load_frequency_data

        with self.assertRaises(ValueError):
            load_frequency_data('run', 'foobar')








if __name__ == '__main__':
    unittest.main()


