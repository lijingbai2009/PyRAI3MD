"""
======================================================================
This module locks the adaptive channel scope to one schema home.

The adaptive workflow refuses channels this build cannot label (its
labels stop at energy, gradient, NAC and SOC). That refusal lives in
`schema/normalize.reconcile_adaptive_properties`, where the final
switch value is known: `dynamics.dodipole` and `dynamics.radiative`
can turn the adaptive switches on, so a check on the raw input would
miss the case the workflow used to catch.

The criterion is both doors: an explicit adaptive switch and an md
switch that enables it are refused at config build, with the channel
named.

Nov 07 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

from pyrai3md.schema.config import PyRAIConfig
from pyrai3md.schema.normalize import normalize


class TestUnlabelledChannelsAreRefusedAtConfigTime(unittest.TestCase):
    """Both the adaptive switch and its md source are refused."""

    def _config(self, adaptive_key=None, dynamics_key=None):
        config = PyRAIConfig()
        config.control.jobtype = 'adaptive'
        if adaptive_key is not None:
            setattr(config.adaptive, adaptive_key, 1)
        if dynamics_key is not None:
            setattr(config.dynamics, dynamics_key, 1)
        return config

    def test_the_explicit_switch_is_refused(self):
        for key in ('sample_dipole', 'sample_transition_dipole',
                    'sample_oscillator_strength'):
            with self.subTest(key):
                with self.assertRaises(ValueError) as caught:
                    normalize(self._config(adaptive_key=key))
                self.assertIn(key, str(caught.exception))

    def test_the_md_switch_that_enables_it_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            normalize(self._config(dynamics_key='dodipole'))
        self.assertIn('sample_dipole', str(caught.exception))

    def test_nac_and_soc_stay_legal(self):
        config = self._config(adaptive_key='sample_soc')
        normalize(config)


if __name__ == '__main__':
    unittest.main()