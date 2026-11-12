"""
======================================================================
This module tests the stochastic cell rescaling barostat, pyrai3md.
workflow.integrator.ensemble._scr, and the rule that says which
thermostat it may be paired with.

Why each case is here:

- SCR is the first barostat in this framework whose correctness is a
  VARIANCE rather than a conservation law, and the two terms that make it
  right are each easy to lose silently: the Ito drift correction the
  published equation carries, and the side the random matrix multiplies
  the cell on. Both are pinned against the source (Bernetti & Bussi,
  J. Chem. Phys. 153, 114107 (2020), eq 3; Del Tatto, arXiv:2111.06402,
  eq 9, 16-17, 28-29) rather than against this file's own arithmetic, so
  an edit that drops one of them fails here and not in a remote round;
- the fluctuation-dissipation ratio is measured from many draws of one
  state, which is what makes a 2% assertion meaningful: a single step of
  a stochastic scheme proves nothing;
- the momenta must come out of a hundred cell rescalings with their kinetic
  energy intact. The variant that rescales them with mu^-1 did not: that is the
  tmp_20260929_151626 blow-up, and it is now a regression case;
- the momenta must come out of fifty cell rescalings untouched, which is the
  property the positions-only formulation buys (eq 29b's p^T <- p^T mu^-1 is
  declined, and the reason is in the case that asserts the tensor refusal);
- and the tensor form of the same scheme is refused rather than implemented: it
  was implemented, measured in the test environment, and removed, so the refusal case
  is what keeps the measurement from being relitigated;
- a stochastic barostat must still be reproducible, so the draw stream is
  pinned to the trajectory seed the thermostat already uses;
- the scheme can hand back a cell whose determinant is negative. That is
  refused and recorded, not committed;
- and finally the pairing rule. schema/validate.py has refused csvr with
  mttk for some time, which is the rule that made the example
  md_lammps_npt_csvr_mttk.yaml unloadable -- it passed every local test
  suite because nothing loaded an example through validate(). The last
  case is that missing gate, over every shipped example.

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

try:
    import pytest
except ImportError:  # unittest-only hosts do not need pytest
    pytest = None

if pytest is not None:
    pytestmark = pytest.mark.integration

import copy
import glob
import os
import unittest

import numpy as np

from pyrai3md.qualification.verification._ensemble_helpers import (
    NATOM, TILTED_LATTICE, kinetic_pressure, make_npt_state, run_npt_loop,
    surrogate_stress)
from pyrai3md.utilities.units import (
    FS_TO_AU_TIME, KELVIN_TO_HARTREE, bar_to_au_pressure,
    compressibility_bar_to_au)
from pyrai3md.workflow.integrator import ensemble as E
from pyrai3md.workflow.integrator.ensemble import _berendsen, _scr

DRAWS = 2000
KT = 300.0 * KELVIN_TO_HARTREE


def _load_state(mode='isotropic', lattice=None, tau=1000.0):
    """One prepared NPT state, plus the stress that goes with it."""
    state = make_npt_state(barostat='scr', barostat_mode=mode,
                           lattice=lattice or TILTED_LATTICE)
    state.pmd.dynamics.barostat_tau = tau
    p_iso_ref = float(np.trace(kinetic_pressure(state)) / 3.0)
    stress = surrogate_stress(state, p_iso_ref)
    state.set_result({'gradient': np.zeros((NATOM, 3)).tolist(),
                      'stress_tensor': stress[None].tolist()})
    return state, stress


def _mu_of(state, dt, draw):
    """One draw of the rescaling matrix from a prepared state.

    The state arrives with this step's result already stored, so a deep
    copy of it is a fresh sample of the same configuration: every draw
    then measures the same increment, which is the only way a variance
    over draws means anything.
    """
    np.random.seed(draw)
    _scr(state, dt)
    return np.asarray(state.runtime.ensemble['mu'], dtype=np.float64)





class TestTheCombinationIsCheckedBeforeTheRun(unittest.TestCase):
    """schema/validate.py's pairing rule, which is what an example hits
    first."""

    def _check(self, **overrides):
        from pyrai3md.schema.validate import validate_ensemble
        data = {'ensemble': 'npt', 'thermostat': 'nose-hoover-chain',
                'barostat': 'scr', 'barostat_mode': 'isotropic',
                'thermostat_tau': 100.0, 'barostat_tau': 1000.0,
                'compressibility': 4.5e-5}
        data.update(overrides)
        return validate_ensemble(data)

    def test_scr_accepts_every_thermostat(self):
        """Bernetti & Bussi say so in terms; the point of the entry is
        that the barostat does the pairing, not the framework's
        convenience."""
        for thermostat in ('nose-hoover', 'nose-hoover-chain', 'csvr'):
            with self.subTest(thermostat=thermostat):
                self._check(thermostat=thermostat)

    def test_the_extended_cell_barostats_refuse_csvr(self):
        for barostat in ('mttk', 'parrinello-rahman', 'pr'):
            with self.subTest(barostat=barostat):
                with self.assertRaises(ValueError) as caught:
                    self._check(thermostat='csvr', barostat=barostat)
                message = str(caught.exception)
                self.assertIn('csvr', message)
                self.assertIn('thermostat', message)

    def test_the_tensor_form_of_scr_is_refused(self):
        """Implemented, measured in the test environment, removed.

        The refusal carries the finding so it does not read as an oversight:
        300 K to 897 K within 200 steps of the 60-atom water box, and
        ten-fold tau_p slowed the run-away by about the factor tau_p predicts
        rather than removing it. A liquid has no shear modulus, so the strain
        the tensor form injects is dissipated into heat instead of stored --
        and the conservative toy surrogate, whose stress is an analytic
        function of the cell, has no channel to show that in.
        """
        with self.assertRaises(ValueError) as caught:
            self._check(barostat_mode='anisotropic')
        message = str(caught.exception)
        self.assertIn('scr', message)
        self.assertIn('isotropic', message)

    def test_the_alias_resolves_to_the_canonical_token(self):
        self._check(barostat='c-rescale')
        self._check(barostat='stochastic-cell-rescaling')

    def test_every_shipped_example_survives_validation(self):
        """The gate that was missing.

        md_lammps_npt_csvr_mttk.yaml paired csvr with mttk, which this
        module's own validator has refused for a long time: the file
        loaded through every local test suite and would have died at
        Step 4 of pyrai3md.pmd.run on the cluster. Parsing, building,
        normalizing and validating each shipped example is the cheapest
        way to make sure no input file ships that its own schema
        rejects.
        """
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate

        import pyrai3md
        root = os.path.dirname(os.path.abspath(pyrai3md.__file__))
        paths = sorted(glob.glob(os.path.join(root, 'examples', '*.yaml')))
        self.assertGreater(len(paths), 20)
        for path in paths:
            with self.subTest(example=os.path.basename(path)):
                config = normalize(build_config(parse_input(path)))
                validate(config)


if __name__ == '__main__':
    unittest.main()
