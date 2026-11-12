"""
======================================================================
QMMM runs the constraint subsystem exactly as ONIOM does.

The constraint block is accepted for QMMM because nothing in the chain
below it is layer-aware: a constraint names global atoms, its energy and
gradient corrections live in the framework (PyRAIState) and are applied by
`workflow/constraints/runtime.py::finalize` AFTER the composite assembled
its energy, and the optimizer/integrator read the same fix mask for every
composite scheme. Lifting the gate therefore needs evidence that the two
steps really are separate, which is what these tests measure:

- the input pipeline accepts the block on a shipped QMMM example;
- finalize shifts the total energy once, leaves `composite_terms`
  untouched, and reports the correction in `energy_contributions`;
- a `fix` on MM atoms reaches the same mask the optimizer uses, while the
  QM atoms stay movable;
- `hess_atoms` derives from that mask, which is the channel a QMMM
  TS/IRC run needs (an explicit partial set with no fix behind it is still
  refused).

Nov 03 2026
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

from pyrai3md.data.state import PyRAIState
from pyrai3md.planning.assembly import _derive_curvature_atoms
from pyrai3md.data.runtime.state import RuntimeState
from pyrai3md.qualification.verification.test_qmmm_composite import (
    NATOM, _energy_raw, _field_raw, _outcome, _qm_raw, _qmmm_tasks)
from pyrai3md.schema.config import PyRAIConfig, SystemConstraintConfig
from pyrai3md.workflow.constraints import projection, runtime
from pyrai3md.workflow.multiscale.composite import assemble_qmmm

# One bias across the QM/MM boundary and a whole-atom hold on the MM side:
# the bias must be allowed to move a QM atom and the fix must not.
BLOCK = {
    'units': {'length': 'angstrom', 'angle': 'degree', 'energy': 'hartree'},
    'cvs': [{'name': 'r1', 'kind': 'bond', 'atoms': [1, 4]}],
    'biases': [{'id': 'stretch', 'cv': 'r1', 'target': 0.90,
                'force_constant': 0.05}],
    'fixes': [{'id': 'hold_mm', 'atoms': [4, 5, 6], 'direction': 'xyz'}],
}

QM_INDEX = [0, 1, 2]
COORDINATES = [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.9, 0.0],
               [3.0, 3.0, 3.0], [3.9, 3.0, 3.0], [3.0, 3.9, 3.0]]


def _qmmm_state(block=None):
    """A six-atom QMMM state (QM = atoms 1-3) with an optional constraint block."""
    config = PyRAIConfig()
    config.control.jobtype = 'sp'
    config.control.title = 'constraint_qmmm'
    config.control.theory = ['molcas', 'lammps']
    config.control.multiscale_type = 'qmmm'
    config.system.atoms = ['O', 'H', 'H', 'O', 'H', 'H']
    config.system.mass = [16.0, 1.0, 1.0, 16.0, 1.0, 1.0]
    config.system.natom = NATOM
    config.system.coordinates = [list(row) for row in COORDINATES]
    config.system.charges = 0
    config.system.multiplicity = 1
    config.system.nstate = 1
    config.system.highlevel = list(QM_INDEX)
    config.system.embedding = True
    config.system.q_qm = [0.1, 0.2, -0.3]
    if block is not None:
        config.system.constraints = SystemConstraintConfig(**block)
    state = PyRAIState(runtime=RuntimeState(total_steps=1))
    state.pmd = config
    return state


class TestQmmmAcceptsTheConstraintBlock(unittest.TestCase):
    """The input pipeline must stop refusing the block on a QMMM spelling."""

    def test_a_shipped_qmmm_example_keeps_its_constraint_block(self):
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate
        here = os.path.dirname(os.path.abspath(__file__))
        source = os.path.join(here, os.pardir, os.pardir, 'examples',
                              'multiscale_qmmm_embedding_sp.yaml')
        work = tempfile.mkdtemp(prefix='constraint_qmmm_')
        try:
            with open(source) as handle:
                text = handle.read()
            text += ('\nsystem_constraint:\n'
                     '  unit_length: angstrom\n'
                     '  unit_angle: degree\n'
                     '  unit_energy: hartree\n'
                     '  fix_atoms: 4-6\n'
                     '  fix_id: hold_mm\n')
            path = os.path.join(work, 'multiscale_qmmm_constraint.yaml')
            with open(path, 'w') as handle:
                handle.write(text)
            raw = parse_input(path)
            config = validate(normalize(build_config(raw)))
            self.assertEqual(config.control.multiscale_type, 'qmmm')
            self.assertIsNotNone(config.system.constraints)
        finally:
            shutil.rmtree(work, ignore_errors=True)


class TestQmmmConstraintAccounting(unittest.TestCase):
    """finalize is a separate step from assembly: one shift, no term change."""

    def _assembled(self):
        tasks = _qmmm_tasks()
        outcomes = [_outcome(tasks[0], _field_raw()),
                    _outcome(tasks[1], _qm_raw()),
                    _outcome(tasks[2], _energy_raw())]
        request = {'natom': NATOM, 'nstate': 1, 'need_gradient': True}
        return assemble_qmmm(tasks, outcomes, request)

    def test_finalize_shifts_the_total_once_and_keeps_the_terms(self):
        data = self._assembled()
        before_energy = float(np.asarray(data['energy'])[0])
        before_terms = {key: list(value) if isinstance(value, list) else value
                        for key, value in data['composite_terms'].items()}
        before_gradient = np.array(data['gradient'], dtype=np.float64)

        state = _qmmm_state(BLOCK)
        runtime.finalize(state, data)

        after_energy = float(np.asarray(data['energy'])[0])
        self.assertEqual(data['composite_terms'], before_terms)
        # Non-identity witness: this bias is stretched, so the total moved.
        self.assertNotEqual(after_energy, before_energy)
        contributions = data.get('energy_contributions') or {}
        self.assertTrue(any(key.endswith('stretch') for key in contributions),
                        contributions)
        self.assertAlmostEqual(
            sum(float(value) for value in contributions.values()),
            after_energy - before_energy, places=12)
        self.assertFalse(np.array_equal(np.asarray(data['gradient']),
                                        before_gradient))
        # The QM atoms are on the bias side, so their rows move; the fixed
        # MM atoms keep only their half of the bias, which the projection
        # removes later (the optimizer/integrator reads the same mask).
        self.assertEqual(np.asarray(data['gradient']).shape,
                         before_gradient.shape)


class TestQmmmFixMask(unittest.TestCase):
    """The mask the optimizer uses comes from the same plan, for QMMM too."""

    def test_fix_holds_the_mm_atoms_and_frees_the_qm_region(self):
        state = _qmmm_state(BLOCK)
        plan = state.constraint_plan()
        self.assertFalse(plan.is_empty)
        mask = projection.build_fix_mask(plan, NATOM)
        self.assertTrue(mask[3:6].all(),
                        'the MM atoms named by fix must be held')
        self.assertFalse(mask[0:3].any(),
                         'the QM region must stay movable')

    def test_without_a_block_nothing_is_held(self):
        plan = _qmmm_state(None).constraint_plan()
        self.assertTrue(plan.is_empty)


class TestQmmmCurvatureAtoms(unittest.TestCase):
    """hess_atoms follows the fix mask, which is how a QMMM TS run starts."""

    def test_a_fix_on_the_mm_region_derives_the_qm_curvature_set(self):
        state = _qmmm_state(BLOCK)
        _derive_curvature_atoms(state, 'ts')
        self.assertEqual(sorted(state.pmd.hessian.hess_atoms),
                         [index + 1 for index in QM_INDEX])

    def test_an_explicit_partial_set_without_a_fix_is_still_refused(self):
        state = _qmmm_state(None)
        state.pmd.hessian.hess_atoms = [index + 1 for index in QM_INDEX]
        with self.assertRaises(ValueError) as caught:
            _derive_curvature_atoms(state, 'ts')
        self.assertIn('no fix', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
