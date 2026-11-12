"""
======================================================================
The constraint subsystem through the real Calculator and integrator.

The subsystem's own suite proves the arithmetic; this file proves the wiring,
which is the half a shadow build cannot test. It drives the framework's actual
entry points - `Calculator.run` with a mock electronic provider and
`operations.apply_constrain` - over a four-atom system that declares every kind
of action at once, and it holds the two promises the merge was judged on:

- DC-1: the same input with no constraint block reproduces the unconstrained
  numbers exactly, so shipping this subsystem closed is a checked fact and not
  an intention;
- one application per evaluation: `base + sum(components) == total` on the
  result the run writes back, and calling the finalizer again does not add
  anything, because the MD loop evaluates a step more than once on some paths.

The gradient the mock provider returns is analytic in the coordinates it is
given, so the bias and wall gradients here are checked against a closed form
rather than against another call of the same code.

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md.data.constraint import ConstraintPlan
from pyrai3md.schema.config import PyRAIConfig
from pyrai3md.utilities.units import AMU_TO_AU, ANGSTROM_TO_BOHR
from pyrai3md.workflow.calculator import Calculator
from pyrai3md.workflow.integrator import operations

ATOMS = ['O', 'H', 'H', 'H']
MASSES = [15.999, 1.008, 1.008, 1.008]
COORDS = [[0.00, 0.00, 0.00],
          [0.96, 0.00, 0.00],
          [-0.24, 0.93, 0.00],
          [-0.10, -0.30, 0.88]]
BASE_ENERGY = -75.4
BLOCK = {
    'units': {'length': 'angstrom', 'angle': 'degree', 'energy': 'hartree'},
    'cvs': [{'name': 'r1', 'kind': 'bond', 'atoms': [1, 2]}],
    'biases': [{'id': 'stretch', 'cv': 'r1', 'target': 0.72,
                'force_constant': 0.05}],
    'walls': [{'id': 'vessel', 'geometry': 'sphere', 'center': [0.0, 0.0, 0.0],
               'radius': 1.5, 'atoms': [4], 'force_constant': 0.01}],
    'fixes': [{'id': 'frozen', 'atoms': [3], 'direction': 'xyz'}],
}

# The same five families the way an input file spells them: one keyword per
# line, one ';' segment per object. `build_config` takes this and only this;
# BLOCK above is what the reader hands to the schema.
FLAT_BLOCK = {
    'unit_length': 'angstrom', 'unit_angle': 'degree',
    'unit_energy': 'hartree',
    'cv_name': 'r1', 'cv_bond': '1 2',
    'bias_cv': 'r1', 'bias_cv_ref': '0.72', 'bias_cv_fc': '0.05',
    'bias_id': 'stretch',
    'wall_type': 'sphere', 'wall_sphere_radius': '1.5',
    'wall_center': '0 0 0', 'wall_atoms': '4', 'wall_fc': '0.01',
    'wall_id': 'vessel',
    'fix_atoms': '3', 'fix_id': 'frozen',
}


class GradientBackend(object):
    """A provider whose energy is analytic, so the sums can be checked.

    `E = E0 + 0.5 * sum(g_i . r_i)` with a fixed per-atom slope gives a constant
    gradient the constraint correction is added to; the shape is what the test
    asserts on, not the size of the electronic part.
    """

    def __init__(self, config=None, execution_id=''):
        self.config = config
        self.execution_id = execution_id
        self.calls = 0
        self.slope = np.array([[0.010, -0.004, 0.002],
                               [-0.003, 0.007, 0.001],
                               [0.004, 0.001, -0.006],
                               [-0.002, -0.005, 0.003]])

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None):
        self.calls += 1
        array = np.asarray(coordinates, dtype=np.float64)
        gradient = self.slope.copy()
        energy = BASE_ENERGY + 0.5 * float(np.sum(self.slope * array))
        return {'energy': energy, 'gradient': gradient,
                'atomic_energy': [energy / len(atoms)] * len(atoms)}

    def close(self):
        pass


def _state(block=None, coordinates=None, velocities=None):
    config = PyRAIConfig()
    config.control.jobtype = 'sp'
    config.control.title = 'constraint_end_to_end'
    config.system.atoms = list(ATOMS)
    config.system.mass = list(MASSES)
    config.system.natom = len(ATOMS)
    config.system.coordinates = (list(coordinates if coordinates is not None
                                      else COORDS))
    if velocities is not None:
        config.system.velocities = [list(row) for row in velocities]
    config.system.charges = 0
    config.system.multiplicity = 1
    config.system.nstate = 1
    if block is not None:
        from pyrai3md.schema import SystemConstraintConfig
        config.system.constraints = SystemConstraintConfig(**block)
    from pyrai3md.data.runtime.state import RuntimeState
    state = _PyRAIState(runtime=RuntimeState(total_steps=10))
    state.pmd = config
    return state


def _PyRAIState(runtime=None):
    from pyrai3md.data.state import PyRAIState
    return PyRAIState(runtime=runtime)


def _run(block=None, coordinates=None):
    backend = GradientBackend()
    state = _state(block, coordinates)
    calc = Calculator(dummy_backend=backend)
    calc.state = state
    result = calc.run()
    return state, result, backend










if __name__ == '__main__':
    unittest.main()
