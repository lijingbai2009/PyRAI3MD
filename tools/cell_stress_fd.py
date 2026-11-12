#!/usr/bin/env python3
"""
======================================================================
This module prices the variable-cell stress channel against an engine.

The question it answers is the one `dev/methods/CELL_OPT_DEVELOPMENT.md`
section 5.3 rests on: is the cell force the optimizer steps on,
`dE/dH = V sigma^T H^-T`, the derivative of the energy the same engine
reports? A stress tensor read back from a file cannot answer that by
itself, so the tool differentiates the engine's own energy along a cell
component and compares.

Two paths are priced, because they are not the same derivative:

  ride  the atoms hold their fractional coordinates while the box moves.
        This is the path the stress identity is stated on, and the one the
        integrator's cell-aware drift takes.
  cart  the atoms hold their Cartesian positions while the box moves.

They differ by exactly the nuclear gradient contracted with the
fractional coordinates,

    dE/dH|_ride = dE/dH|_cart + sum_i g_i[b] * S_i[a]     for slot (a, b)

and the tool checks that identity as well, which is what makes the
comparison a measurement rather than an assertion: on a box whose atoms
are not relaxed the two paths disagree by more than the cell force itself,
and a pricing that only took one of them would report the engine's stress
as wrong.

The verdict is a criterion, not a printout. Exit status is non-zero when
the ride path fails to reproduce the tensor (a sign flip on any slot, or a
ratio outside the tolerance), so the tool can gate a round.

Run it from the package root on a machine that can execute the engine:
    python tools/cell_stress_fd.py [example|path/to/input.yaml] [epsilon]

The first argument is either the name of a shipped example (its fixtures are
taken from `pyrai3md/examples/`) or the path of an input file, in which case the
topology and engine templates beside it are used. Defaults:
`md_lammps_npt_nh_pr` (the equilibrated 60-molecule water box the NPT examples
ship) and a relative strain of 1e-3 per lattice component.

This tool is located in tools/ and is run directly. It does not participate
in test planning or execution logic, and it writes nothing into the package:
its scratch directory is created under the system temp and removed.

Oct 13 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import shutil
import sys
import tempfile

import numpy as np

from pyrai3md.data.build import build_state
from pyrai3md.interface.parser import parse_input
from pyrai3md.schema.adapter import build_config
from pyrai3md.schema.normalize import normalize
from pyrai3md.schema.validate import validate
from pyrai3md.utilities.filesystem import working_directory
from pyrai3md.utilities.lattice import cart_to_frac, cell_volume, frac_to_cart
from pyrai3md.utilities.units import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM
from pyrai3md.workflow import (WORKFLOW_REGISTRY, _graph_backend_execution,
                               _graph_base_flags, resolve_workflow)
from pyrai3md.workflow.calculator import Calculator
from pyrai3md.workflow.dispatch.planner import ExecutionGraphBuilder
from pyrai3md.workflow.optimizer.base import CELL_COMPONENT_SLOTS

# The acceptance criterion, stated in the units the comparison is made in.
# A relative strain of 1e-3 on the shipped water box puts the energy
# difference of one arm at ~1e-7 Hartree, which is three orders above what
# the engine prints and two above the box's own discontinuity floor
# (measured at 5.88e-5 Hartree for a whole-box spelling change, not for a
# strain this small), so the tolerance below is set by the physics of the
# box and not by a wish.
RATIO_TOLERANCE = 0.10
IDENTITY_TOLERANCE = 0.05
# An absolute floor beside the relative one, and it is measured rather than
# chosen: on the shipped box the identity's residual does not shrink with the
# strain (worst slot 3.1e-4 Hartree/Bohr at 1e-3, 8.6e-4 at 1e-4), while the
# ride ratios themselves degrade at 1e-4 (slots yx/zx/zy off by 79/37/13%).
# Both say the same thing - below about 1e-3 the energy differences of this box
# are inside its own discontinuity floor (a pair cutoff of 9.0/10.0 A inside an
# 8.399 A cell), so a relative criterion read there grades the noise. The
# default strain is 1e-3 for that reason, and a residual under this floor is
# the instrument agreeing with itself rather than a claim about the engine.
IDENTITY_FLOOR = 1.0e-3
FIXTURES = ('h2o.data', 'h2o.in.init', 'h2o.in.settings')


def examples_dir():
    """The shipped examples directory, resolved from the package itself."""
    import pyrai3md
    return os.path.join(os.path.dirname(os.path.abspath(pyrai3md.__file__)),
                        'examples')


class Engine(object):
    """One real backend, driven through the production cell channel.

    The state is built by the same pipeline a run uses (parse -> build ->
    normalize -> validate -> build_state), and the cell reaches the backend
    through `system.lattice_vector`, which is what `Calculator._snapshot_cell`
    freezes beside the geometry. Nothing here writes a skill input by hand:
    a pricing that bypassed the channel would price a different question.
    """

    def __init__(self, source, workdir):
        if os.path.isfile(source):
            # An explicit input file: its own directory holds the topology and
            # the engine templates it names, so everything with a fixture
            # extension beside it travels to the scratch directory.
            path = os.path.abspath(source)
            directory = os.path.dirname(path)
            names = [name for name in sorted(os.listdir(directory))
                     if name.endswith(('.data', '.in.init', '.in.settings'))]
        else:
            directory = examples_dir()
            path = os.path.join(directory, source + '.yaml')
            names = list(FIXTURES)
        for name in names:
            shutil.copy2(os.path.join(directory, name),
                         os.path.join(workdir, name))
        raw = parse_input(path)
        self.state = build_state(
            validate(normalize(build_config(raw))), raw_config=raw)
        workflow_cls = resolve_workflow(self.state)
        graph = ExecutionGraphBuilder(WORKFLOW_REGISTRY).build(
            self.state, workflow_cls)
        flags = _graph_base_flags(graph)
        multitask, remotetask, multipool = flags.get(
            'theory', (False, False, 1))
        self.calculator = Calculator(
            self.state, multitask=multitask, remotetask=remotetask,
            multipool=multipool, base_flags=flags,
            backend_execution=_graph_backend_execution(graph))
        self.calculator.initialize()

    def evaluate(self, lattice, coordinates):
        """One evaluation: energy (Hartree), stress (Hartree/Bohr^3), gradient."""
        system = self.state.pmd.system
        rows = np.asarray(lattice, dtype=np.float64)
        lengths = np.linalg.norm(rows, axis=1)
        system.lattice_vector = rows.tolist()
        system.cell = [float(value) for value in
                       (*lengths, 90.0, 90.0, 90.0)]
        system.coordinates = np.asarray(coordinates,
                                       dtype=np.float64).tolist()
        result = self.calculator.run()
        energy = float(np.asarray(result['energy'],
                                 dtype=np.float64).reshape(-1)[0])
        stress = self.state.active_stress()
        if stress is None:
            raise RuntimeError(
                'the evaluation returned no whole-system stress; the '
                'capability this tool prices did not reach the backend')
        gradient = np.asarray(result['gradient'],
                              dtype=np.float64).reshape(-1, 3)
        return energy, np.asarray(stress, dtype=np.float64), gradient


def price(example, epsilon):
    """Price both paths and return the readings with their verdict."""
    workdir = tempfile.mkdtemp(prefix='cell_stress_fd_')
    try:
        with working_directory(workdir):
            engine = Engine(example, workdir)
            system = engine.state.pmd.system
            base = np.asarray(system.lattice_vector, dtype=np.float64)
            coords0 = np.asarray(system.coordinates, dtype=np.float64)
            rows_bohr = base * ANGSTROM_TO_BOHR

            energy0, stress0, grad0 = engine.evaluate(base, coords0)
            conjugate = (cell_volume(rows_bohr) * stress0.T
                         @ np.linalg.inv(rows_bohr).T).reshape(-1)
            truth = conjugate[list(CELL_COMPONENT_SLOTS)]
            frac0 = cart_to_frac(coords0 * ANGSTROM_TO_BOHR, rows_bohr)
            chain = np.array([
                float(np.sum(grad0[:, slot % 3] * frac0[:, slot // 3]))
                for slot in CELL_COMPONENT_SLOTS])

            rows = []
            for slot in CELL_COMPONENT_SLOTS:
                axis, column = slot // 3, slot % 3
                step = epsilon * float(np.linalg.norm(base[axis]))
                delta = np.zeros((3, 3))
                delta[axis, column] = step
                plus, minus = base + delta, base - delta
                step_bohr = step * ANGSTROM_TO_BOHR

                cart = (engine.evaluate(plus, coords0)[0]
                        - engine.evaluate(minus, coords0)[0]
                        ) / (2.0 * step_bohr)
                ride = (engine.evaluate(
                            plus,
                            frac_to_cart(frac0, plus * ANGSTROM_TO_BOHR)
                            * BOHR_TO_ANGSTROM)[0]
                        - engine.evaluate(
                            minus,
                            frac_to_cart(frac0, minus * ANGSTROM_TO_BOHR)
                            * BOHR_TO_ANGSTROM)[0]) / (2.0 * step_bohr)
                rows.append((slot, float(truth[len(rows)]), float(cart),
                             float(ride), float(chain[len(rows)])))
            return {'example': example, 'epsilon': epsilon,
                    'energy': energy0,
                    'gradient_norm': float(np.sqrt(np.sum(grad0 ** 2))),
                    'stress': stress0, 'rows': rows}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def report(readings):
    """Print the table and return (verdict, reasons)."""
    rows = readings['rows']
    print('example            : %s' % readings['example'])
    print('relative strain    : %.1e per lattice component'
          % readings['epsilon'])
    print('E (Hartree)        : %.10f' % readings['energy'])
    print('|gradient| (Ha/Bohr): %.6f' % readings['gradient_norm'])
    print()
    print('%-5s %-15s %-15s %-15s %-9s %-9s' %
          ('slot', 'V s^T H^-T', 'FD cart', 'FD ride', 'ride/', 'cart/'))
    reasons = []
    for (slot, value, cart, ride, _chain) in rows:
        ratio_ride = ride / value if value else float('nan')
        ratio_cart = cart / value if value else float('nan')
        print('%-5d %-15.6e %-15.6e %-15.6e %-9.4f %-9.4f' %
              (slot, value, cart, ride, ratio_ride, ratio_cart))
        if value and np.sign(ride) != np.sign(value):
            reasons.append('slot %d: the ride path flips the sign of the '
                           'cell force' % slot)
        elif value and abs(ratio_ride - 1.0) > RATIO_TOLERANCE:
            reasons.append('slot %d: the ride path is off by %.1f%%, above '
                           'the %.0f%% tolerance'
                           % (slot, 100.0 * abs(ratio_ride - 1.0),
                              100.0 * RATIO_TOLERANCE))
    print()
    print('identity bound: max(%.0f%% of the largest term, %.1e Hartree/Bohr)'
          % (100.0 * IDENTITY_TOLERANCE, IDENTITY_FLOOR))
    print('%-5s %-15s %-15s %-15s' %
          ('slot', 'ride - cart', 'chain sum', 'identity error'))
    for (slot, _value, cart, ride, chain) in rows:
        error = (ride - cart) - chain
        scale = max(abs(ride), abs(cart), abs(chain), 1.0e-30)
        print('%-5d %-15.6e %-15.6e %-15.6e' % (slot, ride - cart, chain,
                                                error))
        bound = max(IDENTITY_TOLERANCE * scale, IDENTITY_FLOOR)
        if abs(error) > bound:
            reasons.append('slot %d: the two paths do not differ by the '
                           'gradient contraction, so the instrument itself '
                           'is not consistent (error %.3e above a bound of '
                           '%.3e)' % (slot, error, bound))
    return (not reasons), reasons


def main(argv):
    example = argv[1] if len(argv) > 1 else 'md_lammps_npt_nh_pr'
    epsilon = float(argv[2]) if len(argv) > 2 else 1.0e-3
    readings = price(example, epsilon)
    passed, reasons = report(readings)
    print()
    if passed:
        print('VERDICT: the stress channel is the derivative of the engine\'s '
              'own energy along the ride path.')
        return 0
    print('VERDICT: FAILED')
    for reason in reasons:
        print('  - %s' % reason)
    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
