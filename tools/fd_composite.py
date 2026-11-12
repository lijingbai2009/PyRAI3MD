#!/usr/bin/env python3
"""
======================================================================
This module prices the composite surface's gradient against its own energy.

The question it answers is not the one `tools/verify.py` section I asks.
That section checks an internal identity - the reported gradient is the force
the engine returned plus the gas and electrostatic corrections, term by term -
and an identity of that kind holds even when every term is wrong together.
What a consumer of the composite surface needs is that the energy and the
gradient belong to the same surface, so this tool differentiates the
composite's own energy along a few Cartesian degrees of freedom and compares
the central difference with the gradient the same evaluation reported.

Each evaluation runs the production pipeline (parse -> build -> normalize ->
validate -> build_state -> Calculator), and every displacement gets a fresh
state and calculator. Reusing one calculator across displacements would leave
the geometry channel cached, which is exactly the failure family this
instrument exists to catch.

The displacement is taken in Angstrom along one Cartesian axis; the reported
gradient is Hartree/Bohr, so the difference is converted with
0.529177210903 Angstrom/Bohr. The composite's components (E_classic, U_main,
U_gas, QMMM_el, E_QM) are differenced as well, so a mismatch is attributed
instead of just reported.

The verdict is a criterion, not a printout. Exit status is non-zero when any
probed degree of freedom misses the tolerance, so the tool can gate a round.

Run it from the package root on a machine that can execute the engine:
    python tools/fd_composite.py [example|path/to/input.yaml] [--delta 0.005]

The first argument is either the name of a shipped example (its topology
fixtures are taken from `pyrai3md/examples/`) or the path of an input file, in
which case the `.data`/`.in.init`/`.in.settings` files beside it travel with it.
Defaults: `multicenter_qmmm_sp`, five degrees of freedom (the three atoms of
the first center on z, one classical atom on z, one on x) and a 0.005 Angstrom
step.

This tool is located in tools/ and is run directly. It does not participate in
test planning or execution logic, and it writes nothing into the package: its
scratch directory is created under the system temp and removed.

Nov 6 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import argparse
import copy
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
from pyrai3md.utilities.units import ANGSTROM_TO_BOHR
from pyrai3md.utilities.filesystem import working_directory
from pyrai3md.workflow.calculator import Calculator

DEFAULT_EXAMPLE = 'multicenter_qmmm_sp'
DEFAULT_DOFS = ((9, 2), (10, 2), (11, 2), (0, 2), (30, 0))
DEFAULT_DELTA = 5.0e-3
DEFAULT_TOLERANCE = 1.0e-3
FIXTURES = ('h2o.data', 'h2o.in.init', 'h2o.in.settings')
COMPONENTS = ('e_classic', 'u_main', 'u_gas', 'qmmm_el')


def examples_dir():
    """The shipped examples directory, resolved from the package itself."""
    import pyrai3md
    return os.path.join(os.path.dirname(os.path.abspath(pyrai3md.__file__)),
                        'examples')


class Engine(object):
    """One composite surface, evaluated at whatever geometry it is given."""

    def __init__(self, source, workdir):
        if os.path.isfile(source):
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
        self.raw = parse_input(path)
        self.config = validate(normalize(build_config(self.raw)))
        self.coordinates = np.asarray(self.config.system.coordinates,
                                      dtype=np.float64)

    def evaluate(self, coordinates):
        """One evaluation: energy (Hartree), gradient (Ha/Bohr), components."""
        config = copy.deepcopy(self.config)
        config.system.coordinates = np.asarray(coordinates,
                                               dtype=np.float64).tolist()
        state = build_state(config, raw_config=copy.deepcopy(self.raw))
        calculator = Calculator(state=state).initialize()
        result = calculator.run()
        energy = float(np.asarray(result['energy'],
                                  dtype=np.float64).reshape(-1)[0])
        gradient = result.get('gradient')
        if gradient is None:
            raise RuntimeError('the evaluation returned no gradient')
        gradient = np.asarray(gradient, dtype=np.float64)
        if gradient.ndim == 3:
            gradient = gradient[0]
        if gradient.shape != self.coordinates.shape:
            raise RuntimeError(
                'gradient shape %r does not match the %d-atom geometry'
                % (gradient.shape, len(self.coordinates)))
        reading = {
            'energy': energy,
            'gradient': gradient,
            'components': {},
        }
        components = result.get('qmmm_components') or {}
        for key in COMPONENTS:
            value = components.get(key)
            if value is not None:
                reading['components'][key] = float(value)
        qm_energy = components.get('qm_energy')
        if qm_energy is not None:
            reading['components']['qm_energy'] = float(
                np.asarray(qm_energy, dtype=np.float64).reshape(-1)[0])
        return reading


def price(source, delta, dofs, tolerance, keep):
    """Difference the energy along each degree of freedom and judge the result."""
    workdir = tempfile.mkdtemp(prefix='fd_composite_')
    try:
        with working_directory(workdir):
            engine = Engine(source, workdir)
            base = engine.evaluate(engine.coordinates)
            rows = []
            for atom, axis in dofs:
                if atom >= len(engine.coordinates):
                    raise SystemExit(
                        'degree of freedom (%d, %d) is outside the '
                        '%d-atom system'
                        % (atom, axis, len(engine.coordinates)))
                plus = engine.coordinates.copy()
                minus = engine.coordinates.copy()
                plus[atom][axis] += delta
                minus[atom][axis] -= delta
                forwarded = engine.evaluate(plus)
                backward = engine.evaluate(minus)
                slope = ((forwarded['energy'] - backward['energy'])
                         / (2.0 * delta) * ANGSTROM_TO_BOHR)
                reported = float(base['gradient'][atom][axis])
                terms = {}
                for key in forwarded['components']:
                    if key not in backward['components']:
                        continue
                    terms[key] = (
                        (forwarded['components'][key]
                         - backward['components'][key])
                        / (2.0 * delta) * ANGSTROM_TO_BOHR)
                rows.append({
                    'atom': atom,
                    'axis': axis,
                    'slope': slope,
                    'reported': reported,
                    'residual': slope + reported,
                    'terms': terms,
                })
            return {'source': source, 'delta': delta,
                    'tolerance': tolerance, 'base': base, 'rows': rows}
    finally:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)


def report(readings):
    """Print the readings and return (verdict, reasons)."""
    print('example            : %s' % readings['source'])
    print('displacement       : %.1e Angstrom' % readings['delta'])
    print('E (Hartree)        : %.10f' % readings['base']['energy'])
    print('|gradient| (Ha/Bohr): %.6f' % float(
        np.sqrt(np.sum(readings['base']['gradient'] ** 2))))
    print()
    print('%-10s %-15s %-15s %-12s' %
          ('dof', 'dE/dr', '-F', '|diff|'))
    reasons = []
    for row in readings['rows']:
        label = '%d %s' % (row['atom'] + 1, 'xyz'[row['axis']])
        residual = abs(row['residual'])
        print('%-10s %-15.6e %-15.6e %-12.3e'
              % (label, row['slope'], -row['reported'], residual))
        if residual > readings['tolerance']:
            reasons.append(
                '%s: the energy slope %.6e and the reported -F %.6e differ '
                'by %.3e, above %.1e'
                % (label, row['slope'], -row['reported'], residual,
                   readings['tolerance']))
        for key, value in sorted(row['terms'].items()):
            print('           %-10s d/dr=% .6e' % (key, value))
    return (not reasons), reasons


def _parse_dofs(values):
    """`--dof 9,2` entries, as (atom, axis) with a 0-based atom index."""
    dofs = []
    for value in values:
        parts = value.replace(',', ' ').split()
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                'a degree of freedom is "atom,axis": %r' % value)
        atom, axis = int(parts[0]) - 1, int(parts[1])
        if axis not in (0, 1, 2):
            raise argparse.ArgumentTypeError(
                'axis must be 0, 1 or 2: %r' % value)
        dofs.append((atom, axis))
    return dofs


def main(argv):
    parser = argparse.ArgumentParser(
        description='Difference a composite surface\'s energy against its '
                    'reported gradient.')
    parser.add_argument('source', nargs='?', default=DEFAULT_EXAMPLE,
                        help='shipped example name or path to an input yaml')
    parser.add_argument('--delta', type=float, default=DEFAULT_DELTA,
                        help='displacement in Angstrom (default %.3f)'
                             % DEFAULT_DELTA)
    parser.add_argument('--dof', action='append', metavar='ATOM,AXIS',
                        help='1-based atom and 0-based axis; repeatable '
                             '(default: the first center and two controls)')
    parser.add_argument('--tol', type=float, default=DEFAULT_TOLERANCE,
                        help='acceptance bound in Ha/Bohr (default %.1e)'
                             % DEFAULT_TOLERANCE)
    parser.add_argument('--keep', action='store_true',
                        help='keep the scratch directory')
    args = parser.parse_args(argv)
    dofs = _parse_dofs(args.dof) if args.dof else list(DEFAULT_DOFS)
    readings = price(args.source, args.delta, dofs, args.tol, args.keep)
    passed, reasons = report(readings)
    print()
    if passed:
        print('VERDICT: the composite gradient is the derivative of the '
              'energy the same evaluation reports.')
        return 0
    print('VERDICT: FAILED')
    for reason in reasons:
        print('  - %s' % reason)
    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))