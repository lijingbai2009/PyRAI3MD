"""
======================================================================
Contract tests for a multi-center run inside a periodic box (M7a of the
MD closure, dev/methods/PBC_DEVELOPMENT.md section 4.2,
dev/MULTICENTER_DEVELOPMENT.md section 5.2).

The ruling this suite encodes: multi-center means more than one method in
`theory`/`reference` and does not fix which methods may be combined; the one
capability requirement is that the OUTERMOST layer - the task that owns the
whole-system geometry, `planning/task.py::resolve_system_task` - can carry a
box. A center is a molecular calculation on a reconstructed cluster, so a
center method that declares `supports_pbc: false` is legitimate, and the same
method refusing a box as the outermost layer is the point of the gate.

Nothing here executes an engine: the claims are about the configuration, the
task queue and the geometry a center backend is handed, and they are checked
with the real skill manifests so the capability rule is tested against what
the registry actually declares.

Two boxes are used deliberately. `theory: [pyscf, lammps]` puts a molecular
method at the centers and a periodic method outside, which is the combination
the ruling opens. `theory: [lammps, pyscf]` inverts it, and the refusal must
name the outermost rule rather than the mere presence of a molecular backend.

Oct 08 2026
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

import os
import time
import unittest

import numpy as np

from pyrai3md.data.build import build_state
from pyrai3md.data.runtime.state import RuntimeState
from pyrai3md.schema import normalize, validate
from pyrai3md.schema.adapter import build_config
from pyrai3md.utilities.lattice import continuous_view
from pyrai3md.workflow.records.diagnostics import (REGION_RECONSTRUCTION,
                                                   RecordedStepError,
                                                   check_recorded_step)

# Two waters in a 9 Angstrom cube. The second one is written straddling the
# bottom face: its oxygen is at z = 0.50 and both hydrogens at z = 8.9133, so
# the main-image O-H distance reads 8.4 Angstrom while the bond, across the
# face, is the 0.958 Angstrom of the first water. A box nine times the bond is
# what lets the perception derive crossing counts instead of refusing.
_BOX = [[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]]
_ATOMS = ['O', 'H', 'H', 'O', 'H', 'H']
_COORDS = [[1.00000, 0.00000, 0.11734],
           [1.00000, 0.75754, -0.46936],
           [1.00000, -0.75754, -0.46936],
           [4.00000, 0.00000, 0.50000],
           [4.00000, 0.75754, 8.91330],
           [4.00000, -0.75754, 8.91330]]


def _config(theory, n_center=2, pbc=(1, 1, 1), highlevel=None):
    """A sectioned input for `n_center` molecular centers in a periodic box."""
    highlevel = ([[1, 2, 3], [4, 5, 6]][:n_center] if highlevel is None
                 else highlevel)
    return {
        'control': {'title': 't_multicenter_periodic', 'jobtype': 'sp',
                    'theory': list(theory)},
        'system': {'atoms': list(_ATOMS),
                   'coordinates': [list(row) for row in _COORDS],
                   'pbc': list(pbc), 'lattice_vector': [list(row)
                                                        for row in _BOX],
                   'cell': [9.0, 9.0, 9.0, 90.0, 90.0, 90.0],
                   'highlevel': highlevel,
                   'charges': 0, 'multiplicity': 1,
                   'nstate': 1},
        'pyscf': {'method': 'hf', 'basis': 'sto3g'},
        'lammps': {'force_field': 'tip3p'},
    }


def _load(config):
    return validate(normalize(build_config(config)))




def _tasks(state):
    return [task for _identity, task in state.iter_tasks()]


# The sections a LAMMPS `write_data` file can open, as the word before the
# optional `# <style>` comment.
_DATA_SECTIONS = ('Masses', 'Atoms', 'Bonds', 'Angles', 'Dihedrals',
                  'Impropers', 'Velocities', 'Coordinates', 'Forces',
                  'Pairs', 'Special Bonds', 'Fixes', 'Bonds Global',
                  'Angles Global', 'Dihedrals Global', 'Impropers Global',
                  'Pair Coeffs', 'Bond Coeffs', 'Angle Coeffs',
                  'Dihedral Coeffs', 'Improper Coeffs', 'Groups',
                  'Molecules', 'Restart')
_ELEMENTS = (('C', 12.011), ('H', 1.008), ('O', 15.999))


def _element_of(mass):
    """The one element a force-field mass stands for, or None."""
    for label, reference in _ELEMENTS:
        if abs(mass - reference) < 0.02:
            return label
    return None


def _read_lammps_data(path):
    """The rows, types, charges and box of one LAMMPS data file.

    Read the way this package's own writer demands them
    (`skills/lammps_skill/service/builder/input_builder.py:948-996`: atom ids
    must map onto 1..natom, and coordinate row `i` belongs to id `i + 1`), so
    what a criterion compares is what a run would use - not a second opinion
    about the same file.
    """
    import os
    import re

    with open(path, encoding='utf-8') as handle:
        lines = handle.read().splitlines()
    starts = [index for index, line in enumerate(lines)
              if line.split('#')[0].strip() in _DATA_SECTIONS]
    blocks = {}
    for position, index in enumerate(starts):
        name = lines[index].split('#')[0].strip()
        stop = (starts[position + 1] if position + 1 < len(starts)
                else len(lines))
        blocks[name] = [row.split() for row in lines[index + 1:stop]
                        if row.strip()]
    masses = {int(row[0]): float(row[1]) for row in blocks.get('Masses', [])}
    rows = blocks.get('Atoms', [])
    atoms = {}
    for row in rows:
        atom_id = int(row[0])
        atoms[atom_id] = {'molecule': int(row[1]), 'type': int(row[2]),
                          'charge': float(row[3]),
                          'coordinates': [float(value) for value in row[4:7]]}
    assert sorted(atoms) == list(range(1, len(atoms) + 1)), (
        'the data file %s has atom ids outside 1..natom' % path)
    elements = []
    for atom_id in sorted(atoms):
        mass = masses[atoms[atom_id]['type']]
        label = _element_of(mass)
        if label is None:
            raise AssertionError(
                'row %d of %s carries mass %g, which this suite cannot name: '
                'the type dictionary is shared across the whole carrier family '
                'and holds elements beyond carbon, hydrogen and oxygen, so only '
                'the masses a row actually uses are read'
                % (atom_id, path, mass))
        elements.append(label)
    bounds, tilt = {}, (0.0, 0.0, 0.0)
    for line in lines:
        parts = line.split()
        if len(parts) == 4 and parts[2:] in (['xlo', 'xhi'], ['ylo', 'yhi'],
                                             ['zlo', 'zhi']):
            bounds[parts[2]], bounds[parts[3]] = (float(parts[0]),
                                                 float(parts[1]))
        elif len(parts) == 5 and parts[2:] == ['xy', 'xz', 'yz']:
            tilt = (float(parts[0]), float(parts[1]), float(parts[2]))
    lengths = [bounds['xhi'] - bounds['xlo'], bounds['yhi'] - bounds['ylo'],
               bounds['zhi'] - bounds['zlo']]
    cutoffs = []
    init_path = os.path.splitext(path)[0] + '.in.init'
    if os.path.isfile(init_path):
        with open(init_path, encoding='utf-8') as handle:
            for line in handle:
                parts = line.split()
                if len(parts) > 2 and parts[0] == 'pair_style':
                    cutoffs = [float(value) for value in parts[2:]
                               if re.match(r'^[0-9.]+$', value)]
    return {'atoms': atoms, 'elements': elements, 'masses': masses,
            'molecules': {atom_id: atoms[atom_id]['molecule']
                          for atom_id in atoms},
            'charges': [atoms[atom_id]['charge'] for atom_id in sorted(atoms)],
            'lengths': lengths, 'tilt': tilt,
            'cutoff': max(cutoffs) if cutoffs else 0.0}


def _lammps_pair_types(text):
    """The atom types a settings block gives coefficients of its own.

    Returns `(diagonal types, wildcard seen)`. This family's templates carry one
    `pair_coeff i i` row per type and no cross rows: `pair_modify mix arithmetic`
    in the init template is what turns those into the 4,851 cross interactions, so
    a type with no diagonal row has no interaction with anything.
    """
    types = set()
    wildcard = False
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == 'pair_coeff':
            if parts[1] == '*' or parts[2] == '*':
                wildcard = True
                continue
            first, second = int(parts[1]), int(parts[2])
            if first == second:
                types.add(first)
    return types, wildcard


















class TestTheForceFieldCarrierOfTheCrossingPair(unittest.TestCase):
    """What the two `optimize_cell_*_lammps` inputs promise, checked on the files.

    The pair is the first case in the shipped set whose box is carried by a
    force field that has a topology of its own, so its certification depends on a
    data file the input does not restate: 1581 rows, a 98-type dictionary, one
    ethylene written as the LAST molecule. Every one of those facts can be broken
    by an edit that leaves the run running - a centre selected at rows 1-6 is 24
    atoms more of water in PySCF and still converges, a box that no longer matches
    the topology's own edges relaxes a cell the engine never had - which is why
    they are checked here rather than trusted.
    """

    NAMES = ('optimize_cell_meci_lammps.yaml', 'optimize_cell_mecp_lammps.yaml')
    # The centre: molecule 526 of 526, rows 1576-1581 by atom id.
    CENTRE = [1576, 1577, 1578, 1579, 1580, 1581]

    def _examples(self):
        return os.path.abspath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'examples'))

    def _load_example(self, name):
        from pyrai3md.interface.parser import parse_input
        return parse_input(os.path.join(self._examples(), name))

    def _path(self, name):
        return os.path.join(self._examples(), name)

    def test_the_topology_rows_are_the_atoms_the_input_names(self):
        for name in self.NAMES:
            with self.subTest(example=name):
                raw = self._load_example(name)
                data = _read_lammps_data(self._path('c2h4_water25.data'))
                declared = list(raw['atoms'])
                self.assertEqual(len(declared), len(data['elements']),
                                 'the input and the topology count atoms apart')
                self.assertEqual(declared, data['elements'],
                                 'the input and the data file stop naming the '
                                 'same row somewhere in their 1581 atoms')
                # The non-identity witness: the centre is the tail of this
                # carrier, so reading it at the head - where every other gap
                # example puts it - names six water atoms, not six solute ones.
                self.assertEqual(['O', 'H', 'H'], declared[:3])
                self.assertNotEqual(declared[:6], ['C', 'C', 'H', 'H', 'H', 'H'])

    def test_the_declared_centre_is_one_whole_molecule_of_the_data(self):
        for name in self.NAMES:
            with self.subTest(example=name):
                raw = self._load_example(name)
                data = _read_lammps_data(self._path('c2h4_water25.data'))
                highlevel = list(raw['molecule']['highlevel'])
                self.assertEqual([self.CENTRE], [sorted(rows)
                                                 for rows in highlevel],
                                 'the centre is no longer the solute rows')
                self.assertEqual(['C', 'C', 'H', 'H', 'H', 'H'],
                                 [data['elements'][row - 1] for row in self.CENTRE])
                molecules = {data['molecules'][row] for row in self.CENTRE}
                self.assertEqual(len(molecules), 1,
                                 'the centre spans more than one molecule of the '
                                 'topology, so it is not a whole solute')

    def test_the_declared_box_is_the_box_of_the_topology(self):
        for name in self.NAMES:
            with self.subTest(example=name):
                raw = self._load_example(name)
                data = _read_lammps_data(self._path('c2h4_water25.data'))
                cell = [float(value) for value in raw['molecule']['cell']]
                self.assertEqual([round(value, 6) for value in data['lengths']],
                                 [round(value, 6) for value in cell[:3]],
                                 'the input relaxes a box the topology does not '
                                 'describe: %s against %s'
                                 % (cell[:3], data['lengths']))
                self.assertEqual([90.0, 90.0, 90.0], cell[3:])
                self.assertEqual(data['tilt'], (0.0, 0.0, 0.0),
                                 'the topology is triclinic while the input '
                                 'declares an orthorhombic cell')
                # The condition `optimize_cell_lammps.yaml` records for a
                # variable-cell force-field box: the pair cutoff has to stay
                # under the smallest half extent, or the energy is not smooth in
                # its own cell and the lattice gradient reads a discontinuity.
                self.assertGreater(min(data['lengths']) / 2.0, data['cutoff'],
                                   'the cell is smaller than its own cutoff')

    def test_the_charge_of_the_topology_is_the_charge_declared(self):
        for name in self.NAMES:
            with self.subTest(example=name):
                raw = self._load_example(name)
                data = _read_lammps_data(self._path('c2h4_water25.data'))
                self.assertAlmostEqual(sum(data['charges']),
                                       float(raw['molecule']['charges']),
                                       places=6,
                                       msg='the box is charged while the input '
                                           'says it is not')
                self.assertAlmostEqual(
                    sum(data['charges'][row - 1] for row in self.CENTRE), 0.0,
                    places=6, msg='the centre carries charge into the quantum '
                                  'region, whose own charge the input declares '
                                  'as zero')

    def test_the_shared_template_covers_every_type_the_carrier_uses(self):
        # The carrier's two template files are byte-identical to water25's, and
        # that is a claim with two halves: ffauto writes one 98-type dictionary
        # for the whole family, and its settings block carries one diagonal
        # `pair_coeff i i` row per type while `pair_modify mix arithmetic` in the
        # init template turns those into the cross interactions. The solute's
        # carbon and hydrogen rows are covered only while both halves hold, and
        # if they stop holding the engine fails mid-run with a message about
        # incomplete pair coefficients - so it is pinned here rather than left to
        # look like an accidental copy.
        carrier = _read_lammps_data(self._path('c2h4_water25.data'))
        water = _read_lammps_data(self._path('water25.data'))
        self.assertEqual(carrier['masses'], water['masses'],
                         'the two carriers no longer share one type dictionary')
        with open(self._path('c2h4_water25.in.settings'),
                  encoding='utf-8') as reader:
            typed, wildcard = _lammps_pair_types(reader.read())
        with open(self._path('c2h4_water25.in.init'),
                  encoding='utf-8') as reader:
            self.assertIn('pair_modify mix', reader.read(),
                          'the template has no mixing rule, so one diagonal row '
                          'per type leaves every cross interaction undefined')
        for label, handle in (('carrier', carrier), ('water25', water)):
            used = {atom['type'] for atom in handle['atoms'].values()}
            missing = [] if wildcard else sorted(used - typed)
            self.assertEqual(missing, [],
                             '%s uses types with no pair coefficients of their '
                             'own: %s' % (label, missing[:10]))

    def test_the_plan_hands_the_box_to_the_force_field_layer(self):
        for name in self.NAMES:
            with self.subTest(example=name):
                raw = self._load_example(name)
                state = build_state(validate(normalize(build_config(raw))))
                tasks = _tasks(state)
                whole = [task for task in tasks
                         if task.get('region') == 'real']
                self.assertEqual(len(whole), 1,
                                 'the whole-system layer is no longer one task')
                outer = whole[0]['config']['skill_config']
                self.assertEqual(whole[0].get('skill_name'), 'lammps_skill')
                self.assertTrue(any(int(flag) for flag in outer['pbc']),
                                'the box left the layer that carries it')
                self.assertTrue(outer.get('stress'),
                                'a variable-cell run whose outer layer returns '
                                'no stress has no lattice gradient to take')
                centres = [task for task in tasks
                           if task.get('center_index') is not None]
                self.assertTrue(centres)
                for task in centres:
                    skill = task['config']['skill_config']
                    if skill.get('natom') is not None:
                        self.assertEqual(skill['natom'], len(self.CENTRE),
                                         'the centre is not the six solute rows')
                selected = [task for task in tasks
                            if task.get('region_indices')
                            and task.get('skill_name') == 'lammps_skill']
                self.assertTrue(selected,
                                'no force-field layer was cut for the declared '
                                'centre')
                for task in selected:
                    self.assertEqual(
                        list(task['config']['skill_config']['atom_selection']),
                        self.CENTRE)


if __name__ == '__main__':
    unittest.main(verbosity=2)
