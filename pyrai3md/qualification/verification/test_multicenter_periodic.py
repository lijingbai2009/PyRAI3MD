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


class _CollectingLogger:
    """The outlet a finding reaches: one line per report, kept for the test."""

    def __init__(self):
        self.lines = []

    def write(self, line):
        self.lines.append(line)


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


class TestOutermostLayerCarriesTheBox(unittest.TestCase):
    """The capability gate reads the outer method, not every method."""

    def test_a_molecular_center_under_a_periodic_environment_loads(self):
        validated = _load(_config(['pyscf', 'lammps']))
        self.assertIsNotNone(validated.dynamics or validated.system)

    def test_the_same_methods_inverted_are_refused_by_name(self):
        # lammps at the centers would be fine; pyscf as the whole-system
        # provider is not, and the message has to say which layer failed.
        with self.assertRaises(ValueError) as caught:
            _load(_config(['lammps', 'pyscf']))
        message = str(caught.exception)
        self.assertIn('whole-system method', message)
        self.assertIn('outermost layer', message)
        self.assertIn('pyscf', message)

    def test_a_single_center_periodic_molecular_run_is_still_refused(self):
        # The rule generalizes rather than narrows: with one method that
        # method is the outermost, so the old refusal stands.
        config = _config(['pyscf'], n_center=1)
        config['system']['highlevel'] = [[1, 2, 3]]
        config['system']['coordinates'] = [list(_COORDS[0]),
                                           list(_COORDS[1]),
                                           list(_COORDS[2])]
        config['system']['atoms'] = ['O', 'H', 'H']
        with self.assertRaises(ValueError) as caught:
            _load(config)
        self.assertIn('whole-system method', str(caught.exception))

    def test_the_pre_m7_refusal_message_is_gone(self):
        # The refusal used to say multi-center does not support pbc at all.
        # Nothing on this path may still say that, whatever else it says.
        try:
            _load(_config(['pyscf', 'lammps']))
        except ValueError as error:
            self.assertNotIn('multi-center', str(error))
            raise


class TestCenterTaskIsAMolecularSubsystem(unittest.TestCase):
    """A center is handed a connected cluster, no cell and no periodicity."""

    @classmethod
    def setUpClass(cls):
        cls.config = _load(_config(['pyscf', 'lammps']))
        cls.state = build_state(cls.config)
        cls.tasks = _tasks(cls.state)

    def _centers(self):
        return [task for task in self.tasks
                if task.get('region') != 'all' and task.get('center_index')
                is not None]

    def test_the_queue_holds_both_a_whole_system_and_the_centers(self):
        self.assertTrue(self._centers(), 'no center task was planned')
        whole = [task for task in self.tasks if task.get('region') == 'real'
                 and task.get('center_index') is None]
        self.assertTrue(whole, 'no whole-system baseline task was planned')

    def test_a_center_config_declares_no_periodicity(self):
        for task in self._centers():
            with self.subTest(region=task.get('region_indices')):
                skill = task['config']['skill_config']
                self.assertEqual([0, 0, 0], [int(flag) for flag in
                                             skill.get('pbc', [0, 0, 0])])
                self.assertFalse(
                    any(float(value) for row in
                        (skill.get('lattice_vector') or []) for value in row),
                    'a center subsystem must not inherit the box')

    def test_the_whole_system_config_keeps_the_box(self):
        whole = [task for task in self.tasks
                 if task.get('region') == 'real'
                 and task.get('center_index') is None][0]
        skill = whole['config']['skill_config']
        self.assertTrue([int(flag) for flag in skill.get('pbc', [])],
                        'the outer layer must see the periodicity')
        self.assertEqual(3, np.asarray(skill.get('lattice_vector'),
                                       dtype=np.float64).shape[0])

    def test_a_torn_center_is_reassembled_before_the_backend_sees_it(self):
        """The sliced cluster's bond is the bond, not the box.

        Center 2 of this fixture is the water torn by the z face: its wrapped
        O-H distance is ~8.4 Angstrom and its real one is 0.96. The static
        region geometry must carry the second number, which is what
        `continuous_view` is for; the raw wrapped slice is shown alongside so
        the test cannot pass by accident on a geometry that never straddled.
        """
        centers = [task for task in self._centers()
                   if 4 in (task.get('region_indices') or [])]
        self.assertTrue(centers, 'the straddling center was not planned')
        task = next(task for task in centers
                    if task.get('skill_name') == 'pyscf_skill')
        indices = list(task['region_indices'])
        skill = task['config']['skill_config']
        sliced = np.asarray(skill['coordinates'], dtype=np.float64)
        self.assertEqual(len(indices), sliced.shape[0])
        # oxygen first, then its two hydrogens, in region order
        rebuilt = float(np.linalg.norm(sliced[0] - sliced[1]))
        self.assertLess(rebuilt, 1.05,
                        'the center was handed a torn molecule: O-H reads %.3f'
                        % rebuilt)
        wrapped = np.asarray(_COORDS, dtype=np.float64)[indices]
        raw = float(np.linalg.norm(wrapped[0] - wrapped[1]))
        self.assertGreater(raw, 6.0,
                           'the fixture stopped straddling the face (%.3f)'
                           % raw)

    def test_the_identity_view_is_untouched_for_a_non_periodic_run(self):
        config = _load(_config(['pyscf', 'lammps'], pbc=(0, 0, 0)))
        state = build_state(config)
        centers = [task for task in _tasks(state)
                   if task.get('center_index') is not None
                   and task.get('region') != 'all']
        for task in centers:
            sliced = np.asarray(task['config']['skill_config']['coordinates'],
                                dtype=np.float64)
            direct = np.asarray(_COORDS, dtype=np.float64)[
                list(task['region_indices'])]
            np.testing.assert_allclose(sliced, direct)


class TestTheRegionTaskIsHandedItsRows(unittest.TestCase):
    """A LAMMPS region task is handed the topology rows of its region.

    The user ships one system-level `{project}.data`; the framework's whole
    contribution to a subsystem is the list of rows it computes, and that list
    is ascending because a center is a SET of atoms, so
    `schema/adapter.py::_parse_center_indices` canonicalises the declaration
    (dev/skills/LAMMPS_SKILL_DEVELOPMENT.md, `atom_selection`). Ascending
    matters downstream: the skill's per-step writer puts coordinate row `i` on
    the row numbered `i + 1`, so the row numbers it is handed have to be the
    order the geometry was sliced in.
    """

    # Declared out of ascending order on purpose: this is what the
    # canonicalisation has to absorb, and what the two input carriers have to
    # agree on.
    _SHUFFLED = [[3, 1, 2], [6, 4, 5]]
    _ASCENDING = [[1, 2, 3], [4, 5, 6]]

    @classmethod
    def setUpClass(cls):
        cls.config = _load(_config(['pyscf', 'lammps'],
                                   highlevel=cls._SHUFFLED))
        cls.state = build_state(cls.config)
        cls.tasks = _tasks(cls.state)

    def _lammps_tasks(self):
        return [task for task in self.tasks
                if task.get('skill_name') == 'lammps_skill']

    def _sliced(self):
        # The tasks a region was cut for: L(real) carries the region label
        # 'real' with no indices, and it is the one that must NOT be handed a
        # selection.
        return [task for task in self._lammps_tasks()
                if task.get('region_indices')]

    def test_a_declared_center_is_canonicalised_to_ascending_rows(self):
        tasks = self._sliced()
        self.assertTrue(tasks, 'no low-layer center task was planned')
        for task in tasks:
            indices = list(task['region_indices'])
            selection = task['config']['skill_config']['atom_selection']
            self.assertEqual([index + 1 for index in indices], selection)
            self.assertEqual(sorted(indices), indices,
                             'the plan kept a declared order')
        self.assertEqual([[0, 1, 2], [3, 4, 5]],
                         [list(task['region_indices']) for task in tasks],
                         'the shuffled declaration did not come out ascending')

    def test_every_input_carrier_describes_the_same_centers(self):
        # The nested yaml list, the classic string and the ascending list all
        # go through one canonicalisation, so the plan cannot depend on which
        # spelling a user happened to write.
        carriers = {
            'shuffled': _config(['pyscf', 'lammps'],
                                highlevel=self._SHUFFLED),
            'ascending': _config(['pyscf', 'lammps'],
                                 highlevel=self._ASCENDING),
            'string': _config(['pyscf', 'lammps'],
                              highlevel='3 1 2; 6 4 5'),
        }
        plans = {}
        for label, raw in carriers.items():
            state = build_state(validate(normalize(build_config(raw))))
            plans[label] = [tuple(task['region_indices'])
                            for task in _tasks(state)
                            if task.get('region_indices')]
        self.assertEqual(plans['shuffled'], plans['ascending'])
        self.assertEqual(plans['shuffled'], plans['string'])
        self.assertEqual([(0, 1, 2), (0, 1, 2), (3, 4, 5), (3, 4, 5)],
                         plans['shuffled'])

    def test_the_whole_system_task_is_handed_no_rows(self):
        whole = [task for task in self._lammps_tasks()
                 if task.get('region') == 'all'
                 or not (task.get('region_indices') or [])]
        self.assertTrue(whole, 'no whole-system LAMMPS task was planned')
        for task in whole:
            # Nothing written is the whole system: the skill's own default is
            # the empty selection, so L(real) keeps the path it always had.
            self.assertNotIn('atom_selection',
                             task['config']['skill_config'],
                             'L(real) must keep the whole-system path')

    def test_a_skill_without_the_field_is_handed_nothing(self):
        # The declared-key check is what keeps a QM task's config at the
        # shape it always had: a field the schema never lists must not
        # appear because the framework wrote it.
        qm = [task for task in self.tasks
              if task.get('skill_name') == 'pyscf_skill']
        self.assertTrue(qm)
        for task in qm:
            self.assertNotIn('atom_selection', task['config']['skill_config'])

    def test_the_rows_and_the_handed_geometry_describe_one_cluster(self):
        # The subsystem the skill cuts is the subsystem the runtime slices:
        # row `i` of the handed geometry is row `selection[i] - 1` of the
        # whole-system continuous view. That is the contract the skill's
        # writer relies on when it puts coordinate row i on atom id i + 1, so
        # it is checked through the selection rather than the indices.
        for task in self._sliced():
            indices = list(task['region_indices'])
            skill = task['config']['skill_config']
            selection = list(skill['atom_selection'])
            self.assertEqual(len(selection), len(indices))
            self.assertEqual(skill['natom'], len(indices))
            view = np.asarray(
                continuous_view(_COORDS, self.config.system.image, _BOX),
                dtype=np.float64)
            np.testing.assert_allclose(
                np.asarray(skill['coordinates'], dtype=np.float64),
                view[[row - 1 for row in selection]], atol=1e-8)


    def test_a_three_layer_plan_hands_the_mid_shell_rows_too(self):
        """`L(I_c)` is a subsystem of the same kind, one shell wider.

        The mid-layer shell of a three-layer plan is an index set like any
        other, so the same channel serves it; only the low layer can be cut
        from a topology template, which is what the declared-key check keeps
        straight. Certified at plan level: this machine has no engine that can
        serve as a middle layer, so no energy number is claimed here.
        """
        raw = {
            'control': {'title': 't_three_layer_rows', 'jobtype': 'sp',
                        'theory': ['pyscf', 'xtb', 'lammps']},
            'system': {
                'atoms': ['O', 'H', 'H', 'O', 'H', 'H', 'C', 'H', 'H'],
                'coordinates': [[1.0, 0.0, 0.117], [1.0, 0.757, -0.469],
                                [1.0, -0.757, -0.469], [4.0, 0.0, 0.500],
                                [4.0, 0.757, 8.913], [4.0, -0.757, 8.913],
                                [2.0, 2.0, 2.0], [3.0, 2.0, 2.0],
                                [4.0, 2.0, 2.0]],
                'pbc': [1, 1, 1],
                'cell': [9.0, 9.0, 9.0, 90.0, 90.0, 90.0],
                'highlevel': [[1, 2, 3], [4, 5, 6]],
                'midlevel': [[7], [8, 9]],
                'charges': 0, 'multiplicity': 1, 'nstate': 1},
            'pyscf': {'method': 'hf', 'basis': 'sto3g'},
            'xtb': {'method': 'gfn2'},
            'lammps': {'method': 'ff', 'lammps_project': 'h2o'},
        }
        state = build_state(validate(normalize(build_config(raw))))
        tasks = _tasks(state)
        # 3-layer = 1 + 4C
        self.assertEqual(9, len(tasks))
        inter = [task for task in tasks
                 if task.get('region') == 'inter'
                 and task.get('skill_name') == 'lammps_skill']
        self.assertEqual(2, len(inter), 'no low-layer mid-shell task')
        for task in inter:
            indices = list(task['region_indices'])
            self.assertEqual([index + 1 for index in indices],
                             task['config']['skill_config']['atom_selection'])
        for task in tasks:
            if task.get('skill_name') != 'lammps_skill':
                self.assertNotIn('atom_selection',
                                 task['config']['skill_config'])


class TestGradientScatterIsFrameBlind(unittest.TestCase):
    """A per-atom lattice shift leaves the scatter-add mapping exact.

    The region was translated by an integer number of cell vectors per atom,
    which is constant during one evaluation, so its derivative w.r.t. the
    global coordinates is the identity and the local rows go back to the
    global slots by index with no transform. This test pins that claim on the
    reconstruction the center above actually received.
    """

    def test_scatter_by_index_returns_to_the_global_slots(self):
        from pyrai3md.workflow.multiscale.regions import project_gradient

        indices = [3, 4, 5]
        local = np.arange(3 * 3, dtype=np.float64).reshape(1, 3, 3)
        assembled = project_gradient(local, indices, 6)
        np.testing.assert_allclose(assembled[0, indices, :], local[0])
        self.assertEqual(0.0, float(np.abs(assembled[:, [0, 1, 2], :]).max()))

    def test_unwrapping_is_a_translation_only(self):
        shifted = continuous_view(_COORDS, [[0, 0, 0]] * 6,
                                  _BOX)
        # A row of counts of (0,0,1) is exactly one lattice vector added.
        moved = continuous_view(_COORDS, [[0, 0, 1]] * 6, _BOX)
        np.testing.assert_allclose(shifted + np.asarray(_BOX)[2], moved,
                                   rtol=0, atol=1e-12)


class TestTheLivePathAgreesWithTheStaticConfig(unittest.TestCase):
    """What the backend is called with must be what the config promised.

    The static `skill_config` and the runtime slice are two different code
    paths into the same geometry, and only the second one is what the engine
    evaluates. This drives the real work-item assembly with a recording
    backend, so a center that received its cluster unwrapped in the config but
    still torn at call time cannot pass.
    """

    @classmethod
    def setUpClass(cls):
        from pyrai3md.workflow.calculator import Calculator

        cls.state = build_state(_load(_config(['pyscf', 'lammps'])))

        class _Recorder:
            """Backend double that keeps every call it was asked to make."""

            name = 'recorder'

            def __init__(self, config=None, execution_id=''):
                del config, execution_id
                self.calls = []

            def calculate(self, atoms, coordinates, point_charges=None,
                          target_states=None, cell=None):
                self.calls.append({'natoms': len(atoms),
                                   'coordinates': np.asarray(
                                       coordinates, dtype=np.float64),
                                   'cell': cell})
                natom = len(atoms)
                return {'energy': [-76.0],
                        'gradient': [[[0.0] * 3] * natom]}

        recorder = _Recorder()
        calculator = Calculator(state=cls.state, dummy_backend=recorder)
        calculator.initialize()
        calculator.run()
        cls.calls = recorder.calls

    def test_every_center_call_got_a_connected_cluster(self):
        centers = [call for call in self.calls if call['natoms'] == 3]
        self.assertTrue(centers, 'no center-sized call reached a backend')
        for call in centers:
            distance = float(np.linalg.norm(call['coordinates'][0]
                                            - call['coordinates'][1]))
            self.assertLess(distance, 1.05,
                            'a center was evaluated on a torn molecule: %.3f'
                            % distance)

    def test_only_the_whole_system_call_is_given_the_cell(self):
        for call in self.calls:
            if call['natoms'] == 3:
                self.assertIsNone(call['cell'],
                                  'a region call was handed the global cell')
            elif call['natoms'] == 6:
                self.assertIsNotNone(call['cell'],
                                     'the whole-system call got no cell')


class TestTheRegionWatchCertifiesARecordedFrame(unittest.TestCase):
    """What the frame is asked about the region it implies, and what costs.

    A periodic run cuts each center out of the continuous view rebuilt from
    this frame's coordinates and crossing counts. Counts kept in fractional
    units are exact for the cell they were accrued under, so a variable-cell
    run is the case that has to be watched rather than argued about: the way a
    wrong image shows up is not an exception but a center handed to its backend
    as fragments a box length apart. These readings tell the difference, taken
    on the frame the run itself would record: a center whose counts reconnect it,
    a count off by one cell, a cell halved under the same counts, and a frame
    carrying no counts at all.
    """

    def _watched(self):
        """A state the watch can report to: runtime record, log line, frame."""
        state = build_state(_load(_config(['pyscf', 'lammps'])))
        state.runtime = RuntimeState(current_step=1)
        state.logger = _CollectingLogger()
        return state, state._build_history_frame()

    def test_a_center_whose_counts_reconnect_it_says_nothing(self):
        state, frame = self._watched()
        # The fixture's second water straddles the z face and the seed gave it
        # the count that closes the bond; that is the configuration the watch
        # must be able to look at and stay quiet about.
        self.assertTrue(any(any(int(value) for value in row)
                            for row in frame['system']['image']),
                        'the fixture lost its crossing counts')
        check_recorded_step(state, frame)
        self.assertNotIn(REGION_RECONSTRUCTION, state.runtime.diagnostics)
        self.assertEqual([], state.logger.lines)

    def test_a_count_off_by_one_cell_is_a_finding(self):
        state, frame = self._watched()
        image = [list(row) for row in frame['system']['image']]
        image[4] = [value + 1 for value in image[4]]
        frame['system']['image'] = image
        check_recorded_step(state, frame)
        entry = state.runtime.diagnostics[REGION_RECONSTRUCTION]
        self.assertEqual(1, entry['count'])
        self.assertFalse(entry['aborted'], 'a warn-only run keeps integrating')
        line = ' '.join(state.logger.lines)
        self.assertIn('center 2', line, 'the finding has to name the center')
        self.assertIn('4 and 5', line, 'and the two atoms it measured')

    def test_a_deforming_cell_is_the_case_the_watch_exists_for(self):
        # Half the box, the same counts: the pair that was 0.958 A across the
        # face is now 3.99 A, past half of the shortest extent of the cell it
        # is read against. This is the negative statement of section 4.1 of
        # dev/methods/PBC_DEVELOPMENT.md measured rather than assumed, so the
        # bound is not a threshold the check was tuned to sit under.
        state, frame = self._watched()
        frame['system']['lattice_vector'] = [
            [value * 0.5 for value in row]
            for row in frame['system']['lattice_vector']]
        check_recorded_step(state, frame)
        self.assertIn(REGION_RECONSTRUCTION, state.runtime.diagnostics)

    def test_a_torn_center_with_no_counts_at_all_is_a_finding(self):
        # An empty image field means "nothing crossed", which is what a
        # structure whose perception refused is left with too. The cut then
        # takes the identity path, and the straddling bond is as torn there as
        # it ever was, so the watch is not allowed to read absence as consent.
        state, frame = self._watched()
        frame['system']['image'] = []
        check_recorded_step(state, frame)
        entry = state.runtime.diagnostics[REGION_RECONSTRUCTION]
        self.assertEqual(1, entry['count'])
        raw = float(np.linalg.norm(np.asarray(_COORDS[3], dtype=np.float64)
                                   - np.asarray(_COORDS[4], dtype=np.float64)))
        self.assertIn('%.3f A apart, at or beyond' % raw,
                      ' '.join(state.logger.lines),
                      'the finding must quote the wrapped reading it saw')

    def test_a_debug_run_stops_on_the_same_finding(self):
        state, frame = self._watched()
        state.pmd.control.debug = True
        image = [list(row) for row in frame['system']['image']]
        image[4] = [value + 1 for value in image[4]]
        frame['system']['image'] = image
        with self.assertRaises(RecordedStepError) as caught:
            check_recorded_step(state, frame)
        self.assertIn('torn cluster', str(caught.exception))

    def test_the_frames_that_cut_no_region_are_asked_nothing(self):
        # Non-periodic: nothing was rebuilt. A run with one center: the whole
        # system goes to the backend as it stands, so a tear is not something
        # the reconstruction introduced. Both must leave no entry behind.
        state = build_state(_load(_config(['pyscf', 'lammps'],
                                          pbc=(0, 0, 0))))
        state.runtime = RuntimeState(current_step=1)
        state.logger = _CollectingLogger()
        frame = state._build_history_frame()
        check_recorded_step(state, frame)
        self.assertNotIn(REGION_RECONSTRUCTION, state.runtime.diagnostics)

        single = build_state(_load(_config(['pyscf', 'lammps'], n_center=1)))
        single.runtime = RuntimeState(current_step=1)
        single.logger = _CollectingLogger()
        check_recorded_step(single, single._build_history_frame())
        self.assertNotIn(REGION_RECONSTRUCTION, single.runtime.diagnostics)




class TestTheShippedPeriodicMultiCenterExample(unittest.TestCase):
    """What the two `multicenter_pbc_*` inputs promise, checked on the files.

    The example is the certification a real engine provides, so its shape is
    worth pinning here: an edited input could quietly stop being the case the
    document points at (the two centers no longer being disjoint subsets of the
    topology they name, that topology no longer shipped, the box landing on the
    center instead of the outer layer) and the LAMMPS run would then fail for a
    reason that has nothing to do with what it was written to demonstrate.
    """

    NAMES = ('multicenter_pbc_nve.yaml', 'multicenter_pbc_npt.yaml')

    def _load_example(self, name):
        import os

        from pyrai3md.interface.parser import parse_input
        examples = os.path.abspath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'examples'))
        return parse_input(os.path.join(examples, name))

    def test_two_disjoint_centers_are_cut_from_one_system_topology(self):
        """What the two inputs demonstrate, checked on the files.

        Both centers are proper subsets of one 60-atom topology, they share no
        atom, and the low-layer task of each is handed its region's rows in the
        order the plan declared them. That is the shape the `atom_selection`
        channel exists for: the earlier one-water shape (a center covering the
        whole system, so that a single topology trivially served both the
        baseline and the correction) demonstrated nothing about the cut. The
        declared-order invariant itself is pinned in
        `TestTheRegionTaskIsHandedItsRows` and in
        `skills/lammps_skill/tests/test_ff_model.py::TestSelection`, because the
        classic string form of a center sorts inside a group and a shipped input
        has to read the same in both modes.
        """
        for name in self.NAMES:
            with self.subTest(example=name):
                raw = self._load_example(name)
                state = build_state(validate(normalize(build_config(raw))))
                natom = len(state.pmd.system.atoms)
                self.assertEqual(60, natom,
                                 'the box the topology describes moved')
                centers = [task for task in _tasks(state)
                           if task.get('center_index') is not None
                           and task.get('region') != 'all']
                self.assertTrue(centers, 'no center was planned')
                selected = {}
                for task in centers:
                    skill = task['config']['skill_config']
                    if task.get('skill_name') != 'lammps_skill':
                        continue
                    indices = list(task['region_indices'])
                    self.assertEqual([index + 1 for index in indices],
                                     skill['atom_selection'])
                    self.assertEqual(3, len(indices),
                                     'a center stopped being one water')
                    self.assertLess(len(indices), natom,
                                    'the center covers the system again')
                    selected[task['center_index']] = list(
                        skill['atom_selection'])
                self.assertEqual({0, 1}, set(selected),
                                 'the plan no longer has two centers')
                first, second = selected[0], selected[1]
                self.assertFalse(set(first) & set(second),
                                 'the two centers share a topology row')
                self.assertEqual([10, 11, 12], first)
                self.assertEqual([22, 23, 24], second)

    def test_the_outer_task_keeps_the_box_and_the_center_gives_it_up(self):
        for name in self.NAMES:
            with self.subTest(example=name):
                state = build_state(validate(normalize(
                    build_config(self._load_example(name)))))
                whole = [task for task in _tasks(state)
                         if task.get('region') == 'real']
                centers = [task for task in _tasks(state)
                           if task.get('center_index') is not None
                           and task.get('region') != 'all']
                self.assertTrue(whole and centers)
                for task in centers:
                    skill = task['config']['skill_config']
                    self.assertFalse(
                        any(int(flag) for flag in (skill.get('pbc') or [])),
                        'a center inherited the periodicity')
                for task in whole:
                    self.assertTrue(
                        any(int(flag) for flag in
                            (task['config']['skill_config'].get('pbc') or [])),
                        'the outermost layer lost the box')

    def test_the_topology_each_input_names_is_shipped_beside_it(self):
        import os

        examples = os.path.abspath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'examples'))
        for name in self.NAMES:
            with self.subTest(example=name):
                raw = self._load_example(name)
                project = (raw.get('lammps') or {}).get('lammps_project')
                self.assertTrue(project,
                                '%s leaves the LAMMPS project to an '
                                'environment default, so it cannot be run by '
                                'one command from the examples directory'
                                % name)
                for suffix in ('.data', '.in.init', '.in.settings'):
                    self.assertTrue(
                        os.path.exists(os.path.join(examples,
                                                    project + suffix)),
                        'the topology %s%s is not shipped beside the input'
                        % (project, suffix))





if __name__ == '__main__':
    unittest.main(verbosity=2)
