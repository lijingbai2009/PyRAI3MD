"""
======================================================================
Tests for the additive QM/MM (ESPF external field) composite.

The LAMMPS refactor (#432) split QM/MM into two standalone profiles:
`qmmm_field` (q_Q=+1 A/B Coulomb probe, atomic-unit V/E) and
`qmmm_energy` (full FF with q_qm, E_classic = U_main - U_gas - qmmm_el).
The framework plans `qmmm_field -> QM -> qmmm_energy`, feeds the Molcas
ESPF field, and stitches E = E_QM + E_classic.

Coverage:
- task expansion and role classification (field 0, QM +1, energy +1)
- assembler: energy, QM-row correction, MM rows, diagnostics
- native→framework conversion (energy Hartree, gradient = -F factor)
- `.esfp` external-field construction
- the three-stage calculator path with mock backends
- config injection and the LAMMPS inert mirror fields

Oct 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from pyrai3md.workflow.multiscale.composite import (
    TaskOutcome, assemble_qmmm,
)
from pyrai3md.workflow.records.conversion import (
    apply_result_units, convert_native_energy, convert_native_gradient,
    resolve_result_units,
)


NATOM = 6
QIDX = [0, 1, 2]


class _BackendA:
    pass


class _BackendB:
    pass


def _skills():
    return {
        'a': {'skill_name': 'molcas_skill', 'role': 'qm',
              'backend': _BackendA, 'capabilities': ['energy', 'gradient']},
        'b': {'skill_name': 'lammps_skill', 'role': 'qm2',
              'backend': _BackendB, 'capabilities': ['energy', 'gradient']},
    }


def _qmmm_tasks():
    return [
        {'task_id': 'field', 'role': 'field', 'qmmm_role': 'field',
         'contribution': 0, 'region': 'all', 'region_indices': []},
        {'task_id': 'qm', 'role': 'qm', 'qmmm_role': 'qm',
         'contribution': 1, 'region': 'model',
         'region_indices': list(QIDX)},
        {'task_id': 'energy', 'role': 'energy', 'qmmm_role': 'energy',
         'contribution': 1, 'region': 'all', 'region_indices': []},
    ]


def _outcome(task, raw, contribution=None, success=True):
    return TaskOutcome(
        task_id=task['task_id'],
        contribution=(task['contribution'] if contribution is None
                      else contribution),
        raw=raw,
        region_indices=list(task['region_indices']),
        region=task['region'],
        success=success,
    )


def _field_block():
    return {
        'qm_atoms': list(QIDX),
        'qm_atom_ids': [index + 1 for index in QIDX],
        'probe_charge': 1.0,
        'v_mm_native': np.array([-1.0, -2.0, -3.0]),
        'e_mm_native': np.zeros((3, 3)),
        'sanity': {'v_vs_u_gg': 0.0, 'e_vs_f_gg': 0.0},
        'units': {'energy': 'real', 'force': 'real'},
    }


def _legacy_field_block():
    """A field block from a producer that returns no native V/E columns."""
    block = _field_block()
    block.pop('v_mm_native', None)
    block.pop('e_mm_native', None)
    return block


def _field_raw():
    return {
        'energy': np.array([0.0]),
        'gradient': np.zeros((1, NATOM, 3)),
        'qmmm_field': _field_block(),
    }


def _energy_block():
    return {
        'qm_atoms': list(QIDX),
        'qm_atom_ids': [index + 1 for index in QIDX],
        'q_qm': [0.1, 0.2, -0.3],
        'u_main_native': -10.0,
        'u_gas_native': -4.0,
        'qmmm_el_native': -3.5,
        'e_classic_native': -2.5,
        'u_gas_gradient_native': np.array([[-1.0, 0.0, 0.0],
                                           [-2.0, 0.0, 0.0],
                                           [-3.0, 0.0, 0.0]]),
        'qmmm_el_gradient_native': np.array([[-0.5, 0.0, 0.0],
                                             [-0.5, 0.0, 0.0],
                                             [-0.5, 0.0, 0.0]]),
        'sanity': {'parts': 0.0},
        'units': {'energy': 'real', 'force': 'real'},
    }


def _energy_atomic():
    return {
        'e_classic': -2.5,
        'u_main': -10.0,
        'u_gas': -4.0,
        'qmmm_el': -3.5,
        'qmmm_el_gradient': np.array([[0.1, 0.0, 0.0],
                                      [0.1, 0.0, 0.0],
                                      [0.1, 0.0, 0.0]]),
        'u_gas_gradient': np.array([[0.2, 0.0, 0.0],
                                    [0.2, 0.0, 0.0],
                                    [0.2, 0.0, 0.0]]),
        'native_unit_set': 'real',
    }


def _reduce(main, el, gas):
    """The skill's own Q-row reduction, so a fixture states the same
    arithmetic the backend applies instead of pretending the composite
    still does it."""
    reduced = np.array(main, dtype=np.float64, copy=True)
    reduced[0:3] = reduced[0:3] - el - gas
    return reduced


def _energy_raw(qmmm_el_gradient=None, u_gas_gradient=None):
    atomic = _energy_atomic()
    if qmmm_el_gradient is not None:
        atomic['qmmm_el_gradient'] = qmmm_el_gradient
    if u_gas_gradient is not None:
        atomic['u_gas_gradient'] = u_gas_gradient
    main = np.zeros((NATOM, 3))
    main[0:3, :] = 10.0
    main[3:6, :] = 5.0
    return {
        'energy': np.array([-10.0]),
        'gradient': _reduce(main, atomic['qmmm_el_gradient'],
                            atomic['u_gas_gradient'])[np.newaxis, :, :],
        'qmmm': _energy_block(),
        'qmmm_atomic': atomic,
    }


def _qm_raw():
    return {
        'energy': np.array([-10.0]),
        'gradient': np.ones((1, 3, 3)),
    }


class TestQmmmExpansion(unittest.TestCase):
    """The plan expands into field, QM and energy tasks."""

    def test_order_roles_contributions(self):
        from pyrai3md.planning.task import _expand_qmmm
        tasks = _expand_qmmm(
            _skills(),
            SimpleNamespace(highlevel=list(QIDX), embedding=True))
        self.assertEqual([task['role'] for task in tasks],
                         ['field', 'qm', 'energy'])
        self.assertEqual([task['contribution'] for task in tasks],
                         [0, 1, 1])
        self.assertEqual(tasks[0]['qmmm_role'], 'field')
        self.assertEqual(tasks[0]['region'], 'all')
        self.assertEqual(tasks[0]['binding_role'], 'qm2')
        self.assertFalse(tasks[0]['provides_electronic'])
        self.assertIn('field', tasks[0]['capabilities'])
        self.assertNotIn('stress', tasks[0]['capabilities'])
        self.assertEqual(tasks[1]['qmmm_role'], 'qm')
        self.assertEqual(tasks[1]['region'], 'model')
        self.assertEqual(tasks[1]['region_indices'], QIDX)
        self.assertTrue(tasks[1]['provides_electronic'])
        self.assertNotIn('stress', tasks[1]['capabilities'])
        self.assertEqual(tasks[2]['qmmm_role'], 'energy')
        self.assertEqual(tasks[2]['region'], 'all')
        self.assertEqual(tasks[2]['binding_role'], 'qm2')
        self.assertFalse(tasks[2]['provides_electronic'])
        self.assertIn('qmmm', tasks[2]['capabilities'])
        self.assertNotIn('stress', tasks[2]['capabilities'])

    def test_capability_tokens_spell_the_jobtype(self):
        from pyrai3md.planning.assembly import spell_skill_jobtype
        self.assertEqual(
            spell_skill_jobtype(['energy', 'gradient', 'field'],
                                'energy gradient'),
            'energy gradient field')
        self.assertEqual(
            spell_skill_jobtype(['energy', 'gradient', 'qmmm'],
                                'energy gradient'),
            'energy gradient qmmm')

    def test_without_embedding_two_tasks(self):
        from pyrai3md.planning.task import _expand_qmmm
        tasks = _expand_qmmm(
            _skills(),
            SimpleNamespace(highlevel=list(QIDX), embedding=False))
        self.assertEqual([task['role'] for task in tasks], ['qm', 'energy'])
        self.assertEqual([task['contribution'] for task in tasks], [1, 1])
        self.assertNotIn('field', [task['qmmm_role'] for task in tasks])

    def test_primary_and_system_providers(self):
        from pyrai3md.planning.task import (
            _expand_qmmm, resolve_primary_task, resolve_system_task)
        tasks = _expand_qmmm(
            _skills(),
            SimpleNamespace(highlevel=list(QIDX), embedding=True))
        self.assertEqual(resolve_primary_task(tasks)['task_id'], 'qm')
        self.assertEqual(resolve_system_task(tasks)['task_id'], 'energy')

    def test_electronic_and_stress_caps_reach_the_tasks(self):
        from pyrai3md.planning.task import _expand_qmmm
        skills = _skills()
        skills['a']['capabilities'] = ['energy', 'gradient', 'nac', 'soc']
        skills['b']['capabilities'] = ['energy', 'gradient', 'stress']
        tasks = _expand_qmmm(
            skills,
            SimpleNamespace(highlevel=list(QIDX), embedding=True))
        by_role = {task['qmmm_role']: task for task in tasks}
        # Surface hopping: the QM provider carries the couplings.
        self.assertIn('nac', by_role['qm']['capabilities'])
        self.assertIn('soc', by_role['qm']['capabilities'])
        self.assertNotIn('stress', by_role['qm']['capabilities'])
        # The whole-system classical task carries the structural request.
        self.assertIn('stress', by_role['energy']['capabilities'])


class TestAssembleQmmm(unittest.TestCase):
    """E = E_QM + E_classic; Q rows get the classical correction."""

    def setUp(self):
        self.tasks = _qmmm_tasks()
        self.outcomes = [
            _outcome(self.tasks[0], _field_raw()),
            _outcome(self.tasks[1], _qm_raw()),
            _outcome(self.tasks[2], _energy_raw()),
        ]
        self.request = {'natom': NATOM, 'nstate': 1,
                        'need_gradient': True}

    def test_energy_is_qm_plus_e_classic(self):
        data = assemble_qmmm(self.tasks, self.outcomes, self.request)
        self.assertAlmostEqual(float(data['energy'][0]), -12.5)
        components = data['qmmm_components']
        self.assertAlmostEqual(components['e_classic'], -2.5)
        self.assertAlmostEqual(components['u_main'], -10.0)
        self.assertAlmostEqual(components['qmmm_el'], -3.5)
        self.assertEqual(components['q_qm'], [0.1, 0.2, -0.3])
        self.assertEqual(components['qm_atoms'], QIDX)

    def test_qm_rows_add_the_classical_correction(self):
        data = assemble_qmmm(self.tasks, self.outcomes, self.request)
        gradient = np.asarray(data['gradient'])
        # QM 1.0 + (full Q 10.0 - F_el 0.1 - F_gas 0.2) = 10.7 on x;
        # y/z carry only the full Q row and the QM value.
        np.testing.assert_allclose(
            gradient[0, 0:3, :],
            [[10.7, 11.0, 11.0],
             [10.7, 11.0, 11.0],
             [10.7, 11.0, 11.0]])
        # MM rows keep the full U_main rows.
        np.testing.assert_allclose(gradient[0, 3:6, :], 5.0)

    def test_composite_terms_use_the_two_physical_components(self):
        data = assemble_qmmm(self.tasks, self.outcomes, self.request)
        terms = data['composite_terms']
        self.assertEqual(terms['labels'], ['high[qm]', 'all[energy]'])
        self.assertEqual(terms['contributions'], [1, 1])
        self.assertAlmostEqual(terms['energies'][0][0], -10.0)
        self.assertAlmostEqual(terms['energies'][1][0], -2.5)

    def test_two_tasks_refused(self):
        with self.assertRaises(ValueError):
            assemble_qmmm(self.tasks[:2], self.outcomes[:2], self.request)

    def test_negative_contribution_refused(self):
        outcomes = list(self.outcomes)
        outcomes[2] = _outcome(self.tasks[2], _energy_raw(),
                              contribution=-1)
        with self.assertRaises(ValueError):
            assemble_qmmm(self.tasks, outcomes, self.request)

    def test_missing_field_block_refused(self):
        raw = _field_raw()
        raw.pop('qmmm_field')
        outcomes = list(self.outcomes)
        outcomes[0] = _outcome(self.tasks[0], raw)
        with self.assertRaises(ValueError):
            assemble_qmmm(self.tasks, outcomes, self.request)

    def test_missing_energy_atomic_block_refused(self):
        raw = _energy_raw()
        raw.pop('qmmm_atomic')
        outcomes = list(self.outcomes)
        outcomes[2] = _outcome(self.tasks[2], raw)
        with self.assertRaises(ValueError):
            assemble_qmmm(self.tasks, outcomes, self.request)

    def test_partition_mismatch_refused(self):
        raw = _energy_raw()
        raw['qmmm']['qm_atom_ids'] = [1, 2, 4]
        outcomes = list(self.outcomes)
        outcomes[2] = _outcome(self.tasks[2], raw)
        with self.assertRaises(ValueError):
            assemble_qmmm(self.tasks, outcomes, self.request)

    def test_inconsistent_native_components_refused(self):
        raw = _energy_raw()
        raw['qmmm_atomic']['e_classic'] = -1.0
        outcomes = list(self.outcomes)
        outcomes[2] = _outcome(self.tasks[2], raw)
        with self.assertRaises(ValueError):
            assemble_qmmm(self.tasks, outcomes, self.request)

    def test_q_qm_length_refused(self):
        raw = _energy_raw()
        raw['qmmm']['q_qm'] = [0.1, 0.2]
        outcomes = list(self.outcomes)
        outcomes[2] = _outcome(self.tasks[2], raw)
        with self.assertRaises(ValueError):
            assemble_qmmm(self.tasks, outcomes, self.request)

    def test_energy_only_has_no_gradient(self):
        request = dict(self.request, need_gradient=False)
        data = assemble_qmmm(self.tasks, self.outcomes, request)
        self.assertNotIn('gradient', data)

    def test_stress_is_the_classical_u_main_tensor(self):
        raw = _energy_raw()
        raw['stress_tensor'] = np.eye(3).reshape(1, 3, 3) * 7.0
        outcomes = list(self.outcomes)
        outcomes[2] = _outcome(self.tasks[2], raw)
        data = assemble_qmmm(self.tasks, outcomes, self.request)
        np.testing.assert_allclose(data['stress_tensor'][0],
                                   7.0 * np.eye(3))
        self.assertEqual(data['qmmm_components']['stress'],
                         'classical_u_main')

    def test_scheme_and_field_rows_are_recorded_for_the_log(self):
        qm_raw = _qm_raw()
        qm_raw['qmmm_external_potential'] = [
            [0.1, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.0]]
        outcomes = list(self.outcomes)
        outcomes[1] = _outcome(self.tasks[1], qm_raw)
        data = assemble_qmmm(self.tasks, outcomes, self.request)
        self.assertEqual(data['composite_scheme'], 'qmmm')
        self.assertEqual(
            data['qmmm_components']['external_potential'],
            qm_raw['qmmm_external_potential'])

    def test_two_tasks_without_embedding(self):
        from pyrai3md.workflow.multiscale.composite import assemble_qmmm
        tasks = [self.tasks[1], self.tasks[2]]
        # Isolated QM: no field term at all, so only dU_gas leaves the Q rows.
        raw = _energy_raw(qmmm_el_gradient=np.zeros((3, 3)))
        raw['qmmm_atomic']['e_classic'] = -6.0
        raw['qmmm_atomic']['qmmm_el'] = 0.0
        raw['qmmm']['qmmm_el_native'] = 0.0
        raw['qmmm']['e_classic_native'] = -6.0
        raw['qmmm']['qmmm_el_gradient_native'] = np.zeros((3, 3))
        outcomes = [_outcome(tasks[0], _qm_raw()),
                    _outcome(tasks[1], raw)]
        data = assemble_qmmm(tasks, outcomes, self.request)
        self.assertAlmostEqual(float(data['energy'][0]), -16.0)
        gradient = np.asarray(data['gradient'])
        # QM 1.0 + full Q 10.0 - F_gas 0.2 (no F_el while isolated).
        np.testing.assert_allclose(
            gradient[0, 0:3, :],
            [[10.8, 11.0, 11.0],
             [10.8, 11.0, 11.0],
             [10.8, 11.0, 11.0]])
        np.testing.assert_allclose(gradient[0, 3:6, :], 5.0)
        self.assertIsNone(data['qmmm_components']['field'])

    def test_field_gradient_default_removes_f_el(self):
        # A request without the key keeps the ESPF contract: F_el is
        # removed, so the Q rows read 10.7 as in the pinned case above.
        data = assemble_qmmm(self.tasks, self.outcomes, self.request)
        self.assertIs(
            data['qmmm_components']['qm_contains_field_gradient'], True)
        np.testing.assert_allclose(
            np.asarray(data['gradient'])[0, 0:3, 0], [10.7, 10.7, 10.7])

    def test_field_gradient_off_keeps_f_el_on_the_classical_side(self):
        # F_el is not removed: the Q rows gain exactly the 0.1 per x that
        # the default case subtracts, and nothing else moves.
        request = dict(self.request, qm_contains_field_gradient=False)
        default = assemble_qmmm(self.tasks, self.outcomes, self.request)
        kept = assemble_qmmm(self.tasks, self.outcomes, request)
        self.assertIs(
            kept['qmmm_components']['qm_contains_field_gradient'], False)
        delta = (np.asarray(kept['gradient'])
                 - np.asarray(default['gradient']))
        np.testing.assert_allclose(delta[0, 0:3, 0], 0.1)
        np.testing.assert_allclose(delta[0, 0:3, 1:], 0.0)
        np.testing.assert_allclose(delta[0, 3:6, :], 0.0)
        self.assertGreater(float(np.abs(delta).max()), 0.0)
        # The switch never moves an energy.
        self.assertAlmostEqual(float(kept['energy'][0]),
                               float(default['energy'][0]), places=12)

    def test_moving_the_reduction_into_the_skill_changed_no_bit(self):
        # The load-bearing judgement for the cutover: before, the composite
        # subtracted dU_gas and dqmmm_el from the Q rows of the full U_main
        # gradient; now the skill hands the reduced array and the composite
        # only adds dqmmm_el back when the provider lacks it. Same inputs,
        # same floats, so equality must be exact, not within a tolerance.
        def legacy_stitch(qm_gradient, main, el, gas, contains_field):
            field = el if contains_field else np.zeros_like(el)
            out = np.array(main, dtype=np.float64, copy=True)
            out[:, QIDX, :] = (qm_gradient
                               + main[:, QIDX, :] - field - gas)
            return out

        atomic = _energy_atomic()
        main = np.zeros((1, NATOM, 3))
        main[0, 0:3, :] = 10.0
        main[0, 3:6, :] = 5.0
        qm_gradient = np.ones((1, 3, 3))
        for contains_field in (True, False):
            request = dict(self.request,
                           qm_contains_field_gradient=contains_field)
            data = assemble_qmmm(self.tasks, self.outcomes, request)
            expected = legacy_stitch(qm_gradient, main,
                                     atomic['qmmm_el_gradient'],
                                     atomic['u_gas_gradient'], contains_field)
            np.testing.assert_array_equal(np.asarray(data['gradient']),
                                          expected)


# The route this replaced, frozen here from the LAMMPS skill's own
# constants so the comparison is arithmetic and not a call into the table
# under test: the skill used to hand back native `forces` and its
# `service/conversion.py::to_framework` produced
# `energy = native * KCAL_TO_HARTREE` and
# `gradient = (-forces) * (KCAL_TO_HARTREE * BOHR_TO_ANGSTROM)`.
_LEGACY_HARTREE_TO_KCAL = 627.5094740631
_LEGACY_HARTREE_TO_EV = 27.211386245988
_LEGACY_BOHR_TO_ANGSTROM = 0.529177210903


def _legacy_to_framework(result):
    kcall = 1.0 / _LEGACY_HARTREE_TO_KCAL
    ev = 1.0 / _LEGACY_HARTREE_TO_EV
    bohr = _LEGACY_BOHR_TO_ANGSTROM
    units = result.get('_lammps_units', 'real')
    energy_factor = kcall if units == 'real' else ev
    gradient_factor = (kcall if units == 'real' else ev) * bohr
    out = dict(result)
    out.pop('_conversion', None)
    out.pop('_lammps_units', None)
    if out.get('energy') is not None:
        out['energy'] = np.asarray(out['energy'], float) * energy_factor
    if out.get('forces') is not None:
        gradient = -np.asarray(out.pop('forces'), float) * gradient_factor
        out['gradient'] = gradient.reshape((1,) + gradient.shape)
    return out


class TestSkillConversion(unittest.TestCase):
    """The manifest units declaration owns the native→Properties factors."""

    def test_energy_factor(self):
        value = convert_native_energy(-627.5094740631, 'real',
                                      'lammps_skill')
        self.assertAlmostEqual(value, -1.0, places=10)

    def test_gradient_is_scaled_not_flipped(self):
        # The sign belongs to the label: `gradient` is dU/dR, and the skill
        # applies it. The framework only scales.
        gradient = convert_native_gradient([[0.0008433, 0.0, 0.0]], 'real',
                                           'lammps_skill')
        self.assertGreater(gradient[0, 0], 0.0)
        # 0.0008433 kcal/mol/Angstrom x 8.432975640616191e-4 = 7.11152835773e-7
        self.assertAlmostEqual(gradient[0, 0], 7.111528357731634e-07,
                               places=18)

    def test_apply_conversion_removes_the_marker(self):
        native = {
            'energy': np.array([627.5094740631]),
            'gradient': np.array([[1.0, 0.0, 0.0]]),
            '_native_units': 'real',
        }
        converted = apply_result_units(native, 'lammps_skill')
        self.assertAlmostEqual(float(converted['energy'][0]), 1.0,
                               places=10)
        self.assertNotIn('_native_units', converted)
        # A (natom, 3) native array arrives with the state axis.
        self.assertEqual(converted['gradient'].shape, (1, 1, 3))

    def test_unmarked_result_is_unchanged(self):
        raw = {'energy': [-1.0]}
        self.assertIs(apply_result_units(raw, 'lammps_skill'), raw)

    def test_renamed_declaration_key_is_read(self):
        # #464/3.4: the written key is the skill's declaration, not a
        # hardcoded name. Renaming `e_classic` in `qmmm.results` must move
        # the value the framework reads; ignoring the declaration leaves
        # the renamed block unread, which refuses here.
        from unittest import mock
        from pyrai3md.workflow.calculator import Calculator
        raw = {'qmmm': {
            'e_classic_renamed': -627.5094740631,
            'u_main_native': -10.0,
            'u_gas_native': -4.0,
            'qmmm_el_native': -3.5,
            'qmmm_el_gradient_native': np.zeros((1, 3)),
            'u_gas_gradient_native': np.zeros((1, 3)),
        }}
        outcome = {'raw': raw}
        task = {'task_id': 'energy', 'skill_name': 'lammps_skill'}
        declaration = {'results': {'e_classic': 'e_classic_renamed',
                                   'default_units': 'real'}}
        with mock.patch('pyrai3md.skills.query.get_declaration',
                        return_value=declaration):
            atomic = Calculator()._qmmm_atomic_block(outcome, task)
        self.assertAlmostEqual(atomic['e_classic'], -1.0, places=10)

    def test_new_route_reproduces_the_legacy_route_bitwise(self):
        # Moving the conversion from the skill module to the framework may
        # not change one bit: the skill now hands `gradient` (sign applied)
        # instead of `forces`, and the factor table is the same product of
        # the same three constants.
        forces = np.array([[1.0, -2.5, 3.0], [0.7, 0.0, -0.1]])
        energy = np.array([1234.5, -627.5094740631])
        legacy = _legacy_to_framework({
            'energy': energy, 'forces': forces, '_conversion': 'lammps',
            '_lammps_units': 'real'})
        current = apply_result_units({
            'energy': energy,
            'gradient': -np.asarray(forces, dtype=np.float64),
            '_native_units': 'real'}, 'lammps_skill')
        np.testing.assert_array_equal(current['energy'], legacy['energy'])
        np.testing.assert_array_equal(current['gradient'], legacy['gradient'])

    def test_metal_unit_set_scales_differently(self):
        # One factor at a time: flipping the unit set must move both the
        # energy and the gradient by the eV factors, and nothing else.
        real = apply_result_units({'energy': np.array([1.0]),
                                   'gradient': np.array([[1.0, 0.0, 0.0]]),
                                   '_native_units': 'real'}, 'lammps_skill')
        metal = apply_result_units({'energy': np.array([1.0]),
                                    'gradient': np.array([[1.0, 0.0, 0.0]]),
                                    '_native_units': 'metal'},
                                   'lammps_skill')
        self.assertAlmostEqual(float(real['energy'][0]), 1.0 / 627.5094740631,
                               places=12)
        self.assertAlmostEqual(float(metal['energy'][0]), 1.0 / 27.211386245988,
                               places=12)
        self.assertGreater(float(metal['gradient'][0, 0, 0]),
                           float(real['gradient'][0, 0, 0]))

    def test_unknown_unit_set_is_refused(self):
        with self.assertRaises(ValueError):
            apply_result_units({'energy': np.array([1.0]),
                                '_native_units': 'groovy'}, 'lammps_skill')

    def test_skill_without_a_declaration_never_scales(self):
        # molcas emits framework units already: no declaration, so its
        # factors are 1.0 and a result cannot be silently rescaled.
        self.assertEqual(resolve_result_units('molcas_skill', 'real'),
                         (1.0, 1.0))
        raw = {'energy': np.array([-1.0]), 'gradient': np.array([[1.0, 0.0,
                                                                 0.0]])}
        self.assertIs(apply_result_units(raw, 'molcas_skill'), raw)


class TestExternalPotentialBuilder(unittest.TestCase):
    """The .esfp handoff is read and its partition is checked."""

    def _calculator(self):
        from pyrai3md.workflow.calculator import Calculator
        return Calculator()

    def _backend(self, rows):
        tmp = tempfile.mkdtemp(prefix='qmmm_field_')
        with open(os.path.join(tmp, 'probe.esfp'), 'w') as handle:
            handle.write('# PyRAI3MD ESPF probe (atomic units)\n')
            for row in rows:
                handle.write(' '.join(f'{value:.16e}' for value in row)
                             + '\n')
        return SimpleNamespace(calcdir=tmp)

    def test_the_returned_block_is_the_handoff(self):
        # The field task hands V/E back in its own result block; that
        # block is the channel, in the QM task's atom order (#3.3), and
        # its native columns reach the QM side in atomic units.
        calculator = self._calculator()
        backend = self._backend([[9.0, 9.0, 9.0, 9.0]] * 3)
        qm_task = {'task_id': 'qm', 'region_indices': list(QIDX)}
        result = calculator._qmmm_external_potential(
            backend, qm_task, _field_block(), 'lammps_skill')
        factor = convert_native_energy(1.0, 'real', 'lammps_skill')
        np.testing.assert_allclose(
            np.asarray(result),
            [[-1.0 * factor, 0.0, 0.0, 0.0],
             [-2.0 * factor, 0.0, 0.0, 0.0],
             [-3.0 * factor, 0.0, 0.0, 0.0]])

    def test_missing_unit_set_uses_the_declared_default(self):
        # `qmmm.results.default_units` is the second spelling of the same
        # declaration, the one `_qmmm_atomic_block` already reads.
        block = _field_block()
        block.pop('units')
        result = self._calculator()._qmmm_external_potential(
            self._backend([]), {'task_id': 'qm',
                                'region_indices': list(QIDX)},
            block, 'lammps_skill')
        factor = convert_native_energy(1.0, 'real', 'lammps_skill')
        self.assertEqual(result[0][0], -1.0 * factor)

    def test_missing_unit_set_and_default_refused(self):
        block = _field_block()
        block.pop('units')
        with self.assertRaises(ValueError):
            self._calculator()._qmmm_external_potential(
                self._backend([]), {'task_id': 'qm',
                                    'region_indices': list(QIDX)},
                block, 'no_such_skill')

    def test_the_block_and_the_esfp_file_agree(self):
        # The two channels are one number: the block's native columns
        # converted are bit-for-bit the .esfp rows the same task wrote.
        calculator = self._calculator()
        rows = _field_block()
        rows['v_mm_native'] = np.array([627.5094740631, -1255.0189481262,
                                        1882.5284221893])
        rows['e_mm_native'] = np.array([[627.5094740631, 0.0, 0.0],
                                        [0.0, -627.5094740631, 0.0],
                                        [0.0, 0.0, 627.5094740631]])
        block_row = calculator._qmmm_external_potential(
            self._backend([]), {'task_id': 'qm',
                                'region_indices': list(QIDX)},
            rows, 'lammps_skill')
        from pyrai3md.skills.lammps_skill.service.parser.qmmm_output import \
            esfp_rows
        file_row = [[float(value) for value in line.split()[1:]]
                    for line in
                    esfp_rows([index + 1 for index in QIDX],
                              rows['v_mm_native'], rows['e_mm_native'],
                              'real')]
        np.testing.assert_allclose(np.asarray(block_row),
                                   np.asarray(file_row), rtol=0, atol=1e-12)

    def test_rows_follow_the_qm_order(self):
        calculator = self._calculator()
        rows = [[1.0, 0.1, 0.0, 0.0],
                [2.0, 0.2, 0.0, 0.0],
                [3.0, 0.3, 0.0, 0.0]]
        backend = self._backend(rows)
        qm_task = {'task_id': 'qm', 'region_indices': list(QIDX)}
        result = calculator._qmmm_external_potential(
            backend, qm_task, _legacy_field_block())
        np.testing.assert_allclose(np.asarray(result), rows)

    def test_missing_esfp_refused(self):
        calculator = self._calculator()
        backend = SimpleNamespace(calcdir=tempfile.mkdtemp())
        with self.assertRaises(RuntimeError):
            calculator._qmmm_external_potential(
                backend, {'task_id': 'qm', 'region_indices': list(QIDX)},
                _legacy_field_block())

    def test_partition_mismatch_refused(self):
        calculator = self._calculator()
        backend = self._backend([[1.0, 0.0, 0.0, 0.0]] * 3)
        block = _field_block()
        block['qm_atom_ids'] = [1, 2, 4]
        with self.assertRaises(ValueError):
            calculator._qmmm_external_potential(
                backend, {'task_id': 'qm',
                          'region_indices': list(QIDX)}, block)

    def test_non_finite_refused(self):
        calculator = self._calculator()
        rows = [[1.0, 0.1, 0.0, 0.0],
                [2.0, float('nan'), 0.0, 0.0],
                [3.0, 0.3, 0.0, 0.0]]
        backend = self._backend(rows)
        with self.assertRaises(ValueError):
            calculator._qmmm_external_potential(
                backend, {'task_id': 'qm',
                          'region_indices': list(QIDX)},
                _legacy_field_block())

    def test_row_count_refused(self):
        calculator = self._calculator()
        backend = self._backend([[1.0, 0.0, 0.0, 0.0]] * 2)
        with self.assertRaises(ValueError):
            calculator._qmmm_external_potential(
                backend, {'task_id': 'qm',
                          'region_indices': list(QIDX)},
                _legacy_field_block())


class _FieldBackend:
    """Auxiliary A/B probe: writes .esfp and returns the native block."""

    def __init__(self, config=None, execution_id=''):
        self.config = config
        self.calls = 0
        self.calcdir = tempfile.mkdtemp(prefix='qmmm_field_')

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None, qmmm_inputs=None):
        self.calls += 1
        self.qmmm_inputs = qmmm_inputs
        with open(os.path.join(self.calcdir, 'probe.esfp'), 'w') as handle:
            handle.write('# PyRAI3MD ESPF probe (atomic units)\n')
            for value, field in zip([1.0, 2.0, 3.0],
                                    [[0.1, 0.0, 0.0],
                                     [0.2, 0.0, 0.0],
                                     [0.3, 0.0, 0.0]]):
                handle.write(' '.join(f'{number:.16e}' for number in
                                      (value, *field)) + '\n')
        natom = len(atoms)
        return {
            'energy': np.array([0.0]),
            'gradient': -np.zeros((natom, 3)),
            'qmmm_field': _field_block(),
            '_native_units': 'real',
        }


class _QMBackend:
    """Region QM backend: records the field and returns q_qm."""

    supports_external_potential = True

    def __init__(self, config=None, execution_id=''):
        self.config = config
        self.calls = 0
        self.external_potential = None

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None, external_potential=None):
        self.calls += 1
        self.external_potential = external_potential
        natom = len(atoms)
        return {
            'energy': np.array([-10.0]),
            'gradient': np.ones((1, natom, 3)),
            'computed_charges': [[0.1, 0.2, -0.3]],
        }


class _QmCouplingBackend(_QMBackend):
    """The same region QM leg, answering with one coupling row."""

    supports_nac_request = True

    def calculate(self, *args, **kwargs):
        raw = super().calculate(*args, **kwargs)
        raw['nac'] = [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]
        return raw


class _EnergyBackend:
    """Full-FF backend: records qmmm_inputs and returns the native block."""

    def __init__(self, config=None, execution_id=''):
        self.config = config
        self.calls = 0
        self.qmmm_inputs = None

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None, qmmm_inputs=None):
        self.calls += 1
        self.qmmm_inputs = qmmm_inputs
        natom = len(atoms)
        block = _energy_block()
        # The transport test pins the Q-row correction to zero; the
        # assembler test above owns the non-zero correction math.
        block['u_gas_gradient_native'] = np.zeros((3, 3))
        block['qmmm_el_gradient_native'] = np.zeros((3, 3))
        return {
            'energy': np.array([-10.0]),
            # A LAMMPS result arrives as dU/dR in native units; -0.0 keeps
            # the sign the engine's own zeros would have produced.
            'gradient': -np.zeros((natom, 3)),
            'qmmm': block,
            '_native_units': 'real',
        }


class _EnergyBackendWithField(_EnergyBackend):
    """Same cold run, with a nonzero field gradient reduced off the Q rows."""

    def calculate(self, atoms, coordinates, **kwargs):
        raw = _EnergyBackend.calculate(self, atoms, coordinates, **kwargs)
        # q_qm * E is the force; as a gradient the native block carries
        # -q*E, and the reduction removes it from the Q rows the same way
        # the skill's _reduce_main_gradient does.
        el_gradient = np.array([[-0.5, 0.0, 0.0]] * 3, dtype=np.float64)
        raw['qmmm']['qmmm_el_gradient_native'] = el_gradient
        gradient = np.array(raw['gradient'], dtype=np.float64, copy=True)
        gradient[0:3] = gradient[0:3] - el_gradient
        raw['gradient'] = gradient
        return raw


class TestQmmmThreeStage(unittest.TestCase):
    """The calculator runs field -> QM -> energy and transports the data."""

    def _make_state(self):
        from pyrai3md.data.state import PyRAIState
        from pyrai3md.data.runtime.state import RuntimeState
        from pyrai3md.schema.config import PyRAIConfig

        config = PyRAIConfig()
        config.system.atoms = ['O', 'H', 'H', 'O', 'H', 'H']
        config.system.coordinates = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0],
        ]
        config.system.natom = NATOM
        config.system.charges = 0
        config.system.multiplicity = 1
        config.system.nstate = 1
        config.system.embedding = True
        config.system.read_charge = False
        state = PyRAIState(runtime=RuntimeState(total_steps=1))
        state.pmd = config
        return state

    def _task(self, role, api, region, indices, contribution,
              provides_electronic, skill_name):
        return {
            'role': role,
            'task': 0,
            'task_id': role,
            'task_slot': {'field': 0, 'qm': 1, 'energy': 2}[role],
            'layer_id': {'field': 2, 'qm': 0, 'energy': 2}[role],
            'static_id': 'tmp_%s_MOCK' % role,
            'contribution': contribution,
            'region': region,
            'region_indices': list(indices),
            'region_charge': None,
            'region_multiplicity': None,
            'skill_name': skill_name,
            'api': api,
            'capabilities': ['energy', 'gradient'],
            'config': {'results': {}},
            'center_index': 0 if role == 'qm' else None,
            'provides_electronic': provides_electronic,
            'composite': 'qmmm',
            'qmmm_role': role,
            'binding_role': role,
            'partition_indices': list(QIDX),
            'plan_version': 1,
            'plan_fingerprint': None,
            'property_request_spec': None,
        }

    def test_a_region_coupling_reaches_the_whole_system(self):
        # What the first `meci` run that actually received a coupling died
        # on: `shapes (18,) and (4743,) not aligned`. The region provider
        # answers per region atom, the search steps per system atom, and
        # before the coupling channel existed nobody noticed, because a
        # fabricated zero row was dropped below the direction floor before
        # anyone took its dot product.
        from pyrai3md.workflow.calculator import Calculator
        state = self._make_state()
        state.add_task('theory', self._task(
            'field', _FieldBackend, 'all', [], 0, False, 'lammps_skill'))
        state.add_task('theory', self._task(
            'qm', _QmCouplingBackend, 'model', QIDX, 1, True,
            'molcas_skill'))
        state.add_task('theory', self._task(
            'energy', _EnergyBackend, 'all', [], 1, False,
            'lammps_skill'))

        result = Calculator(state=state).initialize().run()
        rows = np.asarray(result['nac'])
        self.assertEqual(rows.shape, (1, NATOM, 3))
        self.assertEqual(rows[0][0].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(rows[0][2].tolist(), [6.0, 7.0, 8.0])
        self.assertEqual(rows[0][3].tolist(), [0.0, 0.0, 0.0])

    def test_field_then_qm_then_energy(self):
        from pyrai3md.workflow.calculator import Calculator
        state = self._make_state()
        state.add_task('theory', self._task(
            'field', _FieldBackend, 'all', [], 0, False, 'lammps_skill'))
        state.add_task('theory', self._task(
            'qm', _QMBackend, 'model', QIDX, 1, True, 'molcas_skill'))
        state.add_task('theory', self._task(
            'energy', _EnergyBackend, 'all', [], 1, False,
            'lammps_skill'))

        calculator = Calculator(state=state).initialize()
        result = calculator.run()

        field_backend = calculator.backends['theory'][0]
        qm_backend = calculator.backends['theory'][1]
        energy_backend = calculator.backends['theory'][2]
        self.assertEqual(field_backend.calls, 1)
        self.assertEqual(qm_backend.calls, 1)
        self.assertEqual(energy_backend.calls, 1)

        rows = np.asarray(qm_backend.external_potential, dtype=float)
        self.assertEqual(rows.shape, (3, 4))
        # The field block is the handoff, converted to atomic units: the
        # same numbers the .esfp channel carries. Reading the native
        # columns raw embedded the QM in a 627x field (#464 regression).
        factor = convert_native_energy(1.0, 'real', 'lammps_skill')
        np.testing.assert_allclose(
            rows, [[-1.0 * factor, 0.0, 0.0, 0.0],
                   [-2.0 * factor, 0.0, 0.0, 0.0],
                   [-3.0 * factor, 0.0, 0.0, 0.0]])

        payload = energy_backend.qmmm_inputs
        self.assertEqual(payload['q_qm'], [0.1, 0.2, -0.3])
        np.testing.assert_allclose(payload['v_mm'], [-1.0, -2.0, -3.0])
        np.testing.assert_allclose(payload['e_mm'], np.zeros((3, 3)))

        e_classic = convert_native_energy(-2.5, 'real', 'lammps_skill')
        self.assertAlmostEqual(float(np.asarray(result['energy'])[0]),
                               -10.0 + e_classic, places=10)
        gradient = np.asarray(result['gradient'])
        np.testing.assert_allclose(gradient[0, 0:3, :], 1.0)
        np.testing.assert_allclose(gradient[0, 3:6, :], 0.0)
        self.assertEqual(result['qmmm_components']['q_qm'],
                         [0.1, 0.2, -0.3])
        self.assertAlmostEqual(
            result['qmmm_components']['e_classic'], e_classic, places=12)

    def _run_field_force(self, contains):
        """One three-stage run with a nonzero F_el on the Q rows."""
        from pyrai3md.workflow.calculator import Calculator
        state = self._make_state()
        if contains is not None:
            state.pmd.system.qm_contains_field_gradient = contains
        state.add_task('theory', self._task(
            'field', _FieldBackend, 'all', [], 0, False, 'lammps_skill'))
        state.add_task('theory', self._task(
            'qm', _QMBackend, 'model', QIDX, 1, True, 'molcas_skill'))
        state.add_task('theory', self._task(
            'energy', _EnergyBackendWithField, 'all', [], 1, False,
            'lammps_skill'))
        return Calculator(state=state).initialize().run()

    def test_field_gradient_switch_reaches_the_assembler(self):
        # system.qm_contains_field_gradient travels
        # calculator -> request -> assemble_qmmm: with the switch off the
        # Q rows keep the classical field force instead of removing it.
        f_el = convert_native_gradient([[-0.5, 0.0, 0.0]] * 3, 'real',
                                     'lammps_skill')
        default = self._run_field_force(None)
        kept = self._run_field_force(False)
        delta = (np.asarray(kept['gradient'])[0, 0:3, :]
                 - np.asarray(default['gradient'])[0, 0:3, :])
        np.testing.assert_allclose(delta, f_el, atol=1e-12)
        self.assertGreater(float(np.abs(delta).max()), 1e-6)
        np.testing.assert_allclose(
            np.asarray(kept['gradient'])[0, 3:6, :],
            np.asarray(default['gradient'])[0, 3:6, :])
        self.assertAlmostEqual(float(np.asarray(kept['energy'])[0]),
                               float(np.asarray(default['energy'])[0]),
                               places=12)
        self.assertIs(
            kept['qmmm_components']['qm_contains_field_gradient'], False)
        self.assertIs(
            default['qmmm_components']['qm_contains_field_gradient'], True)


class TestFieldGradientSwitchSchema(unittest.TestCase):
    """The switch is a framework input with the ESPF default."""

    def test_system_default_is_contains(self):
        from pyrai3md.schema.config import PyRAIConfig, System
        self.assertIs(System().qm_contains_field_gradient, True)
        self.assertIs(
            PyRAIConfig().system.qm_contains_field_gradient, True)

    def test_yaml_and_classic_spellings(self):
        from pyrai3md.schema.config import System
        self.assertIs(
            System.model_validate(
                {'qm_contains_field_gradient': False}
            ).qm_contains_field_gradient, False)
        from pyrai3md.schema.adapter import MOLECULE_TYPES
        self.assertEqual(MOLECULE_TYPES['qm_contains_field_gradient'], 'b')


class TestQmmmTaskConfigInjection(unittest.TestCase):
    """build_state injects the two profiles and the plan identity."""

    @classmethod
    def setUpClass(cls):
        from pyrai3md.kernel.environment import Environment
        Environment().initialize()

    def _build(self, read_charge=False):
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.config import PyRAIConfig
        config = PyRAIConfig()
        config.control.jobtype = 'energy gradient'
        config.control.theory = ['molcas', 'lammps']
        config.control.multiscale_type = 'qmmm'
        config.system.atoms = ['O', 'H', 'H', 'O', 'H', 'H']
        config.system.coordinates = [
            [1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0],
            [5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0],
        ]
        config.system.natom = NATOM
        config.system.charges = 0
        config.system.multiplicity = 1
        config.system.nstate = 1
        config.system.highlevel = list(QIDX)
        config.system.q_qm = [0.1, 0.2, -0.3]
        config.system.embedding = True
        config.system.read_charge = bool(read_charge)
        return build_state(config)

    def test_field_task_gets_the_probe_profile(self):
        state = self._build()
        tasks = [task for _, task in state.iter_tasks()]
        self.assertEqual([task['qmmm_role'] for task in tasks],
                         ['field', 'qm', 'energy'])
        field = tasks[0]
        self.assertEqual(field['composite'], 'qmmm')
        self.assertEqual(field['binding_role'], 'qm2')
        self.assertEqual(field['partition_indices'], QIDX)
        skill_config = field['config']['skill_config']
        self.assertEqual(skill_config['profile'], 'qmmm_field')
        self.assertEqual(skill_config['jobtype'], 'energy gradient field')
        self.assertEqual(skill_config['qm_atoms'], QIDX)
        self.assertEqual(int(skill_config['keep_tmp']), 1)
        # q_qm belongs to the energy task; the field probe sets Q=+1 and
        # ignores it, so the forwarded molecule.q_qm is left in place
        # rather than cleared.
        self.assertEqual([float(value) for value in skill_config['q_qm']],
                         [0.1, 0.2, -0.3])

    def test_qm_task_gets_the_esfp_field(self):
        state = self._build()
        tasks = [task for _, task in state.iter_tasks()]
        qm = tasks[1]
        self.assertEqual(qm['composite'], 'qmmm')
        self.assertEqual(qm['binding_role'], 'qm')
        self.assertEqual(qm['region_indices'], QIDX)
        skill_config = qm['config']['skill_config']
        self.assertEqual(skill_config['embedding'], 'field')
        self.assertEqual(skill_config['read_charge'], 'none')
        self.assertEqual(skill_config['highlevel'], [])
        self.assertEqual(skill_config['midlevel'], [])
        self.assertEqual(skill_config['natom'], 3)
        self.assertEqual(skill_config['pbc'], [0, 0, 0])

    def test_energy_task_gets_the_profile_and_q_qm(self):
        state = self._build()
        tasks = [task for _, task in state.iter_tasks()]
        energy = tasks[2]
        self.assertEqual(energy['composite'], 'qmmm')
        self.assertEqual(energy['binding_role'], 'qm2')
        skill_config = energy['config']['skill_config']
        self.assertEqual(skill_config['profile'], 'qmmm_energy')
        self.assertEqual(skill_config['jobtype'], 'energy gradient qmmm')
        self.assertEqual(skill_config['qm_atoms'], QIDX)
        self.assertTrue(skill_config['embedding'])
        self.assertEqual([float(value) for value in skill_config['q_qm']],
                         [0.1, 0.2, -0.3])

    def test_read_charge_is_ignored(self):
        state = self._build(read_charge=True)
        tasks = [task for _, task in state.iter_tasks()]
        field, qm, energy = tasks
        self.assertNotIn('request_charge',
                         field['config']['skill_config'])
        # QMMM theory[0] uses the field V/E; a read_charge request does
        # not switch it to charge embedding.
        self.assertEqual(qm['config']['skill_config']['embedding'],
                         'field')
        self.assertTrue(energy['config']['skill_config']['embedding'])

    def test_plan_identity_survives_serialization(self):
        state = self._build()
        entries = state.to_dict()['theory']
        self.assertEqual([entry['qmmm_role'] for entry in entries],
                         ['field', 'qm', 'energy'])
        self.assertEqual(entries[0]['composite'], 'qmmm')
        self.assertEqual(entries[0]['binding_role'], 'qm2')
        self.assertEqual(entries[2]['partition_indices'], QIDX)


class TestEspfCharges(unittest.TestCase):
    """Molcas ESPF 0-order charges -> computed_charges."""

    def _write(self, text):
        import tempfile
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, 'ESPF.DATA')
        with open(path, 'w') as handle:
            handle.write(text)
        return tmp

    def test_two_column_rows(self):
        from pyrai3md.skills.molcas_skill.service.parser.espf_charges import (
            parse_espf_charges)
        calcdir = self._write('# ESPF charges\n1 -0.83\n2 0.415\n'
                              '3 0.415\n')
        self.assertEqual(parse_espf_charges(calcdir, 3),
                         [[-0.83, 0.415, 0.415]])

    def test_one_column_rows(self):
        from pyrai3md.skills.molcas_skill.service.parser.espf_charges import (
            parse_espf_charges)
        calcdir = self._write('header\n-0.5\n0.25\n0.25\n')
        self.assertEqual(parse_espf_charges(calcdir, 3),
                         [[-0.5, 0.25, 0.25]])

    def test_wrong_count_returns_none(self):
        from pyrai3md.skills.molcas_skill.service.parser.espf_charges import (
            parse_espf_charges)
        calcdir = self._write('1 -0.5\n2 0.25\n')
        self.assertIsNone(parse_espf_charges(calcdir, 3))

    # --- the block the deployed build actually prints --------------------
    # Spacing copied from `tmp_20261031_183037/examples/multiscale_qmmm_ts/
    # yaml/runtime/.../tmp_high_qm_MOLCAS/MOLCAS.log`, not from a guess. The
    # module banner matters: a block is attributed to the module running when
    # it was printed, and only `rasscf` prints one per state.

    BANNER = ('--- Start Module: rasscf at Sat Oct 31 18:48:13 2026 ---\n'
              '\n')

    def _block(self, qm, total):
        rows = ''.join('        Charge on %-14s =    %8.4f\n' % pair
                       for pair in qm)
        return ('++    ESPF analysis\n'
                '      -------------\n'
                '\n'
                '      Expectation values of the ESPF operators:\n'
                '\n'
                + rows + '\n'
                '      Total ESPF charge     =     %8.4f\n'
                '\n'
                '      Total ESPF dipole moment (debye):\n'
                '               X=  1.7134E+00\n' % total)

    BLOCK = ('++    ESPF analysis\n'
             '      -------------\n'
             '\n'
             '      Expectation values of the ESPF operators:\n'
             '\n'
             '        Charge on O1          =    -0.7164\n'
             '        Charge on H2          =     0.3605\n'
             '        Charge on H3          =     0.3587\n'
             '\n'
             '      Total ESPF charge     =     0.0028\n'
             '\n'
             '      Total ESPF dipole moment (debye):\n'
             '               X=  1.7134E+00\n')

    def _write_log(self, text, name='MOLCAS.log'):
        import tempfile
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, name), 'w') as handle:
            handle.write(text)
        return tmp

    def test_analysis_block_is_read_from_the_log(self):
        from pyrai3md.skills.molcas_skill.service.parser.espf_charges import (
            parse_espf_charges)
        self.assertEqual(
            parse_espf_charges(self._write_log(self.BANNER + self.BLOCK), 3),
            [[-0.7164, 0.3605, 0.3587]])

    def test_every_root_block_comes_back_in_root_order(self):
        # The shape measured on the deployed `embedding_sp` run: two blocks
        # inside &RASSCF (one per root, in root order), then more from
        # &CASPT2. The reader hands all of the root-bound rows over and does
        # not choose among them: which state embeds is the framework's call,
        # and a single-row answer to a two-state run is what left a crossing
        # search with one state's charges for both surfaces.
        from pyrai3md.skills.molcas_skill.service.parser.espf_charges import (
            parse_espf_charges)
        root1 = self._block([('O1', -0.5990), ('H2', 0.3001),
                             ('H3', 0.3001)], 0.0012)
        root2 = self._block([('O1', 0.1215), ('H2', -0.0480),
                             ('H3', -0.0480)], 0.0255)
        caspt2 = ('--- Start Module: caspt2 at Sat Oct 31 20:00:58 2026 ---\n'
                  + self._block([('O1', -0.5730), ('H2', 0.2873),
                                 ('H3', 0.2873)], 0.0017))
        log = self.BANNER + root1 + root2 + caspt2
        calcdir = self._write_log(log)
        self.assertEqual(parse_espf_charges(calcdir, 3),
                         [[-0.5990, 0.3001, 0.3001],
                          [0.1215, -0.0480, -0.0480]])
        # The block &CASPT2 printed is never reachable: it is a
        # response/multi-state quantity with no state to bind it to.
        self.assertNotIn([-0.5730, 0.2873, 0.2873],
                         parse_espf_charges(calcdir, 3))
        # The reader is not the one that chooses. Trimming the rows to the
        # framework's `nstate` would discard a row this run's own template
        # computed more roots than that field says, and a discarded row is a
        # state the framework can no longer address.
        self.assertEqual(len(parse_espf_charges(calcdir, 3)), 2)

    def test_sections_are_read_in_file_order_across_spin_manifolds(self):
        # Two singlets then one triplet, each &RASSCF section printing its own
        # roots: the rows come back in that order, which is the global
        # 0-based state numbering the framework's `target_states` uses. A
        # reader that reset its count per section could not be addressed by
        # that index at all.
        from pyrai3md.skills.molcas_skill.service.parser.espf_charges import (
            parse_espf_charges)
        singlet = self._block([('O1', -0.5990), ('H2', 0.3001),
                               ('H3', 0.3001)], 0.0012)
        second = self._block([('O1', 0.1215), ('H2', -0.0480),
                              ('H3', -0.0480)], 0.0255)
        triplet_banner = ('--- Start Module: rasscf at '
                          'Sat Oct 31 22:16:54 2026 ---\n\n')
        triplet = self._block([('O1', 0.0545), ('H2', -0.0137),
                               ('H3', -0.0137)], 0.0271)
        log = self.BANNER + singlet + second + triplet_banner + triplet
        self.assertEqual(parse_espf_charges(self._write_log(log), 3),
                         [[-0.5990, 0.3001, 0.3001],
                          [0.1215, -0.0480, -0.0480],
                          [0.0545, -0.0137, -0.0137]])

    def test_block_at_the_end_of_a_module_is_still_counted(self):
        # The last block of a section is closed by the NEXT module's banner,
        # not by a text line, so counting only on prose would drop it.
        from pyrai3md.skills.molcas_skill.service.parser.espf_charges import (
            parse_espf_charges)
        log = (self.BANNER + self._block([('O1', -0.6), ('H2', 0.3),
                                          ('H3', 0.3)], 0.0)
               + '--- Start Module: caspt2 at Sat Oct 31 20:00:58 2026 ---\n'
               + ' nothing else here\n')
        self.assertEqual(
            parse_espf_charges(self._write_log(log), 3),
            [[-0.6, 0.3, 0.3]])

    def test_log_block_of_the_wrong_length_is_refused(self):
        # Not truncated, not padded: three charges cannot answer for four
        # atoms, and a reader that guessed would hand the classical stage a
        # charge set that silently mislabels one atom.
        from pyrai3md.skills.molcas_skill.service.parser.espf_charges import (
            parse_espf_charges)
        calcdir = self._write_log(self.BANNER + self.BLOCK)
        self.assertIsNone(parse_espf_charges(calcdir, 4))
        self.assertIsNone(parse_espf_charges(calcdir, 2))

    def test_no_block_is_no_charges(self):
        from pyrai3md.skills.molcas_skill.service.parser.espf_charges import (
            parse_espf_charges)
        self.assertIsNone(parse_espf_charges(
            self._write_log(self.BANNER + ' RASSCF completed\n'
                            ' Happy landing\n'), 3))

    def test_block_outside_rasscf_is_not_used(self):
        # A CASPT2-section block alone yields nothing: picking it is what
        # handed the classical stage a 0.0276 net charge instead of the
        # active state's 0.0012.
        from pyrai3md.skills.molcas_skill.service.parser.espf_charges import (
            parse_espf_charges)
        log = ('--- Start Module: caspt2 at Sat Oct 31 20:00:58 2026 ---\n'
               + self.BLOCK)
        self.assertIsNone(parse_espf_charges(self._write_log(log), 3))

    def test_sidecar_is_preferred_over_the_log(self):
        from pyrai3md.skills.molcas_skill.service.parser.espf_charges import (
            parse_espf_charges)
        calcdir = self._write_log(self.BANNER + self.BLOCK)
        with open(os.path.join(calcdir, 'ESPF.DATA'), 'w') as handle:
            handle.write('-0.5\n0.25\n0.25\n')
        self.assertEqual(parse_espf_charges(calcdir, 3),
                         [[-0.5, 0.25, 0.25]])

    def test_missing_file_returns_none(self):
        from pyrai3md.skills.molcas_skill.service.parser.espf_charges import (
            parse_espf_charges)
        self.assertIsNone(parse_espf_charges('/no/such/dir', 3))


class TestQmmmChargeSource(unittest.TestCase):
    """An untouched store must not read as a zero q_qm."""

    def test_untouched_store_does_not_answer(self):
        from pyrai3md.data.charges import ChargeStore
        from pyrai3md.workflow.calculator import Calculator
        calc = Calculator()
        calc._state = SimpleNamespace(
            pmd=SimpleNamespace(system=SimpleNamespace(q_qm=[])),
            charge_store=ChargeStore(3, managed_indices=[0, 1, 2]))
        with self.assertRaises(ValueError):
            calc._qmmm_charges({'task_id': 'qm', 'region_indices': [0, 1]},
                               {}, None)

    def test_seeded_store_answers(self):
        from pyrai3md.data.charges import ChargeStore
        from pyrai3md.workflow.calculator import Calculator
        store = ChargeStore(3, managed_indices=[0, 1, 2])
        store.update([0, 1], [0.1, 0.2])
        calc = Calculator()
        calc._state = SimpleNamespace(
            pmd=SimpleNamespace(system=SimpleNamespace(q_qm=[])),
            charge_store=store)
        self.assertEqual(
            calc._qmmm_charges({'task_id': 'qm', 'region_indices': [0, 1]},
                               {}, None),
            [0.1, 0.2])


class TestQmmmReadChargeModes(unittest.TestCase):
    """QMMM ignores read_charge; theory[0] always uses the field V/E."""

    def _data(self, **overrides):
        data = {
            'theory': ['molcas', 'lammps'], 'reference': [],
            'multiscale_type': 'qmmm', 'highlevel': list(QIDX),
            'natom': NATOM, 'atoms': ['O'] * NATOM,
            'embedding': True, 'read_charge': False,
            'constraints': None,
        }
        data.update(overrides)
        return data

    def test_embedding_true_read_charge_true_is_not_repurposed(self):
        # QMMM embeds through the MM field every step, so the switch is
        # cleared rather than read: no charge embedding appears behind it.
        from pyrai3md.schema.validate import validate_system
        out = validate_system(self._data(embedding=True, read_charge=True))
        self.assertFalse(out['read_charge'])

    def test_embedding_false_ignores_read_charge(self):
        from pyrai3md.schema.validate import validate_system
        out = validate_system(self._data(embedding=False,
                                         read_charge=True))
        self.assertFalse(out['read_charge'])

    def test_standalone_ignores_embedding_either_way(self):
        # `embedding` is a multiscale switch: with one theory and no region
        # there is nothing for it to steer, and the run is embedded exactly
        # when read_charge names the file. Neither value of embedding is read,
        # so both spellings keep the switch on.
        from pyrai3md.schema.validate import validate_system
        base = {'theory': ['molcas'], 'reference': [],
                'multiscale_type': '', 'highlevel': [],
                'natom': NATOM, 'atoms': ['O'] * NATOM,
                'read_charge': True, 'constraints': None}
        for extra in ({}, {'embedding': True}, {'embedding': False}):
            out = validate_system(dict(base, **extra))
            self.assertTrue(out['read_charge'], extra)

    def test_qmmm_drops_read_charge_without_a_warning(self):
        # QMMM takes its field from the MM stage every step, so the switch is
        # not consulted - and, being not consulted, its two values are not
        # told apart: no warning, no refusal, just a cleared flag.
        import warnings
        from pyrai3md.schema.validate import validate_system
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = validate_system(self._data(embedding=True,
                                             read_charge=True))
        self.assertFalse(out['read_charge'])
        self.assertEqual([str(item.message) for item in caught], [])


class TestLammpsInertMirrors(unittest.TestCase):
    """The framework's string embedding values reach the LAMMPS schema."""

    def test_string_values_accepted(self):
        from pyrai3md.skills.lammps_skill.schema.config import System
        system = System(embedding='none', read_charge='none')
        self.assertEqual(system.embedding, 'none')
        self.assertEqual(system.read_charge, 'none')

    def test_qmmm_keywords_declared(self):
        from pyrai3md.planning.assembly import _skill_declares
        self.assertTrue(_skill_declares('lammps_skill', 'qm_atoms'))
        self.assertTrue(_skill_declares('lammps_skill', 'q_qm'))
        self.assertTrue(_skill_declares('molcas_skill', 'embedding'))
        self.assertTrue(_skill_declares('molcas_skill', 'read_charge'))


class TestQmmmChargeBalancing(unittest.TestCase):
    """The ESPF fit's charge residual is averaged out; nothing else is."""

    def _calc(self, charges):
        from pyrai3md.workflow.calculator import Calculator
        calc = Calculator()
        calc._state = SimpleNamespace(
            pmd=SimpleNamespace(system=SimpleNamespace(q_qm=[],
                                                        charges=charges)))
        return calc

    def test_readback_residual_is_averaged_over_the_region(self):
        # Measured on the deployed build for the active state of a neutral
        # water: -0.5990 / 0.3001 / 0.3001, a 1.2e-3 residual that no
        # denser grid removes. qmmm_energy needs a neutral cell, so the
        # residual is spread evenly and the shape of the fit is kept.
        calc = self._calc(0)
        values = calc._qmmm_charges(
            {'task_id': 'qm', 'region_indices': [0, 1, 2]},
            {'computed_charges': [[-0.5990, 0.3001, 0.3001]]}, None)
        self.assertAlmostEqual(sum(values), 0.0, places=12)
        for value, wanted in zip(values, (-0.5994, 0.2997, 0.2997)):
            self.assertAlmostEqual(value, wanted, places=12)

    def test_a_declared_q_qm_is_never_rescaled(self):
        # The witness that the averaging is bound to the readback: the same
        # 1.2e-3-style residual arriving through the static fallback is
        # returned verbatim, because rewriting a number the input spelled
        # out is not the framework's call to make.
        calc = self._calc(0)
        energy_task = {'config': {'skill_config': {'q_qm':
                                                   [0.1, 0.2, -0.25]}}}
        self.assertEqual(
            calc._qmmm_charges({'task_id': 'qm', 'region_indices': [0, 1, 2]},
                               {}, energy_task),
            [0.1, 0.2, -0.25])

    def test_residual_inside_the_print_resolution_is_left_alone(self):
        calc = self._calc(0)
        values = calc._qmmm_charges(
            {'task_id': 'qm', 'region_indices': [0, 1, 2]},
            {'computed_charges': [[-0.5990, 0.30005, 0.29895]]}, None)
        self.assertEqual(values, [-0.5990, 0.30005, 0.29895])

    def test_a_non_zero_target_is_matched_not_zeroed(self):
        # A cation's QM region must come back at +1, not at neutral.
        calc = self._calc(1)
        values = calc._qmmm_charges(
            {'task_id': 'qm', 'region_indices': [0, 1, 2]},
            {'computed_charges': [[0.6000, 0.2000, 0.2012]]}, None)
        self.assertAlmostEqual(sum(values), 1.0, places=12)

    def test_a_per_center_charge_list_is_not_guessed_at(self):
        # Multi-center runs state one charge per center; which entry owns
        # this region is undefined, so the fit is handed on unchanged and
        # the classical stage's own refusal is what reports it.
        calc = self._calc([0, 1])
        values = calc._qmmm_charges(
            {'task_id': 'qm', 'region_indices': [0, 1, 2]},
            {'computed_charges': [[-0.5990, 0.3001, 0.3001]]}, None)
        self.assertEqual(values, [-0.5990, 0.3001, 0.3001])


class TestPerStateChargeRows(unittest.TestCase):
    """A per-state charge payload is addressed by state, not by row 0."""

    ROWS = [[-0.5990, 0.3001, 0.3001], [0.1215, -0.0480, -0.0480]]

    def _calc(self):
        from pyrai3md.workflow.calculator import Calculator
        calc = Calculator()
        calc._state = SimpleNamespace(
            pmd=SimpleNamespace(system=SimpleNamespace(
                q_qm=[0.1, 0.2, -0.3], charges=0)),
            charge_store=None)
        return calc

    def _ask(self, state):
        return self._calc()._qmmm_charges(
            {'task_id': 'qm', 'region_indices': [0, 1, 2]},
            {'computed_charges': self.ROWS}, None, state=state)

    def test_the_requested_state_gets_its_own_row(self):
        # Both rows come off one run: root 1 and root 2 of the deployed
        # `embedding_sp` log. Taking row 0 whatever the state says is how a
        # two-state search ended up embedding both surfaces with the ground
        # state's charges, so the two answers have to differ.
        first, second = self._ask(0), self._ask(1)
        self.assertEqual([round(value, 4) for value in first],
                         [-0.5994, 0.2997, 0.2997])
        self.assertGreater(second[0], 0.0)
        self.assertAlmostEqual(sum(second), 0.0, places=9)

    def test_a_state_the_run_never_printed_is_refused_by_name(self):
        # Not row 0, and not a silent zero: the readback is declined and the
        # fallback chain (charge store, then the declared q_qm) answers.
        self.assertEqual(self._ask(5), [0.1, 0.2, -0.3])


class TestCrossingRequestsTheCoupling(unittest.TestCase):
    """A search that spans the branching plane asks the engine for it."""

    @classmethod
    def setUpClass(cls):
        from pyrai3md.kernel.environment import Environment
        Environment().initialize()

    def _jobtypes(self, method, theory=('molcas', 'lammps'), i=0, j=1):
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.config import PyRAIConfig
        config = PyRAIConfig()
        config.control.jobtype = 'optimize'
        config.control.theory = list(theory)
        if len(theory) > 1:
            config.control.multiscale_type = 'qmmm'
        config.system.atoms = ['O', 'H', 'H', 'O', 'H', 'H']
        config.system.coordinates = [
            [1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0],
            [5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.5],
        ]
        config.system.natom = 6
        config.system.charges = 0
        config.system.multiplicity = 1
        config.system.nstate = 2
        config.system.state_spin = [0, 0]
        # The declaration is what says which coupling is computed at all: a
        # crossing search that needs one names its pair here, and a pair left
        # out is shielded rather than guessed (docs MULTICENTER 4.4.5).
        config.system.coupling = [[i, j]]
        if len(theory) > 1:
            config.system.highlevel = [0, 1, 2]
            config.system.embedding = True
        config.optimize.opt_method = method
        config.optimize.opt_state_i = i
        config.optimize.opt_state_j = j
        state = build_state(config)
        return {task.get('qmmm_role') or task['skill_name']:
                task['config']['skill_config'].get('jobtype')
                for _, task in state.iter_tasks()}

    def test_meci_asks_and_the_classical_legs_do_not(self):
        jobs = self._jobtypes('meci')
        self.assertEqual(jobs['qm'], 'energy gradient nac')
        # The token rides on the leg that can answer it. A force field has no
        # derivative coupling, and handing it one spells a jobtype its own
        # profile resolution refuses.
        self.assertEqual(jobs['field'], 'energy gradient field')
        self.assertEqual(jobs['energy'], 'energy gradient qmmm')

    def test_mecp_never_asks(self):
        # A spin-forbidden pair has no derivative coupling to span the plane
        # with, and the optimizer says so through `uses_derivative_coupling`.
        self.assertEqual(self._jobtypes('mecp')['qm'], 'energy gradient')

    def test_a_skill_without_couplings_is_not_asked(self):
        jobs = self._jobtypes('meci', theory=('xtb',))
        self.assertEqual(list(jobs.values()), ['energy gradient'])


MC_NATOM = 6
MC_A = [0, 1]
MC_B = [3, 4]
MC_UNION = [0, 1, 3, 4]
# The classical main rows: 1.0 on center A, 2.0 on center B, 5.0 on MM.
MC_MAIN = {0: 1.0, 1: 1.0, 2: 5.0, 3: 2.0, 4: 2.0, 5: 5.0}


def _mc_task(task_id, role, contribution, region, indices, center,
             skill='lammps_skill'):
    return {'task_id': task_id, 'role': role, 'qmmm_role': role,
            'contribution': contribution, 'region': region,
            'region_indices': list(indices), 'center_index': center,
            'skill_name': skill, 'provides_electronic': role == 'qm',
            'config': {'skill_config': {'embedding': 'field'}}}


def _mc_field_block(indices, potentials):
    return {
        'qm_atoms': list(indices),
        'qm_atom_ids': [index + 1 for index in indices],
        'probe_charge': 1.0,
        'v_mm_native': np.array(potentials, dtype=np.float64),
        'e_mm_native': np.zeros((len(indices), 3)),
        'sanity': {'v_vs_u_gg': 0.0, 'e_vs_f_gg': 0.0},
        'units': {'energy': 'real', 'force': 'real'},
    }


def _mc_energy_raw(el_x, gas_x, q_qm=(-0.1, 0.1, 0.3, 0.4),
                   u_main=-10.0, u_gas=-4.0, qmmm_el=-3.5, e_classic=-2.5,
                   centers=None):
    """One whole-system classical surface over the union of the centers.

    ``el_x``/``gas_x`` are the four union-row x components of the two
    removed gradients, in QM-set order; the returned ``gradient`` is the
    reduced array the skill hands over (main - el - gas on the Q rows,
    untouched elsewhere), so the fixture states the backend's arithmetic
    rather than re-deriving it in the assembler.
    """
    el = np.zeros((len(MC_UNION), 3))
    gas = np.zeros((len(MC_UNION), 3))
    for row in range(len(MC_UNION)):
        el[row, 0] = float(el_x[row])
        gas[row, 0] = float(gas_x[row])
    main = np.array([[MC_MAIN[index]] * 3 for index in range(MC_NATOM)])
    reduced = main.copy()
    for row, index in enumerate(MC_UNION):
        reduced[index] = main[index] - el[row] - gas[row]
    atomic = {
        'e_classic': e_classic,
        'u_main': u_main,
        'u_gas': u_gas,
        'qmmm_el': qmmm_el,
        'qmmm_el_gradient': el,
        'u_gas_gradient': gas,
        'native_unit_set': 'real',
    }
    if centers is not None:
        atomic['u_gas_centers'] = list(centers)
    return {
        'energy': np.array([u_main]),
        'gradient': reduced[np.newaxis, :, :],
        'qmmm': {
            'qm_atoms': list(MC_UNION),
            'qm_atom_ids': [index + 1 for index in MC_UNION],
            'q_qm': list(q_qm),
            'embedding': True,
            'sanity': {'parts': 0.0},
            'units': {'energy': 'real', 'force': 'real'},
        },
        'qmmm_atomic': atomic,
    }


def _mc_qm_raw(energy, value, charges):
    n = len(energy) if isinstance(energy, (list, tuple)) else 1
    indices = MC_A if value == 'A' else MC_B
    return {
        'energy': np.array(energy if isinstance(energy, (list, tuple))
                           else [energy]),
        'gradient': np.full((n, len(indices), 3),
                            1.0 if value == 'A' else 2.0),
        'computed_charges': np.array([list(charges)]),
    }


class TestAssembleQmmmTwoCenters(unittest.TestCase):
    """C=2 through one assembly, not the single-center one run twice.

    The arithmetic is written out in absolute numbers on purpose: the point
    of this class is that each center keeps its own energy and gradient rows
    while the classical surface, its two reductions and the identity guard
    stay single.
    """

    def _run(self, energy_b=-20.0, q_b=(0.3, 0.4), grad_b=2.0,
             energy_a=-10.0,
             el=(0.1, 0.1, 0.1, 0.1), gas=(0.2, 0.2, 0.2, 0.2),
             centers=(-1.5, -2.5), configuration=None, multi_center=False,
             nac_a=None, nac_b=None, pair_axis=None):
        tasks = [
            _mc_task('field', 'field', 0, 'all', [], 0),
            _mc_task('field_task1', 'field', 0, 'all', [], 1),
            _mc_task('qm', 'qm', 1, 'model', MC_A, 0, skill='molcas_skill'),
            _mc_task('qm_task1', 'qm', 1, 'model', MC_B, 1,
                     skill='molcas_skill'),
            _mc_task('energy', 'energy', 1, 'all', [], None),
        ]
        energy_b_values = np.asarray(energy_b, dtype=float).reshape(-1)
        q_b_raw = {
            'energy': energy_b_values,
            'gradient': np.full((energy_b_values.size, len(MC_B), 3),
                                grad_b),
            'computed_charges': np.array([list(q_b)]),
        }
        outcomes = [
            _outcome(tasks[0], {
                'energy': np.array([0.0]),
                'gradient': np.zeros((1, MC_NATOM, 3)),
                'qmmm_field': _mc_field_block(MC_A, [-1.0, -2.0])}),
            _outcome(tasks[1], {
                'energy': np.array([0.0]),
                'gradient': np.zeros((1, MC_NATOM, 3)),
                'qmmm_field': _mc_field_block(MC_B, [-3.0, -4.0])}),
            _outcome(tasks[2], self._qm_raw(energy_a, 'A', (-0.1, 0.1),
                                            nac_a)),
            _outcome(tasks[3], self._with_nac(q_b_raw, nac_b)),
            _outcome(tasks[4], _mc_energy_raw(
                el, gas, q_qm=(-0.1, 0.1) + tuple(q_b), centers=centers)),
        ]
        request = {'natom': MC_NATOM, 'nstate': 1, 'need_gradient': True}
        if configuration is not None:
            request['active_configuration'] = tuple(configuration)
        if multi_center:
            request['multi_center'] = True
        if pair_axis is not None:
            request['pair_axis'] = [list(pick) for pick in pair_axis]
        return assemble_qmmm(tasks, outcomes, request)

    @staticmethod
    def _qm_raw(energy, value, charges, nac=None):
        raw = _mc_qm_raw(energy, value, charges)
        if nac is not None:
            raw['nac'] = np.asarray(nac, dtype=np.float64)
        return raw

    @staticmethod
    def _with_nac(raw, nac):
        if nac is None:
            return raw
        tagged = dict(raw)
        tagged['nac'] = np.asarray(nac, dtype=np.float64)
        return tagged

    def test_energy_is_both_centers_plus_one_classical_surface(self):
        data = self._run()
        # -10 + -20 + -2.5, with only one E_classic in the sum.
        self.assertAlmostEqual(float(data['energy'][0]), -32.5)
        components = data['qmmm_components']
        self.assertEqual(components['n_center'], 2)
        self.assertEqual(components['qm_atoms'], MC_UNION)
        self.assertEqual(components['qm_energy'], [-30.0])
        self.assertAlmostEqual(components['e_classic'], -2.5)
        self.assertAlmostEqual(components['u_gas'], -4.0)

    def test_components_report_each_center_separately(self):
        centers = self._run()['qmmm_components']['centers']
        self.assertEqual([entry['center_index'] for entry in centers],
                         [0, 1])
        self.assertEqual([entry['qm_atoms'] for entry in centers],
                         [MC_A, MC_B])
        self.assertEqual([entry['qm_energy'] for entry in centers],
                         [[-10.0], [-20.0]])
        self.assertEqual(centers[1]['q_qm'], [0.3, 0.4])
        # The in-cell removal is a per-center number, not just a total.
        self.assertEqual([entry['u_gas'] for entry in centers],
                         [-1.5, -2.5])

    def test_couplings_land_as_one_block_per_center(self):
        # Each center answers its own pair on its own atoms; the composite
        # keeps the blocks apart and leaves every other row zero. Nothing is
        # summed across centers, which is why no aggregation rule is needed.
        nac_a = np.ones((1, len(MC_A), 3))
        nac_b = np.full((1, len(MC_B), 3), 2.0)
        data = self._run(multi_center=True, nac_a=nac_a, nac_b=nac_b)
        nac = np.asarray(data['nac'])
        self.assertEqual(nac.shape, (2, MC_NATOM, 3))
        np.testing.assert_allclose(nac[0, MC_A, :], 1.0)
        np.testing.assert_allclose(nac[1, MC_B, :], 2.0)
        quiet = [index for index in range(MC_NATOM)
                 if index not in MC_UNION]
        np.testing.assert_allclose(nac[:, quiet, :], 0.0)
        # Each block owns its own pair: center A's rows are dark in pair 2.
        np.testing.assert_allclose(nac[1, MC_A, :], 0.0)
        np.testing.assert_allclose(nac[0, MC_B, :], 0.0)

    def test_a_pair_axis_builds_two_composite_surfaces(self):
        # Crossing search at C=2: each center names its own pair, and the
        # objective is the difference of the two composite surfaces - which is
        # the SUM of the per-center gaps. The classical term appears once, on
        # both surfaces, because it carries no electronic state.
        data = self._run(energy_a=[-10.0, -12.0], energy_b=[-20.0, -21.0],
                         multi_center=True, pair_axis=[[0, 1], [1, 0]])
        energy = np.asarray(data['energy'])
        self.assertEqual(energy.shape, (2,))
        self.assertAlmostEqual(float(energy[0]), -10.0 + -21.0 + -2.5)
        self.assertAlmostEqual(float(energy[1]), -12.0 + -20.0 + -2.5)
        gradient = np.asarray(data['gradient'])
        self.assertEqual(gradient.shape, (2, MC_NATOM, 3))
        # The MM rows are the same on both surfaces.
        quiet = [index for index in range(MC_NATOM) if index not in MC_UNION]
        np.testing.assert_allclose(gradient[0, quiet, :],
                                   gradient[1, quiet, :])
        # The gap the search follows is the sum of the per-center gaps.
        gap = float(energy[0] - energy[1])
        self.assertAlmostEqual(
            gap, (-10.0 - -20.0) - (-12.0 - -21.0))

    def test_each_center_answers_at_its_own_active_state(self):
        # Two ladders of two states, and an active configuration that names a
        # different state for each: the scalar is E_classic plus each center's
        # own pick, not the difference of two composite surfaces.
        data = self._run(energy_a=[-10.0, -12.0],
                         energy_b=[-20.0, -21.0],
                         multi_center=True, configuration=(1, 0))
        self.assertAlmostEqual(float(data['energy'][0]),
                               -12.0 + -20.0 + -2.5)
        self.assertEqual(tuple(data['active_configuration']), (1, 0))

    def test_a_state_outside_a_centers_ladder_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self._run(energy_a=[-10.0, -12.0], energy_b=[-20.0],
                      multi_center=True, configuration=(0, 1))
        self.assertIn('names state 1', str(caught.exception))

    def test_each_center_keeps_its_own_gradient_rows(self):
        gradient = np.asarray(self._run()['gradient'])[0]
        # A: QM 1.0 + (main 1.0 - F_el 0.1 - F_gas 0.2) = 1.7 on x, and the
        # unreduced y/z of the main row plus the QM value.
        np.testing.assert_allclose(gradient[MC_A[0]], [1.7, 2.0, 2.0])
        np.testing.assert_allclose(gradient[MC_A[1]], [1.7, 2.0, 2.0])
        # B: QM 2.0 + (main 2.0 - 0.3) = 3.7.
        np.testing.assert_allclose(gradient[MC_B[0]], [3.7, 4.0, 4.0])
        np.testing.assert_allclose(gradient[MC_B[1]], [3.7, 4.0, 4.0])
        # MM rows keep the full U_main rows.
        np.testing.assert_allclose(gradient[[2, 5], :], 5.0)

    def test_composite_terms_name_the_centers(self):
        terms = self._run()['composite_terms']
        self.assertEqual(terms['labels'],
                         ['high[qm][1]', 'high[qm][2]', 'all[energy]'])
        self.assertEqual(terms['contributions'], [1, 1, 1])
        self.assertEqual([row[0] for row in terms['energies']],
                         [-10.0, -20.0, -2.5])

    def test_the_active_surface_configuration_survives_the_assembly(self):
        # A multi-center MD step stages the surface the next step predicts
        # from, and that surface is indexed per center. The configuration is
        # what the request was built with, so the assembly has to hand it
        # back: dropping it left the second step of a two-center QMMM run
        # indexing an empty tuple.
        data = self._run(configuration=(0, 0))
        self.assertEqual(data['active_configuration'], (0, 0))
        self.assertAlmostEqual(float(data['active_energy']), -32.5)
        np.testing.assert_allclose(
            np.asarray(data['active_gradient'], dtype=float),
            np.asarray(data['gradient'], dtype=float)[0])

    def test_a_request_without_a_configuration_is_not_answered_with_one(self):
        # A caller that does not name the per-center states (the single-center
        # and legacy spellings) gets an empty configuration back, not a tuple
        # the assembly invented: staging an invented (0, 0) would make the
        # next step's hops read a center state nobody chose.
        data = self._run()
        self.assertEqual(data['active_configuration'], ())

    def test_second_center_with_nothing_to_contribute(self):
        # Degenerate witness: no energy, no charge and no coupling from the
        # second center leaves the first center's rows and the total exactly
        # what a one-center run on the same main rows gives.
        data = self._run(energy_b=0.0, q_b=(0.0, 0.0), grad_b=0.0,
                         el=(0.1, 0.1, 0.0, 0.0),
                         gas=(0.2, 0.2, 0.0, 0.0))
        gradient = np.asarray(data['gradient'])[0]
        np.testing.assert_allclose(gradient[MC_A[0]], [1.7, 2.0, 2.0])
        # B's rows then hold the untouched main row, and nothing else.
        np.testing.assert_allclose(gradient[MC_B[0]], [2.0, 2.0, 2.0])
        self.assertAlmostEqual(float(data['energy'][0]), -12.5)

    def test_overlapping_centers_are_refused(self):
        tasks = [_mc_task('qm', 'qm', 1, 'model', [0, 1], 0,
                         skill='molcas_skill'),
                 _mc_task('qm_task1', 'qm', 1, 'model', [1, 3], 1,
                          skill='molcas_skill'),
                 _mc_task('energy', 'energy', 1, 'all', [], None)]
        outcomes = [
            _outcome(tasks[0], _mc_qm_raw(-1.0, 'A', (0.1, 0.1))),
            _outcome(tasks[1], {
                'energy': np.array([-1.0]),
                'gradient': np.zeros((1, 2, 3)),
                'computed_charges': np.array([[0.2, 0.2]])}),
            _outcome(tasks[2], _mc_energy_raw(
                (0.0,) * 4, (0.0,) * 4, q_qm=(0.1, 0.1, 0.2, 0.2))),
        ]
        with self.assertRaises(ValueError) as caught:
            assemble_qmmm(tasks, outcomes,
                          {'natom': MC_NATOM, 'nstate': 1,
                           'need_gradient': False})
        self.assertIn('overlap', str(caught.exception))

    def test_classical_identity_is_still_one_statement(self):
        # U_main - U_gas - qmmm_el must equal E_classic whatever the center
        # count: the guard is about the classical surface, not per center.
        tasks = [_mc_task('qm', 'qm', 1, 'model', MC_A, 0,
                          skill='molcas_skill'),
                 _mc_task('qm_task1', 'qm', 1, 'model', MC_B, 1,
                          skill='molcas_skill'),
                 _mc_task('energy', 'energy', 1, 'all', [], None)]
        outcomes = [
            _outcome(tasks[0], _mc_qm_raw(-1.0, 'A', (-0.1, 0.1))),
            _outcome(tasks[1], _mc_qm_raw(-1.0, 'B', (0.3, 0.4))),
            _outcome(tasks[2], _mc_energy_raw((0.0,) * 4, (0.0,) * 4,
                                              e_classic=-9.9)),
        ]
        with self.assertRaises(ValueError) as caught:
            assemble_qmmm(tasks, outcomes,
                          {'natom': MC_NATOM, 'nstate': 1,
                           'need_gradient': False})
        self.assertIn('do not add up', str(caught.exception))


class _MccFieldBackend:
    """One center's A/B probe: records the charge set it was fed.

    The field a center is solved in is the only channel the other centers
    reach it through, so the probe is where a charge round's output has to
    reappear. Each instance writes the `.esfp` handoff for one center's
    atoms and keeps every ``q_qm`` array it was called with.
    """

    center_ids = ()
    potentials = ()

    def __init__(self, config=None, execution_id=''):
        self.config = config
        self.calls = 0
        self.feeds = []
        self.calcdir = tempfile.mkdtemp(prefix='qmmm_mcc_field_')

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None, qmmm_inputs=None):
        self.calls += 1
        self.feeds.append([float(value) for value in
                           ((qmmm_inputs or {}).get('q_qm') or [])])
        with open(os.path.join(self.calcdir, 'probe.esfp'), 'w') as handle:
            handle.write('# PyRAI3MD ESPF probe (atomic units)\n')
            for atom_id, value in zip(self.center_ids, self.potentials):
                handle.write('%d %s\n' % (
                    atom_id,
                    ' '.join(f'{number:.16e}' for number in
                             (value, 0.0, 0.0, 0.0))))
        return {
            'energy': np.array([0.0]),
            'gradient': -np.zeros((len(atoms), 3)),
            'qmmm_field': _mc_field_block(
                [index - 1 for index in self.center_ids],
                list(self.potentials)),
            '_native_units': 'real',
        }


def _mcc_field_backend(center_ids, potentials):
    """Bind one probe fixture to the ids and V rows it answers with."""
    return type('MccField', (_MccFieldBackend,), {
        'center_ids': tuple(center_ids),
        'potentials': tuple(potentials),
    })


class _MccChargeBackend:
    """A center's cheap 0-order round: no gradient, charges per round.

    The scripted sequence is the round's answer, not the test's premise:
    what the test reads is the residual the loop derived from these rows
    and the field the next probe was fed.
    """

    sequence = ()
    supports_external_potential = True

    def __init__(self, config=None, execution_id=''):
        self.config = config
        self.calls = 0
        self.fields = []

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None, external_potential=None):
        self.calls += 1
        self.fields.append(external_potential)
        charges = list(self.sequence)[
            self.calls - 1] if self.calls <= len(self.sequence) \
            else list(self.sequence)[-1]
        return {
            'energy': np.array([-1.0]),
            'computed_charges': np.array([list(charges)]),
        }


def _mcc_charge_backend(sequence):
    return type('MccCharge', (_MccChargeBackend,), {
        'sequence': tuple(tuple(group) for group in sequence),
    })


class _MccQmBackend:
    """A center's property round: the same embedded solve, with gradient."""

    supports_external_potential = True

    def __init__(self, config=None, execution_id=''):
        self.config = config
        self.calls = 0
        self.fields = []

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None, external_potential=None):
        self.calls += 1
        self.fields.append(external_potential)
        return {
            'energy': np.array([-10.0]),
            'gradient': np.zeros((1, len(atoms), 3)),
            'computed_charges': np.zeros((1, len(atoms))),
        }


class _MccEnergyBackend:
    """The whole-system classical surface, fed the union charge set.

    It answers the way the LAMMPS skill does: one native ``qmmm`` block in
    the engine's unit set, from which the calculator builds the atomic
    block. Writing the atomic block here would test the fixture.
    """

    centers_native = (-1.5, -2.5)

    def __init__(self, config=None, execution_id=''):
        self.config = config
        self.calls = 0
        self.feeds = []

    def calculate(self, atoms, coordinates, point_charges=None,
                  target_states=None, cell=None, qmmm_inputs=None):
        self.calls += 1
        self.feeds.append([float(value) for value in
                           ((qmmm_inputs or {}).get('q_qm') or [])])
        el = np.zeros((len(MC_UNION), 3))
        gas = np.zeros((len(MC_UNION), 3))
        main = np.array([[MC_MAIN[index]] * 3 for index in range(MC_NATOM)])
        reduced = main.copy()
        for row, index in enumerate(MC_UNION):
            reduced[index] = main[index] - el[row] - gas[row]
        block = {
            'qm_atoms': list(MC_UNION),
            'qm_atom_ids': [index + 1 for index in MC_UNION],
            'q_qm': list((qmmm_inputs or {}).get('q_qm') or []),
            'embedding': True,
            # U_main -10 = U_gas -4 + qmmm_el -3.5 + E_classic -2.5.
            'u_main_native': -10.0,
            'u_gas_native': -4.0,
            'qmmm_el_native': -3.5,
            'e_classic_native': -2.5,
            'u_gas_gradient_native': gas,
            'qmmm_el_gradient_native': el,
            'sanity': {'parts': 0.0},
            'units': {'energy': 'real', 'force': 'real'},
        }
        if self.centers_native:
            block['u_gas_centers_native'] = list(self.centers_native)
        return {
            'energy': np.array([-10.0]),
            'gradient': reduced[np.newaxis, :, :],
            'qmmm': block,
            '_native_units': 'real',
        }


def _mcc_energy_backend(centers=(-1.5, -2.5)):
    return type('MccEnergy', (_MccEnergyBackend,), {
        'centers_native': tuple(centers) if centers else None,
    })


class TestQmmmChargeIteration(unittest.TestCase):
    """The multi-center charge rounds: cold start, snapshot, stop reason.

    The queue here is what the planner expands for two centers - two
    probes, two cheap rounds, two property rounds, one classical surface -
    and the numbers read are the ones the loop recorded, not the ones the
    fixture scripted.
    """

    def _make_state(self, sc_charge, maxiter=10, threshold=0.02,
                    centers=(MC_A, MC_B)):
        from pyrai3md.data.center import build_center_plan
        from pyrai3md.data.state import PyRAIState
        from pyrai3md.data.runtime.state import RuntimeState
        from pyrai3md.schema.config import PyRAIConfig

        config = PyRAIConfig()
        config.system.atoms = ['O', 'H', 'O', 'H', 'H', 'H']
        config.system.coordinates = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [3.0, 3.0, 3.0], [4.0, 3.0, 3.0], [3.0, 4.0, 3.0],
            [9.0, 9.0, 9.0],
        ]
        config.system.natom = MC_NATOM
        config.system.charges = 0
        config.system.multiplicity = 1
        config.system.nstate = 1
        config.system.embedding = True
        config.system.read_charge = False
        config.system.highlevel = [list(group) for group in centers]
        config.system.sc_charge = sc_charge
        config.system.sc_charge_maxiter = maxiter
        config.system.sc_charge_threshold = threshold
        state = PyRAIState(runtime=RuntimeState(total_steps=1))
        state.pmd = config
        centers = [list(group) for group in config.system.highlevel]
        state.bind_partition_plan(build_center_plan({
            'highlevel': centers,
            'ci': [[1]] * len(centers),
            'spin': [[0]] * len(centers),
            'root': [0] * len(centers),
            'charges': [0] * len(centers),
        }))
        return state

    def _task(self, task_id, role, api, indices, center, contribution,
              skill_name, slot):
        return {
            'role': 'theory',
            'task': 0,
            'task_id': task_id,
            'task_slot': slot,
            'layer_id': 0 if role == 'qm' else 2,
            'static_id': 'tmp_%s_MOCK' % task_id,
            'contribution': contribution,
            'region': 'model' if indices else 'all',
            'region_indices': list(indices),
            'region_charge': None,
            'region_multiplicity': None,
            'skill_name': skill_name,
            'api': api,
            'capabilities': (['energy'] if role == 'qm_charge'
                             else ['energy', 'gradient']),
            'config': {'results': {}},
            'center_index': center,
            'provides_electronic': role == 'qm',
            'provides_charges': role == 'qm_charge',
            'composite': 'qmmm',
            'qmmm_role': role,
            'binding_role': role,
            'partition_indices': list(MC_UNION),
            'plan_version': 1,
            'plan_fingerprint': None,
            'property_request_spec': None,
        }

    def _queue(self, field_a, field_b, charge_a, charge_b):
        return [
            self._task('field', 'field', field_a, [], 0, 0,
                       'lammps_skill', 0),
            self._task('field_task1', 'field', field_b, [], 1, 0,
                       'lammps_skill', 1),
            self._task('qm_charge', 'qm_charge', charge_a, MC_A, 0, 0,
                       'molcas_skill', 2),
            self._task('qm_charge_task1', 'qm_charge', charge_b, MC_B, 1, 0,
                       'molcas_skill', 3),
            self._task('qm', 'qm', _MccQmBackend, MC_A, 0, 1,
                       'molcas_skill', 4),
            self._task('qm_task1', 'qm', _MccQmBackend, MC_B, 1, 1,
                       'molcas_skill', 5),
            self._task('energy', 'energy', _mcc_energy_backend(), [], None, 1,
                       'lammps_skill', 6),
        ]

    def _run(self, sc_charge, maxiter=10, threshold=0.02,
             charges_a=((-0.2, 0.2), (-0.1, 0.1), (-0.09, 0.09)),
             charges_b=((0.4, -0.4), (0.2, -0.2), (0.19, -0.19))):
        from pyrai3md.workflow.calculator import Calculator
        state = self._make_state(sc_charge, maxiter, threshold)
        for task in self._queue(
                _mcc_field_backend([1, 2], [-1.0, -2.0]),
                _mcc_field_backend([4, 5], [-3.0, -4.0]),
                _mcc_charge_backend(charges_a),
                _mcc_charge_backend(charges_b)):
            state.add_task('theory', task)
        calculator = Calculator(state=state).initialize()
        result = calculator.run()
        return calculator, result

    def test_first_probe_is_fed_zero_charges(self):
        # R7 cold start: the first round cannot read the store, because a
        # charge carried over from the previous frame would make this
        # frame's answer depend on the trajectory it is supposed to predict.
        calculator, _result = self._run(sc_charge=False)
        probes = [backend for backend in calculator.backends['theory']
                  if isinstance(backend, _MccFieldBackend)]
        self.assertEqual(len(probes), 2)
        for probe in probes:
            self.assertEqual(probe.feeds[0], [0.0, 0.0, 0.0, 0.0])

    def test_one_round_when_sc_charge_is_off(self):
        calculator, result = self._run(sc_charge=False)
        iteration = result['qmmm_charge_iteration']
        self.assertFalse(iteration['sc_charge'])
        self.assertEqual(iteration['rounds_run'], 1)
        self.assertEqual(iteration['maxiter'], 1)
        self.assertEqual(iteration['stopped_by'], 'single round')
        rounds = [backend.calls for backend in calculator.backends['theory']
                  if isinstance(backend, _MccChargeBackend)]
        self.assertEqual(rounds, [1, 1])
        # The single round's charges are the field source of the property
        # round, so the last probe call must already see them.
        probes = [backend for backend in calculator.backends['theory']
                  if isinstance(backend, _MccFieldBackend)]
        self.assertEqual(probes[0].calls, 2)
        self.assertEqual(probes[0].feeds[1], [-0.2, 0.2, 0.4, -0.4])

    def test_one_round_when_asked_for_one(self):
        # `sc_charge: true` with `sc_charge_maxiter: 1` is the shipped
        # `multicenter_qmmm_sp.yaml`: the iteration is asked for, and the
        # budget says one. It is not a stall, so the log must not read
        # `maxiter` - that word would tell a reader the fixed point escaped.
        calculator, result = self._run(sc_charge=True, maxiter=1)
        iteration = result['qmmm_charge_iteration']
        self.assertTrue(iteration['sc_charge'])
        self.assertEqual(iteration['rounds_run'], 1)
        self.assertEqual(iteration['maxiter'], 1)
        self.assertFalse(iteration['converged'])
        self.assertEqual(iteration['stopped_by'], 'single round')
        rounds = [backend.calls for backend in calculator.backends['theory']
                  if isinstance(backend, _MccChargeBackend)]
        self.assertEqual(rounds, [1, 1])
        probes = [backend for backend in calculator.backends['theory']
                  if isinstance(backend, _MccFieldBackend)]
        self.assertEqual(probes[0].calls, 2)
        self.assertEqual(probes[0].feeds[1], [-0.2, 0.2, 0.4, -0.4])

    def test_residuals_and_stop_by_tolerance(self):
        calculator, result = self._run(sc_charge=True, threshold=0.02)
        iteration = result['qmmm_charge_iteration']
        self.assertTrue(iteration['sc_charge'])
        # Center A: 0.2, 0.1, 0.01; center B: 0.4, 0.2, 0.01. The round is
        # accepted by the largest of the two, which is what makes the
        # answer independent of the order the centers were declared in.
        self.assertEqual(len(iteration['residuals']), 3)
        for value, expected in zip(iteration['residuals'], [0.4, 0.2, 0.01]):
            self.assertAlmostEqual(value, expected)
        self.assertEqual(iteration['rounds_run'], 3)
        self.assertTrue(iteration['converged'])
        self.assertEqual(iteration['stopped_by'], 'tolerance')
        centers = iteration['centers']
        self.assertEqual([entry['center_index'] for entry in centers], [0, 1])
        for entry, expected in ((centers[0], [0.2, 0.1, 0.01]),
                                (centers[1], [0.4, 0.2, 0.01])):
            self.assertEqual(len(entry['residuals']), 3)
            for value, want in zip(entry['residuals'], expected):
                self.assertAlmostEqual(value, want)
        rounds = [backend.calls for backend in calculator.backends['theory']
                  if isinstance(backend, _MccChargeBackend)]
        self.assertEqual(rounds, [3, 3])

    def test_maxiter_stops_an_unconverged_run(self):
        _calculator, result = self._run(
            sc_charge=True, maxiter=2, threshold=0.001,
            charges_a=((-0.2, 0.2), (-0.1, 0.1)),
            charges_b=((0.4, -0.4), (0.2, -0.2)))
        iteration = result['qmmm_charge_iteration']
        self.assertFalse(iteration['converged'])
        self.assertEqual(iteration['stopped_by'], 'maxiter')
        self.assertEqual(iteration['rounds_run'], 2)
        self.assertEqual(iteration['maxiter'], 2)

    def test_second_probe_sees_both_centers_of_the_previous_round(self):
        # R11 Jacobi: a round solves every center in the snapshot the
        # previous round left, so center A's field holds center B's new
        # charges and not a zero or a stale value.
        calculator, _result = self._run(sc_charge=True, threshold=0.001,
                                        maxiter=2)
        probes = [backend for backend in calculator.backends['theory']
                  if isinstance(backend, _MccFieldBackend)]
        # probe calls: round-1 field (zeros), then the accepted-charge
        # re-probe before the property round.
        self.assertEqual(probes[0].feeds[-1], [-0.1, 0.1, 0.2, -0.2])
        self.assertEqual(probes[1].feeds[-1], [-0.1, 0.1, 0.2, -0.2])

    def test_a_charge_round_without_a_readback_is_refused(self):
        # R8: the round's number feeds another center's field. Falling back
        # to the store or to molecule.q_qm here would iterate on a value
        # nothing computed and still report a converged run.
        class _Silent(_MccChargeBackend):
            def calculate(self, atoms, coordinates, **kwargs):
                self.calls += 1
                return {'energy': np.array([-1.0])}

        state = self._make_state(True)
        for task in self._queue(
                _mcc_field_backend([1, 2], [-1.0, -2.0]),
                _mcc_field_backend([4, 5], [-3.0, -4.0]),
                _mcc_charge_backend(((-0.1, 0.1),)),
                _Silent):
            state.add_task('theory', task)
        from pyrai3md.workflow.calculator import Calculator
        calculator = Calculator(state=state).initialize()
        with self.assertRaises(ValueError) as caught:
            calculator.run()
        self.assertIn('no computed_charges readback', str(caught.exception))

    def test_the_classical_surface_is_fed_the_accepted_charges(self):
        _calculator, result = self._run(sc_charge=True, threshold=0.02)
        iteration = result['qmmm_charge_iteration']
        self.assertEqual(iteration['rounds_run'], 3)
        components = result['qmmm_components']
        self.assertEqual(components['q_qm'], [-0.09, 0.09, 0.19, -0.19])
        self.assertEqual(components['qm_atoms'], MC_UNION)

    def test_one_center_has_no_charge_round(self):
        # The C=1 anchor at the iteration level: one center has no other
        # center's charge to iterate on, so the queue holds no charge duty,
        # the charges come from the property solve, and no iteration is
        # reported.
        from pyrai3md.workflow.calculator import Calculator
        state = self._make_state(False, centers=(MC_UNION,))
        for task in [
                self._task('field', 'field',
                           _mcc_field_backend([1, 2, 4, 5],
                                              [-1.0, -2.0, -3.0, -4.0]),
                           [], None, 0, 'lammps_skill', 0),
                self._task('qm', 'qm', _MccQmBackend, MC_UNION, 0, 1,
                           'molcas_skill', 1),
                self._task('energy', 'energy',
                           _mcc_energy_backend(centers=(-4.0,)), [], None,
                           1, 'lammps_skill', 2)]:
            state.add_task('theory', task)
        calculator = Calculator(state=state).initialize()
        result = calculator.run()
        self.assertIsNone(result.get('qmmm_charge_iteration'))
        self.assertEqual(result['qmmm_components']['n_center'], 1)
        self.assertEqual(calculator._qmmm_iteration, None)


class TestChargeIterationKeyFace(unittest.TestCase):
    """sc_charge is one input question with defaults that answer it alone.

    The three keys are one question's three numbers: whether the centers
    polarize each other, how many rounds that may take, and how small a
    charge move has to be before the round counts. A run that does not ask
    the question must read exactly like the run that did before the keys
    existed.
    """

    def _system(self, **values):
        from pyrai3md.schema.validate import validate_system
        base = {'atoms': ['O', 'H', 'O', 'H', 'H', 'H'], 'natom': 6,
                'theory': ['molcas', 'lammps'],
                'multiscale_type': 'qmmm',
                'highlevel': [[0, 1], [3, 4]],
                'embedding': True}
        return validate_system(dict(base, **values))

    def test_defaults_do_not_iterate(self):
        from pyrai3md.schema.config import PyRAIConfig, System
        self.assertIs(System().sc_charge, False)
        self.assertEqual(System().sc_charge_maxiter, 10)
        self.assertAlmostEqual(System().sc_charge_threshold, 1e-2)
        self.assertIs(PyRAIConfig().system.sc_charge, False)

    def test_yaml_spelling_and_type_codes(self):
        from pyrai3md.schema.config import System
        from pyrai3md.schema.adapter import MOLECULE_TYPES
        system = System.model_validate(
            {'sc_charge': True, 'sc_charge_maxiter': 4,
             'sc_charge_threshold': 1e-4})
        self.assertIs(system.sc_charge, True)
        self.assertEqual(system.sc_charge_maxiter, 4)
        self.assertAlmostEqual(system.sc_charge_threshold, 1e-4)
        self.assertEqual(MOLECULE_TYPES['sc_charge'], 'b')
        self.assertEqual(MOLECULE_TYPES['sc_charge_maxiter'], 'i')
        self.assertEqual(MOLECULE_TYPES['sc_charge_threshold'], 'f')

    def test_iteration_needs_a_second_center(self):
        from pyrai3md.schema.validate import validate_system
        with self.assertRaises(ValueError) as caught:
            validate_system({
                'atoms': ['O', 'H', 'H', 'O'], 'natom': 4,
                'theory': ['molcas', 'lammps'],
                'multiscale_type': 'qmmm',
                'highlevel': [0, 1], 'embedding': True,
                'sc_charge': True})
        self.assertIn('one QM region', str(caught.exception))

    def test_iteration_needs_the_field(self):
        with self.assertRaises(ValueError) as caught:
            self._system(sc_charge=True, embedding=False)
        self.assertIn('embedding', str(caught.exception))

    def test_a_round_must_be_countable_and_a_threshold_positive(self):
        with self.assertRaises(ValueError):
            self._system(sc_charge=True, sc_charge_maxiter=0)
        with self.assertRaises(ValueError):
            self._system(sc_charge=True, sc_charge_threshold=0.0)
        with self.assertRaises(ValueError):
            self._system(sc_charge=True, sc_charge_threshold=float('nan'))

    def test_an_absent_threshold_is_not_a_bad_threshold(self):
        # The keys have model defaults, so an input that names only
        # sc_charge must not be refused for the two numbers it did not
        # write. Every pre-existing QMMM input is of this shape.
        self.assertNotIn('sc_charge', self._system())
        self.assertTrue(self._system(sc_charge=True)['sc_charge'])

    def test_converter_echoes_the_iteration_keys(self):
        from pyrai3md.qualification.certification.converter import (
            _build_molecule_pairs)
        _header, pairs = _build_molecule_pairs({
            'sc_charge': True, 'sc_charge_maxiter': 4,
            'sc_charge_threshold': 0.001})
        pairs = dict(pairs)
        self.assertIs(pairs['sc_charge'], True)
        self.assertEqual(pairs['sc_charge_maxiter'], 4)
        self.assertEqual(pairs['sc_charge_threshold'], 0.001)

    def test_converter_keeps_the_default_quiet(self):
        # A key at its default is not an echo: the generated classic input
        # must read the way it did before the keys existed.
        from pyrai3md.qualification.certification.converter import (
            _build_molecule_pairs)
        _header, pairs = _build_molecule_pairs({})
        spelled = ' '.join('%s %s' % (key, value) for key, value in pairs)
        self.assertNotIn('sc_charge', spelled)


if __name__ == '__main__':
    unittest.main()
