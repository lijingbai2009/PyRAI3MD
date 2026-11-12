"""
======================================================================
Test suite for the run-log result formatting utilities
(pyrai3md/utilities/print_results.py).

Tests cover:
- format_result_block state population section ('population' key,
  'final_population' fallback, omission when absent).
- control.verbose level gating of the missing-property sections:
  verbose < 3 drops every 'Not computed' section (whole sections
  and the pair/state lines embedded inside valid sections),
  verbose >= 3 restores them.

Aug 25 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import unittest

import numpy as np

from pyrai3md.utilities.print_results import (format_ml_report,
                                              format_result_block)

_SYSTEM = {
    'atoms': ['X'],
    'coordinates': [[0.0, 0.0, 0.0]],
    'current_state': 0,
}


class TestFormatResultBlockPopulation(unittest.TestCase):
    """State population section of the run-log result block."""

    def test_population_key_printed(self):
        block = format_result_block(
            dict(_SYSTEM), {'population': [0.25, 0.75]})
        self.assertIn('&state population', block)
        self.assertIn('State   1', block)
        self.assertIn('0.2500000000000000', block)
        self.assertIn('0.7500000000000000', block)

    def test_final_population_fallback(self):
        block = format_result_block(
            dict(_SYSTEM), {'final_population': [1.0, 3e-24]})
        self.assertIn('&state population', block)
        self.assertIn('1.0000000000000000', block)

    def test_population_key_wins_over_final_population(self):
        block = format_result_block(
            dict(_SYSTEM),
            {'population': [0.5, 0.5], 'final_population': [1.0, 0.0]})
        self.assertIn('0.5000000000000000', block)
        self.assertNotIn('1.0000000000000000', block)

    def test_no_population_omits_section(self):
        block = format_result_block(dict(_SYSTEM), {'energy': [1.0]})
        self.assertNotIn('&state population', block)

    def test_empty_population_omits_section(self):
        block = format_result_block(
            dict(_SYSTEM), {'population': [], 'final_population': []})
        self.assertNotIn('&state population', block)


# Section titles carrying the 'Not computed' marker when their
# property is missing (whole-section suppression under verbose < 3).
_MISSING_SECTIONS = (
    '&point charges in e (atom, q)',
    '&velocities in Bohr/au',
    '&kinetic energy in Hartree',
    '&gradient in Hartree/Bohr',
    '&nonadiabatic coupling in Hartree/Bohr',
    '&spin-orbit coupling in cm-1',
    '&dipole moment in Debye',
    '&transition dipole moment in Debye',
    '&thermodynamic properties',
)


_LEVEL_SYSTEM = {
    'atoms': ['C', 'H'],
    'coordinates': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    'velocities': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    'lattice_vector': [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
    'cell': [5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
    'volume': 125.0,
    'pbc': [1, 1, 1],
    'current_state': 0,
}

_LEVEL_RESULTS = {
    'energy': [-1.0],
    'gradient': [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
    'stress_tensor': [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]]],
    'norm_mode': [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
    'hessian': [[1.0, 0.0], [0.0, 1.0]],
}


class TestFormatResultBlockLevels(unittest.TestCase):
    """The 0-3 content levels: progress, structure, properties, everything."""

    def _block(self, level, system=None, results=None):
        return format_result_block(
            dict(system or _LEVEL_SYSTEM), dict(results or _LEVEL_RESULTS),
            step=1, verbose=level)

    def test_level_zero_keeps_progress_and_the_explaining_numbers(self):
        block = self._block(0)
        for title in ('&coordinates', '&cell', '&velocities', '&gradient',
                      '&stress tensor', '&normal modes', '&hessian'):
            self.assertNotIn(title, block)
        self.assertIn('&electronic energy in Hartree', block)
        self.assertIn('Step        1', block)

    def test_level_one_adds_structure_but_not_properties(self):
        block = self._block(1)
        self.assertIn('&coordinates in Angstrom', block)
        self.assertIn('&cell in Angstrom', block)
        self.assertNotIn('&velocities', block)
        self.assertNotIn('&gradient', block)
        self.assertNotIn('&stress tensor', block)

    def test_level_two_adds_properties_but_not_the_full_matrix(self):
        block = self._block(2)
        for title in ('&velocities', '&gradient', '&stress tensor',
                      '&normal modes'):
            self.assertIn(title, block)
        self.assertNotIn('&hessian', block)

    def test_level_three_adds_the_hessian_and_missing_markers(self):
        block = self._block(3)
        self.assertIn('&hessian', block)
        self.assertIn('Not computed', block)

    def test_the_cell_section_states_the_lattice_and_volume(self):
        block = self._block(1)
        self.assertIn('&cell in Angstrom', block)
        self.assertIn('a:', block)
        self.assertIn('volume (A^3):', block)
        self.assertIn('pbc:', block)
        self.assertIn('125.00000000', block)

    def test_a_non_periodic_run_has_no_cell_section_below_level_three(self):
        system = dict(_LEVEL_SYSTEM)
        system.pop('lattice_vector')
        system.pop('cell')
        system.pop('volume')
        system.pop('pbc')
        self.assertNotIn('&cell', self._block(1, system=system))
        self.assertNotIn('&cell', self._block(2, system=system))
        self.assertIn('&cell', self._block(3, system=system))


class TestFormatResultBlockVerbose(unittest.TestCase):
    """'Not computed' sections follow the control verbose level."""

    def test_default_verbose_suppresses_missing_sections(self):
        # The minimal system keeps coordinates and the electronic
        # state present; every other property is missing and must
        # stay silent below verbose 3 (the default level 2).
        block = format_result_block(dict(_SYSTEM), {'energy': [1.0]})
        self.assertNotIn('Not computed', block)
        for title in _MISSING_SECTIONS:
            self.assertNotIn(title, block)
        # Present properties survive the suppression.
        self.assertIn('&coordinates in Angstrom', block)
        self.assertIn('&electronic state', block)
        self.assertIn('&electronic energy in Hartree', block)

    def test_verbose_three_shows_missing_sections(self):
        block = format_result_block(
            dict(_SYSTEM), {'energy': [1.0]}, verbose=3)
        for title in _MISSING_SECTIONS:
            self.assertIn(title, block)
        self.assertIn('Not computed', block)

    def test_missing_state_section_follows_the_level(self):
        # current_state absent: the section obeys the same gate.
        system = dict(_SYSTEM)
        system.pop('current_state')
        self.assertNotIn('&electronic state', format_result_block(
            system, {}))
        self.assertIn('&electronic state', format_result_block(
            system, {}, verbose=3))

    def test_embedded_soc_missing_line_follows_the_level(self):
        # The pair map declares the section; a missing value prints
        # a 'Not computed' line inside it, gated by the verbose
        # level (the whole section drops when every line is gone).
        system = dict(_SYSTEM, soc_pair_map=[[0, 1]])
        self.assertNotIn('&spin-orbit coupling',
                         format_result_block(system, {}))
        block = format_result_block(system, {}, verbose=3)
        self.assertIn('&spin-orbit coupling in cm-1', block)
        self.assertIn('Not computed', block)

    def test_embedded_dipole_missing_line_follows_the_level(self):
        # A malformed per-state moment embeds a 'Not computed' line
        # inside the dipole section.
        results = {'dipole': [[0.1]]}
        self.assertNotIn('&dipole moment',
                         format_result_block(dict(_SYSTEM), results))
        block = format_result_block(dict(_SYSTEM), results, verbose=3)
        self.assertIn('&dipole moment in Debye', block)
        self.assertIn('State   1  Not computed', block)

    def test_embedded_transition_dipole_line_follows_the_level(self):
        system = dict(_SYSTEM, transition_pair=[[0, 1]])
        self.assertNotIn('&transition dipole moment',
                         format_result_block(system, {}))
        block = format_result_block(system, {}, verbose=3)
        self.assertIn('&transition dipole moment in Debye', block)
        self.assertIn('Not computed', block)

    def test_the_transition_rows_are_the_declared_pairs_and_nothing_else(self):
        # The section printed one row per state and opened with `1 - 1` - a
        # transition dipole of the ground state to itself, which no engine
        # computes and which the readers then filled with zeros (#448). Three
        # states declaring two ground-to-state pairs print two rows, labelled by
        # the pairs themselves and numbered 1-based as the section has always
        # been read. The schema half of the same rule - the default that opened
        # with the self pair - is pinned by
        # `test_state_declaration.py::TestTransitionRowsAreDeclaredOnly`, which
        # was measured red when that default was put back.
        import re
        results = {'transition_dipole': [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]}
        block = format_result_block(
            dict(_SYSTEM, transition_pair=[[0, 1], [0, 2]]), results, verbose=3)
        rows = [line for line in block.splitlines() if '<M>=' in line]
        self.assertEqual(2, len(rows))
        labels = [tuple(int(x) for x in re.search(r'(\d+)\s*-\s*(\d+)\s*$',
                                 line).groups())
                  for line in rows]
        self.assertEqual([(1, 2), (1, 3)], labels)
        self.assertIsNone(re.search(r'\b1\s+-\s+1\b', block))

    def test_embedded_nac_missing_line_follows_the_level(self):
        system = dict(_SYSTEM, nac_pair_map=[[0, 1]])
        self.assertNotIn('&nonadiabatic coupling',
                         format_result_block(system, {}))
        block = format_result_block(system, {}, verbose=3)
        self.assertIn('&nonadiabatic coupling', block)
        self.assertIn('Not computed', block)

    def test_partial_thermodynamics_keeps_alignment(self):
        # Only G computed: the section survives at both levels; the
        # missing H/S columns print 'Not computed' from verbose 3
        # upward and stay blank below.
        results = {'free_energy': [-0.5]}
        quiet = format_result_block(dict(_SYSTEM), results)
        self.assertIn('&thermodynamic properties', quiet)
        self.assertNotIn('Not computed', quiet)
        loud = format_result_block(dict(_SYSTEM), results, verbose=3)
        self.assertIn('&thermodynamic properties', loud)
        self.assertIn('Not computed', loud)



_TRAINING = {
    'train_errors': {'energy_grad': {'energy': [0.0686, 0.0915],
                                     'gradient': [0.0087, 0.0110]},
                     'nac': {'nac': [0.0123]}},
    'train_metric_units': {'energy': 'eV', 'gradient': 'eV/angstrom',
                           'nac': 'eV/angstrom'},
    'train_wall_seconds': 22.5,
    'train_arm': 'local',
    'train_data': '/data/TRAIN.json',
    'train_modeldir': '/run/NN-TEST',
    'train_fitlogs': ['/run/NN-TEST/energy_grad/fitlog_0.txt',
                      '/run/NN-TEST/energy_grad/fitlog_1.txt'],
    'train_prediction_std': {'ndata': 4, 'energy': 0.001,
                             'gradient': 0.002, 'nac': None, 'soc': None},
    'train_summary': '/run/NNPES_train.summary.txt',
}

_TRAIN_SYSTEM = {
    'atoms': ['C', 'C'],
    'coordinates': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    'natom': 2,
    'nstate': 2,
    'nac_pair_map': [[0, 1]],
    'soc_pair_map': [],
    'current_state': 1,
}


class TestFormatResultBlockTraining(unittest.TestCase):
    """Run-log report of a model-training result."""

    def test_training_sections_are_printed(self):
        block = format_result_block(
            dict(_TRAIN_SYSTEM), dict(_TRAINING),
            record_report=format_ml_report(dict(_TRAINING), dict(_TRAIN_SYSTEM)))
        self.assertIn('&nn training', block)
        self.assertIn('&nn validation mean absolute error', block)
        self.assertIn('energy_grad_v0', block)
        self.assertIn('energy_grad_v1', block)
        self.assertIn('nac_v0', block)
        self.assertIn('Wall clock:       22.5 s', block)
        self.assertIn('eV/angstrom', block)
        self.assertIn('Number of atoms:  2', block)
        self.assertIn('Training data:    /data/TRAIN.json', block)
        self.assertIn('Model folder:     /run/NN-TEST', block)
        self.assertIn('Fit log:          /run/NN-TEST/energy_grad/fitlog_0.txt',
                      block)
        self.assertIn('&nn prediction max std', block)
        self.assertIn('0.001000', block)
        self.assertNotIn('Summary file:', block)

    def test_property_sections_of_a_step_that_took_none(self):
        # A training run computes no energy, no gradient and no velocity,
        # so it has no active state and no kinetic energy to report either;
        # printing them would put observations in the log that were never
        # taken. The input geometry is real and stays.
        block = format_result_block(
            dict(_TRAIN_SYSTEM), dict(_TRAINING), verbose=3,
            record_report=format_ml_report(dict(_TRAINING), dict(_TRAIN_SYSTEM)))
        self.assertIn('&coordinates in Angstrom', block)
        for title in ('&electronic state', '&kinetic energy',
                      '&electronic energy', '&gradient'):
            self.assertNotIn(title, block)

    def test_an_untrained_channel_is_blank_not_zero(self):
        block = format_result_block(
            dict(_TRAIN_SYSTEM), dict(_TRAINING),
            record_report=format_ml_report(dict(_TRAINING), dict(_TRAIN_SYSTEM)))
        rows = [line for line in block.splitlines()
                if line.strip().startswith('energy_grad_v')]
        self.assertTrue(rows)
        for row in rows:
            self.assertIn('--', row)
            self.assertNotIn('0.00000000', row)

    def test_adaptive_summary_leaves_data_and_model_to_the_nn_report(self):
        results = {'adaptive': {
            'n_iterations': 2, 'converged': False,
            'reason': 'maxiter exhausted',
            'final_dataset': '/run/data_iter-2_10.json',
            'final_modeldir': '/run/NN-test-iter-2',
            'iterations': [
                {'iteration': 1, 'external': True, 'ntraj': 2,
                 'completed': 0, 'triggered': True, 'selected': 2,
                 'new_frames': 0, 'average_length': 100.0,
                 'ground_percent': 0.0},
                {'iteration': 2, 'external': False, 'ntraj': 2,
                 'completed': 0, 'triggered': True, 'selected': 2,
                 'new_frames': 2, 'average_length': 150.0,
                 'ground_percent': 50.0}]}}
        block = format_result_block(dict(_TRAIN_SYSTEM), results,
                                    include_system=False,
                                    dynamics={'step': 200})
        self.assertIn('&adaptive sampling summary', block)
        self.assertIn('Elapsed iteration:  2', block)
        self.assertIn('Final convergence:  False', block)
        self.assertIn('Ground %', block)
        self.assertIn('50.0', block)
        self.assertIn('75.0', block)
        # The table is the per-iteration trend, not a repetition of each
        # iteration's full record.
        for removed in ('External', 'Ntraj', 'Triggered', 'Selected',
                        'Completed'):
            self.assertNotIn(removed, block)
        self.assertNotIn('Final dataset:', block)
        self.assertNotIn('Final model:', block)
        for title in ('&coordinates in Angstrom', '&cell in Angstrom',
                      '&electronic state', '&kinetic energy',
                      '&velocities in Bohr/au'):
            self.assertNotIn(title, block)

    def test_a_property_result_is_unaffected(self):
        block = format_result_block(dict(_TRAIN_SYSTEM),
                                    {'energy': [1.0, 2.0]})
        self.assertNotIn('&nn training', block)
        self.assertIn('&electronic energy in Hartree', block)


class TestMultiscaleLog(unittest.TestCase):
    """Region-scoped coordinates/charges and the QMMM decomposition."""

    def _system(self):
        return {
            'atoms': ['C', 'H', 'O', 'N'],
            'coordinates': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0], [9.0, 9.0, 9.0]],
            'natom': 4,
            'highlevel': [0, 1],
            'midlevel': [2],
            'embedding': True,
            'point_charges': [[-0.1, 0.0, 0.0, 0.0],
                              [0.1, 1.0, 0.0, 0.0],
                              [0.2, 2.0, 0.0, 0.0],
                              [0.3, 9.0, 9.0, 9.0]],
            'current_state': 0,
        }

    def test_oniom_shows_highlevel_and_midlevel_charges_only(self):
        results = {
            'energy': [-1.0],
            'composite_scheme': 'oniom',
            'composite_terms': {'labels': ['high[qm]'], 'energies': [[-1.0]],
                                'contributions': [1]},
            'composite_total': [-1.0],
        }
        block = format_result_block(self._system(), results, verbose=2)
        self.assertIn('&coordinates in Angstrom (highlevel)', block)
        self.assertIn('&highlevel point charges in e (atom, q)', block)
        self.assertIn('&midlevel point charges in e (atom, q)', block)
        self.assertNotIn('&point charges in e (atom, q)', block)
        # The low-only atom is not printed as a coordinate row.
        self.assertNotIn('   9.0000000000000000', block)
        self.assertNotIn('&external charges', block)

    def test_qmmm_shows_qm_region_and_decomposition(self):
        results = {
            'energy': [-2.0],
            'composite_scheme': 'qmmm',
            'composite_terms': {'labels': ['high[qm]', 'all[energy]'],
                                'energies': [[-1.5], [-0.5]],
                                'contributions': [1, 1]},
            'composite_total': [-2.0],
            'qmmm_components': {
                'embedding': 'potential',
                'qm_energy': [-1.5],
                'e_classic': -0.5,
                'u_main': -10.0,
                'u_gas': -4.0,
                'qmmm_el': -5.5,
                'qm_contains_field_gradient': True,
                'q_qm': [-0.1, 0.1],
                'qm_atoms': [0, 1],
                'external_potential': [[0.01, 0.0, 0.0, 0.0],
                                       [0.02, 0.0, 0.0, 0.0]],
            },
        }
        block = format_result_block(self._system(), results, verbose=2)
        self.assertIn('&coordinates in Angstrom (QM region)', block)
        self.assertIn('&qmmm 0-order multipole charges in e (atom, q)',
                      block)
        self.assertIn('&qmmm external potential/field in a.u.', block)
        self.assertIn('&qmmm energy decomposition in Hartree', block)
        self.assertIn('U_main', block)
        self.assertIn('qmmm_el', block)
        self.assertIn('field gradient', block)
        self.assertIn('qm provider', block)
        self.assertNotIn('   9.0000000000000000', block)
        self.assertNotIn('&highlevel point charges', block)

    def test_multicenter_nac_is_printed_per_center(self):
        system = {
            'atoms': ['O', 'H', 'O', 'H'],
            'coordinates': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            'natom': 4,
            'highlevel': [[0, 1], [2, 3]],
            'current_state': 0,
        }
        centers = [
            {'active': 0, 'population': [], 'charge': 0, 'multiplicity': 1,
             'high_indices': [0, 1], 'mid_indices': [],
             'nac': np.array([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
             'nac_pair_map': [[0, 1]], 'soc': None, 'soc_pair_map': []},
            {'active': 0, 'population': [], 'charge': 0, 'multiplicity': 1,
             'high_indices': [2, 3], 'mid_indices': [],
             'nac': np.array([[[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]),
             'nac_pair_map': [[0, 1]], 'soc': None, 'soc_pair_map': []},
        ]
        results = {'energy': [-2.0], 'composite_scheme': 'qmmm'}
        block = format_result_block(system, results, verbose=2,
                                    centers=centers)
        self.assertIn('&center 1 nonadiabatic coupling', block)
        self.assertIn('&center 2 nonadiabatic coupling', block)
        self.assertNotIn('&nonadiabatic coupling', block)
        self.assertIn('   1.0000000000000000', block)
        self.assertIn('   4.0000000000000000', block)

    def test_multicenter_qmmm_shows_per_center_0_order_charges(self):
        system = {
            'atoms': ['O', 'H', 'O', 'H'],
            'coordinates': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            'natom': 4,
            'highlevel': [[0, 1], [2, 3]],
            'embedding': True,
            'point_charges': [[-0.8, 0.0, 0.0, 0.0],
                              [0.4, 1.0, 0.0, 0.0],
                              [-0.7, 2.0, 0.0, 0.0],
                              [0.3, 3.0, 0.0, 0.0]],
            'current_state': 0,
        }
        results = {
            'energy': [-2.0],
            'composite_scheme': 'qmmm',
            'qmmm_components': {
                'embedding': 'potential',
                'qm_atoms': [0, 1, 2, 3],
                'qm_energy': [-1.5, -1.5],
                'e_classic': -0.5,
                'u_main': -10.0,
                'u_gas': -4.0,
                'qmmm_el': -5.5,
                'qm_contains_field_gradient': True,
                'external_potential': None,
            },
        }
        centers = [
            {'active': 0, 'population': [], 'charge': 0, 'multiplicity': 1,
             'high_indices': [0, 1], 'mid_indices': []},
            {'active': 0, 'population': [], 'charge': 0, 'multiplicity': 1,
             'high_indices': [2, 3], 'mid_indices': []},
        ]
        block = format_result_block(system, results, verbose=2,
                                    centers=centers)
        self.assertIn('&qmmm 0-order multipole charges in e (atom, q): '
                      'center 1', block)
        self.assertIn('&qmmm 0-order multipole charges in e (atom, q): '
                      'center 2', block)
        self.assertEqual(block.count('0-order multipole charges'), 2)
        self.assertIn('Center   1  Current state: 1', block)

    def test_qmmm_field_force_owner_follows_the_switch(self):
        components = {
            'embedding': 'potential',
            'qm_energy': [-1.5],
            'e_classic': -0.5,
            'u_main': -10.0,
            'u_gas': -4.0,
            'qmmm_el': -5.5,
            'q_qm': [-0.1, 0.1],
            'qm_atoms': [0, 1],
        }
        results = {'energy': [-2.0], 'composite_scheme': 'qmmm'}
        for value, text in ((False, 'classical side'),
                            (True, 'qm provider')):
            results['qmmm_components'] = dict(
                components, qm_contains_field_gradient=value)
            block = format_result_block(self._system(), results, verbose=2)
            self.assertIn('field gradient', block)
            self.assertIn(text, block)

    def test_qmmm_without_embedding_shows_no_charges(self):
        results = {
            'energy': [-2.0],
            'composite_scheme': 'qmmm',
            'qmmm_components': {
                'embedding': 'none',
                'qm_energy': [-1.5],
                'e_classic': -0.5,
                'u_main': -10.0,
                'u_gas': -4.0,
                'qmmm_el': 0.0,
                'qm_contains_field_gradient': True,
                'q_qm': [-0.1, 0.1],
                'qm_atoms': [0, 1],
                'external_potential': None,
            },
        }
        block = format_result_block(self._system(), results, verbose=2)
        self.assertNotIn('0-order multipole', block)
        self.assertNotIn('external potential/field', block)
        self.assertNotIn('field gradient', block)
        self.assertIn('&qmmm energy decomposition in Hartree', block)

    def test_multicenter_oniom2_shows_per_center_embedding(self):
        system = {
            'atoms': ['N', 'H', 'O', 'H'],
            'coordinates': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            'natom': 4,
            'highlevel': [[0, 1], [3]],
            'midlevel': [],
            'embedding': True,
            'point_charges': [[0.0, 0.0, 0.0, 0.0]] * 4,
            'current_state': 0,
        }
        results = {
            'energy': [-1.0],
            'composite_scheme': 'oniom',
            # The L(real) row: +1 on the ammonium atoms, neutral water.
            'oniom_outer_charges': [[0.6, 0.0, 0.0, 0.0],
                                    [0.2, 1.0, 0.0, 0.0],
                                    [-0.4, 2.0, 0.0, 0.0],
                                    [0.2, 3.0, 0.0, 0.0]],
        }
        centers = [
            {'active': 0, 'population': [], 'charge': 1, 'multiplicity': 1,
             'high_indices': [0, 1], 'mid_indices': []},
            {'active': 0, 'population': [], 'charge': 0, 'multiplicity': 1,
             'high_indices': [3], 'mid_indices': []},
        ]
        block = format_result_block(system, results, verbose=2,
                                    centers=centers)
        self.assertIn('&center 1 highlevel point charges in e (atom, q)',
                      block)
        self.assertIn('&center 1 embedding point charges in e (atom, q)',
                      block)
        self.assertIn('&center 2 highlevel point charges in e (atom, q)',
                      block)
        self.assertIn('&center 2 embedding point charges in e (atom, q)',
                      block)
        # Center 1's environment is atoms 3-4, center 2's is atoms 1-2.
        self.assertIn('       3 O         -0.4000000000', block)
        self.assertIn('       4 H          0.2000000000', block)
        self.assertIn('&region charge & spin multiplicity', block)
        self.assertIn('Center   1  charge    1  multiplicity    1', block)
        self.assertIn('L(real)    charge    1  multiplicity    1', block)
        self.assertIn('Center   1  Current state: 1', block)
        self.assertNotIn('&point charges in e (atom, q)', block)

    def test_multicenter_oniom3_shows_per_center_midlevel(self):
        system = {
            'atoms': ['N', 'O', 'H', 'H'],
            'coordinates': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            'natom': 4,
            'highlevel': [[0], [2]],
            'midlevel': [[1], [3]],
            'embedding': True,
            'point_charges': [[0.0, 0.0, 0.0, 0.0],
                              [-0.2, 1.0, 0.0, 0.0],
                              [0.0, 2.0, 0.0, 0.0],
                              [0.2, 3.0, 0.0, 0.0]],
            'current_state': 0,
        }
        results = {'energy': [-1.0], 'composite_scheme': 'oniom'}
        centers = [
            {'active': 0, 'population': [], 'charge': 0, 'multiplicity': 1,
             'high_indices': [0], 'mid_indices': [1]},
            {'active': 0, 'population': [], 'charge': 0, 'multiplicity': 1,
             'high_indices': [2], 'mid_indices': [3]},
        ]
        block = format_result_block(system, results, verbose=2,
                                    centers=centers)
        self.assertIn('&center 1 midlevel point charges in e (atom, q)', block)
        self.assertIn('&center 2 midlevel point charges in e (atom, q)', block)
        self.assertNotIn('&midlevel point charges in e (atom, q)', block)

    def test_single_center_oniom2_shows_its_embedding_field(self):
        system = self._system()
        system['midlevel'] = []
        results = {
            'energy': [-1.0],
            'composite_scheme': 'oniom',
            'oniom_outer_charges': [
                [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                [0.2, 2.0, 0.0, 0.0], [0.3, 9.0, 9.0, 9.0]],
        }
        block = format_result_block(system, results, verbose=2)
        self.assertIn('&highlevel point charges in e (atom, q)', block)
        self.assertIn('&embedding point charges in e (atom, q)', block)
        self.assertNotIn('&center 1', block)

    def test_result_centers_reads_the_plan_and_active_configuration(self):
        from types import SimpleNamespace
        from pyrai3md.kernel.logger import _result_centers
        plan = SimpleNamespace(
            n_center=2,
            center=lambda index: SimpleNamespace(
                high_indices=(index,), mid_indices=(),
                charge=1 if index == 0 else 0, multiplicity=1))
        state = SimpleNamespace(
            partition_plan=plan,
            pmd=SimpleNamespace(
                results={'active_configuration': [0, 1]}))
        centers = _result_centers(state)
        self.assertEqual([center['charge'] for center in centers], [1, 0])
        self.assertEqual([center['active'] for center in centers], [0, 1])
        self.assertEqual(centers[1]['high_indices'], [1])

    def test_path_summary_has_barrier_and_profile(self):
        from pyrai3md.utilities.print_results import format_path_summary
        block = format_path_summary(
            [{'step': 2, 'energy': -152.3878677587,
              'barrier': 0.1253435285}],
            [{'index': 0, 'role': 'reactant', 'energy': -152.5281727945},
             {'index': 1, 'role': 'image', 'energy': -152.4629558045},
             {'index': 4, 'role': 'product', 'energy': -152.4558203436}])
        self.assertIn('&path summary', block)
        self.assertIn('Step 2', block)
        self.assertIn('Barrier 0.1253435285', block)
        self.assertIn('  1   reactant', block)
        self.assertIn('  2   image', block)
        self.assertIn('  5   product', block)

    def test_external_charges_have_their_own_block(self):
        results = {'energy': [-1.0],
                   'external_charges': [[0.5, 0.0, 0.0, 5.0]]}
        block = format_result_block({'atoms': ['C'], 'coordinates':
                                     [[0.0, 0.0, 0.0]]}, results,
                                    verbose=2)
        self.assertIn('&external charges in e (q x y z)', block)
        self.assertNotIn('&point charges in e (atom, q)', block)


if __name__ == '__main__':
    unittest.main()
