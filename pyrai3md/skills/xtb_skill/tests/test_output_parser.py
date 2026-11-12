"""
Unit tests for xtb_skill result extraction (output_parser).

Verifies the molcas-aligned return contract with mock calcdir
files: engrad-based energy/gradient extraction and the
computed_charges channel ((1, natom), omitted when xTB wrote no
charge file).
These tests do NOT require an xTB installation.

Run: pytest xtb_skill/tests/test_output_parser.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import os

import numpy as np
import pytest

from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..service.parser.output_parser import (
    parse_output, format_results, ANG_TO_BOHR)

# Water molecule keeps the mock files compact
ATOMS = ['O', 'H', 'H']
COORDS = [
    [0.000000000, 0.000000000, 0.117300000],
    [0.000000000, 0.757200000, -0.469200000],
    [0.000000000, -0.757200000, -0.469200000],
]

ENERGY = -5.070000000
GRADIENT = [0.001, -0.002, 0.003, -0.004, 0.005, -0.006,
            0.003, -0.003, 0.003]


def _make_config(method='gfn2', profile='gfn2_gradient'):
    raw = {
        'atoms': ATOMS,
        'coordinates': COORDS,
        'charges': 0,
        'multiplicity': 1,
        'nstate': 1,
        'state_spin': [0],
        'profile': profile,
        'method': method,
    }
    return normalize(build_config(raw))


def _write_engrad(calcdir, project):
    lines = [
        '#',
        'The current total energy in Eh',
        '',
        f'    {ENERGY:.9f}',
        'The current gradient in Eh/bohr',
        '',
    ] + [f'    {g:.9f}' for g in GRADIENT]
    calcdir.joinpath(f'{project}.engrad').write_text(
        '\n'.join(lines) + '\n')


class TestEnergyGradientExtraction:
    """engrad-based ground-state energy and gradient."""

    def test_energy_and_gradient_from_engrad(self, tmp_path):
        config = _make_config()
        _write_engrad(tmp_path, config.xtb_project)

        results = parse_output(config, str(tmp_path), workflow=None)

        assert np.allclose(results['energy'], [ENERGY])
        assert results['gradient'].shape == (1, 3, 3)
        assert np.allclose(results['gradient'][0],
                           np.array(GRADIENT).reshape((3, 3)))

    def test_missing_engrad_raises(self, tmp_path):
        config = _make_config()

        with pytest.raises(RuntimeError, match='engrad file not found'):
            parse_output(config, str(tmp_path), workflow=None)

    def test_engrad_without_energy_raises(self, tmp_path):
        config = _make_config()
        (tmp_path / f'{config.xtb_project}.engrad').write_text('#\n')

        with pytest.raises(RuntimeError, match='no energy'):
            parse_output(config, str(tmp_path), workflow=None)


class TestComputedChargesContract:
    """Atomic charges follow the computed_charges (ns, na) channel."""

    def test_charges_returned_as_state_resolved_matrix(self, tmp_path):
        config = _make_config()
        _write_engrad(tmp_path, config.xtb_project)
        (tmp_path / 'charges').write_text('-0.585\n0.292\n0.293\n')

        results = parse_output(config, str(tmp_path), workflow=None)

        assert 'computed_charges' in results
        assert 'atom_charges' not in results
        assert results['computed_charges'].shape == (1, 3)
        assert np.allclose(results['computed_charges'][0],
                           [-0.585, 0.292, 0.293])

    def test_gfn_ff_reads_gfnff_charges_file(self, tmp_path):
        config = _make_config(method='gfn_ff', profile='gfn_ff_gradient')
        _write_engrad(tmp_path, config.xtb_project)
        (tmp_path / 'gfnff_charges').write_text('-0.300\n0.150\n0.150\n')

        results = parse_output(config, str(tmp_path), workflow=None)

        assert results['computed_charges'].shape == (1, 3)
        assert results['computed_charges'][0, 0] == pytest.approx(-0.3)

    def test_a_charge_file_older_than_this_runs_geometry_is_ignored(
            self, tmp_path):
        # A reused calcdir can hold the previous step's charge file. Only the
        # mtime differs between the two halves of this test, so the flip is
        # the guard's doing and nothing else.
        config = _make_config()
        _write_engrad(tmp_path, config.xtb_project)
        charge = tmp_path / 'charges'
        charge.write_text('-0.585\n0.292\n0.293\n')
        geometry = tmp_path / f'{config.xtb_project}.xyz'
        geometry.write_text('3\n\nO 0 0 0\nH 0 1 0\nH 0 0 1\n')
        old, fresh = 1_700_000_000, 1_700_000_100
        os.utime(charge, (old, old))
        os.utime(geometry, (fresh, fresh))

        stale = parse_output(config, str(tmp_path), workflow=None)
        assert 'computed_charges' not in stale

        os.utime(charge, (fresh + 100, fresh + 100))
        live = parse_output(config, str(tmp_path), workflow=None)
        assert live['computed_charges'].shape == (1, 3)

    def test_missing_charge_file_omits_key(self, tmp_path):
        config = _make_config()
        _write_engrad(tmp_path, config.xtb_project)

        results = parse_output(config, str(tmp_path), workflow=None)

        # Schema default empty list means not computed: no key at all
        assert 'computed_charges' not in results

    def test_format_results_serializes_charges(self, tmp_path):
        config = _make_config()
        _write_engrad(tmp_path, config.xtb_project)
        (tmp_path / 'charges').write_text('-0.585\n0.292\n0.293\n')

        formatted = format_results(
            parse_output(config, str(tmp_path), workflow=None), config)

        assert isinstance(formatted['computed_charges'], list)
        assert len(formatted['computed_charges']) == 1
        assert len(formatted['computed_charges'][0]) == 3


# Periodic GFN-FF lattice-gradient record: the first three rows are
# the lattice vectors H (Bohr), the next three the lattice gradient G
# (Hartree/Bohr). stress = G.T @ H / abs(det(H)) (Hartree/Bohr^3).
# H matches the requested cell (LATTICE_ANG Angstrom -> Bohr) so the
# parser's staleness guard accepts it.
LATTICE_ANG = 6.3
GRADLATT_H = np.eye(3) * (LATTICE_ANG * ANG_TO_BOHR)
GRADLATT_G = np.array([[0.01, 0.0, 0.0],
                       [0.0, 0.02, 0.0],
                       [0.0, 0.0, 0.03]])
GRADLATT_STRESS = GRADLATT_G.T @ GRADLATT_H / abs(np.linalg.det(GRADLATT_H))


def _write_gradlatt(calcdir, name='gradlatt', h=None, g=None):
    h = GRADLATT_H if h is None else h
    g = GRADLATT_G if g is None else g
    lines = ['$gradlatt',
             '  cycle =    1  SCF energy =   -5.0700  |dE/dlatt| =  0.0374']
    for row in h:
        lines.append('  '.join('%18.12f' % v for v in row))
    for row in g:
        lines.append('  '.join('%18.12E' % v for v in row))
    lines.append('$end')
    calcdir.joinpath(name).write_text('\n'.join(lines) + '\n')


def _stress_config():
    raw = {
        'atoms': ATOMS,
        'coordinates': COORDS,
        'charges': 0,
        'multiplicity': 1,
        'nstate': 1,
        'state_spin': [0],
        'profile': 'gfn_ff_gradient',
        'method': 'gfn_ff',
        'stress': True,
        'pbc': [1, 1, 1],
        'lattice_vector': [[LATTICE_ANG, 0.0, 0.0],
                           [0.0, LATTICE_ANG, 0.0],
                           [0.0, 0.0, LATTICE_ANG]],
    }
    return normalize(build_config(raw))


class TestStressTensor:
    """Whole-system stress tensor from the periodic gradlatt record."""

    def test_stress_from_gradlatt(self, tmp_path):
        config = _stress_config()
        _write_engrad(tmp_path, config.xtb_project)
        _write_gradlatt(tmp_path)

        results = parse_output(config, str(tmp_path), workflow=None)

        assert results['stress_tensor'].shape == (1, 3, 3)
        assert np.allclose(results['stress_tensor'][0], GRADLATT_STRESS)

    def test_stress_zero_filled_when_not_requested(self, tmp_path):
        config = _make_config(method='gfn_ff', profile='gfn_ff_gradient')
        _write_engrad(tmp_path, config.xtb_project)
        _write_gradlatt(tmp_path)

        results = parse_output(config, str(tmp_path), workflow=None)

        assert results['stress_tensor'].shape == (1, 3, 3)
        assert np.allclose(results['stress_tensor'], 0.0)

    def test_missing_gradlatt_is_nonblocking(self, tmp_path):
        config = _stress_config()
        _write_engrad(tmp_path, config.xtb_project)

        results = parse_output(config, str(tmp_path), workflow=None)

        assert np.allclose(results['stress_tensor'], 0.0)

    def test_stale_gradlatt_cell_is_rejected(self, tmp_path):
        # A gradlatt whose cell does not match the requested lattice is
        # a leftover of a previous run: the stress stays zero-filled.
        config = _stress_config()
        _write_engrad(tmp_path, config.xtb_project)
        _write_gradlatt(tmp_path, h=np.eye(3) * 10.0)

        results = parse_output(config, str(tmp_path), workflow=None)

        assert np.allclose(results['stress_tensor'], 0.0)

    def test_an_unwritten_lattice_gradient_is_not_reported_as_zero_stress(
            self, tmp_path):
        # Verbatim record of a periodic GFN1-xTB gradient run on the
        # test environment (xtb 6.7.1): the cell matches the request, the
        # header norm reads `|dE/dlatt| =  0.000000`, and the six body rows
        # are exact zeros - the engine wrote the record without computing the
        # lattice gradient. The same coord file under GFN-FF prints
        # `|dE/dlatt| =  0.008680` with a nonzero block, so a step that read
        # this as "no stress" would be told its box sits at equilibrium by a
        # file that says nothing of the kind.
        config = _stress_config()
        _write_engrad(tmp_path, config.xtb_project)
        tmp_path.joinpath('gradlatt').write_text(
            '$gradlatt\n'
            '  cycle =      1    SCF energy =    -6.20114076166'
            '   |dE/dlatt| =  0.000000\n'
            '   18.89726124620440      0.00000000000000'
            '      0.00000000000000\n'
            '    0.00000000000000     18.89726124620440'
            '      0.00000000000000\n'
            '    0.00000000000000      0.00000000000000'
            '     18.89726124620440\n'
            '   0.0000000000000E+00   0.0000000000000E+00'
            '   0.0000000000000E+00\n'
            '   0.0000000000000E+00   0.0000000000000E+00'
            '   0.0000000000000E+00\n'
            '   0.0000000000000E+00   0.0000000000000E+00'
            '   0.0000000000000E+00\n'
            '$end\n')

        with pytest.warns(UserWarning, match='exact zeros'):
            results = parse_output(config, str(tmp_path), workflow=None)

        assert results['stress_tensor'].shape == (1, 3, 3)
        assert np.allclose(results['stress_tensor'], 0.0)
