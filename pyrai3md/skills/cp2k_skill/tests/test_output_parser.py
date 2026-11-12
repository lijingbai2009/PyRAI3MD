"""
Unit tests for cp2k_skill result extraction (output_parser).

Verifies the return contract with mock calcdir files following the
demo-verified CP2K output formats: total energy from the log,
gradient = -force from the force xyz, TDDFPT excitation energies /
oscillator strengths and tdfrc excited-state forces, and the
fin_coord channel for GEO_OPT / standalone MD.
These tests do NOT require a CP2K installation.

Run: pytest cp2k_skill/tests/test_output_parser.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import numpy as np
import pytest

from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..service.jobs import Job, plan_jobs
from ..service.workflow import compile_workflow
from ..service.parser.output_parser import (parse_output, merge_job_results,
                                            format_results)
from ..service.parser.log_parser import parse_log

# Water molecule keeps the mock outputs compact
ATOMS = ['O', 'H', 'H']
COORDS = [
    [0.000000000, 0.000000000, 0.117300000],
    [0.000000000, 0.757200000, -0.469200000],
    [0.000000000, -0.757200000, -0.469200000],
]

PROJECT = 'water'
S0_ENERGY = -76.408953144857

# Mock forces in hartree/bohr (gradient = -force)
FORCES = [
    [0.001000, 0.002000, -0.003000],
    [-0.000500, -0.001000, 0.001500],
    [-0.000500, -0.001000, 0.001500],
]


def _raw_config(profile, method, nstate=1, **kwargs):
    raw = {
        'atoms': ATOMS,
        'coordinates': COORDS,
        'charges': 0,
        'multiplicity': 1,
        'nstate': nstate,
        'state_spin': [0] * nstate,
        'profile': profile,
        'method': method,
        'functional': 'pbe',
        'project': PROJECT,
    }
    raw.update(kwargs)
    return raw


def _write_log(calcdir, lines):
    path = calcdir / f'{PROJECT}.log'
    path.write_text('\n'.join(lines) + '\n')
    return path


def _write_stress_file(calcdir, lines):
    """Write the dedicated {project}_stress_tensor.dat print file."""
    path = calcdir / f'{PROJECT}_stress_tensor.dat'
    path.write_text('\n'.join(lines) + '\n')
    return path


def _energy_log():
    return [
        ' SCF WAVEFUNCTION - SEED',
        ' *** SCF run converged ***',
        '',
        f' ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: {S0_ENERGY}',
        '',
        ' Total energy:                              '
        f'{S0_ENERGY:18.10f}',
    ]


def _write_force_file(calcdir, forces=FORCES):
    lines = [
        ' ATOMIC FORCES in [a.u.]',
        '',
        '   # Atom   Kind   Element          X             '
        '      Y                  Z',
    ]
    kinds = {'O': 1, 'H': 2}
    for index, (atom, (fx, fy, fz)) in enumerate(zip(ATOMS, forces), 1):
        lines.append(
            f'   {index:5d}   {kinds[atom]:5d}   {atom:2s}   '
            f'{fx:18.10f} {fy:18.10f} {fz:18.10f}')
    path = calcdir / f'{PROJECT}-force-1_0.xyz'
    path.write_text('\n'.join(lines) + '\n')
    return path


def _write_pos_file(calcdir, frames):
    """Render a multi-frame xyz trajectory ({project}-pos-1.xyz)."""
    lines = []
    for tag, coords in frames:
        lines.append(str(len(coords)))
        lines.append(tag)
        for atom, (x, y, z) in zip(ATOMS, coords):
            lines.append(f'{atom}  {x:12.6f} {y:12.6f} {z:12.6f}')
    path = calcdir / f'{PROJECT}-pos-1.xyz'
    path.write_text('\n'.join(lines) + '\n')
    return path


def _write_tdfrc(calcdir, blocks):
    """Render tdfrc force blocks, one per requested excited state."""
    lines = []
    for state, excitation, osc, forces in blocks:
        lines.append(
            f'# STATE NR.    {state}   {excitation:.5f} eV  '
            f'Oscillator strength:  {osc:.5e}')
        lines.append(str(len(forces)))
        lines.append('')
        for fx, fy, fz in forces:
            lines.append(f'{fx:18.10f} {fy:18.10f} {fz:18.10f}')
    path = calcdir / f'{PROJECT}-td_forces-1_0.tdfrc'
    path.write_text('\n'.join(lines) + '\n')
    return path


@pytest.fixture
def dft_gradient_setup():
    config = normalize(build_config(_raw_config('dft_gradient', 'dft')))
    return config, compile_workflow('dft_gradient')


@pytest.fixture
def tddft_setup():
    raw = _raw_config('tddft_energy', 'tddft', nstate=3,
                      target_states=[0, 1, 2])
    config = normalize(build_config(raw))
    return config, compile_workflow('tddft_energy')


class TestDftEnergyAndGradient:
    """Ground-state energy and the gradient = -force contract."""

    def test_energy_from_log(self, tmp_path, dft_gradient_setup):
        config, workflow = dft_gradient_setup
        _write_log(tmp_path, _energy_log())
        _write_force_file(tmp_path)
        results = parse_output(config, str(tmp_path), workflow)
        assert results['energy'][0] == pytest.approx(S0_ENERGY)

    def test_gradient_is_negative_force(self, tmp_path,
                                        dft_gradient_setup):
        config, workflow = dft_gradient_setup
        _write_log(tmp_path, _energy_log())
        _write_force_file(tmp_path)
        results = parse_output(config, str(tmp_path), workflow)
        expected = -np.array(FORCES)
        assert np.allclose(results['gradient'][0], expected)

    def test_missing_force_file_raises(self, tmp_path,
                                        dft_gradient_setup):
        config, workflow = dft_gradient_setup
        _write_log(tmp_path, _energy_log())
        with pytest.raises(RuntimeError, match='no atomic forces'):
            parse_output(config, str(tmp_path), workflow)

    def test_format_results_converts_arrays(self, tmp_path,
                                            dft_gradient_setup):
        config, workflow = dft_gradient_setup
        _write_log(tmp_path, _energy_log())
        _write_force_file(tmp_path)
        results = parse_output(config, str(tmp_path), workflow)
        formatted = format_results(results, config)
        assert isinstance(formatted['energy'], list)
        assert isinstance(formatted['gradient'], list)


class TestTddftResults:
    """Excitation energies, oscillator strengths and tdfrc forces."""

    @staticmethod
    def _tddft_log(excitations):
        lines = [
            ' *** SCF run converged ***',
            f' Total energy:   {S0_ENERGY:18.10f}',
            '',
            ' TDDFPT| State               eV              '
            'X           Y           Z        a.u.          f',
        ]
        for state, (exc, osc) in enumerate(excitations, 1):
            lines.append(
                f' TDDFPT|   {state}   {exc:9.5f}   '
                f'0.000000  0.000000  0.100000  0.010000  {osc:.5e}')
        return lines

    def test_no_declared_pair_leaves_the_transition_channel_empty(
            self, tmp_path):
        """A ground-state-only run answers with no row, not a zero row.

        The reader used to pad one (`npair = len(pairs) if pairs else 1`),
        which put a zero transition dipole and a zero oscillator strength in
        the results of a run that never asked for a transition property - and
        a spectrum plot then draws a dark line nobody computed.
        """
        config = normalize(build_config(
            _raw_config('tddft_energy', 'tddft', nstate=1)))
        assert [list(pair) for pair in config.transition_pair] == []
        _write_log(tmp_path, _energy_log())
        results = parse_output(config, str(tmp_path),
                               compile_workflow('tddft_energy'))
        assert np.asarray(results['transition_dipole']).shape == (0, 3)
        assert np.asarray(results['osc_strength']).shape == (0,)

    def test_excited_state_energies(self, tmp_path, tddft_setup):
        config, workflow = tddft_setup
        _write_log(tmp_path, self._tddft_log(
            [(5.00000, 0.010000), (7.50000, 0.040000)]))
        results = parse_output(config, str(tmp_path), workflow)
        ev_to_hartree = 1.0 / 27.211386245988
        assert results['energy'][0] == pytest.approx(S0_ENERGY)
        assert results['energy'][1] == pytest.approx(
            S0_ENERGY + 5.0 * ev_to_hartree)
        assert results['energy'][2] == pytest.approx(
            S0_ENERGY + 7.5 * ev_to_hartree)
        assert results['excitation'][1] == pytest.approx(5.0)

    def test_excited_state_gradients_from_tdfrc(self, tmp_path):
        """Row 0 comes from the force file, the excited row from the tdfrc block.

        A TDDFT run writes both, so a gradient step reads both: a crossing
        between a ground state and an excited one needs a force on each, and a
        zero-filled row 0 has the right shape to pass every check downstream
        while describing a surface nobody computed.
        """
        raw = _raw_config('tddft_gradient', 'tddft', nstate=3,
                          target_states=[0, 1])
        config = normalize(build_config(raw))
        workflow = compile_workflow('tddft_gradient')
        _write_log(tmp_path, self._tddft_log([(5.0, 0.01)]))
        _write_force_file(tmp_path)
        forces = [[0.01, -0.02, 0.03]] * 3
        _write_tdfrc(tmp_path, [(1, 5.0, 0.01, forces)])
        results = parse_output(config, str(tmp_path), workflow)
        assert np.allclose(results['gradient'][1], -np.array(forces))
        assert np.allclose(results['gradient'][0],
                           -np.asarray(FORCES, dtype=np.float64))
        assert np.max(np.abs(results['gradient'][0])) > 0.0

    def test_a_tddft_run_without_ground_state_forces_is_refused(self, tmp_path):
        """The missing force file is named, not read as a flat surface."""
        raw = _raw_config('tddft_gradient', 'tddft', nstate=3,
                          target_states=[0, 1])
        config = normalize(build_config(raw))
        workflow = compile_workflow('tddft_gradient')
        _write_log(tmp_path, self._tddft_log([(5.0, 0.01)]))
        _write_tdfrc(tmp_path, [(1, 5.0, 0.01, [[0.01, -0.02, 0.03]] * 3)])
        with pytest.raises(RuntimeError, match='no atomic forces'):
            parse_output(config, str(tmp_path), workflow)

    def test_tdfrc_block_count_mismatch_raises(self, tmp_path):
        raw = _raw_config('tddft_gradient', 'tddft', nstate=3,
                          target_states=[0, 1, 2])
        config = normalize(build_config(raw))
        workflow = compile_workflow('tddft_gradient')
        _write_log(tmp_path, self._tddft_log([(5.0, 0.01), (7.5, 0.04)]))
        # Only one block although two excited states are requested
        _write_tdfrc(tmp_path, [(1, 5.0, 0.01, [[0.0] * 3] * 3)])
        with pytest.raises(RuntimeError, match='tdfrc'):
            parse_output(config, str(tmp_path), workflow)


class TestTwoSpinManifolds:
    """CP2K prints one excited-state table per spin manifold, restarting numbers.

    The table rows below are transcribed from a real run that asked for both
    manifolds (`cp2k_demo/sp/bulk.log`, whose banners read `multiplicity 1` then
    `multiplicity 3` and whose two tables are each numbered 1..10). Only the
    `Total energy:` line is this fixture's own, because the demo system is not
    the water trimer the rest of this file parses.

    The reason this needs a test rather than a comment: with two tables numbered
    alike, the printed number alone describes two different surfaces, and a
    reader keyed on the number hands back the right count of rows with the wrong
    ones in them. That is not detected anywhere downstream - the shapes agree.
    """

    # Verbatim from cp2k_demo/sp/bulk.log:276 (banner) and :284-286 (rows).
    SINGLET_BANNER = ' R-TDDFPT states of multiplicity 1'
    SINGLET_ROWS = [
        ' TDDFPT|      1       3.94485   2.5439E+00  2.0296E-04 '
        '-1.9796E-04   6.25468E-01',
        ' TDDFPT|      2       3.94675   9.0953E-03  3.6198E-04 '
        '3.7413E-05   8.01177E-06',
        ' TDDFPT|      3       3.99014  -1.8325E-04  2.2943E+00 '
        '-4.7728E-05   5.14567E-01',
    ]
    # Verbatim from cp2k_demo/sp/bulk.log:1583 (banner) and :1590-1592 (rows).
    TRIPLET_BANNER = ' R-TDDFPT states of multiplicity 3'
    TRIPLET_ROWS = [
        ' TDDFPT|      1       3.76173   0.0000E+00  0.0000E+00 '
        '0.0000E+00   0.00000E+00',
        ' TDDFPT|      2       3.76947   0.0000E+00  0.0000E+00 '
        '0.0000E+00   0.00000E+00',
        ' TDDFPT|      3       3.77130   0.0000E+00  0.0000E+00 '
        '0.0000E+00   0.00000E+00',
    ]

    EV_TO_HARTREE = 1.0 / 27.211386245988

    def _log(self):
        lines = [
            ' *** SCF run converged ***',
            f' Total energy:                              '
            f'{S0_ENERGY:18.10f}',
            '',
            self.SINGLET_BANNER,
            '         State    Excitation        Transition dipole (a.u.)'
            '        Oscillator',
            '         number   energy (eV)       x           y           z'
            '     strength (a.u.)',
        ]
        lines += self.SINGLET_ROWS
        lines.append('')
        lines.append(self.TRIPLET_BANNER)
        lines += self.TRIPLET_ROWS
        return lines

    def _config(self, profile='tddft_energy', state_spin=(0, 0, 2),
                target_states=(0, 1, 2)):
        raw = _raw_config(profile, 'tddft', nstate=len(state_spin),
                          state_spin=list(state_spin),
                          target_states=list(target_states))
        return normalize(build_config(raw))

    def test_the_triplet_row_lands_on_the_triplet_and_not_the_second_singlet(
            self, tmp_path):
        """Row 2 is the triplet at 3.76173 eV, row 1 the singlet at 3.94485 eV.

        The two tables both print the number 1, so the energies are the only
        thing that can say which row a line belongs to once the spin is known:
        read the wrong way round, row 1 carries 3.76173 and the singlet is gone
        while every shape check still passes.
        """
        from ..service.parser.state_map import resolve_log_row

        config = self._config()
        _write_log(tmp_path, self._log())
        _write_force_file(tmp_path)
        results = parse_output(config, str(tmp_path),
                               compile_workflow('tddft_energy'))
        ev = self.EV_TO_HARTREE
        assert results['energy'][1] == pytest.approx(S0_ENERGY + 3.94485 * ev)
        assert results['energy'][2] == pytest.approx(S0_ENERGY + 3.76173 * ev)
        assert results['excitation'][1] == pytest.approx(3.94485)
        assert results['excitation'][2] == pytest.approx(3.76173)
        # The collision the banner resolves: two printed rows share the number 1.
        data = parse_log(str(tmp_path / f'{PROJECT}.log'), config)
        numbers = [(row['multiplicity'], row['state']) for row
                   in data['tddft_states']]
        assert (1, 1) in numbers and (3, 1) in numbers
        assert resolve_log_row(config, 1, 1) == 1
        assert resolve_log_row(config, 3, 1) == 2

    def test_extra_roots_the_engine_prints_are_not_packed(self, tmp_path):
        """NSTATES solves more roots than the run follows; the surplus is ignored.

        A request for one excited singlet and one triplet is answered by three of
        each in this excerpt. Rows 2 and 3 of a manifold belong to no framework
        row here, and taking them would be the same kind of error as the
        collision above, just quieter.
        """
        from ..service.parser.state_map import resolve_log_row

        config = self._config()
        _write_log(tmp_path, self._log())
        data = parse_log(str(tmp_path / f'{PROJECT}.log'), config)
        assert resolve_log_row(config, 1, 2) is None
        assert resolve_log_row(config, 3, 3) is None
        results = parse_output(config, str(tmp_path),
                               compile_workflow('tddft_energy'))
        ev = self.EV_TO_HARTREE
        assert results['energy'][1] == pytest.approx(S0_ENERGY + 3.94485 * ev)

    def test_the_oscillator_strength_of_the_singlet_survives_the_triplet(self,
                                                                        tmp_path):
        """A spin-forbidden triplet prints zero; that zero must not reach row 1.

        Keying on the printed number alone leaves the last table winning, and the
        winner here is the triplet with its vanishing oscillator strength, so the
        spectrum an excitation analysis reports turns dark without any error.
        """
        config = self._config()
        _write_log(tmp_path, self._log())
        results = parse_output(config, str(tmp_path),
                               compile_workflow('tddft_energy'))
        # Rows follow the declared transition pairs in order: [0, 1] then
        # [0, 2] (the self pair is no longer part of the default table).
        assert results['osc_strength'][0] == pytest.approx(0.625468)
        assert results['osc_strength'][1] == pytest.approx(0.0)

    def test_blocks_printed_with_restarting_numbers_still_find_their_rows(
            self, tmp_path):
        """The header counts per manifold, and the energies carry the attribution.

        This is the reading the test round came for, and the parser did not
        have to guess it: under the per-manifold counting each of these two blocks
        matches exactly one requested row, so no tie-break is used and nothing is
        warned about. Under the continuing counting the same pair would collide,
        which is how a run reports which counting an engine actually used.
        """
        config = self._config()
        _write_log(tmp_path, self._log())
        _write_force_file(tmp_path)
        singlet_forces = [[0.01, 0.02, 0.03]] * 3
        triplet_forces = [[-0.04, 0.05, -0.06]] * 3
        _write_tdfrc(tmp_path, [
            (1, 3.94485, 0.625468, singlet_forces),
            (1, 3.76173, 0.0, triplet_forces),
        ])
        results = parse_output(config, str(tmp_path),
                               compile_workflow('tddft_gradient'))
        assert np.allclose(results['gradient'][1], -np.array(singlet_forces))
        assert np.allclose(results['gradient'][2], -np.array(triplet_forces))

    def test_a_block_no_reading_explains_is_refused(self, tmp_path):
        """A header number outside both readings is a failure, not a dropped row.

        The block here claims to be the fourth root, which is neither the
        per-manifold nor the continuing count for this request. Silently leaving
        it out would return one surface short of the pair a crossing search
        steps on.
        """
        config = self._config()
        _write_log(tmp_path, self._log())
        _write_force_file(tmp_path)
        _write_tdfrc(tmp_path, [
            (1, 3.94485, 0.625468, [[0.01, 0.02, 0.03]] * 3),
            (9, 3.76173, 0.0, [[-0.04, 0.05, -0.06]] * 3),
        ])
        with pytest.raises(RuntimeError, match='neither reading'):
            parse_output(config, str(tmp_path),
                         compile_workflow('tddft_gradient'))


class TestFinalGeometryChannel:
    """fin_coord from the trajectory for GEO_OPT and MD."""

    FINAL_COORDS = [
        [0.000000000, 0.000000000, 0.120000000],
        [0.000000000, 0.760000000, -0.480000000],
        [0.000000000, -0.760000000, -0.480000000],
    ]

    def test_optimization_returns_fin_coord(self, tmp_path):
        config = normalize(build_config(_raw_config('dft_opt_min', 'dft')))
        workflow = compile_workflow('dft_opt_min')
        log = _energy_log() + ['', ' GEOMETRY OPTIMIZATION COMPLETED']
        _write_log(tmp_path, log)
        _write_pos_file(tmp_path, [('frame 1', COORDS),
                                   ('frame 2', self.FINAL_COORDS)])
        results = parse_output(config, str(tmp_path), workflow)
        assert np.allclose(results['fin_coord'], self.FINAL_COORDS)
        assert results['opt_converge'] is True
        assert results['energy'][0] == pytest.approx(S0_ENERGY)

    def test_md_returns_last_frame(self, tmp_path):
        config = normalize(build_config(_raw_config('dft_md', 'dft')))
        workflow = compile_workflow('dft_md')
        _write_log(tmp_path, _energy_log())
        _write_pos_file(tmp_path, [('step 1', COORDS),
                                   ('step 2', self.FINAL_COORDS)])
        results = parse_output(config, str(tmp_path), workflow)
        assert np.allclose(results['fin_coord'], self.FINAL_COORDS)

    def test_missing_trajectory_raises(self, tmp_path):
        config = normalize(build_config(_raw_config('dft_opt_min', 'dft')))
        workflow = compile_workflow('dft_opt_min')
        _write_log(tmp_path, _energy_log())
        with pytest.raises(RuntimeError, match='final geometry'):
            parse_output(config, str(tmp_path), workflow)


# A CP2K analytical stress block (compressive-positive, GPa); the
# parser flips the sign to the tensile-positive potential stress.
STRESS_BLOCK = [
    ' STRESS TENSOR [GPa]',
    '            X               Y               Z',
    ' X     1.00000000     0.50000000     0.00000000',
    ' Y     0.50000000     2.00000000     0.00000000',
    ' Z     0.00000000     0.00000000     3.00000000',
    ' 1/3 Trace (X,Y,Z)          2.00000000',
    ' Determinant                  5.25000000',
]
STRESS_GPA = np.array([[1.0, 0.5, 0.0],
                       [0.5, 2.0, 0.0],
                       [0.0, 0.0, 3.0]])


class TestStressTensor:
    """Whole-system stress extraction: parse when requested, else zero."""

    def test_zero_filled_when_not_requested(self, tmp_path):
        config = normalize(build_config(_raw_config('dft_gradient', 'dft')))
        workflow = compile_workflow('dft_gradient')
        _write_log(tmp_path, _energy_log() + [''] + STRESS_BLOCK)
        _write_force_file(tmp_path)
        results = parse_output(config, str(tmp_path), workflow)
        assert results['stress_tensor'].shape == (1, 3, 3)
        assert np.allclose(results['stress_tensor'], 0.0)

    def test_stress_parsed_and_negated(self, tmp_path):
        config = normalize(build_config(
            _raw_config('dft_gradient', 'dft', stress=True)))
        workflow = compile_workflow('dft_gradient')
        _write_log(tmp_path, _energy_log() + [''] + STRESS_BLOCK)
        _write_force_file(tmp_path)
        results = parse_output(config, str(tmp_path), workflow)
        assert results['stress_tensor'].shape == (1, 3, 3)
        # compressive-positive GPa -> tensile-positive (negated)
        assert np.allclose(results['stress_tensor'][0], -STRESS_GPA)

    def test_missing_block_is_refused_when_requested(self, tmp_path):
        config = normalize(build_config(
            _raw_config('dft_gradient', 'dft', stress=True)))
        workflow = compile_workflow('dft_gradient')
        _write_log(tmp_path, _energy_log())
        _write_force_file(tmp_path)
        with pytest.raises(RuntimeError, match='carries no stress block'):
            parse_output(config, str(tmp_path), workflow)

    def test_missing_block_stays_zero_when_not_requested(self, tmp_path):
        config = normalize(build_config(_raw_config('dft_gradient', 'dft')))
        workflow = compile_workflow('dft_gradient')
        _write_log(tmp_path, _energy_log())
        _write_force_file(tmp_path)
        results = parse_output(config, str(tmp_path), workflow)
        assert np.allclose(results['stress_tensor'], 0.0)

    def test_dedicated_file_preferred_over_log(self, tmp_path):
        # The builder writes {project}_stress_tensor.dat; it must win
        # over any block left in the main log.
        config = normalize(build_config(
            _raw_config('dft_gradient', 'dft', stress=True)))
        workflow = compile_workflow('dft_gradient')
        _write_log(tmp_path, _energy_log() + [''] + STRESS_BLOCK)
        _write_force_file(tmp_path)
        _write_stress_file(tmp_path, [
            ' STRESS TENSOR [GPa]',
            '            X               Y               Z',
            ' X     9.00000000     0.00000000     0.00000000',
            ' Y     0.00000000     9.00000000     0.00000000',
            ' Z     0.00000000     0.00000000     9.00000000',
        ])
        results = parse_output(config, str(tmp_path), workflow)
        assert np.allclose(results['stress_tensor'][0],
                           -9.0 * np.eye(3))

    def test_real_log_format_prefixed_bar(self, tmp_path):
        # Real CP2K main-log block: 'STRESS|' module prefix, lowercase
        # x/y/z labels, E-notation, bar unit, followed by an eigenvalue
        # block whose x/y/z rows must NOT be mistaken for the tensor.
        config = normalize(build_config(
            _raw_config('dft_gradient', 'dft', stress=True)))
        workflow = compile_workflow('dft_gradient')
        real_block = [
            ' STRESS| Analytical stress tensor [bar]',
            ' STRESS|                        x                   y'
            '                   z',
            ' STRESS|      x        1.00000000000E+00'
            '   5.00000000000E-01   0.00000000000E+00',
            ' STRESS|      y        5.00000000000E-01'
            '   2.00000000000E+00   0.00000000000E+00',
            ' STRESS|      z        0.00000000000E+00'
            '   0.00000000000E+00   3.00000000000E+00',
            ' STRESS| 1/3 Trace                    2.00000000000E+00',
            ' STRESS| Determinant                  5.25000000000E+00',
            ' STRESS| Eigenvectors and eigenvalues of the analytical'
            ' stress tensor [bar]',
            ' STRESS|                        1                   2'
            '                   3',
            ' STRESS| Eigenvalues   1.0E+00   2.0E+00   3.0E+00',
            ' STRESS|      x          1.0   0.0   0.0',
            ' STRESS|      y          0.0   1.0   0.0',
            ' STRESS|      z          0.0   0.0   1.0',
        ]
        _write_log(tmp_path, _energy_log() + [''] + real_block)
        _write_force_file(tmp_path)
        results = parse_output(config, str(tmp_path), workflow)
        # bar -> GPa (1e-4), compressive-positive -> tensile (negated)
        expected = -STRESS_GPA * 1.0e-4
        assert np.allclose(results['stress_tensor'][0], expected)


class TestCellFromTheTrajectory:
    """The frame that gives the coordinates also gives the cell, when the
    run was periodic -- and only then."""

    LATTICE = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]

    def test_quoted_lattice_is_read(self, tmp_path):
        from ..service.parser.log_parser import parse_xyz_cell

        tag = ('Properties=types=1:exact=yes '
               'Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0"')
        path = _write_pos_file(tmp_path, [(tag, COORDS)])
        assert np.allclose(parse_xyz_cell(str(path)), self.LATTICE)

    def test_braced_lattice_with_a_leading_word_is_read(self, tmp_path):
        # Versions differ in whether the value is quoted or braced, and
        # some stamp a word in front of the nine numbers.
        from ..service.parser.log_parser import parse_xyz_cell

        tag = 'Properties=exact=yes Lattice={"Cell" 10 0 0 0 10 0 0 0 10}'
        path = _write_pos_file(tmp_path, [(tag, COORDS)])
        assert np.allclose(parse_xyz_cell(str(path)), self.LATTICE)

    def test_a_comment_without_a_cell_reads_as_none(self, tmp_path):
        from ..service.parser.log_parser import parse_xyz_cell

        path = _write_pos_file(tmp_path, [('frame 1', COORDS)])
        assert parse_xyz_cell(str(path)) == []

    def test_the_last_frame_with_a_cell_wins(self, tmp_path):
        from ..service.parser.log_parser import parse_xyz_cell

        quoted = 'Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0"'
        grown = 'Lattice="11.0 0.0 0.0 0.0 11.0 0.0 0.0 0.0 11.0"'
        path = _write_pos_file(tmp_path, [(quoted, COORDS),
                                          ('plain', self.FINAL),
                                          (grown, self.FINAL)])
        lattice = parse_xyz_cell(str(path))
        assert lattice[0][0] == pytest.approx(11.0)

    FINAL = COORDS

    def test_optimization_reports_fin_cell(self, tmp_path):
        config = normalize(build_config(_raw_config('dft_opt_min', 'dft')))
        workflow = compile_workflow('dft_opt_min')
        _write_log(tmp_path, _energy_log())
        _write_pos_file(tmp_path, [
            ('Lattice="10 0 0 0 10 0 0 0 10"', COORDS),
            ('Lattice="10.5 0 0 0 10.5 0 0 0 10.5"', COORDS)])
        results = parse_output(config, str(tmp_path), workflow)
        assert results['fin_cell'] == pytest.approx(
            [10.5, 0.0, 0.0, 0.0, 10.5, 0.0, 0.0, 0.0, 10.5])

    def test_a_molecular_run_reports_no_cell(self, tmp_path):
        # No Lattice= in the comment means no cell was ever printed; the
        # channel stays absent rather than invented from the input.
        config = normalize(build_config(_raw_config('dft_opt_min', 'dft')))
        workflow = compile_workflow('dft_opt_min')
        _write_log(tmp_path, _energy_log())
        _write_pos_file(tmp_path, [('frame 1', COORDS)])
        results = parse_output(config, str(tmp_path), workflow)
        assert 'fin_cell' not in results

    def test_a_zero_lattice_is_not_a_cell(self, tmp_path):
        # A molecular structure written by ASE carries exactly this: the
        # slot is there, the box is not. Reporting it would be a cell of
        # zero volume, which the framework correctly refuses.
        from ..service.parser.log_parser import parse_xyz_cell

        tag = ('Lattice="0 0 0 0 0 0 0 0 0" '
               'Properties=species:S:1:pos:R:3 pbc="F F F"')
        path = _write_pos_file(tmp_path, [(tag, COORDS)])
        assert parse_xyz_cell(str(path)) == []

    def test_numbers_after_the_lattice_do_not_leak_in(self, tmp_path):
        # energy= and pbc= follow the nine components; only the cell may
        # be read out of a line that carries other numbers too.
        from ..service.parser.log_parser import parse_xyz_cell

        tag = ('Lattice="10 0 0 0 10 0 0 0 10" energy=-170.5 '
               'pbc="T T T"')
        path = _write_pos_file(tmp_path, [(tag, COORDS)])
        lattice = parse_xyz_cell(str(path))
        assert np.allclose(lattice, [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])


class TestTwoManifoldsAreStitchedByRow:
    """A split request is answered by two runs; each row comes from its own.

    The tables below are transcribed from the A/B run made in the test environment
    machine (CP2K 2025.2): the SAME ethylene input, the same basis and cutoff,
    differing only in `RKS_TRIPLETS` and carrying no `&SOC`. `T` printed
    `multiplicity 3` and no singlet table at all (`out_T.log:263`, rows at
    :269-270); `F` printed `multiplicity 1` (`out_F.log:263`, rows at :269-270).
    Both printed the same SCF baseline, `Total energy: -13.56482679723119`
    (line 206 of each), which is what makes one axis out of two runs safe to
    build.

    What these tests exist to catch: a merge that takes a row from the wrong run
    still produces `(nstate,)` with four finite numbers in it. Nothing down the
    chain can see that the triplet's energy came out of a singlet's table, so
    the claim is checked where it is made and the agreement is asserted where
    two runs overlap.
    """

    # Verbatim, both runs.
    S0 = -13.56482679723119
    HEADER = [
        '         State    Excitation        Transition dipole (a.u.)'
        '        Oscillator',
        '         number   energy (eV)       x           y           z'
        '     strength (a.u.)',
        '         ----------------------------------------------------'
        '----',
    ]
    SINGLET_LOG = [
        ' *** SCF run converged ***',
        f' Total energy:                              {S0:18.10f}',
        '',
        ' R-TDDFPT states of multiplicity 1',
        ' Transition dipoles calculated using length formulation',
        '',
    ] + HEADER + [
        ' TDDFPT|      1       1.49462   1.5144E-06  3.6088E-07  5.4721E+00'
        '   1.09646E+00',
        ' TDDFPT|      2       7.15881  -4.1730E-01 -4.1733E-01  2.3686E-05'
        '   6.10876E-02',
    ]
    TRIPLET_LOG = [
        ' *** SCF run converged ***',
        f' Total energy:                              {S0:18.10f}',
        '',
        ' R-TDDFPT states of multiplicity 3',
        ' Transition dipoles calculated using length formulation',
        '',
    ] + HEADER + [
        ' TDDFPT|      1      -1.41009   0.0000E+00  0.0000E+00  0.0000E+00'
        '  -0.00000E+00',
        ' TDDFPT|      2       5.99128   0.0000E+00  0.0000E+00  0.0000E+00'
        '   0.00000E+00',
    ]

    EV_TO_HARTREE = 1.0 / 27.211386245988

    def _request(self):
        """S0, two singlets and one triplet: the shape the A/B pair answers."""
        raw = _raw_config('tddft_energy', 'tddft', nstate=4,
                          state_spin=[0, 0, 0, 2],
                          target_states=[0, 1, 2, 3])
        return normalize(build_config(raw))

    def _jobs(self, config):
        return plan_jobs(config, compile_workflow(config.profile))

    def _stage(self, tmp_path, lines):
        """Write each job's log under its own project name, as the runs do."""
        for project, body in lines.items():
            (tmp_path / f'{project}.log').write_text('\n'.join(body) + '\n')

    def _parsed(self, tmp_path, config, overrides=None):
        results = []
        lines = {'water_S': list(self.SINGLET_LOG),
                 'water_T': list(self.TRIPLET_LOG)}
        for project, body in (overrides or {}).items():
            lines[project] = body
        self._stage(tmp_path, lines)
        for job in self._jobs(config):
            results.append((job, parse_output(job.config, str(tmp_path),
                                             compile_workflow('tddft_energy'))))
        return results

    def test_each_row_comes_from_the_run_that_solved_it(self, tmp_path):
        config = self._request()
        pairs = self._parsed(tmp_path, config)
        merged = merge_job_results([job for job, _ in pairs],
                                   [res for _, res in pairs], config)
        ev = self.EV_TO_HARTREE
        assert merged['energy'][0] == pytest.approx(self.S0)
        # S1 and S2 out of the singlet run, T1 out of the triplet one - the four
        # numbers the engine printed, in the framework's row order.
        assert merged['energy'][1] == pytest.approx(self.S0 + 1.49462 * ev)
        assert merged['energy'][2] == pytest.approx(self.S0 + 7.15881 * ev)
        assert merged['energy'][3] == pytest.approx(self.S0 - 1.41009 * ev)
        assert merged['excitation'][1:] == pytest.approx(
            [1.49462, 7.15881, -1.41009])

    def test_the_root_the_run_did_not_answer_stays_unpacked(self, tmp_path):
        # The triplet run solves NSTATES 1 and CP2K still prints a second row
        # (5.99128 eV, the surplus the Davidson space produced). It names no
        # framework row and must not be packed as if it did.
        config = self._request()
        triplet_job = self._jobs(config)[1]
        assert triplet_job.config.tddft_nstates == 1
        self._stage(tmp_path, {'water_T': self.TRIPLET_LOG})
        parsed = parse_output(triplet_job.config, str(tmp_path),
                              compile_workflow('tddft_energy'))
        assert parsed['excitation'][3] == pytest.approx(-1.41009)
        assert parsed['excitation'][1] == 0.0
        assert parsed['excitation'][2] == 0.0

    def test_the_two_runs_must_agree_on_the_ground_state(self, tmp_path):
        # The A/B showed both runs print the same SCF baseline. If one of them
        # lands elsewhere, the two are not describing one geometry and one
        # reference, and the merge has no right to pick either.
        drifted = [line.replace(f'{self.S0:18.10f}',
                               f'{self.S0 + 1.0e-6:18.10f}')
                   for line in self.TRIPLET_LOG]
        config = self._request()
        pairs = self._parsed(tmp_path, config, overrides={'water_T': drifted})
        with pytest.raises(RuntimeError, match='row 0'):
            merge_job_results([job for job, _ in pairs],
                              [res for _, res in pairs], config)

    def test_a_table_answered_by_the_wrong_run_is_refused_where_claimed(
            self, tmp_path):
        # Stage the triplet table under the singlet run's project: its row maps
        # onto a 2S=2 surface while this run solves 2S=0. The refusal belongs in
        # the pack, because a merge that simply took rows by owner would hide the
        # swap and the singlet run would have printed nothing it was asked for.
        config = self._request()
        singlet_job = self._jobs(config)[0]
        self._stage(tmp_path, {'water_S': self.TRIPLET_LOG})
        with pytest.raises(RuntimeError, match='belongs to the other run'):
            parse_output(singlet_job.config, str(tmp_path),
                         compile_workflow('tddft_energy'))

    def test_one_run_returns_the_dict_it_parsed(self, tmp_path):
        # The identity arm: a single-manifold request never reaches the merge
        # logic at all, so every shipped case keeps its own numbers untouched.

        config = _raw_config('tddft_energy', 'tddft', nstate=2,
                             state_spin=[0, 2])
        single = normalize(build_config(config))
        job = Job(None, single)
        _write_log(tmp_path, self.TRIPLET_LOG)
        parsed = parse_output(single, str(tmp_path),
                              compile_workflow('tddft_energy'))
        assert merge_job_results([job], [parsed], single) is parsed

    def test_a_pair_both_runs_printed_differently_is_refused(self):
        # No engine files here: the pair channels are indexed by `transition_pair`
        # order, and two runs filling one position with different numbers means
        # the split was not disjoint.

        config = self._request()
        jobs = [Job(0, config), Job(2, config)]
        first = {'transition_dipole': np.array([[0.5, 0.0, 0.0]])}
        second = {'transition_dipole': np.array([[0.9, 0.0, 0.0]])}
        with pytest.raises(RuntimeError, match='disjoint'):
            merge_job_results(jobs, [first, second], config)

    def test_a_single_owner_channel_that_disagrees_is_refused(self):

        config = self._request()
        jobs = [Job(0, config), Job(2, config)]
        first = {'frequencies': np.array([100.0, 200.0])}
        second = {'frequencies': np.array([100.0, 210.0])}
        with pytest.raises(RuntimeError, match='one run per evaluation'):
            merge_job_results(jobs, [first, second], config)

    def test_a_channel_only_the_first_run_has_does_not_become_zero(self):
        # The merge must not invent an answer for a channel the other run never
        # produced; it keeps the owner's value.

        config = self._request()
        jobs = [Job(0, config), Job(2, config)]
        first = {'soc': np.array([1.25])}
        second = {}
        merged = merge_job_results(jobs, [first, second], config)
        assert merged['soc'].tolist() == [1.25]
