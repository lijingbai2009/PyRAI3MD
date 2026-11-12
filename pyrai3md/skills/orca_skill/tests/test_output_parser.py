"""
Unit tests for orca_skill result extraction (output_parser).

Verifies the molcas-aligned return contract with mock calcdir
files: fin_coord/opt_converge channel for optimizations, the
pair-driven transition_dipole/osc_strength extraction and the
engrad-based ground-state gradient path.
These tests do NOT require an ORCA installation.

Run: pytest orca_skill/tests/test_output_parser.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import numpy as np
import pytest

from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..service.workflow import compile_workflow
from ..service.parser.output_parser import parse_output, format_results

# Water molecule keeps the mock logs compact
ATOMS = ['O', 'H', 'H']
COORDS = [
    [0.000000000, 0.000000000, 0.117300000],
    [0.000000000, 0.757200000, -0.469200000],
    [0.000000000, -0.757200000, -0.469200000],
]

# Optimized geometry block printed by ORCA after the relaxation
OPT_COORDS = [
    [0.000000000, 0.000000000, 0.120000000],
    [0.000000000, 0.760000000, -0.480000000],
    [0.000000000, -0.760000000, -0.480000000],
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
        'functional': 'pbe0',
        'basis': 'sto-3g',
    }
    raw.update(kwargs)
    return raw


def _coord_block(coords):
    """Render a CARTESIAN COORDINATES (ANGSTROEM) log block.

    The parser skips the two lines following the header (separator
    and column caption), matching the real ORCA layout.
    """
    lines = [
        'CARTESIAN COORDINATES (ANGSTROEM)',
        '---------------------------------------------',
        '   x           y           z',
    ]
    for atom, (x, y, z) in zip(ATOMS, coords):
        lines.append(f'  {atom}   {x:12.6f} {y:12.6f} {z:12.6f}')
    return lines


def _absorption_rows():
    """Render the transition dipole absorption spectrum section.

    Row columns after the symmetry label: X, Y, Z, |D|^2, |D|,
    excitation (eV), oscillator strength.
    """
    return [
        'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS',
        'state          X           Y           Z',
        '   1( 1A  )   0.100000   0.000000   0.000000   0.010000'
        '   0.100000   5.000000   0.010000',
        '   2( 1A  )   0.000000   0.200000   0.000000   0.040000'
        '   0.200000   7.500000   0.040000',
    ]


def _write_log(calcdir, project, lines):
    path = calcdir / f'{project}.out'
    path.write_text('\n'.join(lines) + '\n')
    return path


def _write_xyz(calcdir, project, coords):
    """Render the {project}.xyz file ORCA updates each opt cycle."""
    lines = [str(len(coords)), f'{project} optimized geometry']
    for atom, (x, y, z) in zip(ATOMS, coords):
        lines.append(f'{atom}  {x:12.6f} {y:12.6f} {z:12.6f}')
    path = calcdir / f'{project}.xyz'
    path.write_text('\n'.join(lines) + '\n')
    return path


@pytest.fixture
def optimize_setup():
    config = normalize(build_config(_raw_config('dft_opt_min', 'dft')))
    return config, compile_workflow('dft_opt_min')


@pytest.fixture
def tddft_setup():
    raw = _raw_config('tddft_energy', 'tddft', nstate=3)
    config = normalize(build_config(raw))
    return config, compile_workflow('tddft_energy')


class TestOptimizationReturnContract:
    """fin_coord / opt_converge channel for optimize profiles."""

    def test_optimize_returns_fin_coord_and_opt_converge(
            self, tmp_path, optimize_setup):
        config, workflow = optimize_setup
        # The log keeps the input geometry while the relaxed
        # structure lands in {project}.xyz; fin_coord must come
        # from the xyz file.
        lines = _coord_block(COORDS)
        lines.append('FINAL SINGLE POINT ENERGY       -76.100000000')
        lines.append('OPTIMIZATION RUN DONE')
        _write_log(tmp_path, config.project, lines)
        _write_xyz(tmp_path, config.project, OPT_COORDS)

        results = parse_output(config, str(tmp_path), workflow)

        assert 'fin_coord' in results
        assert 'coordinates' not in results
        assert results['fin_coord'].shape == (3, 3)
        assert np.allclose(results['fin_coord'], OPT_COORDS)
        assert results['opt_converge'] is True
        assert results['energy'][0] == pytest.approx(-76.1)

    def test_optimize_falls_back_to_log_without_xyz(
            self, tmp_path, optimize_setup):
        config, workflow = optimize_setup
        lines = _coord_block(OPT_COORDS)
        lines.append('OPTIMIZATION RUN DONE')
        _write_log(tmp_path, config.project, lines)

        results = parse_output(config, str(tmp_path), workflow)

        assert np.allclose(results['fin_coord'], OPT_COORDS)

    def test_optimize_malformed_xyz_falls_back_to_log(
            self, tmp_path, optimize_setup):
        config, workflow = optimize_setup
        lines = _coord_block(OPT_COORDS)
        lines.append('OPTIMIZATION RUN DONE')
        _write_log(tmp_path, config.project, lines)
        (tmp_path / f'{config.project}.xyz').write_text(
            '3\nbroken\nonly one atom line\n')

        results = parse_output(config, str(tmp_path), workflow)

        assert np.allclose(results['fin_coord'], OPT_COORDS)

    def test_optimize_unconverged_reports_false(
            self, tmp_path, optimize_setup):
        config, workflow = optimize_setup
        lines = _coord_block(OPT_COORDS)
        lines.append('FINAL SINGLE POINT ENERGY       -76.050000000')
        # No OPTIMIZATION RUN DONE marker: truncated step budget
        _write_log(tmp_path, config.project, lines)

        results = parse_output(config, str(tmp_path), workflow)

        assert results['opt_converge'] is False
        assert results['fin_coord'].shape == (3, 3)

    def test_optimize_missing_final_geometry_raises(
            self, tmp_path, optimize_setup):
        config, workflow = optimize_setup
        _write_log(tmp_path, config.project,
                   ['FINAL SINGLE POINT ENERGY       -76.100000000'])

        with pytest.raises(RuntimeError, match='no final geometry'):
            parse_output(config, str(tmp_path), workflow)

    def test_format_results_serializes_fin_coord(
            self, tmp_path, optimize_setup):
        config, workflow = optimize_setup
        lines = _coord_block(OPT_COORDS)
        lines.append('OPTIMIZATION RUN DONE')
        _write_log(tmp_path, config.project, lines)
        _write_xyz(tmp_path, config.project, OPT_COORDS)

        formatted = format_results(
            parse_output(config, str(tmp_path), workflow), config)

        assert isinstance(formatted['fin_coord'], list)
        assert len(formatted['fin_coord']) == 3


class TestTransitionPairContract:
    """transition_dipole/osc_strength extracted per transition_pair."""

    def test_default_pairs_follow_ground_to_state(
            self, tmp_path, tddft_setup):
        config, workflow = tddft_setup
        _write_log(tmp_path, config.project, _absorption_rows())

        results = parse_output(config, str(tmp_path), workflow)

        # normalize derives the declared ground-to-state pairs, so a 3-state
        # run has rows for [0, 1] and [0, 2]: the self pair [0, 0] is not one
        # of them and no longer takes a zero row (#448).
        assert results['transition_dipole'].shape == (2, 3)
        assert results['osc_strength'].shape == (2,)
        assert np.allclose(results['transition_dipole'][0],
                           [0.1, 0.0, 0.0])
        assert np.allclose(results['transition_dipole'][1],
                           [0.0, 0.2, 0.0])
        assert results['osc_strength'][0] == pytest.approx(0.01)
        assert results['osc_strength'][1] == pytest.approx(0.04)

    def test_excited_to_excited_pair_stays_zero(
            self, tmp_path, tddft_setup):
        config, workflow = tddft_setup
        config.system.transition_pair = [[0, 2], [1, 2]]
        _write_log(tmp_path, config.project, _absorption_rows())

        results = parse_output(config, str(tmp_path), workflow)

        assert results['transition_dipole'].shape == (2, 3)
        assert np.allclose(results['transition_dipole'][0],
                           [0.0, 0.2, 0.0])
        # ORCA computes no excited-state to excited-state transitions
        assert np.allclose(results['transition_dipole'][1], 0.0)
        assert results['osc_strength'][1] == 0.0

    def test_excitation_stays_state_resolved(self, tmp_path, tddft_setup):
        config, workflow = tddft_setup
        _write_log(tmp_path, config.project, _absorption_rows())

        results = parse_output(config, str(tmp_path), workflow)

        assert results['excitation'].shape == (3,)
        assert results['excitation'][0] == 0.0
        assert results['excitation'][1] == pytest.approx(5.0)
        assert results['excitation'][2] == pytest.approx(7.5)


class TestGroundStateGradient:
    """Ground-state DFT gradient reads the .engrad file."""

    def test_dft_gradient_from_engrad(self, tmp_path):
        config = normalize(build_config(
            _raw_config('dft_gradient', 'dft')))
        workflow = compile_workflow('dft_gradient')

        engrad = [
            '#',
            '# The current total energy in Eh',
            '#',
            '    -76.123456789',
            '#',
            '# The current gradient in Eh/bohr',
            '#',
        ] + ['  0.00%d' % (i + 1) for i in range(9)]
        (tmp_path / f'{config.project}.engrad').write_text(
            '\n'.join(engrad) + '\n')

        results = parse_output(config, str(tmp_path), workflow)

        assert results['energy'][0] == pytest.approx(-76.123456789)
        assert results['gradient'].shape == (1, 3, 3)
        assert results['gradient'][0, 0, 0] == pytest.approx(0.001)
        assert results['gradient'][0, 2, 2] == pytest.approx(0.009)


class TestSocRootMapping:
    """nroots derivation and ORCA->framework SOC state mapping.

    ORCA computes nroots singlet and nroots triplet roots with
    'Triplets true', so normalize requests max(excited singlets,
    triplets) instead of nstate - 1, and the parser maps the
    combined singlet-first ORCA state numbering back onto the
    framework state list (S0, excited singlets, then triplets).
    """

    @staticmethod
    def _soc_config(coupling=None):
        raw = _raw_config('tddft_soc', 'tddft', nstate=6,
                          state_spin=[0, 0, 0, 2, 2, 2])
        if coupling is not None:
            raw['coupling'] = coupling
        return normalize(build_config(raw))

    def test_mixed_spin_nroots_is_max_of_singlet_triplet(self):
        # [0,0,0,2,2,2]: 2 excited singlets, 3 triplets -> nroots 3
        config = self._soc_config()
        assert config.tddft_nroots == 3
        assert config.nsing_root == 2
        assert config.ntrip_root == 3

    def test_pure_singlet_nroots_unchanged(self):
        config = normalize(build_config(
            _raw_config('tddft_energy', 'tddft', nstate=3)))
        assert config.tddft_nroots == 2
        assert config.nsing_root == 2
        assert config.ntrip_root == 0

    def test_socme_row_mapping(self):
        # The table numbers the triplet root from 1 and the singlet root from
        # 0 (0 is the ground state); the framework orders S0, S1..Sn, T1..Tm.
        from ..service.parser.output_parser import _map_orca_socme_pair
        config = self._soc_config()
        assert config.nsing_root == 2          # S1, S2; nsinglet = 3 with S0
        assert _map_orca_socme_pair(1, 0, config) == (0, 3)
        assert _map_orca_socme_pair(1, 1, config) == (1, 3)
        assert _map_orca_socme_pair(3, 2, config) == (2, 5)

    @staticmethod
    def _socme_table(rows):
        """A DoSOC SOCME block in the shape the deployment prints."""
        lines = [
            '      -----------------------------------------------------------------'
            '---------------',
            '                      CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS'
            '                  ',
            '      -----------------------------------------------------------------'
            '---------------',
            '           Root                          <T|HSO|S>  (Re, Im) cm-1     '
            '                ',
            '         T      S             Z                     X                   '
            '  Y           ',
            '      -----------------------------------------------------------------'
            '---------------',
        ]
        for triplet_root, singlet_root, magnitude in rows:
            lines.append(
                '         %d      %d    (   0.00 ,%8.2f )    (   0.00 ,    0.00 )'
                '    (  -0.00 ,   -0.00 )'
                % (triplet_root, singlet_root, magnitude))
        return lines

    def test_pack_soc_uses_mapping(self, tmp_path):
        config = self._soc_config()
        workflow = compile_workflow('tddft_soc')
        pairs = [pair for pair in config.soc_pair_map if len(pair) == 2]
        # DoSOC numbers excited roots only, so no pair through S0 is asked
        # for (schema/normalize.py) and every row below has an element.
        assert pairs == [[1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]]
        nsinglet = config.nsing_root + 1
        rows = [(triplet - nsinglet + 1, singlet, 10.0 * singlet + triplet)
                for singlet, triplet in pairs]
        log = [
            'FINAL SINGLE POINT ENERGY  -76.000000000',
            # `tddft_soc` carries the gradient step, and a charge-free deck
            # writes EnGrad for it (no NACME in this profile), so the log
            # holds the block the parser counts.
            *TestTddftGradientPacking._gradient_block(1.0),
        ] + self._socme_table(rows) + ['SOC stabilization of the ground '
                                       'state: -14971.1619 cm-1']
        _write_log(tmp_path, config.project, log)

        results = parse_output(config, str(tmp_path), workflow)

        assert len(np.asarray(results['soc'])) == len(pairs)
        for singlet, triplet in pairs:
            expected = 10.0 * singlet + triplet
            assert results['soc'][pairs.index([singlet, triplet])] == \
                pytest.approx(expected)

    def test_a_declared_pair_the_log_does_not_answer_is_no_row(self, tmp_path):
        # It used to be recorded as 0.0. Zero says "these two states do not
        # couple", which is a physics statement the run never made, and an
        # ISC rate in a surface-hopping run reads it as its own input. The row
        # for (2, 5) is left out of the log below, and the reader names the
        # pair rather than inventing its value.
        config = self._soc_config()
        workflow = compile_workflow('tddft_soc')
        pairs = [pair for pair in config.soc_pair_map if len(pair) == 2]
        nsinglet = config.nsing_root + 1
        rows = [(triplet - nsinglet + 1, singlet, 10.0 * singlet + triplet)
                for singlet, triplet in pairs if [singlet, triplet] != [2, 5]]
        log = [
            'FINAL SINGLE POINT ENERGY  -76.000000000',
            *TestTddftGradientPacking._gradient_block(1.0),
        ] + self._socme_table(rows) + ['SOC stabilization of the ground '
                                       'state: -14971.1619 cm-1']
        _write_log(tmp_path, config.project, log)

        with pytest.raises(RuntimeError, match=r'SOC pair \[2, 5\]'):
            parse_output(config, str(tmp_path), workflow)


class TestNacBlocksAreAddressedByTheRequest:
    """NACME blocks carry no root, so their order is their identity."""

    HEADER = [
        '---------------------------------',
        'CARTESIAN NON-ADIABATIC COUPLINGS',
        '          <GS|d/dx|ES>',
        '---------------------------------',
    ]

    @classmethod
    def _block(cls, scale, natom=6):
        # The row shape the deployment prints (round `tmp_20261101_224544`).
        rows = ['   %d   C   :   %0.9f    %0.9f   %0.9f'
                % (index + 1, scale if index == 0 else 0.0, 0.0,
                   scale if index == 1 else 0.0)
                for index in range(natom)]
        return cls.HEADER + rows + ['Norm of the NACs                   '
                                    '...    %0.9f' % scale]

    _COORDS = _coord_block(COORDS)

    def _config(self):
        return normalize(build_config(
            _raw_config('tddft_nac', 'tddft', nstate=3)))

    def test_blocks_in_deck_order_map_onto_the_requested_pairs(self,
                                                               tmp_path):
        config = self._config()
        pairs = [pair for pair in config.target_nac_pair if len(pair) == 2]
        assert pairs == [[0, 1], [0, 2]]
        lines = self._COORDS + self._block(0.1) + self._block(0.2) + [
            'FINAL SINGLE POINT ENERGY       -76.100000000']
        _write_log(tmp_path, config.project, lines)

        results = parse_output(config, str(tmp_path),
                               compile_workflow('tddft_nac'))
        rows = np.asarray(results['nac'])
        assert rows.shape[0] == 2
        assert rows[0][0].tolist() == pytest.approx([0.1, 0.0, 0.0])
        assert rows[1][0].tolist() == pytest.approx([0.2, 0.0, 0.0])

    def test_a_block_count_that_does_not_match_is_refused(self, tmp_path):
        config = self._config()
        lines = self._COORDS + self._block(0.1) + [
            'FINAL SINGLE POINT ENERGY       -76.100000000']
        _write_log(tmp_path, config.project, lines)

        with pytest.raises(RuntimeError, match='1 non-adiabatic coupling block'):
            parse_output(config, str(tmp_path),
                         compile_workflow('tddft_nac'))


class TestFrequencyParsing:
    """Frequency extraction prefers {project}.hess over the log.

    The $vibrational_frequencies block of the .hess file (the layout
    the legacy wigner sampler read) is the primary source; the log's
    VIBRATIONAL FREQUENCIES section only serves as a fallback, and
    both missing fails fast.
    """

    HESS_FREQS = [0.0, 0.0, 0.0, 0.0, 0.0, 1595.0, 3657.0, 3756.0, 3832.0]
    LOG_FREQS = [1111.0, 2222.0, 3333.0]

    @staticmethod
    def _frequency_setup():
        config = normalize(build_config(
            _raw_config('dft_frequency', 'dft')))
        return config, compile_workflow('dft_frequency')

    def _write_hess(self, tmp_path, project, freqs):
        lines = ['$vibrational_frequencies', str(len(freqs))]
        lines += [f'{i:5d}   {f:14.6f}' for i, f in enumerate(freqs)]
        lines.append('$ir_spectrum')
        (tmp_path / f'{project}.hess').write_text('\n'.join(lines) + '\n')

    def _write_freq_log(self, tmp_path, project, freqs):
        lines = ['VIBRATIONAL FREQUENCIES', '----------------------']
        lines += [f'   {i}:   {f:12.2f} cm**-1' for i, f in enumerate(freqs)]
        lines.append('')
        _write_log(tmp_path, project, lines)

    def test_hess_frequencies_preferred_over_log(self, tmp_path):
        config, workflow = self._frequency_setup()
        self._write_hess(tmp_path, config.project, self.HESS_FREQS)
        self._write_freq_log(tmp_path, config.project, self.LOG_FREQS)

        results = parse_output(config, str(tmp_path), workflow)

        assert len(results['frequencies']) == 9
        assert results['frequencies'][5] == pytest.approx(1595.0)
        assert results['frequencies'][8] == pytest.approx(3832.0)

    def test_log_fallback_without_hess(self, tmp_path):
        config, workflow = self._frequency_setup()
        self._write_freq_log(tmp_path, config.project, self.LOG_FREQS)

        results = parse_output(config, str(tmp_path), workflow)

        assert list(results['frequencies']) == pytest.approx(self.LOG_FREQS)

    def test_missing_frequencies_fail_fast(self, tmp_path):
        config, workflow = self._frequency_setup()
        _write_log(tmp_path, config.project, ['no frequencies here'])

        with pytest.raises(RuntimeError):
            parse_output(config, str(tmp_path), workflow)


class TestTddftGradientPacking:
    """Gradient blocks are distributed along the irootlist/trootlist
    print order.

    ORCA prints one CARTESIAN GRADIENT block per requested root:
    irootlist[1:] + trootlist + [0] when the ground state (singlet
    root 0) is requested, otherwise irootlist + trootlist, with
    triplet entries carrying the state offset nsing_root + root. The
    block count must match the requested root count.
    """

    @staticmethod
    def _gradient_block(scale):
        # Real ORCA layout: header, separator, blank line, then one
        # 'index atom : dx dy dz' line per atom (parser skips 3 lines
        # after the header).
        lines = [
            'CARTESIAN GRADIENT',
            '---------------------------------------------',
            '',
        ]
        for i, atom in enumerate(ATOMS, start=1):
            lines.append(
                f'   {i}   {atom}   :   '
                f'{scale * 0.1:12.6f} {scale * 0.2:12.6f} '
                f'{scale * 0.3:12.6f}')
        return lines

    @staticmethod
    def _gradient_config(target_states):
        raw = _raw_config(
            'tddft_gradient', 'tddft', nstate=5,
            state_spin=[0, 0, 0, 2, 2], target_states=target_states)
        return normalize(build_config(raw))

    def test_excited_only_order_is_irootlist_plus_trootlist(self, tmp_path):
        # targets [1, 4]: irootlist [1], trootlist [2] -> print order
        # S1, T2 (framework states 1, 4); no ground-state block.
        config = self._gradient_config([1, 4])
        workflow = compile_workflow('tddft_gradient')

        log = ['FINAL SINGLE POINT ENERGY  -76.000000000']
        log += self._gradient_block(1.0)
        log += self._gradient_block(2.0)
        _write_log(tmp_path, config.project, log)

        results = parse_output(config, str(tmp_path), workflow)

        gradient = results['gradient']
        assert np.allclose(gradient[1], 1.0 * np.array([0.1, 0.2, 0.3]))
        assert np.allclose(gradient[4], 2.0 * np.array([0.1, 0.2, 0.3]))
        # Ground state not requested -> stays zero
        assert np.allclose(gradient[0], 0.0)
        assert np.allclose(gradient[2], 0.0)
        assert np.allclose(gradient[3], 0.0)

    def test_ground_state_block_printed_last(self, tmp_path):
        # targets [0, 1, 4]: irootlist [0, 1], trootlist [2] -> print
        # order S1, T2, then S0.
        config = self._gradient_config([0, 1, 4])
        workflow = compile_workflow('tddft_gradient')

        log = ['FINAL SINGLE POINT ENERGY  -76.000000000']
        log += self._gradient_block(1.0)
        log += self._gradient_block(2.0)
        log += self._gradient_block(3.0)
        _write_log(tmp_path, config.project, log)

        results = parse_output(config, str(tmp_path), workflow)

        gradient = results['gradient']
        assert np.allclose(gradient[1], 1.0 * np.array([0.1, 0.2, 0.3]))
        assert np.allclose(gradient[4], 2.0 * np.array([0.1, 0.2, 0.3]))
        assert np.allclose(gradient[0], 3.0 * np.array([0.1, 0.2, 0.3]))

    def test_block_count_mismatch_fails_fast(self, tmp_path):
        config = self._gradient_config([0, 1, 4])
        workflow = compile_workflow('tddft_gradient')

        log = ['FINAL SINGLE POINT ENERGY  -76.000000000']
        log += self._gradient_block(1.0)
        log += self._gradient_block(2.0)
        _write_log(tmp_path, config.project, log)

        with pytest.raises(RuntimeError):
            parse_output(config, str(tmp_path), workflow)


# Energies of the reference ORCA 6.1 TDDFT/TDA run captured in the
# remote qualification artifacts (C2H2, pbe0/sto-3g, D3(BJ)).
STATE_ENERGY_REF = [-77.4830733993217,
                    -77.20140037970403,
                    -77.11007624732251]

# Block emitted by ORCA-CIS/TD-DFT for each gradient root.
_ENERGY_LOG = [
    'TD-DFT/TDA EXCITED STATES (SINGLETS)',
    'STATE  1:  E=   0.281673 au      7.665 eV    61820.1 cm**-1 '
    '<S**2> =   0.000000 Mult 1',
    'STATE  2:  E=   0.372997 au     10.150 eV    81863.4 cm**-1 '
    '<S**2> =   0.000000 Mult 1',
    'CIS/TD-DFT TOTAL ENERGY',
    '    E(SCF)  =    -77.481199485 Eh',
    '    DE(CIS) =      0.281673020 Eh (Root  1)',
    '    ----------------------------- ---------',
    '    E(tot)  =    -77.199526466 Eh',
    # The excited root, not the ground state, owns this energy.
    'Dispersion correction           -0.001873914',
    'FINAL SINGLE POINT ENERGY       -77.201400379704',
]

_PROPERTY = """$CIS_Energies
   &GeometryIndex 1
   &e0 [&Type "Double"]      -7.7481199485230405e+01  "Ground state energy"
   &mode [&Type "String"] "TDA" "CIS mode: CIS, RPA, TDA, TD-DFT"
   &totalEnergy [&Type "ArrayOfDoubles", &Dim (2,1)] "Total energy of each state"
                                                         0

0                                     -7.7199526465612735e+01
1                                     -7.7108202333231219e+01
   &Root [&Type "ArrayOfIntegers", &Dim (2,1)] "Root index within the block"
                                                         0

0                                                        0
1                                                        1
$End
$VdW_Correction
   &GeometryIndex 1
   &vdW [&Type "Double"]      -1.8739140912968928e-03
$End
$Single_Point_Data
   &GeometryIndex 1
   &FinalEnergy [&Type "Double"]      -7.7201400379704026e+01  "Final energy"
   &Converged [&Type "Boolean"] true
$End
"""


def _write_property(calcdir, project, text):
    path = calcdir / f'{project}.property.txt'
    path.write_text(text)
    return path


class TestTddftEnergyReference:
    """The ground-state reference must not be the excited-root energy.

    ORCA writes FINAL SINGLE POINT ENERGY (and the .engrad $energy)
    as the total energy of the state whose nuclear gradient was
    printed last, so taking it as the S0 reference double-counts the
    excitation energy and makes an NVE trajectory appear to lose
    energy at exactly the rate the gap varies.
    """

    @staticmethod
    def _config(profile='tddft_energy'):
        raw = _raw_config(profile, 'tddft', nstate=3, target_states=[1])
        return normalize(build_config(raw))

    def test_property_file_drives_the_energies(self, tmp_path):
        config = self._config()
        workflow = compile_workflow('tddft_energy')
        _write_log(tmp_path, config.project, _ENERGY_LOG)
        _write_property(tmp_path, config.project, _PROPERTY)

        results = parse_output(config, str(tmp_path), workflow)

        assert results['energy'] == pytest.approx(
            STATE_ENERGY_REF, abs=1e-9)

    def test_log_only_fallback_reproduces_the_reference(self, tmp_path):
        config = self._config()
        workflow = compile_workflow('tddft_energy')
        _write_log(tmp_path, config.project, _ENERGY_LOG)

        results = parse_output(config, str(tmp_path), workflow)

        # The corrections are recovered by subtracting the labelled
        # excitation of the gradient root, so only the 6-decimal
        # STATE line limits the second excited state.
        assert results['energy'] == pytest.approx(
            STATE_ENERGY_REF, abs=1e-6)

    def test_unusable_property_file_falls_back(self, tmp_path):
        workflow = compile_workflow('tddft_energy')
        for text in ('', 'not a property file\n'):
            config = self._config()
            _write_log(tmp_path, config.project, _ENERGY_LOG)
            _write_property(tmp_path, config.project, text)

            results = parse_output(config, str(tmp_path), workflow)

            assert results['energy'] == pytest.approx(
                STATE_ENERGY_REF, abs=1e-6)

    def test_property_inconsistent_with_log_is_rejected(self, tmp_path):
        # A correction term the parser does not enumerate (solvation)
        # leaves no root able to reproduce the final energy; the log
        # path then recovers the reference by subtraction.
        config = self._config()
        workflow = compile_workflow('tddft_energy')
        _write_log(tmp_path, config.project, _ENERGY_LOG)
        _write_property(tmp_path, config.project,
                        _PROPERTY.replace('-1.8739140912968928e-03',
                                          '-0.5000000000000000e-03'))

        results = parse_output(config, str(tmp_path), workflow)

        assert results['energy'] == pytest.approx(
            STATE_ENERGY_REF, abs=1e-6)

    def test_root_labels_drive_the_state_mapping(self, tmp_path):
        # Both roots carry a labelled DE line, printed root 2 first:
        # subtracting the wrong one shifts the reference by the
        # 0.091 Eh root separation.
        config = self._config()
        workflow = compile_workflow('tddft_energy')
        log = [
            'CIS/TD-DFT TOTAL ENERGY',
            '    E(SCF)  =    -77.481199485 Eh',
            '    DE(CIS) =      0.372997152 Eh (Root  2)',
            '    E(SCF)  =    -77.481199485 Eh',
            '    DE(CIS) =      0.281673020 Eh (Root  1)',
            'Dispersion correction           -0.001873914',
            'FINAL SINGLE POINT ENERGY       -77.201400379704',
        ]
        _write_log(tmp_path, config.project, log)

        results = parse_output(config, str(tmp_path), workflow)

        assert results['energy'][0] == pytest.approx(
            STATE_ENERGY_REF[0], abs=1e-6)
        assert results['energy'][1] == pytest.approx(
            STATE_ENERGY_REF[1], abs=1e-6)

    def test_ground_state_profile_keeps_the_final_energy(self, tmp_path):
        # A pure DFT gradient genuinely owns the final energy: the
        # log carries no E(SCF) block, so the historic behaviour
        # stays untouched.
        config = normalize(build_config(_raw_config('dft_energy', 'dft')))
        workflow = compile_workflow('dft_energy')
        _write_log(tmp_path, config.project,
                   ['FINAL SINGLE POINT ENERGY       -76.100000000'])

        results = parse_output(config, str(tmp_path), workflow)

        assert results['energy'][0] == pytest.approx(-76.1)

class TestNoCouplingIsNoRows:
    """A coupling the deck never asked for travels as no rows."""

    @staticmethod
    def _log(tmp_path, config):
        lines = (_coord_block(COORDS) + _absorption_rows()
                 + TestTddftGradientPacking._gradient_block(1.0)
                 + ['FINAL SINGLE POINT ENERGY       -76.100000000'])
        _write_log(tmp_path, config.project, lines)

    def test_a_deck_without_the_switch_leaves_the_axis_empty(self, tmp_path):
        # The pre-allocated `np.zeros((nnac, natom, 3))` used to travel as an
        # answer: a crossing search read it as a coupling of exactly zero,
        # dropped the direction, and printed a branching plane one dimension
        # wide for a run in which the engine never computed a coupling. The
        # charge arm is the one whose deck leaves `NACME` out (the engine
        # refuses it beside point charges), so nothing was asked and nothing
        # is reported.
        config = normalize(build_config(
            _raw_config('tddft_nac', 'tddft', nstate=3,
                        point_charges=[[0.5, 5.0, 5.0, 5.0]])))
        workflow = compile_workflow('tddft_nac')
        self._log(tmp_path, config)
        # The warning belongs to the builder that leaves the switch out; the
        # parser only sees a deck that never asked.
        results = parse_output(config, str(tmp_path), workflow)
        assert np.asarray(results['nac']).shape[0] == 0

    def test_a_written_switch_with_no_block_is_refused(self, tmp_path):
        # The other arm: the deck asked (`NACME true`) and the log carries no
        # vector, so either the reader's header is wrong or the run died after
        # TD-DFT. Naming it beats answering with nothing for a switch that was
        # written - that is how the SOC reader's wrong header went unnoticed.
        config = normalize(build_config(
            _raw_config('tddft_nac', 'tddft', nstate=3)))
        workflow = compile_workflow('tddft_nac')
        self._log(tmp_path, config)
        with pytest.raises(RuntimeError, match='NACME true'):
            parse_output(config, str(tmp_path), workflow)
