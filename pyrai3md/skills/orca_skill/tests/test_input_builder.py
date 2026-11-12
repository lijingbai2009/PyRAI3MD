"""
Unit tests for orca_skill input generation (input_builder).

Verifies the accuracy-acceleration keywords on the ORCA input:
density fitting (RIJCOSX + auxiliary basis after the basis set),
Grimme dispersion correction, and the DEFGRID integration grid
line, including their off switches.
Also verifies the embedding point charges: the input carries a quoted
%pointcharges file reference and the rows are written to a side file.
These tests do NOT require an ORCA installation.

Run: pytest orca_skill/tests/test_input_builder.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import os

import pytest

from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..schema.validate import validate
from ..service.builder.input_builder import (
    build_input,
    build_pointcharges_file,
    pointcharges_filename,
    write_input_file,
)
from ..service.workflow import compile_workflow

# Water molecule keeps the geometry block compact
ATOMS = ['O', 'H', 'H']
COORDS = [
    [0.000000000, 0.000000000, 0.117300000],
    [0.000000000, 0.757200000, -0.469200000],
    [0.000000000, -0.757200000, -0.469200000],
]


def _make_config(**kwargs):
    raw = {
        'atoms': ATOMS,
        'coordinates': COORDS,
        'charges': 0,
        'multiplicity': 1,
        'nstate': 1,
        'state_spin': [0],
        'profile': 'dft_energy',
        'method': 'dft',
        'functional': 'pbe0',
        'basis': 'sto-3g',
    }
    raw.update(kwargs)
    return validate(normalize(build_config(raw)))


def _make_tddft_config(profile, **kwargs):
    raw = {
        'atoms': ATOMS,
        'coordinates': COORDS,
        'charges': 0,
        'multiplicity': 1,
        'nstate': 3,
        'state_spin': [0, 0, 0],
        'profile': profile,
        'method': 'tddft',
        'functional': 'pbe0',
        'basis': 'sto-3g',
    }
    raw.update(kwargs)
    return validate(normalize(build_config(raw)))


def _first_lines(config, count=2):
    workflow = compile_workflow('dft_energy')
    return build_input(config, workflow).split('\n')[:count]


class TestDensityFittingKeywords:
    """RIJCOSX and the auxiliary basis follow the basis set."""

    def test_defaults_write_rijcosx_and_aux_basis(self):
        config = _make_config()
        main_line, _ = _first_lines(config)
        tokens = main_line.split()
        basis_index = tokens.index('sto-3g')
        assert tokens[basis_index + 1] == 'RIJCOSX'
        assert tokens[basis_index + 2] == 'def2/J'

    def test_disabled_omits_both_keywords(self):
        config = _make_config(density_fitting=False)
        main_line, _ = _first_lines(config)
        assert 'RIJCOSX' not in main_line
        assert 'def2/J' not in main_line

    def test_custom_aux_basis_is_forwarded(self):
        config = _make_config(aux_basis='cc-pVDZ/C')
        main_line, _ = _first_lines(config)
        assert 'RIJCOSX cc-pVDZ/C' in main_line


class TestDispersionCorrectionKeyword:
    """dispersion_correction maps onto the ORCA dispersion keyword."""

    def test_default_d3_writes_d3bj(self):
        config = _make_config()
        main_line, _ = _first_lines(config)
        assert 'D3BJ' in main_line

    @pytest.mark.parametrize('value, keyword', [
        ('d3zero', 'D3ZERO'),
        ('d4', 'D4'),
    ])
    def test_alternative_corrections(self, value, keyword):
        config = _make_config(dispersion_correction=value)
        main_line, _ = _first_lines(config)
        assert keyword in main_line

    def test_none_disables_keyword(self):
        config = _make_config(dispersion_correction='none')
        main_line, _ = _first_lines(config)
        assert 'D3BJ' not in main_line
        assert 'D3ZERO' not in main_line
        assert 'D4' not in main_line

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError,
                           match='dispersion correction'):
            _make_config(dispersion_correction='d99')


class TestGridLevelKeyword:
    """grid_level 1-3 emits a dedicated '! DEFGRID<n>' line."""

    def test_default_writes_defgrid2_line(self):
        config = _make_config()
        main_line, grid_line = _first_lines(config)
        assert 'DEFGRID' not in main_line
        assert grid_line == '! DEFGRID2'

    @pytest.mark.parametrize('level', [1, 3])
    def test_other_levels(self, level):
        config = _make_config(grid_level=level)
        _, grid_line = _first_lines(config)
        assert grid_line == f'! DEFGRID{level}'

    def test_zero_keeps_orca_default_grid(self):
        config = _make_config(grid_level=0)
        lines = _first_lines(config)
        assert all('DEFGRID' not in line for line in lines)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match='grid_level'):
            _make_config(grid_level=5)


class TestNacmePlacement:
    """NACME is a %tddft-block switch, not a keyword-line keyword.

    ORCA rejects NACME on the main '!' line (input parsing aborts),
    so the nac step must place it inside the %tddft block.
    """

    def test_nacme_inside_tddft_block(self):
        config = _make_tddft_config('tddft_nac')
        workflow = compile_workflow('tddft_nac')
        content = build_input(config, workflow)
        main_line = content.split('\n')[0]
        assert 'NACME' not in main_line
        assert '  NACME true' in content
        # The switch sits between %tddft and its closing 'end'
        block = content.split('%tddft')[1].split('end')[0]
        assert 'NACME true' in block


class TestNacmeIsRefusedBesidePointCharges:
    """ORCA cannot do NACME with embedded charges, so it is not asked.

    The engine aborts at input check - `Error (ORCA_CIS): The NACMEs are
    currently not compatible with point charges!` (`main_input_check.cpp:10261`),
    return code 25 - so the switch is left out when the run carries background
    charges, and the step says what it has: no coupling measured. The two arms
    differ only by the charges, so the surviving arm proves the keyword is
    otherwise written.
    """

    def test_charges_present_drop_the_switch_and_say_so(self):
        config = _make_tddft_config('tddft_nac', point_charges=POINT_CHARGES)
        workflow = compile_workflow('tddft_nac')
        with pytest.warns(UserWarning, match='point charges'):
            content = build_input(config, workflow)
        assert 'NACME' not in content

    def test_charges_arriving_as_an_array_are_still_charges(self):
        # The framework hands region rows as arrays, and `x or []` on one is
        # an ambiguity error, not a test - so the check asks for None and a
        # length rather than truthiness.
        import numpy as np

        config = _make_tddft_config('tddft_nac')
        config.system.point_charges = np.asarray(POINT_CHARGES)
        workflow = compile_workflow('tddft_nac')
        with pytest.warns(UserWarning, match='point charges'):
            content = build_input(config, workflow)
        assert 'NACME' not in content

    def test_without_charges_the_switch_is_written(self):
        config = _make_tddft_config('tddft_nac')
        content = build_input(config, compile_workflow('tddft_nac'))
        assert '  NACME true' in content

    def test_the_coupling_deck_does_not_also_ask_for_a_gradient(self):
        # Measured in the test environment (round `tmp_20261101_221306`, ORCA
        # 6.1.1): a deck with `EnGrad` and `NACME true` finishes the TD-DFT
        # part and then aborts on `CANNOT OPEN FILE <project>.nacme.tmp`,
        # return code 64. One charge-free deck cannot carry both, and the
        # coupling is what the run asked for, so the gradient step is not
        # requested here.
        config = _make_tddft_config('tddft_nac')
        content = build_input(config, compile_workflow('tddft_nac'))
        assert 'EnGrad' not in content
        # The root list stays: it is also the NACME request, and without it
        # the run prints the coupling of root 1 only (measured on round
        # `tmp_20261101_224544`).
        assert '  irootlist' in content

    def test_the_charged_deck_asks_for_the_gradient_instead(self):
        # The other half of the same rule: with charges the NACME switch is
        # gone, so the gradient can be asked for - that is the crossing
        # search's deck, and its gap gradient is what makes the plane
        # two-dimensional as far as the coupling-less reading goes.
        config = _make_tddft_config('tddft_nac', point_charges=POINT_CHARGES)
        with pytest.warns(UserWarning, match='point charges'):
            content = build_input(config, compile_workflow('tddft_nac'))
        assert 'EnGrad' in content
        assert '  irootlist' in content

    def test_the_soc_deck_keeps_the_gradient(self):
        # `tddft_soc` has no NACME switch, so nothing competes with EnGrad;
        # the test run of round `tmp_20261101_221306` printed exactly
        # one `CARTESIAN GRADIENT` block with this deck.
        config = _make_tddft_config('tddft_soc', nstate=6,
                                    state_spin=[0, 0, 0, 2, 2, 2])
        content = build_input(config, compile_workflow('tddft_soc'))
        assert 'EnGrad' in content
        assert '  DoSOC true' in content


class TestFrequencyKeywords:
    """AnFreq/NumFreq selection per method.

    ORCA has no analytical TDDFT second derivatives, so excited-state
    frequency runs always use NumFreq regardless of num_freq; the
    num_freq switch only controls ground-state DFT. RIJCOSX stays on
    for TDDFT since it is compatible with excited-state gradients and
    numerical Hessians.
    """

    def test_tddft_frequency_forces_numfreq(self):
        # num_freq=False must not yield AnFreq for TDDFT
        config = _make_tddft_config('tddft_frequency', num_freq=False)
        workflow = compile_workflow('tddft_frequency')
        main_line = build_input(config, workflow).split('\n')[0]
        assert 'NumFreq' in main_line
        assert 'AnFreq' not in main_line

    def test_tddft_frequency_keeps_density_fitting(self):
        config = _make_tddft_config('tddft_frequency')
        workflow = compile_workflow('tddft_frequency')
        main_line = build_input(config, workflow).split('\n')[0]
        assert 'RIJCOSX' in main_line
        assert 'def2/J' in main_line

    def test_tddft_gradient_keeps_density_fitting(self):
        config = _make_tddft_config('tddft_gradient', target_states=[1])
        workflow = compile_workflow('tddft_gradient')
        main_line = build_input(config, workflow).split('\n')[0]
        assert 'RIJCOSX' in main_line

    def test_dft_frequency_keeps_anfreq_by_default(self):
        config = _make_config(profile='dft_frequency')
        workflow = compile_workflow('dft_frequency')
        main_line = build_input(config, workflow).split('\n')[0]
        assert 'AnFreq' in main_line
        assert 'NumFreq' not in main_line
        # Ground-state Hessians work with RIJCOSX
        assert 'RIJCOSX' in main_line

    def test_dft_frequency_numfreq_switch(self):
        config = _make_config(profile='dft_frequency', num_freq=True)
        workflow = compile_workflow('dft_frequency')
        main_line = build_input(config, workflow).split('\n')[0]
        assert 'NumFreq' in main_line


class TestEnGradExclusion:
    """EnGrad never shares the keyword line with Opt/OptTS.

    ORCA treats an EnGrad input as an energy+gradient single point
    and overrides the optimization task, so optimizations (ground
    and excited state) must run without EnGrad; gradient single
    points keep it.
    """

    def test_dft_optimization_drops_engrad(self):
        config = _make_config(profile='dft_opt_min')
        workflow = compile_workflow('dft_opt_min')
        main_line = build_input(config, workflow).split('\n')[0]
        assert 'Opt' in main_line
        assert 'EnGrad' not in main_line

    def test_transition_state_search_drops_engrad(self):
        config = _make_config(profile='dft_opt_ts')
        workflow = compile_workflow('dft_opt_ts')
        main_line = build_input(config, workflow).split('\n')[0]
        assert 'OptTS' in main_line
        assert 'EnGrad' not in main_line

    def test_excited_state_optimization_drops_engrad(self):
        # current_state > 0 exercised the old EnGrad branch
        config = _make_tddft_config(
            'tddft_opt_min', current_state=1, target_states=[1])
        workflow = compile_workflow('tddft_opt_min')
        main_line = build_input(config, workflow).split('\n')[0]
        assert 'Opt' in main_line
        assert 'EnGrad' not in main_line

    def test_excited_state_ts_search_drops_engrad(self):
        config = _make_tddft_config(
            'tddft_opt_ts', current_state=1, target_states=[1])
        workflow = compile_workflow('tddft_opt_ts')
        main_line = build_input(config, workflow).split('\n')[0]
        assert 'OptTS' in main_line
        assert 'EnGrad' not in main_line

    def test_gradient_single_points_keep_engrad(self):
        config = _make_config(profile='dft_gradient')
        workflow = compile_workflow('dft_gradient')
        main_line = build_input(config, workflow).split('\n')[0]
        assert 'EnGrad' in main_line

        config = _make_tddft_config('tddft_gradient', target_states=[1])
        workflow = compile_workflow('tddft_gradient')
        main_line = build_input(config, workflow).split('\n')[0]
        assert 'EnGrad' in main_line


class TestAlwaysOnPrintKeywords:
    """PrintBasis and the MO dump are part of every input.

    The keyword line always carries PrintBasis, the %scf block always
    dumps the MO coefficients, and the (illegal) %output
    Print[P_Hessian] block is never written; frequency runs produce
    {project}.hess automatically.
    """

    def test_printbasis_on_keyword_line(self):
        config = _make_config()
        workflow = compile_workflow('dft_energy')
        main_line = build_input(config, workflow).split('\n')[0]
        assert 'PrintBasis' in main_line

    def test_scf_block_prints_mos(self):
        config = _make_config()
        workflow = compile_workflow('dft_energy')
        input_text = build_input(config, workflow)
        assert 'Print[P_MOs] 1' in input_text

    def test_frequency_input_has_no_output_block(self):
        config = _make_config(profile='dft_frequency')
        workflow = compile_workflow('dft_frequency')
        input_text = build_input(config, workflow)
        assert '%output' not in input_text
        assert 'P_Hessian' not in input_text

    def test_orca_mem_maps_to_maxcore(self):
        # orca_mem (MB per core) rides into the %maxcore line
        config = _make_config(orca_mem=4000)
        workflow = compile_workflow('dft_energy')
        input_text = build_input(config, workflow)
        assert '%maxcore 4000' in input_text


class TestSocKeywords:
    """SOC reference state contract.

    ORCA DoSOC is only implemented for a closed-shell singlet (RHF)
    reference (#129 regression: main_input_check.cpp aborted a UKS
    reference). Triplet target states enter through 'Triplets true'
    in the %tddft block, so the input keeps a singlet reference even
    when the target states are triplets.
    """

    def test_soc_drops_uks_reference(self):
        config = _make_tddft_config(
            'tddft_soc', nstate=6,
            state_spin=[0, 0, 0, 2, 2, 2])
        workflow = compile_workflow('tddft_soc')
        main_line = build_input(config, workflow).split('\n')[0]
        assert 'UKS' not in main_line

    def test_soc_enables_triplets_and_dosoc(self):
        config = _make_tddft_config(
            'tddft_soc', nstate=6,
            state_spin=[0, 0, 0, 2, 2, 2])
        workflow = compile_workflow('tddft_soc')
        input_text = build_input(config, workflow)
        assert 'Triplets true' in input_text
        assert 'DoSOC true' in input_text

    def test_soc_keeps_singlet_reference(self):
        config = _make_tddft_config(
            'tddft_soc', nstate=6,
            state_spin=[0, 0, 0, 2, 2, 2])
        workflow = compile_workflow('tddft_soc')
        input_text = build_input(config, workflow)
        assert '* xyz 0 1' in input_text

    def test_soc_rejects_triplet_ground_state(self):
        # DoSOC needs an RHF reference; a triplet ground state must
        # fail fast at normalization instead of dying remotely.
        import pytest
        with pytest.raises(ValueError):
            _make_tddft_config(
                'tddft_soc', nstate=3,
                multiplicity=3, state_spin=[2, 2, 2])


class TestMixedSpinGradientRoots:
    """Mixed singlet/triplet gradient runs select the right roots.

    The %tddft block requests nroots = max(excited singlets,
    triplets) and enables 'Triplets true' whenever triplet states
    are present. Gradient roots are split between irootlist (singlet
    roots, 0 = ground state) and trootlist (triplet roots, 1-based),
    both ascending.
    """

    @staticmethod
    def _tddft_lines(input_text):
        block = input_text.split('%tddft')[1].split('end')[0]
        return [line.strip() for line in block.splitlines()
                if line.strip()]

    def test_balanced_mixed_spin_gradient(self):
        # 3 singlets (S0 + 2 excited) and 2 triplets -> nroots 2;
        # target 1 (S1) is singlet root 1, target 4 (T2) is triplet
        # root 4 - 2 = 2.
        config = _make_tddft_config(
            'tddft_gradient', nstate=5,
            state_spin=[0, 0, 0, 2, 2], target_states=[1, 4])
        assert config.tddft_nroots == 2
        workflow = compile_workflow('tddft_gradient')
        lines = self._tddft_lines(build_input(config, workflow))
        assert 'nroots 2' in lines
        assert 'irootlist 1' in lines
        assert 'trootlist 2' in lines
        assert 'Triplets true' in lines
        assert 'DoSOC true' not in lines

    def test_triplet_heavy_trootlist_offset(self):
        # 2 singlets + 3 triplets -> nroots 3, nsing_root 1; target
        # 4 (T3) is triplet root 4 - 1 = 3.
        config = _make_tddft_config(
            'tddft_gradient', nstate=5,
            state_spin=[0, 0, 2, 2, 2], target_states=[1, 4])
        assert config.tddft_nroots == 3
        workflow = compile_workflow('tddft_gradient')
        lines = self._tddft_lines(build_input(config, workflow))
        assert 'nroots 3' in lines
        assert 'irootlist 1' in lines
        assert 'trootlist 3' in lines

    def test_ground_state_rides_in_irootlist(self):
        # The ground state is singlet root 0 and only ever appears
        # in irootlist, never in trootlist.
        config = _make_tddft_config(
            'tddft_gradient', nstate=5,
            state_spin=[0, 0, 0, 2, 2], target_states=[0, 1, 4])
        workflow = compile_workflow('tddft_gradient')
        lines = self._tddft_lines(build_input(config, workflow))
        assert 'irootlist 0, 1' in lines
        assert 'trootlist 2' in lines

    def test_pure_singlet_gradient_keeps_irootlist(self):
        config = _make_tddft_config(
            'tddft_gradient', nstate=3,
            state_spin=[0, 0, 0], target_states=[1, 2])
        workflow = compile_workflow('tddft_gradient')
        lines = self._tddft_lines(build_input(config, workflow))
        assert 'irootlist 1, 2' in lines
        assert not any(line.startswith('trootlist') for line in lines)
        assert 'Triplets true' not in lines


# ONIOM electrostatic embedding hands the QM region the outer-layer
# charges. ORCA refuses them inline ("Quotes expected after scanning
# %pointcharges") and reads them from a side file instead, so the input
# carries a quoted reference and the rows live in {project}.pc: charge in
# atomic units first, then the position in Angstrom, the frame the
# geometry block already uses.
POINT_CHARGES = [
    [-0.834, -1.3130, 0.0000, -0.0310],
    [0.417, -1.8700, 0.7570, 0.1651],
]


class TestPointChargesAreFileReferenced:

    def _config(self, point_charges):
        return _make_config(point_charges=point_charges)

    def test_the_input_carries_a_quoted_reference_only(self):
        config = self._config(POINT_CHARGES)
        text = build_input(config, compile_workflow('dft_energy'))
        assert f'%pointcharges "{config.project}.pc"' in text
        # The rows are gone from the input: the reference line is the
        # whole block, and a charge value appearing anywhere else would
        # mean the inline form is still being written.
        assert '-0.834' not in text

    def test_the_file_opens_with_the_count_and_keeps_row_order(self):
        config = self._config(POINT_CHARGES)
        rows = build_pointcharges_file(config).strip().split('\n')
        assert rows[0].strip() == '2'
        assert len(rows) == 3
        for row, expected in zip(rows[1:], POINT_CHARGES):
            assert [float(v) for v in row.split()] == pytest.approx(
                expected, abs=1e-9)

    def test_write_input_file_puts_the_side_file_beside_the_input(
            self, tmp_path):
        config = self._config(POINT_CHARGES)
        input_path = write_input_file(
            config, compile_workflow('dft_energy'), str(tmp_path))
        charge_path = tmp_path / pointcharges_filename(config)
        assert os.path.dirname(input_path) == str(tmp_path)
        assert charge_path.exists()
        assert charge_path.read_text().strip().split('\n')[0].strip() == '2'

    def test_no_charges_leaves_no_reference_and_no_file(self, tmp_path):
        config = self._config([])
        assert build_pointcharges_file(config) == ''
        assert '%pointcharges' not in build_input(
            config, compile_workflow('dft_energy'))
        write_input_file(config, compile_workflow('dft_energy'),
                         str(tmp_path))
        assert not (tmp_path / pointcharges_filename(config)).exists()


class TestTddftNeedsAnExcitedState:
    """`method: tddft` on a one-state manifold is refused here.

    The builder derives `%tddft nroots {tddft_nroots}` from the state
    manifold, so a ground-state-only run asks ORCA for zero roots and the
    engine aborts at input scan time with "IROOT must be smaller than or
    equal to NROOTS" (main_input_check.cpp) -- which is what a remote
    multiscale round measured on the ORCA point-charge MEP case. Refusing
    it in validation names the contradiction; degrading to DFT would
    answer a question the caller did not ask.
    """

    def test_one_state_tddft_is_refused_by_name(self):
        with pytest.raises(ValueError, match='no excited state'):
            _make_config(profile='tddft_energy', method='tddft',
                         nstate=1, state_spin=[0])

    def test_an_excited_root_still_passes(self):
        # Negative control on the same knob: the refusal is not an
        # identity over every tddft config, it fires only when the
        # manifold carries no excited state.
        config = _make_tddft_config('tddft_energy')
        assert config.tddft_nroots == 2
        assert 'nroots 2' in build_input(config, compile_workflow(
            'tddft_energy'))

    def test_ground_state_dft_on_the_same_manifold_is_accepted(self):
        config = _make_config(profile='dft_energy', method='dft',
                              nstate=1, state_spin=[0])
        assert '%tddft' not in build_input(
            config, compile_workflow('dft_energy'))
