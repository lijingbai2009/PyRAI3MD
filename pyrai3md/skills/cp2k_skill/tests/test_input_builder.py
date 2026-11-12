"""
Unit tests for cp2k_skill input generation (input_builder).

Verifies the CP2K input parameter consistency: cutoff passthrough,
functional mapping (PBE / PBE0 with HF exchange), dispersion
correction with the correct REFERENCE_FUNCTIONAL, element-level
KIND basis/potential blocks, ADMM auxiliary basis, and the
GEO_OPT / MD motion sections.
These tests do NOT require a CP2K installation.

Run: pytest cp2k_skill/tests/test_input_builder.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import pytest

from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..schema.validate import validate
from ..service.builder.input_builder import build_input
from ..service.workflow import compile_workflow

# Water molecule keeps the input compact
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
        'functional': 'pbe',
        'cutoff': 400.0,
        'ref_cutoff': 50.0,
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
        'functional': 'pbe',
    }
    raw.update(kwargs)
    return validate(normalize(build_config(raw)))


def _build(config, profile):
    return build_input(config, compile_workflow(profile))


class TestCutoffPassthrough:
    """cutoff / ref_cutoff ride into &MGRID verbatim."""

    def test_default_cutoffs(self):
        config = _make_config()
        text = _build(config, 'dft_energy')
        assert 'CUTOFF 400' in text
        assert 'REL_CUTOFF 50' in text

    def test_custom_cutoffs(self):
        config = _make_config(cutoff=600.0, ref_cutoff=80.0)
        text = _build(config, 'dft_energy')
        assert 'CUTOFF 600' in text
        assert 'REL_CUTOFF 80' in text

    def test_invalid_cutoff_raises(self):
        with pytest.raises(ValueError):
            _make_config(cutoff=-1.0)


class TestFunctionalMapping:
    """functional maps onto the &XC section."""

    def test_pbe_writes_xc_functional(self):
        config = _make_config(functional='pbe')
        text = _build(config, 'dft_energy')
        assert '&XC_FUNCTIONAL PBE' in text
        assert '&HF' not in text

    def test_pbe0_writes_scaled_pbe_with_hf(self):
        config = _make_config(functional='pbe0')
        text = _build(config, 'dft_energy')
        assert 'SCALE_X 0.75' in text
        assert '&HF' in text
        assert 'FRACTION 0.25' in text

    def test_unsupported_functional_raises(self):
        with pytest.raises(ValueError, match='Unsupported CP2K functional'):
            _make_config(functional='b3lyp')


class TestDispersionCorrection:
    """dispersion maps onto &VDW_POTENTIAL; the reference functional
    always follows the active functional."""

    def test_d3(self):
        config = _make_config(dispersion='d3')
        text = _build(config, 'dft_energy')
        assert '&VDW_POTENTIAL' in text
        assert 'TYPE DFTD3' in text
        assert 'REFERENCE_FUNCTIONAL PBE' in text

    def test_d3bj(self):
        config = _make_config(dispersion='d3bj')
        text = _build(config, 'dft_energy')
        assert 'TYPE DFTD3(BJ)' in text

    def test_pbe0_d3_keeps_pbe0_reference(self):
        config = _make_config(functional='pbe0', dispersion='d3')
        text = _build(config, 'dft_energy')
        assert 'REFERENCE_FUNCTIONAL PBE0' in text

    def test_empty_disables_section(self):
        config = _make_config(dispersion='')
        text = _build(config, 'dft_energy')
        assert '&VDW_POTENTIAL' not in text

    def test_unknown_dispersion_raises(self):
        with pytest.raises(ValueError):
            _make_config(dispersion='d4')


class TestKindBlocks:
    """Element-level basis/potential ride into the &KIND blocks."""

    def test_default_basis_carries_valence_suffix(self):
        config = _make_config()
        text = _build(config, 'dft_energy')
        assert 'BASIS_SET DZVP-MOLOPT-SR-GTH-q6' in text
        assert 'BASIS_SET DZVP-MOLOPT-SR-GTH-q1' in text
        assert 'POTENTIAL GTH-PBE' in text

    def test_user_basis_overrides_default(self):
        config = _make_config(basis={'H': 'SZV-MOLOPT-SR-GTH-q1'})
        text = _build(config, 'dft_energy')
        assert 'BASIS_SET SZV-MOLOPT-SR-GTH-q1' in text
        # Oxygen keeps the normalized default
        assert 'BASIS_SET DZVP-MOLOPT-SR-GTH-q6' in text

    def test_basis_file_list(self):
        config = _make_config(
            basis_file=['BASIS_MOLOPT', 'BASIS_MOLOPT_LnPP2'])
        text = _build(config, 'dft_energy')
        assert 'BASIS_SET_FILE_NAME BASIS_MOLOPT' in text
        assert 'BASIS_SET_FILE_NAME BASIS_MOLOPT_LnPP2' in text


class TestAdmm:
    """An auxiliary_basis dict enables ADMM."""

    def test_admm_section_and_aux_fit(self):
        config = _make_config(auxiliary_basis={'H': 'admm-dzp-q1'})
        text = _build(config, 'dft_energy')
        assert '&AUXILIARY_DENSITY_MATRIX_METHOD' in text
        assert 'BASIS_PROJECTION' in text
        assert 'BASIS_SET AUX_FIT admm-dzp-q1' in text

    def test_no_admm_without_auxiliary_basis(self):
        config = _make_config()
        text = _build(config, 'dft_energy')
        assert '&AUXILIARY_DENSITY_MATRIX_METHOD' not in text
        assert 'AUX_FIT' not in text

    def test_tddft_admm_disables_purification(self):
        config = _make_tddft_config(
            'tddft_energy', auxiliary_basis={'H': 'admm-dzp-q1'})
        text = _build(config, 'tddft_energy')
        assert 'ADMM_PURIFICATION_METHOD NONE' in text


class TestRunTypeAndPrint:
    """RUN_TYPE and the &PRINT section follow the profile steps."""

    def test_energy_profile(self):
        config = _make_config()
        text = _build(config, 'dft_energy')
        assert 'RUN_TYPE ENERGY' in text
        assert '&FORCES' not in text
        # FFT grid factorization guard against arbitrary cell sizes
        assert 'EXTENDED_FFT_LENGTHS TRUE' in text

    def test_gradient_profile_prints_forces(self):
        config = _make_config(profile='dft_gradient')
        text = _build(config, 'dft_gradient')
        assert 'RUN_TYPE ENERGY_FORCE' in text
        assert '&FORCES ON' in text
        assert 'force_unit hartree/bohr' in text

    def test_geometry_optimization(self):
        config = _make_config(profile='dft_opt_min')
        text = _build(config, 'dft_opt_min')
        assert 'RUN_TYPE GEO_OPT' in text
        assert '&MOTION' in text
        assert '&GEO_OPT' in text
        assert '&TRAJECTORY' in text

    def test_standalone_md(self):
        config = _make_config(profile='dft_md')
        text = _build(config, 'dft_md')
        assert 'RUN_TYPE MD' in text
        assert '&MD' in text
        assert 'ENSEMBLE' in text
        assert 'CSVR' in text
        # MD runs extrapolate the density between steps
        assert 'EXTRAPOLATION ASPC' in text


class TestCellAndPoisson:
    """Periodic systems use the lattice vectors; molecules get a
    vacuum box with the WAVELET solver. pbc is the single periodicity
    authority (dev/methods/PBC_DEVELOPMENT.md section 3.2): a
    non-zero lattice alone never turns a run periodic."""

    def test_molecular_vacuum_box(self):
        config = _make_config()
        text = _build(config, 'dft_energy')
        assert 'PERIODIC NONE' in text
        assert 'PSOLVER WAVELET' in text

    def test_periodic_cell(self):
        config = _make_config(
            pbc=[1, 1, 1],
            lattice_vector=[
                [10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 15.0]])
        text = _build(config, 'dft_energy')
        assert 'A 10.00000000 0.00000000 0.00000000' in text
        assert 'B 0.00000000 12.00000000 0.00000000' in text
        assert 'PERIODIC XYZ' in text
        assert 'PSOLVER PERIODIC' in text

    def test_nonzero_lattice_without_pbc_stays_molecular(self):
        # PBC_DEVELOPMENT.md 3.2 reproduction: a non-zero lattice with
        # pbc off used to be misread as periodic (PERIODIC XYZ). pbc is
        # now the authority, so this stays a molecular vacuum box.
        config = _make_config(lattice_vector=[
            [10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 15.0]])
        text = _build(config, 'dft_energy')
        assert 'PERIODIC NONE' in text
        assert 'PSOLVER WAVELET' in text
        assert 'PERIODIC XYZ' not in text

    def test_partial_periodicity_rejected(self):
        # Only full 3D periodicity is supported in this stage; mixed
        # pbc is refused by the skill validation.
        with pytest.raises(ValueError, match='partial periodicity'):
            _make_config(
                pbc=[1, 1, 0],
                lattice_vector=[
                    [10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 15.0]])

    def test_pbc_without_lattice_rejected(self):
        with pytest.raises(ValueError):
            _make_config(pbc=[1, 1, 1])


class TestDipoleCorrection:
    """The slab correction maps onto &DFT SURF_DIP and only accepts the
    cell shape CP2K implements (normal along one Cartesian axis)."""

    BOX = [[10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 30.0]]

    def test_off_by_default(self):
        config = _make_config(
            pbc=[1, 1, 1], lattice_vector=self.BOX)
        text = _build(config, 'dft_energy')
        assert 'SURF_DIP' not in text

    def test_keywords_written_inside_dft(self):
        config = _make_config(
            pbc=[1, 1, 1], lattice_vector=self.BOX,
            dipole_correction=True)
        text = _build(config, 'dft_energy')
        assert '  SURF_DIP .TRUE.' in text
        assert '  SURF_DIP_DIR Z' in text
        assert text.index('SURF_DIP .TRUE.') > text.index('&DFT')
        assert text.index('SURF_DIP .TRUE.') < text.index('&END DFT')

    def test_non_periodic_rejected(self):
        with pytest.raises(ValueError, match='requires a periodic slab cell'):
            _make_config(dipole_correction=True)

    def test_tilted_cell_rejected(self):
        lattice = [list(row) for row in self.BOX]
        lattice[2][0] = 0.5
        with pytest.raises(ValueError, match='normal must be the c vector'):
            _make_config(
                pbc=[1, 1, 1], lattice_vector=lattice,
                dipole_correction=True)

    def test_out_of_plane_a_rejected(self):
        lattice = [list(row) for row in self.BOX]
        lattice[0][2] = 0.5
        with pytest.raises(ValueError, match='normal must be the c vector'):
            _make_config(
                pbc=[1, 1, 1], lattice_vector=lattice,
                dipole_correction=True)

    def test_full_strain_scope_rejected(self):
        with pytest.raises(ValueError, match='cannot be combined'):
            _make_config(
                pbc=[1, 1, 1], lattice_vector=self.BOX,
                dipole_correction=True, stress=True,
                stress_method='num', stress_scope='full')

    def test_iso_strain_scope_allowed(self):
        config = _make_config(
            pbc=[1, 1, 1], lattice_vector=self.BOX,
            dipole_correction=True, stress=True,
            stress_method='num', stress_scope='iso')
        text = _build(config, 'dft_gradient')
        assert 'SURF_DIP .TRUE.' in text


class TestTddfpt:
    """TDDFPT section follows the excited-state contract."""

    def test_nstates_derived_from_nstate(self):
        config = _make_tddft_config('tddft_energy')
        text = _build(config, 'tddft_energy')
        assert '&TDDFPT' in text
        # nstate=3 -> 2 excited roots
        assert 'NSTATES 2' in text

    def test_gradient_lists_target_states(self):
        config = _make_tddft_config(
            'tddft_gradient', target_states=[0, 1, 2])
        text = _build(config, 'tddft_gradient')
        assert '&TDDFPT' in text
        forces_block = text.split('&TDDFPT')[1].split('&END TDDFPT')[0]
        assert '&FORCES' in forces_block
        assert 'LIST' in forces_block

    def test_triplet_roots_switch_the_triplet_block_on(self):
        # RKS_TRIPLETS follows the spin content of the request: CP2K solves the
        # triplet block only when told to, and the count of triplet roots the
        # framework holds comes from `state_spin`. Measured in the test round
        # tmp_20261014_185533, `T` replaces the singlet solve rather than adding to
        # it (that job printed only `R-TDDFPT states of multiplicity 3`), so this
        # switch alone does not give two surfaces in one run. The energy profile
        # carries it: a gradient step across two manifolds is refused at the LIST
        # (see `test_a_two_manifold_force_request_is_refused_by_name`).
        config = _make_tddft_config('tddft_energy', nstate=3,
                                    state_spin=[0, 0, 2],
                                    target_states=[0, 1, 2])
        text = _build(config, 'tddft_energy')
        assert 'RKS_TRIPLETS T' in text

    def test_an_all_singlet_request_keeps_the_triplet_block_off(self):
        # The identity arm: every shipped case asks for singlets, and their input
        # must not change by one byte. Without this arm the switch above could be
        # a constant `T` that nobody notices.
        config = _make_tddft_config('tddft_gradient', target_states=[0, 1, 2])
        text = _build(config, 'tddft_gradient')
        assert 'RKS_TRIPLETS F' in text

    def test_nstates_counts_the_larger_manifold(self):
        # NSTATES is per manifold: CP2K solves that many roots in the singlet
        # block and that many in the triplet one (measured on
        # cp2k_demo/sp/bulk.log, NSTATES 10 answering with two tables of 10).
        config = _make_tddft_config('tddft_energy', nstate=3,
                                    state_spin=[0, 0, 2],
                                    target_states=[0, 1, 2])
        assert config.ntrip_root == 1
        assert config.nsing_root == 1
        assert config.tddft_nstates == 2
        assert 'NSTATES 2' in _build(config, 'tddft_energy')

    def test_a_two_manifold_force_request_is_refused_by_name(self):
        # Measured, not deduced: asking for the first singlet and the first
        # triplet as `LIST 1 3` made CP2K abort mid-print with
        # `State List contains invalid state` / `TDDFPT Print Forces: Invalid
        # State` (round tmp_20261014_142736, tddft_st_forces_test.log:674),
        # because LIST numbers roots inside one manifold and takes no spin
        # argument. Refusing at the input keeps an engine run from being spent on
        # an input known to abort.
        config = _make_tddft_config('tddft_gradient', nstate=3,
                                    state_spin=[0, 0, 2],
                                    target_states=[0, 1, 2])
        with pytest.raises(ValueError, match='no spin argument'):
            _build(config, 'tddft_gradient')

    def test_one_manifold_lists_the_numbers_it_always_listed(self):
        # The positive arm for the refusal above, and the byte-identity claim
        # behind the LIST change: inside one manifold the written numbers are the
        # rows themselves, exactly as before.
        config = _make_tddft_config('tddft_gradient', target_states=[0, 1, 2])
        text = _build(config, 'tddft_gradient')
        forces_block = text.split('&TDDFPT')[1].split('&END TDDFPT')[0]
        assert 'LIST 1 2' in forces_block

    def test_a_tddft_run_prints_the_ground_state_forces_too(self):
        # A crossing between a ground state and an excited one needs a force on
        # each, and CP2K writes both files in one calculation (the demo input at
        # cp2k_demo/td_grad/azo-1.inp asks for both and its directory holds both
        # products). Leaving row 0 at zero would hand the search a flat surface
        # with the right shape.
        config = _make_tddft_config('tddft_gradient', target_states=[0, 1, 2])
        text = _build(config, 'tddft_gradient')
        assert 'filename force' in text
        assert 'filename td_forces' in text

    def test_an_all_singlet_dft_run_prints_only_ground_state_forces(self):
        config = _make_config(profile='dft_gradient')
        text = _build(config, 'dft_gradient')
        assert 'filename force' in text
        assert 'filename td_forces' not in text

    def test_excited_state_optimization_targets_state(self):
        config = _make_tddft_config(
            'tddft_opt_min', current_state=1, target_states=[1])
        text = _build(config, 'tddft_opt_min')
        assert '&OPTIMIZE' in text
        assert 'STATE' in text

    def test_active_state_beyond_nstates_raises(self):
        with pytest.raises(ValueError, match='exceeds'):
            _make_tddft_config(
                'tddft_energy', nstate=2, state_spin=[0, 0],
                current_state=2)

    # The box the strain route needs: pbc is the authority on periodicity, so a
    # request that resolves to 'num' without one is refused by that check, not
    # silently run as a molecular calculation.
    _BOX = {'pbc': [1, 1, 1],
            'lattice_vector': [[6.3, 0.0, 0.0], [0.0, 6.3, 0.0],
                               [0.0, 0.0, 6.3]]}

    def test_stress_under_tddft_is_routed_to_the_strain_path(self):
        """A TDDFT tensor request goes to the route that can answer it.

        CP2K prints the TDDFT forces and then aborts when an analytical tensor
        is asked of it (`response_solver.F:943`, measured in the test environment in
        rounds tmp_20261010_133424 and tmp_20261010_190544, where the engine had
        already died and the task was still recorded as a pass). This case used
        to assert the refusal of the whole combination; that refusal was a dead
        end for a framework-driven variable-cell run, which asks for the tensor
        through the `stress` capability and has no opinion about routes
        (`planning/resolver.py::_derive_stress`). The route is now resolved from
        the method, and the input proves which one was taken: the strain route
        sends CP2K no stress keyword at all, because it builds the tensor from
        its own +/- strain single points.
        """
        config = _make_tddft_config('tddft_gradient', stress=True, **self._BOX)
        assert config.calculation.stress_method == 'num'
        text = _build(config, 'tddft_gradient')
        assert 'STRESS_TENSOR' not in text, (
            'the engine tensor keyword is in an input that was routed to the '
            'strain path - the route the input takes is the route that must '
            'be spoken of here')
        assert '&TDDFPT' in text, 'the excited-state block vanished with it'

    def test_a_tddft_tensor_request_without_a_periodic_box_is_still_refused(self):
        # Resolving the route does not relax what the route requires: an
        # aperiodic cell cannot be strained, and that is said by name.
        with pytest.raises(ValueError, match='full 3D periodicity'):
            _make_tddft_config('tddft_gradient', stress=True)

    def test_an_explicit_engine_tensor_under_tddft_is_refused_as_written(self):
        # The negative control: the resolution fills what is absent. A caller
        # who wrote `stress_method: analytical` on an excited state asked for
        # the combination the engine aborts on, and still gets told so before
        # a run is bought and spent on it.
        with pytest.raises(ValueError, match='Stress Tensor not available'):
            _make_tddft_config('tddft_gradient', stress=True,
                               stress_method='analytical', **self._BOX)

    def test_ground_state_stress_still_validates(self):
        # The refusal names the excited-state channel, not the keyword: the
        # same request under `method: dft` is what this skill computes
        # (`tests/dft_stress.yaml` runs it in the test environment).
        config = _make_config(profile='dft_gradient', method='dft',
                              stress=True, pbc=[1, 1, 1],
                              lattice_vector=[[6.3, 0.0, 0.0],
                                              [0.0, 6.3, 0.0],
                                              [0.0, 0.0, 6.3]])
        assert config.stress is True

    def test_properties_sibling_of_dft(self):
        # CP2K rejects &PROPERTIES nested in &DFT ('unknown
        # subsection PROPERTIES of section DFT')
        config = _make_tddft_config('tddft_energy')
        text = _build(config, 'tddft_energy')
        assert '&PROPERTIES' in text.split('&END DFT')[1]
        dft_block = text.split('&DFT')[1].split('&END DFT')[0]
        assert '&PROPERTIES' not in dft_block

    def test_dipole_form_inside_dipole_moments(self):
        # DIPOLE_FORM is a &DIPOLE_MOMENTS keyword, not &TDDFPT
        config = _make_tddft_config('tddft_energy')
        text = _build(config, 'tddft_energy')
        tddfpt_block = text.split('&TDDFPT')[1].split('&END TDDFPT')[0]
        assert '&DIPOLE_MOMENTS' in tddfpt_block
        dipole_block = tddfpt_block.split(
            '&DIPOLE_MOMENTS')[1].split('&END DIPOLE_MOMENTS')[0]
        assert 'DIPOLE_FORM' in dipole_block
