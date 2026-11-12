"""
Unit tests for the two LAMMPS QM/MM tasks.

No LAMMPS process is started here. The tests cover the profile/validation
gate, the partition charge rules, the in-cell Q-Q evaluator, the two
assembly paths, the file writers/readers, and the native-to-framework
conversion. Real-engine behavior lives in test_qmmm_engine.py.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import os

import numpy as np
import pytest

from ..schema.adapter import build_config
from ..schema.method import resolve_profile
from ..schema.normalize import normalize
from ..schema.validate import (PERSISTENT_PROFILES, QMMM_PROFILES,
                               validate)
from ..service.builder.ff_model import build_ff_model
from ..service.builder.qmmm import (in_cell_qq, in_cell_qq_by_centers,
                                    plan_partition,
                                    validate_partition_charges)
from ._result_units import apply_result_units
from ..service.parser.output_parser import to_native_gradient
from ..service.utils import FORCE_TO_HARTREE_PER_BOHR
from ..service.parser.qmmm_output import (ESPF_FIELD_SIGN,
                                          assemble_qmmm_energy,
                                          assemble_qmmm_field,
                                          read_esfp, write_esfp,
                                          write_qm_force, write_qm_multipole,
                                          write_qmmm_energy)

BASE = {
    'lammps_project': 'h2o', 'method': 'ff',
    'atoms': ['O', 'H', 'H', 'O', 'H', 'H'],
    'coordinates': [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0],
                    [5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0]],
    'charges': 0, 'multiplicity': 1, 'natom': 6,
    'qm_atoms': [0, 1, 2], 'lammps_units': 'real',
    'execution_mode': 'local', 'pbc': [0, 0, 0],
    'lammps_path': '',
}

TEN_LATTICE = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
TEN_POSITIONS = [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0]]
TEN_CHARGES = [-0.830, 0.415, 0.415]


def _config(profile, **overrides):
    values = dict(BASE, profile=profile)
    values.update(overrides)
    return validate(normalize(build_config(values)))


def test_the_two_tasks_are_standalone_profiles(tmp_path, monkeypatch):
    (tmp_path / 'h2o.data').write_text('LAMMPS Description\n')
    monkeypatch.chdir(tmp_path)
    assert QMMM_PROFILES == ('qmmm_field', 'qmmm_energy')
    assert 'qmmm_field' in PERSISTENT_PROFILES
    assert 'qmmm_energy' in PERSISTENT_PROFILES
    assert _config('qmmm_field').calculation.profile == 'qmmm_field'
    assert _config('qmmm_energy', q_qm=TEN_CHARGES).q_qm == TEN_CHARGES


def test_old_qmmm_switch_on_ff_gradient_is_refused():
    with pytest.raises(ValueError, match='qmmm_field|qmmm_energy'):
        _config('ff_gradient', qmmm=True)


def test_qm_atoms_is_required_and_checked():
    with pytest.raises(ValueError, match='qm_atoms'):
        _config('qmmm_field', qm_atoms=[])
    with pytest.raises(ValueError, match='outside'):
        _config('qmmm_field', qm_atoms=[0, 1, 6])
    with pytest.raises(ValueError, match='duplicate'):
        _config('qmmm_field', qm_atoms=[0, 0, 1])
    with pytest.raises(ValueError, match='whole system'):
        _config('qmmm_field', qm_atoms=list(range(6)))


def test_profile_is_resolved_from_the_duty_tokens():
    # The framework plan appends the field/qmmm capability token; the
    # skill owns the profile names. A bare energy/gradient jobtype stays
    # on the ordinary ff_* profiles.
    assert resolve_profile('ff', 'energy gradient field') == 'qmmm_field'
    assert resolve_profile('ff', 'energy gradient qmmm') == 'qmmm_energy'
    assert resolve_profile('ff', 'energy gradient') == 'ff_gradient'
    assert resolve_profile('ff', 'energy') == 'ff_energy'


def test_energy_assembly_without_embedding_keeps_qmmm_el():
    partition = {'qm_index': [0, 1, 2], 'qm_ids': [1, 2, 3],
                 'removed': {}}
    block = assemble_qmmm_energy(
        -10.0, np.zeros((6, 3)), partition, np.array(TEN_CHARGES),
        None, None, TEN_POSITIONS, TEN_LATTICE, 'real', embedding=False)
    assert block['embedding'] is False
    assert block['qmmm_el_native'] == 0.0
    assert np.allclose(block['qmmm_el_gradient_native'], 0.0)
    assert block['e_classic_native'] == pytest.approx(
        -10.0 - block['u_gas_native'])


def test_field_ignores_energy_inputs_and_energy_checks_q_qm(tmp_path,
                                                             monkeypatch):
    # The probe sets Q=+1 at run time; q_qm/qmmm_field_file are the
    # energy task's inputs and are simply unused here, not refused.
    # The validating leg reaches the topology check, so stage a data file
    # rather than inheriting whatever the working directory happens to hold.
    (tmp_path / 'h2o.data').write_text('LAMMPS Description\n')
    monkeypatch.chdir(tmp_path)
    field = _config('qmmm_field', q_qm=TEN_CHARGES,
                    qmmm_field_file='/nonexistent/esfp')
    assert field.q_qm == TEN_CHARGES
    assert field.qmmm_field_file == '/nonexistent/esfp'
    with pytest.raises(ValueError, match='q_qm|order'):
        _config('qmmm_energy', q_qm=[0.0, 0.0])
    with pytest.raises(ValueError, match='non-finite'):
        _config('qmmm_energy', q_qm=[float('nan'), 0.0, 0.0])
    with pytest.raises(ValueError, match='does not exist'):
        _config('qmmm_energy', qmmm_field_file='/nonexistent/esfp')


def test_the_shipped_topology_loads_through_the_validator(monkeypatch):
    # The shipped fixture is a package resource, so it is read from its own
    # directory by name: the skill asks for `{lammps_project}.data` in the
    # launch directory, and this is the one test that says the file that
    # ships actually passes the real loader.
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    assert _config('qmmm_field').lammps_project == 'h2o'


def test_qmmm_boolean_is_retired_on_the_new_profiles():
    with pytest.raises(ValueError, match='retired'):
        _config('qmmm_field', qmmm=True)


def test_partition_charge_rules():
    model = {'sections': {
        'Atoms_header': '# full',
        'Atoms': [
            '1 1 1 -0.830 1.0 1.0 1.0',
            '2 1 2 0.415 2.0 1.0 1.0',
            '3 1 2 0.415 1.0 2.0 1.0',
            '4 2 1 -0.830 5.0 5.0 5.0',
            '5 2 2 0.415 6.0 5.0 5.0',
            '6 2 2 0.415 5.0 6.0 5.0',
        ],
    }}
    partition = plan_partition(model, [0, 1, 2])
    assert partition['qm_ids'] == [1, 2, 3]
    assert partition['qm_index'] == [0, 1, 2]
    validate_partition_charges(partition, partition['qm_charges'], 'field')
    validate_partition_charges(partition, TEN_CHARGES, 'energy')
    # A charged QM region is admissible on both flows: the declared per-center
    # charge is what the framework balances to, and neither side is forced
    # neutral (the A/B background cancels between the probe's two legs).
    validate_partition_charges(partition, [0.0, 0.0, 0.1], 'energy')
    validate_partition_charges(partition, [0.0, 0.0, 0.1], 'field')
    # What is still refused is a vector that does not name every QM atom: the
    # unprobed centers are environment charges.
    with pytest.raises(Exception, match='values for'):
        validate_partition_charges(partition, [0.0, 0.1], 'energy')
    with pytest.raises(Exception, match='values for'):
        validate_partition_charges(partition, [0.0, 0.1], 'field')


def test_in_cell_qq_matches_the_known_periodic_value():
    result = in_cell_qq(TEN_POSITIONS, TEN_CHARGES, TEN_LATTICE, 'real')
    assert result['energy'] == pytest.approx(-188.31948461, rel=1e-9)
    assert len(result['forces']) == 3
    assert np.all(np.isfinite(result['forces']))


def test_field_assembly_uses_half_potential_and_checks_group_tally():
    partition = {'qm_index': [0, 1, 2], 'qm_ids': [1, 2, 3]}
    raw = {
        'pe_a': {1: 1.0, 2: 2.0, 3: 3.0},
        'pe_b': {1: 0.0, 2: 0.0, 3: 0.0},
        'forces_a': np.array([[1.0, 0.0, 0.0],
                              [2.0, 0.0, 0.0],
                              [3.0, 0.0, 0.0]]),
        'forces_b': np.zeros((3, 3)),
        'groups': {'A': {'energy': 12.0, 'force': [6.0, 0.0, 0.0]},
                   'B': {'energy': 0.0, 'force': [0.0, 0.0, 0.0]}},
    }
    block = assemble_qmmm_field(raw, partition, 'real')
    assert block['v_mm_native'].tolist() == [2.0, 4.0, 6.0]
    assert block['e_mm_native'][:, 0].tolist() == [1.0, 2.0, 3.0]
    assert block['sanity']['v_vs_u_gg'] == pytest.approx(0.0)
    assert block['probe_charge'] == 1.0


def test_field_kspace_pin_fixes_alpha_only():
    # A and B must agree on the reciprocal sum: LAMMPS re-derives alpha and
    # the extent of that sum from qsqsum at every init, and zeroing the MM
    # charges changes qsqsum. The split parameter pinned here is alpha. The
    # extent cannot be pinned - ewald accepts `kspace_modify mesh` and
    # ignores it, and `kspace_modify cutoff` is refused outright on the
    # deployed build - so the probe carries an accuracy from `ewald_accuracy`
    # and both runs converge instead. Measured on the 1581-atom box: 1e-6
    # leaves a 5.3e-4 relative residual and 1e-8 leaves 2.1e-6, with 1e-10
    # no longer moving the readings.
    from ..service.builder.qmmm_field import field_kspace_setup
    assert field_kspace_setup([8.399091] * 3, 10.0, 'pppm', '1e-8') == \
        'kspace_style pppm 1e-8\nkspace_modify gewald 0.320000'
    assert 'mesh' not in field_kspace_setup([8.399091] * 3, 10.0,
                                            'pppm', '1e-8')
    assert field_kspace_setup([8.399091] * 3, 10.0, 'ewald', '1e-8') == \
        'kspace_style ewald 1e-8\nkspace_modify gewald 0.320000'
    assert field_kspace_setup([2.0] * 3, None, 'pppm', '1e-8') == \
        'kspace_style pppm 1e-8\nkspace_modify gewald 3.200000'
    assert field_kspace_setup([], 10.0, 'pppm', '1e-8') == ''
    # The token is the caller's value, not a constant folded into the
    # builder: a looser accuracy has to appear verbatim, or every case that
    # declares 1e-4 would silently run at the default.
    assert field_kspace_setup([8.399091] * 3, 10.0, 'ewald', '1e-4') == \
        'kspace_style ewald 1e-4\nkspace_modify gewald 0.320000'


def test_field_cutoff_comes_from_the_last_pair_argument():
    from ..service.builder.qmmm_field import _coul_cutoff
    model = {'init': {'styles': {'pair': {'segments': [
        ['lj/charmm/coul/long', ['9.0', '10.0', '10.0']],
        ['coul/long', ['12.0']],
    ]}}}}
    assert _coul_cutoff(model) == 10.0
    assert _coul_cutoff({'init': {}}) is None


def test_field_cutoff_reads_a_plain_pair_style_too():
    # The model segments a hybrid declaration only, so a plain
    # `pair_style` has to be read off the init line its entry points at.
    # Missing it is not a crash: the caller silently falls back to half
    # the box and pins an alpha whose real-space truncation is not the
    # erfc(3.2) the pin is specified by.
    from ..service.builder.qmmm_field import _coul_cutoff
    plain = {'init': {
        'lines': ['units real',
                  'pair_style lj/charmm/coul/long 9.0 10.0',
                  'kspace_style ewald 0.0001'],
        'styles': {'pair': {'index': 1, 'hybrid': False, 'wrapper': '',
                            'segments': [], 'unobserved': False}}}}
    assert _coul_cutoff(plain) == 10.0
    # A comment after the arguments must not be taken for one.
    commented = {'init': {
        'lines': ['pair_style coul/long 11.0  # rc'],
        'styles': {'pair': {'index': 0, 'hybrid': False, 'segments': []}}}}
    assert _coul_cutoff(commented) == 11.0
    # Negative control: a hybrid wrapper with no observed sub-style is
    # not answered from the raw line, where the per-sub-style argument
    # boundaries cannot be told apart.
    unobserved = {'init': {
        'lines': ['pair_style hybrid lj/charmm/coul/long 9.0 10.0 10.0'],
        'styles': {'pair': {'index': 0, 'hybrid': True, 'wrapper': 'hybrid',
                            'segments': [], 'unobserved': True}}}}
    assert _coul_cutoff(unobserved) is None


def test_field_probe_dumps_full_precision():
    # The cold path reads V from the pe dump; LAMMPS's default %g keeps
    # six digits and would quantise V at ~5e-4.
    from ..service.builder.qmmm_field import _probe_run
    text = _probe_run('A', 'h2o_qmmm_field', '/tmp/calc')
    assert text.count('format float %20.17g') == 2


def test_energy_assembly_uses_qv_and_in_cell_gas():
    partition = {'qm_index': [0, 1, 2], 'qm_ids': [1, 2, 3],
                 'removed': {}}
    v = np.array([1.0, 2.0, 3.0])
    e = np.zeros((3, 3))
    q = np.array(TEN_CHARGES)
    block = assemble_qmmm_energy(
        -10.0, np.zeros((6, 3)), partition, q, v, e,
        TEN_POSITIONS, TEN_LATTICE, 'real')
    assert block['qmmm_el_native'] == pytest.approx(float(q @ v))
    assert block['u_main_native'] == pytest.approx(-10.0)
    assert block['e_classic_native'] == pytest.approx(
        -10.0 - block['u_gas_native'] - float(q @ v))
    assert block['u_gas_gradient_native'].shape == (3, 3)
    assert block['qmmm_el_gradient_native'].shape == (3, 3)


def _distinct_gradient():
    """A gradient whose every row and column differs, so an index bug shows."""
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],
                     [-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0],
                     [-7.0, -8.0, -9.0]])


def test_reduced_gradient_moves_the_q_rows_only():
    partition = {'qm_index': [0, 1, 2], 'qm_ids': [1, 2, 3], 'removed': {}}
    v = np.array([1.0, 2.0, 3.0])
    e = np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    main = _distinct_gradient()
    block = assemble_qmmm_energy(-10.0, -main, partition, np.array(TEN_CHARGES),
                                 v, e, TEN_POSITIONS, TEN_LATTICE, 'real')
    mm = block['mm_gradient_native']
    # MM rows are handed through as a copy, never as arithmetic.
    np.testing.assert_array_equal(mm[3:6], main[3:6])
    np.testing.assert_allclose(mm[0:3],
                               main[0:3] - block['u_gas_gradient_native']
                               - block['qmmm_el_gradient_native'],
                               rtol=0.0, atol=1e-14)
    # Non-identity witness: with a nonzero field the Q rows really moved.
    assert float(np.abs(mm[0:3] - main[0:3]).max()) > 0.0


def test_zero_auxiliaries_reduce_to_a_bitwise_copy():
    from ..service.parser.qmmm_output import _reduce_main_gradient
    main = _distinct_gradient()
    zeros = np.zeros((3, 3))
    out = _reduce_main_gradient(main, zeros, zeros, {'qm_index': [0, 1, 2]})
    # x - 0.0 is exact, so an isolated QM run cannot drift by even one bit.
    np.testing.assert_array_equal(out, main)


def test_reduction_refuses_a_broken_partition():
    from ..service.parser.qmmm_output import _reduce_main_gradient
    main = _distinct_gradient()
    three = np.zeros((3, 3))
    with pytest.raises(ValueError, match='duplicate'):
        _reduce_main_gradient(main, np.zeros((2, 3)), np.zeros((2, 3)),
                              {'qm_index': [0, 0]})
    with pytest.raises(ValueError, match='outside'):
        _reduce_main_gradient(main, three, three, {'qm_index': [0, 1, 9]})
    with pytest.raises(ValueError, match='shape'):
        _reduce_main_gradient(main, np.zeros((2, 3)), three,
                              {'qm_index': [0, 1, 2]})


def test_grad_file_declares_units_and_sign_convention(tmp_path):
    from ..service.parser.qmmm_output import write_grad
    main = _distinct_gradient()
    write_grad(str(tmp_path), 't', main, 'real')
    text = (tmp_path / 't.grad').read_text()
    assert 'dU/dR' in text and 'real' in text
    rows = [line for line in text.splitlines() if not line.startswith('#')]
    assert len(rows) == 6
    assert rows[0].split()[0] == '1'
    back = np.array([[float(value) for value in row.split()[1:]]
                     for row in rows])
    np.testing.assert_allclose(back, main, rtol=0.0, atol=1e-12)


def test_esfp_round_trip_is_atomic_units(tmp_path):
    v = np.array([-3.275, -3.664, -3.664])
    e = np.array([[0.433, 0.433, 0.753],
                  [0.322, 0.592, 0.982],
                  [0.592, 0.322, 0.982]])
    write_esfp(str(tmp_path), 'h2o_qmmm_field', v, e, 'real')
    text = (tmp_path / 'h2o_qmmm_field.esfp').read_text()
    v_back, e_back = read_esfp(text, 'real')
    assert np.allclose(v_back, v, atol=1e-12)
    assert np.allclose(e_back, e, atol=1e-12)
    assert ESPF_FIELD_SIGN == 1.0
    assert 'atomic units' in text


def test_energy_and_force_writers(tmp_path):
    write_qm_multipole(str(tmp_path), 't', [1, 2, 3], TEN_CHARGES)
    write_qmmm_energy(str(tmp_path), 't', -4.0, -9.0, -0.1, 5.1)
    write_qm_force(str(tmp_path), 't', 'u_gas', [1, 2, 3],
                   np.zeros((3, 3)))
    write_qm_force(str(tmp_path), 't', 'qmmm_el', [1, 2, 3],
                   np.ones((3, 3)))
    qm = (tmp_path / 't.qm_multipole').read_text()
    assert qm.splitlines()[2].split()[0] == '1'
    energies = (tmp_path / 't.qmmm').read_text()
    for label in ('E_classic', 'U_main', 'U_gas', 'qmmm_el'):
        assert label in energies
    assert (tmp_path / 't.u_gas.force').exists()
    assert (tmp_path / 't.qmmm_el.force').exists()


def test_energy_model_splits_qm_types_and_zeroes_only_q_q_lj(
        tmp_path, monkeypatch):
    (tmp_path / 'probe.data').write_text("""LAMMPS Description

2 atoms
0 bonds
0 angles
0 dihedrals
0 impropers

1 atom types
0 bond types
0 angle types
0 dihedral types
0 improper types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 1.0

Atoms # full

1 1 1 0.0 1.0 1.0 1.0
2 1 1 0.0 2.0 1.0 1.0
""")
    (tmp_path / 'probe.in.init').write_text("""units real
atom_style full
boundary f f f
pair_style lj/cut/coul/cut 9.0
pair_modify mix arithmetic
""")
    (tmp_path / 'probe.in.settings').write_text(
        'pair_coeff 1 1 0.102 3.188\n')
    monkeypatch.chdir(tmp_path)
    config = validate(normalize(build_config({
        'lammps_project': 'probe', 'profile': 'qmmm_energy', 'method': 'ff',
        'atoms': ['O', 'O'], 'coordinates': [[1.0, 1.0, 1.0],
                                             [2.0, 1.0, 1.0]],
        'charges': 0, 'multiplicity': 1, 'natom': 2,
        'qm_atoms': [0], 'q_qm': [0.0], 'lammps_units': 'real',
        'lammps_path': '', 'execution_mode': 'local', 'pbc': [0, 0, 0],
    })))
    model = build_ff_model(config, 'qmmm_energy')
    assert model['qmmm']['new_types'] == [2]
    rows = {}
    for row in model['coeffs']['pair']:
        ids = tuple(sorted((int(row[0]), int(row[1]))))
        rows.setdefault(ids, []).append(row)
    # M-M keeps its LJ, Q-M keeps the copied LJ, Q-Q has LJ zeroed.
    assert float(rows[(1, 1)][0][2]) == pytest.approx(0.102)
    assert float(rows[(1, 2)][0][2]) == pytest.approx(0.102)
    assert float(rows[(2, 2)][0][2]) == pytest.approx(0.0)
    qm_row = next(line for line in model['sections']['Atoms']
                  if line.startswith('1 '))
    assert qm_row.split()[2] == '2'


def test_field_model_zeroes_every_non_coulomb_row(
        tmp_path, monkeypatch):
    (tmp_path / 'probe.data').write_text("""LAMMPS Description

2 atoms
0 bonds
0 angles
0 dihedrals
0 impropers

1 atom types
0 bond types
0 angle types
0 dihedral types
0 improper types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 1.0

Atoms # full

1 1 1 0.0 1.0 1.0 1.0
2 1 1 0.0 2.0 1.0 1.0
""")
    (tmp_path / 'probe.in.init').write_text("""units real
atom_style full
boundary f f f
pair_style lj/cut/coul/cut 9.0
pair_modify mix arithmetic
""")
    (tmp_path / 'probe.in.settings').write_text(
        'pair_coeff 1 1 0.102 3.188\n')
    monkeypatch.chdir(tmp_path)
    config = validate(normalize(build_config({
        'lammps_project': 'probe', 'profile': 'qmmm_field', 'method': 'ff',
        'atoms': ['O', 'O'], 'coordinates': [[1.0, 1.0, 1.0],
                                             [2.0, 1.0, 1.0]],
        'charges': 0, 'multiplicity': 1, 'natom': 2,
        'qm_atoms': [0], 'lammps_units': 'real',
        'lammps_path': '', 'execution_mode': 'local', 'pbc': [0, 0, 0],
    })))
    model = build_ff_model(config, 'qmmm_field')
    assert 'new_types' not in model['qmmm']
    for row in model['coeffs']['pair']:
        # qmmm_field is a pure Coulomb probe; every non-Coulomb pair
        # parameter is zero, not just the Q-Q one.
        assert float(row[2]) == pytest.approx(0.0)


def test_skill_hands_gradient_and_framework_scales_once():
    # Two steps, two owners: the skill turns the engine's force into the
    # `gradient` label (sign is what the label MEANS) and still reports it
    # in the run's native unit; the framework scales by the unit its
    # manifest declared.
    forces = [[1.0, 0.0, 0.0]]
    native = {'energy': [627.5094740631],
              'gradient': to_native_gradient(forces),
              '_native_units': 'real'}
    converted = apply_result_units(dict(native), 'lammps_skill')
    assert converted['energy'][0] == pytest.approx(1.0)
    assert converted['gradient'][0, 0, 0] == pytest.approx(
        -FORCE_TO_HARTREE_PER_BOHR['real'])
    assert '_native_units' not in converted
    # A second pass must not scale again (the marker is gone): the call
    # returns the caller's own object, untouched.
    again = apply_result_units(converted, 'lammps_skill')
    assert again is converted
    assert again['energy'][0] == pytest.approx(1.0)


PERIODIC_DATA = """LAMMPS Description

4 atoms
0 bonds
0 angles
0 dihedrals
0 impropers

1 atom types
0 bond types
0 angle types
0 dihedral types
0 improper types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 1.0

Atoms # full

1 1 1 0.0 1.0 1.0 1.0
2 1 1 -0.83 5.0 5.0 5.0
3 1 1 0.415 6.0 5.0 5.0
4 1 1 0.415 5.0 6.0 5.0
"""

PERIODIC_INIT = """units real
atom_style full
boundary p p p
pair_style lj/charmm/coul/long 9.0 10.0
pair_modify mix arithmetic
kspace_style ewald 0.0001
"""


def _field_config(tmp_path, monkeypatch, pbc, cell, accuracy=None):
    """A qmmm_field config for one QM atom in a box, staged and validated.

    One QM atom and a net-neutral three-atom MM water: ``qmmm_field``
    overwrites the probe charge to +1 itself, and only the MM partition
    has to be neutral for the A/B Ewald backgrounds to cancel.

    ``accuracy`` leaves the ``ewald_accuracy`` key unset when it is None, so
    the caller can tell the default apart from a declared value.
    """
    (tmp_path / 'probe.data').write_text(PERIODIC_DATA)
    (tmp_path / 'probe.in').write_text('# PyRAI3MD probe\n')
    (tmp_path / 'probe.in.init').write_text(PERIODIC_INIT)
    (tmp_path / 'probe.in.settings').write_text(
        'pair_coeff 1 1 0.102 3.188\n')
    monkeypatch.chdir(tmp_path)
    raw = {
        'lammps_project': 'probe', 'profile': 'qmmm_field', 'method': 'ff',
        'atoms': ['O', 'O', 'H', 'H'],
        'coordinates': [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0],
                        [6.0, 5.0, 5.0], [5.0, 6.0, 5.0]],
        'charges': 0, 'multiplicity': 1, 'natom': 4,
        'qm_atoms': [0], 'lammps_units': 'real', 'lammps_path': '',
        'execution_mode': 'local', 'pbc': pbc, 'cell': cell,
    }
    if accuracy is not None:
        raw['ewald_accuracy'] = accuracy
    return validate(normalize(build_config(raw)))


def _field_deck(tmp_path, monkeypatch, pbc, cell, accuracy=None):
    """The cold qmmm_field script for that config, as text."""
    config = _field_config(tmp_path, monkeypatch, pbc, cell, accuracy)
    from ..service.workflow import compile_workflow
    from ..service.builder.qmmm_field import write_field_input_file
    basename = write_field_input_file(
        config, compile_workflow('qmmm_field'), str(tmp_path),
        build_ff_model(config, 'qmmm_field'))
    return (tmp_path / f'{basename}.in').read_text()


def _field_package(tmp_path, monkeypatch, pbc, cell):
    """The resident qmmm_field package: its script text and file names."""
    config = _field_config(tmp_path, monkeypatch, pbc, cell)
    from ..service.workflow import compile_workflow
    from ..service.builder.qmmm_field import write_field_static_input_files
    package = write_field_static_input_files(
        config, compile_workflow('qmmm_field'), str(tmp_path),
        build_ff_model(config, 'qmmm_field'))
    script = (tmp_path / ('%s.in' % package['script'])).read_text()
    return script, package


class TestFieldDeckFollowsTheDeclaredPeriodicity:
    """The A/B probe's Ewald split is pinned only when the run says pbc.

    Both the pin and the kspace channel of the two tallies hang off
    ``_is_periodic(config)``, and ``pbc`` is the framework's periodicity
    authority. A periodic box declared without ``pbc`` therefore does not
    become a molecular run: the data template still carries a box and the
    settings tail still re-issues ``kspace_style``, so the engine runs
    Ewald while the deck asks neither tally for the kspace term and never
    pins alpha. That is the A/B mismatch measured in the test round
    ``tmp_20261031_005246`` (three differing G vectors, one of them under
    ``WARNING: System is not charge neutral``), and it is why the pin has
    to be asserted on an assembled deck and not only on
    ``field_kspace_setup``, which was correct all along.
    """

    def test_periodic_deck_pins_the_split_and_tallies_kspace(
            self, tmp_path, monkeypatch):
        script = _field_deck(tmp_path, monkeypatch, [1, 1, 1],
                             [10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
        assert 'kspace_style ewald 1e-8' in script
        # alpha = 3.2/rc with rc the Coulomb cutoff of lj/charmm/coul/long.
        assert 'kspace_modify gewald %.6f' % (3.2 / 10.0) in script
        assert 'compute QMMM_PEQ QMMM_QM pe/atom pair kspace' in script
        assert ('compute QMMM_GG QMMM_QM group/group QMMM_MM'
                ' pair yes kspace yes') in script

    def test_declared_accuracy_replaces_the_template_token(
            self, tmp_path, monkeypatch):
        # The template asks for 1e-4 as `0.0001`. A case that declares
        # ewald_accuracy must see its own token on both lines the deck
        # carries, with no trace of the template's spelling left - otherwise
        # the key is decorative and the probe still runs the loose sum that
        # cost the 1581-atom box 5.3e-4 of its V.
        script = _field_deck(tmp_path, monkeypatch, [1, 1, 1],
                             [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
                             accuracy='1e-4')
        assert 'kspace_style ewald 1e-4' in script
        assert '0.0001' not in script
        assert '1e-8' not in script

    def test_molecular_deck_carries_no_kspace_channel(
            self, tmp_path, monkeypatch):
        # Negative control on the same builder: the assertions above are
        # not identities over every field deck. With pbc off there is no
        # Ewald sum to split, so the pin and the kspace tallies must both
        # be absent rather than merely different.
        script = _field_deck(tmp_path, monkeypatch, [0, 0, 0], [])
        assert 'kspace_modify gewald' not in script
        assert 'compute QMMM_PEQ QMMM_QM pe/atom pair\n' in script
        assert ('compute QMMM_GG QMMM_QM group/group QMMM_MM'
                ' pair yes\n') in script
        assert 'kspace yes' not in script


class TestResidentPackageLeavesTheBoxChangeable:
    """The resident field package must not hold a dump, and must still
    consume its per-atom compute.

    A declared dump is what makes `change_box all triclinic` illegal on the
    deployed build (`Cannot change box ortho/triclinic with dumps defined`,
    `src/change_box.cpp:308`), and the first tilted step of a variable-cell
    search took the whole resident session down with it. The compute still
    needs a consumer to be made current, so the package carries a `reduce`
    over an atom-style alias instead: asserting only "no dump" would also
    pass a deck that never refreshes V.
    """

    def test_periodic_package_has_no_dump_but_a_consumer(
            self, tmp_path, monkeypatch):
        script, package = _field_package(
            tmp_path, monkeypatch, [1, 1, 1],
            [10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
        assert 'dump ' not in script
        assert 'dump_modify' not in script
        assert 'variable QMMM_PEQV atom c_QMMM_PEQ' in script
        assert 'compute QMMM_PEQS all reduce sum v_QMMM_PEQV' in script
        # A reduce nobody references is never invoked, and an uninvoked
        # compute is what hands the library read a stale array: the
        # deployment's second evaluation read V from the first one and broke
        # the A/B identity by 2.65 of 249. The thermo line is the consumer.
        assert ('thermo_style custom step pe c_QMMM_GG c_QMMM_GG[1] '
                'c_QMMM_GG[2] c_QMMM_GG[3] c_QMMM_PEQS' in script)
        assert 'compute QMMM_PEQ QMMM_QM pe/atom pair kspace' in script
        # The consumer owns no file, so the package declares no artifact
        # beside the input package it genuinely needs.
        assert not [name for name in package['files']
                    if name.endswith('.dump')]

    def test_molecular_package_declares_the_consumer_too(
            self, tmp_path, monkeypatch):
        # Non-identity witness for the pair above: the consumer is not a
        # periodic-only decoration, so a molecular package carries it as
        # well while keeping neither the pin nor the kspace channel.
        script, _ = _field_package(tmp_path, monkeypatch, [0, 0, 0], [])
        assert 'variable QMMM_PEQV atom c_QMMM_PEQ' in script
        assert 'c_QMMM_PEQS' in script
        assert 'compute QMMM_PEQ QMMM_QM pe/atom pair\n' in script
        assert 'kspace_modify gewald' not in script


THREE_WATERS = [
    '1 1 1 -0.830 1.0 1.0 1.0',
    '2 1 2 0.415 2.0 1.0 1.0',
    '3 1 2 0.415 1.0 2.0 1.0',
    '4 2 1 -0.830 5.0 5.0 5.0',
    '5 2 2 0.415 6.0 5.0 5.0',
    '6 2 2 0.415 5.0 6.0 5.0',
    '7 3 1 -0.830 9.0 9.0 9.0',
    '8 3 2 0.415 10.0 9.0 9.0',
    '9 3 2 0.415 9.0 10.0 9.0',
]

WATER_Q = [-0.830, 0.415, 0.415]

# The rendered data file the QM/MM charge write works on: a bare section
# header, one line before the rows (schema/adapter writes the style comment
# into the parsed model, not into the file the writer re-reads).
DATA_TEXT = ('three waters\n\n9 atoms\n\n3 atom types\n\n'
             '0.0 20.0 x\n0.0 20.0 y\n0.0 20.0 z\n\n'
             'Atoms\n\n' + '\n'.join(THREE_WATERS) + '\n')


def _three_water_model():
    """Two QM waters and one MM water, in one data section."""
    return {'sections': {'Atoms_header': '# full',
                         'Atoms': list(THREE_WATERS)}}


class TestProbeIsNarrowedToOneCenter:
    """``qm_probe`` makes one deck measure one center of the QM set.

    With one center the probe is the QM set and every name below collapses
    to what a single-center run already produced; the assertions that pin
    the single-center wording are what keeps that true, because the whole
    point of putting the other centers on the environment side is that they
    survive the A/B subtraction, and nothing else in the deck says so.
    """

    def test_partition_separates_probe_environment_and_mm(self):
        partition = plan_partition(_three_water_model(), list(range(6)),
                                   probe_atoms=[3, 4, 5])
        assert partition['qm_ids'] == [1, 2, 3, 4, 5, 6]
        assert partition['probe_ids'] == [4, 5, 6]
        assert partition['probe_index'] == [3, 4, 5]
        # The unprobed center is environment, the third water is MM, and the
        # two together are all the A/B difference keeps.
        assert partition['env_ids'] == [1, 2, 3]
        assert partition['mm_ids'] == [7, 8, 9]

    def test_no_probe_means_the_whole_qm_set(self):
        # Identity anchor: omitting the keyword is not a different scheme,
        # it is the one-center deck with the names it always had.
        partition = plan_partition(_three_water_model(), [0, 1, 2])
        assert partition['probe_ids'] == partition['qm_ids'] == [1, 2, 3]
        assert partition['env_ids'] == []

    def test_probe_outside_the_qm_set_is_refused(self):
        with pytest.raises(Exception, match='outside qm_atoms'):
            plan_partition(_three_water_model(), [0, 1, 2],
                           probe_atoms=[7, 8])

    def test_deck_declares_the_probe_as_the_measured_group(self):
        from ..service.builder.qmmm_field import _groups
        narrowed = _groups(plan_partition(_three_water_model(),
                                          list(range(6)),
                                          probe_atoms=[0, 1, 2]))
        plain = _groups(plan_partition(_three_water_model(), [0, 1, 2]))
        assert 'group QMMM_QM id 1 2 3' in narrowed
        assert 'group QMMM_MM subtract all QMMM_QM' in narrowed
        # The probed center of the two-center partition and the whole QM set
        # of the one-center run produce the same deck text: the difference
        # between them is in the environment charges, not the group names.
        assert narrowed == plain

    def test_field_carries_a_charged_unprobed_center(self):
        partition = plan_partition(_three_water_model(), list(range(6)),
                                   probe_atoms=[3, 4, 5])
        # The center this probe does not measure sits in the environment at
        # its declared charge; a charged one is now admissible, and what the
        # check still requires is that the vector names every QM atom.
        validate_partition_charges(partition, WATER_Q + WATER_Q, 'field')
        validate_partition_charges(
            partition, [0.0, 0.0, 0.1] + WATER_Q, 'field')
        with pytest.raises(Exception, match='values for'):
            validate_partition_charges(partition, WATER_Q, 'field')

    def test_single_center_without_charges_is_a_no_op(self):
        # The topology-time call: no `q_qm` has arrived, there is nothing to
        # shape, and a charged MM cell is no longer refused.
        partition = plan_partition(_three_water_model(), [0, 1])
        validate_partition_charges(partition, None, 'field')

    def test_assemble_reports_the_probed_center_only(self):
        partition = plan_partition(_three_water_model(), list(range(6)),
                                   probe_atoms=[3, 4, 5])
        raw = {
            'pe_a': {atom_id: 1.0 for atom_id in range(1, 10)},
            'pe_b': {atom_id: 0.0 for atom_id in range(1, 10)},
            'forces_a': [[0.0, 0.0, 0.0]] * 3,
            'forces_b': [[0.0, 0.0, 0.0]] * 3,
            'groups': {'A': {'energy': 6.0, 'force': [0.0, 0.0, 0.0]},
                       'B': {'energy': 0.0, 'force': [0.0, 0.0, 0.0]}},
        }
        block = assemble_qmmm_field(raw, partition, 'real')
        assert block['qm_atoms'] == [3, 4, 5]
        assert block['qm_atom_ids'] == [4, 5, 6]
        assert block['v_mm_native'].shape == (3,)


class TestFieldRowsNameTheirAtoms:
    """The handoff carries the atom id, so a row cannot be renumbered."""

    def test_write_and_read_keep_the_id_column(self, tmp_path):
        v = np.array([-3.275, -3.664])
        e = np.array([[0.433, 0.433, 0.753], [0.322, 0.592, 0.982]])
        write_esfp(str(tmp_path), 'c1_probe', v, e, 'real', qm_ids=[4, 5])
        text = (tmp_path / 'c1_probe.esfp').read_text()
        rows = [line.split() for line in text.splitlines()
                if line.strip() and not line.startswith('#')]
        assert [int(row[0]) for row in rows] == [4, 5]
        assert len(rows[0]) == 5
        v_back, e_back = read_esfp(text, 'real')
        np.testing.assert_allclose(v_back, v, rtol=0, atol=1e-12)
        np.testing.assert_allclose(e_back, e, rtol=0, atol=1e-12)

    def test_positional_file_still_reads(self):
        # A hand-written standalone handoff has no id column; the reader
        # keeps serving it, which is why the column is additive and not a
        # silent re-interpretation of the first number.
        from ..service.parser.qmmm_output import esfp_values
        v = np.array([-3.275, -3.664])
        e = np.array([[0.433, 0.433, 0.753], [0.322, 0.592, 0.982]])
        v_au, e_au = esfp_values(v, e, 'real')
        text = '# legacy\n' + '\n'.join(
            ' '.join(f'{number:.16e}'
                     for number in (value, *field))
            for value, field in zip(v_au, e_au))
        rows = [line.split() for line in text.splitlines()
                if line.strip() and not line.startswith('#')]
        assert len(rows[0]) == 4
        v_back, e_back = read_esfp(text, 'real')
        np.testing.assert_allclose(v_back, v, rtol=0, atol=1e-12)
        np.testing.assert_allclose(e_back, e, rtol=0, atol=1e-12)


class TestEnvironmentChargesReachTheDataFile:
    """The unprobed center's charges are written by atom id, not by row."""

    def test_only_the_named_rows_change(self, tmp_path):
        from ..service.builder.qmmm_energy import _rewrite_qm_charges
        (tmp_path / 'probe.data').write_text(DATA_TEXT)
        _rewrite_qm_charges(str(tmp_path), 'probe', _three_water_model(),
                            [1, 2, 3, 4, 5, 6],
                            [-0.9, 0.45, 0.45, -0.7, 0.35, 0.35])
        lines = (tmp_path / 'probe.data').read_text().splitlines()
        charges = [float(row.split()[3])
                   for row in lines
                   # An `Atoms # full` row is id mol type q x y z: the count
                   # and box lines also start with a digit, and are not rows.
                   if len(row.split()) == 7 and row.split()[0].isdigit()]
        assert charges[:6] == [-0.9, 0.45, 0.45, -0.7, 0.35, 0.35]
        # The MM water is untouched: its topology charges are the classical
        # surface's own, and nothing in the QM/MM path may rewrite them.
        assert charges[6:] == [-0.830, 0.415, 0.415]

    def test_a_missing_id_is_refused(self, tmp_path):
        from ..service.builder.qmmm_energy import _rewrite_qm_charges
        (tmp_path / 'probe.data').write_text(DATA_TEXT)
        with pytest.raises(Exception, match='lacks QM atom ids'):
            _rewrite_qm_charges(str(tmp_path), 'probe', _three_water_model(),
                                [1, 99], [-0.9, 0.9])


class TestGasIsRemovedInsideEachCenter:
    """The in-cell Q-Q removal narrows to one center at a time.

    With one center this is the whole QM set and the numbers are the ones
    the single-center path already reports; the second assertion below is
    the one that says the generalization is not an identity.
    """

    POSITIONS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                 [10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]
    CHARGES = [0.5, -0.5, 0.5, -0.5]
    UNION = [0, 1, 2, 3]

    def test_single_center_equals_the_union_call(self):
        whole = in_cell_qq(self.POSITIONS, self.CHARGES, None, 'real')
        grouped = in_cell_qq_by_centers(self.POSITIONS, self.CHARGES, None,
                                        'real', [self.UNION], self.UNION)
        assert grouped['energy'] == pytest.approx(whole['energy'])
        np.testing.assert_allclose(grouped['forces'], whole['forces'])
        np.testing.assert_allclose(grouped['per_atom'], whole['per_atom'])

    def test_the_inter_center_pair_stays_in_the_classical_surface(self):
        whole = in_cell_qq(self.POSITIONS, self.CHARGES, None, 'real')
        grouped = in_cell_qq_by_centers(self.POSITIONS, self.CHARGES, None,
                                        'real', [[0, 1], [2, 3]], self.UNION)
        # The dropped pairs are the four cross terms: the 1-3 and 2-4 pairs
        # at 10 A carry -k*0.25/10 each, the 1-4 and 2-3 pairs at sqrt(10^2
        # +1) - 10 is not a rounding, so compare against the explicit sum.
        k = 332.06371
        import math
        cross = 0.0
        for left in (0, 1):
            for right in (2, 3):
                dx = self.POSITIONS[right][0] - self.POSITIONS[left][0]
                cross += k * self.CHARGES[left] * self.CHARGES[right] \
                    / math.hypot(dx, 0.0)
        assert grouped['energy'] == pytest.approx(whole['energy'] - cross)
        # Each half keeps only its own pair, and the two halves are the same
        # number because the two pairs are placed and charged identically.
        assert grouped['per_center'][0] == pytest.approx(
            grouped['per_center'][1])
        assert sum(grouped['per_center']) == pytest.approx(
            grouped['energy'])

    def test_a_center_outside_the_qm_set_is_refused(self):
        with pytest.raises(Exception, match='outside the QM set'):
            in_cell_qq_by_centers(self.POSITIONS, self.CHARGES, None, 'real',
                                  [[0, 1], [2, 9]], self.UNION)

    def test_two_centers_sharing_an_atom_are_refused(self):
        with pytest.raises(Exception, match='share atoms'):
            in_cell_qq_by_centers(self.POSITIONS, self.CHARGES, None, 'real',
                                  [[0, 1], [1, 2]], self.UNION)

    def test_energy_assembly_reports_the_per_center_shares(self):
        model = _three_water_model()
        partition = plan_partition(model, [0, 1, 3, 4])
        partition['qm_centers'] = [[0, 1], [3, 4]]
        positions = [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0],
                     [5.0, 5.0, 5.0], [6.0, 5.0, 5.0]]
        q_qm = [-0.83, 0.83, -0.7, 0.7]
        block = assemble_qmmm_energy(
            main_energy=-100.0, main_forces=np.zeros((6, 3)),
            partition=partition, q_qm=q_qm,
            v_mm=[-1.0, -2.0, -3.0, -4.0],
            e_mm=np.zeros((4, 3)),
            positions=positions, lattice=None, units='real')
        parts = in_cell_qq_by_centers(positions, np.asarray(q_qm), None,
                                      'real', [[0, 1], [3, 4]],
                                      partition['qm_index'])
        assert block['u_gas_native'] == pytest.approx(parts['energy'])
        assert block['u_gas_centers_native'] == pytest.approx(
            parts['per_center'])
        assert len(block['u_gas_centers_native']) == 2
