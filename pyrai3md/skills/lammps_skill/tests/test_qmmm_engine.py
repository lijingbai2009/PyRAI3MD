"""
Real-engine acceptance tests for the two QM/MM tasks.

A skip means no LAMMPS (or no resident Python module) is available on the
host; it is never a pass. The engine location is the skill's own
``lammps_path`` keyword, read from qualification/pmd_env.yaml, with a
verified PATH ``lmp`` as the developer fallback.

The tests run the 6-atom water fixture: qmmm_field cold/hot, qmmm_energy
cold/hot, and the sequential field -> energy handoff.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from ..backend import LammpsBackend


def _deployment_lammps_path() -> str:
    path = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                        'qualification', 'pmd_env.yaml')
    try:
        import yaml
        with open(path, 'r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
        return str((((data.get('skills') or {}).get('lammps_skill') or {})
                    .get('lammps_path')) or '')
    except Exception:                               # noqa: BLE001
        return ''


def _path_lammps() -> str:
    binary = shutil.which('lmp')
    if not binary:
        return ''
    try:
        result = subprocess.run([binary, '-h'], capture_output=True,
                                text=True, timeout=10)
    except Exception:                               # noqa: BLE001
        return ''
    return binary if 'LAMMPS' in (result.stdout + result.stderr) else ''


def _resolve_lammps() -> str:
    for candidate in (_deployment_lammps_path(), _path_lammps()):
        if candidate and (os.path.isdir(candidate)
                          or os.path.isfile(candidate)):
            return candidate
    return ''


LAMMPS = _resolve_lammps()

pytestmark = pytest.mark.skipif(
    not LAMMPS,
    reason='no LAMMPS found: define skills.lammps_skill.lammps_path in '
           'qualification/pmd_env.yaml, or put a verified lmp on PATH. '
           'This harness runs real engine processes, so a skip is not a '
           'pass')


DATA = """LAMMPS Description

6  atoms
4  bonds
2  angles
0  dihedrals
0  impropers

2  atom types
1  bond types
1  angle types
0  dihedral types
0  improper types

0.0 10.00000000 xlo xhi
0.0 10.00000000 ylo yhi
0.0 10.00000000 zlo zhi

Masses

1 15.999
2 1.008

Atoms

1 1 1 -0.830 1.00000000 1.00000000 1.00000000
2 1 2 0.415 2.00000000 1.00000000 1.00000000
3 1 2 0.415 1.00000000 2.00000000 1.00000000
4 2 1 -0.830 5.00000000 5.00000000 5.00000000
5 2 2 0.415 6.00000000 5.00000000 5.00000000
6 2 2 0.415 5.00000000 6.00000000 5.00000000

Bonds

1 1 1 2
2 1 1 3
3 1 4 5
4 1 4 6

Angles

1 1 2 1 3
2 1 5 4 6
"""

INIT = """units real
atom_style full
boundary f f f
bond_style harmonic
angle_style harmonic
pair_style lj/cut/coul/cut 9.0
pair_modify mix arithmetic
special_bonds amber
"""

SETTINGS = """pair_coeff 1 1 0.102 3.188
pair_coeff 2 2 0.00 0.4000
pair_coeff 1 2 0.00 1.7753
bond_coeff 1 450.0 0.9572
angle_coeff 1 55.0 104.52
"""

COORDS = [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0],
          [5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0]]
ATOMS = ['O', 'H', 'H', 'O', 'H', 'H']
Q_QM = [-0.1831, 0.0916, 0.0916]


def _launch(tmp_path, monkeypatch):
    (tmp_path / 'h2o.data').write_text(DATA)
    (tmp_path / 'h2o.in.init').write_text(INIT)
    (tmp_path / 'h2o.in.settings').write_text(SETTINGS)
    monkeypatch.chdir(tmp_path)


def _config(profile, resident=False, **overrides):
    values = {
        'lammps_project': 'h2o', 'profile': profile, 'method': 'ff',
        'atoms': list(ATOMS), 'coordinates': [list(row) for row in COORDS],
        'charges': 0, 'multiplicity': 1, 'natom': 6,
        'qm_atoms': [0, 1, 2], 'lammps_units': 'real',
        'lammps_path': LAMMPS, 'lammps_nproc': 1, 'lammps_thread': 1,
        'execution_mode': 'local', 'python_enabled': resident,
        'python_path': sys.executable, 'keep_tmp': 1, 'pbc': [0, 0, 0],
    }
    values.update(overrides)
    return values


def _run_field(tmp_path, monkeypatch, resident=False, suffix='field'):
    _launch(tmp_path, monkeypatch)
    config = _config('qmmm_field', resident=resident)
    backend = LammpsBackend(config, calcdir=str(tmp_path / suffix))
    try:
        result = backend.calculate(config['atoms'], config['coordinates'])
    finally:
        backend.close()
    return result, tmp_path / suffix


def _run_energy(tmp_path, monkeypatch, field, resident=False,
                suffix='energy'):
    _launch(tmp_path, monkeypatch)
    config = _config('qmmm_energy', resident=resident, q_qm=Q_QM)
    backend = LammpsBackend(config, calcdir=str(tmp_path / suffix))
    try:
        result = backend.calculate(
            config['atoms'], config['coordinates'],
            qmmm_inputs={
                'q_qm': Q_QM,
                'v_mm': field['qmmm_field']['v_mm_native'],
                'e_mm': field['qmmm_field']['e_mm_native'],
            })
    finally:
        backend.close()
    return result, tmp_path / suffix


def test_field_cold_emits_esfp_and_group_check(tmp_path, monkeypatch):
    result, calc = _run_field(tmp_path, monkeypatch)
    block = result['qmmm_field']
    assert np.asarray(block['v_mm_native']).shape == (3,)
    assert np.asarray(block['e_mm_native']).shape == (3, 3)
    assert float(np.sum(block['v_mm_native'])) == pytest.approx(
        block['u_gg_native'], abs=1e-4)
    assert (calc / 'h2o_qmmm_field.esfp').exists()
    assert not (calc / 'h2o_qmmm_field.energy').exists()
    assert result['energy'] == [0.0]


def test_energy_cold_writes_the_full_contract(tmp_path, monkeypatch):
    field, _ = _run_field(tmp_path, monkeypatch)
    result, calc = _run_energy(tmp_path, monkeypatch, field)
    block = result['qmmm']
    assert block['u_main_native'] == pytest.approx(
        result['energy'][0], rel=0, abs=1e-12)
    assert block['e_classic_native'] == pytest.approx(
        block['u_main_native'] - block['u_gas_native']
        - block['qmmm_el_native'], abs=1e-8)
    for name in ('h2o_qmmm_energy.energy', 'h2o_qmmm_energy.force',
                 'h2o_qmmm_energy.qm_multipole', 'h2o_qmmm_energy.qmmm',
                 'h2o_qmmm_energy.u_gas.force',
                 'h2o_qmmm_energy.qmmm_el.force',
                 'h2o_qmmm_energy.grad'):
        assert (calc / name).exists(), name


def test_field_hot_matches_cold(tmp_path, monkeypatch):
    cold, _ = _run_field(tmp_path, monkeypatch, suffix='cold')
    hot, calc = _run_field(tmp_path, monkeypatch, resident=True,
                           suffix='hot')
    for key in ('v_mm_native', 'e_mm_native'):
        assert np.allclose(hot['qmmm_field'][key],
                           cold['qmmm_field'][key], atol=1e-4)
    assert (calc / 'h2o_qmmm_field.esfp').exists()


def test_energy_hot_matches_cold(tmp_path, monkeypatch):
    field, _ = _run_field(tmp_path, monkeypatch, suffix='field_cold')
    cold, _ = _run_energy(tmp_path, monkeypatch, field, suffix='energy_cold')
    hot, calc = _run_energy(tmp_path, monkeypatch, field, resident=True,
                            suffix='energy_hot')
    assert hot['qmmm']['u_main_native'] == pytest.approx(
        cold['qmmm']['u_main_native'], abs=1e-8)
    assert hot['qmmm']['u_gas_native'] == pytest.approx(
        cold['qmmm']['u_gas_native'], abs=1e-8)
    assert hot['qmmm']['e_classic_native'] == pytest.approx(
        cold['qmmm']['e_classic_native'], abs=1e-7)
    assert hot['qmmm']['qmmm_el_native'] == pytest.approx(
        cold['qmmm']['qmmm_el_native'], abs=1e-4)
    for name in ('h2o_qmmm_energy.energy', 'h2o_qmmm_energy.force',
                 'h2o_qmmm_energy.qmmm'):
        assert (calc / name).exists(), name


def _table_rows(path):
    """The numeric atom rows of one force/gradient file, in atom order.

    Three shapes arrive here: a cold LAMMPS dump (`ITEM: ATOMS fx fy fz`, with
    a box-bounds block BEFORE that header which is not atom data), the
    resident dump (`ITEM: ATOMS id fx fy fz`), and the skill's own writers
    (comment lines then `id x y z`).
    """
    rows, dump, collecting, has_id = [], False, False, False
    for line in open(path):
        tokens = line.split()
        if not tokens or tokens[0].startswith('#'):
            continue
        if tokens[0] == 'ITEM:':
            dump = True
            # 'ITEM: NUMBER OF ATOMS' also contains the word.
            if len(tokens) > 1 and tokens[1] == 'ATOMS':
                columns = tokens[2:]
                has_id = bool(columns) and columns[0] == 'id'
                collecting = True
            continue
        if dump and not collecting:
            continue
        numbers = tokens[1:] if (has_id or len(tokens) == 4) else tokens
        rows.append([float(value) for value in numbers])
    return rows


def test_grad_file_holds_the_reduced_surface(tmp_path, monkeypatch):
    field, _ = _run_field(tmp_path, monkeypatch, suffix='field_grad')
    result, calc = _run_energy(tmp_path, monkeypatch, field,
                               suffix='energy_grad')
    block = result['qmmm']
    grad = np.array(_table_rows(str(calc / 'h2o_qmmm_energy.grad')))
    np.testing.assert_allclose(grad,
                               np.asarray(result['gradient'], dtype=float),
                               rtol=0.0, atol=1e-12)
    # The engine's own file stays the UNREDUCED U_main force. Adding it back
    # to grad must leave exactly the two auxiliaries on the QM rows and
    # nothing anywhere else: what the reduction removed, in file order.
    main = np.array(_table_rows(str(calc / 'h2o_qmmm_energy.force')))
    qm = [int(index) for index in block['qm_atoms']]
    removed = (-np.asarray(block['u_gas_gradient_native'], dtype=float)
               - np.asarray(block['qmmm_el_gradient_native'], dtype=float))
    # %g in the cold dump costs six significant digits; the file is the
    # least precise thing in this comparison, so the line follows it.
    limit = 1e-5 * max(1.0, float(np.abs(main).max()))
    residual = grad + main
    for row, index in enumerate(qm):
        np.testing.assert_allclose(residual[index], removed[row], atol=limit)
    mm = [index for index in range(main.shape[0]) if index not in qm]
    np.testing.assert_allclose(residual[mm], np.zeros((len(mm), 3)),
                               atol=limit)
    # Non-identity witness: on this fixture the reduction is not nothing.
    assert float(np.abs(removed).max()) > limit

# ---------------------------------------------------------------------------
# Two centers, same engine.
#
# A multi-center run is not the single-center one executed twice: the probe of
# one center has to see the OTHER center's 0-order charges as its environment,
# and the classical surface has to remove the in-cell Q-Q pairs inside each
# center while leaving the inter-center pair in place.
#
# The fixture is three waters in a 12 A non-periodic box: atoms 1-3 and 4-6
# are the two QM centers, atoms 7-9 are the classical region. The third water
# carries zero charge on purpose. A probe measures the potential of everything
# outside itself, so with a charged MM water the two centers would each see
# their own MM pair as well and the reciprocity reading below would compare
# two different sums. With the classical water inert, q_A . V(A) is exactly
# the A-B inter-center pair and q_B . V(B) is the same pair from the other
# side.

MC_DATA = """LAMMPS Description

9  atoms
6  bonds
3  angles
0  dihedrals
0  impropers

2  atom types
1  bond types
1  angle types
0  dihedral types
0  improper types

0.0 12.00000000 xlo xhi
0.0 12.00000000 ylo yhi
0.0 12.00000000 zlo zhi

Masses

1 15.999
2 1.008

Atoms

1 1 1 -0.830 1.00000000 1.00000000 1.00000000
2 1 2 0.415 2.00000000 1.00000000 1.00000000
3 1 2 0.415 1.00000000 2.00000000 1.00000000
4 2 1 -0.830 5.00000000 5.00000000 5.00000000
5 2 2 0.415 6.00000000 5.00000000 5.00000000
6 2 2 0.415 5.00000000 6.00000000 5.00000000
7 3 1 0.000 9.00000000 1.00000000 9.00000000
8 3 2 0.000 10.00000000 1.00000000 9.00000000
9 3 2 0.000 9.00000000 2.00000000 9.00000000

Bonds

1 1 1 2
2 1 1 3
3 1 4 5
4 1 4 6
5 1 7 8
6 1 7 9

Angles

1 1 2 1 3
2 1 5 4 6
3 1 8 7 9
"""

# The only change from the cluster fixture's declarations is the box: the
# third water has to fit beside the two centers. The pair cutoff stays at the
# 9.0 A the 6-atom fixture uses, because the two centers are 6.93 A apart and
# a smaller cutoff makes the inter-center Coulomb invisible - which is the
# one thing these tests are reading.
MC_INIT = INIT.replace('0.0 10.00000000', '0.0 12.00000000')

MC_COORDS = [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0],
             [5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0],
             [9.0, 1.0, 9.0], [10.0, 1.0, 9.0], [9.0, 2.0, 9.0]]
MC_ATOMS = ['O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H']

CENTER_A = [0, 1, 2]
CENTER_B = [3, 4, 5]
UNION = CENTER_A + CENTER_B
CHARGE_A = [-0.1832, 0.0916, 0.0916]
CHARGE_B = [0.1400, -0.0600, -0.0800]


def _mc_launch(tmp_path, monkeypatch):
    (tmp_path / 'h2o.data').write_text(MC_DATA)
    (tmp_path / 'h2o.in.init').write_text(MC_INIT)
    (tmp_path / 'h2o.in.settings').write_text(SETTINGS)
    monkeypatch.chdir(tmp_path)


def _mc_config(profile, **overrides):
    values = {
        'lammps_project': 'h2o', 'profile': profile, 'method': 'ff',
        'atoms': list(MC_ATOMS),
        'coordinates': [list(row) for row in MC_COORDS],
        'charges': 0, 'multiplicity': 1, 'natom': 9,
        'lammps_units': 'real', 'lammps_path': LAMMPS,
        'lammps_nproc': 1, 'lammps_thread': 1,
        'execution_mode': 'local', 'python_enabled': False, 'keep_tmp': 1,
        'pbc': [0, 0, 0],
    }
    values.update(overrides)
    return values


def _mc_run(tmp_path, monkeypatch, suffix, charges, **overrides):
    """One cold LAMMPS run of a multi-center duty, straight from the config.

    `charges` is the array the QM set is fed with: the union for the two
    two-center cases, one center's own rows for a standalone one-center
    reading. It goes into the config (what the energy profile pins) and into
    the call payload (what the framework actually hands a stage), because a
    cold run must answer to the same numbers either way.
    """
    _mc_launch(tmp_path, monkeypatch)
    config = _mc_config(**overrides)
    backend = LammpsBackend(config, calcdir=str(tmp_path / suffix))
    try:
        result = backend.calculate(
            config['atoms'], config['coordinates'],
            qmmm_inputs={'q_qm': [float(value) for value in charges]})
    finally:
        backend.close()
    return result


def test_the_two_probes_report_the_same_pair_energy(tmp_path, monkeypatch):
    union_charges = list(CHARGE_A) + list(CHARGE_B)
    a = _mc_run(tmp_path, monkeypatch, 'mc_field_a', union_charges,
                profile='qmmm_field',
                qm_atoms=list(UNION), qm_probe=list(CENTER_A))['qmmm_field']
    b = _mc_run(tmp_path, monkeypatch, 'mc_field_b', union_charges,
                profile='qmmm_field',
                qm_atoms=list(UNION), qm_probe=list(CENTER_B))['qmmm_field']
    # Each probe answers for its own center only, by id.
    assert [int(index) for index in a['qm_atom_ids']] == [1, 2, 3]
    assert [int(index) for index in b['qm_atom_ids']] == [4, 5, 6]
    v_a = np.asarray(a['v_mm_native'], dtype=float)
    v_b = np.asarray(b['v_mm_native'], dtype=float)
    assert v_a.shape == (3,) and v_b.shape == (3,)
    el_ab = float(np.dot(CHARGE_A, v_a))
    el_ba = float(np.dot(CHARGE_B, v_b))
    # q_A . V(B on A) and q_B . V(A on B) are the same inter-center Coulomb
    # pair seen from either side. If the environment charges never reached a
    # probe, one of the two would be zero and this would fail by name.
    assert el_ab == pytest.approx(el_ba, rel=1e-4, abs=1e-6)
    # Non-identity witness: the pair is not nothing on this geometry.
    assert abs(el_ab) > 1e-4


def test_a_probe_narrowed_to_one_center_is_not_the_union_probe(tmp_path,
                                                               monkeypatch):
    # The probe's own center sits in both A/B legs and cancels, so narrowing
    # the probe to one center has to CHANGE that center's rows: measured over
    # the union partition, the same environment includes the intra-center
    # charges the narrowed probe drops.
    charges = list(CHARGE_A) + list(CHARGE_B)
    union = _mc_run(tmp_path, monkeypatch, 'mc_field_union', charges,
                    profile='qmmm_field', qm_atoms=list(UNION))
    narrowed = _mc_run(tmp_path, monkeypatch, 'mc_field_narrow', charges,
                       profile='qmmm_field',
                       qm_atoms=list(UNION), qm_probe=list(CENTER_A))
    union_rows = np.asarray(union['qmmm_field']['v_mm_native'], dtype=float)
    probe_rows = np.asarray(narrowed['qmmm_field']['v_mm_native'], dtype=float)
    assert union_rows.shape == (6,) and probe_rows.shape == (3,)
    delta = probe_rows - union_rows[:3]
    assert float(np.abs(delta).max()) > 1e-4


def test_the_in_cell_removal_is_split_by_center(tmp_path, monkeypatch):
    charges = list(CHARGE_A) + list(CHARGE_B)
    two = _mc_run(tmp_path, monkeypatch, 'mc_energy_two', charges,
                  profile='qmmm_energy',
                  qm_atoms=list(UNION), qm_centers=[CENTER_A, CENTER_B],
                  q_qm=charges)['qmmm']
    per_center = [float(value) for value in two['u_gas_centers_native']]
    assert sum(per_center) == pytest.approx(float(two['u_gas_native']),
                                            rel=1e-9, abs=1e-9)
    # Each center's number must be the standalone one-center reading of that
    # same water, not a share of a total the fixture happens to split.
    one_a = _mc_run(tmp_path, monkeypatch, 'mc_energy_a', list(CHARGE_A),
                    profile='qmmm_energy',
                    qm_atoms=list(CENTER_A), q_qm=list(CHARGE_A))['qmmm']
    one_b = _mc_run(tmp_path, monkeypatch, 'mc_energy_b', list(CHARGE_B),
                    profile='qmmm_energy',
                    qm_atoms=list(CENTER_B), q_qm=list(CHARGE_B))['qmmm']
    assert per_center[0] == pytest.approx(float(one_a['u_gas_native']),
                                          rel=1e-9, abs=1e-9)
    assert per_center[1] == pytest.approx(float(one_b['u_gas_native']),
                                          rel=1e-9, abs=1e-9)


CHARGE_A_CHARGED = [value + 1.0 / 3.0 for value in CHARGE_A]   # net +1


def test_a_charged_center_keeps_its_own_charge_set(tmp_path, monkeypatch):
    """One center at +1, the other neutral: the decomposition still splits.

    The per-center removal is taken with the charges that center carries, so
    the charged center's number has to equal its own standalone reading and
    nothing of it may leak into the neutral center's.
    """
    charges = list(CHARGE_A_CHARGED) + list(CHARGE_B)
    two = _mc_run(tmp_path, monkeypatch, 'chg_two', charges,
                  profile='qmmm_energy',
                  qm_atoms=list(UNION), qm_centers=[CENTER_A, CENTER_B],
                  q_qm=charges)['qmmm']
    per_center = [float(value) for value in two['u_gas_centers_native']]
    assert sum(per_center) == pytest.approx(float(two['u_gas_native']),
                                            rel=1e-9, abs=1e-9)
    one_a = _mc_run(tmp_path, monkeypatch, 'chg_one_a',
                    list(CHARGE_A_CHARGED), profile='qmmm_energy',
                    qm_atoms=list(CENTER_A),
                    q_qm=list(CHARGE_A_CHARGED))['qmmm']
    one_b = _mc_run(tmp_path, monkeypatch, 'chg_one_b', list(CHARGE_B),
                    profile='qmmm_energy',
                    qm_atoms=list(CENTER_B), q_qm=list(CHARGE_B))['qmmm']
    assert per_center[0] == pytest.approx(float(one_a['u_gas_native']),
                                          rel=1e-9, abs=1e-9)
    assert per_center[1] == pytest.approx(float(one_b['u_gas_native']),
                                          rel=1e-9, abs=1e-9)
    residual = (float(two['u_main_native']) - float(two['u_gas_native'])
                - float(two['qmmm_el_native'])
                - float(two['e_classic_native']))
    assert abs(residual) < 1e-4 * max(1.0, abs(float(two['u_main_native'])))
    # The neutral center's probe now has a charged environment: its own A/B
    # identity still closes.
    probe_b = _mc_run(tmp_path, monkeypatch, 'chg_field_b', charges,
                      profile='qmmm_field', qm_atoms=list(UNION),
                      qm_probe=list(CENTER_B))['qmmm_field']
    v = np.asarray(probe_b['v_mm_native'], dtype=float)
    assert float(np.sum(v)) == pytest.approx(
        float(probe_b['u_gg_native']), abs=1e-4 * max(
            1.0, float(np.abs(v).sum())))


def test_the_resident_probe_carries_the_other_center(tmp_path, monkeypatch):
    """The same reading through a resident session, not only a cold run.

    `python_enabled: true` keeps one LAMMPS process per duty and moves the
    frame over a socket, so the environment charges have to reach the deck by
    a different route (per-atom `set`) than the cold path's rewritten data
    file. The witness is the fixture itself: the classical water carries zero
    charge, so any potential a probe reports at its center can only come from
    the other center's charges.
    """
    charges = list(CHARGE_A) + list(CHARGE_B)
    resident = {'python_enabled': True, 'python_path': sys.executable}
    a = _mc_run(tmp_path, monkeypatch, 'mc_res_a', charges,
                profile='qmmm_field', qm_atoms=list(UNION),
                qm_probe=list(CENTER_A), **resident)['qmmm_field']
    b = _mc_run(tmp_path, monkeypatch, 'mc_res_b', charges,
                profile='qmmm_field', qm_atoms=list(UNION),
                qm_probe=list(CENTER_B), **resident)['qmmm_field']
    v_a = np.asarray(a['v_mm_native'], dtype=float)
    v_b = np.asarray(b['v_mm_native'], dtype=float)
    assert v_a.shape == (3,) and v_b.shape == (3,)
    # Nonzero, and nonzero by the only possible source.
    assert float(np.abs(v_a).max()) > 1e-4
    el_ab = float(np.dot(CHARGE_A, v_a))
    el_ba = float(np.dot(CHARGE_B, v_b))
    assert el_ab == pytest.approx(el_ba, rel=1e-4, abs=1e-6)


Q_QM_CHARGED = [value + 1.0 / 3.0 for value in Q_QM]   # net +1


def _charged_mm_data(extra):
    """The same fixture with one MM atom's charge raised by `extra`."""
    lines = DATA.splitlines(True)
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 7 and parts[0] == '5':
            parts[3] = '%.4f' % (float(parts[3]) + extra)
            out.append(' '.join(parts) + '\n')
        else:
            out.append(line)
    return ''.join(out)


def test_a_charged_mm_cell_keeps_the_ab_identity(tmp_path, monkeypatch):
    """A net charge on the environment does not break the probe.

    The rule used to refuse a charged partition outright, on the theory that
    the two probe legs' Ewald backgrounds would not cancel. Measured here and
    on the 60-atom box: a cell charged by +0.3 (here) and -0.4 (there) leaves
    the A/B identity at the same 1e-5 level as the neutral cell, because a
    uniform background is the same in both legs of one evaluation, and a
    surviving offset would cancel between `E_QM` and `qmmm_el` anyway.
    """
    (tmp_path / 'h2o.data').write_text(_charged_mm_data(0.3))
    (tmp_path / 'h2o.in.init').write_text(INIT)
    (tmp_path / 'h2o.in.settings').write_text(SETTINGS)
    monkeypatch.chdir(tmp_path)
    config = _config('qmmm_field')
    backend = LammpsBackend(config, calcdir=str(tmp_path / 'charged'))
    try:
        result = backend.calculate(config['atoms'], config['coordinates'])
    finally:
        backend.close()
    block = result['qmmm_field']
    v = np.asarray(block['v_mm_native'], dtype=float)
    assert float(np.sum(v)) == pytest.approx(
        float(block['u_gg_native']), abs=1e-4 * max(
            1.0, float(np.abs(v).sum())))


def test_a_charged_qm_region_runs_the_energy_flow(tmp_path, monkeypatch):
    """`charges: +1` is a legal QM region, and the decomposition still closes."""
    field, _ = _run_field(tmp_path, monkeypatch)
    _launch(tmp_path, monkeypatch)
    config = _config('qmmm_energy', charges=1, q_qm=Q_QM_CHARGED,
                     embedding=True)
    backend = LammpsBackend(config, calcdir=str(tmp_path / 'charged_energy'))
    try:
        result = backend.calculate(
            config['atoms'], config['coordinates'],
            qmmm_inputs={
                'q_qm': Q_QM_CHARGED,
                'v_mm': field['qmmm_field']['v_mm_native'],
                'e_mm': field['qmmm_field']['e_mm_native'],
            })
    finally:
        backend.close()
    block = result['qmmm']
    residual = (float(block['u_main_native']) - float(block['u_gas_native'])
                - float(block['qmmm_el_native'])
                - float(block['e_classic_native']))
    scale = max(1.0, abs(float(block['u_main_native'])))
    assert abs(residual) < 1e-4 * scale
    # The charged set reaches the sum, so qmmm_el is built from +1, not 0.
    assert abs(float(block['qmmm_el_native'])) > 1e-6


if __name__ == '__main__':
    unittest.main()
