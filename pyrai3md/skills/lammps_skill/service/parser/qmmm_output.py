"""
======================================================================
This is a standalone LAMMPS calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Readers, assemblers and writers for the two QM/MM tasks.

The files are deliberately small and explicit:

* ``.esfp`` carries the qmmm_field potential/field in atomic units for
  Molcas (V Ex Ey Ez, qm_atoms order);
* ``.qm_multipole`` carries the q_qm charges the qmmm_energy task
  received (id q, e);
* ``.qmmm`` carries the native energy decomposition;
* ``.u_gas.force`` and ``.qmmm_el.force`` carry the QM-row force
  corrections in native LAMMPS units (force, ``F = -dU/dR``);
* ``.grad`` carries the reduced classical gradient, all atoms, in native
  units and in gradient sign (``dU/dR``).

No reader here reads ``.esfp`` or ``.qm_multipole`` back from disk: the
in-memory arrays are the contract; the files are audit/handoff output.
"""

import os

import numpy as np

from ..builder.qmmm import in_cell_qq, in_cell_qq_by_centers
from ..utils import (ENERGY_TO_HARTREE, FORCE_TO_HARTREE_PER_BOHR)

# The field column is the force on a +1 probe. If the target Molcas ESPF
# reader defines the field with the opposite sign, flip it here only;
# the native arrays used by the QM/MM force assembly never change.
ESPF_FIELD_SIGN = 1.0


def _read_dump_rows(path: str):
    """Read the numeric rows of a one-frame LAMMPS custom dump."""
    lines = open(path, 'r').read().splitlines()
    header = None
    for line in lines:
        if line.startswith('ITEM: ATOMS'):
            header = line.split()[2:]
            start = lines.index(line) + 1
            break
    if header is None:
        raise ValueError(f'no ITEM: ATOMS header in {path}')
    rows = []
    for line in lines[start:]:
        if not line.strip():
            continue
        tokens = line.split()
        if len(tokens) != len(header):
            break
        rows.append(tokens)
    return header, rows


def read_probe_pe(path: str, expected_ids) -> dict:
    """Read ``id c_QMMM_PEQ`` probe rows into ``{id: value}``."""
    header, rows = _read_dump_rows(path)
    try:
        id_col = header.index('id')
        value_col = next(index for index, name in enumerate(header)
                         if name.startswith('c_'))
    except (ValueError, StopIteration):
        raise ValueError(f'probe dump {path} lacks id/c_* columns: {header}')
    values = {int(row[id_col]): float(row[value_col]) for row in rows}
    expected = {int(value) for value in expected_ids}
    if set(values) != expected:
        raise ValueError(
            f'probe dump {path} ids {sorted(values)} do not match the QM '
            f'atoms {sorted(expected)}')
    return values


def read_probe_force(path: str, expected_ids):
    """Read ``id fx fy fz`` probe rows into an ``(nQ, 3)`` array."""
    header, rows = _read_dump_rows(path)
    try:
        id_col = header.index('id')
        force_col = header.index('fx')
    except ValueError:
        raise ValueError(f'probe dump {path} lacks id/fx columns: {header}')
    by_id = {int(row[id_col]): [float(value) for value in
                                row[force_col:force_col + 3]]
             for row in rows}
    expected = [int(value) for value in expected_ids]
    if set(by_id) != set(expected):
        raise ValueError(
            f'probe dump {path} ids {sorted(by_id)} do not match the QM '
            f'atoms {sorted(expected)}')
    return np.asarray([by_id[atom_id] for atom_id in expected],
                      dtype=np.float64)


def read_field_groups(path: str) -> dict:
    """Read one ``QMMM_FIELD`` group-tally line from a probe file."""
    groups = {}
    for line in open(path, 'r').read().splitlines():
        tokens = line.split()
        if len(tokens) == 6 and tokens[0] == 'QMMM_FIELD':
            groups[tokens[1]] = {
                'energy': float(tokens[2]),
                'force': [float(value) for value in tokens[3:6]],
            }
    if not groups:
        raise ValueError(f'field group tally {path} carries no QMMM_FIELD line')
    return groups


def assemble_qmmm_field(raw: dict, partition: dict, units: str) -> dict:
    """Assemble native V/E from the A/B probe rows.

    ``V_i = 2 (peA_i - peB_i)`` because ``compute pe/atom`` allocates half
    a pair energy to each atom; ``E_i = FA_i - FB_i`` has no such factor.
    The A group tally is the independent check: with q_Q=+1 its energy is
    the sum of the per-Q potentials and its force the sum of the fields.

    The rows belong to the probed center, which under one center is the
    whole QM set: ``qm_atoms``/``qm_atom_ids`` therefore always name the
    atoms these V/E rows are sampled at, never a wider partition. The
    environment the difference isolates is whatever sits outside the probe
    group, so with several centers it carries the MM charges **and** the
    other centers' current 0-order charges, and the group/group check spans
    both -- which is the point of putting them there.
    """
    qm_ids = [int(value) for value in
              (partition.get('probe_ids') or partition['qm_ids'])]
    qm_index = [int(value) for value in
                (partition.get('probe_index') or partition['qm_index'])]
    pe_a = raw['pe_a']
    pe_b = raw['pe_b']
    v = np.asarray([2.0 * (pe_a[atom_id] - pe_b[atom_id])
                    for atom_id in qm_ids], dtype=np.float64)
    e = np.asarray(raw['forces_a'], dtype=np.float64) \
        - np.asarray(raw['forces_b'], dtype=np.float64)
    groups = raw['groups']['A']
    if 'B' not in raw['groups']:
        raise ValueError('qmmm_field A/B group tallies are incomplete')
    v_residual = float(v.sum() - groups['energy'])
    e_residual = float(np.abs(e.sum(axis=0)
                              - np.asarray(groups['force'])).max())
    scale = max(1.0, float(np.abs(v).sum()),
                float(np.abs(e).sum()))
    if abs(v_residual) > 1e-4 * scale:
        # Both operands, not just their difference: a residual of the size of
        # the scale says the two sides describe different states, and which
        # side moved is the first question a reader asks. The deployment's
        # `md`/`mep` legs failed this gate on their second evaluation at
        # 2.65 / 249 and 23.0 / 160, and the operands are what tells a stale
        # extraction from a build that splits kspace differently.
        raise ValueError(
            f'qmmm_field V does not sum to the group/group Q-M energy: '
            f'residual={v_residual} (scale={scale}; sum(V)={float(v.sum())}, '
            f'group/group={float(groups["energy"])}); the two probes did not '
            f'converge to the same reciprocal sum (check the accuracy token '
            f'on the probe kspace_style line - the extent of an ewald sum is '
            f'not settable, so a loose accuracy leaves this residual and no '
            f'mesh setting will remove it)')
    if e_residual > 1e-4 * scale:
        raise ValueError(
            f'qmmm_field E does not sum to the group/group Q-M force: '
            f'residual={e_residual} (sum(E)='
            f'{[float(value) for value in e.sum(axis=0)]}, '
            f'group/group={[float(value) for value in groups["force"]]})')
    return {
        'qm_atoms': qm_index,
        'qm_atom_ids': qm_ids,
        'probe_charge': 1.0,
        'v_mm_native': v,
        'e_mm_native': e,
        'u_gg_native': float(groups['energy']),
        'f_gg_native': np.asarray(groups['force'], dtype=np.float64),
        'sanity': {'v_vs_u_gg': v_residual,
                   'e_vs_f_gg': e_residual},
        'units': {'energy': units, 'force': units},
    }


def assemble_qmmm_energy(main_energy: float, main_forces, partition: dict,
                         q_qm, v_mm, e_mm, positions, lattice,
                         units: str, embedding: bool = True) -> dict:
    """Assemble the QM/MM energy decomposition in native units.

    ``U_gas``/``F_gas`` come from the Python in-cell Q-Q evaluation; the
    QM-MM electrostatic term is the dot product of the received q_qm with
    the Flow-1 potential and field. ``main_energy`` and ``main_forces`` are
    the full-FF U_main values; the engine's own ``.energy``/``.force`` files
    stay exactly those, while the returned block carries the REDUCED
    classical surface (``mm_gradient_native``) that the composite consumes:
    ``E_classic`` is the matching scalar.

    ``embedding=False`` means the QM calculation is isolated: the QM-MM
    electrostatics are NOT in E_QM, so they stay in the classical term
    (``E_classic = U_main - U_gas``) and only ``F_gas`` is removed from the
    QM rows. The V/E arrays are then not required and ``qmmm_el`` and its
    gradient are reported as zero, which makes the QM rows a bit-for-bit
    copy of the main gradient.
    """
    q_qm = np.asarray(q_qm, dtype=np.float64)
    n_qm = len(partition['qm_ids'])
    if q_qm.shape != (n_qm,):
        raise ValueError(f'q_qm shape {q_qm.shape} != ({n_qm},)')
    if np.any(~np.isfinite(q_qm)):
        raise ValueError('q_qm contains a non-finite value')
    if embedding:
        v_mm = np.asarray(v_mm, dtype=np.float64)
        e_mm = np.asarray(e_mm, dtype=np.float64)
        if v_mm.shape != (n_qm,):
            raise ValueError(f'v_mm shape {v_mm.shape} != ({n_qm},)')
        if e_mm.shape != (n_qm, 3):
            raise ValueError(f'e_mm shape {e_mm.shape} != ({n_qm}, 3)')
        if np.any(~np.isfinite(v_mm)) or np.any(~np.isfinite(e_mm)):
            raise ValueError('V/E contains a non-finite value')
        qmmm_el = float(np.dot(q_qm, v_mm))
        f_el = q_qm[:, None] * e_mm
    else:
        qmmm_el = 0.0
        f_el = np.zeros((n_qm, 3), dtype=np.float64)
    centers = [list(group) for group in (partition.get('qm_centers') or [])]
    if len(centers) > 1:
        # Several centers: the in-cell removal narrows to the pairs inside
        # each one, and the inter-center pair stays in the classical surface
        # where it has a force on both sides.
        gas = in_cell_qq_by_centers(positions, q_qm, lattice, units, centers,
                                    partition['qm_index'])
    else:
        gas = in_cell_qq(positions, q_qm, lattice, units)
    e_classic = float(main_energy) - gas['energy'] - qmmm_el
    parts = (float(main_energy) - gas['energy'] - qmmm_el
             - e_classic)
    if abs(parts) > 1e-8 * max(1.0, abs(float(main_energy))):
        raise ValueError(
            f'E_classic residual {parts} exceeds the native energy scale')
    gas_gradient = -np.asarray(gas['forces'], dtype=np.float64)
    el_gradient = -f_el
    main_gradient = -np.asarray(main_forces, dtype=np.float64)
    mm_gradient = _reduce_main_gradient(main_gradient, gas_gradient,
                                        el_gradient, partition)
    return {
        'qm_atoms': partition['qm_index'],
        'qm_atom_ids': list(partition['qm_ids']),
        'q_qm': q_qm,
        'embedding': bool(embedding),
        'u_main_native': float(main_energy),
        'u_gas_native': float(gas['energy']),
        # The per-center shares of the in-cell removal, when the QM set is
        # partitioned: the sum alone cannot say which center's pairs came
        # out of the classical surface.
        'u_gas_centers_native': gas.get('per_center'),
        'qmmm_el_native': qmmm_el,
        'e_classic_native': e_classic,
        'u_gas_gradient_native': gas_gradient,
        'qmmm_el_gradient_native': el_gradient,
        # The classical surface the composite consumes: dU/dR of
        # ``U_main - U_gas - qmmm_el`` on the QM rows, untouched elsewhere.
        'mm_gradient_native': mm_gradient,
        'removed': dict(partition.get('removed') or {}),
        'sanity': {'parts': parts},
        'units': {'energy': units, 'gradient': units},
    }


def _reduce_main_gradient(main_gradient, gas_gradient, el_gradient,
                          partition):
    """Remove the QM-row terms the QM provider owns from the main gradient.

    Every reduction this skill makes on the gradient side happens here, so
    the cold and the resident path cannot drift. A copy is taken first: the
    MM rows are handed through untouched, and the QM rows are the only ones
    arithmetic is applied to.

    The partition check is the load-bearing guard, not an arithmetic one.
    The reduction itself is exact by construction, so no residual can report
    a bug; a duplicated or out-of-range QM row index would silently reduce
    the wrong atom, and that is what is refused here.
    """
    qm_index = [int(index) for index in partition['qm_index']]
    natom = len(main_gradient)
    if len(set(qm_index)) != len(qm_index):
        raise ValueError('qm_index carries a duplicate atom index')
    if any(index < 0 or index >= natom for index in qm_index):
        raise ValueError(
            f'qm_index {qm_index} reaches outside the {natom}-atom system')
    if gas_gradient.shape != (len(qm_index), 3):
        raise ValueError(
            f'U_gas gradient shape {gas_gradient.shape} != '
            f'({len(qm_index)}, 3)')
    if el_gradient.shape != (len(qm_index), 3):
        raise ValueError(
            f'qmmm_el gradient shape {el_gradient.shape} != '
            f'({len(qm_index)}, 3)')
    reduced = np.array(main_gradient, dtype=np.float64, copy=True)
    reduced[qm_index] = (reduced[qm_index] - gas_gradient - el_gradient)
    return reduced


def esfp_values(v_native, e_native, units: str):
    """Convert native V/E to the atomic-unit ``.esfp`` columns."""
    v = np.asarray(v_native, dtype=np.float64) * ENERGY_TO_HARTREE[units]
    e = (np.asarray(e_native, dtype=np.float64)
         * FORCE_TO_HARTREE_PER_BOHR[units] * ESPF_FIELD_SIGN)
    return v, e


def read_esfp(text: str, units: str):
    """Read an ``.esfp`` payload into native V/E.

    Rows are ``V Ex Ey Ez`` or ``id V Ex Ey Ez``; the id column is carried
    so a reader can bind a row to an atom instead of to a position, and it
    is dropped here because this caller wants the arrays in file order.
    """
    v, e = [], []
    for line in text.splitlines():
        content = line.split('#')[0].strip()
        if not content:
            continue
        tokens = content.split()
        if len(tokens) == 5:
            tokens = tokens[1:]
        if len(tokens) != 4:
            raise ValueError(f'bad .esfp row: {line!r}')
        v.append(float(tokens[0]) / ENERGY_TO_HARTREE[units])
        e.append([float(value) / FORCE_TO_HARTREE_PER_BOHR[units]
                  / ESPF_FIELD_SIGN for value in tokens[1:4]])
    return (np.asarray(v, dtype=np.float64),
            np.asarray(e, dtype=np.float64))


def esfp_rows(qm_ids, v_native, e_native, units: str):
    """The ``id V Ex Ey Ez`` text rows of one probe result (atomic units).

    The id is an integer: it is an atom number, and a reader that has to
    match it against another atom number must not depend on how a float
    formatter happened to print it.
    """
    v, e = esfp_values(v_native, e_native, units)
    rows = []
    for atom_id, value, field in zip(qm_ids, v, e):
        rows.append('%d ' % int(atom_id) + ' '.join(
            f'{number:.16e}' for number in (value, *field)))
    return rows


def write_esfp(calcdir: str, title: str, v_native, e_native,
               units: str, qm_ids=None) -> None:
    """Write the Molcas ESPF potential/field file (atomic units).

    The id column is what makes a row addressable: under several centers
    each probe writes only its own center's rows, and the reader has to
    know which atoms they are, not where in a list they landed.
    """
    if qm_ids is None:
        qm_ids = list(range(1, len(np.asarray(v_native)) + 1))
    rows = ['# PyRAI3MD ESPF probe (atomic units): id V Ex Ey Ez']
    rows.extend(esfp_rows(qm_ids, v_native, e_native, units))
    with open(os.path.join(calcdir, f'{title}.esfp'), 'w') as handle:
        handle.write('\n'.join(rows) + '\n')


def write_qm_multipole(calcdir: str, title: str, qm_ids, q_qm) -> None:
    """Record the q_qm charges the energy task received."""
    rows = ['# PyRAI3MD QM 0-order charges (e), qm_atoms order',
            '# id q']
    for atom_id, charge in zip(qm_ids, q_qm):
        rows.append(f'{int(atom_id)} {float(charge):.16e}')
    with open(os.path.join(calcdir, f'{title}.qm_multipole'), 'w') as handle:
        handle.write('\n'.join(rows) + '\n')


def write_qmmm_energy(calcdir: str, title: str, u_main, u_gas,
                      qmmm_el, e_classic) -> None:
    """Write the native energy decomposition."""
    rows = [
        '# PyRAI3MD QM/MM energy decomposition (native LAMMPS units)',
        f'E_classic {float(e_classic):.16e}',
        f'U_main {float(u_main):.16e}',
        f'U_gas {float(u_gas):.16e}',
        f'qmmm_el {float(qmmm_el):.16e}',
    ]
    with open(os.path.join(calcdir, f'{title}.qmmm'), 'w') as handle:
        handle.write('\n'.join(rows) + '\n')


def write_qm_force(calcdir: str, title: str, suffix: str, qm_ids,
                   forces) -> None:
    """Write one QM-row native force file."""
    rows = [f'# PyRAI3MD {suffix} force (native), qm_atoms order',
            '# id fx fy fz']
    for atom_id, force in zip(qm_ids, forces):
        rows.append(f'{int(atom_id)} ' + ' '.join(
            f'{float(value):.16e}' for value in force))
    with open(os.path.join(calcdir, f'{title}.{suffix}.force'), 'w') as handle:
        handle.write('\n'.join(rows) + '\n')


def write_grad(calcdir: str, title: str, mm_gradient, units: str) -> None:
    """Write the reduced classical gradient of one QM/MM evaluation.

    One row per atom, in id order, and the sign convention is stated in the
    header because it is the one thing a reader cannot infer from the
    extension: this file is dU/dR, while ``{title}.force`` beside it is the
    engine's own ``F = -dU/dR`` of the UNREDUCED U_main. The two, plus the
    QM-row auxiliary force files, must satisfy
    ``grad = -force + u_gas.force + qmmm_el.force`` on the QM rows and
    ``grad = -force`` elsewhere; that identity is checked across the files,
    never inside this writer.
    """
    rows = ['# PyRAI3MD reduced classical gradient (dU/dR, native units)',
            f'# units {units}: energy per mol per Angstrom, '
            'one row per atom in id order',
            '# id gx gy gz']
    for atom_id, value in enumerate(np.asarray(mm_gradient, dtype=np.float64),
                                    start=1):
        rows.append(f'{atom_id} ' + ' '.join(
            f'{float(number):.16e}' for number in value))
    with open(os.path.join(calcdir, f'{title}.grad'), 'w') as handle:
        handle.write('\n'.join(rows) + '\n')