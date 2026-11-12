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

QM/MM topology handling: the partition, the QM region's bonded rows and
the per-partition net charges.

Two Q-Q terms must never be confused, and this module exists to keep
them apart:

* the **in-cell** Q-Q interaction (bonded, LJ and the bare Coulomb sum).
  A QM calculation already contains it, so the classical side must not
  add it again.
* the interaction of the QM region with its **own periodic images**.
  That term is long-range, the QM calculation does not contain it, and
  it stays in the LAMMPS result.

Only the first is removed here (as bonded rows; the bare non-bonded
part is subtracted numerically from the whole-cell result, see
service/parser/qmmm_output.py). Nothing in this
module excludes or scales the image term.

The module is pure topology: it reads and rewrites the parsed model,
builds no script and writes no file.
"""

from .exceptions import FFModelUnsupported

# Entity sections and the column holding their type id. The member atom
# ids follow it, exactly as TYPE_SECTIONS in ff_model reads them.
_ENTITY_SECTIONS = (('Bonds', 'bonds', 1), ('Angles', 'angles', 1),
                    ('Dihedrals', 'dihedrals', 1),
                    ('Impropers', 'impropers', 1))

# atom_style -> index of the charge column. The styles listed as None
# carry no charge at all, which a QM/MM run cannot use.
_CHARGE_COLUMN = {'charge': 2, 'full': 3, 'atomic': None,
                  'molecular': None, 'bond': None}

# A QM calculation's charges are often quoted to four decimals (the
# Molcas population analysis in this repository is), so their sum is
# neutral to that precision, not to machine precision. A tolerance below
# the quoted resolution would refuse a real charge set for its rounding.
_CHARGE_TOLERANCE = 1e-3


def _atom_style(model: dict):
    """Return the atom_style named by the Atoms header comment, or ''."""
    header = model['sections'].get('Atoms_header', '') or ''
    if '#' not in header:
        return ''
    tokens = header.split('#', 1)[1].strip().split()
    return tokens[0].lower() if tokens else ''


def _is_int(token) -> bool:
    try:
        int(str(token).strip())
        return True
    except (TypeError, ValueError):
        return False


def _charge_index(model: dict, tokens: list) -> int:
    """Locate the per-atom charge column of an Atoms row.

    With an atom_style in the header the column is exact. Without one the
    leading column count is inferred from the token count, and a row that
    could be either a charged or an uncharged style is refused rather
    than guessed: reading the molecule id as a charge would silently
    change every electrostatic interaction.

    Parameters
    ----------
    model : dict
        Model carrying the parsed data sections.
    tokens : list of str
        One Atoms row, split.

    Returns
    -------
    int
        Index of the charge column.

    Raises
    ------
    FFModelUnsupported
        When the style has no charge column or cannot be resolved.
    """
    style = _atom_style(model)
    if style:
        if style not in _CHARGE_COLUMN:
            raise FFModelUnsupported(
                f"qmmm needs a per-atom charge column and atom_style "
                f"{style!r} has none")
        index = _CHARGE_COLUMN[style]
        if index is None:
            raise FFModelUnsupported(
                f"qmmm needs a per-atom charge column and atom_style "
                f"{style!r} has none")
        return index

    # No style comment: count trailing image flags (they come in threes)
    # and take the three columns before them as the coordinates.
    count = len(tokens)
    trailing = 0
    while trailing < count and _is_int(tokens[count - 1 - trailing]):
        trailing += 1
    if trailing not in (0, 3):
        raise FFModelUnsupported(
            f'qmmm cannot locate the Atoms columns: {trailing} trailing '
            f'integer columns. Add an "# <atom_style>" comment to the '
            f'Atoms section header.')
    coord_start = count - trailing - 3
    if coord_start == 4:
        # id mol type q x y z
        return 3
    raise FFModelUnsupported(
        f'qmmm cannot resolve the charge column from {count} Atoms '
        f'columns without an atom_style comment; write "Atoms # full" '
        f'(or the style actually used) in the data file')


def _atoms_rows(model: dict):
    """Return (atom id, tokens, charge) for every Atoms row."""
    rows = model['sections'].get('Atoms') or []
    parsed = []
    for raw in rows:
        content = raw.partition('#')[0]
        tokens = content.split()
        if not tokens:
            continue
        if not _is_int(tokens[0]):
            raise FFModelUnsupported(
                f'Atoms row does not start with an atom id: {raw!r}')
        index = _charge_index(model, tokens)
        try:
            charge = float(tokens[index])
        except (IndexError, ValueError):
            raise FFModelUnsupported(
                f'Atoms row has no readable charge in column {index}: '
                f'{raw!r}')
        parsed.append((int(tokens[0]), tokens, charge))
    return parsed


def plan_partition(model: dict, qm_atoms, probe_atoms=None) -> dict:
    """Resolve the QM/MM partition and the two net charges.

    ``qm_atoms`` is the whole QM set: every atom whose 0-order charge the
    classical surface carries. ``probe_atoms`` narrows the field task to
    the one center whose potential is wanted, and it defaults to the whole
    QM set, which is what every existing single-center run means. The two
    sets are not the same thing under several centers: the atoms that are
    QM but not probed are *environment* charges, on the same side of the
    A/B difference as the MM atoms, so the probe sees them.

    Parameters
    ----------
    model : dict
        Model whose data sections were just parsed.
    qm_atoms : sequence of int
        0-based framework atom indices of the whole QM set.
    probe_atoms : sequence of int or None
        0-based framework atom indices of the center being probed; None or
        empty means the whole QM set.

    Returns
    -------
    dict
        ``qm_index``/``qm_ids`` (0-based sorted / 1-based LAMMPS ids of the
        QM set), ``probe_index``/``probe_ids`` (the measured center),
        ``env_ids`` (QM ids that are not probed), ``mm_ids`` (everything
        outside the QM set), ``qm_charges`` and ``mm_charges``.

    Raises
    ------
    FFModelUnsupported
        When the Atoms ids are not 1..natom, an index is outside the
        system, or the probe is not a subset of the QM set.
    """
    rows = _atoms_rows(model)
    ids = [row[0] for row in rows]
    natom = len(ids)
    if sorted(ids) != list(range(1, natom + 1)):
        raise FFModelUnsupported(
            f'qmmm needs Atoms ids 1..{natom} one-to-one (the framework atom '
            f'order); the data file has {len(set(ids))} distinct ids in '
            f'{natom} rows')

    qm_index = sorted({int(index) for index in qm_atoms})
    outside = [index for index in qm_index
               if index < 0 or index >= natom]
    if outside:
        raise FFModelUnsupported(
            f'qm_atoms index outside [0, {natom}): {outside}')

    charge_of = {atom_id: charge for atom_id, _tokens, charge in rows}
    qm_ids = [index + 1 for index in qm_index]
    qm_set = set(qm_ids)
    mm_ids = [atom_id for atom_id in sorted(ids) if atom_id not in qm_set]

    probe_index = sorted({int(index) for index in (probe_atoms or [])}) \
        or list(qm_index)
    probe_ids = [index + 1 for index in probe_index]
    probe_outside = [atom_id for atom_id in probe_ids if atom_id not in qm_set]
    if probe_outside:
        raise FFModelUnsupported(
            f'qm_probe atoms {probe_outside} are outside qm_atoms: the probe '
            f'measures a center, and a center must be part of the QM set')
    probe_set = set(probe_ids)
    env_ids = [atom_id for atom_id in qm_ids if atom_id not in probe_set]

    qm_charges = [charge_of[atom_id] for atom_id in qm_ids]
    mm_charges = [charge_of[atom_id] for atom_id in mm_ids]

    return {
        'qm_index': qm_index,
        'qm_ids': qm_ids,
        'probe_index': probe_index,
        'probe_ids': probe_ids,
        'env_ids': env_ids,
        'mm_ids': mm_ids,
        'qm_charges': qm_charges,
        'mm_charges': mm_charges,
    }


def in_cell_qq_by_centers(positions, charges, lattice, units: str,
                          centers, qm_index) -> dict:
    """The in-cell Q-Q term, summed over the pairs **inside** one center.

    With a single QM region this is exactly :func:`in_cell_qq`: the QM
    calculation already contains every pair among its own atoms, so the
    classical surface must not add them again. With several centers the
    statement narrows to each center, and the pairs *between* centers are
    deliberately left in the classical surface: they are the only place
    where the inter-center Coulomb term has a force on both centers' atoms,
    because a QM provider answers the gradient of its own region alone.
    Removing them here would leave each center pulled by the other with no
    reaction anywhere in the result.

    Positions and charges are given in ``qm_index`` order (the whole QM set
    in global atom order); the returned arrays keep that order, so the
    caller reduces the QM rows of the main gradient in one pass.

    Parameters
    ----------
    positions, charges, lattice, units
        As for :func:`in_cell_qq`.
    centers : sequence of sequence of int
        Per-center global (0-based framework) atom indices.
    qm_index : sequence of int
        The whole QM set, 0-based, ascending.

    Returns
    -------
    dict
        Same shape as :func:`in_cell_qq`: ``energy``, ``per_atom`` and
        ``forces`` in QM-set order.

    Raises
    ------
    FFModelUnsupported
        When a center names an atom outside the QM set, or two centers
        share an atom.
    """
    slot_of = {int(index): slot for slot, index in enumerate(qm_index)}
    energy = 0.0
    per_center = []
    per_atom = [0.0] * len(qm_index)
    forces = [[0.0, 0.0, 0.0] for _ in qm_index]
    claimed = set()
    for group in centers:
        indices = [int(index) for index in group]
        outside = [index for index in indices if index not in slot_of]
        if outside:
            raise FFModelUnsupported(
                f'qm_centers names atoms {outside} outside the QM set '
                f'{list(qm_index)}')
        shared = claimed & set(indices)
        if shared:
            raise FFModelUnsupported(
                f'qm_centers share atoms {sorted(shared)}; a QM atom belongs '
                f'to one center, or it would carry two 0-order charges')
        claimed.update(indices)
        if len(indices) < 2:
            per_center.append(0.0)
            continue
        slots = [slot_of[index] for index in indices]
        part = in_cell_qq([positions[slot] for slot in slots],
                          [charges[slot] for slot in slots],
                          lattice, units)
        energy += float(part['energy'])
        per_center.append(float(part['energy']))
        for slot, value, force in zip(slots, part['per_atom'],
                                      part['forces']):
            per_atom[slot] += float(value)
            forces[slot] = [forces[slot][axis] + float(force[axis])
                            for axis in range(3)]
    return {'energy': energy, 'per_atom': per_atom, 'forces': forces,
            'per_center': per_center}


def validate_partition_charges(partition: dict, q_qm, mode: str) -> None:
    """Check the shape of a charge partition; any net charge is admissible.

    Neither side has to be neutral. The QM charge is declared per center in
    the input (`molecule.charges`, alongside `ci`/`coupling`) and the MM
    charges come from the user's topology; the framework balances each
    center's fitted 0-order charges to that declared value
    (`Calculator._balance_qmmm_charges`), so the gas removal of a center is
    taken with the charges that center actually carries.

    A net charge in the cell is *not* a problem for the two flows, and the
    earlier rule that refused one was over-cautious in two directions:

    * ``qmmm_field`` takes V from the A/B difference of two runs that differ
      only in the MM charges. Whatever a uniform background does to the
      absolute level of the potential, it is the *same* level in both legs of
      one evaluation and cancels in the difference - measured on the 60-atom
      box with the MM cell deliberately charged by -0.4 e: residual of the
      A/B identity 2.0e-5 against a tally of -8.156 kcal/mol, the same order
      as the neutral case. A uniform offset that did survive would still
      cancel in the composite total, because `E_QM` is linear in V with the
      QM net charge and `qmmm_el = sum q V` subtracts the same charge with
      the same sign.
    * ``qmmm_energy`` evaluates the full force field on the supplied charges;
      the Ewald background is LAMMPS's own convention and no term of the
      decomposition needs a neutral cell.

    What is still checked here is shape: the QM charge vector has to name
    every QM atom, because the unprobed centers are environment charges and a
    short vector would silently leave some of them at their topology values.

    Parameters
    ----------
    partition : dict
        Result of :func:`plan_partition`.
    q_qm : sequence of float
        QM charges used by the flow (the data charges for ``field``, the
        fitted 0-order charges for ``energy``).
    mode : str
        ``'field'`` or ``'energy'``.

    Raises
    ------
    FFModelUnsupported
        When the QM charge vector has the wrong length.
    """
    if mode not in ('field', 'energy'):
        raise ValueError(f'unknown QM/MM charge mode: {mode!r}')
    # `q_qm` arrives as a list or as the numpy array the assembly built, so
    # this tests for absence rather than leaning on truthiness, which an
    # array does not answer.
    q_values = [] if q_qm is None else [float(value) for value in q_qm]
    if mode == 'field' and not q_values:
        # The topology-time check, made while the model is built: there is
        # nothing to shape yet.
        return
    if len(q_values) != len(partition['qm_ids']):
        raise FFModelUnsupported(
            f'q_qm has {len(q_values)} values for '
            f'{len(partition["qm_ids"])} QM atoms; the probe needs the '
            f'whole QM set because the unprobed centers are environment')


def apply_qmmm(model: dict, qm_atoms, centers=None) -> dict:
    """Remove the QM region's bonded rows from the topology in place.

    A row whose member atoms are all in the QM region describes an
    interaction the QM calculation already contains, so it is dropped. A
    row that straddles the partition describes a QM-MM bonded term, which
    this version refuses explicitly: dropping it would silently delete a
    physical interaction, and keeping it would double count it against
    the point-charge model. Unknown topology is refused the same way -
    "no row" is not evidence that no covalent bond crosses the boundary.

    ``centers`` (the per-center atom groups of a multi-center QMMM run)
    adds a third refusal: a row whose members are all QM but split across
    two centers is a covalent bond *between* centers, which neither center's
    wavefunction describes. Dropping it would delete the bond; leaving it to
    the force field would let a bond no QM region owns stretch and bend.

    Parameters
    ----------
    model : dict
        Model whose data sections were just parsed (modified in place).
    qm_atoms : sequence of int
        0-based framework atom indices of the QM region.
    centers : sequence of sequence of int or None
        Per-center 0-based atom indices; None keeps the single-center
        behavior.

    Returns
    -------
    dict
        The partition plus the number of rows removed per section.

    Raises
    ------
    FFModelUnsupported
        On a cross-boundary or cross-center entity row, an unreadable one,
        or centers that do not tile the QM region.
    """
    partition = plan_partition(model, qm_atoms)
    validate_partition_charges(partition, partition['qm_charges'], 'energy')
    qm_set = set(partition['qm_ids'])
    center_sets = [{int(index) + 1 for index in group}
                   for group in (centers or [])]
    if len(center_sets) > 1:
        # The in-cell subtraction is done per center, so an unclaimed QM
        # atom would leave its pairs in the classical surface while the
        # center that does own them already contains them.
        covered = set().union(*center_sets)
        if covered != qm_set:
            raise FFModelUnsupported(
                f'qm_centers cover {sorted(covered)} but the QM region is '
                f'{sorted(qm_set)}: every QM atom must belong to one center.')
    partition['qm_centers'] = [[int(index) for index in group]
                               for group in (centers or [])]
    sections = model['sections']
    counts = sections['counts']
    removed = {}

    def _owner(atom_id):
        for slot, group in enumerate(center_sets):
            if atom_id in group:
                return slot
        return None

    for name, kind, type_column in _ENTITY_SECTIONS:
        rows = sections.get(name) or []
        kept = []
        for raw in rows:
            content = raw.partition('#')[0].split()
            members = content[type_column + 1:]
            try:
                member_ids = [int(token) for token in members]
            except ValueError:
                raise FFModelUnsupported(
                    f'{name} row does not carry integer atom ids: {raw!r}')
            inside = [atom_id for atom_id in member_ids
                      if atom_id in qm_set]
            if not inside:
                kept.append(raw)
                continue
            if len(inside) == len(member_ids):
                if len({_owner(atom_id) for atom_id in inside}) > 1:
                    raise FFModelUnsupported(
                        f'{name} row binds atoms of two QM centers '
                        f'({raw!r}): no center owns that bond, so it is '
                        f'refused rather than deleted or left classical')
                continue
            raise FFModelUnsupported(
                f'{name} row crosses the QM/MM partition ({raw!r}): v1 '
                f'refuses a cross-boundary bonded term rather than dropping '
                f'or double counting it')
        sections[name] = kept
        counts[kind] = len(kept)
        removed[kind] = len(rows) - len(kept)

    model['audit']['qmmm_removed'] = removed
    return dict(partition, removed=removed)


# LAMMPS's Coulomb energy prefactor q_i q_j / r, by units style. These are
# the numbers the engine itself uses (force.cpp qqrd2e), not a general
# physical constant: real is kcal*A/mol/e^2, metal is eV*A/e^2. A
# difference here would show up as a non-zero residual in the
# non-periodic check `compute group/group(qm,all) - group/group(qm,mm)`
# against this sum, which is why that check is a runtime guard.
COULOMB_PREFACTOR = {'real': 332.06371, 'metal': 14.399645}


def _minimum_image(vector, lattice):
    """The shortest periodic image of a displacement.

    LAMMPS evaluates a pair inside the cell through the minimum image,
    so the correction term has to use the same representative: the
    subtraction is an identity only when both sides describe the same
    pair distance. A search over the 27 neighbouring cells is exact for
    any cell this framework carries and is cheap for a QM region.

    Parameters
    ----------
    vector : sequence of float
        ``r_i - r_j`` in Angstrom.
    lattice : (3, 3) list of float or None
        Cell matrix (rows a, b, c), or None for a non-periodic cell.

    Returns
    -------
    list of float
        The displacement with the smallest length.
    """
    if lattice is None:
        return [float(value) for value in vector]
    best = [float(value) for value in vector]
    best_norm = sum(value * value for value in best)
    for n0 in (-1, 0, 1):
        for n1 in (-1, 0, 1):
            for n2 in (-1, 0, 1):
                if n0 == n1 == n2 == 0:
                    continue
                candidate = [
                    float(vector[axis])
                    + n0 * float(lattice[0][axis])
                    + n1 * float(lattice[1][axis])
                    + n2 * float(lattice[2][axis])
                    for axis in range(3)]
                norm = sum(value * value for value in candidate)
                if norm < best_norm:
                    best, best_norm = candidate, norm
    return best


def in_cell_qq(positions, charges, lattice, units: str) -> dict:
    """The bare in-cell Q-Q Coulomb term, its per-atom shares and forces.

    This is the one piece of the classical surface a QM calculation
    already contains and LAMMPS does not know to leave out. It is
    evaluated here from the QM coordinates and charges, with the same
    minimum image LAMMPS uses, so the whole-cell result can subtract it
    exactly without a second engine evaluation.

    The interaction with the periodic images is not part of this term:
    it stays in the LAMMPS result (see `assemble_qmmm_energy`).

    Parameters
    ----------
    positions : sequence of (3,) sequence of float
        Wrapped QM positions, Angstrom.
    charges : sequence of float
        Per-atom charges of the same atoms.
    lattice : (3, 3) list of float or None
        Cell matrix, or None for a non-periodic cell.
    units : str
        LAMMPS units style ('real' or 'metal').

    Returns
    -------
    dict
        ``energy`` (native units), ``per_atom`` (each atom's full
        interaction energy with the rest of Q, summing to twice
        ``energy``) and ``forces`` (native force on each atom, which is
        minus the gradient of ``energy``).
    """
    prefactor = COULOMB_PREFACTOR[units]
    count = len(positions)
    energy = 0.0
    per_atom = [0.0] * count
    forces = [[0.0, 0.0, 0.0] for _ in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            vector = _minimum_image(
                [float(positions[i][axis]) - float(positions[j][axis])
                 for axis in range(3)], lattice)
            distance2 = sum(value * value for value in vector)
            if distance2 <= 0.0:
                raise FFModelUnsupported(
                    f'two QM atoms ({i}, {j}) are at the same position; the '
                    f'in-cell Q-Q term the QM calculation already contains '
                    f'cannot be subtracted')
            distance = distance2 ** 0.5
            pair = prefactor * float(charges[i]) * float(charges[j]) / distance
            energy += pair
            per_atom[i] += pair
            per_atom[j] += pair
            factor = prefactor * float(charges[i]) * float(charges[j]) \
                / distance2 ** 1.5
            for axis in range(3):
                value = factor * vector[axis]
                forces[i][axis] += value
                forces[j][axis] -= value
    return {'energy': energy, 'per_atom': per_atom, 'forces': forces}
