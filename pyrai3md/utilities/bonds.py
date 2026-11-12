"""
======================================================================
This module is the bond topology and image service of PyRAI3MD: the
bond-length lookup, bond perception, the bond graph, connected
components and offsets, component unfolding, and the initial image
counts of a wrapped structure.

The bond-length table is `schema.reference.BOND_LENGTHS`; this module
owns the lookup and the 1.2 Angstrom default an unlisted pair receives.
The heavy-atom entries carry the single-bond distance because perception
tests `distance < bond_lib(a, b) * 1.2`; the data's own comment records
the D-28 measurement behind that choice.

A wrapped snapshot keeps every atom inside the cell, so the integer
crossing counts are the only thing that says which copy of the cell a
bond connects: a molecule whose two halves sit on opposite faces is
continuous only once those crossings are declared. LAMMPS reconstructs
its unwrapped positions as ``x + image * box``, reads no crossings from
a data file that does not list them, and reports the resulting long bond
as "WARNING: Inconsistent image flags". The counts therefore have to be
derived from the geometry before the structure is handed to an engine.

The derivation is a bond-graph propagation, ported from the author's
FFautomator package (ffauto/molecule/graph.py, generate_connections /
generate_connected_pairs / generate_fragments /
generate_intuitive_coordinates; Copyright (c) 2026 Jingbai Li. All
Rights Reserved.): perceive bonds by minimum-image distance, walk each
connected component as a spanning tree, and let each child inherit its
parent's count plus the image step the bond needed. The wrapped
coordinates are never moved; only the integer counts are read off.

Pure functions only: this module never stores state. The authority for
the seeded counts is the schema System.image field.

Conventions
-----------
- Row-vector layout, as in utilities.lattice: the lattice matrix B has
  the cell vectors as rows and S = R @ B^-1 gives fractional
  coordinates.
- Lengths in Angstrom; image counts are integers per axis.
- pbc defaults to fully periodic here (unlike lattice.normalize_pbc,
  which reads None as non-periodic), because the question this module
  answers only exists for a periodic cell.
- An empty result means "nothing could be derived", not "the crossings
  are zero".

The module imports numpy, `.lattice` and `..schema.reference` only; it
must not import workflow, data or skills (framework-skill isolation).

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import warnings

import numpy as np

from ..schema.reference import BOND_LENGTHS
from .lattice import (cart_to_frac, half_box_bound, min_image_displacement,
                       normalize_pbc)

# A pair is bonded when its shortest image distance falls below
# bond_lib(pair) * BOND_SCALE, the tolerance the ported perception used
# (there, a covalent-radius sum times 1.2; here, the repository's own
# typical-bond-length table times the same factor, which selects the
# same 40 O-H bonds in the shipped water box).
BOND_SCALE = 1.2

# Perception probes every atom pair, so its cost is O(N^2). Below this
# atom count a single configuration-time derivation is worth the wall
# time; above it the derivation is refused rather than silently
# stalling a large run. Raising the cap without a cell list buys time,
# not scale.
PERCEPTION_MAX_ATOMS = 4096


def bond_lib(atom1, atom2):
    """Look up the typical bond length for a given atom pair.

    Parameters
    ----------
    atom1 : str
        First atom element symbol.
    atom2 : str
        Second atom element symbol.

    Returns
    -------
    float
        Typical bond length in Angstrom. Returns 1.2 for
        unrecognized pairs.
    """
    label = '%s-%s' % (atom1, atom2)
    if label in BOND_LENGTHS:
        return BOND_LENGTHS[label]
    return 1.2


def periodic_flags(pbc):
    """Return three 0/1 periodicity flags, defaulting to all periodic.

    This is the topology-side rule: `None` means fully periodic here,
    unlike `lattice.normalize_pbc(None)` which means non-periodic. The
    two contracts are kept separate because the question a topology
    propagates through only exists for a periodic cell.
    """
    if pbc is None:
        return np.ones(3, dtype=np.int64)
    return np.asarray(normalize_pbc(pbc), dtype=np.int64)


def perceive_bonds(atoms, coordinates, lattice_vector, pbc=None,
                   scale=BOND_SCALE):
    """Find bonded pairs and the image step each bond needs.

    Parameters
    ----------
    atoms : list
        Element symbols, one per atom, in framework order.
    coordinates : array-like
        Wrapped Cartesian coordinates, shape (natom, 3), Angstrom.
    lattice_vector : array-like
        Lattice matrix, shape (3, 3), rows are cell vectors (Angstrom).
    pbc : array-like, optional
        Three 0/1 periodicity flags; None means fully periodic. Axes
        that are not periodic are never shifted.
    scale : float, optional
        Multiplier on the typical bond length of the pair.

    Returns
    -------
    list of tuple
        ``(i, j, delta)`` for each perceived bond, where ``delta`` is
        the per-axis integer count atom j adds to atom i's count to sit
        in i's nearest image. The shift is what the perception
        established, so the tree edges of derive_initial_image agree
        with it by construction; a bond the tree reaches by another
        route is not covered by it and is validated afterwards.
    """
    coords = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
    if coords.shape[0] < 2:
        return []
    if lattice_vector is None:
        # Aperiodic perception: the same radius rule, with every image step
        # zero because there is no cell to shift by. The routine was written to
        # seed periodic image flags and returned nothing without a cell; a bond
        # graph is the same question asked of a molecule, so the answer now has
        # one shape either way (design document section 3.8 item 1), and
        # `derive_initial_image` keeps its own early return for a system with no
        # periodic axis, so seeding is unaffected.
        return [(i, j, (0, 0, 0))
                for i in range(coords.shape[0] - 1)
                for j in range(i + 1, coords.shape[0])
                if float(np.linalg.norm(coords[j] - coords[i]))
                < bond_lib(atoms[i], atoms[j]) * scale]

    lattice = np.asarray(lattice_vector, dtype=np.float64).reshape(3, 3)
    try:
        # The conversion is the singularity test; its refusal means there is
        # no fractional representation, and the old 27-shift answer was the
        # empty table. The certificate's own refusal below is a different
        # finding and propagates.
        cart_to_frac(coords, lattice)
    except ValueError:
        return []
    flags = periodic_flags(pbc)
    inv_lattice = np.linalg.inv(lattice)
    bonds = []
    for i in range(coords.shape[0] - 1):
        gap = coords[i + 1:] - coords[i]
        reduced = min_image_displacement(gap, lattice, pbc=flags)
        delta = np.rint((reduced - gap) @ inv_lattice).astype(np.int64)
        lengths = np.linalg.norm(reduced, axis=1)
        for offset, value in enumerate(lengths):
            j = i + 1 + offset
            if value < bond_lib(atoms[i], atoms[j]) * scale:
                bonds.append((i, j, tuple(int(n) for n in delta[offset])))
    return bonds


def perceive_bonds_focus(atoms, coordinates, lattice_vector, focus,
                         pbc=None, scale=BOND_SCALE):
    """Perceive only the bonds reachable from a set of focal atoms.

    `perceive_bonds` probes every pair, so its cost is O(N^2) and a periodic
    constraint would pay that on every evaluation. The constraint core does not
    need the whole graph: it needs the connected component of the atoms its
    variables name. Those are found by scanning outward from the focal atoms and
    adding each newly bonded partner as a new scan, which costs O(N) per atom in
    the reached set and nothing for the rest of the box.

    The rows are the same shape `perceive_bonds` returns, oriented with the
    lower index first so a pair reached from either end is the same row, and the
    winner among the candidate images is chosen by the same rule on the same
    candidate list. A bond entirely outside the reached set is not returned: it
    cannot change how the focal component is unfolded.

    Parameters
    ----------
    atoms : list
        Element symbols, one per atom, in framework order.
    coordinates : array-like
        Wrapped Cartesian coordinates, shape (natom, 3), Angstrom.
    lattice_vector : array-like
        Lattice matrix, shape (3, 3), rows are cell vectors (Angstrom).
    focus : sequence
        0-based atom indices whose components are wanted.
    pbc : array-like, optional
        Three 0/1 periodicity flags; None means fully periodic.
    scale : float, optional
        Multiplier on the typical bond length of the pair.

    Returns
    -------
    list of tuple
        (i, j, delta) rows with i < j, sorted, in the same convention as
        `perceive_bonds`.
    """
    if lattice_vector is None:
        return []
    coords = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
    natom = coords.shape[0]
    if natom < 2:
        return []
    lattice = np.asarray(lattice_vector, dtype=np.float64).reshape(3, 3)
    try:
        cart_to_frac(coords, lattice)
    except ValueError:
        return []
    flags = periodic_flags(pbc)
    inv_lattice = np.linalg.inv(lattice)

    reached = set(int(index) for index in focus if 0 <= int(index) < natom)
    found = {}
    queue = sorted(reached)
    scanned = set()
    while queue:
        i = queue.pop(0)
        if i in scanned:
            continue
        scanned.add(i)
        others = np.array([j for j in range(natom) if j != i], dtype=np.int64)
        if not others.size:
            continue
        gap = coords[others] - coords[i]
        reduced = min_image_displacement(gap, lattice, pbc=flags)
        delta = np.rint((reduced - gap) @ inv_lattice).astype(np.int64)
        lengths = np.linalg.norm(reduced, axis=1)
        for offset, value in enumerate(lengths):
            j = int(others[offset])
            if value >= bond_lib(atoms[i], atoms[j]) * scale:
                continue
            step = tuple(int(n) for n in delta[offset])
            if i < j:
                row = (i, j, step)
            else:
                row = (j, i, tuple(-value for value in step))
            # Reached from the other end of the same pair is the same bond;
            # the first scan that found it owns the row, so a tie between two
            # equal-length images resolves the same way whichever end looked.
            found.setdefault(row[:2], row)
            if j not in reached:
                reached.add(j)
                queue.append(j)
    return [found[key] for key in sorted(found)]


def _listed_pairs():
    """The element combinations the radius table actually names.

    Read off `BOND_LENGTHS`' own keys rather than a second list or a probe
    over a hand-written symbol list: extending the table extends the
    topology service in the same step, and a pair the table names can no
    longer be missed because the symbols to probe were stale. The probe and
    the derived set were measured equal (20 pairs each, no difference), so
    this is not a change of coverage.
    """
    return {tuple(sorted(key.split('-'))) for key in BOND_LENGTHS
            if '-' in key}


class UnknownPairError(ValueError):
    """An element combination the radius table does not list."""


class ComponentError(ValueError):
    """The bond graph cannot be walked as declared."""


def component_offsets(neighbours, natom, flags):
    """Walk every component from its lowest atom.

    Returns ``(offsets, components, fragments)``: the per-atom integer
    fractional offset from the anchor, the anchor index of each atom's
    component, and one list of member indices per component. The anchor
    is the lowest atom index and the traversal is breadth-first in the
    order the adjacency was built, so the answer does not depend on how
    a caller sorted its bond rows.
    """
    offsets = np.zeros((natom, 3), dtype=np.int64)
    components = np.arange(natom, dtype=np.int64)
    fragments = []
    visited = np.zeros(natom, dtype=bool)
    for anchor in range(natom):
        if visited[anchor]:
            continue
        visited[anchor] = True
        fragment = [anchor]
        queue = [anchor]
        cursor = 0
        while cursor < len(queue):
            current = queue[cursor]
            cursor += 1
            for child, step in neighbours[current]:
                if visited[child]:
                    continue
                visited[child] = True
                offsets[child] = offsets[current] + step
                components[child] = anchor
                fragment.append(child)
                queue.append(child)
        fragments.append(fragment)
    offsets *= flags
    return offsets, components, fragments


def unfold_components(coordinates, lattice, bonds, pbc=None):
    """Unfold every connected component along its bond graph.

    Each component's anchor is its lowest atom index, which is also the order
    the graph is walked in, so the answer does not depend on how the bond table
    was sorted. The anchor keeps the stored coordinate; every other atom is
    placed at its anchor's position plus the shortest image of the bond that
    reaches it, so a molecule whose halves sit on opposite cell faces comes out
    continuous instead of torn at the face.

    Parameters
    ----------
    coordinates : array-like
        Stored (wrapped) Cartesian coordinates, shape (natom, 3).
    lattice : array-like or None
        Row-vector lattice matrix, shape (3, 3). None means no cell, and with
        it the expansion is the identity.
    bonds : sequence
        (i, j, delta) rows; `delta` is the per-axis integer image count atom j
        adds to atom i. An empty table leaves every atom in a component of one.
    pbc : sequence, optional
        Three 0/1 periodicity flags. None means fully periodic, matching
        `bonds.periodic_flags`: a shift is only meaningful on an axis the cell
        repeats along, and the offset on a non-periodic axis is dropped.

    Returns
    -------
    tuple
        ``(expanded, offsets, components)``: unfolded Cartesian coordinates
        (natom, 3), the integer fractional offset of each atom from its stored
        position (natom, 3), and the anchor atom index of the component each
        atom belongs to (natom,). `expanded` is exactly
        ``coordinates + offsets @ lattice``, so the write-back is the offset
        table: a correction landed on unfolded atom k belongs to stored atom k,
        and because a fixed delta makes d(expanded)/d(stored) the identity, a
        Jacobian row indexed on the unfolded array is already indexed on the
        stored one.

    Raises
    ------
    ComponentError
        A bond references an atom the structure does not have.
    """
    coords = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
    natom = coords.shape[0]
    offsets = np.zeros((natom, 3), dtype=np.int64)
    components = np.arange(natom, dtype=np.int64)
    if lattice is None or natom == 0:
        return coords, offsets, components

    matrix = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    flags = periodic_flags(pbc)
    neighbours = [[] for _ in range(natom)]
    for row in bonds:
        i, j, delta = int(row[0]), int(row[1]), row[2]
        if not (0 <= i < natom and 0 <= j < natom):
            raise ComponentError(
                'bond %d-%d references an atom outside the structure, which '
                'has %d atoms' % (i + 1, j + 1, natom))
        step = np.asarray(delta, dtype=np.int64).reshape(3) * flags
        neighbours[i].append((j, step))
        neighbours[j].append((i, -step))

    offsets, components, _fragments = component_offsets(
        neighbours, natom, flags)
    # The unfolded structure is the stored one shifted by whole cells, not a
    # geometry re-derived by walking bond vectors from the anchor. The two are
    # the same structure mathematically, and only this form keeps the promise
    # the return makes: an atom whose offset is zero keeps its stored numbers
    # exactly, so a molecule that never crosses a face evaluates to what it
    # evaluated to before any of this existed.
    expanded = coords + offsets @ matrix
    return expanded, offsets, components


def bond_graph(atoms, coordinates, lattice_vector=None, pbc=None,
               scale=BOND_SCALE):
    """The bond graph of a structure, as one answer for one geometry.

    This is the topology service `perceive_bonds` was never asked to be: it
    returns the pairs *and* the connected components, and it refuses rather than
    guessing when an element combination that could be bonded is missing from
    the radius table. `bond_lib` answers an unknown pair with 1.2 Angstrom, which
    is the right tolerance for seeding image flags (a missed bond there costs a
    zero count) and the wrong one for choosing which atoms a constraint binds:
    there, a silent default decides physics (design document section 3.8 item 1).

    Parameters
    ----------
    atoms : list
        Element symbols, one per atom.
    coordinates : array-like
        Cartesian coordinates, shape (natom, 3), in the unit of the table
        (Angstrom).
    lattice_vector : array-like, optional
        Row-vector lattice, shape (3, 3). None is an aperiodic structure.
    pbc : array-like, optional
        Three 0/1 flags; None means fully periodic when a cell is given.
    scale : float, optional
        Multiplier on the typical bond length of the pair.

    Returns
    -------
    dict
        'bonds': the (i, j, delta) rows, 'atoms': the symbols read back, and
        'components': a (natom,) label per atom, the lowest index of its
        connected component. A structure with no bonds returns empty pairs and
        one component per atom, which is a fact about the input, not a failure.

    Raises
    ------
    UnknownPairError
        Two atoms sit within the default cutoff of a pair the table does not
        list, so the answer would rest on that default.
    """
    symbols = [str(item) for item in (atoms or [])]
    coords = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
    natom = coords.shape[0]
    if len(symbols) != natom:
        raise UnknownPairError(
            'the bond graph needs one element symbol per atom, got %d symbols '
            'for %d atoms' % (len(symbols), natom))
    bonds = perceive_bonds(symbols, coords, lattice_vector, pbc=pbc,
                           scale=scale)
    listed = _listed_pairs()
    matrix = None
    flags = periodic_flags(pbc)
    if lattice_vector is not None:
        matrix = np.asarray(lattice_vector, dtype=np.float64).reshape(3, 3)
    for i in range(natom - 1):
        for j in range(i + 1, natom):
            pair = tuple(sorted((symbols[i], symbols[j])))
            if pair in listed:
                continue
            if matrix is None:
                length = float(np.linalg.norm(coords[j] - coords[i]))
            else:
                reduced = min_image_displacement(
                    coords[j] - coords[i], matrix, pbc=flags)
                length = float(np.linalg.norm(reduced))
            # `bond_lib`'s default is 1.2 Angstrom, so anything at or below
            # the default cutoff of a pair nobody listed is a decision the
            # table cannot make. The comparison is the one the listed branch
            # uses (`>= bond_lib(...) * scale` continues), read from the other
            # side: a pair exactly on the cutoff would have bonded, so the
            # table has to answer it.
            if length <= 1.2 * scale:
                raise UnknownPairError(
                    "the element pair %s-%s (atoms %d-%d, %.3f Angstrom apart) "
                    'is not in the bond-radius table, which would fall back to '
                    'a 1.2 Angstrom cutoff here; list the atoms explicitly or '
                    'extend schema/reference.py::BOND_LENGTHS'
                    % (symbols[i], symbols[j], i + 1, j + 1, length))
    neighbours = [[] for _ in range(natom)]
    for i, j, delta in bonds:
        step = np.asarray(delta, dtype=np.int64).reshape(3)
        neighbours[int(i)].append((int(j), step))
        neighbours[int(j)].append((int(i), -step))
    _offsets, components, _fragments = component_offsets(
        neighbours, natom, periodic_flags(pbc))
    return {'bonds': bonds, 'atoms': symbols, 'components': components}


def bond_pairs_by_type(graph, label):
    """The bonded pairs whose element combination matches `label`.

    'O-H', 'H-O' and 'o-h' name the same pair: an element combination has no
    order. Atoms are 0-based inside the graph and 1-based in every framework
    input, so the rows come back 1-based, the same shape an explicit `bonds`
    list has.

    'X-H' is the one wildcard the spelling carries: X stands for any element
    that is not hydrogen, so a box of water, an alcohol or a mixed protic
    solvent holds every its X-H bonds in one line without the input having to
    know which heavy atoms are present. It is the combination the shortcut
    exists for, and only that one: 'X-X' or 'X-C' would make X mean whatever
    the reader needed it to mean.
    """
    parts = [part.strip().capitalize()
             for part in str(label).replace('-', ' ').split()]
    if len(parts) != 2:
        raise ValueError(
            "bond type %r needs two element symbols, as in 'O-H'" % (label,))
    symbols = graph['atoms']
    wildcard = 'X' in parts
    if wildcard:
        other = [part for part in parts if part != 'X']
        if other != ['H']:
            raise ValueError(
                "bond type %r is not a wildcard: 'X' stands for any element "
                "other than hydrogen, so it pairs with 'H' and with nothing "
                "else" % (label,))
    out = []
    for i, j, _delta in graph['bonds']:
        pair = tuple(sorted((symbols[int(i)], symbols[int(j)])))
        if wildcard:
            if 'H' in pair and pair != ('H', 'H'):
                out.append([int(i) + 1, int(j) + 1])
        elif pair == tuple(sorted(parts)):
            out.append([int(i) + 1, int(j) + 1])
    return sorted(out)


def unwrapped_bond_lengths(coordinates, image, lattice_vector, bonds):
    """Return the continuous bond length of each ``(i, j, _)`` bond.

    ``bonds`` is what perceive_bonds returned, and only the atom pair is
    read from it: the point of the check is to re-derive the distance
    from the counts being validated, not from the shift that produced
    them.
    """
    coords = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
    lattice = np.asarray(lattice_vector, dtype=np.float64).reshape(3, 3)
    counts = np.asarray(image, dtype=np.int64).reshape(-1, 3)
    fractional = coords @ np.linalg.inv(lattice) + counts
    lengths = np.empty(len(bonds), dtype=np.float64)
    for position, (i, j, _delta) in enumerate(bonds):
        vector = (fractional[j] - fractional[i]) @ lattice
        lengths[position] = float(np.linalg.norm(vector))
    return lengths


def derive_initial_image(atoms, coordinates, lattice_vector, pbc=None):
    """Reconstruct the crossing counts a wrapped snapshot implies.

    A spanning forest is walked over the perceived bonds: the root of
    each component is its lowest atom index, which is also the order the
    forest is grown in, so the answer does not depend on insertion
    order. A root starts at zero and every child inherits its parent's
    count plus the image step of the bond that reached it. Each
    component is then shifted back by the rounded mean of its own
    counts, which leaves a molecule nominally in the cell it was
    wrapped into: that keeps the choice canonical for the caller and
    matches what ``fix recenter`` in LAMMPS expects to find.

    The tree is what makes the answer exact where it can be: each bond
    the propagation followed ends up at the image step the perception
    established for it, by construction. A bond the tree reached by
    another route (a ring) is not covered by that argument, so every
    perceived bond, tree edge or not, is then measured against
    ``half_box_bound``, which is LAMMPS' own criterion. That check is a
    consistency gate, not a proof against an unseen shift convention:
    its teeth are in rejecting a structure whose counts cannot work, not
    in re-deriving a distance the propagation already fixed.

    Parameters
    ----------
    atoms : list
        Element symbols, one per atom.
    coordinates : array-like
        Wrapped Cartesian coordinates, shape (natom, 3), Angstrom.
    lattice_vector : array-like
        Lattice matrix, shape (3, 3), rows are cell vectors (Angstrom).
    pbc : array-like, optional
        Three 0/1 periodicity flags; None means fully periodic.

    Returns
    -------
    list of list of int
        (natom, 3) crossing counts. A structure that sits wholly inside
        the cell is a correct answer of all zeros, not a refusal; [] is
        returned when there is nothing to derive from (fewer than two
        atoms, more atoms than PERCEPTION_MAX_ATOMS, a non-periodic or
        unreadable cell, no perceived bonds) or when the counts cannot
        satisfy every bond, in which case a warning says which bond and
        how long it stayed.
    """
    coords = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
    natom = coords.shape[0]
    if natom < 2 or natom > PERCEPTION_MAX_ATOMS:
        return []
    if lattice_vector is None or not np.all(np.isfinite(coords)):
        return []

    lattice = np.asarray(lattice_vector, dtype=np.float64).reshape(3, 3)
    flags = periodic_flags(pbc)
    if not flags.any():
        return []

    bonds = perceive_bonds(atoms, coords, lattice, pbc=flags)
    if not bonds:
        return []

    neighbours = [[] for _ in range(natom)]
    for i, j, delta in bonds:
        step = np.asarray(delta, dtype=np.int64)
        neighbours[i].append((j, step))
        neighbours[j].append((i, -step))

    counts, _components, fragments = component_offsets(
        neighbours, natom, flags)
    for fragment in fragments:
        shift = np.rint(counts[fragment].mean(axis=0)).astype(np.int64)
        counts[fragment] -= shift

    bound = half_box_bound(lattice)
    if bound <= 0.0:
        return []
    lengths = unwrapped_bond_lengths(coords, counts, lattice, bonds)
    too_long = lengths >= bound
    if too_long.any():
        # A component that winds around the cell (a ring closing on the
        # far side) has no consistent assignment: the spanning tree
        # satisfies its bonds and the closing edge then crosses the box.
        # Shipping those counts would be worse than shipping none.
        offending = int(np.argmax(too_long))
        warnings.warn(
            'Periodic image flags are not derivable for this structure: '
            'bond %d-%d would stay %.3f A long, at or beyond half of the '
            'shortest cell extent (%.3f A). The crossing counts are left '
            'unset.' % (bonds[offending][0] + 1, bonds[offending][1] + 1,
                        float(lengths[offending]), bound),
            stacklevel=2,
        )
        return []
    return [[int(value) for value in row] for row in counts]


__all__ = [
    'BOND_SCALE',
    'PERCEPTION_MAX_ATOMS',
    'ComponentError',
    'UnknownPairError',
    'bond_lib',
    'bond_graph',
    'bond_pairs_by_type',
    'component_offsets',
    'derive_initial_image',
    'periodic_flags',
    'perceive_bonds',
    'perceive_bonds_focus',
    'unfold_components',
    'unwrapped_bond_lengths',
]