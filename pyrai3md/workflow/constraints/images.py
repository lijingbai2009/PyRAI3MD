"""
======================================================================
The periodic image services the constraint core reaches through.

Constraints measure distances, and a distance in a periodic cell is only a
number once someone says which copy of the cell each atom sits in. The stored
representation answers that with a wrapped coordinate plus an integer crossing
count, and the count is what makes a molecule whose halves sit on opposite faces
one molecule instead of two fragments.

This module is the seam between that representation and the numeric core:

- the bond table comes from `utilities.bonds.perceive_bonds`, the
  framework's one perception routine. It works in Angstrom because its radius
  table (`utilities.bonds.bond_lib`) is in Angstrom, while the constraint core
  works in Bohr. Only the integer counts cross over: the unfolded Bohr geometry
  is the stored Bohr geometry plus `offsets @ cell`, which is exact and is never
  a second unit conversion.
- the half-box gate. A constrained distance approaching half of the shortest
  interplanar spacing sits where the nearest image ties, so the variable jumps
  between steps and the solver is being asked to follow a discontinuity. That is
  a declaration error, not a convergence failure, and it is refused with the axis
  and the difference.
- the same-component refusal. An angle is a chain of two bonds and a torsion is
  a chain of three, so they have an image rule as long as their atoms sit in one
  connected component. A distance between two molecules, an aligned RMSD over a
  composite, or a group center the bond graph does not connect has no rule to
  follow, and the answer stays a refusal instead of an invented convention.

Every entry point has an identity path when the view has no cell, which is what
keeps a molecule-in-a-box run reading exactly the numbers it read before this
existed. The module reads a state-shaped object by attribute and imports neither
the state class nor the config model.

Sep 30 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from pyrai3md.utilities import bonds
from pyrai3md.utilities import geometric
from pyrai3md.utilities import lattice
from pyrai3md.utilities.units import ANGSTROM_TO_BOHR

# Perception needs element symbols to read the radius table. A view without them
# cannot be unfolded, and the identity path below is the honest answer: no bond
# graph, so no multi-atom variable gets an image rule, so it is refused by name
# rather than measured across the box.
_NO_ELEMENTS = 'no atom symbols were read'


class ImageError(ValueError):
    """A periodic constraint cannot be given an image rule."""


class Expansion(object):
    """One read of the unfolded geometry: the numbers and how to write back.

    Attributes
    ----------
    coordinates : numpy.ndarray
        (natom, 3) unfolded coordinates, in the core's unit (Bohr).
    offsets : numpy.ndarray
        (natom, 3) integer fractional shift from the stored position. A
        correction on unfolded atom k belongs to stored atom k, and the stored
        coordinate is the unfolded one minus `offsets @ cell`, so this table is
        the write-back map.
    components : numpy.ndarray
        (natom,) anchor atom index of each atom's connected component.
    bonds : list
        The (i, j, delta) rows perception produced.
    cell : numpy.ndarray or None
        (3, 3) row-vector lattice in Bohr, None when there is no cell.
    pbc : numpy.ndarray
        Three 0/1 periodicity flags used for the fold.
    """

    __slots__ = ('coordinates', 'offsets', 'components', 'bonds', 'cell',
                 'pbc')

    def __init__(self, coordinates, offsets, components, bonds, cell, pbc):
        self.coordinates = coordinates
        self.offsets = offsets
        self.components = components
        self.bonds = bonds
        self.cell = cell
        self.pbc = pbc

    @property
    def periodic(self):
        """True when a cell was read, i.e. when anything was unfolded at all."""
        return self.cell is not None

    def within_component(self, indices):
        """True when the atoms of one variable are all in one molecule."""
        if not self.periodic or self.components is None:
            return True
        return len(set(int(self.components[int(atom)])
                       for atom in indices)) <= 1


def read_expansion(view, focus=None):
    """Unfold the stored geometry along its perceived bond graph.

    A view with no cell gets the identity expansion: the stored coordinates, zero
    offsets, one component per atom, no bonds. Callers then run the same code on
    both paths.

    `focus` names the atoms whose components are needed. Passing it switches the
    bond search to `perceive_bonds_focus`, which scans outward from those atoms
    instead of probing every pair: measured on a 600-atom water box, the full
    probe costs 295 ms and the focal scan of one water costs 3.1 ms for the same
    rows. The two routes are the same rule, not two conventions - and when the
    focus already covers half the structure the full probe is the cheaper one, so
    the choice is made on size rather than on which argument was given.

    Parameters
    ----------
    view : object
        Anything exposing `coordinates` (Bohr), `cell` (Bohr or None), `atoms`,
        and a periodic flag; a `pbc` attribute is read when present.
    focus : sequence, optional
        0-based atom indices whose connected components the caller needs. None
        means the whole structure.

    Returns
    -------
    Expansion
    """
    coordinates = np.asarray(view.coordinates, dtype=np.float64)
    natom = coordinates.shape[0]
    cell = getattr(view, 'cell', None)
    flags = getattr(view, 'pbc', None)
    # A view that does not say which axes repeat is read as fully periodic: the
    # cell is what makes this branch run at all, and a shift is only dropped on
    # an axis the caller has ruled out.
    pbc = (np.ones(3, dtype=np.int64) if flags is None else
           np.asarray(flags, dtype=np.int64).reshape(-1))
    identity = Expansion(coordinates, np.zeros((natom, 3), dtype=np.int64),
                         np.arange(natom, dtype=np.int64), [], None, pbc)
    if cell is None or not getattr(view, 'periodic', True):
        return identity
    matrix = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    atoms = list(getattr(view, 'atoms', None) or [])
    if len(atoms) != natom:
        return Expansion(coordinates, np.zeros((natom, 3), dtype=np.int64),
                         np.arange(natom, dtype=np.int64), [], matrix, pbc)
    scale = 1.0 / ANGSTROM_TO_BOHR
    focal = sorted(set(int(index) for index in (focus or ())))
    if focal and 2 * len(focal) < natom:
        rows = bonds.perceive_bonds_focus(
            atoms, coordinates * scale, matrix * scale, focal, pbc=pbc)
    else:
        rows = bonds.perceive_bonds(atoms, coordinates * scale,
                                    matrix * scale, pbc=pbc)
    if not rows:
        return Expansion(coordinates, np.zeros((natom, 3), dtype=np.int64),
                         np.arange(natom, dtype=np.int64), [], matrix, pbc)
    _unfolded, offsets, components = bonds.unfold_components(
        coordinates * scale, matrix * scale, rows, pbc=pbc)
    return Expansion(coordinates + offsets @ matrix, offsets, components, rows,
                     matrix, pbc)


def require_same_component(expansion, indices, label):
    """Refuse a variable the bond graph does not connect."""
    members = [int(atom) for atom in indices]
    if expansion.periodic and not expansion.within_component(members):
        raise ImageError(
            '%s names atoms %s, which lie in %d separate connected components '
            'of the perceived bond graph; a periodic image rule exists for a '
            'variable inside one molecule, and this version does not invent one '
            'for the rest'
            % (label, [atom + 1 for atom in members],
               len(set(int(expansion.components[atom]) for atom in members))))
    return members


def require_half_box(expansion, rows, label, declared=None):
    """Refuse a constrained distance at or beyond half the shortest spacing.

    Both numbers are checked: the separation the geometry shows right now, and
    the length the declaration asks the constraint to hold. The second is what a
    user gets wrong without noticing - "hold this bond at 5 Bohr" is a
    discontinuous variable in a cell whose shortest gap is 8 Bohr, whatever the
    atoms happen to be doing at the step the run starts.

    The message carries the axis: the pair that ties says which cell direction is
    too short for the declared distance, and by how much.
    """
    if not expansion.periodic:
        return
    scale = 1.0 / ANGSTROM_TO_BOHR
    bound = lattice.half_box_bound(expansion.cell * scale)
    if bound <= 0.0:
        raise ImageError('%s: the cell has no usable interplanar spacing'
                         % label)
    # `bound` is in Angstrom because the interplanar spacing routine reads the
    # radius table's unit; everything compared below is in Bohr, so the bound
    # crosses to Bohr once here rather than being scaled a second time.
    limit = bound * ANGSTROM_TO_BOHR
    array = np.asarray(expansion.coordinates, dtype=np.float64)
    inverse = np.linalg.inv(expansion.cell)
    for position, row in enumerate(rows):
        atoms = [int(atom) for atom in row]
        worst = None
        for left in range(len(atoms) - 1):
            for right in range(left + 1, len(atoms)):
                delta = array[atoms[right]] - array[atoms[left]]
                length = float(np.linalg.norm(delta))
                if declared is not None:
                    index = min(position, len(declared) - 1)
                    length = max(length, float(declared[index]))
                if length <= limit:
                    continue
                if worst is None or length > worst[0]:
                    spread = np.abs(delta @ inverse)
                    axis = int(np.argmax(spread))
                    worst = (length, atoms[left], atoms[right], axis,
                             float(spread[axis]))
        if worst is not None:
            length, left, right, axis, fraction = worst
            raise ImageError(
                "%s: atoms %d-%d sit %.4f Bohr apart against a declared length "
                'of %.4f, and the larger of the two is at or beyond half of the '
                'shortest interplanar spacing (%.4f Bohr), which leaves the '
                'nearest image tied along axis %s (fractional separation %.4f); '
                'the constraint is discontinuous there and this version refuses '
                'it rather than solving across the tie'
                % (label, left + 1, right + 1,
                   float(np.linalg.norm(array[right] - array[left])),
                   float(declared[position]) if declared is not None else 0.0,
                   limit, 'abc'[axis], fraction))


def group_center(expansion, indices, masses=None):
    """The center of a group, taken on the unfolded geometry.

    A centroid averaged over wrapped coordinates puts `0.01` and `L - 0.01` at
    `L / 2`, which is the center of nothing. On the unfolded geometry a connected
    group's center is the molecule's own center, and shifting the molecule by one
    cell shifts the center by one cell.
    """
    members = np.asarray(require_same_component(expansion, indices,
                                                'group center'), dtype=np.int64)
    try:
        return geometric.group_center(expansion.coordinates, members, masses=masses)
    except geometric.CVSingularError as error:
        raise ImageError(str(error)) from error


def fold_back(expansion, coordinates):
    """Wrap an unfolded geometry and report the crossings it implies.

    This is the second half of "solve on the unfolded coordinates, then fold":
    the counts returned are the counts the expansion started from plus whatever
    the solve carried an atom across a face. A caller that writes the folded
    coordinates without these hands the engine a pair that contradicts itself.

    Returns
    -------
    tuple
        (wrapped coordinates in Bohr, (natom, 3) integer crossing counts). The
        counts are absolute, not an increment: they are read off the unfolded
        position the caller is folding, which already contains the shift the
        expansion carried, so adding the expansion's own offsets on top would
        charge every crossed face twice and hand the next engine call an image
        count one cell too far out.
    """
    array = np.asarray(coordinates, dtype=np.float64)
    if not expansion.periodic:
        return array, np.zeros(array.shape, dtype=np.int64)
    fractional = array @ np.linalg.inv(expansion.cell)
    wrapped_frac, crossings = lattice.wrap_fractional_with_image(
        fractional, expansion.pbc)
    return wrapped_frac @ expansion.cell, crossings
