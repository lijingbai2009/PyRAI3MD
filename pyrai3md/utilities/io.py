"""
======================================================================
This module parses input: molecular coordinate files, point charges,
initial conditions and the text conversions those readers need. It owns
no output formatting; structure tables live in
`utilities/structure_formatter.py` and run-log sections in
`utilities/print_results.py`. Migrated from classic Utils/coordinates.py
with dependencies on the Atom class replaced by schema.reference lookups.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import re
import sys

import numpy as np

from ..schema.reference import PERIODIC_TABLE, ELEMENT_SYMBOLS

# The extended-XYZ cell key, the one `workflow/records/output.py::lattice_comment`
# writes into a frame's comment line: nine Angstrom entries, row-major.
_LATTICE_KEY = re.compile(r'Lattice\s*=\s*"([^"]*)"')


def string2float(x):
    """Convert a list of strings (e,x,y,z) to a 2D float array.

    Parameters
    ----------
    x : list of str
        Lines containing element and coordinates.

    Returns
    -------
    list of list of float
        Parsed coordinate values.
    """
    x = [[float(i) for i in row.split()[1: 4]] for row in x]
    return x


def reverse_string2float(x):
    """Convert a list of strings (e,x,y,z) to a 2D float array (last 3 columns).

    Parameters
    ----------
    x : list of str
        Lines containing element and coordinates.

    Returns
    -------
    list of list of float
        Parsed coordinate values from the last three columns.
    """
    x = [[float(i) for i in row.split()[-3:]] for row in x]
    return x


def complex2string(x):
    """Convert a 2D complex array to a 2D string array.

    Parameters
    ----------
    x : list of list of complex
        Input complex array.

    Returns
    -------
    list of list of str
        String representation of the array.
    """
    x = [[str(i) for i in row] for row in x]
    return x


def string2complex(x):
    """Convert a 2D string array back to a 2D complex array.

    Parameters
    ----------
    x : list of list of str
        Input string array.

    Returns
    -------
    list of list of complex
        Parsed complex array.
    """
    x = [[complex(i) for i in row] for row in x]
    return x


def verify_xyz(mol):
    """Determine the coordinate file type.

    Parameters
    ----------
    mol : str, np.ndarray, list, or dict
        Input molecule specification.

    Returns
    -------
    str
        File type: 'xyz', 'tinker', 'array', 'dict', or 'unknown'.
    """
    if isinstance(mol, str):
        if os.path.exists('%s.xyz' % mol):
            with open('%s.xyz' % mol, 'r') as ld_input:
                xyzfile = ld_input.read().splitlines()
            flag = xyzfile[2].split()[1]

            try:
                float(flag)
                xyztype = 'xyz'
            except ValueError:
                xyztype = 'tinker'
        else:
            sys.exit('\n  FileNotFoundError\n  PyRAI3MD: looking for coordinate file %s' % mol)

    elif isinstance(mol, np.ndarray):
        xyztype = 'array'

    elif isinstance(mol, list):
        xyztype = 'array'

    elif isinstance(mol, dict):
        xyztype = 'dict'
    else:
        xyztype = 'unknown'

    return xyztype


def read_coord(mol):
    """Read xyz coordinates and velocities from files.

    Reads the .xyz file for atomic coordinates and looks for .velo
    or .velocity.xyz files for velocities. If no velocity file is
    found, zeros are returned.

    Parameters
    ----------
    mol : str
        Molecule name (without extension). The function looks for
        mol.xyz, mol.velo, and mol.velocity.xyz.

    Returns
    -------
    tuple
        (atoms, coord, velo) where atoms is (natom, 1), coord is
        (natom, 3), and velo is (natom, 3).
    """
    with open('%s.xyz' % mol) as xyzfile:
        file = xyzfile.read().splitlines()

    natom = int(file[0])
    atoms = []
    coord = []
    for i, line in enumerate(file[2: 2 + natom]):
        e, x, y, z = line.split()[0:4]
        atoms.append(e)
        coord.append([x, y, z])

    if os.path.exists('%s.velo' % mol):
        velo = read_float_from_text('%s.velo' % mol)

    elif os.path.exists('%s.velocity.xyz' % mol):
        velo = read_float_from_text('%s.velocity.xyz' % mol)

    else:
        velo = np.zeros([natom, 3])

    atoms = np.array(atoms).reshape((-1, 1))
    coord = np.array(coord).astype(float)

    return atoms, coord, velo


def read_xyz_frame(path, atoms=None, lattice_required=False):
    """Read one XYZ frame, and the cell its comment line declares.

    The counterpart of what this framework writes: a frame's second line may
    carry the extended-XYZ `Lattice="..."` key that
    `workflow/records/output.py::lattice_comment` appends to a periodic frame,
    and a reader that ignores it puts a molecule sitting on two sides of a cell
    face back into a box it never had.

    Parameters
    ----------
    path : str
        The XYZ file. One frame: the count, the comment, then the atoms.
    atoms : sequence of str, optional
        The element symbols this frame has to carry, in order. A band whose
        endpoints were read on another order would be interpolated between two
        different molecules, so the mismatch is refused rather than sorted out.
    lattice_required : bool
        Whether the comment line has to declare a cell. A variable-cell band
        interpolates the lattice between its endpoints, so an endpoint without
        one has no ramp to sit on.

    Returns
    -------
    tuple
        `(atoms, coordinates, lattice)`: symbols as read, `(natom, 3)`
        coordinates in Angstrom, and the `(3, 3)` cell in Angstrom or None.

    Raises
    ------
    ValueError
        When the file is not one XYZ frame, when the atom order differs from
        `atoms`, or when a cell was required and the frame declares none.
    """
    if not os.path.isfile(path):
        raise ValueError(
            'the structure file %r does not exist' % (path,))
    with open(path) as handler:
        lines = handler.read().splitlines()
    if len(lines) < 2:
        raise ValueError(
            'the structure file %r is not an XYZ frame: %d line(s)'
            % (path, len(lines)))
    try:
        count = int(lines[0].split()[0])
    except (IndexError, ValueError):
        raise ValueError(
            'the structure file %r does not start with an atom count'
            % (path,)) from None
    body = lines[2:2 + count]
    if len(body) != count:
        raise ValueError(
            'the structure file %r declares %d atoms and carries %d'
            % (path, count, len(body)))
    symbols = []
    coordinates = []
    for line in body:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(
                'the structure file %r carries a line that is not '
                '"symbol x y z": %r' % (path, line))
        symbols.append(str(fields[0]))
        coordinates.append([float(fields[1]), float(fields[2]),
                            float(fields[3])])
    if atoms is not None:
        wanted = [str(name) for name in atoms]
        if symbols != wanted:
            raise ValueError(
                'the structure file %r holds %s and this run is built on %s: a '
                'path interpolated between two different atom orders is not a '
                'path' % (path, ' '.join(symbols), ' '.join(wanted)))
    lattice = read_lattice_comment(lines[1])
    if lattice_required and lattice is None:
        raise ValueError(
            'the structure file %r declares no cell and this run relaxes the '
            'box: a variable-cell band interpolates the lattice between its two '
            'endpoints, so both have to carry the Lattice="..." comment line '
            'the framework writes' % (path,))
    return symbols, np.asarray(coordinates, dtype=np.float64), lattice


def read_lattice_comment(comment):
    """The cell one frame's comment line declares, or None.

    Nine Angstrom entries in row-major order, the layout
    `records.lattice_comment` writes. A comment without the key, or with a
    count that is not nine, is None: a partial cell is not a cell, and reading
    three of nine entries as a lattice would put the atoms in a box nobody
    declared.
    """
    match = _LATTICE_KEY.search(str(comment or ''))
    if match is None:
        return None
    try:
        values = [float(value) for value in match.group(1).split()]
    except ValueError:
        return None
    if len(values) != 9:
        return None
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def read_charge(mol):
    """Read point charges from a .charge file.

    Parameters
    ----------
    mol : str
        Molecule name (without extension). Looks for mol.charge.

    Returns
    -------
    np.ndarray
        Point charges array, shape (ncharge, 4) as [q, x, y, z].
    """
    if not os.path.exists('%s.charge' % mol):
        sys.exit('\n  FileNotFoundError\n  PyRAI3MD: looking for charge file %s.charge' % mol)

    with open('%s.charge' % mol) as charge_file:
        file = charge_file.read().splitlines()

    ncharge = int(file[0])
    charges = []
    for i, line in enumerate(file[2: 2 + ncharge]):
        q, x, y, z = line.split()[0:4]
        charges.append([q, x, y, z])

    charges = np.array(charges).astype(float)

    return charges


def read_initcond(mol):
    """Read coordinates and velocities from initial condition list.

    Parameters
    ----------
    mol : list
        List of atom specifications, each containing at minimum
        [element, x, y, z] and optionally [vx, vy, vz, mass, charge].

    Returns
    -------
    tuple
        (atoms, coord, velo) arrays.
    """
    natom = len(mol)
    atoms = []
    coord = []
    velo = np.zeros((natom, 3))
    for i, line in enumerate(mol):

        if len(line) >= 9:
            e, x, y, z, vx, vy, vz, m, chrg = line[0:9]
            atoms.append(e)
            coord.append([x, y, z])
            velo[i, 0: 3] = float(vx), float(vy), float(vz)
        elif 9 > len(line) >= 7:
            e, x, y, z, vx, vy, vz = line[0:7]
            atoms.append(e)
            coord.append([x, y, z])
            velo[i, 0: 3] = float(vx), float(vy), float(vz)
        else:
            e, x, y, z = line[0:4]
            atoms.append(e)
            coord.append([x, y, z])
            velo[i, 0: 3] = 0.0, 0.0, 0.0

    atoms = np.array(atoms).reshape((-1, 1))
    coord = np.array(coord).astype(float)

    return atoms, coord, velo


def read_float_from_text(txt):
    """Read float data from a text file.

    Handles Fortran-style 'D' exponent notation by replacing 'D' with 'e'.

    Parameters
    ----------
    txt : str
        Path to the text file.

    Returns
    -------
    np.ndarray
        Parsed float array.
    """
    with open(txt, 'r') as ld_input:
        ftxt = ld_input.read().splitlines()

    flist = []
    for fx in ftxt:
        fx = fx.replace('D', 'e')
        flist.append(fx.split())

    farray = np.array(flist).astype(float)

    return farray


def lookup_amber(name):
    """Look up the element symbol for an AMBER atom type.

    Parameters
    ----------
    name : str
        AMBER atom type name.

    Returns
    -------
    str
        Element symbol.
    """
    amber_dict = {
        'C': 'C',
        'CT': 'C',
        'CA': 'C',
        'CM': 'C',
        'CC': 'C',
        'CV': 'C',
        'CW': 'C',
        'CR': 'C',
        'CB': 'C',
        'C*': 'C',
        'CN': 'C',
        'CK': 'C',
        'CQ': 'C',
        'C2R': 'C',
        'C3R': 'C',
        'N': 'N',
        'NA': 'N',
        'NB': 'N',
        'NC': 'N',
        'N*': 'N',
        'N2': 'N',
        'N3': 'N',
        'O': 'O',
        'OW': 'O',
        'OH': 'O',
        'OS': 'O',
        'OT': 'O',
        'O2': 'O',
        'S': 'S',
        'SH': 'S',
        'P': 'P',
        'H': 'H',
        'HW': 'H',
        'HO': 'H',
        'HR': 'H',
        'HS': 'H',
        'HT': 'H',
        'HA': 'H',
        'HC': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'H4': 'H',
        'H5': 'H',
        'HP': 'H',
        'LAH': 'H',
        'Cl-': 'Cl',
        'F': 'F'
    }

    if name in amber_dict.keys():
        atom = amber_dict[name]
    else:
        atom = name[0]
        print('Do not find atom type %s, use %s instead' % (name, atom))

    return atom


def atomic_number(x):
    """Convert element symbols to atomic numbers.

    Parameters
    ----------
    x : list of str
        List of element symbols.

    Returns
    -------
    list of int
        List of atomic numbers.
    """
    z = [PERIODIC_TABLE[atom] for atom in x]
    return z


def numerize_xyz(xyz):
    """Convert element symbols in coordinate lists to atomic numbers.

    Parameters
    ----------
    xyz : list
        List of molecules, each containing atoms with [symbol, x, y, z].

    Returns
    -------
    list
        Same structure with element symbols replaced by atomic numbers.
    """
    new_xyz = []
    for mol in xyz:
        new_mol = []
        for atom in mol:
            name, cx, cy, cz = atom
            z = PERIODIC_TABLE[name]
            new_mol.append([int(z), float(cx), float(cy), float(cz)])
        new_xyz.append(new_mol)

    return new_xyz


def mark_atom(xyz, marks):
    """Mark atoms for different basis set specification.

    Parameters
    ----------
    xyz : list
        Coordinate list with [element, x, y, z] entries.
    marks : list
        Replacement labels for each atom.

    Returns
    -------
    list
        Modified coordinate list with marked atoms.
    """
    new_xyz = []

    for n, line in enumerate(xyz):
        e, x, y, z = line
        e = marks[n].split()[0]
        new_xyz.append([e, x, y, z])

    return new_xyz
