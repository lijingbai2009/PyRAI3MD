"""
======================================================================
Explicit atom selection for the system constraint subsystem.

Selection is the step that turns what the user wrote into a plain array of
framework atom indices. Three sources are accepted, and they are mutually
exclusive inside one selector: an explicit index list or range, an element
set, and a named group. An element set is a union of matching atoms; it is
never a chemical molecule, so a group intended to mean "each water" must
list its members explicitly.

Indices arriving from user input are 1-based in every format, including
JSON and YAML. parse_index_list keeps them 1-based and validates them, and
to_zero_based performs the single subtraction that the framework contract
requires. A normalized plan and a restored runtime file are already 0-based
and must never pass through to_zero_based again.

Topology-derived selection is refused rather than guessed. The framework
carries no bond graph: data/structure was removed during the PBC stage A+B
refactor, and the repository's only bond perception routine,
utilities/bonds.py::perceive_bonds, requires a lattice and exists to
seed periodic image counts. A second, competing radius table is prohibited
by the design document (dev/methods/CONSTRAINT_DEVELOPMENT.md section 3.8),
so 'bond_type' and connectivity-derived groups raise UnsupportedSelectionError
and point at that section.

The module must stay a leaf: numpy only, no imports from schema, data,
workflow or skills.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

# Selector keys that need a bond graph the framework does not have. The
# schema recognizes them so a typo is not mistaken for one of them, and the
# resolver refuses them.
UNSUPPORTED_KEYS = ('bond_type', 'molecule', 'connectivity', 'fragment',
                    'by_topology')

# Selector keys accepted by resolve_indices, exactly one per selector.
SUPPORTED_KEYS = ('atoms', 'element', 'group')


class SelectionError(ValueError):
    """Raised when a selector is malformed, empty or out of range."""


class UnsupportedSelectionError(SelectionError):
    """Raised when a selector needs a bond graph the framework lacks."""


def parse_index_list(values, natom=None):
    """Parse 1-based indices written as integers or as 'lo-hi' ranges.

    Parameters
    ----------
    values : int, str, list or tuple
        One index, one 'lo-hi' range, or a mixed list of both. Commas are
        accepted so a single token '1,3,5' behaves as the list [1, 3, 5].
    natom : int, optional
        Atom count used for the range check when given.

    Returns
    -------
    numpy.ndarray
        int64 array of 1-based indices, duplicates preserved in the order
        written so the caller can decide whether repeating is an error.
    """
    if values is None:
        raise SelectionError('empty atom selection')
    if isinstance(values, (int, np.integer)):
        values = [values]
    elif isinstance(values, str):
        values = values.replace(',', ' ').split()
    elif isinstance(values, (list, tuple, np.ndarray)):
        flat = []
        for item in values:
            if isinstance(item, str):
                flat.extend(item.replace(',', ' ').split())
            else:
                flat.append(item)
        values = flat
    else:
        raise SelectionError(
            'unsupported atom selection type: %s' % type(values).__name__)

    indices = []
    for token in values:
        if isinstance(token, (int, np.integer)):
            indices.append(int(token))
            continue
        text = str(token).strip()
        if not text:
            continue
        if '-' in text[1:] or text.startswith('-'):
            head, _, tail = text.partition('-')
            if head and tail:
                start, stop = int(head), int(tail)
                if stop < start:
                    raise SelectionError(
                        'inverted atom index range: %s' % text)
                indices.extend(range(start, stop + 1))
                continue
        try:
            indices.append(int(text))
        except ValueError:
            raise SelectionError(
                'malformed atom index token: %r' % token)

    if not indices:
        raise SelectionError('empty atom selection')
    array = np.asarray(indices, dtype=np.int64)
    if np.any(array < 1):
        raise SelectionError(
            'atom indices are 1-based; got %d' % int(array.min()))
    if natom is not None and np.any(array > natom):
        raise SelectionError(
            'atom index %d exceeds natom %d' % (int(array.max()), int(natom)))
    return array


def to_zero_based(indices, natom=None):
    """Convert a 1-based index array to the framework's 0-based order.

    This is the single place the subtraction happens. Repeated application
    on an already-typed plan is a defect, which is why plans carry 0-based
    arrays and this helper is called only during normalization.
    """
    array = np.asarray(indices, dtype=np.int64).reshape(-1)
    if array.size == 0:
        raise SelectionError('empty atom selection')
    if np.any(array < 1):
        raise SelectionError(
            'index %d is already 0-based or negative' % int(array.min()))
    converted = array - 1
    if natom is not None and np.any(converted >= natom):
        raise SelectionError(
            'atom index %d exceeds natom %d' % (int(array.max()), int(natom)))
    return converted


def resolve_indices(spec, atoms=None, groups=None, natom=None, key='atoms'):
    """Resolve one selector into a 0-based index array.

    Parameters
    ----------
    spec : dict or list
        Either a mapping such as {'atoms': [1, '4-6']}, {'element': ['O']}
        or {'group': 'substrate'}, or a bare index list under `key`.
    atoms : list, optional
        Element symbols in framework order. Required by 'element'.
    groups : dict, optional
        Mapping of group name to a selector resolved the same way. Group
        references are expanded recursively with cycle detection.
    natom : int, optional
        Atom count for the range check. Defaults to len(atoms) when available.
    key : str
        Selector name used when `spec` is a bare list, for error messages.

    Returns
    -------
    numpy.ndarray
        int64 0-based indices in a stable order, duplicates removed.
    """
    if natom is None and atoms is not None:
        natom = len(atoms)

    if not isinstance(spec, dict):
        spec = {key: spec}

    unsupported = [name for name in UNSUPPORTED_KEYS if name in spec]
    if unsupported:
        raise UnsupportedSelectionError(
            "selection source '%s' resolves bonded pairs or connected "
            "components, which this selector cannot use: it needs atoms. "
            "Expand a bond type on a constraint record, or list the atoms "
            "explicitly (dev/methods/CONSTRAINT_DEVELOPMENT.md section "
            "3.8; the graph itself is utilities/bonds.bond_graph)"
            % unsupported[0])

    present = [name for name in SUPPORTED_KEYS if name in spec]
    if not present:
        raise SelectionError(
            'no selection source given; expected one of %s'
            % ', '.join(SUPPORTED_KEYS))
    if len(present) > 1:
        raise SelectionError(
            'selection sources are mutually exclusive in this version: got '
            + ' and '.join("'%s'" % name for name in present))

    name = present[0]
    value = spec[name]

    if name == 'atoms':
        one_based = parse_index_list(value, natom=natom)
        resolved = to_zero_based(one_based, natom=natom)
    elif name == 'element':
        if atoms is None:
            raise SelectionError("'element' selection needs the atom list")
        symbols = _element_list(value)
        lookup = {}
        for position, symbol in enumerate(atoms):
            lookup.setdefault(_canonical(symbol), []).append(position)
        picked = []
        for symbol in symbols:
            hits = lookup.get(symbol, [])
            if not hits:
                raise SelectionError(
                    "'element' selection matches no atom: %s" % symbol)
            picked.extend(hits)
        resolved = np.asarray(picked, dtype=np.int64)
    else:
        resolved = _resolve_group(value, atoms=atoms, groups=groups,
                                  natom=natom, seen=[])

    resolved = _stable_unique(resolved)
    if resolved.size == 0:
        raise SelectionError("'%s' selection expanded to nothing" % name)
    if natom is not None and np.any(resolved >= natom):
        raise SelectionError(
            "'%s' selection references atom %d beyond natom %d"
            % (name, int(resolved.max()) + 1, int(natom)))
    return resolved


def resolve_bond_pairs(pairs, natom=None):
    """Validate explicit bond pairs given as 1-based atom index pairs.

    Explicit pairs are allowed to describe a geometric distance between two
    atoms that are not bonded; only 1-based bounds and self-pairing are
    checked. The returned array is 0-based with each pair ordered
    (smaller, larger), which is what deduplication and the Jacobian builder
    expect.
    """
    if pairs is None:
        raise SelectionError('empty bond list')
    rows = []
    for entry in pairs:
        if isinstance(entry, (int, np.integer, str)):
            entry = [entry]
        items = list(entry)
        if len(items) != 2:
            raise SelectionError(
                'a bond needs two atom indices, got %r' % (entry,))
        one_based = parse_index_list(items, natom=natom)
        zero_based = to_zero_based(one_based, natom=natom)
        if zero_based[0] == zero_based[1]:
            raise SelectionError(
                'self bond is not a distance: atom %d' % int(one_based[0]))
        rows.append((int(min(zero_based)), int(max(zero_based))))
    if not rows:
        raise SelectionError('empty bond list')
    array = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
    order = np.lexsort((array[:, 1], array[:, 0]))
    return array[order]


def resolve_index_rows(rows, width, natom=None, label='selection'):
    """Validate explicit atom rows of a fixed width, keeping their order.

    An angle and a torsion are ordered variables: `1-2-3` and `3-2-1` are the
    same angle only because the middle atom is the vertex, and a dihedral's
    sign flips if the ends are swapped. `resolve_bond_pairs` cannot be reused
    here for exactly that reason: it sorts each pair so that deduplication and
    the Jacobian builder see one canonical order, and sorting a torsion would
    silently un-negate it.

    Parameters
    ----------
    rows : list
        Each entry is `width` 1-based atom indices, as integers, one-element
        strings split by the caller, or a mix.
    width : int
        Indices per row: 3 for an angle, 4 for a torsion.
    natom : int, optional
        Upper bound for the 1-based indices.
    label : str
        Names the declaration in the message.

    Returns
    -------
    np.ndarray
        (n, width) 0-based indices in the order they were given.

    Raises
    ------
    SelectionError
        A row has the wrong length, repeats an atom, or is out of range.
    """
    if rows is None:
        raise SelectionError('%s: empty atom list' % label)
    out = []
    for position, entry in enumerate(rows):
        if isinstance(entry, (int, np.integer, str)):
            entry = [entry]
        items = list(entry)
        if len(items) != int(width):
            raise SelectionError(
                '%s: row %d needs %d atom indices, got %r'
                % (label, position + 1, int(width), entry))
        zero_based = to_zero_based(parse_index_list(items, natom=natom),
                                   natom=natom)
        if len(set(int(value) for value in zero_based)) != len(zero_based):
            raise SelectionError(
                '%s: row %d repeats an atom, which is not a %d atom variable'
                % (label, position + 1, int(width)))
        out.append([int(value) for value in zero_based])
    if not out:
        raise SelectionError('%s: empty atom list' % label)
    return np.asarray(out, dtype=np.int64).reshape(-1, int(width))


def _element_list(value):
    """Normalize an element selector into canonical symbols."""
    if isinstance(value, str):
        value = value.replace(',', ' ').split()
    symbols = []
    for item in value:
        symbol = _canonical(item)
        if not symbol:
            raise SelectionError('empty element symbol')
        symbols.append(symbol)
    if not symbols:
        raise SelectionError("'element' selection is empty")
    return symbols


def _canonical(symbol):
    """Canonicalize an element symbol: 'cl' and 'Cl' become 'Cl'."""
    text = str(symbol).strip()
    if not text:
        return ''
    return text[0].upper() + text[1:].lower()


def _resolve_group(name, atoms, groups, natom, seen):
    """Expand a named group, detecting cycles and unknown names."""
    if groups is None:
        raise SelectionError(
            "group '%s' referenced but no group is defined" % name)
    key = str(name).strip()
    if key in seen:
        raise SelectionError(
            'circular group reference: %s' % ' -> '.join(seen + [key]))
    if key not in groups:
        raise SelectionError(
            "unknown group '%s'; defined groups: %s"
            % (key, ', '.join(sorted(str(k) for k in groups)) or 'none'))
    seen = seen + [key]
    return resolve_indices(groups[key], atoms=atoms, groups=groups,
                           natom=natom) if not isinstance(groups[key], dict) \
        else _resolve_group_spec(groups[key], atoms, groups, natom, seen)


def _resolve_group_spec(spec, atoms, groups, natom, seen):
    """Expand a group whose definition is itself a selector mapping."""
    if 'group' in spec:
        return _resolve_group(spec['group'], atoms=atoms, groups=groups,
                              natom=natom, seen=seen)
    return resolve_indices(spec, atoms=atoms, groups=groups, natom=natom)


def _stable_unique(indices):
    """Sort and drop repeats while keeping an int64 array."""
    array = np.asarray(indices, dtype=np.int64).reshape(-1)
    if array.size <= 1:
        return array
    _, position = np.unique(array, return_index=True)
    return np.sort(array[np.sort(position)])
