"""
======================================================================
This module provides ChargeStore, the per-atom partial-charge store
backing ONIOM electrostatic embedding (docs charge management).

ChargeStore is a pure data + derivation object. It owns a single
global per-atom charge array and answers two kinds of questions:

- which point charges embed a given region (embedding_charges), and
- which point charges describe the whole system for output, logging and
  MD charge files (point_charges).

Only the managed region (highlevel union midlevel) ever carries a
non-zero charge; every other atom stays at 0. The store never decides
WHICH atoms to update or embed -- that routing lives in the workflow
layer (workflow/multiscale/charges.py) which maps task results onto global atom
indices. ChargeStore only holds values and derives [q, x, y, z] rows
from a coordinate array handed to it, so it imports nothing from the
workflow/calculator layer (one-way dependency).

Coordinates are always supplied by the caller in Angstrom; the skill
backend converts to Bohr when it writes its own embedding file. The
store itself is unit-agnostic for coordinates and stores charges in
elementary charge units (e).

Sep 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os

import numpy as np

from ..utilities.units import BOHR_TO_ANGSTROM


# Bumped whenever the serialized charge-store layout changes. Persisted
# with the store so restart can migrate or refuse stale records.
CHARGE_STORE_VERSION = '1.0'


class ChargeStoreError(ValueError):
    """Raised for any charge-store structural/IO inconsistency."""


def read_charge_rows(path):
    """Read a legacy '.charge' file into ``[q, x, y, z]`` rows.

    The format is: first line the charge count, second line a comment,
    then one ``q x y z`` row per charge (coordinates in Angstrom). It is
    shared by ``read_charge`` (per-atom values, only q consumed) and the
    ``external_charges`` channel (all four columns are the external
    point).

    Parameters
    ----------
    path : str
        Path to the charge file.

    Returns
    -------
    list of list of float
        One ``[q, x, y, z]`` row per charge, in file order.

    Raises
    ------
    FileNotFoundError
        When the file does not exist.
    ChargeStoreError
        On a malformed header, a row-count mismatch, a short row, or a
        non-finite value.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            "\n  FileNotFoundError\n  PyRAI3MD: looking for charge "
            "file %s" % path)
    with open(path) as charge_file:
        lines = charge_file.read().splitlines()
    if not lines:
        raise ChargeStoreError("Charge file %s is empty." % path)
    try:
        ncharge = int(lines[0].split()[0])
    except (ValueError, IndexError):
        raise ChargeStoreError(
            "Charge file %s: bad charge count on the first line." % path)
    raw_rows = lines[2:2 + ncharge]
    if len(raw_rows) != ncharge:
        raise ChargeStoreError(
            "Charge file %s declares %d charges but holds %d rows."
            % (path, ncharge, len(raw_rows)))
    rows = []
    for raw in raw_rows:
        fields = raw.split()
        if len(fields) < 4:
            raise ChargeStoreError(
                "Charge file %s row must be 'q x y z': %r."
                % (path, raw))
        try:
            row = [float(value) for value in fields[:4]]
        except ValueError:
            raise ChargeStoreError(
                "Charge file %s has a non-numeric row: %r."
                % (path, raw))
        if not all(np.isfinite(row)):
            raise ChargeStoreError(
                "Charge file %s has a non-finite row: %r."
                % (path, raw))
        rows.append(row)
    return rows


EMBEDDING_KINDS = ('point', 'field')
EMBEDDING_ROLES = ('atom', 'external')


def read_embedding_file(path):
    """Read a `{title}.charge` embedding document.

    One file carries one embedding, and its first line says which:

        point angstrom external      # rows: q x y z, arbitrary external points
        point angstrom atom          # rows: q x y z, one per mid-shell atom
        field au                     # rows: V Fx Fy Fz, one per QM atom

    The declaration is what keeps one keyword from meaning three things: the
    kind, the length unit and (for point charges) whether the rows are tied to
    atoms or float freely are properties of the data, read once here, and the
    regime that consumes them no longer infers them from its own shape.

    `point` rows come back as ``[q, x, y, z]`` in Angstrom; `field` rows as
    ``[V, Fx, Fy, Fz]`` in atomic units. A file that opens with a bare count is
    the pre-declaration layout: it is ``point angstrom`` with no declared
    role, and the regime that consumes it decides what the rows are (the ONIOM
    mid shell or a standalone background field), which is the only reading
    those files ever had. Anything else is refused rather than guessed.

    Parameters
    ----------
    path : str
        Path to the embedding file.

    Returns
    -------
    dict
        ``{'kind', 'unit', 'role', 'rows'}``; ``role`` is empty for a field.

    Raises
    ------
    FileNotFoundError
        When the file does not exist.
    ChargeStoreError
        On an unreadable declaration, a row of the wrong width, a
        non-numeric or non-finite value, or a legacy count that does not
        match the rows it declares.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            "\n  FileNotFoundError\n  PyRAI3MD: looking for embedding "
            "file %s" % path)
    with open(path) as handle:
        lines = handle.read().splitlines()

    declaration = None
    body = []
    for line in lines:
        text = line.strip()
        if not text or text.startswith('#'):
            continue
        if declaration is None:
            declaration = text
            continue
        body.append(text)
    if declaration is None:
        raise ChargeStoreError("Embedding file %s is empty." % path)

    tokens = declaration.split()
    if tokens[0].isdigit():
        # Count-line layout: point charges in Angstrom, and no declared role -
        # the two regimes that read it have always meant different things by
        # the same rows, so this layout keeps that reading and says so.
        return {'kind': 'point', 'unit': 'angstrom', 'role': '',
                'rows': read_charge_rows(path)}

    kind = tokens[0].lower()
    if kind == 'point':
        if len(tokens) != 3:
            raise ChargeStoreError(
                "Embedding file %s: a point declaration is "
                "'point <angstrom|bohr> <atom|external>', got %r."
                % (path, declaration))
        unit = tokens[1].lower()
        role = tokens[2].lower()
        if unit not in ('angstrom', 'bohr'):
            raise ChargeStoreError(
                "Embedding file %s: unsupported length unit %r."
                % (path, tokens[1]))
        if role not in EMBEDDING_ROLES:
            raise ChargeStoreError(
                "Embedding file %s: row role must be atom or external, "
                "got %r." % (path, tokens[2]))
        rows = _embedding_rows(path, body, 4, 'q x y z')
        factor = 1.0 if unit == 'angstrom' else BOHR_TO_ANGSTROM
        return {'kind': 'point', 'unit': unit, 'role': role,
                'rows': [[row[0], row[1] * factor, row[2] * factor,
                          row[3] * factor] for row in rows]}

    if kind == 'field':
        if len(tokens) != 2 or tokens[1].lower() not in ('au', 'a.u.'):
            raise ChargeStoreError(
                "Embedding file %s: a field declaration is 'field au', "
                "got %r." % (path, declaration))
        return {'kind': 'field', 'unit': 'au', 'role': '',
                'rows': _embedding_rows(path, body, 4, 'V Fx Fy Fz')}

    raise ChargeStoreError(
        "Embedding file %s: the first line must declare 'point <unit> "
        "<role>' or 'field au', or be a charge count; got %r."
        % (path, declaration))


def _embedding_rows(path, body, width, layout):
    """Parse fixed-width numeric rows of an embedding document."""
    rows = []
    for raw in body:
        fields = raw.split()
        if len(fields) != width:
            raise ChargeStoreError(
                "Embedding file %s row must be '%s': %r."
                % (path, layout, raw))
        try:
            row = [float(value) for value in fields]
        except ValueError:
            raise ChargeStoreError(
                "Embedding file %s has a non-numeric row: %r." % (path, raw))
        if not all(np.isfinite(row)):
            raise ChargeStoreError(
                "Embedding file %s has a non-finite row: %r." % (path, raw))
        rows.append(row)
    if not rows:
        raise ChargeStoreError("Embedding file %s has no rows." % path)
    return rows


def embedding_region_by_center(mid_centers):
    """Per-center midlevel, the ONIOM electrostatic embedding source.

    ``molecule.highlevel`` and ``molecule.midlevel`` are disjoint input
    regions, so the midlevel of a center is exactly the shell that
    embeds its high-level model. The highlevel charges are stored and
    reported for output, yet never embed the high-level calculation
    (docs charge management).

    This is pure set logic on global atom indices so both the data
    layer (build_state sizing the managed region) and the workflow
    layer (charge routing / embedding) share one definition without a
    circular import.

    Parameters
    ----------
    mid_centers : sequence of sequence of int
        Per-center midlevel global indices.

    Returns
    -------
    dict
        {center_index: [midlevel global indices, in handed order]}.
    """
    return {c: [int(i) for i in (mid or [])]
            for c, mid in enumerate(list(mid_centers or []))}


def managed_region_indices(high_centers, mid_centers):
    """Sorted global indices of the managed region (highlevel ∪ midlevel).

    The managed region is exactly the atoms ChargeStore tracks and
    point_charges() reports: every highlevel atom plus every midlevel
    atom, in global atom order. The two input regions are disjoint.

    Parameters
    ----------
    high_centers : sequence of sequence of int
        Per-center highlevel global indices (H_c).
    mid_centers : sequence of sequence of int
        Per-center midlevel global indices (I_c).

    Returns
    -------
    list of int
        Sorted, duplicate-free global atom indices.
    """
    managed = set()
    for group in (high_centers or []):
        managed.update(int(i) for i in group)
    for group in (mid_centers or []):
        managed.update(int(i) for i in group)
    return sorted(managed)


class ChargeStore:
    """Per-atom partial-charge store for ONIOM electrostatic embedding.

    Parameters
    ----------
    natom : int
        Total number of atoms in the system (global atom count). The
        charge array is sized to this so every global atom index is
        addressable; unmanaged atoms stay at 0.
    managed_indices : sequence of int, optional
        Global atom indices of the managed region (highlevel union
        midlevel). Only these atoms are ever updated; point_charges()
        reports the whole system with the unmanaged atoms at 0.
        Duplicates are dropped and the set is sorted.

    Attributes
    ----------
    natom : int
        Total number of atoms.
    atom_charges : numpy.ndarray
        Shape (natom,) float array of per-atom partial charges (e).
        Only managed-region atoms are ever updated; the rest stay 0.
    managed_indices : numpy.ndarray
        Sorted unique global indices of the managed region.
    """

    def __init__(self, natom, managed_indices=None):
        self.natom = int(natom)
        if self.natom < 0:
            raise ChargeStoreError(
                "ChargeStore natom must be non-negative, got %d."
                % self.natom)
        self.atom_charges = np.zeros(self.natom, dtype=float)
        self.managed_indices = self._normalize_indices(managed_indices)
        self._validate_indices(self.managed_indices, 'managed_indices')
        # Whether any source has written a charge yet. A QMMM run uses
        # the store as the q_qm source only after a seed or a computed
        # charge arrived; an untouched store must not read as zeros.
        self.has_values = False

    @staticmethod
    def _normalize_indices(indices):
        """Return a sorted, duplicate-free int array of global indices."""
        if indices is None:
            return np.zeros(0, dtype=int)
        return np.array(sorted({int(i) for i in indices}), dtype=int)

    def _validate_indices(self, indices, label):
        """Check every index is a valid global atom index."""
        if len(indices) and (indices.min() < 0 or indices.max() >= self.natom):
            raise ChargeStoreError(
                "ChargeStore %s out of range [0, %d): %s."
                % (label, self.natom, indices.tolist()))

    def load_file(self, path, indices):
        """Load static point charges from a legacy '{title}.charge' file.

        The legacy format is: first line the charge count, second line a
        comment, then one 'q x y z' row per charge (coordinates in
        Angstrom). Only the charge column q is consumed; it is written
        onto atom_charges at the given global indices, matched to the
        file rows in order. The file coordinates are ignored -- the
        embedding field always follows the live geometry, while the
        loaded q values stay fixed for the whole run (read_charge).

        Parameters
        ----------
        path : str
            Path to the charge file (run working directory).
        indices : sequence of int
            Global atom indices receiving the file rows, in file order
            (the midlevel indices for a read_charge initialization).

        Returns
        -------
        int
            Number of charges loaded.

        Raises
        ------
        FileNotFoundError
            When the charge file does not exist (legacy semantics).
        ChargeStoreError
            On a malformed file or an index/row-count mismatch.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                "\n  FileNotFoundError\n  PyRAI3MD: looking for charge "
                "file %s" % path)
        document = read_embedding_file(path)
        if document['kind'] != 'point':
            raise ChargeStoreError(
                "Charge file %s declares a field; the mid shell takes point "
                "charges, one row per mid atom." % path)
        if document['role'] == 'external':
            raise ChargeStoreError(
                "Charge file %s declares external points; the mid shell needs "
                "'point <unit> atom' rows in region order." % path)
        rows = document['rows']
        ncharge = len(rows)
        idx = np.array([int(i) for i in indices], dtype=int)
        if idx.shape[0] != ncharge:
            raise ChargeStoreError(
                "Charge file %s holds %d charges but %d target indices "
                "were given." % (path, ncharge, idx.shape[0]))
        values = [row[0] for row in rows]
        self._validate_indices(idx, 'load_file indices')
        self.update(idx, values)
        return ncharge

    def update(self, indices, values):
        """Write per-atom partial charges onto the given global indices.

        Parameters
        ----------
        indices : sequence of int
            Global atom indices to update.
        values : sequence of float
            Partial charges (e), one per index, in the same order.

        Raises
        ------
        ChargeStoreError
            On a length mismatch or an out-of-range index.
        """
        idx = np.array([int(i) for i in indices], dtype=int)
        vals = np.asarray(values, dtype=float).reshape(-1)
        if idx.shape[0] != vals.shape[0]:
            raise ChargeStoreError(
                "ChargeStore.update got %d indices but %d values."
                % (idx.shape[0], vals.shape[0]))
        self._validate_indices(idx, 'update indices')
        self.atom_charges[idx] = vals
        if vals.shape[0]:
            self.has_values = True

    def embedding_charges(self, indices, coords):
        """Build the [q, x, y, z] point-charge rows embedding a region.

        Parameters
        ----------
        indices : sequence of int
            Global atom indices of the region to embed (e.g. one
            center's mid shell).
        coords : numpy.ndarray
            Shape (natom, 3) current coordinates in Angstrom.

        Returns
        -------
        numpy.ndarray
            Shape (len(indices), 4) array of [q, x, y, z] rows in the
            order of indices. q comes from atom_charges; xyz from the
            supplied coordinates. The backend converts Angstrom to Bohr.
        """
        idx = np.array([int(i) for i in indices], dtype=int)
        self._validate_indices(idx, 'embedding indices')
        coords = np.asarray(coords, dtype=float)
        out = np.zeros((idx.shape[0], 4), dtype=float)
        if idx.shape[0]:
            out[:, 0] = self.atom_charges[idx]
            out[:, 1:4] = coords[idx]
        return out

    def point_charges(self, coords):
        """Build the whole-system [q, x, y, z] rows for output/logging.

        Reports every atom in global atom order; atoms outside the
        managed region carry the 0 they were initialized with. This is
        the array synced into System.point_charges and written to the
        MD charge files, so row i is atom i.

        Parameters
        ----------
        coords : numpy.ndarray
            Shape (natom, 3) current coordinates in Angstrom.

        Returns
        -------
        numpy.ndarray
            Shape (natom, 4) array of [q, x, y, z] rows.
        """
        return self.embedding_charges(np.arange(self.natom, dtype=int),
                                      coords)

    def to_dict(self):
        """Serialize the store to a plain JSON-compatible dict."""
        return {
            'version': CHARGE_STORE_VERSION,
            'natom': self.natom,
            'atom_charges': self.atom_charges.tolist(),
            'managed_indices': self.managed_indices.tolist(),
        }

    @classmethod
    def from_dict(cls, data):
        """Rebuild a ChargeStore from its serialized dict form.

        Parameters
        ----------
        data : dict
            Mapping produced by to_dict().

        Returns
        -------
        ChargeStore
            Restored store carrying the persisted charges.
        """
        if not data:
            return None
        store = cls(int(data['natom']), data.get('managed_indices'))
        charges = data.get('atom_charges')
        if charges is not None:
            restored = np.asarray(charges, dtype=float).reshape(-1)
            if restored.shape[0] != store.natom:
                raise ChargeStoreError(
                    "ChargeStore.from_dict got %d charges for natom %d."
                    % (restored.shape[0], store.natom))
            store.atom_charges = restored
            store.has_values = True
        return store
