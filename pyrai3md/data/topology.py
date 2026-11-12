"""
======================================================================
Group and topology records for the constraint subsystem.

A group is a named, ordered list of atom indices. This module stores the
result of resolving the selectors, nothing more: no adjacency, no bond
perception, no connected components. The framework carries no bond graph
(dev/methods/CONSTRAINT_DEVELOPMENT.md section 3.8), so a group whose members
are meant to be "each water" has to list them; what this table can hold is
exactly what a selector can spell out.

The order of a member list is part of the record. Group membership feeds atom
order into every downstream array, so a reshuffle would silently rebind
targets, anchors and per-atom gradients.

Records only. No algorithm here reads or mutates a framework state object.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import hashlib
import json

import numpy as np


class GroupTable(object):
    """A name-to-member table, already resolved to 0-based indices.

    Parameters
    ----------
    entries : dict
        Mapping of group name to a sequence of 0-based atom indices.
    """

    def __init__(self, entries=None):
        self._entries = {}
        for name, members in (entries or {}).items():
            self._entries[str(name)] = self._checked(str(name), members)

    @staticmethod
    def _checked(name, members):
        array = np.asarray(members, dtype=np.int64).reshape(-1)
        if array.size == 0:
            raise ValueError("group '%s' has no members" % name)
        if np.any(array < 0):
            raise ValueError(
                "group '%s' holds a 0-based-violating index %d"
                % (name, int(array.min())))
        if np.unique(array).size != array.size:
            raise ValueError(
                "group '%s' lists an atom twice" % name)
        return array

    def names(self):
        """Group names in insertion order."""
        return list(self._entries.keys())

    def __contains__(self, name):
        return str(name) in self._entries

    def __len__(self):
        return len(self._entries)

    def members(self, name):
        """0-based member indices of one group, as an int64 array."""
        key = str(name)
        if key not in self._entries:
            raise KeyError(
                "unknown group '%s'; defined: %s"
                % (key, ', '.join(self._entries) or 'none'))
        return self._entries[key]

    def add(self, name, members):
        """Insert or replace one group."""
        key = str(name)
        self._entries[key] = self._checked(key, members)
        return self._entries[key]

    def overlaps(self):
        """Pairs of groups that share at least one atom.

        Overlap is legal for additive potentials, whose energies simply sum,
        but the caller must know about it before treating two groups as
        independent molecules.
        """
        names = list(self._entries)
        found = []
        for position, left in enumerate(names):
            for right in names[position + 1:]:
                if np.intersect1d(self._entries[left],
                                  self._entries[right]).size:
                    found.append((left, right))
        return found

    def to_dict(self):
        """Plain-data view, JSON safe."""
        return {name: [int(v) for v in members]
                for name, members in self._entries.items()}

    @classmethod
    def from_dict(cls, data):
        """Rebuild a table from `to_dict` output."""
        return cls(entries=data or {})

    def fingerprint(self):
        """Stable digest of names and member order."""
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def atom_count(groups):
    """Largest index across a GroupTable, or any list of index arrays."""
    highest = -1
    if isinstance(groups, GroupTable):
        for name in groups.names():
            highest = max(highest, int(groups.members(name).max()))
        return highest + 1
    for members in (groups or []):
        array = np.asarray(members, dtype=np.int64).reshape(-1)
        if array.size:
            highest = max(highest, int(array.max()))
    return highest + 1
