"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Nov 15 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Phase maps of the phaseless coupling loss.

A non-adiabatic coupling changes sign with the arbitrary phase of the
electronic states it connects, and the same state phase has to be used in
every pair that state appears in. This module turns that constraint into a
table of sign choices: every row is one legal assignment of state phases,
every column is one input pair, and the entry says whether that pair keeps
its sign or flips it.

The full table is built from the state count alone, from the complete
lexicographic pair list with the first state fixed at +1. A caller that
carries only some pairs selects their columns and drops duplicate rows,
which is why a partial or disconnected pair set needs no separate path.

Rows are cached per (state count, ordered pair list) and must be treated as
read-only: the loss class keeps its own device copy.

"""

from functools import lru_cache
from itertools import product

import numpy as np


def _check_nstate(nstate):
    """Return the state count as an int, refusing one no pair can be built from."""
    nstate = int(nstate)
    if nstate < 2:
        raise ValueError(
            "a phaseless coupling map needs at least two states, got %d"
            % nstate)
    return nstate


def _check_pairs(nstate, pairs):
    """Return the pairs as ordered sorted tuples one level.

    The sign of a pair is s_i * s_j, which does not depend on which end is
    written first, so a declared pair is looked up in sorted order while its
    position in the input list, which is the channel order of the coupling
    tensor, is preserved.

    Parameters
    ----------
    nstate : int
        Number of electronic states.
    pairs : sequence
        Declared state pairs, ordered as the coupling channels are.

    Returns
    -------
    tuple of tuple of int
        One (min, max) entry per input channel, in input order.

    Raises
    ------
    ValueError
        If no pair was declared, a pair is malformed, repeated or outside
        the state ladder.
    """
    if pairs is None:
        raise ValueError("no coupling pair was declared for a phaseless loss")
    ordered = []
    seen = set()
    for entry in pairs:
        if len(entry) != 2:
            raise ValueError(
                "a coupling pair carries two state indices, got %r"
                % (entry,))
        a, b = int(entry[0]), int(entry[1])
        if not (0 <= a < nstate and 0 <= b < nstate):
            raise ValueError(
                "coupling pair %s carries a state index outside the %d "
                "state(s) the model was trained for" % (list(entry), nstate))
        if a == b:
            raise ValueError(
                "coupling pair %s joins a state to itself" % list(entry))
        key = (min(a, b), max(a, b))
        if key in seen:
            raise ValueError(
                "coupling pair %s is declared more than once" % list(key))
        seen.add(key)
        ordered.append(key)
    if not ordered:
        raise ValueError("no coupling pair was declared for a phaseless loss")
    return tuple(ordered)


@lru_cache(maxsize=64)
def full_phase_map(nstate):
    """Build the phase table of the complete pair set.

    The first state is fixed at +1 because flipping every state leaves all
    pair signs unchanged, which cuts the enumeration from 2**nstate to
    2**(nstate-1) without losing a distinct candidate.

    Parameters
    ----------
    nstate : int
        Number of electronic states.

    Returns
    -------
    tuple of tuple of int
        Complete pair list, lexicographic, as ((0, 1), (0, 2), ...).
    numpy.ndarray
        Table of shape (2**(nstate-1), npair): 0 keeps the pair sign, 1
        flips it. The array is cached and must not be modified.
    """
    nstate = _check_nstate(nstate)
    pairs = tuple((i, j) for i in range(nstate)
                  for j in range(i + 1, nstate))

    # One row per assignment of the trailing states; the leading state is 0.
    phases = np.asarray(
        [(0,) + tail for tail in product((0, 1), repeat=nstate - 1)],
        dtype=np.int64)
    first = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    second = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    return pairs, phases[:, first] ^ phases[:, second]


@lru_cache(maxsize=256)
def reduced_phase_map(nstate, pairs):
    """Build the phase table of the pairs a caller actually carries.

    Parameters
    ----------
    nstate : int
        Number of electronic states.
    pairs : tuple of tuple of int
        Declared pairs in channel order. Must be hashable, which is what
        makes it usable as a cache key; a list is converted by the caller.

    Returns
    -------
    numpy.ndarray
        Table of shape (K, npair) with the pair axis in the declared order
        and duplicate rows removed. States that no declared pair touches
        do not multiply the candidate rows, so K is 2**(nstate-q) for q
        connected components. The array is cached and must not be modified.
    """
    nstate = _check_nstate(nstate)
    ordered = _check_pairs(nstate, pairs)
    complete, full = full_phase_map(nstate)
    column = {pair: position for position, pair in enumerate(complete)}
    columns = [column[pair] for pair in ordered]
    return np.unique(full[:, columns], axis=0)
