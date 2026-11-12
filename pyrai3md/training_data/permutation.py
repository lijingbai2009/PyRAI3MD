"""
======================================================================
This module provides the permute-map side of training data: reading the
legacy one-based atom-order file, checking that every row is a real
permutation of the atom list, and expanding a dataset by it.

An atom relabelling moves the atom-indexed channels only. Energies,
spin-orbit values, oscillator strengths and transition dipoles describe
the electronic states, not labelled nuclei, so they are copied as they
stand. No NAC sign is flipped here: the vector's phase is settled by the
training loss, not by a nuclear relabelling.

Oct 23 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from .dataset import TrainingData


def _disabled(path):
    """Whether a permutation-map setting asks for no map."""
    if path is None:
        return True
    return str(path).strip().lower() in ('', 'no', 'none')


def read_permutation_map(path, natom, symbols=None):
    """Read a legacy permutation-map file.

    Parameters
    ----------
    path : str
        Whitespace-separated atom orders, one row per permutation,
        1-based as the legacy format wrote them. '' or 'No' means no map.
    natom : int
        Number of atoms the row must cover.
    symbols : sequence or None
        Element labels of the atoms. When given, every row must preserve
        them; an atom relabelling that swaps different elements is not a
        symmetry operation of the molecule and is refused.

    Returns
    -------
    ndarray or None
        ``(nmap, natom)`` zero-based atom indices, or None when the
        setting asks for no map.

    Raises
    ------
    ValueError
        If a row is not a permutation of the atom list, or it moves an
        atom onto a different element.
    """
    if _disabled(path):
        return None

    raw = np.loadtxt(str(path))
    rows = np.atleast_2d(raw)
    if rows.size == 0:
        raise ValueError("permutation map %s is empty" % path)
    if rows.ndim != 2 or rows.shape[1] != int(natom):
        raise ValueError(
            "permutation map %s must have one column per atom (%d), got "
            "shape %r" % (path, natom, rows.shape))

    rounded = np.rint(rows)
    if not np.allclose(rows, rounded):
        raise ValueError(
            "permutation map %s carries a non-integer atom index" % path)
    indices = rounded.astype(int) - 1

    expected = np.arange(int(natom))
    labels = None if symbols is None else np.asarray(symbols, dtype=object)
    for row_number, index in enumerate(indices, start=1):
        if not np.array_equal(np.sort(index), expected):
            raise ValueError(
                "permutation map %s row %d is not a permutation of the %d "
                "atom indices 1..%d: %r"
                % (path, row_number, natom, natom, (index + 1).tolist()))
        if labels is not None and not np.array_equal(labels[index], labels):
            raise ValueError(
                "permutation map %s row %d changes the element order; an "
                "atom relabelling may only exchange equivalent atoms"
                % (path, row_number))
    return indices


def expand_permutations(dataset, maps):
    """Return a dataset with one copy per map appended to every frame.

    The order is source-frame major: each frame is followed by its own
    permutations before the next source frame. That keeps the copies of
    one geometry together, which is what the trainer's positional
    validation split needs to keep most of a group on one side.

    Parameters
    ----------
    dataset : TrainingData or dict
        Source frames.
    maps : ndarray or None
        ``(nmap, natom)`` zero-based atom orders, as returned by
        :func:`read_permutation_map`. None or an empty array returns the
        source dataset unchanged.

    Returns
    -------
    TrainingData
        The expanded dataset, carrying ``group_id`` (source frame) and
        ``permutation_id`` (0 for the original frame, k for map k).
    """
    if not isinstance(dataset, TrainingData):
        dataset = TrainingData.from_mapping(dataset)
    if maps is None:
        return dataset
    maps = np.asarray(maps, dtype=int)
    if maps.size == 0:
        return dataset

    natom = dataset.natom
    if maps.ndim != 2 or maps.shape[1] != natom:
        raise ValueError(
            "a permutation map must be (nmap, natom=%d), got %r"
            % (natom, maps.shape))
    expected = np.arange(natom)
    for index in maps:
        if not np.array_equal(np.sort(index), expected):
            raise ValueError("a permutation map row is not a permutation "
                             "of the atom indices")
    if dataset.symbols is not None:
        labels = dataset.symbols[0]
        for index in maps:
            if not np.array_equal(labels[index], labels):
                raise ValueError(
                    "a permutation map row changes the element order; an "
                    "atom relabelling may only exchange equivalent atoms")

    nframe, nmap = dataset.nframe, maps.shape[0]
    group = np.repeat(np.arange(nframe), nmap + 1)
    permutation = np.tile(
        np.concatenate(([0], np.arange(1, nmap + 1))), nframe)
    atom_index = np.empty((nframe * (nmap + 1), natom), dtype=int)
    atom_index[0::nmap + 1] = expected
    for offset, index in enumerate(maps, start=1):
        atom_index[offset::nmap + 1] = index

    xyz = np.take_along_axis(dataset.xyz[group],
                             atom_index[:, :, None], axis=1)
    grad = None
    if dataset.grad is not None:
        grad = np.take_along_axis(dataset.grad[group],
                                  atom_index[:, None, :, None], axis=2)
    nac = None
    if dataset.nac is not None:
        nac = np.take_along_axis(dataset.nac[group],
                                 atom_index[:, None, :, None], axis=2)

    def copy(name):
        value = getattr(dataset, name)
        return None if value is None else value[group]

    return TrainingData(
        xyz=xyz,
        symbols=None if dataset.symbols is None else dataset.symbols[group],
        energy=copy('energy'),
        grad=grad,
        nac=nac,
        soc=copy('soc'),
        transition_dipole=copy('transition_dipole'),
        transition_dipole_norm=copy('transition_dipole_norm'),
        osc_strength=copy('osc_strength'),
        charge=copy('charge'),
        cell=copy('cell'),
        pbc=copy('pbc'),
        unit=dataset.unit,
        group_id=group,
        permutation_id=permutation,
        source=dataset.source,
    )