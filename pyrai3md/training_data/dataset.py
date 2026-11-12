"""
======================================================================
This module provides the framework-level training data container used
by the adaptive-sampling and train workflows.

The container is deliberately independent of PyRAIState and of any
skill: it reads and writes one JSON file, appends frames without
guessing at the caller's intent, and reports every shape disagreement
as an error. The permute-map expansion lives beside it so a skill is
never asked to reorder atoms.

Oct 23 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import os
import tempfile

import numpy as np

FORMAT_NAME = 'pyrai3md-training-data'
FORMAT_VERSION = 1

# Per-frame arrays. Each is optional, but when present every frame must
# carry the same number of leading entries.
_FRAME_FIELDS = (
    'energy',
    'grad',
    'nac',
    'soc',
    'transition_dipole',
    'transition_dipole_norm',
    'osc_strength',
    'charge',
    'cell',
    'pbc',
    'group_id',
    'permutation_id',
)


def _optional_array(raw, key, dtype=np.float64):
    """Return one optional array, or None when the key is empty."""
    value = raw.get(key)
    if value is None:
        return None
    array = np.asarray(value, dtype=dtype)
    return array if array.size else None



def _read_xyz(raw, source):
    """Read geometries and symbols from the legacy or canonical spelling.

    Rows are either ``[x, y, z]`` or ``[symbol, x, y, z]``. Mixing the
    two inside one file is refused: a dataset whose atom labels exist for
    only some frames cannot be checked for element-preserving permutations.
    """
    if 'xyz' not in raw:
        raise ValueError(
            "%s carries no 'xyz' geometries" % source)

    frames = []
    symbols = []
    labelled = None
    for index, frame in enumerate(raw['xyz']):
        coordinates = []
        labels = []
        for row_index, row in enumerate(frame):
            if len(row) >= 4:
                has_label = True
                label = str(row[0])
                tail = row[-3:]
            elif len(row) == 3:
                has_label = False
                label = None
                tail = row
            else:
                raise ValueError(
                    "%s frame %d row %d carries an atom row of %d entries; "
                    "expected [x, y, z] or [symbol, x, y, z]"
                    % (source, index, row_index, len(row)))
            if labelled is None:
                labelled = has_label
            elif labelled != has_label:
                raise ValueError(
                    "%s frame %d row %d mixes labelled and unlabelled atom "
                    "rows; a data file must pick one spelling so "
                    "permutations can be checked for element preservation"
                    % (source, index, row_index))
            try:
                coordinates.append([float(value) for value in tail])
            except (TypeError, ValueError):
                raise ValueError(
                    "%s frame %d carries a non-numeric coordinate: %r"
                    % (source, index, row))
            if has_label:
                labels.append(label)
        frames.append(coordinates)
        symbols.append(labels)

    xyz = np.asarray(frames, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError("%s: xyz must be (nframe, natom, 3)" % source)
    if labelled:
        return xyz, np.asarray(symbols, dtype=object)
    # A file with numeric rows may name the elements once for the whole
    # set. The nn_pes skill reads the same key
    # (skills/nn_pes_skill/service/trainer_data.py), so ignoring it here
    # dropped labels the skill kept and the two readers disagreed on the
    # field set; the shape rule is the skill's.
    declared = raw.get('symbols')
    if declared is not None:
        declared = np.asarray(declared, dtype=object)
        if declared.shape != xyz.shape[:2]:
            raise ValueError(
                "%s: symbols must be (nframe, natom) = %r, got %r"
                % (source, xyz.shape[:2], declared.shape))
        return xyz, declared
    return xyz, None


def _declared_count(raw, key, source='<inline>'):
    value = raw.get(key)
    if value is None or value == '':
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError("%s declares %s=%r, which is not an integer"
                         % (source, key, value))


class TrainingData:
    """One training dataset: frames of geometries and their labels.

    Parameters
    ----------
    xyz : array
        Coordinates, ``(nframe, natom, 3)`` Angstrom.
    symbols : array or None
        Element labels, ``(nframe, natom)``. Every frame must use the same
        atom order.
    energy, grad, nac, soc : array or None
        Scalar labels and their derivatives. ``grad`` is
        ``(nframe, nstate, natom, 3)``, ``nac`` is
        ``(nframe, nnac, natom, 3)``, ``soc`` is ``(nframe, nsoc)`` cm-1.
    transition_dipole : array or None
        ``(nframe, npair, 3)`` vectors; a vector in space, invariant
        under an atom relabelling.
    transition_dipole_norm : array or None
        ``(nframe, npair)`` magnitudes; derived from the vectors when
        the file omits them.
    osc_strength : array or None
        ``(nframe, npair)`` dimensionless oscillator strengths.
    charge, cell, pbc : array or None
        Optional per-frame auxiliary data, carried unchanged.
    unit : dict or None
        The file's self-describing unit block, carried unchanged.
    group_id, permutation_id : array or None
        Provenance of an expanded dataset: the source frame index and
        the applied map index (0 is the original frame).
    source : str
        Where the data came from, for error messages and provenance.
    nstate, nnac, nsoc, npair : int or None
        Declared counts. They are only used to fill a count the arrays
        cannot derive; a declaration that contradicts an array is refused.
    """

    def __init__(self, xyz, symbols=None, energy=None, grad=None, nac=None,
                 soc=None, transition_dipole=None,
                 transition_dipole_norm=None, osc_strength=None,
                 charge=None, cell=None, pbc=None, unit=None,
                 group_id=None, permutation_id=None, source='',
                 nstate=None, nnac=None, nsoc=None, npair=None):
        self.source = str(source or '')
        self.unit = dict(unit) if isinstance(unit, dict) else {}
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self.symbols = None if symbols is None else np.asarray(
            symbols, dtype=object)
        self.energy = self._coerce_2d(energy)
        self.grad = self._coerce_atom_gradient(grad, 'grad')
        self.nac = self._coerce_atom_gradient(nac, 'nac')
        self.soc = self._coerce_2d(soc)
        self.transition_dipole = self._coerce_vector(transition_dipole,
                                                     'transition_dipole')
        self.transition_dipole_norm = self._coerce_2d(
            transition_dipole_norm)
        self.osc_strength = self._coerce_2d(osc_strength)
        self.charge = self._coerce_1d(charge)
        self.cell = self._coerce_aux(cell, 6)
        self.pbc = self._coerce_aux(pbc, 3, dtype=bool)
        self.group_id = self._coerce_1d(group_id, dtype=int)
        self.permutation_id = self._coerce_1d(permutation_id, dtype=int)
        self.nframe, self.natom = self.xyz.shape[0], self.xyz.shape[1]

        self.nstate = self._derive_count('nstate', nstate,
                                         self.energy, self.grad)
        self.nnac = self._derive_count('nnac', nnac, self.nac)
        self.nsoc = self._derive_count('nsoc', nsoc, self.soc)
        self.npair = self._derive_count('npair', npair,
                                        self.transition_dipole,
                                        self.transition_dipole_norm,
                                        self.osc_strength)

        if self.transition_dipole is not None and \
                self.transition_dipole_norm is None:
            self.transition_dipole_norm = np.linalg.norm(
                self.transition_dipole, axis=-1)

        self._validate()

    # -- shape coercion -------------------------------------------------

    @staticmethod
    def _coerce_2d(value):
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        if array.size == 0:
            return None
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("expected a (nframe, nchannel) array, got %r"
                             % (array.shape,))
        return array

    @staticmethod
    def _coerce_1d(value, dtype=np.float64):
        if value is None:
            return None
        array = np.asarray(value, dtype=dtype)
        if array.size == 0:
            return None
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1:
            raise ValueError("expected a per-frame array, got %r"
                             % (array.shape,))
        return array

    @staticmethod
    def _coerce_aux(value, width, dtype=np.float64):
        """A per-frame row of fixed width (cell/pbc).

        A single-frame file may write the row flat; it is lifted to one
        frame here so the frame-axis checks stay uniform.
        """
        if value is None:
            return None
        array = np.asarray(value, dtype=dtype)
        if array.size == 0:
            return None
        if array.ndim == 1 and array.shape[0] == width:
            array = array.reshape(1, width)
        if array.ndim != 2 or array.shape[1] != width:
            raise ValueError("expected a per-frame array with %d values, "
                             "got %r" % (width, array.shape))
        return array

    @staticmethod
    def _coerce_vector(value, name):
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        if array.size == 0:
            return None
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("%s must be (nframe, nchannel, 3), got %r"
                             % (name, array.shape))
        return array

    @staticmethod
    def _coerce_atom_gradient(value, name):
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        if array.size == 0:
            return None
        if array.ndim != 4 or array.shape[3] != 3:
            raise ValueError("%s must be (nframe, nchannel, natom, 3), "
                             "got %r" % (name, array.shape))
        return array

    @staticmethod
    def _derive_count(name, declared, *arrays):
        derived = None
        for array in arrays:
            if array is None:
                continue
            count = int(array.shape[1])
            if derived is not None and count != derived:
                raise ValueError(
                    "%s arrays disagree about the channel count: %d vs %d"
                    % (name, derived, count))
            derived = count
        if declared is not None:
            declared = int(declared)
            if derived is not None and declared != derived:
                raise ValueError(
                    "the data declares %s=%d but its arrays carry %d"
                    % (name, declared, derived))
            return declared
        return 0 if derived is None else derived

    # -- validation -----------------------------------------------------

    def _require_frames(self, value, name):
        if value is not None and int(np.asarray(value).shape[0]) != self.nframe:
            raise ValueError(
                "%s carries %d frames but xyz carries %d"
                % (name, np.asarray(value).shape[0], self.nframe))

    def _validate(self):
        if self.xyz.ndim != 3 or self.xyz.shape[2] != 3:
            raise ValueError("xyz must be (nframe, natom, 3)")
        if self.nframe < 1 or self.natom < 1:
            raise ValueError("a training dataset needs at least one frame "
                             "and one atom")

        for name in _FRAME_FIELDS:
            self._require_frames(getattr(self, name), name)

        if self.symbols is not None:
            if self.symbols.shape != (self.nframe, self.natom):
                raise ValueError(
                    "symbols must be (nframe, natom) = %r, got %r"
                    % ((self.nframe, self.natom), self.symbols.shape))
            first = self.symbols[0]
            for frame in self.symbols[1:]:
                if not np.array_equal(frame, first):
                    raise ValueError(
                        "every frame must use the same atom order; symbols "
                        "differ between frames")

        if self.grad is not None:
            if self.grad.shape[1] != self.nstate or \
                    self.grad.shape[2] != self.natom:
                raise ValueError(
                    "grad must be (nframe, nstate, natom, 3); got %r for "
                    "nstate=%d, natom=%d"
                    % (self.grad.shape, self.nstate, self.natom))
        if self.nac is not None:
            if self.nac.shape[1] != self.nnac or \
                    self.nac.shape[2] != self.natom:
                raise ValueError(
                    "nac must be (nframe, nnac, natom, 3); got %r for "
                    "nnac=%d, natom=%d"
                    % (self.nac.shape, self.nnac, self.natom))
        if self.energy is not None and self.energy.shape[1] != self.nstate:
            raise ValueError("energy must be (nframe, nstate)")

        for name, value in (
                ('transition_dipole', self.transition_dipole),
                ('transition_dipole_norm', self.transition_dipole_norm),
                ('osc_strength', self.osc_strength)):
            if value is not None and value.shape[1] != self.npair:
                raise ValueError("%s must carry npair=%d channels, got %d"
                                 % (name, self.npair, value.shape[1]))

        self._check_aux_shape('charge', self.charge)
        self._check_aux_shape('cell', self.cell)
        self._check_aux_shape('pbc', self.pbc)

    def _check_aux_shape(self, name, value):
        if value is None:
            return
        expected = {'charge': (), 'cell': (6,), 'pbc': (3,)}[name]
        if expected and value.ndim == 2 and value.shape[1] == expected[0]:
            return
        if expected and value.ndim == 1 and value.shape[0] == expected[0]:
            return
        if not expected:
            return
        raise ValueError("%s must be a per-frame array with %d values"
                         % (name, expected[0]))

    # -- construction ---------------------------------------------------

    @classmethod
    def from_mapping(cls, raw, source='<inline>'):
        """Build a dataset from a JSON-style mapping.

        Parameters
        ----------
        raw : dict
            Parsed JSON, or the inline payload a skill receives.
        source : str
            Name used in error messages; a file's own 'source' wins.
        """
        if not isinstance(raw, dict):
            raise ValueError("%s does not carry a training-data mapping"
                             % source)
        source = str(raw.get('source') or source)
        xyz, symbols = _read_xyz(raw, source)
        declared_symbols = raw.get('symbols')
        if declared_symbols is not None:
            declared_symbols = np.asarray(declared_symbols, dtype=object)
            if symbols is None:
                symbols = declared_symbols
            elif not np.array_equal(symbols, declared_symbols):
                raise ValueError(
                    "%s carries atom symbols both inside xyz and beside it, "
                    "and the two disagree" % source)
        declared_atoms = _declared_count(raw, 'natom', source)
        if declared_atoms is not None and declared_atoms != xyz.shape[1]:
            raise ValueError(
                "%s declares natom=%d but carries geometries of %d atoms"
                % (source, declared_atoms, xyz.shape[1]))
        return cls(
            xyz=xyz,
            symbols=symbols,
            energy=_optional_array(raw, 'energy'),
            grad=_optional_array(raw, 'grad'),
            nac=_optional_array(raw, 'nac'),
            soc=_optional_array(raw, 'soc'),
            transition_dipole=_optional_array(raw, 'transition_dipole'),
            transition_dipole_norm=_optional_array(
                raw, 'transition_dipole_norm'),
            osc_strength=_optional_array(raw, 'osc_strength'),
            charge=_optional_array(raw, 'charge'),
            cell=_optional_array(raw, 'cell'),
            pbc=_optional_array(raw, 'pbc', dtype=bool),
            unit=raw.get('unit'),
            group_id=_optional_array(raw, 'group_id', dtype=int),
            permutation_id=_optional_array(raw, 'permutation_id', dtype=int),
            source=source,
            nstate=_declared_count(raw, 'nstate'),
            nnac=_declared_count(raw, 'nnac'),
            nsoc=_declared_count(raw, 'nsoc'),
            npair=_declared_count(raw, 'npair'),
        )

    def to_payload(self):
        """Return a JSON-safe mapping of the dataset."""
        payload = {
            'format': FORMAT_NAME,
            'version': FORMAT_VERSION,
            'natom': int(self.natom),
            'nstate': int(self.nstate),
            'nnac': int(self.nnac),
            'nsoc': int(self.nsoc),
            'npair': int(self.npair),
            'xyz': self.xyz.tolist(),
        }
        if self.symbols is not None:
            payload['symbols'] = self.symbols.tolist()
        for name in _FRAME_FIELDS:
            value = getattr(self, name)
            if value is not None:
                payload[name] = value.tolist()
        if self.unit:
            payload['unit'] = dict(self.unit)
        if self.source:
            payload['source'] = self.source
        return payload

    # -- mutation -------------------------------------------------------

    def append(self, other):
        """Append another dataset and return self.

        A dataset is a statement about one molecule and one channel
        layout. Frames are only appended when every one of those
        statements agrees; anything else is refused rather than padded.
        """
        other = _as_dataset(other)
        if (self.natom, self.nstate, self.nnac, self.nsoc, self.npair) != \
                (other.natom, other.nstate, other.nnac, other.nsoc,
                 other.npair):
            raise ValueError(
                "cannot append a dataset with natom/nstate/nnac/nsoc/npair "
                "= %r to one with %r"
                % ((other.natom, other.nstate, other.nnac, other.nsoc,
                    other.npair),
                   (self.natom, self.nstate, self.nnac, self.nsoc,
                    self.npair)))

        self._append_symbols(other)
        self.xyz = np.concatenate((self.xyz, other.xyz), axis=0)

        for name in _FRAME_FIELDS:
            mine = getattr(self, name)
            theirs = getattr(other, name)
            if (mine is None) != (theirs is None):
                raise ValueError(
                    "cannot append: one dataset carries '%s' and the other "
                    "does not" % name)
            if mine is not None:
                setattr(self, name, np.concatenate((mine, theirs), axis=0))

        self._append_unit(other)
        self.nframe = int(self.xyz.shape[0])
        self._validate()
        return self

    def _append_symbols(self, other):
        if (self.symbols is None) != (other.symbols is None):
            raise ValueError(
                "cannot append: one dataset carries atom symbols and the "
                "other does not")
        if self.symbols is None:
            return
        if not np.array_equal(self.symbols[0], other.symbols[0]):
            raise ValueError(
                "cannot append: the two datasets order their atoms "
                "differently; a training set needs one atom order")
        self.symbols = np.concatenate((self.symbols, other.symbols), axis=0)

    def _append_unit(self, other):
        if self.unit and other.unit and self.unit != other.unit:
            raise ValueError(
                "cannot append: the two datasets declare different unit "
                "blocks (%r vs %r)" % (self.unit, other.unit))
        if not self.unit:
            self.unit = dict(other.unit)

    def subset(self, indices, source=None):
        """Return a new dataset carrying the named frames, in that order.

        The receiver is left untouched, so a caller that may reject the
        result (a trial prune of a suspect frame) pays nothing to undo
        it. Indices are frame positions, not group or permutation ids.
        """
        index = np.asarray(indices, dtype=int)
        if index.ndim != 1:
            raise ValueError("subset indices must be a 1-D list, got %r"
                             % (index.shape,))
        if index.size and (index.min() < 0 or index.max() >= self.nframe):
            raise ValueError(
                "subset indices must lie in [0, %d), got [%d, %d]"
                % (self.nframe, int(index.min()), int(index.max())))
        if np.unique(index).size != index.size:
            raise ValueError(
                "subset indices name a frame twice; a training set holds "
                "each frame once")
        values = {
            'xyz': self.xyz[index],
            'symbols': (None if self.symbols is None
                        else self.symbols[index]),
            'unit': dict(self.unit or {}),
            'source': self.source if source is None else source,
        }
        for name in _FRAME_FIELDS:
            value = getattr(self, name)
            values[name] = None if value is None else value[index]
        return TrainingData(
            nstate=self.nstate, nnac=self.nnac, nsoc=self.nsoc,
            npair=self.npair, **values)

    # -- persistence ----------------------------------------------------

    def save(self, path):
        """Write the dataset as canonical JSON, atomically."""
        path = os.fspath(path)
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode='w', dir=directory, prefix='.training-data-',
            suffix='.json', delete=False)
        try:
            with handle:
                json.dump(self.to_payload(), handle)
            os.replace(handle.name, path)
        except BaseException:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise

    def __len__(self):
        return self.nframe

    def __repr__(self):
        return ("TrainingData(nframe=%d, natom=%d, nstate=%d, nnac=%d, "
                "nsoc=%d, npair=%d)"
                % (self.nframe, self.natom, self.nstate, self.nnac,
                   self.nsoc, self.npair))


def _as_dataset(value):
    if isinstance(value, TrainingData):
        return value
    return TrainingData.from_mapping(value)


def load(path):
    """Read one training-data JSON file."""
    path = os.fspath(path)
    with open(path, 'r') as handle:
        raw = json.load(handle)
    return TrainingData.from_mapping(raw, source=path)


def save(dataset, path):
    """Write one dataset to ``path`` (see TrainingData.save)."""
    _as_dataset(dataset).save(path)


def from_payload(mapping, source='<inline>'):
    """Build a dataset from an inline, JSON-safe mapping."""
    return TrainingData.from_mapping(mapping, source=source)
