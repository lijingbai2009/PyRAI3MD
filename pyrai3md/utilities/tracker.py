"""
======================================================================
Geometry parameter tracking of the md trajectory.

The legacy GeomTracker read a small declaration and compared one
distance per step against a fixed bound: a pair of atoms ('dist') or two
groups compared through their centers ('frag'). This module keeps that
reading as a pure function of a coordinate array and the declaration.
Stopping the trajectory on the reading is the workflow's decision; what
lives here is only what the number is and whether it is over.

Lengths are in the unit of the coordinates handed in - Angstrom for the
framework's stored geometry - and the comparison is the legacy one:
`value > threshold` reports, equality does not. A threshold list shorter
than the tracked objects is broadcast with its last value, which is what
the legacy reader did; a list with no entry at all is refused, because a
declaration that names an object and no bound would silently monitor
nothing.

The declaration is validated by `plan()`, and the schema calls the same
function when a System model is built, so an input that passes at load
time cannot fail halfway through a trajectory. Atom numbers in the
declaration are 0-based here, as everywhere inside the framework; the
one decrement from the external 1-based spelling happens in the input
adapter.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from . import geometric


# The two shapes the legacy keyword offers. A 'dist' entry is one atom pair;
# a 'frag' declaration is exactly two groups compared through their centers.
TRACK_TYPES = ('dist', 'frag')

# Human-readable unit of the tracked lengths, for the run log and for the
# reason a stop is reported with. The number itself is never converted: it is
# whatever the coordinates carry.
LENGTH_UNIT = 'Angstrom'


class TrackError(ValueError):
    """A tracking declaration or a coordinate set it cannot be read against."""


def _read(source, name):
    """Read one field from a system model or from a plain mapping."""
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def plan(system):
    """The normalized tracking declaration of one run, or None.

    Parameters
    ----------
    system : object or dict
        A System model or the mapping a frame carries, exposing
        `track_type`, `track_index`, `track_thrhd` and `track_stop`.

    Returns
    -------
    dict or None
        ``{'type', 'groups', 'thresholds', 'stop'}`` with 0-based atom
        numbers, or None when the run declares no tracking.

    Raises
    ------
    TrackError
        The declaration cannot be read as written. A type without an
        index or a bound, an index without a type, an unknown type, a
        'dist' entry that is not a pair, a 'frag' declaration that is not
        two groups, and a `track_stop` outside 0/1 are all refused. The
        legacy `track_stop` 2 (suppress surface hopping) is deliberately
        not implemented and is named in the refusal.
    """
    kind = _read(system, 'track_type')
    index = _read(system, 'track_index') or []
    thresholds = _read(system, 'track_thrhd') or []
    stop = _read(system, 'track_stop')
    stop = 0 if stop is None else int(stop)

    if kind is None or not str(kind).strip():
        if index or thresholds or stop:
            raise TrackError(
                'track_index/track_thrhd/track_stop were given without a '
                'track_type; the run would monitor nothing')
        return None

    kind = str(kind).strip().lower()
    if kind not in TRACK_TYPES:
        raise TrackError(
            "unknown track_type %r; expected one of %s"
            % (kind, ', '.join(TRACK_TYPES)))

    groups = [list(group) if isinstance(group, (list, tuple)) else [group]
              for group in index]
    if not groups:
        raise TrackError("track_type '%s' needs track_index" % kind)
    if not thresholds:
        raise TrackError(
            "track_type '%s' needs at least one track_thrhd" % kind)

    if kind == 'frag':
        if len(groups) != 2:
            raise TrackError(
                "track_type 'frag' compares exactly two groups through their "
                'centers, got %d group(s)' % len(groups))
        for number, group in enumerate(groups):
            if not group:
                raise TrackError(
                    'track_index group %d is empty; a fragment center needs '
                    'at least one atom' % (number + 1))
    else:
        for number, group in enumerate(groups):
            if len(group) != 2:
                raise TrackError(
                    "track_index group %d is a 'dist' entry and needs exactly "
                    'two atoms, got %d' % (number + 1, len(group)))

    if stop not in (0, 1):
        raise TrackError(
            'track_stop must be 0 (report) or 1 (stop the trajectory), got %d;'
            ' 2 (suppress surface hopping) is not implemented'
            % stop)

    return {
        'type': kind,
        'groups': [[int(atom) for atom in group] for group in groups],
        'thresholds': [float(value) for value in thresholds],
        'stop': stop,
    }


def threshold_at(thresholds, number):
    """The bound of the number-th tracked object, legacy broadcast included.

    A declaration with fewer thresholds than objects repeats its last value,
    and extra thresholds past the last object are never read. That is the
    legacy reader's rule and it is kept on purpose; an empty list never
    reaches this function because `plan()` refuses it.
    """
    return float(thresholds[min(int(number), len(thresholds) - 1)])


def evaluate(spec, coordinates):
    """Read every tracked parameter at one geometry.

    Parameters
    ----------
    spec : dict or None
        The declaration from `plan()`.
    coordinates : array-like
        (natom, 3) positions in the unit the thresholds are declared in.

    Returns
    -------
    list of dict or None
        One row per tracked object when a declaration exists, None
        otherwise. A row carries the 1-based atom numbers it names (the
        run log prints input numbering), the distance, its threshold and
        the `value > threshold` judgment.

    Raises
    ------
    TrackError
        The geometry is not (natom, 3) or a declared atom number is out
        of range. A silent skip would report a run as tracked when one of
        its objects was never measured.
    """
    if not spec:
        return None
    array = np.asarray(coordinates, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise TrackError(
            'tracked coordinates must have shape (natom, 3), got %r'
            % (array.shape,))
    natom = int(array.shape[0])

    if spec['type'] == 'frag':
        centers = []
        for number, group in enumerate(spec['groups']):
            members = _members(group, natom, 'frag group %d' % (number + 1))
            center, _picked, _weights = geometric.group_center(array, members)
            centers.append(np.asarray(center, dtype=np.float64))
        value = float(np.linalg.norm(centers[0] - centers[1]))
        threshold = threshold_at(spec['thresholds'], 0)
        return [{
            'kind': 'frag',
            'groups': [[int(atom) + 1 for atom in group]
                       for group in spec['groups']],
            'distance': value,
            'threshold': threshold,
            'exceeded': value > threshold,
        }]

    rows = []
    for number, group in enumerate(spec['groups']):
        members = _members(group, natom, 'dist group %d' % (number + 1))
        # The value only: `utilities/cv.distance` also builds the Jacobian and
        # refuses a zero-length pair, but a collapsed pair is exactly what a
        # monitor has to report as 0.0 rather than abort on.
        value = float(np.linalg.norm(array[members[0]] - array[members[1]]))
        threshold = threshold_at(spec['thresholds'], number)
        rows.append({
            'kind': 'dist',
            'atoms': [int(atom) + 1 for atom in members],
            'distance': value,
            'threshold': threshold,
            'exceeded': value > threshold,
        })
    return rows


def exceeded(rows):
    """Whether any tracked parameter is over its bound."""
    return any(bool(row.get('exceeded')) for row in (rows or []))


def stop_reason(rows):
    """The run-log sentence of the first exceeded parameter, or ''."""
    for row in rows or []:
        if not row.get('exceeded'):
            continue
        if row.get('kind') == 'frag':
            what = 'fragment distance %s' % ' / '.join(
                _span(group) for group in row.get('groups') or [])
        else:
            what = 'distance %s' % _span(row.get('atoms') or [])
        return ('%s is %.4f %s, above %.4f %s'
                % (what, float(row['distance']), LENGTH_UNIT,
                   float(row['threshold']), LENGTH_UNIT))
    return ''


def _members(group, natom, label):
    """One group as 0-based indices, with the range refused by name."""
    members = [int(atom) for atom in group]
    for atom in members:
        if atom < 0 or atom >= natom:
            raise TrackError(
                '%s names atom %d, outside the %d-atom system'
                % (label, atom + 1, natom))
    return members


def _span(atoms):
    """A compact 1-based range for the run log: [2, 5] is '2-5'."""
    values = [int(atom) for atom in atoms]
    if not values:
        return '-'
    if len(values) == 1:
        return str(values[0])
    return '%d-%d' % (values[0], values[-1])