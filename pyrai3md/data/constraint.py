"""
======================================================================
Data records of the system constraint subsystem.

Three kinds of persistent record live here, with the ownership split the
design document defines in section 6:

- ConstraintPlan is the immutable numeric plan. It is derived once from the
  normalized configuration and carries explicit 0-based atom indices,
  canonical parameters in internal units, resolved CV and group references,
  the unit and semantic version and a fingerprint. No executable object and
  no cache goes into it.
- ConstraintState belongs to one trajectory. It holds what only the run knows:
  the targets and anchors bound at the reference geometry, the metadynamics
  history, the absolute accepted step, the history version and the deposition
  tokens already spent.
- ConstraintResult is the output of a single evaluation: corrected geometry,
  per-item energies and gradients, the three distinct gradient roles, the
  projection diagnostics and the provenance that ties numbers to geometry and
  history.

Records only. These classes never run an algorithm, never read a framework
state object and never mutate another record in place; the controller and the
potential functions do the work and return new records.

Units
-----
Every stored numeric is internal: length Bohr, angle radian, energy Hartree,
force constant Hartree per Bohr squared (or per radian squared for an angular
variable), wall constant Hartree per Bohr to the power of the wall exponent.
Input units are converted once, in schema normalization, and a record that
carries `units_converted=True` must never be converted again.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import copy
import hashlib
import json

import numpy as np

# Bump when the meaning of a stored field changes, so a restored runtime file
# refuses to be read by an incompatible build.
PLAN_SEMANTIC_VERSION = 1
STATE_SEMANTIC_VERSION = 1

EVALUATION_PURPOSES = ('single', 'md', 'optimization_trial', 'hessian',
                       'structure')

# The five action kinds. A plan that holds none of them is empty, which is the
# single condition every consumer tests before doing any work.
ACTION_KINDS = ('bias', 'wall', 'fix', 'constraint', 'metadynamics')


def _as_index_array(values, name):
    """0-based int64 indices, rejecting an empty or negative list."""
    array = np.asarray(values, dtype=np.int64).reshape(-1)
    if array.size == 0:
        raise ValueError('%s needs at least one atom index' % name)
    if np.any(array < 0):
        raise ValueError(
            '%s holds a negative index; plans are 0-based' % name)
    return array


def _as_pair_array(values, name):
    """0-based (n, 2) index pairs, ordered within each pair."""
    array = np.asarray(values, dtype=np.int64)
    if array.size == 0 or array.ndim != 2 or array.shape[1] != 2:
        raise ValueError('%s must be a non-empty list of atom pairs' % name)
    if np.any(array < 0):
        raise ValueError('%s holds a negative index; plans are 0-based' % name)
    low = np.minimum(array[:, 0], array[:, 1])
    high = np.maximum(array[:, 0], array[:, 1])
    stacked = np.stack((low, high), axis=1)
    if np.any(stacked[:, 0] == stacked[:, 1]):
        raise ValueError('%s contains a self pair' % name)
    order = np.lexsort((stacked[:, 1], stacked[:, 0]))
    return stacked[order]


def _endpoint_value(item):
    """One ramp endpoint as numbers: a scalar, or a vector of them."""
    if isinstance(item, np.ndarray):
        return tuple(_endpoint_value(value) for value in item.tolist())
    if isinstance(item, (list, tuple)):
        return tuple(_endpoint_value(value) for value in item)
    return float(item)


def _tolist(value):
    """JSON-safe view of an array, None or scalar."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_tolist(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _tolist(item) for key, item in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


class EvaluationContext(object):
    """What one evaluation is for, and when it happens.

    A potential must never read a module-level step counter; every time- or
    history-dependent quantity comes from this object, which is also what
    makes a repeated evaluation reproducible.

    Attributes
    ----------
    purpose : str
        One of EVALUATION_PURPOSES.
    step : int
        Absolute accepted step number.
    time : float
        Physical time in the framework's time unit (atomic units).
    dt : float
        Timestep in atomic units, or 0.0 when the workflow has none.
    evaluation_id : str
        Unique id of this evaluation, used to keep logging and retries honest.
    cell : numpy.ndarray or None
        Row-vector lattice of this evaluation, Bohr; None for a non-periodic
        system.
    history_version : int
        Metadynamics history counter at evaluation time.
    trajectory_id : str
        Identifies the trajectory that owns the state, so a batch of trial
        evaluations cannot share a deposition token by accident.
    """

    __slots__ = ('purpose', 'step', 'time', 'dt', 'evaluation_id', 'cell',
                 'history_version', 'trajectory_id')

    def __init__(self, purpose='single', step=0, time=0.0, dt=0.0,
                 evaluation_id='', cell=None, history_version=0,
                 trajectory_id='trajectory'):
        if purpose not in EVALUATION_PURPOSES:
            raise ValueError(
                'unknown evaluation purpose %r; expected one of %s'
                % (purpose, ', '.join(EVALUATION_PURPOSES)))
        self.purpose = purpose
        self.step = int(step)
        self.time = float(time)
        self.dt = float(dt)
        self.evaluation_id = str(evaluation_id)
        self.cell = (None if cell is None
                     else np.asarray(cell, dtype=np.float64))
        self.history_version = int(history_version)
        self.trajectory_id = str(trajectory_id)

    def to_dict(self):
        """Plain-data view."""
        return {'purpose': self.purpose, 'step': self.step,
                'time': self.time, 'dt': self.dt,
                'evaluation_id': self.evaluation_id,
                'cell': _tolist(self.cell),
                'history_version': self.history_version,
                'trajectory_id': self.trajectory_id}

    @classmethod
    def from_dict(cls, data):
        """Rebuild from `to_dict` output."""
        data = data or {}
        return cls(purpose=data.get('purpose', 'single'),
                   step=data.get('step', 0), time=data.get('time', 0.0),
                   dt=data.get('dt', 0.0),
                   evaluation_id=data.get('evaluation_id', ''),
                   cell=data.get('cell'),
                   history_version=data.get('history_version', 0),
                   trajectory_id=data.get('trajectory_id', 'trajectory'))


class CVRecord(object):
    """One resolved scalar collective variable.

    Attributes
    ----------
    id : str
    kind : str
        'bond', 'angle' or 'dihedral'.
    indices : numpy.ndarray
        0-based atom indices, 2, 3 or 4 of them by kind.
    periodic : bool
        True for a dihedral, where differences wrap.
    unit : str
        'bohr' or 'radian', recorded so a mismatch fails loudly.
    """

    KIND_ATOMS = {'bond': 2, 'angle': 3, 'dihedral': 4}

    __slots__ = ('id', 'kind', 'indices', 'periodic', 'unit')

    def __init__(self, id, kind, indices, periodic=False, unit='bohr'):
        if kind not in self.KIND_ATOMS:
            raise ValueError(
                "unknown CV kind %r; expected one of %s"
                % (kind, ', '.join(sorted(self.KIND_ATOMS))))
        array = _as_index_array(indices, "cv '%s'" % id)
        if array.size != self.KIND_ATOMS[kind]:
            raise ValueError(
                "cv '%s' of kind %s needs %d atoms, got %d"
                % (id, kind, self.KIND_ATOMS[kind], array.size))
        if kind == 'dihedral':
            periodic = True
        if unit not in ('bohr', 'radian'):
            raise ValueError("cv '%s' has an unknown unit %r" % (id, unit))
        self.id = str(id)
        self.kind = kind
        self.indices = array
        self.periodic = bool(periodic)
        self.unit = unit

    def to_dict(self):
        """Plain-data view."""
        return {'id': self.id, 'kind': self.kind,
                'indices': _tolist(self.indices), 'periodic': self.periodic,
                'unit': self.unit}

    @classmethod
    def from_dict(cls, data):
        """Rebuild from `to_dict` output."""
        return cls(id=data['id'], kind=data['kind'],
                   indices=data['indices'],
                   periodic=data.get('periodic', False),
                   unit=data.get('unit', 'bohr'))


class ScheduleRecord(object):
    """A parameter ramp keyed on the absolute accepted step.

    Exactly one of `scale` and `value` is given: `scale` multiplies the
    declared parameter, `value` names the absolute endpoints. Endpoints are
    held outside the window, so a step before the start repeats the first
    value instead of extrapolating.

    Attributes
    ----------
    id, kind, parameter, start_step, end_step, scale, value
    time_dependent : bool
        True when the ramp is keyed on physical time rather than steps; the
        first version does not accept it.
    """

    __slots__ = ('id', 'kind', 'parameter', 'start_step', 'end_step',
                 'scale', 'value', 'time_dependent')

    def __init__(self, id, kind='linear', parameter='force_constant',
                 start_step=0, end_step=0, scale=None, value=None,
                 time_dependent=False):
        if kind != 'linear':
            raise ValueError(
                "schedule '%s' has unsupported kind %r; the first version "
                "implements 'linear'" % (id, kind))
        if scale is not None and value is not None:
            raise ValueError(
                "schedule '%s' gives both 'scale' and 'value'; they are "
                "mutually exclusive" % id)
        if scale is None and value is None:
            raise ValueError(
                "schedule '%s' gives neither 'scale' nor 'value'" % id)
        if time_dependent:
            raise ValueError(
                "schedule '%s' is time keyed; only the accepted step is "
                "supported in the first version" % id)
        start = int(start_step)
        end = int(end_step)
        if end <= start:
            raise ValueError(
                "schedule '%s' needs end_step greater than start_step "
                "(%d, %d)" % (id, start, end))
        for name, pair in (('scale', scale), ('value', value)):
            if pair is None:
                continue
            if len(pair) != 2:
                raise ValueError(
                    "schedule '%s' %s needs two endpoints" % (id, name))
        self.id = str(id)
        self.kind = kind
        self.parameter = str(parameter)
        self.start_step = start
        self.end_step = end
        self.scale = None if scale is None else (float(scale[0]),
                                                 float(scale[1]))
        self.value = None if value is None else self._endpoints(id, value)
        self.time_dependent = False

    @staticmethod
    def _endpoints(schedule_id, value):
        """Two endpoints of one shape: a ramp interpolates component by component.

        A geometry parameter is a vector (three semiaxes, three box edges, one
        cell matrix), so both of its endpoints are vectors of the same length.
        The pair is checked here rather than at the first step because a
        mismatched endpoint is a property of the declaration, and a run that
        only fails where the two shapes part company has been told about it
        after the fact.
        """
        low = _endpoint_value(value[0])
        high = _endpoint_value(value[1])
        if isinstance(low, tuple) != isinstance(high, tuple):
            raise ValueError(
                "schedule '%s' ramps a vector against a scalar; an endpoint "
                'has the shape of the parameter it replaces' % schedule_id)
        if isinstance(low, tuple) and len(low) != len(high):
            raise ValueError(
                "schedule '%s' ramps %d numbers against a %d-number endpoint; "
                'an endpoint has the shape of the parameter it replaces'
                % (schedule_id, len(low), len(high)))
        return (low, high)

    def to_dict(self):
        """Plain-data view."""
        return {'id': self.id, 'kind': self.kind,
                'parameter': self.parameter, 'start_step': self.start_step,
                'end_step': self.end_step, 'scale': _tolist(self.scale),
                'value': _tolist(self.value),
                'time_dependent': self.time_dependent}

    @classmethod
    def from_dict(cls, data):
        """Rebuild from `to_dict` output."""
        return cls(id=data['id'], kind=data.get('kind', 'linear'),
                   parameter=data.get('parameter', 'force_constant'),
                   start_step=data.get('start_step', 0),
                   end_step=data.get('end_step', 0), scale=data.get('scale'),
                   value=data.get('value'),
                   time_dependent=data.get('time_dependent', False))


class ConstraintSpec(object):
    """One declared action: a bias, a wall, a fix, a constraint or a history.

    The record is deliberately broad: it stores what an action needs after
    normalization (explicit atoms, a CV reference, canonical parameters) and
    leaves the mathematics to the module that owns the kind. `family` names the
    concrete potential so a plan can be audited without re-reading the input.

    Attributes
    ----------
    id, kind, family
    cv : str or None
        Referenced CV id.
    atoms : numpy.ndarray or None
        0-based atoms the action acts on.
    pairs : numpy.ndarray or None
        0-based (n, 2) atom pairs, for bond kinds.
    group : str or None
        Referenced group name.
    parameters : dict
        Canonical numeric parameters in internal units.
    region : str or None
        'all', 'upper' or 'lower' for a one-sided potential.
    target : float or numpy.ndarray or None
        Canonical target; None when it must be bound at the reference.
    reference : str or None
        'initial' when the target or anchor is captured at binding time.
    schedule : ScheduleRecord or None
        The one parameter ramp, when the action declares exactly one.
    schedules : tuple
        Additional ramps of the same action. A wall whose radius and force
        constant both move carries two, and they are applied in this order;
        `schedule` keeps its meaning as the first entry so every existing
        reader of it stays correct.
    diagnostics : dict
        Solver settings such as tolerance and iteration caps.
    """

    __slots__ = ('id', 'kind', 'family', 'cv', 'atoms', 'pairs', 'group',
                 'parameters', 'region', 'target', 'reference', 'schedule',
                 'schedules', 'diagnostics')

    def __init__(self, id, kind, family, cv=None, atoms=None, pairs=None,
                 group=None, parameters=None, region=None, target=None,
                 reference=None, schedule=None, diagnostics=None,
                 schedules=None):
        if kind not in ACTION_KINDS:
            raise ValueError(
                "spec '%s' has unknown kind %r; expected one of %s"
                % (id, kind, ', '.join(ACTION_KINDS)))
        self.id = str(id)
        self.kind = kind
        self.family = str(family)
        self.cv = None if cv is None else str(cv)
        self.atoms = None if atoms is None else _as_index_array(
            atoms, "spec '%s' atoms" % id)
        self.pairs = None if pairs is None else _as_pair_array(
            pairs, "spec '%s' pairs" % id)
        self.group = None if group is None else str(group)
        self.parameters = dict(parameters or {})
        if region not in (None, 'all', 'upper', 'lower'):
            raise ValueError(
                "spec '%s' has unsupported region %r" % (id, region))
        self.region = region
        self.target = target
        if reference not in (None, 'initial', 'explicit'):
            raise ValueError(
                "spec '%s' has unsupported reference %r" % (id, reference))
        self.reference = reference
        self.schedule = schedule
        extra = tuple(() if schedules is None else tuple(schedules))
        if extra and schedule is None:
            self.schedule, extra = extra[0], extra[1:]
        self.schedules = extra
        self.diagnostics = dict(diagnostics or {})

    def all_schedules(self):
        """Every parameter ramp of this action, in application order."""
        return tuple(() if self.schedule is None else (self.schedule,)) \
            + tuple(self.schedules)

    def needs_reference(self):
        """True when a target or anchor must be captured at binding time."""
        if self.reference == 'initial':
            return True
        if self.kind in ('fix', 'constraint') and self.target is None:
            return True
        return self.kind == 'bias' and self.target is None

    def to_dict(self):
        """Plain-data view."""
        return {'id': self.id, 'kind': self.kind, 'family': self.family,
                'cv': self.cv, 'atoms': _tolist(self.atoms),
                'pairs': _tolist(self.pairs), 'group': self.group,
                'parameters': _tolist(self.parameters), 'region': self.region,
                'target': _tolist(self.target), 'reference': self.reference,
                'schedule': None if self.schedule is None
                else self.schedule.to_dict(),
                'schedules': [record.to_dict() for record in self.schedules],
                'diagnostics': _tolist(self.diagnostics)}

    @classmethod
    def from_dict(cls, data):
        """Rebuild from `to_dict` output."""
        schedule = data.get('schedule')
        extra = data.get('schedules') or []
        return cls(id=data['id'], kind=data['kind'], family=data['family'],
                   cv=data.get('cv'), atoms=data.get('atoms'),
                   pairs=data.get('pairs'), group=data.get('group'),
                   parameters=data.get('parameters'),
                   region=data.get('region'), target=data.get('target'),
                   reference=data.get('reference'),
                   schedule=None if schedule is None
                   else ScheduleRecord.from_dict(schedule),
                   schedules=[ScheduleRecord.from_dict(item)
                              for item in extra] or None,
                   diagnostics=data.get('diagnostics'))


class ConstraintPlan(object):
    """The immutable numeric plan of one run.

    Attributes
    ----------
    version : int
        Semantic version of the stored meaning.
    units : dict
        Record of the input units that were converted, for auditing.
    groups : GroupTable
    cvs : dict
        CV id to CVRecord.
    actions : tuple
        ConstraintSpec records, in declaration order.
    schedules : dict
        Schedule id to ScheduleRecord.
    fingerprint : str
        Digest of the canonical form; a restart refuses a mismatch.
    units_converted : bool
        Guards against a second unit pass on an already-typed plan.
    notes : tuple
        What the input left for the readers to decide: a value broadcast over
        several objects, a ramp whose stop came from the length of the run, a
        wall geometry `wall_type` does not read. The run record prints these.
        They are prose about the spelling of a declaration, not part of its
        numeric content, which is why `to_dict` leaves them out and a reworded
        note cannot break a restart.
    """

    __slots__ = ('version', 'units', 'groups', 'cvs', 'actions', 'schedules',
                 'fingerprint', 'units_converted', 'notes')

    def __init__(self, groups=None, cvs=None, actions=None, schedules=None,
                 units=None, version=PLAN_SEMANTIC_VERSION,
                 units_converted=True, notes=None):
        from pyrai3md.data.topology import GroupTable
        self.version = int(version)
        self.units = dict(units or {})
        self.groups = groups if isinstance(groups, GroupTable) \
            else GroupTable(groups or {})
        self.cvs = dict(cvs or {})
        self.actions = tuple(actions or ())
        self.schedules = dict(schedules or {})
        self.units_converted = bool(units_converted)
        self.notes = tuple(str(item) for item in (notes or ()))
        self.fingerprint = self._fingerprint()
        self._check_references()

    @classmethod
    def empty(cls):
        """The plan of a run that asks for nothing (the DC-1 sentinel).

        This is the single condition every consumer tests before it builds a
        controller, registers a finalizer or touches a history.
        """
        return cls(units_converted=True)

    def _fingerprint(self):
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]

    def _check_references(self):
        for spec in self.actions:
            if spec.cv is not None and spec.cv not in self.cvs:
                raise KeyError(
                    "action '%s' references unknown cv '%s'"
                    % (spec.id, spec.cv))
            if spec.group is not None and spec.group not in self.groups:
                raise KeyError(
                    "action '%s' references unknown group '%s'"
                    % (spec.id, spec.group))
            if spec.schedule is not None and spec.schedule.id \
                    not in self.schedules:
                raise KeyError(
                    "action '%s' references unknown schedule '%s'"
                    % (spec.id, spec.schedule.id))

    @property
    def is_empty(self):
        """True when no action is declared."""
        return len(self.actions) == 0

    def ids(self, kind=None):
        """Action ids in declaration order, optionally filtered by kind."""
        return [spec.id for spec in self.actions
                if kind is None or spec.kind == kind]

    def spec(self, action_id):
        """One action by id, raising when it is absent."""
        for item in self.actions:
            if item.id == str(action_id):
                return item
        raise KeyError("unknown action '%s'" % action_id)

    def to_dict(self):
        """Plain-data, JSON-safe view."""
        return {'version': self.version, 'units': _tolist(self.units),
                'groups': self.groups.to_dict(),
                'cvs': {name: record.to_dict()
                        for name, record in self.cvs.items()},
                'actions': [spec.to_dict() for spec in self.actions],
                'schedules': {name: record.to_dict()
                              for name, record in self.schedules.items()},
                'units_converted': self.units_converted}

    @classmethod
    def from_dict(cls, data):
        """Rebuild from `to_dict` output; the fingerprint recomputes."""
        from pyrai3md.data.topology import GroupTable
        data = data or {}
        plan = cls(
            groups=GroupTable.from_dict(data.get('groups')),
            cvs={name: CVRecord.from_dict(record)
                 for name, record in (data.get('cvs') or {}).items()},
            actions=[ConstraintSpec.from_dict(record)
                     for record in (data.get('actions') or ())],
            schedules={name: ScheduleRecord.from_dict(record)
                       for name, record
                       in (data.get('schedules') or {}).items()},
            units=data.get('units'),
            version=data.get('version', PLAN_SEMANTIC_VERSION),
            units_converted=data.get('units_converted', True))
        return plan

    def __eq__(self, other):
        return isinstance(other, ConstraintPlan) \
            and self.fingerprint == other.fingerprint

    def __ne__(self, other):
        return not self.__eq__(other)


def _checked(token):
    """A deposition token must name a trajectory, a step and an action.

    A None or half-filled key would be stored as an opaque entry that no
    replay can reproduce, which is precisely the double-deposition bug the
    token exists to prevent.
    """
    parts = tuple(token) if token is not None else ()
    if len(parts) != 3 or any(part is None or part == '' for part in parts):
        raise ValueError(
            'a deposition token needs (trajectory_id, accepted_step, '
            'item_id), got %r' % (token,))
    return parts


class HillRecord(object):
    """One deposited metadynamics Gaussian.

    Attributes
    ----------
    center : numpy.ndarray
        CV values at deposition, internal units, one entry per CV.
    width : numpy.ndarray
        Per-CV width, internal units, all positive.
    height : float
        Peak height in Hartree, positive.
    token : tuple
        (trajectory_id, accepted_step, item_id), the idempotency key.
    """

    __slots__ = ('center', 'width', 'height', 'token')

    def __init__(self, center, width, height, token=None):
        centres = np.asarray(center, dtype=np.float64).reshape(-1)
        widths = np.asarray(width, dtype=np.float64).reshape(-1)
        if centres.size == 0:
            raise ValueError('a hill needs at least one CV')
        if widths.size != centres.size:
            raise ValueError(
                'hill width has %d entries for %d CVs'
                % (widths.size, centres.size))
        if np.any(widths <= 0.0):
            raise ValueError('hill widths must be positive')
        if float(height) <= 0.0:
            raise ValueError('hill height must be positive')
        self.center = centres
        self.width = widths
        self.height = float(height)
        self.token = None if token is None else tuple(token)

    def to_dict(self):
        """Plain-data view."""
        return {'center': _tolist(self.center), 'width': _tolist(self.width),
                'height': self.height, 'token': _tolist(self.token)}

    @classmethod
    def from_dict(cls, data):
        """Rebuild from `to_dict` output."""
        return cls(center=data['center'], width=data['width'],
                   height=data['height'], token=data.get('token'))


class ConstraintState(object):
    """Per-trajectory runtime state of the constraint subsystem.

    Everything that only the run knows lives here, and everything in it is
    reproducible from a checkpoint: bound targets and anchors, the reference
    center, the hill list, the absolute accepted step, the history version and
    the deposition tokens already spent. Restoring never re-captures a target
    from the current geometry, which is why the bound values are stored rather
    than recomputed.

    A copy is deep by construction: two trajectories launched from one plan
    must not share hill lists, step counters or writable arrays.
    """

    __slots__ = ('targets', 'anchors', 'reference_center', 'hills',
                 'last_committed_step', 'history_version', 'tokens',
                 'diagnostics')

    def __init__(self, targets=None, anchors=None, reference_center=None,
                 hills=None, last_committed_step=-1, history_version=0,
                 tokens=None, diagnostics=None):
        self.targets = dict(targets or {})
        self.anchors = {} if anchors is None else dict(anchors)
        self.reference_center = None if reference_center is None \
            else np.asarray(reference_center, dtype=np.float64)
        self.hills = list(hills or ())
        self.last_committed_step = int(last_committed_step)
        self.history_version = int(history_version)
        self.tokens = set(tokens or ())
        self.diagnostics = dict(diagnostics or {})

    def copy(self):
        """A deep, independent copy."""
        return copy.deepcopy(self)

    def with_hill(self, hill):
        """Return a new state with one more hill; never mutate in place."""
        if not isinstance(hill, HillRecord):
            raise TypeError('with_hill expects a HillRecord')
        fresh = self.copy()
        fresh.hills.append(hill)
        fresh.history_version = self.history_version + 1
        return fresh

    def spent(self, token):
        """True when this deposition token has already been used."""
        return _checked(token) in self.tokens

    def mark(self, token):
        """Return a new state that records a spent deposition token."""
        fresh = self.copy()
        fresh.tokens.add(_checked(token))
        return fresh

    def to_dict(self):
        """Plain-data view."""
        anchors = {name: _tolist(value)
                   for name, value in self.anchors.items()}
        return {'targets': _tolist(self.targets), 'anchors': anchors,
                'reference_center': _tolist(self.reference_center),
                'hills': [hill.to_dict() for hill in self.hills],
                'last_committed_step': self.last_committed_step,
                'history_version': self.history_version,
                'tokens': sorted(_tolist(token) for token in self.tokens),
                'diagnostics': _tolist(self.diagnostics)}

    @classmethod
    def from_dict(cls, data):
        """Rebuild from `to_dict` output."""
        data = data or {}
        anchors = {}
        for name, value in (data.get('anchors') or {}).items():
            anchors[name] = None if value is None \
                else np.asarray(value, dtype=np.float64)
        tokens = set(tuple(token) if isinstance(token, list) else token
                     for token in (data.get('tokens') or ()))
        return cls(targets=data.get('targets'), anchors=anchors,
                   reference_center=data.get('reference_center'),
                   hills=[HillRecord.from_dict(record)
                          for record in (data.get('hills') or ())],
                   last_committed_step=data.get('last_committed_step', -1),
                   history_version=data.get('history_version', 0),
                   tokens=tokens, diagnostics=data.get('diagnostics'))


class ConstraintStatistics(object):
    """Free degrees of freedom and the kinetic energy that belongs to them.

    `free_dof` is `mobile components - rank(A) - com removals`, so it can be
    zero. It is never clamped to one: a fully frozen system has no temperature
    to report, and pretending otherwise hides the mistake behind a number.
    """

    __slots__ = ('free_dof', 'free_kinetic', 'rank', 'mobile_components',
                 'com_removed', 'residual', 'iterations')

    def __init__(self, free_dof=0, free_kinetic=0.0, rank=0,
                 mobile_components=0, com_removed=0, residual=None,
                 iterations=0):
        self.free_dof = int(free_dof)
        self.free_kinetic = float(free_kinetic)
        self.rank = int(rank)
        self.mobile_components = int(mobile_components)
        self.com_removed = int(com_removed)
        self.residual = residual
        self.iterations = int(iterations)

    def temperature(self, boltzmann_in_hartree):
        """Kinetic temperature, or None when there is nothing to average."""
        if self.free_dof <= 0:
            return None
        return 2.0 * self.free_kinetic / (self.free_dof * boltzmann_in_hartree)

    def to_dict(self):
        """Plain-data view."""
        return {'free_dof': self.free_dof,
                'free_kinetic': self.free_kinetic, 'rank': self.rank,
                'mobile_components': self.mobile_components,
                'com_removed': self.com_removed,
                'residual': _tolist(self.residual),
                'iterations': self.iterations}

    @classmethod
    def from_dict(cls, data):
        """Rebuild from `to_dict` output."""
        data = data or {}
        return cls(free_dof=data.get('free_dof', 0),
                   free_kinetic=data.get('free_kinetic', 0.0),
                   rank=data.get('rank', 0),
                   mobile_components=data.get('mobile_components', 0),
                   com_removed=data.get('com_removed', 0),
                   residual=data.get('residual'),
                   iterations=data.get('iterations', 0))


class ConstraintResult(object):
    """Output of one evaluation, per the field list of the design document.

    The four gradient-like arrays mean different things and are kept apart on
    purpose:

    - `gradient_total` is the true derivative of the physical potential,
      base plus every additive term, at the reported geometry.
    - `gradient_projected` has the frozen and constrained directions removed;
      an optimizer or a diagnostic reads it, and it is generally NOT a
      derivative of the total with respect to Cartesian coordinates.
    - `gradient_for_integrator` is what a Verlet kick uses.
    - `constraint_impulse` is a Lagrange quantity from the position solve. It
      is an impulse, not a force from a potential, and adding it to an energy
      gradient is a category error.

    A base value that was never computed stays None; zeros are never
    manufactured in its place.
    """

    __slots__ = ('coordinates', 'velocities', 'delta_coordinates',
                 'energy_correction', 'gradient_correction', 'components',
                 'energy_base', 'energy_total', 'gradient_base',
                 'gradient_total', 'gradient_projected',
                 'gradient_for_integrator', 'constraint_impulse',
                 'stress_correction', 'diagnostics', 'provenance')

    def __init__(self, coordinates=None, velocities=None,
                 delta_coordinates=None, energy_correction=0.0,
                 gradient_correction=None, components=None,
                 energy_base=None, energy_total=None, gradient_base=None,
                 gradient_total=None, gradient_projected=None,
                 gradient_for_integrator=None, constraint_impulse=None,
                 stress_correction=None, diagnostics=None, provenance=None):
        self.coordinates = _array(coordinates)
        self.velocities = _array(velocities)
        self.delta_coordinates = _array(delta_coordinates)
        self.energy_correction = float(energy_correction or 0.0)
        self.gradient_correction = _array(gradient_correction)
        self.components = dict(components or {})
        self.energy_base = energy_base
        self.energy_total = energy_total
        self.gradient_base = _array(gradient_base)
        self.gradient_total = _array(gradient_total)
        self.gradient_projected = _array(gradient_projected)
        self.gradient_for_integrator = _array(gradient_for_integrator)
        self.constraint_impulse = _array(constraint_impulse)
        self.stress_correction = _array(stress_correction)
        self.diagnostics = dict(diagnostics or {})
        self.provenance = dict(provenance or {})

    @staticmethod
    def empty(context=None, coordinates=None):
        """The result of an evaluation that had nothing to do.

        No (natom, 3) array is allocated, so an empty plan cannot grow a
        history frame or a dispatch payload by carrying empty arrays (DC-3).
        """
        result = ConstraintResult()
        result.coordinates = None if coordinates is None \
            else np.asarray(coordinates, dtype=np.float64)
        result.provenance = {
            'plan_empty': True,
            'evaluation_id': '' if context is None else context.evaluation_id,
            'step': 0 if context is None else context.step,
        }
        return result

    @property
    def is_empty(self):
        """True when no action contributed and nothing was projected."""
        return not self.components and abs(self.energy_correction) == 0.0 \
            and self.gradient_correction is None

    def flat_contributions(self, prefix='constraint'):
        """Flat name-to-number map for the existing `energy_contributions`.

        The reader in utilities/print_results.py expects a plain dict of
        numbers, so the per-item breakdown is flattened as
        `<prefix>.<kind>.<id>`. Subtotals are left out; a consumer must be
        able to sum every row without double counting.
        """
        flat = {}
        for name, item in self.components.items():
            kind = str(item.get('kind', 'item'))
            energy = item.get('energy')
            if energy is None:
                continue
            flat['%s.%s.%s' % (prefix, kind, name)] = float(energy)
        return flat

    def to_dict(self):
        """Plain-data view, recursive so no ndarray survives conversion."""
        return {'coordinates': _tolist(self.coordinates),
                'velocities': _tolist(self.velocities),
                'delta_coordinates': _tolist(self.delta_coordinates),
                'energy_correction': self.energy_correction,
                'gradient_correction': _tolist(self.gradient_correction),
                'components': _tolist(self.components),
                'energy_base': _tolist(self.energy_base),
                'energy_total': _tolist(self.energy_total),
                'gradient_base': _tolist(self.gradient_base),
                'gradient_total': _tolist(self.gradient_total),
                'gradient_projected': _tolist(self.gradient_projected),
                'gradient_for_integrator': _tolist(
                    self.gradient_for_integrator),
                'constraint_impulse': _tolist(self.constraint_impulse),
                'stress_correction': _tolist(self.stress_correction),
                'diagnostics': _tolist(self.diagnostics),
                'provenance': _tolist(self.provenance)}

    @classmethod
    def from_dict(cls, data):
        """Rebuild from `to_dict` output."""
        data = data or {}
        return cls(
            coordinates=data.get('coordinates'),
            velocities=data.get('velocities'),
            delta_coordinates=data.get('delta_coordinates'),
            energy_correction=data.get('energy_correction', 0.0),
            gradient_correction=data.get('gradient_correction'),
            components=data.get('components'),
            energy_base=data.get('energy_base'),
            energy_total=data.get('energy_total'),
            gradient_base=data.get('gradient_base'),
            gradient_total=data.get('gradient_total'),
            gradient_projected=data.get('gradient_projected'),
            gradient_for_integrator=data.get('gradient_for_integrator'),
            constraint_impulse=data.get('constraint_impulse'),
            stress_correction=data.get('stress_correction'),
            diagnostics=data.get('diagnostics'),
            provenance=data.get('provenance'))


def _array(value):
    """float64 ndarray view of `value`, or None."""
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def build_fix_mask(plan, natom):
    """(natom, 3) boolean mask of frozen Cartesian components.

    Overlapping fixes are merged component by component. Two fixes that
    freeze the same component of the same atom are consistent when their
    anchors are the same geometry, which they are by construction: the
    anchor is captured once, at the reference.

    This reads only the plan and returns an array, so it lives with the
    records rather than with the workflow that applies it. The planning
    layer needs the same mask to decide which atoms a curvature search
    may move, and importing workflow for that would invert the layers.
    """
    mask = np.zeros((int(natom), 3), dtype=bool)
    for spec in plan.actions:
        if spec.kind != 'fix':
            continue
        axes = [int(axis) for axis in spec.parameters.get('axes', (0, 1, 2))]
        for position in _members(spec, plan):
            for axis in axes:
                mask[position, axis] = True
    return mask


def _members(spec, plan):
    """Atoms a fix acts on, from the explicit list or the named group."""
    if spec.atoms is not None and len(spec.atoms):
        return [int(position) for position in spec.atoms]
    if spec.group is not None:
        return [int(position) for position in plan.groups.members(spec.group)]
    raise ValueError("fix '%s' acts on no atom" % spec.id)
