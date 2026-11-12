"""
======================================================================
The bridge between a framework state object and the constraint core.

The numeric core of this subsystem takes arrays and an EvaluationContext. The
framework hands over a state object whose accessors have real quirks, and the
merge round is the wrong moment to discover them, so they are read here and
tested against a stub:

- `mass_au` can be None. PyRAIState returns None when the mass array length
  disagrees with natom or when a mass is not positive, rather than raising, so
  every consumer has to decide what to do. For a constraint it is a failure:
  a projected geometry or a SHAKE correction without masses is meaningless.
- the cell is authoritative in `lattice_vector`, and its default is three zero
  rows. `cell` carries the six-parameter spelling whose default is the
  "not filled in" sentinel (99, 99, 99, 90, 90, 90); reading that as a lattice
  would hand a molecule a 99 Angstrom box, which is a bug this project already
  fixed once on the PBC line. So: a lattice exists only when `lattice_vector`
  is non-zero and invertible.
- `current_state` and `nstate` raise for a multi-center run. The constraint core
  never needs them: it works on the global assembled gradient, which is exactly
  what `active_gradient()` exposes for both C=1 and C>1. This module therefore
  must not read them, and a test proves it by making them raise.
- a periodic run is met by capability, not by the presence of a cell. A wall and
  a fix need no image rule and are checked by their own geometric tests; a strict
  constraint is read on the unfolded geometry this module derives once per
  evaluation and leaves on the view; a collective variable over atoms the bond
  graph does not connect stays refused by name (design document section 5.2).
  `System.image` being non-zero is one of the ways that system shows up.

Units: the framework is Angstrom for coordinates and the lattice, atomic units
of mass for masses, Bohr per atomic time for velocities and Hartree per Bohr for
gradients. This module converts lengths to Bohr at the boundary and keeps one
unit system inside.

The module reads a state-shaped object by attribute and imports neither the
state class (`data.state`) nor the config model (`schema.config`): it is handed a
state and returns arrays, which is what keeps the constraint core testable
without a live run. `data.constraint` is a record module and is imported.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from pyrai3md.data.constraint import EvaluationContext
from pyrai3md.utilities.units import ANGSTROM_TO_BOHR
from pyrai3md.utilities.lattice import normalize_pbc
from pyrai3md.workflow.constraints import images
from pyrai3md.workflow.constraints import schedule
from pyrai3md.workflow.constraints import wall

# A lattice whose rows are all below this is the default "no cell" spelling.
CELL_FLOOR = 1.0e-8


class StateAdapterError(ValueError):
    """The state object cannot be read into the constraint core."""


class SystemView(object):
    """One consistent read of a state, in the units the core expects.

    Attributes
    ----------
    atoms : list
        Element symbols in framework order.
    coordinates : numpy.ndarray
        (natom, 3) in Bohr.
    coordinates_angstrom : numpy.ndarray
        The same numbers in Angstrom, kept because the framework's own write
        back is in Angstrom and a round trip must not go through a conversion.
    velocities : numpy.ndarray or None
        (natom, 3) Bohr per atomic time, None when the state carries none.
    masses : numpy.ndarray or None
        (natom,) atomic units of mass.
    natom : int
    cell : numpy.ndarray or None
        (3, 3) row-vector lattice in Bohr, None for a non-periodic system.
    periodic : bool
        True when any periodic axis is on or an atom has crossed a cell face.
    step : int
        Absolute current step from the runtime counter.
    time : float
        Simulation time in atomic units.
    dt : float
        Timestep in atomic units.
    pbc : numpy.ndarray
        The three 0/1 periodicity flags as read, all zeros when the system
        declares none.
    expansion : object or None
        The unfolded geometry `require_usable` read for a periodic system, kept
        on the view so the solve and the fold use the same bond graph the guards
        just passed instead of perceiving a second one. None for a non-periodic
        view.
    """

    __slots__ = ('atoms', 'coordinates', 'coordinates_angstrom', 'velocities',
                 'masses', 'natom', 'cell', 'periodic', 'step', 'time', 'dt',
                 'pbc', 'expansion')

    def __init__(self, atoms, coordinates, velocities, masses, natom, cell,
                 periodic, step, time, dt, pbc=None, expansion=None):
        self.atoms = list(atoms or [])
        self.coordinates_angstrom = coordinates
        self.coordinates = coordinates * ANGSTROM_TO_BOHR
        self.velocities = velocities
        self.masses = masses
        self.natom = int(natom)
        self.cell = cell
        self.periodic = bool(periodic)
        self.step = int(step)
        self.time = float(time)
        self.dt = float(dt)
        self.pbc = np.asarray(pbc if pbc is not None else [0, 0, 0],
                              dtype=np.int64).reshape(-1)
        self.expansion = expansion


def read_state(state, purpose='single', evaluation_id='', trajectory_id=None):
    """Read a state-shaped object into a SystemView and an EvaluationContext.

    Parameters
    ----------
    state : object
        Anything exposing the PyRAIState read surface: `atoms`, `coordinates`,
        `velocities`, `mass_au`, `pmd.system`, `pmd.dynamics`, `runtime`.
    purpose : str
        Why this evaluation happens; see EvaluationContext.
    evaluation_id : str
        Identifies this evaluation, so a retry can be recognised.
    trajectory_id : str, optional
        Defaults to the runtime object's identity, which is per trajectory in a
        batch, so two runs started from one plan cannot spend the same
        deposition token.

    Returns
    -------
    tuple
        (SystemView, EvaluationContext).

    Raises
    ------
    StateAdapterError
        A required accessor is missing or holds a shape that cannot be used.
    """
    if state is None:
        raise StateAdapterError('no state object was given')
    system = _system(state)
    atoms = _as_list(getattr(state, 'atoms', None))
    coordinates = _matrix(getattr(state, 'coordinates', None), 'coordinates')
    if coordinates is None:
        raise StateAdapterError('the state carries no coordinates')
    natom = int(getattr(system, 'natom', 0) or coordinates.shape[0])
    if coordinates.shape[0] != natom:
        raise StateAdapterError(
            'coordinates hold %d atoms but natom is %d'
            % (coordinates.shape[0], natom))
    if atoms and len(atoms) != natom:
        raise StateAdapterError(
            'the atom list holds %d symbols but natom is %d'
            % (len(atoms), natom))

    velocities = _matrix(getattr(state, 'velocities', None), 'velocities')
    if velocities is not None and velocities.shape != coordinates.shape:
        raise StateAdapterError(
            'velocities shape %s does not match coordinates %s'
            % (velocities.shape, coordinates.shape))

    masses = _masses(state, natom)
    cell = _cell(system)
    periodic, flags = _periodic(system, cell)
    step, time, dt = _clock(state)

    view = SystemView(atoms=atoms, coordinates=coordinates,
                      velocities=velocities, masses=masses, natom=natom,
                      cell=cell, periodic=periodic, step=step, time=time,
                      dt=dt, pbc=flags)
    context = EvaluationContext(purpose=purpose, step=step, time=time, dt=dt,
                                 evaluation_id=evaluation_id, cell=cell,
                                 trajectory_id=str(trajectory_id
                                                  if trajectory_id is not None
                                                  else _default_trajectory(state)))
    return view, context


# Which capabilities need an image rule to say anything at all.
#
# A wall measures one point against one declared center: an atom's distance past
# a surface, a group center against a sphere. None of that asks which copy of the
# cell an atom is in, because no separation between two stored atoms is involved.
# A fix masks components. So both work in a periodic cell with no bond graph, and
# the granularity of the periodic guard is the capability, not the presence of a
# cell (user ruling, 2026-09-30).
#
# A strict constraint measures a separation inside a molecule, so it needs the
# unfolded geometry, and this module reads it once per evaluation and hands the
# same table to the solve through `SystemView.expansion`. A bias or a history
# potential is asked the same question per variable rather than refused per kind:
# a cv whose atoms the bond graph connects has a periodic image and is served on
# the unfolded geometry, and one that straddles two molecules - a centre-of-mass
# distance between two waters - has none, and `_validate_images` refuses it there
# by name. Refusing the whole kind would have thrown away the served half to
# avoid inventing a rule for the other half.
IMAGE_INDEPENDENT = ('wall', 'fix')


def require_usable(view, plan):
    """Refuse a combination the core cannot honour, before any work is done.

    Returns the view unchanged so a caller can chain it. An empty plan asks for
    nothing at all: that is the DC-1 path, and no check here runs for it.
    """
    if plan is None or plan.is_empty:
        return view
    if view.periodic:
        # One read of the bond graph per evaluation, kept on the view: the
        # guards below, the solve and the fold all use it, and a second
        # perception would be a second answer. The image rules are then asked
        # per variable, not per action kind: a cv inside one connected molecule
        # has a periodic image and is served, and the one that straddles two
        # molecules is refused by name where it is read.
        view.expansion = images.read_expansion(
            view, focus=_image_focus(plan, view.natom))
        _validate_images(view, plan)
    if _needs_masses(plan) and view.masses is None:
        raise StateAdapterError(
            'this plan needs per-atom masses and the state could not supply '
            'them (mass_au is None: the mass array is missing, the wrong '
            'length, or not positive)')
    highest = _highest_index(plan)
    if highest >= view.natom:
        raise StateAdapterError(
            'the plan references atom %d but the system has %d atoms; the plan '
            'was built for a different structure'
            % (highest + 1, view.natom))
    return view


def _image_focus(plan, natom):
    """The atoms whose connected components the plan actually measures.

    A strict constraint unfolds the molecule its rows sit in, a collective
    variable measures a separation inside one molecule too, and a centroid wall
    needs its group's center; nothing else in a plan reads a separation between
    two stored atoms. Naming those atoms lets the bond graph be built outward
    from them instead of over every pair in the box.
    """
    focus = set()
    for spec in plan.actions:
        if spec.kind == 'constraint':
            for row in _condition_rows(spec):
                focus.update(int(atom) for atom in row)
        elif spec.kind in ('bias', 'metadynamics'):
            for indices in _variable_atoms(spec, plan):
                focus.update(indices)
        elif spec.kind == 'wall':
            if spec.parameters.get('center') is not None:
                continue
            if spec.atoms is not None and len(spec.atoms):
                focus.update(int(atom) for atom in spec.atoms)
            elif spec.group is not None:
                members = plan.groups.members(spec.group)
                focus.update(int(atom) for atom in np.asarray(members).reshape(-1))
    return sorted(index for index in focus if 0 <= index < natom)


def _variable_atoms(spec, plan):
    """The atom lists one action measures, each as 0-based indices.

    A bias carries one named cv, a gaussian history potential can carry several,
    and an aligned RMSD names its block on the action. They come back in the same
    shape so one question can be asked of all of them: does the bond graph
    connect this set, which is what makes a periodic image of the variable exist.
    """
    out = []
    for name in _variable_names(spec):
        record = plan.cvs.get(str(name))
        if record is not None:
            out.append([int(atom) for atom in
                        np.asarray(record.indices).reshape(-1)])
    if not out and spec.atoms is not None and len(spec.atoms):
        out.append([int(atom) for atom in spec.atoms])
    return out


def _variable_names(spec):
    """The cv names one action is written in.

    An aligned RMSD names none: its variable is the deviation of the atoms the
    action itself carries, so it comes back empty and `_variable_atoms` falls
    through to `spec.atoms`.
    """
    if spec.cv:
        return [str(spec.cv)]
    return [str(name) for name in (spec.parameters.get('cvs') or ())]



def _validate_images(view, plan):
    """Pass the periodic plan through the image rules it depends on."""
    expansion = view.expansion
    if expansion is None:
        return
    for spec in plan.actions:
        if spec.kind == 'constraint':
            rows = _condition_rows(spec)
            label = "constraint '%s'" % spec.id
            for row in rows:
                images.require_same_component(expansion, row, label)
            declared = None
            if spec.reference == 'explicit' and spec.target is not None:
                declared = [float(spec.target)] * len(rows)
            images.require_half_box(expansion, rows, label, declared=declared)
        elif spec.kind == 'wall':
            declared = spec.parameters.get('center')
            if declared is not None:
                center = np.asarray(declared, dtype=np.float64)
            elif spec.parameters.get('target_kind', 'atom') != 'atom':
                # A centroid wall's center is the group's own continuous center,
                # which is exactly the thing a wrapped coordinate cannot give.
                center = _wall_center_from_images(spec, plan, view, expansion)
            else:
                # An atom-target wall with no declared center has always meant
                # the origin. In a cell the origin is a face, and
                # `periodic_check` says so rather than inventing a center.
                center = None
            for values in _wall_geometry_variants(spec):
                wall.periodic_check(
                    spec, plan, expansion.coordinates, view.cell,
                    center=center, dt=view.dt, velocities=view.velocities,
                    parameters=values, check_members=False)
            # The declared geometry: this is where "an atom that starts outside
            # is being held out of the wall" applies, because the declaration
            # says what the run starts with and a ramp only moves the surface
            # afterwards. Only while the run has not moved: a one-sided penalty
            # surface is crossed by thermal motion and pushes the atom back, so
            # re-asking the question at a later step would abort a run that is
            # behaving the way a confining wall behaves.
            wall.periodic_check(spec, plan, expansion.coordinates, view.cell,
                                center=center, dt=view.dt,
                                velocities=view.velocities,
                                check_members=view.step == 0)
        elif spec.kind == 'fix':
            continue
        elif spec.kind in ('bias', 'metadynamics'):
            # The variable is measured on the unfolded geometry, so a bond that
            # straddles a face reads as a bond and not as an 8 Bohr break. What
            # cannot be given an image rule is a variable whose atoms the graph
            # does not connect - a distance between two molecules - and that is
            # refused here, naming the action and the atoms, rather than being
            # averaged across the seam.
            for name in _variable_names(spec):
                record = plan.cvs.get(str(name))
                if record is None:
                    continue
                indices = [int(atom) for atom in
                           np.asarray(record.indices).reshape(-1)]
                label = "%s '%s' cv '%s'" % (spec.kind, spec.id, name)
                images.require_same_component(expansion, indices, label)
                if spec.kind == 'bias' and str(record.kind) == 'bond':
                    declared = ([float(spec.target)] if spec.target is not None
                                and spec.reference != 'initial' else None)
                    images.require_half_box(expansion, [indices], label,
                                            declared=declared)
            for indices in _variable_atoms(spec, plan):
                if spec.kind == 'metadynamics' and spec.family == 'rmsd':
                    images.require_same_component(
                        expansion, indices,
                        "metadynamics '%s' rmsd block" % spec.id)


def _wall_geometry_variants(spec):
    """The geometries one periodic wall has to fit, ramp endpoints included.

    `delay` and `stop` bound when a value is reached, not which values exist, so
    a wall whose radius or cell moves is checked against every value the ramp
    produces, not against the one the run happens to start with. A wall with no
    geometry ramp gets the single declared entry, which is what it always was.
    """
    variants = schedule.endpoint_variants(spec, spec.parameters)
    return variants or [None]


def _condition_rows(spec):
    """The atom rows one strict constraint declares."""
    if spec.family == 'shake':
        return np.asarray(spec.pairs, dtype=np.int64).reshape(-1, 2).tolist()
    rows = spec.parameters.get('rows') or []
    return [[int(atom) for atom in row] for row in rows]


def _wall_center_from_images(spec, plan, view, expansion):
    """The center a centroid wall was bound to, taken on the unfolded geometry.

    A wrapped coordinate cannot give this: averaging 0.01 and L-0.01 returns L/2,
    which is a point in the middle of the cell rather than the center of the
    molecule. The unfolded geometry is what makes the anchor mean what its name
    says, and a group the bond graph does not connect is refused there rather
    than averaged across a face.
    """
    members = None
    if spec.atoms is not None and len(spec.atoms):
        members = np.asarray(spec.atoms, dtype=np.int64)
    elif spec.group is not None:
        members = plan.groups.members(spec.group)
    if members is None or not np.size(members):
        raise StateAdapterError(
            "wall '%s' acts on no atom, so its center cannot be read"
            % spec.id)
    weighted = spec.parameters.get('target_kind') == 'com'
    point, _members, _weights = images.group_center(
        expansion, [int(atom) for atom in members],
        masses=view.masses if weighted else None)
    return point


def solve_geometry(view):
    """The geometry the core measures on: unfolded when there is a cell.

    A non-periodic view returns its stored coordinates, which is the same array,
    so nothing about a molecule-in-a-box run changes.
    """
    expansion = getattr(view, 'expansion', None)
    if expansion is None:
        return np.asarray(view.coordinates, dtype=np.float64)
    return expansion.coordinates


def masses_or_raise(view, action_id=''):
    """Masses for an action that cannot run without them."""
    if view.masses is None:
        raise StateAdapterError(
            'action %s needs masses and the state has none'
            % ("'%s'" % action_id if action_id else 'under evaluation'))
    return view.masses


def to_angstrom(coordinates):
    """Convert core coordinates back to the framework's Angstrom spelling.

    The framework writes geometry in Angstrom, so the merge round converts here
    and nowhere else in the write-back path.
    """
    return np.asarray(coordinates, dtype=np.float64) / ANGSTROM_TO_BOHR


def _default_trajectory(state):
    """A stand-in trajectory key, good enough to keep tokens per run.

    Callers that launch several trajectories should pass `trajectory_id`
    explicitly; this exists so a single run is never blocked by the omission.
    """
    runtime = getattr(state, 'runtime', None)
    return id(runtime if runtime is not None else state)


def _system(state):
    pmd = getattr(state, 'pmd', None)
    system = getattr(pmd, 'system', None)
    if system is None:
        raise StateAdapterError(
            'the state carries no pmd.system, so natom and the lattice cannot '
            'be read')
    return system


def _as_list(value):
    if value is None:
        return []
    return list(value)


def _matrix(value, label):
    """A float (n, 3) array, or None when nothing was given."""
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return None
    if array.ndim != 2 or array.shape[1] != 3:
        raise StateAdapterError(
            '%s must be (natom, 3), got shape %s' % (label, array.shape))
    if not np.all(np.isfinite(array)):
        raise StateAdapterError('%s holds a non-finite number' % label)
    return array


def _masses(state, natom):
    """Atomic-unit masses, with the framework's None kept as None."""
    values = getattr(state, 'mass_au', None)
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != natom:
        raise StateAdapterError(
            'mass_au holds %d entries for %d atoms' % (array.size, natom))
    if np.any(array <= 0.0):
        raise StateAdapterError('mass_au holds a non-positive mass')
    return array


def _cell(system):
    """The authoritative lattice in Bohr, or None when there is no lattice.

    `lattice_vector` is the runtime authority in this project; the six-parameter
    `cell` and its 99-sentinel default are deliberately not consulted, because a
    molecule must not inherit a 99 Angstrom box.
    """
    rows = getattr(system, 'lattice_vector', None)
    array = np.asarray(rows, dtype=np.float64) if rows else None
    if array is None or array.size == 0:
        return None
    array = array.reshape(-1)
    if array.size != 9 or np.all(np.abs(array) < CELL_FLOOR):
        return None
    matrix = array.reshape(3, 3) * ANGSTROM_TO_BOHR
    if abs(float(np.linalg.det(matrix))) < 1.0e-12:
        raise StateAdapterError(
            'a lattice is declared but it is singular; the geometry would be '
            'read into a collapsed cell')
    return matrix


def _periodic(system, cell):
    """Whether the run is periodic, and the three flags as read.

    The flags are `utilities/lattice.normalize_pbc`'s reading, the one
    the integrator shares (`integrator/operations._is_periodic`), so a
    scalar, string-ish or three-axis spelling of `pbc` answers the same
    in both places. An atom that has crossed a face still makes the run
    periodic with the flags off -- that override stays, because at that
    point the stored coordinate no longer says where the atom is.
    """
    flags = np.asarray(normalize_pbc(getattr(system, 'pbc', None)),
                       dtype=np.int64)
    periodic = bool(np.any(flags != 0))
    if periodic and cell is None:
        raise StateAdapterError(
            'pbc is on but no lattice is available; a periodic system without a '
            'cell cannot be constrained')
    image = getattr(system, 'image', None)
    if image is not None and np.size(image):
        if np.any(np.asarray(image, dtype=np.int64) != 0):
            periodic = True
    return periodic, flags


def _clock(state):
    runtime = getattr(state, 'runtime', None)
    step = int(getattr(runtime, 'current_step', 0) or 0) if runtime else 0
    time = float(getattr(runtime, 'simulation_time', 0.0)
                 or 0.0) if runtime else 0.0
    dt = float(getattr(state, 'timestep', 0.0) or 0.0)
    return step, time, dt


def _needs_masses(plan):
    """Whether any action in the plan cannot be computed without masses."""
    for spec in plan.actions:
        if spec.kind == 'constraint':
            return True
        if spec.kind == 'metadynamics' and spec.family == 'rmsd' \
                and spec.parameters.get('weight') == 'mass':
            return True
        if spec.kind == 'wall':
            # A 'com' target is a mass-weighted centroid; a spring or a cage
            # around one without masses is not a number this subsystem can
            # invent.
            if spec.parameters.get('target_kind') == 'com':
                return True
        if spec.kind == 'fix':
            # the fix mask itself needs no mass, but the DOF report that comes
            # with it does, and an inconsistent freeze is the defect this
            # subsystem exists to close
            return True
    return False


def _highest_index(plan):
    highest = -1
    for spec in plan.actions:
        for group in (spec.atoms, spec.pairs):
            if group is None or not np.size(group):
                continue
            highest = max(highest, int(np.asarray(group, dtype=np.int64).max()))
        if spec.group is not None:
            members = plan.groups.members(spec.group)
            if len(members):
                highest = max(highest, int(members.max()))
    for record in plan.cvs.values():
        highest = max(highest, int(record.indices.max()))
    return highest
