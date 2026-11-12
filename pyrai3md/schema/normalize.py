"""
======================================================================
This module provides normalization functions for PyRAI3MD configuration
objects. It computes derived quantities from the raw input, such as
atom counts, total electrons, alpha/beta electron distributions,
state counts, and pair counts.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import warnings

import numpy as np

from pydantic import BaseModel

from pyrai3md.utilities import bonds
from pyrai3md.utilities import selection
from pyrai3md.utilities.lattice import cell6_to_matrix
from pyrai3md.utilities.units import (
    ANGSTROM_TO_BOHR,
    KCAL_TO_HARTREE,
    CM_TO_HARTREE,
    EV_TO_HARTREE,
)
from .config import flatten_config_dump, partition_to_submodels
from .config import (Config, DEFAULT, FIELD_TO_SUBMODEL, SUBMODEL_CLASSES,
                     RESTORED_NESTED_SECTIONS,
                     SystemConstraintConfig,
                     ConstraintInputError, UnsupportedCapabilityError)
from .validate import (check_periodic, check_ramp_shape,
                       check_ramp_positive)
from .reference import PERIODIC_TABLE, ATOMIC_MASS
from .validate import JOBTYPE_ALIASES, RETIRED_JOBTYPES
from ..utilities.bonds import derive_initial_image
from ..utilities.lattice import (cell6_to_matrix, matrix_to_cell6,
                                 parse_cell,
                                 cell_volume, normalize_pbc)

def normalize_jobtype(data):
    """Normalize the jobtype string by applying aliases and deduplication.

    This runs before validate_task() so that the validator sees
    canonical names. Only performs normalization (alias replacement,
    lowercasing, deduplication) — legality checks are done by validate.

    Parameters
    ----------
    data : dict
        Configuration dictionary to normalize.

    Returns
    -------
    dict
        Updated dictionary with normalized jobtype string.
    """
    jobtype = data.get('jobtype', 'energy')
    if not isinstance(jobtype, str):
        jobtype = str(jobtype)

    # Lowercase and split
    tokens = jobtype.lower().split()
    if not tokens:
        data['jobtype'] = 'energy'
        return data

    # Apply aliases (keep original token if not in alias map)
    normalized = []
    for token in tokens:
        if token in RETIRED_JOBTYPES:
            # Refused once, here, with the replacement: keeping the token
            # for a later stage to report it as merely unknown loses the
            # sentence that says what to write instead.
            raise ValueError(
                "jobtype '%s' has been removed: %s."
                % (token, RETIRED_JOBTYPES[token]))
        normalized.append(JOBTYPE_ALIASES.get(token, token))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in normalized:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    data['jobtype'] = ' '.join(unique)
    return data


def compute_derived_quantities(data):
    """Compute derived quantities from the configuration data.

    Calculates:
    - natom from the atoms list
    - total_electron from atomic numbers and charge
    - nalpha/nbeta from total_electron and multiplicity
    - nstate from ci (active states)
    - npair, nnac, nsoc from coupling and spin information

    Parameters
    ----------
    data : dict
        Configuration dictionary to normalize.

    Returns
    -------
    dict
        Normalized configuration dictionary with computed fields.
    """
    # Compute natom
    atoms = data.get('atoms', [])
    natom = len(atoms)
    data['natom'] = natom

    # Derive nuclear masses (amu) from the element list when the input
    # does not carry them: the integrators, thermostats/barostats and
    # the kinetic-energy records all read system.mass and silently
    # skip when it is empty (a mass-less MD run never moves). An
    # explicitly provided mass list of the right length always wins;
    # an unknown element symbol leaves the user value untouched so the
    # downstream guards and logs report it.
    mass = data.get('mass') or []
    if natom > 0 and len(mass) != natom:
        derived = []
        for symbol in atoms:
            sym = str(symbol).split('_')[0]  # Strip isotope label
            value = ATOMIC_MASS.get(sym)
            if value is None:
                value = ATOMIC_MASS.get(sym[:1].upper() + sym[1:].lower())
            if value is None:
                derived = None
                break
            derived.append(float(value))
        if derived is not None:
            data['mass'] = derived

    # Compute total_electron from atoms and charge
    charges = data.get('charges', 0)
    multiplicity = data.get('multiplicity', 1)
    # Multi-center: charges/multiplicity may be per-center lists. The
    # global derived electron counts use the full-system L(real)
    # aggregate -- the sum of the center charges and the paired
    # (unpaired-electron cancellation) multiplicity, the same rule
    # data.center.build_center_plan applies (docs MULTICENTER 8). The
    # per-center values stay in System.charges/multiplicity for the
    # center plan; only the derived scalars use the aggregate.
    from ..data.center import aggregate_system_multiplicity
    agg_charge = (sum(int(c) for c in charges)
                  if isinstance(charges, (list, tuple)) else charges)
    agg_multiplicity = (aggregate_system_multiplicity(multiplicity)
                        if isinstance(multiplicity, (list, tuple))
                        else multiplicity)
    if natom > 0:
        total_electron = 0
        for symbol in atoms:
            sym = str(symbol).split('_')[0]  # Strip isotope label
            z = PERIODIC_TABLE.get(sym, PERIODIC_TABLE.get(sym.upper(), 0))
            total_electron += z
        total_electron -= agg_charge  # Subtract charge (positive charge = fewer electrons)
        data['total_electron'] = total_electron

        # Compute nalpha/nbeta from multiplicity
        unpaired = agg_multiplicity - 1
        nalpha = (total_electron + unpaired) // 2
        nbeta = total_electron - nalpha
        data['nalpha'] = nalpha
        data['nbeta'] = nbeta

    # Multi-center (nested ci): the authoritative per-center plan is
    # built by data.center.build_center_plan, validated strictly by
    # schema.validate.validate_centers and bound to PyRAIState (docs
    # R4, r4a). The flat single-valued scalar fields are no longer a
    # first-center projection; they stay at neutral Pydantic-valid
    # defaults so the model validates, and every C>1 consumer reads the
    # per-center electronic_spec through the bound partition plan
    # (PyRAIState.nstate / current_state raise for C>1, so the flat
    # scalars are never consumed as a fake global state axis).
    ci = data.get('ci', [1])
    if _is_nested_list(ci):
        data['nstate'] = 1
        data['state_spin'] = []
        data['npair'] = 0
        data['nnac'] = 0
        data['nsoc'] = 0
        data['transition_pair'] = [[]]
        data['nac_pair_map'] = [[]]
        data['soc_pair_map'] = [[]]
        return _convert_lattice(data)

    # Compute nstate from state_spin (preferred) or ci
    state_spin = data.get('state_spin', [])
    if state_spin:
        # state_spin is explicitly set — use its length
        data['nstate'] = len(state_spin)
    elif ci:
        nstate = sum(ci) if isinstance(ci, list) else ci
        data['nstate'] = nstate

    # Build state_spin from ci and spin (only if not already set)
    spin = data.get('spin', [0])
    if not state_spin:
        if ci and spin and isinstance(ci, list) and isinstance(spin, list):
            if len(ci) != len(spin):
                raise ValueError(
                    "molecule.spin: len(ci)=%d but len(spin)=%d; they "
                    "must match (one spin value per ci state block)."
                    % (len(ci), len(spin)))
            state_spin = []
            for n_states, s in zip(ci, spin):
                two_s = s * 2  # Convert S to 2S (0->0, 1->2, 2->4, ...)
                state_spin.extend([two_s] * n_states)
            data['state_spin'] = state_spin
        elif spin:
            # Fallback: treat spin as-is (already in 2S or single value)
            data['state_spin'] = spin

    # Coupling declaration. One derivation, shared with the per-center
    # plan: `coupling` is the authority and `nac_pair_map` / `soc_pair_map`
    # are its internal labels - which the nested input formats may also
    # carry as an extra entry, in which case they fill `coupling` in and
    # must then agree with it (docs MULTICENTER 4.4.5).
    from ..data.center import derive_pair_declaration
    state_spin = data.get('state_spin', []) or spin
    nstate = int(data.get('nstate', 1) or 1)
    coupling, nac_pairs, soc_pairs = derive_pair_declaration(
        data.get('coupling'), data.get('nac_pair_map'),
        data.get('soc_pair_map'), state_spin, nstate, 'molecule.coupling')
    data['coupling'] = coupling
    data['npair'] = len(coupling)
    data['nnac'] = len(nac_pairs)
    data['nsoc'] = len(soc_pairs)
    data['nac_pair_map'] = [list(pair) for pair in nac_pairs]
    data['soc_pair_map'] = [list(pair) for pair in soc_pairs]

    # The transition channels' pair list is derived here too, once. Every
    # skill used to rebuild it as `[[0, i] for i in range(nstate)]`, which
    # includes the self pair [0, 0] - a "transition dipole of S0 to S0" row
    # that no engine computes and every consumer then reads as zero
    # (#448). A declaration the input makes is kept as it stands.
    declared_transition = [list(pair) for pair in
                           (data.get('transition_pair') or [])
                           if len(pair) == 2]
    for pair in declared_transition:
        if pair[0] == pair[1]:
            raise ValueError(
                "molecule.transition_pair: %r names one state twice; a"
                " transition is between two states, and the row it produced"
                " used to be read as a transition dipole of zero." % (pair,))
    data['transition_pair'] = (declared_transition
                               or [[0, i] for i in range(1, nstate)])

    return _convert_lattice(data)


def _is_nested_list(value):
    """True when value is a non-empty list whose first item is a list.

    Distinguishes the nested per-center form (list[center][...]) from
    the flat single-center form of ci/spin/coupling.
    """
    return (isinstance(value, (list, tuple)) and len(value) > 0
            and isinstance(value[0], (list, tuple)))


def _convert_lattice(data):
    """Normalize the cell representation and derive lattice_vector.

    Representation priority: explicit lattice (6 parameters) over
    cell (9 components or 6 parameters) over a pre-existing
    lattice_vector. The 6-parameter default cell [99, 99, 99, 90, 90,
    90] is the "not filled" sentinel: no lattice_vector is derived
    from it, so a non-periodic run never carries a fake 99 Angstrom
    cubic cell (which downstream would silently look periodic).
    Periodicity is decided solely by pbc, normalized here to exactly
    three 0/1 flags; a conflicting cell/lattice_vector pair is an
    error instead of a silent overwrite. The volume is derived from
    the valid lattice_vector when not explicitly given.

    Shared by the single-center and multi-center derivation paths.
    """
    # pbc: the single normalized periodicity authority (three 0/1).
    if data.get('pbc') is not None:
        data['pbc'] = normalize_pbc(data['pbc'])

    lattice = data.get('lattice', [])
    cell = data.get('cell', [])
    lattice_vector = data.get('lattice_vector') or []

    def _non_zero(matrix):
        try:
            arr = np.asarray(matrix, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        return arr.size > 0 and bool(np.any(arr))

    cell_is_sentinel = (len(cell) == 6
                        and np.allclose(np.asarray(cell, dtype=np.float64),
                                        np.asarray(DEFAULT['cell'])))

    if lattice and len(lattice) == 6:
        data['lattice_vector'] = cell6_to_matrix(lattice).tolist()
    elif cell and len(cell) == 9:
        new_lattice = [[cell[0], cell[1], cell[2]],
                       [cell[3], cell[4], cell[5]],
                       [cell[6], cell[7], cell[8]]]
        if _non_zero(lattice_vector) and not np.allclose(
                np.asarray(lattice_vector, dtype=np.float64),
                np.asarray(new_lattice, dtype=np.float64)):
            raise ValueError(
                "conflicting periodic cell input: the 9-component cell "
                "%r does not match lattice_vector %r" % (cell, lattice_vector))
        data['lattice_vector'] = new_lattice
    elif cell and len(cell) == 6 and not cell_is_sentinel:
        new_lattice = cell6_to_matrix(cell).tolist()
        if _non_zero(lattice_vector):
            back = matrix_to_cell6(lattice_vector)
            if not np.allclose(back, np.asarray(cell, dtype=np.float64),
                               rtol=1e-5, atol=1e-6):
                raise ValueError(
                    "conflicting periodic cell input: cell %r does not "
                    "match lattice_vector %r (derived cell6 %r)"
                    % (cell, lattice_vector, back))
        data['lattice_vector'] = new_lattice
    elif cell and len(cell) == 6 and cell_is_sentinel:
        # "Not filled" sentinel: keep the explicit lattice_vector when
        # given, otherwise leave the all-zero default (no cell).
        pass

    derived = data.get('lattice_vector')
    if _non_zero(derived):
        volume = cell_volume(derived)
        if not data.get('volume'):
            data['volume'] = volume

    return data


def derive_current_state(data):
    """Derive system.current_state from the dynamics root.

    Fill-if-default derivation: root (&MD) is the initial electronic
    state of a dynamics run; by this point the adapter has already
    converted the external 1-based spelling to the internal 0-based
    index, and the system ledger starts there. When current_state is
    still at its default (0) it is
    filled with root. An explicitly declared non-default
    current_state (e.g. written by the certification converter for
    skill tests) is never overridden.

    Parameters
    ----------
    data : dict
        Flattened configuration dictionary.

    Returns
    -------
    dict
        Updated dictionary with current_state derived from root.
    """
    current_state = data.get('current_state', 0)
    if current_state == 0:
        root = data.get('root', 0)
        # A multi-center root is a list; the flat current_state scalar
        # takes the first center's root as a temporary projection. The
        # authoritative per-center initial states live in the partition
        # plan (data.center). Deletion gate: R4, when PyRAIState tracks
        # a per-center active state.
        if isinstance(root, (list, tuple)):
            root = root[0] if root else 0
        data['current_state'] = root
    return data


def normalize_history(data):
    """Normalize the history snapshot buffer length.

    Electronic dynamics algorithms (FSSH and friends) need at least
    the last 4 timesteps, so any input value below 4 is raised to 4.
    PyRAIState itself only sees the normalized max_length and never
    interprets the physical meaning of the bound.

    Parameters
    ----------
    data : dict
        Configuration dictionary to normalize.

    Returns
    -------
    dict
        Updated dictionary with history >= 4.
    """
    try:
        history = int(data.get('history', 4))
    except (TypeError, ValueError):
        history = 4
    data['history'] = max(history, 4)
    return data


# ---------------------------------------------------------------------
# The constraint block: input mapping in, numeric plan out.
#
# `schema/config.py` declares what may be declared and `schema/adapter.py`
# reads the classic text into the same mapping; this region is the static half
# of the two-step process the design document fixes (section 5). Names become
# 0-based atom indices, every number becomes an internal unit, a `bond_type`
# selection becomes the pairs the bond graph perceives, a ramp becomes a
# ScheduleRecord - and anything that cannot be served is refused here, before a
# backend, a scratch directory or a remote job exists.
#
# The runtime half, capturing the targets and anchors that mean "the initial
# geometry", belongs to the controller. A restarted run must never re-capture a
# target from the structure it resumed at.
# ---------------------------------------------------------------------

# Conversion factor from the input unit of a quantity to its internal unit.
_LENGTH_FACTOR = {'angstrom': ANGSTROM_TO_BOHR, 'bohr': 1.0}
_ANGLE_FACTOR = {'degree': float(np.pi) / 180.0, 'radian': 1.0}
_ENERGY_FACTOR = {'hartree': 1.0, 'ev': EV_TO_HARTREE,
                  'kcal': KCAL_TO_HARTREE, 'cm': CM_TO_HARTREE}


# Action kinds a plan can hold, in the order default ids are minted.
_ID_PREFIX = {'bias': 'bias', 'wall': 'wall', 'fix': 'fix',
              'constraint': 'constraint', 'metadynamics': 'metad'}


def config_from_mapping(mapping):
    """Build the input model from a parsed mapping (text, JSON or YAML)."""
    from pydantic import ValidationError

    try:
        return SystemConstraintConfig(**(mapping or {}))
    except ConstraintInputError:
        raise
    except ValidationError as error:  # pydantic's message is the useful one
        raise ConstraintInputError(
            'constraint input rejected: %s' % error) from error

def _factors(units):
    """Conversion factors implied by the declared input units."""
    length = _LENGTH_FACTOR[units.length]
    angle = _ANGLE_FACTOR[units.angle]
    energy = _ENERGY_FACTOR[units.energy]
    return length, angle, energy


def _number(value, label):
    """A finite float, or a failure naming what was wrong."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ConstraintInputError('%s is not a number: %r' % (label, value))
    if not np.isfinite(number):
        raise ConstraintInputError('%s must be finite, got %r'
                                   % (label, value))
    return number


def _axes(direction):
    """Frozen Cartesian components of a direction spelling, as a list."""
    lookup = {'x': 0, 'y': 1, 'z': 2}
    return sorted(lookup[char] for char in direction)



class _Normalizer(object):
    """Shared state of one normalization pass."""

    def __init__(self, config, atoms, natom, masses, pbc,
                coordinates=None, lattice=None, md_step=None):
        from pyrai3md.data.topology import GroupTable
        self.config = config
        self.atoms = list(atoms or [])
        self.coordinates = coordinates
        self.lattice = lattice
        self._graph = None
        self.natom = int(natom if natom is not None else len(self.atoms))
        self.masses = masses
        self.pbc = pbc
        # The length of the run, which is what closes a ramp whose `stop` the
        # input left open. None says the run declares no step count, and a ramp
        # that needs one is refused rather than guessed at.
        self.md_step = md_step
        self.length, self.angle, self.energy = _factors(config.units)
        self.groups = GroupTable()
        self.cvs = {}
        self.schedules = {}
        self.actions = []
        self.notes = list(config.notes)
        self._counters = {}
        self._seen_ids = set()

    def graph(self, label):
        """The bond graph of the structure this plan was built on.

        Built on demand, because perceiving it is only owed to a declaration
        that asks which atoms are bonded. The refusal when no structure reached
        this layer states what is missing: a plan assembled from an atom count
        alone, which is what a unit test or a remote stub does, cannot produce a
        bond graph.
        """
        if self._graph is None:
            if self.coordinates is None or not len(self.atoms):
                raise UnsupportedCapabilityError(
                    "%s: 'bond_type' perceives the bond graph from the "
                    'structure, and this plan was built without one; pass the '
                    'coordinates and the cell, or list the atom pairs '
                    'explicitly' % label)
            try:
                self._graph = bonds.bond_graph(
                    self.atoms, self.coordinates,
                    lattice_vector=self.lattice, pbc=self.pbc)
            except bonds.UnknownPairError as error:
                raise UnsupportedCapabilityError(
                    '%s: %s' % (label, error)) from error
        return self._graph

    def bond_type_pairs(self, value, label):
        """Every bonded pair of one element combination, as 1-based rows."""
        # The graph is read outside the try: its own refusals (no structure
        # reached this layer, an element pair the radius table does not list)
        # are the answer, not a ValueError to be relabeled.
        graph = self.graph(label)
        try:
            rows = bonds.bond_pairs_by_type(graph, value)
        except ValueError as error:
            raise ConstraintInputError('%s: %s' % (label, error)) from error
        if not rows:
            # The wildcard's empty answer has one specific reason worth
            # naming: it matched no hydrogen bonded to a heavy atom, which is
            # not the same statement as one named pair being absent.
            wildcard = 'X' in [part.upper()
                               for part in str(value).replace('-', ' ').split()]
            reason = ('the structure has no hydrogen bonded to a heavy atom'
                      if wildcard else
                      'list the atoms explicitly if the pair is not a bond')
            raise ConstraintInputError(
                '%s: no bonded %s pair was perceived in this structure; %s'
                % (label, value, reason))
        return rows

    def mint(self, kind, given):
        """A stable id: the declared one, or `<kind><n>` by appearance."""
        name = str(given or '')
        if not name:
            self._counters[kind] = self._counters.get(kind, 0) + 1
            name = '%s%d' % (_ID_PREFIX.get(kind, kind), self._counters[kind])
        if name in self._seen_ids:
            raise ConstraintInputError(
                "duplicate action id '%s'; ids must be unique within one "
                "plan" % name)
        self._seen_ids.add(name)
        return name

    def require_index(self, index, label):
        """Range check a 0-based index."""
        if index >= self.natom:
            raise ConstraintInputError(
                '%s references atom %d beyond natom %d'
                % (label, index + 1, self.natom))

    def resolve(self, selector, label):
        """One selector mapping to 0-based indices, range checked.

        A group reference is answered from the already-resolved table rather
        than by re-running the selector: the table holds 0-based indices, so
        feeding it back through the 1-based parser would shift every member
        by one. The cycle check that a nested group would need therefore
        happens in normalize_groups, where group definitions are still raw.
        """
        if isinstance(selector, dict) and 'group' in selector:
            name = str(selector['group'])
            try:
                indices = self.groups.members(name)
            except KeyError as error:
                raise ConstraintInputError(
                    '%s: %s' % (label, error.args[0] if error.args else error))
            return indices.copy()
        try:
            indices = selection.resolve_indices(
                selector, atoms=self.atoms or None, natom=self.natom)
        except selection.UnsupportedSelectionError as error:
            raise UnsupportedCapabilityError('%s: %s' % (label, error))
        except selection.SelectionError as error:
            raise ConstraintInputError('%s: %s' % (label, error))
        for index in indices:
            self.require_index(int(index), label)
        return indices

    def length_to_internal(self, value, label):
        return _number(value, label) * self.length

    def angle_to_internal(self, value, label):
        return _number(value, label) * self.angle

    def energy_to_internal(self, value, label):
        return _number(value, label) * self.energy

    def force_constant(self, value, power, angular, label):
        """Convert a force-like constant: energy over variable to `power`."""
        number = _number(value, label) * self.energy
        scale = (self.angle if angular else self.length) ** power
        return number / scale

    def cell_to_internal(self, rows, label):
        """Row-vector lattice of a wall boundary, converted to Bohr.

        Both spellings the framework already accepts for a cell are accepted
        here, through the one conversion that exists for each: nine numbers are
        the row-major matrix, six are the classic
        ``[a, b, c, alpha, beta, gamma]`` parameters, whose angles are degrees in
        that convention (the same reading `schema/adapter.py` applies to
        `molecule.cell`, so a wall and the run's own cell cannot disagree about
        what `[8, 8, 8, 90, 90, 120]` means).
        """
        matrix = np.asarray(rows, dtype=np.float64).reshape(-1)
        if matrix.size not in (9, 6):
            raise ConstraintInputError(
                "%s needs nine matrix numbers or six cell parameters, got %d"
                % (label, matrix.size))
        if matrix.size == 6:
            lengths = matrix[:3] * self.length
            if np.any(lengths <= 0.0):
                raise ConstraintInputError(
                    '%s: the three cell lengths must be positive, got %r'
                    % (label, matrix[:3].tolist()))
            array = np.asarray(
                parse_cell(np.concatenate(
                    (lengths, matrix[3:])).tolist()),
                dtype=np.float64)
        else:
            array = np.asarray(parse_cell(matrix.tolist()),
                               dtype=np.float64) * self.length
        if not np.all(np.isfinite(array)):
            raise ConstraintInputError('%s holds a non-finite number' % label)
        if abs(float(np.linalg.det(array))) < 1.0e-12:
            raise ConstraintInputError('%s matrix is singular' % label)
        return array



    def number_width(self, value, unit, label):
        """Convert a Gaussian width written in the CV's own unit.

        A width belongs to a collective variable, so it carries the variable's
        unit: Bohr for a bond, radian for an angle or a dihedral.
        """
        number = _number(value, '%s width' % label)
        scaled = (self.angle_to_internal(number, '%s width' % label)
                  if unit == 'radian' else
                  self.length_to_internal(number, '%s width' % label))
        if scaled <= 0.0:
            raise ConstraintInputError('%s widths must be positive' % label)
        return scaled

def normalize_groups(normalizer):
    """Resolve every named group before anything can reference it.

    A group may name another group, which is legal as long as the chain ends
    at an explicit selection. Declarations are resolved in dependency order
    with a cycle test, and each is range checked as it lands.
    """
    pending = {}
    for record in normalizer.config.groups:
        name = str(record.name or '')
        if not name:
            raise ConstraintInputError('a group needs a name')
        if name in pending:
            raise ConstraintInputError("group '%s' is declared twice" % name)
        pending[name] = record
    order = []
    for name in pending:
        _visit_group(name, pending, order, [])
    for name in order:
        record = pending[name]
        selector = record.selector("group '%s'" % name)
        normalizer.groups.add(
            name, normalizer.resolve(selector, "group '%s'" % name))


def _visit_group(name, pending, order, stack):
    """Depth-first ordering of group definitions by their references."""
    if name in order:
        return
    if name in stack:
        raise ConstraintInputError(
            'circular group reference: %s' % ' -> '.join(stack + [name]))
    record = pending.get(name)
    if record is None:
        raise ConstraintInputError("unknown group '%s'" % name)
    for reference in _group_references(record):
        _visit_group(reference, pending, order, stack + [name])
    order.append(name)


def _group_references(record):
    """Names a group definition points at, if any."""
    found = []
    if getattr(record, 'group', ''):
        found.append(str(record.group))
    return found


def normalize_cvs(normalizer):
    """Resolve named collective variables and inline ones from a bias."""
    for record in normalizer.config.cvs:
        name = str(record.name or '')
        if not name:
            raise ConstraintInputError('a cv needs a name')
        if name in normalizer.cvs:
            raise ConstraintInputError("cv '%s' is declared twice" % name)
        kind = str(record.kind or '')
        indices = normalizer.resolve(
            {'atoms': record.atoms}, "cv '%s'" % name)
        normalizer.cvs[name] = _make_cv(name, kind, indices)


def _make_cv(name, kind, indices):
    """Infer a missing kind from the atom count and build the record."""
    from pyrai3md.data.constraint import CVRecord
    count = int(np.asarray(indices).size)
    if not kind:
        kind = {2: 'bond', 3: 'angle', 4: 'dihedral'}.get(count, '')
    if not kind:
        raise ConstraintInputError(
            "cv '%s' has %d atoms, which is not a bond, angle or dihedral"
            % (name, count))
    if kind == 'bond' and count != 2:
        raise ConstraintInputError("cv '%s' of kind bond needs 2 atoms" % name)
    if kind == 'angle' and count != 3:
        raise ConstraintInputError(
            "cv '%s' of kind angle needs 3 atoms" % name)
    if kind == 'dihedral' and count != 4:
        raise ConstraintInputError(
            "cv '%s' of kind dihedral needs 4 atoms" % name)
    unit = 'bohr' if kind == 'bond' else 'radian'
    return CVRecord(id=name, kind=kind, indices=indices,
                    periodic=(kind == 'dihedral'), unit=unit)


def _endpoints(value):
    """Two endpoints, however the input spelled them.

    The text reader keeps 'scale 0 1' as one string, YAML gives a list and a
    dotted JSON can give a space-separated string too, so all three land here.
    A vector parameter (three semiaxes, three box edges, one cell) gives two
    *vectors*, which stay nested so one component cannot be read as a different
    component's endpoint. A wrong count is reported as such rather than
    truncated, because a truncated endpoint pair silently changes what a ramp
    does.
    """
    if value is None or value == '':
        return None
    if isinstance(value, str):
        parts = value.replace(',', ' ').split()
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ConstraintInputError(
            'a schedule needs two endpoints, got %r' % (value,))
    return [_endpoint(item) for item in parts]


def _endpoint(item):
    """One endpoint: a number, or a vector of them."""
    if isinstance(item, (list, tuple)):
        return [_number(part, 'schedule endpoint') for part in item]
    if isinstance(item, str):
        parts = item.replace(',', ' ').split()
        if len(parts) == 1:
            return _number(parts[0], 'schedule endpoint')
        return [_number(part, 'schedule endpoint') for part in parts]
    return _number(item, 'schedule endpoint')


def _inline_schedule(normalizer, record, label):
    """Register an action's parameter ramps and return their names, in order.

    A ramp is spelled inline on the record, which is what the flat front-end
    produces from `bias_bond_fc_start` and friends: the five suffixes sit on the
    keyword of the parameter they smooth, so a ramp cannot mean two different
    things depending on which spelling wrote it, and there is no shared schedule
    section to name one.

    One action may ramp several parameters (a wall whose radius and force
    constant both move), so the result is a list; the order is the order the
    ramps were declared, which is also the order they are applied in.
    """
    from pyrai3md.data.constraint import ScheduleRecord
    inline = list(getattr(record, 'schedules', ()) or ())
    if not inline:
        return []
    names = []
    for position, item in enumerate(inline):
        body = dict(item)
        base = str(label.split("'")[1])
        # The generated name keeps the shape the single-ramp spelling always
        # used, so a plan's schedule ids do not change when a second ramp is
        # added to the same action.
        auto = ('schedule_of_%s' % base if position == 0
                else 'schedule_of_%s_%d' % (base, position + 1))
        name = str(body.pop('id', '') or auto)
        if name in normalizer.schedules:
            raise ConstraintInputError(
                "schedule '%s' is declared twice" % name)
        leftover = sorted(key for key in body
                          if key not in ('kind', 'parameter', 'start_step',
                                         'end_step', 'scale', 'value'))
        if leftover:
            raise ConstraintInputError(
                '%s: inline schedule carries unknown keys %s'
                % (label, ', '.join(leftover)))
        body['scale'] = _endpoints(body.get('scale'))
        body['value'] = _endpoints(body.get('value'))
        if 'start_step' not in body:
            raise ConstraintInputError(
                "%s: an inline schedule needs a start step" % label)
        if body.get('end_step') is None:
            # `stop` is the length of the run, which only this layer can see:
            # the reader hands a ramp over with its stop open, and the run's own
            # step count closes it here. The source is named in the note,
            # because "it stops at the last step" is a decision the input did
            # not make in numbers.
            if not normalizer.md_step:
                raise ConstraintInputError(
                    "%s: ramp '%s' declares no stop and the run declares no "
                    'step count for it to stop at (md.step)' % (label, name))
            body['end_step'] = int(normalizer.md_step)
            normalizer.notes.append(
                "%s ramp '%s' stops at the last step of the run (md.step = %d)"
                % (label, name, body['end_step']))
        try:
            normalizer.schedules[name] = ScheduleRecord(
                id=name, kind=body.get('kind', 'linear'),
                parameter=body.get('parameter', 'force_constant'),
                start_step=body.get('start_step', 0),
                end_step=body.get('end_step', 0),
                scale=body.get('scale'), value=body.get('value'))
        except ValueError as error:
            raise ConstraintInputError(
                '%s: inline schedule: %s' % (label, error)) from error
        names.append(name)
    return names


def _attach_schedule(normalizer, name, label, conversions, declared=None):
    """Resolve a schedule reference and check it ramps a real parameter.

    The returned record carries values already converted to internal units.
    A 'scale' ramp would multiply the declared constant, whose declared unit
    and the internal unit differ by a factor of the variable conversion, so
    the two endpoints would be mixed; it is refused instead of guessed, and
    'value' is the spelling that stays unambiguous.

    Three checks run here rather than at the first step, because a ramp that
    cannot be resolved is a property of the declaration, not of the geometry:
    the endpoints must have the shape of the value they replace, a parameter
    that must stay positive may not be ramped to zero, and a triclinic cell is
    turned into endpoint matrices, since no single factor converts a quantity
    that mixes lengths and angles.
    """
    from pyrai3md.data.constraint import ScheduleRecord
    if not name:
        return None
    if name not in normalizer.schedules:
        raise ConstraintInputError(
            "%s references unknown schedule '%s'" % (label, name))
    record = normalizer.schedules[name]
    if record.parameter not in conversions:
        raise ConstraintInputError(
            "%s: schedule ramps '%s', which %s does not carry; supported: %s"
            % (label, record.parameter, label, ', '.join(sorted(conversions))))
    factor = conversions[record.parameter]
    if record.scale is not None:
        if abs(factor - 1.0) > 1.0e-12:
            raise UnsupportedCapabilityError(
                "%s: a 'scale' schedule on a converted parameter is "
                "ambiguous; declare the endpoints with 'value' instead"
                % label)
        check_ramp_positive(record, label)
        return record
    if record.value is None:
        raise ConstraintInputError(
            "%s: schedule '%s' gives neither scale nor value" % (label, name))
    if record.parameter == 'cell':
        value = [normalizer.cell_to_internal(
            item, "%s schedule '%s' cell endpoint" % (label, name)).tolist()
            for item in record.value]
    else:
        value = [_convert_endpoint(item, factor) for item in record.value]
    base = None if declared is None else declared.get(record.parameter)
    check_ramp_shape(record.parameter, value, base, label, name)
    record = ScheduleRecord(
        id=record.id, kind=record.kind, parameter=record.parameter,
        start_step=record.start_step, end_step=record.end_step, value=value)
    check_ramp_positive(record, label)
    normalizer.schedules[record.id] = record
    return record


def _convert_endpoint(value, factor):
    """One endpoint in internal units: a number, or a vector of them."""
    if isinstance(value, (list, tuple)):
        return [float(item) * factor for item in value]
    return float(value) * factor


def _attach_schedules(normalizer, names, label, conversions, declared=None):
    """Resolve every ramp of one action into (first, rest).

    `ConstraintSpec` keeps `schedule` as the first ramp and `schedules` as the
    remainder, so a reader that knows only the old single-ramp shape still sees
    the parameter it expects while an action may now ramp several.
    """
    records = [item for item in (_attach_schedule(normalizer, name, label,
                                                  conversions, declared)
                                 for name in names) if item is not None]
    if not records:
        return None, None
    return records[0], (records[1:] or None)


def normalize_biases(normalizer):
    """Harmonic biases, one scalar CV each."""
    from pyrai3md.data.constraint import ConstraintSpec
    for record in normalizer.config.biases:
        label = "bias '%s'" % (record.id or '?')
        name = normalizer.mint('bias', record.id)
        label = "bias '%s'" % name
        cv_name, cv_record = _bias_cv(normalizer, record, name, label)
        angular = cv_record.unit == 'radian'
        parameters = {}
        if record.force_constant is None:
            raise ConstraintInputError(
                "%s needs a force_constant" % label)
        parameters['force_constant'] = normalizer.force_constant(
            record.force_constant, 2, angular,
            "%s force_constant" % label)
        if parameters['force_constant'] <= 0.0:
            raise ConstraintInputError(
                "%s force_constant must be positive" % label)
        target = None
        reference = 'initial'
        if record.target is not None:
            target = (normalizer.angle_to_internal(record.target, label)
                      if angular else
                      normalizer.length_to_internal(record.target, label))
            reference = 'explicit'
        region = record.region
        if cv_record.kind == 'dihedral' and region in ('upper', 'lower'):
            raise UnsupportedCapabilityError(
                "%s: a one-sided periodic region needs a declared origin and "
                "direction, which the first version does not define" % label)
        if region in ('inside', 'outside'):
            raise UnsupportedCapabilityError(
                "%s: region %r is not defined yet" % (label, region))
        first, extra = _attach_schedules(
            normalizer, _inline_schedule(normalizer, record, label), label,
            {'force_constant': 1.0}, declared=parameters)
        normalizer.actions.append(ConstraintSpec(
            id=name, kind='bias', family='harmonic', cv=cv_name,
            target=target, reference=reference, region=region,
            parameters=parameters, schedule=first, schedules=extra))


def _bias_cv(normalizer, record, name, label):
    """The CV a bias acts on: named, or inline from the atoms it listed."""
    if record.cv:
        if record.atoms:
            raise ConstraintInputError(
                "%s: give either a named cv or an inline selection, not both"
                % label)
        if str(record.cv) not in normalizer.cvs:
            raise ConstraintInputError(
                "%s references unknown cv '%s'" % (label, record.cv))
        return str(record.cv), normalizer.cvs[str(record.cv)]
    selector = record.selector(label)
    indices = normalizer.resolve(selector, label)
    cv_name = 'cv_of_%s' % name
    if cv_name in normalizer.cvs:
        raise ConstraintInputError(
            "%s: inline cv name '%s' already taken" % (label, cv_name))
    kind = ''
    record_cv = _make_cv(cv_name, kind, indices)
    normalizer.cvs[cv_name] = record_cv
    return cv_name, record_cv


def normalize_walls(normalizer):
    """The wall geometries, their molecule groups and their anchor.

    Three separations are worth naming, because each one is a rule the input
    states and the plan then obeys without second-guessing it:

    - the grouping says **what the wall acts on**, the target says **how a
      group's center is weighted**, and the law says **what the potential does
      with the distance**. They are independent keywords, so a wall over three
      molecules of three atoms each is one declaration and not a mode.
    - the center is `center` if the input gave coordinates, else the center of
      the `ref_atoms` the input named, else the center of the selection. All
      three are captured on the reference geometry and then fixed, so a restart
      cannot move the wall it resumed under.
    - a grouping whose molecules hold more than one atom means nothing to a wall
      whose target is a single atom, because that wall already evaluates every
      atom on its own. It is dropped, and the drop is written into `notes`: a
      declaration that reaches the run changed has to be readable there.
    """
    from pyrai3md.data.constraint import ConstraintSpec
    for record in normalizer.config.walls:
        name = normalizer.mint('wall', record.id)
        label = "wall '%s'" % name
        indices = _confined_atoms(normalizer, record, label,
                                  groups=record.molecules)
        if record.force_constant is None:
            raise ConstraintInputError("%s needs a force_constant" % label)
        power = int(record.power)
        if power < 2:
            raise ConstraintInputError(
                "%s power must be at least 2 so the wall is differentiable"
                % label)
        law = str(record.law or 'hard')
        parameters = {
            'geometry': record.geometry,
            'power': power,
            'target_kind': record.target,
            'law': law,
        }
        if law == 'soft':
            # The argument of the soft law is a ratio of lengths, so it is
            # dimensionless and its constant converts as an energy and nothing
            # else. That is the one place the two laws of one wall differ in
            # units, and it is why they do not share a conversion helper.
            parameters['force_constant'] = normalizer.energy_to_internal(
                record.force_constant, '%s force_constant' % label)
        else:
            parameters['force_constant'] = normalizer.force_constant(
                record.force_constant, power, False,
                '%s force_constant' % label)
        if parameters['force_constant'] <= 0.0:
            raise ConstraintInputError(
                "%s force_constant must be positive" % label)
        if record.center:
            if len(record.center) != 3:
                raise ConstraintInputError(
                    "%s center needs three numbers" % label)
            parameters['center'] = [
                normalizer.length_to_internal(value, '%s center' % label)
                for value in record.center]
        elif record.ref_atoms:
            anchor = normalizer.resolve({'atoms': record.ref_atoms},
                                        '%s ref_atoms' % label)
            parameters['anchor'] = [int(value) for value in anchor]
        molecules = _wall_molecules(normalizer, record, indices, label)
        if molecules:
            parameters['molecules'] = molecules
        _wall_shape(normalizer, record, parameters, label)
        if record.target in ('com', 'cog') and normalizer.masses is None \
                and record.target == 'com':
            raise ConstraintInputError(
                "%s target 'com' needs per-atom masses" % label)
        first, extra = _attach_schedules(
            normalizer, _inline_schedule(normalizer, record, label), label,
            {'force_constant': 1.0, 'radius': normalizer.length,
             'semiaxes': normalizer.length, 'lengths': normalizer.length,
             # A triclinic ramp is handed over as endpoint matrices already in
             # internal units, because a cell mixes lengths and angles and no
             # single factor converts it.
             'cell': 1.0}, declared=parameters)
        normalizer.actions.append(ConstraintSpec(
            id=name, kind='wall', family=record.geometry,
            atoms=indices, group=record.group or None,
            parameters=parameters, schedule=first, schedules=extra,
            # An explicit point is the only center that is already in numbers;
            # every other case is captured from the reference geometry, which is
            # what keeps a restart from re-reading its own drift as the wall's
            # anchor.
            reference='explicit' if record.center else 'initial'))


def _wall_molecules(normalizer, record, indices, label):
    """The wall's groups as 0-based member lists, or None for one whole group.

    The structured form gives the groups; the text form gives one list whose
    last number is the size of a molecule, which the reader has already cut up.
    Both arrive here as lists of atom numbers, 1-based until this line.
    """
    given = [list(group) for group in (record.molecules or [])]
    if not given:
        return None
    if str(record.target) == 'atom' and any(len(group) > 1 for group in given):
        normalizer.notes.append(
            "%s: 'wall_groups' has molecules of more than one atom but the "
            "target is 'atom', which evaluates every atom on its own; the "
            'grouping is ignored' % label)
        return None
    members = [int(index) for index in indices]
    groups = []
    for group in given:
        resolved = [int(index) for index in
                    normalizer.resolve({'atoms': group},
                                       '%s groups' % label)]
        for index in resolved:
            if index not in members:
                raise ConstraintInputError(
                    "%s: a group holds atom %d, which the wall's own selection "
                    'does not contain; the atoms a wall groups must be the '
                    'atoms it confines' % (label, index + 1))
        if not resolved:
            raise ConstraintInputError('%s has an empty molecule group' % label)
        groups.append(resolved)
    covered = sorted(index for group in groups for index in group)
    if covered != sorted(members):
        raise ConstraintInputError(
            "%s: the groups cover %d atom(s) of the %d the wall selects; every "
            'confined atom has to be in one molecule, or the run leaves it '
            'unrestrained and nothing says so'
            % (label, len(covered), len(members)))
    return groups


def _confined_atoms(normalizer, record, label, groups=None):
    """The confined atoms: the selection, the grouped atoms, or all of them.

    Three readings, in that order. An explicit selection is the answer. With
    none, a wall that cuts its atoms into molecules is confining exactly those
    molecules, so the groups are the selection - the alternative is to make the
    same atom list be typed twice, once to select and once to group. With neither
    the archived fallback holds and the wall acts on every atom, as it did when
    `constrain` was left empty; a migration that turned that into an error would
    refuse inputs that used to run. Both fallbacks are written into `notes`,
    because a declaration that reaches the run as "these atoms" or "every atom" is
    a different statement from the one the input seems to make.
    """
    if record.atoms or record.element or record.group:
        return normalizer.resolve(record.selector(label), label)
    tokens = [atom for group in (groups or ())
              for atom in (group if isinstance(group, (list, tuple))
                           else [group])]
    if tokens:
        # The numbers go through the same one decrement every atom number here
        # gets, so a range spelled in a group is as good as a list of them.
        indices = sorted({int(index) for index in normalizer.resolve(
            {'atoms': tokens}, '%s groups' % label)})
        normalizer.notes.append(
            "%s: no atom selection, so it confines the %d atoms its groups "
            'name' % (label, len(indices)))
        return indices
    if not normalizer.natom:
        raise ConstraintInputError(
            '%s has no atom selection and the system size is unknown' % label)
    normalizer.notes.append(
        "%s: no atom selection, so it acts on all %d atoms"
        % (label, normalizer.natom))
    return list(range(normalizer.natom))


def _wall_shape(normalizer, record, parameters, label):
    """Copy and convert the geometry numbers one wall shape needs."""
    geometry = parameters['geometry']
    if geometry == 'sphere':
        if record.radius is None:
            raise ConstraintInputError("%s sphere needs a radius" % label)
        radius = normalizer.length_to_internal(record.radius,
                                               '%s radius' % label)
        if radius <= 0.0:
            raise ConstraintInputError("%s radius must be positive" % label)
        parameters['radius'] = radius
    elif geometry == 'ellipsoid':
        axes = [normalizer.length_to_internal(value, '%s semiaxis' % label)
                for value in record.semiaxes]
        if len(axes) != 3 or any(value <= 0.0 for value in axes):
            raise ConstraintInputError(
                "%s ellipsoid needs three positive semiaxes" % label)
        parameters['semiaxes'] = axes
    elif geometry == 'box':
        lengths = [normalizer.length_to_internal(value, '%s length' % label)
                   for value in record.lengths]
        if len(lengths) != 3 or any(value <= 0.0 for value in lengths):
            raise ConstraintInputError(
                "%s box needs three positive full edge lengths" % label)
        parameters['lengths'] = lengths
    else:
        if not record.cell:
            raise ConstraintInputError(
                "%s triclinic needs a cell matrix" % label)
        parameters['cell'] = normalizer.cell_to_internal(
            record.cell, '%s cell' % label).tolist()


def normalize_fixes(normalizer):
    """Frozen coordinate components anchored at the initial geometry."""
    from pyrai3md.data.constraint import ConstraintSpec
    for record in normalizer.config.fixes:
        name = normalizer.mint('fix', record.id)
        label = "fix '%s'" % name
        selector = record.selector(label)
        indices = normalizer.resolve(selector, label)
        normalizer.actions.append(ConstraintSpec(
            id=name, kind='fix', family='mask', atoms=indices,
            group=record.group or None, reference='initial',
            parameters={'axes': _axes(record.direction)}))


CONSTRAINT_WIDTH = {'bond': 2, 'angle': 3, 'dihedral': 4}
# The family a strict constraint is filed under. Bond rows keep the name the
# plan has always carried, so a serialized plan of a bond-only run reads the
# same after this change; the two new kinds are what routes them to the
# multi-condition solver instead of the pair solver.
CONSTRAINT_FAMILY = {'bond': 'shake', 'angle': 'angle', 'dihedral': 'dihedral'}


def normalize_hard_constraints(normalizer):
    """Strict geometric constraints: bond lengths, angles and torsions."""
    from pyrai3md.data.constraint import ConstraintSpec
    seen = {}
    for record in normalizer.config.constraints:
        name = normalizer.mint('constraint', record.id)
        label = "constraint '%s'" % name
        kind = str(record.kind or 'bond')
        if kind not in CONSTRAINT_WIDTH:
            raise ConstraintInputError(
                "%s: unknown constraint kind %r; expected one of %s"
                % (label, kind, ', '.join(sorted(CONSTRAINT_WIDTH))))
        if record.bond_type and kind != 'bond':
            raise ConstraintInputError(
                "%s: 'bond_type' selects bonded pairs, so it expands a bond "
                'constraint only; list the %d-atom rows explicitly for a %s '
                'constraint' % (label, CONSTRAINT_WIDTH[kind], kind))
        rows = record.bonds
        if not rows and record.bond_type:
            rows = normalizer.bond_type_pairs(record.bond_type, label)
        if not rows:
            raise ConstraintInputError(
                "%s needs an explicit list of %s atom rows"
                % (label, 'bond' if kind == 'bond' else '%d'
                   % CONSTRAINT_WIDTH[kind]))
        if kind == 'bond':
            try:
                pairs = selection.resolve_bond_pairs(
                    rows, natom=normalizer.natom)
            except selection.UnsupportedSelectionError as error:
                raise UnsupportedCapabilityError('%s: %s' % (label, error))
            except selection.SelectionError as error:
                raise ConstraintInputError('%s: %s' % (label, error))
        else:
            try:
                pairs = None
                rows = selection.resolve_index_rows(
                    record.bonds, CONSTRAINT_WIDTH[kind],
                    natom=normalizer.natom, label=label)
            except selection.UnsupportedSelectionError as error:
                raise UnsupportedCapabilityError('%s: %s' % (label, error))
            except selection.SelectionError as error:
                # The row resolver names the declaration itself, so the label
                # is not pasted in front of it a second time.
                raise ConstraintInputError(str(error))
        target = None
        reference = 'initial'
        if record.target is not None:
            if kind == 'bond':
                target = normalizer.length_to_internal(
                    record.target, '%s target' % label)
            else:
                target = normalizer.angle_to_internal(
                    record.target, '%s target' % label)
            reference = 'explicit'
            if target <= 0.0:
                raise ConstraintInputError(
                    "%s target must be positive (a bond length is; an angle "
                    "or torsion target of 0 rad is a straight line, which the "
                    "solver cannot differentiate)" % label)
        merged = []
        if kind == 'bond':
            for pair in pairs.tolist():
                key = tuple(int(value) for value in pair)
                if key in seen and reference == 'explicit':
                    if (seen[key] is not None
                            and abs(seen[key] - target) > 1.0e-12):
                        raise ConstraintInputError(
                            '%s: atoms %d-%d carry two different targets '
                            '(%.8f, %.8f)'
                            % (label, key[0] + 1, key[1] + 1, seen[key],
                               target))
                    continue
                seen[key] = target
                merged.append(list(key))
        else:
            merged = rows.tolist()
        if not merged:
            continue
        normalizer.actions.append(ConstraintSpec(
            id=name, kind='constraint', family=CONSTRAINT_FAMILY[kind],
            pairs=None if pairs is None else np.asarray(
                merged, dtype=np.int64),
            atoms=None if kind == 'bond' else np.asarray(
                merged, dtype=np.int64).reshape(-1),
            target=target,
            reference=reference,
            parameters={} if kind == 'bond' else {
                'cv_kind': kind, 'rows': merged},
            diagnostics={'tolerance': float(record.tolerance),
                         'max_iterations': int(record.max_iterations),
                         'velocity_constraint': str(
                             record.velocity_constraint)}))


def _delta_temperature(record, label):
    """The well-tempered `dT` in Kelvin, or 0.0 for plain deposition.

    A bias factor is a ratio of temperatures, so it names nothing on its own:
    `gamma` and `T` arrive together or the declaration is refused here, at input
    time, rather than becoming a silent no-op in the run.
    """
    gamma = float(record.bias_factor)
    temperature = float(record.temperature)
    if gamma == 0.0 and temperature == 0.0:
        return 0.0
    if gamma <= 1.0:
        raise ConstraintInputError(
            "%s: 'bias_factor' is the ratio (T + dT) / T of well-tempered "
            'metadynamics, so it must be above 1; got %r (1 is the plain '
            'deposition, which needs no keyword)' % (label, gamma))
    if temperature <= 0.0:
        raise ConstraintInputError(
            "%s: 'bias_factor' %r needs 'temperature' in Kelvin to turn into "
            'a dT' % (label, gamma))
    return (gamma - 1.0) * temperature


def normalize_metadynamics(normalizer):
    """History potentials: the CV Gaussians and the aligned RMSD."""
    from pyrai3md.data.constraint import ConstraintSpec
    for record in normalizer.config.metadynamics:
        label = "metadynamics '%s'" % (record.id or '?')
        if record.kind == 'rmsd':
            _normalize_rmsd_history(normalizer, record, label)
            continue
        name = normalizer.mint('metadynamics', record.id)
        label = "metadynamics '%s'" % name
        names = [str(item) for item in record.cvs]
        if not names:
            raise ConstraintInputError('%s needs at least one cv' % label)
        for item in names:
            if item not in normalizer.cvs:
                raise ConstraintInputError(
                    "%s references unknown cv '%s'" % (label, item))
        units = [normalizer.cvs[item].unit for item in names]
        widths = [normalizer.number_width(value, unit, label)
                  for value, unit in zip(_pad_widths(record.width, len(names)),
                                         units)]
        if any(value <= 0.0 for value in widths):
            raise ConstraintInputError(
                '%s widths must all be positive' % label)
        if record.height is None:
            raise ConstraintInputError('%s needs a height' % label)
        height = normalizer.energy_to_internal(record.height,
                                               '%s height' % label)
        if height <= 0.0:
            raise ConstraintInputError('%s height must be positive' % label)
        parameters = {
            'cv_names': names, 'height': height, 'widths': widths,
            'pace': int(record.pace),
            'delta_temperature': _delta_temperature(record, label)}
        first, extra = _attach_schedules(
            normalizer, _inline_schedule(normalizer, record, label), label,
            {'height': 1.0}, declared=parameters)
        normalizer.actions.append(ConstraintSpec(
            id=name, kind='metadynamics', family='gaussian',
            atoms=None, cv=names[0], schedule=first, schedules=extra,
            parameters=parameters,
            diagnostics={'deposit_at_zero': False}))


def _normalize_rmsd_history(normalizer, record, label):
    """Turn an RMSD declaration into a hill variable over D^2.

    The declared width is an RMSD *length*, because that is the number a user
    has in mind, and the variable is the squared deviation, because the square
    root has a singular derivative exactly at zero deviation, which is where a
    converged run sits. So the width is converted as a length and then squared,
    which keeps `sigma` in the same unit as the variable (Bohr squared).
    """
    from pyrai3md.data.constraint import ConstraintSpec
    name = normalizer.mint('metadynamics', record.id)
    label = "metadynamics '%s'" % name
    if record.cvs:
        raise ConstraintInputError(
            "%s of kind rmsd is written in the aligned deviation, not in a cv;"
            " drop 'cvs'" % label)
    selector = record.selector(label)
    indices = normalizer.resolve(selector, label)
    if len(indices) < 2:
        raise ConstraintInputError(
            '%s aligns %d atoms; a deviation against a reference needs at '
            'least two' % (label, len(indices)))
    if len(record.width) != 1:
        raise ConstraintInputError(
            '%s needs exactly one width' % label)
    sigma = normalizer.length_to_internal(record.width[0], '%s width' % label)
    if sigma <= 0.0:
        raise ConstraintInputError('%s width must be positive' % label)
    if record.height is None:
        raise ConstraintInputError('%s needs a height' % label)
    height = normalizer.energy_to_internal(record.height, '%s height' % label)
    if height <= 0.0:
        raise ConstraintInputError('%s height must be positive' % label)
    target = _rmsd_reference_block(normalizer, record, indices, label)
    if record.weight == 'mass' and normalizer.masses is None:
        raise ConstraintInputError(
            "%s weight 'mass' needs per-atom masses" % label)
    parameters = {'height': height, 'widths': [sigma * sigma],
                  'pace': int(record.pace), 'weight': record.weight,
                  'delta_temperature': _delta_temperature(record, label)}
    first, extra = _attach_schedules(
        normalizer, _inline_schedule(normalizer, record, label), label,
        {'height': 1.0}, declared=parameters)
    normalizer.actions.append(ConstraintSpec(
        id=name, kind='metadynamics', family='rmsd',
        atoms=indices,
        parameters=parameters, schedule=first, schedules=extra,
        target=target,
        reference='explicit' if target is not None else 'initial',
        diagnostics={'deposit_at_zero': False,
                     'variable_unit': 'bohr2'}))


def _rmsd_reference_block(normalizer, record, indices, label):
    """The declared reference geometry, or None when it binds at the start.

    `reference initial` is the usual case: the run is measured against the
    structure it started from. Explicit coordinates are legitimate for a
    reference read out of a previous run, and they arrive as either a flat list
    of 3k numbers or k rows of three.
    """
    given = list(record.reference)
    if not given:
        return None
    if len(given) == 1 and isinstance(given[0], str):
        if given[0].lower() != 'initial':
            raise ConstraintInputError(
                "%s reference must be 'initial' or %d coordinates"
                % (label, 3 * len(indices)))
        return None
    numbers = []
    for item in given:
        if isinstance(item, str):
            numbers.extend(item.split())
        else:
            numbers.append(item)
    if len(numbers) != 3 * len(indices):
        raise ConstraintInputError(
            '%s reference holds %d numbers for %d atoms'
            % (label, len(numbers), len(indices)))
    flat = [normalizer.length_to_internal(value, '%s reference' % label)
            for value in numbers]
    return np.asarray(flat, dtype=np.float64).reshape(len(indices), 3)


def _pad_widths(values, count):
    """One width for many CVs is repeated; a longer list is a mistake."""
    numbers = list(values)
    if not numbers:
        raise ConstraintInputError('metadynamics needs a width per cv')
    if len(numbers) == 1 and count > 1:
        return numbers * count
    if len(numbers) != count:
        raise ConstraintInputError(
            'metadynamics has %d cvs but %d widths'
            % (count, len(numbers)))
    return numbers


def normalize_config(config, atoms=None, natom=None, masses=None, pbc=None,
                     coordinates=None, lattice=None, md_step=None):
    """Normalize a validated config into an immutable plan.

    Parameters
    ----------
    config : SystemConstraintConfig or dict
        The declaration, already strict-checked.
    atoms : list, optional
        Element symbols in framework order, needed by an element selector.
    natom : int, optional
        Atom count for range checks. Defaults to len(atoms).
    masses : array-like, optional
        Per-atom masses, required by a wall whose target is 'com'.
    pbc : array-like, optional
        Periodicity flags. Any periodic axis is refused in this version.
    md_step : int, optional
        The number of steps the run declares. It closes a parameter ramp whose
        `stop` the input left open, which is the one piece of the run's own
        length a constraint can inherit.

    Returns
    -------
    ConstraintPlan
        Empty when the declaration is empty, which is the caller's cue to do
        nothing at all.
    """
    from pyrai3md.data.constraint import ConstraintPlan
    if not isinstance(config, SystemConstraintConfig):
        config = config_from_mapping(config)
    normalizer = _Normalizer(config, atoms, natom, masses, pbc,
                           coordinates=coordinates, lattice=lattice,
                           md_step=md_step)
    if config.is_empty():
        return ConstraintPlan.empty()
    check_periodic(config, normalizer.pbc)
    if normalizer.natom <= 0:
        raise ConstraintInputError(
            'constraint normalization needs an atom count')
    normalize_groups(normalizer)
    normalize_cvs(normalizer)
    normalize_biases(normalizer)
    normalize_walls(normalizer)
    normalize_fixes(normalizer)
    normalize_hard_constraints(normalizer)
    normalize_metadynamics(normalizer)
    if not normalizer.actions:
        raise ConstraintInputError(
            'the constraint section declares only names, with no action to '
            'carry them')
    return ConstraintPlan(groups=normalizer.groups, cvs=normalizer.cvs,
                          actions=normalizer.actions,
                          schedules=normalizer.schedules,
                          notes=normalizer.notes,
                          units={'length': config.units.length,
                                 'angle': config.units.angle,
                                 'energy': config.units.energy,
                                 'length_to_bohr': normalizer.length,
                                 'angle_to_radian': normalizer.angle,
                                 'energy_to_hartree': normalizer.energy},
                          units_converted=True)


def build_plan(mapping, atoms=None, natom=None, masses=None, pbc=None,
             coordinates=None, lattice=None, md_step=None):
    """Parse-and-normalize convenience: a raw mapping straight to a plan.

    `coordinates` and `lattice` are what a connectivity-derived selector
    needs; a plan that never asks which atoms are bonded does not look at
    them.
    """
    return normalize_config(config_from_mapping(mapping), atoms=atoms,
                            natom=natom, masses=masses, pbc=pbc,
                            coordinates=coordinates, lattice=lattice,
                            md_step=md_step)


def normalize_constraints(data):
    """Build the constraint plan once, at the input boundary, and refuse early.

    The plan itself is not stored here: the runtime derives it from
    `System.constraints` and caches it on the state, because the geometry, the
    masses and the periodic flags a plan is normalized against belong to a run,
    not to a configuration. What this call buys is the input-time failure the
    design document requires (section 14, C0-6): an unsupported capability, a
    periodic system under a constraint, a `bond_type` selection, a duplicated
    id or an out-of-range index raises here, before any backend, scratch
    directory or remote job exists. A configuration that reaches a backend with
    a broken constraint block would report a constraint that was never applied.

    The block is read as it stands: this pass no longer folds the archive's flat
    restraint keys into it. Those keys are refused by `schema.config.System`
    before a config can be built at all, so a folded-in legacy term and a
    new-style record can never describe one physical term twice (#272).

    Building the plan twice - once here as a refusal check, once in the runtime
    as the live object - is deliberate and cheap: it is a few hundred
    microseconds against a calculation that costs seconds, and it means the two
    are produced by the same function on the same data, so an assertion that
    they agree is a real assertion.
    """
    block = data.get('constraints')
    if not block:
        # No block: no plan, nothing to check. This is DC-1, the case every
        # existing input still takes, and it is now the only way a run can be
        # unconstrained - the archive's flat restraint keywords are refused at
        # the model (`config.RETIRE_CONSTRAINT_KEYS`) rather than folded in.
        return data
    atoms = list(data.get('atoms') or [])
    natom = int(data.get('natom') or len(atoms))
    if not natom:
        raise ValueError(
            'the &constraint block was given but the system has no atom '
            'count, so no constraint can be resolved: give atoms or natom')
    build_plan(block, atoms=atoms, natom=natom,
               masses=data.get('mass') or None,
               pbc=data.get('pbc') or None,
               coordinates=data.get('coordinates') or None,
               lattice=data.get('lattice_vector') or None,
               # The same value the runtime hands over: a ramp whose `stop` the
               # input left open closes at the run's step count, and this pass
               # must know it or an input-time refusal would fire on a run the
               # runtime accepts.
               md_step=int(data.get('step') or 0) or None)
    return data


def normalize_method_labels(data):
    """Strip the retired `qm.`/`abinit.` prefixes from both method queues.

    The two queues are `theory` and `reference`; the prefixes are the old
    section spellings, kept so an archived input still runs. Stripping them
    here means every later reader (Maestro, planning, the task queue) sees
    one plain label, and no consumer has to know the old spelling.

    Parameters
    ----------
    data : dict or BaseModel
        Configuration dictionary or model to update.

    Returns
    -------
    dict or BaseModel
        Same type, with plain method labels.
    """
    from pydantic import BaseModel
    from ..skills.query import method_label

    def _stripped(values):
        return [method_label(item) for item in (values or [])]

    if isinstance(data, BaseModel):
        return data.model_copy(update={
            'theory': _stripped(getattr(data, 'theory', [])),
            'reference': _stripped(getattr(data, 'reference', [])),
        })
    data['theory'] = _stripped(data.get('theory', []))
    data['reference'] = _stripped(data.get('reference', []))
    return data


def normalize_skill_name(data):
    """Resolve skill_name from theory[0] via the query mechanism.

    In input mode, the user specifies the theory method (e.g. 'molcas').
    This function maps it to the corresponding skill_name (e.g. 'molcas_skill')
    using the query mechanism.

    Accepts both short labels ('pyscf') and prefixed format
    ('qm.pyscf') -- any prefix is stripped before resolution.

    Does nothing if skill_name is already set or theory is empty.

    Parameters
    ----------
    data : dict or BaseModel
        Configuration dictionary or model to update.

    Returns
    -------
    dict or BaseModel
        Updated configuration with skill_name resolved.
    """
    from pydantic import BaseModel

    if isinstance(data, BaseModel):
        skill_name = getattr(data, 'skill_name', '')
        theory = getattr(data, 'theory', [])
    else:
        skill_name = data.get('skill_name', '')
        theory = data.get('theory', [])

    if not skill_name and theory:
        from ..skills.query import method_label, resolve_label
        label = method_label(theory[0])
        skill_info = resolve_label(label)
        if skill_info:
            resolved = skill_info['skill_name']
            if isinstance(data, BaseModel):
                data = data.model_copy(update={'skill_name': resolved})
            else:
                data['skill_name'] = resolved
    return data






def seed_initial_image(data):
    """Fill in the crossing counts a wrapped periodic snapshot implies.

    The framework stores wrapped (main image) coordinates, so a snapshot
    whose molecules straddle a cell face is only continuous once the
    atoms that crossed are counted on the far side. Those counts cannot
    be recovered afterwards: the integrator accrues them from the moment
    it starts watching, and an equilibrated frame handed in as input has
    already crossed. Deriving them here, once, from the geometry the
    input already carries, is what lets both execution paths hand the
    engine the same (coordinate, count) pair.

    A run that declared its counts (or already accrued them, as a
    restart does) is left alone, and so is anything the derivation
    cannot answer: this is a seed, not a repair, and a wrong count would
    be worse than the missing one it replaces.
    """
    if data.get('image'):
        return data
    pbc = data.get('pbc') or []
    if not any(int(flag) for flag in pbc):
        return data
    lattice_vector = data.get('lattice_vector') or []
    atoms = data.get('atoms') or []
    coordinates = data.get('coordinates') or []
    if len(atoms) != len(coordinates):
        return data
    try:
        image = derive_initial_image(atoms, coordinates, lattice_vector,
                                     pbc=pbc)
    except (TypeError, ValueError):
        # The shape checks here are guards against malformed input, not
        # a supported call path: a structure this layer cannot read is
        # left exactly as it arrived, and validate_task reports it.
        return data
    if image and any(any(row) for row in image):
        # An all-zero answer needs no statement: the empty field already
        # reads that way downstream, and rows of zeros would ride along
        # in every checkpoint and history frame of a run that never
        # crosses anything.
        data['image'] = image
    return data


def _adaptive_switch(value):
    """Read one adaptive property switch as 0/1."""
    try:
        return 1 if int(value or 0) else 0
    except (TypeError, ValueError):
        return 0


def reconcile_adaptive_properties(nested_data):
    """Reconcile the adaptive property switches with the &md keywords.

    The adaptive switches are the global property plan; the &md flags say
    what the dynamics engine evaluates. The union is used, and the one
    direction that cannot be satisfied silently is made explicit:

    - adaptive on, md off: switch the md keyword on (nac -> nactype=nac,
      soc -> dosoc=1, dipole -> dodipole=1, radiative -> radiative=1);
    - adaptive off, md on: warn and switch the adaptive keyword on;
    - both off: the property is skipped.

    Energy and gradient are the ML training baseline and are required.

    Parameters
    ----------
    nested_data : dict
        Config model dump carrying the 'adaptive' and 'dynamics' sections.
    """
    adaptive = nested_data.get('adaptive')
    dynamics = nested_data.get('dynamics')
    if not isinstance(adaptive, dict) or not isinstance(dynamics, dict):
        return

    def warn(message):
        warnings.warn(message, UserWarning, stacklevel=3)

    if (_adaptive_switch(adaptive.get('sample_energy', 1)) == 0
            or _adaptive_switch(adaptive.get('sample_grad', 1)) == 0):
        raise ValueError(
            "adaptive sampling trains on energy and gradient; "
            "sample_energy and sample_grad must be 1")

    want_nac = _adaptive_switch(adaptive.get('sample_nac', 0)) == 1
    md_nac = str(dynamics.get('nactype', 'ktdc')).strip().lower() == 'nac'
    if want_nac and not md_nac:
        dynamics['nactype'] = 'nac'
        warn("adaptive sample_nac is on; dynamics.nactype switched to 'nac'")
    elif md_nac and not want_nac:
        adaptive['sample_nac'] = 1
        warn("dynamics.nactype is 'nac' but adaptive sample_nac is off; "
             "enabling sample_nac")

    for adaptive_key, md_key in (
            ('sample_soc', 'dosoc'),
            ('sample_dipole', 'dodipole')):
        want = _adaptive_switch(adaptive.get(adaptive_key, 0)) == 1
        md_has = _adaptive_switch(dynamics.get(md_key, 0)) == 1
        if want and not md_has:
            dynamics[md_key] = 1
            warn("%s is on; dynamics.%s switched on" % (adaptive_key, md_key))
        elif md_has and not want:
            adaptive[adaptive_key] = 1
            warn("dynamics.%s is on but adaptive %s is off; enabling %s"
                 % (md_key, adaptive_key, adaptive_key))

    want_radiative = (
        _adaptive_switch(adaptive.get('sample_transition_dipole', 0)) == 1
        or _adaptive_switch(adaptive.get('sample_oscillator_strength', 0)) == 1)
    md_radiative = _adaptive_switch(dynamics.get('radiative', '0')) == 1
    if want_radiative and not md_radiative:
        dynamics['radiative'] = '1'
        warn("an adaptive radiative channel is on; dynamics.radiative "
             "switched on")
    elif md_radiative and not want_radiative:
        adaptive['sample_transition_dipole'] = 1
        adaptive['sample_oscillator_strength'] = 1
        warn("dynamics.radiative is on but the adaptive radiative switches "
             "are off; enabling sample_transition_dipole and "
             "sample_oscillator_strength")

    # The channels this build labels stop at energy, gradient, NAC and SOC
    # (`records/adaptive.TRAINING_UNITS`). Refused here, after both the
    # adaptive and the md side have spoken, because `dodipole`/`radiative`
    # can turn these on; the workflow's start-up guard used to restate it.
    for key in ('sample_dipole', 'sample_transition_dipole',
                'sample_oscillator_strength'):
        if _adaptive_switch(adaptive.get(key, 0)) == 1:
            raise ValueError(
                "adaptive %s is on, but this build does not label that "
                "channel" % key)


def normalize(configs):
    """Normalize a PyRAIConfig or the config within a pipeline dict.

    Accepts either a PyRAIConfig instance directly or a pipeline
    context dict with key 'config'. Applies all normalization
    transformations and returns the same type.

    Parameters
    ----------
    configs : PyRAIConfig or dict
        Config instance or pipeline context dict with key 'config'.

    Returns
    -------
    PyRAIConfig or dict
        Normalized config (same type as input).
    """
    if isinstance(configs, BaseModel):
        config = configs
    else:
        config = configs['config']

    nested_data = config.model_dump()
    reconcile_adaptive_properties(nested_data)
    data = flatten_config_dump(nested_data)
    data = normalize_jobtype(data)
    data = normalize_method_labels(data)
    data = compute_derived_quantities(data)
    data = seed_initial_image(data)
    data = derive_current_state(data)
    data = normalize_history(data)
    data = normalize_skill_name(data)
    data = normalize_constraints(data)
    submodel_data = partition_to_submodels(data)
    # Nested sections the flat namespace cannot represent: the workflow
    # sections share the dispatch keywords (multitask/remotetask/
    # multipool appear in several of them), and the environment
    # snapshot owns keys no submodel field list claims -- both would be
    # rebuilt from defaults. Restore them from the model dump, then
    # overlay the normalized values of the keys the submodels actually
    # own (e.g. normalize_history mutates dynamics.history).
    for submodel_name in RESTORED_NESTED_SECTIONS:
        if submodel_name not in nested_data:
            continue
        section = dict(nested_data[submodel_name])
        for key, value in submodel_data.get(submodel_name, {}).items():
            if key in section:
                section[key] = value
        submodel_data[submodel_name] = section
    result = Config.model_validate(submodel_data)

    if isinstance(configs, BaseModel):
        return result
    configs['config'] = result
    return configs
