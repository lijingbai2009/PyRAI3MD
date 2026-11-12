"""
======================================================================
Canonical multi-center partition plan (dev/MULTICENTER_DEVELOPMENT.md
sections 3-5, refactor phase R1).

This module is the single source of truth for the "center" partition
axis. Every calculation carries a length-C center plan; the ordinary
single-center run is C=1. C=1 and C>1 differ only in data length and
physical strategy, never in data source or code path.

The plan is a lightweight, frozen, serializable data contract built
from the normalized framework configuration. It answers, per center:

- which atoms belong to the high/mid region (0-based),
- the static electronic definition (ci, spin, nstate, state_spin,
  coupling pairs, NAC/SOC pair maps, initial root).

Mutable per-step electronic data (current_state, density) never lives
here; only the static electronic_spec does.

Design rules enforced by this module:
- external region/coupling/root indices are 1-based, internal are
  0-based; the values reaching build_center_plan are ALREADY
  normalized to 0-based by the adapter, so this module never
  re-decrements.
- four identities stay separate: trajectory_index / center_index
  (contiguous 0-based) / task_slot+task_id / local_state_index
  (within [0, nstate[c])).

Sep 08 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Tuple


# Bumped whenever the serialized center-plan layout or derivation
# semantics change. Persisted with the plan so restart can migrate or
# refuse stale records (docs section 11).
# 1.1: CenterSpec carries per-center charge/multiplicity; PartitionPlan
#      carries the aggregated full-system (L(real)) charge/multiplicity.
# 1.2: an undeclared coupling stays undeclared. The plan no longer
#      auto-generates adjacent pairs, so a persisted 1.1 plan would carry
#      pair maps its input never declared.
CENTER_PLAN_VERSION = '1.2'


class CenterPlanError(ValueError):
    """Raised for any structural/per-center plan inconsistency.

    Carries a fully qualified field path plus the human (1-based) and
    internal (0-based) center ordinals so the message never needs a
    second lookup (docs section 5.2).
    """


def _center_tag(center_index):
    """Human + internal center ordinal for error messages."""
    return "center %d (c=%d)" % (center_index + 1, center_index)


def _is_nested(value):
    """True when value is a non-empty list whose first item is a list.

    Used to tell the flat (single-center) form from the nested
    (per-center) form of ci/spin.
    """
    return (isinstance(value, (list, tuple)) and len(value) > 0
            and isinstance(value[0], (list, tuple)))


def _is_per_center(value, group_depth):
    """True when value is a per-center (nested) list of groups.

    ``group_depth`` is the nesting depth of ONE center's group: 1 for
    ci/spin/highlevel (a group is a flat list of scalars) and 2 for
    coupling (a group is a list of [int, int] pairs).

    The generic ``_is_nested`` cannot separate coupling's flat
    single-center form ``[[0, 1], [1, 2]]`` (a list of pairs) from a
    nested per-center form ``[[[0, 1]], [[1, 2]]]`` (a list of pair
    lists): both have a list as their first item. Descending
    ``group_depth`` levels from ``value[0]`` disambiguates them -- only
    the nested form keeps finding lists all the way down.
    """
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return False
    node = value[0]
    for _ in range(group_depth):
        if not isinstance(node, (list, tuple)):
            return False
        if len(node) == 0:
            # An empty group is an explicit per-center "no entries"
            # marker (e.g. coupling [] for one center), so treat the
            # container as nested rather than flat.
            return True
        node = node[0]
    return True


def _as_int(value, path):
    """Coerce value to a plain int, rejecting bool and truncating float.

    docs section 5.2: indices must be integers, not bool or a
    silently truncated float.
    """
    if isinstance(value, bool):
        raise CenterPlanError(
            "%s: expected an integer index, got bool %r." % (path, value))
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            raise CenterPlanError(
                "%s: expected an integer index, got float %r "
                "(truncation is not allowed)." % (path, value))
        raise CenterPlanError(
            "%s: expected an integer index, got non-integral float %r."
            % (path, value))
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            raise CenterPlanError(
                "%s: expected an integer index, got %r." % (path, value))
    raise CenterPlanError(
        "%s: expected an integer index, got %r of type %s."
        % (path, value, type(value).__name__))


@dataclass(frozen=True)
class ElectronicSpec:
    """Static electronic definition of one center (docs 3.2).

    All state indices are LOCAL to the center, 0-based, within
    [0, nstate). Pair maps keep the normalized (sorted) pair order; the
    NAC sign-flip / SOC conjugate rule for reversed pairs is a consumer
    responsibility recorded here, not applied to the stored data.
    """

    ci: Tuple[int, ...]
    spin: Tuple[int, ...]
    nstate: int
    state_spin: Tuple[int, ...]
    coupling: Tuple[Tuple[int, int], ...]
    nac_pair_map: Tuple[Tuple[int, int], ...]
    soc_pair_map: Tuple[Tuple[int, int], ...]
    initial_root: int

    @property
    def nnac(self):
        return len(self.nac_pair_map)

    @property
    def nsoc(self):
        return len(self.soc_pair_map)

    @property
    def npair(self):
        return len(self.coupling)

    def to_dict(self):
        return {
            'ci': list(self.ci),
            'spin': list(self.spin),
            'nstate': self.nstate,
            'state_spin': list(self.state_spin),
            'coupling': [list(p) for p in self.coupling],
            'nac_pair_map': [list(p) for p in self.nac_pair_map],
            'soc_pair_map': [list(p) for p in self.soc_pair_map],
            'initial_root': self.initial_root,
        }


@dataclass(frozen=True)
class CenterSpec:
    """Static geometry + electronic definition of one center (docs 3.2).

    high_indices/mid_indices are 0-based atom indices. mid_indices is
    empty for a two-layer center. A plain single-center run with no
    region definition has empty high_indices (it covers the real
    system).

    charge/multiplicity are this center region's electronic structure
    (docs 4/8): the high-level region task H(c) and its low-level
    correction L(c) use them, and a three-layer mid region I_c inherits
    them one-to-one. The full-system baseline L(real) uses the
    PartitionPlan aggregate instead.
    """

    center_index: int
    high_indices: Tuple[int, ...]
    mid_indices: Tuple[int, ...]
    electronic_spec: ElectronicSpec
    charge: int = 0
    multiplicity: int = 1

    @property
    def nstate(self):
        return self.electronic_spec.nstate

    def to_dict(self):
        return {
            'center_index': self.center_index,
            'high_indices': list(self.high_indices),
            'mid_indices': list(self.mid_indices),
            'electronic_spec': self.electronic_spec.to_dict(),
            'charge': self.charge,
            'multiplicity': self.multiplicity,
        }


@dataclass(frozen=True)
class PartitionPlan:
    """Versioned, fingerprinted center partition plan (docs 3.2).

    n_center is C; n_layer is 2 or 3 (three-layer ONIOM) or 1 for a
    plain single-method run. centers are in declared order and never
    re-sorted. supplied records which per-center fields the user gave
    explicitly, so `missing` stays distinguishable from an explicit
    `[]`/`0` (docs 4.4.4).

    system_charge/system_multiplicity are the aggregated full-system
    (L(real)) electronic definition (docs 8): system_charge is the sum
    of the per-center charges; system_multiplicity comes from pairing
    off the centers' unpaired electrons, i.e.
    ``(sum(mult_c - 1) % 2) + 1`` (triplet + triplet -> singlet). For a
    neutral closed-shell single center they reduce to that center's own
    charge/multiplicity.
    """

    version: str
    plan_fingerprint: str
    n_center: int
    n_layer: int
    centers: Tuple[CenterSpec, ...]
    supplied: Dict[str, bool] = field(default_factory=dict)
    system_charge: int = 0
    system_multiplicity: int = 1

    def center(self, c):
        """Bounded center read interface (docs 3.3)."""
        if not isinstance(c, int) or isinstance(c, bool):
            raise CenterPlanError(
                "center index must be an int, got %r." % (c,))
        if c < 0 or c >= self.n_center:
            raise CenterPlanError(
                "center index %d out of range [0, %d)."
                % (c, self.n_center))
        return self.centers[c]

    @property
    def is_single(self):
        return self.n_center == 1

    def to_dict(self):
        return {
            'version': self.version,
            'plan_fingerprint': self.plan_fingerprint,
            'n_center': self.n_center,
            'n_layer': self.n_layer,
            'centers': [c.to_dict() for c in self.centers],
            'supplied': dict(self.supplied),
            'system_charge': self.system_charge,
            'system_multiplicity': self.system_multiplicity,
        }


# ----------------------------------------------------------------------
# Per-center electronic derivation
# ----------------------------------------------------------------------

def derive_state_spin(ci, spin, path):
    """Expand per-spin-block ci/spin into a per-state 2S list.

    ci and spin must have equal length; each block contributes ci[i]
    states with spin quantum number S=spin[i], stored as 2S. This is
    the strict replacement for the old `zip(ci, spin)` which silently
    truncated to the shorter list (docs 5.2, R0 defect).
    """
    if len(ci) != len(spin):
        raise CenterPlanError(
            "%s: len(ci)=%d but len(spin)=%d; they must match."
            % (path, len(ci), len(spin)))
    state_spin = []
    for n_states, s in zip(ci, spin):
        two_s = s * 2
        state_spin.extend([two_s] * n_states)
    return state_spin


def classify_pairs(coupling, state_spin, nstate, path):
    """Split coupling pairs into NAC (same spin) and SOC (diff spin).

    Every pair must be exactly two distinct integers within
    [0, nstate). Out-of-range pairs raise instead of falling back to
    the NAC default (docs 5.2). Duplicate unordered pairs raise.
    """
    nac_pairs = []
    soc_pairs = []
    seen = set()
    for pair in coupling:
        if len(pair) != 2:
            raise CenterPlanError(
                "%s: coupling pair %r must have exactly two states."
                % (path, list(pair)))
        a = _as_int(pair[0], path)
        b = _as_int(pair[1], path)
        if a == b:
            raise CenterPlanError(
                "%s: self-coupling pair [%d, %d] is not allowed."
                % (path, a, b))
        for idx in (a, b):
            if idx < 0 or idx >= nstate:
                raise CenterPlanError(
                    "%s: state %d exceeds this center's internal "
                    "range 0..%d." % (path, idx, nstate - 1))
        key = (min(a, b), max(a, b))
        if key in seen:
            raise CenterPlanError(
                "%s: duplicate coupling pair [%d, %d]." % (path, a, b))
        seen.add(key)
        if state_spin[a] == state_spin[b]:
            nac_pairs.append((a, b))
        else:
            soc_pairs.append((a, b))
    return tuple(nac_pairs), tuple(soc_pairs)


def _pair_rows(pairs):
    """The valid pair rows of a pair list, as lists of two ints.

    A row that is not exactly two integers is the "not filled" sentinel
    the config defaults carry ([[]]), never a declaration, so it reads as
    absent. Anything else is passed through so :func:`classify_pairs`
    names the offender instead of a filter hiding it.
    """
    rows = []
    for pair in pairs or []:
        row = list(pair)
        if len(row) == 2 or len(row) == 0:
            rows.append(row)
        else:
            raise CenterPlanError(
                "coupling pair %r must have exactly two states." % (row,))
    return [row for row in rows if row]


def derive_pair_declaration(coupling, nac_pairs, soc_pairs, state_spin,
                            nstate, path):
    """Reconcile a coupling declaration with its derived pair maps.

    This is the one derivation of the coupling axis, shared by the flat
    single-center path (schema.normalize) and the per-center plan. The
    three keys are one statement in two spellings: `coupling` is the input
    vocabulary and the only one a classic input file can carry, while
    `nac_pair_map` / `soc_pair_map` are the internal labels every consumer
    reads - and, in the nested yaml/json formats, an extra entry a user may
    fill in instead. Either side fills in the other; a disagreement is a
    broken input, so it raises. Overwriting whichever side the caller
    happened to look at last silently discarded what was declared.

    Nothing is invented. Adjacent pairs used to be supplied for any run
    with two or more states, which read back as a computed coupling for a
    pair nobody asked about - and made the "neither side declares
    anything" case unreachable, because the authority was never empty. A
    pair that is not listed is shielded; whether a run may go without the
    coupling it consumes is answered where the need is known
    (planning/assembly.py::_require_declared_couplings).

    Parameters
    ----------
    coupling : list of [int, int] or None
        Declared 0-based state pairs.
    nac_pairs, soc_pairs : list of [int, int] or None
        The pair maps as declared or as previously derived.
    state_spin : list of int
        2S per state; same-spin pairs are NAC, different-spin ones SOC.
    nstate : int
        Number of states the pairs are indexed against.
    path : str
        Field path prefixing every error.
    Returns
    -------
    tuple
        (coupling, nac_pairs, soc_pairs) with the authority filled in:
        coupling as list of list of int, the two maps as tuples of tuples.
    """
    coup = _pair_rows(coupling)
    nac_in = _pair_rows(nac_pairs)
    soc_in = _pair_rows(soc_pairs)

    if not coup and (nac_in or soc_in):
        coup = [list(p) for p in nac_in] + [list(p) for p in soc_in]
    elif not coup:
        # Nothing on either side: no couplings, and none invented. Whether
        # the run may go without them is the requester's question, answered
        # where the need is known (planning/assembly.py::
        # _require_declared_couplings), not by guessing here.
        return [], (), ()

    if len(state_spin) < nstate:
        raise CenterPlanError(
            "%s: a coupling pair was declared but state_spin describes "
            "only %d of the %d states; ci/spin must define every state "
            "before a pair can be told apart as NAC or SOC."
            % (path, len(state_spin), nstate))

    nac, soc = classify_pairs(coup, state_spin, nstate, path)
    if nac_in or soc_in:
        if {tuple(p) for p in nac_in} != set(nac) or \
                {tuple(p) for p in soc_in} != set(soc):
            raise ValueError(
                "%s: coupling %r and the declared pair maps "
                "nac_pair_map=%r soc_pair_map=%r are one statement, and "
                "these two disagree. Correct one side; the framework will "
                "not overwrite the other."
                % (path, coup, nac_in, soc_in))
    return coup, nac, soc


def pairs_to_coupling(nac_pairs, soc_pairs):
    """The `coupling` spelling of a declaration made only in pair maps.

    Used where a declaration has to be rendered into a format that carries
    only `coupling` (the classic input writer). It does not classify or
    validate: the derivation that owns those rules is
    :func:`derive_pair_declaration`, which runs when the configuration is
    loaded.
    """
    return [list(p) for p in _pair_rows(nac_pairs) + _pair_rows(soc_pairs)]


def derive_center_electronic(ci, spin, coupling, root, center_index=0):
    """Derive one center's ElectronicSpec with strict per-center checks.

    Parameters
    ----------
    ci : list of int
        Number of states per spin block (non-negative, sum > 0).
    spin : list of int
        Total spin S per block; must match ci in length.
    coupling : list of [int, int] or None
        0-based local state pairs. Absent or empty means this center
        declares no couplings: nothing is invented for it, and a pair a
        center does not list is shielded rather than zero-filled.
    root : int
        Initial local state, 0-based, within [0, nstate).
    center_index : int
        Internal center ordinal for error messages.

    Returns
    -------
    ElectronicSpec
    """
    path_ci = "molecule.ci[%d]" % center_index
    path_spin = "molecule.spin[%d]" % center_index

    ci = [_as_int(v, path_ci) for v in ci]
    spin = [_as_int(v, path_spin) for v in spin]

    if not ci:
        raise CenterPlanError(
            "%s: ci must be a non-empty list of state counts." % path_ci)
    for v in ci:
        if v < 0:
            raise CenterPlanError(
                "%s: negative state count %d is not allowed." % (path_ci, v))
    if sum(ci) <= 0:
        raise CenterPlanError(
            "%s: total number of states must be > 0, got 0." % path_ci)
    for v in spin:
        if v < 0:
            raise CenterPlanError(
                "%s: negative spin %d is not allowed." % (path_spin, v))

    nstate = sum(ci)
    state_spin = derive_state_spin(ci, spin, "molecule")

    # One derivation for the coupling axis, shared with the flat path.
    coupling, nac_pairs, soc_pairs = derive_pair_declaration(
        coupling, None, None, state_spin, nstate,
        "molecule.coupling[%d]" % center_index)

    root = _as_int(root, "md.root[%d]" % center_index)
    if root < 0 or root >= nstate:
        raise CenterPlanError(
            "md.root[%d]: %s has nstate=%d; the input root=%d (1-based) "
            "is invalid; expected 1..%d."
            % (center_index, _center_tag(center_index),
               nstate, root + 1, nstate))

    return ElectronicSpec(
        ci=tuple(ci),
        spin=tuple(spin),
        nstate=nstate,
        state_spin=tuple(state_spin),
        coupling=tuple(tuple(p) for p in coupling),
        nac_pair_map=nac_pairs,
        soc_pair_map=soc_pairs,
        initial_root=root,
    )


# ----------------------------------------------------------------------
# Center-count / group helpers
# ----------------------------------------------------------------------

def count_geometry_centers(highlevel):
    """Number of geometric centers implied by highlevel.

    Flat non-empty list -> 1; nested list -> len; empty -> 1 (a plain
    run materializes one center covering the real system, docs 4.4.1).
    """
    if not highlevel:
        return 1
    if _is_nested(highlevel):
        return len(highlevel)
    return 1


def as_center_groups(value, n_center, path, default=None, group_depth=1,
                     broadcast=False):
    """Normalize a per-center field to exactly n_center groups.

    Flat (single-center) input wraps into one group when n_center == 1.
    For n_center > 1 the behavior splits by ``broadcast``:

    - broadcast=False (region fields highlevel/midlevel): a flat value
      or a wrong group count is rejected -- regions are never guessed
      or replicated (docs 4.4.2/4.4.3).
    - broadcast=True (electronic fields ci/spin/coupling/charges/
      multiplicity): an incomplete setting is completed by copying the
      FIRST center's group to every missing center (docs 4.4). A flat
      value counts as one provided group; k provided groups with
      1 <= k < n_center pad centers k..n_center-1 from group 0; k >
      n_center is an error; k == n_center is used as-is.

    A missing value (None) falls back to `default` replicated per
    center.

    ``group_depth`` (see _is_per_center) separates a flat single-center
    value from a nested per-center one: 1 for ci/spin/highlevel/
    charges/multiplicity, 2 for coupling (whose flat form is already a
    list of pairs).
    """
    if value is None:
        if default is None:
            return [None] * n_center
        return [list(default) for _ in range(n_center)]

    if _is_per_center(value, group_depth):
        groups = [list(g) for g in value]
    else:
        # Flat form: one provided group.
        groups = [list(value)]

    if len(groups) == n_center:
        return groups

    if len(groups) > n_center:
        raise CenterPlanError(
            "%s: expected %d center groups, got %d."
            % (path, n_center, len(groups)))

    # Fewer groups than centers.
    if not broadcast:
        if len(groups) == 1 and n_center != 1:
            raise CenterPlanError(
                "%s: a flat value cannot be broadcast to %d centers; "
                "write one group per center." % (path, n_center))
        raise CenterPlanError(
            "%s: expected %d center groups, got %d." % (path, n_center,
                                                        len(groups)))

    # Broadcast: fill every missing center from the first center's
    # group (docs 4.4: an incomplete multi-center setting copies the
    # first center).
    first = groups[0]
    while len(groups) < n_center:
        groups.append(list(first))
    return groups


def aggregate_system_multiplicity(multiplicities):
    """Full-system multiplicity from per-center multiplicities (docs 8).

    The centers' unpaired electrons pair off (antiferromagnetic-style
    cancellation), so the residual unpaired count is the parity of the
    total unpaired electrons and the system multiplicity is 1 or 2:

        total_unpaired = sum(mult_c - 1) % 2
        system_multiplicity = total_unpaired + 1

    Examples: triplet + triplet -> singlet, doublet + doublet ->
    singlet, triplet + doublet -> doublet, all-singlet -> singlet.
    """
    total_unpaired = sum(int(m) - 1 for m in multiplicities)
    return (total_unpaired % 2) + 1


def as_center_scalars(value, n_center, path, default=0, broadcast=True):
    """Normalize a per-center SCALAR field to exactly n_center values.

    root/charges/multiplicity carry one scalar per center (unlike
    ci/spin/coupling whose group is a list), so they bypass
    as_center_groups. A scalar counts as one provided value; a list is
    the per-center form. With ``broadcast`` an incomplete setting
    (fewer than n_center values) copies the FIRST center's value to
    every missing center (docs 4.4); more than n_center is an error.
    A missing value (None) falls back to ``default`` per center.
    """
    if value is None:
        return [default] * n_center
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    if not values:
        return [default] * n_center
    if len(values) == n_center:
        return values
    if len(values) > n_center:
        raise CenterPlanError(
            "%s: expected %d center values, got %d."
            % (path, n_center, len(values)))
    if not broadcast:
        raise CenterPlanError(
            "%s: a scalar value cannot be broadcast to %d centers; "
            "write one value per center." % (path, n_center))
    first = values[0]
    while len(values) < n_center:
        values.append(first)
    return values


# ----------------------------------------------------------------------
# Plan builder
# ----------------------------------------------------------------------

def _region_indices(highlevel, midlevel, n_center):
    """Normalize highlevel/midlevel to per-center 0-based index tuples."""
    high_groups = as_center_groups(
        highlevel, n_center, "molecule.highlevel", default=[])
    if midlevel:
        # midlevel is the per-center environment of a three-layer ONIOM
        # and must correspond one-to-one with the highlevel centers;
        # highlevel is the sole source of the center count, so a
        # midlevel group-count mismatch is a hard error, never a
        # broadcast or a guess (docs 4/5).
        n_mid = len(midlevel) if _is_nested(midlevel) else 1
        if n_mid != n_center:
            raise CenterPlanError(
                "molecule.midlevel: %d region group(s) given but "
                "highlevel defines %d center(s); midlevel must match "
                "highlevel one-to-one (highlevel is the sole source of "
                "the center count)." % (n_mid, n_center))
        mid_groups = as_center_groups(
            midlevel, n_center, "molecule.midlevel", default=[])
    else:
        mid_groups = [[] for _ in range(n_center)]
    highs = []
    mids = []
    for c in range(n_center):
        highs.append(tuple(_as_int(i, "molecule.highlevel[%d]" % c)
                           for i in high_groups[c]))
        mids.append(tuple(_as_int(i, "molecule.midlevel[%d]" % c)
                          for i in mid_groups[c]))
    return highs, mids


def _fingerprint(payload):
    """Stable content fingerprint (never Python's randomized hash)."""
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]



def _read_flat_fields(source):
    """Pull the center-relevant fields from a config or flat dict.

    Accepts a PyRAIConfig (reads .system/.dynamics), a System-like
    model, or a flat dict (normalize pipeline medium).
    """
    system = getattr(source, 'system', None)
    dynamics = getattr(source, 'dynamics', None)
    if system is not None:
        get_sys = lambda k, d=None: getattr(system, k, d)  # noqa: E731
    elif isinstance(source, dict):
        get_sys = lambda k, d=None: source.get(k, d)  # noqa: E731
    else:
        get_sys = lambda k, d=None: getattr(source, k, d)  # noqa: E731

    if dynamics is not None:
        root = getattr(dynamics, 'root', 0)
    elif isinstance(source, dict):
        root = source.get('root', 0)
    else:
        root = getattr(source, 'root', 0)

    return {
        'highlevel': get_sys('highlevel', []),
        'midlevel': get_sys('midlevel', []),
        'ci': get_sys('ci', [1]),
        'spin': get_sys('spin', [0]),
        'coupling': get_sys('coupling', None),
        'charges': get_sys('charges', 0),
        'multiplicity': get_sys('multiplicity', 1),
        'root': root,
    }


def build_center_plan(source, supplied=None):
    """Build the canonical PartitionPlan from a normalized config.

    Parameters
    ----------
    source : PyRAIConfig or System-like model or dict
        Normalized configuration (0-based indices already applied by
        the adapter).
    supplied : dict or None
        Optional provenance map {'coupling': bool, ...} recording which
        fields the user gave explicitly. Absent -> legacy semantics
        (an empty coupling field auto-generates adjacent pairs).

    Returns
    -------
    PartitionPlan

    Raises
    ------
    CenterPlanError
        On any structural or per-center inconsistency; raised before
        any backend directory, thread or Slurm submission (docs 5.1).
    """
    supplied = dict(supplied or {})
    f = _read_flat_fields(source)

    highlevel = f['highlevel'] or []
    midlevel = f['midlevel'] or []
    n_center = count_geometry_centers(highlevel)

    # A plain run has no region definition; a nested ci/spin/root with
    # multiple groups but no highlevel cannot guess atom partitioning
    # (docs 4.4.1).
    if n_center == 1 and not highlevel:
        for name in ('ci', 'spin', 'root'):
            value = f[name]
            if name == 'root':
                if isinstance(value, (list, tuple)) and len(value) > 1:
                    raise CenterPlanError(
                        "md.root: %d center groups given but no "
                        "highlevel defines multiple centers." % len(value))
            elif _is_nested(value) and len(value) > 1:
                raise CenterPlanError(
                    "molecule.%s: %d center groups given but no "
                    "highlevel defines multiple centers."
                    % (name, len(value)))

    n_layer = 3 if midlevel else (2 if highlevel else 1)

    highs, mids = _region_indices(highlevel, midlevel, n_center)

    ci_groups = as_center_groups(f['ci'], n_center, "molecule.ci",
                                 default=[1], broadcast=True)
    spin_groups = as_center_groups(f['spin'], n_center, "molecule.spin",
                                   default=[0], broadcast=True)
    # A flat empty coupling (missing / C=1 default []) means "not
    # provided" and falls back to the legacy adjacent-pair default per
    # center; an explicit nested empty group [] means "this center has
    # no pairs" (docs 4.4.5).
    coupling_raw = f['coupling']
    if not coupling_raw:
        coupling_raw = None
    coupling_groups = as_center_groups(
        coupling_raw, n_center, "molecule.coupling", default=None,
        group_depth=2, broadcast=True)

    # Per-center scalars (root/charges/multiplicity): an incomplete
    # multi-center setting copies the first center (docs 4.4).
    root_groups = as_center_scalars(
        f['root'], n_center, "md.root", default=0, broadcast=True)
    charge_groups = [_as_int(v, "molecule.charges[%d]" % c)
                     for c, v in enumerate(as_center_scalars(
                         f['charges'], n_center, "molecule.charges",
                         default=0, broadcast=True))]
    mult_groups = [_as_int(v, "molecule.multiplicity[%d]" % c)
                   for c, v in enumerate(as_center_scalars(
                       f['multiplicity'], n_center, "molecule.multiplicity",
                       default=1, broadcast=True))]
    for c, m in enumerate(mult_groups):
        if m < 1:
            raise CenterPlanError(
                "molecule.multiplicity[%d]: %s multiplicity must be "
                ">= 1, got %d." % (c, _center_tag(c), m))

    centers = []
    for c in range(n_center):
        spec = derive_center_electronic(
            ci_groups[c], spin_groups[c], coupling_groups[c],
            root_groups[c], center_index=c)
        centers.append(CenterSpec(
            center_index=c,
            high_indices=highs[c],
            mid_indices=mids[c],
            electronic_spec=spec,
            charge=charge_groups[c],
            multiplicity=mult_groups[c],
        ))

    centers = tuple(centers)
    # Full-system (L(real)) aggregate: charge is the sum of the center
    # charges; multiplicity pairs off the centers' unpaired electrons
    # (docs 8).
    system_charge = sum(charge_groups)
    system_multiplicity = aggregate_system_multiplicity(mult_groups)
    payload = {
        'version': CENTER_PLAN_VERSION,
        'n_center': n_center,
        'n_layer': n_layer,
        'centers': [c.to_dict() for c in centers],
        'system_charge': system_charge,
        'system_multiplicity': system_multiplicity,
    }
    return PartitionPlan(
        version=CENTER_PLAN_VERSION,
        plan_fingerprint=_fingerprint(payload),
        n_center=n_center,
        n_layer=n_layer,
        centers=centers,
        supplied=supplied,
        system_charge=system_charge,
        system_multiplicity=system_multiplicity,
    )
