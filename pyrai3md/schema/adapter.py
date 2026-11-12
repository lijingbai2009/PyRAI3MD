"""
======================================================================
This module provides adapter functions for converting interface-layer
raw dictionaries (sectioned format) into unified PyRAI3MD Config
objects. It handles section flattening, type conversion, and Pydantic
model construction.

The adapter bridges the gap between Interface (pure splitting, raw
strings) and Schema (structure validation, typed values).

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import copy
import json
from pydantic import BaseModel

from .config import (Config, FIELD_TO_SUBMODEL,
                     AMBIGUOUS_FIELD_NAMES,
                     SUBMODEL_CLASSES, NESTED_FRAMEWORK_SECTIONS,
                     NESTED_SECTION_FIELDS, RETIRE_ENVIRONMENT_KEYS,
                     DEFAULT)
from ..utilities.config_merge import deep_merge
from ..utilities.read_tools import ReadVal, ReadIndex


# Type mapping for each section's keywords
# Maps keyword name -> ReadVal type code
CONTROL_TYPES = {
    'title': 's',
    # jobtype is handled specially in _convert_types (full space-separated
    # string preserved; ReadVal('s') would keep only the first token).
    'jobtype': 's',
    'theory': 'sl', 'reference': 'sl',   # Formal keywords
    'qm': 'sl', 'abinit': 'sl',          # Backward compatible aliases
    'driver': 's',                       # Execution driver (main/skill/skill name)
    'multiscale_type': 's',              # qmmm/oniom for two theory entries
    'gl_seed': 'i',
    'verbose': 'i', 'silent': 'i',
    # Global test switch. 'b' is the boolean code of read_tools.ReadVal,
    # which accepts only true/1/false/0 in the classic text.
    'debug': 'b',
}

ADAPTIVE_TYPES = {
    'remote_train': 'i', 'load': 'i', 'transfer': 'i', 'pop_step': 'i',
    'maxiter': 'i', 'maxsample': 'i', 'record_step': 'i', 'keep_md': 'i',
    # `pop_tolerance` stays in the table so a classic input carrying it is
    # converted and then refused by name with the `md_maxpop` pointer, rather
    # than disappearing as an unknown keyword.
    'check_population': 'i', 'pop_tolerance': 'f',
    'check_energy_order': 'i', 'gap_tolerance': 'f',
    'sample_energy': 'i', 'sample_grad': 'i', 'sample_nac': 'i',
    'sample_soc': 'i', 'sample_dipole': 'i',
    'sample_transition_dipole': 'i', 'sample_oscillator_strength': 'i',
    'dynsample': 'i', 'maxdiscard': 'i', 'maxmae': 'f',
    'analyze_pes': 'i', 'analyze_gap': 's', 'analyze_trace': 'i',
    'analyze_coverage': 'i', 'analyze_window': 'i',
    'analyze_local_tolerance': 'f', 'analyze_global_tolerance': 'f',
    'predict_mode': 'i', 'gap_threshold': 'f',
    'explore_udd': 'i', 'explore_gap': 'i',
    'udd_mode': 's', 'udd_amplitude': 'f', 'udd_width': 'f',
    'udd_start_iteration': 'i',
    'gap_method': 's', 'gap_pair': 'il',
    'gap_height': 'f', 'gap_width': 'f',
    'multitask': 'b', 'remotetask': 'b', 'multipool': 'i',
    'md_multitask': 'b', 'md_remotetask': 'b', 'md_multipool': 'i',
    'reference_multitask': 'b', 'reference_remotetask': 'b',
    'reference_multipool': 'i',
    'refine': 'i', 'refine_num': 'i', 'refine_start': 'i', 'refine_end': 'i',
    'refine_gap': 'f',
    'maxenergy': 'f', 'minenergy': 'f', 'dynenergy': 'f', 'inienergy': 'f',
    'fwdenergy': 'i', 'bckenergy': 'i',
    'maxgrad': 'f', 'mingrad': 'f', 'dyngrad': 'f', 'inigrad': 'f',
    'fwdgrad': 'i', 'bckgrad': 'i',
    'maxnac': 'f', 'minnac': 'f', 'dynnac': 'f', 'ininac': 'f',
    'fwdnac': 'i', 'bcknac': 'i',
    'maxsoc': 'f', 'minsoc': 'f', 'dynsoc': 'f', 'inisoc': 'f',
    'fwdsoc': 'i', 'bcksoc': 'i',
}

# Fragment groups of the center axis, read as 0-based index lists by the
# composite planner. It is `center_groups` and not `groups` because the
# constraint block names its atom groups `groups` too, and the two were one
# field until #272 (see `config.RETIRE_CONSTRAINT_KEYS['groups']`).
MOLECULE_TYPES = {
    'ci': 'il', 'spin': 'il', 'coupling': 'g',
    'current_state': 'i',
    'qmmm_key': 's', 'qmmm_xyz': 's',
    'q_qm': 'fl',
    'external_charges': 's',
    'highlevel': 'si', 'midlevel': 'si',
    'highref': 'si', 'midref': 'si',
    'embedding': 's', 'read_charge': 's',
    'qm_contains_field_gradient': 'b',
    'sc_charge': 'b', 'sc_charge_maxiter': 'i',
    'sc_charge_threshold': 'f', 'sc_charge_damping': 'f',
    'center_groups': 'g',
    'track_type': 's', 'track_index': 'g', 'track_thrhd': 'fl', 'track_stop': 'i',
    'cell': 'fl', 'lattice': 'fl', 'pbc': 'il',
    # Molecule-scope dispatch keywords (tri-state; see System).
    'multitask': 'b', 'remotetask': 'b', 'multipool': 'i',
}

# Custom ReadIndex start offsets for 1-based input indices.
# Fields not listed here use the default start=0.
INDEX_START = {
    'coupling': 1,     # state indices are 1-based per documentation
    'root': 1,         # initial state is externally 1-based (internal 0-based)
    'highlevel': 1,    # atom indices in QM region
    'midlevel': 1,     # atom indices in QM2 region
    'highref': 1,      # atom indices in the reference high layer
    'midref': 1,       # atom indices in the reference mid layer
    'track_index': 1,  # tracking atom indices
}

MD_TYPES = {
    'initcond': 's', 'ninitcond': 'i', 'method': 's', 'format': 's',
    # The dispatch pair, alongside the pool size below. `&hessian`, `&search`,
    # `&train` and `&prediction` have carried all three since the fan-out was
    # opened there; `&MD` had only `multipool`, so a classic input that stated
    # `multitask true` handed the reader an untabled keyword - and #395 put
    # that line on the writer, so the table had to follow or the writer would
    # be stating something the parser does not know the type of.
    # `qualification/verification/test_qualification_slurm.py::
    # TestMdSectionEnsembleEcho` is what refused to let that pass.
    'multitask': 'b', 'remotetask': 'b',
    'multipool': 'i',
    'randvelo': 'i', 'temp': 'f', 'step': 'i', 'size': 'f',
    'root': 'i', 'activestate': 'i', 'history': 'i',
    'sfhp': 's', 'nactype': 's',
    'phasecheck': 'i',
    'gap': 'f', 'gapsoc': 'f', 'addsoc': 'f', 'dosoc': 'i',
    'dodipole': 'i', 'radiative': 's',
    'substep': 'i', 'rk4': 's', 'integrate': 'i', 'deco': 's',
    'adjust': 'i', 'reflect': 'i', 'maxh': 'i',
    # FSSH kernel implementation selection and profiling switch
    'fssh_rk4': 's', 'fssh_dpdt': 's', 'fssh_matb': 's',
    'fssh_ktdc': 's', 'fssh_decoherence': 's', 'fssh_profile': 'i',
    'perf_profile': 'i',
    'fssh_force_hop': 'i',
    'thermodelay': 'i', 'dissipate': 'i',
    # Ensemble control (NVT thermostats / NPT barostats). pressure is
    # read in bar; temp->temperature is aliased in FIELD_MAP.
    'thermostat': 's', 'barostat': 's', 'ensemble': 's', 'pressure': 'f',
    'thermostat_chain': 'i', 'thermostat_tau': 'f', 'barostat_tau': 'f',
    'compressibility': 'f', 'barostat_mode': 's',
    'thermostat_mass': 'f', 'barostat_mass': 'f',
    # Detection thresholds of an NN-driven md run (None/empty = not judged).
    'md_maxenergy': 'f', 'md_maxgrad': 'f',
    'md_maxnac': 'f', 'md_maxsoc': 'f', 'md_maxpop': 'f',
    'direct': 'i', 'buffer': 'i', 'checkpoint': 'i',
    'excess': 'f', 'scale': 'f', 'target': 'f',
    'reset': 'i', 'resetstep': 'i',
}

OPTIMIZE_TYPES = {
    'multipool': 'i',
    'opt_method': 's', 'opt_maxit': 'i',
    'opt_energy': 'f', 'opt_grms': 'f', 'opt_gmax': 'f',
    'opt_drms': 'f', 'opt_dmax': 'f',
    'opt_csrms': 'f', 'opt_csmax': 'f',
    'opt_cerms': 'f', 'opt_cemax': 'f',
    'opt_pressure': 'f', 'opt_cell_coupling': 's',
    'opt_trust_radius': 'f',
    'opt_use_rfo': 'b',
    # Scalar or one index per center; the list spelling is read by the
    # explicit branch in `_convert_types` below, because 'il' would turn a
    # single-token value into a one-entry list and the scalar path of a
    # single-center run must stay an int.
    'opt_state_i': 'i', 'opt_state_j': 'i',
    'optimize_cell': 'b',
    # The path family (`optimize.method` mep / irc / neb). Every one of these is
    # a scalar the classic `.input` carrier has to coerce, and none of them names
    # the section, so all of them take the `opt_` prefix.
    'opt_path_constraint': 's', 'opt_path_radius': 'f',
    'opt_path_inner_maxit': 'i', 'opt_path_inner_tol': 'f',
    'opt_irc_direction': 's', 'opt_irc_mode': 's', 'opt_irc_mode_file': 's',
    'opt_neb_reactant': 's', 'opt_neb_product': 's', 'opt_neb_ts': 's',
    'opt_neb_images': 'i', 'opt_neb_interpolate': 's',
    'opt_neb_geodesic_threshold': 'f', 'opt_neb_geodesic_scaler': 'f',
    'opt_neb_geodesic_min_neighbors': 'i',
    'opt_neb_spring': 'f', 'opt_neb_climb': 'b',
    'opt_neb_climb_threshold': 'f', 'opt_neb_step': 'f',
}

# Optimize-section keywords that keep their own spelling. The section prefixes
# its bare keywords (`method` -> `opt_method`) so they cannot collide with a
# skill section that defines the same word; a keyword that already carries the
# section's own name has no such collision, and prefixing it would invent a
# keyword no model owns, which is then dropped on the way to the submodel.
OPTIMIZE_PASSTHROUGH_KEYS = frozenset({'optimize_cell'})

HESSIAN_TYPES = {
    'multitask': 'b', 'remotetask': 'b', 'multipool': 'i',
    # `atoms` is deliberately absent: it is a list, and the section types table
    # is the list of scalars ReadVal is allowed to coerce.
    'hess_driver': 's', 'hess_mode': 's', 'hess_step': 'f',
    'hess_vibration': 'b', 'hess_thermo': 'b', 'hess_molden': 'b',
    'hess_temperature': 'f', 'hess_pressure': 'f', 'hess_scale': 'f',
    'hess_cutoff': 'f', 'hess_alpha': 'f', 'hess_symmetry': 'i',
    'hess_linear': 'b',
}

SEARCH_TYPES = {
    'multitask': 'b', 'remotetask': 'b', 'multipool': 'i',
}

# Dispatch-layer keywords shared by every workflow section.
DISPATCH_TYPES = {
    'multitask': 'b', 'remotetask': 'b', 'multipool': 'i',
}

# Composite workflow section keywords.
COMPOSITE_TYPES = {
    'executor': 's', 'max_workers': 'i',
}

# Read-only extraction section keywords. `path` is deliberately absent:
# it is one path or a list of paths, and this table is the scalar
# coercion surface (the same rule that keeps `atoms` out of HESSIAN_TYPES).
READ_TYPES = {
    'project': 's', 'profile': 's',
    'energy': 'b', 'gradient': 'b', 'nac': 'b', 'soc': 'b',
}

# Section type map for the constraint block. The block types itself: its own
# pydantic model owns the unit conversion, the index base and the capability
# refusals, so ReadVal must not touch a single value inside it. An empty map is
# the explicit statement of that, not an oversight. Both section names carry the
# same block - `constraint` is what a new input writes and `system_constraint`
# is the retired spelling that still routes - and both are listed, because
# `_raw_user_keys` reads this map to decide which sections are the framework's
# and therefore which keys a user actually provided.
CONSTRAINT_TYPES = {}

SECTION_TYPES = {
    'constraint': CONSTRAINT_TYPES,
    'system_constraint': CONSTRAINT_TYPES,
    'control': CONTROL_TYPES,
    'adaptive': ADAPTIVE_TYPES,
    'molecule': MOLECULE_TYPES,
    'md': MD_TYPES,
    'optimize': OPTIMIZE_TYPES,
    'hessian': HESSIAN_TYPES,
    'search': SEARCH_TYPES,
    'read': READ_TYPES,
    'composite': COMPOSITE_TYPES,
}

# Framework sections kept nested through the pipeline (section name
# -> submodel name); defined in config.py and shared with the
# normalize/validate repartitioning.

# Field name mapping: section keyword -> Config field name
FIELD_MAP = {
    'qm': 'theory',            # Backward compatible alias
    'abinit': 'reference',     # Backward compatible alias
    'temp': 'temperature',
    'lattice': 'lattice_vector',
}

# Fields that use semicolon-separated center parsing
_CENTER_INDEX_FIELDS = frozenset({'highlevel', 'midlevel',
                                  'highref', 'midref'})

# Per-center scalar-list fields (ci/spin): each center is a list of
# integers (state counts / spin quantum numbers). These are NOT
# indices, so no 1-based decrement is applied. A classic string uses
# ';' to split centers ('2 2; 2 2'); a typed value is passed through
# unchanged (normalized entry).
_CENTER_INT_LIST_FIELDS = frozenset({'ci', 'spin'})

# Per-center pair field (coupling): each center is a list of state
# pairs. Classic string form uses ';' to split centers, ',' to split
# pairs and 1-based state indices ('1 2, 3 4; 1 2'); a typed value is
# the normalized 0-based entry and is passed through unchanged.
_CENTER_PAIR_FIELDS = frozenset({'coupling'})

# Per-center scalar field (root): one initial state per center,
# 1-based in every external spelling (classic string '1; 3', typed
# int/list alike) and decremented once to the internal 0-based value
# by _parse_center_scalars(start=INDEX_START['root']).
_CENTER_SCALAR_FIELDS = frozenset({'root'})

# Per-center scalar electronic fields (charges/multiplicity): one
# physical scalar per center, parsed exactly like root (';' splits
# centers, a typed int/list passes through, never an index so never
# decremented). A scalar broadcasts to every center downstream
# (data.center.as_center_scalars); charge may be negative.
_CENTER_SCALAR_LIST_FIELDS = frozenset({'charges', 'multiplicity'})

# Explicit classic-input token for a center with no coupling pairs.
# An empty semicolon group is an error; 'none' is the only way to ask
# for "this center has no pairs" in classic input (docs 4.2).
_CENTER_NONE_TOKEN = 'none'


def _parse_center_indices(raw_value, field_name, start=1, raw=True):
    """Parse highlevel/midlevel/highref/midref center groups.

    Supports three input formats:
    - Semicolon-separated string: '1-3 5; 11-14'
      -> [[0,1,2,4], [10,11,12,13]]  (0-based)
    - Flat list of ints: [1,2,3] or simple string '1 2 3'
      -> [[0,1,2]]  (single center, backward compatible)
    - Nested list: [[1,2],[3,4]]
      -> [[0,1],[2,3]]  (multi-center)

    Empty groups (consecutive semicolons, leading/trailing semicolons)
    raise ValueError.

    A center is a SET of atoms, so its indices come back ascending whatever
    order the input listed them in. The string form gets that from
    ``ReadIndex(sort=True)``; the typed branches sort here, because the point
    of the rule is that a yaml list, a classic keyword and a programmatic
    assignment all describe one center -- and everything downstream (the region
    geometry slice, the per-center charge/state lists indexed by CENTER rather
    than by atom, and the topology row numbers handed to a classical low layer,
    which it renumbers by position) then reads the same order. Duplicates and
    out-of-range indices are not silently repaired here; they are refused by
    ``schema.validate.validate_centers``.

    The ``raw`` flag separates the two entry points required by
    dev/MULTICENTER_DEVELOPMENT.md section 4.3: raw classic/YAML/JSON
    input carries 1-based indices and is decremented once (raw=True);
    an already-normalized programmatic entry passes 0-based typed
    values with raw=False so they are never decremented a second time.
    String input is always treated as raw (1-based) regardless of the
    flag, because a string can only come from raw input.

    Parameters
    ----------
    raw_value : str or list
        Raw input value from the configuration.
    field_name : str
        Field name for error messages.
    start : int
        1-based offset for input indices (default 1).
    raw : bool
        True (default) decrements typed lists by ``start``; False keeps
        typed lists as already-normalized 0-based values.

    Returns
    -------
    list of list of int
        Nested list of 0-based indices per center.
    """
    # Already typed (list) input from Pydantic or programmatic API
    if isinstance(raw_value, list):
        if not raw_value:
            return [[]]
        offset = start if raw else 0
        # Nested list: [[1,2],[3,4]]
        if isinstance(raw_value[0], list):
            return [sorted(int(x) - offset for x in group)
                    for group in raw_value]
        # Flat list: [1,2,3] -> single center
        return [sorted(int(x) - offset for x in raw_value)]

    # String input
    if not isinstance(raw_value, str):
        raw_value = str(raw_value)
    raw_value = raw_value.strip()
    if not raw_value:
        return [[]]

    # Check for semicolons
    if ';' not in raw_value:
        # No semicolons: single center, use ReadIndex('s')
        tokens = raw_value.split()
        converter = ReadIndex('s', start=start, sort=True)
        return [converter(tokens)]

    # Semicolon-separated groups
    groups = raw_value.split(';')
    centers = []
    for group_str in groups:
        group_str = group_str.strip()
        if not group_str:
            raise ValueError(
                "Empty center group in '%s' (consecutive or "
                "leading/trailing semicolons are not allowed)."
                % field_name)
        tokens = group_str.split()
        converter = ReadIndex('s', start=start, sort=True)
        indices = converter(tokens)
        if not indices:
            raise ValueError(
                "Empty center group in '%s'." % field_name)
        centers.append(indices)
    return centers


# Full-width semicolon (U+FF1B) look-alike that users may type by
# mistake; it never splits centers and gets an explicit replacement
# hint instead of being silently folded into a token (docs 4.2).
_FULLWIDTH_SEMICOLON = '\uff1b'


def _reject_fullwidth_semicolon(raw_value, field_name):
    """Raise with a replacement hint when a full-width ';' is present."""
    if isinstance(raw_value, str) and _FULLWIDTH_SEMICOLON in raw_value:
        raise ValueError(
            "Full-width semicolon '\uff1b' in '%s' is not a center "
            "separator; replace it with the ASCII ';'." % field_name)


def _empty_center_group_error(field_name):
    return ValueError(
        "Empty center group in '%s' (consecutive or leading/trailing "
        "semicolons are not allowed)." % field_name)


# The one workflow keyword whose value is a mapping rather than a scalar:
# each entry names a hyperparameter of the skill being scanned and carries
# the candidate values the grid varies it over.
SEARCH_GRID_FIELD = 'grid'


def _grid_number(token):
    """Read one candidate, keeping integers integral.

    ``true``/``false`` are read as booleans because that is how the
    classic text spells a boolean everywhere else
    (``_format_section_value``), and a candidate for a boolean keyword has
    to arrive as one: the skill's own check refuses the string 'true'
    where it expects true.
    """
    if isinstance(token, str):
        lowered = token.strip().lower()
        if lowered == 'true':
            return True
        if lowered == 'false':
            return False
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        # Not a number: keep the text. Whether the keyword accepts a
        # string is the skill's question (its declared type and its
        # hyperparameter vocabulary), and `query.check_searchable` answers
        # it when the grid is planned. Deciding it here would give the
        # framework a second copy of a type table it does not own.
        return str(token).strip()


def _reject_grid_fullwidth_semicolon(raw_value, field_name):
    """Refuse a full-width ';' in a grid line with the separator it belongs to.

    The shared helper names a center separator; a grid's ';' separates the
    entries of the scan, so the hint has to say which one the reader meant.
    """
    if _FULLWIDTH_SEMICOLON in raw_value:
        raise ValueError(
            "Full-width semicolon '\uff1b' in %s is not a group separator; "
            "replace it with the ASCII ';'." % field_name)


def _parse_search_grid(raw_value, field_name=SEARCH_GRID_FIELD):
    """Read &search grid into the mapping of candidate lists a scan plans.

    Three spellings describe one grid:

        yaml / json      grid: {depth: [1, 2], nn_size: [8, 16]}
        classic text     grid depth 1 2; nn_size 8 16
        programmatic     the same mapping

    The classic carrier puts the whole grid on **one** keyword line and
    separates the entries with ';', because the section reader keeps only
    the last value of a repeated key: two `grid` lines would quietly drop
    one keyword, and a scan planned from the surviving half is not the scan
    that was asked for. ';' is the same group separator the multi-center
    indices use, so the rule is not new to a reader.

    A name written twice is refused rather than resolved by order, and an
    entry without candidates is refused rather than read as an empty list
    (an empty list expands the grid to no points at all). A single candidate
    may be written as a bare value; it becomes the one-entry list the plan
    expects.

    Parameters
    ----------
    raw_value : dict or str
        The value as it arrived from any of the three carriers.
    field_name : str
        Field name for the messages.

    Returns
    -------
    dict
        Skill keyword to its list of candidate values.

    Raises
    ------
    ValueError
        For a carrier this parser cannot read, an entry without candidates
        or without a keyword, or a name claimed twice. A candidate whose
        *type* is wrong for its keyword is not refused here: the skill
        that declared the keyword judges that when the grid is planned.
    """
    grid = {}

    def _add(keyword, candidates):
        keyword = str(keyword).strip()
        if not keyword:
            raise ValueError(
                "%s carries a candidate list without a keyword: write "
                "'%s <keyword> <values...>'" % (field_name, field_name))
        if keyword in grid:
            raise ValueError(
                "%s names '%s' twice; one keyword is one axis of the grid, "
                "so write its candidates once" % (field_name, keyword))
        if isinstance(candidates, (list, tuple)):
            values = list(candidates)
        else:
            values = [candidates]
        if not values:
            raise ValueError(
                "%s entry '%s' carries no candidate value; an empty list "
                "would expand the grid to no points at all"
                % (field_name, keyword))
        grid[keyword] = values

    if isinstance(raw_value, dict):
        for keyword, candidates in raw_value.items():
            _add(keyword, [_grid_number(str(item)) if isinstance(item, str)
                           else item for item in candidates]
                 if isinstance(candidates, (list, tuple))
                 else (_grid_number(str(candidates))
                       if isinstance(candidates, str) else candidates))
        return grid

    if isinstance(raw_value, str):
        _reject_grid_fullwidth_semicolon(raw_value, field_name)
        groups = [raw_value] if raw_value.strip() else []
    elif isinstance(raw_value, list):
        # One keyword per line: the section reader keeps every value a
        # repeated key carried, and each is its own group here.
        groups = []
        for item in raw_value:
            if not isinstance(item, str):
                raise ValueError(
                    "%s cannot read the entry %r; write one keyword with its "
                    "candidates per line or per ';'-separated group"
                    % (field_name, item))
            _reject_grid_fullwidth_semicolon(item, field_name)
            if item.strip():
                groups.append(item.strip())
    else:
        raise ValueError(
            "%s must be a mapping of keyword to candidate list (yaml/json) or "
            "a ';'-separated group per keyword (classic input); got %r"
            % (field_name, raw_value))

    text = '; '.join(groups)
    if not text.strip():
        return {}
    for group in text.split(';'):
        group = group.strip()
        if not group:
            raise ValueError(
                "Empty %s group in '%s' (consecutive or leading/trailing "
                "semicolons are not allowed)." % (field_name, text))
        tokens = group.split()
        _add(tokens[0], [_grid_number(token) for token in tokens[1:]])
    return grid


def _parse_center_int_lists(raw_value, field_name):
    """Parse ci/spin per-center integer lists.

    Classic string: '2 2; 2 2' -> [[2,2],[2,2]]; '2 2' -> [2,2] (flat,
    single center, backward compatible). Values are state counts and
    spin quantum numbers, never indices, so no 1-based decrement is
    applied. Typed input is passed through unchanged (normalized
    entry).

    Parameters
    ----------
    raw_value : str or list
        Raw classic string or already-typed value.
    field_name : str
        Field name for error messages.

    Returns
    -------
    list
        Flat list (single center) or nested list (per center), or the
        typed value unchanged.
    """
    if not isinstance(raw_value, str):
        return raw_value
    _reject_fullwidth_semicolon(raw_value, field_name)
    text = raw_value.strip()
    if not text:
        return []
    if ';' not in text:
        return [int(t) for t in text.split()]
    centers = []
    for group in text.split(';'):
        group = group.strip()
        if not group:
            raise _empty_center_group_error(field_name)
        centers.append([int(t) for t in group.split()])
    return centers


def _parse_center_pairs(raw_value, field_name, start=1):
    """Parse coupling per-center state pairs.

    Classic string: '1 2, 3 4; 1 2' -> [[[0,1],[2,3]], [[0,1]]] with
    1-based state indices decremented once; the explicit 'none' token
    marks a center with no pairs. A single-center string ('1 2, 3 4')
    returns the flat pair list [[0,1],[2,3]] for backward
    compatibility. Typed input is the normalized 0-based entry and is
    passed through unchanged (docs 4.3: legacy typed coupling arrays
    keep their normalized base instead of being re-decremented).

    Parameters
    ----------
    raw_value : str or list
        Raw classic string or already-typed value.
    field_name : str
        Field name for error messages.
    start : int
        1-based offset for classic string state indices (default 1).

    Returns
    -------
    list
        Flat pair list (single center), nested per-center pair lists,
        or the typed value unchanged.
    """
    if not isinstance(raw_value, str):
        return raw_value
    _reject_fullwidth_semicolon(raw_value, field_name)
    text = raw_value.strip()
    if not text:
        return []
    if ';' not in text:
        converter = ReadIndex('g', start=start, sort=False)
        return converter(text.split())
    centers = []
    for group in text.split(';'):
        group = group.strip()
        if not group:
            raise _empty_center_group_error(field_name)
        if group.lower() == _CENTER_NONE_TOKEN:
            centers.append([])
            continue
        converter = ReadIndex('g', start=start, sort=False)
        centers.append(converter(group.split()))
    return centers


def _parse_center_scalars(raw_value, field_name, start=0):
    """Parse per-center scalar values (root / charges / multiplicity).

    Classic string: ';' splits centers ('1; 3' -> [1, 3] after the
    start shift); a single group returns a scalar for backward
    compatibility ('2' -> 2 - start). Typed values (int or list) are
    the same external spelling and shift identically: root is
    1-based externally at EVERY entry point (yaml, classic text,
    agent config and programmatic generators) and 0-based internally,
    matching the coupling/highlevel convention. charges/multiplicity
    are physical scalars, never indices, and use start=0. An empty
    classic string returns the internal default 0 (never -start).

    Parameters
    ----------
    raw_value : str, int or list
        Raw classic string or typed external value.
    field_name : str
        Field name for error messages.
    start : int
        External index base subtracted from every value (1 for root,
        0 for physical scalars).

    Returns
    -------
    int or list
        Scalar (single center) or per-center list of internal values.
    """
    if isinstance(raw_value, str):
        _reject_fullwidth_semicolon(raw_value, field_name)
        text = raw_value.strip()
        if not text:
            return 0
        if ';' not in text:
            return int(text) - start
        values = []
        for group in text.split(';'):
            group = group.strip()
            if not group:
                raise _empty_center_group_error(field_name)
            values.append(int(group) - start)
        return values
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, int):
        return raw_value - start
    if isinstance(raw_value, (list, tuple)):
        return [int(v) - start for v in raw_value]
    return raw_value


# ---------------------------------------------------------------------
# The classic `&constraint` section
# ---------------------------------------------------------------------
# The interface layer splits the file into `&section` headers and
# `keyword value` lines and types nothing; `&molecule` is interpreted
# here, and so is `&constraint`. The constraint section is the same flat
# shape with one addition: a value is a list of ';' separated segments,
# and a segment is either one object (one bond, one wall, one history
# potential) or the value, for one object, of a parameter. ',' groups the
# fields of a segment and spaces separate the tokens inside a field, as
# in the center-group parsers above.
#
# What the reader deliberately does not do is give the section a meaning:
# units, index base, geometry, capability and the plan itself stay with
# the constraint schema, which is why the output is the very mapping a
# YAML or JSON `constraints` block produces. Both spellings then reach
# the same validators with the same values, and the text form adds no
# capability the structured form lacks.
#
# Three rules are what makes a long declaration readable, and each leaves
# a visible trace in the run record:
#
# - A parameter line may carry fewer segments than the family has objects.
#   One segment applies to every object; a shorter list pads with its last
#   value. Either way the fill-in is written to `notes`, because a number
#   that reaches an object the input never named must be readable there.
# - More segments than objects is a refusal: the objects were miscounted
#   somewhere, and dropping the extras would drop a declaration.
# - `wall_type` names which wall geometries are read at all. A geometry the
#   keyword does not name contributes no object and its parameter lines are
#   reported in `notes` instead of being dropped in silence.
#
# Atom indices stay 1-based here: the project decrements them once, in the
# schema, and a range like '1-12' is expanded by ReadIndex.

# Section names the classic input may use. `system_constraint` is the
# retired spelling of the same block: it keeps working so an archived
# input still runs, while every new input and example writes `constraint`.
CONSTRAINT_SECTION_NAMES = ('constraint', 'system_constraint')

# `wall_type` selects which wall geometries are read. Absent or `none`
# reads none of them, which is the default-closed statement of the block.
CONSTRAINT_WALL_TYPES = ('sphere', 'ellipsoid', 'box', 'triclinic')
CONSTRAINT_WALL_NONE = 'none'
CONSTRAINT_WALL_OBJECTS = {
    'sphere': 'wall_sphere_radius',
    'ellipsoid': 'wall_ellipsoid_semiaxes',
    'box': 'wall_box_lengths',
    'triclinic': 'wall_triclinic_cell',
}

CONSTRAINT_UNIT_KEYS = {'unit_length': 'length', 'unit_angle': 'angle',
                        'unit_energy': 'energy'}

# One parameter ramp is spelled as a family of five keys around the key it
# ramps: `bias_bond_fc_start`, `_end`, `_scale`, `_delay`, `_stop`.
# `start` and `end` are exact values and fall back, per object, to that
# object's own base value, so an omitted ramp means no change rather than
# another component's ramp. `delay` defaults to the first step and `stop`
# is left for the schema to resolve from the length of the run, where the
# whole input is in hand.
CONSTRAINT_RAMP_SUFFIXES = ('_start', '_end', '_scale', '_delay', '_stop')
CONSTRAINT_RAMP_DELAY_DEFAULT = 1

# keyword -> (section field, segment kind, extra record fields) for the
# lines that declare objects. One ';' segment is one object, so
# `bias_bond 1 2; 3 4` declares two biases and each becomes one record.
CONSTRAINT_OBJECT_KEYS = {
    'group_name': ('groups', 'name', 'token', ()),
    'cv_bond': ('cvs', 'atoms', 'index', (('kind', 'bond'),)),
    'cv_angle': ('cvs', 'atoms', 'index', (('kind', 'angle'),)),
    'cv_dihedral': ('cvs', 'atoms', 'index', (('kind', 'dihedral'),)),
    'shake_bond': ('constraints', 'bonds', 'row2', (('kind', 'bond'),)),
    'shake_bond_type': ('constraints', 'bond_type', 'token',
                        (('kind', 'bond'),)),
    'shake_angle': ('constraints', 'bonds', 'row3', (('kind', 'angle'),)),
    'shake_dihedral': ('constraints', 'bonds', 'row4',
                       (('kind', 'dihedral'),)),
    'bias_bond': ('biases', 'atoms', 'index', ()),
    'bias_angle': ('biases', 'atoms', 'index', ()),
    'bias_dihedral': ('biases', 'atoms', 'index', ()),
    'bias_cv': ('biases', 'cv', 'token', ()),
    'wall_sphere_radius': ('walls', 'radius', 'number',
                           (('geometry', 'sphere'),)),
    'wall_ellipsoid_semiaxes': ('walls', 'semiaxes', 'vector3',
                                (('geometry', 'ellipsoid'),)),
    'wall_box_lengths': ('walls', 'lengths', 'vector3',
                         (('geometry', 'box'),)),
    'wall_triclinic_cell': ('walls', 'cell', 'cell',
                            (('geometry', 'triclinic'),)),
    'fix_atoms': ('fixes', 'atoms', 'index', ()),
    'fix_element': ('fixes', 'element', 'tokens', ()),
    'fix_group': ('fixes', 'group', 'token', ()),
    'meta_cv': ('metadynamics', 'cvs', 'tokens', ()),
    'meta_rmsd': ('metadynamics', 'atoms', 'index', (('kind', 'rmsd'),)),
}

# keyword -> (section field, record field, segment kind, object group) for
# the lines that give one value per object. A group of None aligns over
# every object of the family, in the object order below; a named group
# aligns over that object line's objects only, so `bias_angle_ref` can
# never land on a bond.
CONSTRAINT_PARAM_KEYS = {
    'group_atoms': ('groups', 'atoms', 'index', ('group_name',)),
    'group_element': ('groups', 'element', 'tokens', ('group_name',)),
    'group_group': ('groups', 'group', 'token', ('group_name',)),
    'cv_name': ('cvs', 'name', 'token', None),
    'bias_bond_ref': ('biases', 'target', 'number', ('bias_bond',)),
    'bias_angle_ref': ('biases', 'target', 'number', ('bias_angle',)),
    'bias_dihedral_ref': ('biases', 'target', 'number', ('bias_dihedral',)),
    'bias_cv_ref': ('biases', 'target', 'number', ('bias_cv',)),
    'bias_bond_fc': ('biases', 'force_constant', 'number', ('bias_bond',)),
    'bias_angle_fc': ('biases', 'force_constant', 'number', ('bias_angle',)),
    'bias_dihedral_fc': ('biases', 'force_constant', 'number',
                         ('bias_dihedral',)),
    'bias_cv_fc': ('biases', 'force_constant', 'number', ('bias_cv',)),
    'bias_bond_region': ('biases', 'region', 'token', ('bias_bond',)),
    'bias_angle_region': ('biases', 'region', 'token', ('bias_angle',)),
    'bias_dihedral_region': ('biases', 'region', 'token',
                             ('bias_dihedral',)),
    'bias_cv_region': ('biases', 'region', 'token', ('bias_cv',)),
    'bias_id': ('biases', 'id', 'token', None),
    'shake_bond_ref': ('constraints', 'target', 'number',
                       ('shake_bond', 'shake_bond_type')),
    'shake_angle_ref': ('constraints', 'target', 'number', ('shake_angle',)),
    'shake_dihedral_ref': ('constraints', 'target', 'number',
                           ('shake_dihedral',)),
    'shake_tol': ('constraints', 'tolerance', 'number', None),
    'shake_iter': ('constraints', 'max_iterations', 'step', None),
    'shake_vel': ('constraints', 'velocity_constraint', 'token', None),
    'shake_id': ('constraints', 'id', 'token', None),
    'wall_atoms': ('walls', 'atoms', 'index', None),
    'wall_element': ('walls', 'element', 'tokens', None),
    'wall_group': ('walls', 'group', 'token', None),
    'wall_center': ('walls', 'center', 'vector3', None),
    # The anchor as atoms rather than coordinates: one segment names the atoms
    # whose center fixes that wall's center at the reference geometry.
    'wall_ref': ('walls', 'ref_atoms', 'index', None),
    # One segment is one wall's molecule partition: atom numbers whose last
    # number says how many of them a molecule holds. `1-3 7-9 3` is therefore
    # the two molecules [1,2,3] and [7,8,9].
    'wall_groups': ('walls', 'molecules', 'groups', None),
    'wall_law': ('walls', 'law', 'token', None),
    'wall_power': ('walls', 'power', 'step', None),
    'wall_fc': ('walls', 'force_constant', 'number', None),
    'wall_target': ('walls', 'target', 'token', None),
    'wall_id': ('walls', 'id', 'token', None),
    'fix_dir': ('fixes', 'direction', 'token', None),
    'fix_id': ('fixes', 'id', 'token', None),
    'meta_height': ('metadynamics', 'height', 'number', None),
    'meta_width': ('metadynamics', 'width', 'numbers', None),
    'meta_pace': ('metadynamics', 'pace', 'step', None),
    'meta_gamma': ('metadynamics', 'bias_factor', 'number', None),
    'meta_temp': ('metadynamics', 'temperature', 'number', None),
    'meta_weight': ('metadynamics', 'weight', 'token', ('meta_rmsd',)),
    'meta_rmsd_ref': ('metadynamics', 'reference', 'reference',
                      ('meta_rmsd',)),
    'meta_id': ('metadynamics', 'id', 'token', None),
}

# The rampable base keys: (record parameter, endpoint kind, object group).
# Anything that is not listed here refuses the `_start` family, because a
# parameter with no implementation must not be smoothed over by a ramp
# that appears to work. A strict constraint carries no potential, so it
# has nothing to ramp.
CONSTRAINT_RAMP_KEYS = {
    'bias_bond_fc': ('biases', 'force_constant', 'number', ('bias_bond',)),
    'bias_angle_fc': ('biases', 'force_constant', 'number', ('bias_angle',)),
    'bias_dihedral_fc': ('biases', 'force_constant', 'number',
                         ('bias_dihedral',)),
    'bias_cv_fc': ('biases', 'force_constant', 'number', ('bias_cv',)),
    'wall_fc': ('walls', 'force_constant', 'number', None),
    'wall_sphere_radius': ('walls', 'radius', 'number',
                           ('wall_sphere_radius',)),
    'wall_ellipsoid_semiaxes': ('walls', 'semiaxes', 'vector3',
                                ('wall_ellipsoid_semiaxes',)),
    'wall_box_lengths': ('walls', 'lengths', 'vector3',
                         ('wall_box_lengths',)),
    'wall_triclinic_cell': ('walls', 'cell', 'cell',
                            ('wall_triclinic_cell',)),
    'meta_height': ('metadynamics', 'height', 'number', None),
}

# A geometry ramp may be written per component instead of as a whole
# vector: `wall_x_start` moves one semiaxis, one box edge or one cell
# length and leaves the others at the object's own base value. The
# positions are those of the geometry vector, so x/y/z are the three
# lengths of a cell and alpha/beta/gamma its three angles.
CONSTRAINT_COMPONENT_KEYS = {'wall_x': 0, 'wall_y': 1, 'wall_z': 2,
                             'wall_alpha': 3, 'wall_beta': 4,
                             'wall_gamma': 5}
CONSTRAINT_COMPONENT_FIELDS = {'semiaxes': 3, 'lengths': 3, 'cell': 6}

# Keywords that are refused by name. Each one is a spelling the subsystem
# has actually carried at some point, so the answer states what changed
# rather than calling it unknown.
CONSTRAINT_REFUSED_KEYS = {
    'shake_fc': ('a strict constraint carries no potential, so there is no '
                 'force constant to set; write bias_bond_fc'),
    'shake_bond_fc': ('a strict constraint carries no potential; write '
                      'bias_bond_fc'),
    'shake_angle_fc': ('a strict constraint carries no potential; write '
                       'bias_angle_fc'),
    'shake_dihedral_fc': ('a strict constraint carries no potential; write '
                          'bias_dihedral_fc'),
    'bias_ref': ('a parameter name carries the type of its variable; write '
                 'bias_bond_ref, bias_angle_ref, bias_dihedral_ref or '
                 'bias_cv_ref'),
    'bias_fc': ('a parameter name carries the type of its variable; write '
                'bias_bond_fc, bias_angle_fc, bias_dihedral_fc or '
                'bias_cv_fc'),
    'bias_region': ('a parameter name carries the type of its variable; '
                    'write bias_bond_region, bias_angle_region, '
                    'bias_dihedral_region or bias_cv_region'),
    'fix_ref': ('a frozen atom returns to the geometry the run starts from, so '
                'there is no reference to choose; fix_dir says which components '
                'freeze'),
    # The structured spelling of the block: records keyed by the schema's own
    # field names. That mapping is what the reader produces and what normalize
    # consumes; it was never a second way to write an input file, because a
    # section read two ways says nothing about which reading the run applies.
    # Each refusal names the flat line of the same family.
    'version': ('no version keyword is read; the grammar is the one the reader '
                'of this build implements'),
    'units': ('write unit_length, unit_angle or unit_energy'),
    'notes': ('notes is what the reader reports back to the log, not a '
              'declaration; the input has no notes section'),
    'schedules': ('a ramp is written on the parameter it smooths, as '
                  'bias_bond_fc_start, bias_bond_fc_end, bias_bond_fc_scale, '
                  'bias_bond_fc_delay, bias_bond_fc_stop (the same five '
                  'suffixes on any rampable key); there is no schedule '
                  'section to name'),
    'groups': ('write group_name for the groups and group_atoms, '
               'group_element or group_group for what is in them'),
    'cvs': ('write cv_bond, cv_angle or cv_dihedral for the variables and '
            'cv_name for their names'),
    'constraints': ('write shake_bond, shake_bond_type, shake_angle or '
                    'shake_dihedral for the rows and shake_*_ref, shake_tol, '
                    'shake_iter, shake_vel for their settings'),
    'biases': ('write bias_bond, bias_angle, bias_dihedral or bias_cv for the '
               'biases and bias_*_ref, bias_*_fc, bias_*_region for their '
               'parameters'),
    'walls': ('write wall_type to say which geometries are read, then '
              'wall_sphere_radius, wall_ellipsoid_semiaxes, wall_box_lengths '
              'or wall_triclinic_cell for the walls and wall_* for their '
              'parameters'),
    'fixes': ('write fix_atoms, fix_element or fix_group for the frozen atoms '
              'and fix_dir for the components'),
    'metadynamics': ('write meta_cv or meta_rmsd for the history potentials '
                     'and meta_height, meta_width, meta_pace, meta_gamma, '
                     'meta_temp for their parameters'),
}
for _legacy in ('cbond', 'cangle', 'cdihedral', 'tbond', 'tangle',
               'tdihedral', 'fbond', 'fangle', 'fdihedral'):
    CONSTRAINT_REFUSED_KEYS[_legacy] = (
        "the legacy `&molecule` spelling; it is still read there by the "
        'legacy adapter, but `&constraint` writes shake_bond, '
        'shake_angle and shake_dihedral instead')
del _legacy


def _constraint_keyword_order():
    """The object lines of every family, in the order objects concatenate.

    A parameter line is matched against a count, so the count has to have
    one order. It is this table, not the order the keywords happen to
    appear in the file, because a reader of the input must be able to
    count the objects without re-reading the file.
    """
    return (
        ('groups', ('group_name',)),
        ('cvs', ('cv_bond', 'cv_angle', 'cv_dihedral')),
        ('constraints', ('shake_bond', 'shake_bond_type', 'shake_angle',
                         'shake_dihedral')),
        ('biases', ('bias_bond', 'bias_angle', 'bias_dihedral', 'bias_cv')),
        ('walls', ('wall_sphere_radius', 'wall_ellipsoid_semiaxes',
                   'wall_box_lengths', 'wall_triclinic_cell')),
        ('fixes', ('fix_atoms', 'fix_element', 'fix_group')),
        ('metadynamics', ('meta_cv', 'meta_rmsd')),
    )


CONSTRAINT_FAMILIES = _constraint_keyword_order()


def _constraint_token_text(token):
    """One token as text: a typed YAML or JSON entry reads like a string."""
    return token if isinstance(token, str) else str(token)


def _constraint_segment_tokens(item):
    """The tokens of one segment: a scalar is a token, a list is a row."""
    if isinstance(item, (list, tuple)):
        return [_constraint_token_text(word) for word in item]
    return [_constraint_token_text(item)]


def _constraint_segments(raw, keyword):
    """Split one value into segments, each a list of tokens.

    A segment is one object or one per-object value. A doubled, leading or
    trailing ';' is a refusal rather than an empty object, and the
    full-width look-alike is refused with the replacement named, exactly as
    the center-group parsers do.

    The rule is the same for every keyword, which is the point of offering
    YAML and JSON at all: in text the ';' carries the object boundary, and here
    the list does. One element is one object, and that object's tokens are
    either a scalar or their own list - `wall_atoms: [[1, 2, 3]]` is one wall
    on three atoms, `[[1, 2], [3, 4]]` two walls, exactly as
    `wall_atoms 1 2 3` and `wall_atoms 1 2 3; 4 5` read in the classic input.
    """
    if isinstance(raw, str):
        _reject_fullwidth_semicolon(raw, keyword)
        text = raw.strip()
        if not text:
            raise ValueError("'%s' needs at least one segment" % keyword)
        segments = []
        for piece in text.split(';'):
            piece = piece.strip()
            if not piece:
                raise ValueError(
                    "'%s' has an empty segment: ';' separates the objects "
                    "of one line, so it cannot be doubled or trailing"
                    % keyword)
            segments.append(piece.replace(',', ' ').split())
        return segments
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise ValueError("'%s' needs at least one segment" % keyword)
        return [_constraint_segment_tokens(item) for item in raw]
    return [[_constraint_token_text(raw)]]


def _constraint_floats(tokens, keyword, expected=None):
    """Numbers for one segment, with the count the keyword owes."""
    try:
        numbers = [float(word) for word in tokens]
    except ValueError:
        raise ValueError("'%s' expects numbers, got '%s'"
                         % (keyword, ' '.join(tokens)))
    if expected is not None and len(numbers) != expected:
        raise ValueError("'%s' needs %d numbers per segment, got %d ('%s')"
                         % (keyword, expected, len(numbers),
                            ' '.join(tokens)))
    if not numbers:
        raise ValueError("'%s' needs at least one number" % keyword)
    return numbers


def _constraint_indices(tokens, keyword):
    """1-based atom indices of one segment, ranges expanded, order kept.

    The decrement happens once, in the schema, so `start=0` here is not an
    off-by-one: it is the statement that the reader keeps the input's base.
    """
    try:
        indices = ReadIndex('s', start=0, sort=False)(tokens)
    except (ValueError, IndexError, TypeError):
        raise ValueError("'%s' expects 1-based atom indices or a range like "
                         "'5-8', got '%s'" % (keyword, ' '.join(tokens)))
    return indices


def _constraint_molecules(tokens, keyword):
    """One molecule-partition segment: atom numbers, then the molecule size.

    The trailing number is what splits the list into molecules, so a segment
    that does not divide by it had either the count or the atoms miswritten;
    guessing which would move atoms between molecules without saying so. The
    groups come back 1-based like every other atom number here, since the schema
    is where the one decrement happens.
    """
    numbers = _constraint_indices(tokens, keyword)
    if len(numbers) < 2:
        raise ValueError(
            "'%s' needs atom numbers and a trailing molecule size, got '%s'"
            % (keyword, ' '.join(tokens)))
    per = int(numbers[-1])
    atoms = numbers[:-1]
    if per < 1:
        raise ValueError("'%s' gives a molecule size of %d; a molecule holds at "
                         'least one atom' % (keyword, per))
    if len(atoms) % per:
        raise ValueError(
            "'%s' lists %d atoms with %d per molecule, which does not divide "
            'out; a partial molecule would leave %d atom(s) unconfined'
            % (keyword, len(atoms), per, len(atoms) % per))
    return [atoms[start:start + per] for start in range(0, len(atoms), per)]


def _constraint_read(kind, segment, keyword):
    """One segment as the value its keyword declares."""
    tokens = [_constraint_token_text(word) for word in segment]
    if kind == 'token':
        if len(tokens) != 1:
            raise ValueError("'%s' takes one word per segment, got '%s'"
                             % (keyword, ' '.join(tokens)))
        return tokens[0]
    if kind == 'tokens':
        return list(tokens)
    if kind == 'index':
        return _constraint_indices(tokens, keyword)
    if kind in ('row2', 'row3', 'row4'):
        arity = int(kind[-1])
        indices = _constraint_indices(tokens, keyword)
        if len(indices) % arity:
            raise ValueError(
                "'%s' states whole rows: %d atom numbers is not a multiple of "
                "the %d an object of this kind has ('%s')"
                % (keyword, len(indices), arity, ' '.join(tokens)))
        return [list(indices[start:start + arity])
                for start in range(0, len(indices), arity)]
    if kind == 'number':
        return _constraint_floats(tokens, keyword, 1)[0]
    if kind == 'numbers':
        return _constraint_floats(tokens, keyword)
    if kind == 'vector3':
        return _constraint_floats(tokens, keyword, 3)
    if kind == 'cell':
        numbers = _constraint_floats(tokens, keyword)
        if len(numbers) not in (6, 9):
            raise ValueError(
                "'%s' takes six cell parameters (a b c alpha beta gamma) or "
                'nine matrix components, got %d' % (keyword, len(numbers)))
        return numbers
    if kind == 'scale':
        return _constraint_floats(tokens, keyword, 2)
    if kind == 'groups':
        return _constraint_molecules(tokens, keyword)
    if kind == 'step':
        if len(tokens) != 1:
            raise ValueError("'%s' takes one integer per segment, got '%s'"
                             % (keyword, ' '.join(tokens)))
        try:
            return int(tokens[0])
        except ValueError:
            raise ValueError("'%s' needs an integer step count, got '%s'"
                             % (keyword, tokens[0]))
    if kind == 'reference':
        if len(tokens) == 1 and tokens[0].lower() == 'initial':
            return 'initial'
        return _constraint_floats(tokens, keyword)
    raise ValueError("'%s' asks for the unknown segment kind '%s'"
                     % (keyword, kind))


def _constraint_align(segments, count, keyword, notes):
    """Segments to one per object: broadcast, pad, or refuse overlong."""
    if not segments:
        return [None] * count
    if len(segments) > count:
        raise ValueError(
            "'%s' gives %d segments for %d objects; every object needs one "
            'value and dropping the extras would drop a declaration'
            % (keyword, len(segments), count))
    if count and len(segments) == 1 and count > 1:
        notes.append("'%s' broadcast over %d objects" % (keyword, count))
        return list(segments) * count
    if len(segments) < count:
        notes.append("'%s' padded with its last value to %d objects"
                     % (keyword, count))
        return list(segments) + [segments[-1]] * (count - len(segments))
    return list(segments)


def _constraint_group_positions(positions, group, total):
    """The object indices one parameter aligns against."""
    if group is None:
        return list(range(total))
    found = []
    for key in group:
        found.extend(positions.get(key, []))
    return found


def _constraint_wall_types(values):
    """The geometries `wall_type` activates, in the order it names them."""
    raw = values.get('wall_type')
    if raw is None:
        return []
    names = [_constraint_token_text(word).lower()
             for segment in _constraint_segments(raw, 'wall_type')
             for word in segment]
    if not names:
        raise ValueError("'wall_type' names no geometry; write %s or %s"
                         % (', '.join(CONSTRAINT_WALL_TYPES),
                            CONSTRAINT_WALL_NONE))
    for name in names:
        if name not in CONSTRAINT_WALL_TYPES and name != CONSTRAINT_WALL_NONE:
            raise ValueError(
                "'wall_type' does not know the geometry '%s'; expected %s "
                'or %s' % (name, ', '.join(CONSTRAINT_WALL_TYPES),
                           CONSTRAINT_WALL_NONE))
    if CONSTRAINT_WALL_NONE in names and len(names) > 1:
        raise ValueError(
            "'wall_type' cannot mix '%s' with %s: the switch is either off "
            'or it names the geometries to read'
            % (CONSTRAINT_WALL_NONE,
               ', '.join(sorted(set(names) - {CONSTRAINT_WALL_NONE}))))
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "'wall_type' names %s twice; one instance of a geometry is one "
            "';' segment of its own parameter line"
            % ', '.join(duplicates))
    if CONSTRAINT_WALL_NONE in names:
        return []
    return names


def _constraint_objects(section_key, object_keys, values, wall_types, notes):
    """Read one family's object lines into records and their positions."""
    order = list(object_keys)
    if section_key == 'walls':
        order = [CONSTRAINT_WALL_OBJECTS[name] for name in wall_types]
        for key in object_keys:
            if key in values and key not in order:
                notes.append(
                    "'%s' ignored: 'wall_type' does not name that geometry"
                    % key)
    else:
        # Objects concatenate in the order their lines appear in the file, and
        # the parameter lines count them the same way. The table above is the
        # list of object lines a family has, not an order they must be written
        # in: a file that starts with `fix_element` means its first `fix_dir`
        # to belong to that element selection.
        order = [key for key in values if key in object_keys]
    records, positions = [], {}
    for key in order:
        raw = values.get(key)
        if raw is None:
            continue
        _family, field, kind, extra = CONSTRAINT_OBJECT_KEYS[key]
        for segment in _constraint_segments(raw, key):
            record = dict(extra)
            record[field] = _constraint_read(kind, segment, key)
            positions.setdefault(key, []).append(len(records))
            records.append(record)
    return records, positions


def _constraint_params(section_key, records, positions, values, notes):
    """Give every object the parameter values its family declares."""
    for key, spec in CONSTRAINT_PARAM_KEYS.items():
        family, field, kind, group = spec
        if family != section_key or key not in values:
            continue
        segment_source = _constraint_segments(values[key], key)
        indices = _constraint_group_positions(positions, group, len(records))
        if not indices:
            if not records and section_key == 'walls':
                notes.append("'%s' ignored: 'wall_type' reads no wall" % key)
                continue
            raise ValueError(
                "'%s' has no objects to apply to; the object line of the "
                'same family comes first' % key)
        aligned = _constraint_align(segment_source, len(indices), key, notes)
        for position, segment in zip(indices, aligned):
            if segment is None:
                continue
            if key in CONSTRAINT_NULLABLE_KEYS and _constraint_is_none(segment):
                records[position][field] = []
                continue
            records[position][field] = _constraint_read(kind, segment, key)


# The wall keys whose "no value" is a decision rather than a gap: one wall can
# name its own center while the next takes the default, and a positional line has
# to be able to say so. `none` is the same word `wall_type` already uses to
# switch a wall off, and the writer emits it back for an object that declares
# nothing, so a mixed block survives the round trip instead of being refused.
CONSTRAINT_NULLABLE_KEYS = ('wall_center', 'wall_ref', 'wall_groups')


def _constraint_is_none(segment):
    """True for the one-word segment that means 'this object declares none'."""
    return len(segment) == 1 and str(segment[0]).strip().lower() == 'none'


def _constraint_ramp_endpoints(kind, segment, keyword, base):
    """One ramp endpoint as the value the base keyword would have read."""
    if segment is None:
        return None
    value = _constraint_read(kind, segment, keyword)
    if isinstance(base, list) and isinstance(value, list) \
            and len(value) != len(base):
        raise ValueError(
            "'%s' ramps %d numbers against a base of %d; an endpoint has the "
            'same shape as the value it ramps' % (keyword, len(value),
                                                 len(base)))
    return value


def _constraint_base_ramps(section_key, records, positions, values, notes):
    """Turn the `<key>_start/_end/_scale/_delay/_stop` families into ramps.

    One ramp is an inline schedule on the record it belongs to, which is the
    same ScheduleRecord a structured `schedules:` entry builds, so a ramp
    cannot mean two different things depending on which spelling wrote it.
    """
    for base, spec in CONSTRAINT_RAMP_KEYS.items():
        family, parameter, kind, group = spec
        if family != section_key:
            continue
        given = {suffix: values.get(base + suffix)
                 for suffix in CONSTRAINT_RAMP_SUFFIXES}
        if not any(item is not None for item in given.values()):
            continue
        indices = _constraint_group_positions(positions, group, len(records))
        if not indices:
            if not records and section_key == 'walls':
                notes.append(
                    "'%s' ignored: 'wall_type' reads no wall" % base)
                continue
            raise ValueError(
                "'%s_start' ramps no object; the object line of the same "
                'family comes first' % base)
        aligned = {suffix: _constraint_align(
            _constraint_segments(given[suffix], base + suffix)
            if given[suffix] is not None else [], len(indices),
            base + suffix, notes) for suffix in CONSTRAINT_RAMP_SUFFIXES}
        for position, where in zip(indices, range(len(indices))):
            record = records[position]
            start = _constraint_ramp_endpoints(
                kind, aligned['_start'][where], base + '_start',
                record.get(parameter))
            end = _constraint_ramp_endpoints(
                kind, aligned['_end'][where], base + '_end',
                record.get(parameter))
            scale = None
            if aligned['_scale'][where] is not None:
                scale = _constraint_read('scale', aligned['_scale'][where],
                                         base + '_scale')
            delay = aligned['_delay'][where]
            stop = aligned['_stop'][where]
            delay = (CONSTRAINT_RAMP_DELAY_DEFAULT if delay is None else
                     _constraint_read('step', delay, base + '_delay'))
            stop = (None if stop is None else
                    _constraint_read('step', stop, base + '_stop'))
            if scale is not None and (start is not None or end is not None):
                raise ValueError(
                    "'%s' ramps with _scale or with _start/_end, never both: "
                    'the two answers are a factor and a value' % base)
            if scale is None and start is None and end is None:
                raise ValueError(
                    "'%s_delay' and '%s_stop' bound a ramp that has no "
                    "endpoints: give %s_start, %s_end or %s_scale"
                    % (base, base, base, base, base))
            schedule = {'parameter': parameter, 'start_step': delay}
            if stop is not None:
                schedule['end_step'] = stop
            if scale is not None:
                schedule['scale'] = scale
            else:
                reference = record.get(parameter)
                if reference is None:
                    raise ValueError(
                        "'%s' ramps a value this object does not carry: give "
                        'the base %s as well, or both endpoints'
                        % (base, base.split('_')[-1]))
                schedule['value'] = [
                    reference if start is None else start,
                    reference if end is None else end]
            record.setdefault('schedules', []).append(schedule)


def _constraint_component_field(record):
    """The geometry field of one wall object that components can ramp.

    The name is the parameter a ramp writes and the second value is what the
    components are positions in. A nine-number cell is a matrix, whose rows are
    not x, y, z and alpha, so it reports no components at all and a component
    ramp against it is refused rather than pointed at a row nobody meant.
    """
    for name, width in CONSTRAINT_COMPONENT_FIELDS.items():
        value = record.get(name)
        if not isinstance(value, list):
            continue
        if name == 'cell' and len(value) != 6:
            return None, []
        if len(value) >= width:
            return name, value
    return None, []


def _constraint_component_ramps(records, values, notes):
    """Ramp one component of a wall geometry, leaving the others at base.

    A component ramp is one schedule on the whole vector, because two
    schedules on the same parameter would apply in an order nothing states.
    The components therefore share the delay and the stop of the object, and
    say so when they do not.
    """
    present = {}
    for stem in CONSTRAINT_COMPONENT_KEYS:
        given = {suffix: values.get(stem + suffix)
                 for suffix in CONSTRAINT_RAMP_SUFFIXES}
        if any(item is not None for item in given.values()):
            present[stem] = given
    if not present:
        return
    groups = {}
    for stem in sorted(present,
                       key=lambda item: CONSTRAINT_COMPONENT_KEYS[item]):
        given = present[stem]
        index = CONSTRAINT_COMPONENT_KEYS[stem]
        eligible = []
        for position, record in enumerate(records):
            name, base = _constraint_component_field(record)
            if name is None or index >= len(base):
                continue
            eligible.append((position, name, base))
        if not eligible:
            raise ValueError(
                "'%s' ramps no wall geometry with that component: a sphere "
                'has one radius, ramp it with wall_sphere_radius_start, and a '
                'nine-component cell has no angles to ramp, so write the six '
                'parameters instead' % stem)
        shapes = sorted({name for _position, name, _base in eligible})
        if len(shapes) > 1:
            raise ValueError(
                "'%s' would ramp %s at once, which are different vectors; "
                'write the ramp on the geometry line you mean '
                '(%s_start)' % (stem, ', '.join(shapes), shapes[0]))
        parameter = shapes[0]
        line = CONSTRAINT_GEOMETRY_LINES[parameter]
        for suffix in CONSTRAINT_RAMP_SUFFIXES:
            if given[suffix] is not None and line + suffix in values:
                raise ValueError(
                    "'%s' and '%s' ramp the same %s twice, once per "
                    'component and once as a vector; write it one way'
                    % (stem + suffix, line + suffix, parameter))
        entry = groups.setdefault(parameter, {'objects': eligible,
                                              'stems': {}})
        if entry['objects'] != eligible:
            raise ValueError(
                "'%s' and the other component keys of %s do not count the same "
                'walls; one vector ramp is one schedule per wall, so its '
                'components need the same objects' % (stem, parameter))
        entry['stems'][stem] = dict(
            (suffix, _constraint_align(
                _constraint_segments(given[suffix], stem + suffix)
                if given[suffix] is not None else [], len(eligible),
                stem + suffix, notes))
            for suffix in CONSTRAINT_RAMP_SUFFIXES)
    for parameter, entry in groups.items():
        for where, (position, _name, base) in enumerate(entry['objects']):
            schedule = _constraint_one_component_ramp(
                parameter, base, entry['stems'], where)
            if schedule is None:
                continue
            records[position].setdefault('schedules', []).append(schedule)


CONSTRAINT_GEOMETRY_LINES = {
    'semiaxes': 'wall_ellipsoid_semiaxes',
    'lengths': 'wall_box_lengths',
    'cell': 'wall_triclinic_cell',
}


def _constraint_one_component_ramp(parameter, base, stems, where):
    """The one schedule a set of component ramps composes for one object."""
    start = list(base)
    end = list(base)
    ramps = False
    delay = None
    stop = None
    for stem, index in sorted(CONSTRAINT_COMPONENT_KEYS.items(),
                              key=lambda item: item[1]):
        if index >= len(base):
            continue
        aligned = stems.get(stem)
        if aligned is None:
            continue
        segment = dict((suffix, aligned[suffix][where])
                       for suffix in CONSTRAINT_RAMP_SUFFIXES)
        if segment['_scale'] is not None:
            raise ValueError(
                "'%s_scale' is refused: a scale multiplies a value whose unit "
                "differs from the internal one, so ramp '%s_start' and "
                "'%s_end' with the numbers you mean"
                % (stem, stem, stem))
        if segment['_start'] is not None:
            ramps = True
            start[index] = _constraint_read('number', segment['_start'],
                                            stem + '_start')
        if segment['_end'] is not None:
            ramps = True
            end[index] = _constraint_read('number', segment['_end'],
                                          stem + '_end')
        if segment['_delay'] is not None:
            value = _constraint_read('step', segment['_delay'],
                                     stem + '_delay')
            if delay is not None and delay != value:
                raise ValueError(
                    "'%s_delay' and the ramp of the other components of one "
                    '%s disagree; the components of a vector ramp share '
                    'their schedule' % (stem, parameter))
            delay = value
        if segment['_stop'] is not None:
            value = _constraint_read('step', segment['_stop'],
                                     stem + '_stop')
            if stop is not None and stop != value:
                raise ValueError(
                    "'%s_stop' and the ramp of the other components of one "
                    '%s disagree; the components of a vector ramp share '
                    'their schedule' % (stem, parameter))
            stop = value
    if not ramps:
        if delay is not None or stop is not None:
            raise ValueError(
                "'%s' bounds a ramp that has no endpoints: give %s_start or "
                '%s_end' % (parameter, parameter, parameter))
        return None
    schedule = {'parameter': parameter, 'value': [start, end],
                'start_step': (CONSTRAINT_RAMP_DELAY_DEFAULT if delay is None
                               else delay)}
    if stop is not None:
        schedule['end_step'] = stop
    return schedule


def _check_constraint_keywords(values):
    """Refuse an unknown keyword, and name the family it belongs to."""
    known = set(CONSTRAINT_OBJECT_KEYS) | set(CONSTRAINT_PARAM_KEYS)
    known |= set(CONSTRAINT_UNIT_KEYS)
    known.add('wall_type')
    for base in CONSTRAINT_RAMP_KEYS:
        known.update(base + suffix for suffix in CONSTRAINT_RAMP_SUFFIXES)
    for stem in CONSTRAINT_COMPONENT_KEYS:
        known.update(stem + suffix for suffix in CONSTRAINT_RAMP_SUFFIXES)
    unknown = sorted(set(values) - known)
    if not unknown:
        return
    refused = [key for key in unknown if key in CONSTRAINT_REFUSED_KEYS]
    if refused:
        raise ValueError(
            ' '.join("'%s' is refused: %s." % (key, CONSTRAINT_REFUSED_KEYS[key])
                     for key in refused))
    stems = sorted({key.split('_')[0] for key in unknown})
    near = sorted(key for key in known if key.split('_')[0] in stems)
    detail = ('the same family writes %s' % ', '.join(near[:8])
              if near else 'see the keyword table of the design document')
    raise ValueError(
        "&constraint does not know %s; %s. The section is flat: one keyword "
        'per line, one object per \';\' segment, and no sub-blocks'
        % (', '.join("'%s'" % key for key in unknown), detail))


def _read_constraint_section(section_data):
    """Classic `&constraint` lines into the structured constraint mapping.

    This is the only spelling of the section, in all three carriers: the text
    file's `keyword value` lines, and the YAML or JSON mapping whose keys are
    the same keywords. The structured records this returns are the reader's own
    output and the schema's input, not a second way to write an input file.
    """
    if not isinstance(section_data, dict):
        raise ValueError(
            'the &constraint section needs keyword value lines, got %r'
            % (section_data,))
    values = {}
    for key, item in section_data.items():
        name = str(key).lower()
        values[name] = item
    _check_constraint_keywords(values)
    notes = []
    result = {}
    units = {}
    for key, field in CONSTRAINT_UNIT_KEYS.items():
        if key not in values:
            continue
        segments = _constraint_segments(values[key], key)
        if len(segments) != 1:
            raise ValueError(
                "'%s' takes one unit name; %d segments would mean the run "
                'converts one quantity twice' % (key, len(segments)))
        units[field] = _constraint_read('token', segments[0], key)
    if units:
        result['units'] = units
    wall_types = _constraint_wall_types(values)
    if not wall_types:
        ignored = sorted(key for key in values
                         if key in CONSTRAINT_OBJECT_KEYS
                         and CONSTRAINT_OBJECT_KEYS[key][0] == 'walls')
        ignored += [key for key in values
                    if key in CONSTRAINT_PARAM_KEYS
                    and CONSTRAINT_PARAM_KEYS[key][0] == 'walls']
        if ignored:
            notes.append(
                "wall_type reads no wall, so %s is ignored"
                % ', '.join("'%s'" % key for key in sorted(set(ignored))))
    for section_key, object_keys in CONSTRAINT_FAMILIES:
        records, positions = _constraint_objects(
            section_key, object_keys, values, wall_types, notes)
        _constraint_params(section_key, records, positions, values, notes)
        _constraint_base_ramps(section_key, records, positions, values, notes)
        if section_key == 'walls' and records:
            _constraint_component_ramps(records, values, notes)
        if records:
            result[section_key] = records
    if notes:
        result['notes'] = notes
    return result


def _constraint_declaration(section_data):
    """Read one `&constraint` section, in any of the three carriers.

    The keywords are the same in the classic text, in YAML and in JSON, and so
    is the meaning of one segment: in text a ';' separates the objects, in a
    list one element is one object. `_constraint_segments` holds both.
    """
    return _read_constraint_section(section_data)


# ---------------------------------------------------------------------
# The same grammar, written the other way
# ---------------------------------------------------------------------
# `convert` in the qualification layer emits a classic `.input` from a YAML or
# JSON test config, and the constraint block is the one section whose text is
# not a plain `keyword value` list. The writer lives next to the reader because
# the two are one grammar: a segment separator, an object order and a broadcast
# rule changed on one side only would produce an input file that does not read
# back as the run that wrote it.

# The record fields a structured block may carry without the flat grammar being
# able to state them. `kind` and `region` have defaults the reader restores, so
# only a value that differs from the model's own default is a refusal.
def _reject_workflow_resource_keywords(raw_config):
    """Refuse the compute-resource keywords in a framework section.

    Threads and MPI ranks are consumed by a calculation, and a
    calculation is either a skill's job or a framework-launched child
    process. A workflow section is neither: md/optimize/adaptive/hessian/
    search/composite drive the integration loop and the dispatch shape,
    and the work they spread goes to more processes
    (``multitask``/``multipool``), never to more threads or ranks inside
    one engine. There is therefore no meaning for ``thread``, ``nproc``,
    their ``<skill>_`` spellings, or the retired ``num_thread`` here, and
    reading a keyword without refusing it is the silent drop this
    repository has removed twice already (#299's &md keywords, #300's
    environment keys): the input looks configured, the value is dropped.

    The sections are taken from SECTION_TYPES, the same table
    `_raw_user_keys` uses to decide which sections belong to the
    framework, so this refusal cannot grow a second list of section
    names that drifts from the first.

    Parameters
    ----------
    raw_config : dict or None
        Untouched parsed input.

    Raises
    ------
    ValueError
        When a framework workflow section carries a resource keyword.
    """
    if not isinstance(raw_config, dict):
        return
    offenders = []
    for section in sorted(SECTION_TYPES):
        if not isinstance(raw_config.get(section), dict):
            continue
        found = sorted(
            key for key in raw_config[section]
            if str(key).lower() in ('thread', 'nproc', 'num_thread')
            or str(key).lower().endswith(('_thread', '_nproc')))
        if found:
            offenders.append('%s: %s' % (
                section, ', '.join('`%s`' % key for key in found)))
    if not offenders:
        return
    raise ValueError(
        "a framework workflow section carries the compute-resource "
        "keywords (%s), and none of them is consumed there: a workflow "
        "parallelises with processes (multitask/multipool), not with "
        "threads or ranks inside one engine. Set the keyword where the "
        "calculation actually runs -- `<skill>_thread`/`<skill>_nproc` "
        "(or plain `thread`/`nproc`) in the skill section (&pyscf/&xtb/"
        "&molcas/...)"
        % '; '.join(offenders))


def build_config(raw_config):
    """Build PyRAIConfig from raw configuration dict.

    This is the pipeline entry point. It initializes the runtime
    environment (loading package/user configs, populating Maestro),
    then converts the raw input into a typed PyRAIConfig.

    Parameters
    ----------
    raw_config : dict
        Raw configuration from interface layer. Can be either:
        - Sectioned format: {'control': {...}, 'molecule': {...}, ...}
        - Flat format: {'title': 'xxx', 'jobtype': 'sp', ...}

    Returns
    -------
    PyRAIConfig
        Typed and validated configuration model.
    """
    from ..kernel.environment import Environment
    env = Environment()
    env.initialize()
    _apply_environment_overrides(env, raw_config)
    _reject_workflow_resource_keywords(raw_config)
    config = _build_pydantic_config(raw_config, env_config=env.get_config())
    # Freeze the merged environment into the framework config so the
    # dispatch subsystem reads it through PyRAIState (state.pmd.
    # environment) instead of the kernel Environment singleton.
    config.environment = _build_environment_model(env)
    return config


def _build_environment_model(env):
    """Snapshot the merged environment source into the schema model.

    Reads the kernel environment after runtime overrides, so the
    priority chain Package < ~/.pyrai3md < YAML/JSON < Classic Input
    is fully respected. Only the merged values are copied; no new
    keys are introduced.

    Compute-resource counts are deliberately not carried here: a thread
    count belongs to the calculation that uses it, so it travels as
    `<skill>_thread`/`<skill>_nproc` in the skill section and the
    framework reads them from the task queue when it checks the machine
    (#306 retires the old run-wide `num_thread` floor of #303).

    Parameters
    ----------
    env : Environment
        Initialized kernel environment singleton.

    Returns
    -------
    Environment
        Schema environment model carried by state.pmd.environment.
    """
    from .config import DEFAULT, Environment
    env_section = env.combined_config.get('environment', {}) or {}
    if not isinstance(env_section, dict):
        env_section = {}
    export = env.get_environment_variables()
    return Environment(
        shell=env.get_shell(),
        environment={str(k): str(v) for k, v in export.items()},
        slurm=env.get_slurm() or {},
        # The run's own scratch root: where the framework writes its
        # <title>.log (#306, kernel/logger.RunLogger). An empty value
        # keeps the historical behaviour of writing beside the launch
        # directory.
        scratch_path=str(env_section.get('scratch_path',
                                         DEFAULT['scratch_path']) or ''),
    )


def _apply_environment_overrides(env, raw_config):
    """Propagate the runtime input environment and slurm sections (additive).

    The classic input '&environment' section (or the 'environment'
    entry of a config.json) overrides the package/user pmd_env.yaml
    values so the dispatch subsystem reads one merged source:
    Package < ~/.pyrai3md < YAML/JSON < Classic Input.

    Keys matching schema fields (shell, program_path, ...) and
    free-form key-value pairs both land in the combined environment
    section without schema expansion -- except the retired dispatch
    keys, which are refused (config.RETIRE_ENVIRONMENT_KEYS). Writing
    them read as a setting while nothing consumed them, which is the
    same silent-drop failure #299 removed the &md keywords of.

    The two sections are read independently, and the order below is the
    reason it works: an input that declares `slurm` and no
    `&environment` is the normal case (the grid scan example is one), so
    a slurm merge placed after the environment guard would never run for
    it. `Environment.get_slurm` -- which the job-script builder reads --
    looks the directives up at the top level, not inside `&environment`,
    because they describe the allocation a dispatched child asks the
    cluster for rather than the shell a process starts in. Both the YAML
    block and the classic `&SLURM` section arrive here the same way, so
    the two carriers cannot disagree about what a child submits with.

    Parameters
    ----------
    env : Environment
        Initialized kernel environment singleton.
    raw_config : dict
        Original runtime input configuration.

    Raises
    ------
    ValueError
        If the input environment section carries a retired key.
    """
    if not isinstance(raw_config, dict):
        return
    # The input's overrides are for this run alone: the kernel environment is
    # a process singleton, so without the rollback the next input built in
    # this process would read the present input's directives as host defaults.
    env.reset_input_overrides()
    slurm = raw_config.get('slurm')
    if isinstance(slurm, dict) and slurm:
        env.apply_slurm_overrides(slurm)
    overrides = raw_config.get('environment')
    if not (isinstance(overrides, dict) and overrides):
        return
    retired = sorted(str(key).lower() for key in overrides
                     if str(key).lower() in RETIRE_ENVIRONMENT_KEYS)
    if retired:
        raise ValueError(
            "&environment %s %s retired and read as a setting while "
            "nothing consumed it: %s"
            % (', '.join('`%s`' % key for key in retired),
               'is' if len(retired) == 1 else 'are',
               '; '.join('%s -> %s'
                         % (key, RETIRE_ENVIRONMENT_KEYS[key])
                         for key in retired)))
    env.apply_runtime_overrides(overrides)


def _build_pydantic_config(raw_config, env_config=None):
    """Convert interface raw dict (sectioned) to unified Config.

    Internal function that performs the actual Pydantic Config
    construction. Called by build_config after Environment init.

    Steps:
    1. Expand dotted keys ('section.keyword' agent format)
    2. Apply environment defaults (package + user config)
    3. Flatten sections (control/molecule/md/...) into Config fields
    4. Type conversion (raw strings -> typed values via ReadVal)
    5. Build Config via Pydantic validation

    Parameters
    ----------
    raw_config : dict
        Raw configuration from interface layer.
    env_config : dict, optional
        Combined environment defaults (package + user).

    Returns
    -------
    Config
        Validated Config instance with typed values.
    """
    if isinstance(raw_config, BaseModel):
        raw_dict = raw_config.model_dump()
        # Group flat fields into submodel dicts
        submodel_data = _partition_fields(raw_dict)
        return Config.model_validate(submodel_data)

    if not isinstance(raw_config, dict):
        raise TypeError("Unsupported raw_config type: %s" % type(raw_config).__name__)

    # Expand the dotted flat agent format ('section.keyword' keys)
    # into nested sections before any merging.
    raw_config = _expand_dotted_keys(raw_config)

    # Reject pure flat keyword input before the environment merge
    # (env defaults introduce sections that would mask the flat
    # format). Top-level scalars without framework meaning
    # (skill_name, theory, atoms, ...) stay accepted.
    if not any(isinstance(v, dict) for v in raw_config.values()):
        _framework_keys = (set(FIELD_TO_SUBMODEL) | set(FIELD_MAP)
                             | set(NESTED_FRAMEWORK_SECTIONS)
                             | set(NESTED_SECTION_FIELDS))
        offending = sorted(
            key for key in raw_config if key in _framework_keys)
        if offending:
            raise TypeError(
                "Flat config input is not supported (keys: %s): use "
                "sectioned config ({'control': {...}, ...}) or the "
                "dotted flat format ({'control.jobtype': 'md', "
                "...})." % ', '.join(offending[:8]))

    # Apply environment defaults before processing
    if env_config:
        raw_config = _apply_env_defaults(env_config, raw_config)

    flat_dict = _flatten_sections(raw_config)

    # Convert types
    typed_dict = _convert_types(flat_dict)

    # Partition flat fields into submodel groups and build nested Config
    submodel_data = _partition_fields(typed_dict)
    return Config.model_validate(submodel_data)


# A top-level dispatch keyword (agent mode flattens the sections) names
# the workflow it configures. The jobtype picks the section; a basic
# property run keeps the molecule-scope meaning.
_AMBIGUOUS_HOME_BY_JOBTYPE = {
    'md': 'dynamics',
    'surface_hopping': 'dynamics',
    'adaptive': 'adaptive',
    'optimize': 'optimize',
    'path': 'optimize',
    'mep': 'optimize',
    'irc': 'optimize',
    'neb': 'optimize',
    'frequency': 'hessian',
    'hessian': 'hessian',
    'num_hess': 'hessian',
    'num_freq': 'hessian',
    'search': 'search',
    'scan': 'search',
    'train': 'train',
    'prediction': 'prediction',
}


def _ambiguous_home(flat_dict, key):
    """The section a top-level dispatch keyword belongs to for this run.

    The jobtype names the workflow; if that section does not declare the
    keyword (``optimize`` declares ``multipool`` but not ``multitask``),
    the molecule-scope section is the one that does.
    """
    tokens = str(flat_dict.get('jobtype') or '').strip().lower().split()
    home = _AMBIGUOUS_HOME_BY_JOBTYPE.get(
        tokens[0] if tokens else '', 'system')
    model = SUBMODEL_CLASSES.get(home)
    if model is None or key not in model.model_fields:
        home = 'system'
    return home


def _partition_fields(flat_dict):
    """Partition a flat dict into submodel-grouped dicts for Config.

    Uses FIELD_TO_SUBMODEL to map each field name to its parent
    submodel. Unknown fields are silently dropped (they belong to
    skill-managed models: Properties, Calculation, Environment).
    Nested framework sections (NESTED_FRAMEWORK_SECTIONS) are routed
    by section name and MERGED into their submodel: a section dict
    replaces only the keys it carries, never the whole submodel, so
    fields routed from top-level keys of the same submodel (e.g. the
    example YAML's top-level atoms/coordinates, which map to system
    alongside the nested molecule section) are preserved.

    Parameters
    ----------
    flat_dict : dict
        Flat dictionary with Config field names as keys.

    Returns
    -------
    dict
        Nested dict suitable for Config.model_validate(), e.g.
        {'control': {...}, 'system': {...}, ...}.
    """
    submodel_data = {name: {} for name in SUBMODEL_CLASSES}
    for key, value in flat_dict.items():
        target = NESTED_SECTION_FIELDS.get(key)
        if target is not None and isinstance(value, dict):
            submodel, field = target
            submodel_data[submodel][field] = value
            continue
        if (key in NESTED_FRAMEWORK_SECTIONS
                and isinstance(value, dict)):
            submodel_data[NESTED_FRAMEWORK_SECTIONS[key]].update(value)
            continue
        parent = FIELD_TO_SUBMODEL.get(key)
        if parent is not None:
            if key in AMBIGUOUS_FIELD_NAMES:
                # Agent-mode configs carry the dispatch keywords at the
                # top level. The jobtype names which workflow is being
                # configured, so route there instead of the historical
                # last-model-wins choice.
                parent = _ambiguous_home(flat_dict, key)
            submodel_data[parent][key] = value
    return submodel_data


def _flatten_nested(nested_dict):
    """Flatten a nested submodel dict into a single-level dict.

    The 'environment' snapshot is excluded: it carries dispatch
    policy fields that must never leak into skill config merging.

    Parameters
    ----------
    nested_dict : dict
        Nested dict like {'control': {'title': 'x', ...}, 'system': {...}}.

    Returns
    -------
    dict
        Flat dict with all leaf values.
    """
    flat = {}
    for submodel_name, submodel_data in nested_dict.items():
        # `environment` is a machine snapshot and `read` is a run
        # directive; neither is a skill parameter, so neither may reach
        # the skill merge below (its label names would overwrite a
        # skill's own `nac`/`soc` vocabulary).
        if submodel_name in ('environment', 'read'):
            continue
        if isinstance(submodel_data, dict):
            flat.update(submodel_data)
        else:
            flat[submodel_name] = submodel_data
    return flat


def _apply_env_defaults(env_config, raw_config):
    """Apply environment defaults to raw config.

    Values in raw_config (runtime input) take priority over env_config.
    Empty input values (None, '', [], {}) count as unset and fall
    back to the environment defaults; False and 0 are valid values
    and never fall back.

    Parameters
    ----------
    env_config : dict
        Combined environment defaults (package + user).
    raw_config : dict
        Runtime input configuration.

    Returns
    -------
    dict
        Merged configuration with env defaults applied.
    """
    return deep_merge(env_config, raw_config)






def _expand_dotted_keys(raw_config):
    """Expand 'section.keyword' dotted keys into nested sections.

    The dotted flat format is the agent data medium: a single-level
    JSON dict whose keys carry the section prefix, e.g.
    {'md.multipool': 4, 'control.jobtype': 'adaptive'}. Splitting on
    the first dot restores the sectioned dict consumed by the rest of
    the pipeline, keeping the per-section dispatch keywords
    (multitask/remotetask/multipool) unambiguous.

    Keys without a dot stay at the top level (skill_name, theory,
    atoms, ...).

    Parameters
    ----------
    raw_config : dict
        Raw configuration, possibly carrying dotted keys.

    Returns
    -------
    dict
        Configuration with dotted keys expanded into nested section
        dicts; unchanged when no dotted keys are present.

    Raises
    ------
    ValueError
        For malformed dotted keys or when a dotted key targets a
        section name already present as a top-level entry.
    """
    existing = {
        key.lower() for key in raw_config
        if isinstance(key, str) and '.' not in key
    }
    expanded = {}
    for key, value in raw_config.items():
        if not isinstance(key, str) or '.' not in key:
            expanded[key] = value
            continue
        section_name, _, section_key = key.partition('.')
        section_name = section_name.strip().lower()
        section_key = section_key.strip()
        if not section_name or not section_key:
            raise ValueError(
                "Malformed dotted config key '%s': expected "
                "'section.keyword'." % key)
        if section_name in existing:
            raise ValueError(
                "Dotted config key '%s' conflicts with the "
                "top-level '%s' entry; keep one format per "
                "section." % (key, section_name))
        expanded.setdefault(section_name, {})[section_key] = value
    return expanded


def _flatten_sections(raw_config):
    """Map section-based keys to flat Config field names.

    Handles mixed format: sectioned dicts are flattened, non-dict
    top-level values are preserved as-is.

    The optimize section keys are prefixed with 'opt_' to avoid
    name collisions with skill sections (e.g. both optimize and
    pyscf define 'method').  The prefixed names match the
    FIELD_TO_SUBMODEL entries generated for the Optimize submodel.

    Workflow sections sharing the dispatch-layer keyword names
    (NESTED_FRAMEWORK_SECTIONS) stay nested under their section
    name; flattening them into the shared namespace would let later
    sections silently overwrite earlier ones.

    Parameters
    ----------
    raw_config : dict
        Mixed config: {'control': {...}, 'skill_name': 'xxx', ...}

    Returns
    -------
    dict
        Flat dictionary with all keywords merged (nested framework
        sections kept as sub-dicts).
    """
    flat = {}
    for section_name, section_data in raw_config.items():
        if not isinstance(section_data, dict):
            # Preserve non-dict top-level values (skill_name, theory, atoms, etc.)
            key_lower = section_name.lower()
            config_key = FIELD_MAP.get(key_lower, key_lower)
            flat[config_key] = section_data
            continue
        section_lower = section_name.lower()
        if section_lower in CONSTRAINT_SECTION_NAMES:
            # The classic text of this section is flat `keyword value` lines, so
            # the generic loop above could carry it; what it cannot carry is the
            # meaning of a ';' segment. `_read_constraint_section` gives that,
            # and emits the same mapping the structured `constraints` block of a
            # YAML or JSON input emits, so one set of validators sees both.
            flat['constraint'] = _constraint_declaration(section_data)
            continue
        if section_lower in NESTED_SECTION_FIELDS:
            # Handed over as it came: the inner keys are the constraint
            # schema's vocabulary and its values are nested lists, so neither
            # FIELD_MAP renaming nor ReadVal conversion may run on them.
            flat[section_lower] = section_data
            continue
        if section_lower in NESTED_FRAMEWORK_SECTIONS:
            nested = {}
            for key, value in section_data.items():
                config_key = FIELD_MAP.get(key.lower(), key.lower())
                # Prefix optimize section keys to avoid collisions
                # with skill sections (e.g. method, maxit). Dispatch
                # keywords, keywords that already name the section, and
                # keys already prefixed keep their canonical names.
                if (section_lower == 'optimize'
                        and not config_key.startswith('opt_')
                        and config_key not in DISPATCH_TYPES
                        and config_key not in OPTIMIZE_PASSTHROUGH_KEYS):
                    config_key = 'opt_' + config_key
                nested[config_key] = value
            flat[section_lower] = _convert_types(
                nested, dict(DISPATCH_TYPES,
                             **SECTION_TYPES.get(section_lower, {})))
            continue

        for key, value in section_data.items():
            key_lower = key.lower()
            # Apply field name mapping
            config_key = FIELD_MAP.get(key_lower, key_lower)
            # Store with type info for later conversion
            flat[config_key] = value

    return flat


def _parse_lattice_value(value):
    """Parse a lattice input into the 3x3 lattice_vector matrix.

    The spellings and their one reading live in `utilities.lattice.parse_cell`;
    this name is kept because the adapter's field map and tests call it.
    """
    from ..utilities.lattice import parse_cell
    return parse_cell(value)


def _convert_center_or_lattice(key, value):
    """Convert one shared center/lattice keyword.

    Returns ``(handled, converted)``. The flat adapter and the skill-dict
    unification both call this, so `highlevel`/`ci`/`coupling`/`root`/
    `charges`/`lattice` are read the same way in every input mode.

    Parameters
    ----------
    key : str
        Keyword spelling (already reverse-mapped by the adapter).
    value : str
        The raw classic-input value.

    Returns
    -------
    tuple(bool, object)
        Whether this keyword was handled, and the converted value.
    """
    if key in _CENTER_INDEX_FIELDS:
        centers = _parse_center_indices(value, key,
                                        start=INDEX_START.get(key, 1))
        return True, centers[0] if len(centers) == 1 else centers
    if key in _CENTER_INT_LIST_FIELDS:
        return True, _parse_center_int_lists(value, key)
    if key in _CENTER_PAIR_FIELDS:
        return True, _parse_center_pairs(value, key,
                                         start=INDEX_START.get(key, 1))
    if key in _CENTER_SCALAR_FIELDS:
        return True, _parse_center_scalars(
            value, key, start=INDEX_START.get(key, 0))
    if key in _CENTER_SCALAR_LIST_FIELDS:
        return True, _parse_center_scalars(
            value, key, start=INDEX_START.get(key, 0))
    if key in ('lattice', 'lattice_vector'):
        return True, _parse_lattice_value(value)
    return False, value


def _decrement_track_index(value):
    """Typed track groups from the external 1-based spelling to 0-based.

    The classic string path already decrements through ReadIndex; a typed
    value (yaml, agent config, programmatic build) arrived unchanged and named
    atoms one apart from the same declaration written as text. Track groups are
    kept in the order given: a pair is not a center set, so sorting would be a
    second convention for the same number.
    """
    offset = INDEX_START.get('track_index', 1)
    groups = []
    for group in value:
        atoms = group if isinstance(group, (list, tuple)) else [group]
        groups.append([int(atom) - offset for atom in atoms])
    return groups


def _convert_types(flat_dict, type_map=None):
    """Convert raw string values to proper types using ReadVal.

    Parameters
    ----------
    flat_dict : dict
        Flat dictionary with raw string values.
    type_map : dict or None
        Explicit key -> type-code map (nested framework sections);
        None builds the combined framework type map.

    Returns
    -------
    dict
        Dictionary with typed values.
    """
    # Build combined type map
    if type_map is None:
        all_types = {}
        all_types.update(CONTROL_TYPES)
        all_types.update(ADAPTIVE_TYPES)
        all_types.update(MOLECULE_TYPES)
        all_types.update(MD_TYPES)
        all_types.update(OPTIMIZE_TYPES)
        all_types.update(HESSIAN_TYPES)
        all_types.update(SEARCH_TYPES)
        all_types.update(DISPATCH_TYPES)
        all_types.update(COMPOSITE_TYPES)
    else:
        all_types = type_map

    converted = {}
    for key, value in flat_dict.items():
        # Apply field name mapping for type lookup
        lookup_key = key
        for orig, mapped in FIELD_MAP.items():
            if mapped == key:
                lookup_key = orig
                break

        # jobtype: preserve full string (may contain space-separated tokens)
        # Do not use ReadVal('s') which only keeps the first token.
        if lookup_key == 'jobtype':
            if isinstance(value, list):
                converted[key] = ' '.join(str(v) for v in value)
            elif isinstance(value, str):
                converted[key] = value.strip()
            else:
                converted[key] = str(value)
            continue

        # A typed track_index (yaml / agent / programmatic input) carries
        # 1-based atom numbers like every other atom selection; the classic
        # string path below decrements through ReadIndex, so only the typed
        # spelling needs this branch.
        if lookup_key == 'track_index' and isinstance(value, (list, tuple)):
            converted[key] = _decrement_track_index(value)
            continue

        handled, converted_value = _convert_center_or_lattice(
            lookup_key, value)
        if handled:
            converted[key] = converted_value
            continue

        # &search grid: a mapping of the skill's hyperparameters to their
        # candidate lists. Owned here rather than by the type map because the
        # classic carrier hands it over as one structured string, and the
        # keyword names belong to a skill the adapter does not read.
        # A per-center pair travels as the classic carrier's space-separated
        # list (`state_i 0 0`); one token stays an int so a single-center run's
        # scalar path is untouched (read values are already typed here).
        if lookup_key in ('opt_state_i', 'opt_state_j'):
            if not isinstance(value, str):
                # Already typed (yaml/json): a list stays a list, an int stays
                # an int; the schema's Union is what decides.
                converted[key] = value
                continue
            tokens = value.split()
            try:
                converted[key] = ([int(token) for token in tokens]
                                  if len(tokens) > 1 else int(tokens[0]))
            except ValueError:
                converted[key] = value
            continue

        if lookup_key == SEARCH_GRID_FIELD:
            converted[key] = _parse_search_grid(value, key)
            continue

        type_code = all_types.get(lookup_key)

        if type_code is None:
            # No type info - pass through
            converted[key] = value
            continue

        if not isinstance(value, str):
            # Already typed (YAML/JSON entry). String-typed keywords
            # accept plain scalars by coercing to their canonical
            # string form (e.g. YAML `initcond: 0` -> '0'); booleans
            # are excluded because YAML 1.1 folds off/on/no/yes into
            # them and the original spelling is unrecoverable. Every
            # other type code passes typed values through.
            if (type_code == 's'
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)):
                converted[key] = str(value)
            else:
                converted[key] = value
            continue

        # Convert using ReadVal or ReadIndex
        try:
            tokens = value.split()
            if type_code == 'g':
                # Group index: use ReadIndex with optional custom start
                start = INDEX_START.get(lookup_key, 0)
                converter = ReadIndex('g', start=start, sort=False)
                converted[key] = converter(tokens)
            elif type_code == 'si':
                # Single index list: use ReadIndex('s') with optional custom start
                start = INDEX_START.get(lookup_key, 0)
                converter = ReadIndex('s', start=start, sort=True)
                converted[key] = converter(tokens)
            else:
                converter = ReadVal(type_code)
                converted[key] = converter(tokens)
        except (ValueError, IndexError, KeyError):
            # If conversion fails, keep raw value
            converted[key] = value

    return converted



def read_json(filename):
    """Read a configuration dictionary from a JSON file.

    Parameters
    ----------
    filename : str
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(filename, 'r') as f:
        return json.load(f)



def _raw_user_keys(raw_config):
    """Collect the field spellings the user actually provided.

    The typed Config loses provenance: a field whose value equals the
    framework default is indistinguishable from one the user never
    set, so the Config overlay in _merge_one_skill_config cannot rely
    on values alone. The raw input keeps the record -- every key
    present in a raw FRAMEWORK section was written by the user (or by
    the certification converter on the user's behalf). Skill sections
    are excluded: their keys reach the merged dict verbatim through
    the highest-priority deep merge and must not mark framework
    canonical fields as user-set. Keys are lower-cased, dotted agent
    spellings ('md.temp') contribute their field part when the dotted
    head is a framework section, and FIELD_MAP aliases resolve to the
    canonical field name so the provenance test matches the flattened
    Config fields.

    Parameters
    ----------
    raw_config : dict or None
        Original raw input (sectioned, dotted-flat or classic parse).

    Returns
    -------
    set of str
        Canonical lower-case framework field names the user provided.
    """
    framework_sections = frozenset(SECTION_TYPES) | {'environment'}
    keys = set()
    if not isinstance(raw_config, dict):
        return keys

    def _add(key):
        name = str(key).lower()
        keys.add(name)
        canonical = FIELD_MAP.get(name)
        if canonical:
            keys.add(canonical)

    for key, value in raw_config.items():
        key_lower = str(key).lower()
        if isinstance(value, dict):
            if key_lower not in framework_sections:
                continue  # skill / skills sections: not provenance
            for sub_key in value:
                _add(sub_key)
        else:
            if '.' in key_lower:
                # Dotted agent spelling: only a framework-section head
                # counts as framework provenance.
                head, tail = key_lower.split('.', 1)
                if head in framework_sections:
                    _add(tail)
            else:
                _add(key_lower)
    return keys


def _control_debug(config, raw_config):
    """Read the global test switch out of either representation of the input.

    `config` is the validated framework Config, `raw_config` the untouched
    input. Either may be absent -- a skill test that drives the merge
    directly hands over only one of them -- so both are asked and the
    answer is the same question in two spellings ('true' in a classic
    section, True in yaml).
    """
    for holder in (getattr(getattr(config, 'control', None), 'debug', None),
                   (raw_config or {}).get('control', {})
                   if isinstance((raw_config or {}).get('control'), dict)
                   else None):
        if isinstance(holder, str):
            if holder.strip().lower() in ('true', '1'):
                return True
        elif bool(holder):
            return True
    return False


def _control_driver(config, raw_config):
    """Read the execution driver out of either representation of the input.

    Same two-spelling question as `_control_debug` above: the validated
    framework Config and the untouched input, either of which may be absent.
    Anything but 'main' means the skill receives the whole jobtype and
    therefore owns the geometry, which is what decides constraint ownership
    (dev/methods/CONSTRAINT_DEVELOPMENT.md section 31).
    """
    for holder in (getattr(getattr(config, 'control', None), 'driver', None),
                   (raw_config or {}).get('control', {})
                   if isinstance((raw_config or {}).get('control'), dict)
                   else None):
        if isinstance(holder, str) and holder.strip():
            return holder.strip()
    return 'main'


def _merge_one_skill_config(maestro, skill_name, config=None, raw_config=None):
    """Merge skill runtime config from Maestro with user input.

    config.json and pmd_env.yaml share the same structure:
    skill-specific settings are under ``skills.<skill_name>``.
    Input text format uses ``<skill_name>`` as top-level section key.

    Priority (highest to lowest), layered over the Maestro defaults
    base (package pmd_env.yaml):
    0. control.debug, in the one direction that means something: when the
       run was started as a test, every module under test is tested. It
       sits outside the ladder because the ladder would otherwise put a
       skill's own spelling of the same name above the switch that is
       supposed to include it (lammps_skill calls its per-step artifact
       dump `debug` too, and skills/lammps_skill/tests owns that spelling).
    1. Skill section from input file (&pyscf, &molcas, etc.)
    2. User raw_config['skills'][skill_name] (config.json / pmd_env.yaml)
    3. Framework Config (validated, filtered to framework fields)

    Parameters
    ----------
    maestro : Maestro
        The populated Maestro instance.
    skill_name : str
        Name of the skill.
    config : dict, Config, or None
        Framework Config from user input.
    raw_config : dict or None
        Original user input.

    Returns
    -------
    dict or None
        Merged config dict, or None if no config provided.
    """
    # Get skill runtime config from Maestro (package pmd_env.yaml)
    skill_runtime = maestro.get_config(skill_name)

    # Start with Maestro defaults as base. Drop None values: an empty
    # yaml entry (e.g. 'slurm_memory:' with no value) parses as None
    # and would fail the skill schema validation for typed fields.
    if skill_runtime:
        merged = {
            key: value for key, value in copy.deepcopy(skill_runtime).items()
            if value is not None
        }
    else:
        merged = {}

    # Overlay framework Config -- but only fields the user actually
    # set. Framework defaults (empty strings, zeros) must NOT
    # overwrite skill-specific defaults from Maestro or skill section.
    # Provenance: a field counts as user-set when its spelling appears
    # in the raw input -- even when the value coincides with the
    # framework default (an explicit 'temperature: 300' must survive,
    # not fall through to the skill default) -- or when it differs
    # from the default (normalize-derived values such as nstate from
    # ci). Everything else keeps the layer below.
    if config is not None:
        if hasattr(config, 'model_dump'):
            config_dict = config.model_dump()
        elif isinstance(config, dict):
            config_dict = dict(config)
        else:
            config_dict = {}
        framework_defaults = Config().model_dump()
        # config_dict and framework_defaults are both nested:
        # {'control': {...}, 'system': {...}, ...}
        # Flatten both for per-field comparison
        flat_user = _flatten_nested(config_dict)
        flat_defaults = _flatten_nested(framework_defaults)
        user_keys = _raw_user_keys(raw_config)
        # Control.debug reaches the skill here like any other framework
        # field the user actually set. It is not enough on its own: the
        # priority ladder of this function puts the skill's own input
        # section ABOVE the framework Config, so a skill that spelled out
        # debug would outrank the global switch it is supposed to be
        # subject to. The force below is applied after every layer for
        # exactly that reason.
        # Framework-only state selectors that a multi-center run names PER
        # CENTER. A skill solves one region and receives its own state list
        # through `target_states`, so a per-center sequence here would be dead
        # data at best and a validation failure at worst (every skill's field is
        # a scalar). Scalars keep travelling: a single-center run's root and
        # state indices are what the skill has always been handed.
        _PER_CENTER_ONLY = ('root', 'opt_state_i', 'opt_state_j')
        for key, value in flat_user.items():
            if key in _PER_CENTER_ONLY and isinstance(value, (list, tuple)):
                continue
            if key in user_keys or value != flat_defaults.get(key):
                # The framework carries the embedding / read_charge
                # decisions as bools. Skills with a bool field take the
                # value as it is; the string-vocabulary skill (Molcas)
                # folds the same bool in its own field validator. A
                # generic bool -> 'charge'/'legacy' rewrite here broke
                # every bool-typed skill section (measured on the
                # multiscale examples: xtb SkillConfig rejected
                # embedding='charge').
                merged[key] = value

    # Overlay state fields directly.  The framework normalize may
    # recompute skill-specific fields (nstate, state_spin) from
    # framework defaults (ci, spin), overwriting user values.
    # Sources: top level (scalar passthrough such as skill_name) and
    # the molecule section (sectioned/dotted input, where System
    # fields live).
    _SKILL_STATE_KEYS = (
        'nstate', 'state_spin', 'target_states', 'target_nac_pair',
        'nact_elec', 'nact_orb', 'nclosed', 'active',
        'ci_roots', 'ci_spin',
    )
    if raw_config and isinstance(raw_config, dict):
        state_sources = [raw_config]
        molecule_section = raw_config.get('molecule')
        if isinstance(molecule_section, dict):
            state_sources.append(molecule_section)
        for source in state_sources:
            for key in _SKILL_STATE_KEYS:
                # Skip null values: they mean "not provided" and must
                # not overwrite framework-derived values.
                if key in source and source[key] is not None:
                    merged[key] = source[key]

    # Skill-internal dynamics step limits: maxit_md/md_maxiter are
    # skill parameters that the framework Config drops (no submodel
    # owns them), so forward the user values from any input section
    # (typically &md) into the skill config. The skill section merge
    # below keeps the highest priority.
    _MD_SKILL_KEYS = ('maxit_md', 'md_maxiter')
    if raw_config and isinstance(raw_config, dict):
        sources = [raw_config]
        sources.extend(
            value for value in raw_config.values()
            if isinstance(value, dict))
        for key in _MD_SKILL_KEYS:
            for source in sources:
                if source.get(key) is not None:
                    merged[key] = source[key]

    # Keep state_spin consistent with nstate.  The framework may
    # derive state_spin from its own (default) nstate before the raw
    # input overrides nstate, leaving a length mismatch that breaks
    # skill normalization.  Drop the stale value so the skill schema
    # regenerates it.
    nstate = merged.get('nstate')
    state_spin = merged.get('state_spin')
    if (
        isinstance(nstate, int) and nstate >= 1
        and isinstance(state_spin, (list, tuple))
        and len(state_spin) != nstate
    ):
        merged['state_spin'] = []

    # Overlay user's skill-specific input from raw_config
    if raw_config and isinstance(raw_config, dict):
        # Source 1: config.json format -- raw_config['skills'][skill_name]
        user_skill_config = (
            raw_config.get('skills', {}).get(skill_name, {})
            if isinstance(raw_config.get('skills'), dict)
            else {}
        )

        # Source 2: input text format -- support both &pyscf and &pyscf_skill
        skill_info = maestro.get_skill_info(skill_name) or {}
        short_name = skill_info.get('name', '')

        # Check both &pyscf (short) and &pyscf_skill (full)
        skill_section = raw_config.get(short_name, {}) if short_name else {}
        if not isinstance(skill_section, dict):
            skill_section = {}
        skill_section_full = raw_config.get(skill_name, {})
        if not isinstance(skill_section_full, dict):
            skill_section_full = {}

        # Source 2b: the same block in the agent `.config` dialect. The
        # converter keeps every skill's keywords under `skill_sections`,
        # keyed by the skill stem, because two skills spelling the same
        # keyword (`method`) cannot share a flat namespace - and it then
        # flattens ONLY the first `control.theory` entry to the top level.
        # Without this reader the second skill's own block reaches nobody:
        # the primary's flat `method` is the only one in the file, so it
        # lands on every skill, each other skill's profile resolution is
        # refused for an unknown method, and it falls back to its default
        # profile - which for a gradient jobtype is an energy profile.
        sectioned = raw_config.get('skill_sections')
        skill_section_sectioned = (
            sectioned.get(short_name)
            if isinstance(sectioned, dict) and short_name else None)
        if not isinstance(skill_section_sectioned, dict):
            skill_section_sectioned = {}

        # The user's own statements, lowest to highest: the config.json
        # block, then the agent dialect's `skill_sections`, then the input
        # section under its short name, then under the full package name.
        # One ordered list, so the order is the code and a new dialect
        # joins by appending to it.
        for name, mapping in (
                ('skills.%s' % skill_name, user_skill_config),
                ('skill_sections.%s' % short_name, skill_section_sectioned),
                (short_name, skill_section),
                (skill_name, skill_section_full)):
            if mapping:
                merged = deep_merge(merged, mapping)

        # Source 3: a skill may declare extra input sections beside its
        # own block in its manifest. Each is folded in under its own key,
        # which is all the framework knows about it -- what the section
        # contains and what that means stays inside the skill.
        for sub in (skill_info.get('subsections') or []):
            subsection = raw_config.get(sub)
            if isinstance(subsection, dict):
                base = merged.get(sub)
                merged[sub] = dict(base) if isinstance(base, dict) else {}
                merged[sub] = deep_merge(merged[sub], subsection)

    # Layer 0. Applied after every layer above so a skill section cannot
    # outrank the global switch it is subject to. Only the positive
    # direction forces: a run that did not ask to be debugged must not
    # silence a skill test that asked for its own artifacts, which is what
    # skills/lammps_skill/tests/ff_gradient_persistent.yaml does.
    if _control_debug(config, raw_config):
        merged['debug'] = True
    # Constraint ownership travels with the driver, and for the same reason
    # `debug` does: a skill section cannot outrank the switch that decides who
    # moves the atoms. `main` means the framework integrates and constrains, so
    # the skill's own constraint keywords are inert; anything else means the
    # skill owns the jobtype, so the framework's `&system_constraint` block is
    # inert. The receiving skill reads this value and applies the opposite
    # control (skills/lammps_skill/schema/validate.py).
    # Only for a run that has a skill section at all: an input that hands the
    # skill nothing must keep saying so (TestFrameworkForwardsTheSwitch), and a
    # skill built from its own defaults is a skill that drives itself, which is
    # the same answer the driver would have carried.
    if merged:
        merged['driver'] = _control_driver(config, raw_config)

    unify_skill_flat_dict(merged)

    return merged if merged else None


# ---------------------------------------------------------------------
# Skill flat-dict unification (framework side, schema/adapter layer)
#
# Concept contract: unification is NOT normalize/validate. Type
# legality belongs to the skill (its build_config / normalize /
# validate reject bad values and report back, exactly like an AI
# agent call); shape determinism belongs to the framework: the three
# input modes (yaml / classic .input / agent .config) must hand every
# skill backend a key-by-key identically typed flat dict for the
# shared vocabulary. When a skill reports a type error in API mode,
# the fix belongs in this module only -- never in a new ad-hoc patch
# function next to it.
# ---------------------------------------------------------------------

# Shared framework/skill state vocabulary beyond the classic
# SECTION_TYPES tables: keywords the framework Config models as typed
# fields but the classic tables do not code. The certification
# converter writes them as text tokens; the unified keyword design
# keeps both ends aligned (dev/SKILL_APP_DEVELOPMENT.md section 6.2).
_SKILL_UNIFICATION_EXTRA_TYPES = {
    'nstate': 'i',
    'state_spin': 'il',
    'target_states': 'il',
    'ci_roots': 'il',
    'ci_spin': 'il',
    'active': 'il',
    'nact_elec': 'i',
    'nact_orb': 'i',
    'nclosed': 'i',
    'inactive_orb': 'i',
    'maxit_md': 'i',
    'md_maxiter': 'i',
}


def _skill_unification_types():
    """Combined type-code map for the shared framework/skill vocabulary.

    Single source of truth: the classic SECTION_TYPES code tables (the
    documented shared keyword design) plus the state-key extras and
    the FIELD_MAP alias spellings, so a merged dict carrying either
    spelling of an aliased keyword converts identically.

    Returns
    -------
    dict
        Keyword -> type-code map used by unify_skill_flat_dict.
    """
    combined = {}
    for table in (MOLECULE_TYPES, MD_TYPES, OPTIMIZE_TYPES,
                  DISPATCH_TYPES):
        combined.update(table)
    combined.update(_SKILL_UNIFICATION_EXTRA_TYPES)
    for source, target in FIELD_MAP.items():
        if source in combined:
            combined.setdefault(target, combined[source])
    return combined


_SKILL_UNIFICATION_TYPES = _skill_unification_types()


def unify_skill_flat_dict(merged):
    """Unify shared-vocabulary keys of a merged skill flat dict.

    Converts classic-text spellings (e.g. ci '3', coupling '1 2, 2 3',
    pbc '1 1 1', highlevel '1-6') into the canonical typed forms the
    skill schemas expect, so the yaml / classic-input / agent-config
    entry points hand every skill backend the same value shapes.

    Rules:
    - Vocabulary: _SKILL_UNIFICATION_TYPES (the shared keyword design)
      plus the center-parsed field sets. Skill-private keywords are
      unknown to the framework and pass through untouched (the skill
      normalize owns them).
    - 's'-coded keywords keep their string form (already canonical).
    - Typed values pass through, except float-coded keys carrying a
      JSON int (and float lists carrying ints) which are cast to the
      canonical float shape so the modes agree on types as well.
    - Multi-center strings (';'-separated) pass through: the
      per-center typed projection in planning/assembly.py owns them.
    - Unparsable values are kept raw so the skill schema reports the
      original error (the agent-mode feedback channel; in API mode it
      locates the unification gap to fix here, and only here).
    - Keys are never renamed (FIELD_MAP applies to spelling lookup
      only), so skill-section echoes and existing artifacts stay
      byte-identical.

    Parameters
    ----------
    merged : dict
        Merged flat skill config from _merge_one_skill_config,
        modified in place.

    Returns
    -------
    dict
        The same dict, with shared-vocabulary values unified.
    """
    if not isinstance(merged, dict):
        return merged

    for key, value in list(merged.items()):
        code = _SKILL_UNIFICATION_TYPES.get(key)
        if not isinstance(value, str):
            # Typed value (YAML/JSON): enforce the canonical numeric
            # shape of the shared vocabulary, so a JSON int and a
            # YAML float spelling of the same number do not differ
            # across modes (e.g. agent 'temperature: 300' vs yaml
            # typed 300.0).
            if (code == 'f' and isinstance(value, int)
                    and not isinstance(value, bool)):
                merged[key] = float(value)
            elif (code == 'fl'
                    and isinstance(value, (list, tuple)) and value
                    and all(isinstance(item, (int, float))
                            and not isinstance(item, bool)
                            for item in value)):
                merged[key] = [float(item) for item in value]
            continue
        if not value.strip():
            continue
        if ';' in value:
            continue
        is_center = (key in _CENTER_INDEX_FIELDS
                     or key in _CENTER_INT_LIST_FIELDS
                     or key in _CENTER_PAIR_FIELDS
                     or key in _CENTER_SCALAR_FIELDS
                     or key in _CENTER_SCALAR_LIST_FIELDS)
        if code == 's':
            continue
        if code is None and not is_center:
            continue
        try:
            handled, converted_value = _convert_center_or_lattice(key, value)
            if handled:
                merged[key] = converted_value
            else:
                tokens = value.split()
                if code == 'g':
                    converter = ReadIndex(
                        'g', start=INDEX_START.get(key, 0), sort=False)
                elif code == 'si':
                    converter = ReadIndex(
                        's', start=INDEX_START.get(key, 0), sort=True)
                else:
                    converter = ReadVal(code)
                merged[key] = converter(tokens)
        except (ValueError, IndexError, KeyError, TypeError):
            continue  # keep raw; the skill schema reports the original
    return merged
