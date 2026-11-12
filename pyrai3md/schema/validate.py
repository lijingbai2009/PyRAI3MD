"""
======================================================================
This module provides validation functions and the Task schema for
PyRAI3MD configuration objects. It checks the consistency of control
parameters, system parameters, electronic settings, and dynamics
settings.

The Task model (merged from former task.py) describes a computation
task to be executed by a skill.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import warnings

import numpy as np
from pydantic import BaseModel

from .config import flatten_config_dump
from .config import (DEFAULT, FIELD_TO_SUBMODEL, SUBMODEL_CLASSES,
                     NESTED_FRAMEWORK_SECTIONS,
                     ConstraintInputError, UnsupportedCapabilityError)
from ..utilities.lattice import parse_cell, validate_lattice


# Basic jobtype options (freely composable, space-separated)
BASIC_JOBTYPES = frozenset({
    'energy', 'gradient', 'nac', 'soc', 'frequency', 'absorption',
    'dipole', 'transition', 'hessian', 'num_freq', 'num_hess',
})

# Advanced jobtype options (standalone only, not composable)
ADVANCED_JOBTYPES = frozenset({
    'md', 'adaptive', 'train', 'prediction', 'search', 'optimize',
    'surface_hopping', 'read',
})

# Ranking options of the hyperparameter grid scan (&search).
SEARCH_SORT_STATS = ('max', 'mean')
SEARCH_SORT_ORDERS = ('asc', 'desc')

# Backward-compatible aliases: old_name -> canonical_name
# Value None means the option has been removed with no direct replacement.
JOBTYPE_ALIASES = {
    'sp': 'energy',
    'predict': 'prediction',
    'opt': 'optimize',
    'freq': 'frequency',
    'frequence': 'frequency',
    'hess': 'hessian',
    'numfreq': 'num_freq',
    'numhess': 'num_hess',
    'namd': 'md',
}

# Jobtype tokens that were removed with no replacement. Refused where
# the alias table is applied, so no later stage has to recognise them
# (#5.9): a token parked here as a silent no-op used to travel through
# normalize untouched and fail much later as an unknown jobtype.
RETIRED_JOBTYPES = {
    'hop': ("surface hopping is `md` with `nactype nac` and `sfhp`; there is "
            "no separate hop jobtype"),
    'hybrid': ("a hybrid calculation is a composite run (control.theory with "
               "a multiscale scheme), not a jobtype"),
}

# Built-in theory method names that have no skill directory of their
# own (machine-learning potentials). QC program labels (molcas,
# orca, cp2k, ...) come from the dynamic skill registry, see
# _valid_theory_methods().
BUILTIN_THEORY_METHODS = frozenset({'nn', 'e2n2'})

# Static fallback used when the skill registry cannot be discovered
# (e.g. a partial installation). Kept in sync with the shipped skills.
FALLBACK_THEORY_METHODS = BUILTIN_THEORY_METHODS | frozenset({
    'molcas', 'bagel', 'orca', 'xtb', 'openqp', 'pyscf',
    'lammps', 'cp2k', 'qdm',
})

# Cached dynamic method set (None = not resolved yet)
_THEORY_METHOD_CACHE = None

# Ensemble control vocabularies (dev/methods/ENSEMBLE_DEVELOPMENT.md).
# Values are compared case-insensitively; aliases are normalized to
# the canonical token before membership checks.
VALID_ENSEMBLES = frozenset({'nve', 'nvt', 'npt'})
VALID_THERMOSTATS = frozenset(
    {'off', 'nose-hoover', 'nose-hoover-chain', 'csvr'})
VALID_BAROSTATS = frozenset(
    {'off', 'berendsen', 'parrinello-rahman', 'mttk', 'scr'})
VALID_BAROSTAT_MODES = frozenset({'isotropic', 'anisotropic'})
THERMOSTAT_ALIASES = {
    'nh': 'nose-hoover', 'nose': 'nose-hoover',
    'nhc': 'nose-hoover-chain', 'nose-hoover-chains': 'nose-hoover-chain',
    'csnr': 'csvr',
}
BAROSTAT_ALIASES = {
    'pr': 'parrinello-rahman', 'parrinello': 'parrinello-rahman',
    'parrinello-rahman-npt': 'parrinello-rahman',
    'c-rescale': 'scr', 'stochastic-cell-rescaling': 'scr',
}

# Which particle thermostat a cell algorithm may be combined with.
#
# This is a statement about the literature, not a preference, and each
# entry names the scheme a pairing is: an extended-cell barostat carries
# a Nosé-Hoover chain in its own derivation, so it has no defined
# stationary measure once the particles are rescaled stochastically
# instead, while a first-order stochastic barostat was derived to accept
# any thermostat. An unlisted pairing is refused here rather than left to
# show up as a drifting trajectory twenty thousand steps later;
# tmp_20260929_121646 measured exactly that (csvr + parrinello-rahman
# reached T_inst 4339 K against a 300 K target).
_NH_FAMILY = frozenset({'nose-hoover', 'nose-hoover-chain'})
BAROSTAT_THERMOSTATS = {
    # Berendsen weak coupling is a relaxation scheme with no ensemble
    # claim, so it is not picky about what it relaxes.
    'berendsen': VALID_THERMOSTATS - {'off'},
    # Parrinello-Rahman (J. Appl. Phys. 52, 7182 (1981)) and MTTK
    # (Martyna, Tobias, Klein, J. Chem. Phys. 101, 4177 (1994)) are
    # deterministic extended-Lagrangian cell dynamics: the chain on the
    # particle momenta is part of how they are derived.
    'parrinello-rahman': _NH_FAMILY,
    'mttk': _NH_FAMILY,
    # Stochastic cell rescaling (Bernetti & Bussi, J. Chem. Phys. 153,
    # 114107 (2020); Del Tatto, J. Chem. Phys. / arXiv:2111.06402 for the
    # fully anisotropic form) is a first-order stochastic cell equation,
    # and its own text states it "can in principle be combined with any
    # thermostat to sample the isothermal-isobaric ensemble".
    'scr': VALID_THERMOSTATS - {'off'},
}
BAROSTAT_PAIRING_REASON = {
    'berendsen': 'Berendsen weak coupling (JCP 81, 3684 (1984)) is a '
                 'relaxation scheme and constrains no thermostat choice',
    'parrinello-rahman': 'the extended-Lagrangian cell is derived with a '
                         'Nosé-Hoover chain on the particle momenta',
    'mttk': 'MTTK is Parrinello-Rahman plus a chain on the cell momentum; '
            'its derivation assumes a chain on the particles too',
    'scr': 'stochastic cell rescaling is a first-order cell equation and '
           'was derived to pair with any thermostat',
}


def _valid_theory_methods():
    """Resolve the set of valid theory methods.

    Combines the built-in methods with the program labels registered
    by the installed skills (skills/query.discover_all), so newly
    added skills become valid without touching this whitelist. When
    the registry cannot be discovered, the static fallback keeps
    validation functional instead of blocking it.

    Returns
    -------
    frozenset
        Valid theory method labels.
    """
    global _THEORY_METHOD_CACHE
    if _THEORY_METHOD_CACHE is not None:
        return _THEORY_METHOD_CACHE

    try:
        from ..skills.query import discover_all
        skill_labels = frozenset(discover_all().keys())
    except Exception as error:
        # A broken skill package must not silently narrow the accepted
        # method list to the built-in fallback; say so and keep going.
        warnings.warn(
            "skill discovery failed (%s: %s); theory methods are checked "
            "against the built-in fallback list"
            % (type(error).__name__, error),
            UserWarning, stacklevel=2)
        skill_labels = frozenset()

    if skill_labels:
        _THEORY_METHOD_CACHE = BUILTIN_THEORY_METHODS | skill_labels
    else:
        _THEORY_METHOD_CACHE = FALLBACK_THEORY_METHODS
    return _THEORY_METHOD_CACHE


def _valid_reference_methods():
    """Resolve the set of valid reference methods.

    Same as the theory methods except 'nn': reference chooses the
    reference QC electronic property calculator for recomputing
    QC-data during adaptive sampling.

    Returns
    -------
    frozenset
        Valid reference method labels.
    """
    return _valid_theory_methods() - {'nn'}


def parse_jobtype(jobtype_str):
    """Parse a jobtype string into canonical tokens with classification.

    Handles:
    1. Space-separated multi-token strings
    2. Alias replacement (sp -> energy, predict -> prediction, etc.)
    3. Classification into basic vs advanced

    Parameters
    ----------
    jobtype_str : str
        Raw jobtype string from input (e.g. 'sp', 'energy gradient', 'md').

    Returns
    -------
    tuple of (list[str], bool, bool)
        tokens: canonical token list (deduplicated, order preserved)
        is_basic: True if all tokens are basic jobtypes
        is_advanced: True if exactly one advanced jobtype token

    Raises
    ------
    ValueError
        If any token is unrecognized, or if basic/advanced are mixed,
        or if multiple advanced tokens are given.
    """
    if not isinstance(jobtype_str, str):
        jobtype_str = str(jobtype_str)

    raw_tokens = jobtype_str.lower().split()
    if not raw_tokens:
        raise ValueError(
            "jobtype is empty. Basic options: %s. Advanced options: %s"
            % (sorted(BASIC_JOBTYPES), sorted(ADVANCED_JOBTYPES))
        )

    # Apply aliases
    canonical = []
    for token in raw_tokens:
        if token in RETIRED_JOBTYPES:
            raise ValueError(
                "jobtype '%s' has been removed: %s."
                % (token, RETIRED_JOBTYPES[token]))
        if token in JOBTYPE_ALIASES:
            canonical.append(JOBTYPE_ALIASES[token])
        elif token in BASIC_JOBTYPES or token in ADVANCED_JOBTYPES:
            canonical.append(token)
        else:
            raise ValueError(
                "Unknown jobtype '%s'. "
                "Basic options: %s. Advanced options: %s"
                % (token, sorted(BASIC_JOBTYPES), sorted(ADVANCED_JOBTYPES))
            )

    # Deduplicate while preserving order
    seen = set()
    tokens = []
    for t in canonical:
        if t not in seen:
            seen.add(t)
            tokens.append(t)

    # Classify
    basic_tokens = [t for t in tokens if t in BASIC_JOBTYPES]
    advanced_tokens = [t for t in tokens if t in ADVANCED_JOBTYPES]

    if basic_tokens and advanced_tokens:
        raise ValueError(
            "Cannot combine basic jobtypes %s with advanced jobtypes %s. "
            "Basic options are freely composable; advanced options must be used alone."
            % (basic_tokens, advanced_tokens)
        )

    if len(advanced_tokens) > 1:
        raise ValueError(
            "Advanced jobtypes cannot be combined: %s. "
            "Use only one advanced option at a time."
            % advanced_tokens
        )

    is_basic = len(basic_tokens) > 0 and len(advanced_tokens) == 0
    is_advanced = len(advanced_tokens) == 1 and len(basic_tokens) == 0

    return tokens, is_basic, is_advanced


def validate_task(data):
    """Validate task/control parameters.

    Checks that jobtype is recognized and valid, title is non-empty,
    and ncpu values are positive. Also validates theory/reference methods.

    Parameters
    ----------
    data : dict
        Configuration dictionary to validate.

    Returns
    -------
    dict
        Validated configuration dictionary.

    Raises
    ------
    ValueError
        If jobtype is not recognized or methods are invalid.
    """
    # Parse and validate jobtype
    jobtype_str = data.get('jobtype', 'energy')
    tokens, is_basic, is_advanced = parse_jobtype(jobtype_str)
    data['jobtype'] = ' '.join(tokens)

    title = data.get('title', 'pyrai3md')
    if not title:
        data['title'] = 'pyrai3md'

    # Validate theory and reference methods
    data = validate_methods(data)

    # Composite selector for exactly two theory entries.
    multiscale = str(data.get('multiscale_type', '') or '').strip().lower()
    if multiscale not in ('', 'qmmm', 'oniom'):
        raise ValueError(
            "multiscale_type must be 'qmmm' or 'oniom'; got %r"
            % (data.get('multiscale_type'),))
    theory = data.get('theory', []) or []
    jobtype_tokens = str(data.get('jobtype', '') or '').strip().lower().split()
    adaptive = bool(jobtype_tokens) and jobtype_tokens[0] == 'adaptive'
    if multiscale and len(theory) not in (2, 3):
        raise ValueError(
            "multiscale_type=%r is only meaningful with two or three "
            "theory methods; got %d." % (multiscale, len(theory)))
    if len(theory) == 3 and multiscale == 'qmmm':
        raise ValueError(
            "three theory methods are always ONIOM3; "
            "multiscale_type='qmmm' is not allowed.")
    data['multiscale_type'] = multiscale

    # QMMM takes exactly two theory providers (QM first, MM last) and no
    # reference queue. It shares the ordinary jobtype set: the QMMM plan
    # spells each task's per-step request from the jobtype capabilities, so
    # optimize/md/surface_hopping and the numerical Hessian tokens reach
    # the same three stages as `energy gradient`. Adaptive sampling is the
    # exception: its reference queue is the labelling plan and may carry a
    # QMMM composite of its own (reference[0] QM, reference[1] MM).
    if multiscale == 'qmmm':
        if len(theory) != 2:
            raise ValueError(
                "multiscale_type='qmmm' needs exactly two theory methods "
                "(QM first, MM last); got %d." % len(theory))
        if data.get('reference') and not adaptive:
            raise ValueError(
                "multiscale_type='qmmm' takes both providers from "
                "control.theory; control.reference must be empty.")
        if adaptive and len(data.get('reference') or []) > 2:
            raise ValueError(
                "adaptive QMMM reference carries %d methods; the QMMM "
                "reference plan takes one QM provider and one MM provider."
                % len(data.get('reference') or []))

    return data


def _multiscale_kind(data):
    """Return 'qmmm', 'oniom', or None for the current theory queue."""
    theory = data.get('theory', []) or []
    multiscale = str(data.get('multiscale_type', '') or '').strip().lower()
    if len(theory) == 3:
        return 'oniom'
    if len(theory) == 2:
        return 'qmmm' if multiscale == 'qmmm' else 'oniom'
    return None


def _theory_is_classical(label):
    """Whether a theory label resolves to an mm-role-only skill.

    Used for the ONIOM2 + MM gate: an MM outer method is a classical
    low layer, not a quantum one.
    """
    from ..skills.query import role_of
    return role_of(label) == 'mm'


def is_standalone_queue(theory, reference, highlevel=None, midlevel=None,
                        highref=None, midref=None):
    """Whether this system has one electronic provider and no partition.

    `read_charge` means two different things by regime, and both callers have
    to agree on where the line is: under a multiscale queue it selects the
    ONIOM mid-shell charges, while here it selects the background field read
    from '{title}.charge'. One predicate, so the validation rule and the task
    queue cannot drift apart.

    The reference partition (highref/midref) counts as a partition: an
    adaptive run whose reference has its own layered plan must not be
    mistaken for a standalone read_charge run.
    """
    if _multiscale_kind({'theory': theory or [],
                         'reference': reference or []}) is not None:
        return False
    return (not (highlevel or midlevel or highref or midref)
            and len(reference or []) <= 1)


def _validate_system_shape(data):
    """Atom, coordinate, mass and per-atom-charge arrays agree in size.

    These are the arrays the whole pipeline indexes by atom, so a length
    that disagrees with `atoms` is refused here rather than met as an
    index error three layers down.
    """
    atoms = data.get('atoms', [])
    coordinates = data.get('coordinates', [])
    mass = data.get('mass', [])
    natom = data.get('natom', 0)

    # Check natom consistency
    if atoms and natom and len(atoms) != natom:
        raise ValueError(
            "natom (%d) does not match atoms list length (%d)"
            % (natom, len(atoms))
        )

    # Check coordinates dimension
    if coordinates and atoms:
        if len(coordinates) != len(atoms):
            raise ValueError(
                "coordinates length (%d) does not match atoms length (%d)"
                % (len(coordinates), len(atoms))
            )

    # Check mass dimension
    if mass and atoms:
        if len(mass) != len(atoms):
            raise ValueError(
                "mass length (%d) does not match atoms length (%d)"
                % (len(mass), len(atoms))
            )

    # Per-atom charges are whole-system: row i is atom i. A run that
    # writes them must size the array to the system so a region slice
    # is an index slice, never a guess.
    point_charges = data.get('point_charges') or []
    if point_charges:
        natom_ref = len(atoms) or int(natom or 0)
        if natom_ref and len(point_charges) != natom_ref:
            raise ValueError(
                "point_charges has %d rows but the system has %d atoms; "
                "the array is whole-system and row i is atom i."
                % (len(point_charges), natom_ref))
        for atom, row in enumerate(point_charges):
            if len(row) < 4:
                raise ValueError(
                    "point_charges row %d must be [q, x, y, z]; got %r."
                    % (atom + 1, row))
            if not np.all(np.isfinite(np.asarray(row[:4], dtype=float))):
                raise ValueError(
                    "point_charges row %d contains a non-finite value."
                    % (atom + 1))

    # Arbitrary external charges (legacy '.charge' rows) are a
    # standalone background field: they have no atom indices, so they
    # cannot be mixed with a multiscale partition and carry no
    # periodic images.
    external_charges = str(data.get('external_charges') or '').strip()
    if external_charges:
        if data.get('highlevel') or data.get('midlevel'):
            raise ValueError(
                "external_charges is for a standalone system with no "
                "highlevel/midlevel partition; use point_charges/embedding "
                "for ONIOM/QMMM charge embedding.")
        if any(int(flag) for flag in (data.get('pbc') or [])):
            raise ValueError(
                "external_charges positions are arbitrary and carry no "
                "periodic images; a periodic run is not supported.")
        # The file is read here on purpose (#7.6, reviewed and kept): the
        # read is a *refusal* of a field file the run would otherwise meet
        # at its first step, and the parse is the runtime's own
        # (`data.charges.read_charge_rows`), not a second grammar.
        from ..data.charges import ChargeStoreError, read_charge_rows
        try:
            read_charge_rows(external_charges)
        except FileNotFoundError:
            raise ValueError(
                "external_charges file '%s' was not found."
                % external_charges)
        except ChargeStoreError as exc:
            raise ValueError(str(exc))
        data['external_charges'] = external_charges


def _validate_embedding(data, multiscale_kind):
    """The embedding / read_charge switches, read by the run's regime.

    `embedding` decides charge embedding vs mechanical where a
    multiscale partition exists; `read_charge` names a background field
    file and is consulted only where a file can supply it. The normalized
    values are written back onto `data`.
    """
    # The two switches answer two questions, and each regime consults only
    # the one that has an answer there:
    #   gas       - `embedding` is a multiscale switch and is not read; the
    #               run is embedded exactly when read_charge names
    #               '{title}.charge', which then supplies the field itself.
    #   oniom     - `embedding` decides charge embedding vs mechanical;
    #               read_charge is consulted only when embedding is on, and
    #               then it replaces the computed mid-shell charges with the
    #               file's.
    #   qmmm      - `embedding` decides the internal V/E vs mechanical;
    #               read_charge is not consulted at all, because the MM stage
    #               supplies the field every step (a file would freeze it).
    embedding = data.get('embedding', False)
    read_charge = data.get('read_charge', False)
    # Written back by `_validate_system_shape`, which runs first.
    external_charges = str(data.get('external_charges') or '').strip()
    standalone = is_standalone_queue(data.get('theory'),
                                     data.get('reference'),
                                     data.get('highlevel'),
                                     data.get('midlevel'),
                                     data.get('highref'),
                                     data.get('midref'))
    if standalone and read_charge and external_charges:
        raise ValueError(
            "read_charge and external_charges both name a background "
            "embedding file ('{title}.charge' and '%s'); pick one."
            % external_charges)
    if not standalone:
        if multiscale_kind == 'qmmm' or not embedding:
            read_charge = False
    midlevel = data.get('midlevel', []) or []
    theory_methods = data.get('theory', []) or []
    highlevel = data.get('highlevel') or []
    single_center = not (isinstance(highlevel[0], (list, tuple))
                         if highlevel else False)
    classical_outer = (len(theory_methods) == 2
                       and _theory_is_classical(theory_methods[1]))
    if embedding and multiscale_kind == 'oniom' and not midlevel:
        if not highlevel:
            raise ValueError(
                "ONIOM embedding=true needs a highlevel region (and a "
                "midlevel for the three-layer form); both are empty.")
        if classical_outer:
            raise ValueError(
                "ONIOM2 with a classical (MM) outer method and "
                "embedding=true is not supported: use "
                "control.multiscale_type 'qmmm' with embedding=true, where "
                "the QM calculation receives the MM field V/E.")
        if not single_center and read_charge:
            raise ValueError(
                "ONIOM2 embedding=true with several centers cannot read one "
                "static charge file: each center's environment is the atoms "
                "outside its own highlevel, so the file rows would have no "
                "single binding. Use read_charge=false and let the "
                "full-system low-level charges fill every environment.")
        # Single- and multi-center ONIOM2 with a quantum outer method: each
        # high-level model is embedded in the outer region's charges. With
        # several centers that region is every atom outside the center's
        # own highlevel, including another center's high-level atoms; the
        # L(real) term supplies those rows (one evaluation, no cross-center
        # charge iteration).
    if (not embedding and multiscale_kind == 'oniom' and not midlevel
            and single_center and classical_outer):
        warnings.warn(
            "ONIOM2 with a classical (MM) outer method and embedding=false "
            "is the same additive combination as QMMM embedding=false; use "
            "control.multiscale_type 'qmmm' for that spelling.",
            UserWarning)

    data['embedding'] = embedding
    data['read_charge'] = read_charge


def _validate_qmmm_ownership(data, multiscale_kind):
    """A QMMM declaration owns its field and its region.

    The ESPF external field is injected by the planner, never by the
    ONIOM embedding keywords; the QM region is a proper, duplicate-free
    subset; a pinned charge set must address the whole QM set; and the
    charge iteration needs a field and more than one center to iterate
    over.
    """
    # QMMM ownership (dev/methods/QMMM_DEVELOPMENT.md section 2): the ESPF
    # external field is injected by the planner, never by the ONIOM
    # embedding keywords; the QM region is a proper, duplicate-free
    # subset. qmmm_key no longer selects QMMM by itself.
    if data.get('qmmm_key') and multiscale_kind != 'qmmm':
        raise ValueError(
            "system.qmmm_key is retired as a QMMM declaration: write "
            "control.multiscale_type qmmm with exactly two theory methods "
            "(QM first, MM last) and molecule.highlevel for the QM region.")
    if multiscale_kind == 'qmmm':
        highlevel = data.get('highlevel') or []
        if not highlevel:
            raise ValueError(
                "QMMM needs molecule.highlevel (the QM region); it is "
                "empty.")
        # One region per center, or one region total. The per-center shape
        # is checked by validate_centers, which owns the region rules
        # (non-empty, in range, duplicate-free, centers disjoint) and runs
        # right after this; what stays here is only what no other scheme
        # requires: the classical region must exist, and the charge set is
        # addressed to the whole QM set, never to one center.
        centers = highlevel if isinstance(highlevel[0], (list, tuple)) \
            else [highlevel]
        union = sorted({int(index) for group in centers for index in group})
        natom = int(data.get('natom', 0)
                    or len(data.get('atoms', []) or []))
        if natom and len(union) >= natom:
            raise ValueError(
                "QMMM highlevel covers the whole system; the MM region "
                "must be non-empty.")
        # The constraint block is accepted here exactly as it is for ONIOM:
        # constraints act on the whole system through their declared atoms,
        # and their energy/gradient corrections are applied by the framework
        # after assembly (workflow/constraints/runtime.py::finalize), never
        # by a skill or a composite term.
        q_qm = data.get('q_qm') or []
        if q_qm:
            if len(centers) > 1:
                # The classical stage of a multi-center run is fed by the
                # charges each center's own calculation reported; a pinned
                # set would answer for a field nobody built with it.
                raise ValueError(
                    "QMMM with %d centers does not read molecule.q_qm: every "
                    "center's 0-order charge set comes from its own ESPF fit, "
                    "and a charge the input pins would not be the charge the "
                    "field was built from." % len(centers))
            if len(q_qm) != len(union):
                raise ValueError(
                    "QMMM q_qm has %d charges for %d QM atoms; the order is "
                    "the whole QM set in global atom order (%d center(s))."
                    % (len(q_qm), len(union), len(centers)))
            if not np.all(np.isfinite(
                    np.asarray(q_qm, dtype=float))):
                raise ValueError(
                    "QMMM q_qm contains a non-finite value.")
        # The charge iteration answers one question: does each center's
        # field see converged charges from the others. It needs a field, and
        # it needs more than one center to iterate over.
        sc_charge = bool(data.get('sc_charge', False))
        # The two companions have model defaults, so an input that does not
        # name them is not an input that names a bad value: they are checked
        # only where the input spelled one out.
        raw_maxiter = data.get('sc_charge_maxiter')
        raw_threshold = data.get('sc_charge_threshold')
        if raw_threshold is not None:
            threshold = float(raw_threshold)
            if not np.isfinite(threshold) or threshold <= 0.0:
                raise ValueError(
                    "molecule.sc_charge_threshold must be a positive finite "
                    "charge difference; got %r." % (raw_threshold,))
        if raw_maxiter is not None:
            maxiter = int(raw_maxiter)
            if maxiter < 1:
                raise ValueError(
                    "molecule.sc_charge_maxiter counts charge rounds; it "
                    "must be at least 1, and one round is what sc_charge="
                    "false already means. Got %r." % (raw_maxiter,))
        if sc_charge:
            if len(centers) < 2:
                raise ValueError(
                    "molecule.sc_charge iterates the charges the other "
                    "centers contribute to this center's field; with one QM "
                    "region there is no such charge, and iterating would "
                    "solve the same embedded problem twice. Drop "
                    "sc_charge, or name the centers in molecule.highlevel.")
            if not data.get('embedding'):
                raise ValueError(
                    "molecule.sc_charge needs molecule.embedding: the charge "
                    "iteration exists because the other centers' charges "
                    "enter this center's field, and with embedding off the "
                    "QM calculation is isolated and sees no such field.")


def _validate_periodic_system(data):
    """pbc is the one periodicity authority, so it needs a real cell.

    Partial periodicity is refused in this stage: only full 3D periodic or
    non-periodic runs are supported until the low-dimensional solvers and
    the region/embedding model are defined (stage E).
    """
    # Periodic semantics (dev/methods/PBC_DEVELOPMENT.md stage A):
    # pbc is the single normalized periodicity authority, so enabling
    # it requires a valid cell; partial periodicity (mixed 0/1) is
    # rejected in this stage -- only full 3D periodic or non-periodic
    # runs are supported until the low-dimensional solvers and the
    # region/embedding model are defined (stage E).
    pbc = data.get('pbc') or []
    if any(int(p) for p in pbc):
        if sum(1 for p in pbc if int(p)) != 3:
            raise ValueError(
                "partial periodicity (pbc=%r) is not supported in this "
                "version: use full 3D periodicity [1, 1, 1] or disable "
                "pbc (dev/methods/PBC_DEVELOPMENT.md stage E)"
                % (pbc,))
        lattice = data.get('lattice_vector') or []
        flat_lattice = [float(x) for row in lattice for x in row] \
            if lattice and isinstance(lattice[0], (list, tuple)) else []
        if not any(flat_lattice):
            cell = data.get('cell') or []
            if ([float(x) for x in cell]
                    == [float(x) for x in DEFAULT['cell']]):
                # "Not filled" sentinel: PBC is on but no cell given.
                cell = []
            if len(cell) in (6, 9):
                lattice = parse_cell(cell)
            else:
                lattice = []
        validate_lattice(lattice, pbc)
        _reject_unsupported_pbc_backends(data)


def validate_system(data):
    """Validate system parameters for consistency.

    Checks that the number of atoms matches the coordinate dimensions,
    that mass arrays are consistent, and that the ONIOM electrostatic
    embedding keywords (embedding / read_charge) are mutually and
    structurally consistent.

    Parameters
    ----------
    data : dict
        Configuration dictionary to validate.

    Returns
    -------
    dict
        Validated configuration dictionary.

    Raises
    ------
    ValueError
        If atom/coordinate dimensions are inconsistent, or the
        embedding / read_charge keywords are mis-specified.
    """
    multiscale_kind = _multiscale_kind(data)
    _validate_system_shape(data)
    _validate_embedding(data, multiscale_kind)
    _validate_qmmm_ownership(data, multiscale_kind)
    _validate_periodic_system(data)
    return data

    return data


def _reject_unsupported_pbc_backends(data):
    """Reject a periodic run whose OUTERMOST backend lacks PBC support.

    Early, friendly capability gate (dev/methods/PBC_DEVELOPMENT.md
    section 5.4): every skill declares supports_pbc in its manifest,
    surfaced through query.get_metadata -> skills.query.discover_all.
    When pbc is enabled, the method that owns the whole-system geometry
    must be a periodic backend; a method that only ever serves a center
    region is a molecular calculation on a reconstructed cluster and is
    allowed to say `supports_pbc: false`.

    The granularity is the 2026-10-25 ruling on this design: the PBC gate
    follows the theory/reference lengths. With more than one theory method
    the last theory method is the environment/whole-system provider; a
    single reference then labels the first theory method's centre region
    and need not support PBC. With one theory method and any reference, both
    the theory tail and the reference tail own the box. More than one
    reference beside a layered theory queue is refused earlier by
    validate_methods(). `planning/task.py::resolve_system_task` remains the
    runtime authority; this gate is the configuration-time approximation.

    Labels that do not resolve to a registered skill are left alone (the
    per-skill validation is the authoritative backstop); a failed registry
    discovery never blocks.

    Parameters
    ----------
    data : dict
        Flattened configuration with normalized theory/reference.

    Raises
    ------
    ValueError
        When the whole-system provider does not support periodic boundaries.
    """
    pbc = [int(flag) for flag in (data.get('pbc') or [])]
    if not any(pbc):
        return

    jobtype = str(data.get('jobtype', '') or '').strip().lower().split()

    def _methods(value):
        labels = [value] if isinstance(value, str) else list(value or [])
        labels = [_strip_skill_prefix(str(label)) for label in labels]
        return [label for label in labels if label]

    theory = _methods(data.get('theory'))
    reference = _methods(data.get('reference'))

    # The PBC gate follows the method combination, not a universal tail:
    #
    #   theory=1, reference=1  -> both methods own the box
    #   theory>1, reference=1  -> theory tail (environment) owns the box;
    #                             the one reference labels theory[0]/the
    #                             centre region and may be molecular
    #   theory=1, reference>1  -> theory + reference tail own the box
    #   theory>1, reference>1  -> non-adaptive is refused by
    #                             validate_methods(); adaptive allows the
    #                             independent reference plan, whose low
    #                             layer owns the box when it exists
    #
    # A reference-only run keeps the old reference-tail check; an
    # agent/skill-driver run with no theory queue falls back to skill_name.
    adaptive = bool(jobtype) and jobtype[0] == 'adaptive'
    labels = []
    if adaptive and theory:
        # The theory tail always carries the box. The reference tail does
        # so only when its own plan reaches the whole system: a single
        # reference layer beside a single theory method covers every atom,
        # and a layered reference always owns the remaining low region. A
        # single reference layer beside a layered theory queue covers only
        # the neural-network region and may be molecular.
        labels.append(theory[-1])
        if reference and (len(theory) == 1 or len(reference) > 1):
            labels.append(reference[-1])
    elif theory:
        labels.append(theory[-1])
        if len(theory) == 1:
            labels.extend(reference[-1:])
    else:
        labels.extend(reference[-1:])
        skill_name = data.get('skill_name')
        if not reference and skill_name:
            label = _strip_skill_prefix(str(skill_name))
            if label:
                labels.append(label)
    labels = list(dict.fromkeys(labels))
    if not labels:
        return

    try:
        from ..skills.query import discover_all
        registry = discover_all()
    except Exception as error:
        # Without the registry the PBC-support gate cannot run. Skipping
        # it silently would accept a periodic run on a molecular backend;
        # make the skip visible instead.
        warnings.warn(
            "skill discovery failed (%s: %s); the PBC backend gate is "
            "skipped for skill_name %r"
            % (type(error).__name__, error, labels),
            UserWarning, stacklevel=2)
        return
    if not registry:
        return

    for label in labels:
        meta = registry.get(label)
        if meta is None:
            continue
        if meta.get('supports_pbc') is False:
            raise ValueError(
                "the whole-system method '%s' does not support periodic "
                "boundary conditions but pbc is enabled; the outermost layer "
                "must carry the box (cp2k / xtb / lammps). A molecular "
                "backend is still allowed, but only as a center region of a "
                "multi-center run." % label)


def validate_electronic(data):
    """Validate electronic settings for consistency.

    Checks that charge and multiplicity are physically reasonable,
    and that total_electron is non-negative. For a multi-center run
    charges/multiplicity may be per-center lists: each center's
    multiplicity must be >= 1 and the global check uses the
    full-system aggregate (sum of charges, paired multiplicity) that
    normalize derived total_electron from (docs MULTICENTER 8).

    Parameters
    ----------
    data : dict
        Configuration dictionary to validate.

    Returns
    -------
    dict
        Validated configuration dictionary.

    Raises
    ------
    ValueError
        If electronic settings are physically unreasonable.
    """
    from ..data.center import aggregate_system_multiplicity

    multiplicity = data.get('multiplicity', 1)
    if isinstance(multiplicity, (list, tuple)):
        for c, m in enumerate(multiplicity):
            if int(m) < 1:
                raise ValueError(
                    "molecule.multiplicity[%d]: multiplicity must be "
                    ">= 1, got %d" % (c, int(m)))
        agg_multiplicity = aggregate_system_multiplicity(multiplicity)
    else:
        if multiplicity < 1:
            raise ValueError(
                "multiplicity must be >= 1, got %s" % multiplicity)
        agg_multiplicity = multiplicity

    total_electron = data.get('total_electron', 0)
    if total_electron < 0:
        raise ValueError(
            "total_electron is negative (%d). Check charges." % total_electron
        )

    # Check multiplicity consistency with electron count
    if total_electron > 0:
        unpaired = agg_multiplicity - 1
        if unpaired > total_electron:
            raise ValueError(
                "multiplicity (%d) implies %d unpaired electrons, "
                "but total_electron is only %d"
                % (agg_multiplicity, unpaired, total_electron)
            )

    return data


def validate_centers(data):
    """Validate the unified center partition plan (docs section 5).

    Builds the canonical center plan (data.center.build_center_plan),
    which enforces the per-center electronic structure strictly: ci /
    spin length match, non-negative state counts, coupling pairs of two
    distinct in-range local states with no duplicate unordered pair,
    and root within [0, nstate[c]). These checks run for C=1 too.

    The normalized `nstate` is the axis an optimize run's
    `optimize.state_i`/`state_j` indices select from (the plan's
    `sum(ci)` is the fallback when the flat key is absent), so a
    single-center run refuses an index outside [0, nstate) here, at input,
    instead of letting it reach run construction.

    On top of the plan, this validates the geometry region rules that
    were previously only checked inside multi-center planning (docs
    5.2, moved forward to schema): a defined high-level region must be
    non-empty, in range and duplicate-free; for C>1 the high-level
    regions must not overlap; for a three-layer plan the mid region
    must be non-empty, in range, non-overlapping and a superset of the
    high region. The upper-bound index check is skipped until the
    geometry is loaded (natom > 0), but always runs before any backend
    submission.

    Parameters
    ----------
    data : dict
        Flattened, normalized configuration dictionary.

    Returns
    -------
    dict
        The unchanged configuration dictionary.

    Raises
    ------
    ValueError
        On any structural, electronic or region inconsistency; the
        message carries the full field path plus the human (1-based)
        and internal (0-based) center ordinals.
    """
    from ..data.center import build_center_plan, CenterPlanError

    try:
        plan = build_center_plan(data)
    except CenterPlanError as exc:
        raise ValueError(str(exc))

    n_center = plan.n_center

    # An optimize run selects its surface by index into the state axis the
    # run is built with, so the range rule lives here rather than in the
    # workflow that reads the index: an out-of-range surface would otherwise
    # reach run construction before anything refused it. Multi-center runs
    # are skipped - `opt_state_i` names no center, and the authoritative
    # active state there is per center.
    if n_center == 1:
        nstate = int(data.get('nstate') or plan.centers[0].nstate)
        for key, name in (('opt_state_i', 'state_i'),
                          ('opt_state_j', 'state_j')):
            value = data.get(key)
            if value is None:
                continue
            # A list is the per-center spelling; a single-center run may still
            # carry a one-entry list (the broadcast form), and every entry has
            # to answer to this ladder. Scalars keep the original path.
            indices = ([int(item) for item in value]
                       if isinstance(value, (list, tuple)) else [int(value)])
            for index in indices:
                if not 0 <= index < nstate:
                    raise ValueError(
                        "optimize.%s=%d is out of range for nstate=%d (state "
                        "indices are 0-based)" % (name, index, nstate))

    natom = int(data.get('natom', 0) or len(data.get('atoms', []) or []))
    highs = [list(c.high_indices) for c in plan.centers]
    mids = [list(c.mid_indices) for c in plan.centers]
    has_region = any(highs)

    if has_region:
        for c in range(n_center):
            if not highs[c]:
                raise ValueError(
                    "molecule.highlevel[%d]: center %d (c=%d) has an "
                    "empty high-level region." % (c, c + 1, c))
            if len(highs[c]) != len(set(highs[c])):
                raise ValueError(
                    "molecule.highlevel[%d]: center %d (c=%d) has "
                    "duplicate atom indices." % (c, c + 1, c))
            if natom > 0:
                for idx in highs[c]:
                    if idx < 0 or idx >= natom:
                        raise ValueError(
                            "molecule.highlevel[%d]: center %d (c=%d) "
                            "atom index %d out of range [0, %d)."
                            % (c, c + 1, c, idx, natom))

        if n_center > 1:
            for i in range(n_center):
                for j in range(i + 1, n_center):
                    overlap = set(highs[i]) & set(highs[j])
                    if overlap:
                        raise ValueError(
                            "molecule.highlevel: centers %d and %d "
                            "(c=%d, c=%d) overlap at atoms %s."
                            % (i + 1, j + 1, i, j, sorted(overlap)))

        if plan.n_layer == 3:
            for c in range(n_center):
                if not mids[c]:
                    raise ValueError(
                        "molecule.midlevel[%d]: center %d (c=%d) has an "
                        "empty mid-level region (three-layer plan)."
                        % (c, c + 1, c))
                if len(mids[c]) != len(set(mids[c])):
                    raise ValueError(
                        "molecule.midlevel[%d]: center %d (c=%d) has "
                        "duplicate atom indices." % (c, c + 1, c))
                if natom > 0:
                    for idx in mids[c]:
                        if idx < 0 or idx >= natom:
                            raise ValueError(
                                "molecule.midlevel[%d]: center %d (c=%d) "
                                "atom index %d out of range [0, %d)."
                                % (c, c + 1, c, idx, natom))
                overlap = set(highs[c]) & set(mids[c])
                if overlap:
                    raise ValueError(
                        "molecule.highlevel[%d]/midlevel[%d]: center %d "
                        "(c=%d) highlevel and midlevel must be disjoint; "
                        "they overlap at atoms %s."
                        % (c, c, c + 1, c, sorted(overlap)))
            if n_center > 1:
                for i in range(n_center):
                    for j in range(i + 1, n_center):
                        overlap = set(mids[i]) & set(mids[j])
                        if overlap:
                            raise ValueError(
                                "molecule.midlevel: centers %d and %d "
                                "(c=%d, c=%d) overlap at atoms %s."
                                % (i + 1, j + 1, i, j, sorted(overlap)))

    # Capability refusals for C>1 combinations that have no defined
    # multi-center model yet (docs 5.2, section 8). These must fail
    # before planning falls back into a single-center branch and
    # silently changes the physical model.
    if n_center > 1:
        if data.get('center_groups'):
            raise ValueError(
                "multi-center (C=%d) does not support fragment "
                "center_groups in this version." % n_center)
        if data.get('qmmm_key'):
            raise ValueError(
                "multi-center (C=%d) does not support the legacy qmmm "
                "scheme in this version." % n_center)
        # Periodic multi-center runs are allowed since the 2026-10-08
        # ruling: the requirement is on the outermost layer, which
        # validate_system's capability gate reads from the theory tail
        # (`_reject_unsupported_pbc_backends`), not on the centers.

    return data


def _reference_region(value, field_name, adaptive):
    """Return one flat 0-based region from highref/midref.

    The reference partition is a single-centre plan in this build: a
    nested (multi-centre) spelling is refused under adaptive sampling
    rather than silently flattened, because a flattened union would
    change which atoms the reference layer actually covers.
    """
    if not value:
        return []
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        if adaptive:
            raise ValueError(
                "molecule.%s carries multiple centers; the adaptive "
                "reference partition supports one center." % field_name)
        return sorted({int(index) for group in value for index in group})
    return sorted(int(index) for index in value)


def validate_reference_partition(data):
    """Validate the independent reference partition (highref/midref).

    The reference method list is its own layered plan: reference[0] is
    the high layer (``highref``), reference[1] of a three-layer spelling
    the mid layer (``midref``), and the last entry the low layer over
    the remaining atoms. This validator only checks the partition's
    structure; which layer (or the composite total) actually labels the
    neural network is derived later from the atom-set correspondence
    with the theory highlevel.

    Rules:
    1. highref/midref are single-center, duplicate-free and in range.
    2. highref and midref are disjoint.
    3. More than one reference layer needs an effective high region and
       must leave at least one atom for the low layer.
    4. A three-layer reference needs an effective mid region.
    5. A single reference layer, when highref is written explicitly,
       must cover exactly the neural-network surface; otherwise no
       candidate can label it.

    Outside adaptive sampling the fields are inert and pass through
    unchanged.
    """
    jobtype = str(data.get('jobtype', '') or '').strip().lower().split()
    if not jobtype or jobtype[0] != 'adaptive':
        return data

    theory = list(data.get('theory') or [])
    reference = list(data.get('reference') or [])
    if not reference:
        return data

    natom = int(data.get('natom', 0) or len(data.get('atoms') or []))
    highref = _reference_region(data.get('highref'), 'highref', True)
    midref = _reference_region(data.get('midref'), 'midref', True)
    highlevel = _reference_region(data.get('highlevel'), 'highlevel', False)
    midlevel = _reference_region(data.get('midlevel'), 'midlevel', False)

    for name, indices in (('highref', highref), ('midref', midref)):
        if len(indices) != len(set(indices)):
            raise ValueError(
                "molecule.%s has duplicate atom indices." % name)
        if natom:
            for index in indices:
                if index < 0 or index >= natom:
                    raise ValueError(
                        "molecule.%s atom index %d out of range [0, %d)."
                        % (name, index, natom))

    overlap = set(highref) & set(midref)
    if overlap:
        raise ValueError(
            "molecule.highref and midref must be disjoint; they overlap "
            "at atoms %s." % sorted(overlap))

    if len(theory) <= 1:
        nn_surface = set(range(natom)) if natom else set()
    else:
        nn_surface = set(highlevel)

    if len(reference) == 1:
        if highref and nn_surface and set(highref) != nn_surface:
            raise ValueError(
                "molecule.highref covers %d atoms but the neural-network "
                "surface covers %d (theory has %d method(s)); a single "
                "reference layer must cover the same atoms."
                % (len(highref), len(nn_surface), len(theory)))
        return data

    # Layered reference: its high/mid regions define the QM layers; the
    # remaining atoms form the low layer.
    effective_high = highref or (highlevel if len(theory) > 1 else [])
    if not effective_high:
        raise ValueError(
            "reference carries %d layers but molecule.highref is empty; "
            "write the reference high region or (with a layered theory "
            "queue) rely on the highlevel inheritance." % len(reference))

    effective_mid = []
    if len(reference) == 3:
        effective_mid = midref or (midlevel if len(theory) > 2 else [])
        if not effective_mid:
            raise ValueError(
                "reference carries three layers but molecule.midref is "
                "empty; write the reference mid region or (with an ONIOM3 "
                "theory queue) rely on the midlevel inheritance.")

    union = set(effective_high) | set(effective_mid)
    if natom and union >= set(range(natom)):
        raise ValueError(
            "the reference high/mid regions cover the whole system; the "
            "low layer of a layered reference must be non-empty.")
    return data


def _strip_skill_prefix(name):
    """Return the skill label of one method spelling.

    Kept as a name here because this module's tables are built from the
    stripped label; the spelling rule itself lives in
    `skills.query.method_label`, which `normalize` also uses to rewrite
    the queues once.
    """
    from ..skills.query import method_label
    return method_label(name)


def validate_methods(data):
    """Validate theory and reference method lists.

    Rules:
    1. Each method in theory must be a known method (built-in methods
       plus the labels registered by the installed skills, resolved
       via _valid_theory_methods()).
       Accepts both short names ('pyscf') and prefixed format
       ('qm.pyscf') -- the prefix is stripped automatically.
    2. Each method in reference must be a known reference method (no nn).
    3. reference does not support 'nn' -- it chooses the reference QC
       electronic property calculator for recomputing QC-data during
       adaptive sampling.
    4. Empty theory/reference lists are allowed (agent mode uses skill_name).

    Parameters
    ----------
    data : dict
        Configuration dictionary to validate.

    Returns
    -------
    dict
        Validated configuration dictionary. The theory and reference lists
        are checked under their short names (the `qm.`/`abinit.` prefix is
        tolerated); rewriting them is `normalize`'s job.

    Raises
    ------
    ValueError
        If any method name is unrecognized or reference contains 'nn'.
    """
    theory = data.get('theory', [])
    reference = data.get('reference', [])

    if isinstance(theory, str):
        theory = [theory]
    if isinstance(reference, str):
        reference = [reference]

    jobtype = str(data.get('jobtype', '') or '').strip().lower().split()
    adaptive = bool(jobtype) and jobtype[0] == 'adaptive'

    # Check under the short names; `normalize` rewrites the queues once,
    # so this pass stays read-only (the rebuilt Config is normalize's).
    theory = [_strip_skill_prefix(m) for m in theory]
    reference = [_strip_skill_prefix(m) for m in reference]

    valid_theory = _valid_theory_methods()
    valid_reference = _valid_reference_methods()

    for method in theory:
        if method not in valid_theory:
            raise ValueError(
                "Unknown theory method '%s'. Available: %s"
                % (method, sorted(valid_theory))
            )

    for method in reference:
        if method == 'nn':
            raise ValueError(
                "reference does not support 'nn'. "
                "reference chooses the reference QC electronic property "
                "calculator for recomputing QC-data during adaptive sampling. "
                "Available options are the same as theory except for nn."
            )
        if method not in valid_reference:
            raise ValueError(
                "Unknown reference method '%s'. Available: %s"
                % (method, sorted(valid_reference))
            )

    if adaptive and len(reference) > 3:
        raise ValueError(
            "adaptive reference carries %d methods; at most three layers "
            "(reference[0] high, reference[1] mid, last low) are defined."
            % len(reference))

    # Composite roles have one reference calculator at most: more than one
    # reference beside a layered theory queue has no task it could label.
    # Adaptive sampling is the exception: its reference list is an
    # independent layered plan whose regions are defined by highref/midref,
    # and the label source is derived from the region correspondence rather
    # than from list positions.
    if len(theory) > 1 and len(reference) > 1 and not adaptive:
        raise ValueError(
            "theory carries %d methods and reference carries %d; a layered "
            "theory queue accepts at most one reference calculator. Use "
            "theory=[high, low] with reference=[labeller]."
            % (len(theory), len(reference)))

    return data


def validate_skill_name(data):
    """Validate skill_name against the skill registry.

    If skill_name is set, verifies that it corresponds to a known
    skill by checking against the query mechanism.

    Parameters
    ----------
    data : dict
        Configuration dictionary to validate.

    Returns
    -------
    dict
        Validated configuration dictionary.

    Raises
    ------
    ValueError
        If skill_name is set but not found in the registry.
    """
    skill_name = data.get('skill_name', '')
    if not skill_name:
        return data

    from ..skills.query import discover_all
    registry_data = discover_all()
    known_names = {v['skill_name'] for v in registry_data.values()}

    if skill_name not in known_names:
        raise ValueError(
            "skill_name '%s' is not available. "
            "Known skills: %s. "
            "Please check that the skill are installed."
            % (skill_name, sorted(known_names))
        )

    return data


def validate_dynamics(data):
    """Validate dynamics parameters for consistency.

    Checks that step, size, and temperature are physically reasonable.

    Parameters
    ----------
    data : dict
        Configuration dictionary to validate.

    Returns
    -------
    dict
        Validated configuration dictionary.

    Raises
    ------
    ValueError
        If dynamics parameters are invalid.
    """
    # Every value below is a settled submodel field (validate() hands
    # this function the dump of a typed Config), so the class guards
    # that stood here were always true; only root keeps one because the
    # field is Union[int, List[int]] (a per-center list).
    step = data.get('step', 10)
    if step < 0:
        raise ValueError("step must be >= 0, got %s" % step)

    size = data.get('size', 20.67)
    if size <= 0:
        raise ValueError("size (timestep) must be > 0, got %s" % size)

    temperature = data.get('temperature', 300.0)
    if temperature < 0:
        raise ValueError("temperature must be >= 0, got %s" % temperature)

    root = data.get('root', 0)
    if root is not None:
        # root is Union[int, List[int]]: a multi-center run carries one
        # initial state per center.
        roots = root if isinstance(root, (list, tuple)) else [root]
        for value in roots:
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    "root (initial state) is 1-based in the input; expected "
                    "root >= 1, got %r" % (value,))

    # multipool is Field(ge=1) on every submodel that carries it, so an
    # out-of-range value never reaches this stage.

    # The initial-velocity control spells one request three ways: what the
    # first step's kinetic energy should be. The archive chained them on one
    # factor so the last written keyword silently overrode the earlier ones;
    # a run that asks for two has to say which it meant. The test is on the
    # value rather than the key because validate() sees the model dump, where
    # every field is present whether the input wrote it or not.
    requested = [key for key, inert in (('excess', 0.0), ('scale', 1.0),
                                        ('target', 0.0))
                 if float(data.get(key, inert) or inert) != inert]
    if len(requested) > 1:
        raise ValueError(
            "&md %s are three spellings of one request (the kinetic energy of "
            "the first step) and all set it: write one of them. excess is the "
            "energy to add, scale the factor to multiply by, target the energy "
            "to reach, both energies in Hartree"
            % ' and '.join('`%s`' % key for key in requested))

    data = validate_ensemble(data)

    return data


def validate_ensemble(data):
    """Validate the NVE/NVT/NPT ensemble control keywords.

    Checks the ensemble / thermostat / barostat vocabularies and their
    combination rules (dev/methods/ENSEMBLE_DEVELOPMENT.md section 3). The
    checks are case-insensitive and alias-tolerant. Because validate()
    restores the dynamics submodel from the original dump, this
    function only raises on inconsistency; it does not rely on writing
    normalized values back (the runtime reads them case-insensitively).

    Rules:
    - ensemble in {nve, nvt, npt}; thermostat / barostat / barostat_mode
      in their vocabularies.
    - nve: thermostat must be off (the barostat default is inert).
    - nvt: a thermostat is required (the barostat is ignored).
    - npt: a barostat AND a thermostat are required; MTTK may only be
      combined with a Nose-Hoover / Nose-Hoover-chain thermostat.
    - coupling constants and compressibility must be positive.

    Parameters
    ----------
    data : dict
        Flattened configuration dictionary.

    Returns
    -------
    dict
        The unchanged configuration dictionary.

    Raises
    ------
    ValueError
        On an unknown token or an invalid ensemble combination.
    """
    ensemble = str(data.get('ensemble', 'nve') or 'nve').strip().lower()
    thermostat = str(data.get('thermostat', 'off') or 'off').strip().lower()
    barostat = str(data.get('barostat', 'off') or 'off').strip().lower()
    thermostat = THERMOSTAT_ALIASES.get(thermostat, thermostat)
    barostat = BAROSTAT_ALIASES.get(barostat, barostat)
    mode = str(data.get('barostat_mode', 'isotropic')
               or 'isotropic').strip().lower()

    if ensemble not in VALID_ENSEMBLES:
        raise ValueError(
            "Unknown ensemble '%s'. Options: %s"
            % (ensemble, sorted(VALID_ENSEMBLES)))
    if thermostat not in VALID_THERMOSTATS:
        raise ValueError(
            "Unknown thermostat '%s'. Options: %s"
            % (thermostat, sorted(VALID_THERMOSTATS)))
    if barostat not in VALID_BAROSTATS:
        raise ValueError(
            "Unknown barostat '%s'. Options: %s"
            % (barostat, sorted(VALID_BAROSTATS)))
    if mode not in VALID_BAROSTAT_MODES:
        raise ValueError(
            "Unknown barostat_mode '%s'. Options: %s"
            % (mode, sorted(VALID_BAROSTAT_MODES)))

    if ensemble == 'nve':
        if thermostat != 'off':
            raise ValueError(
                "ensemble 'nve' must not set a thermostat "
                "(thermostat='%s'); use 'nvt' or 'npt'." % thermostat)
        if str(data.get('dissipate', '0')).strip() not in ('', '0'):
            # The delay is a statement about when the thermostat starts; an
            # NVE run has no thermostat to start, so accepting the keyword
            # here would let a run believe it was being dissipated.
            raise ValueError(
                "&md dissipate delays the thermostat of an 'nvt'/'npt' run; "
                "ensemble 'nve' has no thermostat to delay. Set "
                "ensemble: nvt (or npt) with a thermostat, or drop the "
                "keyword.")
    elif ensemble == 'nvt':
        if thermostat == 'off':
            raise ValueError(
                "ensemble 'nvt' requires a thermostat "
                "(thermostat='off' is the NVE choice).")
    elif ensemble == 'npt':
        if barostat == 'off':
            raise ValueError(
                "ensemble 'npt' requires a barostat (barostat='off').")
        if thermostat == 'off':
            raise ValueError(
                "ensemble 'npt' requires a thermostat (thermostat='off').")
        allowed = BAROSTAT_THERMOSTATS.get(barostat)
        if allowed is not None and thermostat not in allowed:
            raise ValueError(
                "barostat '%s' cannot be combined with thermostat '%s': %s. "
                "Thermostats this barostat accepts: %s."
                % (barostat, thermostat,
                   BAROSTAT_PAIRING_REASON[barostat], sorted(allowed)))

    # scr's tensor form is refused on a liquid, and the reason is measured
    # rather than assumed (see docs section 8.13): its noise shears the cell
    # by ~1.5e-3 per 1 fs step, and with no shear modulus to store that as
    # elastic energy a liquid dissipates it into heat. On the 60-atom water
    # box the temperature went 300 K -> 897 K within 200 steps and the cell
    # then collapsed; ten-fold tau_p slowed it by about the factor tau_p
    # predicts instead of removing it, and the conservative toy surrogate
    # cannot show the effect at all because an analytic stress has no
    # dissipation channel. The scalar form on ln V is unaffected: bulk
    # compression is reversible, and that branch measured a temperature
    # residue of 0.141 on the same box.
    if barostat == 'scr' and mode == 'anisotropic':
        raise ValueError(
            "barostat 'scr' supports the isotropic cell only: its tensor "
            "form shears the box with no restoring modulus, and that shear "
            "thermalises a liquid (measured 300 K -> 897 K in 200 steps on "
            "the 60-atom water box, then a cell collapse). Use "
            "barostat_mode: isotropic, or parrinello-rahman/mttk for a "
            "tilted cell.")

    for key in ('thermostat_tau', 'barostat_tau'):
        value = data.get(key)
        if value <= 0:
            raise ValueError("%s must be > 0, got %s" % (key, value))
    compressibility = data.get('compressibility')
    if compressibility <= 0:
        raise ValueError(
            "compressibility must be > 0, got %s" % compressibility)

    return data


def validate_adaptive(data):
    """Validate adaptive sampling parameters for consistency.

    Checks that threshold ranges are logical (max >= min),
    and refinement parameters are non-negative.

    Parameters
    ----------
    data : dict
        Configuration dictionary to validate.

    Returns
    -------
    dict
        Validated configuration dictionary.

    Raises
    ------
    ValueError
        If adaptive sampling parameters are inconsistent.
    """
    # Check threshold ranges: max >= min for each property
    for prop in ('energy', 'grad', 'nac', 'soc'):
        max_val = data.get('max' + prop, 0.0)
        min_val = data.get('min' + prop, 0.0)
        if max_val < min_val:
            raise ValueError(
                "max%s (%s) must be >= min%s (%s)"
                % (prop, max_val, prop, min_val)
            )

    # Check refinement parameters
    refine = data.get('refine', 0)
    refine_num = data.get('refine_num', 0)
    if refine < 0:
        raise ValueError("refine must be >= 0, got %s" % refine)
    if refine_num < 0:
        raise ValueError("refine_num must be >= 0, got %s" % refine_num)

    # Check maxiter and maxsample
    maxiter = data.get('maxiter', 1)
    maxsample = data.get('maxsample', 1)
    if maxiter is not None and (not isinstance(maxiter, int) or maxiter < 0):
        raise ValueError("maxiter must be >= 0, got %s" % maxiter)
    if maxsample is not None and (not isinstance(maxsample, int) or maxsample < 0):
        raise ValueError("maxsample must be >= 0, got %s" % maxsample)

    # Screening window and physical checks
    record_step = data.get('record_step', 10)
    if record_step is not None and (not isinstance(record_step, int) or record_step < 0):
        raise ValueError("record_step must be >= 0, got %s" % record_step)
    keep_md = data.get('keep_md', 1)
    if keep_md is not None and (not isinstance(keep_md, int) or keep_md not in (0, 1)):
        raise ValueError("keep_md must be 0 or 1, got %s" % keep_md)
    for key in ('check_population', 'check_energy_order'):
        value = data.get(key, 1)
        if value is not None and (not isinstance(value, int) or value not in (0, 1)):
            raise ValueError("%s must be 0 or 1, got %s" % (key, value))
    for key in ('sample_energy', 'sample_grad', 'sample_nac', 'sample_soc',
                'sample_dipole', 'sample_transition_dipole',
                'sample_oscillator_strength'):
        value = data.get(key, 0)
        if value is not None and (not isinstance(value, int) or value not in (0, 1)):
            raise ValueError("%s must be 0 or 1, got %s" % (key, value))
    if data.get('sample_energy', 1) == 0 or data.get('sample_grad', 1) == 0:
        raise ValueError(
            "adaptive sampling trains on energy and gradient; "
            "sample_energy and sample_grad must be 1")
    for key in ('gap_tolerance',):
        value = data.get(key, 0.0)
        if value is not None and (not isinstance(value, (int, float)) or value < 0):
            raise ValueError("%s must be >= 0, got %s" % (key, value))
    maxmae = data.get('maxmae', 0.0)
    if maxmae is not None and (
            isinstance(maxmae, bool)
            or not isinstance(maxmae, (int, float)) or maxmae < 0):
        raise ValueError(
            "maxmae must be a validation energy MAE ceiling in eV >= 0, "
            "got %s" % (maxmae,))

    # Analysis layer switches (dev/methods/ADAPTIVE_DEVELOPMENT.md section 16).
    for key in ('analyze_pes', 'analyze_trace', 'analyze_coverage'):
        value = data.get(key, 0)
        if value is not None and (not isinstance(value, int) or value not in (0, 1)):
            raise ValueError("%s must be 0 or 1, got %s" % (key, value))
    gap_scope = str(data.get('analyze_gap', 'off') or 'off').strip().lower()
    if gap_scope not in ('off', 's0s1', 'adjacent', 'all'):
        raise ValueError(
            "analyze_gap must be off, s0s1, adjacent or all, got %r"
            % data.get('analyze_gap'))
    window = data.get('analyze_window', 3)
    if window is not None and (not isinstance(window, int) or window < 1):
        raise ValueError("analyze_window must be >= 1, got %s" % window)
    for key in ('analyze_local_tolerance', 'analyze_global_tolerance'):
        value = data.get(key, 0.0)
        if value is not None and (not isinstance(value, (int, float)) or value < 0):
            raise ValueError("%s must be >= 0, got %s" % (key, value))
    predict_mode = data.get('predict_mode', 0)
    if predict_mode is not None and (not isinstance(predict_mode, int) or predict_mode < 0):
        raise ValueError("predict_mode must be >= 0, got %s" % predict_mode)
    gap_threshold = data.get('gap_threshold', 0.03)
    if gap_threshold is not None and (not isinstance(gap_threshold, (int, float)) or gap_threshold < 0):
        raise ValueError("gap_threshold must be >= 0, got %s" % gap_threshold)
    for key in ('explore_udd', 'explore_gap'):
        value = data.get(key, 0)
        if value is not None and (not isinstance(value, int) or value not in (0, 1)):
            raise ValueError("%s must be 0 or 1, got %s" % (key, value))
    udd_mode = str(data.get('udd_mode', 'force') or 'force').lower()
    if udd_mode not in ('force', 'potential'):
        raise ValueError(
            "udd_mode must be force or potential, got %r"
            % data.get('udd_mode'))
    for key in ('udd_amplitude', 'udd_width'):
        value = data.get(key, 0.0)
        if value is not None and (not isinstance(value, (int, float)) or value < 0):
            raise ValueError("%s must be >= 0, got %s" % (key, value))
    start = data.get('udd_start_iteration', 1)
    if start is not None and (not isinstance(start, int) or start < 1):
        raise ValueError(
            "udd_start_iteration must be >= 1, got %s" % start)
    gap_method = str(data.get('gap_method', 'cv') or 'cv').lower()
    if gap_method not in ('cv', 'paper'):
        raise ValueError(
            "gap_method must be cv or paper, got %r" % data.get('gap_method'))
    gap_pair = data.get('gap_pair') or []
    if isinstance(gap_pair, (list, tuple)) and len(gap_pair) not in (0, 2):
        raise ValueError(
            "gap_pair must name zero or two states, got %r" % (gap_pair,))
    for key in ('gap_height', 'gap_width'):
        value = data.get(key, 0.0)
        if value is not None and (not isinstance(value, (int, float)) or value < 0):
            raise ValueError("%s must be >= 0, got %s" % (key, value))

    # Per-stage fan-out pools
    for key in ('md_multipool', 'reference_multipool'):
        value = data.get(key)
        if value is not None and (not isinstance(value, int) or value < 1):
            raise ValueError("%s must be >= 1, got %s" % (key, value))

    return data


def validate_dispatch_sections(nested_data):
    """Validate the dispatch-layer keywords of the nested sections.

    The workflow sections sharing the dispatch keyword names
    (multitask/remotetask/multipool) travel nested through the
    pipeline, so they are validated from the model dump instead of
    the flattened namespace.

    Parameters
    ----------
    nested_data : dict
        Config model dump (submodel name -> section dict).

    Returns
    -------
    dict
        The unchanged nested data.

    Raises
    ------
    ValueError
        If a section carries an invalid multipool.
    """
    for submodel_name in NESTED_FRAMEWORK_SECTIONS.values():
        section_data = nested_data.get(submodel_name) or {}
        multipool = section_data.get('multipool', 1)
        if multipool is not None and (not isinstance(multipool, int) or multipool < 1):
            raise ValueError(
                "%s.multipool must be >= 1, got %s"
                % (submodel_name, multipool))
    return nested_data






# ---------------------------------------------------------------------
# The constraint block's refusals.
#
# `schema/config.py` owns what a constraint declaration may say (its models
# refuse a bad value as they are built) and `schema/normalize.py` owns turning
# one into a plan. What is left, and lives here, is the three rules that decide
# whether a declaration can be served at all: the blocks a periodic system
# cannot carry, the shape of a ramped endpoint against the value it replaces,
# and the sign of a quantity that has to stay positive. The plan build calls
# them on its way through, so the same rule cannot mean two things depending on
# which layer reached it first.
# ---------------------------------------------------------------------

# The ramped quantities that cannot reach zero and keep their meaning. The
# runtime module checks the same list at every step
# (workflow/constraints/schedule.py, POSITIVE_PARAMETERS); the endpoints are
# refused here so a run does not die at its first ramped step for a number the
# input could have said was wrong.
POSITIVE_RAMP_PARAMETERS = ('force_constant', 'radius', 'semiaxes', 'lengths',
                            'height')

# Block kinds a periodic system still cannot serve. A strict constraint, a wall
# and a fix are settled at runtime by capability: the constraint is measured on
# the geometry unfolded along its bond graph, a wall is a boundary against a
# declared center, a fix masks components. A collective variable is not: its
# atoms need not be connected by any bond path, and for such a variable there is
# no image rule to derive, only one to invent (design document section 5.2, and
# Appendix B section 1 layer 4 of
# `dev/methods/CONSTRAINT_DEVELOPMENT.md`).
def check_periodic(config, pbc):
    """Refuse, at input time, the blocks a periodic system cannot serve."""
    flags = np.asarray(pbc or [0, 0, 0], dtype=int).reshape(-1)
    if flags.size != 3:
        raise ConstraintInputError(
            'pbc must hold three flags, got %d' % flags.size)
    if not bool(np.any(flags != 0)) or config.is_empty():
        return
    # The variables are not refused by block any more. Whether a collective
    # variable has a periodic image is a fact about the bond graph, which this
    # layer does not have; `state_adapter._validate_images` asks it per variable
    # at binding time, serves the ones inside a molecule and names the ones that
    # straddle two. What stays an input-time refusal is what no geometry can
    # rescue: the law below.
    centered = [str(record.id or 'wall%d' % (position + 1))
                for position, record in enumerate(config.walls)
                if str(record.law or 'hard') == 'soft']
    if centered:
        raise UnsupportedCapabilityError(
            "a periodic system cannot serve 'wall_law soft' (%s): that law is "
            'a spring toward one point rather than a barrier, so it has no shell '
            'to check against a cell face, and a wrapped coordinate jumps its '
            'offset by a whole lattice vector every time an atom crosses one. '
            'Keep the default `hard` law, or run this wall with pbc=0'
            % ', '.join(centered))


def check_ramp_shape(parameter, value, declared, label, name):
    """An endpoint replaces a value, so it has to be shaped like it."""
    lengths = set()
    for item in value:
        lengths.add(len(item) if isinstance(item, (list, tuple)) else 1)
    if len(lengths) > 1:
        raise ConstraintInputError(
            "%s: schedule '%s' ramps '%s' with endpoints of different shapes "
            '(%s); one ramp moves one geometry'
            % (label, name, parameter,
               ', '.join(str(int(count)) for count in sorted(lengths))))
    if declared is None:
        return
    count = (len(declared) if isinstance(declared, (list, tuple)) else 1)
    if lengths and lengths != {count}:
        raise ConstraintInputError(
            "%s: schedule '%s' ramps '%s' with %d numbers against a declared "
            'value of %d' % (label, name, parameter,
                             sorted(lengths)[0], count))


def _flatten_numbers(items):
    """Every number of an endpoint list, however deeply it is nested."""
    flat = []
    for item in items:
        if isinstance(item, (list, tuple)):
            flat.extend(_flatten_numbers(item))
        else:
            flat.append(float(item))
    return flat


def check_ramp_positive(record, label):
    """A parameter that must stay positive is refused at its endpoints."""
    if record.parameter not in POSITIVE_RAMP_PARAMETERS:
        return
    numbers = record.scale if record.scale is not None else record.value
    flat = _flatten_numbers(numbers)
    if any(value <= 0.0 for value in flat):
        raise ConstraintInputError(
            "%s: schedule '%s' ramps '%s' to a non-positive endpoint (%s). "
            'The subsystem treats a zero or negative force constant, height or '
            'length as a refusal at every step '
            '(workflow/constraints/schedule.py, POSITIVE_PARAMETERS), so a ramp '
            'that turns a term on has to start above zero rather than at it - '
            'and refusing it here says so before the first ramped step costs '
            'anything'
            % (label, record.id, record.parameter,
               ', '.join('%.6g' % value for value in flat)))



def validate_search_section(nested_data):
    """Validate the ranking vocabulary of the grid scan section (&search).

    Only the two enums are checked here. The grid itself is refused by
    the plan (workflow/grid/plan.py) the moment a scan is planned, which
    is where the skill's declaration is known; asking this validator to
    judge candidate values would put the same rule in two places with
    only one of them able to see the vocabulary.

    A misspelled statistic is not allowed to fall back to the default,
    because the default answers a different question than the one the
    user asked: 'sort_stat: average' would rank by the worst member and
    report a ranking nobody requested.

    Parameters
    ----------
    nested_data : dict
        Config model dump (submodel name -> section dict).

    Returns
    -------
    dict
        The unchanged nested data.

    Raises
    ------
    ValueError
        If sort_stat or sort_order names an unranked option.
    """
    section = nested_data.get('search') or {}
    for key, vocabulary in (('sort_stat', SEARCH_SORT_STATS),
                            ('sort_order', SEARCH_SORT_ORDERS)):
        value = section.get(key)
        if value is None:
            continue
        if str(value) not in vocabulary:
            raise ValueError(
                "search.%s must be one of %s, got %r"
                % (key, ', '.join(vocabulary), value))
    return nested_data


def validate_read_section(nested_data):
    """Validate the read-only extraction section (&read).

    A read run extracts labels from finished artifacts: it never starts
    an engine, so it has exactly one theory binding, no reference, no
    multiscale partition and the framework driver. The labels select
    what the skill parser is asked for; an empty selection asks nothing
    and is refused here rather than producing a dataset with no label.

    Parameters
    ----------
    nested_data : dict
        Config model dump (submodel name -> section dict).

    Returns
    -------
    dict
        The unchanged nested data.

    Raises
    ------
    ValueError
        If the section is set without jobtype read, names no path, asks
        for no label, or the run is not a single-theory main-driver run.
    """
    section = nested_data.get('read') or {}
    control = nested_data.get('control') or {}
    tokens = str(control.get('jobtype', '') or '').lower().split()

    if 'read' not in tokens:
        # The section model always exists; only an explicit path or
        # profile makes its presence a user statement. Label defaults
        # (energy/gradient on) are not one.
        if any(str(section.get(key) or '').strip()
               for key in ('path', 'project', 'profile')):
            raise ValueError(
                "the read section (path/project/profile) is only accepted "
                "with control.jobtype 'read', got %r"
                % (str(control.get('jobtype', '') or ''),))
        return nested_data

    path = section.get('path')
    if isinstance(path, (list, tuple)):
        entries = [str(item).strip() for item in path
                   if str(item).strip()]
    else:
        entries = [str(path).strip()] if str(path or '').strip() else []
    if not entries:
        raise ValueError(
            "read.path is required: name a calculation directory, an "
            "output file, or a file listing one path per line")

    requested = [name for name in ('energy', 'gradient', 'nac', 'soc')
                 if bool(section.get(name))]
    if not requested:
        raise ValueError(
            "read asks for no label; enable at least one of "
            "energy/gradient/nac/soc")

    theory = [str(item) for item in (control.get('theory') or [])
              if str(item).strip()]
    reference = [str(item) for item in (control.get('reference') or [])
                 if str(item).strip()]
    if len(theory) != 1:
        raise ValueError(
            "a read run binds exactly one theory skill (the one whose "
            "parser reads the artifacts), got %d" % len(theory))
    if reference:
        raise ValueError(
            "a read run has no reference skill; remove %r"
            % (reference,))
    driver = str(control.get('driver', 'main') or 'main').lower()
    if driver != 'main':
        raise ValueError(
            "a read run uses driver 'main' (it never starts a skill "
            "iteration), got %r" % (driver,))
    system = nested_data.get('system') or {}
    partitioned = bool(control.get('multiscale_type')) or any(
        system.get(key) for key in
        ('highlevel', 'midlevel', 'highref', 'midref'))
    if partitioned:
        raise ValueError(
            "a read run has no multiscale partition; remove "
            "multiscale_type/highlevel/midlevel/highref/midref")
    return nested_data


def validate(configs):
    """Validate a PyRAIConfig or the config within a pipeline dict.

    Accepts either a PyRAIConfig instance directly or a pipeline
    context dict with key 'config'. Runs all validation checks and
    returns the same type as input.

    Parameters
    ----------
    configs : PyRAIConfig or dict
        Config instance or pipeline context dict with key 'config'.

    Returns
    -------
    PyRAIConfig or dict
        Validated config (same type as input).

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    if isinstance(configs, BaseModel):
        config = configs
    else:
        config = configs['config']

    nested_data = config.model_dump()
    data = flatten_config_dump(nested_data)
    data = validate_task(data)
    data = validate_system(data)
    data = validate_electronic(data)
    data = validate_centers(data)
    data = validate_reference_partition(data)
    data = validate_dynamics(data)
    data = validate_adaptive(data)
    data = validate_skill_name(data)
    validate_dispatch_sections(nested_data)
    validate_search_section(nested_data)
    validate_read_section(nested_data)
    # Every check above reads the configuration and returns it
    # unchanged, so there is nothing to rebuild here: a rebuilt Config
    # dumps to the same data as the input (measured over the shipped
    # examples, lock in test_pipeline_builds_the_config_twice). The
    # rebuild that applies values is normalize's; this one was a third
    # full construction of the same model.
    if isinstance(configs, BaseModel):
        return config
    configs['config'] = config
    return configs
