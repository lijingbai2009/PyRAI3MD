"""
======================================================================
This module provides the capability resolution logic for the planning
layer. It resolves jobtype tokens into capability name strings,
derives additional capabilities from dynamics keywords, enforces
multi-backend constraints, and distributes per-skill capabilities.

No Property objects are created -- all capabilities are plain strings.

Aug 01 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from ..schema.config import resolve_hessian_route
from ..schema.optimizer_spec import OPTIMIZER_DECLARATIONS
from ..schema.capabilities import (
    CAPABILITY_NAMES,
    TOKEN_TO_CAPABILITY,
)

# Dependency relationships (capability -> required capabilities)
_DEPENDENCIES = {
    'gradient': ['energy'],
    'hessian': ['energy', 'gradient'],
    'nac': ['energy'],
    'soc': ['energy'],
    'dipole': ['energy'],
    'transition_dipole': ['energy'],
    'overlap': ['energy'],
    'stress': ['energy'],
}

# Framework input sections whose keywords state what has to be computed,
# beyond the jobtype token that names the run. This is the scan surface of
# the planner (docs PLANNING 4.2): a need that is not reachable from here is
# invisible to capability distribution, to the multi-backend ban and to the
# per-task declaration check, which is how a crossing search's derivative
# coupling came to be read only inside the task builder - the plan spelled
# `energy gradient` for every optimize run and the engine printed no coupling
# at all (#33).
_REQUIREMENT_KEYWORDS = {
    'control': ('jobtype', 'driver'),
    'dynamics': ('sfhp', 'nactype', 'radiative', 'dosoc', 'dodipole',
                 'ensemble'),
    'optimize': ('opt_method', 'opt_state_i', 'opt_state_j', 'optimize_cell'),
    'hessian': ('hess_driver',),
    'system': ('coupling', 'nac_pair_map', 'soc_pair_map', 'nstate',
               'state_spin'),
}

# Default capabilities for advanced jobtypes
_ADVANCED_DEFAULTS = {
    'optimize': ['energy', 'gradient'],
    # A dynamics run asks for the couplings only when the coupling method
    # needs them (fssh + nactype=nac, added by _derive_from_dynamics).
    # ktdc builds the coupling from the energy curvature and needs no NAC
    # vector, so the default is the E+G pair every ML potential serves.
    'md': ['energy', 'gradient'],
    'adaptive': ['energy', 'gradient'],
    'search': ['energy', 'gradient'],
    # 'train' and 'prediction' are intentionally NOT expanded: ML
    # tasks go through the skill's own API without capability plans.
}

# All known capability names. 'field' and 'qmmm' are task duties of the
# additive QM/MM plan, not user jobtype tokens: the LAMMPS skill maps
# them (with energy/gradient) to its qmmm_field/qmmm_energy profiles.


# Basic capabilities not subject to multi-backend scoping. 'stress' is
# a whole-system structural property provided by the full-system task
# (resolve_system_task); keeping it here ensures a non-primary backend
# (e.g. the multiscale L(real) baseline) still receives it instead of
# being scoped away. 'field'/'qmmm' are the QMMM task duties; they ride
# with energy/gradient.
_BASIC_CAPABILITIES = frozenset({'energy', 'gradient', 'stress',
                                 'field', 'qmmm'})

# Advanced capabilities subject to multi-backend scoping
_ADVANCED_CAPABILITIES = frozenset({
    'nac', 'soc', 'overlap', 'dipole', 'transition_dipole', 'hessian',
})


def resolve_capabilities(tokens, planning_context=None, skills=None):
    """Resolve jobtype tokens to capability name strings.

    Performs:
    1. Expand advanced jobtype tokens to basic capabilities.
    2. Map tokens to capability name strings.
    3. Resolve dependencies.
    4. Derive additional capabilities from dynamics config.
    5. Enforce multi-backend constraints.
    6. Distribute per-skill capabilities.

    Parameters
    ----------
    tokens : list of str
        Canonical jobtype tokens from parse_jobtype.
    planning_context : dict, optional
        Dynamics keywords from config (sfhp, nactype, dosoc, dodipole, etc.).
    skills : dict, optional
        The nested skills dict {'theory': {...}, 'reference': {...}}.
        Used for multi-backend constraint enforcement.

    Returns
    -------
    dict
        'property_names': list of str -- full capability list.
        'is_basic': bool -- whether this is a basic jobtype.
        'is_advanced': bool -- whether this is an advanced jobtype.
        'backend_capabilities': dict -- {skill_name: [cap_names]}.
        'dosoc': int -- SOC computation flag.
        'dodipole': int -- dipole computation flag.
    """
    from ..schema.validate import BASIC_JOBTYPES, ADVANCED_JOBTYPES

    if planning_context is None:
        planning_context = {}
    if skills is None:
        skills = {'theory': {}, 'reference': {}}

    # Normalize dosoc/dodipole to int (read from planning_context)
    dosoc = _normalize_flag(planning_context.get('dosoc', 0))
    dodipole = _normalize_flag(planning_context.get('dodipole', 0))

    # Determine basic vs advanced
    is_basic = any(t in BASIC_JOBTYPES for t in tokens)
    is_advanced = any(t in ADVANCED_JOBTYPES for t in tokens)

    # Step 1: Expand advanced jobtypes
    expanded = []
    for token in tokens:
        if token in _ADVANCED_DEFAULTS:
            expanded.extend(_ADVANCED_DEFAULTS[token])
        elif token in TOKEN_TO_CAPABILITY:
            expanded.append(token)
        # Skip unknown tokens (workflow handles them)

    # Step 2: Map to capability names
    capabilities = []
    seen = set()
    for token in expanded:
        cap = TOKEN_TO_CAPABILITY.get(token, token)
        if cap not in seen:
            capabilities.append(cap)
            seen.add(cap)

    # Step 3: Resolve dependencies
    capabilities = _resolve_dependencies(capabilities)

    # Step 4: Derive from dynamics config (dosoc/dodipole included)
    capabilities = _derive_from_dynamics(
        capabilities, planning_context
    )

    # Step 4b: Derive the whole-system stress requirement from the
    # ensemble (md + npt) and the cell-optimization switch
    # (optimize + optimize_cell).
    capabilities = _derive_stress(capabilities, tokens, planning_context)

    # Step 4c: the needs a workflow section states itself, read from the
    # same scanned context (a crossing search's derivative coupling).
    capabilities = _derive_from_requests(capabilities, planning_context)

    # Step 5: Check multi-backend constraints
    _check_multi_backend_constraints(
        capabilities, skills,
        tokens=tokens,
        hessian_params=(planning_context or {}).get('hessian_params'))

    # Step 6: Distribute per-skill capabilities
    backend_capabilities = _distribute_capabilities(capabilities, skills)

    return {
        'property_names': sorted(capabilities),
        'is_basic': is_basic,
        'is_advanced': is_advanced,
        'backend_capabilities': backend_capabilities,
        'dosoc': dosoc,
        'dodipole': dodipole,
    }


def _resolve_dependencies(capabilities):
    """Resolve capability dependencies (e.g. gradient requires energy).

    Parameters
    ----------
    capabilities : list of str
        Initial capability list.

    Returns
    -------
    list of str
        Capability list with dependencies included.
    """
    result = list(capabilities)
    seen = set(result)
    queue = list(capabilities)

    while queue:
        cap = queue.pop(0)
        for dep in _DEPENDENCIES.get(cap, []):
            if dep not in seen:
                result.append(dep)
                seen.add(dep)
                queue.append(dep)

    return result


def _normalize_flag(value):
    """Normalize a flag value to int (0 or 1).

    Accepts int, str ('true', '1', 'yes' -> 1; others -> 0).

    Parameters
    ----------
    value : int or str
        Raw flag value.

    Returns
    -------
    int
        0 or 1.
    """
    if isinstance(value, int):
        return value
    s = str(value).lower().strip()
    if s in ('true', '1', 'yes'):
        return 1
    return 0


def _derive_from_dynamics(capabilities, planning_context):
    """Derive additional capabilities from dynamics keywords.

    Parameters
    ----------
    capabilities : list of str
        Current capability list.
    planning_context : dict
        Dynamics settings (sfhp, nactype, dosoc, dodipole, radiative, etc.).

    Returns
    -------
    list of str
        Updated capability list with dynamics-derived additions.
    """
    result = list(capabilities)
    seen = set(result)

    sfhp = str(planning_context.get('sfhp', '')).lower()
    nactype = str(planning_context.get('nactype', '')).lower()
    radiative = str(planning_context.get('radiative', '0')).lower()
    dosoc = _normalize_flag(planning_context.get('dosoc', 0))
    dodipole = _normalize_flag(planning_context.get('dodipole', 0))

    # fssh + dcm -> overlap
    if sfhp == 'fssh' and nactype == 'dcm':
        if 'overlap' not in seen:
            result.append('overlap')
            seen.add('overlap')

    # fssh + nac -> nac. The dynamics default is E+G; only the NAC-vector
    # coupling method needs this channel. ktdc and dcm do not.
    if sfhp == 'fssh' and nactype == 'nac':
        if 'nac' not in seen:
            result.append('nac')
            seen.add('nac')

    # dosoc flag -> soc
    if dosoc:
        if 'soc' not in seen:
            result.append('soc')
            seen.add('soc')

    # dodipole flag -> dipole
    if dodipole:
        if 'dipole' not in seen:
            result.append('dipole')
            seen.add('dipole')

    # radiative=true -> transition_dipole
    if radiative in ('true', '1', 'yes'):
        if 'transition_dipole' not in seen:
            result.append('transition_dipole')
            seen.add('transition_dipole')

    # Resolve new dependencies
    result = _resolve_dependencies(result)

    return result


def crossing_coupling_pair(planning_context):
    """The state pair a crossing search spans its branching plane on.

    None when the run asks for no coupling. `opt_state_i`/`opt_state_j` are
    the search's chasing input - they say which two surfaces are followed,
    not which states exist - so they never define the state ladder; they
    select one pair of the ladder `ci`/`spin` declare. Whether the plane is
    spanned by the derivative coupling is the optimizer class's own
    declaration (`uses_derivative_coupling`), which is why a spin-forbidden
    `mecp` pair asks for none.

    Parameters
    ----------
    planning_context : dict
        Scanned input keywords (driver, opt_method, opt_state_i/j).

    Returns
    -------
    list of int or None
        The two 0-based local state indices, or None.
    """
    if str(planning_context.get('driver') or 'main').lower() != 'main':
        # driver names a skill: that skill plans its own search and the
        # framework adds no request to its jobtype.
        return None
    # The run has to be the one that searches. `optimize.method` can be left
    # at a crossing name in an input whose jobtype is `md`, and that run
    # never steps on a plane. An unstated jobtype is not a refusal here: the
    # caller handed over a partial context (tests, the direct calls).
    jobtype = str(planning_context.get('jobtype') or '').lower().split()
    if jobtype and 'optimize' not in jobtype:
        return None
    method = str(planning_context.get('opt_method') or '')
    if not method:
        return None
    declaration = OPTIMIZER_DECLARATIONS.get(method, {})
    if not declaration.get('uses_derivative_coupling', False):
        return None
    raw_i = planning_context.get('opt_state_i')
    raw_j = planning_context.get('opt_state_j')
    if isinstance(raw_i, (list, tuple)) or isinstance(raw_j, (list, tuple)):
        # A per-center pair. This layer only decides whether the `nac`
        # capability is needed, so a pair being named is enough; which states
        # each center actually pairs is read where the duties are built.
        picks_i = list(raw_i) if isinstance(raw_i, (list, tuple)) else None
        picks_j = list(raw_j) if isinstance(raw_j, (list, tuple)) else None
        if picks_i is None or picks_j is None:
            return [0, 1]
        paired = [i != j for i, j in zip(picks_i, picks_j)]
        return [0, 1] if any(paired) else None
    state_i = int(raw_i or 0)
    if raw_j is None or int(raw_j) == state_i:
        return None
    return [state_i, int(raw_j)]


def _derive_from_requests(capabilities, planning_context):
    """Add the capabilities a workflow section asks for beyond its token.

    A crossing search that spans its plane on the derivative coupling has to
    carry the 'nac' capability: `&ALASKA` computes a coupling only for a pair
    the deck names, so a plan without it spelled `energy gradient` and the
    search ran on a branching plane of one direction (#33). The pair itself is
    checked against `molecule.coupling` where the task is built
    (`planning/assembly.py::_task_property_request`): this layer says what is needed,
    the declaration says which pairs exist.
    """
    if 'nac' in capabilities:
        return list(capabilities)
    if crossing_coupling_pair(planning_context) is None:
        return list(capabilities)
    return _resolve_dependencies(list(capabilities) + ['nac'])


def _derive_stress(capabilities, tokens, planning_context):
    """Derive the whole-system stress capability.

    The stress tensor is required by two framework routes:
    - NPT dynamics: jobtype 'md' with ensemble 'npt' (the barostat
      consumes state.active_stress() as the internal virial).
    - Cell optimization: jobtype 'optimize' with optimize_cell true.

    Both are advanced jobtypes, so the request is carried to the skill
    through the 'stress' capability (injected as a skill config flag),
    never through the skill jobtype spelling.

    Parameters
    ----------
    capabilities : list of str
        Current capability list.
    tokens : list of str
        Canonical jobtype tokens from parse_jobtype.
    planning_context : dict
        Planning context (ensemble, optimize_cell, ...).

    Returns
    -------
    list of str
        Updated capability list with 'stress' appended when required.
    """
    result = list(capabilities)
    seen = set(result)

    ensemble = str(planning_context.get('ensemble', 'nve') or 'nve').lower()
    optimize_cell = bool(planning_context.get('optimize_cell', False))

    needs_stress = (
        ('md' in tokens and ensemble == 'npt')
        or ('optimize' in tokens and optimize_cell)
    )

    if needs_stress and 'stress' not in seen:
        result.append('stress')
        seen.add('stress')
        result = _resolve_dependencies(result)

    return result


def _count_backends(skills):
    """Count total number of theory and reference backends.

    Parameters
    ----------
    skills : dict
        Nested skills dict.

    Returns
    -------
    tuple of (int, int)
        (n_theory, n_reference)
    """
    n_theory = len(skills.get('theory', {}))
    n_reference = len(skills.get('reference', {}))
    return n_theory, n_reference


def _check_multi_backend_constraints(capabilities, skills,
                                     tokens=None, hessian_params=None):
    """Enforce multi-backend constraints.

    An analytical / backend-produced Hessian is forbidden with multiple
    backends. The framework's numerical route is allowed: it differences
    the composite gradient, which is the only matrix an additive composite
    owns. ``hess_driver: main`` is that route.

    Per-binding capability scoping (advanced properties on the first
    theory + first reference pair) is `_distribute_capabilities`.

    Parameters
    ----------
    capabilities : list of str
        Current capability list (read only).
    skills : dict
        Nested skills dict.
    tokens : sequence of str or None
        Canonical jobtype tokens (for the Hessian route).
    hessian_params : dict or None
        ``&hessian`` section values.

    Raises
    ------
    ValueError
        If a backend-owned Hessian is requested with multiple backends.
    """
    n_theory, n_reference = _count_backends(skills)
    multi_backend = (n_theory >= 2) or (n_reference >= 2)

    if not multi_backend:
        return

    # Constraint 1: a single backend cannot own the composite matrix.
    if 'hessian' in capabilities:
        _mode, driver = resolve_hessian_route(tokens or [], hessian_params)
        if driver == 'skill':
            raise ValueError(
                "Multiscale ONIOM/QMMM has no single backend that owns the "
                "composite force-constant matrix: hess_driver 'skill' asks "
                "one term for a matrix that covers only that term. Run the "
                "Hessian on a single-method queue, or let the framework "
                "difference the composite gradient with jobtype "
                "num_hess/num_freq (equivalently &hessian hess_mode: "
                "numerical with hess_driver: main).")


def _distribute_capabilities(capabilities, skills):
    """Distribute capabilities to each skill backend.

    When multiple backends are configured, advanced properties apply
    only to the first theory + first reference backend pair.
    Other backends are restricted to energy + gradient.

    Parameters
    ----------
    capabilities : list of str
        Full capability list.
    skills : dict
        Nested skills dict.

    Returns
    -------
    dict
        {binding role: [capability_names]}
    """
    n_theory, n_reference = _count_backends(skills)
    multi_backend = (n_theory >= 2) or (n_reference >= 2)

    backend_capabilities = {}

    # Get the ordered binding roles (a skill can hold two of them)
    theory_names = list(skills.get('theory', {}).keys())
    reference_names = list(skills.get('reference', {}).keys())

    for i, name in enumerate(theory_names):
        if multi_backend and i > 0:
            # Non-primary theory backend: restricted to basic
            backend_capabilities[name] = sorted(
                c for c in capabilities if c in _BASIC_CAPABILITIES
            )
        else:
            # Primary theory backend (or single backend): full capabilities
            backend_capabilities[name] = sorted(capabilities)

    for i, name in enumerate(reference_names):
        if multi_backend and i > 0:
            # Non-primary reference backend: restricted to basic
            backend_capabilities[name] = sorted(
                c for c in capabilities if c in _BASIC_CAPABILITIES
            )
        else:
            # Primary reference backend (or single backend): full capabilities
            backend_capabilities[name] = sorted(capabilities)

    return backend_capabilities
