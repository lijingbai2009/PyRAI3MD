"""
======================================================================
Planning-side assembly of the task queue.

`build_state` (data/build.py) owns the state object; this module owns
the step that turns a validated Config plus the resolved skill bindings
into the task descriptors PyRAIState carries. The state container does
not import planning, workflow or skill internals; this module is the
composition step that may.

Nov 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import importlib
from copy import deepcopy

import numpy as np

from ..schema.config import DEFAULT
from ..schema.capabilities import CAPABILITY_TO_TOKEN, IMPLIED_BY_HESSIAN
from ..schema.adapter import _merge_one_skill_config
from ..schema.optimizer_spec import (
    OPTIMIZER_DECLARATIONS, implemented_optimizers)
from ..schema.validate import is_standalone_queue, parse_jobtype
from ..data.center import CenterPlanError, build_center_plan
from ..data.geometry import continuous_coordinates
from ..kernel.maestro import find_binding
from ..kernel.logger import get_logger
from ..skills.query import discover_all, get_registry, skill_module_path
from .resolver import crossing_coupling_pair
from .task import resolve_task_queue


# framework resolve the workflow from jobtype; 'skill' or a skill
# program label hands the jobtype to the skill for internal
# execution (framework single shot).
MAIN_DRIVER = 'main'
SKILL_DRIVER_KEYWORD = 'skill'

# Jobtypes that ask whoever answers them to own the iteration. On the route
# that hands the steps over (driver != MAIN_DRIVER) a skill whose profile
# table has no row for one of these cannot be served, and the refusal must
# reach the caller rather than be downgraded to a warning (see
# `_build_skill_flat_and_submodels`).
ENGINE_OWNED_ITERATION_JOBTYPES = frozenset(
    {'md', 'optimize', 'surface_hopping'})

# Fixed canonical order of the jobtype tokens written into skill
# configs: planning capabilities are translated back to tokens and
# spelled in this order so skills always see one canonical form.
JOBTYPE_TOKEN_ORDER = (
    'energy', 'gradient', 'field', 'qmmm', 'nac', 'soc',
    'dipole', 'transition', 'frequency', 'hessian', 'absorption',
)

def _spell_advanced_jobtype(state):
    """Whether an advanced jobtype asks the skill for planned properties.

    md/surface_hopping/adaptive are framework-driven iterations: the skill
    serves one evaluation per step, so its jobtype is the distributed
    property set. A main-driver optimize is the same. Skill-driven
    iterations (driver != main) keep the raw token because the skill plans
    the whole job internally.

    Parameters
    ----------
    state : PyRAIState
        State carrying the jobtype and driver.

    Returns
    -------
    bool
        True when the skill jobtype should be spelled from capabilities.
    """
    jobtype = str(getattr(state, 'jobtype', '') or '').lower()
    driver = getattr(state, 'driver', MAIN_DRIVER)
    if jobtype in ('md', 'surface_hopping', 'adaptive'):
        # The whole point of the skill driver: the skill owns the
        # iteration and resolves it from its own (jobtype, driver) table
        # (`md` + `skill` -> the reference/dynamics profile). Spelling
        # the capability set here hid the iteration token and the qdm
        # reference leg fell back to `*_gradient`, so the benchmark's
        # population record was never written (#467).
        return driver == MAIN_DRIVER
    return jobtype == 'optimize' and driver == MAIN_DRIVER


def spell_skill_jobtype(capabilities, fallback):
    """Spell the skill jobtype string from planning capabilities.

    Capability names are translated back to jobtype tokens
    (transition_dipole -> transition, hessian -> frequency) and
    written in the fixed canonical order. When capabilities are
    unavailable the fallback jobtype is returned as-is.

    Parameters
    ----------
    capabilities : list of str
        Per-skill capabilities distributed by planning.
    fallback : str
        Framework jobtype string used when capabilities are empty.

    Returns
    -------
    str
        Canonical jobtype string for the skill config.
    """
    if not capabilities:
        return fallback
    # A force-constant request already carries its own energy evaluation, and
    # its dependency list adds `energy`/`gradient` to the capability set for
    # every backend. Spelling those as extra tokens would hand the skill
    # `energy gradient frequency`, a jobtype no skill has a profile for, so the
    # capabilities a present capability implies are absorbed rather than
    # repeated. Only `gradient` alongside `energy` keeps its two-token form,
    # which is what the property profiles expect.
    token_set = {
        CAPABILITY_TO_TOKEN[cap] for cap in capabilities
        if cap in CAPABILITY_TO_TOKEN
        and not (cap in IMPLIED_BY_HESSIAN and 'hessian' in capabilities)
    }
    if not token_set:
        return fallback
    return ' '.join(
        token for token in JOBTYPE_TOKEN_ORDER if token in token_set
    )


def _read_skill_jobtype(config):
    """Spell the skill jobtype a read run asks its parser for.

    Energy is always part of the request when any label is: every other
    channel is a derivative or a coupling the engine registers under an
    energy evaluation. The four read labels are the whole vocabulary; a
    read run never asks for a property the section cannot name.
    """
    section = getattr(config, 'read', None)
    tokens = ['energy']
    for name in ('gradient', 'nac', 'soc'):
        if bool(getattr(section, name, False)):
            tokens.append(name)
    return ' '.join(tokens)


MATRIX_CAPABILITIES = ('hessian', 'frequency', 'vibrational_frequency')


def _skill_capabilities(skill_name):
    """The capabilities one skill's own manifest declares, as it spells them.

    Read from the registry (`skills/query.py:126`, which falls back to the
    manifest), not from the planned task list: `skills[category][key]
    ['capabilities']` is what planning assigned to this backend for this jobtype,
    and an optimization run is planned for energy and gradient whatever the
    program can do. The question a curvature producer is chosen on is what the
    skill offers at all.
    """
    registry = get_registry(skill_name) or {}
    names = set()
    for entry in registry.values():
        names.update(str(item) for item in (entry.get('capabilities') or []))
    return names


def _declares_a_matrix(skill_name):
    """True when this skill offers force constants under any of its names."""
    return bool(_skill_capabilities(skill_name)
                & set(MATRIX_CAPABILITIES))


# The two vocabularies one channel: a manifest spells it short ('nac',
# 'soc', the neural skills) or after the physical quantity ('nonadiabatic_
# coupling', 'soc_coupling', the wavefunction skills).
_CHANNEL_CAPABILITY_NAMES = {
    'nac': ('nac', 'nonadiabatic_coupling'),
    'soc': ('soc', 'soc_coupling'),
}


def _declares_a_channel(skill_name, capability):
    """True when this skill offers one coupling channel under any of its names.

    Asking a backend for a coupling it declares no profile for would spell a
    jobtype token it cannot resolve, and drop the request into a leg that has
    no rows to read back.
    """
    if not skill_name:
        return False
    names = _skill_capabilities(skill_name)
    return bool(names & set(_CHANNEL_CAPABILITY_NAMES[capability]))


def _validate_curvature_freezes(state, method_name):
    """A curvature-following search may freeze whole atoms, not components.

    Freezing one Cartesian component of a molecule in vacuum is a statement
    about the laboratory frame, and the frame the input chose is not part of the
    molecule: turn the molecule over and the constraint is a different
    constraint, so the point its optimizer stops at moves with the orientation.
    A relaxation can still be asked to find that point (the criterion is a
    gradient on a chosen submanifold and the search reaches it), but a search
    that follows curvature must be able to rotate the free part of the system
    about what is pinned, because a rotation is one of the directions its
    subspace projects out; freeze an atom's z only and the surviving torque
    force has no direction left to be removed along. Measured on an analytic
    triatomic: the step fell to 1.3e-7 and the model held its one negative
    curvature, while the criterion still read 2.1e-2 Hartree per Bohr sitting on
    the *free* y components - a constraint force the run cannot supply because
    no constraint absorbs a torque.

    A whole-atom freeze keeps that covariance: rotating the free atoms about a
    pinned atom moves only free components, so it stays inside the search's
    subspace. Periodic and field-defined frames are the case where a single
    axis is physical; those are left to their own round (`dev/TODO.md`), which
    is also why this is a refusal and not a rewriting of the input.
    """
    plan = state.constraint_plan()
    if plan.is_empty:
        return
    partial = [record.id for record in plan.actions
               if record.kind == 'fix'
               and len(record.parameters.get('axes', ())) < 3]
    if partial:
        raise ValueError(
            "optimize.method '%s' cannot search under a component-level fix: "
            "%s freezes Cartesian components, which pins a laboratory direction "
            "a free molecule does not have, and the torque the constraint then "
            "leaves behind has no free coordinate to be removed along. Freeze "
            "whole atoms (direction: xyz) to hold a fragment rigid"
            % (method_name, ', '.join(repr(name) for name in partial)))


def _derive_curvature_atoms(state, method_name):
    """Let the constraint plan say which atoms the curvature may move.

    Two keywords answer the same question with different granularity: `fix` says
    what an optimization may not move (per Cartesian component), `hess_atoms`
    says what a numerical Hessian displaces (per atom). During a search the two
    have to be one set, because a matrix holding zero rows for pinned atoms puts
    the lowest curvature *on* a direction the input froze, and a mode-following
    search then climbs it - which is why this pairing was refused outright until
    the constraint plan was made the authority.

    Nothing is written when no atom is frozen: an empty `hess_atoms` already
    means every atom, so a run without constraints keeps the section - and the
    batch it produces - exactly as it was before this derivation existed. An
    input that named a different set is refused rather than overwritten: "which
    atoms was the curvature taken over" has to stay readable from the input.
    """
    plan = state.constraint_plan()
    natom = int(getattr(state.pmd.system, 'natom', 0) or 0)
    declared = [int(atom) for atom in
                (state.hessian_params.get('hess_atoms') or [])]
    if not natom:
        return

    from ..data.constraint import build_fix_mask
    mask = (build_fix_mask(plan, natom)
            if not plan.is_empty else np.zeros((natom, 3), dtype=bool))
    derived = [index + 1 for index in range(natom) if not bool(mask[index].all())]
    if len(derived) == natom:
        # Nothing is frozen, so the search moves every atom. An explicit partial
        # set is still refused: without a fix behind it, the matrix would hold
        # zero rows on atoms the run is otherwise free to move, and the lowest
        # curvature in it would be an atom the *input* pinned while the
        # optimizer believes the rest of the molecule can follow it.
        if declared:
            raise ValueError(
                "optimize.method '%s' takes its active atoms from the "
                "constraint plan, and hess_atoms %s names a subset with no fix "
                "behind it: the curvature would hold zero rows on atoms this "
                "run can move. Declare the frozen atoms under `fixes` and the "
                "search derives the set" % (method_name, sorted(declared)))
        return

    if declared and sorted(declared) != sorted(derived):
        raise ValueError(
            "hess_atoms %s does not match the atoms this run leaves free: the "
            "constraint plan freezes %s, and an optimization that searches a "
            "curvature must search the curvature of what it can move. Drop "
            "hess_atoms (the search derives it) or widen the fix"
            % (sorted(declared),
               [atom + 1 for atom in range(natom) if bool(mask[atom].all())]))
    state.pmd.hessian.hess_atoms = derived


def _validate_driver(state):
    """Validate control.driver and the theory composition.

    Applied uniformly to every jobtype (energy/gradient/optimize/
    md/...): the framework never checks whether the skill supports
    the requested jobtype; unsupported combinations are reported by
    the skill itself.

    Parameters
    ----------
    state : PyRAIState
        State carrying the control section.

    Raises
    ------
    ValueError
        Unknown driver value; a skill driver combined with a
        multi-token theory (skill-driven jobs are exclusive); a
        named driver that does not match the theory skill.
    """
    driver = state.driver
    if driver == MAIN_DRIVER:
        return

    skill_names = sorted(discover_all().keys())
    if driver != SKILL_DRIVER_KEYWORD and driver not in skill_names:
        raise ValueError(
            "Unknown driver: %s. Available: %s"
            % (driver,
               [MAIN_DRIVER, SKILL_DRIVER_KEYWORD] + skill_names))

    theory = list(state.pmd.control.theory or [])
    if len(theory) != 1:
        raise ValueError(
            "driver '%s' requires exactly one theory skill "
            "(skill-driven jobs cannot mix skills), got: %s"
            % (driver, theory))
    if driver != SKILL_DRIVER_KEYWORD and driver != theory[0]:
        raise ValueError(
            "driver '%s' does not match the theory skill '%s'"
            % (driver, theory[0]))


def _task_property_request(descriptor, center_plan, crossing_pair=None):
    """Build the static property request spec for one task (docs 6.3).

    The request is no longer one global target_states: the high-level
    provider H(c) requests its OWN center's local states and coupling
    pairs, while a reference correction reads only its fixed
    reference_state and never another center's active state. The
    per-step evaluation_id and the resolved target_states are filled
    at runtime; planning fixes the static pair/state structure.

    Parameters
    ----------
    descriptor : dict
        Task descriptor from resolve_task_queue.
    center_plan : PartitionPlan or None
        Canonical center plan (None when it could not be built, e.g.
        a mock config in tests).
    crossing_pair : list of int or None
        The two surfaces a crossing search follows (`opt_state_i`/
        `opt_state_j`, 0-based). It selects which declared pair the search's
        branching plane is spanned on; it does not add one, so a pair
        `molecule.coupling` leaves out is refused here rather than asked for
        behind the declaration.

    Returns
    -------
    dict
        {'target_states', 'nac_pairs', 'soc_pairs', 'evaluation_id'}.
    """
    reference_only = descriptor.get('state_policy') == 'reference'
    spec = {
        'target_states': None,
        'nac_pairs': [],
        'soc_pairs': [],
        'evaluation_id': None,
    }
    c = descriptor.get('center_index')
    if (center_plan is None or c is None
            or c >= center_plan.n_center
            or not descriptor.get('provides_electronic')):
        # Baseline L(real), a reference correction, or a queue without
        # a center plan: no per-center active-state request.
        if reference_only:
            spec['target_states'] = [descriptor.get('reference_state') or 0]
        if (crossing_pair is not None
                and descriptor.get('provides_electronic')):
            # No plan means no declaration to consult (mock configs only, see
            # `build_task_queue`); the pair travels unchecked there. Every
            # real run goes through the branch below and has to declare it.
            spec['nac_pairs'] = spec['nac_pairs'] or [list(crossing_pair)]
            # A crossing search reads the pair and nothing else: asking every
            # local state's gradient is the dynamics rule, and here it buys an
            # evaluation per state that the step throws away.
            spec['target_states'] = list(crossing_pair)
        return spec
    es = center_plan.center(c).electronic_spec
    # The provider follows the center's active state; the first version
    # requests every local state so a hop always has its target
    # gradient available (docs 6.3).
    spec['target_states'] = list(range(es.nstate))
    spec['nac_pairs'] = [list(p) for p in es.nac_pair_map]
    spec['soc_pairs'] = [list(p) for p in es.soc_pair_map]
    _require_declared_couplings(descriptor, es)
    if (crossing_pair is not None
            and descriptor.get('provides_electronic')):
        # A crossing search reads the pair and nothing else: asking every
        # local state's gradient is the dynamics rule, and here it buys an
        # evaluation per state that the step throws away. The pair travels
        # only if `molecule.coupling` names it - the declaration says which
        # couplings are computed, `opt_state_i`/`opt_state_j` only which two
        # surfaces the search follows.
        declared = _declared_pair(es.nac_pair_map, crossing_pair)
        if declared is None:
            raise ValueError(
                "optimize.method spans the branching plane of states %d and "
                "%d on their derivative coupling, but this run's "
                "molecule.coupling does not name that pair (declared: %s). "
                "Declare it: a pair that is not declared is shielded, and "
                "the framework does not ask the engine for it behind the "
                "declaration"
                % (crossing_pair[0], crossing_pair[1],
                   [list(p) for p in es.nac_pair_map] or 'none'))
        spec['nac_pairs'] = [declared]
        spec['target_states'] = list(crossing_pair)
    return spec


def _declared_pair(pair_map, wanted):
    """The declared spelling of `wanted`, or None when it is not declared.

    Direction is not part of the identity of a coupling pair: declaring
    [1, 0] asks for the same coupling as [0, 1]. The declaration's own order
    is returned, so the request, the engine's table and the parser's row
    addressing agree.
    """
    target = sorted(int(state) for state in wanted)
    for pair in pair_map:
        if sorted(int(state) for state in pair) == target:
            return [int(state) for state in pair]
    return None


def _require_declared_couplings(descriptor, es):
    """Refuse a request for couplings a run declares none of.

    The two sides of one statement: the request set says the algorithm
    consumes a coupling, `molecule.coupling` says which pairs the engine
    computes. Neither may be left empty while the other is used, and the
    pair maps are never filled in by guesswork - adjacency included, which
    is what made an undeclared pair a shielded one rather than a missing
    quantity (docs MULTICENTER 4.4.5).

    A channel the addressed skill does not offer at all is not a violation
    of the declaration: nothing is asked of it and nothing is read back, and
    the crossing search runs on the gradient difference (the documented
    reading of `meci` on a backend without derivative couplings).
    """
    request = set(descriptor.get('capabilities') or [])
    skill_name = descriptor.get('skill_name')
    for capability in ('nac', 'soc'):
        if capability not in request:
            continue
        if not _declares_a_channel(skill_name, capability):
            continue
        if not getattr(es, capability + '_pair_map'):
            raise ValueError(
                "this run asks %s for %s couplings, but molecule.coupling "
                "declares no pair for it. Name the state pairs whose "
                "coupling is to be computed; a pair that is not declared is "
                "shielded, so there would be nothing to read back"
                % (descriptor.get('skill_name') or 'the backend', capability))


# Framework input sections that carry the dispatch keywords besides the
# molecule scope: every workflow section declares multitask/remotetask/
# multipool (schema.config.MOLECULE_TYPES / *_TYPES).
_DISPATCH_SECTIONS = ('dynamics', 'adaptive', 'optimize', 'hessian',
                      'search')


def _dispatch_requests_remote(config, dispatch_flags):
    """Decide whether the framework will ship this task off-process.

    ``execution_mode`` is a skill keyword: the backend selects its own
    runner from it (``skills/<skill>/service/runner`` returns run_slurm
    only for 'slurm'). The framework's dispatch decision lives in
    multitask/remotetask/multipool, which the skill schema deliberately
    never receives. This is the single place where the two meet: when
    the framework is going to submit the calculation, it says so in the
    skill's own vocabulary, so exactly one component submits.

    True when remotetask is set at any level that can move the work: the
    task's own merged keywords, the molecule scope, or a workflow
    section. The union is taken instead of mapping the jobtype token to
    its section because that mapping is the planner's WORKFLOW_SECTIONS
    table; a second copy of it would drift (CHANGELOG #299).

    Parameters
    ----------
    config : PyRAIConfig
        Validated framework config carrying the molecule and workflow
        sections.
    dispatch_flags : dict
        The task's own multitask/remotetask/multipool triple.

    Returns
    -------
    bool
        True when the framework submits this task's calculation itself.
    """
    if dispatch_flags.get('remotetask'):
        return True
    if getattr(config.system, 'remotetask', False):
        return True
    for section_name in _DISPATCH_SECTIONS:
        section = getattr(config, section_name, None)
        if section is not None and getattr(section, 'remotetask', False):
            return True
    return False


def build_task_queue(state, config, skills, maestro, raw_config=None):
    """Resolve the task queue and register tasks to PyRAIState.

    For each task in the resolved queue:
    1. Merge user skill values (maestro defaults + framework config
       + raw input skill sections) into a flat skill dict (cached
       per skill)
    2. Run the skill schema pipeline (build -> normalize -> validate)
    3. Build the task-specific extended PyRAIConfig (no geometry
       slicing; region slice indices stay in the task metadata)

    Planning never instantiates backends: the task queue is purely
    static, and backend instances are created at runtime by
    Calculator.initialize().

    Parameters
    ----------
    state : PyRAIState
        Target state to register tasks.
    config : PyRAIConfig
        Validated framework configuration.
    skills : dict
        Resolved skills from Maestro.resolve_skills(). Each category's dict is
        keyed by skill name, except where one skill holds two positions of a
        composite, which is keyed `name@role` (`kernel.maestro._binding_key`).
        Look a binding up by (skill_name, role), not by name.
    maestro : Maestro
        Populated Maestro instance.
    raw_config : dict, optional
        Original raw configuration from interface layer.
        Used to extract skill-specific settings (e.g. scratch_path)
        that are not part of the framework Config.
    """

    # Validate the execution driver once for the whole queue (all
    # jobtypes consume control.driver).
    _validate_driver(state)

    # Local import: planning imports schema at load, and schema imports
    # this package's constraint records, so a top-level edge would close
    # the cycle when the first import starts from planning.
    queue = resolve_task_queue(config, skills)

    _apply_standalone_embedding(queue, config)

    center_plan, plan_fingerprint = _bind_center_plan(state, config)

    # Skill binding collision guard: the same program serving several
    # tasks must be distinguished by a unique binding id, never deduped
    # into one shared config (docs 6.2).
    seen_bindings = set()

    # Cache validated skill data (merge + schema pipeline) per skill
    skill_data = {}
    # Cache dispatch-layer keywords (multitask/remotetask/multipool)
    # per skill; extracted from the raw merged dict before schema
    # validation.
    skill_dispatch = {}
    # The state pair a crossing search spans, per cache key: the merge
    # derives it once and every task of that binding reads the same pair.
    skill_crossing = {}

    _check_curvature_plan(state)

    for category in ('theory', 'reference'):
        for descriptor in queue.get(category, []):
            skill_name = descriptor['skill_name']
            backend_class = descriptor['backend']
            # The resolved entry is the *binding*: a composite can put one skill
            # at two positions, and then the second is keyed `skill@role`.
            # (skill_name, role) identifies it either way, so look it up by that
            # pair instead of by name - a name lookup would return whichever
            # binding was stored last.
            dir_name = find_binding(
                skills.get(category, {}), skill_name,
                # A QMMM subtask carries a physical duty role ('mm') that
                # differs from its positional binding role ('qm2'); use the
                # binding role so the config is found either way.
                descriptor.get('binding_role') or descriptor['role'],
            ).get('_dir_name') or skill_name

            # A QMMM task's skill jobtype carries a field/qmmm capability
            # token, so the field and energy tasks of one LAMMPS binding
            # resolve to different profiles. Cache the flattened skill
            # config per (skill, QMMM duty); a plain task keeps one entry
            # per skill.
            qmmm_stage = (descriptor.get('qmmm_role')
                          if descriptor.get('composite') == 'qmmm' else None)
            cache_key = ((skill_name, qmmm_stage) if qmmm_stage
                         else skill_name)

            if cache_key not in skill_data:
                (skill_data[cache_key], skill_dispatch[cache_key],
                 crossing) = _merge_skill_binding(
                    maestro, skill_name, descriptor, state, config,
                    raw_config, skills, center_plan, dir_name,
                    qmmm_stage)
                skill_crossing[cache_key] = crossing
            crossing_pair = skill_crossing.get(cache_key)

            _assemble_task(
                descriptor, skill_name, backend_class,
                skill_data[cache_key], skill_dispatch.get(cache_key, {}),
                config, state, dir_name, center_plan, plan_fingerprint,
                crossing_pair, category, seen_bindings)


def _apply_standalone_embedding(queue, config):
    """Carry a standalone background embedding onto its tasks.
    """
    # provider and the first reference task see the same rows; no other task
    # and no region slice does. Two spellings name the file - the
    # `external_charges` path keyword, and `read_charge: true` in a
    # single-theory queue, where that switch *is* the embedding switch (there
    # is no multiscale plan for `embedding` to steer).
    #
    # One file carries one embedding and declares which on its first line, so
    # a run can never hold both a background charge set and a background
    # field: a point file lands on the charges channel, a field file on the
    # potential channel, and neither is ever combined with the other.
    _external_path = str(
        getattr(config.system, 'external_charges', '') or '').strip()
    if not _external_path and bool(
            getattr(config.system, 'read_charge', False)):
        if is_standalone_queue(getattr(config.system, 'theory', []),
                               getattr(config.system, 'reference', []),
                               getattr(config.system, 'highlevel', []),
                               getattr(config.system, 'midlevel', []),
                               getattr(config.system, 'highref', []),
                               getattr(config.system, 'midref', [])):
            _title = str(getattr(getattr(config, 'control', None),
                                 'title', '') or '')
            _external_path = '%s.charge' % _title
    if _external_path:
        from ..data.charges import read_embedding_file
        _document = read_embedding_file(_external_path)
        _primary = next((t for t in (queue.get('theory') or [])
                         if t.get('provides_electronic')), None)
        _reference = queue.get('reference') or []
        _channel = ('external_charges' if _document['kind'] == 'point'
                    else 'external_potential')
        if _document['kind'] == 'point' and _document['role'] == 'atom':
            raise ValueError(
                "%s declares atom-indexed rows, but a standalone run has no "
                "mid shell to map them onto: write 'point <unit> external' "
                "or run the ONIOM queue this file belongs to."
                % _external_path)
        _rows = [list(row) for row in _document['rows']]
        if _primary is not None:
            _primary[_channel] = _rows
        if _reference:
            _reference[0][_channel] = _rows


def _bind_center_plan(state, config):
    """Build the canonical center plan once and persist it on the state.
    """
    # authoritative per-center electronic source used to stamp the plan
    # fingerprint and each task's static property request. Planning has
    # already validated the config, so this only fails for mock configs
    # in tests, where the geometry-only identity fields still stand.
    try:
        center_plan = build_center_plan(config)
    except CenterPlanError as error:
        # A validated production input should not reach this. Mocks in
        # tests do, and they need the geometry-only identity fields; the
        # warning keeps a real failure from disappearing silently.
        get_logger(__name__).warning('center plan build failed: %s', error)
        center_plan = None
    plan_fingerprint = (center_plan.plan_fingerprint
                        if center_plan is not None else None)

    # Persist the canonical plan on the state (docs R4, r4a): it was
    # built once and discarded before; now it is the authoritative
    # per-center electronic source for the whole run, sizing the
    # electronic carrier and seeding the active surface.
    state.bind_partition_plan(center_plan)
    return center_plan, plan_fingerprint


def _merge_skill_binding(maestro, skill_name, descriptor, state, config,
                        raw_config, skills, center_plan, dir_name,
                        qmmm_stage):
    """Merge, validate and profile one skill binding.

    Returns the flattened skill config, the dispatch keywords read before
    the schema pipeline, and the crossing state pair this binding spans
    (None when it spans none).
    """

    # 1. Merge user values into a flat skill dict
    merged = _merge_one_skill_config(
        maestro, skill_name, config=config,
        raw_config=raw_config,
    ) or {}

    # The QMMM plan's field/qmmm capability tokens own the
    # skill profile; a profile written in the LAMMPS section
    # is a plain-run setting and must not override it.
    if qmmm_stage in ('field', 'energy'):
        merged['profile'] = ''

    # Skill jobtype: basic jobtypes are spelled from the
    # per-skill capabilities distributed by planning
    # (fixed canonical token order); advanced jobtypes
    # (md/surface_hopping/...) pass through as-is since
    # the skill plans the job internally.
    _tokens, is_basic, _adv = parse_jobtype(state.jobtype)
    if 'read' in _tokens:
        # A read run has no capability plan: the section itself names the
        # labels, so the skill jobtype is spelled from `read` rather than
        # from the (empty) distributed capability list. The profile
        # override is written before _build_skill_flat_and_submodels()
        # so its profile re-derivation keeps an explicit choice.
        merged['jobtype'] = _read_skill_jobtype(config)
        read_section = getattr(config, 'read', None)
        read_profile = str(getattr(read_section, 'profile', '') or '').strip()
        if read_profile:
            merged['profile'] = read_profile
    elif is_basic:
        skill_jobtype = spell_skill_jobtype(
            descriptor.get('capabilities') or [],
            state.jobtype,
        )
        merged['jobtype'] = skill_jobtype
    elif _spell_advanced_jobtype(state):
        # The framework plans these iterations; the skill is
        # asked for the per-step properties, not for the
        # iteration token. Override the raw token even when a
        # skill section carried a default, so an md/adaptive
        # request reaches the ML potential as 'energy gradient'
        # (and a ktdc run never demands a NAC profile).
        capabilities = descriptor.get('capabilities') or []
        if capabilities:
            merged['jobtype'] = spell_skill_jobtype(
                capabilities, state.jobtype)
        else:
            merged.setdefault('jobtype', state.jobtype)
    else:
        merged.setdefault('jobtype', state.jobtype)

    # The global environment.shell is reserved for the
    # main framework dispatch scripts; skill run scripts
    # carry only the skill-level shell, which arrives via
    # _merge_one_skill_config (skills.<name>.shell of
    # pmd_env.yaml / classic input).

    # A skill-declared initial-guess file (manifest
    # `initial_guess`, resolved by the interface layer) is
    # filled into that skill's own field, and only when the
    # input left it empty. Nothing here names a skill: the
    # declaration carries the skill and the field.
    _guess = (raw_config or {}).get('initial_guess') or {}
    if (_guess.get('skill') == skill_name
            and _guess.get('field')
            and not merged.get(_guess['field'])):
        merged[_guess['field']] = _guess['path']

    # Adjust skill jobtype based on the execution driver.
    # driver=main + optimize: the framework drives the
    #   optimization loop, the skill provides
    #   energy+gradient.
    # driver=skill/skill name: the skill receives the
    #   jobtype as-is and plans the job internally; the
    #   final structure returns through the fin_coord
    #   channel.
    crossing_pair = None
    if state.jobtype == 'optimize' and \
            state.driver == MAIN_DRIVER:
        method_name = state.optimize_method
        if method_name not in OPTIMIZER_DECLARATIONS:
            raise ValueError(
                "Unknown optimize method: %s. Available: "
                "%s (skill-internal optimization is "
                "selected with control.driver, not "
                "optimize.method)"
                % (method_name,
                   list(OPTIMIZER_DECLARATIONS.keys())))
        if method_name not in implemented_optimizers():
            # A registered name that cannot take a step is refused
            # here, where the input is still cheap to change, rather
            # than after the first evaluation of a run that had
            # nothing to step on. That covers both the unwritten
            # placeholders and a written search whose input it has
            # not been run against (each class says which, on the
            # class itself).
            raise ValueError(
                "optimize.method '%s' is registered but not "
                "accepted for a run yet; the searches this input "
                "may ask for are: %s"
                % (method_name, implemented_optimizers()))
        # `optimize_cell` is not refused here: its planning half (the
        # `stress` capability derived above) is owned by this layer
        # and stays in force, and the refusal of the run side belongs
        # where that run is built (`workflow/optimize.py`).
        if (OPTIMIZER_DECLARATIONS[method_name]['needs_initial_hessian']
                and state.hessian_params.get('hess_driver')
                == SKILL_DRIVER_KEYWORD):
            # The curvature this search asks for before its first
            # step is one backend's own force constants. A run with
            # more than one backend has no single answer to "whose
            # matrix", which is the same reason the planner refuses
            # a 'hessian' capability for such a run
            # (`planning/resolver.py`), and the ask would land on
            # whichever task the fan-out reached first. Refused
            # here rather than at the dispatch, where the geometry
            # has already been paid for.
            n_theory = len(skills.get('theory', {}))
            n_reference = len(skills.get('reference', {}))
            if n_theory >= 2 or n_reference >= 2:
                raise ValueError(
                    "hess_driver 'skill' cannot be answered by a "
                    "multi-backend run (%d theory, %d reference): "
                    "one geometry has one force-constant matrix, "
                    "and this input does not say which backend "
                    "produces it. Set hess_driver 'main' to have "
                    "the framework difference the gradients each "
                    "backend reports")
            elif not _declares_a_matrix(skill_name):
                # The other half of the same question. `xtb` and
                # `cp2k` declare energy, gradient and stress and no
                # force constants (`manifest.yaml:capabilities`), so
                # the ask has nothing behind it: the run would pay
                # for a first evaluation and then be answered with
                # whatever the profile happened to write. Refused by
                # name here, where the input is still cheap, exactly
                # as the multi-backend case is.
                raise ValueError(
                    "hess_driver 'skill' asks %s for force "
                    "constants and that skill declares no matrix "
                    "capability (its manifest offers: %s); set "
                    "hess_driver 'main' to have the framework "
                    "difference the gradients, or run this search "
                    "on a skill that declares one (pyscf offers "
                    "'frequency', orca and molcas offer "
                    "'vibrational_frequency')"
                    % (skill_name,
                       ', '.join(_skill_capabilities(skill_name))
                       or 'none'))
        # A crossing search at C>1 follows the sum of the
        # per-center gaps: each center names its own pair, the
        # assembly builds the two composite surfaces
        # (state 0 = first of each pair, state 1 = second) and
        # `meci`/`mecp` run their single-surface machinery on the
        # difference. The NAC blocks stay per center, so the
        # branching plane is the block union.
        if not OPTIMIZER_DECLARATIONS[method_name]['uses_state_pair']:
            # The searches that do not read the pair would pay for a
            # second gradient on every step and never look at it:
            # `_extract_energy_gradient` takes `active_state`, and
            # the framework's Hessian computation produces force
            # constants for one surface only
            # (`workflow/hessian.py`). Refused rather than ignored.
            _state_i = state.optimize_params.get(
                'opt_state_i', 0)
            _state_j = state.optimize_params.get(
                'opt_state_j', 0)
            # A per-center pair is a tuple/list; the one-surface
            # searches care only about "is a second surface
            # named", so the pair form counts as one.
            if _state_j != _state_i:
                raise ValueError(
                    "optimize.method '%s' reads one surface: it "
                    "optimizes the gradient of opt_state_i=%s, and "
                    "the framework produces no force constants or "
                    "energy difference for a second one. Naming "
                    "opt_state_j=%s asks for a pair only the "
                    "crossing searches use, so with this method it "
                    "would cost an extra gradient every step and be "
                    "discarded"
                    % (method_name, _state_i, _state_j))
        # Spelled, not written as a literal: the step asks for
        # energy and gradient, but the task's own duties ride
        # along, and a QMMM field/energy task is selected by
        # them. Dropping them resolves the classical provider to
        # a plain force-field profile carrying qm_atoms, which
        # its own validation refuses.
        # The need is the planner's: `optimize.method` is scanned
        # with every other requirement keyword, so a crossing
        # search that spans its plane on the derivative coupling
        # already carries the 'nac' capability in the request.
        # What is decided here is whether this backend can answer
        # it: a skill that declares no coupling has no profile for
        # the token, and the search then runs on the gradient
        # difference alone (a documented approximation, not a
        # silent zero).
        request = list(descriptor.get('capabilities') or [])
        from .resolver import crossing_coupling_pair
        crossing_pair = crossing_coupling_pair(
            dict(state.optimize_params,
                 driver=getattr(state, 'driver', 'main'),
                 jobtype='optimize'))
        if crossing_pair is not None and not (
                _declares_a_channel(skill_name, 'nac')):
            request = [cap for cap in request if cap != 'nac']
            crossing_pair = None
        if crossing_pair is not None:
            # The skill refuses to build a deck that asks for a
            # coupling on a one-state table, and a nested input
            # leaves `system.nstate` at its default while its own
            # `ci`/`spin` declare more states - the plan derives
            # the table the run computes, so hand the leg that one.
            c = descriptor.get('center_index')
            if (center_plan is not None and c is not None
                    and c < center_plan.n_center):
                es = center_plan.center(c).electronic_spec
                if es.state_spin:
                    merged['state_spin'] = list(es.state_spin)
                    merged['nstate'] = int(es.nstate)
        merged['jobtype'] = spell_skill_jobtype(
            request, 'energy gradient')

        # Forward the optimized state to the skill so
        # its gradient array covers that state. The
        # index is 0-based and identical to state_i;
        # an explicit user target_states is kept as-is
        # (validated by the skill itself).
        raw_state_i = state.optimize_params.get(
            'opt_state_i', 0)
        # A per-center pair is read where the duties are built;
        # this forwarding speaks the scalar skill field, so a
        # sequence is handed over to nothing here.
        if isinstance(raw_state_i, (list, tuple)):
            raw_state_i = 0
        state_i = int(raw_state_i or 0)
        if state_i > 0 and 'target_states' not in merged:
            merged['target_states'] = [state_i]

    # Extract dispatch-layer keywords
    # (multitask/remotetask/multipool) before the skill
    # schema pipeline validates the merged dict. These are
    # NOT skill schema fields; they belong to the dispatch
    # layer and are read by the planner from the raw task
    # dict.
    dispatch_flags = {
        'multitask': merged.get('multitask'),
        'remotetask': merged.get('remotetask'),
        'multipool': merged.get('multipool'),
    }

    # 2. Fill defaults and resolve profile via query
    return _build_skill_flat_and_submodels(
        dir_name, merged,
        driver=getattr(state, 'driver', 'main'),
    ), dispatch_flags, crossing_pair


def _assemble_task(descriptor, skill_name, backend_class, skill_flat,
                  dispatch_flags, config, state, dir_name, center_plan,
                  plan_fingerprint, crossing_pair, category, seen_bindings):
    """Turn one resolved descriptor into a static task on the state.

    The task carries its own config copy: tasks sharing a skill must not
    mutate each other's flat config, and the plan-owned keywords (region
    slice, QMMM partition, center electronic structure) are per task.
    """

    # 3. Build task-specific config dict (independent per
    # task: deep copy so tasks sharing the same skill do
    # not mutate each other's flat config)
    task_config = {
        'skill_config': deepcopy(skill_flat),
        'results': {},
    }

    # Whole-system stress request (dev/methods/
    # ENSEMBLE_DEVELOPMENT.md section 8): planning derives the
    # 'stress' capability from md+ensemble=npt or
    # optimize+optimize_cell. md/optimize are advanced jobtypes
    # that bypass spell_skill_jobtype, so the request reaches
    # the skill through this boolean config flag (orthogonal to
    # the profile: the skill still runs its gradient profile and
    # additionally emits/reads the stress tensor).
    if 'stress' in (descriptor.get('capabilities') or []):
        task_config['skill_config']['stress'] = True

    # GFN-FF topology continuation. A dynamics run wants the bond list
    # built at its first step to survive into every later step, because
    # a bond that flips mid-trajectory silently changes the surface the
    # next force belongs to. A geometry search wants the opposite: each
    # trial structure must be scored with the topology of that
    # structure. The fixed calculation directory cannot tell the two
    # apart on its own - measured on round tmp_20261017_234656's
    # variable-cell leg, where a carried `gfnff_topo` scored a sane
    # geometry +0.3508098 Hartree high in a controlled rerun, and the
    # run's rows 6-21 read +0.20 while the same frames evaluate fresh
    # to -0.929 - so the framework says which shape this run is.
    _jt_tokens, _jt_basic, _jt_adv = parse_jobtype(state.jobtype)
    if 'md' in _jt_tokens:
        task_config['skill_config']['keep_topology'] = True

    # Region geometry (docs MULTICENTER 6.2 line 214): a
    # sub-region task's backend is built from skill_config, and
    # the skill normalize derives natom from skill_config.atoms
    # -- which then sizes every natom-keyed result array
    # (gradient/nac). The runtime slices the live geometry to
    # the region before calculate() (calculator._build_work_items
    # -> materialize_region), so the static skill_config must
    # carry the SAME region atoms; otherwise the skill allocates
    # a full-system (natom, 3) array and the region result fails
    # to broadcast into it. L(real)/'all' tasks (empty indices)
    # keep the full system. natom is set explicitly so it stays
    # the region size even when a source field is absent.
    _region_idx = list(descriptor.get('region_indices') or [])
    if _region_idx and descriptor.get('region', 'all') != 'all':
        _skill_cfg = task_config.get('skill_config')
        if isinstance(_skill_cfg, dict):
            _sys = config.system
            # The region is cut out of the continuous view, exactly
            # as the runtime slices the live geometry
            # (calculator._build_work_items -> materialize_region),
            # so the static config and the coordinates the backend
            # is called with describe the same connected cluster.
            _geometry = list(getattr(_sys, 'coordinates', []) or [])
            if _geometry:
                _geometry = continuous_coordinates(
                    _geometry, _sys).tolist()
            for _fld, _src in (('atoms', getattr(_sys, 'atoms', [])),
                               ('mass', getattr(_sys, 'mass', [])),
                               ('coordinates', _geometry),
                               ('velocities',
                                getattr(_sys, 'velocities', [])),
                               # The crossing counts travel with the
                               # geometry they were accrued for: the
                               # skill checks one against the other by
                               # atom count, and a region task handed
                               # the global rows is asked about atoms
                               # it does not have.
                               ('image',
                                getattr(_sys, 'image', []))):
                _val = list(_src or [])
                if _val:
                    _skill_cfg[_fld] = [_val[i] for i in _region_idx]
            _skill_cfg['natom'] = len(_region_idx)
            # A sliced region is a molecular subsystem: it is handed
            # no cell (the runtime keeps `task_cell` None for it) and
            # it must not be told the run is periodic, or a
            # non-periodic center backend would refuse a run whose
            # box its outermost layer carries (the 2026-10-08
            # capability ruling; the skill-side check stays the
            # standalone/agent-entry backstop it was written to be).
            _skill_cfg['pbc'] = [0, 0, 0]
            _skill_cfg['cell'] = list(DEFAULT['cell'])
            _skill_cfg['lattice_vector'] = [[0.0] * 3 for _ in range(3)]
            # Under a partition the framework owns the embedding
            # file: it reads '{title}.charge' once, routes the rows
            # through the charge store or the MM field stage, and
            # hands each step the values for its own region. A skill
            # that also opened the file would take the frozen
            # coordinates of an atom-indexed document, or a static
            # copy of a field the composite recomputes every step.
            if _skill_declares(dir_name, 'read_charge'):
                _skill_cfg['read_charge'] = 'none'
            # The rows this task computes, in the order the geometry is
            # handed (ascending: schema/adapter.py canonicalises the
            # center here upstream): a classical backend reads a topology file it does
            # not generate, so a region task has to say which rows of
            # the user's system-level template survive
            # (skills/lammps_skill, `atom_selection`). The declared-key
            # test is what keeps a QM task untouched: writing a keyword
            # its schema never lists would leave a field in the task
            # config that the skill then drops in silence, the failure
            # mode this project has already ruled out for new keywords.
            if _skill_declares(dir_name, 'atom_selection'):
                _skill_cfg['atom_selection'] = [index + 1
                                                for index in _region_idx]

    # QMMM plan-owned configuration (dev/methods/QMMM_DEVELOPMENT.md section 2
    # section 5; dev/skills/LAMMPS_SKILL_DEVELOPMENT.md section
    # 12). The plan adds `field`/`qmmm` capability tokens to the
    # field/energy tasks; the LAMMPS skill resolves its own
    # `qmmm_field`/`qmmm_energy` profile from those tokens. The
    # framework only hands over the partition (`qm_atoms`), the
    # QM 0-order charges (`q_qm`) and the field handoff's
    # `keep_tmp`. The QM provider runs the sliced region under
    # Molcas ESPF potential embedding; its per-step
    # external_potential comes from the field task. None of these
    # keywords is a user-facing switch; the plan owns them.
    if descriptor.get('composite') == 'qmmm':
        _skill_cfg = task_config.get('skill_config')
        if not isinstance(_skill_cfg, dict):
            raise ValueError(
                "QMMM task '%s' has no flat skill config."
                % descriptor.get('task_id'))
        _partition = [int(index) for index in
                      (descriptor.get('partition_indices') or [])]
        _qmmm_role = descriptor.get('qmmm_role')
        if _qmmm_role in ('field', 'energy'):
            if not (_skill_declares(dir_name, 'qm_atoms')):
                raise ValueError(
                    "QMMM MM provider '%s' does not declare "
                    "qm_atoms; the partition cannot be handed over."
                    % skill_name)
            _skill_cfg['qm_atoms'] = list(_partition)
            if _qmmm_role == 'field':
                # Which center this probe measures. Empty is the
                # single-center case and stays unwritten, so a
                # one-center run hands the skill exactly the
                # keywords it has always seen.
                _probe = [int(index) for index in
                          (descriptor.get('probe_indices') or [])]
                if _probe:
                    if not _skill_declares(dir_name, 'qm_probe'):
                        raise ValueError(
                            "QMMM MM provider '%s' does not declare "
                            "qm_probe; the probe cannot be narrowed "
                            "to one center." % skill_name)
                    _skill_cfg['qm_probe'] = _probe
                # The field probe sets Q=+1 and ignores q_qm; it
                # only writes the atomic-unit .esfp handoff the
                # QM stage reads in the V/E mode, so it must
                # survive the backend's own keep_tmp handling.
                _skill_cfg['keep_tmp'] = 1
            else:
                # The center tiling the classical surface removes
                # the in-cell Q-Q pairs by. Omitted for one center,
                # where the QM set already is one group.
                _high = list(
                    getattr(config.system, 'highlevel', []) or [])
                if (len(_high) > 1
                        and isinstance(_high[0], (list, tuple))):
                    if not _skill_declares(dir_name, 'qm_centers'):
                        raise ValueError(
                            "QMMM energy provider '%s' does not "
                            "declare qm_centers; the in-cell Q-Q "
                            "removal cannot be narrowed per center."
                            % skill_name)
                    _skill_cfg['qm_centers'] = [
                        [int(index) for index in group]
                        for group in _high]
                # The LAMMPS qmmm_energy switch: the QM part owns
                # the QM-MM electrostatic correction only when the
                # QM calculation is embedded.
                _skill_cfg['embedding'] = bool(
                    getattr(config.system, 'embedding', False))
                _q_qm = [float(value) for value in
                         (getattr(config.system, 'q_qm', []) or [])]
                if _q_qm:
                    if not _skill_declares(dir_name, 'q_qm'):
                        raise ValueError(
                            "QMMM energy provider '%s' does not "
                            "declare q_qm." % skill_name)
                    _skill_cfg['q_qm'] = list(_q_qm)
        elif _qmmm_role in ('qm', 'qm_charge'):
            # The cheap charge round of multi-center QMMM is the
            # same embedded solve with a narrower jobtype: its
            # capabilities hold no gradient token, so the skill
            # spells an energy-only deck and the ALASKA module is
            # never emitted, while the &ESPF block that carries
            # the 0-order charges stays in place.
            for _key in ('embedding', 'read_charge'):
                if not _skill_declares(dir_name, _key):
                    # Zero coupling: a QM provider that does not
                    # read the ESPF keyword simply ignores the
                    # field. The framework still offers the rows
                    # through the task request; it never refuses
                    # the term because the skill's present keyword
                    # vocabulary is smaller.
                    continue
                _embedding = bool(
                    getattr(config.system, 'embedding', False))
                # QMMM theory[0] uses the field V/E representation;
                # charge embedding is ONIOM-only.
                _skill_cfg[_key] = (
                    ('field' if _embedding else 'none')
                    if _key == 'embedding' else 'none')
            for _key in ('point_charges', 'external_potential'):
                if _skill_declares(dir_name, _key):
                    _skill_cfg[_key] = []
            # The sliced region is the whole QM system for this
            # task; no inner QM/MM partition remains.
            _skill_cfg['highlevel'] = []
            _skill_cfg['midlevel'] = []
            # Which methods may run ESPF embedding is the skill's
            # own rule (`molcas_skill/schema/validate.py::
            # _validate_embedding_for_profile`, run by the backend
            # before it builds a deck). The framework used to
            # restate the vocabulary here; a method list that is
            # the engine's business belongs with the engine.
        else:
            raise ValueError(
                "QMMM task '%s' carries an unknown qmmm_role %r."
                % (descriptor.get('task_id'), _qmmm_role))

    # Region electronic structure (docs MULTICENTER 6.2/8, and
    # the R4 per-task override contract in docs section 6.2
    # line 214): for a multi-center run each task's backend
    # must see its OWN center as a flat single-center system,
    # never the global per-center (nested) lists. The framework
    # Config carries nested ci/spin/coupling/highlevel for C>1;
    # the skill schema is flat-only, so these are projected here
    # before the backend validates its config.
    #
    # The full-system baseline L(real) (center_index None) uses
    # the plan aggregate charge/multiplicity and a single-center
    # ground-state electronic spec; every center-region task
    # (high H(c), its low correction L(c), and the three-layer
    # mid M(I_c)/L(I_c) which inherit the active center) uses
    # that center's charge/multiplicity and ElectronicSpec.
    # highlevel/midlevel are cleared: the runtime slices the
    # region geometry (calculator._build_work_items ->
    # materialize_region) and passes it to backend.calculate(),
    # so the whole sliced subsystem is the QM region and there
    # is no inner QM/MM partition. Injected into the task's
    # skill_config so the first calculation already carries the
    # region values; C=1 keeps the historical merge/default
    # path untouched.
    region_charge = descriptor.get('region_charge')
    region_multiplicity = descriptor.get('region_multiplicity')
    if center_plan is not None and center_plan.n_center > 1:
        _c = descriptor.get('center_index')
        _skill_cfg = task_config.get('skill_config')
        if _c is None:
            region_charge = center_plan.system_charge
            region_multiplicity = center_plan.system_multiplicity
            electronic = {
                'nstate': 1,
                'ci': [1],
                'spin': [0],
                'coupling': [],
                'state_spin': [],
                'nac_pair_map': [],
                'soc_pair_map': [],
                'current_state': 0,
            }
        elif _c < center_plan.n_center:
            _center = center_plan.center(_c)
            region_charge = _center.charge
            region_multiplicity = _center.multiplicity
            _spec = _center.electronic_spec
            electronic = {
                'nstate': _spec.nstate,
                'ci': list(_spec.ci),
                'spin': list(_spec.spin),
                'coupling': [list(p) for p in _spec.coupling],
                'state_spin': list(_spec.state_spin),
                'nac_pair_map': [list(p)
                                 for p in _spec.nac_pair_map],
                'soc_pair_map': [list(p)
                                 for p in _spec.soc_pair_map],
                'current_state': _spec.initial_root,
            }
        else:
            electronic = None
        if isinstance(_skill_cfg, dict):
            if region_charge is not None:
                _skill_cfg['charges'] = region_charge
            if region_multiplicity is not None:
                _skill_cfg['multiplicity'] = region_multiplicity
            if electronic is not None:
                _skill_cfg.update(electronic)
                # The sliced region is the whole QM system for
                # this task; no inner QM/MM region remains.
                _skill_cfg['highlevel'] = []
                _skill_cfg['midlevel'] = []

    # 4. Register the static task to PyRAIState
    # (backend instances are created at runtime by
    # Calculator.initialize(), not here)
    # Per-task skill binding id: distinguishes different
    # configs of the same skill (docs 6.2); unique because
    # task_id is unique within the category.
    skill_binding_id = "%s::%s" % (
        skill_name, descriptor['task_id'])
    binding_key = (category, skill_binding_id)
    if binding_key in seen_bindings:
        raise ValueError(
            "Skill binding collision: '%s' is claimed by two "
            "%s tasks; multi-binding requires a unique "
            "binding id per task (docs 6.2)."
            % (skill_binding_id, category))
    seen_bindings.add(binding_key)

    # Hand the skill the framework's execution decision, and
    # only when the framework really is going remote: a local
    # run keeps whatever the skill section or pmd_env.yaml
    # declared, which is how the skill-side self-submission
    # switch stays in the tester's hands.
    _skill_cfg = task_config.get('skill_config')
    if isinstance(_skill_cfg, dict) and \
            _dispatch_requests_remote(config, dispatch_flags):
        declared = _skill_cfg.get('execution_mode')
        _skill_cfg['execution_mode'] = 'local'
        if declared and str(declared).lower() != 'local':
            get_logger(__name__).warning(
                "remotetask is set for skill '%s', so the "
                "framework submits this calculation itself; the "
                "declared execution_mode '%s' becomes 'local'. "
                "A skill that self-submits inside a remote "
                "runtime queues the same job twice.",
                skill_name, declared)

    task = {
        # Carry every descriptor field first, then override the ones this
        # layer computes. A field planning adds is preserved without a
        # second whitelist to keep in step.
        **descriptor,
        'role': descriptor['role'],
        'task': descriptor['task'],
        'task_id': descriptor['task_id'],
        'task_slot': descriptor.get('task_slot'),
        'layer_id': descriptor['layer_id'],
        'static_id': descriptor['static_id'],
        'contribution': descriptor['contribution'],
        'region': descriptor['region'],
        'region_indices': list(descriptor['region_indices']),
        'region_charge': region_charge,
        'region_multiplicity': region_multiplicity,
        'skill_name': skill_name,
        'api': backend_class,
        'capabilities': descriptor['capabilities'],
        'config': task_config,
        # Dispatch-layer keywords (not validated by skill
        # schema; preserved from raw input for the planner)
        'multitask': dispatch_flags.get('multitask'),
        'remotetask': dispatch_flags.get('remotetask'),
        'multipool': dispatch_flags.get('multipool'),
        # Unified center identity + electronic binding
        # (docs 3.1 / 6.2). center_index is None only for the
        # full-system baseline L(real); a plain single-method
        # task binds center 0.
        'center_index': descriptor.get('center_index'),
        'region_id': descriptor.get('region_id'),
        'provides_electronic': descriptor.get(
            'provides_electronic', False),
        # A cheap charge-only round of multi-center QMMM declares
        # itself a charge provider and not an electronic one: the
        # round must survive the task rebuild without ever being
        # picked up as a center's electronic source.
        'provides_charges': descriptor.get('provides_charges', False),
        'electronic_binding_id': descriptor.get(
            'electronic_binding_id'),
        'state_policy': descriptor.get('state_policy'),
        'reference_state': descriptor.get('reference_state'),
        'transform_spec': descriptor.get('transform_spec'),
        'skill_binding_id': skill_binding_id,
        'plan_version': descriptor.get('plan_version', 0),
        'plan_fingerprint': plan_fingerprint,
        # QMMM plan identity: the composite marker, the physical
        # duty of this task (qm/mm) and the partition. They travel
        # with the task so the calculator can run the MM stage
        # before the QM stage and the assembler can stitch by
        # duty, without re-deriving the scheme from the queue.
        'composite': descriptor.get('composite'),
        'qmmm_role': descriptor.get('qmmm_role'),
        'binding_role': descriptor.get('binding_role'),
        'partition_indices': list(
            descriptor.get('partition_indices') or []),
        # Which center a field probe measures (empty = the whole QM
        # set, the single-center spelling). The energy provider needs
        # the partition; the probe needs its own center, and without
        # this key every center of a multi-center run would be
        # measured over the same union.
        'probe_indices': list(
            descriptor.get('probe_indices') or []),
        # Adaptive label correspondence: which reference layer (or
        # the composite total) labels the neural-network surface,
        # and which auxiliary channels the reference plan
        # produces. Planning owns it; the assembler and the
        # adaptive workflow only read it.
        'label_plan': descriptor.get('label_plan'),
        # Background rows for a standalone embedding; only the
        # primary theory/reference task carries them, and a file
        # declares one kind - charges or a field - so at most one of
        # the two is ever set.
        'external_charges': descriptor.get('external_charges'),
        'external_potential': descriptor.get('external_potential'),
        # Per-task static property request (docs 6.3): the
        # provider H(c) reads its own center, a correction
        # reads only its reference_state.
        'property_request_spec': _task_property_request(
            descriptor, center_plan, crossing_pair=crossing_pair),
    }
    task.pop('backend', None)   # descriptor spelling; the task uses 'api'
    state.add_task(category, task)


def _check_curvature_plan(state):
    """Apply the search's curvature rules once per optimize plan.

    Both rules read the state's constraint plan and write the derived
    atom list, so they are queue-level: evaluating them inside the
    per-binding loop asked the same question of the same geometry again
    for every backend.

    A mode-following search is searchable on a multi-center composite:
    the run is one surface over the whole system's 3N coordinates
    (E = sum_c E_QM(c) + E_classic), the numerical matrix is taken by
    displacing global atoms, and the curvature helpers read the global
    constraint mask, not a center.
    """
    if state.jobtype != 'optimize' or state.driver != MAIN_DRIVER:
        return
    method_name = state.optimize_method
    declaration = OPTIMIZER_DECLARATIONS.get(method_name)
    if declaration is None or not declaration['implemented']:
        # The per-binding check refuses these with the message the input
        # needs; none of the curvature rules can be evaluated before it.
        return
    if declaration['needs_initial_hessian']:
        _validate_curvature_freezes(state, method_name)
        _derive_curvature_atoms(state, method_name)


_SKILL_DECLARED_KEYS = {}


def _skill_declares(dir_name, key: str) -> bool:
    """Whether one skill's own schema lists a configuration keyword.

    The authority is the skill's `query.get_defaults()`, the same mapping
    :func:`_build_skill_flat_and_submodels` fills defaults from, so a keyword
    the skill never registers is never written into its task config by the
    framework. Note that an empty list is not materialized into the merged
    dict by that step, so the presence of a key in the task config says
    nothing about whether the skill has the field.

    Parameters
    ----------
    dir_name : str
        Skill package directory name (e.g. 'lammps_skill').
    key : str
        Keyword name.

    Returns
    -------
    bool
        True when the skill declares it. Cached per skill and keyword.
    """
    if not dir_name:
        return False

    declared = _SKILL_DECLARED_KEYS.get(dir_name)
    if declared is None:
        try:
            query = importlib.import_module(skill_module_path(dir_name))
            declared = set(query.get_defaults())
        except (ImportError, AttributeError):
            declared = set()
        _SKILL_DECLARED_KEYS[dir_name] = declared
    return key in declared


def _build_skill_flat_and_submodels(dir_name, merged, driver='main'):
    """Fill skill defaults and resolve profile via query mechanism.

    The framework does NOT run the skill schema pipeline
    (normalize/validate). The merged flat dict is passed to the
    backend, which runs its own validation.

    Parameters
    ----------
    dir_name : str
        Skill package directory name (e.g. 'molcas_skill').
    merged : dict
        Merged flat skill configuration.
    driver : str
        Control driver of the parent state ('main' or a skill
        driver); forwarded to resolve_profile so skill-driven
        jobtypes select the matching profile.

    Returns
    -------
    dict
        Skill flat dict with defaults filled and profile resolved.
    """
    if not dir_name:
        return dict(merged)


    base = skill_module_path(dir_name, '')

    # Step 1: Fill missing/empty fields from skill defaults (via query)
    skill_query = None
    try:
        skill_query = importlib.import_module(f'{base}.query')
        skill_defaults = skill_query.get_defaults()
        for key, val in skill_defaults.items():
            current = merged.get(key)
            if current is None or current == '' or current == []:
                if val is not None and val != '' and val != []:
                    merged[key] = val
    except (ImportError, AttributeError):
        pass

    # Step 2: Resolve profile via query (method now available from defaults)
    method = str(merged.get('method', ''))
    user_profile = str(merged.get('profile', '') or '')
    # Determine the default profile from skill defaults for priority check
    default_profile = ''
    if skill_query is not None:
        try:
            default_profile = str(
                skill_query.get_defaults().get('profile', '') or ''
            )
        except (ImportError, AttributeError):
            pass
    needs_profile = (not user_profile) or (user_profile == default_profile)

    if needs_profile and method:
        try:
            jobtype = str(merged.get('jobtype', 'energy'))
            if skill_query is not None and hasattr(skill_query, 'resolve_profile'):
                # Every skill takes the driver (#466, 4.3): a driver=skill
                # iteration is refused by the skill's own table, and the
                # framework has nothing to fall back to.
                merged['profile'] = skill_query.resolve_profile(
                    method, jobtype, driver)
        except (ImportError, AttributeError, ValueError) as error:
            if isinstance(error, ValueError) and driver != MAIN_DRIVER and (
                    ENGINE_OWNED_ITERATION_JOBTYPES
                    & set(str(merged.get('jobtype', '')).lower().split())):
                # The profile table answered and said no, on the route that
                # hands the iteration to the skill. A warning here would let
                # the run proceed as a single point wearing the iteration's
                # request and be recorded as a finished calculation, so the
                # skill's refusal reaches the caller unchanged.
                raise
            # Otherwise the skill's own defaults stay in charge, but the
            # substitution is reported: a request for one task quietly served
            # by another profile (a training job run as a property step, or
            # the other way round) reads as a finished calculation either way.
            get_logger(__name__).warning(
                "Profile could not be resolved for skill '%s' "
                "(method='%s', jobtype='%s', driver='%s'): %s. "
                "Keeping profile '%s'.",
                dir_name, method, merged.get('jobtype', 'energy'),
                driver, error, merged.get('profile', ''))

    return merged
