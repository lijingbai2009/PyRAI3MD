"""
======================================================================
This module provides the task queue resolution logic for the planning
layer. It expands resolved skills into a Task Queue where every task
represents one independent backend invocation.

Composite workflows (ONIOM, Fragment, QM/MM) break the assumption
that one role maps to one backend call. resolve_task_queue expands
such workflows into multiple tasks carrying role, task index and
contribution coefficient, so the Calculator can execute the queue
sequentially and the Composite can aggregate results as
sum(contribution x task_result) without knowing the workflow type.

Task descriptors carry region slice indices only (no geometry data).
The Calculator derives atoms/coordinates from the live structure at
execution time, so iterative workflows keep geometry up to date.

Aug 03 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import math

# Capabilities allowed for composite non-primary tasks.
#
# One definition, imported from the module that distributes capabilities in the
# first place. This file used to carry its own `{'energy', 'gradient'}`, and the
# two drifted apart the day 'stress' joined the planning side's set for the
# reason stated there - the multiscale L(real) baseline is a non-primary task
# and still has to be able to report the box force. With the local copy, every
# whole-system low-layer task lost the stress request and a multi-center NPT
# run integrated its barostat on the kinetic term alone (measured:
# `Pvir_bar = -0.000000`).
from .resolver import _BASIC_CAPABILITIES

# The one property in that set that is not a per-atom contribution. A region
# slice - any task whose geometry is a cut-out of the system - has no cell to
# take a stress of, so it is asked for energies and gradients only, and the
# whole-system task is its single source (`PyRAIState.active_stress()`,
# `resolve_system_task`, composite.py's `SYSTEM_STRUCTURAL_KEYS`).
_SYSTEM_ONLY_CAPABILITIES = frozenset({'stress'})


def _task_capabilities(capabilities, region_indices):
    """The capabilities of one task, given the region it computes.

    Parameters
    ----------
    capabilities : list of str
        What the layer's skill was given for this run.
    region_indices : list of int
        The task's region slice; empty means the whole system.

    Returns
    -------
    list of str
        Sorted capabilities, with the whole-system-only properties removed
        from a sliced task.
    """
    if region_indices:
        return sorted(c for c in capabilities
                      if c not in _SYSTEM_ONLY_CAPABILITIES)
    return sorted(capabilities)


def _skill_short_name(skill_name):
    """Return the short skill name without the '_skill' suffix.

    Used to build compact static execution identifiers
    (e.g. 'pyscf' instead of 'pyscf_skill').

    Parameters
    ----------
    skill_name : str
        Full skill identifier, e.g. 'pyscf_skill'.

    Returns
    -------
    str
        Short skill name.
    """
    name = str(skill_name or '')
    if name.endswith('_skill'):
        return name[:-len('_skill')]
    return name


# Region label mapping for human-facing identifiers (the static_id
# directory name and the &composite energy breakdown). The ONIOM
# geometric regions map to the layer names users reason about; any
# other region label ('all', 'fragment', ...) passes through.
REGION_LABELS = {
    'model': 'high',
    'inter': 'mid',
    'real': 'low',
}


def region_label(region):
    """Map a task region label to its human-facing layer name.

    'model' -> 'high', 'inter' -> 'mid', 'real' -> 'low'; any other
    region ('all', 'fragment', ...) is returned unchanged. Shared by
    the static_id directory naming (planning) and the &composite
    energy breakdown (workflow/composite) so both speak one taxonomy.

    Parameters
    ----------
    region : str or None
        Task region label.

    Returns
    -------
    str
        Human-facing layer name.
    """
    return REGION_LABELS.get(region, region)


def _compose_static_id(task, single=False, ordinal=None):
    """Compose the static execution identifier for one task.

    The identifier doubles as the backend calculation directory
    name. Single-shot jobs keep the plain working-directory
    convention of the standalone skill mode; composite/multitask
    jobs are named by REGION, ROLE and (when several tasks share
    them) a 1-based ordinal, then the skill tag:

        static_id = tmp_{SKILL_SHORT}                            (single task)
        static_id = tmp_{region}_{role}[_task{N}]_{SKILL_SHORT}  (composite)

    region is the human-facing layer name (model->high, inter->mid,
    real->low; see region_label). The _task{N} ordinal is present only
    when several tasks share the same (region, role, skill) -- a
    multi-center ONIOM level or a multi-group fragment -- and equals
    the 1-based position in that group (center_index + 1 for ONIOM, so
    center 0 is task1). A lone task in its group (a single-center run,
    the center-less full-system baseline L(real)) omits the ordinal.
    This replaces the former 'tmp_{task_id}_layer{layer_id}_{SKILL}'
    form whose layer number was opaque, whose role duplicated the
    skill tag, and whose per-role running index left the first center
    unnumbered while later centers carried _task1/_task2.

    Examples: 'tmp_MOLCAS', 'tmp_high_qm_task1_OPENQP',
    'tmp_mid_qm2_XTB', 'tmp_low_qm3_XTB'.
    It depends only on static planning information (task region,
    role, selected skill) and never changes during execution.

    Parameters
    ----------
    task : dict
        Task descriptor carrying role, region, skill_name.
    single : bool
        Whether the queue holds exactly one task (no prefix).
    ordinal : int or None
        1-based position within the (region, role, skill) group when
        that group holds several tasks; None omits the _task{N} part.

    Returns
    -------
    str
        Static execution identifier.
    """
    skill_tag = _skill_short_name(task['skill_name']).upper()
    if single:
        return "tmp_%s" % skill_tag
    parts = ['tmp', region_label(task.get('region')),
             str(task.get('role') or 'task')]
    if ordinal is not None:
        parts.append('task%d' % int(ordinal))
    parts.append(skill_tag)
    return '_'.join(parts)


def resolve_composite_type(config, skills):
    """Determine the composite workflow type from skills and config.

    A two-method queue is selected explicitly by
    ``control.multiscale_type`` rather than inferred from the skill
    roles: ``'qmmm'`` asks for the additive QM/MM scheme, while an empty
    value or ``'oniom'`` keeps the historical ONIOM2 meaning. The other
    branches are unchanged:

    - multi-center highlevel/midlevel detected -> multi_center_oniom{2,3}
    - system.center_groups non-empty  -> fragment
    - three theory skills             -> oniom3 (always)
    - two theory skills               -> qmmm when
                                         control.multiscale_type == 'qmmm',
                                         otherwise oniom2
    - otherwise                       -> None (single-shot)

    ``multiscale_type='qmmm'`` with a nested ``molecule.highlevel`` is the
    multi-center QM/MM scheme and returns ``'multi_center_qmmm'``; the flat
    spelling keeps returning ``'qmmm'`` so the single-center task ids and
    the static directory names derived from them never move.

    ``system.qmmm_key`` no longer selects QMMM; it is refused with the
    new spelling (the old keyword had no ownership semantics and the
    reference queue it used was never executed).

    Parameters
    ----------
    config : PyRAIConfig
        Validated framework configuration.
    skills : dict
        Resolved skills {'theory': {...}, 'reference': {...}}.

    Returns
    -------
    str or None
        Composite type identifier, or None for single-shot jobs.
    """
    theory = skills.get('theory', {})
    reference = skills.get('reference', {})

    system = getattr(config, 'system', None)
    control = getattr(config, 'control', None)
    multiscale = str(
        getattr(control, 'multiscale_type', '') or ''
    ).strip().lower()
    jobtype = str(getattr(control, 'jobtype', '') or '').strip().lower()

    # QMMM is decided first so its own gates (two theory methods, the QM
    # regions, no reference queue) refuse a mis-specified input with a
    # QMMM message instead of a misleading ONIOM/PBC one.
    if multiscale == 'qmmm':
        _validate_qmmm_selection(system, theory, reference, jobtype)
        highlevel = getattr(system, 'highlevel', []) or []
        if highlevel and isinstance(highlevel[0], (list, tuple)):
            return 'multi_center_qmmm'
        return 'qmmm'

    qmmm_key = getattr(system, 'qmmm_key', None) if system is not None else None
    if qmmm_key:
        raise ValueError(
            "system.qmmm_key is retired as a QMMM declaration: write "
            "control.multiscale_type qmmm with exactly two theory methods "
            "(QM first, MM last) and molecule.highlevel for the QM region.")

    groups = list(getattr(system, 'center_groups', []) or [])

    # Multi-center detection: check highlevel/midlevel for nested lists
    highlevel = getattr(system, 'highlevel', []) or []
    midlevel = getattr(system, 'midlevel', []) or []
    n_centers = _count_centers(highlevel)

    # Adaptive sampling carries theory and reference as workflow roles: the
    # theory queue runs the MD and the reference queue labels the selected
    # structures. A layered theory queue is still ONIOM; the reference is
    # not part of that sum and must not trigger the generic qmmm branch.
    if jobtype.split() and jobtype.split()[0] == 'adaptive':
        if len(theory) <= 1:
            return None
        if n_centers > 1:
            raise ValueError(
                "adaptive sampling supports one ONIOM centre; got %d."
                % n_centers)
        if len(theory) == 2:
            if not highlevel:
                raise ValueError(
                    "adaptive sampling with two theory methods needs a "
                    "molecule.highlevel region (ONIOM2).")
            return 'oniom2'
        if len(theory) == 3:
            if not midlevel:
                raise ValueError(
                    "adaptive sampling with three theory methods needs "
                    "molecule.midlevel (ONIOM3).")
            return 'oniom3'
        raise ValueError(
            "adaptive sampling supports one, two or three theory methods; "
            "got %d." % len(theory))

    if n_centers > 1:
        # Multi-center ONIOM detected
        if groups or qmmm_key:
            raise ValueError(
                "Multi-center ONIOM conflicts with center_groups/qmmm "
                "settings; use one composite scheme at a time.")
        if len(theory) == 3:
            return 'multi_center_oniom3'
        if len(theory) == 2:
            return 'multi_center_oniom2'
        raise ValueError(
            "Multi-center ONIOM requires 2 or 3 theory skills, "
            "got %d." % len(theory))

    if groups:
        return 'fragment'
    if len(theory) == 3:
        return 'oniom3'
    if len(theory) == 2:
        # multiscale_type is '' or 'oniom' here; 'qmmm' returned above.
        return 'oniom2'
    return None


def _validate_qmmm_selection(system, theory, reference, jobtype):
    """Validate the QMMM selector and its partition before expansion.

    ``control.multiscale_type='qmmm'`` with exactly two theory methods is
    the single spelling of the additive QM/MM scheme: theory[0] is the
    QM provider (Molcas ESPF external field) and theory[1] is the MM
    provider (LAMMPS qmmm, which owns the classical surface and the
    cell). The reference queue is never executed, so it must be empty.

    Parameters
    ----------
    system : schema.config.System or None
        System submodel carrying highlevel/center_groups/constraints.
    theory : dict
        Resolved theory bindings.
    reference : dict
        Resolved reference bindings.
    jobtype : str
        Lower-cased control.jobtype.

    Raises
    ------
    ValueError
        On a count, partition, jobtype or scheme conflict.
    """
    if len(theory) != 2:
        raise ValueError(
            "QMMM needs exactly two theory methods (QM first, MM last); "
            "got %d." % len(theory))
    # Adaptive sampling is allowed to run a QMMM theory queue: the ML
    # backend consumes the field only if it declares that it does. The
    # framework offers the data and never refuses a skill for not using
    # it (zero coupling); a field-independent ML surface is a physics
    # note for the adaptive method-alignment record, not a planning error.
    if reference and not (jobtype.split()
                          and jobtype.split()[0] == 'adaptive'):
        raise ValueError(
            "QMMM takes both providers from control.theory; control.reference "
            "must be empty (got %d entries)." % len(reference))

    highlevel = list(getattr(system, 'highlevel', []) or [])
    if not highlevel:
        raise ValueError(
            "QMMM needs molecule.highlevel (the QM region); it is empty.")
    high_centers, _ = _normalize_centers(highlevel, [], 2)
    n_centers = len(high_centers)
    union = sorted({int(index) for group in high_centers for index in group})
    natom = int(getattr(system, 'natom', 0)
                or len(getattr(system, 'atoms', []) or []))

    # A per-center blank region is the one region rule the schema
    # cannot judge (see _require_region); the rest is refused there.
    _require_region(high_centers, 'QMMM')

    if natom and len(union) >= natom:
        raise ValueError(
            "QMMM highlevel covers the whole system; the MM region must "
            "be non-empty.")

    # q_qm is the QM 0-order charge set. Under one center it is written in
    # molecule.highlevel order, as before; under several centers it is the
    # whole QM set in global atom order (the same order the field rows and
    # the classical charge write-back are addressed in), so its length is
    # the size of the union, never of one center.
    q_qm = list(getattr(system, 'q_qm', []) or [])
    if q_qm:
        if len(q_qm) != len(union):
            raise ValueError(
                "QMMM q_qm has %d charges for %d QM atoms (%d center(s))."
                % (len(q_qm), len(union), n_centers))
        if not all(math.isfinite(float(value)) for value in q_qm):
            raise ValueError("QMMM q_qm contains a non-finite value.")

    groups = list(getattr(system, 'center_groups', []) or [])
    if groups:
        raise ValueError(
            "QMMM conflicts with molecule.center_groups; use one composite "
            "scheme at a time.")

    # Couplings are per-center by construction: each center computes its own
    # NAC/SOC between its own states, `collect_center_electronic` stores them
    # in `state.electronic[c]` in that center's local framing, and the hopping
    # dispatcher runs one ledger per center (`ElectronicStepper`). Nothing is
    # summed across centers, so a multi-center request needs no aggregation
    # rule. A center that declares a single state simply carries an empty
    # coupling - there is nothing for it to hop - and the per-center pair
    # validation (`classify_pairs`) is what refuses a pair outside a center's
    # own ladder.



def _new_task(role, task_idx, contribution, region, region_indices,
              skill_name, backend_class, capabilities, layer_id=0,
              center_index=None, region_id=None, task_slot=None,
              provides_electronic=False, electronic_binding_id=None,
              state_policy=None, reference_state=None,
              transform_spec=None):
    """Create a task descriptor dict (docs 6.2 static task fields).

    Parameters
    ----------
    role : str
        Logical workflow role (qm, qm2, mm, ...).
    task_idx : int
        Execution index within the role.
    contribution : int
        Contribution coefficient for composite aggregation.
    region : str
        Region label ('all', 'real', 'model', 'inter', 'fragment').
    region_indices : list of int
        Atom index slice for the region (empty means full system).
    skill_name : str
        Skill providing this task.
    backend_class : type
        Backend class of the skill.
    capabilities : list of str
        Capabilities allowed for this layer's skill in this run. The
        descriptor records what the task may actually be asked for, so a
        region slice never carries a whole-system property: see
        :func:`_task_capabilities`.
    layer_id : int
        Layer number within the composite workflow (0 for
        single-shot jobs; high=0, mid=1, low=2 for ONIOM).
    center_index : int or None
        Contiguous 0-based center ordinal (docs 3.1). The full-system
        baseline L(real) carries None; a plain single-method task and
        every ONIOM center term carry their center ordinal. C>1 is
        decided by the number of DISTINCT centers, never by this field
        being non-null.
    region_id : str or None
        Unique region identifier for center/layer terms.
    task_slot : int or None
        Frozen queue position within the category (docs 3.1); paired
        with the persistent task_id for restart verification.
    provides_electronic : bool
        True only for the high-level provider H(c) of a center: the
        sole source of that center's NAC/SOC and active-state
        properties, so the property source never depends on the
        mathematical position of L(real) (docs 6.1).
    electronic_binding_id : str or None
        Links the task to its center electronic binding ('c0', ...).
    state_policy : str or None
        'active' follows the center's active state (provider H(c));
        'reference' is a fixed-reference correction/baseline term.
    reference_state : int or None
        Fixed local reference state for 'reference' policy tasks.
    transform_spec : dict or None
        Reserved region->global transform (cap/link atoms); None in
        the first version (no bond breaking, no caps).

    Returns
    -------
    dict
        Task descriptor with semantic identifiers:
        task_id is '{role}' for the first invocation of a role and
        '{role}_task{task_idx}' for repeated invocations.
    """
    task_id = role if task_idx == 0 else "%s_task%d" % (role, task_idx)
    return {
        'role': role,
        'task': task_idx,
        'task_id': task_id,
        'task_slot': task_slot,
        'layer_id': layer_id,
        'contribution': contribution,
        'region': region,
        'region_indices': list(region_indices),
        'region_charge': None,
        'region_multiplicity': None,
        'skill_name': skill_name,
        'backend': backend_class,
        'capabilities': _task_capabilities(capabilities, region_indices),
        'center_index': center_index,
        'region_id': region_id,
        'provides_electronic': provides_electronic,
        'electronic_binding_id': electronic_binding_id,
        'state_policy': state_policy,
        'reference_state': reference_state,
        'transform_spec': transform_spec,
    }


def _restrict_capabilities(capabilities):
    """Restrict capabilities of composite non-primary tasks.

    Composite sub-calculations only contribute energies and
    gradients to the combined result, so advanced properties are
    removed from non-primary tasks.

    Parameters
    ----------
    capabilities : list of str
        Full capability list of the skill.

    Returns
    -------
    list of str
        Restricted capability list.
    """
    return sorted(c for c in capabilities if c in _BASIC_CAPABILITIES)


def _count_centers(highlevel):
    """Count the number of centers from highlevel.

    Parameters
    ----------
    highlevel : list
        Flat list (single center) or nested list (multi-center).

    Returns
    -------
    int
        Number of centers (1 for flat list, len for nested).
    """
    if not highlevel:
        return 0
    if isinstance(highlevel[0], list):
        return len(highlevel)
    return 1


def _normalize_centers(highlevel, midlevel, n_layer):
    """Normalize highlevel/midlevel to nested list format.

    Parameters
    ----------
    highlevel : list
        Flat or nested high-level indices.
    midlevel : list
        Flat or nested mid-level indices.
    n_layer : int
        2 or 3.

    Returns
    -------
    tuple(list of list, list of list or None)
        (highlevel_centers, midlevel_centers) where midlevel_centers
        is None for n_layer == 2.
    """
    # Normalize highlevel
    if not highlevel:
        high_centers = [[]]
    elif isinstance(highlevel[0], list):
        high_centers = [list(g) for g in highlevel]
    else:
        high_centers = [list(highlevel)]

    # Normalize midlevel
    mid_centers = None
    if n_layer == 3:
        if not midlevel:
            mid_centers = [[] for _ in high_centers]
        elif isinstance(midlevel[0], list):
            mid_centers = [list(g) for g in midlevel]
        else:
            mid_centers = [list(midlevel)]

    return high_centers, mid_centers


def _require_region(high_centers, scheme='ONIOM'):
    """Refuse a composite run whose high-level region is blank.

    Region validity is the input schema's rule (`schema/validate.py`
    validate_centers): an empty, out-of-range or repeated index, two
    centers that overlap, and the mid-level shape are all refused
    there with a field-path message. Measured on nine bad regions,
    each was refused upstream before planning ran, so the copy of
    those invariants that stood here could not answer first and was
    removed.

    What the schema cannot judge is a blank region. With no atom index
    to look at, a two-method run with no highlevel and a nested
    ``highlevel: [[], []]`` both read as runs without a region -- the
    schema skips its region block -- yet the composite is still
    selected and its expander has no region to slice. That case is
    refused here, per center.

    Parameters
    ----------
    high_centers : list of list of int
        Per-center high-level indices (0-based), already normalized.
    scheme : str
        Scheme name carried by the message.

    Raises
    ------
    ValueError
        When any center's high-level region is empty.
    """
    for center, group in enumerate(high_centers):
        if not group:
            raise ValueError(
                "%s center %d: highlevel region is empty." % (scheme, center))


def _expand_oniom_centers(theory_skills, system, n_layer,
                          high_centers=None, mid_centers=None):
    """Unified center-based ONIOM expander (docs 6.1, real-first).

    One expander for C=1 and C>1; the mathematical order follows docs
    section 8 (real-first):

      two-layer   1 + 2C:  L(real); per center c: +H(M_c), -L(M_c)
      three-layer 1 + 4C:  L(real); per center c:
                             +M(I_c), -L(I_c), +H(M_c), -M(M_c)

    C=1 naturally yields 3 / 5 terms. The full-system baseline
    L(real) is computed once and carries center_index=None; every
    center term carries its contiguous 0-based center_index. The
    high-level term H(c) of each center is the electronic provider
    (provides_electronic=True) -- the sole source of that center's
    NAC/SOC and active-state properties, so the mathematical position
    of L(real) never decides the property source (docs 6.1).

    Identity stays backward compatible for C=1: the per-role running
    execution index reproduces the historical task_id set ('qm',
    'qm2', 'qm2_task1', ...) and therefore the historical static_id
    directory names, only reordered to real-first. For C>1 the same
    counter keeps every task_id (and static_id) unique across centers;
    the center ordinal is carried by center_index, never parsed back
    out of the id.

    Parameters
    ----------
    theory_skills : dict
        Theory skills {skill_name: info}; insertion order defines the
        layering (1st = high, 2nd = mid/low, 3rd = low).
    system : schema.config.System or None
        System configuration carrying highlevel/midlevel region
        keywords (0-based, already normalized by the adapter).
    n_layer : int
        2 for ONIOM2, 3 for ONIOM3.
    high_centers, mid_centers : list of list of int, optional
        Explicit partition. Theory expansion leaves them None and reads
        molecule.highlevel/midlevel; the adaptive reference plan passes
        molecule.highref/midref so the reference queue has its own
        partition rather than inheriting the theory one.

    Returns
    -------
    list of dict
        Task descriptors in real-first execution order.
    """
    if high_centers is None:
        highlevel = list(getattr(system, 'highlevel', []) or [])
        midlevel = list(getattr(system, 'midlevel', []) or [])
        high_centers, mid_centers = _normalize_centers(
            highlevel, midlevel, n_layer)
    else:
        high_centers = ([list(group) for group in high_centers]
                        if high_centers and isinstance(high_centers[0], list)
                        else [list(high_centers)])
        if n_layer == 3:
            if mid_centers is None:
                mid_centers = [[] for _ in high_centers]
            else:
                mid_centers = ([list(group) for group in mid_centers]
                               if mid_centers and isinstance(
                                   mid_centers[0], list)
                               else [list(mid_centers)])
    n_centers = len(high_centers)

    # Ordered by position, which is the insertion order `resolve_skills` built
    # the dict in, so a skill may now hold two of them (high and low layer of
    # one program) without one dropping out. An entry built by a caller that
    # predates the `skill_name` field is named by its key, and a disambiguated
    # key (`pyscf@qm2`) is named by that field.
    infos = [(info.get('skill_name') or key, info)
             for key, info in theory_skills.items()]
    high_name, high_info = infos[0]
    high_caps = high_info.get('capabilities', [])
    high_role = high_info['role']

    # Per-role running execution index: task_id is '{role}' for index
    # 0 and '{role}_task{idx}' afterwards, so repeated roles across
    # centers stay unique while C=1 reproduces the historical ids.
    role_counter = {}

    def _next_idx(role):
        idx = role_counter.get(role, 0)
        role_counter[role] = idx + 1
        return idx

    tasks = []

    if n_layer == 2:
        low_name, low_info = infos[1]
        restricted_low = _restrict_capabilities(
            low_info.get('capabilities', []))
        low_role = low_info['role']

        # L(real) baseline once: center_index=None
        tasks.append(_new_task(
            low_role, _next_idx(low_role), 1, 'real', [],
            low_name, low_info['backend'], restricted_low,
            layer_id=1, center_index=None, region_id='real',
            state_policy='reference', reference_state=0,
        ))
        # Per center: +H(M_c), -L(M_c)
        for c in range(n_centers):
            h_indices = high_centers[c]
            binding = 'c%d' % c
            tasks.append(_new_task(
                high_role, _next_idx(high_role), 1, 'model', h_indices,
                high_name, high_info['backend'], high_caps,
                layer_id=0, center_index=c,
                region_id='%s_high' % binding,
                provides_electronic=True, electronic_binding_id=binding,
                state_policy='active',
            ))
            tasks.append(_new_task(
                low_role, _next_idx(low_role), -1, 'model', h_indices,
                low_name, low_info['backend'], restricted_low,
                layer_id=1, center_index=c, region_id='%s_low' % binding,
                electronic_binding_id=binding,
                state_policy='reference', reference_state=0,
            ))
        return tasks

    # ONIOM3: 1 + 4C
    mid_name, mid_info = infos[1]
    low_name, low_info = infos[2]
    restricted_mid = _restrict_capabilities(
        mid_info.get('capabilities', []))
    restricted_low = _restrict_capabilities(
        low_info.get('capabilities', []))
    mid_role = mid_info['role']
    low_role = low_info['role']

    # L(real) baseline once
    tasks.append(_new_task(
        low_role, _next_idx(low_role), 1, 'real', [],
        low_name, low_info['backend'], restricted_low,
        layer_id=2, center_index=None, region_id='real',
        state_policy='reference', reference_state=0,
    ))
    # Per center: +M(I_c), -L(I_c), +H(M_c), -M(M_c). The mid-level
    # model region is region_m = H_c ∪ I_c because highlevel and
    # midlevel are disjoint input regions.
    for c in range(n_centers):
        h_indices = high_centers[c]
        i_indices = mid_centers[c] if mid_centers else []
        m_indices = sorted({int(index) for index in
                            (list(h_indices) + list(i_indices))})
        binding = 'c%d' % c
        tasks.append(_new_task(
            mid_role, _next_idx(mid_role), 1, 'inter', m_indices,
            mid_name, mid_info['backend'], restricted_mid,
            layer_id=1, center_index=c, region_id='%s_mid_inter' % binding,
            electronic_binding_id=binding,
            state_policy='reference', reference_state=0,
        ))
        tasks.append(_new_task(
            low_role, _next_idx(low_role), -1, 'inter', m_indices,
            low_name, low_info['backend'], restricted_low,
            layer_id=2, center_index=c, region_id='%s_low_inter' % binding,
            electronic_binding_id=binding,
            state_policy='reference', reference_state=0,
        ))
        tasks.append(_new_task(
            high_role, _next_idx(high_role), 1, 'model', h_indices,
            high_name, high_info['backend'], high_caps,
            layer_id=0, center_index=c, region_id='%s_high' % binding,
            provides_electronic=True, electronic_binding_id=binding,
            state_policy='active',
        ))
        tasks.append(_new_task(
            mid_role, _next_idx(mid_role), -1, 'model', h_indices,
            mid_name, mid_info['backend'], restricted_mid,
            layer_id=1, center_index=c, region_id='%s_mid_model' % binding,
            electronic_binding_id=binding,
            state_policy='reference', reference_state=0,
        ))
    return tasks


def _expand_single(category_skills, center_index=None,
                   provides_electronic=False, region='all',
                   region_indices=None):
    """Expand skills without composite into one task per skill.

    A plain single-method full-system run binds center 0 and provides
    the electronics (docs 6.1); reference-only queues stay unbound
    (center_index=None, provides_electronic=False).

    Parameters
    ----------
    category_skills : dict
        Skills dict of one category {skill_name: info}.
    center_index : int or None
        Center ordinal to bind (0 for a plain single-method run).
    provides_electronic : bool
        Whether the task is the electronic provider of its center.
    region : str
        Region label for the expanded tasks; 'all' by default. Adaptive
        sampling uses 'model' so its reference labels the high layer.
    region_indices : sequence of int or None
        Region indices carried by every expanded task.

    Returns
    -------
    list of dict
        Task descriptors.
    """
    binding = ('c%d' % center_index) if center_index is not None else None
    tasks = []
    for skill_name, info in category_skills.items():
        tasks.append(_new_task(
            role=info['role'],
            task_idx=0,
            contribution=1,
            region=region,
            region_indices=list(region_indices or []),
            skill_name=skill_name,
            backend_class=info['backend'],
            capabilities=info.get('capabilities', []),
            center_index=center_index,
            provides_electronic=provides_electronic,
            electronic_binding_id=binding,
            state_policy='active' if provides_electronic else None,
        ))
    return tasks


def _expand_fragment(theory_skills, system):
    """Expand fragment composite into one task per fragment group.

    Every fragment contributes with coefficient +1.

    Parameters
    ----------
    theory_skills : dict
        Theory skills; the first skill serves all fragments.
    system : schema.config.System or None
        System configuration carrying `center_groups`.

    Returns
    -------
    list of dict
        Task descriptors in execution order.
    """
    groups = list(getattr(system, 'center_groups', []) or [])
    if not theory_skills:
        return []

    info = next(iter(theory_skills.values()))
    skill_name = info.get('skill_name') or next(iter(theory_skills.keys()))
    caps = _restrict_capabilities(info.get('capabilities', []))

    tasks = []
    for idx, group in enumerate(groups):
        tasks.append(_new_task(
            info['role'], idx, 1, 'fragment', list(group),
            skill_name, info['backend'], caps,
        ))
    return tasks


def _expand_qmmm(theory_skills, system, high_centers=None):
    """Expand the additive QM/MM composite into the task queue.

    ``control.multiscale_type='qmmm'`` names the two providers
    positionally: theory[0] is the QM provider (Molcas ESPF external
    field) and theory[1] is the MM provider (LAMMPS). The
    LAMMPS refactor splits QM/MM into two standalone profiles, so one
    evaluation is three stages:

        qmmm_field x C  ->  QM x C  ->  qmmm_energy

    ``qmmm_field`` is the A/B Coulomb probe that emits the environment
    potential and field at the QM atoms; the QM calculation consumes them;
    then ``qmmm_energy`` evaluates the full force field with the QM 0-order
    charges and emits ``E_classic = U_main - U_gas - qmmm_el`` together
    with the QM-row corrections. The field task is auxiliary: it carries
    contribution 0 and never enters the energy sum. The physical components
    are the QM energies (+1 each) and ``E_classic`` (+1).

    Under several centers every center gets its own probe and its own QM
    task, and the classical evaluation stays one task over the whole
    system. The probe partitions by center because each center must feel
    the **others'** current charges as environment charges: the probe
    group holds the atoms whose potential is wanted (at +1) and everything
    outside it -- MM plus the other centers' 0-order charges -- sits on the
    environment side, which the A/B difference then keeps. The energy task
    is addressed by the union of all centers in global atom order, because
    one atom carrying two centers' charges is refused in validation.

    Parameters
    ----------
    theory_skills : dict
        Theory skills {binding: info} in resolution order; exactly two.
    system : schema.config.System or None
        System configuration carrying molecule.highlevel (flat = one
        center, nested = one region per center).
    high_centers : list of list of int, optional
        Explicit QM partition. Theory expansion leaves it None and reads
        molecule.highlevel; the adaptive reference plan passes
        molecule.highref so the reference QM region is its own.

    Returns
    -------
    list of dict
        [field x C?, QM x C, energy] in execution order.
    """
    if high_centers is None:
        highlevel = list(getattr(system, 'highlevel', []) or [])
        high_centers, _ = _normalize_centers(highlevel, [], 2)
    else:
        high_centers = ([list(group) for group in high_centers]
                        if high_centers and isinstance(high_centers[0], list)
                        else [list(high_centers)])
    infos = [(info.get('skill_name') or key, info)
             for key, info in theory_skills.items()]
    if len(infos) != 2:
        raise ValueError(
            "QMMM needs exactly two theory bindings; got %d." % len(infos))
    # Which binding is the quantum side is its declared role, not its
    # position in control.theory: a queue handed in the other order
    # used to hand the QM region to the classical engine, whose field
    # then came from the QM program. The roles are the manifest's
    # (`skills.query.role_of`), the same reading the ONIOM2
    # classical-outer gate uses.
    from ..skills.query import role_of

    def _side(name, info):
        # A resolved binding names its side with 'qm' or 'mm'. Any other
        # label ('qm2', the ONIOM3 mid layer) says nothing about which
        # side of a two-provider additive scheme it is, so it is read as
        # unknown and the documented order decides; only a binding that
        # carries no label at all is looked up in the registry.
        role = str((info or {}).get('role') or '')
        if role:
            return role if role in ('qm', 'mm') else ''
        return role_of(name)

    sides = [(_side(name, info), name, info) for name, info in infos]
    quantum = [(name, info) for side, name, info in sides if side == 'qm']
    classical = [(name, info) for side, name, info in sides if side == 'mm']
    unknown = [name for side, name, _info in sides if not side]
    if unknown:
        # No declared role to read (a binding the registry does not
        # know): the documented spelling is QM first, MM last, so the
        # pair keeps that reading rather than refusing a run the schema
        # accepted.
        (qm_name, qm_info), (mm_name, mm_info) = infos
    elif len(quantum) == 1 and len(classical) == 1:
        (qm_name, qm_info), = quantum
        (mm_name, mm_info), = classical
    else:
        raise ValueError(
            "QMMM needs one quantum (qm) provider and one classical (mm) "
            "provider by their declared roles; got %s."
            % ', '.join('%s=%s' % (name, side or '<unset>')
                        for side, name, _info in sides))

    n_centers = len(high_centers)
    union = sorted({int(index) for group in high_centers for index in group})

    # 'field'/'qmmm' are the QMMM task duties; the LAMMPS skill
    # maps them (with energy/gradient) to its own qmmm_field /
    # qmmm_energy profiles. The QM task carries the electronic request
    # (nac/soc for surface hopping) minus the whole-system-only stress;
    # the energy task is the whole-system classical provider and may
    # answer stress from U_main.
    mm_capabilities = list(mm_info.get('capabilities') or [])
    energy_capabilities = ['energy', 'gradient', 'qmmm']
    if 'stress' in mm_capabilities:
        energy_capabilities.append('stress')
    embedding = bool(getattr(system, 'embedding', False))
    sc_charge = bool(getattr(system, 'sc_charge', False))

    def _duty_id(duty, center):
        """Region identifier: the historical name for one center."""
        return duty if n_centers == 1 else 'c%d_%s' % (center, duty)

    def _owner(center):
        """Center ownership: a lone center stays center-less as before.

        The single-center field task has always carried
        ``center_index=None``, and that value reaches the serialized task
        record and the per-term runtime directory name
        (``term_worker._term_skill_name`` appends ``c<N>`` only when it is
        not None). Several centers do need the owner, because the probe
        partition and the outcome are addressed per center.
        """
        return None if n_centers == 1 else center

    tasks = []
    if embedding:
        for c, group in enumerate(high_centers):
            probe = [int(index) for index in group]
            field_task = _new_task(
                'field', c, 0, 'all', [],
                mm_name, mm_info['backend'],
                ['energy', 'gradient', 'field'],
                layer_id=2, center_index=_owner(c),
                region_id=_duty_id('field', c),
            )
            field_task.update({
                'composite': 'qmmm',
                'qmmm_role': 'field',
                'binding_role': mm_info['role'],
                # The probe measures the potential at THIS center's atoms
                # while every other charge -- MM and the other centers'
                # current 0-order values -- sits on the environment side of
                # the A/B difference. `partition_indices` stays the whole
                # QM set because the environment charges are handed over as
                # one array; `probe_indices` names the center measured.
                'partition_indices': union,
                'probe_indices': probe if n_centers > 1 else [],
            })
            tasks.append(field_task)

    for c, group in enumerate(high_centers):
        indices = [int(i) for i in group]
        binding = 'c%d' % c
        # A charge round solves one center's electrons in the current field
        # and reports nothing but its 0-order charges. It is a separate task
        # because a skill compiles its module chain once per task: the same
        # QM task cannot be an energy-only solve in one round and a gradient
        # solve in the next. Only the centers need it -- with one center the
        # field holds no QM charge, so there is nothing to converge and the
        # property solve is the one and only solve. With `sc_charge` off the
        # iteration is one round long, and the property solve already
        # reports those charges, so a cheap round would be a second solve
        # for a number the run already has.
        if n_centers > 1 and embedding and sc_charge:
            charge_task = _new_task(
                'qm_charge', c, 0, 'model', indices,
                qm_name, qm_info['backend'], ['energy'],
                layer_id=0, center_index=c,
                region_id=_duty_id('qm_charge', c),
                provides_electronic=False, electronic_binding_id=binding,
                state_policy='active',
            )
            charge_task.update({
                'composite': 'qmmm',
                'qmmm_role': 'qm_charge',
                'provides_charges': True,
                'binding_role': qm_info['role'],
                'partition_indices': indices,
            })
            tasks.append(charge_task)

    # The capability channel (`md` with `fssh`, an `optimize` carrier) asks
    # through the same door as the spelled jobtype, and per-center couplings
    # are supported for the reason given in `_validate_qmmm_selection`: each
    # center owns its own states, its own pair map and its own hop.

    for c, group in enumerate(high_centers):
        indices = [int(i) for i in group]
        binding = 'c%d' % c
        qm_capabilities = _task_capabilities(
            list(qm_info.get('capabilities') or ['energy', 'gradient']),
            indices)
        for required in ('energy', 'gradient'):
            if required not in qm_capabilities:
                qm_capabilities.append(required)
        qm_task = _new_task(
            'qm', c, 1, 'model', indices,
            qm_name, qm_info['backend'], qm_capabilities,
            layer_id=0, center_index=c, region_id=_duty_id('qm', c),
            provides_electronic=True, electronic_binding_id=binding,
            state_policy='active',
        )
        qm_task.update({
            'composite': 'qmmm',
            'qmmm_role': 'qm',
            'provides_charges': True,
            'binding_role': qm_info['role'],
            'partition_indices': indices,
        })
        tasks.append(qm_task)

    energy_task = _new_task(
        'energy', 0, 1, 'all', [],
        mm_name, mm_info['backend'], energy_capabilities,
        layer_id=2, center_index=None, region_id='energy',
    )
    energy_task.update({
        'composite': 'qmmm',
        'qmmm_role': 'energy',
        'binding_role': mm_info['role'],
        # The whole QM set, every center, in global atom order: this is the
        # array the 0-order charges are written into the force field from.
        'partition_indices': union,
    })
    tasks.append(energy_task)

    expected = (([0] * n_centers if embedding else [])
                + ([0] * n_centers
                   if (embedding and n_centers > 1 and sc_charge) else [])
                + [1] * n_centers + [1])
    contributions = [task['contribution'] for task in tasks]
    if contributions != expected:
        raise ValueError(
            "QMMM contributions must be %s for embedding=%r, %d center(s); "
            "got %s." % (expected, embedding, n_centers, contributions))
    return tasks


def _flat_region(value):
    """Flatten a highref/midref value to one 0-based atom list.

    Adaptive reference planning is single-center; a nested spelling is
    already refused by schema validation, so flattening here is only the
    tolerant path for an already-normalized programmatic value.
    """
    if not value:
        return []
    if isinstance(value[0], (list, tuple)):
        return sorted({int(index) for group in value for index in group})
    return sorted(int(index) for index in value)


def _reference_regions(config, theory_layers):
    """Resolve the effective highref/midref regions.

    The reference partition is independent of the theory list. Empty
    highref/midref inherit the theory partition when the theory queue is
    layered; with one theory method the high region is the whole system
    unless a proper subset is written (a layered reference needs one).
    """
    system = getattr(config, 'system', None)
    if system is None:
        return [], []
    highref = _flat_region(getattr(system, 'highref', []) or [])
    midref = _flat_region(getattr(system, 'midref', []) or [])
    highlevel = _flat_region(getattr(system, 'highlevel', []) or [])
    midlevel = _flat_region(getattr(system, 'midlevel', []) or [])
    high = highref or (highlevel if theory_layers > 1 else [])
    mid = midref or (midlevel if theory_layers > 2 else [])
    return high, mid


def resolve_reference_type(config, skills):
    """Classify the adaptive reference plan from its own method list.

    The reference list is a layered plan, not a positional mirror of the
    theory list: one method is a single calculation, two are an ONIOM2 or
    QMMM pair (control.multiscale_type decides), and three are ONIOM3.
    """
    reference = skills.get('reference', {}) if skills else {}
    count = len(reference)
    if count == 0:
        return None
    control = getattr(config, 'control', None)
    multiscale = str(
        getattr(control, 'multiscale_type', '') or '').strip().lower()
    if count == 1:
        return 'single'
    if count == 2:
        return 'qmmm' if multiscale == 'qmmm' else 'oniom2'
    if count == 3:
        return 'oniom3'
    raise ValueError(
        "adaptive reference carries %d methods; at most three layers are "
        "defined." % count)


def _expand_reference_plan(config, skills, reference_type):
    """Expand the adaptive reference list into its own task queue."""
    reference_skills = skills.get('reference', {}) if skills else {}
    if not reference_skills:
        return []
    control = getattr(config, 'control', None)
    theory_layers = len(getattr(control, 'theory', []) or [])
    if not theory_layers:
        # Mock/legacy configs may omit control.theory; the resolved
        # theory bindings still name the layer count.
        theory_layers = len(skills.get('theory', {}) or {})
    high, mid = _reference_regions(config, theory_layers)
    system = getattr(config, 'system', None)
    if reference_type == 'single':
        region = 'model' if theory_layers > 1 else 'all'
        indices = high if theory_layers > 1 else []
        return _expand_single(
            reference_skills, region=region, region_indices=indices)
    if reference_type in ('oniom2', 'oniom3'):
        n_layer = 2 if reference_type == 'oniom2' else 3
        return _expand_oniom_centers(
            reference_skills, system, n_layer,
            high_centers=[high] if high else [[]],
            mid_centers=[mid] if n_layer == 3 else None)
    if reference_type == 'qmmm':
        return _expand_qmmm(
            reference_skills, system,
            high_centers=[high] if high else [[]])
    raise ValueError(
        "Unknown adaptive reference plan %r." % (reference_type,))


def resolve_label_plan(config, theory_tasks, reference_tasks):
    """Derive the training-label source from the region correspondence.

    The neural network surface is the theory electronic provider's
    region (all atoms for one theory method, highlevel for a layered
    queue). The reference queue is its own layered plan; the label is
    the reference layer whose region covers exactly that surface, or --
    when no single layer does -- the reference composite total over the
    whole system. This is a set comparison, never a list-length rule.

    Returns a plain dict carried by every reference task:
    ``region_indices``, ``source`` ('layer' or 'total'),
    ``source_task_ids``, ``source_layer``, ``aux``, ``full_system`` and
    ``notes``.
    """
    if not reference_tasks:
        return None
    system = getattr(config, 'system', None)
    natom = int(getattr(system, 'natom', 0)
                or len(getattr(system, 'atoms', []) or []))
    all_indices = list(range(natom))

    provider = resolve_primary_task(theory_tasks)
    nn_indices = (list(provider.get('region_indices') or [])
                  if provider is not None else [])
    if not nn_indices:
        nn_indices = list(all_indices)
    nn_set = set(nn_indices)

    # A physical layer calculation is a +1 term on a region; the
    # full-system 'real' baseline and the -1 subtraction terms are not
    # layer surfaces of their own.
    candidates = []
    for task in reference_tasks:
        if task.get('qmmm_role') == 'qm':
            candidates.append(('high', task))
            continue
        if task.get('qmmm_role'):
            continue
        if task.get('region') == 'real':
            continue
        if int(task.get('contribution', 1)) != 1:
            continue
        layer_id = int(task.get('layer_id', 0) or 0)
        if layer_id == 0:
            candidates.append(('high', task))
        elif layer_id == 1:
            candidates.append(('mid', task))

    matches = []
    for layer, task in candidates:
        indices = set(task.get('region_indices') or [])
        if not indices:
            indices = set(all_indices)
        if indices == nn_set:
            matches.append((layer, task))

    source = ''
    source_layer = None
    source_tasks = []
    if len(matches) == 1:
        source = 'layer'
        source_layer, task = matches[0]
        source_tasks = [task.get('task_id')]
        if source_layer == 'mid':
            # Capability distribution gives the full request to the first
            # reference binding only. When the mid layer is the label
            # source, it must answer the same channel set (NAC/SOC) the
            # run asked for, or the dataset and the label disagree.
            requested = sorted({cap for item in reference_tasks
                                for cap in (item.get('capabilities') or [])})
            task['capabilities'] = _task_capabilities(
                requested, task.get('region_indices') or [])
    else:
        # No single layer covers the surface. The composite total is the
        # label when the reference plan spans the whole system (a low
        # layer exists, or a single all-atom task).
        spans_all = any(
            task.get('region') == 'real'
            or (not task.get('region_indices')
                and task.get('region') == 'all')
            for task in reference_tasks)
        if spans_all and nn_set == set(all_indices):
            source = 'total'
        else:
            raise ValueError(
                "the reference plan does not cover the neural-network "
                "surface: theory covers %d atom(s), reference layers cover "
                "%s. Write molecule.highref/midref so one reference layer "
                "(or the composite total) matches the theory region."
                % (len(nn_indices),
                   [sorted(task.get('region_indices') or all_indices)
                    for _layer, task in candidates]))

    qmmm = (str(getattr(getattr(config, 'control', None),
                        'multiscale_type', '') or '').strip().lower()
            == 'qmmm'
            or any(task.get('composite') == 'qmmm'
                   for task in reference_tasks))
    notes = []
    aux = []
    if qmmm:
        has_mm = any(task.get('qmmm_role') in ('field', 'energy')
                     for task in reference_tasks)
        has_field = any(task.get('qmmm_role') == 'field'
                        for task in reference_tasks)
        if has_mm:
            aux = ['mm_energy', 'q_qm']
            if has_field:
                aux.insert(0, 'field')
            else:
                notes.append(
                    "the QMMM reference runs without a field probe "
                    "(embedding is off): it produces no MM field.")
        else:
            notes.append(
                "the QMMM reference has no MM provider: its QM labels are "
                "not polarized and carry no MM field.")
    else:
        embedding = bool(getattr(system, 'embedding', False)
                         or getattr(system, 'read_charge', False))
        if embedding:
            # The high layer can be embedded by the mid shell of an
            # ONIOM3 queue and by the low shell of an ONIOM2/3 queue.
            has_mid = any(task.get('region') == 'inter'
                          for task in reference_tasks)
            has_low = any(task.get('region') == 'real'
                          for task in reference_tasks)
            if has_mid:
                aux.append('mid_charges')
            if has_low:
                aux.append('low_charges')
            if not (has_mid or has_low):
                notes.append(
                    "electrostatic embedding is on but the reference plan "
                    "has no mid/low layer: the high-level labels are not "
                    "embedded in the environment charges.")
            theory_layers = len(
                getattr(getattr(config, 'control', None),
                        'theory', []) or [])
            if not theory_layers:
                theory_layers = len(theory_tasks)
            if theory_layers > 2 and not has_mid:
                notes.append(
                    "theory carries a midlevel but the reference plan has "
                    "no mid layer: the labels cannot reproduce the ONIOM3 "
                    "mid-shell embedding.")

    plan = {
        'region_indices': list(nn_indices),
        'source': source,
        'source_layer': source_layer,
        'source_task_ids': source_tasks,
        'aux': aux,
        'full_system': len(reference_tasks) > 1,
        'qmmm': qmmm,
        'reference_type': None,
        'notes': notes,
    }
    for task in reference_tasks:
        task['label_plan'] = plan
    return plan


def resolve_task_queue(config, skills):
    """Resolve the Task Queue from jobtype, skills and composite type.

    Every task is one independent backend invocation. The queue
    order is the Calculator execution order. Task descriptors carry
    region slice indices only; geometry is derived from the live
    structure at execution time.

    Parameters
    ----------
    config : PyRAIConfig
        Validated framework configuration.
    skills : dict
        Resolved skills {'theory': {...}, 'reference': {...}} from
        Maestro.resolve_skills(), each entry carrying role, backend
        and capabilities.

    Returns
    -------
    dict
        {'theory': [task, ...], 'reference': [task, ...]}.
        Every task carries its static execution identifier
        ('tmp_{SKILL}' for single-shot jobs, otherwise
        'tmp_{task_id}_layer{layer_id}_{SKILL}').
    """
    theory_skills = skills.get('theory', {})
    reference_skills = skills.get('reference', {})
    system = getattr(config, 'system', None)

    control = getattr(config, 'control', None)
    adaptive_tokens = str(getattr(control, 'jobtype', '') or '') \
        .strip().lower().split()
    adaptive = bool(adaptive_tokens) and adaptive_tokens[0] == 'adaptive'

    composite_type = resolve_composite_type(config, skills)
    reference_type = (resolve_reference_type(config, skills)
                      if adaptive else None)

    queue = {'theory': [], 'reference': []}

    if composite_type is None:
        # Plain single-method full-system run: one task bound to c0
        # (docs 6.1). A reference-only queue stays unbound.
        queue['theory'] = _expand_single(
            theory_skills, center_index=0, provides_electronic=True)
        if adaptive:
            queue['reference'] = _expand_reference_plan(
                config, skills, reference_type)
        else:
            queue['reference'] = _expand_single(reference_skills)
    elif composite_type in ('oniom2', 'oniom3',
                            'multi_center_oniom2', 'multi_center_oniom3'):
        n_layer = 2 if composite_type in (
            'oniom2', 'multi_center_oniom2') else 3
        # Region validity is enforced in the R1 schema; planning only
        # refuses the blank region the schema has no index to judge.
        highlevel = getattr(system, 'highlevel', []) or []
        midlevel = getattr(system, 'midlevel', []) or []
        high_centers, _ = _normalize_centers(highlevel, midlevel, n_layer)
        _require_region(high_centers)
        # One center-based expander for C=1 and C>1 (docs 6.1).
        queue['theory'] = _expand_oniom_centers(
            theory_skills, system, n_layer)
        if adaptive:
            # The reference is its own layered plan (highref/midref),
            # not a positional mirror of the theory list.
            queue['reference'] = _expand_reference_plan(
                config, skills, reference_type)
        else:
            queue['reference'] = _expand_single(reference_skills)
    elif composite_type == 'fragment':
        queue['theory'] = _expand_fragment(theory_skills, system)
        if adaptive:
            queue['reference'] = _expand_reference_plan(
                config, skills, reference_type)
        else:
            queue['reference'] = _expand_single(reference_skills)
    elif composite_type in ('qmmm', 'multi_center_qmmm'):
        queue['theory'] = _expand_qmmm(theory_skills, system)
        if adaptive:
            queue['reference'] = _expand_reference_plan(
                config, skills, reference_type)
        else:
            queue['reference'] = []

    label_plan = None
    if adaptive:
        label_plan = resolve_label_plan(
            config, queue['theory'], queue['reference'])
        if label_plan is not None:
            label_plan['reference_type'] = reference_type

    # Stamp plan_version on every task (1 = current center-aware plan)
    is_composite = (composite_type is not None
                    or reference_type in ('oniom2', 'oniom3', 'qmmm'))
    for category in ('theory', 'reference'):
        for task in queue[category]:
            task.setdefault('plan_version', 1 if is_composite else 0)

    # Freeze the queue position (task_slot) per category: the fast
    # 0-based index into the frozen queue, paired with the persistent
    # task_id for restart verification (docs 3.1).
    for category in ('theory', 'reference'):
        for slot, task in enumerate(queue[category]):
            task['task_slot'] = slot

    # Compose the static execution identifier for every task.
    # Single-shot jobs (exactly one task) keep the plain skill
    # working-directory name; composite/multitask jobs are named by
    # region/role (see _compose_static_id). Groups are per category, so a
    # reference task never changes a theory task's ordinal; a composite
    # reference gets a 'ref_' prefix because it can share region/role/skill
    # with the theory queue.
    total_tasks = len(queue['theory']) + len(queue['reference'])
    single = total_tasks == 1
    groups = {}
    for category in ('theory', 'reference'):
        for task in queue[category]:
            key = (category, task.get('region'), task.get('role'),
                   task.get('skill_name'))
            groups.setdefault(key, []).append(task)
    for (category, _region, _role, _skill), group in groups.items():
        if len(group) > 1:
            ordered = sorted(group, key=lambda t: (
                t.get('center_index') is None,
                t.get('center_index') or 0,
                t.get('task_slot') or 0))
            for ordinal, task in enumerate(ordered, start=1):
                static_id = _compose_static_id(
                    task, single=single, ordinal=ordinal)
                if category == 'reference' and len(queue['reference']) > 1:
                    static_id = 'ref_' + static_id
                task['static_id'] = static_id
        else:
            static_id = _compose_static_id(group[0], single=single)
            if category == 'reference' and len(queue['reference']) > 1:
                static_id = 'ref_' + static_id
            group[0]['static_id'] = static_id

    return queue



def resolve_primary_task(tasks):
    """Return the task whose non-aggregated properties are inherited.

    The composite primary is the electronic provider of the lowest
    center (center 0 for a single-center run), NOT tasks[0]: the
    real-first mathematical order (docs 6.1/8) puts the L(real)
    baseline first, but NAC/SOC and active-state properties always
    come from the high-level provider H(c). Falls back to the first
    task when no provider is flagged (legacy queues restored without
    the center fields).

    Parameters
    ----------
    tasks : list of dict
        Task descriptors of one category in execution order.

    Returns
    -------
    dict or None
        The primary task descriptor, or None for an empty queue.
    """
    if not tasks:
        return None
    providers = [t for t in tasks if t.get('provides_electronic')]
    if providers:
        # Lowest center ordinal wins (center 0 for C=1); a None
        # center sorts last so a real baseline is never the primary.
        providers.sort(key=lambda t: (t.get('center_index') is None,
                                      t.get('center_index') or 0))
        return providers[0]
    return tasks[0]


def resolve_system_task(tasks):
    """Return the task providing whole-system structural properties.

    Structural-track counterpart of resolve_primary_task (dev/methods/
    ENSEMBLE_DEVELOPMENT.md section 4A). The two resolve different property
    classes under one philosophy -- each declares which task provides
    its properties:

    - resolve_primary_task -> the high-level electronic provider H(c),
      the source of NAC/SOC and active-state electronic properties.
    - resolve_system_task  -> the term covering the ENTIRE system, the
      source of whole-system structural properties (stress/virial,
      later hessian/frequencies). It is never ONIOM-summed nor split
      per center.

    Selection (matches the user rule stated in terms of the theory
    method list):
    - a single-method, non-multiscale run: the only task (region 'all'),
      i.e. theory[0];
    - a multiscale run: the full-system low-level term, i.e. the ONIOM
      L(real) baseline (region 'real', center-less), i.e. theory[-1];
    - a multi-center run: the same center-less full-system baseline,
      which spans the whole system, so a single global quantity (e.g.
      the 3x3 stress) is returned, not a per-center one.

    Parameters
    ----------
    tasks : list of dict
        Task descriptors of one category in execution order.

    Returns
    -------
    dict or None
        The whole-system structural provider task, or None for an empty
        queue.
    """
    if not tasks:
        return None
    # QMMM: the energy task is the whole-system classical provider. The
    # field task spans the same atoms but its standard result is a zero
    # placeholder, so it must not answer a whole-system structural query.
    energy = [t for t in tasks if t.get('qmmm_role') == 'energy']
    if energy:
        return energy[0]
    if len(tasks) == 1:
        return tasks[0]
    # Prefer the ONIOM full-system baseline (region 'real'); a
    # center-less one spans the entire system.
    real = [t for t in tasks if t.get('region') == 'real']
    if real:
        centerless = [t for t in real if t.get('center_index') is None]
        return (centerless or real)[0]
    # Then a center-less full-region term (plain whole-system 'all').
    allc = [t for t in tasks
            if t.get('region') == 'all' and t.get('center_index') is None]
    if allc:
        return allc[0]
    # Fallback: the theory tail (the last method = full-system low
    # level) in execution order.
    return tasks[-1]
