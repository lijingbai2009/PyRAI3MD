"""
======================================================================
Composite aggregation for PyRAI3MD.

Provides the mathematical aggregation and projection used by
multi-layer and hybrid calculation schemes:
- ONIOM2: Two-layer ONIOM scheme (contributions +1, +1, -1)
- ONIOM3: Three-layer ONIOM scheme (contributions +1, +1, -1, +1, -1)
- QMMM: Quantum mechanics / molecular mechanics coupling (+1, +1)
- Multi-center ONIOM: N non-overlapping centers (1+2N / 1+4N terms)

Design rules (one-direction dependency):
- The calculator uses the functions of this module.
- This module never imports or uses the calculator.
- This module performs pure mathematical aggregation and projection:
  it never writes to any state, config or result container.

The contribution pattern of a composite task queue is decided in the
planning step (resolve_task_queue) and carried by PyRAIState. Once
the contribution pattern is confirmed, the composite result is just
the sum of contributions multiplied by the task results, so the
patterns are expressed as a registry instead of dedicated classes.

Aug 04 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from ...planning.task import region_label, resolve_system_task
from ...data.runtime.result import task_config_results
from .regions import project_gradient as _scatter_gradient, validate_region

AGGREGATE_KEYS = ('energy', 'gradient')



def aggregate_results(state, category=None):
    """Aggregate task results by contribution coefficients.

    Pure function: aggregates the numerical properties as
    sum(contribution x task_result) over the task queue; all other
    properties are inherited from the primary task (the first task
    in the queue). Never writes to any state or config.

    Parameters
    ----------
    state : PyRAIState
        State with an executed task queue.
    category : str or None
        Restrict aggregation to one backend role (the calculator
        active role); None aggregates both queues (legacy behavior).

    Returns
    -------
    dict
        Combined result dictionary.
    """
    aggregated = {}
    primary_results = None
    queue_tasks = []
    results_by_id = {}

    for task_category, task in state.iter_tasks():
        if category is not None and task_category != category:
            continue
        results = task_config_results(task) or {}
        queue_tasks.append(task)
        results_by_id[task.get('task_id')] = results
        if primary_results is None:
            primary_results = dict(results)
        contribution = task.get('contribution', 1)
        for key in AGGREGATE_KEYS:
            value = results.get(key)
            if value is None:
                continue
            term = contribution * np.asarray(value, dtype=np.float64)
            if key in aggregated:
                aggregated[key] = aggregated[key] + term
            else:
                aggregated[key] = term

    composite = len(queue_tasks) > 1
    data = {key: value
            for key, value in (primary_results or {}).items()
            if not (composite
                    and (str(key).startswith('alternative_')
                         or key in ('predict_model_id', 'n_models')))}
    data.update(aggregated)
    # Structural track (docs 4A): whole-system structural properties
    # come from the full-system provider task, not the electronic
    # primary. The primary inheritance above is only valid for
    # electronic-structure properties.
    system_task = resolve_system_task(queue_tasks)
    if system_task is not None:
        system_results = results_by_id.get(system_task.get('task_id')) or {}
        for key in SYSTEM_STRUCTURAL_KEYS:
            if system_results.get(key) is not None:
                data[key] = system_results[key]
    return data


# ---------------------------------------------------------------------------
# New strict aggregation API (multi-center aware)
# ---------------------------------------------------------------------------

# Result keys that may be linearly combined by contribution coefficients.
# All other keys (nac, soc, dipole, ...) are NOT aggregated by default
# and must be explicitly opted in via the request.
STRICT_AGGREGATE_KEYS = frozenset({'energy', 'gradient'})

# Whole-system structural properties (dev/methods/ENSEMBLE_DEVELOPMENT.md
# section 4A): resolved from the full-system provider task
# (resolve_system_task), NEVER ONIOM-summed, NEVER split per center and
# NEVER refused for multi-center. Distinct from the electronic track
# (STRICT_AGGREGATE_KEYS / PER_CENTER_KEYS) and from the primary-task
# inheritance, which is only valid for electronic-structure properties.
# stress_tensor is the first member; hessian/frequencies/norm_mode/
# reduced_mass/rotation_constants are registered for later migration.
SYSTEM_STRUCTURAL_KEYS = frozenset({'stress_tensor'})

# Locally usable per-center properties (docs 8): normalized, saved
# as-is into state.electronic[c] and consumed by that center's own
# algorithm. They NEVER enter the ONIOM energy/gradient sum and, for
# C>1, are NEVER inherited from the primary task -- each center's
# high-level provider H(c) is the sole source (collect_center_electronic).
PER_CENTER_KEYS = frozenset({'nac', 'soc', 'dipole', 'transition_dipole'})

# Properties with no defined global aggregation rule AND no per-center
# slot in the first version; categorically refused for multi-center
# (docs 8 "undefined global aggregation properties"; no multi-center
# high-order-derivative contract yet). These are expensive, explicitly
# requested properties that a backend only emits on demand, so refusing
# them surfaces a real mis-specification instead of silently dropping
# data the user asked for.
REFUSED_KEYS = frozenset({'overlap', 'hessian'})

# Unconditional backend byproducts with no ONIOM sum rule AND no
# framework consumer yet (docs 8: "no multi-center charge contract
# yet"). Unlike REFUSED_KEYS these ride along with EVERY standard
# calculation -- e.g. xTB writes atomic charges whenever a charge file
# exists -- so refusing them would hard-fail every multi-center ONIOM
# built on such a backend (openqp/xtb included). They are dropped
# silently for multi-center instead; C=1 keeps the legacy pass-through.
# A future per-region charge-scatter contract can promote a key out of
# this set once a consumer needs the full-system array.
IGNORED_KEYS = frozenset({'computed_charges'})

# Union of the non-aggregated keys (never summed by contribution).
_NON_AGGREGATE_KEYS = PER_CENTER_KEYS | REFUSED_KEYS | IGNORED_KEYS

# Display order of the &composite energy breakdown: high level first,
# then mid, then low; any other region (all/fragment) sorts last.
_REGION_ORDER = {'high': 0, 'mid': 1, 'low': 2}


def term_label(task, multi_center=False):
    """Human-facing label of one composite term for the &composite log.

    The label is '<region>[<role>]', where region is the geometric layer
    name (model->high, inter->mid, real->low; see region_label) and role
    is the theory method (qm/qm2/qm3/mm). A multi-center run appends the
    1-based center ordinal '[<center+1>]' so terms of different centers
    stay distinguishable; the center-less full-system baseline L(real)
    and a single-center run omit it.

    Parameters
    ----------
    task : dict
        Task descriptor.
    multi_center : bool
        Whether the evaluation has more than one center.

    Returns
    -------
    str
        Term label, e.g. 'high[qm]', 'mid[qm2][2]'.
    """
    region = region_label(task.get('region'))
    role = task.get('role') or 'task'
    label = "%s[%s]" % (region, role)
    center = task.get('center_index')
    if multi_center and center is not None:
        label = "%s[%d]" % (label, center + 1)
    return label


def _term_sort_key(task):
    """Sort key ordering terms high->mid->low, then by center/slot."""
    region = region_label(task.get('region'))
    center = task.get('center_index')
    return (_REGION_ORDER.get(region, 3),
            center is None, center or 0,
            task.get('task_slot') or 0)


def build_composite_terms(collected, multi_center=False,
                          active_configuration=None):
    """Build the &composite energy-breakdown payload.

    The breakdown is stored as parallel lists (not a list of tuples) so
    it survives the Result container, which converts any list/tuple
    value with np.asarray and would mangle a ragged (label, energies,
    contribution) tuple. Each term keeps the RAW per-state energy array
    its backend computed (never multiplied by the contribution); the
    ONIOM sign/coefficient is implied by the formula and reported
    alongside as 'contributions'.

    'state_indices' carries, per term, the state that term contributes at:
    an integer local state for a multi-center evaluation (given by
    `active_configuration`, docs 8), or None when the term has no state of
    its own and is read at the state the run is driven on. It is computed
    AFTER the ordering below, so an index can never be paired with a
    different term than the label and energy row it belongs to.

    Parameters
    ----------
    collected : list of tuple
        (task, energy_list, contribution) per term, in any order.
    multi_center : bool
        Whether the evaluation has more than one center.
    active_configuration : tuple of int, optional
        Per-center active local states, as the multi-center assembly knows
        them. None (the single-center case) means the assembled total keeps a
        global state axis and each term is read at the driven state.

    Returns
    -------
    dict
        {'labels': [str], 'energies': [list of float],
        'contributions': [int], 'state_indices': [int | None]}
        ordered high->mid->low, per center.
    """
    ordered = sorted(collected, key=lambda item: _term_sort_key(item[0]))
    labels, energies, contributions, state_indices = [], [], [], []
    for task, energy_list, contribution in ordered:
        labels.append(term_label(task, multi_center))
        energies.append([float(e) for e in energy_list])
        contributions.append(int(contribution))
        state_indices.append(
            None if active_configuration is None
            else int(_task_state_index(task, active_configuration)))
    return {'labels': labels, 'energies': energies,
            'contributions': contributions, 'state_indices': state_indices}


class TaskOutcome:
    """Container for one task's raw backend result.

    Attributes
    ----------
    task_id : str
        Task identifier from the task descriptor.
    contribution : int
        Contribution coefficient (+1, -1, ...).
    raw : dict
        Raw result dictionary returned by backend.calculate().
    region_indices : list of int
        Region atom indices (0-based) for gradient scatter.
    region : str
        Region label ('all', 'model', 'real', 'inter', ...).
    success : bool
        Whether the backend call succeeded.
    error : str or None
        Exception text when success is False.
    """

    def __init__(self, task_id, contribution, raw, region_indices,
                 region='all', success=True, error=None):
        self.task_id = task_id
        self.contribution = contribution
        self.raw = raw if raw is not None else {}
        self.region_indices = list(region_indices or [])
        self.region = region
        self.success = success
        self.error = error


def invalidate_task_result(task):
    """Clear stale results in a task config before a new run.

    Ensures that the next backend.calculate() result is the sole
    data source; no fallback to historical non-empty results.

    Parameters
    ----------
    task : dict
        Task descriptor whose config.results should be cleared.
    """
    config = task.get('config')
    if config is None:
        return
    if isinstance(config, dict):
        config['results'] = {}
    elif hasattr(config, 'results'):
        config.results = {}


def _state_broadcast(values, nstate):
    """Repeat a one-state array to every state the result declares.

    A skill that reports one electronic state leaves the caller to say how
    many states the run declared. The "one state means every state"
    contract lives here so every assembly applies it the same way.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] == 1 and nstate > 1:
        values = np.repeat(values, nstate, axis=0)
    return values


def _stress_tensor(value, nstate, label):
    """The generic (nstate, 3, 3) stress tensor of one task.

    A single-state (3, 3) is promoted and broadcast to nstate, and the
    non-finite check lives here because every structural consumer reads
    this one array. ``label`` is the sentence subject, e.g. ``Task 'qm'``,
    so each caller's refusal names its own task.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        raise ValueError(
            "%s stress_tensor has unexpected shape %s "
            "(expected (nstate, 3, 3))." % (label, arr.shape))
    arr = _state_broadcast(arr, nstate)
    if not np.all(np.isfinite(arr)):
        raise ValueError("%s stress_tensor contains NaN/Inf." % label)
    return arr


def normalize_task_result(raw, task, request):
    """Validate and normalise a backend result dictionary.

    Checks:
    - energy is present and finite (scalar or (nstate,) array).
    - gradient, when required, has the correct atom count.
    - No NaN/Inf in numerical arrays.
    - Refused keys for multi-center are not present.
    - Ignored byproduct keys (e.g. computed_charges) are dropped for
      multi-center, not refused (no ONIOM sum rule / consumer yet).

    Parameters
    ----------
    raw : dict
        Raw result from backend.calculate().
    task : dict
        Task descriptor (carries capabilities, region, ...).
    request : dict
        Aggregation request with keys:
        - nstate : int (number of electronic states)
        - nlocal : int (number of local atoms)
        - need_gradient : bool
        - multi_center : bool (refuse advanced keys)

    Returns
    -------
    dict
        Normalised result with energy as float and gradient as
        ndarray of shape (nstate, nlocal, 3) when present.

    Raises
    ------
    ValueError
        On missing/invalid data.
    """
    if not raw:
        raise ValueError(
            "Task '%s' returned an empty result." % task.get('task_id'))

    nstate = int(request.get('nstate', 1))
    nlocal = int(request.get('nlocal', 0))
    need_gradient = bool(request.get('need_gradient', False))
    multi_center = bool(request.get('multi_center', False))

    normalised = {}

    # --- energy ---
    energy = raw.get('energy')
    if energy is None:
        raise ValueError(
            "Task '%s' is missing 'energy'." % task.get('task_id'))
    energy_arr = np.asarray(energy, dtype=np.float64)
    if energy_arr.ndim == 0:
        # Scalar -> single-state array
        energy_arr = energy_arr.reshape(1)
    if energy_arr.ndim != 1:
        raise ValueError(
            "Task '%s' energy has unexpected shape %s."
            % (task.get('task_id'), energy_arr.shape))
    if not np.all(np.isfinite(energy_arr)):
        raise ValueError(
            "Task '%s' energy contains NaN/Inf."
            % task.get('task_id'))
    normalised['energy'] = energy_arr

    # --- gradient ---
    gradient = raw.get('gradient')
    if gradient is not None:
        grad_arr = np.asarray(gradient, dtype=np.float64)
        # Promote (nlocal, 3) -> (1, nlocal, 3)
        if grad_arr.ndim == 2:
            grad_arr = grad_arr[np.newaxis, :, :]
        if grad_arr.ndim != 3 or grad_arr.shape[2] != 3:
            raise ValueError(
                "Task '%s' gradient has unexpected shape %s."
                % (task.get('task_id'), grad_arr.shape))
        if grad_arr.shape[1] != nlocal:
            raise ValueError(
                "Task '%s' gradient has %d atoms but region has %d."
                % (task.get('task_id'), grad_arr.shape[1], nlocal))
        if not np.all(np.isfinite(grad_arr)):
            raise ValueError(
                "Task '%s' gradient contains NaN/Inf."
                % task.get('task_id'))
        normalised['gradient'] = grad_arr
    elif need_gradient:
        raise ValueError(
            "Task '%s' is missing 'gradient' (required by jobtype)."
            % task.get('task_id'))

    # --- stress tensor (structural track, generic (nstate, 3, 3)) ---
    # Whole-system stress/virial consumed by the NPT barostats. Kept in
    # the generic (nstate, 3, 3) format (mirrors gradient's
    # (nstate, natom, 3)) so a future excited-state stress fits; a
    # single-state (3, 3) is promoted and broadcast to nstate.
    stress = raw.get('stress_tensor')
    if stress is not None:
        normalised['stress_tensor'] = _stress_tensor(
            stress, nstate, "Task '%s'" % task.get('task_id'))

    # --- refused keys in multi-center mode (docs 8) ---
    # Only the undefined-global-aggregation keys are refused; the
    # locally usable per-center keys (nac/soc/dipole) pass through and
    # are collected per center by collect_center_electronic, never
    # ONIOM-summed and never inherited from the primary task.
    if multi_center:
        for key in REFUSED_KEYS:
            if key in raw and raw[key] is not None:
                raise ValueError(
                    "Multi-center composite refuses key '%s' from "
                    "task '%s'; no ONIOM sum rule defined."
                    % (key, task.get('task_id')))

    # --- pass through other non-refused keys ---
    for key, value in raw.items():
        if key in STRICT_AGGREGATE_KEYS:
            continue
        if key in SYSTEM_STRUCTURAL_KEYS:
            continue  # normalized explicitly above (structural track)
        if key == 'fin_coord':
            continue  # transport channel, handled by Calculator
        if multi_center and key in IGNORED_KEYS:
            # Unconditional byproduct with no multi-center contract or
            # consumer; dropped so it never reaches the ONIOM sum and
            # never hard-fails a backend that always emits it.
            continue
        normalised[key] = value

    return normalised


def validate_plan(tasks, natom):
    """Validate the integrity of a composite task plan.

    Checks:
    - Region indices are valid (no out-of-range, no duplicates).
    - Each task has a unique task_id.
    - Contribution coefficients are non-zero integers.

    Parameters
    ----------
    tasks : list of dict
        Task descriptors in execution order.
    natom : int
        Total number of atoms in the system.

    Raises
    ------
    ValueError
        On any integrity violation.
    """
    seen_ids = set()
    for task in tasks:
        tid = task.get('task_id', '')
        if tid in seen_ids:
            raise ValueError(
                "Duplicate task_id '%s' in composite plan." % tid)
        seen_ids.add(tid)

        contrib = task.get('contribution', 1)
        # A QMMM field task is an auxiliary probe and a charge round solves
        # for nothing but its 0-order charges: neither contributes energy.
        # Every other task must have a non-zero integer coefficient.
        if (not isinstance(contrib, int)
                or (contrib == 0 and task.get('qmmm_role')
                    not in ('field', 'qm_charge'))):
            raise ValueError(
                "Task '%s' has invalid contribution %r."
                % (tid, contrib))

        indices = task.get('region_indices', [])
        region = task.get('region', 'all')
        if region != 'all' and indices:
            validate_region(indices, natom)


def _project_region_coupling(result, region_indices, natom):
    """Scatter a region provider's coupling rows into system coordinates.

    A derivative coupling is a per-nuclear-coordinate vector of the electronic
    region that produced it, so a region provider answers with
    ``(npair, nlocal, 3)`` while everything that steps on it works in
    whole-system coordinates. The atoms outside the region get no row: the
    wavefunction the derivative acts on does not reach them, which is the same
    statement the region's own gradient already makes. Left unprojected, the
    crossing search fails on the shape - and before this channel existed the
    same mismatch was invisible, because the fabricated zero row was dropped
    below the direction floor before anyone could take its dot product.
    """
    rows = result.get('nac')
    if rows is None or not region_indices:
        return
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim == 3 and array.shape[1] == len(region_indices):
        result['nac'] = np.stack([
            _scatter_gradient(array[k], region_indices, natom)[0]
            for k in range(array.shape[0])])


def _field_skill(field_outcome):
    """Skill label of a field outcome, when its record carries one."""
    return str(getattr(field_outcome, 'skill_name', '') or '')


def native_field_rows(field_block, declaration=None):
    """The V/E rows a field task returned in its own result block.

    The skill's field task already hands the probe's potential and field
    back as `v_mm_native` / `e_mm_native`, labelled by `qm_atom_ids`;
    this reads them into the `[V, Ex, Ey, Ez]` rows the QM handoff
    carries, in the block's own id order.

    Parameters
    ----------
    field_block : dict
        The field result's `qmmm_field` block.
    declaration : dict, optional
        The skill's `qmmm.results` map (logical name -> written key);
        the historical names apply when it declares none.

    Returns
    -------
    list of list of float or None
        One row per probed atom, or None when the block carries no
        native columns (an older producer).

    Raises
    ------
    ValueError
        On a shape or finiteness problem: a truncated V/E must not be
        read as a shorter field.
    """
    if not isinstance(field_block, dict):
        return None
    declared = declaration or {}
    v_mm = field_block.get(declared.get('v_mm') or 'v_mm_native')
    e_mm = field_block.get(declared.get('e_mm') or 'e_mm_native')
    if v_mm is None or e_mm is None:
        return None
    ids = [int(index) for index in (field_block.get('qm_atom_ids') or [])]
    v_mm = np.asarray(v_mm, dtype=np.float64)
    e_mm = np.asarray(e_mm, dtype=np.float64)
    if v_mm.shape != (len(ids),) or e_mm.shape != (len(ids), 3):
        raise ValueError(
            "QMMM field block returned V/E with shapes %s/%s for %d atoms."
            % (v_mm.shape, e_mm.shape, len(ids)))
    if not np.all(np.isfinite(v_mm)) or not np.all(np.isfinite(e_mm)):
        raise ValueError("QMMM field block returned a non-finite V/E.")
    return [[float(v_mm[i])] + [float(value) for value in e_mm[i]]
            for i in range(len(ids))]


def assemble_qmmm(tasks, outcomes, request):
    """Stitch one three-stage QM/MM evaluation into the standard result.

    The queue is ``qmmm_field -> QM -> qmmm_energy``:

    * the field task (LAMMPS ``qmmm_field``) is an auxiliary A/B Coulomb
      probe: it contributes no energy, only the MM potential/field at the
      QM atoms that the QM calculation consumes;
    * the QM task runs the region in that ESPF external field, so its
      energy already contains the QM-MM electrostatics;
    * the energy task (LAMMPS ``qmmm_energy``) evaluates the full force
      field with the QM 0-order charges and reports the REDUCED classical
      surface: ``E_classic = U_main - U_gas - qmmm_el`` and the gradient
      ``dU_main/dR`` with ``dqmmm_el/dR`` and ``dU_gas/dR`` already removed
      from its QM rows. The removed pieces are reported beside it
      (``qmmm_el_gradient``, ``u_gas_gradient``) because one of them is
      reattachable.

    The stitched result is ``E = E_QM + E_classic``, and the gradient is the
    QM provider's own rows plus that reduced classical array.
    ``request['qm_contains_field_gradient']`` decides who carries the
    0-order field term ``-q*E``: ``True`` (the default, the Molcas ESPF
    contract) means the QM gradient already contains it and the classical
    side must not; ``False`` declares a fixed-potential QM provider (a
    polarizable NN, which reports only the derivative at constant V), and
    this function adds ``qmmm_el_gradient`` back onto the QM rows. The two
    branches hold exactly one copy of the term either way. The full
    ``U_main`` MM rows are unchanged, and no branch moves an energy.
    All standard arrays arrive in Hartree / Hartree-per-Bohr; the native
    QM/MM blocks are diagnostics only.

    Parameters
    ----------
    tasks : list of dict
        The three QMMM task descriptors (field, QM, energy).
    outcomes : list of TaskOutcome
        Outcomes index-aligned with tasks; all must have succeeded.
    request : dict
        Aggregation request with ``natom``, ``nstate``, ``need_gradient``
        and ``qm_contains_field_gradient``.

    Returns
    -------
    dict
        ``energy``, optional ``gradient``, ``qmmm_components``,
        ``composite_terms`` and ``composite_total``.

    Raises
    ------
    ValueError
        On a count/contribution/role mismatch, a missing field or energy
        block, a partition mismatch, or non-finite/ill-shaped data.
    """
    field_pairs, charge_pairs, qm_pairs, energy_pairs = _qmmm_duties(
        tasks, outcomes)

    natom = int(request.get('natom', 0))
    nstate = int(request.get('nstate', 1))
    need_gradient = bool(request.get('need_gradient', False))
    # Several centers carry their own state ladders (each its own `ci` and
    # `coupling`), so there is no global state axis to broadcast to: the
    # request's `active_configuration` picks one state per center and the
    # composite exposes a single energy/gradient at that configuration, the
    # same contract `_assemble_multi_center` follows. One center keeps the
    # historical axis, bit for bit.
    multi_center = bool(request.get('multi_center'))
    active_configuration = tuple(request.get('active_configuration') or ())
    # A crossing search names a pair of states per center (milestone M4).
    # The composite then answers TWO surfaces: state 0 is every center at the
    # first state of its own pair, state 1 at the second. The objective the
    # search follows is the difference of those two composites, which is the
    # sum of the per-center gaps, and the mean surface is their average - the
    # single-PES machinery of `meci`/`mecp` unchanged.
    pair_axis = request.get('pair_axis') or None
    # Who owns the 0-order field gradient term F_el = q_qm * E. The QM
    # provider either already contains it (Molcas ESPF, the default) or
    # does not (a fixed-potential model such as a polarizable NN), and the
    # classical Q-row correction is reduced accordingly so the term is
    # counted exactly once. An energy is never affected.
    qm_contains_field_gradient = bool(
        request.get('qm_contains_field_gradient', True))

    energy_task, energy_outcome = energy_pairs[0]

    centers, union, mm_indices = _qmmm_centers(
        qm_pairs, field_pairs, request, natom, nstate, need_gradient,
        multi_center, active_configuration, pair_axis)

    # --- field tasks (embedding only): a dependency, not an energy term --
    field_blocks = []
    if field_pairs:
        for center in centers:
            field_task, field_outcome = center['field_pair']
            field_block = (field_outcome.raw or {}).get('qmmm_field')
            if not isinstance(field_block, dict) or not field_block:
                raise ValueError(
                    "QMMM field task '%s' returned no qmmm_field block."
                    % field_outcome.task_id)
            field_ids = [int(index) for index in
                         (field_block.get('qm_atom_ids') or [])]
            if field_ids != [index + 1 for index in center['indices']]:
                raise ValueError(
                    "QMMM field partition mismatch: the probe labels QM ids "
                    "%s but task '%s' expects %s."
                    % (field_ids, center['task'].get('task_id'),
                       [index + 1 for index in center['indices']]))
            # One reader for the field block (the declared keys, the
            # shape and the finiteness), shared with the calculator's
            # handoff.
            from ...skills.query import get_declaration

            rows = native_field_rows(
                field_block,
                get_declaration(_field_skill(field_outcome) or '', 'qmmm')
                .get('results') or {})
            if rows is None:
                raise ValueError(
                    "QMMM field task '%s' returned no V/E columns."
                    % field_outcome.task_id)
            if len(rows) != len(center['indices']):
                raise ValueError(
                    "QMMM field task '%s' returned V/E for %d of %d QM "
                    "atoms."
                    % (field_outcome.task_id, len(rows),
                       len(center['indices'])))
            field_blocks.append(field_block)

    # --- energy task: the classical component and its Q-row correction --
    energy_raw = energy_outcome.raw or {}
    block = energy_raw.get('qmmm')
    atomic = energy_raw.get('qmmm_atomic')
    if not isinstance(block, dict) or not block:
        raise ValueError(
            "QMMM energy task '%s' returned no qmmm block."
            % energy_outcome.task_id)
    if not isinstance(atomic, dict) or not atomic:
        raise ValueError(
            "QMMM energy task '%s' returned no atomic-unit qmmm block."
            % energy_outcome.task_id)
    block_ids = [int(index) for index in (block.get('qm_atom_ids') or [])]
    if block_ids != [index + 1 for index in union]:
        raise ValueError(
            "QMMM energy partition mismatch: the block labels QM ids %s "
            "but the QM centers together hold %s."
            % (block_ids, [index + 1 for index in union]))

    try:
        e_classic = float(atomic['e_classic'])
        u_main = float(atomic['u_main'])
        u_gas = float(atomic['u_gas'])
        qmmm_el = float(atomic['qmmm_el'])
        qmmm_el_gradient = np.asarray(
            atomic['qmmm_el_gradient'], dtype=np.float64)
        u_gas_gradient = np.asarray(
            atomic['u_gas_gradient'], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "QMMM energy task '%s' atomic block is incomplete: %s."
            % (energy_outcome.task_id, exc))
    if not all(np.isfinite(value)
               for value in (e_classic, u_main, u_gas, qmmm_el)):
        raise ValueError(
            "QMMM energy task '%s' returned a non-finite component."
            % energy_outcome.task_id)
    if (qmmm_el_gradient.shape != (len(union), 3)
            or u_gas_gradient.shape != (len(union), 3)):
        raise ValueError(
            "QMMM energy task '%s' returned Q-row gradients with shapes "
            "%s/%s for %d QM atoms."
            % (energy_outcome.task_id,
               qmmm_el_gradient.shape, u_gas_gradient.shape, len(union)))
    native_residual = u_main - u_gas - qmmm_el - e_classic
    native_block = block.get('units') or atomic.get('native_unit_set')
    if abs(native_residual) > 1e-6 * max(1.0, abs(u_main)):
        raise ValueError(
            "QMMM energy task '%s' components do not add up: "
            "U_main=%r, U_gas=%r, qmmm_el=%r, E_classic=%r (residual %r)."
            % (energy_outcome.task_id, u_main, u_gas, qmmm_el,
               e_classic, native_residual))

    q_qm_values = block.get('q_qm')
    q_qm = [float(value) for value in
            (q_qm_values if q_qm_values is not None else [])]
    if len(q_qm) != len(union):
        raise ValueError(
            "QMMM energy task '%s' returned %d QM charges for %d atoms."
            % (energy_outcome.task_id, len(q_qm), len(union)))
    if not all(np.isfinite(value) for value in q_qm):
        raise ValueError(
            "QMMM energy task '%s' returned a non-finite q_qm."
            % energy_outcome.task_id)
    union_row = {index: slot for slot, index in enumerate(union)}
    q_of_atom = {index: value for index, value in zip(union, q_qm)}

    # E = sum_c E_QM^c + E_classic. The classical surface already carries
    # the double-count removal it owns: qmmm_el is every center's charge set
    # against the total field it was embedded in (MM plus the other
    # centers), and U_gas is the sum of the Q-Q pairs inside each center.
    # That leaves the inter-center 0-order interaction once, on the side that
    # can put a force on both centers' atoms.
    if multi_center:
        # One scalar, or the pair's two composites: each center answers at the
        # state the request names for it, and the classical surface is the
        # same term on both (it carries no electronic state).
        if pair_axis:
            energy = np.array(
                [float(e_classic)
                 + sum(float(center['energy'][center['states'][slot]])
                       for center in centers)
                 for slot in (0, 1)], dtype=np.float64)
        else:
            energy = np.array(
                [float(e_classic)
                 + sum(float(center['energy'][center['state']])
                       for center in centers)], dtype=np.float64)
    else:
        energy = e_classic + sum(center['energy'] for center in centers)

    result = {key: value for key, value in centers[0]['normed'].items()
              if key not in ('energy', 'gradient')}
    result['energy'] = energy
    if multi_center:
        # Couplings stay per center: each block goes on its own center's
        # atoms and the rows outside every center stay zero. No pair crosses
        # a center - no QM provider returns a derivative with respect to
        # another center's atoms - so the union is a block assembly, not a
        # sum, and `collect_center_electronic` keeps the per-center copies in
        # their local framing for the hopping dispatcher.
        result.pop('nac', None)
        result.pop('soc', None)
        nacs, socs = [], []
        for center in centers:
            block = {key: center['normed'][key] for key in ('nac', 'soc')
                     if key in center['normed']}
            if not block:
                continue
            _project_region_coupling(block, center['indices'], natom)
            if 'nac' in block:
                nacs.append(np.asarray(block['nac'], dtype=np.float64))
            if 'soc' in block:
                socs.append(
                    np.asarray(block['soc'], dtype=np.float64).reshape(-1))
        if nacs:
            result['nac'] = np.concatenate(nacs, axis=0)
        if socs:
            result['soc'] = np.concatenate(socs, axis=0)
        result['active_configuration'] = active_configuration
    else:
        _project_region_coupling(result, centers[0]['indices'], natom)

    if need_gradient:
        full = np.asarray(
            energy_raw.get('gradient'), dtype=np.float64)
        if full.ndim == 2:
            full = full[np.newaxis, :, :]
        if (full.ndim != 3 or full.shape[2] != 3
                or full.shape[1] != natom):
            raise ValueError(
                "QMMM energy task '%s' returned a full gradient with "
                "shape %s; expected (nstate, %d, 3)."
                % (energy_outcome.task_id, full.shape, natom))
        full = _state_broadcast(full, nstate)
        if full.shape[0] != nstate:
            raise ValueError(
                "QMMM energy task '%s' returned %d gradient states; "
                "expected %d."
                % (energy_outcome.task_id, full.shape[0], nstate))
        if multi_center:
            nstate_out = 2 if pair_axis else 1
        else:
            nstate_out = nstate
        gradient = np.zeros((nstate_out, natom, 3), dtype=np.float64)
        # QM rows: each center's own gradient, on its own atoms, plus the
        # classical Q-row correction the energy task reported for those rows.
        # The skill already removed both terms it knows the QM side owns
        # (F_el and F_gas) across the whole QM set, so per center this is a
        # row lookup, not a second reduction pass; the switch says whether
        # the field term has to go back.
        for center in centers:
            indices = center['indices']
            rows = [union_row[index] for index in indices]
            qm_gradient = np.asarray(
                center['normed']['gradient'], dtype=np.float64)
            if multi_center:
                qm_corr = full[:1, indices, :]
            else:
                qm_corr = full[:, indices, :]
            if not qm_contains_field_gradient:
                # A QM provider that reports only the fixed-potential
                # derivative does not contain -q*E, and the classical term
                # that was removed for it is exactly that piece.
                qm_corr = qm_corr + qmmm_el_gradient[rows, :]
            if multi_center:
                for slot in range(nstate_out):
                    index = center['states'][slot]
                    gradient[slot, indices, :] = (
                        qm_gradient[index] + qm_corr[0])
            else:
                gradient[:, indices, :] = qm_gradient + qm_corr
        # MM rows: the full U_main force (converted). It keeps the Q-M
        # Coulomb reaction force by the documented v1 contract.
        gradient[:, mm_indices, :] = full[:nstate_out, mm_indices, :]
        result['gradient'] = gradient

    # --- stress: the Flow-2 classical whole-system tensor ---------------
    # qmmm_energy evaluates U_main for the whole system, so its tensor is
    # the structural provider this composite can report. It is converted
    # to atomic units by the calculator's central conversion, exactly as
    # the generic composite path does.
    stress = energy_raw.get('stress_tensor')
    if stress is not None:
        result['stress_tensor'] = _stress_tensor(
            stress, nstate,
            "QMMM energy task '%s'" % energy_outcome.task_id)

    # The QM-only ladder: the diagnostic block and the label channels both
    # read it, so the sum is computed once.
    total_qm = np.zeros(nstate, dtype=np.float64)
    for center in centers:
        total_qm = total_qm + center['energy']
    result['qmmm_components'] = _qmmm_components(
        centers, field_blocks, atomic, block, q_qm, q_of_atom, union,
        qm_contains_field_gradient, e_classic, u_main, u_gas,
        qmmm_el, native_block, total_qm)
    result['composite_scheme'] = 'qmmm'
    collected = [
        (center['task'], [float(value) for value in center['energy']], 1)
        for center in centers
    ] + [(energy_task, [float(e_classic)], 1)]
    result['composite_terms'] = build_composite_terms(
        collected, multi_center=len(centers) > 1)
    result['composite_total'] = energy
    # The active surface the next step predicts from. A multi-center run has
    # no single global state, so the per-center configuration this evaluation
    # was asked to solve comes back out of the assembly; the MD loop stages
    # it and the following step reads it. Dropping it left the second step of
    # a multi-center QMMM run indexing an empty configuration.
    result['active_configuration'] = tuple(
        request.get('active_configuration') or ())
    states = np.asarray(energy, dtype=np.float64).reshape(-1)
    if states.size == 1:
        result['active_energy'] = float(states[0])
        gradient = result.get('gradient')
        if gradient is not None:
            result['active_gradient'] = np.asarray(gradient, dtype=np.float64)[0]

    channels = _qmmm_label_channels(
        tasks, result, centers, union, energy, total_qm, q_qm,
        e_classic, nstate, natom, need_gradient)
    if channels is not None:
        result['label_channels'] = channels
    return result


def _qmmm_label_channels(tasks, result, centers, union, energy, total_qm,
                         q_qm, e_classic, nstate, natom, need_gradient):
    """Build the QM label channels the adaptive reference plan asks for.

    The QM surface the ML potential replaces, plus the environment data
    the reference plan produced. The MM rows are deliberately zero: they
    belong to the classical provider, not to the QM label, while the
    field is the V/F rows handed to the QM calculation. None when the
    plan names no source.
    """
    label_plan = next(
        (task.get('label_plan') for task in tasks
         if task.get('label_plan')), None)
    if label_plan is None:
        return None
    qm_gradient = None
    if need_gradient:
        qm_gradient = np.zeros((nstate, natom, 3), dtype=np.float64)
        for center in centers:
            local = center['normed'].get('gradient')
            if local is None:
                continue
            local = np.asarray(local, dtype=np.float64)
            if local.ndim == 2:
                local = local[np.newaxis, :, :]
            local = _state_broadcast(local, nstate)
            if local.shape[0] != nstate:
                raise ValueError(
                    "QMMM QM task '%s' returned %d gradient states; "
                    "expected %d."
                    % (center['task'].get('task_id'),
                       local.shape[0], nstate))
            qm_gradient[:, center['indices'], :] = local
    source = label_plan.get('source')
    if source == 'total':
        # A theory single-method surface spans the whole system; the
        # QMMM total (QM + classical) is its label, including the MM
        # rows in the gradient.
        label_energy = np.asarray(energy, dtype=np.float64)
        label_gradient = result.get('gradient')
    else:
        label_energy = np.asarray(total_qm, dtype=np.float64)
        label_gradient = qm_gradient
    channels = {
        'source': source or 'layer',
        'region_indices': (list(label_plan.get('region_indices') or [])
                           or list(union)),
        'energy': label_energy,
        'gradient': label_gradient,
        'mm_energy': float(e_classic),
        'q_qm': list(q_qm),
    }
    _carry_couplings(channels, result)
    field_rows = []
    for center in centers:
        rows = (center['outcome'].raw or {}).get(
            'qmmm_external_potential')
        if rows:
            field_rows.extend([list(row) for row in rows])
    if field_rows:
        channels['field'] = field_rows
    return channels


def _qmmm_components(centers, field_blocks, atomic, block, q_qm, q_of_atom,
                    union, qm_contains_field_gradient, e_classic, u_main,
                    u_gas, qmmm_el, native_block, total_qm):
    """Describe the QM/MM evaluation the assembly actually stitched.

    The semantics, the classical decomposition, the per-center rows and
    the approximations the result relies on. Every number here is a
    diagnostic: the energy and gradient the run consumes are assembled
    above, and this block says how to read them.
    """
    qm_config = centers[0]['task'].get('config') or {}
    qm_skill = (qm_config.get('skill_config')
                if isinstance(qm_config, dict) else {}) or {}
    representation = str(qm_skill.get('embedding') or '')
    if representation not in ('none', 'charge', 'potential'):
        representation = ('potential' if bool(block.get('embedding', True))
                          else 'none')
    energy_semantics = {
        'charge': 'esfp_charge_embedding',
        'potential': 'esfp_external_potential_two_flow',
        'none': 'isolated_qm_plus_classical',
    }.get(representation, 'esfp_external_potential_two_flow')
    if qm_contains_field_gradient:
        qm_rows = ('the QM rows are the QM gradient plus the classical Q-row '
                   'correction the skill already reduced '
                   '(dU_main - dqmmm_el - dU_gas); the field term stays '
                   'removed because the QM provider contains it')
    else:
        qm_rows = ('the QM rows are the QM gradient plus the classical Q-row '
                   'correction with dqmmm_el added back: the provider is '
                   'declared not to contain the 0-order field term, so the '
                   'classical side supplies the -q*E it is missing')
    approximations = [
        qm_rows,
        'the MM rows keep the full U_main force, including the Q-M '
        'Coulomb reaction force, which is not part of E_classic',
        'v1 clones the QM LAMMPS types so the Q-Q non-Coulomb terms are '
        'zero; the Q-Q image Coulomb stays in E_classic',
        'the reported stress is the unreduced classical U_main virial: the '
        'terms removed from the QM rows are restored to atoms by the QM '
        'side but not to the cell, so the cell response of qmmm_el and '
        'U_gas stays at the point-charge level',
    ]
    if field_blocks:
        approximations.append(
            'the field is the q_Q=+1 A/B probe; its potential/field is an '
            'approximation to the periodic QM operator')
    if len(centers) > 1:
        approximations.extend([
            # The statement R12 of the design makes explicit: a center's
            # wavefunction responds to where the other centers sit, and no
            # provider returns a derivative with respect to another center's
            # atoms. What the result holds instead is that coupling at the
            # 0-order level, carried by the classical surface.
            'the inter-center coupling is carried by the classical surface '
            'at the 0-order level: no QM provider returns a derivative with '
            'respect to another center\'s atoms, so the response of one '
            'center\'s density to a displacement of another is not in the '
            'gradient',
            'the 0-order inter-center pair is subtracted once per center '
            'from the classical side (qmmm_el uses each center\'s total '
            'field), which leaves exactly one copy of it in the energy',
        ])
    if representation == 'charge':
        approximations.append(
            'charge embedding uses the MM topology charges; under PBC the '
            'ESPF calculation lacks the image contributions the classical '
            'qm_mm term carries')
    per_center = []
    for position, center in enumerate(centers):
        block_center = (field_blocks[position] if field_blocks else None)
        per_center.append({
            'center_index': center['task'].get('center_index'),
            'task_id': center['task'].get('task_id'),
            'qm_atoms': list(center['indices']),
            'qm_energy': [float(value) for value in center['energy']],
            'q_qm': [q_of_atom[index] for index in center['indices']],
            'u_gas': (None if not atomic.get('u_gas_centers')
                      else float(atomic['u_gas_centers'][position])),
            # The rows this center was actually embedded in.
            'external_potential': (center['outcome'].raw or {}).get(
                'qmmm_external_potential'),
            'field': (None if block_center is None else {
                'probe_charge': float(block_center.get('probe_charge', 1.0)),
                'sanity': dict(block_center.get('sanity') or {}),
                'units': dict(block_center.get('units') or {}),
            }),
        })
    return {
        'energy_semantics': energy_semantics,
        'embedding': representation,
        'n_center': len(centers),
        'qm_energy': [float(value) for value in total_qm],
        'e_classic': e_classic,
        'u_main': u_main,
        'u_gas': u_gas,
        'qmmm_el': qmmm_el,
        # Which side owns the 0-order field gradient term of this run.
        'qm_contains_field_gradient': qm_contains_field_gradient,
        'q_qm': q_qm,
        'qm_atoms': list(union),
        # The field rows actually handed to the QM calculation (atomic
        # units), per center; the union view is the concatenation of these.
        'external_potential': (centers[0]['outcome'].raw or {}).get(
            'qmmm_external_potential') if len(centers) == 1 else None,
        'centers': per_center,
        'stress': 'classical_u_main',
        'field': (None if not field_blocks else {
            'probe_charge': float(field_blocks[0].get('probe_charge', 1.0)),
            'sanity': dict(field_blocks[0].get('sanity') or {}),
            'units': dict(field_blocks[0].get('units') or {}),
        }),
        'energy_sanity': dict(block.get('sanity') or {}),
        'units': {
            'framework': {'energy': 'hartree',
                          'gradient': 'hartree/bohr'},
            'native': native_block,
        },
        'approximations': approximations,
    }


def _qmmm_centers(qm_pairs, field_pairs, request, natom, nstate,
                  need_gradient, multi_center, active_configuration, pair_axis):
    """Normalize each QM center and name the atoms the classical side owns.

    One entry per QM center: its region, its normalized result, and --
    with embedding -- the probe that measured it. The centers' union is
    what the classical surface addresses its charges and its Q-row
    corrections by, in global atom order, so those reductions stay one
    pass over the QM rows while the energies and gradients stay per
    center (dev/methods/QMMM_DEVELOPMENT.md section 3.1).

    Returns ``(centers, union, mm_indices)``.
    """
    centers = []
    for position, (qm_task, qm_outcome) in enumerate(qm_pairs):
        indices = [int(index) for index in
                   (qm_task.get('region_indices') or [])]
        if not indices:
            raise ValueError(
                "QMMM QM task '%s' has an empty region."
                % qm_task.get('task_id'))
        normed = normalize_task_result(
            qm_outcome.raw, qm_task,
            {'nstate': nstate, 'nlocal': len(indices),
             'need_gradient': need_gradient})
        energy = np.asarray(normed['energy'], dtype=np.float64).reshape(-1)
        if multi_center:
            if energy.size < 1:
                raise ValueError(
                    "QMMM QM task '%s' returned no state energy."
                    % qm_outcome.task_id)
            if pair_axis:
                picks = tuple(int(value) for value in
                              (pair_axis[position] or (0, 0)))
                if len(picks) != 2:
                    raise ValueError(
                        "QMMM center %d declares %d states in its pair; a "
                        "crossing search reads exactly two."
                        % (position, len(picks)))
            else:
                picks = ((int(active_configuration[position])
                          if position < len(active_configuration) else 0),)
            for index in picks:
                if not 0 <= index < energy.size:
                    raise ValueError(
                        "QMMM center %d has %d states; the request names "
                        "state %d." % (position, energy.size, index))
            state_index = picks[0]
        else:
            state_index = None
            energy = _state_broadcast(energy, nstate)
            if energy.size != nstate:
                raise ValueError(
                    "QMMM QM task '%s' returned %d states; expected %d."
                    % (qm_outcome.task_id, energy.size, nstate))
        centers.append({
            'task': qm_task,
            'outcome': qm_outcome,
            'normed': normed,
            'indices': indices,
            'energy': energy,
            'state': state_index,
            'states': (picks if multi_center else None),
            'field_pair': field_pairs[position] if field_pairs else None,
        })

    union = sorted({index for center in centers
                    for index in center['indices']})
    if len(union) != sum(len(center['indices']) for center in centers):
        raise ValueError(
            "QMMM centers overlap: %d QM atoms across %d centers, the union "
            "holds %d. One atom cannot carry two centers' 0-order charges."
            % (sum(len(center['indices']) for center in centers),
               len(centers), len(union)))
    qm_set = set(union)
    mm_indices = [index for index in range(natom) if index not in qm_set]
    return centers, union, mm_indices


def _qmmm_duties(tasks, outcomes):
    """Group the outcomes by QM/MM duty and check the queue's shape.

    The queue is field -> qm_charge -> qm -> energy. Count and
    contribution checks are here because they describe the plan, not the
    numbers: a queue that does not hold one probe per QM center, one
    charge round per center and exactly one classical evaluation cannot
    be stitched at all.

    Returns ``(field_pairs, charge_pairs, qm_pairs, energy_pairs)``.
    """
    if len(tasks) != len(outcomes) or not tasks:
        raise ValueError(
            "QMMM assembly needs one outcome per task; got %d tasks and %d "
            "outcomes." % (len(tasks), len(outcomes)))
    by_role = {'field': [], 'qm': [], 'energy': [], 'qm_charge': []}
    for task, outcome in zip(tasks, outcomes):
        role = task.get('qmmm_role')
        if role not in by_role:
            raise ValueError(
                "QMMM assembly takes the field, qm_charge, qm and energy "
                "duties only; got roles %s."
                % [t.get('qmmm_role') for t in tasks])
        by_role[role].append((task, outcome))
    qm_pairs = by_role['qm']
    field_pairs = by_role['field']
    charge_pairs = by_role['qm_charge']
    energy_pairs = by_role['energy']
    if not qm_pairs:
        raise ValueError(
            "QMMM assembly needs at least one QM task; got roles %s."
            % [t.get('qmmm_role') for t in tasks])
    if len(energy_pairs) != 1:
        # One classical evaluation of the whole system: the MM-MM terms
        # belong to it once, and per-center classical tasks would count
        # them once per center.
        raise ValueError(
            "QMMM assembly needs exactly one energy task; got %d."
            % len(energy_pairs))
    if field_pairs and len(field_pairs) != len(qm_pairs):
        raise ValueError(
            "QMMM assembly needs one probe per QM center; got %d field and "
            "%d QM tasks." % (len(field_pairs), len(qm_pairs)))
    # Assembly-side half of the queue-shape contract. The execution-side
    # half is `Calculator._run_qmmm`; this function is also a public entry
    # point, so it cannot assume the queue came from that path.
    if not field_pairs and len(tasks) != len(qm_pairs) + 1:
        raise ValueError(
            "QMMM without probes holds %d QM tasks and one energy task; got "
            "%d tasks." % (len(qm_pairs), len(tasks)))
    if charge_pairs and len(charge_pairs) != len(qm_pairs):
        raise ValueError(
            "QMMM assembly needs one charge round per QM center; got %d "
            "charge and %d QM tasks." % (len(charge_pairs), len(qm_pairs)))
    contributions = [outcome.contribution for outcome in outcomes]
    expected = (([0] * len(qm_pairs) if field_pairs else [])
                + [0] * len(charge_pairs)
                + [1] * len(qm_pairs) + [1])
    if contributions != expected:
        raise ValueError(
            "QMMM contributions must be %s; got %s."
            % (expected, contributions))

    for outcome in outcomes:
        if not outcome.success:
            raise ValueError(
                "Task '%s' failed: %s"
                % (outcome.task_id, outcome.error))
    return field_pairs, charge_pairs, qm_pairs, energy_pairs


def _carry_couplings(channels, result):
    """Copy the assembled couplings a label consumer may read.

    The channels are a plain mapping so a remote result.json carries them
    unchanged; a coupling the source itself placed stays where it is.
    """
    for key in ('nac', 'soc'):
        if key not in channels and result.get(key) is not None:
            channels[key] = result[key]
    return channels


def _label_channels_from_terms(plan, normed_by_id, region_by_id, result,
                               total_energy, total_gradient, nstate, natom):
    """Build the training-label channels from the planned source.

    A 'layer' source takes the raw normalized result of the reference
    task whose region matches the neural-network surface; a 'total'
    source takes the assembled composite. The channels are a plain
    mapping so a remote result.json carries them unchanged.
    """
    channels = {
        'source': plan.get('source'),
        'region_indices': list(plan.get('region_indices') or []),
    }
    if plan.get('source') == 'total':
        channels['energy'] = np.asarray(total_energy, dtype=np.float64)
        if total_gradient is not None:
            channels['gradient'] = np.asarray(
                total_gradient, dtype=np.float64)
    else:
        task_ids = list(plan.get('source_task_ids') or [])
        if not task_ids:
            raise ValueError(
                "the label plan names a layer source but no task id.")
        task_id = task_ids[0]
        normed = normed_by_id.get(task_id)
        if normed is None:
            raise ValueError(
                "the label plan names task '%s' but the reference outcome "
                "set does not carry it." % task_id)
        energy = np.asarray(
            normed.get('energy'), dtype=np.float64).reshape(-1)
        energy = _state_broadcast(energy, nstate)
        if energy.size != nstate:
            raise ValueError(
                "the label task '%s' returned %d states; expected %d."
                % (task_id, energy.size, nstate))
        channels['energy'] = energy
        gradient = normed.get('gradient')
        if gradient is not None:
            channels['gradient'] = _scatter_gradient(
                gradient, region_by_id.get(task_id, []), natom)
        coupling = {key: normed[key] for key in ('nac', 'soc')
                    if normed.get(key) is not None}
        if coupling:
            _project_region_coupling(
                coupling, region_by_id.get(task_id, []), natom)
            channels.update(coupling)
    return _carry_couplings(channels, result)


def assemble_results(tasks, outcomes, request):
    """Aggregate task outcomes into a single composite result.

    For each outcome:
    1. Normalise the raw result (shape/state/unit checks).
    2. Scatter the local gradient to the full system.
    3. Multiply by the contribution coefficient.
    4. Accumulate energy and gradient.

    Parameters
    ----------
    tasks : list of dict
        Task descriptors (same order as outcomes).
    outcomes : list of TaskOutcome
        Per-task outcomes from backend.calculate().
    request : dict
        Aggregation request with keys:
        - natom : int (total atoms)
        - nstate : int (electronic states)
        - need_gradient : bool
        - multi_center : bool
        - primary_task_id : str or None

    Returns
    -------
    dict
        Assembled result with 'energy' (ndarray) and optionally
        'gradient' (ndarray (nstate, natom, 3)), plus non-aggregated
        keys from the primary task.

    Raises
    ------
    ValueError
        If any required outcome failed or data is invalid.
    """
    natom = int(request.get('natom', 0))
    need_gradient = bool(request.get('need_gradient', False))
    multi_center = bool(request.get('multi_center', False))

    # QMMM has its own stitching contract (E_QM + mm_mm, QM rows from the
    # QM calculation, MM rows from the classical one); the generic
    # contribution sum would double count qm_mm.
    if request.get('scheme') == 'qmmm':
        return assemble_qmmm(tasks, outcomes, request)

    # Check all outcomes succeeded
    for outcome in outcomes:
        if not outcome.success:
            raise ValueError(
                "Task '%s' failed: %s"
                % (outcome.task_id, outcome.error))

    # C>1 builds NO energy[nstate] public state axis (docs 8): the
    # total result is the ONIOM U(R;s)/G(R;s) at the active
    # configuration; per-center NAC/SOC/dipole are collected separately
    # (collect_center_electronic), never summed nor inherited from the
    # primary task.
    if multi_center:
        return _assemble_multi_center(
            tasks, outcomes, request, natom, need_gradient)

    # ---- C=1 traditional full-state view (unchanged) ----
    nstate = int(request.get('nstate', 1))
    primary_id = request.get('primary_task_id')
    system_id = request.get('system_task_id')

    # Initialise accumulators
    total_energy = np.zeros(nstate, dtype=np.float64)
    total_gradient = None
    if need_gradient:
        total_gradient = np.zeros((nstate, natom, 3), dtype=np.float64)

    primary_result = None
    primary_region = []
    system_result = None
    collected = []
    normed_by_id = {}
    region_by_id = {}
    charges_by_id = {}

    for task, outcome in zip(tasks, outcomes):
        # Build per-task request
        nlocal = (len(outcome.region_indices)
                  if outcome.region_indices
                  else natom)
        task_request = dict(request)
        task_request['nlocal'] = nlocal

        # Normalise
        normed = normalize_task_result(outcome.raw, task, task_request)
        normed_by_id[outcome.task_id] = normed
        region_by_id[outcome.task_id] = list(
            outcome.region_indices or [])
        if task.get('region') in ('inter', 'real'):
            charges_by_id[outcome.task_id] = (
                outcome.raw or {}).get('computed_charges')

        # Collect this term's raw per-state energy for the &composite
        # breakdown (captured before any single->nstate broadcast).
        collected.append((
            task,
            np.asarray(normed['energy'], dtype=float).reshape(-1).tolist(),
            outcome.contribution))

        # Track primary result for non-aggregated keys
        if primary_id is not None and outcome.task_id == primary_id:
            primary_result = dict(normed)
            primary_region = outcome.region_indices
        elif primary_result is None:
            primary_result = dict(normed)
            primary_region = outcome.region_indices

        # Track the whole-system structural provider (docs 4A)
        if system_id is not None and outcome.task_id == system_id:
            system_result = dict(normed)

        contrib = outcome.contribution

        # Energy: weighted sum
        energy = normed['energy']
        # Broadcast single-state energy to match nstate
        energy = _state_broadcast(energy, nstate)
        if energy.shape[0] != nstate:
            raise ValueError(
                "Task '%s' energy has %d states but expected %d."
                % (outcome.task_id, energy.shape[0], nstate))
        total_energy += contrib * energy

        # Gradient: scatter-add then weighted sum
        gradient = normed.get('gradient')
        if gradient is not None and total_gradient is not None:
            global_grad = _scatter_gradient(
                gradient, outcome.region_indices, natom)
            total_gradient += contrib * global_grad

    result = {}
    if primary_result:
        result.update(primary_result)
    # A derivative coupling is a per-nuclear-coordinate vector of the
    # electronic region that produced it, so a region provider hands back
    # (npair, nlocal, 3) while everything that steps on it works in
    # whole-system coordinates. The atoms outside the region get no
    # coupling row - the wavefunction that the derivative acts on does not
    # extend to them - which is the same statement the region's own gradient
    # makes, and it is what lets the crossing search address one vector.
    _project_region_coupling(result, primary_region, natom)
    # Structural track (docs 4A): whole-system structural properties
    # come from the full-system provider task, NOT the electronic
    # primary inherited above.
    if system_result is not None:
        for key in SYSTEM_STRUCTURAL_KEYS:
            if key in system_result:
                result[key] = system_result[key]
    result['energy'] = total_energy
    if total_gradient is not None:
        result['gradient'] = total_gradient
    # &composite energy breakdown: per-term raw energies (high->mid->low)
    # plus the per-state ONIOM total (C=1 carries the full nstate axis).
    result['composite_terms'] = build_composite_terms(
        collected, multi_center=False)
    result['composite_total'] = total_energy
    result['composite_scheme'] = request.get('scheme') or 'oniom'

    label_plan = next(
        (task.get('label_plan') for task in tasks
         if task.get('label_plan')), None)
    if label_plan is not None:
        channels = _label_channels_from_terms(
            label_plan, normed_by_id, region_by_id, result,
            total_energy, total_gradient, nstate, natom)
        if 'mid_charges' in (label_plan.get('aux') or []):
            mid = next((task for task in tasks
                        if task.get('region') == 'inter'), None)
            if mid is not None:
                charges = charges_by_id.get(mid.get('task_id'))
                if charges is not None:
                    channels['mid_charges'] = list(charges)
        if 'low_charges' in (label_plan.get('aux') or []):
            low = next((task for task in tasks
                        if task.get('region') == 'real'), None)
            if low is not None:
                charges = charges_by_id.get(low.get('task_id'))
                if charges is not None:
                    channels['low_charges'] = list(charges)
        result['label_channels'] = channels

    return result


def _task_state_index(task, active_configuration):
    """Local state a composite term contributes at (docs 8).

    A center's high-level provider H(c) (provides_electronic,
    state_policy='active') follows the center's active local state
    s_c; a fixed-reference correction/baseline term -- including the
    L(real) baseline (center_index None) -- contributes at its declared
    reference_state (0 when unspecified, i.e. state independent). The
    reference terms never follow a center's hop, so only a provider
    term changes when its center hops (docs 8: dU equals that center's
    high-level gap).

    Parameters
    ----------
    task : dict
        Task descriptor.
    active_configuration : tuple of int
        Per-center active local states (s_0, ..., s_{C-1}).

    Returns
    -------
    int
        The local state index this term contributes at.
    """
    if (task.get('provides_electronic')
            and task.get('state_policy') == 'active'):
        c = task.get('center_index')
        if c is not None and 0 <= c < len(active_configuration):
            return int(active_configuration[c])
        return 0
    ref = task.get('reference_state')
    return int(ref) if ref is not None else 0


def _select_state_energy(energy, state_index, task_id):
    """One term's energy at the selected local state (docs 8).

    A single-state term is state independent (a reference/baseline);
    otherwise the selected state must be in range.
    """
    arr = np.asarray(energy, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return float(arr[0])
    if 0 <= state_index < arr.size:
        return float(arr[state_index])
    raise ValueError(
        "Task '%s' active state %d is outside its %d computed states."
        % (task_id, state_index, arr.size))


def _select_state_gradient(gradient, state_index, task_id):
    """One term's gradient (nlocal, 3) at the selected local state."""
    arr = np.asarray(gradient, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0]
        if 0 <= state_index < arr.shape[0]:
            return arr[state_index]
        raise ValueError(
            "Task '%s' active state %d is outside its %d computed "
            "gradient states." % (task_id, state_index, arr.shape[0]))
    raise ValueError(
        "Task '%s' gradient has unexpected shape %s."
        % (task_id, arr.shape))


def _assemble_multi_center(tasks, outcomes, request, natom, need_gradient):
    """ONIOM U(R;s) / G(R;s) at the active configuration (docs 8, r4b).

    C>1 builds NO energy[nstate] public state axis. Each term
    contributes its energy/gradient at the state selected by its policy
    (the center's active local state for a provider H(c), the fixed
    reference state for a correction/baseline term), multiplied by its
    contribution coefficient and, for the gradient, scatter-added from
    the region to the full system::

        U(R;s) = sum_t coeff_t * E_t(state_t)
        G(R;s) = sum_t coeff_t * scatter(grad_t(state_t))
        F(R;s) = -G(R;s)   (the integrators negate G; not stored here)

    Per-center NAC/SOC/dipole are NOT summed here;
    collect_center_electronic saves them as-is into state.electronic[c].

    Parameters
    ----------
    tasks : list of dict
        Task descriptors (same order as outcomes).
    outcomes : list of TaskOutcome
        Per-task outcomes (all successful; checked by the caller).
    request : dict
        Aggregation request; reads 'active_configuration'.
    natom : int
        Total atoms in the system.
    need_gradient : bool
        Whether the global gradient G(R;s) is assembled.

    Returns
    -------
    dict
        {'energy': (1,) [U], 'active_energy': U,
        'active_configuration': s}, plus 'gradient' (1, natom, 3) and
        'active_gradient' (natom, 3) when need_gradient.
    """
    active_configuration = tuple(request.get('active_configuration') or ())
    active_energy = 0.0
    active_gradient = (np.zeros((natom, 3), dtype=np.float64)
                       if need_gradient else None)
    collected = []
    system_id = request.get('system_task_id')
    system_result = None

    for task, outcome in zip(tasks, outcomes):
        nlocal = (len(outcome.region_indices)
                  if outcome.region_indices else natom)
        task_request = dict(request)
        task_request['nlocal'] = nlocal
        normed = normalize_task_result(outcome.raw, task, task_request)

        # Track the whole-system structural provider (docs 4A): even
        # for C>1 the full-system baseline supplies a single global
        # stress, so it is never split per center.
        if system_id is not None and outcome.task_id == system_id:
            system_result = dict(normed)

        # Collect this term's raw per-state energy for the &composite
        # breakdown (C>1 keeps every state the backend computed).
        collected.append((
            task,
            np.asarray(normed['energy'], dtype=float).reshape(-1).tolist(),
            outcome.contribution))

        state_index = _task_state_index(task, active_configuration)
        contrib = outcome.contribution
        active_energy += contrib * _select_state_energy(
            normed['energy'], state_index, outcome.task_id)

        gradient = normed.get('gradient')
        if gradient is not None and active_gradient is not None:
            local_grad = _select_state_gradient(
                gradient, state_index, outcome.task_id)
            global_grad = _scatter_gradient(
                local_grad[np.newaxis, :, :], outcome.region_indices, natom)
            active_gradient += contrib * global_grad[0]

    result = {
        'energy': np.array([active_energy], dtype=np.float64),
        'active_energy': float(active_energy),
        'active_configuration': active_configuration,
    }
    # Structural track (docs 4A): whole-system structural properties
    # come from the full-system provider task (the center-less L(real)
    # baseline), giving C>1 the same single authoritative system stress
    # as C=1.
    if system_result is not None:
        for key in SYSTEM_STRUCTURAL_KEYS:
            if key in system_result:
                result[key] = system_result[key]
    if active_gradient is not None:
        result['gradient'] = active_gradient[np.newaxis, :, :]
        result['active_gradient'] = active_gradient
    # &composite energy breakdown: per-term raw energies (high->mid->low,
    # per center). C>1 has no global state axis, so the total is the
    # active-combination U (docs 8), a single value.
    result['composite_terms'] = build_composite_terms(
        collected, multi_center=True, active_configuration=active_configuration)
    result['composite_total'] = np.array([active_energy], dtype=np.float64)
    # The multi-center assembler must name its scheme too: the log's
    # multiscale view keys on it, and without it a C>1 ONIOM run fell back
    # to the plain `&point charges` block over the whole system.
    result['composite_scheme'] = request.get('scheme') or 'oniom'
    return result


def collect_center_electronic(tasks, outcomes, request):
    """Collect per-center electronic properties as-is (docs 8, r4b).

    Each center's high-level provider H(c) (provides_electronic=True)
    is the SOLE source of that center's energy / gradient / NAC / SOC /
    dipole. They are normalized and saved per center in the center's
    LOCAL framing (energy (S_c,), gradient (S_c, A_c, 3), NAC
    (P_nac,c, A_c, 3), SOC (P_soc,c,)) -- never ONIOM-summed and never
    inherited from the primary task. r4c projects them to the global
    frame when a consumer needs it.

    Parameters
    ----------
    tasks : list of dict
        Task descriptors (same order as outcomes).
    outcomes : list of TaskOutcome
        Per-task outcomes.
    request : dict
        Aggregation request (natom / nlocal sizing).

    Returns
    -------
    dict
        {center_index: {'energy':..., 'gradient':..., 'nac':...,
        'soc':..., 'dipole':...}} for storage in state.electronic[c].
        Centers without a successful provider are omitted; only keys
        the provider actually returned are present.
    """
    natom = int(request.get('natom', 0))
    per_center = {}
    for task, outcome in zip(tasks, outcomes):
        if not outcome.success or not task.get('provides_electronic'):
            continue
        c = task.get('center_index')
        if c is None:
            continue
        nlocal = (len(outcome.region_indices)
                  if outcome.region_indices else natom)
        task_request = dict(request)
        task_request['nlocal'] = nlocal
        normed = normalize_task_result(outcome.raw, task, task_request)
        carrier = {}
        if normed.get('energy') is not None:
            carrier['energy'] = normed['energy']
        if normed.get('gradient') is not None:
            carrier['gradient'] = normed['gradient']
        for key in ('nac', 'soc', 'dipole', 'transition_dipole'):
            value = normed.get(key)
            if value is not None:
                carrier[key] = np.asarray(value, dtype=np.float64)
        per_center[int(c)] = carrier
    return per_center
