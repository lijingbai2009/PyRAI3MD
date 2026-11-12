"""
======================================================================
Charge-update routing for ONIOM electrostatic embedding.

This module maps backend charge byproducts onto the global per-atom
ChargeStore and derives the embedding field a high-level task needs.
It answers, per task:

- which tasks are charge SOURCES (the high-level provider H(c) and the
  mid-level inter term M(I_c)),
- how a source's computed_charges scatter onto global atom indices
  (highlevel <- theory[0]; mid shell <- theory[1] inter, highlevel
  part ignored; lowlevel never updates),
- which point charges embed a given high-level task (its own center's
  mid shell).

Design rules (one-direction dependency, mirroring composite.py):
- The calculator uses the functions of this module.
- This module never imports the calculator, the state or any skill.
- It performs pure routing: it reads task descriptors and outcome
  payloads and writes only to the ChargeStore it is handed. Which
  atoms form a center's mid shell is decided by data.charges, so the
  region taxonomy stays single-sourced.

Sep 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from ...kernel.logger import get_logger

logger = get_logger('pyrai3md.charges')


def region_charge_sources(tasks):
    """Split a task queue into its two charge-source groups.

    The high-level provider H(c) of every center (provides_electronic)
    sources the highlevel charges from theory[0]; the mid-level inter
    term M(I_c) (layer_id 1, region 'inter') sources the mid-shell
    charges from theory[1]. Every other term -- the full-system
    baseline L(real), the low-level inter correction -L(I_c) and the
    mid-level model correction -M(M_c) -- returns no charge update.

    Parameters
    ----------
    tasks : list of dict
        Task descriptors of one category in execution order.

    Returns
    -------
    tuple(list of dict, list of dict)
        (high_tasks, mid_inter_tasks).
    """
    high_tasks = [t for t in tasks if t.get('provides_electronic')]
    mid_inter_tasks = [t for t in tasks
                       if t.get('layer_id') == 1
                       and t.get('region') == 'inter']
    return high_tasks, mid_inter_tasks


def _outcome_raw(outcome):
    """Return the raw result dict of a routed outcome or executor dict."""
    if isinstance(outcome, dict):
        return outcome.get('raw') or {}
    return getattr(outcome, 'raw', None) or {}


def _outcome_success(outcome):
    """Return whether an outcome succeeded (missing key means failed)."""
    if isinstance(outcome, dict):
        return bool(outcome.get('success', False))
    return bool(getattr(outcome, 'success', False))


def _charge_row(raw, state=0):
    """Extract the per-atom charge row of the state a task ran.

    Backends pack computed_charges as (nstate, nlocal) - one row per state,
    in the order the engine reports its roots - and the ONIOM path embeds
    with the state its high-level task ran, exactly like `Calculator.
    _qmmm_charges` does. Row 0 is the answer only when that state is 0: a
    payload that carries fewer rows than the requested state writes **no**
    row (the caller logs it) rather than embedding one surface's charges
    under another surface's name. A bare 1-D (nlocal,) array is accepted
    too. Returns None when the backend produced no charges (e.g. openqp)
    or an empty payload.

    Parameters
    ----------
    raw : dict
        Raw backend result dictionary.
    state : int
        0-based index of the state whose row this task embeds with.

    Returns
    -------
    numpy.ndarray or None
        Shape (nlocal,) per-atom charges, or None when unavailable.
    """
    payload = raw.get('computed_charges')
    if payload is None:
        return None
    arr = np.asarray(payload, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        if arr.shape[0] == 0:
            return None
        if not 0 <= int(state) < arr.shape[0]:
            logger.warning(
                "computed_charges carries %d state row(s) but state %d was "
                "requested; no row is written (row 0 would embed one "
                "surface's charges under another's name)",
                arr.shape[0], int(state))
            return None
        return arr[int(state)]
    if arr.ndim == 1:
        return arr
    return None


def _state_index(state, task):
    """The 0-based state whose charge row a task embeds with."""
    if callable(state):
        try:
            return int(state(task))
        except (TypeError, ValueError):
            return 0
    if state is None:
        return 0
    return int(state)


def capture_outer_reference(tasks, outcomes, state=None):
    """Capture the full-system low-level charge row for ONIOM2 embedding.

    A two-layer ONIOM run has no mid-level term: the outer region is filled
    by the whole-system low-level task ``L(real)``. Its row is the
    environment every high-level center is embedded in, so it is captured
    once per evaluation and handed to phase two directly. That is what
    lets several centers share one source: center c's environment is every
    atom outside its own highlevel, including another center's high-level
    atoms, and those rows carry the low-level charges rather than a second
    center's high-level charges (no cross-center charge iteration).

    The persistent ChargeStore is deliberately not written here. Its
    highlevel rows belong to the high-level providers (a source that emits
    no charges must not leave theory[1]'s numbers under the high-level
    name), which is the #450 rule; this capture is a separate, per-evaluation
    array.

    Parameters
    ----------
    tasks : list of dict
        Phase-one task descriptors (non-providers).
    outcomes : list
        Routed outcomes index-aligned with ``tasks``.
    state : int or callable, optional
        State whose row of a per-state ``computed_charges`` payload this
        task embeds with - the same resolver ``update_from_outcomes`` uses.

    Returns
    -------
    numpy.ndarray or None
        Shape (natom,) low-level charges, or None when the queue has no
        full-system term or the term produced no charges (the caller logs
        the consequence once).
    """
    def state_of(task):
        return _state_index(state, task)

    found = False
    for task, outcome in zip(tasks, outcomes):
        if task.get('provides_electronic'):
            continue
        if (task.get('region') != 'real'
                or task.get('layer_id') != 1
                or task.get('center_index') is not None):
            continue
        found = True
        if not _outcome_success(outcome):
            logger.warning(
                "ONIOM2 embedding: the full-system outer task '%s' failed; "
                "the high-level centres run without environment charges.",
                task.get('task_id'))
            return None
        values = _charge_row(_outcome_raw(outcome), state_of(task))
        if values is None:
            logger.warning(
                "ONIOM2 embedding: the full-system outer task '%s' produced "
                "no computed_charges; the high-level centres run without "
                "environment charges.", task.get('task_id'))
            return None
        return np.asarray(values, dtype=float).reshape(-1)
    if found:
        return None
    logger.warning(
        "ONIOM2 embedding: no full-system outer task was found in the queue; "
        "the high-level centres run without environment charges.")
    return None


def update_from_outcomes(charge_store, tasks, outcomes, sources=None,
                         state=None):
    """Scatter backend computed_charges onto the managed region.

    Index-aligned pairing of tasks with their outcomes. For each
    successful source task the per-atom charges are written to the
    global atom indices of the region:

    - high-level provider H(c): every atom of region_indices (= H_c)
      is updated (highlevel charges <- theory[0]).
    - mid-level inter term M(I_c): its region is region_m = H_c ∪ I_c;
      only the midlevel part I_c is written (midlevel charges <-
      theory[1]) and the highlevel part is ignored.
    - any other task, or a source whose backend produced no
      computed_charges, is skipped with an info log (never an error).

    Which rows belong to H_c is a property of the partition, not of the
    batch being scattered, so an embedded run that splits its queue into
    phases hands the whole queue here through ``sources``. Asked of one
    phase alone the answer changes: the phase carrying the mid-level and
    outer terms has no provider in it, so nothing says which rows are the
    high-level ones, and theory[1] writes over them - rows the provider
    would have replaced had it emitted charges at all, which bagel and
    openqp do not.

    Parameters
    ----------
    charge_store : ChargeStore or None
        Store receiving the updates; None makes this a no-op.
    tasks : list of dict
        Task descriptors index-aligned with outcomes.
    outcomes : list
        Routed TaskOutcome objects (or raw executor outcome dicts).
    sources : tuple of two task lists, optional
        ``region_charge_sources`` of the full queue; derived from
        ``tasks`` when not given.
    state : int or callable, optional
        State whose row of a per-state ``computed_charges`` payload each
        task embeds with - an int, or a callable taking the task. The
        calculator passes its own embedding-state resolver, the same one
        the QM/MM path uses; omitted, every task reads row 0 (the
        single-state case).

    Returns
    -------
    int
        Number of tasks whose charges were written.
    """
    if charge_store is None:
        return 0
    high_tasks, mid_inter_tasks = (region_charge_sources(tasks)
                                   if sources is None else sources)

    def state_of(task):
        """The state whose charge row this task embeds with."""
        return _state_index(state, task)
    highlevel_by_center = {}
    for task in high_tasks:
        highlevel_by_center[task.get('center_index')] = set(
            int(i) for i in (task.get('region_indices') or []))
    # ONIOM2: the low full-system term is the outer-charge source. It is
    # the only non-provider term over 'real'; the three-layer queue has
    # inter terms and uses the M(I_c) path below instead.
    oniom2 = not mid_inter_tasks

    updated = 0
    for task, outcome in zip(tasks, outcomes):
        if not _outcome_success(outcome):
            continue
        is_high = bool(task.get('provides_electronic'))
        is_mid_inter = (task.get('layer_id') == 1
                        and task.get('region') == 'inter')
        if not (is_high or is_mid_inter):
            if oniom2 and task.get('region') == 'real':
                raw = _outcome_raw(outcome)
                values = _charge_row(raw, state_of(task))
                if values is None:
                    logger.info(
                        "Task '%s' produced no computed_charges; skipping "
                        "its outer-charge update.", task.get('task_id'))
                    continue
                high = set()
                for group in highlevel_by_center.values():
                    high.update(group)
                outer = [index for index in range(len(values))
                         if index not in high]
                if not outer:
                    continue
                charge_store.update(
                    outer, [values[index] for index in outer])
                updated += 1
            # L(real) in ONIOM3, -L(I_c), -M(M_c): no charge source.
            continue
        raw = _outcome_raw(outcome)
        values = _charge_row(raw, state_of(task))
        region_indices = [int(i) for i in (task.get('region_indices') or [])]
        task_id = task.get('task_id')
        if values is None:
            logger.info(
                "Task '%s' produced no computed_charges; skipping its "
                "charge update.", task_id)
            continue
        if len(values) != len(region_indices):
            logger.warning(
                "Task '%s' computed_charges length %d does not match its "
                "region size %d; skipping its charge update.",
                task_id, len(values), len(region_indices))
            continue
        if is_high:
            charge_store.update(region_indices, values)
            updated += 1
            continue
        # mid inter: keep only its midlevel part I_c, drop H_c.
        exclude = highlevel_by_center.get(task.get('center_index'), set())
        sel_indices = [gi for gi in region_indices if gi not in exclude]
        sel_values = [values[j] for j, gi in enumerate(region_indices)
                      if gi not in exclude]
        if not sel_indices:
            logger.info(
                "Task '%s' midlevel is empty; no charge update.", task_id)
            continue
        charge_store.update(sel_indices, sel_values)
        updated += 1
    return updated


def damp_charges(carried, fitted, damping):
    """One iteration's carried charges, mixed part-way toward the fresh fit.

    A self-consistent multi-center iteration solves each center in the field
    built from the other centers' charges, so the map from one round's charges
    to the next is a fixed point that need not be contractive: with two centers
    polarizing each other the undamped update can oscillate instead of settling.
    A weight below 1 mixes the fit into the charges the round was solved in
    (`q <- (1 - a) q + a q_fit`), which is the standard cure for that; 1.0 is
    the undamped update every run did before the key existed.

    What this is NOT: the convergence test. That one is taken on the raw fit
    against the previous raw fit, before this mixing, because a damped move is
    shorter than the fit by construction and a tolerance read on it would call
    every iteration converged at any damping.

    Parameters
    ----------
    carried : sequence of float
        The charges the finished round was solved in.
    fitted : sequence of float
        The charges that round's provider returned, same order.
    damping : float
        Mixing weight of the fresh fit, in (0, 1]; the schema owns the range.

    Returns
    -------
    list of float
        The charges the next round's field is built from.
    """
    weight = float(damping)
    if weight >= 1.0:
        return [float(value) for value in fitted]
    return [(1.0 - weight) * float(old) + weight * float(new)
            for old, new in zip(carried, fitted)]


def embedding_for_task(charge_store, task, coords, embedding_regions):
    """Build the electrostatic embedding field for one task.

    Only a high-level provider H(c) is embedded, and only by its OWN
    center's midlevel -- never by another center or by the low-level
    region. Every other task embeds nothing.

    Parameters
    ----------
    charge_store : ChargeStore or None
        Store holding the current per-atom charges.
    task : dict
        Task descriptor to embed.
    coords : numpy.ndarray
        Shape (natom, 3) current coordinates in Angstrom.
    embedding_regions : dict
        {center_index: [midlevel global indices]} from data.charges.

    Returns
    -------
    numpy.ndarray or None
        Shape (n_midlevel, 4) [q, x, y, z] embedding rows, or None
        when the task is not an embedded high-level provider (or its
        center has no midlevel).
    """
    if charge_store is None:
        return None
    if not task.get('provides_electronic'):
        return None
    indices = (embedding_regions or {}).get(task.get('center_index'))
    if not indices:
        logger.info(
            "Task '%s' has no midlevel embedding source; running "
            "without electrostatic embedding.", task.get('task_id'))
        return None
    return charge_store.embedding_charges(indices, coords)
