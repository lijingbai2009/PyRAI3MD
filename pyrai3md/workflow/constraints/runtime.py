"""
======================================================================
The framework call sites the constraint subsystem lives behind.

Everything in this directory above it is a pure numeric core: arrays in,
records out. This module is the only place that knows a `PyRAIState` exists,
and it exposes exactly the three hooks the framework has reserved:

- `finalize` runs in the Calculator between the stress conversion and the
  result write-back, and adds every declared additive potential to the
  evaluation the step just produced (design document section 7).
- `constrain_state` is the body of `apply_constrain` in the integrator: the
  freeze restore, the SHAKE position solve, and the velocity handling that
  follows the record's `velocity_constraint` keyword.

- `constraint_split` is what the geometry optimizer judges a constrained step
  on: the gradient with every declared condition's normal component removed,
  beside the value, residual and multiplier of each condition.

All three are gated on the same condition, `plan.is_empty`, and an empty plan
makes all three return before they touch a number: a run with no
`&system_constraint` block pays nothing for this code existing, which is the
default-closed contract the whole subsystem was built under.

Additive potentials are applied here and geometric projection is applied there,
and the split is not cosmetic. A projection changes the geometry, so the energy
and the gradient must describe the projected structure; that is why the
integrator projects before the calculator is asked for a force, and why this
module never rearranges coordinates.

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from pyrai3md.data.constraint import ConstraintState
from pyrai3md.data.geometry import record_wrapped_geometry
from pyrai3md.utilities.lattice import cart_to_frac
from pyrai3md.workflow.constraints import projection
from pyrai3md.workflow.constraints import shake
from pyrai3md.workflow.constraints import controller as constraint_controller
from pyrai3md.workflow.constraints import images
from pyrai3md.workflow.constraints import state_adapter

# The evaluation purposes the framework's own job types map onto. An unknown
# job type evaluates as a single point, which is the reading that adds the
# least: schedules and history depositions key on the accepted step, and a
# purpose that is not recognized earns neither.
PURPOSE_BY_JOBTYPE = {
    'md': 'md',
    'nvt': 'md',
    'npt': 'md',
    'namd': 'md',
    'optimize': 'optimization_trial',
    'opt_ts': 'optimization_trial',
    'hessian': 'hessian',
    'frequency': 'hessian',
}


def purpose_for(state):
    """Why this evaluation is happening, from the job type the run declares."""
    control = getattr(getattr(state, 'pmd', None), 'control', None)
    jobtype = str(getattr(control, 'jobtype', '') or '').lower()
    return PURPOSE_BY_JOBTYPE.get(jobtype, 'single')


def evaluation(state, purpose=None):
    """The triple one step needs: controller, view, context.

    Returns None when the run declares no constraint action, which is the
    caller's signal to do nothing at all. The controller is rebuilt from the
    cached plan, because it holds no mutable state of its own; the geometry
    crossing this boundary is the framework's Angstrom spelling, converted to
    Bohr exactly once, inside the adapter.
    """
    plan = state.constraint_plan()
    if plan.is_empty:
        return None
    controller = constraint_controller.SystemConstraintController(plan)
    view, context = state_adapter.read_state(
        state, purpose=purpose or purpose_for(state))
    state_adapter.require_usable(view, plan)
    if state.constraint_state is None:
        state.constraint_state = controller.bind_reference(
            state_adapter.solve_geometry(view), masses=view.masses,
            cell=view.cell, context=context)
    return controller, view, context


def bind(state):
    """Capture the targets and anchors that mean "the initial geometry".

    The workflow calls this once, before its first step. Lazy binding at the
    first *use* is not enough, because in the MD loop the first use is
    `apply_constrain`, which runs after the integrator has already moved the
    atoms: a bond constraint bound there would hold the length the first step
    happened to land on, and the run would be measuring its own drift as if it
    were the reference. A restarted run never calls this - the stored state
    arrives through `restore`, which is what keeps a restart from silently
    retargeting.
    """
    plan = state.constraint_plan()
    if plan.is_empty or state.constraint_state is not None:
        return state
    controller = constraint_controller.SystemConstraintController(plan)
    view, context = state_adapter.read_state(state, purpose=purpose_for(state))
    state_adapter.require_usable(view, plan)
    state.constraint_state = controller.bind_reference(
        state_adapter.solve_geometry(view), masses=view.masses,
        cell=view.cell, context=context)
    return state


def finalize(state, data, logger=None):
    """Add every declared constraint potential to one evaluation result.

    `data` is the assembled result mapping the Calculator is about to write
    back: `energy` (a scalar, or one value per electronic state) and `gradient`
    (`(natom, 3)` for the active combination a multi-center run integrates with,
    `(ns, natom, 3)` for a state-resolved result, or absent). The correction is
    state independent, so it shifts every state by the same amount and leaves
    the gaps untouched, and a gradient that was never computed stays absent
    rather than gaining a force no wavefunction produced.

    The flat `energy_contributions` breakdown is written here too, which is the
    row the print path already knows how to render; no second channel for a
    per-item energy is opened.
    """
    bundle = evaluation(state)
    if bundle is None:
        return data
    controller, view, context = bundle
    base = {
        'energy': data.get('energy'),
        'gradient': data.get('gradient'),
    }
    potential = controller.evaluate_potentials(
        state_adapter.solve_geometry(view), state.constraint_state, context,
        masses=view.masses)
    totals = controller.combine(base, potential, None, context)
    data['energy'] = _plain(totals.energy_total)
    if totals.gradient_total is not None:
        data['gradient'] = _plain(totals.gradient_total)
    contributions = totals.flat_contributions()
    # The half the engine returned, kept beside the sum it became. Without it the
    # printed decomposition would have to be rebuilt from the rows, and a reader
    # could not tell a measured origin pot from an arithmetic on the total
    # (D-18). A run whose plan holds only geometric holds contributes nothing,
    # and it still gets the row: its decomposition is the origin and itself,
    # which is the answer to "did the constraint change the energy". An empty
    # plan never reaches this line, so an unconstrained run gains no new key.
    data['energy_origin'] = _plain(base['energy'])
    if contributions:
        existing = data.get('energy_contributions')
        if isinstance(existing, dict):
            existing.update(contributions)
        else:
            data['energy_contributions'] = contributions
    if logger is not None and hasattr(logger, 'debug'):
        logger.debug('constraint finalizer applied %d item(s) at step %d'
                     % (len(contributions), context.step))
    return data


def commit(state):
    """Accept the current step into the constraint history.

    Only an accepted step may deposit a hill, so this belongs to the caller that
    decides a step is good, not to the evaluation. A run whose plan holds no
    history action returns without copying anything.
    """
    plan = state.constraint_plan()
    if plan.is_empty or state.constraint_state is None:
        return
    if not any(spec.kind == 'metadynamics' for spec in plan.actions):
        return
    controller = constraint_controller.SystemConstraintController(plan)
    view, context = state_adapter.read_state(state, purpose=purpose_for(state))
    state.constraint_state = controller.commit(
        state_adapter.solve_geometry(view), state.constraint_state, context,
        masses=view.masses)


def constrain_state(state):
    """Project the geometry and the velocities onto what the plan declares.

    The order is the one the design document fixes: frozen components return to
    their anchor, the bond constraints are solved against what is left, and the
    velocities lose the frozen components - plus, for a record that asks for it,
    the bond tangent projection. Coordinates go back in the framework's Angstrom
    spelling, converted at this one boundary.
    """
    bundle = evaluation(state)
    if bundle is None:
        return state
    controller, view, context = bundle
    recorded = state.constraint_state
    # A solve that does not converge raises `ShakeNotConverged` out of
    # `project_positions`, so there is no converged flag left to inspect here:
    # by the time control returns, the geometry satisfies what was declared.
    projected, info = controller.project_positions(
        state_adapter.solve_geometry(view), recorded, context,
        masses=view.masses)
    # The solve already measured what it did. Keeping the reading is what lets a
    # run say "this bond is held" with a number behind it instead of a promise,
    # and `ConstraintState.diagnostics` is where a per-evaluation reading lives
    # (`bind_reference` already files the reference cell there).
    recorded.diagnostics['last_projection'] = {
        key: value for key, value in info.items() if key != 'shift'}
    # Solve on the unfolded geometry, then fold once: writing a wrapped
    # coordinate without the crossing counts it implies hands the next engine
    # call a pair that contradicts itself, which is the failure mode the
    # framework's own image bookkeeping exists to avoid. The fold reports the
    # crossings of *this* fold, counted from the stored wrapped frame, so the
    # commit goes through the one writer that adds them to the ledger the run
    # has accrued instead of replacing it (`data/geometry.py`).
    expansion = getattr(view, 'expansion', None)
    also = None
    if view.velocities is not None:
        # Velocities are Bohr per atomic time on both sides of this boundary -
        # the adapter reads them without a conversion, so nothing converts them
        # back either. They belong to the geometry being committed, so they
        # travel in the same write.
        also = {'velocities': controller.project_velocities(
            view.velocities, recorded, context, masses=view.masses,
            coords=projected)}
    if expansion is None or not expansion.periodic:
        record_wrapped_geometry(state, state_adapter.to_angstrom(projected),
                                also=also)
    else:
        wrapped, crossings = images.fold_back(expansion, projected)
        record_wrapped_geometry(state, state_adapter.to_angstrom(wrapped),
                                crossings, also=also)
    return state


def constraint_split(state, gradient):
    """Split one evaluation's gradient along what the plan declares.

    A constrained optimization is judged on the part of its gradient that the
    declared conditions cannot absorb: on the manifold, each condition's own
    direction carries the force the condition supplies, and a threshold read
    there is a number the run can never reach. The reading belongs to the
    geometry the gradient was taken at, which is why the caller hands the
    gradient over instead of letting this function read a result that has since
    moved; an optimization that declares no condition returns None and is judged
    on its whole gradient, exactly as it always was.

    Returns
    -------
    dict or None
        `tangent`: the (3 * natom,) gradient with the frozen components zeroed
        and every condition normal component removed, the quantity the
        convergence criteria are evaluated on. `conditions`: one row per declared
        condition carrying its id, its atoms as the input numbered them, its
        target, its value at this geometry, the residual between them, and the
        Lagrange multiplier the condition absorbs (Hartree per Bohr for a
        length, Hartree per radian for an angle or a torsion). `rank`: how many
        independent conditions the set really holds. `frozen_components`,
        `mobile_components` and `projection`: the bookkeeping of this step's
        position solve, including the residual it reported. `cell_virial`, on a
        periodic run that declares a condition: the (3, 3) lattice derivative the
        conditions supply, which is the term a variable-cell run's cell force is
        missing because the engine's stress is the potential's and knows nothing
        about a length something else is holding. Absent when the box is fixed or
        the plan declares no condition, so a run that cannot use the term is not
        handed one.
    """
    bundle = evaluation(state)
    if bundle is None:
        return None
    controller, view, _context = bundle
    plan = state.constraint_plan()
    vector = np.asarray(gradient, dtype=np.float64).reshape(-1)
    if vector.size != 3 * view.natom:
        raise ValueError(
            'gradient holds %d components for %d atoms'
            % (vector.size, view.natom))
    geometry = state_adapter.solve_geometry(view)
    mask = projection.build_fix_mask(plan, view.natom)
    records = controller.condition_records(state.constraint_state)
    tangent, multipliers, rank = projection.tangent_split(
        vector, geometry, mask=mask,
        conditions=[(item['kind'], item['atoms']) for item in records] or None)
    rows = []
    for position, record in enumerate(records):
        value = float(shake.condition_value(
            geometry, record['kind'], record['atoms']))
        rows.append({
            'id': record['id'],
            'kind': record['kind'],
            'atoms': [int(atom) + 1 for atom in record['atoms']],
            'target': float(record['target']),
            'value': value,
            'residual': value - float(record['target']),
            'multiplier': float(multipliers[position]),
        })
    frozen = int(mask.sum())
    split = {
        'tangent': tangent,
        'conditions': rows,
        'rank': int(rank),
        'frozen_components': frozen,
        'mobile_components': int(3 * view.natom - frozen),
        'projection': dict((state.constraint_state.diagnostics
                            or {}).get('last_projection') or {}),
    }
    virial = _cell_virial(view, geometry, mask, records, multipliers)
    if virial is not None:
        # The cell force a constrained variable-cell run is missing: the engine's
        # stress is the potential's, and knows nothing about a length the plan is
        # holding. Only a periodic run with a condition gets the key, so a fixed
        # box reads no term it cannot use.
        split['cell_virial'] = virial
    return split


def _cell_virial(view, geometry, mask, records, multipliers):
    """The conditions' cell force at this geometry, or None when there is none.

    The fractional coordinates are taken on the same box and the same geometry the
    condition rows were built from, because the two describe one another: a row is
    a derivative with respect to a position, and the position a variable-cell step
    moves is the fractional one held while the lattice changes
    (`optimizer/base.py::_cell_ride`).
    """
    cell = getattr(view, 'cell', None)
    if cell is None or not getattr(view, 'periodic', False):
        return None
    if multipliers is None or not len(multipliers):
        return None
    conditions = [(item['kind'], item['atoms']) for item in records] or None
    if conditions is None:
        return None
    box = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    fractional = cart_to_frac(np.asarray(geometry, dtype=np.float64), box)
    return projection.cell_virial(geometry, fractional, multipliers, mask=mask,
                                  conditions=conditions).tolist()


def _plain(value):
    """Result-side spelling: an ndarray becomes a list, a scalar stays a float."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is None:
        return None
    return float(value) if np.ndim(value) == 0 else value
