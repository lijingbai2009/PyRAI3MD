"""
======================================================================
Conventional metadynamics over a fixed hill list.

The bias is a sum of Gaussians deposited along collective variables:

    V(s) = sum_h height_h * exp(-1/2 sum_k ((s_k - c_hk) / sigma_hk)^2)

Two variables are written in: a named collective variable (`family`
'gaussian'), and the squared aligned RMSD of a group against a reference
geometry (`family` 'rmsd'), whose value and Jacobian come from
utilities/rmsd.py. The hill kernel above is the same in both cases, which is
the reason the second backend is a family rather than a new potential: only
`_variable` knows which scalar it is handed, and deposition, tokens and the
commit path do not care. For 'rmsd' the variable is D^2 rather than its square
root on purpose: the root has a singular derivative at zero deviation, which is
exactly the point a converged run sits at.

Evaluation is pure. Given a geometry, a plan and a hill list it returns energy
and gradient and changes nothing: no hill is added, no counter moves. Only a
committed, accepted step may add a hill, and adding one returns a new state.
That separation is what keeps a retried evaluation, a batch of trial geometries
and a worker process from depositing twice.

The force at a step comes from the history that existed before that step's
deposition. The design document notes the consequence for a Verlet loop: the
first half-step kick reads the previous frame's stored force, so a newly
deposited hill must not retroactively rewrite the force that describes the step
already taken. The next-step force is produced as a separate value rather than
by mutating the old one, and the helper here is `evaluate` with an explicitly
passed hill list, which makes the two histories impossible to confuse.

Deposition is keyed by (trajectory_id, accepted_step, item_id). A token that
has been spent cannot be spent again, so a restart or a retry replays the same
step without growing the history. No hill is deposited at step zero: the
initial geometry is a sample, not evidence, and starting with a hill there
biases the run against its own starting point before it has moved.

A periodic variable differs through atan2, so a hill centred at 170 degrees
biases -170 degrees as the 20 degrees away it is.

Units: CV values and widths in internal units (Bohr or radian, and Bohr squared
for an aligned RMSD), heights and energies in Hartree, gradients in Hartree per
Bohr.

The module must stay a leaf: numpy, the lab CV and RMSD layers and the lab
records.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from pyrai3md.data.constraint import HillRecord
from pyrai3md.utilities import cv, geometric, rmsd, units
from pyrai3md.workflow.constraints import bias, schedule


class MetadynamicsError(ValueError):
    """A history potential cannot be evaluated or deposited as declared."""


def collect_hills(state, action_id):
    """The hills belonging to one action, oldest first.

    Every hill carries the (trajectory_id, accepted_step, item_id) key it was
    deposited under, so the filter is exact and a hill can never be shared by
    two actions by accident.
    """
    return [hill for hill in getattr(state, 'hills', ())
            if hill.token is not None and hill.token[2] == str(action_id)]


def _hill_terms(hills, values, periodic, action_id, nvar):
    """Yield (hill, differences, sigma, weight) for one geometry.

    One kernel for the energy and for the gradient: the weight a hill contributes
    and the coordinate derivatives of that weight are the same exponential, and
    writing it twice is how a run ends up pushed by a force that is not the
    derivative of the energy it reports.
    """
    values = np.asarray(values, dtype=np.float64)
    for hill in hills:
        centre = np.asarray(hill.center, dtype=np.float64)
        if centre.size != nvar:
            raise MetadynamicsError(
                "a hill of metadynamics '%s' holds %d centres for %d "
                'variables' % (action_id, centre.size, nvar))
        sigma = np.asarray(hill.width, dtype=np.float64)
        differences = np.array([
            cv.periodic_difference(values[position], centre[position])
            if periodic[position] else values[position] - centre[position]
            for position in range(nvar)])
        scaled = differences / sigma
        weight = hill.height * float(np.exp(-0.5 * float(np.dot(scaled,
                                                                scaled))))
        yield hill, differences, sigma, weight


def bias_value(hills, values, periodic, action_id=''):
    """The accumulated bias `V(s)` at one point of the variable space.

    Well-tempered metadynamics scales the height of the hill it is about to
    deposit by `exp(-V(s) / k_B dT)`, so the deposition step needs this number
    and nothing else of the evaluation.
    """
    return float(sum(weight for _hill, _diff, _sigma, weight in _hill_terms(
        hills, values, periodic, action_id or 'history potential',
        np.asarray(values, dtype=np.float64).size)))


def evaluate(spec, plan, coords, hills, context, reference=None, masses=None):
    """Energy and gradient of one history potential at a geometry.

    Parameters
    ----------
    spec : ConstraintSpec
        The metadynamics action, with `kind == 'metadynamics'`.
    plan : ConstraintPlan
    coords : numpy.ndarray
        (natom, 3) coordinates in Bohr.
    hills : list
        HillRecord entries in force for this evaluation. Passing the list
        explicitly is the point: the pre-deposition and post-deposition forces
        of one step differ, and the caller decides which is which.
    context : EvaluationContext
    reference : numpy.ndarray, optional
        (k, 3) reference block of the aligned RMSD, in Bohr, in the atom order
        of `spec.atoms`. Required by family 'rmsd'; it comes from the bound
        state, because re-reading it from the live geometry would reset the
        variable the run is measured against at every step.
    masses : array-like, optional
        Per-atom masses in atomic units of mass, needed only by an RMSD written
        with 'mass' weights.

    Returns
    -------
    dict
        'kind', 'family', 'id', 'cv_names', 'cv_values', 'parameters', 'energy'
        and 'gradient' with shape (natom, 3) in Hartree per Bohr.
    """
    if spec.kind != 'metadynamics':
        raise MetadynamicsError(
            "evaluate() got a '%s' action, not a history potential"
            % spec.kind)
    parameters = schedule.resolve_parameters(spec, context, spec.parameters)
    names, values, jacobians, periodic = _variables(
        spec, plan, coords, parameters, reference=reference, masses=masses)

    natom = int(np.asarray(coords).shape[0])
    widths = np.asarray(parameters.get('widths', ()), dtype=np.float64)
    if widths.size != len(names):
        raise MetadynamicsError(
            "metadynamics '%s' has %d variables but %d widths"
            % (spec.id, len(names), widths.size))
    if np.any(widths <= 0.0):
        raise MetadynamicsError(
            "metadynamics '%s' widths must all be positive" % spec.id)

    gradient = np.zeros((natom, 3), dtype=np.float64)
    energy = 0.0
    for hill, differences, sigma, weight in _hill_terms(
            hills, values, periodic, spec.id, len(names)):
        if weight == 0.0:
            continue
        energy += weight
        # dV/ds_k = -V * (s_k - c_k) / sigma_k^2, then chain into the variable's
        # own coordinate Jacobian so the hill pushes atoms, not variables.
        for position in range(len(names)):
            derivative = -weight * differences[position] / (
                sigma[position] * sigma[position])
            jacobian, indices = jacobians[position]
            gradient += cv.scatter_jacobian(
                jacobian * derivative, indices, natom)
    return {'kind': 'metadynamics', 'family': spec.family, 'id': spec.id,
            'cv_names': names, 'cv_values': values.tolist(),
            'parameters': {'height': float(parameters.get('height', 0.0)),
                           'widths': widths.tolist(),
                           'pace': int(parameters.get('pace', 1)),
                           'hills': int(len(hills))},
            'energy': float(energy), 'gradient': gradient}


def hessian(spec, plan, coords, hills, context, reference=None,
            masses=None):
    """The exact `d2V/dR2` of one history potential at one geometry.

    A hill is `w exp(-1/2 |z|^2)` on the dimensionless scaled distances
    `z_k = (s_k - c_k) / sigma_k`. Because `z` is dimensionless, one
    differentiation brings down a single `1/sigma`, not `1/sigma^2`:

        dV/ds_j     = -V z_j / sigma_j
        d2V/ds_jdk =  V (z_j z_k / (sigma_j sigma_k) - delta_jk / sigma_j^2)

    (writing `1/sigma^2` in the first factor is the trap here: it agrees with the
    right expression at `z = 0`, which is exactly where a hill deposited at the
    current value sits, and diverges once the geometry moves - and it diverges by
    a `z`-dependent amount, measured elementwise between 0.46 and 8.7 on a
    0.567-Bohr width, which is why the mistake reads as "the multi-variable
    coupling is wrong" rather than as one stray power.)

    The coordinate Hessian is then the chain rule over the variables,
    `H = sum_jk V_jk J_j J_k^T + sum_j V_j Q_j`, with `J` the per-variable
    Jacobians the evaluation already builds and `Q` the variable's own second
    derivative from `utilities/geometric`. Hills arrive as an argument, as they
    do for the force: which history is in force at a step is the caller's
    decision.

    Returns `hessian` shaped (3N, 3N) in Hartree per Bohr squared. Raises
    MetadynamicsError for an RMSD history - its optimal rotation moves with the
    geometry, so its curvature is a coupled problem and is not approximated
    here - for a hill of the wrong shape, and for a variable with no derived
    `Q`.
    """
    if spec.kind != 'metadynamics':
        raise MetadynamicsError(
            "hessian() got a '%s' action, not a history potential" % spec.kind)
    parameters = schedule.resolve_parameters(spec, context, spec.parameters)
    names, values, jacobians, periodic = _variables(
        spec, plan, coords, parameters, reference=reference, masses=masses)
    array = np.asarray(coords, dtype=np.float64)
    natom = array.shape[0]
    if spec.family == 'rmsd':
        raise MetadynamicsError(
            "metadynamics '%s' is written on an aligned RMSD: the optimal "
            'rotation depends on the geometry, so its curvature is not the sum '
            'of these terms and is not approximated here' % spec.id)
    variables = []
    for position, name in enumerate(names):
        record = plan.cvs[name]
        flat = np.zeros(natom * 3, dtype=np.float64)
        jacobian, member = jacobians[position]
        atoms = [int(atom) for atom in member]
        for slot, atom in enumerate(atoms):
            flat[atom * 3:atom * 3 + 3] = jacobian[slot]
        second = geometric.cv_hessian_block(record.kind, array, atoms)
        if second is None:
            raise MetadynamicsError(
                "metadynamics '%s' has no second derivative for cv '%s' of "
                'kind %r, and none is assumed'
                % (spec.id, name, record.kind))
        variables.append((flat, second[0], atoms))
    block = np.zeros((natom * 3, natom * 3), dtype=np.float64)
    for hill in hills:
        centre = np.asarray(hill.center, dtype=np.float64)
        sigma = np.asarray(hill.width, dtype=np.float64)
        if centre.size != len(names) or sigma.size != len(names):
            raise MetadynamicsError(
                "a hill of metadynamics '%s' holds %d centres and %d widths "
                'for %d variables'
                % (spec.id, centre.size, sigma.size, len(names)))
        differences = np.array([
            cv.periodic_difference(values[position], centre[position])
            if periodic[position] else values[position] - centre[position]
            for position in range(len(names))])
        scaled = differences / sigma
        weight = hill.height * float(np.exp(-0.5 * float(np.dot(scaled,
                                                                scaled))))
        if weight == 0.0:
            continue
        inverse = 1.0 / sigma
        inverse2 = inverse * inverse
        first = -weight * scaled * inverse
        surface = weight * np.outer(scaled * inverse, scaled * inverse)
        surface[np.diag_indices_from(surface)] -= weight * inverse2
        for left, (flat_left, core_left, atoms_left) in enumerate(variables):
            if first[left] != 0.0:
                for index_a, atom_a in enumerate(atoms_left):
                    for index_b, atom_b in enumerate(atoms_left):
                        block[atom_a * 3:atom_a * 3 + 3,
                              atom_b * 3:atom_b * 3 + 3] += (
                                  first[left] * core_left[index_a, index_b])
            for right in range(len(names)):
                block += surface[left, right] * np.outer(
                    flat_left, variables[right][0])
    return {'kind': 'metadynamics', 'family': spec.family, 'id': spec.id,
            'cv_names': list(names), 'cv_values': np.asarray(values).tolist(),
            'size': natom * 3, 'hessian': block}


def _variables(spec, plan, coords, parameters, reference=None, masses=None,
               gradient=True):
    """The scalars this history potential is written in, with their Jacobians.

    Parameters
    ----------
    gradient : bool
        False asks for the values only, which is what a deposition needs: a
        hill is centred on the variable, and the variable has a value even at a
        geometry whose aligned gradient would be undefined.

    Returns
    -------
    tuple
        (names, values, jacobians, periodic). `jacobians` is one
        (compact_array, indices) pair per variable, in the shape
        `cv.scatter_jacobian` takes, or None per variable when `gradient` is
        False; `periodic` says which variables wrap, so the hill loop can
        difference them correctly without knowing families.
    """
    if spec.family == 'rmsd':
        return _rmsd_variables(spec, coords, reference, masses,
                               gradient=gradient)
    if spec.family != 'gaussian':
        raise MetadynamicsError(
            "metadynamics '%s' of family %r is not implemented"
            % (spec.id, spec.family))
    names = [str(name) for name in parameters.get('cv_names', ())]
    if not names:
        raise MetadynamicsError(
            "metadynamics '%s' binds no cv" % spec.id)
    for name in names:
        if name not in plan.cvs:
            raise MetadynamicsError(
                "metadynamics '%s' references unknown cv '%s'"
                % (spec.id, name))
    values = np.zeros(len(names))
    jacobians = [None] * len(names)
    for position, name in enumerate(names):
        record = plan.cvs[name]
        if gradient:
            value, jacobian = bias._dispatch(record, coords, grad=True)
            jacobians[position] = (jacobian, record.indices)
        else:
            value = bias.variable_value(record, coords)
        values[position] = value
    return names, values, jacobians, [bool(plan.cvs[name].periodic)
                                      for name in names]


def _rmsd_variables(spec, coords, reference, masses, gradient=True):
    """The squared aligned deviation of one group, as a hill variable."""
    indices = np.asarray(spec.atoms, dtype=np.int64).reshape(-1) \
        if spec.atoms is not None else np.zeros(0, dtype=np.int64)
    if indices.size < 2:
        raise MetadynamicsError(
            "metadynamics '%s' aligns a group of at least two atoms and names "
            "%d of them" % (spec.id, indices.size))
    if reference is None:
        raise MetadynamicsError(
            "metadynamics '%s' has no reference geometry: an aligned RMSD "
            "binds its reference once, at the start of the trajectory"
            % spec.id)
    block = np.asarray(reference, dtype=np.float64)
    if block.shape != (indices.size, 3):
        raise MetadynamicsError(
            "metadynamics '%s' holds a reference of shape %s for %d atoms"
            % (spec.id, block.shape, indices.size))
    weights = None
    if spec.parameters.get('weight', 'uniform') == 'mass':
        if masses is None:
            raise MetadynamicsError(
                "metadynamics '%s' weights the deviation by mass and no "
                "masses were given" % spec.id)
        weights = np.asarray(masses, dtype=np.float64).reshape(-1)[indices]
    try:
        result = rmsd.rmsd_squared(
            np.asarray(coords, dtype=np.float64)[indices], block,
            weights=weights, grad=gradient)
    except rmsd.RmsdError as error:
        raise MetadynamicsError(
            "metadynamics '%s' cannot align its group: %s" % (spec.id, error))
    if not gradient:
        return ['rmsd^2'], np.array([result]), [None], [False]
    value, jacobian = result
    return ['rmsd^2'], np.array([value]), [(jacobian, indices)], [False]


def deposition_token(state, spec, context):
    """The (trajectory_id, accepted_step, item_id) key of a candidate hill."""
    return (str(context.trajectory_id), int(context.step), str(spec.id))


def should_deposit(spec, plan, context, state):
    """Whether this accepted step earns a hill.

    Step zero never deposits, a step that already spent its token never
    deposits again, and the interval comes from the declaration.
    """
    parameters = getattr(spec, 'parameters', {}) or {}
    pace = int(parameters.get('pace', 0) or 0)
    if pace < 1:
        raise MetadynamicsError(
            "metadynamics '%s' needs a positive pace" % spec.id)
    token = deposition_token(state, spec, context)
    if state.spent(token):
        return False
    step = int(context.step)
    if step <= 0:
        return False
    return step % pace == 0


def hill_at(spec, plan, coords, context, state, reference=None, masses=None):
    """The hill to deposit now, from the variable values at this geometry.

    `state` is required: a hill without a deposition token cannot be made
    idempotent, and an anonymous hill would slip past both the replay test and
    the per-action filter in `collect_hills`.

    Returns
    -------
    HillRecord or None
        Centred on the current variable values, with the declared per-variable
        width and height and the step's deposition token. None is returned when
        a well-tempered bias has flattened to nothing, because there is no hill
        left to write.
    """
    parameters = dict(getattr(spec, 'parameters', {}) or {})
    if state is None:
        raise MetadynamicsError(
            "metadynamics '%s' cannot deposit without a state: the hill needs "
            "its deposition token" % spec.id)
    _names, centres, _jacobians, periodic = _variables(
        spec, plan, coords, parameters, reference=reference, masses=masses,
        gradient=False)
    widths = np.asarray(parameters.get('widths', ()), dtype=np.float64)
    if widths.size == 1 and centres.size > 1:
        widths = np.repeat(widths, centres.size)
    if widths.size != centres.size:
        raise MetadynamicsError(
            "metadynamics '%s' has %d variables but %d widths"
            % (spec.id, centres.size, widths.size))
    token = deposition_token(state, spec, context)
    height = float(parameters.get('height', 0.0))
    # Well-tempered deposition (Laio and Parrinello, PNAS 99:12562, 2002):
    # the hill just written carries `exp(-V(s)/k_B dT)` of its declared
    # height, so the bias flattens itself against the landscape it is
    # filling and converges to `1/gamma` times the free-energy surface.
    # `delta_temperature` is `(gamma - 1) * T` in Kelvin; 0 means the plain
    # well-to-well deposition this subsystem shipped with.
    delta = float(parameters.get('delta_temperature', 0.0))
    if delta > 0.0:
        accumulated = bias_value(collect_hills(state, spec.id), centres,
                                 periodic, spec.id)
        exponent = -accumulated / (units.KELVIN_TO_HARTREE * delta)
        height *= float(np.exp(min(0.0, exponent)))
        if height == 0.0:
            # The scale is a positive exponential, so it can underflow but never
            # go below zero. A hill of zero height is not a hill - `HillRecord`
            # refuses one, and rightly - so the deposition is reported as
            # nothing to write and the caller spends the step's token without
            # growing the history. This is the flattening well-tempered
            # metadynamics is built to produce, not a failure.
            return None
    try:
        return HillRecord(center=centres, width=widths, height=height,
                          token=token)
    except ValueError as error:
        raise MetadynamicsError(
            "metadynamics '%s' cannot deposit: %s" % (spec.id, error))
