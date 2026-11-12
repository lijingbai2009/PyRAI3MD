"""
======================================================================
This module provides the Optimizer base class for geometry optimization
algorithms. The optimizer determines the next geometry based on
optimization algorithms and convergence criteria, operating exclusively
on PyRAIState.

The optimizer never performs electronic structure calculations. It
reads energy and gradient through the PyRAIState property interface
(state.result), computes the next geometry, and updates the state
through state.update_system().

Coordinate convention:
    PyRAIState stores coordinates as (N, 3) in Angstrom.
    The optimizer internally converts to Bohr (atomic units)
    for numerical optimization, computes the displacement in
    Bohr, converts it back to Angstrom, and updates the state
    through state.update_system(). The workflow and calculator
    never observe flattened coordinates or Bohr values.

    Unit conversion factors (coord_to_bohr, disp_to_angstrom)
    are determined by the Optimize workflow from the skill
    manifest's unit declarations and passed at construction time.

Aug 07 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from abc import ABC, abstractmethod

import numpy as np

from ...kernel.logger import get_logger
from ...utilities.lattice import (CELL_COMPONENT_SLOTS, cart_to_frac,
                                  cell_volume, cell_force_to_stress,
                                  frac_to_cart, matrix_to_cell6,
                                  stress_to_cell_force)
from ...utilities.units import (ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM,
                                au_pressure_to_bar, bar_to_au_pressure)

# Which lattice row each relaxed component sits in: the slots (0, 3, 4, 6, 7, 8)
# of a row-vector 3x3 are rows 0, 1, 1, 2, 2, 2, and the row's own length is what
# divides a lattice step into a strain.
_SLOT_ROWS = (0, 1, 1, 2, 2, 2)

# How small a step's `s . y` may be before a BFGS update stops trusting it, as
# a fraction of |s| |y|. The floor is relative because a trust-region step is
# allowed to be small: measured on a box relaxation whose radius had shrunk to
# its floor, an absolute cut in the units of the problem skipped every remaining
# update, so the curvature model froze at whatever the first steps had taught it
# and the run spent its last fifty evaluations moving 1e-4 Bohr a time.
CURVATURE_FLOOR = 1.0e-12


def usable_curvature_product(s, y, floor=CURVATURE_FLOOR):
    """`s . y` when a quasi-Newton update may be built from this step.

    Returns None when it may not. A negative or zero product is a non-convex
    stretch: the update divides by it, and an inverse built from one is no
    longer positive definite, which is the property every descent direction here
    depends on. A product that is positive but negligible against the lengths of
    the two vectors is the same hazard in the other direction, and that is the
    only case the relative floor removes.
    """
    product = float(np.dot(s, y))
    if product <= 0.0:
        return None
    scale = float(np.linalg.norm(s)) * float(np.linalg.norm(y))
    if scale > 0.0 and product <= floor * scale:
        return None
    return product


class Optimizer(ABC):
    """Abstract base class for geometry optimizers.

    Manages optimization history, iteration tracking, and convergence
    checking. Subclasses implement the specific optimization algorithm
    through _compute_step().

    The optimizer communicates with the outside world exclusively
    through PyRAIState: it reads energy/gradient via state.result
    (property interface) and updates coordinates via
    state.update_system(). Direct access to state.pmd internal
    attributes is not part of the interface.

    Parameters
    ----------
    maxiter : int
        Maximum optimization iterations.
    energy_threshold : float
        Energy convergence threshold (absolute value of delta E).
    gradient_threshold : float
        Gradient RMS convergence threshold.
    gradient_max_threshold : float
        Gradient max-component convergence threshold.
    displacement_threshold : float
        Displacement RMS convergence threshold.
    displacement_max_threshold : float
        Displacement max-component convergence threshold.
    trust_radius : float
        Initial trust radius for step scaling (in Bohr).
    active_state : int
        0-based index of the electronic state to optimize. The
        energy and gradient are read from position active_state of
        the state-resolved result arrays (energy (ns,), gradient
        (ns, na, 3)). Default 0 (ground state).
    active_state_j : int
        0-based index of the second state of a two-state run. When it
        differs from active_state the report carries the gap between
        the two states' energies; when it is the same, no second state
        is requested and no gap row exists. Default 0.
    cell_variables : bool
        Whether the six movable lattice components are part of the optimized
        vector, driven by the stress conjugate to them. Default False,
        which leaves the optimizer exactly the atomic (3N,) search it
        was.
    coord_to_bohr : float
        Factor to convert state coordinates to Bohr.
        The framework constant `utilities.units.ANGSTROM_TO_BOHR`.
        (A hand-typed 1.8897259886 used to sit here, 7.2e-08 relative
        away from it.)
    disp_to_angstrom : float
        Factor to convert displacement from Bohr to Angstrom.
        The framework constant `utilities.units.BOHR_TO_ANGSTROM`.
        (A hand-typed 0.529177210903 used to sit here.)
    """

    # Whether asking for this algorithm by name gets a step. A registered but
    # unwritten optimizer is refused at validation, where the input is still
    # cheap to change, rather than at the first step of a run that has already
    # paid for an evaluation.
    is_implemented = True
    # Whether this search walks several structures rather than one. A path
    # optimizer is driven by the Path workflow (`workflow/path.py`), which owns
    # the calculator and hands it evaluations one at a time, so `run(state)` and
    # `_compute_step` are not its contract - and `Optimize` refuses a path method
    # by this flag rather than by falling into a step that cannot answer.
    is_path = False
    # A search that needs curvature before it can take its first step says so
    # here, and the workflow that owns the calculator goes and gets it.
    needs_initial_hessian = False
    # Whether the second state of `opt_state_i`/`opt_state_j` is read at all.
    # Only the crossing searches solve on a pair; every other search takes the
    # gradient of `active_state` and ignores the second index, so an input that
    # named two surfaces for one of them would pay for a second gradient on
    # every step and never look at it. The input layer refuses that instead.
    uses_state_pair = False

    def __init__(self, maxiter=100, energy_threshold=1e-6,
                 gradient_threshold=3e-4,
                 gradient_max_threshold=4.5e-4,
                 displacement_threshold=3e-4,
                 displacement_max_threshold=1.8e-3,
                 cell_stress_rms_threshold=40.0,
                 cell_stress_max_threshold=60.0,
                 cell_strain_rms_threshold=5.0e-5,
                 cell_strain_max_threshold=7.0e-5,
                 external_pressure=0.0,
                 cell_coupling='anisotropic',
                 trust_radius=0.3,
                 active_state=0,
                 active_state_j=0,
                 cell_variables=False,
                 coord_to_bohr=ANGSTROM_TO_BOHR,
                 disp_to_angstrom=BOHR_TO_ANGSTROM):
        self.maxiter = int(maxiter)
        self.energy_threshold = float(energy_threshold)
        self.gradient_threshold = float(gradient_threshold)
        self.gradient_max_threshold = float(gradient_max_threshold)
        self.displacement_threshold = float(displacement_threshold)
        self.displacement_max_threshold = float(
            displacement_max_threshold)
        # The cell half of a variable-cell run is judged on a pressure (bar) and
        # on a strain (the step of a lattice entry over the length of its own
        # row), not on the atomic thresholds above. A lattice component and a
        # nuclear coordinate are both lengths, but the same Hartree per Bohr on a
        # box twice as long asks four times the pressure it asked on the small
        # one, and a threshold that cannot see the difference is not a threshold
        # on the box (`utilities/lattice.py::cell_force_to_stress`).
        self.cell_stress_rms_threshold = float(cell_stress_rms_threshold)
        self.cell_stress_max_threshold = float(cell_stress_max_threshold)
        self.cell_strain_rms_threshold = float(cell_strain_rms_threshold)
        self.cell_strain_max_threshold = float(cell_strain_max_threshold)
        # `opt_pressure` (bar) turns the objective from the energy into the
        # enthalpy `E + p V`: the lattice derivative of the volume is `V B^-T`,
        # so the quantity the search steps on and the box is judged on is
        # `V (sigma^T + p I) B^-T`, which is `stress_to_cell_force` read on
        # `sigma + p I`. The box is at rest where `sigma = -p I`, the same sign
        # and the same conversion the NPT barostats use
        # (`integrator/ensemble.py`: `g_eps = V(alpha p_kin - sigma - p_ext I)`,
        # whose kinetic term is zero for a search that takes no velocities).
        # `opt_cell_coupling` says how many directions that pressure may move:
        # 'isotropic' scales the box, 'anisotropic' leaves the six relaxed
        # components to answer for themselves.
        self.external_pressure = float(external_pressure)
        self.pressure_au = bar_to_au_pressure(self.external_pressure)
        self.cell_coupling = str(cell_coupling).strip().lower()
        self.trust_radius = float(trust_radius)
        self.active_state = int(active_state)
        self.active_state_j = int(active_state_j)
        self.cell_variables = bool(cell_variables)
        self.coord_to_bohr = float(coord_to_bohr)
        self.disp_to_angstrom = float(disp_to_angstrom)

        # Internal state
        self.iteration = 0
        self.failure_count = 0
        self.history = {
            'energies': [],
            'gradients': [],
            'displacements': [],
            'coordinates': [],
        }
        self.last_displacement = None
        # The geometry the last recorded step was taken from: (Bohr coordinates,
        # Angstrom coordinates, lattice, crossing counts), so a rejection
        # re-steps from there instead of from the trial it refused, and hands
        # back the tally with the positions it belongs to.
        self.step_base = (None, None, None, None)
        # Components the constraint plan took away from the last trial step; zero when the
        # run declares nothing.
        self.last_constraint_projection = 0
        # How far the last cell step rode a component the run had frozen, before
        # the position solve put it back. Zero when the box is fixed or nothing is
        # frozen; see `_frozen_ride`.
        self.last_frozen_ride = 0.0
        # The tangential gradient, condition residuals and multipliers of the
        # evaluation the workflow is currently judging, handed over by
        # `Optimize.run` at the geometry the gradient belongs to. None is the
        # reading of an unconstrained run: the whole gradient is judged.
        self.constraint_split = None
        # The (natom, 3) freeze mask of the evaluation being judged, cached from
        # the same plan `apply_constraint` cuts with. None is "nothing frozen".
        # The step and the curvature update need it *before* the step is cut
        # down, which is why it is taken at the evaluation and not at the step.
        self._frozen_mask = None
        self.logger = get_logger('pyrai3md.optimizer')

    # ---- Public interface ----

    def run(self, state):
        """Compute the next geometry from the current state.

        Reads energy and gradient through state.result (property),
        converts coordinates from state units (Angstrom) to Bohr,
        reshapes (N,3) -> (3N,), delegates to the subclass
        _compute_step() for the algorithm-specific step in Bohr,
        applies trust radius scaling (in Bohr), converts the
        displacement back to Angstrom, reshapes (3N,) -> (N,3),
        and updates the state coordinates via state.update_system().

        A run that relaxes the box carries six more components through
        the same pipeline: the entries of the lattice it may move and the
        stress force conjugate to them join the vector the algorithm steps
        on, so the curvature the optimizer learns covers the shape of the
        cell as one object. The constraint projection stays on the atomic
        half, which is the only half the constraint subsystem has a rule for
        (a plan holding a condition while the box moves is the open A5/D-14
        item), and the six are the triangular entries because that is the
        spelling the engines read back (CELL_COMPONENT_SLOTS).

        Parameters
        ----------
        state : PyRAIState
            Current state with energy/gradient in state.result.
            Coordinates are in Angstrom.

        Returns
        -------
        PyRAIState
            State with updated coordinates (Angstrom), and an updated
            lattice when the run relaxes the cell.
        """
        result = state.result
        energy, gradient = self._objective(state, result)

        if energy is None or gradient is None:
            self.logger.warning(
                "Optimizer.run: energy or gradient not available, "
                "skipping geometry update.")
            return state

        # Read state coordinates (Angstrom) and convert to Bohr
        coords_2d = np.asarray(state.coordinates, dtype=np.float64)
        original_shape = coords_2d.shape
        coords_bohr = coords_2d.reshape(-1) * self.coord_to_bohr
        grad_1d = np.asarray(gradient, dtype=np.float64).reshape(-1)
        atomic_size = coords_bohr.size

        # Cell degrees of freedom, appended to the same vector
        coords_bohr, grad_1d, lattice = self._extend_to_cell(
            state, coords_bohr, grad_1d)

        # The vector the step is solved against, and the one recorded as this
        # evaluation's gradient: the curvature update is fitted to the difference
        # of the recorded vectors, so history and step must be the same quantity
        # or the update learns from a force the run cannot act on.
        step_gradient = self._step_gradient(grad_1d)

        # Update history with current data (Bohr coordinates)
        self._update_history(energy, step_gradient, coords_bohr)

        # Compute optimization step in Bohr (algorithm-specific)
        displacement = self._compute_step(step_gradient)

        # Take away what the run declared immovable first, then cap: the trust
        # radius bounds the distance the run can actually cross, so it is read
        # against the projected move. Capping the unprojected step and cutting
        # afterwards leaves a relaxation with half its atoms pinned taking a
        # quarter of the step it asked for, on the norm of the half it will
        # throw away.
        cell_step = None
        if lattice is not None:
            displacement, cell_step = (displacement[:atomic_size],
                                       displacement[atomic_size:])
        displacement = self.apply_constraint(state, displacement)
        if cell_step is not None:
            displacement = np.concatenate((displacement, cell_step))
        displacement = self._apply_trust_radius(displacement)
        # The cap was taken over the whole (3N + 6) move, the box entries
        # included: they are lengths in the same Bohr as the nuclear components,
        # so a step that left the box uncapped would walk the cell further than
        # the radius the run claims to be respecting. The vector recorded and
        # judged is this capped full move, so the split back into the two halves
        # happens only where the two are applied.
        if lattice is not None:
            cell_step = displacement[atomic_size:]
            direction = self._cell_direction(lattice)
            if direction is not None and cell_step.size:
                # An isotropic run cannot move the box any other way, and the
                # component it is refused is not a residual it has to answer for:
                # it never moved.
                cell_step = np.dot(cell_step, direction) * direction
                displacement = np.concatenate(
                    (displacement[:atomic_size], cell_step))

        # Store displacement for convergence reporting (Bohr)
        self.last_displacement = displacement.copy()
        if self.history['displacements']:
            self.history['displacements'][-1] = displacement.copy()
        else:
            self.history['displacements'].append(displacement.copy())

        # Write the step into the state: atomic half in Cartesian, box half as
        # a strain the atoms ride. The base geometry is kept so a rejection can
        # re-step from the same place rather than from the trial it refused.
        self.step_base = (coords_bohr[:atomic_size].copy(), coords_2d.copy(),
                          lattice,
                          getattr(state.pmd.system, 'image', None))
        self._apply_step(state, coords_bohr[:atomic_size], coords_2d,
                         lattice, displacement)
        self.iteration += 1

        return state

    def _apply_step(self, state, coords_bohr, coords_angstrom, lattice,
                    displacement, image=None):
        """Write one joint step into the state, atoms riding the box.

        The base geometry arrives from the caller rather than from the state,
        because the path that re-steps after a rejection runs while the state
        already holds the geometry that was refused. It arrives in both units
        for the same reason: the two conversion factors the workflow hands in
        are not exact reciprocals, so rebuilding a position from the other
        unit's copy multiplies it by 1 + 1e-11 on every step, and a pinned atom
        drifts 3.6e-6 Bohr over a 200-step saddle search. Only the increments
        are converted.

        The accepted path and the re-step after a rejection call this one
        method, because a run that rode the box when a step is taken and not
        when it is shortened would put its trial geometries on two different
        surfaces while claiming both describe the same quadratic model.

        Parameters
        ----------
        state : PyRAIState
            State whose coordinates and lattice are written.
        coords_bohr : numpy.ndarray
            (3N,) atomic coordinates before the step, in Bohr.
        coords_angstrom : numpy.ndarray
            The same geometry in Angstrom, as the state keeps it.
        lattice : numpy.ndarray or None
            (3, 3) lattice in Bohr before the step, or None for a fixed box.
        displacement : numpy.ndarray
            The step: (3N,) Bohr, or (3N + 6,) when the box is a variable.
        image : list of list of int, optional
            The crossing counts to write back with these coordinates. Given by
            the path that re-steps after a rejection, which returns to a
            geometry whose own tally the refused trial had already moved on
            from; left alone when None, which is the accepted step's case
            because its crossings are counted from the structure it starts at.
        """
        atomic_size = coords_bohr.size
        atomic_disp = displacement[:atomic_size]
        shape = np.asarray(coords_angstrom).shape
        cell_step = (displacement[atomic_size:] if lattice is not None
                     else None)
        new_coords_flat = (np.asarray(coords_angstrom, dtype=np.float64)
                           .reshape(-1) + atomic_disp * self.disp_to_angstrom)
        if cell_step is not None and cell_step.size:
            moved = (coords_bohr + atomic_disp).reshape(shape)
            ride = self._cell_ride(moved, lattice, cell_step) - moved
            new_coords_flat += ride.reshape(-1) * self.disp_to_angstrom
            self.last_frozen_ride = self._frozen_ride(state, ride)
        updates = {'coordinates': new_coords_flat.reshape(shape).tolist()}
        if cell_step is not None and cell_step.size:
            moved_lattice = self._cell_step(lattice, cell_step)
            updates['lattice_vector'] = moved_lattice.tolist()
            # The nine entries and the six cell parameters are two spellings of
            # one box. A state whose lattice moved while `cell` and `volume`
            # stayed at the pre-strain numbers hands a reader that names the
            # box through the six a different geometry than the one the step
            # made - the same rule `dispatcher.py` follows when it loads a
            # strained task.
            updates['cell'] = [float(value) for value in
                               matrix_to_cell6(moved_lattice)]
            updates['volume'] = cell_volume(moved_lattice)
        if image is not None:
            # The crossing counts travel with the coordinates they describe.
            # The workflow wraps every trial geometry and accrues the flags a
            # crossing raises, so a rejected step has to hand back the tally of
            # the geometry it returns to: leaving the accrual of the refused
            # trial in place tells the engine that an atom which never moved has
            # crossed a face, and LAMMPS answers that with "Bond atom missing in
            # image check" (src/domain.cpp) rather than a wrong number.
            updates['image'] = [list(row) for row in image]
        state.update_system(updates)

    def convergence(self, state):
        """Check convergence and produce an optimization report.

        Uses the current state.result for energy and gradient values
        together with the internally stored last displacement to
        evaluate convergence criteria.

        Parameters
        ----------
        state : PyRAIState
            Current state with the latest energy/gradient.

        Returns
        -------
        tuple(bool, dict)
            (converged, report) where report contains Iteration,
            Energy, delta E, Gradient RMS/Max, Displacement RMS/Max,
            and Trust Radius.
        """
        result = state.result
        energy, gradient = self._objective(state, result)

        if energy is None or gradient is None:
            return False, self._empty_report()

        grad_1d = np.asarray(gradient, dtype=np.float64).reshape(-1)

        # Gradient metrics. An optimization that declared no condition judges the
        # whole gradient, exactly as it always did. A constrained one judges the
        # tangential part the workflow handed over: on the manifold each
        # condition's own direction carries the force the condition supplies, so
        # a threshold read on the raw gradient there is unreachable by
        # construction, while the multiplier of that direction is a number the
        # run does report (D-26 closed this by deleting every atom a condition
        # touches, which threw away the tangential half with it).
        split = self.constraint_split
        if split is None:
            grad_rms = float(np.sqrt(np.mean(grad_1d ** 2)))
            grad_max = float(np.max(np.abs(grad_1d)))
        else:
            tangent = np.asarray(split['tangent'], dtype=np.float64)
            # The frozen components come back zeroed from the split, so the sum
            # is over the mobile ones while the denominator stays the mobile
            # count: a free degree of freedom that carries no force is what the
            # threshold is a threshold on.
            mobile = max(int(split['mobile_components']), 1)
            grad_rms = float(np.sqrt(np.sum(tangent ** 2) / mobile))
            grad_max = float(np.max(np.abs(tangent)))

        # Energy change
        if len(self.history['energies']) >= 2:
            delta_e = float(
                self.history['energies'][-1] - self.history['energies'][-2])
        else:
            delta_e = 0.0

        # Displacement metrics. The step a cell run took is one vector, but its
        # halves are not one reading: a box edge moving 0.001 Bohr and an atom
        # moving 0.001 Bohr are not the same event, so each half is counted over
        # its own components and printed on its own line.
        disp = (self.last_displacement
                if self.last_displacement is not None
                else np.zeros_like(grad_1d))
        atom_size = grad_1d.size
        atom_disp = disp[:atom_size]
        cell_disp = (disp[atom_size:] if disp.size > atom_size
                     else np.zeros(0))
        disp_rms = float(np.sqrt(np.sum(atom_disp ** 2) / self._rms_count(
            grad_1d.size, split)))
        disp_max = float(np.max(np.abs(atom_disp)))

        report = {
            'Iteration': self.iteration,
            'Energy': float(energy),
            'Delta E': delta_e,
            'Gradient RMS': grad_rms,
            'Gradient Max': grad_max,
            'Displacement RMS': disp_rms,
            'Displacement Max': disp_max,
            'Trust Radius': self.trust_radius,
        }
        # The cell segment of a variable-cell run: the conjugate of the lattice
        # is the stress times the volume, read at the geometry this gradient
        # belongs to (the lattice the step left behind, kept in history), and it
        # must clear the same gradient thresholds the atoms are judged against.
        cell_ok = True
        if self.cell_variables:
            slots = list(CELL_COMPONENT_SLOTS)
            # The cell readings belong to the geometry this gradient was
            # computed at, not to the box the step proposes. The step moved only
            # the triangular entries, so the entries it did not move are still
            # the state's, and the six it did are the ones history kept.
            flat = self._cell_lattice(state).reshape(-1).copy()
            flat[slots] = self.history['coordinates'][-1][atom_size:]
            lattice = flat.reshape(3, 3)
            cell_force = self._cell_gradient(state, lattice)
            cell_grad_rms = float(np.sqrt(np.mean(cell_force ** 2)))
            cell_grad_max = float(np.max(np.abs(cell_force)))
            cell_disp_rms = (float(np.sqrt(np.mean(cell_disp ** 2)))
                             if cell_disp.size else 0.0)
            cell_disp_max = (float(np.max(np.abs(cell_disp)))
                             if cell_disp.size else 0.0)
            # What those six forces are worth as a pressure, and what those six
            # displacements are worth as a strain: the pair the box is judged on.
            # The stress is rebuilt from the slots rather than read off the
            # engine, because a constrained run's total cell force carries the
            # condition's own lattice term, which no engine reports as stress.
            stress = cell_force_to_stress(cell_force, lattice)
            pressure = au_pressure_to_bar(
                stress[np.tril_indices(3)])
            cell_stress_rms = float(np.sqrt(np.mean(pressure ** 2)))
            cell_stress_max = float(np.max(np.abs(pressure)))
            strain = (np.abs(cell_disp)
                      / np.linalg.norm(lattice, axis=1)[list(_SLOT_ROWS)]
                      if cell_disp.size else np.zeros(0))
            cell_strain_rms = (float(np.sqrt(np.mean(strain ** 2)))
                               if strain.size else 0.0)
            cell_strain_max = (float(np.max(strain)) if strain.size else 0.0)
            report['External Pressure'] = self.external_pressure
            if self.cell_coupling == 'isotropic':
                # The two readings a coupled box answers for: the mean normal
                # stress and the one scale the step moved. A deviatoric stress or
                # a shear residual cannot be relieved by a shape-preserving step,
                # so judging them here would hold the run to a threshold it has no
                # direction to reach; they stay in the six-component rows above.
                normal = float(np.trace(
                    cell_force_to_stress(cell_force, lattice)) / 3.0)
                bar_normal = float(au_pressure_to_bar(normal))
                cell_stress_rms = abs(bar_normal)
                cell_stress_max = abs(bar_normal)
                direction = self._cell_direction(lattice)
                scale = (abs(float(np.dot(cell_disp, direction))
                             / float(np.dot(direction, direction)))
                         if direction is not None and cell_disp.size else 0.0)
                cell_strain_rms = cell_strain_max = scale
            report['Cell Gradient RMS'] = cell_grad_rms
            report['Cell Gradient Max'] = cell_grad_max
            report['Cell Displacement RMS'] = cell_disp_rms
            report['Cell Displacement Max'] = cell_disp_max
            report['Cell Stress RMS'] = cell_stress_rms
            report['Cell Stress Max'] = cell_stress_max
            report['Cell Strain RMS'] = cell_strain_rms
            report['Cell Strain Max'] = cell_strain_max
            report['Cell Volume'] = float(abs(np.linalg.det(lattice))
                                          * self.disp_to_angstrom ** 3)
            # The ride's work on a component the run froze, which the position
            # solve undoes. A run that reports only the net displacement of a
            # pinned atom reports zero and says nothing about the box having
            # carried it there and back.
            report['Frozen Ride Max'] = float(self.last_frozen_ride)
            cell_ok = (cell_stress_rms < self.cell_stress_rms_threshold
                       and cell_stress_max < self.cell_stress_max_threshold
                       and cell_strain_rms < self.cell_strain_rms_threshold
                       and cell_strain_max < self.cell_strain_max_threshold)
        gap = self._state_gap(result)
        if gap is not None:
            # A two-state run says what its two surfaces are worth at this
            # geometry; a crossing search is judged on that number narrowing,
            # and even a single-surface run that names a second state wants to
            # see the gap it is optimizing away from.
            report[self._gap_label()] = gap
        if split is not None:
            # The run log prints these rows; the progress table keeps its seven
            # columns, because the multipliers and residuals are not convergence
            # criteria and a column of them would read as one.
            report['Constraint'] = split

        # All five criteria must be met: energy change, gradient
        # RMS/max and displacement RMS/max (opt_grms/opt_gmax/
        # opt_drms/opt_dmax in the optimize input section). A cell run adds its
        # own four readings over the lattice half of the step, judged in bar and
        # in strain (opt_csrms/opt_csmax/opt_cerms/opt_cemax): a box still under
        # stress has not converged however quiet the atoms are.
        converged = (abs(delta_e) < self.energy_threshold
                     and grad_rms < self.gradient_threshold
                     and grad_max < self.gradient_max_threshold
                     and disp_rms < self.displacement_threshold
                     and disp_max < self.displacement_max_threshold
                     and cell_ok)
        if not self._extra_criteria(report):
            # A saddle search is not finished at a quiet gradient with two
            # downhill directions, and a crossing is not finished with the two
            # surfaces still apart.
            converged = False

        return converged, report

    # ---- Method-specific log and status readings ----

    def status_method_columns(self):
        """Numeric columns this optimizer appends to `opt_status.txt`.

        The columns sit after the cell block and before any composite term
        columns, so the fixed prefix every reader classifies stays intact.
        A method that adds a reading must return the same tuple for every
        step of one run: the writer declares the header once.
        """
        return ('frozen_ride',) if self.cell_variables else ()

    def status_method_values(self, report):
        """Values matching `status_method_columns()`, in the same order."""
        if not self.cell_variables:
            return ()
        value = report.get('Frozen Ride Max')
        return (float('nan') if value is None else float(value),)

    def method_details(self, report):
        """Ordered log-only readings beyond the convergence report.

        Strings are allowed here (they never enter `opt_status.txt`); a
        missing numeric reading is returned as None so the logger prints
        `Not measured` instead of a zero that would read as a measurement.
        """
        return {}

    def apply_constraint(self, state, displacement):
        """Remove from a trial step the directions the run is not allowed to move.

        An optimizer proposes a displacement, and a frozen atom has no
        displaceable direction: the trust radius and the line search do not know
        about the declaration, so the projection happens here, after the step is
        scaled and before it is recorded. Recording the *projected* step is what
        makes the convergence report describe the step actually taken; a run that
        moved 1e-3 Bohr on a frozen atom while reporting zero displacement would
        be describing a step it did not take.

        A bond constraint is not projected out of the step: its positions are
        restored by `constraints/runtime.constrain_state` once the trial geometry
        exists, because the constraint is a curved manifold and a linear
        projection of a displacement is not enough to land on it.

        Harmonic restraints, collective variables and a metadynamics bias need
        no handling in this method at all: they are forces, and the Calculator
        finalizer already adds their energy and gradient to every evaluation
        (dev/methods/CONSTRAINT_DEVELOPMENT.md section 7).

        Parameters
        ----------
        state : PyRAIState
            The state whose plan decides which components are frozen.
        displacement : numpy.ndarray
            (3 * natom,) trial step in Bohr, in the state's atom order.

        Returns
        -------
        numpy.ndarray
            The step with the frozen components zeroed. The same object when the
            run declares no constraints, so an unconstrained optimization is
            byte-identical to what it was before this hook existed.
        """
        vector = np.asarray(displacement, dtype=np.float64)
        plan = state.constraint_plan()
        if plan.is_empty:
            return vector
        from pyrai3md.workflow.constraints import projection

        natom = vector.size // 3
        mask = projection.build_fix_mask(plan, natom)
        if not mask.any():
            return vector
        step = vector.reshape(natom, 3).copy()
        step[mask] = 0.0
        self.last_constraint_projection = int(mask.sum())
        return step.reshape(-1)

    def _cell_lattice(self, state):
        """The run's lattice in Bohr, or None when the box is not a variable.

        A run that asked the box to relax has to be holding a box: a missing or
        singular lattice is refused here rather than optimized around, because a
        cell that keeps its starting shape while the log counts cell steps is a
        claim the run cannot support.
        """
        if not self.cell_variables:
            return None
        rows = getattr(state.pmd.system, 'lattice_vector', None) or []
        array = np.asarray(rows, dtype=np.float64).reshape(-1)
        if array.size != 9 or not np.any(array):
            raise ValueError(
                "optimize_cell relaxes the box, but this system carries no "
                "lattice in system.lattice_vector; there is no box to move")
        lattice = array.reshape(3, 3) * self.coord_to_bohr
        if abs(float(np.linalg.det(lattice))) < 1.0e-12:
            raise ValueError(
                "optimize_cell refuses a singular lattice: the box has no "
                "volume for the stress to be conjugate against")
        return lattice

    def _cell_stress(self, state):
        """The stress tensor the cell force is the conjugate of, or None.

        One surface, one tensor: the whole-system reading of the state this run
        is on. A search whose objective is built out of two surfaces overrides
        this, because its energy is not the active state's and a box stepped on
        one surface while the atoms step on another would put the two halves of
        one vector on different objectives.
        """
        return state.active_stress()

    def _cell_gradient(self, state, lattice_bohr):
        """The derivative of the energy with respect to the six components.

        The framework's stress is the tensile-positive whole-system tensor in
        Hartree per Bohr^3, the reading `P = -Tr/3` is printed from and the one
        the NPT barostats take, and with the row-vector lattice a strain acts on
        as `H -> (I + eps) H`. Solving `dE = V Tr(sigma eps)` for the lattice
        derivative gives

            dE/dH = V sigma^T H^-T

        which for a cubic box of edge h reduces to the checkable `V sigma / h`:
        stretching one axis by dh changes the volume by h^2 dh, so the force on
        that entry is the pressure conjugate times h^2, not times h^3. Using the
        volume alone is off by a factor of the box length, and the box length
        here is 15 Bohr. A missing tensor is refused, not read as a physical
        zero, exactly as the barostat refuses it (`integrator/ensemble.py`).

        A run that declares a strict condition adds the term the engine cannot
        know about. The stress it reports is the potential's, so it does not
        carry the force a held length supplies against the box; the Lagrangian
        the constrained search is actually descending does, and its multiplier is
        the one `constraints/projection.tangent_split` already solved for when it
        split this evaluation's gradient. The subtraction is the sign the split
        uses - `tangent = g - sum_c lambda_c J_c`, so the constraint reaction is
        `-lambda_c J_c` - and it is settled by measurement rather than by this
        sentence (`qualification/verification/test_constraint_cell_coupling.py`
        differences a constrained box against the engine's own energy).
        """
        stress = self._cell_stress(state)
        if stress is None:
            raise ValueError(
                "optimize_cell needs a whole-system stress tensor and this "
                "evaluation returned none. The planning layer derives the "
                "'stress' capability for this run (planning/resolver.py); a "
                "backend without it (only xtb, cp2k and lammps declare one) "
                "cannot relax a box")
        array = np.asarray(stress, dtype=np.float64)
        if array.shape != (3, 3):
            raise ValueError(
                'a whole-system stress must be (3, 3), got %s'
                % (array.shape,))
        if self.pressure_au:
            # The enthalpy's own half of the lattice derivative. It is added to
            # the stress before the conversion, so a run under pressure reports
            # and judges the residual that still has to vanish, not the raw
            # engine stress: at the target the two cancel.
            array = array + self.pressure_au * np.eye(3)
        conjugate = stress_to_cell_force(array, lattice_bohr)
        split = self.constraint_split
        if split is not None and split.get('cell_virial') is not None:
            conjugate = conjugate - np.asarray(split['cell_virial'],
                                               dtype=np.float64)
        slots = conjugate.reshape(-1)[list(CELL_COMPONENT_SLOTS)].copy()
        gap = self._cell_gap_conjugate(state, lattice_bohr)
        if gap is not None:
            # A crossing objective is the mean of two surfaces plus the gap's own
            # square, so the lattice half carries the gap's lattice derivative as
            # well. A single-surface search has nothing to add.
            slots = slots + np.asarray(gap, dtype=np.float64)
        return slots

    def _cell_gap_conjugate(self, state, lattice_bohr):
        """The gap's own lattice derivative, for a two-surface objective.

        None everywhere but a crossing search. A crossing whose backend answers
        with one whole-system tensor keeps returning None: the difference of one
        tensor with itself is zero, and a zero here would be a silence read as a
        number.
        """
        return None

    def _cell_direction(self, lattice):
        """The one direction an isotropic box step may take, in slot space.

        An isotropic relaxation scales the box: one number multiplies every row
        by its own length, so in the six relaxed components the allowed
        direction is the three row lengths standing in their own slots. The step
        is projected onto it here, which is the one place a joint step is
        written, so what the run records, applies and judges is the same move.

        Returns
        -------
        numpy.ndarray or None
        The unit direction, or None for 'anisotropic', where all six relaxed
        components are already free and nothing is projected.
        """
        if self.cell_coupling != 'isotropic':
            return None
        rows = np.linalg.norm(lattice, axis=1)
        direction = np.zeros(len(CELL_COMPONENT_SLOTS), dtype=np.float64)
        for index, slot in enumerate(CELL_COMPONENT_SLOTS):
            if slot // 3 == slot % 3:
                direction[index] = rows[slot // 3]
        norm = float(np.linalg.norm(direction))
        return direction / norm if norm else None

    def _cell_step(self, lattice_bohr, components):
        """The lattice a six-component cell step moves to, still triangular.

        The lattice and the step are both in Bohr here, and the sum is handed
        back in the framework's Angstrom spelling: `lattice_vector` is a state
        field, not an optimizer internal, and writing a Bohr number into an
        Angstrom field inflates the box by the Bohr-to-Angstrom factor on every
        step.
        """
        delta = np.zeros(9, dtype=np.float64)
        delta[list(CELL_COMPONENT_SLOTS)] = components
        return ((lattice_bohr + delta.reshape(3, 3))
                * self.disp_to_angstrom)

    def _cell_ride(self, coords_bohr, lattice_bohr, components):
        """The atomic positions a cell step carries along with the box.

        A cell step changes the box, and an atom whose fractional coordinate is
        unchanged moves in Cartesian space by the same relative amount. That is
        the motion the stress is conjugate against: measured on the equilibrated
        60-molecule LAMMPS water box at a relative strain of 1e-3, holding the
        fractional coordinates and differencing the engine's own energy lands on
        V sigma^T H^-T to within 0.2-7% on all six components, while holding the
        Cartesian ones and differencing misses by factors between -0.47 and +36.9
        and flips the sign on three of them. The same box carries a nuclear
        gradient of 0.22 Hartree per Bohr, which is what makes the two paths
        differ by so much: the difference between them is exactly the gradient's
        contraction with the fractional coordinates, so it vanishes only once the
        atoms are relaxed.

        This is also what the integrator already does. The cell-aware drift
        scales coordinates and lattice together through exp(v_eps dt)
        (`workflow/integrator/verlet.py`), so a variable-cell relaxation that
        pinned the atoms in Cartesian space would relax the box against a
        different definition of the cell force than the one the same framework
        propagates under NPT.
        """
        delta = np.zeros(9, dtype=np.float64)
        delta[list(CELL_COMPONENT_SLOTS)] = components
        new_lattice = lattice_bohr + delta.reshape(3, 3)
        fractional = cart_to_frac(coords_bohr, lattice_bohr)
        return frac_to_cart(fractional, new_lattice)

    def _frozen_ride(self, state, ride):
        """How far the box carried a component the run declared immovable.

        A cell step rides every atom, because holding the fractional coordinates
        is what makes the stress the conjugate force of the step. A frozen atom
        rides too, and the position solve then puts it back on its anchor, which
        is stored in absolute Cartesian and does not move with the box (#276 /
        D-25 on the dynamics side). The net is what a run reports as "the pinned
        atom did not move", and the net is not the whole story: this is the work
        the ride did and the solve undid. Reporting only the net would let a ride
        that carries a pinned atom a whole cell read as a constraint that held.
        """
        plan = state.constraint_plan()
        if plan.is_empty:
            return 0.0
        from pyrai3md.workflow.constraints import projection

        mask = projection.build_fix_mask(plan, ride.shape[0])
        if not mask.any():
            return 0.0
        array = np.asarray(ride, dtype=np.float64)
        return float(np.max(np.abs(array[mask])))

    def _extend_to_cell(self, state, coords_bohr, force):
        """Put the lattice and its conjugate force beside the atomic half."""
        lattice = self._cell_lattice(state)
        if lattice is None:
            return coords_bohr, force, None
        components = lattice.reshape(-1)[list(CELL_COMPONENT_SLOTS)]
        return (np.concatenate((coords_bohr, components)),
                np.concatenate((force, self._cell_gradient(state, lattice))),
                lattice)

    def note_evaluation(self, state):
        """Take the constraint reading of the evaluation this step judges.

        The split is read at the geometry the energy and gradient belong to,
        which is why the workflow calls it before the atoms have moved: after a
        step the state describes a structure no force was computed for, and a
        residual measured there would be a measurement of the optimizer's own
        move rather than of the constraint.

        Parameters
        ----------
        state : PyRAIState
            State carrying the latest energy and gradient.

        Returns
        -------
        dict or None
            The split record from `constraints.runtime.constraint_split`, or
            None when the run declares no condition, which is what keeps an
            unconstrained optimization judged on its whole gradient.
        """
        self.constraint_split = None
        self._frozen_mask = None
        result = state.result
        _energy, gradient = self._objective(state, result)
        if gradient is None:
            return None
        if state.constraint_plan().is_empty:
            return None
        from ..constraints import runtime as constraint_runtime
        from ..constraints import projection

        self.constraint_split = constraint_runtime.constraint_split(
            state, np.asarray(gradient, dtype=np.float64))
        natom = int(np.asarray(state.coordinates, dtype=np.float64)
                    .reshape(-1, 3).shape[0])
        self._frozen_mask = projection.build_fix_mask(state.constraint_plan(),
                                                     natom)
        return self.constraint_split

    def _tangent_of(self, vector):
        """Replace the atomic head of a vector by its tangential part.

        One definition for both the 3N step of an ordinary run and the 3N+6
        vector of a cell run, whose lattice tail belongs to no constraint the
        plan can absorb and is passed through untouched. With nothing declared
        the vector comes back as it went in, which is what keeps an
        unconstrained optimization bit-for-bit what it was.
        """
        split = self.constraint_split
        if split is None:
            return vector
        tangent = np.asarray(split['tangent'], dtype=np.float64)
        vector = np.asarray(vector, dtype=np.float64)
        if tangent.size == vector.size:
            return tangent
        out = vector.copy()
        out[:tangent.size] = tangent
        return out

    def _active_components(self):
        """Boolean (3N,) mask of the components the plan leaves free."""
        if self._frozen_mask is None:
            return None
        return ~np.asarray(self._frozen_mask, dtype=bool).reshape(-1)

    def record_final_evaluation(self, state):
        """Put one evaluation into the history without taking a step.

        A constrained run computes its last structure once more, after the loop
        has stopped moving it, so the energy and gradient the run reports belong
        to the structure it prints. Reporting them beside the previous pair's
        numbers would describe a geometry the run does not have, and the
        displacement is deliberately left as the step that produced this
        geometry: nothing moved to reach it.

        Parameters
        ----------
        state : PyRAIState
            State carrying the final evaluation.
        """
        result = state.result
        energy, gradient = self._objective(state, result)
        if energy is None or gradient is None:
            return
        flat = np.asarray(gradient, dtype=np.float64).reshape(-1)
        coordinates = (np.asarray(state.coordinates, dtype=np.float64)
                       .reshape(-1) * self.coord_to_bohr)
        coordinates, flat, _lattice = self._extend_to_cell(
            state, coordinates, flat)
        self._update_history(energy, self._step_gradient(flat), coordinates)

    # ---- Protected helpers for subclasses ----

    def _extract_pair(self, state):
        """The two surfaces a crossing search reads.

        Returns
        -------
        tuple
            (energy_i, energy_j, gradient_i, gradient_j) with the gradients
            flattened in Bohr-space components, and the signed gap
            `E_j - E_i`. A single-surface evaluation cannot drive a crossing
            search: there is no second surface to bring down to the first, so
            this refuses by name instead of reading one surface twice.
        """
        result = state.result
        if not result:
            raise ValueError(
                '%s needs an evaluation with two states and none was made'
                % type(self).__name__)
        energies = np.asarray(result.get('energy'),
                              dtype=np.float64).reshape(-1)
        gradient = result.get('gradient')
        if gradient is None:
            raise ValueError(
                "%s needs state-resolved gradients and this evaluation "
                "carries none" % type(self).__name__)
        array = np.asarray(gradient, dtype=np.float64)
        if array.ndim != 3:
            raise ValueError(
                "%s needs a state-resolved gradient of shape (nstate, natom, "
                "3); a single-surface gradient of %s shape cannot be brought "
                "to a crossing. Ask the backend for the pair the input named "
                "with opt_state_i and opt_state_j"
                % (type(self).__name__, 'two' if array.ndim == 2 else array.ndim))
        first, second = self.active_state, self.active_state_j
        if first == second:
            raise ValueError(
                "%s is a crossing search: opt_state_i and opt_state_j must "
                "name two different surfaces, and both name %d"
                % (type(self).__name__, first))
        if max(first, second) >= array.shape[0]:
            raise ValueError(
                "%s reads state %d but this evaluation returned %d states; the "
                "run asked for more surfaces than the backend produced"
                % (type(self).__name__, max(first, second) + 1,
                   array.shape[0]))
        return (float(energies[first]), float(energies[second]),
                array[first].reshape(-1), array[second].reshape(-1))

    def _derivative_coupling(self, state):
        """The derivative coupling of the requested pair, or None.

        A conical intersection's branching plane is spanned by the gap
        gradient and the derivative coupling; the second of the two comes from
        the backend as one row of `nac`, addressed through the pair map the
        system carries, exactly the way the surface-hopping integrator
        addresses it (`integrator/fssh.py`). A run whose pair carries no
        coupling is not refused here: the optimizer that needs the plane asks,
        and a spin-forbidden crossing has none to give by physics.
        """
        result = state.result
        nac = result.get('nac') if result else None
        if nac is None:
            return None
        pairs = getattr(state.pmd.system, 'nac_pair_map', None) or []
        index = {}
        for position, pair in enumerate(pairs):
            try:
                left, right = int(pair[0]), int(pair[1])
            except (TypeError, ValueError, IndexError):
                continue
            index[(left, right)] = position
        first, second = self.active_state, self.active_state_j
        rows = np.asarray(nac, dtype=np.float64)
        if (first, second) in index:
            position, sign = index[(first, second)], 1.0
        elif (second, first) in index:
            position, sign = index[(second, first)], -1.0
        else:
            return None
        if position >= rows.shape[0]:
            return None
        return sign * rows[position].reshape(-1)

    def _enthalpy(self, state, energy):
        """Turn an energy into the objective the cell half is stepped on.

        The lattice derivative of a run under `opt_pressure` is
        `V (sigma + p I)^T H^-T` (`_cell_gradient`, :727-732), which is the
        derivative of `E + pV`, not of `E`. So the number the step is judged
        against has to carry the same `pV`, or the trust-region ratio compares a
        measured change of one function with a predicted change of another. That
        is not a refinement. On the analytic instrument of
        `verification/test_optimizer_crossing.py` - one cubic lattice, a stress
        that is the derivative of its own energy, 10 kbar applied - the search
        with this term lands on the enthalpy minimum in four steps with 0.4 bar of
        residual and the radius at its ceiling; with the term taken back out it
        spends all forty steps at the radius floor of 1e-4 Bohr, 0.0145 Bohr off
        that minimum, and 8795 bar of residual left on the box. The same reading
        applies to round `tmp_20261018_082139`'s last thirteen steps, where `Delta
        E` printed as an exact 0.00000000 while the box still moved 0.004 A^3 a
        step: at 10 kbar that is `p dV` of 9.2e-7 Hartree a step, the only thing
        changing on a flat energy, and the ratio was judging it as nothing.
        What the term does NOT explain is the shipped pressure case at 1 kbar on
        the 10 A ethylene cube. Rounds `tmp_20261018_141444` (without this term)
        and `tmp_20261018_173454` (with it) read the same on that box apart from
        the threshold pair they were compared against - endpoint residual
        987.45141829 bar, volume 1000.000000 -> 1000.074059 A^3 - because the one
        accepted lattice step there moved p*dV by 1.692e-5 Hartree against a dE of
        -7.929e-4, and the run's two largest energy changes came with dV exactly
        zero: atomic moves, which no cell term judges. That leg's carrier, not
        this objective, is what #389 changed.

        A run with no applied pressure, or one whose box is not a variable, gets
        its energy back unchanged, so this touches no other search.
        """
        if energy is None or not self.pressure_au:
            return energy
        lattice = self._cell_lattice(state)
        if lattice is None:
            return energy
        return float(energy) + self.pressure_au * float(cell_volume(lattice))

    def _objective(self, state, result):
        """The energy and gradient the search is actually stepping on.

        One surface's evaluation, for every search that optimizes one surface.
        A crossing search overrides this and returns the average energy with
        the gradient of its own objective, so the history, the quasi-Newton
        update, the step rejection and the criteria all read the same pair and
        none of them has to know which search it is inside. The pair is the
        objective in both cases: under an applied pressure that is the enthalpy
        (`_enthalpy`), because the gradient's lattice half already is.
        """
        energy, gradient = self._extract_energy_gradient(result)
        return self._enthalpy(state, energy), gradient

    def _extra_criteria(self, report):
        """A criterion this search adds to the five, or True.

        Rows the search wants printed go into `report`; the run log prints what
        it is given. Returning False says the extra criterion is not met, and
        the workflow keeps stepping.
        """
        return True

    def _extract_energy_gradient(self, result):
        """Extract the active-state energy and gradient from a result.

        Results store state-resolved arrays: energy as (ns,) and
        gradient as (ns, na, 3). The active_state index (0-based,
        consistent with the input state_i) selects the optimized
        state; no index shifting is applied anywhere. A plain
        (na, 3) gradient without a state axis is used as-is for
        backward compatibility.

        Parameters
        ----------
        result : Result or None
            Latest calculation result.

        Returns
        -------
        tuple(float or None, np.ndarray or None)
            (energy, gradient) of the active state. Either entry is
            None when unavailable.
        """
        if not result:
            return None, None

        energy = result.get('energy')
        gradient = result.get('gradient')

        if energy is None or gradient is None:
            return None, None

        # A composite evaluation assembles ONE surface - the one the per-center
        # configuration asks for - and records that configuration in the same
        # result (`multiscale/composite.py`). Its single energy is therefore the
        # requested state already, whatever index that state carries inside each
        # center's ladder; indexing the array by the scalar would address a
        # state the assembled surface never carried, which is what a multi-center
        # excited-state minimization hit (index 1 of a length-1 array).
        energies = np.asarray(energy, dtype=np.float64).reshape(-1)
        assembled = (energies.size == 1
                     and result.get('active_configuration') is not None)

        # Select the active-state energy from the (ns,) array
        energy = float(energies[0] if assembled else energies[self.active_state])

        # Select the active-state gradient from the (ns, na, 3) array; an
        # assembled surface carries one state axis entry for the same reason.
        gradient = np.asarray(gradient, dtype=np.float64)
        if gradient.ndim == 3:
            gradient = gradient[0 if (assembled and gradient.shape[0] == 1)
                                else self.active_state]

        return energy, gradient

    def _gap_label(self):
        """The report key naming the state pair a gap is read over.

        The states are numbered the way the input numbers them, one-based, so
        the row can be matched against the `state_i` and `state_j` it came from.
        """
        return 'Gap (S%d-S%d)' % (self.active_state_j + 1,
                                  self.active_state + 1)

    def _state_gap(self, result):
        """The energy difference between the two states a run asked for.

        None when the run names one state, or when the evaluation returned
        fewer states than the pair needs: a gap between a state that exists and
        one that was never computed is not a number to print.
        """
        if self.active_state_j == self.active_state:
            return None
        energies = result.get('energy') if result else None
        if energies is None:
            return None
        array = np.asarray(energies, dtype=np.float64).reshape(-1)
        if max(self.active_state, self.active_state_j) >= array.size:
            return None
        return float(array[self.active_state_j] - array[self.active_state])

    def _update_history(self, energy, gradient_1d, coordinates_1d):
        """Record the current step data into the optimization history.

        Parameters
        ----------
        energy : float
            Current energy.
        gradient_1d : np.ndarray
            Flattened gradient (3N,).
        coordinates_1d : np.ndarray
            Flattened coordinates (3N,).
        """
        self.history['energies'].append(float(energy))
        self.history['gradients'].append(gradient_1d.copy())
        self.history['coordinates'].append(coordinates_1d.copy())
        # Placeholder for displacement; overwritten after step computation
        self.history['displacements'].append(
            np.zeros_like(gradient_1d))

    def _apply_trust_radius(self, displacement):
        """Scale the displacement vector to the trust radius.

        Parameters
        ----------
        displacement : np.ndarray
            Raw optimization step (3N,).

        Returns
        -------
        np.ndarray
            Scaled displacement within the trust radius.
        """
        step_norm = float(np.linalg.norm(displacement))
        if step_norm > self.trust_radius and step_norm > 0:
            displacement = displacement * (self.trust_radius / step_norm)
        return displacement

    def _empty_report(self):
        """Produce a report with zero/None values.

        Returns
        -------
        dict
            Empty optimization report.
        """
        return {
            'Iteration': self.iteration,
            'Energy': None,
            'Delta E': 0.0,
            'Gradient RMS': 0.0,
            'Gradient Max': 0.0,
            'Displacement RMS': 0.0,
            'Displacement Max': 0.0,
            'Trust Radius': self.trust_radius,
        }

    def _rms_count(self, total_components, split):
        """How many components a displacement RMS is taken over.

        A frozen component cannot move, so including it in the mean dilutes the
        reading toward zero by exactly the fraction of the molecule the input
        pinned: a run with half its atoms frozen would pass a displacement
        threshold at twice the displacement of an unconstrained one. The gradient
        side already divides by `mobile_components`; this is the same rule for
        the step, and with no constraint declared it is the same number the
        mean used before.
        """
        if split is None:
            return max(int(total_components), 1)
        return max(int(split.get('mobile_components', total_components)), 1)

    def _step_gradient(self, gradient):
        """The vector the step is solved against.

        Every search that relaxes a minimum has read the raw gradient and cut
        the frozen components out of the step afterwards, and for a Newton or
        BFGS step that order is harmless: the step is a descent direction on the
        components it keeps. A curvature-following step is not - it is an
        eigenvector of the model, so a zero row left in the model by a frozen
        coordinate is the *lowest* curvature the search can find, and the search
        climbs the direction the input pinned. A search that follows curvature
        therefore solves against the same tangential part of the gradient the
        convergence criteria are already judged on, and overrides this hook.
        """
        return gradient

    @abstractmethod
    def _compute_step(self, gradient):
        """Compute the optimization step (algorithm-specific).

        Subclasses implement the numerical optimization algorithm
        here. The method receives the current flattened gradient
        and returns the displacement vector.

        Parameters
        ----------
        gradient : np.ndarray
            Flattened gradient (3N,).

        Returns
        -------
        np.ndarray
            Displacement vector (3N,).
        """

    def __repr__(self):
        return "%s(iteration=%d, trust_radius=%.4f)" % (
            type(self).__name__, self.iteration, self.trust_radius)
