"""
======================================================================
This module provides the Hessian workflow: the force-constant matrix of
a structure, by the route the input asked for, and the vibrational and
thermodynamic reading of it.

One workflow, two orthogonal decisions.

    who produces it      hess_driver=main   this file displaces and differs
                                              the gradients it is given
                         hess_driver=skill  the backend computes the matrix and
                                              hands it over in one evaluation
    how it is produced   hess_mode=analytic   second derivative of the engine
                         hess_mode=numerical  finite difference of gradients
                         hess_mode=auto       the engine's if it has one

The two are independent because a backend may produce its matrix either way:
ORCA's `num_freq` switches NumFreq over AnFreq, and Molcas runs MCKINLEY inside
its own input. From here both are one evaluation that returns a matrix, which is
why the framework never sees the difference.

`jobtype` spells the pairing this workflow resolves when the input names
neither key: `hessian`/`hess` ask the backend first and differenced second,
`num_hess`/`num_freq` name the framework's own batch, while `frequency` and
`num_freq` additionally ask for the vibrational and thermodynamic tail.
`&hessian hess_mode` overrides the spelling on the method axis - `auto` takes
the analytical matrix when the evaluation returned one and the numerical route
when it did not - and `&hessian hess_driver` overrides it on the producer axis.
A route switch is written into the log, because silently relaxing a different
Hessian than the one the run asked for is the failure mode here.

This is not `control.driver`. That keyword says who drives the run, and one of
its values (`skill`) keeps the run out of this workflow altogether
(`resolve_workflow` hands it to `SingleShot`); this keyword says who produces
one quantity inside a run the framework drives.

The numerical route is a batch: 2 displacements per Cartesian coordinate of the
mobile set, one energy+gradient child job each, and the central difference of
the returned gradients. The analysis half is `workflow/frequency.py`, shared
with nothing else: the same functions read an analytical Hessian that arrived
from a backend and a numerical one assembled here.

The displaced geometry calculations form an independent batch: the batch
algorithms (generate_batch/collect/postprocess) are owned by this module, while
the fan-out is forwarded to the inherited self.dispatch() pipeline (unit
materialization, Launcher and Scheduler composition).

Oct 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import warnings

import numpy as np

from .base import Workflow
from .frequency import thermo_analysis, vibration, write_freq_molden
from ..utilities.selection import resolve_indices
from ..utilities.units import ANGSTROM_TO_BOHR

# The batch child entry is named by module and function: a bound method would
# carry the workflow -- and with it the parent Calculator and its live backends
# -- into any payload that has to be handed to another process.
_EXECUTOR_MODULE = 'pyrai3md.workflow.hessian'

# The state axis a numerical Hessian is differentiated on. A multi-state run
# gets the surface `current_state` names; per-surface force constants are not
# produced by this workflow, and the result is written as one state so no
# reader can mistake the S0 matrix for a state-resolved block.
_DEFAULT_SOURCE = 'auto'

# Who produces the matrix, in the two names `control.driver` already uses for
# the same distinction (the framework, or the skill). Spelled here rather than
# imported from `schema.config::HESS_DRIVERS` because the workflow layer reads
# the section through `state.hessian_params`, the same way it reads the mode.
_MAIN_DRIVER = 'main'
_SKILL_DRIVER = 'skill'
_HESS_DRIVERS = (_MAIN_DRIVER, _SKILL_DRIVER)


def _child_entry(child_state, runtime_idx=None):
    """Execute one displaced geometry through the unified entry point.

    Parameters
    ----------
    child_state : PyRAIState
        Materialized child state of one displacement.
    runtime_idx : str or None
        Runtime identity suffix of the point's directory.

    Returns
    -------
    object
        The outcome of the child calculation.
    """
    from . import execute
    return execute(child_state, runtime_idx=runtime_idx)


class Hessian(Workflow):
    """Force constants, vibrational modes and thermochemical corrections.

    Parameters
    ----------
    calculator : Calculator or None
        Calculator owning the state and the displaced evaluations.
    step_size : float
        Displacement of each Cartesian coordinate, in Angstrom. It is a
        distance, and the gradients are Hartree per Bohr, so the difference
        quotient divides by `2 * step * ANGSTROM_TO_BOHR`: the factor is what
        turns the result into Hartree per Bohr^2 and leaving it out scales
        every frequency by the square root of 1.89.
    source : str
        'analytic' (refuse without a backend matrix), 'numerical' (displace and
        difference) or 'auto' (backend matrix when there is one, displacement
        when there is not). The `hessian`/`frequency` tokens default to 'auto'
        and the `num_*` tokens to 'numerical'; `&hessian hess_mode` overrides
        both.
    driver : str
        Who produces the matrix: 'main' displaces the geometry through the
        framework's own batch, 'skill' asks the backend for the matrix it
        computes. `&hessian hess_driver` names it; left unnamed the mode
        decides, because an analytical matrix can only come from a backend and a
        numerically differentiated gradient can be had from the framework. The
        two are the same question the input asks with `control.driver`, asked of
        one quantity.
    analysis : bool
        Whether the matrix is read into frequencies and normal modes.
    thermo : bool
        Whether the vibrational analysis is carried on into the
        thermodynamic corrections. Implied by `analysis` for the
        `frequency`/`num_freq` jobtypes.
    save_molden : bool
        Whether to write `<title>.freq.molden`, the name the initial-condition
        sampler reads.
    atoms : list or None
        The mobile selection of a partial Hessian, in the 1-based spelling the
        input uses. None is every atom.

    Attributes
    ----------
    matrix : numpy.ndarray or None
        The (3N, 3N) force constants of the last `run`, on the plain shape
        rather than the state-axis shape the result carries.
    """

    name = 'hessian'
    workflow_kind = 'batch'
    # Each displaced geometry needs an energy and a gradient; the gradient is
    # what is differenced, and the energy rides along because that is the
    # combined form the skill backends recognize.
    child_jobtype = 'energy gradient'

    def __init__(self, calculator=None, step_size=0.005, source=None,
                 driver=None,
                 analysis=None, thermo=None, save_molden=None,
                 atoms=None, temperature=None, pressure=None,
                 freq_scale=None, freq_cutoff=None, damping_alpha=None,
                 symmetry=None, linear=None, **kwargs):
        super().__init__(name='hessian', calculator=calculator, maxiter=1,
                         **kwargs)
        state = self.calculator.get_state() if self.calculator else None
        params = _hessian_keywords(state)
        self.step_size = float(params.get('hess_step', step_size) or 0.005)
        # Precedence: an explicit constructor source/driver, then the
        # route the input resolves to (explicit &hessian values, then
        # the jobtype token). One shared resolver keeps planning's
        # multi-backend gate and this workflow from disagreeing about
        # who owns the matrix.
        from ..schema.config import resolve_hessian_route
        tokens = (str(getattr(state, 'jobtype', '') or '').split()
                  if state is not None else [])
        resolved_mode, resolved_driver = resolve_hessian_route(tokens, params)
        if source is not None:
            self.source = str(source).strip().lower()
            if driver is not None:
                self.driver = str(driver).strip().lower()
            elif params.get('hess_driver') is not None:
                self.driver = str(params.get('hess_driver')).strip().lower()
            else:
                self.driver = (_MAIN_DRIVER if self.source == 'numerical'
                               else _SKILL_DRIVER)
        else:
            self.source = str(resolved_mode or _DEFAULT_SOURCE).strip().lower()
            self.driver = str(driver if driver is not None
                              else resolved_driver).strip().lower()
        if self.driver not in _HESS_DRIVERS:
            raise ValueError(
                "unknown hess_driver %r; expected %s"
                % (self.driver, ', '.join(_HESS_DRIVERS)))
        if self.driver == _MAIN_DRIVER and self.source != 'numerical':
            raise ValueError(
                "hess_driver 'main' with hess_mode %r asks the framework for an "
                "analytical force constant, and it has none: it differentiates "
                "the gradients it is already given" % self.source)
        self.analysis = _flag(params, 'hess_vibration', analysis,
                              default=_wants_analysis(state))
        self.thermo = _flag(params, 'hess_thermo', thermo,
                            default=self.analysis)
        self.save_molden = _flag(params, 'hess_molden', save_molden,
                                 default=False)
        self.temperature = float(params.get('hess_temperature', temperature)
                                 or 298.15)
        self.pressure = float(params.get('hess_pressure', pressure) or 1.0)
        self.freq_scale = float(params.get('hess_scale', freq_scale) or 1.0)
        self.freq_cutoff = float(params.get('hess_cutoff', freq_cutoff)
                                 if params.get('hess_cutoff', freq_cutoff)
                                 is not None else 100.0)
        self.damping_alpha = float(params.get('hess_alpha', damping_alpha)
                                   or 4.0)
        self.symmetry = int(params.get('hess_symmetry', symmetry) or 1)
        self.declared_linear = linear if linear is not None else params.get(
            'hess_linear', None)
        self.atoms = atoms if atoms is not None else params.get('hess_atoms', None)
        self.mobile = None
        # The (3N, 3N) matrix of the last `run`, in Hartree per Bohr^2. The
        # result carries the state axis (`postprocess` writes it as one state so
        # no reader can mistake the S0 matrix for a state-resolved block), so a
        # caller that wants the matrix in memory reads it here rather than
        # re-deriving the shape from the result.
        self.matrix = None
        # The (atom, axis) pairs this batch displaces: the mobile atoms minus the
        # components a `fix` freezes. None until `run` has a state to read both
        # from.
        self.rows = None
        self._batch_rows = []
        self._collected = None
        self._reference_coordinates = None
        self._missing_point = None
        # A run that relaxes its box has six more degrees of freedom than its
        # atoms, and the curvature a mode-following search steps on has to
        # describe all of them, so the batch strains the lattice beside
        # displacing the atoms and the matrix it assembles is (3N + 6)^2. The
        # lattice entries are differentiated through the stress rather than
        # through the energy: at the strain this step size implies, the energy
        # difference is second order in it while the cell force is first, and a
        # difference of two second-order numbers is a reading of the engine's
        # print rather than of the surface.
        self.cell_variables = bool(
            (getattr(state, 'optimize_params', None) or {}).get(
                'optimize_cell', False)) if state is not None else False
        self.hybrid_cell_block = bool(
            self.cell_variables and self.driver != _MAIN_DRIVER)
        if self.cell_variables and self.analysis:
            raise ValueError(
                "optimize_cell with a vibrational analysis is not wired: the "
                "analysis turns a force-constant matrix into frequencies through "
                "a mass-weighted basis of the atomic components, and the six "
                "lattice entries have no mass to be weighted by. A box's "
                "curvature is read by the search that asked for it, not as a "
                "frequency")
        self._cell_slots = []
        self._reference_lattice = None

    # ---- Batch structure ----

    def generate_batch(self, state):
        """Generate the +/- displacement tasks of the active components.

        A partial Hessian displaces and reads back only the atoms the input
        selected; the atoms outside it are held, which is the same statement
        the run makes when it freezes them, and the matrix rows of a held atom
        come back as the zeros that a finite difference of a motionless atom
        must give. A `fix` that freezes one component of an atom leaves the other
        two free, so the selection is carried here per component while the input
        still names it per atom (`_active_rows`).

        Parameters
        ----------
        state : PyRAIState
            Current state with the reference geometry.

        Returns
        -------
        list of dict
            One task per displacement sign and Cartesian component, with the
            keys `coordinates`, `atom`, `axis`, `sign`.
        """
        coordinates = np.asarray(state.coordinates, dtype=np.float64)
        self._reference_coordinates = coordinates.tolist()
        rows = self.rows if self.rows is not None else [
            (atom, axis) for atom in (
                self.mobile if self.mobile is not None
                else range(coordinates.shape[0])) for axis in range(3)]
        self._batch_rows = list(rows)
        batch = []
        for atom, axis in rows:
            for sign in (1, -1):
                displaced = coordinates.copy()
                displaced[atom, axis] += sign * self.step_size
                batch.append({
                    'coordinates': displaced.tolist(),
                    'atom': int(atom),
                    'axis': int(axis),
                    'sign': int(sign),
                })
        self._cell_slots = []
        self._reference_lattice = None
        if self.cell_variables:
            batch.extend(self._cell_batch(state, coordinates))
        return batch

    def _cell_batch(self, state, coordinates):
        """The +/- strain tasks of the six lattice entries the run relaxes.

        Each task carries the strained lattice *and* the coordinates riding it,
        because a strain that left the atoms where they were would differentiate
        along a path whose conjugate is not the stress
        (`utilities/lattice.py::stress_to_cell_force`). The entries strained are
        the six the optimizer moves - the row-vector lattice on and below the
        diagonal - which is the spelling the engines read back.
        """
        from ..utilities.lattice import cart_to_frac, frac_to_cart
        from ..utilities.lattice import CELL_COMPONENT_SLOTS

        rows = np.asarray(
            getattr(state.pmd.system, 'lattice_vector', None) or [],
            dtype=np.float64)
        if rows.shape != (3, 3) or not np.any(rows):
            raise ValueError(
                "optimize_cell asks the curvature batch to strain a box this "
                "system does not carry: system.lattice_vector is missing or "
                "zero")
        self._reference_lattice = rows.tolist()
        fractional = cart_to_frac(coordinates, rows)
        tasks = []
        for slot in CELL_COMPONENT_SLOTS:
            self._cell_slots.append(int(slot))
            row, column = slot // 3, slot % 3
            # The step is a length in the lattice's own unit, so a batch that
            # displaces atoms by 0.005 Angstrom strains the box by the same
            # absolute amount; the relative strain that implies is the box's own
            # business and is what the report of a run on a small box says.
            delta = np.zeros((3, 3))
            delta[row, column] = self.step_size
            for sign in (1, -1):
                strained = rows + sign * delta
                tasks.append({
                    'coordinates': frac_to_cart(fractional,
                                                strained).tolist(),
                    'lattice': strained.tolist(),
                    'cell_slot': int(slot),
                    'sign': int(sign),
                })
        return tasks

    def collect(self, state, batch_results):
        """Assemble the force constants from the displaced gradients.

        `H[ia, jb] = (g_jb(+h) - g_jb(-h)) / (2 h)` with h in Bohr, because the
        gradients are Hartree per Bohr; the matrix is then symmetrized, which a
        pair of finite differences is not to within the rounding of the engine's
        own print.

        Parameters
        ----------
        state : PyRAIState
            Current state.
        batch_results : list
            One Result per displaced geometry, in task order.

        Returns
        -------
        numpy.ndarray or None
            The (3N, 3N) Hessian in Hartree per Bohr^2, or None when a
            gradient is missing. `self._missing_point` then names the task that
            failed, because a batch that returns nothing is only diagnosable
            when it says where.
        """
        coordinates = np.asarray(state.coordinates, dtype=np.float64)
        ndof = 3 * coordinates.shape[0]
        divisor = 2.0 * self.step_size * ANGSTROM_TO_BOHR
        cell = len(self._cell_slots)
        size = ndof + cell
        hessian = np.zeros((size, size), dtype=np.float64)
        # The rows are walked in the order the tasks were made, and a partial
        # Hessian's task order is not the component order: the row of task k is
        # the atom and axis that k was built from, not k itself.
        mobile = [3 * int(atom) + int(axis) for atom, axis in self._batch_rows]
        for pair, (atom, axis) in enumerate(self._batch_rows):
            plus, minus = batch_results[2 * pair], batch_results[2 * pair + 1]
            gradient_plus = _active_gradient(plus)
            gradient_minus = _active_gradient(minus)
            if gradient_plus is None or gradient_minus is None:
                self._missing_point = pair
                return None
            difference = (np.asarray(gradient_plus, dtype=np.float64).reshape(-1)
                          - np.asarray(gradient_minus,
                                       dtype=np.float64).reshape(-1))
            hessian[3 * int(atom) + int(axis), :ndof] = difference / divisor
            if cell:
                # The same evaluation carries a stress, so the coupling of this
                # atomic component to the six lattice entries is differenced here
                # as well, at the box the atomic tasks all share. Without these
                # columns the symmetrization below would fold the cell rows'
                # couplings into them at half weight, and a mixed second
                # derivative read at half its value is not a rounding.
                forces = self._cell_force_pair(plus, minus, None)
                if forces is None:
                    self._missing_point = pair
                    return None
                hessian[3 * int(atom) + int(axis), ndof:] = forces / divisor
        if cell:
            mobile = mobile + [ndof + index for index in range(cell)]
            base = 2 * len(self._batch_rows)
            for index, slot in enumerate(self._cell_slots):
                plus = batch_results[base + 2 * index]
                minus = batch_results[base + 2 * index + 1]
                force = self._cell_force_pair(plus, minus, slot)
                gradients = self._gradient_difference(plus, minus)
                if force is None or gradients is None:
                    self._missing_point = len(self._batch_rows) + index
                    return None
                # The row this strain writes is the lattice entry's own: how every
                # atomic force and every other cell force answers to it.
                hessian[ndof + index, :ndof] = gradients / divisor
                hessian[ndof + index, ndof:] = force / divisor
        # Only the mobile block is the vibration problem, so only it is
        # symmetrized and kept: a displaced atom's coupling to a held one is a
        # real number of this finite difference and a fiction as a force
        # constant, and averaging the matrix whole would smear it into the held
        # rows that the analysis then has to ignore anyway. The six lattice
        # entries are always mobile: a run that froze one of them is not a run
        # that asked for a cell block at all.
        block = hessian[np.ix_(mobile, mobile)]
        full = np.zeros_like(hessian)
        full[np.ix_(mobile, mobile)] = 0.5 * (block + block.T)
        return full

    def _gradient_difference(self, plus, minus):
        """The atomic gradient difference of one pair, or None."""
        gradient_plus = _active_gradient(plus)
        gradient_minus = _active_gradient(minus)
        if gradient_plus is None or gradient_minus is None:
            return None
        return (np.asarray(gradient_plus, dtype=np.float64).reshape(-1)
                - np.asarray(gradient_minus, dtype=np.float64).reshape(-1))

    def _cell_force_pair(self, plus, minus, slot):
        """The cell force difference of one task pair, or None.

        Each side is read at the lattice *that side* was evaluated on, because
        the row being written is the derivative of the cell force with respect to
        one lattice entry and a force read at the reference box would difference
        to zero. An atomic task strains nothing, so `slot` is None and both sides
        are read at the reference lattice - which is what makes the coupling of
        an atomic component to the box a derivative at fixed cell.

        The reading goes through the same function the optimizer steps on
        (`utilities/lattice.py::stress_to_cell_force`), so a curvature and a step
        cannot disagree about what the conjugate of a lattice entry is. A missing
        tensor is None here and a named failure in `collect`, exactly as a missing
        gradient is: a zero would be six directions of zero curvature, which is
        the lowest curvature a mode-following search can be pointed at.
        """
        from ..utilities.lattice import stress_to_cell_force

        reference = np.asarray(self._reference_lattice, dtype=np.float64)
        lattices = []
        if slot is None:
            lattices = [reference, reference]
        else:
            delta = np.zeros((3, 3))
            delta[slot // 3, slot % 3] = self.step_size
            lattices = [reference + delta, reference - delta]
        forces = []
        for result, rows in zip((plus, minus), lattices):
            stress = _active_stress(result)
            if stress is None:
                return None
            forces.append(stress_to_cell_force(
                stress, rows * ANGSTROM_TO_BOHR).reshape(-1)[
                    list(self._cell_slots)])
        return forces[0] - forces[1]

    def postprocess(self, state, collected):
        """Write the matrix back and restore the reference geometry.

        Parameters
        ----------
        state : PyRAIState
            Current state.
        collected : numpy.ndarray or None
            Assembled Hessian, or None when the batch came back short.

        Returns
        -------
        numpy.ndarray or None
            The Hessian, passed on so `run` can decide what a None means.
        """
        if collected is not None:
            merged = dict(state.result.to_dict()) if state.result else {}
            merged['hessian'] = collected[np.newaxis, :, :].tolist()
            if self.mobile is not None:
                merged['hessian_atoms'] = [
                    int(atom) for atom in self.mobile]
            self.calculator.update({'results': merged})
        reference = self._reference_coordinates
        if reference is not None:
            self.calculator.update({'system': {'coordinates': reference}})
        return collected

    # ---- Analysis ----

    def analyze(self, state, hessian):
        """Read one Hessian into modes and corrections, and record them.

        The state's own masses and geometry enter as they are, in amu and Bohr,
        because that is the unit pair `vibration` is written in; the analysis
        writes the result keys the model declares, with the leading state
        dimension every array result carries.
        """
        masses = _amu_masses(state)
        coordinates = np.asarray(state.coordinates, dtype=np.float64)
        result = vibration(coordinates * ANGSTROM_TO_BOHR, masses, hessian,
                           mobile=self.mobile)
        linear = result['linear']
        if self.declared_linear is not None:
            linear = bool(self.declared_linear)
        summary = {
            'frequencies': result['frequencies'],
            'norm_mode': result['modes'],
            'reduced_mass': result['reduced_mass'],
            'rotation_constants': result['rotation_constants'],
            'linear': linear,
            'rigid_removed': result['rigid_removed'],
        }
        if self.thermo:
            reference_energy = getattr(self, '_reference_energy', None)
            if reference_energy is None:
                reference_energy = _electronic_energy(state)
            summary['thermo'] = thermo_analysis(
                reference_energy, masses, result['frequencies'],
                result['rotation_constants'], temperature=self.temperature,
                pressure=self.pressure,
                multiplicity=_multiplicity(state),
                scale_factor=self.freq_scale, cutoff=self.freq_cutoff,
                damping_alpha=self.damping_alpha, symmetry=self.symmetry,
                linear=linear)
        if self.save_molden:
            summary['molden'] = write_freq_molden(
                os.path.join(_output_dir(state),
                             '%s.freq.molden' % state.pmd.control.title),
                list(state.atoms), coordinates * ANGSTROM_TO_BOHR,
                result['frequencies'], result['modes'])
        merged = dict(getattr(self, '_reference_result', None)
                      or (state.result.to_dict() if state.result else {}))
        merged.update({
            'frequencies': summary['frequencies'][np.newaxis, :].tolist(),
            'norm_mode': summary['norm_mode'][np.newaxis, :, :, :].tolist(),
            'reduced_mass': summary['reduced_mass'][np.newaxis, :].tolist(),
            'rotation_constants': summary['rotation_constants'][
                np.newaxis, :].tolist(),
        })
        for key in ('zero_point_energy', 'internal_energy', 'enthalpy',
                    'entropy', 'free_energy', 'heat_capacity_cv',
                    'heat_capacity_cp'):
            if 'thermo' in summary:
                merged[key] = [float(summary['thermo'][key])]
        if 'hessian_atoms' not in merged and self.mobile is not None:
            merged['hessian_atoms'] = [int(atom) for atom in self.mobile]
        self.calculator.update({'results': merged})
        return summary

    # ---- Execution ----

    def run(self):
        """Take the Hessian by the route the input asked for, then read it.

        Returns
        -------
        PyRAIState
            The state carrying the matrix, and the modes and corrections when
            the analysis was asked for.

        Raises
        ------
        ValueError
            When the named producer returned no matrix and the input did not
            allow the other route, when the numerical batch came back without a
            gradient, or when the matrix cannot be read at all. The run is
            marked failed either way: a Hessian that silently produced nothing
            would be reported by the tail of the log as a completed property
            calculation.
        """
        state = self.calculator.get_state()
        # The displacement children overwrite the live result; keep the
        # reference evaluation's energy and multiscale decomposition so the
        # final frequency block is anchored to the surface it was taken on.
        self._reference_result = (
            dict(state.result.to_dict()) if state.result is not None else {})
        self._reference_energy = _electronic_energy(state)
        self.mobile = _mobile_selection(state, self.atoms)
        self.rows = _active_rows(state, self.mobile)
        if (self.driver != _SKILL_DRIVER and self.rows
                and len(self.mobile) != len({atom for atom, _ in self.rows})):
            # The numerical batch is narrowed by the constraint plan's fix mask
            # (a held component costs no displacement), so the analysis has to
            # see the same atoms: leaving held atoms in `mobile` diagonalizes
            # rows of zeros beside the mobile block and reports every one of
            # them as a soft mode. Measured on the 60-atom two-center case,
            # where the 54 held atoms turned the mobile block into a 174-entry
            # table whose first 168 rows read `-0.0000 cm-1 imaginary`.
            self.mobile = sorted({int(atom) for atom, _ in self.rows})
        ask_backend = self.driver == _SKILL_DRIVER
        hessian = self._backend_hessian(state) if ask_backend else None
        if ask_backend and hessian is not None and self.hybrid_cell_block:
            # Two producers, one matrix: the atoms come from the backend's own
            # second derivative, the six lattice rows from this framework's strain
            # of the box. Zero-filling those rows is not an option and never was -
            # six directions of zero curvature are the lowest a mode-following
            # search can be pointed at - so the block is measured here or the run
            # fails saying which half has no producer.
            hessian = self._strain_cell_block_onto(state, hessian)
        if hessian is None:
            if ask_backend and self.source != 'auto':
                # The producer was named, so a missing matrix is a failed
                # request rather than a route to relax: `auto` is the only
                # spelling that says "either producer will do".
                named = ('no analytical Hessian in the evaluation'
                         if self.source == 'analytic'
                         else 'no force constant matrix in the evaluation')
                self._fail(
                    state, '%s: hess_driver %r with hess_mode %r asks the '
                           'backend to produce it, and the framework producing '
                           "it by displacement is hess_driver 'main'"
                           % (named, self.driver, self.source))
            run_logger = getattr(state, 'logger', None)
            if run_logger is not None and ask_backend and self.source == 'auto':
                run_logger.write(
                    '\n  Hessian route: the evaluation returned no force '
                    'constant matrix, so the numerical displacement route is '
                    'taken instead (%d gradient evaluations at +/-%.3f A)\n'
                    % (2 * 3 * len(self.mobile or
                                   range(state.pmd.system.natom)),
                       self.step_size))
            route = 'numerical'
            state = self._run_batch(state)
            collected = self._collected
            if collected is None:
                point = ('atom %d axis %d' % self._batch_rows[
                    self._missing_point] if self._missing_point is not None
                    else 'unknown pair')
                self._fail(state, 'the gradient of displacement (%s) is '
                           'missing' % point)
            hessian = collected
        self.matrix = hessian
        if self.analysis:
            self.analyze(state, hessian)
        state.runtime.mark_completed()
        return state

    # ---- Parts ----

    def _backend_hessian(self, state):
        """The matrix the backend computes, from this run's own evaluation.

        A `hessian`/`frequency` jobtype already entitles the skill task to
        produce it: planning spelled the request into the skill's own jobtype
        and resolved the profile for it, so the matrix arrives in the result of
        the evaluation the run was going to do anyway.

        The producer's own method is not this workflow's business: a backend
        that answers with an analytical second derivative and one that answers
        with displacements of its own (ORCA's NumFreq, Molcas' MCKINLEY) both
        hand back the same `hessian` result.
        """
        result = self.calculator.run(target_states=self._target_states(state))
        state = self.calculator.get_state()
        matrix = _matrix_of(result)
        if matrix is None and state.result is not None:
            matrix = _matrix_of(state.result)
        if matrix is None and not _run_asks_for_a_matrix(state):
            matrix = self._requested_backend_matrix(state)
            if matrix is not None:
                self._record_matrix(state, matrix)
        return matrix

    def _requested_backend_matrix(self, state):
        """Ask the backend for its matrix as a job of its own.

        This is the second producer channel, and it exists because a framework
        optimization loop is evaluated for energy and gradient every step: the
        planning layer spells the skill's jobtype that way (`control.driver`
        'main' with `jobtype: optimize`), so the curvature a mode-following
        search wants before its first step is not in what that evaluation
        returns. Asking for it is one dispatched unit on the current geometry,
        not a change to what the loop asks for afterwards.

        The unit runs the same workflow this object does, because `frequency`
        resolves to it on the child side. Its own mode is pinned to 'analytic'
        before it is sent: left to its token it would resolve to 'auto', find no
        matrix from an engine that cannot give one, and start displacing the
        geometry a hundred-fold *inside the worker*. A refusal the caller can
        read is the wanted failure there.
        """
        unit = self.materialize_unit({}, 0, child_jobtype=_MATRIX_JOBTYPE)
        child = unit['state']
        # `apply_child_jobtype` aligns the framework-side jobtype of every task;
        # the skill reads its own `jobtype` out of the flat skill config it was
        # planned with, and the profile was resolved from that. A child whose
        # skill config still says 'energy gradient' answers with a gradient, and
        # the search is told there is no matrix. Both fields are moved here,
        # where a different property is being asked for than the parent run was
        # planned for.
        for _name, task in child.iter_tasks():
            config = task.get('config')
            skill = (config.get('skill_config')
                     if isinstance(config, dict) else None)
            if isinstance(skill, dict):
                skill['jobtype'] = _MATRIX_JOBTYPE
                skill['profile'] = ''
        section = getattr(getattr(child, 'pmd', None), 'hessian', None)
        if section is not None:
            section.hess_mode = 'analytic'
            # The parent wants the matrix, not the spectroscopy: a child that
            # also read the modes into thermochemistry would need atomic masses
            # the search itself does not ask for.
            section.hess_vibration = False
            section.hess_thermo = False
        # One unit is below dispatch's identity allocation (a batch of one keeps
        # today's shared layout), and a force-constant job writing into the
        # directory the optimization loop is evaluating in is not a shared
        # layout anyone asked for, so the identity is taken here.
        self.allocate_runtime_identities([unit])
        if self.local_executor is None:
            self.local_executor = _child_entry
        try:
            outcomes = self.dispatch(units=[unit])
        except ValueError as error:
            raise ValueError(
                "hess_driver 'skill' asked the backend for its force constants "
                "and the request did not go through: %s. A skill with no "
                "Hessian profile cannot answer; the framework can produce the "
                "matrix itself with hess_driver 'main'"
                % error) from error
        return _matrix_of(_outcome_result(outcomes[0] if outcomes else None))

    def _record_matrix(self, state, matrix):
        """Carry the matrix the child produced into this run's own result.

        The jobtype says force constants were computed, and every other route
        leaves `hessian` in the result with the leading state axis. A reader of
        the parent should not be able to tell which producer answered.
        """
        merged = dict(state.result.to_dict()) if state.result else {}
        merged['hessian'] = matrix[np.newaxis, :, :].tolist()
        self.calculator.update({'results': merged})

    def _run_batch(self, state):
        """The numerical route: displaced gradients, differenced."""
        batch = self.generate_batch(state)
        if self.local_executor is None:
            self.local_executor = _child_entry
        results = self.dispatch(units=batch, child_jobtype=self.child_jobtype)
        self._collected = self.collect(state, results)
        if self._collected is not None:
            self.postprocess(state, self._collected)
        return state

    def _strain_cell_block_onto(self, state, atomic):
        """Join the six lattice rows a strained box answers to a backend matrix.

        Twelve evaluations: each relaxed lattice component at plus and minus this
        batch's step, with the atoms riding it. Each pair answers two rows' worth
        of content at once, which is why this is twelve and not eighteen: the
        atomic gradient difference says how every component of the structure
        answers to that strain, and the cell force difference says how the box
        does. The first is the coupling a backend cannot supply at all, so it is
        kept beside the cell block rather than zero-filled, and both are divided
        by the same `2h` the 'main' route uses.

        The backend's block is symmetrized and placed over the atomic corner; the
        lattice rows are symmetrized into their mirrored columns, so a coupling
        appears once with one value rather than twice with two.

        Parameters
        ----------
        state : PyRAIState
            The geometry, box and stress-bearing results of this evaluation.
        atomic : numpy.ndarray
            The backend's (3N, 3N) matrix in Hartree per Bohr^2.

        Returns
        -------
        numpy.ndarray
            (3N + 6, 3N + 6), the backend's block and the strained one joined.

        Raises
        ------
        ValueError
            When the backend matrix does not cover exactly this state's atoms:
            padding a partial matrix with rows no evaluation produced is the
            zero-curvature fiction this route exists to avoid.
        """
        coordinates = np.asarray(state.coordinates, dtype=np.float64)
        size = 3 * coordinates.shape[0]
        matrix = np.asarray(atomic, dtype=np.float64)
        if matrix.shape != (size, size):
            raise ValueError(
                'optimize_cell with hess_driver %r needs the backend matrix to '
                'cover every atom of this state: it answered %s for %d atoms, '
                'and the rows it did not answer would be zeros no evaluation '
                'produced' % (self.driver, matrix.shape, coordinates.shape[0]))
        tasks = self._cell_batch(state, coordinates)
        results = self.dispatch(units=tasks, child_jobtype=self.child_jobtype)
        if results is None or len(results) != len(tasks):
            self._fail(state, 'the cell strain batch did not answer on every '
                              'point, so a backend matrix has no cell block to '
                              'join')
        divisor = 2.0 * self.step_size * ANGSTROM_TO_BOHR
        cell = len(self._cell_slots)
        joined = np.zeros((size + cell, size + cell), dtype=np.float64)
        joined[:size, :size] = 0.5 * (matrix + matrix.T)
        for index in range(cell):
            plus = results[2 * index]
            minus = results[2 * index + 1]
            forces = self._cell_force_pair(plus, minus, self._cell_slots[index])
            gradients = self._gradient_difference(plus, minus)
            if forces is None or gradients is None:
                self._fail(
                    state, 'one of the twelve strained evaluations returned no '
                           '%s, which is a lattice row read as zero rather than '
                           'differenced'
                           % ('stress' if forces is None else 'gradient'))
            row = np.concatenate((gradients / divisor,
                                  np.asarray(forces, dtype=np.float64) / divisor))
            joined[size + index, :] = row
            joined[:, size + index] = row
        return joined

    def _fail(self, state, reason):
        """Stop the run with the reason, rather than with an empty result."""
        state.runtime.mark_failed('Hessian: %s' % reason)
        raise ValueError('Hessian failed: %s' % reason)

    @staticmethod
    def _target_states(state):
        """The surface the matrix is differentiated on."""
        try:
            return [int(state.current_state)]
        except (AttributeError, TypeError, ValueError):
            # A multi-center state has no single surface.
            return None

    # ---- Runtime layout ----

    def runtime_isolated_workdir(self):
        """Give every displaced geometry its own runtime directory.

        A Hessian batch is 6N calculations of the same molecule at the same
        title; unisolated they share one calculation directory and one log, and
        the only trace left of 6N-1 of them is whatever the last one overwrote.
        """
        return True


# ---- Keyword and helper functions ----

def _hessian_keywords(state):
    """The `&hessian` section as the workflow reads it."""
    if state is None:
        return {}
    getter = getattr(state, 'hessian_params', None)
    return dict(getter or {})


# The jobtype that asks a skill for its force constants. `frequency` is the
# spelling, not `hessian`: the capability a Hessian request amounts to is
# reversed back into this token by `CAPABILITY_TO_TOKEN`, and it is this token
# each skill's profile table answers (`pyscf_skill` maps 'frequency' to its
# `_hessian` profile).
_MATRIX_JOBTYPE = 'frequency'

# The jobtype tokens that already entitle a skill task to produce a matrix.
_MATRIX_TOKENS = frozenset({'hessian', 'hess', 'frequency', 'freq',
                            'num_hess', 'numhess', 'num_freq', 'numfreq'})


def _run_asks_for_a_matrix(state):
    """Whether this run's own jobtype already spelled the force-constant request.

    Read from the run's jobtype and not from the task configs: a `hessian` or
    `frequency` job is *what the run asked for*, planning resolved the skill's
    profile for it, and the matrix belongs in the evaluation the run was going
    to do anyway. An `optimize` run asked for something else (the loop spells
    energy and gradient for the skill), so the matrix it wants has to be asked
    for as a job of its own.
    """
    if state is None:
        return False
    tokens = str(getattr(state, 'jobtype', '') or '').lower().split()
    return bool(set(tokens) & _MATRIX_TOKENS)


def _active_rows(state, atoms):
    """The (atom, axis) pairs this batch may displace.

    The selection arrives as atoms (`hess_atoms`, which an optimization run
    derives from the constraint plan), while freezing is per component: a fix on
    x alone leaves y and z free. Displacing a frozen component would spend two
    evaluations on a difference that must come back zero, and the row it writes
    is a fiction about a coordinate the run pinned - the same reason a held atom
    is kept out of the symmetrized block.

    With no constraint declared this returns the mobile atoms' full Cartesian
    triple, which is what every run of this workflow did before the fix mask was
    read here.
    """
    from .constraints import projection

    natom = int(state.pmd.system.natom or len(state.atoms))
    plan = (state.constraint_plan()
            if hasattr(state, 'constraint_plan') else None)
    mask = None
    if plan is not None and not plan.is_empty:
        mask = projection.build_fix_mask(plan, natom)
    rows = []
    for atom in atoms:
        for axis in range(3):
            if mask is not None and mask[int(atom), axis]:
                continue
            rows.append((int(atom), axis))
    return rows


def _outcome_result(outcome):
    """The result mapping of one dispatched outcome, dict or child state.

    A single-point child answers with a plain result dict; a child that ran a
    whole workflow (`frequency` resolves to this one) answers with the state it
    ended on, whose results sit one level down.
    """
    if outcome is None:
        return None
    result = getattr(outcome, 'result', None)
    if result is not None:
        return result
    return outcome if hasattr(outcome, 'get') else None


def _requested_source(state):
    """Which matrix route the jobtype tokens spell."""
    if state is None:
        return None
    tokens = str(getattr(state, 'jobtype', '') or '').lower().split()
    if 'num_hess' in tokens or 'num_freq' in tokens:
        return 'numerical'
    if 'hessian' in tokens or 'frequency' in tokens:
        # The declared default of the analytical tokens is the preference plus
        # its downgrade, not a hard requirement: a backend that answers with
        # frequencies but no matrix has not failed the run's request, and the
        # alternative was built for exactly that case.
        return 'auto'
    return None


def _wants_analysis(state):
    """Whether the run asked for modes and corrections, not only a matrix."""
    if state is None:
        return True
    tokens = str(getattr(state, 'jobtype', '') or '').lower().split()
    if any(token in tokens for token in ('frequency', 'num_freq', 'freq')):
        return True
    return not any(token in tokens for token in ('hessian', 'hess', 'num_hess'))


def _flag(params, key, explicit, default=False):
    """One boolean keyword: the constructor argument wins over the section."""
    if explicit is not None:
        return bool(explicit)
    value = params.get(key, None)
    return default if value is None else bool(value)


def _matrix_of(result):
    """The (3N, 3N) matrix of one result, whatever state dimension it came in."""
    if result is None:
        return None
    matrix = result.get('hessian')
    if matrix is None:
        return None
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim == 3:
        if array.shape[0] == 0:
            return None
        array = array[0]
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        return None
    return 0.5 * (array + array.T)


def _active_gradient(result):
    """The gradient of one displaced evaluation, on the state that owns it."""
    if result is None:
        return None
    gradient = (result.get('gradient') if hasattr(result, 'get')
                else None)
    if gradient is None:
        return None
    array = np.asarray(gradient, dtype=np.float64)
    return array[0] if array.ndim == 3 else array


def _active_stress(result):
    """The whole-system stress of one strained evaluation, (3, 3) or None.

    The same reading the gradient above takes on the state axis: a batch child
    carries one state's tensor behind it, and a run that asked for more than one
    state has no business in a curvature batch that strains a box. Units are the
    framework's by the time a child's result is read back - the calculator
    converts on the way out (`convert_stress_to_atomic`), so this is Hartree per
    Bohr^3 and needs no factor of its own.
    """
    if result is None or not hasattr(result, 'get'):
        return None
    stress = result.get('stress_tensor')
    if stress is None:
        return None
    array = np.asarray(stress, dtype=np.float64)
    if array.ndim == 3:
        array = array[0]
    return array if array.shape == (3, 3) else None


def _mobile_selection(state, atoms):
    """The mobile atoms, 0-based, from the 1-based input spelling."""
    natom = int(state.pmd.system.natom or len(state.atoms))
    if atoms is None or (hasattr(atoms, '__len__') and not len(atoms)):
        return list(range(natom))
    selected = resolve_indices(atoms, natom=natom)
    return sorted(int(index) for index in np.asarray(selected).reshape(-1))


def _amu_masses(state):
    """Atomic masses in amu, the unit the vibrational analysis is written in."""
    masses = getattr(state.pmd.system, 'mass', None)
    array = np.asarray(masses, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(
            'the vibrational analysis needs atomic masses and the system '
            'carries none')
    return array


def _electronic_energy(state):
    """The energy of the surface the Hessian was taken on, in Hartree."""
    result = state.result
    energies = result.get('energy') if result is not None else None
    if energies is None:
        return 0.0
    array = np.asarray(energies, dtype=np.float64).reshape(-1)
    try:
        index = int(state.current_state)
    except (AttributeError, TypeError, ValueError):
        warnings.warn(
            'hessian: the state has no single current_state; using the '
            'ground-state row of the energy array', stacklevel=2)
        index = 0
    return float(array[min(index, array.size - 1)])


def _multiplicity(state):
    """Spin multiplicity of the electronic entropy, guarded away from zero.

    Under several centers `system.multiplicity` is one value per center - the
    input field keeps its shape - and the entropy needs the whole system's,
    which is the plan's paired-off aggregate.
    """
    plan = getattr(state, 'partition_plan', None)
    if plan is not None and plan.n_center > 1:
        value = plan.system_multiplicity or 1
    else:
        value = getattr(state.pmd.system, 'multiplicity', 1) or 1
    if int(value) < 1:
        return 1
    return int(value)


def _output_dir(state):
    """The run's own directory, where its records belong."""
    logger = getattr(state, 'logger', None)
    return (getattr(logger, 'log_dir', None) or os.getcwd())


# Kept as a name: it is what this class was while it only differentiated
# gradients, and it is what the tests and `Optimize` still ask for by name.
NumericalHessian = Hessian
