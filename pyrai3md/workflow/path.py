"""
======================================================================
This module provides the Path workflow: the searches that walk more than
one structure.

    optimize.method: mep   minimum energy path from the geometry given
    optimize.method: irc   the same walk, seeded along the imaginary mode
                           of a transition state
    optimize.method: neb   a nudged elastic band between two structures

It is resolved by `resolve_workflow` from `optimize.method`, so the input
surface is the one the single-structure searches already use: there is no
path jobtype and no path section, and `opt_method` is what decides which of
the two loops a run gets. The run log carries the same `&optimization setup`
and `&optimization progress` blocks as a relaxation: an accepted point writes
the level-gated property block plus the progress block, a band round writes
the progress block only, and a rejected probe appears at `verbose >= 3`.
`path_status.txt`/`path_profile.txt` remain the machine record and gain the
method columns (`settled`, `barrier`) while keeping their existing order. The algorithms live in `workflow/optimizer/`
beside the other searches; this module owns the multi-structure part.

Two execution routes, because the two kinds of walk evaluate differently:

    walker (mep, irc)  one probe at a time, in place on the calculator's own
                       state, the way `scan.py` steps through geometries: a
                       probe's gradient is needed in the same frame that
                       decides the next probe, and the sequence is serial by
                       construction.
    band (neb)         one round is every image, fanned out as a batch with
                       `Dispatcher.dispatch(units=batch)` exactly as the
                       Hessian workflow fans out its displaced geometries, so
                       the images run in parallel (`multipool`) and each gets
                       its own runtime directory (`runtime_isolated_workdir`).

Collections: the ECME algorithm used by LAMMPS is not this; a band here is a
plain nudged elastic band with an optional climbing image.

The records: `path.xyz` (the latest frame), `path_geom.xyz` (every accepted
frame, which is the walked path), `path_profile.txt` (one row per accepted
point or per image), `path_status.txt` (one row per evaluation or per band
round), and `path_forward.xyz`/`path_backward.xyz` for the two sides of an
IRC. None of them is `opt_status.txt`: that reader classifies a record by its
column count (`utilities/print_results.py::classify_header`), so a path table
written through that module's functions would be read as another writer's
record and every existing reader of it would come back empty.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os

import numpy as np

from .base import Workflow
from .constraints import runtime as constraint_runtime
from .optimize import Optimize, resolve_skill_units
from .optimizer import get_optimizer
from .records import lattice_comment
from ..data.geometry import apply_pbc
from ..schema.config import DEFAULT, PATH_METHODS
from ..utilities.geodesic import consistent_pair
from ..utilities.geodesic import interpolate as geodesic_path
from ..utilities.io import read_xyz_frame
from ..utilities.lattice import (CELL_COMPONENT_SLOTS, cart_to_frac,
                                 frac_to_cart, min_image_displacement)
from ..utilities.units import BOHR_TO_ANGSTROM
from ..utilities.print_results import format_path_summary

# The two status tables. A walker's rows are per evaluation and a band's are per
# round, so they do not carry the same columns and neither of them is
# `opt_status.txt`: the walker's `radius` is the mass-weighted radius of a probe
# and the band's `image` is which image the row is about, and calling both of
# them one name would be a column that answers two questions.
PATH_WALKER_COLUMNS = ('step', 'direction', 'energy', 'shift', 'max_d', 'rms_d',
                       'max_g', 'rms_g', 'radius', 'residual', 'settled')
PATH_BAND_COLUMNS = ('step', 'energy', 'shift', 'max_d', 'rms_d', 'max_g',
                     'rms_g', 'image', 'climb', 'barrier')
# The cell readings of a variable-cell path, under the names
# `utilities/opt_formatter.py` uses for the same five numbers, so one reading is
# spelled one way across both records.
PATH_CELL_COLUMNS = ('cell_grms', 'cell_gmax', 'cell_drms', 'cell_dmax',
                     'volume')
# One row per accepted point (walker) or per image (band).
PATH_PROFILE_COLUMNS = ('index', 'direction', 'energy', 'de', 'arc', 'length',
                        'perp_rms', 'perp_max', 'radius', 'role')
PATH_PROFILE_CELL_COLUMN = 'volume'
_PATH_STEP_WIDTH = 6
_PATH_FIELD_WIDTH = 18

# The path row's lowercase names and the convergence-table names they feed.
# The mapping is one-way and lives here because `path_status.txt` keeps the
# lowercase spelling its own readers use.
_PATH_REPORT_KEYS = (
    ('energy', 'Energy'),
    ('shift', 'Delta E'),
    ('max_d', 'Displacement Max'),
    ('rms_d', 'Displacement RMS'),
    ('max_g', 'Gradient Max'),
    ('rms_g', 'Gradient RMS'),
    ('cell_grms', 'Cell Gradient RMS'),
    ('cell_gmax', 'Cell Gradient Max'),
    ('cell_drms', 'Cell Displacement RMS'),
    ('cell_dmax', 'Cell Displacement Max'),
    ('volume', 'Cell Volume'),
)


def path_report_for_log(row):
    """Map a path row onto the common convergence-report names."""
    report = {}
    for key, name in _PATH_REPORT_KEYS:
        value = (row or {}).get(key)
        if value is not None:
            report[name] = value
    return report

# The geodesic smoother's tolerance and evaluation budget. Not keywords: they
# answer how hard the *initial guess* is polished, which is not a question the
# input has - the band is relaxed by the run that follows either way. Bigger
# than the reference's command-line default (15) because this smoother evaluates
# no potential: its work is linear algebra on a small dense system.
_GEODESIC_TOL = 2.0e-3
_GEODESIC_MAXIT = 50
# The seed of the bisection nudges when the run declares none, so a band that
# asked for nothing is still reproducible.
_DEFAULT_GL_SEED = 1


def fixture_path(name):
    """A structure file by the path given, or beside the shipped examples.

    A run resolves an input file against its own working directory, and the
    auxiliary files a shipped example names - a band's endpoint structures, the
    vibration file an IRC reads its mode from - live in the package's
    `examples/` directory, not in the directory the qualification runner creates
    for the case. Measured on the first path round: every case that named a file
    failed with `read_xyz_frame` unable to find it while every case that
    declared its structure inline passed, which is the difference this fallback
    removes. An absolute path, or one that exists as given, is used as it is.
    """
    if not name:
        return name
    if os.path.isabs(name) or os.path.exists(name):
        return name
    beside = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, 'examples',
        name))
    return beside if os.path.exists(beside) else name


def linear_path(start, end, nimages, lattice=None):
    """`nimages` interior geometries evenly spaced between two endpoints.

    Parameters
    ----------
    start, end : array-like
        `(natom, 3)` endpoints in Angstrom.
    nimages : int
        How many interior geometries to produce.
    lattice : array-like or None
        `(3, 3)` cell in Angstrom. Given, the interpolation happens in
        fractional coordinates at that cell, with the second endpoint first
        moved into the same periodic image as the first: a straight Cartesian
        line between two images cut through a face, and the atoms it tears apart
        are the ones the band would then have to stitch back.

    Returns
    -------
    numpy.ndarray
        `(nimages, natom, 3)` interiors, in Angstrom.
    """
    first = np.asarray(start, dtype=np.float64).reshape(-1, 3)
    last = np.asarray(end, dtype=np.float64).reshape(-1, 3)
    blend = np.linspace(0.0, 1.0, int(nimages) + 2)[1:-1]
    if lattice is None:
        return first[None] + blend[:, None, None] * (last - first)[None]
    cell = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    first_fraction = cart_to_frac(first, cell)
    step_fraction = cart_to_frac(min_image_displacement(last - first, cell),
                                 cell)
    fractions = first_fraction[None] + blend[:, None, None] * step_fraction[None]
    return np.asarray([frac_to_cart(fraction, cell) for fraction in fractions])


def build_band(atoms, reactant, product, images, scheme='linear', ts=None,
               lattice=None, scaler=1.7, threshold=3.0, min_neighbors=4,
               rng=None):
    """The chain of a nudged elastic band, and what each image is.

    Parameters
    ----------
    atoms : sequence of str
        Element symbols, for the geodesic metric's pair rest lengths.
    reactant, product : array-like
        `(natom, 3)` endpoints in Angstrom. A periodic pair is brought into
        one frame first (`geodesic.consistent_pair`): the product is moved to
        the image nearest the reactant, which is the same structure and the
        same crystal, so the chain has no whole-box step in it. The record's
        `Lattice=` comment makes the moved frame readable.
    images : int
        How many interior images the band carries. `optimize.neb_images`.
    scheme : str
        'linear' or 'geodesic'.
    ts : array-like or None
        An optional intermediate transition-state guess. Given, the band is
        built as two segments split by their lengths and the guess is inserted
        between them as an image like any other - it is relaxed, not fixed, and
        the profile's `role` column records that it started there.
    lattice : array-like or None
        The cell, for a periodic band. One `(3, 3)` cell applies to every image;
        a pair of them is the two endpoints' cells, and the band's cells then
        run linearly between those two while the metric reads each image at its
        own.
    scaler, threshold, min_neighbors
        The geodesic metric's three parameters.
    rng : numpy.random.Generator or None
        Seed of the bisection nudges.

    Returns
    -------
    tuple
        `(chain, cells, roles)`: `(nimages + 2, natom, 3)` geometries in
        Angstrom, one cell per image or None, and the role of each.
    """
    count = int(images)
    if count < 1:
        raise ValueError(
            'a band carries at least one interior image and this run asked for '
            '%d' % count)
    first = np.asarray(reactant, dtype=np.float64).reshape(-1, 3)
    last = np.asarray(product, dtype=np.float64).reshape(-1, 3)
    key = str(scheme or 'linear').strip().lower()
    single = None
    ladder = None
    if lattice is not None:
        given = np.asarray(lattice, dtype=np.float64)
        if given.ndim == 3:
            ladder = given
        else:
            single = given.reshape(3, 3)
    # The chain is built in one frame. A product file written in the main
    # image may put a straddling molecule on the far side of a face, and a
    # chain that starts in the reactant's frame and ends in the product's
    # has a whole-box step in it - the interpolation itself would tear the
    # molecule through the wall, whatever the metric does.
    pair_cell = (single if single is not None
                 else (ladder[0] if ladder is not None else None))
    first, last = consistent_pair(first, last, pair_cell)

    def interiors(start, end, number):
        if key == 'linear':
            return linear_path(start, end, number, lattice=single)
        if key == 'geodesic':
            return geodesic_path(
                atoms, [start, end], number,
                lattice=(ladder if ladder is not None else single),
                scaler=scaler, threshold=threshold,
                min_neighbors=min_neighbors, rng=rng)
        raise ValueError(
            "Unknown neb_interpolate %r. Options: linear, geodesic" % scheme)

    if ts is None:
        chain = np.concatenate(
            (first[None], interiors(first, last, count), last[None]))
        roles = ['reactant'] + ['image'] * count + ['product']
    else:
        saddle = consistent_pair(first, ts, pair_cell)[1]
        first_leg = float(np.linalg.norm(saddle - first))
        second_leg = float(np.linalg.norm(last - saddle))
        total = first_leg + second_leg
        # Each segment gets at least one interior image, so the split cannot
        # starve a leg however lopsided the endpoints are.
        left = int(min(max(round(count * first_leg / total) if total else 1,
                            1), count - 1))
        chain = np.concatenate(
            (first[None], interiors(first, saddle, left), saddle[None],
             interiors(saddle, last, count - left), last[None]))
        roles = (['reactant'] + ['image'] * left + ['ts']
                 + ['image'] * (count - left) + ['product'])
    cells = None
    if ladder is not None or single is not None:
        pair = (ladder if ladder is not None
                else np.asarray([single, single], dtype=np.float64))
        cells = np.asarray([pair[0] * (1.0 - step) + pair[-1] * step
                            for step in np.linspace(0.0, 1.0, len(chain))])
    return chain, cells, roles


class Path(Workflow):
    """The workflow that drives a path search.

    Parameters
    ----------
    calculator : Calculator or None
        Calculator owning the run's evaluations.
    maxiter : int
        The step bound: path points for a walker, band rounds for a band.
    **kwargs
        Dispatcher parameters (state, multitask, remotetask, multipool,
        local_executor, executor_spec).
    """

    name = 'path'
    workflow_kind = 'iterative'
    # Every image of a band is one energy and one gradient.
    child_jobtype = 'energy gradient'

    def __init__(self, calculator=None, maxiter=100, **kwargs):
        super().__init__(name='path', calculator=calculator, maxiter=maxiter,
                         **kwargs)
        state = self.calculator.get_state() if self.calculator else None
        params = state.optimize_params if state else {}

        self.method = str(params.get('opt_method', 'mep') or 'mep').strip().lower()
        if self.method not in PATH_METHODS:
            raise ValueError(
                "the Path workflow drives %s and this run asks for %r; the "
                "single-structure searches are driven by Optimize"
                % (', '.join(PATH_METHODS), params.get('opt_method')))
        (self.coord_to_bohr, self.disp_to_angstrom,
         self.gradient_factor) = resolve_skill_units(self.calculator,
                                                     self.logger)
        self.maxit = int(params.get('opt_maxit', maxiter) or maxiter)
        self.state_i = int(params.get('opt_state_i', 0) or 0)
        # One state, one surface: the pair a crossing search solves on is
        # refused for a path at the input layer (`planning/assembly.py::build_task_queue`).
        self.target_states = [self.state_i] if self.state_i else None
        self.cell_variables = bool(params.get('optimize_cell', False))
        self.optimize_params = dict(params)
        self.optimizers = self._make_optimizers(state)
        self.output_dir = self._resolve_output_dir(state)
        self.band_roles = None

    # ---- Construction ----

    def _make_optimizers(self, state):
        """One optimizer per walk, and one walk per side of an IRC.

        `irc_direction: both` is two walks from the one saddle, and they are two
        optimizers rather than one driven twice because each carries its own
        sphere, its own micro-iteration and its own list of accepted points;
        sharing one would have the second walk inherit the first's state.
        """
        params = self.optimize_params
        if self.method == 'irc':
            requested = str(params.get('opt_irc_direction', 'both')
                            or 'both').strip().lower()
            branches = (['backward', 'forward'] if requested == 'both'
                        else [requested])
        else:
            branches = [None]
        return [self._build_optimizer(state, branch) for branch in branches]

    def _build_optimizer(self, state, branch):
        """The keyword set one optimizer gets, by method."""
        params = self.optimize_params
        shared = dict(
            maxiter=self.maxit,
            energy_threshold=float(params.get('opt_energy', 1e-6)),
            gradient_threshold=float(params.get('opt_grms', 3e-4)),
            gradient_max_threshold=float(params.get('opt_gmax', 4.5e-4)),
            displacement_threshold=float(params.get('opt_drms', 1.2e-3)),
            displacement_max_threshold=float(params.get('opt_dmax', 1.8e-3)),
            cell_stress_rms_threshold=float(params.get('opt_csrms', 40.0)),
            cell_stress_max_threshold=float(params.get('opt_csmax', 60.0)),
            cell_strain_rms_threshold=float(params.get('opt_cerms', 5.0e-5)),
            cell_strain_max_threshold=float(params.get('opt_cemax', 7.0e-5)),
            external_pressure=float(params.get('opt_pressure', 0.0)),
            cell_coupling=str(params.get('opt_cell_coupling', 'anisotropic')),
            active_state=self.state_i,
            cell_variables=self.cell_variables,
            coord_to_bohr=self.coord_to_bohr,
            disp_to_angstrom=self.disp_to_angstrom,
        )
        radius = params.get('opt_trust_radius')
        if radius:
            shared['trust_radius'] = float(radius)
        if self.method == 'neb':
            shared.update(
                neb_spring=float(params.get('opt_neb_spring', 0.01)),
                neb_climb=bool(params.get('opt_neb_climb', False)),
                neb_climb_threshold=float(
                    params.get('opt_neb_climb_threshold', 1.0e-2)),
                neb_step=float(params.get('opt_neb_step', 0.1)),
            )
        else:
            inner = params.get('opt_path_inner_tol')
            if inner is None or inner == DEFAULT['opt_path_inner_tol']:
                # Unset, the inner tolerance is the run's own gradient
                # threshold: a path point is found when the tangential gradient
                # is as quiet as the run considers converged, so one keyword
                # drives both. The shipped HCN walk is what made this a rule
                # rather than a default: it loosens `opt_grms` to 3e-3 and left
                # the inner tolerance at 3e-4, ten times tighter, so every point
                # spent its whole probe budget hovering around a minimum it was
                # never allowed to call found.
                inner = params.get('opt_grms', 3.0e-4)
            shared.update(
                path_radius=float(params.get('opt_path_radius', 0.1)),
                inner_maxit=int(params.get('opt_path_inner_maxit', 20)),
                inner_tol=float(inner),
                constraint_model=str(params.get('opt_path_constraint', 'sphere')),
            )
        if self.method == 'irc':
            shared.update(
                direction=branch,
                irc_mode=str(params.get('opt_irc_mode', 'hessian')),
                irc_mode_file=fixture_path(
                    str(params.get('opt_irc_mode_file', '') or '')),
            )
            if self.state_i > 0:
                driver = str((getattr(state, 'hessian_params', None) or {}).get(
                    'hess_driver') or '').strip().lower()
                if driver and driver != 'main':
                    # The same reason the saddle search on a variable cell
                    # refuses it (`workflow/optimize.py`): a backend's force
                    # constants are its ground state's, and the mode this walk
                    # needs is one the excited surface would have to supply.
                    raise ValueError(
                        'optimize.method irc on state %d with hess_driver=%r has '
                        'no curvature producer: a backend answers with its '
                        'ground state\'s force constants, and the imaginary mode '
                        'this walk seeds from is the one the surface it is '
                        'walked on would have. Ask the framework for the matrix '
                        '(hess_driver: main), or walk the ground state'
                        % (self.state_i, driver))
        optimizer = get_optimizer(self.method, **shared)
        optimizer.gradient_factor = self.gradient_factor
        return optimizer

    # ---- Execution ----

    def run(self):
        """Walk the path and leave its records behind.

        Returns
        -------
        PyRAIState
            The state holding the primary path's last accepted point (the
            forward branch of an IRC `both`, the product endpoint of a band),
            and the `path_*` result block.
        """
        state = self.calculator.get_state()
        self.output_dir = self._resolve_output_dir(state)
        self._init_output_files()
        run_logger = getattr(state, 'logger', None)

        if self.method == 'neb':
            if run_logger is not None:
                run_logger.log_optimize_setup(self._setup_block(state))
            state = self._run_band(state)
        else:
            if self.optimizers[0].needs_initial_hessian:
                self._supply_initial_hessian(state)
            if run_logger is not None:
                run_logger.log_optimize_setup(self._setup_block(state))
            # `irc_direction: both` is two walks from the one saddle, and the
            # state is walked in place, so the second branch has to be put back
            # on the geometry the run was given. Leaving it on the first
            # branch's last point starts the second walk from the wrong
            # structure: measured on `path_irc` before this reset, the two
            # `saddle` rows read -91.5616686633 and -91.6662865772, and on
            # `path_cell_irc` the forward branch was an empty walk from the
            # backward branch's endpoint (its only row carried the backward
            # walk's final box). `tools/verify.py` section F now reads the pair.
            origin = self._walker_origin(state)
            for index, optimizer in enumerate(self.optimizers):
                if index:
                    self._restore_walker_origin(state, origin)
                state = self._run_walker(state, optimizer)
            self._write_profile(self.optimizers)
            self._write_results(state, self.optimizers)

        run_logger = getattr(state, 'logger', None)
        finished = self._finished()
        if run_logger is not None:
            run_logger.log_optimize_summary(
                finished, self.maxit, None,
                message='' if finished else self._stop_message())
            if self.method == 'neb':
                # The band's machine record is the two path files; put its
                # final status and image energies in the main log too, so a
                # reader of <title>.log does not stop at the setup block.
                status_rows = [dict(optimizer.report())
                               for optimizer in self.optimizers]
                profile = []
                for optimizer in self.optimizers:
                    profile.extend(optimizer.path_points())
                run_logger.write(format_path_summary(status_rows, profile))
        if finished:
            state.runtime.mark_completed()
        else:
            state.runtime.mark_failed(self._stop_message())
        return state

    def _setup_block(self, state):
        """Run-level values for the `&optimization setup` block."""
        optimizer = self.optimizers[0] if self.optimizers else None
        setup = {
            'Method': self.method,
            'Optimizer': type(optimizer).__name__ if optimizer else None,
            'Max steps': int(self.maxit),
        }
        if self.method == 'neb' and optimizer is not None:
            count = getattr(optimizer, 'count', None)
            if count:
                setup['Band size'] = int(count)
            setup['Spring'] = getattr(optimizer, 'spring', None)
            setup['Climbing'] = bool(getattr(optimizer, 'climb', False))
        if self.method == 'irc' and optimizer is not None:
            setup['IRC direction'] = getattr(optimizer, 'direction', None)
            setup['IRC mode'] = getattr(optimizer, 'irc_mode', None)
            setup['IRC imaginary count'] = getattr(
                optimizer, 'imaginary_count', None)
            frequency = getattr(optimizer, 'frequency', None)
            if frequency is not None:
                setup['IRC imaginary frequency'] = float(frequency)
        if self.cell_variables:
            setup['Cell mode'] = 'variable'
            coupling = getattr(optimizer, 'cell_coupling', None)
            if coupling:
                setup['Cell coupling'] = str(coupling)
            pressure = getattr(optimizer, 'external_pressure', None)
            if pressure is not None:
                setup['Pressure target'] = '%.4f bar' % float(pressure)
            system = getattr(getattr(state, 'pmd', None), 'system', None)
            volume = getattr(system, 'volume', None)
            if volume:
                setup['Initial volume'] = '%.6f A^3' % float(volume)
        else:
            setup['Cell mode'] = 'fixed'
        plan = state.constraint_plan() if state is not None else None
        if plan is not None and not plan.is_empty:
            setup['Constraint conditions'] = len(plan.actions)
        return setup

    def _log_path_step(self, state, optimizer, step, write_structure=False):
        """Write one path row to the run log.

        Accepted points get the ordinary property block (gated by verbose)
        plus the convergence table; rejected probes only appear at verbose 3,
        where the extra evaluations are wanted. Band rounds carry no single
        structure to print, so they call the convergence table only.
        """
        run_logger = getattr(state, 'logger', None)
        if run_logger is None:
            return
        row = optimizer.report()
        common = path_report_for_log(row)
        details = optimizer.method_details(row)
        settled = bool(row.get('settled', row.get('converged', False)))
        if write_structure:
            run_logger.log_step(step, state.pmd.model_dump())
        if write_structure or int(getattr(run_logger, 'verbose', 0)) >= 3:
            run_logger.log_optimize_step(
                step, common, settled, details=details)

    @staticmethod
    def _walker_origin(state):
        """The geometry, box and ledger every branch of one IRC starts from.

        A walker is driven on the calculator's state in place, so a second
        branch inherits whatever the first left - the coordinates, the box of a
        variable-cell walk, and the crossing ledger a periodic one accumulated.
        This snapshot is what `_restore_walker_origin` puts back between
        branches; it holds only the fields a walk mutates that decide where the
        next walk begins.
        """
        system = getattr(getattr(state, 'pmd', None), 'system', None)
        if system is None:
            return None
        return {
            'coordinates': [list(row) for row in (system.coordinates or [])],
            'lattice_vector': [list(row)
                               for row in (system.lattice_vector or [])],
            'cell': list(getattr(system, 'cell', None) or []),
            'image': [list(row) for row in (getattr(system, 'image', None)
                                            or [])],
        }

    @staticmethod
    def _restore_walker_origin(state, origin):
        """Put the state back on the geometry its next branch starts from."""
        if not origin:
            return
        state.update_system(origin)

    def _stop_message(self):
        """Why the path stopped short, in the run log's own words.

        A band that has not settled has no reason of its own - it simply ran out
        of rounds - and the framework's phrase for that is the one the
        single-structure loop writes and the qualification runner reads
        (`Maximum optimization steps reached`, reported as PASS[NOT CONVERGED]).
        """
        reasons = [optimizer.stop_reason for optimizer in self.optimizers
                   if optimizer.stop_reason]
        return ('; '.join(reasons) if reasons
                else 'Maximum optimization steps reached')

    def _finished(self):
        """Whether every walk ended by walking rather than by running out.

        A walk that stopped because it reached its step budget did not finish
        the path, and the run says so - the same two-state outcome the
        single-structure loop reports (`converged`, or the step-bound message a
        qualification runner reads as PASS[NOT CONVERGED]).
        """
        bound = 'Maximum optimization steps reached'
        for optimizer in self.optimizers:
            if not optimizer.finished():
                return False
            if bound in optimizer.stop_reason:
                return False
        return True

    # ---- Walkers ----

    def _run_walker(self, state, optimizer):
        """Walk one path point at a time, in place on the calculator's state.

        The order is the Optimize loop's, because it is the same evaluation: the
        optimizer writes the probe into the state, the state is wrapped into its
        cell, a declared condition is landed on its manifold, the backend is
        asked, and the optimizer reads the answer back. What differs is that the
        probe is not the state's own geometry plus a step - it is a geometry on a
        sphere the optimizer is holding - and that the answer may be a probe
        inside the same path point.
        """
        direction = getattr(optimizer, 'direction', None)
        optimizer.begin(state)
        while not optimizer.finished():
            if optimizer.next_trial() is None:
                break
            optimizer.apply_trial(state)
            state = apply_pbc(state)
            constraint_runtime.constrain_state(state)
            self.calculator.sync()
            result = self.calculator.run(target_states=self.target_states)
            state = self.calculator.get_state()
            state.record_step_result()
            optimizer.note_evaluation(state)
            accepted = len(optimizer.points)
            optimizer.observe(state)
            self._write_status_row(optimizer, direction)
            accepted_now = len(optimizer.points) > accepted
            if accepted_now:
                self._write_frame(state, optimizer, direction)
                state.runtime.advance_step()
            self._log_path_step(
                state, optimizer,
                optimizer.report().get('step', 0) or 1,
                write_structure=accepted_now)
        return state

    # ---- The band ----

    def _run_band(self, state):
        """Relax a band, one round of every image at a time."""
        optimizer = self.optimizers[0]
        chain, cells, roles = self._prepare_band(state)
        optimizer.set_band(*self._band_joint(chain, cells), roles=roles)
        optimizer.begin(state)
        self.band_roles = roles
        if self.local_executor is None:
            from .hessian import _child_entry
            self.local_executor = _child_entry
        for round_index in range(self.maxit):
            units = optimizer.band_batch(state)
            results = self.dispatch(units=units, child_jobtype=self.child_jobtype)
            optimizer.band_collect(state, results)
            self._write_band_row(optimizer)
            self._write_band_frames(state, optimizer)
            self._log_path_step(state, optimizer, round_index + 1)
            state.runtime.advance_step()
            if optimizer.finished():
                break
        self._write_profile(self.optimizers)
        self._write_results(state, self.optimizers)
        return state

    def _prepare_band(self, state):
        """The band's chain, from the endpoint files the input named."""
        params = self.optimize_params
        atoms = [str(name) for name in np.asarray(state.atoms).reshape(-1)]
        reactant, reactant_cell = self._endpoint(params.get('opt_neb_reactant'),
                                                atoms)
        product, product_cell = self._endpoint(params.get('opt_neb_product'),
                                              atoms)
        saddle = None
        if str(params.get('opt_neb_ts') or '').strip():
            saddle, _ = self._endpoint(params.get('opt_neb_ts'), atoms)
        lattice = None
        if self.cell_variables:
            if reactant_cell is None or product_cell is None:
                raise ValueError(
                    'optimize_cell relaxes the box of every band image and this '
                    'band\'s endpoints do not both declare one: a variable-cell '
                    'band interpolates the cell between its endpoints, and a '
                    'missing one has no ramp to sit on. Write both endpoint '
                    'files with the Lattice="..." comment line the framework '
                    'writes, or keep the box fixed')
            lattice = [reactant_cell, product_cell]
        else:
            lattice = self._system_cell(state)
        rng = np.random.default_rng(self._seed(state))
        return build_band(
            atoms, reactant, product, int(params.get('opt_neb_images', 8)),
            scheme=str(params.get('opt_neb_interpolate', 'linear')),
            ts=saddle, lattice=lattice,
            scaler=float(params.get('opt_neb_geodesic_scaler', 1.7)),
            threshold=float(params.get('opt_neb_geodesic_threshold', 3.0)),
            min_neighbors=int(params.get('opt_neb_geodesic_min_neighbors', 4)),
            rng=rng)

    def _band_joint(self, chain, cells):
        """The band as joint vectors in Bohr, and its cells.

        The same two halves every other search steps on
        (`optimizer/base.py::_extend_to_cell`): the atomic coordinates, and the
        six relaxed lattice entries when the box is a variable - so a
        variable-cell band's force has one meaning and the curvature batch the
        framework can produce is the shape this vector is.
        """
        joint = np.asarray(chain, dtype=np.float64).reshape(len(chain), -1)
        joint = joint * self.coord_to_bohr
        if cells is None:
            return joint, None
        lattices = np.asarray(cells, dtype=np.float64) * self.coord_to_bohr
        if not self.cell_variables:
            return joint, lattices
        slots = np.asarray([lattice.reshape(-1)[list(CELL_COMPONENT_SLOTS)]
                            for lattice in lattices])
        return np.concatenate((joint, slots), axis=1), lattices

    def _endpoint(self, path, atoms):
        """One endpoint structure, and the cell its comment line declares."""
        name = str(path or '').strip()
        if not name:
            raise ValueError(
                'a band needs both of its endpoints and this run names an empty '
                'file')
        _symbols, coordinates, lattice = read_xyz_frame(
            fixture_path(name), atoms=atoms,
            lattice_required=self.cell_variables)
        return coordinates, lattice

    def _system_cell(self, state):
        """The system's own cell, for a band held in one box.

        An all-zero lattice is no lattice, which is the rule the variable-cell
        searches read a box by (`optimizer/base.py::_cell_lattice`): reading it
        as one would put every image of a non-periodic band through a fractional
        conversion of a singular matrix.
        """
        rows = getattr(getattr(state, 'pmd', None), 'system', None)
        lattice = getattr(rows, 'lattice_vector', None) if rows else None
        if lattice is None:
            return None
        array = np.asarray(lattice, dtype=np.float64).reshape(-1)
        if array.size != 9 or not np.any(array):
            return None
        return array.reshape(3, 3)

    def _seed(self, state):
        """The seed of the interpolation's own draws."""
        control = getattr(getattr(state, 'pmd', None), 'control', None)
        value = getattr(control, 'gl_seed', None) if control else None
        try:
            return int(value)
        except (TypeError, ValueError):
            return _DEFAULT_GL_SEED

    # ---- Curvature for a walker that needs it ----

    def _supply_initial_hessian(self, state):
        """Compute the force constants an IRC seeds its mode from.

        The Hessian computation runs with `analysis=False`, and the modes are
        read here instead: `workflow/hessian.py` refuses a vibration analysis of
        a variable-cell run outright (*"a box's curvature is read by the search
        that asked for it, not as a frequency"*), and what a variable-cell IRC
        wants is the atomic block's mode while the path still carries its box.
        The whole matrix is handed over as well, so a `relu` run gets its inner
        minimizer's starting curvature out of the same computation.
        """
        from .hessian import Hessian

        section = dict(getattr(state, 'hessian_params', None) or {})
        driver = section.get('hess_driver')
        mode = section.get('hess_mode')
        if mode is None:
            mode = 'analytic' if driver == 'skill' else 'numerical'
        curvature = Hessian(calculator=self.calculator, source=mode,
                            driver=driver, analysis=False)
        curvature.run()
        matrix = curvature.matrix
        if matrix is None:
            raise ValueError(
                'optimize.method irc needs the framework Hessian computation for '
                'the imaginary mode of its transition state and it returned no '
                'matrix: the displaced gradients were not available, so there is '
                'no curvature to read a mode from')
        for optimizer in self.optimizers:
            optimizer.set_curvature(np.asarray(matrix, dtype=np.float64))
        return matrix

    # ---- Records ----

    def _resolve_output_dir(self, state):
        """The directory this run's records belong to (the Optimize rule)."""
        return Optimize._resolve_output_dir(self, state)

    def _status_columns(self, optimizer):
        """The header of this kind of row, cell columns included."""
        if optimizer.path_kind == 'band':
            columns = list(PATH_BAND_COLUMNS)
        else:
            columns = list(PATH_WALKER_COLUMNS)
        if self.cell_variables:
            columns += list(PATH_CELL_COLUMNS)
        return tuple(columns)

    def _profile_columns(self):
        columns = list(PATH_PROFILE_COLUMNS)
        if self.cell_variables:
            columns.append(PATH_PROFILE_CELL_COLUMN)
        return tuple(columns)

    @staticmethod
    def _line(columns):
        return ' ' + ' '.join('%-*s' % (_PATH_FIELD_WIDTH, name)
                              for name in columns)

    @staticmethod
    def _values(step, columns, values):
        fields = ['%*d' % (_PATH_STEP_WIDTH, step)]
        for name in columns[1:]:
            value = values.get(name, 0.0)
            if isinstance(value, str):
                fields.append('%*s' % (_PATH_FIELD_WIDTH, value))
            else:
                fields.append('%*.*f' % (_PATH_FIELD_WIDTH, 10, float(value)))
        return ' '.join(fields)

    def _init_output_files(self):
        """Open the four records this run writes."""
        os.makedirs(self.output_dir, exist_ok=True)
        # One directory, one owner per plain name: the run logger and this
        # workflow both write here, and an existing geometry record is truncated
        # rather than appended to across runs.
        with open(self._output_path('path_geom.xyz'), 'w'):
            pass
        with open(self._output_path('path_profile.txt'), 'w') as handle:
            handle.write(self._line(self._profile_columns()) + chr(10))
        with open(self._output_path('path_status.txt'), 'w') as handle:
            handle.write(self._line(self._status_columns(self.optimizers[0]))
                         + chr(10))

    def _output_path(self, name):
        return os.path.join(self.output_dir, name)

    def _append(self, name, text):
        with open(self._output_path(name), 'a') as handle:
            handle.write(text + chr(10))

    def _write_status_row(self, optimizer, direction):
        """One walker evaluation, as a row of `path_status.txt`."""
        row = optimizer.report()
        values = dict(row)
        values['direction'] = direction or 'forward'
        columns = self._status_columns(optimizer)
        self._append('path_status.txt',
                     self._values(row.get('step', 0), columns, values))

    def _write_band_row(self, optimizer):
        """One band round, as a row of `path_status.txt`."""
        row = optimizer.report()
        values = dict(row)
        columns = self._status_columns(optimizer)
        self._append('path_status.txt',
                     self._values(row.get('step', 0), columns, values))

    def _write_profile(self, optimizers):
        """The energy profile, one row per accepted point or per image."""
        columns = self._profile_columns()
        for optimizer in optimizers:
            direction = getattr(optimizer, 'direction', None)
            previous_energy = None
            previous_arc = 0.0
            for point in optimizer.path_points():
                values = dict(point)
                values['direction'] = direction or 'forward'
                values['de'] = (0.0 if previous_energy is None
                                else point['energy'] - previous_energy)
                values['length'] = point['arc'] - previous_arc
                self._append('path_profile.txt',
                             self._values(point['index'] + 1, columns, values))
                previous_energy = point['energy']
                previous_arc = point['arc']

    def _write_frame(self, state, optimizer, direction):
        """One accepted path point, as a frame of the geometry records."""
        atoms = list(state.atoms)
        coordinates = np.asarray(state.coordinates, dtype=np.float64)
        point = optimizer.points[-1]
        comment = 'Point %d  Direction %s  E = %.10f%s' % (
            len(optimizer.points), direction or 'forward', point['energy'],
            lattice_comment(state.pmd.system.model_dump()))
        Optimize._write_xyz(self._output_path('path.xyz'), atoms, coordinates,
                            len(atoms), comment)
        Optimize._write_xyz(self._output_path('path_geom.xyz'), atoms,
                            coordinates, len(atoms), comment, append=True)
        if direction is not None:
            Optimize._write_xyz(
                self._output_path('path_%s.xyz' % direction),
                atoms, coordinates, len(atoms), comment, append=True)

    def _write_band_frames(self, state, optimizer):
        """The band as it stands: every image appended, and as `path.xyz`.

        Each frame carries **its own** box. A variable-cell band steps the six
        lattice components of every image, so the parent state's box is the
        reactant's for every frame, and writing that one for all of them is a
        record that disagrees with its own volume column: measured on
        `path_cell_cineb` in round `tmp_20261021_195813`, `path_status.txt` read
        1000.000 -> 1000.638 A^3 while every `Lattice=` row of `path_geom.xyz`
        read 1000.000. `tools/verify.py` section D's cross-check is what found
        it. A fixed-cell band's `lattice_of` is the snapshot it was built with,
        so the comment is byte-identical there.
        """
        atoms = list(state.atoms)
        system = state.pmd.system.model_dump()
        with open(self._output_path('path.xyz'), 'w'):
            pass
        for index in range(optimizer.count):
            frame = (optimizer.images[index][:optimizer._atomic_size]
                     .reshape(-1, 3) * self.disp_to_angstrom)
            lattice = optimizer.lattice_of(index)
            frame_system = system
            if lattice is not None:
                frame_system = dict(system)
                frame_system['lattice_vector'] = (
                    np.asarray(lattice, dtype=np.float64)
                    * optimizer.disp_to_angstrom).tolist()
            comment = 'Round %d  Image %d  Role %s  E = %.10f%s' % (
                optimizer.iteration, index, optimizer.roles[index],
                optimizer.energies[index], lattice_comment(frame_system))
            Optimize._write_xyz(self._output_path('path_geom.xyz'), atoms,
                                frame, len(atoms), comment, append=True)
            Optimize._write_xyz(self._output_path('path.xyz'), atoms, frame,
                                len(atoms), comment, append=True)

    def _write_results(self, state, optimizers):
        """The `path_*` block, into the state's own results."""
        merged = dict(state.result.to_dict()) if state.result else {}
        merged['path_method'] = self.method
        merged['path_converged'] = self._finished()
        merged['path_status'] = [dict(row) for optimizer in optimizers
                                 for row in optimizer.rows]
        for optimizer in optimizers:
            block = optimizer.results()
            merged.update(block)
            direction = getattr(optimizer, 'direction', None)
            if direction is not None and len(optimizers) > 1:
                # Two IRC branches: the shared keys above carry one side (the
                # last branch written), and each side also has its own block, so
                # a reader that wants both is not left with a list where it
                # expected a number.
                for key, value in block.items():
                    merged['%s_%s' % (key, direction)] = value
        self.calculator.update({'results': merged})


__all__ = ['Path', 'linear_path', 'build_band', 'fixture_path', 'PATH_WALKER_COLUMNS',
           'PATH_BAND_COLUMNS', 'PATH_PROFILE_COLUMNS']