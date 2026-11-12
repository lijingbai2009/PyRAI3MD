"""
======================================================================
This module provides the logging facilities for PyRAI3MD.

get_logger configures console and optional file output with a
consistent format used across all modules.

RunLogger is the run-log master of the framework: it writes the
<title>.log file named after &control title, records the start time
on the first line and the end time, total wall time and final status
on the last line, prints the program logo, the input parameters from
the schema sections, the loaded skills with their calculation
parameters, and forwards the per-step / final result blocks produced
by utilities/print_results.py plus the run-level `&optimization setup`
block, the per-step `&optimization progress` block (common convergence
rows plus method-specific details) and the outcome banner. Screen output is controlled by
control.silent (default silent: file only); the result-block
content level follows control.verbose (missing-property 'Not
computed' sections appear from verbose 3 upward).

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import logging
import os
import sys
import time

import numpy as np

from ..utilities.logo import print_logo
from ..utilities.timer import what_is_time, how_long
from ..utilities.structure_formatter import print_coord, print_charges
from ..utilities.print_results import (
    format_constraint_geometry, format_ml_report, format_result_block,
    snapshot_centers,
)

# What a non-zero md stop code means, in the legacy early-stop vocabulary.
# The run-log tail prints the code as the line before the end line, so a log
# that ended early says why without the reader having to find the step block.
STOP_CODE_MEANINGS = {
    1: 'ill state population',
    2: 'dissociation: tracked geometry bound crossed',
    3: 'prediction error above threshold',
    4: 'calculation failed',
}


def stop_code_note(code):
    """The run-log sentence for one md stop code ('' for the normal code)."""
    try:
        value = int(code or 0)
    except (TypeError, ValueError):
        return ''
    if value == 0:
        return ''
    return 'MD stop code: %d (%s)' % (
        value, STOP_CODE_MEANINGS.get(value, 'unknown'))


def _result_centers(state):
    """Per-center rows for a final (non-step) multi-center result block.

    ``log_step`` reads them from the history snapshot, which carries the
    electronic outcome and populations. A single-point or optimizer result
    has no such snapshot, but the plan and the assembled active
    configuration are enough for the per-center state and the region
    charge / spin block; populations stay empty and the population section
    is skipped.
    """
    plan = getattr(state, 'partition_plan', None)
    if plan is None or int(getattr(plan, 'n_center', 1) or 1) <= 1:
        return None
    pmd = getattr(state, 'pmd', None)
    results = getattr(pmd, 'results', None) or {}
    active = results.get('active_configuration') or []
    electronic = getattr(state, 'electronic', None) or []
    centers = []
    for index in range(int(plan.n_center)):
        spec = plan.center(index)
        try:
            active_state = int(active[index])
        except (TypeError, IndexError, ValueError):
            active_state = 0
        carrier = electronic[index] if index < len(electronic) else {}
        spec_electronic = getattr(spec, 'electronic_spec', None)
        centers.append({
            'active': active_state,
            'population': [],
            'charge': int(getattr(spec, 'charge', 0) or 0),
            'multiplicity': int(getattr(spec, 'multiplicity', 1) or 1),
            'high_indices': [int(value) for value in
                             (getattr(spec, 'high_indices', ()) or ())],
            'mid_indices': [int(value) for value in
                            (getattr(spec, 'mid_indices', ()) or ())],
            'nac': carrier.get('nac'),
            'soc': carrier.get('soc'),
            'nac_pair_map': list(
                getattr(spec_electronic, 'nac_pair_map', ()) or ()),
            'soc_pair_map': list(
                getattr(spec_electronic, 'soc_pair_map', ()) or ()),
        })
    return centers


# Default log format
_LOG_FORMAT = '[%(asctime)s] %(name)s %(levelname)s: %(message)s'
_LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def get_logger(name, level=logging.INFO, log_file=None):
    """Return a configured logger with console and optional file output.

    Parameters
    ----------
    name : str
        Logger name, typically the module name.
    level : int
        Logging level (default: logging.INFO).
    log_file : str or None
        Path to a log file. If provided, log messages will also be
        written to this file.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger


# Extra input sections printed after control/system/environment,
# selected by the leading jobtype token.
JOBTYPE_SECTIONS = {
    'md': ['dynamics'],
    'adaptive': ['dynamics', 'adaptive'],
    'optimize': ['optimize'],
    # The grid a scan is about to run is part of what the run declared,
    # so the heading echoes it the way &dynamics is echoed for a md run.
    'search': ['search'],
}

# Human-readable titles for the calculation section banner, keyed by
# the jobtype token rather than the workflow class: a jobtype is what
# the run does, while the class only says who executes it (a
# skill-driven 'optimize' and a framework-side 'optimize' are the same
# job). Keys must stay a subset of the JOBTYPE_MAP tokens; the
# verification suite guards that.
WORKFLOW_TITLES = {
    'optimize': 'Geometry Optimization',
    'md': 'Nonadiabatic Molecular Dynamics',
    'surface_hopping': 'Nonadiabatic Molecular Dynamics',
    'frequency': 'Vibrational Frequency Analysis',
    'scan': 'Potential Energy Surface Scan',
    'adaptive': 'Adaptive Sampling',
    'search': 'Hyperparameter Grid Search',
    'train': 'Model Training',
    'prediction': 'Model Prediction',
}

# System fields rendered as aligned tables instead of raw values.
_TABLE_FIELDS = {
    'coordinates': 'coord',
    'velocities': 'coord',
    'target_coord': 'coord',
    'point_charges': 'charge',
}

# How many atom numbers one line of the constraint plan is allowed to carry.
# A wall on thirty atoms prints its count, not its list: the numbers are in the
# input, and the point of this block is which entity holds which scope.
_CONSTRAINT_ATOMS_SHOWN = 6


def _count_indices(indices):
    """`n atoms`, or the indices themselves while there are few of them."""
    rows = list(np.asarray(indices, dtype=int).reshape(-1))
    word = 'atom' if len(rows) == 1 else 'atoms'
    if len(rows) > _CONSTRAINT_ATOMS_SHOWN:
        return '%d %s' % (len(rows), word)
    return '%d %s (%s)' % (len(rows), word,
                           ' '.join(str(int(row) + 1) for row in rows))


def _constraint_wall(spec):
    """How one wall is placed and how many objects it confines, in one phrase.

    The three choices a wall input makes are independent, so the record names all
    three: which law, what each object's center is weighted by, and where the
    center came from. `selection` is the case that used to read as the coordinate
    origin, which is the one thing a reader of the log cannot guess from a wall
    that says nothing about a center.
    """
    parameters = spec.parameters or {}
    target = parameters.get('target_kind', 'atom')
    groups = parameters.get('molecules')
    if groups:
        objects = len(groups)
    elif target == 'atom':
        objects = len(np.asarray(spec.atoms if spec.atoms is not None else [],
                                 dtype=np.int64).reshape(-1))
    else:
        objects = 1
    center = ('declared' if parameters.get('center') is not None
              else 'wall_ref' if parameters.get('anchor') is not None
              else 'selection')
    return ('law %s, target %s, %d object%s, center from %s'
            % (parameters.get('law', 'hard'), target, objects,
               '' if objects == 1 else 's', center))


def _constraint_scope(spec):
    """What one action acts on, in the words the input used to say it."""
    if spec.cv:
        return 'cv %s' % spec.cv
    if spec.group:
        return 'group %s' % spec.group
    if spec.pairs is not None:
        return '%d rows' % len(np.asarray(spec.pairs, dtype=int))
    if spec.atoms is not None and len(np.asarray(spec.atoms).reshape(-1)):
        return _count_indices(spec.atoms)
    return ''


class RunLogger:
    """Run-log master for one PyRAI3MD calculation.

    Opens <title>.log in the given directory (current directory by
    default) and orchestrates all log content: start/end lines,
    program header, input parameter sections, skill information and
    the calculation result blocks. Every formatted block is written
    to the file once; screen output happens only when control.silent
    is 0 (debug mode).

    Parameters
    ----------
    state : PyRAIState
        Fully initialized state; title and silent flag come from
        state.pmd.control.
    log_dir : str or None
        Directory for the log file (default: current directory).
    start_time : float or None
        Wall-clock start of the whole run (time.time()); defaults to
        the construction time when omitted.
    log_name : str or None
        File stem of the log (default: the ``&control title``). A
        dispatched stage whose runtime directory carries a product name
        (a fit's ``NN-<title>-round-<k>``) passes that name so the log
        and the directory agree.

    Attributes
    ----------
    title : str
        Calculation title (&control title).
    log_file : str
        Absolute path of the log file.
    start_time : float
        Start timestamp in seconds (defaults to construction time;
        callers pass the run() entry timestamp so the reported wall
        time covers input parsing and environment initialization).
    """

    def __init__(self, state, log_dir=None, start_time=None,
                 log_name=None):
        pmd = state.pmd
        self.title = pmd.control.title
        self.silent = int(pmd.control.silent)
        self.verbose = int(pmd.control.verbose)
        self.log_dir = log_dir or os.getcwd()
        self.log_file = os.path.join(
            self.log_dir, '%s.log' % (log_name or self.title))
        self.start_time = (float(start_time)
                           if start_time is not None else time.time())
        self._handler = self._open_log()
        self._closed = False

    def _open_log(self):
        """Open the run log, creating its directory first.

        ``&environment scratch_path`` names where the log goes (#306),
        and that directory need not exist yet. Opening it blind fails
        with FileNotFoundError, and inside a dispatched child the parent
        only reports the missing result file, so the real cause stays in
        runtime.log.
        """
        os.makedirs(self.log_dir, exist_ok=True)
        return open(self.log_file, 'w')

    def __deepcopy__(self, memo):
        """Create a detached copy for dispatched child states.

        The file handle is not copied (it is not picklable and the
        child must never write into the parent log file); the clone
        reopens its own log file lazily on the first write.

        Returns
        -------
        RunLogger
            Detached copy sharing only the plain attributes.
        """
        clone = RunLogger.__new__(RunLogger)
        clone.title = self.title
        clone.silent = self.silent
        clone.verbose = self.verbose
        clone.log_dir = self.log_dir
        clone.log_file = self.log_file
        clone.start_time = self.start_time
        clone._handler = None
        clone._closed = True
        return clone

    def __getstate__(self):
        """Drop the live file handle for pickling.

        Symmetric with __deepcopy__: a dispatched child state travels
        back to the parent through the multiprocessing pool, and an open
        file handle cannot be pickled. Only the plain attributes travel;
        the handle reopens lazily on the first write, and a log the child
        already finished stays finished (its '_closed' flag travels). A
        state that travelled at all is a child, so '_detached' is set:
        its first write appends to the log rather than truncating it.

        Returns
        -------
        dict
            Picklable state dict, without '_handler'.
        """
        state = dict(self.__dict__)
        state.pop('_handler', None)
        return state

    def __setstate__(self, state):
        """Restore a pickled logger with no handle.

        Parameters
        ----------
        state : dict
            State dict produced by __getstate__().
        """
        state['_closed'] = True
        self.__dict__.update(state)
        self._handler = None

    # ---- Low-level output ----

    def write(self, text):
        """Write a text block to the log file and optionally to screen.

        Parameters
        ----------
        text : str
            Formatted text block.
        """
        if self._closed:
            return
        if self._handler is None:
            self._handler = self._open_log()
        self._handler.write(text)
        self._handler.flush()
        if self.silent == 0:
            print(text, end='', flush=True)

    # ---- Run lifecycle ----

    def write_start_line(self):
        """Write the first line with the run start time."""
        self.write('PyRAI3MD Start: %s\n' % what_is_time())

    def finish(self, status='completed', note=''):
        """Write the last line (end time, wall time, status) and close.

        Parameters
        ----------
        status : str
            Final status text, e.g. 'completed' or 'error: ...'.
        note : str
            Optional sentence written immediately above the end line, so it
            is the second-to-last line of the log (the md stop code, see
            `stop_code_note`). Empty writes nothing and keeps the old shape.
        """
        if self._closed:
            return
        walltime = how_long(self.start_time, time.time())
        head = '\n%s\n' % note if note else '\n'
        self.write('%sPyRAI3MD End: %s Total: %s Status: %s\n' % (
            head, what_is_time(), walltime, status))
        if self._handler is not None:
            self._handler.close()
        self._closed = True

    # ---- Header content ----

    def write_heading(self, state):
        """Write the program logo, input parameters and skill info.

        Parameters
        ----------
        state : PyRAIState
            State with the validated pmd config and the task queue.
        """
        self.write(print_logo())

        # Input parameters from the schema sections
        self._write_input_section(state, 'control')
        self._write_input_section(state, 'system')
        self._write_input_section(state, 'environment')

        jobtype = str(state.jobtype or '')
        for token in jobtype.split():
            extra_sections = JOBTYPE_SECTIONS.get(token)
            if extra_sections:
                for section in extra_sections:
                    self._write_input_section(state, section)
                break

        # What the declared constraints became, and what filled them in.
        self._write_constraint_plan(state)

        # Loaded skills and their calculation parameters
        self._write_skill_section(state)

    def _write_input_section(self, state, name):
        """Write one schema section of the input parameters.

        Parameters
        ----------
        state : PyRAIState
            State holding the pmd config.
        name : str
            Submodel name of state.pmd (control, system, ...).
        """
        submodel = getattr(state.pmd, name, None)
        if submodel is None or not hasattr(submodel, 'model_dump'):
            return

        atoms = getattr(state.pmd.system, 'atoms', [])
        text = '\n  &%s\n%s\n' % (name, '-' * 79)
        for key, value in submodel.model_dump().items():
            table_kind = _TABLE_FIELDS.get(key)
            if table_kind == 'coord' and value and len(value) == len(atoms):
                text += '  %-24s\n%s' % (key, print_coord(atoms, value))
            elif table_kind == 'charge' and value:
                text += '  %-24s\n%s' % (key, print_charges(value))
            else:
                text += '  %-24s %s\n' % (key, value)

        self.write(text)

    def _write_constraint_plan(self, state):
        """Write one line per declared constraint, then what the input left out.

        The `&system` section above prints the block as it was written, ids and
        all blank. This is the other half: what the run actually holds - the
        entity the schema minted from each object, and the notes, which are every
        value the reader broadcast or padded, every wall keyword `wall_type` did
        not read, and every ramp whose stop the run's own step count closed. A
        declaration can reach the run changed, so the change belongs in the
        record of the run rather than only in the input that caused it.

        A run with no constraints writes nothing here: the section is not an
        empty heading, it is absent.
        """
        plan = state.constraint_plan()
        if plan is None or plan.is_empty:
            return
        units = plan.units or {}
        text = '\n  &constraint plan\n%s\n' % ('-' * 79)
        # Whose constraint engine is in force, said once per run: the other
        # party's keywords are inert by the ownership rule, and that is the
        # first thing to look at when a held quantity is not being held.
        text += ("  ownership          framework (control.driver=%r); skill-side "
                 "constraint\n                   keywords are ignored this run\n"
                 % getattr(state, 'driver', 'main'))
        text += ('  input units %s length, %s angle, %s energy;'
                 ' the numbers below are internal (bohr, radian, hartree)\n'
                 % (units.get('length', 'angstrom'),
                    units.get('angle', 'degree'),
                    units.get('energy', 'hartree')))
        for spec in plan.actions:
            ramps = list(spec.all_schedules())
            fields = ['%-10s' % spec.kind, '%-12s' % spec.family,
                      '%-14s' % spec.id, _constraint_scope(spec)]
            if spec.kind == 'wall':
                fields.append(_constraint_wall(spec))
            if ramps:
                fields.append('ramp %s' % ', '.join(
                    '%s(%d-%d)' % (record.parameter, record.start_step,
                                   record.end_step) for record in ramps))
            text += '  %s\n' % ' '.join(item for item in fields if item)
        for name in plan.groups.names():
            text += '  group      %-14s %s\n' % (
                name, _count_indices(plan.groups.members(name)))
        for name, record in sorted(plan.cvs.items()):
            text += '  cv         %-14s %-10s %s\n' % (
                name, record.kind, _count_indices(record.indices))
        if plan.notes:
            text += '\n  notes:\n'
            for note in plan.notes:
                text += '    %s\n' % note
        self.write(text)

    def _write_skill_section(self, state):
        """Write the loaded skill information and calculation parameters.

        Parameters
        ----------
        state : PyRAIState
            State with the registered task queue.
        """
        text = '\n  &skills\n%s\n' % ('-' * 79)
        tasks = list(state.iter_tasks())
        if not tasks:
            text += '  No skill loaded\n'
            self.write(text)
            return

        for category, task in tasks:
            capabilities = ', '.join(task.get('capabilities', []))
            text += (
                '  category     %s\n'
                '  role         %s\n'
                '  task id      %s\n'
                '  skill        %s\n'
                '  api          %s\n'
                '  capabilities %s\n' % (
                    category,
                    task.get('role'),
                    task.get('task_id'),
                    task.get('skill_name'),
                    task.get('api'),
                    capabilities,
                )
            )
            config = task.get('config') or {}
            skill_config = config.get('skill_config')
            if isinstance(skill_config, dict) and skill_config:
                text += '\n  calculation parameters:\n'
                for key, value in skill_config.items():
                    text += '    %-24s %s\n' % (key, value)
            text += '\n'

        self.write(text)

    # ---- Calculation content ----

    def log_workflow_title(self, workflow_cls, jobtype=''):
        """Write the calculation section banner for the jobtype.

        Parameters
        ----------
        workflow_cls : type or None
            Resolved workflow class; None for single-shot
            calculations. Only used to tell a framework-side
            single-point job from a skill-driven single shot.
        jobtype : str
            The run jobtype; its first known token names the banner.
        """
        title = None
        for token in str(jobtype or '').split():
            if token in WORKFLOW_TITLES:
                title = WORKFLOW_TITLES[token]
                break
        if title is None:
            title = ('Single-point Calculation' if workflow_cls is None
                     else 'Single-shot Calculation')

        banner = (
            '\n *---------------------------------------------------*\n'
            ' |%s|\n'
            ' *---------------------------------------------------*\n\n'
            % title.center(51)
        )
        self.write(banner)

    def log_step(self, step, snapshot, thresholds=None):
        """Write one iteration step result block.

        Parameters
        ----------
        step : int
            Iteration step number (1-based).
        snapshot : dict
            Full pmd model dump of the recorded step (typically the
            last history snapshot).
        thresholds : dict or None
            Deviation thresholds keyed by the err_* result keys;
            forwarded to the result block formatter so the
            prediction error section carries the exceeded TRUE/FALSE
            judgment (iterative dynamics only).
        """
        snapshot = snapshot or {}
        block = format_result_block(
            snapshot.get('system'), snapshot.get('results'),
            step=step, runtime=snapshot.get('runtime'),
            thresholds=thresholds, verbose=self.verbose,
            centers=snapshot_centers(snapshot),
            dynamics=snapshot.get('dynamics'))
        self.write(block)

    def log_result(self, state, include_system=True):
        """Write the final result block for non-iterative workflows.

        Parameters
        ----------
        state : PyRAIState
            State after execution with the final pmd results.
        include_system : bool
            Whether to print the system and property sections. A
            workflow that evaluates many structures (adaptive sampling)
            passes False: its parent-state geometry is not a result.
        """
        if state.pmd is None:
            return
        runtime = (state.runtime.to_dict()
                   if state.runtime is not None else None)
        system = state.pmd.system.model_dump()
        results = dict(state.pmd.results)
        block = format_result_block(
            system, results,
            runtime=runtime, verbose=self.verbose,
            centers=_result_centers(state),
            dynamics=state.pmd.dynamics.model_dump(),
            include_system=include_system,
            record_report=format_ml_report(results, system) or None)
        self.write(block)

    def log_optimize_setup(self, setup):
        """Write the run-level optimization setup block.

        One block per run, before the first step: what search is running,
        how long it may run, and the run-level conditions the per-step
        table does not repeat. The names deliberately avoid the
        `OPTIMIZE_TAIL_KEYS` the qualification reader scans for and the
        word `threshold`, which the step-block test counts.

        Parameters
        ----------
        setup : dict
            Ordered run-level values keyed by the labels below; missing
            entries are skipped.
        """
        setup = setup or {}
        order = ('Method', 'Optimizer', 'Max steps', 'Cell mode',
                 'Cell coupling', 'Pressure target', 'Band size',
                 'Spring', 'Climbing', 'IRC direction', 'IRC mode',
                 'IRC imaginary count', 'IRC imaginary frequency',
                 'Constraint conditions', 'Initial volume')
        body = ''
        for key in order:
            value = setup.get(key)
            if value is None:
                continue
            if isinstance(value, bool):
                text = 'True' if value else 'False'
            elif isinstance(value, int):
                text = str(int(value))
            elif isinstance(value, str):
                text = value
            else:
                text = '%.8f' % float(value)
            body += '  %-22s %s\n' % (key + ':', text)
        if not body:
            return
        self.write('\n  &optimization setup\n%s\n%s' % ('-' * 79, body))

    def log_optimize_step(self, step, report, converged,
                          thresholds=None, details=None):
        """Write one geometry-optimization progress block.

        All numeric entries are printed with 8 decimals,
        right-aligned. Entries carrying a convergence threshold
        additionally show the threshold and whether the criterion is
        met (True/False) on the same line. A report that carries the
        `Constraint` record of a constrained run gets the
        `&constraint geometry` section after the judgment lines, in the
        same shape as the `&constraint energies` section of a dynamics
        step block.

        Parameters
        ----------
        step : int
            Optimization step number (1-based).
        report : dict
            Convergence report from the optimizer (Energy, Delta E,
            Gradient RMS/Max, Displacement RMS/Max, Trust Radius, and
            Constraint for a run that declares conditions).
        converged : bool
            Whether the convergence criteria are met at this step.
        thresholds : dict or None
            Convergence thresholds keyed by the report entry they
            judge: 'Delta E', 'Gradient RMS', 'Gradient Max',
            'Displacement RMS' and 'Displacement Max'.
        details : dict or None
            Ordered method-specific readings (TS curvature, crossing
            norms, path arc/image, ...). A value of None prints
            `Not measured`; strings and bools print as themselves; a
            key already printed from `report` is skipped.
        """
        report = report or {}
        thresholds = thresholds or {}
        details = details or {}
        # A two-state run's gap row belongs beside the energies it is read
        # from, and it carries no threshold here: which gap counts as closed is
        # the crossing optimizer's own criterion, not this table's. A cell run's
        # rows come next: what the lattice half of the step is worth as a
        # pressure and as a strain (the pair it is judged on), the same readings
        # in atomic units that `opt_status.txt` has always carried, and the box
        # they belong to.
        gaps = sorted(key for key in report
                      if str(key).startswith('Gap ('))
        cells = [key for key in ('External Pressure',
                                 'Cell Stress RMS', 'Cell Stress Max',
                                 'Cell Strain RMS', 'Cell Strain Max',
                                 'Cell Gradient RMS', 'Cell Gradient Max',
                                 'Cell Displacement RMS',
                                 'Cell Displacement Max', 'Cell Volume',
                                 'Frozen Ride Max')
                 if key in report]
        # Rows a saddle or crossing search adds to the judgment: how many
        # downhill directions the curvature model has, and how far apart the two
        # surfaces still are.
        search = [key for key in ('Crossing Gap', 'Cell Gap Term',
                                  'Cell Gap Term Source',
                                  'Branching Directions',
                                  'Negative Modes (model)',
                                  'Mode Eigenvalue (model)',
                                  'Rigid Directions Removed')
                  if key in report]
        body = '  %-18s %20d\n' % ('Step:', step)
        printed = set()
        for key in (['Iteration', 'Energy', 'Delta E'] + gaps
                    + ['Gradient RMS', 'Gradient Max',
                       'Displacement RMS', 'Displacement Max']
                    + cells + search):
            value = report.get(key)
            if value is None:
                continue
            printed.add(key)
            if isinstance(value, str):
                # A source label, not a measurement: `Cell Gap Term` is the
                # number and this row says who produced it, so a zero that the
                # plan implies and a zero nobody answered cannot read the same.
                body += '  %-18s %20s\n' % (key + ':', value)
                continue
            line = '  %-18s %20.8f' % (key + ':', float(value))
            threshold = thresholds.get(key)
            if threshold is not None:
                threshold = float(threshold)
                if key == 'Delta E':
                    met = abs(float(value)) < threshold
                else:
                    met = float(value) < threshold
                line += '  %-10s %14.8f %8s' % (
                    'threshold', threshold,
                    'True' if met else 'False')
            body += line + '\n'
        # Method-specific readings the convergence report does not carry.
        for key, value in details.items():
            if key in printed:
                continue
            if value is None:
                body += '  %-18s %20s\n' % (key + ':', 'Not measured')
            elif isinstance(value, bool):
                body += '  %-18s %20s\n' % (
                    key + ':', 'True' if value else 'False')
            elif isinstance(value, int):
                body += '  %-18s %20d\n' % (key + ':', int(value))
            elif isinstance(value, str):
                body += '  %-18s %20s\n' % (key + ':', value)
            else:
                body += '  %-18s %20.8f\n' % (key + ':', float(value))
        value = report.get('Trust Radius')
        if value is not None:
            body += '  %-18s %20.8f\n' % ('Trust Radius:', float(value))
        body += '  %-18s %20s\n' % (
            'Converged:', 'True' if converged else 'False')
        # A constrained run also says what its conditions held and what they
        # absorbed: the gradient entries above are judged on the tangential
        # part, and these rows are the reading that was taken off it.
        body += format_constraint_geometry(report.get('Constraint'))
        self.write('\n  &optimization progress\n%s\n%s' % (
            '-' * 79, body))

    def log_optimize_summary(self, converged, steps, energy=None,
                             message='',
                             label='Geometry optimization'):
        """Write the geometry-optimization outcome banner.

        Parameters
        ----------
        converged : bool
            Whether the optimization converged.
        steps : int
            Number of optimization steps performed.
        energy : float or None
            Final energy reported at the last step.
        message : str
            Extra detail appended to the failure line (e.g. the
            failure reason recorded by the runtime).
        label : str
            Banner subject ('Geometry optimization' for the
            framework-driven loop, 'Skill-driven geometry
            optimization' for the SingleShot path).
        """
        unit = 'step' if steps == 1 else 'steps'
        if converged:
            text = '%s converged in %d %s' % (label, steps, unit)
            if energy is not None:
                text += '  (final energy = %.10f)' % float(energy)
        else:
            text = '%s did not converge after %d %s' % (
                label, steps, unit)
            if message:
                text += '  (%s)' % message
        self.write('\n  %s\n  %s\n\n' % ('=' * 79, text))

    def log_ml_summary(self, settled, detail='', message='',
                       label='Machine learning run'):
        """Write the machine-learning outcome banner.

        A training or prediction run reports a record rather than a
        converged geometry, so the banner states which record the run
        came back with instead of a step count. What is printed was
        returned by the skill: the counts and the wall clock come from
        the training or prediction record, and no threshold is applied
        to any of them.

        Parameters
        ----------
        settled : bool
            Whether the run came back with the record its jobtype is
            contracted to carry.
        detail : str
            Record summary read off the returned channels (ensembles,
            channels, wall clock, summary path); printed when the run
            is settled.
        message : str
            Failure reason naming the missing or empty channel.
        label : str
            Banner subject, taken from the jobtype ('Model training'
            or 'Model prediction').
        """
        if settled:
            text = '%s verdict: completed' % label
            if detail:
                text += '  (%s)' % detail
        else:
            text = '%s verdict: failed' % label
            if message:
                text += '  (%s)' % message
        self.write('\n  %s\n  %s\n\n' % ('=' * 79, text))
