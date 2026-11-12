"""
======================================================================
This module provides the SingleShot workflow for skill-driven jobs.

When control.driver is 'skill' or a skill name, the skill receives
the jobtype as-is and plans the whole job internally; the framework
executes exactly one calculator call and settles the outcome from
the skill-reported result:

    calculator.run()                     # single call
    record_step_result                   # standard bookkeeping
    finalize by jobtype                  # outputs + outcome verdict

The run-log result block is written by the pmd tail log_result
(non-iterative workflow convention), not by this workflow.

Finalization is selected dynamically from state.jobtype (the skill
owns the task semantics):
- 'optimize': writes opt.xyz/opt_geom.xyz from the final structure and
  settles from the skill's opt_converge flag. The optimizer progress
  table stays with the Optimize workflow: no framework step, no row.
- 'md': settles on the call having completed and writes a note naming the
  judgement it could not make (the skill-side md convergence flag is not
  defined yet, so a bare 'completed' would read as a checked verdict).
- everything else: plain mark_completed (single-shot semantics). The
  machine-learning jobtypes are not reached here any more: they have
  their own workflow classes (Train, Prediction) and their own shared
  body (MLSingleShot, defined in `workflow/train.py` with the two nodes
  that use it), which
  settle the outcome from the record the skill returned
  (records/ml.py). Co-locating that body here is file organisation
  only: MLSingleShot is not a SingleShot subclass, so a skill-driven
  ML run still gets SingleShot's plain completion rather than an ML
  verdict.

Aug 15 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from .base import Workflow
from .optimize import Optimize
from ..kernel.logger import get_logger

_LOGGER = get_logger(__name__)


class SingleShot(Workflow):
    """Skill-driven single-shot workflow.

    Runs the calculator once; the skill performs the whole job
    (energy/gradient/optimize/md/...) internally. The framework
    never iterates and never retries: the outcome is settled from
    the skill-reported result flags.

    Parameters
    ----------
    calculator : Calculator or None
        Calculator executing the single skill call.
    **kwargs
        Dispatcher parameters (state, multitask, remotetask,
        local_executor, executor_spec).
    """

    name = 'single'
    workflow_kind = 'single'

    @classmethod
    def prepare(cls, state):
        """Apply the ``&train`` permutation map when this call trains.

        A skill-driven training request resolves to this workflow rather
        than to Train, but the data preparation is the same framework-side
        step and still has to run before the backend is initialized.
        """
        if str(getattr(state, 'jobtype', '')).lower() == 'train':
            from .train import prepare_training_data
            prepare_training_data(state)

    def __init__(self, calculator=None, **kwargs):
        super().__init__(name='single', calculator=calculator,
                         **kwargs)

    def run(self):
        """Execute the calculator once and settle the outcome.

        Returns
        -------
        PyRAIState or Result
            The state when a calculator state exists, otherwise
            the raw result.
        """
        if self.local_executor is None:
            calculator = self.calculator

            def _run_calc(unit_state, runtime_idx=None):
                del unit_state, runtime_idx  # calculator owns the live state
                result = calculator.run()
                state = calculator.get_state()
                if state is None:
                    return result

                # Standard step bookkeeping (same as iterative
                # workflows): record the result. The run-log result
                # block is written once by the pmd tail log_result for
                # all non-iterative workflows (writing it here too
                # would duplicate the block).
                state.record_step_result()

                self._finalize(state, result)
                return state

            self.local_executor = _run_calc
        outcomes = self.dispatch()
        return outcomes[0]

    # ---- Jobtype-specific finalization ----

    def _finalize(self, state, result):
        """Settle the outcome by jobtype after the single call.

        Parameters
        ----------
        state : PyRAIState
            State with the skill-updated system (fin_coord already
            consumed by the Calculator).
        result : dict or None
            The skill computation result.
        """
        jobtype = state.jobtype
        if jobtype == 'optimize':
            self._finalize_optimize(state, result)
        elif jobtype == 'md':
            self._finalize_md(state, result)
        else:
            # Plain single-shot jobs (energy/gradient/...): the
            # framework has no verdict beyond a completed call. The
            # machine-learning jobtypes are no longer here -- they have
            # their own workflow classes (workflow/train.py,
            # workflow/prediction.py), which settle from the record the
            # skill returned.
            if getattr(state, 'runtime', None) is not None:
                state.runtime.mark_completed()

    def _finalize_optimize(self, state, result):
        """Write optimization outputs and settle from opt_converge.

        The final structure is the skill-reported geometry already
        applied to the system (fin_coord channel). Gradient and
        displacement report entries are not applicable: the skill
        owns the iteration, so they are recorded as zeros.

        Parameters
        ----------
        state : PyRAIState
            State with the optimized structure.
        result : dict or None
            The skill computation result.
        """
        report = {}
        if result:
            energy = result.get('energy')
            if energy is not None:
                try:
                    report['Energy'] = float(np.asarray(
                        energy, dtype=np.float64).reshape(-1)[0])
                except (IndexError, TypeError, ValueError) as error:
                    _LOGGER.warning(
                        'Final optimization energy could not be read (%s); '
                        'the report entry is omitted', error)

        self._write_optimize_output(state, report)

        # Settle the outcome from the skill-reported convergence
        # flag and write the outcome banner to the run log.
        converged = not (result and result.get('opt_converge') is False)
        run_logger = getattr(state, 'logger', None)
        if run_logger is not None:
            run_logger.log_optimize_summary(
                converged, 1, report.get('Energy'),
                message='skill internal optimization did not converge',
                label='Skill-driven geometry optimization')

        if getattr(state, 'runtime', None) is not None:
            if not converged:
                state.runtime.mark_failed(
                    'Skill internal optimization did not converge')
            else:
                state.runtime.mark_completed()

    def _finalize_md(self, state, result):
        """Settle a skill-driven md run, and name what was not judged.

        The skill-side md convergence flag is not defined yet, so this path
        has no verdict to make -- and a run log that ends in a bare
        `completed` reads as if something had been checked. The note written
        here says which judgement did not run, the same policy the
        qualification runner follows when it prints `not judged` instead of a
        verdict-free pass (dev/QUALIFICATION_DEVELOPMENT.md section 6.6).
        Wire the skill-specific flag here once it is defined.

        Parameters
        ----------
        state : PyRAIState
            State after the md call.
        result : dict or None
            The skill computation result.
        """
        del result  # no verdict flag defined yet
        run_logger = getattr(state, 'logger', None)
        if run_logger is not None:
            run_logger.write(
                'NOTE md verdict not judged: this run was driven inside the '
                'skill, which reports no convergence flag, so the framework '
                'settles only on the call having completed.')
        if getattr(state, 'runtime', None) is not None:
            state.runtime.mark_completed()

    # ---- Optimization output helpers (shared with Optimize) ----

    def _write_optimize_output(self, state, report):
        """Write the final structure of an engine-internal optimization.

        ``opt.xyz`` and ``opt_geom.xyz`` describe the structure the engine
        handed back, so they are real products of this path. The
        optimizer's progress table is not: the iteration happened inside
        the engine (LAMMPS ``minimize``, ORCA ``Opt`` ...), the framework
        neither took a step nor measured a gradient, and the convergence
        verdict is already reported to the run log by
        ``log_optimize_summary``. ``opt_status.txt`` therefore stays what
        it names -- the ``Optimize`` workflow's own per-step record -- and
        is not written here.

        Parameters
        ----------
        state : PyRAIState
            State with the final structure.
        report : dict
            Step report; only its ``Energy`` is recorded.
        """
        atoms = state.atoms
        coordinates = state.coordinates
        if not atoms or not coordinates:
            return

        coords = np.asarray(coordinates, dtype=np.float64)
        natom = len(atoms)
        energy = report.get('Energy', 0.0)
        title = 'Step 1  E = %.10f' % energy

        # Latest structure (opt.xyz) and trajectory (opt_geom.xyz)
        Optimize._write_xyz('opt.xyz', atoms, coords, natom, title)
        Optimize._write_xyz('opt_geom.xyz', atoms, coords, natom,
                           title, append=True)

    def __repr__(self):
        return "SingleShot(skill-driven, jobtype settles outcome)"


