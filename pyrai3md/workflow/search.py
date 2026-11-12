"""
======================================================================
This module provides the GridSearch workflow for hyperparameter scans.

A scan is a plan, a fan-out and a ranking. The plan comes from what the
skill declares searchable and the candidate values the run names in
&search; the fan-out is the Dispatcher's batch path, one materialized
child state per point, each of which trains through the single-shot
path exactly as a 'train' job would; the ranking reads the records those
trainings left and puts the smallest error first.

Nothing here trains a network, evaluates one, or decides that a number
is acceptable. Which folder is complete, what a candidate value means
and what a fit error is measured in all belong to the skill; the scan
asks, and compares what it is told.

The workflow used to be a structural candidate search under this name.
That job is not this one and is reserved for a token of its own: a
hyperparameter scan varies the training, a structure search varies the
geometry, and the two have nothing in common but the word.

Oct 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .base import Workflow
from .grid import (build_context, build_row, first_skill_name, folder_complete,
                   format_plan, format_summary, plan_points, point_model_dir,
                   point_failure, point_record, rank_points, skill_surface,
                   skipped_row, write_point, write_crashed, write_csv)
# The scan's child entry is named by module and function so a worker
# process can resolve it without unpickling the parent.
_EXECUTOR_MODULE = 'pyrai3md.workflow.search'


def _child_entry(child_state, runtime_idx=None):
    """Module-level local execution entry for one grid point.

    It has to be reachable by name: the serial arm calls it as the
    workflow's local executor while the pooled arm resolves it from an
    importable spec, and a bound method would drag the workflow, its
    calculator and the skill's live backend into the worker's payload.

    Parameters
    ----------
    child_state : PyRAIState
        Materialized child state of one point.
    runtime_idx : str or None
        Runtime identity suffix of the point's directory.

    Returns
    -------
    object
        The child state, or a failure marker naming what went wrong.
    """
    from . import execute
    try:
        return execute(child_state, runtime_idx=runtime_idx)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        return {'grid_status': 'error', 'grid_error': str(error)}


class GridSearch(Workflow):
    """Hyperparameter grid scan workflow.

    Batch structure: one task per grid point. Each task is a materialized
    child state whose skill configuration carries that point's values,
    its own model directory, and the training profile the skill declares
    for the method being scanned.

    Parameters
    ----------
    calculator : Calculator or None
        Calculator owning the parent state.
    **kwargs
        Dispatcher parameters (state, multitask, remotetask, multipool,
        local_executor, executor_spec).
    """

    name = 'search'
    workflow_kind = 'batch'

    # Every point runs one training through the single-shot path.
    child_jobtype = 'train'

    def __init__(self, calculator=None, **kwargs):
        super().__init__(name='search', calculator=calculator,
                         executor_spec=(_EXECUTOR_MODULE, '_child_entry'),
                         **kwargs)
        self._context = None

    # ---- Scan plan ----

    def plan(self, state):
        """Build the points of the scan and the context that stamps them.

        Parameters
        ----------
        state : PyRAIState
            Parent state of the scan.

        Returns
        -------
        tuple(list of dict, dict, dict)
            (points, context, section). 'section' is the &search section
            as a plain dict.
        """
        section_model = getattr(getattr(state, 'pmd', None), 'search', None)
        section = (section_model.model_dump() if section_model is not None
                   else {})
        surface = skill_surface(first_skill_name(state))
        context = build_context(state, surface, section)
        points = plan_points(section.get('grid') or {},
                             surface['searchable'], surface['query'])
        limit = int(section.get('max_grid') or 0)
        if limit and len(points) > limit:
            raise ValueError(
                "the grid expands to %d points, over the max_grid of %d "
                "this run declared; narrow the candidate lists"
                % (len(points), limit))
        if not str(section.get('sort_by') or ''):
            section['sort_by'] = context['channels'][0]
        return points, context, section

    # ---- Execution ----

    def run(self):
        """Execute the scan: plan, fan out, read back, rank, report.

        The units are materialized and stamped here rather than handed
        to dispatch() as raw task descriptors: dispatch() only forwards
        coordinates and velocities, and a grid point is a configuration,
        not a geometry. Units that arrive already materialized pass
        through the pipeline untouched, so the per-point skill
        configuration survives both the serial and the pooled arm.

        Returns
        -------
        PyRAIState
            The parent state, carrying the ranked scan in its results.
        """
        state = self.calculator.get_state()
        points, context, section = self.plan(state)
        self._context = context
        keywords = sorted(section.get('grid') or {})
        self._log(format_plan(points, keywords))

        retrieve = int(section.get('retrieve') or 0) != 0
        rows = []
        units = []
        queued = []
        for index, point in enumerate(points):
            directory = point_model_dir(point, context)
            if retrieve:
                # Read what a previous scan left; submit nothing.
                rows.append(build_row(point, context, None, directory))
            elif folder_complete(context, directory):
                rows.append(skipped_row(point, context, directory))
            else:
                unit = self.materialize_unit({}, index,
                                             child_jobtype=self.child_jobtype)
                write_point(unit['state'], point, context)
                units.append(unit)
                queued.append((point, directory))

        outcomes = []
        stopped = ''
        if units:
            if self.local_executor is None:
                self.local_executor = _child_entry
            try:
                outcomes = self.dispatch(units=units,
                                         child_jobtype=self.child_jobtype)
            except Exception as error:
                # A point that dies in the parent -- the submission itself
                # failed, or the child reached a terminal state without
                # writing its result -- propagates out of `dispatch` before
                # this loop runs, and the scan then writes no table at all:
                # the fits that landed on disk lose their place in the
                # ranking, which is the scan's only product. The table is
                # what a scan is for, so the death becomes a row's reason
                # rather than the end of the run.
                stopped = 'batch dispatch stopped (%s: %s)' % (
                    type(error).__name__, error)
        for position, (point, directory) in enumerate(queued):
            outcome = outcomes[position] if position < len(outcomes) else None
            # `build_row` reads the folder whenever the outcome carries no
            # numbers, so a point whose fit landed keeps its row whatever
            # happened to the rest of the batch.
            rows.append(build_row(point, context, point_record(outcome),
                                  directory,
                                  point_failure(outcome) if outcome is not None
                                  else stopped))

        self.postprocess(state, rows, context, section)
        return state

    # ---- Runtime layout ----

    def runtime_root(self):
        """Put the scan's runtime directories under its grid root.

        One tree per scan: the numbered scratch of the points and the
        key-named model folders sit side by side instead of the run
        directories landing in the caller's working directory.

        Returns
        -------
        str or None
            The grid root, or None before the plan ran.
        """
        if self._context is None:
            return None
        return self._context['model_root']

    def runtime_isolated_workdir(self):
        """Isolate each point inside its own runtime directory.

        The training writes its summary and the skill writes its own log
        into the calculation directory; unisolated, a hundred points would
        append into one another's files.

        Returns
        -------
        bool
            True.
        """
        return True

    def _create_copy(self):
        """Copy the scan without the parent's calculator.

        A batch copy exists to carry one unit's state; the pool worker
        resolves the execution entry from the importable spec, never from
        the copy. The parent's Calculator, though, holds the skill's live
        backend, and a neural network backend is a set of built Keras
        models plus a compiled-function cache keyed by weak references,
        none of which survives being sent to a worker process. The scan
        never computes a term in the parent process, so dropping the
        calculator costs nothing it uses: each child builds its own from
        the state it arrives with.

        Returns
        -------
        GridSearch
            A copy with no calculator and no injected executor.
        """
        copy = super()._create_copy()
        copy.calculator = None
        copy.local_executor = None
        return copy

    # ---- Reporting ----

    def postprocess(self, state, rows, context, section):
        """Rank the points, write the scan's products, settle the run.

        Parameters
        ----------
        state : PyRAIState
            Parent state of the scan.
        rows : list of dict
            One row per point (ranked in place).
        context : dict
            Scan context.
        section : dict
            The &search section as a plain dict.
        """
        keywords = sorted(section.get('grid') or {})
        ranked = rank_points(rows, section['sort_by'],
                             section.get('sort_stat') or 'max',
                             section.get('sort_order') or 'asc')
        failed = [row for row in ranked if row['status'] == 'failed']
        skipped = [row for row in ranked if row['status'] == 'skipped']

        summary_path = write_csv(
            '%s/summary.csv' % context['model_root'], ranked, keywords,
            section['sort_by'])
        crashed_path = write_crashed(
            '%s/crashed.txt' % context['model_root'], ranked)

        self._log(format_summary(ranked, keywords, section['sort_by'],
                                 aliases=context.get('role_aliases')))
        merged = dict(state.result.to_dict()) if state.result else {}
        merged['grid_search'] = {
            'skill': context['label'],
            'profile': context['profile'],
            'nsearch': len(ranked),
            'nfailed': len(failed),
            'nskipped': len(skipped),
            'sort_by': section['sort_by'],
            'sort_stat': section.get('sort_stat') or 'max',
            'grid_root': context['model_root'],
            'summary_csv': summary_path,
            'crashed': crashed_path,
            'points': [_public(row) for row in ranked],
        }
        self.calculator.update({'results': merged})

        if getattr(state, 'runtime', None) is not None:
            if failed and len(failed) == len(ranked):
                state.runtime.mark_failed(
                    'all %d grid points produced no training record'
                    % len(ranked))
            else:
                state.runtime.mark_completed()

    def _log(self, block):
        """Write one scan table to the run log.

        Parameters
        ----------
        block : str
        The rendered table.
        """
        run_logger = getattr(self.calculator.get_state(), 'logger', None)
        if run_logger is not None and block:
            run_logger.write(block)

    def __repr__(self):
        return "GridSearch(hyperparameter scan, batch of train calls)"


def _public(row):
    """Reduce one scan row to the plain dict stored in the results."""
    return {
        'rank': row.get('rank'),
        'key': row['key'],
        'values': row['values'],
        'modeldir': row['modeldir'],
        'status': row['status'],
        'reason': row['reason'],
        'wall_seconds': row['wall'],
        'sort_value': row.get('stat'),
        'errors': {'%s_v%d %s' % (role, index, channel): value
                   for (role, channel, index), value
                   in sorted(row['columns'].items())},
        'units': row['units'],
    }
