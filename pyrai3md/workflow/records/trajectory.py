"""
======================================================================
This module provides the trajectory result containers returned by
iterative workflows.

A single-trajectory task is the trajectory-count-1 case of a
multi-trajectory task, so the universal return architecture is an
ordered arrangement of the per-trajectory outcomes arranged by the
planning-order idx. The containers are pure data holders with
read-only access: the trajectory history stays owned by each
trajectory state, and downstream modules (e.g. adaptive sampling)
read it through the existing PyRAIState history interface without
an extra copy layer.

Aug 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from .deviation import DEVIATION_KEYS
from .output import MdOutputWriter


def _is_state(outcome):
    """Check whether an outcome behaves like a PyRAIState.

    Duck-typed on the history interface so this module never
    imports the data package (and tolerates test doubles).

    Parameters
    ----------
    outcome : object
        Outcome returned by one trajectory execution.

    Returns
    -------
    bool
        True when the outcome exposes get_history/get_history_step.
    """
    return (hasattr(outcome, 'get_history')
            and callable(getattr(outcome, 'get_history')))


def _plain_list(value):
    """Frame data as plain Python: a list of lists, an ndarray, or None.

    A summary crosses the dispatch boundary (a worker's result goes back to
    the parent through `pool.map`) and is read by a sampler that knows
    nothing about the frame layout, so nothing numpy-typed leaves here.
    """
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _state_population(system):
    """Diagonal populations of the recorded step, from its density matrix.

    The one density-matrix reading is the output writer's: the trajectory
    file and a summary must not report two different populations for one
    step.
    """
    density = system.get('state_density')
    nstate = int(system.get('nstate') or 0)
    if not nstate and density is not None:
        shape = np.asarray(density).shape
        nstate = int(shape[0]) if shape else 0
    return MdOutputWriter._density_population(density, nstate)


class TrajectoryResult:
    """One trajectory record of an iterative workflow run.

    Parameters
    ----------
    idx : int
        Planning-order index of the trajectory (batch position).
    outcome : object
        Outcome returned by the trajectory execution; normally the
        final PyRAIState, but Result objects or failure markers are
        tolerated as is.
    runtime_idx : str
        Runtime identity suffix allocated to the trajectory (empty
        string when the run carries a single trajectory).
    """

    def __init__(self, idx, outcome, runtime_idx=''):
        self.idx = int(idx)
        self.outcome = outcome
        self.runtime_idx = str(runtime_idx or '')

    def summary(self):
        """The latest recorded step of this trajectory, as one mapping.

        What adaptive sampling asks of a finished trajectory: where it is,
        what the structure and the nuclear state are there, what the
        electronic surfaces cost there, and how far the surrogate was from
        being trusted. It is a reading of the newest history frame, so every
        value is the step as the loop recorded it (the pre-hop active state
        and the geometry the constraints declared), which is the same frame
        the trajectory files were written from.

        The frame is read by reference, not copied: it is the largest object
        in the state and a summary is one row of it. Nothing is aggregated
        here either - a set of trajectories exposes its own summaries and the
        consumer decides what to do with them.

        Returns
        -------
        dict
            Empty when the outcome carries no state or no recorded step;
            otherwise the keys `idx`, `runtime_idx`, `step`, `time`,
            `coordinates`, `velocities`, `active_state`, `energy`, `epot`,
            `kinetic`, `gradient`, `population`, `image` and `deviations`
            (the six `err_*` keys, with the ones the backend did not report
            left out rather than filled with a zero that would read as a
            measured agreement). A multi-center run adds `active_config`.
        """
        state = self.state
        if state is None:
            return {}
        frame = state.get_history_step(-1, copy=False)
        if not frame:
            return {}
        system = frame.get('system') or {}
        results = frame.get('results') or {}
        nuclear = frame.get('nuclear_sample') or {}
        energies = _plain_list(results.get('energy'))
        active = system.get('current_state')
        active = int(active) if active is not None else None
        kinetic = nuclear.get('K_pre')
        summary = {
            'idx': self.idx,
            'runtime_idx': self.runtime_idx,
            'step': frame.get('step'),
            'time': frame.get('time'),
            'coordinates': _plain_list(system.get('coordinates')),
            'velocities': _plain_list(system.get('velocities')),
            'active_state': active,
            'energy': energies,
            'epot': (float(energies[active])
                     if energies and active is not None
                     and active < len(energies) else None),
            'kinetic': float(kinetic) if kinetic is not None else None,
            'gradient': _plain_list(nuclear.get('active_gradient')),
            'population': _state_population(system),
            'image': _plain_list(system.get('image')),
            'deviations': {key: results[key]
                           for key in DEVIATION_KEYS
                           if key in results and results[key] is not None},
        }
        configuration = nuclear.get('active_config_pre')
        if configuration:
            summary['active_config'] = [int(value) for value in configuration]
        return summary

    @property
    def state(self):
        """Final state of the trajectory (read-only).

        Returns
        -------
        PyRAIState or None
            The outcome when it behaves like a PyRAIState,
            otherwise None.
        """
        return self.outcome if _is_state(self.outcome) else None

    def history(self, n=None):
        """Read the recorded trajectory history.

        Parameters
        ----------
        n : int, optional
            Number of most recent snapshots to return; None returns
            everything stored.

        Returns
        -------
        list
            Chronological history snapshots; [] when the outcome
            carries no state or no history.
        """
        state = self.state
        if state is None:
            return []
        return state.get_history(n) or []

    def history_step(self, index=-1):
        """Read one recorded history snapshot.

        Parameters
        ----------
        index : int, optional
            Relative index into the chronological snapshot list
            (-1 = newest).

        Returns
        -------
        dict or None
            The snapshot, or None when the outcome carries no state
            or the step does not exist.
        """
        state = self.state
        if state is None:
            return None
        return state.get_history_step(index)

    def center_history_step(self, center, index=-1):
        """Read one center's electronic record from a history frame.

        Exposes the per-center outcome (energy/gradient/nac/soc/
        density in the center's LOCAL framing), the center's region
        atoms and the pre-hop active configuration of one recorded
        step (docs R6, r6c). Read-only convenience over the structured
        HistoryFrame; the whole-frame access stays in history_step().

        Parameters
        ----------
        center : int
            Center index (0-based).
        index : int, optional
            Relative frame index (-1 = newest).

        Returns
        -------
        dict or None
            {'electronic', 'region', 'active_config_pre'} of the
            center, or None when the outcome carries no state or no
            such frame/center.
        """
        state = self.state
        if state is None:
            return None
        frame = state.get_history_step(index)
        if not frame:
            return None
        outcome = frame.get('electronic_outcome') or {}
        carrier = outcome.get(center)
        if carrier is None:
            carrier = outcome.get(str(center)) or {}
        centers = frame.get('centers') or []
        region = (list(centers[center].get('high_indices', []))
                  if 0 <= center < len(centers) else [])
        nuclear = frame.get('nuclear_sample') or {}
        pre_config = tuple(nuclear.get('active_config_pre') or ())
        return {
            'electronic': dict(carrier),
            'region': region,
            'active_config_pre': pre_config,
        }

    def __repr__(self):
        return "TrajectoryResult(idx=%d, runtime_idx='%s')" % (
            self.idx, self.runtime_idx)


class TrajectorySet:
    """Ordered collection of the trajectory outcomes of one run.

    The universal return structure of iterative workflows: the
    trajectories are arranged by the planning-order idx, and a
    single-trajectory run is the trajectory-count-1 case.

    Parameters
    ----------
    entries : iterable of (idx, outcome) or TrajectoryResult
        Trajectory records; plain (idx, outcome) pairs are wrapped
        into TrajectoryResult objects. The set is kept sorted by idx
        regardless of the input order.
    """

    def __init__(self, entries=None):
        self._entries = []
        for entry in entries or []:
            if isinstance(entry, TrajectoryResult):
                self._entries.append(entry)
            else:
                idx, outcome = entry
                self._entries.append(TrajectoryResult(idx, outcome))
        self._entries.sort(key=lambda record: record.idx)

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __getitem__(self, idx):
        """Return the trajectory record with the given planning idx.

        Parameters
        ----------
        idx : int
            Planning-order index of the trajectory.

        Returns
        -------
        TrajectoryResult
            The record carrying the requested idx.

        Raises
        ------
        KeyError
            If no trajectory carries the requested idx.
        """
        for record in self._entries:
            if record.idx == idx:
                return record
        raise KeyError("No trajectory with idx=%d" % idx)

    @property
    def states(self):
        """Final states of all trajectories in idx order (read-only).

        Returns
        -------
        list
            One entry per trajectory; None where the outcome carries
            no state.
        """
        return [record.state for record in self._entries]

    def summaries(self):
        """The latest recorded step of every trajectory, in idx order.

        Returns
        -------
        list of dict
        One `TrajectoryResult.summary()` per trajectory; an empty
        mapping where the outcome carries no recorded step. A batch
        of one is the single-trajectory case, so the shape is the same
        either way.
        """
        return [record.summary() for record in self._entries]

    def __repr__(self):
        return "TrajectorySet(%d trajectories)" % len(self._entries)
