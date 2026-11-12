"""
======================================================================
This module provides the framework-side step profile of the resident
LAMMPS backend: what one evaluate costs the process that asked for it,
as opposed to what it costs the engine.

Three tables describe one resident step, and each is written by its own
owner so they never overwrite one another:

- ``pyrai3md_step.perf``    the workflow loop (framework package, written
  by the md workflow),
- ``lammps_backend.perf``   this module: the backend around the session,
- ``lammps_resident.perf``  the service: inside the engine.

Read together they answer the question a slow MD step actually poses -
the engine reported a fraction of the step while the run took many times
that - and read apart they show the transport and the result conversion
that sit in neither of the other two. The row format is the one the
service already writes, so all three parse the same way.

The skill keeps its own copy of this accumulator rather than importing
the framework's because of the framework-skill isolation rule (see
utilities/lattice.py): a skill must not depend on framework internals.

Sep 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import time

# File name and refresh cadence of this table. The cadence matches the
# framework's and the service's so one step's three tables are sampled at
# the same steps and can be laid side by side.
PROFILE_FILE = 'lammps_backend.perf'
PROFILE_EVERY = 100


def perf_row(phase, steps, seconds):
    """One '[perf]' row, in the format the framework's table prints.

    The framework's PhaseProfiler (utilities/timer.py) renders the
    same row for its own phases; the two sides are separate by the
    isolation rule, so the format is pinned by a criterion that lays
    the rows side by side instead of by an import.
    """
    return ('[perf] phase=%s steps=%d total_s=%.4f ms_per_step=%.3f'
            % (phase, steps, seconds, 1000.0 * seconds / steps))


class StepProfiler:
    """Accumulate wall time per named phase of a resident step.

    Attributes
    ----------
    stats : dict
        {phase name: [call count, total seconds]}, in first-seen order.
    """

    def __init__(self):
        self.stats = {}
        self._signature = None

    def record(self, name, elapsed):
        """Accumulate one timed phase.

        Parameters
        ----------
        name : str
            Phase label; rows keep first-seen order.
        elapsed : float
            Wall seconds of this occurrence.
        """
        entry = self.stats.setdefault(name, [0, 0.0])
        entry[0] += 1
        entry[1] += float(elapsed)

    def measure(self, name):
        """Time the wrapped block as phase ``name``.

        Two clock reads per phase are nanoseconds against the
        milliseconds they measure, so the timing stays unconditional.

        Parameters
        ----------
        name : str
            Phase label.

        Returns
        -------
        _PhaseTimer
            Context manager recording the block on exit.
        """
        return _PhaseTimer(self, name)

    def rows(self, steps):
        """Render one row per phase, in milliseconds per evaluated step.

        Parameters
        ----------
        steps : int
            Steps measured; the divisor of every total.

        Returns
        -------
        list of str
            The rendered rows; empty before the first step.
        """
        steps = int(steps)
        if steps <= 0:
            return []
        lines = []
        total = 0.0
        for name, (calls, seconds) in self.stats.items():
            if not calls:
                continue
            total += seconds
            lines.append(perf_row(name, steps, seconds))
        lines.append(perf_row('TOTAL', steps, total))
        return lines

    def refresh(self, directory, steps, extra=()):
        """Rewrite the table when this profiler's numbers moved.

        Parameters
        ----------
        directory : str
            Calculation directory of this session.
        steps : int
            Steps measured so far.
        extra : iterable of str, optional
            Rows of a nested table (the client's own buckets), appended
            under a header so they are never added to the rows above.
        """
        signature = (steps, len(self.stats), len(extra),
                     tuple(round(seconds, 4)
                           for _calls, seconds in self.stats.values()))
        if signature == self._signature:
            return
        self._signature = signature
        lines = self.rows(steps)
        if not lines or not directory:
            return
        nested = list(extra)
        if nested:
            lines.append('# client buckets (inside session_evaluate above)')
            lines.extend(nested)
        try:
            with open(os.path.join(directory, PROFILE_FILE), 'w',
                      encoding='utf-8') as handle:
                handle.write('\n'.join(lines) + '\n')
        except OSError:
            # A table that cannot be written must not end a calculation
            # that is going fine; the service treats its own the same way.
            pass


class _PhaseTimer:
    """Context manager returned by StepProfiler.measure()."""

    __slots__ = ('_profiler', '_name', '_started')

    def __init__(self, profiler, name):
        self._profiler = profiler
        self._name = name
        self._started = 0.0

    def __enter__(self):
        self._started = time.perf_counter()
        return self

    def __exit__(self, *args):
        self._profiler.record(self._name,
                              time.perf_counter() - self._started)
        return False
