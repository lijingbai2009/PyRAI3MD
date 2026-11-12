"""
======================================================================
This module provides timing utilities for PyRAI3MD: human-readable
timestamp and elapsed-time formatting for the run log, a simple
Timer class with manual start/stop and context manager usage
patterns, and the PhaseProfiler that attributes one workflow step's
wall time to the phases it is made of.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import time


# Name of the framework-side phase table, and how often it is refreshed
# during a run. Both mirror the resident LAMMPS service's own table
# (skills/lammps_skill/service/persistent/server.py: _PERF_FILE,
# _PERF_EVERY) so the two halves of one step read as one format, and so
# a phase table survives a run that never reaches a clean shutdown.
PHASE_FILE = 'pyrai3md_step.perf'
PHASE_EVERY = 100


def what_is_time():
    """Return the current date and time as a readable string.

    Returns
    -------
    str
        Current time formatted as 'Mon DD YYYY  HH:MM:SS'.
    """
    return time.strftime('%b %d %Y  %H:%M:%S')


def how_long(start, end):
    """Format the elapsed time between two timestamps.

    Parameters
    ----------
    start : float
        Start timestamp in seconds (time.time()).
    end : float
        End timestamp in seconds (time.time()).

    Returns
    -------
    str
        Elapsed time formatted as 'DD:HH:MM:SS'.
    """
    # Round to the nearest second: truncation reported sub-second
    # runs as 00:00:00:00 even when the wall clock advanced.
    elapsed = max(round(end - start), 0)
    days, remainder = divmod(elapsed, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%02d:%02d:%02d:%02d' % (days, hours, minutes, seconds)


class Timer:
    """Simple timer with context manager support for profiling.

    Parameters
    ----------
    name : str
        Label for this timer instance.

    Attributes
    ----------
    name : str
        Timer label.
    start_time : float or None
        Timestamp when the timer was started.
    elapsed : float
        Elapsed time in seconds after stopping.

    Examples
    --------
    Manual usage:
    >>> t = Timer('my_task')
    >>> t.start()
    >>> # ... do work ...
    >>> t.stop()
    0.123

    Context manager usage:
    >>> with Timer('my_task') as t:
    ...     # ... do work ...
    >>> print(t.elapsed)
    """

    def __init__(self, name=''):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0

    def start(self):
        """Start the timer."""
        self.start_time = time.perf_counter()

    def stop(self):
        """Stop the timer and return elapsed time.

        Returns
        -------
        float
            Elapsed time in seconds.
        """
        if self.start_time is None:
            return 0.0
        self.elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class PhaseProfiler:
    """Attribute one iterative step's wall time to the phases it contains.

    The resident LAMMPS service has reported its own per-phase table for
    a while, and that is how a measured 10000-step NVE run came to show
    1.48 ms per step inside the engine. The number that mattered was the
    other one: 8.7 ms per step of framework wall time. The gap between
    them was invisible because nothing over the loop itself had a timer,
    so this profiler exists to turn "the framework must be spending it
    somewhere" into a ranked table with units.

    The row format is deliberately identical to the service's, because
    the two tables describe one step and are read side by side.

    Examples
    --------
    >>> profiler = PhaseProfiler()
    >>> with profiler.measure('verlet_i'):
    ...     pass
    >>> profiler.render(1)
    ['[perf] phase=verlet_i steps=1 total_s=0.0000 ms_per_step=0.000']
    """

    def __init__(self):
        self.stats = {}
        self.wall = 0.0
        self._signature = None

    def mark_wall(self, elapsed):
        """Record the wall time of the whole measured span.

        The phase rows answer "where inside the step", but that is only a
        usable answer once the table also says how much of the step they
        account for: a run whose rows sum to a fifth of its wall clock has
        found a gap, not a bottleneck. The owner of the loop stamps this
        number, and it renders as a comment rather than a phase row so a
        reader never adds it to the total it is meant to be compared with.

        Parameters
        ----------
        elapsed : float
            Wall seconds of the span the phases were timed inside.
        """
        self.wall = float(elapsed)

    def record(self, name, elapsed):
        """Accumulate one timed phase.

        Parameters
        ----------
        name : str
            Phase label; rows keep first-seen order, which is the order
            the loop runs them in.
        elapsed : float
            Wall seconds of this occurrence.
        """
        entry = self.stats.setdefault(name, [0, 0.0])
        entry[0] += 1
        entry[1] += float(elapsed)

    def measure(self, name):
        """Time the block as phase `name`.

        Two clock reads per phase are nanoseconds against the
        milliseconds they measure, which is why this stays unconditional
        rather than sitting behind a switch nobody turns on before a run
        goes wrong.

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

    def render(self, steps):
        """Render the table, in milliseconds per evaluated step.

        Parameters
        ----------
        steps : int
            Number of steps measured; the divisor of every total.

        Returns
        -------
        list of str
            One '[perf] phase=...' row per recorded phase, in loop order,
            plus the total, plus the wall row the total is compared
            against; empty when nothing was recorded yet.
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
            lines.append('[perf] phase=%s steps=%d total_s=%.4f ms_per_step=%.3f'
                         % (name, steps, seconds, 1000.0 * seconds / steps))
        lines.append('[perf] phase=TOTAL steps=%d total_s=%.4f ms_per_step=%.3f'
                     % (steps, total, 1000.0 * total / steps))
        if self.wall > 0.0:
            lines.append('# wall total_s=%.4f ms_per_step=%.3f '
                         'coverage=%.3f'
                         % (self.wall, 1000.0 * self.wall / steps,
                            total / self.wall))
        return lines

    def refresh(self, directory, steps, extra=()):
        """Rewrite the table only if this profiler's numbers moved.

        A workflow object is reused across the trajectories of a batch,
        so without a signature the periodic write would dump another
        trajectory's whole table on every cadence tick of the next one,
        and the file would keep whatever the last write left in it.

        Parameters
        ----------
        directory : str
            Run directory to refresh the table in.
        steps : int
            Steps measured so far.
        extra : iterable of str, optional
            Rows of a nested breakdown (a component's own table, timed
            inside one row of this one) appended under a header, so the
            two never add up twice in this table's total.
        """
        signature = (steps, len(self.stats), len(extra),
                     tuple(round(seconds, 4)
                           for _calls, seconds in self.stats.values()))
        if signature == self._signature:
            return
        self._signature = signature
        self.write(directory, steps, extra=extra)

    def write(self, directory, steps, extra=()):
        """Refresh the table inside the run directory.

        Called on a cadence rather than at the end of the run, because a
        workflow that dies mid-trajectory (or finishes without a clean
        shutdown of its backends) still has to leave its timings behind:
        that is the difference between a number the next investigation
        can use and one it has to re-derive. Failures are swallowed for
        the same reason the service swallows them: a table that cannot be
        written must not end a calculation that is going fine.

        Parameters
        ----------
        directory : str
            Run directory holding the log and the md.* records.
        steps : int
            Steps measured so far.
        extra : iterable of str, optional
            Rows of a nested breakdown, appended under a header of their
            own so a reader never adds a component's phases to the step
            rows that already contain them.
        """
        lines = self.render(steps)
        if not lines or not directory:
            return
        nested = list(extra)
        if nested:
            lines.append('# nested breakdown (inside the rows above)')
            lines.extend(nested)
        try:
            with open(os.path.join(directory, PHASE_FILE), 'w',
                      encoding='utf-8') as handle:
                handle.write('\n'.join(lines) + '\n')
        except OSError:
            pass


class _PhaseTimer:
    """Context manager returned by PhaseProfiler.measure()."""

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
