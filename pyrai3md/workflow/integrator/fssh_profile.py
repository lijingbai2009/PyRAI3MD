"""
======================================================================
This module provides the function-level timing support of the FSSH
electronic propagator (dev/methods/FSSH_DEVELOPMENT.md,
sections 21 and 22).

The profiler accumulates call counts and wall time for every timed
FSSH function (dPdt, rk4, matB, ktdc, decoherence, hopping,
velocity adjustment) and renders the 'FSSH Performance' summary
together with the active numpy/numba implementation table.

Numba warm-up: the first call of a Numba kernel includes the JIT
compilation time and must never enter the statistics; the warm_up()
helper runs one dummy call of every selected Numba kernel before
the measurement starts.

Aug 26 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import time

import numpy as np

# Canonical timing entry order of the performance table.
PROFILE_FUNCTIONS = ('dPdt', 'RK4', 'matB', 'kTDC', 'decoherence',
                     'hopping', 'velocity adjustment')


class FsshProfiler:
    """Function-level timing accumulator for one FSSH run.

    Attributes
    ----------
    stats : dict
        {function name: [call count, total seconds]}.
    implementations : dict
        {function name: implementation label} ('numpy' / 'numba'),
        rendered next to the timing table.
    """

    def __init__(self):
        self.stats = {name: [0, 0.0] for name in PROFILE_FUNCTIONS}
        self.implementations = {}

    def record(self, name, elapsed):
        """Accumulate one timed call of a profiled function.

        Parameters
        ----------
        name : str
            Profiled function name (a PROFILE_FUNCTIONS member).
        elapsed : float
            Wall time of the call in seconds.
        """
        entry = self.stats.setdefault(name, [0, 0.0])
        entry[0] += 1
        entry[1] += float(elapsed)

    def wrap(self, name, func):
        """Wrap a kernel so every call is timed under 'name'.

        Parameters
        ----------
        name : str
            Profiled function name.
        func : callable
            The selected kernel implementation.

        Returns
        -------
        callable
            Timing wrapper with the same call signature.
        """
        profiler = self

        def timed(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            profiler.record(name, time.perf_counter() - start)
            return result

        return timed

    def set_implementations(self, implementations):
        """Store the active implementation table.

        Parameters
        ----------
        implementations : dict
            {function name: 'numpy' or 'numba'}.
        """
        self.implementations = dict(implementations)

    def summary(self):
        """Render the FSSH Performance and Implementation tables.

        Returns
        -------
        str
            Multi-line report string (design document section 21
            layout).
        """
        lines = ['', 'FSSH Performance',
                 '-' * 48,
                 '%-18s %10s %10s' % ('Function', 'Calls', 'Total')]
        total = 0.0
        for name in PROFILE_FUNCTIONS:
            calls, elapsed = self.stats.get(name, [0, 0.0])
            if calls == 0:
                continue
            total += elapsed
            lines.append('%-18s %10d %9.2f s' % (name, calls, elapsed))
        lines.append('%-18s %10s %9.2f s' % ('Total', '', total))

        if self.implementations:
            lines.extend(['', 'Implementation', '-' * 48])
            for name, impl in self.implementations.items():
                lines.append('%-18s %10s' % (name, impl))
        return '\n'.join(lines)


def warm_up(kernels):
    """Run one dummy call of every selected Numba kernel.

    Excludes the JIT compilation time from later measurements
    (design document section 22). NumPy kernels need no warm-up
    and are skipped silently.

    Parameters
    ----------
    kernels : dict
        {kernel label: (callable, is_numba)} pairs; only entries
        flagged as Numba are exercised.
    """
    nstate = 2
    A = np.eye(nstate, dtype=np.complex128) / nstate
    H = np.diag([0.0, 0.1]).astype(np.complex128)
    D = np.zeros((nstate, nstate), dtype=np.complex128)
    dHD = np.zeros((nstate, nstate), dtype=np.complex128)
    E = np.array([0.0, 0.1])

    for label, (kernel, is_numba) in kernels.items():
        if not is_numba:
            continue
        if label == 'dPdt':
            kernel(A, H, D)
        elif label == 'rk4':
            kernel(A, H, D, dHD, dHD, 0.5)
        elif label == 'matB':
            kernel(A, H, D)
        elif label == 'ktdc':
            kernel(0, 1, E, E, E, 1.0, 0.5)
        elif label == 'decoherence':
            kernel(A.copy(), H, 0.5, 0, 0.1, 0.01)
