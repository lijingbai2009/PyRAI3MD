"""
======================================================================
Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

"""Module-level fake execution entry for multiprocessing tests.

Pool workers import this entry through the executor spec injected
into the Dispatcher/Scheduler; it records nothing and returns the
jobtype carried by the child state so order and correctness are
verifiable.
"""


def execute(state, runtime_idx=None):
    """Return the child state jobtype (echo executor)."""
    return getattr(state, 'jobtype', '')
