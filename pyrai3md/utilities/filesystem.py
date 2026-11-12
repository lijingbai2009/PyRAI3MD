"""
======================================================================
This module provides filesystem utility functions for PyRAI3MD,
including directory creation, safe file copying, and context manager
for temporary working directory changes.

The three helpers exist so that a run, a tool or a test can state what
it wants from the filesystem once instead of writing the same
`os.makedirs` / `shutil.copy2` / `chdir`-and-restore block at every
call site. The directory change is the one that must not be hand-rolled:
a bare `os.chdir` leaves the process pointing elsewhere when the body
raises, and a qualification round reads artifacts by the current
directory. `working_directory` restores the previous directory on every
exit path, which is why the framework's own rule against process-wide
chdir (dev/SKILL_APP_DEVELOPMENT.md section 5.6) points here.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import shutil
from contextlib import contextmanager


def ensure_dir(path):
    """Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str
        Directory path to ensure exists.

    Returns
    -------
    str
        The same path that was passed in.
    """
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def safe_copy(src, dst):
    """Copy one file, preserving its metadata.

    The copy is not swallowed: a missing source is an OSError with the
    pair that failed, because a silently absent fixture reads later as a
    wrong number rather than as a missing file.

    Parameters
    ----------
    src : str
        Source file path.
    dst : str
        Destination file path.

    Returns
    -------
    str
        The destination path.
    """
    try:
        shutil.copy2(src, dst)
    except OSError as error:
        raise OSError('copy failed: %s -> %s (%s)'
                      % (src, dst, error)) from error
    return dst


@contextmanager
def working_directory(path):
    """Context manager for temporarily changing the working directory.

    The previous directory is restored on every exit path, including an
    exception inside the body.

    Parameters
    ----------
    path : str
        Directory to change to. Restores the original directory
        when the context exits.

    Examples
    --------
    >>> with working_directory('/tmp/calc'):
    ...     # do work in /tmp/calc
    ...     pass
    """
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield previous
    finally:
        os.chdir(previous)
