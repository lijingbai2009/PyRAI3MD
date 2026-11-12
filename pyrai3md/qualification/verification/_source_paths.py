"""
======================================================================
Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

"""Module-level source paths for the guards that read the source tree.

Three verification guards scan the tree itself (the runner modules' cwd
handling, the thread-ownership rules, the mpi launcher templates). They used to
spell their patterns relative to the current directory, which is not a fixed
place: run from the repository root, `pyrai3md/skills/...` matches nothing, and
on a case-insensitive filesystem the leading `pyrai3md` resolves to the project
directory `PyRAI3MD`, so the sweep then walks `build/lib/pyrai3md/` - a stale
copy of the package kept by `setup.py build` - and reports removed APIs as
present. Anchoring at this file makes the answer independent of where pytest was
started, and the build directory is excluded by construction because the pattern
root is the source tree's parent.
"""

import glob
import os

# This file lives at <source parent>/pyrai3md/qualification/verification/, so
# three levels up from its directory is the directory that contains `pyrai3md`.
SOURCE_PARENT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def source_tree(pattern):
    """Every path matching one tree pattern, as absolute paths.

    Absolute so a caller can open what it found without knowing where the
    pattern was anchored; a guard that compares against a literal tree path
    reads clearer through `relative_to_source`.
    """
    found = glob.glob(os.path.join(SOURCE_PARENT, pattern), recursive=True)
    return sorted(os.path.abspath(path) for path in found)


def relative_to_source(path):
    """One path of the source tree, spelled relative to the tree's parent."""
    return os.path.relpath(path, SOURCE_PARENT)


def source_path(path):
    """The absolute path of one tree-relative path."""
    return os.path.join(SOURCE_PARENT, path)


def source_read(path):
    """The text of one source-tree file, named relative to the tree or not."""
    with open(source_path(path), encoding='utf-8') as handle:
        return handle.read()
