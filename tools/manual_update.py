#!/usr/bin/env python3
"""
======================================================================
Manual update helper: list the files a running install is missing.

Reinstalling a package replaces the whole installed tree, which can
disrupt a calculation that is still running from it. This tool compares
the repository's `pyrai3md/` package with the installed one and prints
the `cp` commands that would bring the install up to date, file by
file. It never copies anything and never deletes anything.

Usage
-----
    python tools/manual_update.py
    python tools/manual_update.py --source /path/to/pyrai3md

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import argparse
import filecmp
import os
import shlex
import sys
from pathlib import Path

_SKIP_DIRS = {'__pycache__'}
_SKIP_SUFFIXES = ('.pyc', '.pyo')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='List files that differ between the repository '
                    'package and the installed pyrai3md package.')
    parser.add_argument(
        '--source', default=str(Path(__file__).resolve().parents[1]
                                / 'pyrai3md'),
        help='source package directory (default: the repository package)')
    args = parser.parse_args(argv)

    try:
        import pyrai3md
    except ModuleNotFoundError:
        print('pyrai3md is not installed; nothing to compare.')
        return 1

    source = Path(args.source).resolve()
    installed = Path(pyrai3md.__file__).resolve().parent
    if not source.is_dir():
        print('source package not found: %s' % source)
        return 1
    if source == installed:
        print('The installed package is this source tree (editable '
              'install); there is nothing to copy.')
        return 0

    added, changed = _compare(source, installed)
    extras = _extras(source, installed)

    print('Source:    %s' % source)
    print('Installed: %s' % installed)
    print('')
    if not added and not changed:
        print('Every source file is already installed byte-identically.')
    if added:
        print('New files (%d)' % len(added))
        print('---------------------------')
        for relative in added:
            print(' cp %s %s' % (
                shlex.quote(str(source / relative)),
                shlex.quote(str(installed / relative))))
    if changed:
        print('Changed files (%d)' % len(changed))
        print('---------------------------')
        for relative in changed:
            print(' cp %s %s' % (
                shlex.quote(str(source / relative)),
                shlex.quote(str(installed / relative))))
    if extras:
        print('Installed files with no source counterpart (%d)'
              % len(extras))
        print('---------------------------')
        for relative in extras:
            print(' only installed: %s' % (installed / relative))
    return 0


def _compare(source, installed):
    """Relative paths that are new or differ, in sorted order."""
    added = []
    changed = []
    for root, dirs, files in os.walk(source):
        dirs[:] = sorted(name for name in dirs
                         if name not in _SKIP_DIRS)
        for name in sorted(files):
            if name.endswith(_SKIP_SUFFIXES) or name == '.DS_Store':
                continue
            path = Path(root) / name
            relative = path.relative_to(source)
            target = installed / relative
            if not target.is_file():
                added.append(relative)
            elif not filecmp.cmp(path, target, shallow=False):
                changed.append(relative)
    return added, changed


def _extras(source, installed):
    """Installed files that the source tree does not carry."""
    extras = []
    for root, dirs, files in os.walk(installed):
        dirs[:] = sorted(name for name in dirs
                         if name not in _SKIP_DIRS)
        for name in sorted(files):
            if name.endswith(_SKIP_SUFFIXES) or name == '.DS_Store':
                continue
            path = Path(root) / name
            relative = path.relative_to(installed)
            if not (source / relative).is_file():
                extras.append(relative)
    return extras


if __name__ == '__main__':
    sys.exit(main())