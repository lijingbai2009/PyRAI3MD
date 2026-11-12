"""
======================================================================
This module provides the read-only extraction flow of the framework.

A `jobtype read` run never starts an engine. It walks the paths named by
the `read` section, hands each one to the skill parser of the single
theory binding, and returns the extracted frames to its caller. No
training-data file, no trajectory and no result artifact is written: the
caller (the training-data tool, or a Python caller of pmd.run) owns the
filtering, statistics and persistence. The ordinary run log is bypassed
as well, so a read run leaves the working directory untouched.

The skill side is a module-level `read_calculation(config, calcdir,
project=None, profile=None)` re-exported by each skill's backend module.
It is deliberately not the Backend class: constructing a backend
validates the engine installation and resolves a runner, which a machine
holding only finished artifacts does not have.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import importlib
import os

import numpy as np

from ..schema.validate import parse_jobtype

_LABELS = ('energy', 'gradient', 'nac', 'soc')

# File suffixes that identify one calculation's artifacts. The project
# stem is the part before the suffix; a run directory may be named
# `<project>_1` while the files inside keep the run title.
_ARTIFACT_SUFFIXES = (
    '.md.xyz', '.md.energies', '.md.pop', '.sh.xyz', '.sh.energies',
    '.engrad', '.opt.xyz', '.out', '.log', '.inp', '.json', '.xyz',
    '.data',
)


class ReadExtractionError(RuntimeError):
    """One or more read paths could not be extracted."""


def is_read_jobtype(state):
    """Whether this state asks for the read-only extraction flow."""
    jobtype = str(getattr(state, 'jobtype', '') or '').lower()
    tokens, _basic, _advanced = parse_jobtype(jobtype)
    return 'read' in tokens


def read_calculations(state):
    """Extract labels from the finished artifacts named by `read.path`.

    Parameters
    ----------
    state : PyRAIState
        State built from a `jobtype read` input. Its single theory task
        carries the merged skill config, and `state.pmd.read` names the
        paths and labels.

    Returns
    -------
    dict
        JSON-safe payload: title, skill, paths, requested labels, one
        frame per path (symbols, coordinates, requested label channels,
        unit block, missing labels) and the labels missing from every
        frame.

    Raises
    ------
    ReadExtractionError
        When any named path could not be extracted.
    NotImplementedError
        When the theory skill has no file-based reader.
    """
    from .records.adaptive import TRAINING_UNITS
    from .records.conversion import apply_result_units

    config = getattr(state, 'pmd', None)
    section = getattr(config, 'read', None) if config is not None else None
    if section is None:
        raise ReadExtractionError(
            "this state carries no read section; set control.jobtype "
            "'read' and add a read block")

    requested = [name for name in _LABELS if bool(getattr(section, name))]
    paths = _resolve_paths(section.path, section.project)

    tasks = list(getattr(state, 'theory', []) or [])
    if len(tasks) != 1:
        raise ReadExtractionError(
            "a read run binds exactly one theory task, got %d"
            % len(tasks))
    task = tasks[0]
    skill_name = str(task.get('skill_name', '') or '')
    api = str(task.get('api', '') or '')
    module_path = api.rsplit('.', 1)[0] if '.' in api else ''
    if not module_path:
        raise ReadExtractionError(
            "cannot resolve the skill module of this read task")
    module = importlib.import_module(module_path)
    read_calculation = getattr(module, 'read_calculation', None)
    if read_calculation is None:
        raise NotImplementedError(
            "skill '%s' has no file-based reader. A read run needs a "
            "calculation artifact on disk; in-process engines (pyscf, "
            "qdm, nn_pes) compute and return in memory and write no "
            "directory a reader could reopen." % skill_name)

    skill_config = dict((task.get('config') or {}).get('skill_config') or {})
    profile = str(getattr(section, 'profile', '') or '').strip() or None

    frames = []
    failures = []
    for calcdir, project, display in paths:
        try:
            result = read_calculation(
                skill_config, calcdir,
                project=project, profile=profile)
            result = apply_result_units(result, skill_name)
            frames.append(_frame(result, display, project,
                                 skill_name, TRAINING_UNITS, requested))
        except Exception as error:
            failures.append((display, error))

    if failures:
        lines = ['read path failed for %d of %d entries:'
                 % (len(failures), len(paths))]
        for display, error in failures:
            lines.append('  %s: %s' % (display, error))
        raise ReadExtractionError('\n'.join(lines))

    missing = [name for name in requested
               if all(name in frame['missing'] for frame in frames)]

    return {
        'title': str(getattr(getattr(config, 'control', None),
                             'title', '') or ''),
        'skill': skill_name,
        'paths': [display for _calcdir, _project, display in paths],
        'requested': requested,
        'frames': frames,
        'missing': missing,
    }


def _frame(result, display, project, skill_name, default_units,
           requested):
    """Normalize one skill result to the JSON-safe frame contract."""
    if not isinstance(result, dict):
        raise ValueError(
            "skill '%s' returned %s, not a frame mapping"
            % (skill_name, type(result).__name__))
    symbols = result.get('symbols')
    coordinates = result.get('coordinates')
    if coordinates is None:
        raise ValueError(
            "skill '%s' read no geometry from this path; a training "
            "frame needs coordinates" % skill_name)
    coordinates = _to_list(coordinates)
    if symbols is None:
        raise ValueError(
            "skill '%s' read no atom symbols from this path"
            % skill_name)
    unit = dict(default_units)
    unit.update(result.get('unit') or {})
    frame = {
        'path': display,
        'project': project,
        'symbols': list(symbols),
        'coordinates': coordinates,
        'unit': unit,
    }
    for name in _LABELS:
        value = result.get(name)
        frame[name] = None if value is None else _to_list(value)
    frame['missing'] = [name for name in requested
                        if frame.get(name) is None]
    return frame


def _to_list(value):
    """JSON-safe list from a numpy array, list or scalar."""
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _resolve_paths(value, project):
    """Expand read.path into (calcdir, project, display) triples.

    A single file whose non-comment lines all resolve to existing paths
    is a path list; its entries are anchored to the list file's
    directory. Every other entry is a calculation directory or an
    output file, whose parent directory and file stem locate the
    artifacts.
    """
    entries = _as_entries(value)
    if len(entries) == 1 and os.path.isfile(entries[0]) \
            and _looks_like_path_list(entries[0]):
        list_path = os.path.abspath(os.path.expanduser(entries[0]))
        base = os.path.dirname(list_path)
        with open(list_path) as handle:
            entries = []
            for line in handle:
                entry = line.strip()
                if not entry or entry.startswith('#'):
                    continue
                if not os.path.isabs(entry):
                    entry = os.path.join(base, entry)
                entries.append(entry)
        if not entries:
            raise ReadExtractionError(
                "read.path list file carries no path: %s" % list_path)

    resolved = []
    for entry in entries:
        path = os.path.abspath(os.path.expanduser(str(entry)))
        if os.path.isdir(path):
            calcdir = path
            default = _discover_project(path) \
                or os.path.basename(os.path.normpath(path))
        elif os.path.isfile(path):
            calcdir = os.path.dirname(path)
            default = os.path.splitext(os.path.basename(path))[0]
        else:
            raise ReadExtractionError(
                "read.path entry does not exist: %s" % entry)
        resolved.append((calcdir, project or default, path))
    return resolved


def _as_entries(value):
    """One path string or a sequence of paths as a list."""
    if value is None:
        return []
    if isinstance(value, str):
        entries = [value]
    elif isinstance(value, (list, tuple)):
        entries = [str(item) for item in value]
    else:
        entries = [str(value)]
    return [entry for entry in entries if str(entry).strip()]


def _looks_like_path_list(path):
    """Whether a text file lists one existing path per non-comment line."""
    try:
        with open(path) as handle:
            entries = [line.strip() for line in handle
                       if line.strip() and not line.strip().startswith('#')]
    except (OSError, UnicodeDecodeError):
        return False
    if not entries:
        return False
    base = os.path.dirname(os.path.abspath(path))
    for entry in entries:
        candidate = entry if os.path.isabs(entry) \
            else os.path.join(base, entry)
        if not os.path.exists(candidate):
            return False
    return True


def _discover_project(calcdir):
    """The unique artifact stem in a directory, or an empty string.

    A framework trajectory directory is named `<title>_<idx>` while its
    files keep `<title>.md.xyz` / `<title>.md.energies`. The stem is the
    part before a known artifact suffix; exactly one stem is the run
    title. More than one means the directory holds unrelated files and
    the caller must name the project explicitly.
    """
    stems = set()
    try:
        names = os.listdir(calcdir)
    except OSError:
        return ''
    for name in names:
        for suffix in _ARTIFACT_SUFFIXES:
            if name.endswith(suffix):
                stem = name[:-len(suffix)]
                if stem:
                    stems.add(stem)
                break
    if len(stems) == 1:
        return stems.pop()
    return ''