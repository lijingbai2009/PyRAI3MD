"""
======================================================================
This module provides the initial-condition resolution of the md
workflow: it turns the &md initcond/ninitcond keywords into the list
of trajectory starting points (coordinates + velocities) consumed
by the multi-trajectory fan-out.

initcond semantics:
- '0' / 'none'         use the geometry/velocities carried by the
                       state; replicated across trajectories when
                       ninitcond > 1
- 'read' / 'load'      read ninitcond frames from <title>.init.xyz in the
                       current working directory (one frame per
                       initial condition, atom lines carry
                       coordinates and velocities)
- 'wigner' / 'boltzmann'
                       vibrational sampling from the frequency
                       record '<title>.freq.<format>' in the
                       current working directory (workflow/
                       sampling.py); the generator is seeded from
                       the control gl_seed

A caller that already owns the starting points (the adaptive chain
selecting structures from finished trajectories) bypasses the three
sources above through adopt_initial_conditions(); the batch it hands
over is validated here and used as the only source, never merged with
or overridden by the initcond keyword.

Aug 14 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os

import numpy as np

from .sampling import load_frequency_data, sample_initial_conditions


def normalize_initcond(value):
    """Normalize the initcond keyword to its canonical form.

    Parameters
    ----------
    value : str or int
        Raw initcond keyword from the dynamics section.

    Returns
    -------
    str
        One of '0', 'none', 'read', 'load', 'wigner', 'boltzmann'.

    Raises
    ------
    ValueError
        If the value is not recognized.
    """
    key = str(value).strip().lower()
    # Legacy numeric semantics: 0 = use the state geometry,
    # nonzero = sampling (defaults to wigner).
    if key == '1':
        key = 'wigner'
    if key in ('0', 'none', 'read', 'load', 'wigner', 'boltzmann'):
        return key
    raise ValueError("Unrecognized initcond value: %r" % (value,))


def read_init_xyz(path):
    """Parse a multi-frame `<title>.init.xyz` (coordinates + velocities).

    Each frame follows the standard xyz layout: a natom line, a
    comment line, then natom atom lines of an element symbol
    followed by six floats (X Y Z VX VY VZ).

    Parameters
    ----------
    path : str
        Path of the ensemble file, named `<title>.init.xyz`.

    Returns
    -------
    list of tuple
        One (coordinates, velocities) pair per frame; each array has
        shape (natom, 3).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty or a frame is malformed (wrong natom,
        missing atom lines, or non-numeric columns).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Initial condition file not found: %s" % path)
    with open(path) as handler:
        lines = handler.read().splitlines()

    frames = []
    position = 0
    total = len(lines)
    while position < total:
        if not lines[position].strip():
            position += 1
            continue
        try:
            natom = int(lines[position].split()[0])
        except (IndexError, ValueError):
            raise ValueError(
                "Malformed frame header in %s (line %d): %r" % (
                    path, position + 1, lines[position]))
        position += 2  # skip the header and the comment line
        if position + natom > total:
            raise ValueError(
                "Truncated frame in %s: expected %d atom lines, "
                "got %d" % (path, natom, total - position))
        coordinates = []
        velocities = []
        for line in lines[position: position + natom]:
            tokens = line.split()
            if len(tokens) < 7:
                raise ValueError(
                    "Atom line needs element + 6 columns "
                    "(X Y Z VX VY VZ) in %s: %r" % (path, line))
            try:
                values = [float(token) for token in tokens[1:7]]
            except ValueError:
                raise ValueError(
                    "Non-numeric columns in %s: %r" % (path, line))
            coordinates.append(values[0:3])
            velocities.append(values[3:6])
        position += natom
        frames.append((np.array(coordinates), np.array(velocities)))
    if not frames:
        raise ValueError("No frames found in %s" % path)
    return frames


def write_init_xyz(path, batch, atoms, comment='Init'):
    """Write trajectory starting points in the ``<title>.init.xyz`` layout.

    The layout is the one read_init_xyz() expects: an atom count, one
    comment line, then one line per atom with the element symbol, the
    coordinates and the velocities. The same content is used for the
    warm-up product ``<title>.init`` and the input file
    ``<title>.init.xyz``.

    Parameters
    ----------
    path : str
        Destination file.
    batch : list of dict
        One {'coordinates', 'velocities'} mapping per trajectory.
    atoms : sequence of str
        Element labels of the system.
    comment : str
        Second line of every frame.

    Returns
    -------
    str
        The path written.
    """
    labels = list(atoms)
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, 'w') as handle:
        for index, item in enumerate(batch):
            coordinates = np.asarray(item['coordinates'], dtype=np.float64)
            velocities = np.asarray(item['velocities'], dtype=np.float64)
            if coordinates.shape != (len(labels), 3):
                raise ValueError(
                    "initial condition %d coordinates have shape %s but the "
                    "system has %d atoms"
                    % (index, coordinates.shape, len(labels)))
            if velocities.shape != (len(labels), 3):
                raise ValueError(
                    "initial condition %d velocities have shape %s but the "
                    "system has %d atoms"
                    % (index, velocities.shape, len(labels)))
            handle.write('%d\n' % len(labels))
            handle.write('%s\n' % comment)
            for label, xyz, velo in zip(labels, coordinates, velocities):
                handle.write(
                    '%s %.10f %.10f %.10f %.10f %.10f %.10f\n'
                    % (label, xyz[0], xyz[1], xyz[2],
                       velo[0], velo[1], velo[2]))
    return destination


def adopt_initial_conditions(batch, natom):
    """Validate an externally supplied batch of trajectory starting points.

    Used when the caller owns the starting points (adaptive sampling
    selecting structures from finished trajectories) instead of the
    initcond keyword. The returned batch is the only source for the run:
    an empty payload raises here rather than falling back to the state
    geometry or <title>.init.xyz, so a caller that failed to select any
    structure cannot be mistaken for a run that started normally.

    Parameters
    ----------
    batch : list or None
        One {'coordinates', 'velocities'} mapping per trajectory. The
        two values may be lists or numpy arrays and are normalized to
        nested lists; the input is never mutated.
    natom : int
        Atom count of the run; every frame must match it.

    Returns
    -------
    list of dict
        Normalized {'coordinates', 'velocities'} descriptors, one per
        trajectory, in payload order.

    Raises
    ------
    ValueError
        If the payload is None or empty, an item is not a mapping, a
        key is missing, or a frame does not have the (natom, 3) shape.
    """
    if batch is None:
        raise ValueError(
            "adopt_initial_conditions is enabled but initial_conditions "
            "is None; the md workflow does not fall back to initcond or "
            "<title>.init.xyz when the caller owns the starting points.")
    try:
        items = list(batch)
    except TypeError:
        raise ValueError(
            "initial_conditions must be a sequence of "
            "{'coordinates', 'velocities'} mappings, got %r."
            % (type(batch).__name__,))
    if not items:
        raise ValueError(
            "adopt_initial_conditions is enabled but initial_conditions "
            "is empty; a run needs at least one starting point.")

    natom = int(natom)
    adopted = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                "initial_conditions[%d] must be a mapping with "
                "'coordinates' and 'velocities', got %r."
                % (index, type(item).__name__))
        missing = [key for key in ('coordinates', 'velocities')
                   if item.get(key) is None]
        if missing:
            raise ValueError(
                "initial_conditions[%d] is missing %s."
                % (index, ', '.join(missing)))
        coordinates = np.asarray(item['coordinates'], dtype=np.float64)
        velocities = np.asarray(item['velocities'], dtype=np.float64)
        if coordinates.shape != (natom, 3):
            raise ValueError(
                "initial_conditions[%d] coordinates have shape %s but "
                "the system has %d atoms." % (
                    index, coordinates.shape, natom))
        if velocities.shape != (natom, 3):
            raise ValueError(
                "initial_conditions[%d] velocities have shape %s but "
                "the system has %d atoms." % (
                    index, velocities.shape, natom))
        adopted.append({'coordinates': coordinates.tolist(),
                        'velocities': velocities.tolist()})
    return adopted


def resolve_initial_conditions(state):
    """Resolve the trajectory starting points of one md run.

    Parameters
    ----------
    state : PyRAIState
        State carrying the dynamics section (initcond/ninitcond) and
        the current system geometry/velocities.

    Returns
    -------
    list of dict
        One {'coordinates', 'velocities'} descriptor per trajectory
        (length == ninitcond), in frame order.

    Raises
    ------
    ValueError
        If ninitcond < 1, the initcond value is unrecognized, or the
        `<title>.init.xyz` file carries fewer frames than ninitcond.
    FileNotFoundError
        If `<title>.init.xyz` (read/load) or the frequency file
        (wigner/boltzmann) is missing.
    """
    dynamics = state.pmd.dynamics
    initcond = normalize_initcond(dynamics.initcond)
    ninitcond = int(dynamics.ninitcond)
    if ninitcond < 1:
        raise ValueError(
            "ninitcond must be >= 1, got %d" % ninitcond)
    # One naming rule for every file the initial-condition step reads: the
    # run title from the control section, the same source the frequency
    # record (`<title>.freq.<format>`) is resolved from.
    control = state.pmd.control
    title = str(getattr(control, 'title', '') or 'pyrai3md')
    gl_seed = int(getattr(control, 'gl_seed', 1) or 1)

    if initcond in ('0', 'none'):
        system = state.pmd.system
        coordinates = np.asarray(system.coordinates)
        velocities = np.asarray(system.velocities)
        return [{'coordinates': coordinates.tolist(),
                 'velocities': velocities.tolist()}
                for _ in range(ninitcond)]

    if initcond in ('read', 'load'):
        init_path = '%s.init.xyz' % title
        frames = read_init_xyz(init_path)
        if len(frames) < ninitcond:
            raise ValueError(
                "%s carries %d frames but ninitcond=%d" % (
                    init_path, len(frames), ninitcond))
        return [{'coordinates': coords.tolist(),
                 'velocities': velos.tolist()}
                for coords, velos in frames[:ninitcond]]

    # Vibrational sampling: one draw per initial condition from the
    # frequency record, seeded by the control gl_seed (the root
    # runtime samples once before the fan-out, so the ensemble is
    # insensitive to the worker scheduling order).
    freqdata = load_frequency_data(title, dynamics.format)
    ensemble = sample_initial_conditions(
        freqdata, ninitcond, method=initcond,
        temp=float(dynamics.temperature), seed=gl_seed)
    return [{'coordinates': np.asarray(
                 sample['coordinates']).tolist(),
             'velocities': np.asarray(
                 sample['velocities']).tolist()}
            for sample in ensemble]
