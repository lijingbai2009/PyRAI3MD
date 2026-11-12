"""
======================================================================
This is a standalone OpenMolcas calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module is the main parsing entry point.
It selects the appropriate parser based on the calculation profile.

"""

import os
import numpy as np

from .log_parser import parse_log
from .orbital_parser import manage_orbital_files


def _extract_velocities(config, calcdir):
    """Extract the final velocities (reserved fin_velocity channel).

    The molcas skill performs no skill-internal MD propagation that
    reports final velocities, so the fin_velocity return channel is
    reserved with an empty value; the uniform channel keeps every
    skill result contract aligned.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str
        Calculation directory.

    Returns
    -------
    list
        Empty list (no velocity output).
    """
    return []


def parse_output(config, calcdir: str, workflow) -> dict:
    """Parse Molcas output files based on the calculation profile.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str
        Calculation directory.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    dict
        Parsed results dictionary.
    """
    project = config.project
    log_path = os.path.join(calcdir, f'{project}.log')

    # Parse the log file
    results = parse_log(log_path, config, workflow=workflow)

    # Map the log completion status onto the framework opt_converge
    # contract: only the geometry-optimization profile iterates
    # internally, so the marker is reported there and dropped
    # elsewhere to keep non-optimize results clean.
    log_not_converged = results.pop('log_not_converged', False)
    if config.profile == 'casscf_geometry_optimization':
        results['opt_converge'] = not log_not_converged

    # Final structure from skill-internal iteration (optimize/dynamics),
    # returned through the framework fin_coord channel. An optimization
    # that stopped at its step budget has no final structure to report:
    # the missing file is the status, not an error.
    fin_coord = parse_fin_coord(config, calcdir,
                                not_converged=log_not_converged)
    if len(fin_coord):
        results['fin_coord'] = fin_coord

    # Reserved fin_velocity return channel (no velocity output)
    results['fin_velocity'] = _extract_velocities(config, calcdir)

    # Manage orbital files for subsequent steps
    manage_orbital_files(config, calcdir)

    return results


def parse_fin_coord(config, calcdir: str, not_converged: bool = False):
    """Read the final structure from the Molcas iteration output.

    Geometry optimization writes the relaxed structure to
    $Project.Opt.xyz (SLAPAF capitalizes the extension on some
    versions; the lowercase spelling is probed as a fallback);
    dynamics (ab initio / surface hopping) writes the full
    trajectory to $Project.md.xyz, whose last frame is the final
    structure. Coordinates are in Angstrom, matching the framework
    fin_coord channel contract ((natom, 3)).

    Other profiles perform no internal iteration and return an empty
    list, leaving the framework coordinates untouched.

    An optimization that exhausted its step budget stops SLAPAF with
    _RC_NOT_CONVERGED_ and never writes the final structure. That is
    a reported outcome (opt_converge=False), not a read failure, so
    no coordinates are returned instead of raising.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str
        Calculation directory.
    not_converged : bool, optional
        Whether the run log recorded a SLAPAF budget stop. A missing
        structure then yields an empty array.

    Returns
    -------
    np.ndarray
        Final coordinates (natom, 3) in Angstrom, or an empty array
        for non-iterative profiles and unconverged optimizations.

    Raises
    ------
    RuntimeError
        When a converged iterative profile finished but its output
        file is missing or carries no parseable frame.
    ValueError
        When the final frame atom count disagrees with the system.
    """
    profile = config.profile
    if profile == 'casscf_geometry_optimization':
        candidates = [f'{config.project}.Opt.xyz',
                      f'{config.project}.opt.xyz']
    elif profile in ('casscf_abinitio_dynamics',
                     'casscf_surface_hopping'):
        candidates = [f'{config.project}.md.xyz']
    else:
        return np.zeros((0, 3))

    xyz_path = None
    for name in candidates:
        path = os.path.join(calcdir, name)
        if os.path.exists(path):
            xyz_path = path
            break
    if xyz_path is None:
        if not_converged and profile == 'casscf_geometry_optimization':
            return np.zeros((0, 3))
        raise RuntimeError(
            f'Molcas {profile} finished but the final structure file '
            f'{candidates[0]} is missing in {calcdir}.'
        )

    frames = _parse_xyz_frames(xyz_path)
    if not frames:
        raise RuntimeError(
            f'Molcas {profile} output file {xyz_path} carries no '
            f'parseable XYZ frame.'
        )

    coords = np.array(frames[-1])
    if len(coords) != config.natom:
        raise ValueError(
            f'Molcas final structure in {xyz_path} has {len(coords)} '
            f'atoms, but the system has {config.natom}.'
        )
    return coords


def _parse_xyz_frames(xyz_path: str) -> list:
    """Parse every frame of a (multi-frame) XYZ file.

    Each frame is a natom count line, a comment line and natom
    coordinate lines; the comment content is ignored.

    Parameters
    ----------
    xyz_path : str
        Path to the XYZ file.

    Returns
    -------
    list of list of float
        Coordinates (natom, 3) per frame, in file order.
    """
    with open(xyz_path, 'r') as f:
        lines = f.read().splitlines()

    frames = []
    i = 0
    n = len(lines)
    while i < n:
        parts = lines[i].split()
        if not parts:
            i += 1
            continue
        try:
            natom = int(parts[0])
        except ValueError:
            i += 1
            continue
        frame = []
        for atom_line in lines[i + 2: i + 2 + natom]:
            tokens = atom_line.split()
            if len(tokens) < 4:
                break
            try:
                frame.append([float(tokens[1]), float(tokens[2]),
                              float(tokens[3])])
            except ValueError:
                break
        if len(frame) == natom:
            frames.append(frame)
            i += 2 + natom
        else:
            i += 1
    return frames


def format_results(results: dict, config) -> dict:
    """Format parsed results for output.

    Converts numpy arrays to lists for JSON serialization.

    Parameters
    ----------
    results : dict
        Raw parsed results.
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    dict
        Formatted results with numpy arrays converted to lists.
    """
    formatted = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            formatted[key] = value.tolist()
        else:
            formatted[key] = value

    return formatted
