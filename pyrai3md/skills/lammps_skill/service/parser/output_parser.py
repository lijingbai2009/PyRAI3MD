"""
======================================================================
This is a standalone LAMMPS calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module parses LAMMPS output files (log, force dump and xyz
dump) into standardized calculation results.

Unit conversion
---------------
The framework workflows only consume hartree-based units, so the
parser converts all results before returning:
  - real:  kcal/mol -> hartree, kcal/mol/Angstrom -> hartree/bohr
  - metal: eV -> hartree, eV/Angstrom -> hartree/bohr
The gradient is the negative of the LAMMPS force.

Energy normalization
--------------------
The internal-iteration profiles (ff_optimize and the MD profiles)
run with 'thermo_modify norm yes' (aligned with the reference
inputs), so the log energies are per-atom averages while the
framework stores total energies. For these workflows the parsed
energy is multiplied by natom before the unit conversion.

Final geometry and velocities
-----------------------------
The internal-iteration profiles dump the final structure with
'write_dump all xyz {project}_{profile}.xyz modify sort id' (no
element list: the row order is the atom order); the last xyz frame
is returned through the fin_coord channel (natom, 3, Angstrom),
matching the framework contract used by the other skills. The MD
profiles additionally dump the final coordinates + velocities to the
7-column custom dump '{project}_{profile}.cond' (id x y z vx vy vz);
the velocities are converted back to Bohr/au and returned through
the fin_velocity channel.

"""

import os

import numpy as np

from ..builder.input_builder import bound_and_tilt_to_lattice
from ..utils import LAMMPS_VELOCITY_TO_AU
from ..utils import PRESSURE_TO_BAR
from ..utils import (output_basename, read_log_tail,
                     step_data_files)
from .log_parser import parse_log, parse_stress_file


def pressure6_to_stress_tensor(components6, lammps_units: str) -> np.ndarray:
    """Turn LAMMPS's six pressure components into the framework tensor.

    The ordering (xx yy zz xy xz yz), the negation to tensile-positive
    and the unit normalization into bar live here and nowhere else, so
    the log/dump parser and the resident interface cannot drift apart.

    Parameters
    ----------
    components6 : sequence of float
        ``pxx pyy pzz pxy pxz pyz`` in the native units style.
    lammps_units : str
        LAMMPS units style ('real' or 'metal').

    Returns
    -------
    np.ndarray
        The symmetric (3, 3) potential stress in bar.
    """
    pxx, pyy, pzz, pxy, pxz, pyz = components6
    pconf = np.array([[pxx, pxy, pxz],
                      [pxy, pyy, pyz],
                      [pxz, pyz, pzz]])
    factor = PRESSURE_TO_BAR.get(lammps_units, 1.0)
    # LAMMPS pressure is compressive-positive; the framework contract
    # wants the tensile-positive potential stress (sigma = -P).
    return -pconf * factor


def _build_stress_tensor(config, lammps_units, stress_path):
    """Reconstruct the tensile-positive stress tensor (1, 3, 3).

    The six virial pressure components are ordered xx yy zz xy xz yz
    (dev/skills/STRESS_DEVELOPMENT.md section 4.1); the symmetric matrix is
    negated to the tensile-positive potential stress and normalized
    into bar (the manifest units.stress), so the framework applies the
    single bar -> Hartree/Bohr^3 factor. The components come from the
    ``{title}.stress`` file the builder prints after 'run 0': the log's
    thermo table quantizes each component to eight significant digits,
    and a step that did not write its stress file is a failed step, not a
    step whose stress is zero.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    lammps_units : str
        LAMMPS units style ('real' or 'metal').
    stress_path : str or None
        Path of the dedicated stress file for this run, or None when the
        run writes none.

    Returns
    -------
    np.ndarray
        The (1, 3, 3) stress tensor in bar.

    Raises
    ------
    RuntimeError
        When stress was asked for and the file is missing or malformed.
    """
    stress = np.zeros((1, 3, 3))
    if not getattr(config, 'stress', False):
        return stress
    if stress_path is None:
        raise RuntimeError(
            'stress was requested but the run writes no stress file; the '
            'builder decides the file names, so this is a wiring bug')
    components = parse_stress_file(stress_path)
    if components is None:
        raise RuntimeError(
            f"LAMMPS stress file {stress_path} is missing or holds no "
            f"'STRESS <xx> <yy> <zz> <xy> <xz> <yz>' line; the stress "
            f"tensor was requested and no other channel is read.")
    stress[0] = pressure6_to_stress_tensor(components, lammps_units)
    return stress


def read_energy_file(energy_path: str, log_path=None) -> float:
    """Read the single potential energy the run printed for itself.

    The writer ends every single-point step with
    ``variable pyrai_pe equal pe`` plus ``print "${pyrai_pe}"``, which
    carries fifteen significant digits where the thermo table in the log
    carries eight, so the file is the energy's only source and its
    absence is reported rather than worked around by scanning the log.

    Parameters
    ----------
    energy_path : str
        Path of the ``{title}.energy`` file.
    log_path : str, optional
        Run log to quote (its last lines) when the energy is unreadable.

    Returns
    -------
    float
        The potential energy in the run's native LAMMPS unit.

    Raises
    ------
    RuntimeError
        When the file is missing or holds no readable number.
    """
    if not os.path.exists(energy_path):
        raise RuntimeError(
            f'LAMMPS energy file not found: {energy_path}.'
            + _quote_log_tail(log_path))
    values = []
    with open(energy_path, 'r') as handle:
        for line in handle:
            tokens = line.split()
            for token in tokens:
                try:
                    values.append(float(token))
                except ValueError:
                    continue
    if not values:
        raise RuntimeError(
            f'LAMMPS energy file {energy_path} holds no number.'
            + _quote_log_tail(log_path))
    # print overwrites, so one value is expected; the last is taken to
    # stay readable even for a template that appended its own prints.
    return values[-1]


def _quote_log_tail(log_path) -> str:
    """Append the end of the run log to an error, when it can be read."""
    if not log_path:
        return ''
    tail = read_log_tail(log_path)
    return f'\nLAMMPS log tail:\n{tail}' if tail else ''


def _parse_force_dump(dump_path: str, natom: int) -> np.ndarray:
    """Parse the LAMMPS custom force dump file.

    The builder dumps 'fx fy fz' with 'modify sort id' after 'run 0'
    ('{title}.force'), so the rows of the frame are already
    ordered by atom id and carry no id column; the row order is used
    directly. Frames that include an id column are sorted by id
    defensively.

    Parameters
    ----------
    dump_path : str
        Path to the force dump file.
    natom : int
        Expected number of atoms.

    Returns
    -------
    np.ndarray
        Force array of shape (natom, 3) in native LAMMPS units,
        ordered by atom id.

    Raises
    ------
    RuntimeError
        If the dump file is missing or has no complete frame.
    """
    if not os.path.exists(dump_path):
        raise RuntimeError(
            f"LAMMPS force dump file not found: {dump_path}"
        )

    with open(dump_path, 'r') as f:
        lines = f.read().splitlines()

    # Split the dump into frames and keep the last complete one
    frame_starts = [
        n for n, line in enumerate(lines) if line.startswith('ITEM: ATOMS')
    ]
    if not frame_starts:
        raise RuntimeError(
            f"LAMMPS force dump file contains no data: {dump_path}"
        )

    header = lines[frame_starts[-1]].split()
    columns = header[2:]  # tokens after 'ITEM:' and 'ATOMS'
    id_column = columns.index('id') if 'id' in columns else None
    fx_column = columns.index('fx')
    fy_column = columns.index('fy')
    fz_column = columns.index('fz')

    records = []
    for line in lines[frame_starts[-1] + 1:]:
        tokens = line.split()
        if len(tokens) != len(columns):
            break
        atom_id = int(tokens[id_column]) if id_column is not None else len(records) + 1
        records.append((
            atom_id,
            float(tokens[fx_column]),
            float(tokens[fy_column]),
            float(tokens[fz_column]),
        ))

    if len(records) != natom:
        raise RuntimeError(
            f"LAMMPS force dump file {dump_path} has {len(records)} "
            f"atoms, but the system defines {natom}."
        )

    records.sort(key=lambda record: record[0])
    return np.array([record[1:] for record in records], dtype=float)


def _parse_xyz_dump(dump_path: str, natom: int) -> np.ndarray:
    """Parse a LAMMPS xyz dump file and return the last frame.

    Handles the 'write_dump all xyz' / 'dump ... xyz' output format:
    each frame is a natom line, a comment line (e.g. 'Atoms.
    Timestep: 100') and natom '<type> x y z' rows. The skill emits no
    'modify element' clause, so the leading column is the atom type
    id and is ignored; the frames are sorted by atom id ('modify sort
    id'), so the row order is the atom order.

    Parameters
    ----------
    dump_path : str
        Path to the xyz dump file.
    natom : int
        Expected number of atoms.

    Returns
    -------
    np.ndarray
        Coordinate array of shape (natom, 3) in Angstrom from the
        last complete frame.

    Raises
    ------
    RuntimeError
        If the dump file is missing or has no complete frame.
    """
    if not os.path.exists(dump_path):
        raise RuntimeError(
            f"LAMMPS xyz dump file not found: {dump_path}"
        )

    with open(dump_path, 'r') as f:
        lines = f.read().splitlines()

    last_frame = None
    n = 0
    while n < len(lines):
        if not lines[n].strip():
            n += 1
            continue
        try:
            frame_natom = int(lines[n].strip())
        except ValueError:
            n += 1
            continue
        if frame_natom != natom or n + 2 + natom > len(lines):
            n += 1
            continue
        # Skip the comment line and read the coordinate rows
        frame = []
        valid = True
        for line in lines[n + 2:n + 2 + natom]:
            tokens = line.split()
            if len(tokens) < 4:
                valid = False
                break
            try:
                frame.append([
                    float(tokens[-3]), float(tokens[-2]), float(tokens[-1])
                ])
            except ValueError:
                valid = False
                break
        if valid and len(frame) == natom:
            last_frame = frame
            n += 2 + natom
        else:
            n += 1

    if last_frame is None:
        raise RuntimeError(
            f"LAMMPS xyz dump file {dump_path} contains no complete "
            f"frame with {natom} atoms."
        )

    return np.array(last_frame, dtype=float)


def _parse_cond_dump(dump_path: str, natom: int) -> tuple:
    """Parse the LAMMPS cond dump file (final coordinates + velocities).

    The MD profiles dump the final structure and velocities with
    'write_dump all custom {project}_{profile}.cond id x y z vx vy vz
    modify sort id' (7 columns, no element information). The
    coordinates match the final xyz dump; the velocity columns feed
    the fin_velocity channel.

    Parameters
    ----------
    dump_path : str
        Path to the cond dump file.
    natom : int
        Expected number of atoms.

    Returns
    -------
    tuple of np.ndarray
        (coordinates, velocities), each of shape (natom, 3) in
        native LAMMPS units (Angstrom; Angstrom/fs for real,
        Angstrom/ps for metal), ordered by atom id.

    Raises
    ------
    RuntimeError
        If the dump file is missing, has no complete frame, or the
        atom count does not match.
    """
    if not os.path.exists(dump_path):
        raise RuntimeError(
            f"LAMMPS cond dump file not found: {dump_path}"
        )

    with open(dump_path, 'r') as f:
        lines = f.read().splitlines()

    frame_starts = [
        n for n, line in enumerate(lines) if line.startswith('ITEM: ATOMS')
    ]
    if not frame_starts:
        raise RuntimeError(
            f"LAMMPS cond dump file contains no data: {dump_path}"
        )

    header = lines[frame_starts[-1]].split()
    columns = header[2:]  # tokens after 'ITEM:' and 'ATOMS'
    try:
        id_column = columns.index('id') if 'id' in columns else None
        x_column = columns.index('x')
        y_column = columns.index('y')
        z_column = columns.index('z')
        vx_column = columns.index('vx')
        vy_column = columns.index('vy')
        vz_column = columns.index('vz')
    except ValueError:
        raise RuntimeError(
            f"LAMMPS cond dump file {dump_path} lacks the expected "
            f"id/x/y/z/vx/vy/vz columns: {columns}"
        )

    records = []
    for line in lines[frame_starts[-1] + 1:]:
        tokens = line.split()
        if len(tokens) != len(columns):
            break
        atom_id = int(tokens[id_column]) if id_column is not None else len(records) + 1
        records.append((
            atom_id,
            [float(tokens[x_column]), float(tokens[y_column]),
             float(tokens[z_column])],
            [float(tokens[vx_column]), float(tokens[vy_column]),
             float(tokens[vz_column])],
        ))

    if len(records) != natom:
        raise RuntimeError(
            f"LAMMPS cond dump file {dump_path} has {len(records)} "
            f"atoms, but the system defines {natom}."
        )

    records.sort(key=lambda record: record[0])
    coordinates = np.array([record[1] for record in records], dtype=float)
    velocities = np.array([record[2] for record in records], dtype=float)
    return coordinates, velocities


def parse_cond_cell(dump_path: str):
    """Read the simulation cell of the last frame of a custom dump.

    A LAMMPS text dump states its box as the *bounding* planes, which is
    the frame :func:`bound_and_tilt_to_lattice` inverts, and it writes that
    block whether or not the run was allowed to move it. For a fixed-cell
    profile the answer is therefore the cell the run started with, which the
    framework notices and skips; the value of reading it is the case that
    matters -- ``fix npt`` with cell degrees of freedom ends on a box the
    framework never asked for, and until now that box stayed in the file
    while the state went on claiming the old one.

    The tilt names are taken from the frame header (``ITEM: BOX BOUNDS
    xy xz yz pp pp pp``) rather than assumed positionally: LAMMPS lists
    them in the order their lines appear, and reading that order is what
    keeps a sheared box from being decoded as a different sheared box.

    Parameters
    ----------
    dump_path : str
        Path to the multi-frame custom dump.

    Returns
    -------
    list of list of float or None
        Lower-triangular (3, 3) lattice in Angstrom, or None when the file
        carries no box block at all (a dump written without a box).
    """
    with open(dump_path, encoding='utf-8', errors='replace') as handle:
        lines = handle.read().splitlines()

    frame = None
    for index, line in enumerate(lines):
        if line.startswith('ITEM: BOX BOUNDS'):
            frame = index
    if frame is None:
        return None

    header = lines[frame]
    tokens = header.split()
    # 'xy xz yz' style names come between 'BOUNDS' and the first flag.
    tilts_in_order = [name for name in tokens[3:] if not name.startswith('p')]
    numbers = []
    for offset in range(1, 4):
        if frame + offset >= len(lines):
            raise RuntimeError(
                f"LAMMPS dump file {dump_path} has a truncated BOX BOUNDS "
                f"block ({len(lines)} lines)."
            )
        numbers.append(lines[frame + offset].split())

    bounds = {'xlo': float(numbers[0][0]), 'xhi': float(numbers[0][1]),
              'ylo': float(numbers[1][0]), 'yhi': float(numbers[1][1]),
              'zlo': float(numbers[2][0]), 'zhi': float(numbers[2][1])}
    tilts = {'xy': 0.0, 'xz': 0.0, 'yz': 0.0}
    for position, name in enumerate(tilts_in_order):
        if name not in tilts:
            raise RuntimeError(
                f"LAMMPS dump file {dump_path} declares an unknown tilt "
                f"{name!r} in {header!r}."
            )
        tilts[name] = float(numbers[position][2])
    return bound_and_tilt_to_lattice(bounds, tilts)


def parse_output(config: dict, calcdir: str, workflow) -> dict:
    """Parse LAMMPS output files based on the calculation workflow.

    A single point reads its numbers from the files it printed for itself
    ({title}.energy, {title}.force, {title}.stress) and does not scan the
    log for them; the internal-iteration profiles keep the log as their
    energy channel, because there the thermo table is the trajectory output.
    The log follows the per-task dump output basename
    ('{project}_{profile}.log', '{project}_force.log' for
    ff_gradient, see service.utils.output_basename).
    For the internal-iteration workflows (ff_optimize and the MD
    profiles) the final structure is read from
    '{project}_{profile}.xyz' and returned through the fin_coord
    channel; the MD profiles additionally read the final velocities
    from '{project}_{profile}.cond', convert them back to Bohr/au and
    return them through the fin_velocity channel. The log energy is a
    per-atom average ('thermo_modify norm yes') and is multiplied by
    natom to recover the total energy. Results are converted to
    hartree (energy) and hartree/bohr (gradient) according to the
    lammps_units style.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    calcdir : str
        Calculation directory.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    dict
        Parsed results dictionary with 'energy' (hartree),
        'gradient' (hartree/bohr, shape (1, natom, 3)) and, for the
        internal-iteration workflows, 'fin_coord' (natom, 3,
        Angstrom); the MD profiles additionally carry 'fin_velocity'
        (natom, 3, Bohr/au).

    Raises
    ------
    RuntimeError
        If required output files are missing or malformed.
    """
    project = config.lammps_project
    natom = config.natom
    lammps_units = config.lammps_units
    title = output_basename(project, workflow.profile)
    names = step_data_files(title, gradient=workflow.needs_gradient,
                            stress=bool(getattr(config, 'stress', False)))

    log_path = os.path.join(calcdir, f'{title}.log')
    if not os.path.exists(log_path):
        raise RuntimeError(
            f"LAMMPS log file not found: {log_path}"
        )

    with open(log_path, 'r') as f:
        log_lines = f.read().splitlines()

    if workflow.is_optimization or workflow.is_md:
        # Internal iteration: LAMMPS ran a trajectory or a relaxation of
        # its own, so the only channel that holds every step is the log's
        # thermo table. It runs with 'thermo_modify norm yes', so the
        # per-atom average is multiplied back to a total energy.
        energy = parse_log(log_lines) * natom
    else:
        # Single point: the printed energy file is the source (see
        # read_energy_file); the log below is read for diagnosis only.
        energy = read_energy_file(os.path.join(calcdir, names['energy']),
                                 log_path)
    energy_native = float(energy)

    if workflow.is_optimization or workflow.is_md:
        # Internal iteration: the final structure comes from the
        # write_dump xyz output (fin_coord channel, Angstrom);
        # missing final geometry is a contract violation.
        xyz_path = os.path.join(
            calcdir, f'{project}_{workflow.profile}.xyz')
        fin_coord = _parse_xyz_dump(xyz_path, natom)
        results = {
            'energy': np.array([energy_native]),
            # Internal-iteration profiles run LAMMPS's own minimize/MD;
            # they do not hand a final gradient back through this channel.
            'gradient': to_native_gradient(np.zeros((natom, 3))),
            # The whole-system stress channel is only populated for the
            # single-point route, so this stays zero-filled.
            'stress_tensor': np.zeros((1, 3, 3)),
            'fin_coord': fin_coord,
        }
        # MD profiles additionally report the final velocities through
        # the fin_velocity channel (converted back to Bohr/au), and the
        # cell the box ended on through fin_cell: 'fix npt' carries the
        # cell as a degree of freedom, so the run can finish in a box the
        # framework never declared, and the coordinates it returned belong
        # to that box.
        if getattr(workflow, 'profile', '') == 'ff_optimize_cell':
            # An engine-owned box: read back what the relaxation stopped on.
            from .log_parser import parse_cell_file
            numbers = parse_cell_file(os.path.join(
                calcdir, f'{project}_{workflow.profile}.cell'))
            if numbers is None:
                raise ValueError(
                    'ff_optimize_cell moved a box the engine never reported: '
                    'no readable CELL line in %s, and keeping the framework\'s '
                    'request box would return coordinates that belong to no '
                    'declared cell' % project)
            lx, ly, lz, tilt_xy, tilt_xz, tilt_yz = numbers
            results['fin_cell'] = [lx, 0.0, 0.0,
                                   tilt_xy, ly, 0.0,
                                   tilt_xz, tilt_yz, lz]
        if workflow.is_md:
            cond_path = os.path.join(
                calcdir, f'{project}_{workflow.profile}.cond')
            _, velocities = _parse_cond_dump(cond_path, natom)
            velocity_factor = LAMMPS_VELOCITY_TO_AU[lammps_units]
            results['fin_velocity'] = velocities * velocity_factor
            cell_lattice = parse_cond_cell(cond_path)
            if cell_lattice is not None:
                # Flattened rows: the six cell parameters the state also
                # keeps are derived from these by the framework, which owns
                # that formula (a skill may not import it).
                results['fin_cell'] = [float(value)
                                       for row in cell_lattice
                                       for value in row]
        return results

    # Native engine forces; the framework conversion owns the sign and
    # the unit factor.
    forces = None
    if workflow.needs_gradient:
        force_path = os.path.join(calcdir, names['force'])
        forces = _parse_force_dump(force_path, natom)
    else:
        forces = np.zeros((natom, 3))

    results = {
        'energy': np.array([energy_native]),
        'gradient': to_native_gradient(forces),
        'stress_tensor': _build_stress_tensor(
            config, lammps_units,
            os.path.join(calcdir, names['stress'])
            if 'stress' in names else None),
    }

    return results


def to_native_gradient(forces):
    """The `gradient` label (dU/dR) of one native LAMMPS force array.

    LAMMPS dumps F = -dU/dR. The framework's `gradient` label is dU/dR, and
    the sign is part of what that label MEANS, not part of a unit
    conversion, so the skill applies it here and the framework only scales.
    The array stays in this run's native gradient unit (kcal/mol/Angstrom
    or eV/Angstrom).
    """
    return -np.asarray(forces, dtype=np.float64)


def format_results(results: dict, config: dict) -> dict:
    """Format parsed results for output.

    Converts numpy arrays to lists for JSON serialization.

    Parameters
    ----------
    results : dict
        Raw parsed results.
    config : dict
        Configuration dictionary.

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


def format_persistent_results(response: dict, config, workflow) -> dict:
    """Convert one resident-service reply into framework results.

    The resident path takes the numbers straight out of the LAMMPS
    interface instead of parsing a log, which changes two things and
    only two. The energy is already the total potential energy of the
    system, so the per-atom normalization the log parser reverses for
    the internal-iteration profiles ('thermo_modify norm yes') must not
    be applied here. And the force and pressure numbers arrive as arrays
    rather than as file contents, so there is nothing to locate. The
    unit conversion itself stays in this module, with the same factors,
    so the two paths cannot disagree about what a LAMMPS number means.

    Parameters
    ----------
    response : dict
        Validated RESULT message: ``energy_raw`` plus ``forces_raw`` and
        ``pressure6_raw`` when those capabilities were requested, all in
        native LAMMPS units.
    config : SkillConfig
        Runtime configuration (natom, lammps_units, stress).
    workflow : WorkflowGraph
        Compiled workflow graph (needs_gradient).

    Returns
    -------
    dict
        ``energy`` (1,) and ``gradient`` (natom, 3), both still in this
        run's native LAMMPS units (the caller marks the unit set), plus
        ``stress_tensor`` (1, 3, 3) in bar.
    """
    natom = int(config.natom)
    lammps_units = config.lammps_units

    energy = float(response['energy_raw'])

    if workflow.needs_gradient:
        forces = np.asarray(response['forces_raw'], dtype=float)
    else:
        forces = np.zeros((natom, 3))

    stress = np.zeros((1, 3, 3))
    if getattr(config, 'stress', False):
        stress[0] = pressure6_to_stress_tensor(
            response['pressure6_raw'], lammps_units)

    return {
        'energy': np.array([energy]),
        'gradient': to_native_gradient(forces),
        'stress_tensor': stress,
    }
