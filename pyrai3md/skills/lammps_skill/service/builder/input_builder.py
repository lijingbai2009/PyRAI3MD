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

This module builds LAMMPS input and data files from the
configuration dictionary and the compiled workflow.

The skill does not generate force field templates or topology files
(tooling to be added later). It reads the user-provided topology file
'{project}.data' from the launch directory, updates the coordinates
in the data file, and assembles the final input script according to
the compiled workflow. The optional user templates ('{project}.in',
'{project}.in.init', '{project}.in.settings') carry force-field
customizations; when the main template is absent the input script is
fully generated from the workflow:
  - single point (ff_energy/ff_gradient): the template body is kept
    (when present) and a 'run 0' evaluation section is appended
  - internal minimization (ff_optimize): the execution commands of
    the template body are dropped and a minimize section is generated
  - standalone MD (ff_nve/ff_nvt/ff_npt): the execution commands of
    the template body are dropped and a dynamics section is generated
    (thermo control, velocity initialization, timestep, ensemble fix,
    run, final write_dump outputs and restart data)

Dump output contract (element-independent)
------------------------------------------
The generated dumps never carry a 'modify element' clause: the atom
types of the data file may be redundant, so the element symbols are
owned by the user-provided data/framework, not by the dumps. Every
dump is sorted by atom id so the row order is the atom order:
  - final structure:    '{project}_{profile}.xyz'
  - final coordinates + velocities (MD only): '{project}_{profile}.cond'
    (custom dump with id x y z vx vy vz)
  - forces (ff_gradient): '{title}.force' (custom dump with
    fx fy fz)
'{project}.xyz' is reserved for the framework-provided initial
structure file (element + coordinates) used by read_dump templates.

Per-task file naming
--------------------
The generated input script shares the dump output basename so
different tasks in the same directory never overwrite each other:
'{project}_{profile}.in' ('{project}_force.in' for ff_gradient),
see service.utils.output_basename. The run script and the LAMMPS
log follow the same basename.

For the internal-iteration profiles the generated thermo control uses
'thermo_modify norm yes' (aligned with the reference inputs), so the
log energies are per-atom averages; the parser multiplies them back
by natom to recover the total energy for the framework.

Shake constraint switch (legacy-aligned unfix mechanism)
----------------------------------------------------------
The &lammps shake keywords (lammps_shake/shake_fix_id/shake_keywords)
are the single control source shared with the framework MD shake
constraint implementation. The builder strips every existing
'unfix {shake_fix_id}' line from the user templates, then applies
exactly one state decided by lammps_shake:
  - lammps_shake=0 (default, legacy behavior): the shake fix defined
    in the templates (e.g. TIP3P water) is disabled by appending a
    single 'unfix {shake_fix_id}' before 'run 0'.
  - lammps_shake=1 (synchronized with MD shake): no unfix is emitted,
    so the template shake fix stays active. If the templates define
    no such fix, the raw 'shake_keywords' fix command is injected
    instead (error if empty).

"""

import os
import re

from ..persistent.protocol import TILT_EPSILON
from ..utils import VELOCITY_TO_LAMMPS, output_basename, step_data_files
from .ff_model import (is_subsystem, model_path, render_data, render_init,
                       render_settings, save_model)

# Execution commands owned by the skill for the internal-iteration
# profiles (ff_optimize / ff_nve / ff_nvt / ff_npt). Lines of the
# user template starting with these commands are dropped so the
# generated sections stay the single source of execution control.
EXECUTION_COMMANDS = (
    'thermo ', 'thermo_style ', 'thermo_modify ',
    'dump ', 'dump_modify ', 'undump ',
    'write_dump ', 'write_data ',
    'velocity ', 'timestep ',
    'fix ', 'unfix ',
    'run ', 'minimize ', 'min_style ', 'min_modify ',
)


def _is_periodic(config) -> bool:
    """Whether this config's evaluation is periodic (pbc is the authority)."""
    return any(int(flag) for flag in (getattr(config, 'pbc', None) or []))


def _read_template(filename: str) -> str:
    """Read a user template file from the launch directory.

    Parameters
    ----------
    filename : str
        Template file name.

    Returns
    -------
    str
        Template content, empty string if the file does not exist.
    """
    if not os.path.exists(filename):
        return ''
    with open(filename, 'r') as f:
        return f.read()


def _extract_kspace_line(project: str, accuracy: str = '') -> str:
    """Return the kspace_style command from the '{project}.in.init' template.

    LAMMPS invalidates a kspace_style that was defined before read_data
    once the simulation box becomes triclinic (it aborts with "Must
    redefine kspace_style after changing to triclinic box"). A
    variable-cell (NPT) step rewrites the data-file box with tilt
    factors, so the kspace command has to be re-issued after read_data.
    This helper pulls the user's kspace_style line verbatim from the
    in.init template so it can be replayed in the generated header.

    The accuracy token is the one thing a caller may replace: the style
    name stays exactly what the template chose, because the two reciprocal
    styles are different methods and the skill has no business swapping
    them. What it does have to decide is how far the sum is carried - see
    `ewald_accuracy` in schema/reference.py.

    Parameters
    ----------
    project : str
        Template base name ('{project}.in.init').
    accuracy : str, optional
        Accuracy token to write in place of the template's. Empty, or a
        line carrying no token at all, replays the template unchanged.

    Returns
    -------
    str
        The kspace_style command (without trailing newline), or an
        empty string when the template defines none.
    """
    content = _read_template(f'{project}.in.init')
    for line in content.splitlines():
        if not line.strip().startswith('kspace_style'):
            continue
        if not accuracy:
            return line.strip()
        tokens = line.split()
        if len(tokens) < 3:
            return line.strip()
        tokens[2] = accuracy
        return ' '.join(tokens)
    return ''


def _reissue_kspace(model, project: str, accuracy: str = '') -> str:
    """The kspace line a step replays after read_data, '' for a subsystem.

    Both paths ask through this one function so the subprocess header and the
    resident package cannot drift apart. A subsystem (a cut-out center, see
    `atom_selection`) is evaluated non-periodically and its model already
    dropped the kspace declaration; re-issuing the user's template line after
    it would put the Ewald sum back exactly where the multi-center difference
    term must not carry one.
    """
    if model is not None and is_subsystem(model):
        return ''
    return _extract_kspace_line(project, accuracy)


def _strip_lines(content: str, shake_fix_id: str) -> str:
    """Strip include/read_data/unfix lines from template content.

    The include/read_data lines are re-injected by the builder in a
    fixed order, and the unfix lines are dropped so the shake switch
    becomes the sole source of the shake fix state.

    Parameters
    ----------
    content : str
        Template content.
    shake_fix_id : str
        Shake fix identifier (e.g. 'fShakeTIP3P').

    Returns
    -------
    str
        Filtered content.
    """
    kept = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith('include '):
            continue
        if stripped.startswith('read_data '):
            continue
        if shake_fix_id and stripped.startswith(f'unfix {shake_fix_id}'):
            continue
        kept.append(line)
    return '\n'.join(kept)


def _strip_execution_lines(content: str, shake_fix_id: str) -> str:
    """Strip include/read_data/unfix and execution lines from a template.

    Used by the internal-iteration profiles (ff_optimize and the MD
    profiles): the execution commands (thermo/dump/velocity/timestep/
    fix/run/minimize/...) are dropped from the user template body so
    the skill-generated sections stay the single source of execution
    control, while force-field related custom content is preserved.

    Parameters
    ----------
    content : str
        Template content.
    shake_fix_id : str
        Shake fix identifier (e.g. 'fShakeTIP3P').

    Returns
    -------
    str
        Filtered content.
    """
    kept = []
    shake_pattern = None
    if shake_fix_id:
        # Keep the shake fix definition so the shake switch stays
        # the sole source of the shake fix state (the fix lives in
        # the template, the unfix decision lives in _build_shake_section)
        shake_pattern = re.compile(
            rf'^\s*fix\s+{re.escape(shake_fix_id)}\s+.*shake'
        )
    for line in content.splitlines():
        stripped = line.strip()
        if shake_pattern and shake_pattern.match(stripped.split('#')[0]):
            kept.append(line)
            continue
        if any(stripped.startswith(command) for command in EXECUTION_COMMANDS):
            continue
        kept.append(line)
    return _strip_lines('\n'.join(kept), shake_fix_id)


def _find_shake_fix(contents: list, shake_fix_id: str) -> bool:
    """Check whether the templates define the shake fix.

    Scans all template contents for a 'fix {shake_fix_id} ... shake'
    definition, matching the legacy ffa/moltemplate convention where
    the TIP3P shake fix lives in '{project}.in.settings'.

    Parameters
    ----------
    contents : list of str
        Template contents to scan.
    shake_fix_id : str
        Shake fix identifier.

    Returns
    -------
    bool
        True if a shake fix definition is found.
    """
    if not shake_fix_id:
        return False
    pattern = re.compile(rf'^\s*fix\s+{re.escape(shake_fix_id)}\s+.*shake')
    for content in contents:
        for line in content.splitlines():
            stripped = line.split('#')[0]
            if pattern.match(stripped):
                return True
    return False


def _build_velocity_section(config: dict) -> str:
    """Build the Velocities section lines for the data file.

    Converts the framework velocities (Bohr/au) to the LAMMPS data
    file velocity units of the active lammps_units style (real:
    Angstrom/fs, metal: Angstrom/ps) and lists them by atom id.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    str
        Velocities section content (header + one line per atom),
        empty string when the system carries no velocities.
    """
    velocities = config.velocities
    if not velocities:
        return ''

    natom = config.natom
    if len(velocities) != natom:
        raise RuntimeError(
            f"System defines {len(velocities)} velocities, but "
            f"{natom} atoms."
        )

    factor = VELOCITY_TO_LAMMPS[config.lammps_units]
    lines = ['Velocities', '']
    for n, velocity in enumerate(velocities):
        vx, vy, vz = (float(component) * factor for component in velocity)
        lines.append(f'{n + 1} {vx:20.10f} {vy:20.10f} {vz:20.10f}')
    return '\n'.join(lines)


def _drop_velocity_section(lines: list) -> list:
    """Drop an existing Velocities section from data file lines.

    The section runs from the 'Velocities' header until the next
    section header (or end of file), so it can be replaced by the
    current velocities of the system.

    Parameters
    ----------
    lines : list of str
        Data file lines.

    Returns
    -------
    list of str
        Lines without the Velocities section.
    """
    result = []
    skipping = False
    for line in lines:
        first_token = line.split('#')[0].strip()
        if first_token == 'Velocities':
            skipping = True
            continue
        if skipping and first_token and not first_token[0].isdigit():
            # Reached the next section header
            skipping = False
        if not skipping:
            result.append(line)
    return result


# LAMMPS atom_style -> number of leading columns before x y z.
# id [mol] [type] [q] x y z (image flags optionally trail the coords).
_ATOM_STYLE_LEADING = {
    'atomic': 2,       # id type
    'charge': 3,       # id type q
    'molecular': 3,    # id mol type
    'bond': 3,         # id mol type
    'full': 4,         # id mol type q
}


def _is_int_token(token):
    """True when a token is a plain integer (image flags, ids)."""
    return re.fullmatch(r'[-+]?\d+', token.strip()) is not None


def _is_float_token(token):
    """True when a token parses as a float."""
    try:
        float(token)
        return True
    except (TypeError, ValueError):
        return False


def _resolve_periodic_lattice(config):
    """Return the (3, 3) lattice matrix when the run is periodic.

    pbc is the periodicity authority: the box is rewritten from the
    framework cell only when at least one axis is periodic. Returns
    None for a non-periodic run (the template box is then kept as-is).
    The lattice prefers lattice_vector and falls back to the 6 cell
    parameters; the framework convention is lower-triangular (a along
    x, b in the xy plane), which maps directly onto the LAMMPS box
    lengths and tilt factors.
    """
    pbc = [int(p) for p in (getattr(config, 'pbc', None) or [])]
    if not any(pbc):
        return None
    lattice = getattr(config, 'lattice_vector', None) or []
    flat = [float(x) for row in lattice for x in row] \
        if lattice and isinstance(lattice[0], (list, tuple)) else []
    if any(flat):
        return [[float(x) for x in row] for row in lattice]
    cell = list(getattr(config, 'cell', None) or [])
    if len(cell) == 6 and any(float(x) for x in cell):
        from math import radians, cos, sin, sqrt
        a, b, c = (float(cell[0]), float(cell[1]), float(cell[2]))
        al, be, ga = (radians(float(cell[3])), radians(float(cell[4])),
                      radians(float(cell[5])))
        ca, cb, cg, sg = cos(al), cos(be), cos(ga), sin(ga)
        sz2 = 1.0 + 2.0 * ca * cb * cg - ca ** 2 - cb ** 2 - cg ** 2
        sz = sqrt(max(sz2, 0.0))
        return [[a, 0.0, 0.0],
                [b * cg, b * sg, 0.0],
                [c * cb, c * (ca - cb * cg) / sg, c * sz / sg]]
    return None


def lattice_to_true_box(lattice, xlo: float, ylo: float,
                        zlo: float) -> dict:
    """Convert a framework lattice into the true LAMMPS domain box.

    The framework keeps a lower-triangular lattice (a along x, b in the
    xy plane), whose diagonal is the box length and whose off-diagonal
    elements are LAMMPS's tilt factors. With the origin kept at the
    template's lo planes this gives the numbers LAMMPS stores internally
    (``boxlo``/``boxhi`` plus ``xy``/``xz``/``yz``); the data file then
    states the *bounding* planes of the same box, which is what
    :func:`box_to_bound_lines` derives.

    Parameters
    ----------
    lattice : list of list of float
        (3, 3) lattice matrix.
    xlo, ylo, zlo : float
        Box origin kept from the data template.

    Returns
    -------
    dict
        ``{'xlo', 'ylo', 'zlo', 'xhi', 'yhi', 'zhi', 'xy', 'xz', 'yz'}``.

    Raises
    ------
    RuntimeError
        If a box length is not positive, or the lattice is not
        lower-triangular.
    """
    lx, ly, lz = float(lattice[0][0]), float(lattice[1][1]), \
        float(lattice[2][2])
    xy, xz, yz = float(lattice[1][0]), float(lattice[2][0]), \
        float(lattice[2][1])
    if lx <= 0.0 or ly <= 0.0 or lz <= 0.0:
        raise RuntimeError(
            f"LAMMPS: the periodic lattice diagonal must be positive "
            f"for the simulation box, got {lattice!r}")
    # The data-file box can only express the lower-triangular lattice
    # (a along x, b in the xy plane, tilt factors xy/xz/yz). The
    # framework re-triangularizes variable-cell commits
    # (workflow.integrator.verlet._scale_cell), so upper-triangular
    # components here mean the convention was violated upstream:
    # fail fast instead of silently dropping them and writing a box
    # that no longer matches the coordinates.
    upper = max(abs(float(lattice[0][1])), abs(float(lattice[0][2])),
                abs(float(lattice[1][2])))
    if upper > 1e-8 * max(lx, ly, lz):
        raise RuntimeError(
            f"LAMMPS data writer requires a lower-triangular lattice "
            f"(a along x, b in the xy plane); got upper-triangular "
            f"components ~{upper:.3e} in {lattice!r}.")

    return {'xlo': xlo, 'ylo': ylo, 'zlo': zlo,
            'xhi': xlo + lx, 'yhi': ylo + ly, 'zhi': zlo + lz,
            'xy': xy, 'xz': xz, 'yz': yz}


def box_to_bound_numbers(box: dict) -> dict:
    """A true box in the frame LAMMPS takes cell numbers in.

    A data file states the **bounding** planes, so the shear pulls each
    plane outward: ``xlo_bound = xlo + min(0, xy, xz, xy+xz)`` and
    ``xhi_bound = xhi + max(0, xy, xz, xy+xz)``, with only ``yz`` shifting
    y (the read_data convention). The tilt factors travel unchanged.

    The resident path sends these numbers rather than the true ones because
    that is the frame both of its consumers speak: ``read_data`` for the
    subprocess path, and ``change_box all <axis> final`` for a live domain,
    which echoes them back through ``extract_box`` untouched. Measured on
    23Jun2022u1, applying the true bounds of a tilted cell leaves the domain
    one shear short of the cell the subprocess path computed on, and the two
    energies differ by 0.04 kcal/mol at the very first tilted step.

    Parameters
    ----------
    box : dict
        True box numbers from :func:`lattice_to_true_box`.

    Returns
    -------
    dict
        The same nine keys, six of them pulled out to the bounding planes.
    """
    xy, xz, yz = box['xy'], box['xz'], box['yz']
    x_pull = (min(0.0, xy, xz, xy + xz), max(0.0, xy, xz, xy + xz))
    y_pull = (min(0.0, yz), max(0.0, yz))
    return {
        'xlo': box['xlo'] + x_pull[0], 'xhi': box['xhi'] + x_pull[1],
        'ylo': box['ylo'] + y_pull[0], 'yhi': box['yhi'] + y_pull[1],
        'zlo': box['zlo'], 'zhi': box['zhi'],
        'xy': xy, 'xz': xz, 'yz': yz,
    }


def box_to_bound_lines(box: dict) -> list:
    """Render a true box as the lines a LAMMPS data file carries.

    One rendering of :func:`box_to_bound_numbers`, so the file and the
    resident payload cannot drift apart in their box convention. The
    labelled tilt line is appended when the box is tilted, which is what
    makes a triclinic box round-trip through :func:`lattice_to_true_box`
    and back.

    Parameters
    ----------
    box : dict
        True box numbers from :func:`lattice_to_true_box`.

    Returns
    -------
    list of str
        Three bound lines, plus the 'xy xz yz' line when tilted.
    """
    bound = box_to_bound_numbers(box)
    lines = [
        f"{bound['xlo']:.10f} {bound['xhi']:.10f} xlo xhi",
        f"{bound['ylo']:.10f} {bound['yhi']:.10f} ylo yhi",
        f"{bound['zlo']:.10f} {bound['zhi']:.10f} zlo zhi",
    ]
    if any(abs(tilt) > TILT_EPSILON
           for tilt in (bound['xy'], bound['xz'], bound['yz'])):
        lines.append(f"{bound['xy']:.10f} {bound['xz']:.10f} "
                     f"{bound['yz']:.10f} xy xz yz")
    return lines


def bound_and_tilt_to_lattice(bounds: dict, tilts: dict) -> list:
    """Recover the framework lattice from a box stated as bounding planes.

    Inverse of :func:`lattice_to_true_box` followed by
    :func:`box_to_bound_numbers`, which is the frame everything LAMMPS
    *writes* uses: a data file's box lines and a dump's
    ``ITEM: BOX BOUNDS`` are the planes that enclose the sheared domain, not
    its true bounds. Pulling the shear back out therefore has to undo exactly
    the ``min``/``max`` the forward direction applied, and nothing else, so a
    cell that travelled out to the engine and back arrives unchanged.

    This exists because the engine can also *move* the box (``fix npt`` with
    cell degrees of freedom), and the framework needs that final cell back as
    a lattice, not as the plane pair it was printed in.

    Parameters
    ----------
    bounds : dict
        Bounding planes, with the keys ``xlo``/``xhi``/``ylo``/``yhi``/
        ``zlo``/``zhi``.
    tilts : dict
        Tilt factors ``xy``/``xz``/``yz`` in Angstrom.

    Returns
    -------
    list of list of float
        Lower-triangular (3, 3) lattice matrix: a along x, b in the xy plane.

    Raises
    ------
    RuntimeError
        If a recovered box length is not positive, which means the planes and
        the tilts do not describe one domain.
    """
    xy = float(tilts['xy'])
    xz = float(tilts['xz'])
    yz = float(tilts['yz'])
    x_pull = (min(0.0, xy, xz, xy + xz), max(0.0, xy, xz, xy + xz))
    y_pull = (min(0.0, yz), max(0.0, yz))
    lx = float(bounds['xhi']) - float(bounds['xlo']) - x_pull[1] + x_pull[0]
    ly = float(bounds['yhi']) - float(bounds['ylo']) - y_pull[1] + y_pull[0]
    lz = float(bounds['zhi']) - float(bounds['zlo'])
    if lx <= 0.0 or ly <= 0.0 or lz <= 0.0:
        raise RuntimeError(
            f"LAMMPS: the bounding planes {bounds!r} with tilts "
            f"{tilts!r} do not enclose a domain (recovered lengths "
            f"{lx:.6g} {ly:.6g} {lz:.6g}).")
    return [[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]]




def parse_template_box_origin(lines: list, data_template: str = ''):
    """Locate the three bound lines of a data file and their lo values.

    Parameters
    ----------
    lines : list of str
        Data file lines.
    data_template : str, optional
        File name to quote in the error of an incomplete box.

    Returns
    -------
    tuple of (tuple, dict)
        ``(xlo, ylo, zlo)`` and ``{label: (line_index, lo_value)}``.

    Raises
    ------
    RuntimeError
        When the file does not carry a complete box.
    """
    box_idx = {}
    for n, line in enumerate(lines):
        match = re.match(
            r'\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+(xlo xhi|ylo yhi|zlo zhi)\s*$',
            line)
        if match:
            box_idx[match.group(3)] = (n, float(match.group(1)))
    if len(box_idx) != 3:
        raise RuntimeError(
            f"LAMMPS data file {data_template} has no complete box "
            f"(xlo xhi / ylo yhi / zlo zhi); cannot apply the "
            f"framework cell.")
    origin = (box_idx['xlo xhi'][1], box_idx['ylo yhi'][1],
              box_idx['zlo zhi'][1])
    return origin, box_idx


# A subsystem's container: an orthorhombic box of this half-width in each
# direction, centred on the coordinate origin. The number is not physics - the
# model carries `boundary f f f` and no kspace command, so nothing in what the
# engine computes depends on it - and its only job is to hold the cluster:
# LAMMPS refuses an atom outside the bounds of a non-periodic run at
# `read_data` ("Did not assign all atoms correctly"), and with more than one MPI
# rank an atom outside the box is not merely refused at load time but lost from
# the domain decomposition ("Bond atoms 1 3 missing on proc 0", measured on the
# deployment).
_SUBSYSTEM_BOX_HALF = 50.0

_AXIS_LABELS = (('xlo xhi', 0), ('ylo yhi', 1), ('zlo zhi', 2))


def _subsystem_box(lines, coordinates, data_template):
    """Write the container a subsystem is evaluated in, and drop its tilt.

    The handed geometry is the continuous view of the region, so a molecule
    torn by a box face arrives with an atom a whole cell length away from the
    main image: any container tied to the template's bound lines is a guess
    about where the cluster will be, and the cluster moves every step. A fixed
    box around the origin is not a guess - a subsystem's coordinates stay
    within one cell length of the box they were cut out of, and 50 A is far
    more than any cell this framework carries - so the atoms never leave it,
    neither at load time nor under MPI.

    Parameters
    ----------
    lines : list of str
        Data file lines, as rendered for this step.
    coordinates : list of list of float
        The geometry the task was handed, (natom, 3).
    data_template : str
        File name to quote in an error.

    Returns
    -------
    list of str
        The same lines with the three bound lines replaced and any tilt line
        blanked (indices are preserved because a tilt line is emptied, not
        removed).

    Raises
    ------
    RuntimeError
        When a coordinate lies outside the container, which means the system
        is not the molecular-scale cluster this rule assumes.
    """
    box_idx = parse_template_box_origin(lines, data_template)[1]
    limit = _SUBSYSTEM_BOX_HALF
    for row in coordinates:
        for axis, value in enumerate(row[:3]):
            if abs(float(value)) > limit:
                raise RuntimeError(
                    f"LAMMPS: the subsystem handed to {data_template} has an "
                    f"atom at {float(value):.3f} along axis "
                    f"'xyz'[{axis}] outside the +/-{limit:.0f} A container "
                    f"a non-periodic run is evaluated in; that is no longer a "
                    f"molecular "
                    f"cluster of this cell, so the geometry is suspect rather "
                    f"than the box being grown")

    for label, axis in _AXIS_LABELS:
        lines[box_idx[label][0]] = (f'{-limit:.8f} {limit:.8f} {label}')

    # A tilt line describes shear, which a container with no periodicity and
    # no Ewald sum has no use for; blanking it in place keeps every line index
    # the style entries reference.
    zline = box_idx['zlo zhi'][0]
    if zline + 1 < len(lines):
        tokens = lines[zline + 1].split()
        tilted = (len(tokens) in (3, 6)
                  and all(_is_float_token(token) for token in tokens[:3])
                  and (len(tokens) == 3
                       or tokens[3:] == ['xy', 'xz', 'yz']))
        if tilted:
            lines[zline + 1] = ''
    return lines


def resolve_live_box(config, template_origin) -> dict:
    """This step's box, in the frame the engine takes cell numbers in.
    Shared by the data-file writer and the resident path, so the box a live
    LAMMPS instance is moved to is derived by exactly the same arithmetic
    the cold path writes into the data file. The numbers are the bounding
    planes and tilt factors of :func:`box_to_bound_numbers`, not the true
    bounds: ``change_box`` and ``read_data`` both take the cell in that
    frame, and a tilted step sent in true bounds lands on a different domain
    than the subprocess path computes on.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration (pbc, lattice_vector, cell).
    template_origin : tuple of float
        ``(xlo, ylo, zlo)`` kept from the data template.

    Returns
    -------
    dict or None
        Box numbers, or None for a non-periodic run.
    """
    lattice = _resolve_periodic_lattice(config)
    if lattice is None:
        return None
    return box_to_bound_numbers(lattice_to_true_box(lattice, *template_origin))


def _rewrite_data_box(lines, lattice, data_template):
    """Rewrite the simulation box lines from the framework lattice.

    Keeps the template's lo origin and sets hi = lo + length, so the
    box follows the current cell (a variable-cell NPT step resizes the
    box instead of keeping the stale template size). Triclinic tilt
    factors (xy, xz, yz) are written to the LAMMPS bounding-box
    convention; the 'xy xz yz' line is inserted after 'zlo zhi' when
    the cell is tilted and dropped when it becomes orthorhombic.
    """
    origin, box_idx = parse_template_box_origin(lines, data_template)
    box = lattice_to_true_box(lattice, *origin)
    rendered = box_to_bound_lines(box)

    lines[box_idx['xlo xhi'][0]] = rendered[0]
    lines[box_idx['ylo yhi'][0]] = rendered[1]
    lines[box_idx['zlo zhi'][0]] = rendered[2]

    # Tilt line: three floats immediately after 'zlo zhi', either
    # bare or self-labelled with the trailing 'xy xz yz' tokens (both
    # spellings are valid LAMMPS; the labelled form is written here
    # and both forms are detected so a triclinic template is rewritten
    # instead of getting a second tilt line inserted).
    zline = box_idx['zlo zhi'][0]
    tilt_idx = None
    nxt = zline + 1
    if nxt < len(lines):
        toks = lines[nxt].split()
        if (len(toks) == 6 and toks[3:] == ['xy', 'xz', 'yz']
                and all(_is_float_token(t) for t in toks[:3])):
            tilt_idx = nxt
        elif (len(toks) == 3 and all(_is_float_token(t) for t in toks)
                and not any(_is_int_token(t) for t in toks)):
            tilt_idx = nxt
    if len(rendered) > 3:
        if tilt_idx is not None:
            lines[tilt_idx] = rendered[3]
        else:
            lines.insert(zline + 1, rendered[3])
    elif tilt_idx is not None:
        lines.pop(tilt_idx)
    return lines


def _atom_coord_start(tokens, style, line, data_template):
    """Return (coord_start, n_image) for one Atoms line.

    coord_start is the index of the x column; n_image is the number of
    trailing image-flag columns to preserve (0 or 3). With a known
    atom_style the leading column count is exact; otherwise a heuristic
    is used (trailing integer tokens are image flags, the three columns
    before them are the coordinates). An ambiguous line raises instead
    of guessing (dev/methods/PBC_DEVELOPMENT.md section 3.3).
    """
    n = len(tokens)
    if style in _ATOM_STYLE_LEADING:
        lead = _ATOM_STYLE_LEADING[style]
        if n == lead + 3:
            return lead, 0
        if n == lead + 6 and all(_is_int_token(t) for t in tokens[lead + 3:]):
            return lead, 3
        raise RuntimeError(
            f"LAMMPS: cannot locate the coordinates in Atoms line for "
            f"atom_style '{style}' ({n} columns) in {data_template}: "
            f"{line!r}")
    # Heuristic: count trailing integer tokens (image flags come in 3).
    n_int = 0
    while n_int < n and _is_int_token(tokens[n - 1 - n_int]):
        n_int += 1
    if n_int not in (0, 3):
        raise RuntimeError(
            f"LAMMPS: ambiguous Atoms line in {data_template} "
            f"(trailing integer columns={n_int}): {line!r}. Add an "
            f"'# <atom_style>' comment to the Atoms section header so "
            f"the coordinate columns can be resolved unambiguously.")
    if n - n_int < 3:
        raise RuntimeError(
            f"LAMMPS: malformed Atoms line in {data_template} "
            f"(too few columns): {line!r}")
    return n - n_int - 3, n_int


def _image_rows(config: dict, natom: int):
    """Per-atom crossing counts to write, or None when not periodic.

    A periodic run states its images explicitly instead of letting
    read_data derive them from box containment: the two agree while every
    atom sits inside the cell, and only the explicit form stays right once
    an atom has crossed a face and come back. Both execution paths take
    the numbers from the same evaluation snapshot, which is what keeps the
    subprocess and resident results comparable. A non-periodic run has no
    images to declare and is left exactly as it was.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    natom : int
        Expected atom count of the run.

    Returns
    -------
    list of list of int or None
        (natom, 3) crossing counts, or None for a non-periodic run.
    """
    system = getattr(config, 'system', None)
    if not any(int(flag) for flag in (getattr(system, 'pbc', None) or [])):
        return None
    rows = getattr(system, 'image', None) or []
    if len(rows) != natom:
        # No crossings recorded yet (a fresh run, or a snapshot from
        # before the image channel): every atom is where it started.
        rows = [[0, 0, 0]] * natom
    return [[int(value) for value in row] for row in rows]


def _with_image_columns(tokens, n_image, image):
    """Write one atom's crossing counts into its trailing columns."""
    flags = ['%d' % int(value) for value in image]
    if n_image:
        return tokens[:-3] + flags
    return tokens + flags


def _rewrite_atoms(lines, atoms_start, style, coordinates, natom,
                   image_rows, template):
    """Rewrite the Atoms section's coordinate columns in place.

    Keyed on the atom id (id-1 is the framework index), so a template
    whose atoms are ordered differently from the framework still lands
    each coordinate on the right row. Every id must appear exactly once
    and the set must cover 1..natom: a partial map would silently
    displace one atom and leave another where the template put it.

    Parameters
    ----------
    lines : list of str
        Data-file lines; modified in place.
    atoms_start : int
        Index of the first Atoms row.
    style : str or None
        atom_style from the section header, when it carried one.
    coordinates : sequence
        Framework coordinates, (natom, 3).
    natom : int
        Atom count.
    image_rows : list or None
        Per-atom image flags to write, or None to keep the template's.
    template : str
        Template name, for the refusals.

    Raises
    ------
    RuntimeError
        On a malformed row, an id outside 1..natom, a duplicate id, or a
        map that does not cover every atom.
    """
    seen_ids = set()
    updated = 0
    for n in range(atoms_start, len(lines)):
        line = lines[n]
        if not line.strip():
            if updated > 0:
                break
            continue
        content, sep, comment = line.partition('#')
        tokens = content.split()
        if len(tokens) < 3:
            raise RuntimeError(
                f"Malformed Atoms line in {template}: {line}"
            )
        if not _is_int_token(tokens[0]):
            # Reached the next section header (non-numeric first token).
            if updated > 0:
                break
            raise RuntimeError(
                f"Malformed Atoms line in {template}: {line}"
            )
        atom_id = int(tokens[0])
        if not (1 <= atom_id <= natom):
            raise RuntimeError(
                f"LAMMPS: atom id {atom_id} in {template} is out of "
                f"range [1, {natom}]: {line!r}"
            )
        if atom_id in seen_ids:
            raise RuntimeError(
                f"LAMMPS: duplicate atom id {atom_id} in {template}: "
                f"{line!r}"
            )
        seen_ids.add(atom_id)

        start, n_image = _atom_coord_start(
            tokens, style, line, template)
        xyz = coordinates[atom_id - 1]
        tokens[start:start + 3] = [f'{float(xyz[0]):20.10f}',
                                   f'{float(xyz[1]):20.10f}',
                                   f'{float(xyz[2]):20.10f}']
        if image_rows is not None:
            tokens = _with_image_columns(tokens, n_image,
                                         image_rows[atom_id - 1])
        lines[n] = ' '.join(tokens) + (sep + comment if sep else '')
        updated += 1

    if updated != natom or seen_ids != set(range(1, natom + 1)):
        raise RuntimeError(
            f"LAMMPS data file {template} atom ids {sorted(seen_ids)} "
            f"do not map one-to-one onto 1..{natom} (the framework atom "
            f"order); found {updated} atom lines."
        )


def write_data_file(config: dict, calcdir: str, model=None):
    """Write the LAMMPS data file with updated box and coordinates.

    Reads the user-provided '{project}.data' topology template from the
    launch directory (the template itself is never modified, contract
    #142) and writes the per-step copy into calcdir. When a compressed
    force-field model is supplied, the template text is rendered from
    it instead, which carries the renumbered types and the pruned counts
    while leaving the box, coordinate, charge and velocity handling of
    this function untouched.

    - Box: when the run is periodic (pbc on) the simulation box is
      rebuilt from the framework lattice_vector, keeping the template's
      lo origin and following the current cell (so a variable-cell NPT
      step resizes the box instead of keeping the stale template size).
      Triclinic tilt factors are written to the LAMMPS bounding-box
      convention. A non-periodic run keeps the template box, except for a
      subsystem (`model['selection']`), which is given the fixed container
      :func:`_subsystem_box` describes.
    - Coordinates: each Atoms line is updated by atom id (id-1 is the
      framework index), replacing only the coordinate columns. The
      atom_style is read from the Atoms header comment when present;
      otherwise the coordinate columns are resolved by a heuristic that
      preserves any trailing image flags. id/mol/type/charge columns are
      preserved. A periodic run also carries the three trailing image
      columns, written from this step's snapshot (added when the template
      has none), so the engine is told which image each atom sits in
      rather than inferring it from box containment.

    When the system carries velocities (e.g. a continuing MD), an
    existing Velocities section is replaced; otherwise the template
    content stays untouched.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    calcdir : str
        Calculation directory.
    model : dict or None
        Compressed force-field model; None reads the template file.

    Raises
    ------
    RuntimeError
        If the data template is missing, the Atoms section or box is
        malformed, the atom ids do not map onto 1..natom, or a
        coordinate column cannot be resolved unambiguously.
    """
    project = config.lammps_project
    coordinates = config.coordinates
    natom = config.natom
    image_rows = _image_rows(config, natom)

    data_template = f'{project}.data'
    if model is not None:
        lines = render_data(model).splitlines()
    else:
        if not os.path.exists(data_template):
            raise RuntimeError(
                f"LAMMPS topology file not found: {data_template}"
            )

        with open(data_template, 'r') as f:
            lines = f.read().splitlines()

    # Rewrite the simulation box from the framework cell (periodic only)
    # BEFORE locating the Atoms section, since a tilt-line insertion
    # shifts the following line indices.
    lattice = _resolve_periodic_lattice(config)
    if lattice is not None:
        lines = _rewrite_data_box(lines, lattice, data_template)
    elif model is not None and is_subsystem(model):
        # A subsystem has no use for the template's box: the handed geometry is
        # the continuous view, so an atom of a face-straddling molecule sits
        # outside those bounds, and a non-periodic LAMMPS run refuses atoms
        # outside its box - under MPI it loses them from the domain
        # decomposition instead. It gets the fixed container, which the cluster
        # cannot walk out of mid-run.
        lines = _subsystem_box(lines, coordinates, data_template)

    # Locate the Atoms section header (may carry a style comment,
    # e.g. 'Atoms # full'), then skip the blank separator line.
    header_idx = -1
    for n, line in enumerate(lines):
        if line.split('#')[0].strip() == 'Atoms':
            header_idx = n
            break
    if header_idx < 0:
        raise RuntimeError(
            f"LAMMPS data file has no Atoms section: {data_template}"
        )
    style = None
    if '#' in lines[header_idx]:
        comment = lines[header_idx].split('#', 1)[1].strip().split()
        if comment:
            style = comment[0].lower()

    atoms_start = header_idx + 1
    if atoms_start < len(lines) and not lines[atoms_start].strip():
        atoms_start += 1

    # Replace the coordinate columns atom by atom, keyed on the atom id.
    _rewrite_atoms(lines, atoms_start, style, coordinates, natom,
                   image_rows, data_template)

    # Refresh the Velocities section when the system carries velocities
    velocity_section = _build_velocity_section(config)
    if velocity_section:
        lines = _drop_velocity_section(lines)
        while lines and not lines[-1].strip():
            lines.pop()
        lines.extend(['', velocity_section, ''])

    data_path = os.path.join(calcdir, f'{project}.data')
    with open(data_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_input_init_file(config: dict, calcdir: str, model=None):
    """Copy the optional '{project}.in.init' template to calcdir.

    With a compressed force-field model the content is rendered from it:
    the model owns the ``*_style hybrid`` declarations, which must lose
    every sub-style its trimmed coefficient table starves (LAMMPS aborts
    on an unused hybrid sub-style). Otherwise the template is copied.

    The unfix shake lines are stripped so the shake switch stays the
    sole source of the shake fix state.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    calcdir : str
        Calculation directory.
    model : dict or None
        Compressed force-field model; None keeps the verbatim template.
    """
    project = config.lammps_project
    shake_fix_id = config.shake_fix_id

    if model is not None:
        content = render_init(model)
    else:
        content = _read_template(f'{project}.in.init')
    if not content:
        return

    content = _strip_lines(content, shake_fix_id)
    init_path = os.path.join(calcdir, f'{project}.in.init')
    with open(init_path, 'w') as f:
        f.write(content + '\n')


def write_input_settings_file(config: dict, calcdir: str, model=None):
    """Copy the optional '{project}.in.settings' template to calcdir.

    The settings file typically carries the force field parameters
    and the legacy TIP3P shake fix definition
    ('fix fShakeTIP3P tip3p shake ...'). With a compressed force-field
    model the content is rendered from it, so the coefficients carry
    only the referenced types and the constraint declarations are
    emitted according to the single decision recorded in the model;
    otherwise the template is copied with the unfix lines stripped so
    the shake switch stays the sole source of the shake fix state.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    calcdir : str
        Calculation directory.
    model : dict or None
        Compressed force-field model; None copies the template.
    """
    project = config.lammps_project
    shake_fix_id = config.shake_fix_id

    if model is not None:
        content = render_settings(model)
    else:
        content = _read_template(f'{project}.in.settings')
        if not content:
            return
        content = _strip_lines(content, shake_fix_id)

    settings_path = os.path.join(calcdir, f'{project}.in.settings')
    with open(settings_path, 'w') as f:
        f.write(content if content.endswith('\n') else content + '\n')


def write_input_package(config: dict, target_dir: str, model=None):
    """Write the files a LAMMPS run loads, plus what it was rendered from.

    Every execution path assembles its launch directory here, so the
    topology, the ``*_style`` declarations and the coefficient table are
    always three renderings of the same model. Writing them one by one at
    each call site allowed a path to pass the model to two of the three
    writers, which is a silent physics change: the pruned settings drop
    the coefficient rows of a hybrid sub-style the system never uses,
    while a verbatim ``in.init`` still declares that sub-style, and
    LAMMPS aborts at run time with "Pair hybrid sub-style is not used".

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    target_dir : str
        Directory receiving the package.
    model : dict or None
        Compressed force-field model; None keeps the verbatim templates and
        writes no model file.
    """
    write_data_file(config, target_dir, model)
    write_input_init_file(config, target_dir, model)
    write_input_settings_file(config, target_dir, model)
    if model is not None:
        # The compressed model is part of the package: these four files are
        # one rendering, and keeping the source beside its output is what
        # lets a kept directory answer "what was the engine actually
        # given?". The backend holds the model in memory, so nothing reads
        # this back to run a step.
        save_model(model_path(config.lammps_project, target_dir), model)


def _build_shake_section(config: dict, calcdir: str, model=None) -> str:
    """Build the shake constraint section for the input script.

    Applies the &lammps shake switch to the templates (see module
    docstring for the legacy-aligned unfix mechanism). With a
    compressed force-field model the constraint state is already
    resolved there and the declarations travel with the rendered
    settings, so this section only has to inject the raw keywords when
    the model found no fix to enable.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    calcdir : str
        Calculation directory.
    model : dict or None
        Compressed force-field model; None keeps the unfix mechanism.

    Returns
    -------
    str
        Shake section lines (may be empty).

    Raises
    ------
    RuntimeError
        If lammps_shake=1 but no shake fix definition is found in
        the templates and shake_keywords is empty.
    """
    project = config.lammps_project
    lammps_shake = int(config.lammps_shake)
    shake_fix_id = config.shake_fix_id
    shake_keywords = str(config.shake_keywords).strip()

    if model is not None:
        constraints = model.get('constraints', {})
        if not constraints.get('enabled'):
            # Nothing was emitted into the settings, so there is no fix
            # to disable: no 'unfix' either.
            return ''
        if constraints.get('decl'):
            return ''
        if not shake_keywords:
            raise RuntimeError(
                f"lammps_shake=1 but no 'fix {shake_fix_id} ... shake' "
                f"definition was found in {project}.in.settings, and "
                f"shake_keywords is empty. Provide the raw fix command "
                f"via shake_keywords."
            )
        return f'{shake_keywords}\n'

    # Scan all template contents for the shake fix definition
    template_contents = [
        _read_template(f'{project}.in'),
        _read_template(f'{project}.in.init'),
        _read_template(f'{project}.in.settings'),
    ]
    has_shake_fix = _find_shake_fix(template_contents, shake_fix_id)

    section = ''
    if lammps_shake == 0:
        # Legacy behavior: disable the template shake fix via unfix.
        # No unfix is needed if the templates define no such fix.
        if has_shake_fix:
            section += f'unfix {shake_fix_id}\n'
    else:
        # Shake enabled (synchronized with the MD shake constraint):
        # the template fix stays active; inject shake_keywords only
        # if the templates define no shake fix at all.
        if not has_shake_fix:
            if not shake_keywords:
                raise RuntimeError(
                    f"lammps_shake=1 but no 'fix {shake_fix_id} ... "
                    f"shake' definition was found in {project}.in, "
                    f"{project}.in.init or {project}.in.settings, "
                    f"and shake_keywords is empty. Provide the raw "
                    f"fix command via shake_keywords."
                )
            section += f'{shake_keywords}\n'

    return section


def _build_thermo_section(config: dict) -> str:
    """Build the thermo output control for internal-iteration profiles.

    Aligned with the reference inputs: 'thermo_style multi' with
    per-atom normalization. The parser multiplies the normalized
    PotEng by natom to recover the total energy for the framework.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    str
        Thermo control lines.
    """
    dump_freq = int(config.dump_freq)
    return (
        f'thermo {dump_freq}\n'
        'thermo_style multi\n'
        'thermo_modify norm yes\n'
    )


def _build_optimize_section(config: dict, workflow) -> str:
    """Build the minimization section for the ff_optimize profile.

    Aligned with the reference min input: 'min_style cg' +
    'minimize etol ftol maxiter maxeval', followed by the final
    structure dump (element-independent, sorted by atom id) and the
    restart data.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    workflow : WorkflowGraph
        Compiled workflow graph (provides the profile token used in
        the dump file name).

    Returns
    -------
    str
        Minimization section lines.
    """
    project = config.lammps_project
    box_relax = str(getattr(config, 'box_relax', 'off') or 'off').strip().lower()

    section = _build_thermo_section(config)
    if box_relax in ('iso', 'aniso'):
        section += '\n'
        section += _box_relax_fix(config, box_relax)
    section += '\n'
    section += f'min_style {config.min_style}\n'
    section += (
        f'minimize {config.min_etol:g} {config.min_ftol:g} '
        f'{int(config.min_maxiter)} {int(config.min_maxeval)}\n'
    )
    section += '\n'
    section += (
        f'write_dump all xyz {project}_{workflow.profile}.xyz '
        f'modify sort id\n'
    )
    if box_relax in ('iso', 'aniso'):
        # What the engine ended the relaxation with, printed after `minimize` so
        # it is the answer box and not the request one. `print ... file`
        # overwrites, so the file holds this run's single line, and the parser
        # refuses to guess a box it cannot read.
        section += (
            'variable pyrai_lx equal lx\n'
            'variable pyrai_ly equal ly\n'
            'variable pyrai_lz equal lz\n'
            'variable pyrai_lxy equal xy\n'
            'variable pyrai_lxz equal xz\n'
            'variable pyrai_lzy equal yz\n'
            f'print "CELL ${{pyrai_lx}} ${{pyrai_ly}} ${{pyrai_lz}} '
            f'${{pyrai_lxy}} ${{pyrai_lxz}} ${{pyrai_lzy}}" '
            f'file {project}_{workflow.profile}.cell\n'
        )
        section += 'unfix PYRAI_BOX\n'
    section += 'write_data  restart.data\n'
    return section


# The largest relative box move one line search may take. The engine's own
# default is 1e-4, which was measured on the test build to hold a 100 kbar
# target to 0.01% of an edge over sixty line-search steps while `vmax 0.1` let
# the same target move it 4% (`CHANGELOG.md` #362) - a box relaxation written
# without this keyword spends its budget rather than reaching its answer.
BOX_RELAX_VMAX = 0.01


def _box_relax_fix(config, style):
    """The `fix box/relax` line, in the units the run declares.

    The grammar is the one this LAMMPS build accepts, measured rather than read
    off current documentation: `box/relax iso P` and `box/relax aniso P` take a
    single pressure value, `couple iso` / `pre` / `dilate none` do not exist, and
    the shear styles need a triclinic box (`CHANGELOG.md` #362). The pressure
    arrives in bar, the unit every pressure keyword in this package uses, and is
    converted to the engine's own unit for the declared `units` spelling.
    """
    pressure = float(getattr(config, 'box_relax_pressure', 0.0) or 0.0)
    if str(getattr(config, 'lammps_units', 'real')).strip().lower() == 'real':
        # `units real` reads a pressure as atm.
        pressure = pressure / BAR_PER_ATM
    return (f'fix PYRAI_BOX all box/relax {style} {pressure:g} '
            f'vmax {BOX_RELAX_VMAX:g}\n')


# Bar per atmosphere, stated here because a skill may not reach into the
# framework's unit table (`service/persistent/profile.py`).
BAR_PER_ATM = 1.01325


def _build_md_fix_section(config: dict, ensemble: str) -> str:
    """Build the ensemble fix lines for a standalone MD profile.

    Aligned with the reference nve/nvt/npt inputs:
      - nve: 'fix PYRAI_MD all nve'
      - nvt: 'fix PYRAI_MD all nvt temp {T} {T} {Tdamp}'
      - npt: 'fix PYRAI_MD all npt temp {T} {T} {Tdamp} x/y/z
              pressure controls drag 2.0' plus the recenter fix

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    ensemble : str
        MD ensemble token ('nve', 'nvt', 'npt').

    Returns
    -------
    str
        Fix command lines.
    """
    temperature = float(config.temperature)
    thermo_damp = float(config.thermo_damp)

    if ensemble == 'nve':
        return 'fix PYRAI_MD all nve\n'

    if ensemble == 'nvt':
        return (
            f'fix PYRAI_MD all nvt temp {temperature:g} '
            f'{temperature:g} {thermo_damp:g}\n'
        )

    pressure = float(config.pressure)
    pressure_damp = float(config.pressure_damp)
    # The cell coupling follows the run's barostat_mode, the same authority
    # the framework integrator reads when it drives the step itself:
    # 'isotropic' scales the volume as one scalar, 'anisotropic' lets the
    # three axes move on their own. The anisotropic form keeps being written
    # as three same-pressure axis clauses instead of the equivalent 'aniso'
    # keyword, so the text a run produced before the field existed is
    # reproduced byte for byte.
    mode = str(getattr(config, 'barostat_mode', '') or 'isotropic')
    isotropic = mode.strip().lower() != 'anisotropic'
    coupling = (f'iso {pressure:g} {pressure:g} {pressure_damp:g}'
                if isotropic else
                f'x {pressure:g} {pressure:g} {pressure_damp:g} '
                f'y {pressure:g} {pressure:g} {pressure_damp:g} '
                f'z {pressure:g} {pressure:g} {pressure_damp:g}')
    section = (
        f'fix PYRAI_MD all npt temp {temperature:g} {temperature:g} '
        f'{thermo_damp:g} {coupling} drag 2.0\n'
    )
    section += 'fix PYRAI_RECENTER all recenter INIT INIT INIT\n'
    return section


def _build_md_section(config: dict, workflow) -> str:
    """Build the dynamics section for the standalone MD profiles.

    Aligned with the reference nve/nvt/npt inputs: thermo control,
    velocity initialization (only when the data file carries no
    velocities), timestep, the ensemble fix, the run, and the final
    outputs. The final structure is dumped to
    '{project}_{profile}.xyz' and the final coordinates + velocities
    to the 7-column '{project}_{profile}.cond' custom dump; both are
    element-independent and sorted by atom id.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    workflow : WorkflowGraph
        Compiled workflow graph (carries the ensemble token and the
        profile token used in the dump file names).

    Returns
    -------
    str
        Dynamics section lines.
    """
    project = config.lammps_project

    section = _build_thermo_section(config)
    section += '\n'

    # Initial velocities: the data file carries them when the system
    # provides velocities; otherwise sample from the Maxwell-Boltzmann
    # distribution (aligned with the reference nve input)
    if not config.velocities:
        section += (
            f'velocity all create {float(config.temperature):g} '
            f'{int(config.md_seed)}\n'
        )
        section += '\n'

    section += f'timestep {float(config.md_timestep):g}\n'
    section += _build_md_fix_section(config, workflow.ensemble)
    section += '\n'
    section += f'run {int(config.md_nstep)}\n'
    section += 'unfix PYRAI_MD\n'
    section += '\n'
    section += (
        f'write_dump all xyz {project}_{workflow.profile}.xyz '
        f'modify sort id\n'
    )
    section += (
        f'write_dump all custom {project}_{workflow.profile}.cond '
        f'id x y z vx vy vz modify sort id\n'
    )
    section += 'write_data  restart.data\n'
    return section


def build_script_head(config: dict, workflow, calcdir: str, model=None, *,
                      internal: bool = False,
                      label: str = 'single-point section') -> str:
    """Build everything up to (but not including) the run commands.

    This is the text that loads the topology and the force field: the
    legacy-order includes and read_data, the kspace re-issue, the user
    template body (stripped of execution commands when the profile runs
    its own iteration) and the shake decision. Both execution styles
    consume it: the subprocess path appends the run tail and writes the
    result to disk, the resident path feeds it to a live LAMMPS instance
    once. Keeping a single producer is what lets the two paths be
    compared at all.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    workflow : WorkflowGraph
        Compiled workflow graph.
    calcdir : str
        Directory the shake section is built against.
    model : dict or None
        Compressed force-field model; None keeps the verbatim templates and
        writes no model file.
    internal : bool
        Strip the template's execution commands (internal-iteration
        profiles) instead of keeping them (single point).
    label : str
        Section comment naming the tail that follows.

    Returns
    -------
    str
        Script head text, ending after the shake section.
    """
    project = config.lammps_project
    shake_fix_id = config.shake_fix_id
    template = _read_template(f'{project}.in')

    # Header: load topology and settings in the legacy order
    header = ''
    if os.path.exists(f'{project}.in.init'):
        header += f'include "{project}.in.init"\n'
    # A subsystem carries no box. `boundary` is refused after the simulation
    # box exists (measured: "Boundary command after simulation box is
    # defined"), so it goes here, ahead of read_data, and it overrides the
    # periodic default the engine would otherwise apply to the template's box.
    if model is not None and is_subsystem(model):
        header += 'boundary f f f\n'
    header += f'read_data "{project}.data"\n'
    # Re-issue the kspace_style after read_data: a triclinic box (an
    # NPT variable-cell step rewrites the data-file box with tilt
    # factors) invalidates the kspace command set earlier in in.init,
    # and LAMMPS aborts unless it is redefined. Harmless for an
    # orthorhombic box (NVT / single point). Skipped for a subsystem,
    # which has no box to redefine one for. The style name is the
    # template's; only the accuracy token is replaced, by ewald_accuracy.
    kspace_line = _reissue_kspace(model, project, config.ewald_accuracy)
    if kspace_line:
        header += f'{kspace_line}\n'
    if model is not None or os.path.exists(f'{project}.in.settings'):
        header += f'include "{project}.in.settings"\n'
    header += '\n'

    if internal:
        body = _strip_execution_lines(template, shake_fix_id)
    else:
        body = _strip_lines(template, shake_fix_id)

    shake_section = _build_shake_section(config, calcdir, model)

    return (header + body + '\n'
            + f'\n# PyRAI3MD {label}\n' + shake_section)


def build_thermo_section(config: dict) -> str:
    """Build the single-point output control that is set up once.

    Declaring a compute is a setup command, not an evaluation, so the
    resident path keeps it in the loaded script and issues only
    :func:`build_run_section` per step.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    str
        The pressure compute (when stress is asked for), thermo style and
        thermo frequency.
    """
    # When the whole-system stress is requested, a virial-only pressure
    # compute is added; its six components (xx yy zz xy xz yz) are
    # surfaced both through thermo_style (log fallback) and, after
    # 'run 0', written to a dedicated file the parser reads
    # (dev/skills/STRESS_DEVELOPMENT.md section 4.1/4.2). 'NULL virial'
    # excludes the kinetic term, giving the potential (configurational)
    # pressure the parser turns into the tensile-positive stress.
    if getattr(config, 'stress', False):
        return (
            'compute Pconf all pressure NULL virial\n'
            'thermo_style custom step pe vol '
            'c_Pconf[1] c_Pconf[2] c_Pconf[3] '
            'c_Pconf[4] c_Pconf[5] c_Pconf[6]\n'
            'thermo 1\n'
        )
    return (
        'thermo_style custom step pe etotal\n'
        'thermo 1\n'
    )


def build_run_section(config: dict, workflow, *, files: bool = True) -> str:
    """Build the commands that evaluate the geometry currently loaded.

    This is the part of the single-point tail that actually runs and, when
    ``files`` is set, writes. The resident path issues it once per step,
    after that step's box and coordinates have been applied, and never
    twice in a row: the thermo setup stays in the loaded script, because
    re-declaring a compute the instance already owns is not what the
    subprocess path does either.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    workflow : WorkflowGraph
        Compiled workflow graph (single-point profile).
    files : bool
        Keep the lines naming an output file, which is what ``debug``
        decides on the resident path.

    Returns
    -------
    str
        The 'run 0' line and the writes that follow it.

    Notes
    -----
    ``run 0`` is expensive for a reason that has been measured rather than
    assumed: it is a run of zero steps, so what it answers with comes from the
    run's opening (``force->init_style``, the reciprocal-sum configuration, a
    full neighbour build), which a resident run therefore pays once per step -
    10001 ``PPPM initialization`` blocks for a 10000-step run whose cell never
    changed, 1.268 ms of a 4.20 ms step.

    Issuing ``run 1 pre no`` instead takes that evaluation from the step loop
    and skips the opening, and it works: 201 openings became 1, and the measured
    price of one line fell from 1.624 ms to 1.135 ms per step. It is not used,
    because the two lines are not the same calculation to the last bit. The
    resident instance keeps its neighbour list across steps and renews it
    incrementally, where every ``run 0`` rebuilds it, and the different order of
    additions shows up first at 1e-15 in kcal/mol and then, through a chaotic
    2 ps trajectory, at 1.26e-05 in potential energy: identical step 0, diverging
    from step 8268, and the byte-for-byte parity this path is judged on is gone.
    The whole step came to 1.116x, under the 1.15x adoption gate, so nothing was
    given up that was worth having. Measurements and the two refusals are in
    ``dev/skills/LAMMPS_SKILL_DEVELOPMENT.md`` section 11.12.

    Every per-step result is a text file of its own under the run's stem
    (``{title}.energy``, ``{title}.coords``, ``{title}.force``,
    ``{title}.stress``), which is what the parser reads back: the log is a
    diagnosis channel, not a data source. The energy is printed through an
    equal-style variable because the thermo table quantizes to eight
    significant digits while ``print`` of a variable keeps fifteen.
    Coordinates, forces and stress land in LAMMPS' own text dump format;
    this build writes per-atom numbers with ``%g`` and offers no legal way
    to widen it (``dump_modify format``, ``write_dump … modify format``,
    and the ``cdm``/``json`` styles are all refused), so the numeric text
    of those files carries six significant digits by the engine's choice.
    """
    project = config.lammps_project
    title = output_basename(project, workflow.profile)
    wants_stress = bool(getattr(config, 'stress', False))
    names = step_data_files(title, gradient=workflow.needs_gradient,
                            stress=wants_stress)

    run_section = 'run 0\n'
    if files:
        # Energy and geometry are written for every single-point profile,
        # gradient or not: they are what a person reads when a step is
        # wrong. print overwrites (measured), so a repeated calculate
        # leaves the last step, exactly as the dumps below do.
        run_section += (
            'variable pyrai_pe equal pe\n'
            f'print "${{pyrai_pe}}" file {names["energy"]}\n'
            f'write_dump all custom {names["coords"]} '
            f'x y z modify sort id\n'
        )
    # Force write_dump for gradient workflows: emitted after 'run 0'
    # so the forces are computed; no id column is dumped because
    # 'modify sort id' guarantees the row order is the atom order.
    if files and workflow.needs_gradient:
        run_section += (
            f'write_dump all custom {names["force"]} '
            f'fx fy fz modify sort id\n'
        )
    # Stress file write: a single-point 'run 0' evaluates the Pconf
    # compute, so the six global components are captured into equal-style
    # variables and appended to a dedicated file (more reliable than
    # fix ave/time, which needs a propagating trajectory).
    if files and wants_stress:
        run_section += (
            'variable pyrai_sxx equal c_Pconf[1]\n'
            'variable pyrai_syy equal c_Pconf[2]\n'
            'variable pyrai_szz equal c_Pconf[3]\n'
            'variable pyrai_sxy equal c_Pconf[4]\n'
            'variable pyrai_sxz equal c_Pconf[5]\n'
            'variable pyrai_syz equal c_Pconf[6]\n'
            f'print "STRESS ${{pyrai_sxx}} ${{pyrai_syy}} ${{pyrai_szz}} '
            f'${{pyrai_sxy}} ${{pyrai_sxz}} ${{pyrai_syz}}" '
            f'file {names["stress"]}\n'
        )

    return run_section


def build_step_tail(config: dict, workflow, *, files: bool = True) -> str:
    """Build the single-point evaluation tail.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    workflow : WorkflowGraph
        Compiled workflow graph (single-point profile).
    files : bool
        Keep the lines that name an output file. The subprocess path
        needs them (its parser reads those files); the resident path
        under ``python_enabled`` without ``debug`` extracts the same
        numbers through the LAMMPS interface and must not leave dumps
        behind.

    Returns
    -------
    str
        Thermo control plus :func:`build_run_section`, which is what the
        subprocess path appends to its input script.
    """
    return (build_thermo_section(config)
            + build_run_section(config, workflow, files=files))


def write_static_input_files(config: dict, workflow, package_dir: str,
                             model) -> dict:
    """Write the one-time input package a resident instance loads.

    The package is written once, at initialization, into the same
    directory and under the same names the subprocess path uses, and is
    then loaded by the service without running: the per-step evaluation
    comes from the returned ``step_tail``, issued after that step's box
    and coordinates have been applied. Coordinates and box travel over the
    socket afterwards, so the files are never rewritten per step. The
    package is produced by :func:`write_input_package`, the very call the
    subprocess path makes, so the loaded topology, declarations and
    coefficients are identical by construction.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    workflow : WorkflowGraph
        Compiled workflow graph (single point).
    package_dir : str
        Directory receiving the package: the backend's calculation
        directory, the same one the subprocess path writes into.
    model : dict
        Compressed force-field model; the resident path requires one.

    Returns
    -------
    dict
        ``{'script', 'kspace', 'step_tail', 'files', 'order', 'origin',
        'box', 'natom'}``: the script basename to load, the kspace line to
        re-issue after a domain conversion, the command lines to evaluate
        one step, the sha256 of every package file, the atom id order, the
        data-file box origin and this step's box as a data file states it.
    """
    import hashlib

    project = config.lammps_project
    basename = output_basename(project, workflow.profile)

    os.makedirs(package_dir, exist_ok=True)
    write_input_package(config, package_dir, model)

    # ``debug`` decides only whether a step also writes the files the
    # subprocess path writes. The script stops before the run section: the
    # service applies a box and a geometry first and then issues that
    # section, at initialization and at every step alike.
    debug = bool(getattr(config, 'debug', False))
    script = build_script_head(config, workflow, package_dir, model)
    script += build_thermo_section(config)
    with open(os.path.join(package_dir, f'{basename}.in'), 'w') as f:
        f.write(script)
    step_tail = build_run_section(config, workflow, files=debug).splitlines()

    with open(os.path.join(package_dir, f'{project}.data')) as f:
        data_lines = f.read().splitlines()
    origin, _box_idx = parse_template_box_origin(data_lines, f'{project}.data')

    # The digest covers exactly the files this package consists of. Walking
    # the directory instead would fold in whatever else a run leaves there
    # (a log, an earlier step's dump, the socket directory) and make the
    # session identity depend on unrelated history.
    names = (f'{project}.data', f'{project}.in.init',
             f'{project}.in.settings', f'{basename}.in')
    files = {}
    for name in sorted(names):
        path = os.path.join(package_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, 'rb') as f:
            files[name] = hashlib.sha256(f.read()).hexdigest()

    return {
        'script': basename,
        'kspace': _reissue_kspace(model, project, config.ewald_accuracy),
        'step_tail': step_tail,
        'files': files,
        # write_data_file already refuses anything but ids 1..natom in
        # one-to-one order, so the resident mapping needs no separate
        # id list.
        'order': list(range(1, int(config.natom) + 1)),
        'origin': list(origin),
        'box': resolve_live_box(config, origin),
        'natom': int(config.natom),
    }


def write_input_file(config: dict, workflow, calcdir: str, model=None):
    """Write the LAMMPS input script for the compiled workflow.

    Assembles the final '{project}_{profile}.in' ('{project}_force.in'
    for ff_gradient, see service.utils.output_basename) in calcdir:
      1. inject 'include in.init' / 'read_data' / 'include in.settings'
         at the top (legacy order), dropping the template's own
         include/read_data/unfix lines
      2. apply the shake switch (unfix mechanism, see module docstring)
      3. append the workflow-specific tail:
         - single point: 'thermo_style custom step pe etotal' +
           'thermo 1', 'run 0', optional force write_dump
         - ff_optimize: the generated minimize section (the template
           execution commands are dropped)
         - ff_nve/ff_nvt/ff_npt: the generated dynamics section (the
           template execution commands are dropped)

    The main input template is optional: when absent the input script is
    fully generated. The data file and the optional in.init/in.settings
    files are written alongside. The per-task input name never collides
    with the user template '{project}.in'.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    workflow : WorkflowGraph
        Compiled workflow graph.
    calcdir : str
        Calculation directory.
    model : dict or None
        Compressed force-field model; None keeps the verbatim templates and
        writes no model file.
    """
    project = config.lammps_project
    basename = output_basename(project, workflow.profile)

    # Write the topology and auxiliary templates into calcdir
    write_input_package(config, calcdir, model)

    if workflow.is_optimization or workflow.is_md:
        # Internal-iteration profiles: the execution commands of the
        # template body are dropped and the tail is fully generated
        if workflow.is_optimization:
            tail = _build_optimize_section(config, workflow)
            section_label = 'minimization section'
        else:
            tail = _build_md_section(config, workflow)
            section_label = f'{workflow.ensemble} dynamics section'

        head = build_script_head(config, workflow, calcdir, model,
                                 internal=True, label=section_label)
        input_path = os.path.join(calcdir, f'{basename}.in')
        with open(input_path, 'w') as f:
            f.write(head)
            f.write(tail)
        return

    # Single-point evaluation (ff_energy / ff_gradient)
    head = build_script_head(config, workflow, calcdir, model)
    input_path = os.path.join(calcdir, f'{basename}.in')
    with open(input_path, 'w') as f:
        f.write(head)
        f.write(build_step_tail(config, workflow))
