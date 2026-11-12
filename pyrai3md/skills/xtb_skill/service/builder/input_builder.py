"""
======================================================================
This is a standalone xTB calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module builds xTB input files and command-line arguments
from the configuration dictionary and the compiled workflow.

"""

import os

from ...schema.method import AVAILABLE_METHOD
from ...schema.method import method_supports_pbc

# Angstrom to Bohr conversion factor
ANG_TO_BOHR = 1.0 / 0.529177249


def _has_pbc(config) -> bool:
    """Check whether periodic boundary conditions are active.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.

    Returns
    -------
    bool
        True if any pbc flag is set.
    """
    pbc = config.pbc
    return any(int(x) != 0 for x in pbc)


def build_xtb_args(config, workflow) -> str:
    """Build the xTB command-line arguments from the config and workflow.

    Assembles the method flag (e.g. '--gfn 2' or '--gfnff'), the
    gradient flag, the total charge, and the number of unpaired
    electrons. Parallelization is controlled by OMP_NUM_THREADS in
    the run script templates.

    The '--grad' flag is always appended: the output parser reads
    the total energy from the engrad file for every profile, and
    xTB only writes that file when '--grad' is requested, so an
    energy-only run would otherwise fail at parse time.

    Parameters
    ----------
    config : SkillConfig
        Configuration container. Modified in place to store
        'xtb_args' for the run script templates.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    str
        Command-line argument string for the xTB executable.
    """
    method = config.method
    charges = config.charges
    uhf = config.uhf

    args = AVAILABLE_METHOD[method]['cli_flag']

    # The parser unconditionally reads the engrad file, which xTB
    # only writes for '--grad' runs; keep it on for energy profiles
    # as well.
    args += ' --grad'

    # Atomic charges are a returned property, not a diagnostic: the
    # framework's `computed_charges` channel reads xTB's `charges`
    # (GFN-x) or `gfnff_charges` (GFN-FF) file, which xTB only writes
    # when the population analysis is requested. ONIOM charge
    # embedding consumes those charges, so the request is unconditional
    # rather than a profile toggle.
    args += ' --pop'

    if charges != 0:
        args += f' --chrg {charges}'

    if uhf > 0:
        args += f' --uhf {uhf}'

    config.environment.xtb_args = args
    return args


def write_xyz_file(config, calcdir: str):
    """Write the XYZ coordinate file for xTB.

    The file is named $XTB_PROJECT.xyz and passed to xTB as the
    geometry input for molecular (non-periodic) calculations.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    calcdir : str
        Calculation directory.
    """
    project = config.xtb_project
    atoms = config.atoms
    coordinates = config.coordinates
    natom = len(atoms)

    xyz_path = os.path.join(calcdir, f'{project}.xyz')
    with open(xyz_path, 'w') as f:
        f.write(f'{natom}\n')
        f.write(f'{project} geometry\n')
        for atom, coord in zip(atoms, coordinates):
            f.write(f'{atom:>4s}  {coord[0]:16.10f}  {coord[1]:16.10f}  {coord[2]:16.10f}\n')


def write_coord_file(config, calcdir: str):
    """Write the Turbomole coord file for periodic xTB calculations.

    Periodic calculations (GFN-FF only) use the Turbomole coord
    format with coordinates in Bohr and an explicit lattice section.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    calcdir : str
        Calculation directory.
    """
    project = config.xtb_project
    atoms = config.atoms
    coordinates = config.coordinates
    pbc = config.pbc
    lattice_vector = config.lattice_vector

    coord = '$coord\n'
    for atom, xyz in zip(atoms, coordinates):
        coord += '%24.16f%24.16f%24.16f %-5s\n' % (
            float(xyz[0]) * ANG_TO_BOHR,
            float(xyz[1]) * ANG_TO_BOHR,
            float(xyz[2]) * ANG_TO_BOHR,
            atom,
        )

    periodic_dim = sum(int(x) != 0 for x in pbc)
    if periodic_dim > 0 and len(lattice_vector) == 3:
        coord += f'$periodic {periodic_dim}\n$lattice angstrom\n'
        for active, vector in zip(pbc, lattice_vector):
            if int(active) != 0:
                coord += '%24.16f%24.16f%24.16f\n' % (
                    float(vector[0]), float(vector[1]), float(vector[2])
                )

    coord += '$end\n'

    coord_path = os.path.join(calcdir, f'{project}.coord')
    with open(coord_path, 'w') as f:
        f.write(coord)


def write_embedding_file(config, calcdir: str) -> str:
    """Write the point-charge embedding file for xTB.

    Background point charges [q, x, y, z] are written to
    $XTB_PROJECT.pc in Bohr units, and the corresponding
    $embedding section is returned for the inp file.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    calcdir : str
        Calculation directory.

    Returns
    -------
    str
        The $embedding inp section, empty if no point charges.
    """
    project = config.xtb_project
    point_charges = config.point_charges

    if len(point_charges) == 0:
        return ''

    pc_path = os.path.join(calcdir, f'{project}.pc')
    with open(pc_path, 'w') as f:
        f.write(f'{len(point_charges)}\n')
        for line in point_charges:
            q, x, y, z = line
            f.write('%24.16f%24.16f%24.16f%24.16f\n' % (
                float(q),
                float(x) * ANG_TO_BOHR,
                float(y) * ANG_TO_BOHR,
                float(z) * ANG_TO_BOHR,
            ))

    return f'$embedding\ninput={project}.pc\n$end\n'


def write_input_file(config, workflow, calcdir: str):
    """Write the xTB geometry and input files.

    Selects the geometry format (Turbomole coord for periodic GFN-FF,
    XYZ otherwise), builds the command-line arguments, and assembles
    the inp file from an optional user template and the embedding
    section.

    An optional user template '<project>.inp' in the launch directory
    is prepended, allowing extra xTB keywords (e.g. solvent models).

    Parameters
    ----------
    config : SkillConfig
        Configuration container. Modified in place to store
        'xtb_args' and 'geometry_file' for the run script templates.
    workflow : WorkflowGraph
        Compiled workflow graph.
    calcdir : str
        Calculation directory.
    """
    project = config.xtb_project

    # Write geometry file according to the periodicity
    if _has_pbc(config):
        if not method_supports_pbc(config.method):
            # The schema asks this question too, but it can only see a `pbc`
            # the input wrote on the skill itself. A framework run carries the
            # cell on the system and hands it over per evaluation
            # (backend.py:205-215), so this is where the cell actually becomes
            # geometry: measured in the test environment, an unguarded run asked
            # GFN2-xTB for a periodic cell, the coord file was written with
            # `$periodic 3`, and xtb 6.7.1 stopped inside its own SCF with
            # `scf: Multipoles not available with PBC`.
            raise ValueError(
                f"xTB: method '{config.method}' does not support pbc, but a "
                f"periodic cell reached this run; only the methods marked "
                f"periodic in schema/method.py write a cell xTB accepts")
        write_coord_file(config, calcdir)
        config.environment.geometry_file = f'{project}.coord'
    else:
        write_xyz_file(config, calcdir)
        config.environment.geometry_file = f'{project}.xyz'

    # Build command-line arguments for the run script
    build_xtb_args(config, workflow)

    # Read optional user template from the launch directory
    template_path = f'{project}.inp'
    template = ''
    if os.path.exists(template_path):
        with open(template_path, 'r') as f:
            template = f.read()

    # Append the point-charge embedding section
    embedding_section = write_embedding_file(config, calcdir)

    inp_path = os.path.join(calcdir, f'{project}.inp')
    with open(inp_path, 'w') as f:
        f.write(template)
        f.write(embedding_section)
