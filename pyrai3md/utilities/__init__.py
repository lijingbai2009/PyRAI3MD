"""
======================================================================
This module is the public API for the PyRAI3MD utilities package.
It exposes input parsing, structure formatting, the bond library,
geometry tools, lattice math, periodic image-flag derivation, read
tools, unit conversions, filesystem utilities, and the Timer class.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .io import (
    read_coord,
    read_charge,
    read_initcond,
    read_float_from_text,
    verify_xyz,
    string2float,
    reverse_string2float,
    complex2string,
    string2complex,
    lookup_amber,
    atomic_number,
    numerize_xyz,
    mark_atom,
)
from .structure_formatter import (
    print_coord,
    print_charges,
    print_matrix,
    print_cell,
    print_pbc,
    print_turbocoord,
)
from .bonds import bond_lib, derive_initial_image, perceive_bonds
from .geometric import BND, AGL, DHD
from .lattice import (
    LATTICE_EPS,
    cell6_to_matrix,
    matrix_to_cell6,
    cell_volume,
    normalize_pbc,
    boundary_condition,
    validate_lattice,
    cart_to_frac,
    frac_to_cart,
    wrap_fractional,
    wrap_fractional_with_image,
    unwrap_coordinates,
    min_image_displacement,
)
from .read_tools import ReadVal, ReadIndex
from .units import (
    BOHR_TO_ANGSTROM,
    ANGSTROM_TO_BOHR,
    AMU_TO_AU,
    AU_TO_AMU,
    HARTREE_TO_EV,
    EV_TO_HARTREE,
    HARTREE_TO_KCAL,
    KCAL_TO_HARTREE,
    CM_TO_HARTREE,
    HARTREE_TO_CM,
    NAC_ENERGY_GAP_EPS,
    bohr_to_angstrom,
    angstrom_to_bohr,
    amu_to_au,
    hartree_to_ev,
    ev_to_hartree,
)
from .timer import Timer
from .filesystem import ensure_dir, safe_copy, working_directory
from .logo import print_logo, __version__ as logo_version
