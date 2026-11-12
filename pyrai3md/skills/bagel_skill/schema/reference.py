"""
======================================================================
This is a standalone BAGEL calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides reference data for the BAGEL skill,
including periodic table, basis sets, and default parameters.

"""

# Periodic table: element symbols with their atomic numbers (electrons)
PERIODIC_TABLE = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20,
    21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30,
    31: 31, 32: 32, 33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40,
    41: 41, 42: 42, 43: 43, 44: 44, 45: 45, 46: 46, 47: 47, 48: 48, 49: 49, 50: 50,
    51: 51, 52: 52, 53: 53, 54: 54, 55: 55, 56: 56, 57: 57, 58: 58, 59: 59, 60: 60,
    61: 61, 62: 62, 63: 63, 64: 64, 65: 65, 66: 66, 67: 67, 68: 68, 69: 69, 70: 70,
    71: 71, 72: 72, 73: 73, 74: 74, 75: 75, 76: 76, 77: 77, 78: 78, 79: 79, 80: 80,
    81: 81, 82: 82, 83: 83, 84: 84, 85: 85, 86: 86,
}

# BAGEL supported basis sets (cluster-validated list)
AVAILABLE_BASIS = [
    'sto-3g',
    '3-21g',
    '6-31g',
    'svp',
    'tzvpp',
    'qzvpp',
    # Dunning correlation-consistent basis sets
    'cc-pvdz',
    'cc-pvtz',
    'cc-pvqz',
    'cc-pv5z',
    'cc-pv6z',
    # Core-valence variants
    'cc-pcvdz',
    'cc-pcvtz',
    'cc-pcvqz',
    'cc-pcv5z',
    'cc-pcvdz-dk',
    'cc-pcvtz-dk',
    # Augmented variants
    'aug-cc-pvdz',
    'aug-cc-pvtz',
    'aug-cc-pvqz',
    'aug-cc-pv5z',
    'aug-cc-pv6z',
    'aug-cc-pcvdz',
    'aug-cc-pcvtz',
    'aug-cc-pcvqz',
    'aug-cc-pcv5z',
    'aug-cc-pcvdz-dk',
    'aug-cc-pcvtz-dk',
    'aug-cc-pcvqz-dk',
    # Weighted core-valence variants
    'aug-cc-pwcvdz',
    'aug-cc-pwcvtz',
    'aug-cc-pwcvqz',
    'aug-cc-pwcv5z',
    # Doubly augmented variants
    'd-aug-cc-pvdz',
    'd-aug-cc-pvtz',
    'd-aug-cc-pvqz',
    'd-aug-cc-pv5z',
    # ANO basis set
    'ano-rcc',
]

# BAGEL supported density-fitting basis sets. An empty df_basis
# config value is resolved by similarity matching against this list
# (see input_builder._match_df_basis), defaulting to svp-jkfit.
AVAILABLE_DF_BASIS = [
    'svp-jkfit',
    'tzvpp-jkfit',
    'qzvpp-jkfit',
    'cc-pvdz-jkfit',
    'cc-pvtz-jkfit',
    'cc-pvqz-jkfit',
    'cc-pv5z-jkfit',
]

# Default convergence thresholds
DEFAULT_CONVERGENCE = {
    'scf_energy': 1e-8,
    'maxit_scf': 500,
    'maxit_opt': 100,
}

# Available calculation profiles
AVAILABLE_PROFILES = [
    'casscf_energy',
    'casscf_gradient',
    'casscf_nac',
    'casscf_frequency',
    'casscf_geometry_optimization',
    'casscf_dipole',
    'casscf_transition_dipole',
    'xmscaspt2_energy',
    'xmscaspt2_gradient',
    'xmscaspt2_nac',
    'xmscaspt2_frequency',
    'xmscaspt2_geometry_optimization',
    'xmscaspt2_dipole',
    'xmscaspt2_transition_dipole',
]

# Default calculation parameters
DEFAULT = {
    'execution_mode': 'local',
    'shell': '',
    'project': 'bagel_calc',
    'profile': 'casscf_energy',
    'method': 'casscf',
    'basis': 'cc-pvdz',
    'df_basis': '',
    'charges': 0,
    'multiplicity': 1,
    'nstate': 1,
    # State information (from framework)
    'ci': [1],
    'spin': [0],
    'coupling': [],
    'nact_elec': 2,
    'nact_orb': 2,
    'nclosed': 0,
    'active': [],
    'ci_roots': [1],
    'ci_spin': [0],
    'target_states': [0],
    'target_nac_pair': [[]],
    # Optimization target states (0-based): equal values relax a
    # single state (opttype energy); a distinct pair optimizes their
    # minimum-energy conical intersection (opttype meci).
    'opt_state_i': 0,
    'opt_state_j': None,
    'scf_converge': DEFAULT_CONVERGENCE['scf_energy'],
    # XMS-CASPT2 level shift feeding the smith 'shift' keyword.
    'pt2_shift': 0.2,
    'maxit_scf': DEFAULT_CONVERGENCE['maxit_scf'],
    'maxit_opt': DEFAULT_CONVERGENCE['maxit_opt'],
    'bagel_project': 'BAGEL',
    'bagel_path': '',
    'bagel_nproc': 1,
    'blas': '',
    'lapack': '',
    'boost': '',
    'mpi': '',
    # MPI launcher executable; every skill that starts an MPI job names
    # the same launcher (mpirun), and no/none runs BAGEL directly with
    # multi-threading instead of MPI.
    'mpiexe': 'mpirun',
    'mkl': '',
    'arch': '',
    'keep_tmp': 0,
    'verbose': 4,
    'solvent': 'gas',
    'temperature': 298.15,
    # External potential (from framework)
    'shape': 'ellipsoid',
    'factor': 40,
    'scale': 1.0,
    # Periodic settings (from framework)
    'pbc': [0, 0, 0],
    'cell': [99.0, 99.0, 99.0, 90.0, 90.0, 90.0],
    'lattice_vector': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    'kpoint': [1, 1, 1],
    # Spectroscopy (from framework)
    'ir_width': 100.0,
    'ir_scale': 1.0,
    'uv_width': 100.0,
    'nics_ring': [0],
    'nics_scan': [0.0],
    'slurm_time': '01:00:00',
    'slurm_partition': 'default',
    'slurm_ntasks': 1,
    'slurm_memory': 4000,
    'slurm_command': '',
    # Compute resources, two spellings each (#306): the bare name is
    # what a standalone input writes, the prefixed name what the
    # framework forwards; the larger of a pair wins.
    'thread': 1,
    'bagel_thread': 1,
    'nproc': 1,
}
