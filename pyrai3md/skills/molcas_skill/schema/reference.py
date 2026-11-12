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

This module provides reference data for the Molcas skill,
including periodic table, state labels, basis sets, and default parameters.

"""

# Spin state label mapping: 2S -> label
STATE_LABEL = {
    0: 's',
    1: 'd',
    2: 't',
    3: 'quan',
    4: 'quin',
    5: 'sext',
    6: 'sept',
    7: 'oct',
    8: 'non',
    9: 'dec',
    10: 'und',
}

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

# OpenMolcas supported basis sets (the list inherited from the example-input
# note; dev/skills/MOLCAS_SKILL_DEVELOPMENT.md section 5.8)
AVAILABLE_BASIS = [
    # Pople-style basis sets
    '3-21g',
    '4-31g',
    '5-21g',
    '6-31g',
    '6-31g**',
    'g-311g**',
    # ANO-S basis sets
    'ano-s-mb',
    'ano-s-vdzp',
    'ano-s-vtzp',
    # ANO-L basis sets
    'ano-l-mb',
    'ano-l-vdzp',
    'ano-l-vtzp',
    # ANO-RCC basis sets
    'ano-rcc-mb',
    'ano-rcc-vdzp',
    'ano-rcc-vtzp',
    # Dunning correlation-consistent basis sets
    'cc-pvdz',
    'cc-pvtz',
    'cc-pvqz',
    'aug-cc-pvdz',
    'aug-cc-pvtz',
    'aug-cc-pvqz',
    # Karlsruhe def2 basis sets
    'def2-svp',
    'def2-tzvp',
    'def2-tzvpp',
]

# Default convergence thresholds
DEFAULT_CONVERGENCE = {
    'scf_energy': 1e-8,
    'scf_density': 1e-6,
    'opt_energy': 1e-6,
    'opt_gradient': 3e-4,
    'opt_step': 100,
    'caspt2_imaginary_shift': 0.2,
    'caspt2_ipea_shift': 0.25,
}

# CASPT2 variants accepted by caspt2_type.
CASPT2_TYPES = ('ss', 'ms', 'xms', 'rms', 'xdw')

# Available calculation profiles
AVAILABLE_PROFILES = [
    'hf_energy',
    'casscf_energy',
    'casscf_gradient',
    'casscf_nac',
    'casscf_properties',
    'casscf_geometry_optimization',
    'casscf_abinitio_dynamics',
    'casscf_surface_hopping',
    'casscf_soc',
    'casscf_absorption',
    'casscf_frequency',
    'caspt2_energy',
    'caspt2_properties',
    'caspt2_absorption',
    'caspt2_frequency',
]

# Default CI parameters for singlet states in SOC dual RASSCF
SINGLET_DEFAULT = {
    'ci_roots': [3],
    'ci_spin': [0],
}

# Default CI parameters for triplet states in SOC dual RASSCF
TRIPLET_DEFAULT = {
    'ci_roots': [2],
    'ci_spin': [2],
}

# Default calculation parameters
DEFAULT = {
    'execution_mode': 'local',
    'shell': '',
    'project': 'molcas_calc',
    'profile': 'casscf_energy',
    'method': 'casscf',
    'basis': 'ano-rcc-vdzp',
    'charges': 0,
    'multiplicity': 1,
    'nstate': 1,
    # State information (from framework)
    'ci': [1],
    'spin': [0],
    'coupling': [],
    'nact_elec': 2,
    'nact_orb': 2,
    'ras1_orb': 0,
    'ras2_orb': 0,
    'ras3_orb': 0,
    'inactive_orb': 0,
    'active': [],
    'ci_roots': [1],
    'ci_spin': [0],
    'state_spin': [],
    # Electrostatic embedding (ESPF). Registered here so the classic
    # input carrier writes the two keywords into the skill section;
    # the &MOLECULE carrier writes them as well.
    'embedding': 'none',
    'read_charge': 'none',
    # Target states for gradient evaluation (0-based); empty falls
    # back to [0] during normalization.
    'target_states': [0],
    # CASPT2 multistate switch (xmultistate all); default enabled
    'xms': True,
    # Empty resolves from xms; accepted values are CASPT2_TYPES.
    'caspt2_type': '',
    'dwms_zeta': 1.0,
    'dwtype': 3,
    'nocsf': True,
    'caspt2_ipea_shift': None,
    'caspt2_real_shift': 0.0,
    'caspt2_imaginary_shift': DEFAULT_CONVERGENCE['caspt2_imaginary_shift'],
    'singlet_ci_roots': [3],
    'singlet_ci_spin': [0],
    'triplet_ci_roots': [2],
    'triplet_ci_spin': [2],
    'scf_converge': DEFAULT_CONVERGENCE['scf_energy'],
    'opt_converge': False,
    'opt_maxit': 100,
    # RASSCF ITERATIONS first number (max CI iterations); the second
    # number (super-CI orbital optimization) is fixed at 100.
    'maxit_scf': 200,
    # Optimization target states (0-based): single-state relaxation
    # when equal; same-spin pair triggers MECI, cross-spin pair MECP.
    'opt_state_i': 0,
    'opt_state_j': None,
    # Molecular dynamics: number of MD steps (FOREACH loop bound)
    # and the MOLCAS_MAXITER export limit (0 = auto, maxit_md x 10).
    'maxit_md': 1,
    'md_maxiter': 0,
    'molcas_project': 'MOLCAS',
    'molcas_mem': 4000,
    'molcas_nproc': 1,
    'molcas_print': 3,
    'molcas_path': '',
    'keep_tmp': 0,
    'verbose': 1,
    'solvent': 'gas',
    'temperature': 298.15,
    # Dynamics propagation (from framework &MD): timestep in au,
    # decoherence correction and thermostat ('off'/'nve' -> NVE,
    # 'nose'/'nose-hoover'/'nvt' -> Nose-Hoover NVT).
    'size': 20.67,
    'deco': '0.1',
    'thermostat': 'off',
    # Initial MD state (0-based); becomes RASSCF MDRLXR (1-based).
    'root': 1,
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
    'molcas_thread': 1,
    'nproc': 1,
}
