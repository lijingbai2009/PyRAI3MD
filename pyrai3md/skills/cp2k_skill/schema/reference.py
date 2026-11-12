"""
======================================================================
This is a standalone CP2K calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides reference data for the CP2K skill,
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

# Default element-level basis set / pseudopotential names filled by
# normalize when the user does not override them per element. The
# MOLOPT/ADMM names carry a -qN valence-electron suffix (demo
# verified, e.g. DZVP-MOLOPT-SR-GTH-q4 for C); elements outside the
# table fall back to the unsuffixed name and rely on the user
# override.
DEFAULT_BASIS = 'DZVP-MOLOPT-SR-GTH'
DEFAULT_AUX_BASIS = 'admm-dzp'
DEFAULT_POTENTIAL = 'GTH-PBE'
VALENCE_ELECTRON = {
    'H': 1, 'He': 2,
    'Li': 3, 'Be': 4, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
    'Na': 9, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8,
    'K': 9, 'Ca': 10,
    'Cu': 11, 'Zn': 12,
    'Br': 7, 'I': 7,
}

# Supported exchange-correlation functionals (validated in
# schema/validate.py); pbe0 is assembled as SCALE_X 0.75 + &HF
# FRACTION 0.25 following the verified bulk.inp demo layout.
AVAILABLE_FUNCTIONAL = [
    'pbe',
    'pbe0',
]

# Default convergence thresholds
DEFAULT_CONVERGENCE = {
    'scf_energy': 1e-6,
    'opt_energy': 1e-6,
    'opt_gradient': 3e-4,
    'opt_step': 100,
    'tddft': 1e-5,
}

# Available calculation profiles
AVAILABLE_PROFILES = [
    'dft_energy',
    'dft_gradient',
    'dft_opt_min',
    'dft_md',
    'tddft_energy',
    'tddft_gradient',
    'tddft_transition_dipole',
    'tddft_opt_min',
    'tddft_md',
]

# Grimme dispersion correction values accepted by the schema, mapped
# onto the &VDW_POTENTIAL POTENTIAL_TYPE keyword. REFERENCE_FUNCTIONAL
# always follows the calculation functional (validate enforces it).
# An empty string or 'none' disables the correction.
DISPERSION_KEYWORD = {
    'd3': 'DFTD3',
    'd3bj': 'DFTD3(BJ)',
}

# Stress evaluation routes.
# 'analytical' asks CP2K for STRESS_TENSOR ANALYTICAL: the ground-state
# (SCF) tensor, the route the engine prints and the one to prefer for a
# DFT run -- a numerical strain scan of the same box would cost twelve
# self-consistent field cycles for the quantity it already reports.
# 'num' is the external finite-difference strain route owned by this
# skill: no STRESS_TENSOR keyword is sent to CP2K at all, and the stress
# is built from the energies of +/- strain single points. That is what
# makes an excited-state (TDDFT) tensor reachable, since the engine
# supplies no tensor for a TDDFT run -- neither an analytical one
# (response_solver.F aborts) nor a numerical one (measured on the
# deployment, which is why no STRESS_TENSOR NUMERICAL route is offered
# here any more).
STRESS_METHODS = ('analytical', 'num')

# Strain scopes of the 'num' route: 'full' samples the six independent
# components of the tensor (twelve perturbed single points), 'iso'
# samples the uniform dilation only (two points) and reports the trace
# contracted onto the diagonal.
STRESS_SCOPES = ('full', 'iso')

# Default calculation parameters
DEFAULT = {
    'execution_mode': 'local',
    'shell': '',
    'project': 'cp2k_calc',
    'profile': 'dft_energy',
    'method': 'dft',
    'functional': 'pbe',
    # Plane-wave cutoffs (Ry): cutoff -> &MGRID CUTOFF, ref_cutoff ->
    # &MGRID REL_CUTOFF (written explicitly, never left to CP2K).
    'cutoff': 400.0,
    'ref_cutoff': 50.0,
    # Basis set / pseudopotential library files (multiple allowed,
    # e.g. BASIS_MOLOPT + BASIS_ADMM_UUZ for ADMM runs).
    'basis_file': ['BASIS_MOLOPT'],
    'potential_file': ['POTENTIAL'],
    'auxiliary_basis_file': [],
    # Element-level overrides (normalize fills defaults per atom):
    # {'H': 'DZVP-MOLOPT-SR-GTH', ...}
    'basis': {},
    'potential': {},
    'auxiliary_basis': {},
    # Grimme dispersion correction: d3/d3bj, empty or 'none'
    # disables &VDW_POTENTIAL.
    'dispersion': '',
    # Whole-system stress tensor request + evaluation method.
    # 'analytical' is the engine's ground-state tensor; 'num' is the
    # finite-difference strain route driven by this skill and the only
    # one that reaches an excited-state tensor. Empty means the route was
    # left to this skill, which answers from the method the run was given:
    # a TDDFT run has no analytical tensor to ask for, so it resolves to
    # 'num' and a ground-state run to 'analytical'. A name written down is
    # never overwritten - an explicit 'analytical' on a TDDFT run is still
    # refused by name.
    'stress': False,
    'stress_method': '',
    # 'num' strain step (dimensionless) and scope. The starting range
    # recommended in dev/skills/CP2K_SKILL_DEVELOPMENT.md section 6.6
    # (step size) is
    # 1e-4 ~ 1e-3; validate refuses a step outside (0, 1e-2].
    'stress_delta': 1.0e-3,
    'stress_scope': 'full',
    # Slab dipole correction (&DFT SURF_DIP / SURF_DIP_DIR). CP2K
    # implements it only for a slab whose normal is one Cartesian axis;
    # the direction is fixed to Z, the axis the skill's slab cells use.
    'dipole_correction': False,
    'charges': 0,
    'multiplicity': 1,
    'nstate': 1,
    # State information (from framework)
    'ci': [1],
    'spin': [0],
    'coupling': [],
    # TDDFPT settings
    'tddft_nstates': 0,       # computed by normalize from nstate/target_states
    'tddft_dipole_form': 'length',
    'tddft_maxiter': 100,
    'tddft_converge': DEFAULT_CONVERGENCE['tddft'],
    # Requested singlet/triplet root counts (set by normalize from
    # state_spin)
    'nsing_root': 0,
    'ntrip_root': 0,
    # Target states for gradient/optimization (0-based)
    'target_states': [0],
    'scf_converge': DEFAULT_CONVERGENCE['scf_energy'],
    # &SCF MAX_SCF limit
    'scf_maxit': 200,
    'opt_converge': False,
    'opt_maxit': 100,
    # Standalone &MOTION MD settings (driver=skill profiles only;
    # framework AIMD steps through dft_gradient instead)
    'md_nstep': 100,
    'md_timestep': 0.5,       # fs
    'md_ensemble': 'NVT',     # NVE / NVT (CSVR thermostat)
    'cp2k_project': 'CP2K',
    'cp2k_path': '',          # CP2K_HOME installation directory
    'cp2k_exe': '',           # executable; empty -> $CP2K_HOME/exe/local/cp2k.psmp
    'setup_script': '',       # optional toolchain install/setup script to source
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
    # Compute resources, two spellings each (#306): the bare name is what
    # a standalone input writes, the prefixed name what the framework
    # forwards from &cp2k; the larger of a pair wins.
    'thread': 1,
    'cp2k_thread': 1,
    'nproc': 1,
    'cp2k_nproc': 1,
}
