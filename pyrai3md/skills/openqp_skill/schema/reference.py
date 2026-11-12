"""
======================================================================
This is a standalone OpenQP calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides reference data for the OpenQP skill,
including periodic table, state labels, basis sets, functionals,
and default parameters.

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

# OpenQP supported basis sets (loaded from the openqp basis library)
AVAILABLE_BASIS = [
    # Pople-style basis sets
    '3-21g',
    '4-31g',
    '6-31g',
    '6-31g*',
    '6-31g**',
    '6-31+g*',
    '6-311g',
    '6-311g*',
    '6-311g**',
    # Dunning correlation-consistent basis sets
    'cc-pvdz',
    'cc-pvtz',
    'cc-pvqz',
    'aug-cc-pvdz',
    'aug-cc-pvtz',
    # Karlsruhe def2 basis sets
    'def2-svp',
    'def2-tzvp',
    'def2-tzvpp',
    # Minimal basis
    'sto-3g',
]

# OpenQP supported DFT functionals (passed to libxc by name).
# hf calculations use an empty functional.
AVAILABLE_FUNCTIONALS = [
    '',        # hf (no functional)
    'bhhlyp',
    'b3lyp',
    'b3lyp5',
    'pbe',
    'pbe0',
    'bp86',
    'blyp',
    'b97',
    'b97m',
    'm06',
    'wb97',
    'wb97x',
    'cam-b3lyp',
]

# Default convergence thresholds
DEFAULT_CONVERGENCE = {
    'scf_conv': 1e-6,
    'td_conv': 1e-6,
    'opt_rmsd_grad': 1e-4,
    'opt_max_grad': 3e-4,
    'hess_dx': 0.01,
}

# Available calculation profiles
AVAILABLE_PROFILES = [
    'hf_energy',
    'hf_gradient',
    'hf_geometry_optimization',
    'hf_frequency',
    'tddft_energy',
    'tddft_gradient',
    'tddft_geometry_optimization',
    'tddft_nonadiabatic_dynamics',
    'tddft_frequency',
    'mrsf_energy',
    'mrsf_gradient',
    'mrsf_nac',
    'mrsf_soc',
    'mrsf_soc_gradient',
    'mrsf_geometry_optimization',
    'mrsf_nonadiabatic_dynamics',
    'mrsf_frequency',
]

# Default calculation parameters
DEFAULT = {
    'execution_mode': 'local',
    'shell': '',
    'project': 'openqp_calc',
    'profile': 'tddft_energy',
    'method': 'tddft',
    'basis': '6-31g*',
    'functional': 'bhhlyp',
    'td_type': 'tda',
    'charges': 0,
    'multiplicity': 1,
    'nstate': 1,
    # State information (from framework)
    'ci': [1],
    'spin': [0],
    'coupling': [],
    # Which states the run evaluates, 0-based. Registered here because this
    # table is the only route a skill keyword takes into the framework's
    # generated config: a field declared by the schema but absent from this
    # dict is dropped by the converter and the run silently falls back to
    # state 0 (the `tddft_gradient` leg of tmp_20261110_093321 asked the
    # skill carrier for OpenQP `grad=2,3` and the framework carriers for
    # `grad=1`).
    'target_states': [0],
    # SCF settings
    'scf_type': '',      # derived by normalize (rhf/rohf/uhf)
    'scf_maxit': 100,
    'scf_converge': DEFAULT_CONVERGENCE['scf_conv'],
    # TD settings
    'td_maxit': 50,
    'td_converge': DEFAULT_CONVERGENCE['td_conv'],
    # Guess and NAC settings.
    # nac_type vocabulary: 'ktdc' (framework kTDC dynamics: per-step
    # energies + gradients only, runtype=data), 'dcm' (nacme through
    # MO overlap with the previous-step orbitals, runtype=prop),
    # 'nac' (numerical NAC vectors: the *_nac single-point profiles
    # drive runtype=nac, and inside a *_nonadiabatic_dynamics profile
    # the same request stays on runtype=data and adds
    # [properties] nac=nac).
    # Inside *_nonadiabatic_dynamics the default 'ktdc' keeps a missing
    # framework nactype on the energy+gradient path.
    'guess_type': 'auto',
    'nac_type': 'ktdc',
    # NAC vector settings. runtype=nac answers one state pair, the
    # runtype=data vector path answers every declared pair. The worker count
    # is not a deck setting: `openqp_nproc` gives the MPI ranks and
    # `thread`/`openqp_thread` the OpenMP threads per rank (see
    # RETIRED_RESOURCES in schema/adapter.py).
    'nac_dt': 20.67,
    'nac_dx': 0.0001,
    'nac_bp': True,
    # Spin-orbit settings.
    # 'dosoc' is the framework &md spelling forwarded into the skill
    # config; it is the request side of SOC and selects the two-manifold
    # run.  OpenQP refuses SOC together with any NAC flavour, so a request
    # carrying both is rejected at validate, not resolved here.
    # 'nstate_s' / 'nstate_t' size the singlet and triplet ladders; 0 means
    # derive from state_spin.  'soc_2e' is NOT registered here: an absent
    # key must stay distinguishable from an explicit 0, so the field
    # defaults to None and normalize maps None -> OpenQP's own default 1.
    'dosoc': 0,
    'nstate_s': 0,
    'nstate_t': 0,
    # Scalar-relativistic DKH order of the reference (OpenQP default 0).
    'scal_rel': 0,
    # Geometry optimization settings
    'opt_maxit': 100,
    'opt_optimizer': 'bfgs',
    'opt_rmsd_grad': DEFAULT_CONVERGENCE['opt_rmsd_grad'],
    'opt_max_grad': DEFAULT_CONVERGENCE['opt_max_grad'],
    # Hessian settings. No thread key: see the NAC note above.
    'hess_dx': DEFAULT_CONVERGENCE['hess_dx'],
    # OpenQP runtime
    'openqp_project': 'OQP',
    'openqp_path': '',
    'keep_tmp': 0,
    'verbose': 4,
    'temperature': 298.15,
    # MPI switch (default: no-MPI, OpenMP only)
    'use_mpi': False,
    'mpi_command': '',
    # Slurm settings
    'slurm_time': '01:00:00',
    'slurm_partition': 'default',
    'slurm_ntasks': 1,
    'slurm_memory': 4000,
    'slurm_command': '',
    # Compute resources (#306). `thread` keeps two spellings: the bare
    # name is what a standalone input writes, the prefixed name what the
    # framework forwards, and the larger of the pair wins. The rank count
    # has one spelling only (#427): `openqp_nproc`. Both `nproc` and the
    # old `mpi_nproc` are refused by name.
    'thread': 1,
    'openqp_thread': 1,
    'openqp_nproc': 1,
    # Framework-provided cell and spectroscopy defaults. The names and
    # values match the other skills: a keyword absent from this table is
    # dropped by the converter, and the skill then sees only its schema
    # default.
    'solvent': 'gas',
    'pbc': [0, 0, 0],
    'cell': [99.0, 99.0, 99.0, 90.0, 90.0, 90.0],
    'lattice_vector': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    'kpoint': [1, 1, 1],
    'ir_width': 100.0,
    'ir_scale': 1.0,
    'uv_width': 100.0,
    'nics_ring': [0],
    'nics_scan': [0.0],
}
