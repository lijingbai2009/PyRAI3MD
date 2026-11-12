"""
======================================================================
This is a standalone quantum chemistry calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 30, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module defines the available methods, tasks, and profile mappings.
This is the single source of truth for method-task compatibility.

"""

# Profile to method mapping
PROFILE_METHOD_MAP = {
    # HF profiles
    'hf_energy': 'hf',
    'hf_gradient': 'hf',
    'hf_hessian': 'hf',
    'hf_opt_min': 'hf',
    'hf_opt_ts': 'hf',
    'hf_mo': 'hf',
    'hf_esp': 'hf',
    'hf_population': 'hf',
    'hf_ir': 'hf',
    'hf_nmr': 'hf',
    # DFT profiles
    'dft_energy': 'dft',
    'dft_gradient': 'dft',
    'dft_hessian': 'dft',
    'dft_opt_min': 'dft',
    'dft_opt_ts': 'dft',
    'dft_mo': 'dft',
    'dft_esp': 'dft',
    'dft_population': 'dft',
    'dft_ir': 'dft',
    'dft_nmr': 'dft',
    # MP2 profiles
    'mp2_energy': 'mp2',
    'mp2_gradient': 'mp2',
    'mp2_mo': 'mp2',
    'mp2_esp': 'mp2',
    'mp2_population': 'mp2',
    # TDDFT profiles
    'tddft_energy': 'tddft',
    'tddft_gradient': 'tddft',
    'tddft_opt_min': 'tddft',
    'tddft_uv_vis': 'tddft',
    'tddft_mo': 'tddft',
    'tddft_esp': 'tddft',
    'tddft_population': 'tddft',
    # TDA profiles
    'tda_energy': 'tda',
    'tda_gradient': 'tda',
    'tda_opt_min': 'tda',
    'tda_uv_vis': 'tda',
    'tda_mo': 'tda',
    'tda_esp': 'tda',
    'tda_population': 'tda',
}

# Profile to step chain mapping
PROFILE_STEP_MAP = {
    # HF profiles
    'hf_energy': ['molecule', 'reference', 'results'],
    'hf_gradient': ['molecule', 'reference', 'gradient', 'results'],
    'hf_hessian': ['molecule', 'reference', 'hessian', 'results'],
    'hf_opt_min': ['molecule', 'reference', 'optimize', 'results'],
    'hf_opt_ts': ['molecule', 'reference', 'optimize_ts', 'results'],
    'hf_mo': ['molecule', 'reference', 'mo_analysis', 'results'],
    'hf_esp': ['molecule', 'reference', 'esp_analysis', 'results'],
    'hf_population': ['molecule', 'reference', 'population_analysis', 'results'],
    'hf_ir': ['molecule', 'reference', 'ir_analysis', 'results'],
    'hf_nmr': ['molecule', 'reference', 'nmr_analysis', 'results'],
    # DFT profiles
    'dft_energy': ['molecule', 'reference', 'results'],
    'dft_gradient': ['molecule', 'reference', 'gradient', 'results'],
    'dft_hessian': ['molecule', 'reference', 'hessian', 'results'],
    'dft_opt_min': ['molecule', 'reference', 'optimize', 'results'],
    'dft_opt_ts': ['molecule', 'reference', 'optimize_ts', 'results'],
    'dft_mo': ['molecule', 'reference', 'mo_analysis', 'results'],
    'dft_esp': ['molecule', 'reference', 'esp_analysis', 'results'],
    'dft_population': ['molecule', 'reference', 'population_analysis', 'results'],
    'dft_ir': ['molecule', 'reference', 'ir_analysis', 'results'],
    'dft_nmr': ['molecule', 'reference', 'nmr_analysis', 'results'],
    # MP2 profiles
    'mp2_energy': ['molecule', 'reference', 'mp2', 'results'],
    'mp2_gradient': ['molecule', 'reference', 'mp2', 'gradient', 'results'],
    'mp2_mo': ['molecule', 'reference', 'mp2', 'mo_analysis', 'results'],
    'mp2_esp': ['molecule', 'reference', 'mp2', 'esp_analysis', 'results'],
    'mp2_population': ['molecule', 'reference', 'mp2', 'population_analysis', 'results'],
    # TDDFT profiles
    'tddft_energy': ['molecule', 'reference', 'tddft', 'results'],
    'tddft_gradient': ['molecule', 'reference', 'tddft', 'gradient', 'results'],
    'tddft_opt_min': ['molecule', 'reference', 'tddft', 'optimize', 'results'],
    'tddft_uv_vis': ['molecule', 'reference', 'tddft', 'uv_analysis', 'results'],
    'tddft_mo': ['molecule', 'reference', 'tddft', 'mo_analysis', 'results'],
    'tddft_esp': ['molecule', 'reference', 'tddft', 'esp_analysis', 'results'],
    'tddft_population': ['molecule', 'reference', 'tddft', 'population_analysis', 'results'],
    # TDA profiles
    'tda_energy': ['molecule', 'reference', 'tda', 'results'],
    'tda_gradient': ['molecule', 'reference', 'tda', 'gradient', 'results'],
    'tda_opt_min': ['molecule', 'reference', 'tda', 'optimize', 'results'],
    'tda_uv_vis': ['molecule', 'reference', 'tda', 'uv_analysis', 'results'],
    'tda_mo': ['molecule', 'reference', 'tda', 'mo_analysis', 'results'],
    'tda_esp': ['molecule', 'reference', 'tda', 'esp_analysis', 'results'],
    'tda_population': ['molecule', 'reference', 'tda', 'population_analysis', 'results'],
}

# Available methods and their capabilities
# This is used for validation and runtime method selection
AVAILABLE_METHOD = {
    'hf': {
        'energy': 'reference',
        'gradient': 'reference',
        'hessian': 'reference',
        'opt_min': 'reference',
        'opt_ts': 'reference',
        'opt_irc': None,
        'mo': 'reference',
        'esp': 'reference',
        'population': 'reference',
        'ir': 'infrared',
        'raman': None,
        'uv_vis': None,
        'nmr': 'shielding',
        'post_method': None,
        'functional': False,
        'reference': None,  # Will be set at runtime
        'infrared': None,   # Will be set at runtime
        'shielding': None,  # Will be set at runtime
        'mp2': False,
        'tddft': False,
    },
    'mp2': {
        'energy': 'mp2',
        'gradient': 'mp2',
        'hessian': None,
        'opt_min': None,
        'opt_ts': None,
        'opt_irc': None,
        'mo': 'reference',
        'esp': 'reference',
        'population': 'reference',
        'ir': None,
        'raman': None,
        'uv_vis': None,
        'nmr': None,
        'post_method': 'mp2',
        'functional': False,
        'reference': None,  # Will be set at runtime
        'mp2': None,        # Will be set at runtime
        'tddft': False,
    },
    'dft': {
        'energy': 'reference',
        'gradient': 'reference',
        'hessian': 'reference',
        'opt_min': 'reference',
        'opt_ts': 'reference',
        'opt_irc': None,
        'mo': 'reference',
        'esp': 'reference',
        'population': 'reference',
        'ir': 'infrared',
        'raman': None,
        'uv_vis': None,
        'nmr': 'shielding',
        'post_method': None,
        'functional': True,
        'reference': None,  # Will be set at runtime
        'infrared': None,   # Will be set at runtime
        'shielding': None,  # Will be set at runtime
        'mp2': False,
        'tddft': False,
    },
    'tddft': {
        'energy': 'tddft',
        'gradient': 'tddft',
        'hessian': None,
        'opt_min': 'tddft',
        'opt_ts': None,
        'opt_irc': None,
        'mo': 'reference',
        'esp': 'reference',
        'population': 'tddft',
        'ir': None,
        'raman': None,
        'uv_vis': 'tddft',
        'nmr': None,
        'post_method': 'tddft',
        'functional': True,
        'reference': None,  # Will be set at runtime
        'mp2': False,
        'tddft': None,      # Will be set at runtime
    },
    'tda': {
        'energy': 'tddft',
        'gradient': 'tddft',
        'hessian': None,
        'opt_min': 'tddft',
        'opt_ts': None,
        'opt_irc': None,
        'mo': 'reference',
        'esp': 'reference',
        'population': 'tddft',
        'ir': None,
        'raman': None,
        'uv_vis': 'tddft',
        'nmr': None,
        'post_method': 'tddft',
        'functional': True,
        'reference': None,  # Will be set at runtime
        'mp2': False,
        'tddft': None,      # Will be set at runtime
    },
}


def initialize_method_classes():
    """Initialize PySCF method classes at runtime.

    This function must be called before using AVAILABLE_METHOD.
    It imports PySCF modules and populates the class references.
    """
    from pyscf import scf, mp, dft, tddft
    # `pyscf.prop` is the separate properties extension (pip install
    # git+https://github.com/pyscf/properties), and pyscf wheels do not all
    # ship it: it is present in 2.13 and absent from the 2.14 wheel this
    # project's local environment uses. Importing it unconditionally here made
    # every HF and DFT job fail with "No module named 'pyscf.prop'" long before
    # any property was asked for, so the import is optional and only the two
    # capabilities that need it report the absence, at the call that needs it.
    try:
        from pyscf.prop import infrared, nmr
    except ImportError:
        infrared = nmr = None
        PROPERTIES_AVAILABLE = False
    else:
        PROPERTIES_AVAILABLE = True
    globals()['PROPERTIES_AVAILABLE'] = PROPERTIES_AVAILABLE

    # HF
    AVAILABLE_METHOD['hf']['reference'] = (scf.RHF, scf.UHF)
    if infrared is not None:
        AVAILABLE_METHOD['hf']['infrared'] = (
            infrared.rhf.Infrared, infrared.uhf.Infrared)
    if nmr is not None:
        AVAILABLE_METHOD['hf']['shielding'] = (nmr.RKS, nmr.UKS)

    # MP2
    AVAILABLE_METHOD['mp2']['reference'] = (scf.RHF, scf.UHF)
    AVAILABLE_METHOD['mp2']['mp2'] = (mp.RMP2, mp.UMP2)

    # DFT
    AVAILABLE_METHOD['dft']['reference'] = (dft.RKS, dft.UKS)
    if infrared is not None:
        AVAILABLE_METHOD['dft']['infrared'] = (
            infrared.rks.Infrared, infrared.uks.Infrared)
    if nmr is not None:
        AVAILABLE_METHOD['dft']['shielding'] = (nmr.RKS, nmr.UKS)

    # TDDFT
    AVAILABLE_METHOD['tddft']['reference'] = (dft.RKS, dft.UKS)
    AVAILABLE_METHOD['tddft']['tddft'] = tddft.TDDFT

    # TDA
    AVAILABLE_METHOD['tda']['reference'] = (dft.RKS, dft.UKS)
    AVAILABLE_METHOD['tda']['tddft'] = tddft.TDA


# Mapping from jobtype string to profile suffix.
# Used to derive the profile from method + jobtype when profile is
# not explicitly provided (e.g. classic input flow).
_JOBERTYPE_SUFFIX_MAP = {
    'energy':              '_energy',
    'gradient':            '_gradient',
    'energy gradient':     '_gradient',
    'frequency':           '_hessian',
    'energy frequency':    '_hessian',
    'optimize':            '_opt_min',
    'absorption':          '_uv_vis',
    # Molecular dynamics propagates on energy/gradient surfaces
    'md':                  '_gradient',
}


def resolve_profile(method, jobtype):
    """Derive the profile from method and jobtype.

    When the profile is not explicitly provided, this function
    determines the correct profile by combining the method name
    with the task implied by the jobtype.

    Parameters
    ----------
    method : str
        Computational method (e.g. 'hf', 'dft', 'tddft').
    jobtype : str
        Job type string (e.g. 'energy', 'energy gradient', 'optimize').

    Returns
    -------
    str
        Profile name (e.g. 'dft_energy', 'tddft_gradient').

    Raises
    ------
    ValueError
        If the method or jobtype is not recognized.
    """
    method = method.lower()
    jobtype = jobtype.lower().strip()

    if method not in AVAILABLE_METHOD:
        raise ValueError(f"Cannot resolve profile: unknown method '{method}'")

    suffix = _JOBERTYPE_SUFFIX_MAP.get(jobtype)
    if suffix is None:
        raise ValueError(
            f"Cannot resolve profile: jobtype '{jobtype}' does not map "
            f"to a known profile suffix. Available: {sorted(_JOBERTYPE_SUFFIX_MAP)}"
        )

    profile = f'{method}{suffix}'
    if profile not in PROFILE_METHOD_MAP:
        raise ValueError(
            f"Resolved profile '{profile}' (method={method}, "
            f"jobtype={jobtype}) is not defined in PROFILE_METHOD_MAP"
        )

    return profile
