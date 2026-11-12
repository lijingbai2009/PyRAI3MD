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

This module builds and dispatches PySCF calculations based on
the configuration and compiled workflow.

For PySCF (a Python-native library), "building" means:
1. Constructing the PySCF Mole and mean-field objects
2. Dispatching to the appropriate service function based on workflow steps

"""


from ..energy import energy
from ..gradient import gradient
from ..hessian import hessian
from ..optimize import opt_min, opt_ts
from ..analysis import mo, esp, population
from ..spectra import ir, uv_vis, nmr
from ...schema.method import initialize_method_classes


# Map step names to service functions
STEP_DISPATCH_MAP = {
    'reference': energy,
    'tddft': energy,          # TDDFT/TDA step: energy() handles via post_method
    'tda': energy,            # TDDFT/TDA step: energy() handles via post_method
    'gradient': gradient,
    'hessian': hessian,
    'optimize': opt_min,
    'optimize_ts': opt_ts,
    'mo_analysis': mo,
    'esp_analysis': esp,
    'population_analysis': population,
    'ir_analysis': ir,
    'uv_analysis': uv_vis,
    'nmr_analysis': nmr,
}


def build_runtime(config, workflow) -> dict:
    """Build and execute the PySCF calculation based on workflow.

    For PySCF, building means constructing the runtime objects
    (Mole, mean-field, post-HF) and executing the calculation.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    dict
        Runtime dictionary with results.
    """
    return dispatch_calculation(config, workflow)


def dispatch_calculation(config, workflow) -> dict:
    """Dispatch the calculation based on the workflow steps.

    The workflow defines the ordered list of steps to execute.
    The last non-'results' step determines the main calculation function.
    For most profiles, the service functions handle the full chain internally
    (e.g., gradient() calls energy() first).

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    dict
        Runtime dictionary with results.
    """
    # Ensure PySCF method classes are loaded before dispatching
    initialize_method_classes()

    steps = workflow.steps

    # Find the main calculation step (last step before 'results')
    calc_steps = [s for s in steps if s != 'results' and s != 'molecule']

    if not calc_steps:
        raise ValueError(f"No calculation steps found in workflow: {steps}")

    # The last calculation step is the main one to call
    # Service functions internally call their dependencies
    main_step = calc_steps[-1]

    if main_step in STEP_DISPATCH_MAP:
        func = STEP_DISPATCH_MAP[main_step]
        runtime = func(config)
    else:
        # Fallback: try to use service module dynamically
        runtime = energy(config)

    return runtime
