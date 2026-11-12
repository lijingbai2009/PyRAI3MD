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

This module defines the BAGEL input file templates.
BAGEL uses JSON format for input, so these templates define
the structure of each JSON block (molecule, casscf, forces, etc.).

Note: These are not string templates with format placeholders.
The BAGEL input is built programmatically as Python dicts and
serialized to JSON. These constants define default parameter values.

"""

# Default CASSCF parameters
CASSCF_DEFAULTS = {
    'natocc': True,
    'maxiter': 500,
}

# Default XMS-CASPT2 method definition (embedded in forces/hessian/
# optimize blocks as an array entry; smith maxiter and method-level
# maxiter align with maxit_scf)
CASPT2_DEFAULTS = {
    'smith': {
        'method': 'caspt2',
        'ms': True,
        'xms': True,
        'sssr': True,
        'shift': 0.2,
        'thresh': 1.0e-8,
        'maxiter': 500,
        'frozen': True,
    },
    'maxiter': 500,
}

# Default standalone smith block for XMS-CASPT2 single-point
# energies (BAGEL has no standalone caspt2-titled block; the energy
# chain runs casscf followed by this smith block)
SMITH_ENERGY_DEFAULTS = {
    'method': 'caspt2',
    'ms': True,
    'xms': True,
    'sssr': True,
    'shift': 0.2,
}

# Default force computation parameters
FORCE_DEFAULTS = {
    'export': True,
}

# Default NACME (non-adiabatic coupling) parameters
NACME_DEFAULTS = {
    'nacmtype': 'interstate',
}
