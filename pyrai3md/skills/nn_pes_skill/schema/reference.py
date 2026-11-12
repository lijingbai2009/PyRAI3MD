"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the reference values of the nn_pes skill: the unit
constants the boundary converts with, and the accepted spellings of the
inbound training-label units.

The constants are the skill's own copy. A skill never imports framework
modules, so the numbers are restated here and guarded by a test.

"""

# Length conversions (identical to the framework constants)
BOHR_TO_ANGSTROM = 0.52917721090299996
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# Energy conversions
HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

# State spin labels (aligned with the framework convention)
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

# The unit convention a training file is expected to carry. It is the
# framework's own convention, the one every engine backend reports: energies
# in Hartree, gradients and couplings in Hartree/Bohr, and spin-orbit
# couplings in cm-1 (this container trains no SOC model, so the last one is
# a declaration only). Nothing selects it and no keyword overrides it,
# because the only rule a training file has to satisfy is that its energy
# and its gradient describe one surface: the gradient must be the derivative
# of that energy with respect to the coordinates the network is fed, and the
# skill arranges that itself (service/trainer_data.py:match_labels).
LABEL_ENERGY_UNIT = 'hartree'
LABEL_GRADIENT_UNIT = 'hartree/bohr'
LABEL_SOC_UNIT = 'cm-1'

# The units the training metrics are printed in. A validation error is
# stored in the scaled units of the fit, so it is converted back through the
# scaler before anyone reads it (service/trainer.py:physical_errors), and
# eV and eV/Angstrom are what a chemist compares against.
METRIC_ENERGY_UNIT = 'eV'
METRIC_GRADIENT_UNIT = 'eV/angstrom'
METRIC_SOC_UNIT = 'cm-1'

# Unit of the gradient and of the couplings this backend reports.
OUTPUT_GRADIENT_UNIT = 'hartree/bohr'
