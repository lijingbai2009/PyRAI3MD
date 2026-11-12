"""
======================================================================
This module provides physical unit conversion constants and helper
functions for PyRAI3MD. It consolidates constants that were previously
scattered as hardcoded values across multiple classic modules.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

# Length conversions
BOHR_TO_ANGSTROM = 0.52917721090299996
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# Mass conversions
AMU_TO_AU = 1822.8884871474306  # atomic mass unit to atomic units (electron mass)
AU_TO_AMU = 1.0 / AMU_TO_AU

# Energy conversions
HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
HARTREE_TO_KCAL = 627.5094740631
KCAL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL
HARTREE_TO_KJ = 2625.4996394799
KJ_TO_HARTREE = 1.0 / HARTREE_TO_KJ

# Frequency/wavenumber conversions
CM_TO_HARTREE = 4.556335252e-6
HARTREE_TO_CM = 1.0 / CM_TO_HARTREE

# ---------------------------------------------------------------------
# SI pieces of the vibrational and thermodynamic analysis
#
# The harmonic frequency of one normal mode is the square root of an
# eigenvalue of the Hessian mass-weighted in amu and Bohr^2, and the
# thermodynamic corrections are sums of the quantized-oscillator and
# ideal-gas formulas. Both are written in SI in every textbook, so the
# constants below are the SI values (all six are exact under the 2019 SI
# redefinition except the atomic mass unit and the Bohr radius, which are
# CODATA) and the conversion into Hartree happens once, at the end, through
# J_TO_HARTREE. Keeping the formulas in SI and the conversion in one place is
# what makes the framework's numbers comparable with the analytical path:
# PySCF's `thermo.thermo` does the same arithmetic internally, and the two
# agree on the water test to the printed digit.
#
# `HARTREE_TO_CM / sqrt(AMU_TO_AU)` is the same 5140.489... that the legacy
# OQP `frequency` module carried as a named constant; deriving it from the
# table above instead of copying the number keeps one source of truth, and its
# 4e-7 relative difference from the literal is far below any tolerance this
# analysis is judged against.
# ---------------------------------------------------------------------
SPEED_OF_LIGHT_CM = 2.99792458e10        # cm/s
PLANCK_CONSTANT = 6.62607015e-34         # J s
BOLTZMANN_CONSTANT = 1.380649e-23        # J/K
AVOGADRO_CONSTANT = 6.02214076e23        # 1/mol
GAS_CONSTANT = BOLTZMANN_CONSTANT * AVOGADRO_CONSTANT  # J/(mol K)
AMU_TO_KG = 1.66053906660e-27            # kg
BOHR_TO_METER = BOHR_TO_ANGSTROM * 1.0e-10
# One Hartree in Joule, per particle. `HARTREE_TO_KJ` above is a molar
# quantity (kiloJoule per mole, the chemistry convention), so dividing a
# Joule by it silently loses a factor of Avogadro's number: the thermodynamic
# sums came out 6e23 times too small that way, which reads as a zero point
# energy of 0.0000000000 rather than as an obvious unit error. Hence the
# particle version here, and the molar one stays where it belongs.
HARTREE_TO_JOULE = HARTREE_TO_KJ * 1.0e3 / AVOGADRO_CONSTANT
J_TO_HARTREE = 1.0 / HARTREE_TO_JOULE
HARTREE_TO_JOULE_PER_MOLE = HARTREE_TO_KJ * 1.0e3
# Boltzmann constant in Hartree/K: the same quantity as KELVIN_TO_HARTREE
# above, reached through SI. The two agree to 6e-8 relative (measured), which
# is why the thermodynamic sums use this one and the existing constants stay
# where they are.
BOLTZMANN_TO_HARTREE_PER_K = BOLTZMANN_CONSTANT * J_TO_HARTREE
# Gas constant per mole in Hartree/(mol K), for the rare molar formula.
GAS_CONSTANT_TO_HARTREE_PER_MOL_K = (GAS_CONSTANT
                                      / HARTREE_TO_JOULE_PER_MOLE)
# cm^-1 -> GHz (the unit `rotation_constants` is declared in): nu/c cancels a
# length, so it is just the speed of light.
CM_TO_GHZ = SPEED_OF_LIGHT_CM / 1.0e9
# sqrt(Hartree / (amu Bohr^2)) -> cm^-1, the vibrational frequency factor.
VIBRATION_TO_CM = HARTREE_TO_CM / AMU_TO_AU ** 0.5
# Standard-state pressure of the thermodynamic corrections, in Pa. One bar, so
# that the framework's numbers and the analytical backend's are read against
# the same reference state: a backend asked for 1 bar gets 1e5 Pa here too.
PRESSURE_STANDARD_PA = 1.0e5
BAR_TO_PA = 1.0e5

# Temperature conversions
KELVIN_TO_HARTREE = 3.1668114e-6
HARTREE_TO_KELVIN = 1.0 / KELVIN_TO_HARTREE

# Pressure conversions
# The atomic unit of pressure is Hartree / Bohr^3. One Hartree/Bohr^3
# equals 2.942101496e13 Pa (29421.0 GPa), so one bar (1e5 Pa) is a very
# small fraction of it. Input pressure keywords (Dynamics.pressure) are
# read in bar and converted here into atomic units for the barostats.
HARTREE_PER_BOHR3_TO_PA = 2.942101496e13
PA_TO_HARTREE_PER_BOHR3 = 1.0 / HARTREE_PER_BOHR3_TO_PA
HARTREE_PER_BOHR3_TO_BAR = HARTREE_PER_BOHR3_TO_PA / 1.0e5
BAR_TO_HARTREE_PER_BOHR3 = 1.0 / HARTREE_PER_BOHR3_TO_BAR
ATM_TO_BAR = 1.01325
BAR_TO_ATM = 1.0 / ATM_TO_BAR
ATM_TO_HARTREE_PER_BOHR3 = ATM_TO_BAR * BAR_TO_HARTREE_PER_BOHR3
# One GPa (1e9 Pa) in Hartree/Bohr^3; CP2K prints its analytical stress
# tensor in GPa, so the framework-central stress conversion needs it.
GPA_TO_HARTREE_PER_BOHR3 = 1.0e9 * PA_TO_HARTREE_PER_BOHR3
HARTREE_PER_BOHR3_TO_GPA = 1.0 / GPA_TO_HARTREE_PER_BOHR3
# Isothermal compressibility beta is an inverse pressure: one bar^-1
# equals HARTREE_PER_BOHR3_TO_BAR in atomic units (Bohr^3 / Hartree).
COMPRESSIBILITY_BAR_INV_TO_AU = HARTREE_PER_BOHR3_TO_BAR

# ---------------------------------------------------------------------
# Native units a skill may declare for the numbers it returns
#
# A skill is allowed to compute in its own units and to say so in its
# manifest; the framework converts. The manifest names the unit of the
# run's unit set (e.g. 'real' -> 'kcal/mol'), and the two tables below
# turn that unit name into the factor applied to the result.
#
# The gradient factors are written as the SAME product the LAMMPS skill
# used while it still owned this conversion
# (energy factor x BOHR_TO_ANGSTROM, its
# FORCE_TO_HARTREE_PER_BOHR = KCAL_TO_HARTREE * BOHR_TO_ANGSTROM). That is
# deliberate: a factor re-derived through a different association of the
# same three constants would differ in the last bit, and the move of the
# conversion from the skill to the framework is only provable if both
# routes produce the identical number.
# ---------------------------------------------------------------------
NATIVE_ENERGY_TO_HARTREE = {
    'hartree': 1.0,
    'ev': EV_TO_HARTREE,
    'kcal/mol': KCAL_TO_HARTREE,
    'kj/mol': KJ_TO_HARTREE,
}

# A gradient is dU/dR, so an energy-per-length unit converts as the energy
# factor times the length factor (1/Angstrom = BOHR_TO_ANGSTROM /Bohr).
NATIVE_GRADIENT_TO_HARTREE_PER_BOHR = {
    'hartree/bohr': 1.0,
    'kcal/mol/angstrom': KCAL_TO_HARTREE * BOHR_TO_ANGSTROM,
    'ev/angstrom': EV_TO_HARTREE * BOHR_TO_ANGSTROM,
}

# Time conversions
# One atomic unit of time is 2.4188843265857e-17 s, so one femtosecond
# (the unit of the thermostat/barostat coupling constants tau_T / tau_p)
# equals ~41.34 atomic time units.
AU_TIME_TO_FS = 0.024188843265857
FS_TO_AU_TIME = 1.0 / AU_TIME_TO_FS

# Nonadiabatic coupling threshold (Hartree)
# Adiabatic energy gaps below this value are treated as degenerate:
# the NAC vector component is set to zero instead of dividing by the
# gap. Reused by surface hopping (FSSH NAC.veloc, frustrated hopping
# gap checks) and kept numerically identical to the self-contained
# constant of the qdm_skill (the two are independent by design).
NAC_ENERGY_GAP_EPS = 1e-12


def bohr_to_angstrom(x):
    """Convert distance from Bohr to Angstrom."""
    return x * BOHR_TO_ANGSTROM


def angstrom_to_bohr(x):
    """Convert distance from Angstrom to Bohr."""
    return x * ANGSTROM_TO_BOHR


def amu_to_au(x):
    """Convert mass from atomic mass units to atomic units."""
    return x * AMU_TO_AU



def hartree_to_ev(x):
    """Convert energy from Hartree to eV."""
    return x * HARTREE_TO_EV


def ev_to_hartree(x):
    """Convert energy from eV to Hartree."""
    return x * EV_TO_HARTREE




def bar_to_au_pressure(x):
    """Convert pressure from bar to atomic units (Hartree/Bohr^3)."""
    return x * BAR_TO_HARTREE_PER_BOHR3


def au_pressure_to_bar(x):
    """Convert pressure from atomic units (Hartree/Bohr^3) to bar."""
    return x * HARTREE_PER_BOHR3_TO_BAR




def compressibility_bar_to_au(x):
    """Convert isothermal compressibility from bar^-1 to Bohr^3/Hartree."""
    return x * COMPRESSIBILITY_BAR_INV_TO_AU
