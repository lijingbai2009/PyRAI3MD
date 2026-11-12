"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 2, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Half-linear Landau-Zener crossing model with a flat lower surface.

Diabatic Hamiltonian (x in Bohr, values in Hartree):
    V11 =  A * x       (upper surface: slope +A)
    V22 =  0           (lower surface: zero slope)
    V12 =  B (constant coupling)

Analytic derivatives:
    dV11/dx =  A
    dV22/dx =  0
    dV12/dx =  0

The model complements the textbook lz1 linear crossing model: one
diabatic surface keeps the constant slope while the other stays
flat, so both premises of the GSH force reconstruction (EQ 7 / EQ 8
of the Zhu-Nakamura internal-conversion kernel) still hold exactly
(constant diabatic slopes everywhere, constant coupling). The slope
difference at the crossing is |F1 - F2| = lz_a (half of lz1), so
the single-passage Landau-Zener exponent doubles at equal crossing
velocity: P = exp(-2*pi*lz_b^2 / (lz_a*v_c)).

The approach side (x < 0) climbs the sloped branch exactly like
lz1, but beyond the crossing the lower adiabatic surface flattens
out (E_lower -> 0^- as x -> +inf): a non-hopping trajectory coasts
at constant velocity instead of running away on a descending
surface, which relaxes the benchmark propagation-window constraint;
a trajectory that hops onto the upper surface climbs the +lz_a
slope and may turn back (frustrated-hop territory).

"""

import numpy as np

from .base import AnalyticModel, DoubleSlopeLandauZener


class Lz3(DoubleSlopeLandauZener, AnalyticModel):
    """Half-linear Landau-Zener model, flat lower surface.

    A straight diabatic surface of slope +lz_a (V11 = lz_a*x)
    crossing a flat surface (V22 = 0) at x = 0, coupled by the
    constant off-diagonal term lz_b. The crossing gap is 2*lz_b and
    the adiabatic character-swap width scales as lz_b / lz_a.
    """

    name = 'lz3'
    PARAMETER_NAMES = ('lz_a', 'lz_b')

    lz_a = 0.04   # Hartree/Bohr, sloped diabatic surface slope
    lz_b = 0.02   # Hartree, constant coupling (gap = 2 * lz_b)

    def hamiltonian(self, x):
        """Evaluate the diabatic Hamiltonian at x.

        Parameters
        ----------
        x : float
            Model coordinate in Bohr.

        Returns
        -------
        numpy.ndarray
            (2, 2) Hamiltonian matrix (Hartree).
        """
        return np.array([[self.lz_a * float(x), self.lz_b],
                         [self.lz_b, 0.0]])

    def hamiltonian_gradient(self, x):
        """Evaluate the analytic dH/dx at x.

        The derivative is coordinate independent: the diabatic
        surfaces are straight lines and the coupling is constant.

        Parameters
        ----------
        x : float
            Model coordinate in Bohr (accepted for interface
            uniformity, unused).

        Returns
        -------
        numpy.ndarray
            (2, 2) derivative matrix (Hartree/Bohr).
        """
        return np.array([[self.lz_a, 0.0], [0.0, 0.0]])

    def adiabatic_gradients(self, x):
        """Analytic adiabatic dE/dx (lower, upper) at x.

        The eigenvalues of [[lz_a*x, lz_b], [lz_b, 0]] are
        E+- = lz_a*x/2 +- sqrt((lz_a*x/2)^2 + lz_b^2), which
        differentiate to the closed form below.

        Parameters
        ----------
        x : float
            Model coordinate in Bohr.

        Returns
        -------
        tuple(float, float)
            (dE_lower/dx, dE_upper/dx) in Hartree/Bohr.
        """
        x = float(x)
        s = np.sqrt((self.lz_a * x / 2.0) ** 2 + self.lz_b ** 2)
        swap = self.lz_a ** 2 * x / (4.0 * s)
        return float(self.lz_a / 2.0 - swap), float(self.lz_a / 2.0 + swap)

    def crossing_velocity(self, mass, x0, v0):
        """Velocity at the crossing point (x = 0).

        Energy conservation on the lower adiabatic surface
        E_lower(x) = lz_a*x/2 - sqrt((lz_a*x/2)^2 + lz_b^2) fixes
        the crossing-point velocity of a trajectory launched from
        (x0, v0) toward the crossing.

        Parameters
        ----------
        mass : float
            Model mass (atomic units).
        x0 : float
            Launch coordinate (Bohr).
        v0 : float
            Launch velocity (Bohr/au).

        Returns
        -------
        float
            Crossing-point velocity (Bohr/au); 0.0 when the launch
            energy does not reach the crossing.
        """
        mass = float(mass)
        x0 = float(x0)
        v0 = float(v0)
        e_lower_x0 = (self.lz_a * x0 / 2.0
                      - np.sqrt((self.lz_a * x0 / 2.0) ** 2
                                + self.lz_b ** 2))
        ekin_c = 0.5 * mass * v0 ** 2 + e_lower_x0 + self.lz_b
        return float(np.sqrt(max(2.0 * ekin_c / mass, 0.0)))

