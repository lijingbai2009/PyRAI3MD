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

Textbook Landau-Zener linear crossing model.

Diabatic Hamiltonian (x in Bohr, values in Hartree):
    V11 =  A * x
    V22 = -A * x
    V12 =  B (constant coupling)

Analytic derivatives:
    dV11/dx =  A
    dV22/dx = -A
    dV12/dx =  0

The model exists as the calibration target of the Zhu-Nakamura
global surface hopping (GSH) integrator. Its diabatic surfaces are
linear (constant slopes everywhere) and the coupling is constant, so
the two premises of the GSH force reconstruction (EQ 7 / EQ 8 of the
Zhu-Nakamura internal-conversion kernel) hold exactly: the sampling
points never see a decaying diabatic slope, and the avoided-crossing
character-swap width B / A can be tuned narrow enough to fit inside
the two-step sampling window. The Tully models violate both premises
(curved diabats, wide swap region), which is why GSH is calibrated
here instead of on the Tully set.

The adiabatic gap at the crossing is 2 B; the calibration case pins
a gap threshold above this value so the three-step crossing detector
admits the hop evaluation.

"""

import numpy as np

from .base import AnalyticModel


class Lz1(AnalyticModel):
    """Landau-Zener linear crossing benchmark model.

    Two straight diabatic surfaces with slopes +/- A crossing at
    x = 0, coupled by the constant off-diagonal term B. The crossing
    gap is 2 B and the adiabatic character-swap width scales as
    B / A.
    """

    name = 'lz1'
    PARAMETER_NAMES = ('lz_a', 'lz_b')

    lz_a = 0.04   # Hartree/Bohr, diabatic surface slope
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
        x = float(x)
        v11 = self.lz_a * x
        return np.array([[v11, self.lz_b], [self.lz_b, -v11]])

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
        return np.array([[self.lz_a, 0.0], [0.0, -self.lz_a]])

    def adiabatic_gradients(self, x):
        """Analytic adiabatic dE/dx (lower, upper) at x.

        The lower (upper) adiabatic eigenvalue of [[lz_a*x, lz_b],
        [lz_b, -lz_a*x]] differentiates to -/+ lz_a^2*x/d with
        d = sqrt((lz_a*x)^2 + lz_b^2).

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
        d = np.sqrt((self.lz_a * x) ** 2 + self.lz_b ** 2)
        slope = self.lz_a ** 2 * x / d
        return float(-slope), float(slope)

    def crossing_velocity(self, mass, x0, v0):
        """Velocity at the crossing point (x = 0).

        Energy conservation on the lower adiabatic surface
        E_lower(x) = -sqrt((lz_a*x)^2 + lz_b^2) fixes the
        crossing-point velocity of a trajectory launched from
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
        e_lower_x0 = -np.sqrt((self.lz_a * x0) ** 2
                              + self.lz_b ** 2)
        ekin_c = 0.5 * mass * v0 ** 2 + e_lower_x0 + self.lz_b
        return float(np.sqrt(max(2.0 * ekin_c / mass, 0.0)))

    def landau_zener_probability(self, v):
        """Single-passage Landau-Zener hopping probability.

        The analytic textbook result for the linear crossing:
        P = exp(-pi * lz_b^2 / (lz_a * v)) (hbar = 1 in Hartree
        atomic units). This is the calibration target of the
        Zhu-Nakamura surface hopping benchmark.

        Parameters
        ----------
        v : float
            Velocity at the crossing point (Bohr/au).

        Returns
        -------
        float
            The analytic hopping probability.
        """
        v = float(v)
        if v <= 0:
            return 0.0
        return float(np.exp(-np.pi * self.lz_b ** 2
                            / (self.lz_a * v)))
