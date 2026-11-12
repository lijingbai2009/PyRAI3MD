"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 24, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Tully Model 1 (simple avoided crossing), Tully 1990 example I.

Diabatic Hamiltonian (x in Bohr, values in Hartree):
    V11 =  A * sign(x) * (1 - exp(-B |x|))
    V22 = -V11
    V12 =  C * exp(-D x^2)

Analytic derivatives:
    dV11/dx = A * B * exp(-B |x|)   (x = 0 takes the right limit)
    dV12/dx = -2 C D x * exp(-D x^2)

"""

import numpy as np

from .base import AnalyticModel


class Tully1(AnalyticModel):
    """Simple avoided crossing benchmark model.

    Standard parameters from Tully 1990 example I.
    """

    name = 'tully1'
    PARAMETER_NAMES = ('tully_a', 'tully_b', 'tully_c', 'tully_d')

    tully_a = 0.01    # Hartree, asymptotic diabatic splitting
    tully_b = 1.6     # 1/Bohr, wall steepness
    tully_c = 0.005   # Hartree, coupling strength
    tully_d = 1.0     # 1/Bohr^2, coupling width

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
        v11 = (self.tully_a * np.sign(x)
               * (1.0 - np.exp(-self.tully_b * abs(x))))
        v12 = self.tully_c * np.exp(-self.tully_d * x * x)
        return np.array([[v11, v12], [v12, -v11]])

    def hamiltonian_gradient(self, x):
        """Evaluate the analytic dH/dx at x.

        At x = 0 the kink derivative of V11 takes the right limit
        A*B, matching the analytic NAC calibration point
        d01(0) = -A*B/(2C).

        Parameters
        ----------
        x : float
            Model coordinate in Bohr.

        Returns
        -------
        numpy.ndarray
            (2, 2) derivative matrix (Hartree/Bohr).
        """
        x = float(x)
        dv11 = self.tully_a * self.tully_b * np.exp(
            -self.tully_b * abs(x))
        dv12 = (-2.0 * self.tully_c * self.tully_d * x
                * np.exp(-self.tully_d * x * x))
        return np.array([[dv11, dv12], [dv12, -dv11]])
