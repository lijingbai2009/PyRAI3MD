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

This module defines the abstract base class of all analytic quantum
dynamics models. A model only defines its diabatic Hamiltonian H(x)
and its analytic derivative dH/dx; the calculator and the reference
solver share this single physical source.

"""

import numpy as np


class DoubleSlopeLandauZener:
    """The crossing law two half-linear models share.

    lz2 and lz3 differ in which surface is flat, but their half-linear
    slope difference is the same ``lz_a``, so both answer the textbook
    single-passage probability ``exp(-2*pi*lz_b^2 / (lz_a*v))`` (hbar = 1
    in Hartree atomic units). lz1's slope difference is half as large and
    keeps its own method.
    """

    def landau_zener_probability(self, v):
        """Single-passage Landau-Zener hopping probability.

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
        return float(np.exp(-2.0 * np.pi * self.lz_b ** 2
                            / (self.lz_a * v)))


class AnalyticModel:
    """Abstract base class for 1D two-state analytic models.

    Every model defines a real symmetric 2x2 diabatic Hamiltonian
    H(x) and its analytic derivative dH/dx as functions of the model
    coordinate x (Bohr). Standard parameters live on the class and
    may be overridden per instance through the constructor.

    Parameters
    ----------
    params : dict, optional
        Parameter overrides keyed by the class parameter names.

    Raises
    ------
    ValueError
        If an override names a parameter the model does not define.
    """

    #: Human readable model label (subclasses override).
    name = 'analytic'

    def __init__(self, params=None):
        declared = set(self._parameter_names())
        for key, value in (params or {}).items():
            if value is None:
                continue
            if key not in declared:
                raise ValueError(
                    "Unknown parameter '%s' for model '%s'; available: "
                    "%s" % (key, self.name, sorted(declared)))
            setattr(self, key, float(value))

    @classmethod
    def _parameter_names(cls):
        """Return the model parameter names.

        Parameters are the class attributes listed in the class-level
        PARAMETER_NAMES tuple (subclasses declare their own).

        Returns
        -------
        tuple of str
            Parameter names accepted by the constructor.
        """
        return tuple(getattr(cls, 'PARAMETER_NAMES', ()))

    def hamiltonian(self, x):
        """Evaluate the diabatic Hamiltonian at x.

        Parameters
        ----------
        x : float
            Model coordinate in Bohr.

        Returns
        -------
        numpy.ndarray
            Real symmetric (2, 2) Hamiltonian matrix (Hartree).
        """
        raise NotImplementedError

    def hamiltonian_gradient(self, x):
        """Evaluate the analytic derivative dH/dx at x.

        Parameters
        ----------
        x : float
            Model coordinate in Bohr.

        Returns
        -------
        numpy.ndarray
            Real symmetric (2, 2) derivative matrix (Hartree/Bohr).
        """
        raise NotImplementedError

    def numerical_hamiltonian_gradient(self, x, dx=1e-5):
        """Central-difference dH/dx (test/diagnostic helper only).

        Parameters
        ----------
        x : float
            Model coordinate in Bohr.
        dx : float
            Displacement step.

        Returns
        -------
        numpy.ndarray
            Numerical (2, 2) derivative matrix (Hartree/Bohr).
        """
        return (np.asarray(self.hamiltonian(x + dx))
                - np.asarray(self.hamiltonian(x - dx))) / (2.0 * dx)
