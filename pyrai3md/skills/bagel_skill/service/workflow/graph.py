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

This module defines the WorkflowGraph class that represents
the BAGEL computation step dependency chain and execution order.

"""


class WorkflowGraph:
    """Represents a sequence of BAGEL computation steps to execute.

    A workflow graph is compiled from a profile and defines the
    ordered list of BAGEL JSON blocks needed for a calculation.

    Attributes
    ----------
    profile : str
        The calculation profile name.
    steps : list of str
        Ordered list of BAGEL computation step names.
    is_casscf : bool
        Whether this is a CASSCF-level calculation.
    is_caspt2 : bool
        Whether this is a CASPT2-level calculation.
    needs_gradient : bool
        Whether the workflow computes gradients.
    needs_nac : bool
        Whether the workflow computes NAC couplings.
    needs_frequency : bool
        Whether the workflow computes frequencies.
    needs_optimize : bool
        Whether the workflow runs a geometry optimization.
    needs_dipole : bool
        Whether the workflow computes dipole properties (forces
        block with the dipole keyword).
    """

    def __init__(self, profile: str, steps: list):
        self.profile = profile
        self.steps = steps
        self.is_casscf = profile.startswith('casscf')
        self.is_caspt2 = 'caspt2' in profile
        self.needs_gradient = 'gradient' in profile
        self.needs_nac = profile.endswith('_nac')
        self.needs_frequency = 'frequency' in profile
        self.needs_optimize = 'geometry_optimization' in profile
        self.needs_dipole = 'dipole' in profile

    def __repr__(self):
        return f"WorkflowGraph(profile='{self.profile}', steps={self.steps})"

    def has_step(self, step_name: str) -> bool:
        """Check if a step is in the workflow.

        Parameters
        ----------
        step_name : str
            BAGEL step name (e.g. 'casscf', 'forces').

        Returns
        -------
        bool
            True if the step is in the workflow.
        """
        return step_name.lower() in [s.lower() for s in self.steps]

    def get_step_index(self, step_name: str) -> int:
        """Get the index of a step in the workflow.

        Parameters
        ----------
        step_name : str
            BAGEL step name.

        Returns
        -------
        int
            Index of the step, -1 if not found.
        """
        lower_steps = [s.lower() for s in self.steps]
        try:
            return lower_steps.index(step_name.lower())
        except ValueError:
            return -1
