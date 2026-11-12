"""
======================================================================
This is a standalone xTB calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module defines the WorkflowGraph class that represents
the xTB calculation step chain and execution order.

"""


class WorkflowGraph:
    """Represents a sequence of xTB calculation steps to execute.

    A workflow graph is compiled from a profile and defines the
    ordered list of xTB steps needed for a calculation.

    Attributes
    ----------
    profile : str
        The calculation profile name.
    steps : list of str
        Ordered list of step names ('energy', 'gradient').
    needs_gradient : bool
        Whether the workflow requires the gradient (--grad flag).
    """

    def __init__(self, profile: str, steps: list):
        self.profile = profile
        self.steps = steps
        self.needs_gradient = 'gradient' in steps

    def __repr__(self):
        return f"WorkflowGraph(profile='{self.profile}', steps={self.steps})"

    def has_step(self, step_name: str) -> bool:
        """Check if a step is in the workflow.

        Parameters
        ----------
        step_name : str
            Step name (e.g. 'energy', 'gradient').

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
            Step name.

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
