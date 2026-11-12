"""
======================================================================
This is a standalone CP2K calculation skill
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
the CP2K input step chain and execution requirements.

"""


class WorkflowGraph:
    """Represents a sequence of CP2K input steps to assemble.

    A workflow graph is compiled from a profile and defines the
    ordered list of input steps needed for a calculation, together
    with derived execution flags consumed by the input builder.

    Attributes
    ----------
    profile : str
        The calculation profile name.
    steps : list of str
        Ordered list of CP2K input step names.
    method : str
        Electronic structure method ('dft' or 'tddft').
    is_tddft : bool
        Whether the workflow needs the TDDFPT block.
    is_optimization : bool
        Whether the workflow is a geometry optimization (GEO_OPT).
    is_md : bool
        Whether the workflow is a standalone &MOTION MD run.
    """

    def __init__(self, profile: str, steps: list):
        self.profile = profile
        self.steps = list(steps)
        self.method = profile.split('_')[0]
        self.is_tddft = self.has_step('tddft')
        self.is_optimization = self.has_step('optimize')
        self.is_md = self.has_step('md')

    def __repr__(self):
        return f"WorkflowGraph(profile='{self.profile}', steps={self.steps})"

    def has_step(self, step_name: str) -> bool:
        """Check if a step is in the workflow.

        Parameters
        ----------
        step_name : str
            CP2K input step name (e.g. 'gradient', 'tddft').

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
            CP2K input step name.

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
