"""
======================================================================
This is a standalone ORCA calculation skill
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
the ORCA input step chain and execution requirements.

"""


class WorkflowGraph:
    """Represents a sequence of ORCA input steps to assemble.

    A workflow graph is compiled from a profile and defines the
    ordered list of input steps needed for a calculation, together
    with derived execution flags consumed by the input builder.

    Attributes
    ----------
    profile : str
        The calculation profile name.
    steps : list of str
        Ordered list of ORCA input step names.
    method : str
        Electronic structure method ('dft' or 'tddft').
    is_tddft : bool
        Whether the workflow needs the TDDFT block.
    is_soc : bool
        Whether the workflow needs spin-orbit coupling (DoSOC).
    is_optimization : bool
        Whether the workflow is a geometry optimization.
    is_transition_state : bool
        Whether the optimization targets a transition state (OptTS).
    is_frequency : bool
        Whether the workflow needs a frequency/Hessian calculation.
    """

    def __init__(self, profile: str, steps: list):
        self.profile = profile
        self.steps = list(steps)
        self.method = profile.split('_')[0]
        self.is_tddft = self.has_step('tddft')
        self.is_soc = self.profile.endswith('_soc')
        self.is_optimization = self.has_step('optimize') or self.has_step('optimize_ts')
        self.is_transition_state = self.has_step('optimize_ts')
        self.is_frequency = self.has_step('hessian')

    def __repr__(self):
        return f"WorkflowGraph(profile='{self.profile}', steps={self.steps})"

    def has_step(self, step_name: str) -> bool:
        """Check if a step is in the workflow.

        Parameters
        ----------
        step_name : str
            ORCA input step name (e.g. 'gradient', 'tddft').

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
            ORCA input step name.

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
