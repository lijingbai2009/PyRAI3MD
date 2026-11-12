"""
======================================================================
This is a standalone quantum chemistry calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 30, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module defines the WorkflowGraph class that represents
the PySCF computation step dependency chain and execution order.

"""


class WorkflowGraph:
    """Represents a sequence of PySCF computation steps to execute.

    A workflow graph is compiled from a profile and defines the
    ordered list of PySCF computation steps needed for a calculation.

    Attributes
    ----------
    profile : str
        The calculation profile name.
    steps : list of str
        Ordered list of computation step names.
    needs_gradient : bool
        Whether the workflow computes gradients.
    needs_hessian : bool
        Whether the workflow computes Hessians.
    needs_optimize : bool
        Whether the workflow performs geometry optimization.
    needs_analysis : bool
        Whether the workflow performs analysis (MO, ESP, population).
    needs_spectra : bool
        Whether the workflow computes spectra (IR, UV-Vis, NMR).
    """

    def __init__(self, profile: str, steps: list):
        self.profile = profile
        self.steps = steps
        self.needs_gradient = 'gradient' in steps
        self.needs_hessian = 'hessian' in steps
        self.needs_optimize = any(s in steps for s in ['optimize', 'optimize_ts'])
        self.needs_analysis = any(s in steps for s in [
            'mo_analysis', 'esp_analysis', 'population_analysis'
        ])
        self.needs_spectra = any(s in steps for s in [
            'ir_analysis', 'nmr_analysis', 'uv_analysis'
        ])

    def __repr__(self):
        return f"WorkflowGraph(profile='{self.profile}', steps={self.steps})"

    def has_step(self, step_name: str) -> bool:
        """Check if a step is in the workflow.

        Parameters
        ----------
        step_name : str
            Step name (e.g. 'gradient', 'hessian', 'mo_analysis').

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
