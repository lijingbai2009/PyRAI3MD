"""
======================================================================
This is a standalone OpenMolcas calculation skill
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
the Molcas module dependency chain and execution order.

"""


class WorkflowGraph:
    """Represents a sequence of Molcas modules to execute.

    A workflow graph is compiled from a profile and defines the
    ordered list of Molcas modules needed for a calculation.

    Attributes
    ----------
    profile : str
        The calculation profile name.
    modules : list of str
        Ordered list of Molcas module names.
    requires_dual_rasscf : bool
        Whether the workflow needs singlet+triplet RASSCF before RASSI.
    requires_rassi_absorption : bool
        Whether the appended RASSI step runs in absorption mode
        (transition dipoles requested on top of another task).
    """

    def __init__(self, profile: str, modules: list,
                 requires_dual_rasscf: bool = False,
                 requires_rassi_absorption: bool = False,
                 method: str = '',
                 tasks=frozenset(),
                 is_generic: bool = False,
                 requires_rassi_soc: bool = False,
                 requires_mclr: bool = False,
                 requires_esfp: bool = False,
                 embedding_mode: str = 'none',
                 two_pass: bool = False,
                 pass_plan=None):
        self.profile = profile
        self.modules = modules
        self.requires_dual_rasscf = requires_dual_rasscf
        self.requires_rassi_absorption = requires_rassi_absorption
        self.method = method
        self.tasks = frozenset(tasks)
        self.is_generic = is_generic
        self.requires_rassi_soc = requires_rassi_soc
        self.requires_mclr = requires_mclr
        self.requires_esfp = requires_esfp
        self.embedding_mode = embedding_mode
        self.two_pass = two_pass
        self.pass_plan = list(pass_plan or [])

    def __repr__(self):
        return f"WorkflowGraph(profile='{self.profile}', modules={self.modules})"

    def has_module(self, module_name: str) -> bool:
        """Check if a module is in the workflow.

        Parameters
        ----------
        module_name : str
            Molcas module name (e.g. 'RASSCF', 'RASSI').

        Returns
        -------
        bool
            True if the module is in the workflow.
        """
        return module_name.upper() in [m.upper() for m in self.modules]

    def get_module_index(self, module_name: str) -> int:
        """Get the index of a module in the workflow.

        Parameters
        ----------
        module_name : str
            Molcas module name.

        Returns
        -------
        int
            Index of the module, -1 if not found.
        """
        upper_modules = [m.upper() for m in self.modules]
        try:
            return upper_modules.index(module_name.upper())
        except ValueError:
            return -1
