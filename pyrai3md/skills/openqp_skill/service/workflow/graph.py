"""
======================================================================
This is a standalone OpenQP calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module defines the WorkflowGraph class that represents
the OpenQP input section chain and execution settings.

"""


class WorkflowGraph:
    """Represents the OpenQP input structure for one calculation.

    A workflow graph is compiled from a profile and defines the
    ordered list of OpenQP input sections, the runtype, and the
    execution flags needed to build the input file.

    Attributes
    ----------
    profile : str
        The calculation profile name.
    sections : list of str
        Ordered list of OpenQP input section names.
    runtype : str
        OpenQP [input]runtype (energy, grad, prop, optimize, hess).
    is_nac_profile : bool
        Whether the workflow involves nonadiabatic coupling
        (runtype=prop), i.e. the [guess]file2 orbital continuation
        mechanism may be needed for nac_type=dcm.
    is_soc : bool
        Whether the run computes spin-orbit coupling.  Two-manifold by
        itself does not imply SOC; the parser's SOC branch does.  Named as
        in the orca skill's WorkflowGraph.
    two_manifold : bool
        Whether both spin ladders are solved and published.  OpenQP labels
        every export with the manifold it holds in that case
        (`energies_s`, `grad_t_2`, ...), which is the difference between
        reading the right file and reading nothing.
    nac_vectors : bool
        Whether the run asks for numerical NAC vectors through
        [properties] nac=nac, the one coupling flavour runtype=data answers.
    """

    def __init__(self, profile: str, sections: list, runtype: str,
                 is_nac_profile: bool = False, is_soc: bool = False,
                 two_manifold: bool = False, nac_vectors: bool = False):
        self.profile = profile
        self.sections = sections
        self.runtype = runtype
        self.is_nac_profile = is_nac_profile
        self.is_soc = is_soc
        self.two_manifold = two_manifold
        self.nac_vectors = nac_vectors

    def __repr__(self):
        return (f"WorkflowGraph(profile='{self.profile}', sections={self.sections}, "
                f"runtype='{self.runtype}', is_soc={self.is_soc}, "
                f"two_manifold={self.two_manifold})")

    def has_section(self, section_name: str) -> bool:
        """Check if a section is in the workflow.

        Parameters
        ----------
        section_name : str
            OpenQP input section name (e.g. 'scf', 'tdhf', 'nac').

        Returns
        -------
        bool
            True if the section is in the workflow.
        """
        return section_name.lower() in [s.lower() for s in self.sections]

    def get_section_index(self, section_name: str) -> int:
        """Get the index of a section in the workflow.

        Parameters
        ----------
        section_name : str
            OpenQP input section name.

        Returns
        -------
        int
            Index of the section, -1 if not found.
        """
        lower_sections = [s.lower() for s in self.sections]
        try:
            return lower_sections.index(section_name.lower())
        except ValueError:
            return -1
