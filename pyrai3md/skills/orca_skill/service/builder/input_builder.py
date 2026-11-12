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

This module assembles the complete ORCA input file from the compiled
workflow and configuration. Coordinates are written in Angstrom
(framework unit, ORCA default).

"""

import os
import warnings

from ...schema.reference import DISPERSION_KEYWORD
from ..utils import writes_engrad, writes_nacme
from ..templates.orca_templates import (
    KEYWORD_TEMPLATE,
    PAL_TEMPLATE,
    MAXCORE_TEMPLATE,
    SCF_TEMPLATE,
    GEOM_TEMPLATE,
    TDDFT_TEMPLATE,
    POINTCHARGES_TEMPLATE,
    POINTCHARGES_FILE_TEMPLATE,
    GEOMETRY_TEMPLATE,
)


def write_input_file(config, workflow, calcdir: str) -> str:
    """Build and write the ORCA input file to the calculation directory.

    Combines build_input() with file writing. The file is named
    $ORCA_PROJECT.inp and placed in calcdir. Embedding charges go to a
    $ORCA_PROJECT.pc side file in the same directory.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.
    calcdir : str
        Calculation directory.

    Returns
    -------
    str
        Path to the written input file.
    """
    input_content = build_input(config, workflow)
    project = config.project
    input_path = os.path.join(calcdir, f'{project}.inp')
    with open(input_path, 'w') as f:
        f.write(input_content)

    # The input references the charges by bare file name, so the file
    # has to sit beside it before the run starts.
    charge_content = build_pointcharges_file(config)
    if charge_content:
        charge_path = os.path.join(
            calcdir, pointcharges_filename(config))
        with open(charge_path, 'w') as f:
            f.write(charge_content)

    return input_path


def build_input(config, workflow) -> str:
    """Build the complete ORCA input file content.

    Assembles the input file from the workflow steps:
    - Main keyword line: functional, basis, and job keywords
      (EnGrad, Opt, OptTS, AnFreq/NumFreq, PrintBasis, TightSCF);
      EnGrad is mutually exclusive with Opt/OptTS -- ORCA treats an
      EnGrad input as a gradient single point and overrides the
      optimization task; NACME lives inside the %tddft block, not on
      the keyword line
    - SOC calculations run on a closed-shell singlet (RHF) reference
      with DoSOC and Triplets true in %tddft; ORCA rejects any other
      reference for spin-orbit coupling
    - %pal and %maxcore resource blocks
    - %scf block (MaxIter aligned with the scf_maxit config, plus
      Print[P_MOs] 1 to dump the MO coefficients)
    - %geom block for optimizations (MaxIter aligned with opt_maxit)
    - %tddft block (nroots, TDA, irootlist for excited states)
    - %pointcharges file reference for embedding charges (the rows live
      in a side file; ORCA refuses them inline)
    - Geometry block in Angstrom

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    str
        Complete ORCA input file content.
    """
    blocks = []

    blocks.append(build_keyword_line(config, workflow))

    # Integration grid level rides on its own keyword line: ORCA
    # accepts multiple '!' lines and DEFGRID2 keeps the default
    # layout of the main line untouched.
    if config.grid_level in (1, 2, 3):
        blocks.append(f'! DEFGRID{config.grid_level}')

    blocks.append(PAL_TEMPLATE.format(nproc=config.nproc))
    blocks.append(MAXCORE_TEMPLATE.format(memory=config.orca_mem))
    blocks.append(SCF_TEMPLATE.format(scf_maxit=config.scf_maxit))

    if workflow.is_tddft:
        blocks.append(build_tddft_block(config, workflow))

    if workflow.is_optimization:
        blocks.append(GEOM_TEMPLATE.format(opt_maxit=config.opt_maxit))

    pointcharge_block = build_pointcharges_block(config)
    if pointcharge_block:
        blocks.append(pointcharge_block)

    blocks.append(build_geometry_block(config))

    return '\n'.join(blocks)


def build_keyword_line(config, workflow) -> str:
    """Build the main ORCA keyword line.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    str
        Keyword line starting with '!'.
    """
    functional = config.functional
    basis = config.basis
    num_freq = config.num_freq

    keywords = []

    keywords.append(functional)
    keywords.append(basis)

    # Density fitting: RIJCOSX plus the matching auxiliary basis
    # follow the basis set (both omitted when disabled). RIJCOSX is
    # fully compatible with TDDFT gradients and numerical Hessians.
    if config.density_fitting:
        keywords.append('RIJCOSX')
        if config.aux_basis:
            keywords.append(config.aux_basis)

    # Grimme dispersion correction (d3 -> D3BJ, d3zero -> D3ZERO,
    # d4 -> D4); empty or unrecognized values skip the keyword
    # (validate normalizes 'none' to '' beforehand).
    dispersion_keyword = DISPERSION_KEYWORD.get(
        config.dispersion_correction, '')
    if dispersion_keyword:
        keywords.append(dispersion_keyword)

    # EnGrad requests an energy+gradient single point and must never share
    # the keyword line with Opt/OptTS (ORCA would override the optimization
    # task) or with NACME (measured: the TD-DFT part finishes and then ORCA
    # aborts on the missing `.nacme.tmp`, return code 64). The rule lives in
    # one predicate because the parser has to know whether to expect a
    # gradient block at all.
    if writes_engrad(workflow, config):
        keywords.append('EnGrad')

    if workflow.is_transition_state:
        keywords.append('OptTS')
    elif workflow.is_optimization:
        keywords.append('Opt')

    if workflow.is_frequency:
        # ORCA has no analytical TDDFT second derivatives, so
        # excited-state frequencies always run with NumFreq; the
        # num_freq switch only controls ground-state DFT.
        if workflow.is_tddft or num_freq:
            keywords.append('NumFreq')
        else:
            keywords.append('AnFreq')

    # Always print the basis function table for inspection
    keywords.append('PrintBasis')

    keywords.append('TightSCF')

    return KEYWORD_TEMPLATE.format(keywords=' '.join(keywords))


def _split_target_roots(target_states, config):
    """Split framework target states into ORCA singlet/triplet roots.

    ORCA selects singlet gradient roots via 'irootlist' (root 0 is
    the ground state) and triplet roots via 'trootlist' (1-based
    within the triplet roots). The framework orders states as S0,
    excited singlets, then triplets, so a target s <= nsing_root is
    singlet root s and a target f > nsing_root is triplet root
    f - nsing_root. Both lists are returned in ascending order.

    Parameters
    ----------
    target_states : list of int
        Framework state indices (0-based, ground state included).
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    tuple of list
        (irootlist, trootlist) root numbers, each ascending.
    """
    irootlist = sorted(s for s in target_states
                       if s <= config.nsing_root)
    trootlist = sorted(s - config.nsing_root for s in target_states
                       if s > config.nsing_root)
    return irootlist, trootlist


def build_tddft_block(config, workflow) -> str:
    """Build the %tddft block.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    str
        The %tddft block text.
    """
    tddft_nroots = config.tddft_nroots
    target_states = config.target_states

    lines = [f'  nroots {tddft_nroots}']

    # Tamm-Dancoff approximation switch
    if config.tda:
        lines.append('  TDA true')

    # Excited-state gradients: singlet roots (including root 0 for
    # the ground state) go to irootlist and triplet roots to
    # trootlist, both in ascending order.
    if writes_engrad(workflow, config) or writes_nacme(workflow, config):
        # The root list is also the NACME request: without it a run prints the
        # coupling of root 1 only (measured: round `tmp_20261101_224544`
        # printed one `CARTESIAN NON-ADIABATIC COUPLINGS` block for `IROOT 1`
        # while the request named two pairs). `EnGrad` stays separate - it is
        # the keyword ORCA refuses beside NACME. A coupling-only deck names
        # the roots of the pairs, so the block count is the pair count and the
        # rows can be addressed by that order.
        roots = target_states
        if writes_nacme(workflow, config) and not writes_engrad(workflow, config):
            roots = sorted({int(max(pair)) for pair in
                            (config.target_nac_pair or [])
                            if len(pair) == 2})
        irootlist, trootlist = _split_target_roots(roots, config)
        if irootlist:
            lines.append(
                f"  irootlist {', '.join(str(r) for r in irootlist)}")
        if trootlist:
            lines.append(
                f"  trootlist {', '.join(str(r) for r in trootlist)}")

    # Excited-state geometry optimization: relax the current state
    # (singlet roots via irootlist, triplet roots via trootlist).
    if workflow.is_optimization and config.current_state > 0:
        if config.current_state <= config.nsing_root:
            lines.append(f'  irootlist {config.current_state}')
        else:
            root = config.current_state - config.nsing_root
            lines.append(f'  trootlist {root}')

    # Triplet roots: ORCA only computes triplet excitations when
    # 'Triplets true' is set - required for SOC (DoSOC) and for any
    # run whose state list contains triplet states.
    if workflow.is_soc or config.ntrip_root > 0:
        lines.append('  Triplets true')

    # Spin-orbit coupling: ORCA requires a closed-shell singlet
    # (RHF) reference and computes the singlet-triplet SOC matrix
    # from the triplet roots enabled above.
    if workflow.is_soc:
        lines.append('  DoSOC true')

    # Nonadiabatic coupling vectors between S0 and excited states;
    # ORCA reads the NACME switch from the %tddft block (it is not
    # a main keyword-line keyword).
    if workflow.has_step('nac'):
        if not writes_nacme(workflow, config):
            # ORCA refuses this combination outright - `Error (ORCA_CIS): The
            # NACMEs are currently not compatible with point charges!`
            # (main_input_check.cpp:10261) - so the switch is not written. The
            # step then reports no coupling row, which is what the run has.
            warnings.warn(
                "ORCA: NACME is omitted because this run embeds point "
                "charges; the engine refuses the combination, so no "
                "nonadiabatic coupling vector will be reported",
                UserWarning, stacklevel=2)
        else:
            lines.append('  NACME true')

    return TDDFT_TEMPLATE.format(tddft_lines='\n'.join(lines))


def build_pointcharges_block(config) -> str:
    """Build the %pointcharges line for embedding charges.

    ORCA reads background charges from an external file only; an inline
    block is rejected at scan time. The reference carries a bare file
    name, resolved against the calculation directory where ORCA runs.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    str
        The %pointcharges line, empty if no point charges.
    """
    if len(config.point_charges) == 0:
        return ''

    return POINTCHARGES_TEMPLATE.format(
        filename=pointcharges_filename(config))


def pointcharges_filename(config) -> str:
    """Name of the point-charge file next to the input.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    str
        Bare file name, without the calculation directory.
    """
    return f'{config.project}.pc'


def build_pointcharges_file(config) -> str:
    """Build the point-charge file content.

    The first line is the charge count; each following line gives the
    charge in atomic units and its position in Angstrom, the frame the
    geometry block already uses.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    str
        The file text, empty if no point charges.
    """
    point_charges = config.point_charges

    if len(point_charges) == 0:
        return ''

    charge_lines = []
    for pc in point_charges:
        q, x, y, z = float(pc[0]), float(pc[1]), float(pc[2]), float(pc[3])
        charge_lines.append(f'{q:20.12f} {x:20.12f} {y:20.12f} {z:20.12f}')

    return POINTCHARGES_FILE_TEMPLATE.format(
        ncharges=len(charge_lines),
        charge_block='\n'.join(charge_lines),
    )


def build_geometry_block(config) -> str:
    """Build the geometry block in Angstrom.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    str
        The geometry block text.
    """
    atoms = config.atoms
    coordinates = config.coordinates
    charges = config.charges
    multiplicity = config.multiplicity

    coord_lines = []
    for atom, coord in zip(atoms, coordinates):
        coord_lines.append(
            f'{atom:>4s}  {coord[0]:16.10f}  {coord[1]:16.10f}  {coord[2]:16.10f}'
        )

    return GEOMETRY_TEMPLATE.format(
        charge=charges,
        mult=multiplicity,
        coord_block='\n'.join(coord_lines) + '\n',
    )
