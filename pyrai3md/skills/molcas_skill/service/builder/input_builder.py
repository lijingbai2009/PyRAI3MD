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

This module assembles the complete Molcas input file from the compiled
workflow and configuration. It dispatches each Molcas module to the
corresponding builder function in module_builders.py.

"""

import os
import shutil
import warnings

from .module_builders import (
    build_gateway,
    build_seward,
    build_rasscf,
    build_scf,
    build_alaska,
    build_rassi,
    build_rassi_caspt2_soc,
    build_caspt2,
    build_slapaf,
    build_dynamix,
    build_surfacehop,
    build_mckinley,
)
from .embedding import resolve_embedding_field, build_esfp_block
from ..templates.molcas_templates import (
    COPY_TEMPLATE,
    COPY_JOBMIX_TEMPLATE,
    COPY_RUNFILE2_TEMPLATE,
    COPY_JOBIPH_S_TEMPLATE,
    COPY_JOBIPH_T_TEMPLATE,
    ESPF_CLEANUP_TEMPLATE,
    EXPORT_MAXITER_TEMPLATE,
    DO_WHILE_TEMPLATE,
    END_DO_TEMPLATE,
    COPY_XYZ_TEMPLATE,
    COPY_VELOCITY_TEMPLATE,
    COPY_STRORB_AS_RASORB_TEMPLATE,
    FOREACH_TEMPLATE,
    END_FOREACH_TEMPLATE,
    ALASKA_TEMPLATE,
)


def write_input_file(config, workflow, calcdir: str) -> str:
    """Build and write the Molcas input file to the calculation directory.

    Combines build_input() with file writing. The file is named
    $MOLCAS_PROJECT.inp and placed in calcdir.

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
    input_content = build_input(config, workflow, calcdir=calcdir)
    project = config.project or 'MOLCAS'
    input_path = os.path.join(calcdir, f'{project}.inp')
    with open(input_path, 'w') as f:
        f.write(input_content)
    return input_path


def _spin_alaska_blocks(config, target_states, state_spin, singlet,
                        root_offset, nac_pairs, need_gradient, need_nac,
                        has_alaska):
    """Emit the ALASKA blocks one spin group of a dual-RASSCF job needs.

    The singlet and triplet groups carry the same two asks with their own
    root offset: a gradient block after their RASSCF job, and a NAC block
    per same-spin pair. The root numbers of a wrapper-injected state are
    already rebased onto this group's ladder (``root_offset``), so the
    caller after RASSI reads blocks in ladder order.
    """
    blocks = []
    if need_gradient and has_alaska:
        targets = [
            s for s in target_states
            if 0 <= s < len(state_spin)
            and (state_spin[s] == 0) == bool(singlet)
        ]
        if targets:
            blocks.append(build_alaska(
                config, states=targets, root_offset=root_offset))
    if need_nac and has_alaska and nac_pairs:
        blocks.append(build_alaska(
            config, nac_mode=True, root_offset=root_offset,
            nac_pairs=nac_pairs,
            nocsf=bool(getattr(config, 'nocsf', True))))
    return blocks


def build_input(config, workflow, calcdir=None) -> str:
    """Build the complete Molcas input file content.

    Iterates over the workflow modules and dispatches each to the
    appropriate builder function. Handles special cases:
    - SOC dual RASSCF: singlet then triplet, with COPY commands;
      gradient requests (jobtype contains 'gradient') insert one
      ALASKA block per target state after its own spin RASSCF job
      (singlet roots then triplet roots, renumbered from 1), and NAC
      requests (jobtype contains 'nac') insert one ALASKA NAC block
      per same-spin pair after its own spin RASSCF job (triplet pairs
      renumbered by the number of singlet states), all before RASSI
    - Absorption: passes absorption flag to RASSI; a workflow that
      also contains CASPT2 uses the multistate RASSI template and
      copies JobMix to JOB001 before RASSI
    - NAC: passes nac_mode to ALASKA; a gradient request in the same
      jobtype adds ROOT blocks alongside the NAC blocks
    - Surface hopping: includes SURFACEHOP module
    - Geometry optimization: the input exports MOLCAS_MAXITER and
      wraps SEWARD/RASSCF/SLAPAF in a '>> Do while' loop. The mode
      derives from opt_state_i/opt_state_j: equal indices give a
      single-state relaxation (Rlxroot), a same-spin pair adds an
      'Ediff n m' GATEWAY constraint (MECI), and a cross-spin pair
      adds a bare 'Ediff' constraint with dual singlet/triplet RASSCF
      jobs whose Rlxroot keywords fix the crossing (MECP; triplet
      roots are renumbered by the singlet count)
    - Dynamics (ab initio MD / surface hopping): Molcas propagates
      the nuclei internally. The header copies the coordinate and
      velocity files into the working directory and exports
      MOLCAS_MAXITER (must exceed the MD step count); a
      '>> FOREACH ITER in (1 .. maxit_md)' loop wraps
      SEWARD..DYNAMIX, RASSCF carries MDRLXR and reads RasOrb for
      orbital continuation, and ALASKA stays bare
    - Frequency: the input exports MOLCAS_MAXITER (100 x natom) as
      its first line so internal iteration budgets stay generous
      while MCKINLEY evaluates the numerical hessian
    - Orbital guess: every fileorb line targets
      $MOLCAS_PROJECT.RasOrb and is only emitted when a staged user
      guess ({project}.StrOrb) exists; the StrOrb suffix is reserved
      for locking the user guess file, and without it RASSCF uses
      its default orbitals

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.
    calcdir : str, optional
        Calculation directory probed for the staged {project}.StrOrb
        guess. Defaults to the guess_orbital setting when omitted.

    Returns
    -------
    str
        Complete Molcas input file content.
    """
    modules = workflow.modules
    profile = workflow.profile

    # Generic property profiles and any ESPF run use the phase-based
    # builder below; legacy gas-phase profiles keep their existing
    # loop/dynamics/frequency logic. A mixed ladder joins the grouped
    # builder whatever the profile is called: one RASSCF job per manifold
    # is what the declared states need for their energies alone, and the
    # single-pass legacy templates answered a `casscf_gradient` ask with one
    # singlet job that was then asked for roots living in the other
    # manifold (#449).
    if groups_by_manifold(config, workflow):
        return _build_generic_input(config, workflow, calcdir=calcdir)

    is_soc = workflow.requires_dual_rasscf
    # Absorption profiles and transition requests appended to another
    # task both run RASSI in absorption mode (MEES/DIPRint=0).
    is_absorption = (
        profile in ('casscf_absorption', 'caspt2_absorption')
        or getattr(workflow, 'requires_rassi_absorption', False)
    )
    is_nac = profile == 'casscf_nac'
    is_optimize = profile == 'casscf_geometry_optimization'
    is_dynamics = profile in (
        'casscf_abinitio_dynamics', 'casscf_surface_hopping')
    is_frequency = profile in (
        'casscf_frequency', 'caspt2_frequency')
    has_caspt2 = 'CASPT2' in [m.upper() for m in modules]

    # Staged user orbital guess ({project}.StrOrb): gates every
    # fileorb line and the StrOrb->RasOrb COPY in the dynamics
    # header. Without it RASSCF keeps its default orbitals.
    has_strorb = _has_strorb(config, calcdir)
    warn_missing_orbital_guess(config, calcdir, has_strorb)

    # Optimization mode and crossing constraint, derived from the
    # requested target states (see _resolve_optimize_mode).
    opt_mode = None
    constraint_states = None
    if is_optimize:
        opt_mode = _resolve_optimize_mode(config)
        state_i = config.opt_state_i
        state_j = config.opt_state_j
        if opt_mode == 'meci':
            constraint_states = f' {state_i + 1} {state_j + 1}'
        elif opt_mode == 'mecp':
            constraint_states = ''

    # Dynamics MD root and iteration limit. Initial velocities are
    # mandatory: DYNAMIX reads them from $Project.velocity.xyz
    # (velo=1) and cannot fall back on a thermal guess here.
    md_root = None
    md_maxiter = None
    if is_dynamics:
        if not config.velocities:
            raise ValueError(
                'Molcas dynamics requires initial velocities '
                '(DYNAMIX velo=1 reads $Project.velocity.xyz)'
            )
        # MDRLXR tracks the initial MD state selected by
        # current_state (0-based, pushed from the framework system
        # or the agent config), not target_states, which normalize
        # keeps sorted.
        md_root = config.current_state + 1
        maxit_md = config.maxit_md or 1
        md_maxiter = config.md_maxiter or maxit_md * 10
        if md_maxiter <= maxit_md:
            raise ValueError(
                f'md_maxiter={md_maxiter} must exceed the MD step '
                f'count maxit_md={maxit_md}'
            )

    # Gradient requests: the SOC dual-RASSCF workflow always carries
    # the ALASKA module, but it is only emitted when the jobtype
    # requests gradients.  NAC requests are handled the same way: the
    # jobtype must carry a 'nac' token, and NAC blocks are emitted per
    # spin group in SOC workflows.
    jobtype_tokens = config.jobtype.split() if config.jobtype else []
    need_gradient = 'gradient' in jobtype_tokens
    need_nac = 'nac' in jobtype_tokens
    has_alaska = 'ALASKA' in [m.upper() for m in modules]
    target_states = config.target_states
    state_spin = config.state_spin
    n_singlet = sum(1 for s in state_spin if s == 0)

    # NAC pairs grouped by spin.  Only same-spin pairs are valid NAC
    # requests (cross-spin pairs are SOC couplings handled by RASSI).
    # NAC requires at least two states of the same spin: a spin group
    # with a single state has no NAC and emits no ALASKA NAC block.
    nac_pairs = [
        p for p in config.target_nac_pair if len(p) == 2
    ]
    singlet_nac_pairs = [
        p for p in nac_pairs
        if 0 <= p[0] < len(state_spin) and 0 <= p[1] < len(state_spin)
        and state_spin[p[0]] == 0 and state_spin[p[1]] == 0
    ]
    triplet_nac_pairs = [
        p for p in nac_pairs
        if 0 <= p[0] < len(state_spin) and 0 <= p[1] < len(state_spin)
        and state_spin[p[0]] != 0 and state_spin[p[1]] != 0
    ]
    if is_soc and need_nac and not singlet_nac_pairs and not triplet_nac_pairs:
        raise ValueError(
            'jobtype requests NAC but no same-spin state pair exists: '
            'NAC requires at least two states of the same spin.'
        )
    if (not is_soc and need_nac and not nac_pairs
            and config.nstate < 2):
        raise ValueError(
            'jobtype requests NAC but nstate < 2: '
            'NAC requires at least two states.'
        )

    input_blocks = []
    job_id = 0

    for module in modules:
        module_upper = module.upper()

        if module_upper == 'GATEWAY':
            input_blocks.append(build_gateway(
                config, soc=is_soc, constraint=constraint_states,
                use_xfield=False))
            if is_optimize:
                # GATEWAY stays outside the loop so SLAPAF-updated
                # coordinates are re-read on every iteration.
                input_blocks.append(DO_WHILE_TEMPLATE)
            elif is_dynamics:
                # GATEWAY stays outside the FOREACH loop: it defines
                # the starting geometry once, while DYNAMIX updates
                # the coordinates in $WorkDir on every MD step.
                input_blocks.append(FOREACH_TEMPLATE.format(
                    nstep=config.maxit_md or 1))

        elif module_upper == 'SEWARD':
            input_blocks.append(build_seward(config, soc=is_soc))

        elif module_upper == 'RASSCF':
            if is_optimize and opt_mode == 'mecp':
                # Cross-spin crossing: one RASSCF job per spin, each
                # relaxing its own crossing state. The triplet root
                # is renumbered by the number of singlet states.
                # Each spin job is followed by a bare ALASKA so the
                # constrained gradient of its crossing state reaches
                # SLAPAF; the singlet RunFile is locked as RUNFILE2
                # before the triplet job overwrites it.
                input_blocks.append(build_rasscf(
                    config, spin='singlet',
                    rlxroot=config.opt_state_i + 1,
                    fileorb=has_strorb))
                input_blocks.append(ALASKA_TEMPLATE)
                input_blocks.append(COPY_RUNFILE2_TEMPLATE)
                input_blocks.append(build_rasscf(
                    config, spin='triplet',
                    rlxroot=config.opt_state_j - n_singlet + 1,
                    fileorb=has_strorb))
                input_blocks.append(ALASKA_TEMPLATE)
            elif is_optimize:
                # Single-state or same-spin crossing relaxation.
                input_blocks.append(build_rasscf(
                    config, rlxroot=config.opt_state_i + 1,
                    fileorb=has_strorb))
            elif is_dynamics:
                # Initial-state MDRLXR plus RasOrb orbital
                # continuation between FOREACH iterations (only when
                # a StrOrb guess is staged).
                input_blocks.append(build_rasscf(
                    config, mdrlxr=md_root, fileorb=has_strorb))
            elif is_soc:
                # Dual RASSCF: singlet first, then triplet. Gradient
                # and NAC ALASKA blocks follow their own spin RASSCF
                # job (singlet roots before the triplet job, triplet
                # roots after it, all before RASSI) so each root
                # number refers to the correct JobIph.
                input_blocks.append(build_rasscf(
                    config, spin='singlet', fileorb=has_strorb))
                job_id += 1
                input_blocks.append(COPY_TEMPLATE.format(job_id=job_id))

                input_blocks.extend(_spin_alaska_blocks(
                    config, target_states, state_spin, singlet=True,
                    root_offset=0, nac_pairs=singlet_nac_pairs,
                    need_gradient=need_gradient, need_nac=need_nac,
                    has_alaska=has_alaska))

                input_blocks.append(build_rasscf(
                    config, spin='triplet', fileorb=has_strorb))
                job_id += 1
                input_blocks.append(COPY_TEMPLATE.format(job_id=job_id))

                input_blocks.extend(_spin_alaska_blocks(
                    config, target_states, state_spin, singlet=False,
                    root_offset=n_singlet, nac_pairs=triplet_nac_pairs,
                    need_gradient=need_gradient, need_nac=need_nac,
                    has_alaska=has_alaska))
            else:
                input_blocks.append(build_rasscf(
                    config, fileorb=has_strorb))

        elif module_upper == 'SCF':
            input_blocks.append(build_scf(config))

        elif module_upper == 'ALASKA':
            if is_soc:
                # Already emitted between the dual RASSCF jobs
                continue
            if is_dynamics:
                # DYNAMIX selects the gradient state via MDRLXR, so
                # ALASKA stays bare inside the MD loop.
                input_blocks.append(ALASKA_TEMPLATE)
                continue
            blocks = []
            if is_nac and need_nac:
                # NAC request: one NAC=i j block per pair.  A gradient
                # request in the same jobtype adds ROOT blocks after.
                blocks.append(build_alaska(
                    config, nac_mode=True,
                    nocsf=bool(getattr(config, 'nocsf', True))))
            if not is_nac or need_gradient:
                blocks.append(build_alaska(config))
            input_blocks.append(''.join(blocks))

        elif module_upper == 'RASSI':
            if is_soc:
                input_blocks.append(build_rassi(config, soc=True, dual_rasscf=True))
            elif is_absorption:
                input_blocks.append(
                    build_rassi(config, absorption=True, caspt2=has_caspt2)
                )
            else:
                input_blocks.append(build_rassi(config, soc=False))

        elif module_upper == 'CASPT2':
            input_blocks.append(build_caspt2(config))
            if is_absorption and has_caspt2:
                # CASPT2 absorption: the following RASSI Nrof JobIph
                # block must read the multistate CASPT2 states, which
                # are stored in JobMix rather than JobIph.
                input_blocks.append(COPY_JOBMIX_TEMPLATE)

        elif module_upper == 'SLAPAF':
            input_blocks.append(build_slapaf(config))
            if is_optimize:
                input_blocks.append(END_DO_TEMPLATE)

        elif module_upper == 'DYNAMIX':
            input_blocks.append(build_dynamix(config))
            if is_dynamics:
                input_blocks.append(END_FOREACH_TEMPLATE)

        elif module_upper == 'SURFACEHOP':
            input_blocks.append(build_surfacehop(config))

        elif module_upper == 'MCKINLEY':
            input_blocks.append(build_mckinley(config))

    if is_optimize:
        # MOLCAS_MAXITER bounds the Do while relaxation loop and is
        # also picked up by SLAPAF as its default iteration limit.
        input_blocks.insert(
            0, EXPORT_MAXITER_TEMPLATE.format(maxit=config.opt_maxit or 100))

    if is_dynamics:
        # Copy the initial coordinates and velocities into the Molcas
        # working directory, then export the total substitution step
        # limit (must exceed the MD step count). The header lines are
        # emitted as one contiguous block; the StrOrb guess COPY
        # (renamed to RasOrb) joins only when a guess is staged.
        header = COPY_XYZ_TEMPLATE + COPY_VELOCITY_TEMPLATE
        if has_strorb:
            header += COPY_STRORB_AS_RASORB_TEMPLATE
        header += EXPORT_MAXITER_TEMPLATE.format(maxit=md_maxiter)
        input_blocks.insert(0, header)

    if is_frequency:
        # MCKINLEY displaces every atom for the numerical hessian;
        # export a generous iteration budget (100 x natom) up front.
        input_blocks.insert(0, EXPORT_MAXITER_TEMPLATE.format(
            maxit=max(config.natom, 1) * 100))

    return '\n'.join(input_blocks)


def _spin_groups(config, dual):
    """Return (label, global_indices, offset) groups for this pass."""
    state_spin = list(getattr(config, 'state_spin', []) or [])
    if not dual:
        return [(None, list(range(len(state_spin))), 0)]
    groups = []
    for label in ('singlet', 'triplet'):
        if label == 'singlet':
            indices = [i for i, s in enumerate(state_spin) if s == 0]
        else:
            indices = [i for i, s in enumerate(state_spin) if s != 0]
        if indices:
            groups.append((label, indices, indices[0]))
    return groups


# Property-shaped profiles that can be answered by the grouped builder
# when the declared ladder spans two manifolds. The SOC, absorption,
# frequency, optimization and dynamics profiles keep their own writers.
_MIXED_LADDER_PROFILES = frozenset({
    'casscf_energy', 'casscf_gradient', 'casscf_nac', 'caspt2_energy',
})


def groups_by_manifold(config, workflow):
    """Whether this deck is written by the grouped (per-manifold) builder.

    One question, asked by the dispatch in `build_input` and by
    `rasscf_pass_count`, so the writer and the reader cannot disagree
    about how many RASSCF blocks a round holds.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        Compiled workflow for this run.

    Returns
    -------
    bool
        True when the deck groups its RASSCF passes by spin manifold.
    """
    if getattr(workflow, 'is_generic', False):
        return True
    profile = str(getattr(workflow, 'profile', '') or '')
    return deck_is_dual(config) and profile in _MIXED_LADDER_PROFILES


def deck_is_dual(config):
    """Whether the declared ladder spans more than one spin manifold.

    The deck's shape follows the declaration, not the token combination:
    a run that names a singlet and a triplet state needs one pass per
    manifold for its energies alone. Which gradients or couplings each
    manifold then carries is decided per group inside the loop, not by
    dropping a manifold the input declared.
    """
    state_spin = list(getattr(config, 'state_spin', []) or [])
    return len(set(state_spin)) > 1


def _pass_groups(config, workflow, plan):
    """Choose the RASSCF groups for a pass."""
    if deck_is_dual(config):
        return _spin_groups(config, True)
    return _spin_groups(config, False)


def rasscf_pass_count(config, workflow):
    """How many `&RASSCF` blocks one round of this deck prints.

    One source for a fact the builder and the reader both need: the
    reader used to re-derive it from the profile name, which meant a
    deck that grouped by spin (the generic path) was labelled as if it
    had never grouped, and its second pass carried the first pass's spin.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        The compiled workflow the deck was built from.

    Returns
    -------
    int
        Blocks per round, at least one.
    """
    if groups_by_manifold(config, workflow):
        return max(1, len(_pass_groups(config, workflow, {})))
    if getattr(workflow, 'requires_dual_rasscf', False):
        # The legacy SOC branch always writes one singlet and one
        # triplet job, whatever the ladder declares.
        return 2
    profile = str(getattr(workflow, 'profile', '') or '')
    if profile == 'casscf_geometry_optimization':
        # The same question the builder asks when it decides whether to
        # write one RASSCF job or two; 'meci' relaxes both surfaces
        # inside one spin job, so only 'mecp' doubles the blocks.
        try:
            return 2 if _resolve_optimize_mode(config) == 'mecp' else 1
        except ValueError:
            # A declaration the builder would have refused has no deck
            # shape to report; the reader does not get to re-adjudicate it.
            return 1
    return 1


def deck_loops(workflow):
    """Whether this deck reprints its blocks inside one input.

    SLAPAF's `Do while` and the `FOREACH` dynamics loop repeat the whole
    set of blocks every iteration, so only those logs need the
    "keep the last iteration" reset. A generic property pass is evaluated
    once per call, and resetting there would discard the first pass.
    """
    profile = str(getattr(workflow, 'profile', '') or '')
    return profile in ('casscf_geometry_optimization',
                       'casscf_abinitio_dynamics', 'casscf_surface_hopping')


def _local_states(config, indices):
    return [s for s in (config.target_states or []) if s in indices]


def nac_deck_order(pairs, state_spin):
    """The order in which the deck prints the requested couplings.

    One `&ALASKA` block is emitted per pair, group by group: the singlet
    group first, then the triplet group (`_pass_groups` / the dual-RASSCF
    branch), and inside a group the request's own order. The parser maps its
    rows back through this function, so what it returns is in the order the
    input declared - the identity of a row is the pair, not the slot the
    engine happened to print it in.

    Parameters
    ----------
    pairs : list of [int, int]
        The requested NAC pairs, 0-based global state indices.
    state_spin : list of int
        2S per state, the same list the deck groups on.

    Returns
    -------
    list of list of int
        The same pairs, grouped singlet-then-triplet, each group in the
        request's order.
    """
    valid = [list(pair) for pair in pairs if len(pair) == 2]
    singlet = [pair for pair in valid
               if state_spin[pair[0]] == 0 and state_spin[pair[1]] == 0]
    rest = [pair for pair in valid if pair not in singlet]
    return singlet + rest


def _local_nac_pairs(config, indices):
    return [pair for pair in (config.target_nac_pair or [])
            if len(pair) == 2 and pair[0] in indices and pair[1] in indices]


def _build_generic_pass(config, workflow, plan, calcdir, has_strorb):
    """Build one property pass (gas phase or ESPF)."""
    method = workflow.method
    state_spin = list(getattr(config, 'state_spin', []) or [])
    blocks = []

    blocks.append(build_gateway(
        config, soc=workflow.requires_rassi_soc, use_xfield=False))
    blocks.append(build_seward(config, soc=workflow.requires_rassi_soc))
    # The field is resolved once - every pass embeds the same classical
    # charges - but the block is written per `&RASSCF`: the ESP integrals
    # live in one pair of files, so a second manifold whose pass had no
    # `&ESPF` of its own would read the file the first pass left behind
    # and embed a density that is not its own (#448).
    field = (resolve_embedding_field(config, calcdir)
             if plan.get('espf') else None)

    groups = _pass_groups(config, workflow, plan)
    caspt2_soc = (method == 'caspt2' and plan.get('soc'))
    caspt2_multistate = (
        str(getattr(config, 'caspt2_type', '') or 'xms').lower() != 'ss')
    jobiph_names = []
    for group_index, (label, indices, offset) in enumerate(groups):
        local_targets = _local_states(config, indices)
        local_pairs = _local_nac_pairs(config, indices)
        local_gradient = bool(plan.get('gradient') and local_targets)
        local_nac = bool(plan.get('nac') and local_pairs)
        nocsf = bool(getattr(config, 'nocsf', True))

        if field is not None:
            if group_index:
                blocks.append(ESPF_CLEANUP_TEMPLATE)
            blocks.append(build_esfp_block(field))

        blocks.append(build_rasscf(
            config, spin=label, fileorb=has_strorb))

        if method == 'caspt2':
            blocks.append(build_caspt2(
                config, gradient=local_gradient or local_nac))
            if caspt2_soc:
                name = 'JOBIPH_T' if label == 'triplet' else 'JOBIPH_S'
                jobiph_names.append(name)
                blocks.append(
                    COPY_JOBIPH_T_TEMPLATE if name == 'JOBIPH_T'
                    else COPY_JOBIPH_S_TEMPLATE)
            if local_gradient:
                blocks.append(build_alaska(
                    config, states=local_targets, root_offset=offset))
            if local_nac:
                blocks.append(build_alaska(
                    config, nac_mode=True, root_offset=offset,
                    nac_pairs=local_pairs, nocsf=nocsf))
        else:
            # CASSCF: ALASKA selects roots/pairs itself.
            if local_nac:
                blocks.append(build_alaska(
                    config, nac_mode=True, root_offset=offset,
                    nac_pairs=local_pairs, nocsf=nocsf))
            if local_gradient:
                blocks.append(build_alaska(
                    config, states=local_targets, root_offset=offset))
            if workflow.requires_dual_rasscf and plan.get('soc'):
                job_id = 1 if label == 'singlet' else 2
                blocks.append(COPY_TEMPLATE.format(job_id=job_id))

    if plan.get('soc'):
        if method == 'caspt2':
            blocks.append(build_rassi_caspt2_soc(
                config, multistate=caspt2_multistate,
                jobiph_names=jobiph_names or ['JOBIPH_S']))
        else:
            blocks.append(build_rassi(
                config, soc=True,
                dual_rasscf=workflow.requires_dual_rasscf))
    elif plan.get('absorption'):
        if method == 'caspt2':
            blocks.append(COPY_JOBMIX_TEMPLATE)
        blocks.append(build_rassi(
            config, absorption=True, caspt2=(method == 'caspt2')))

    return ''.join(blocks)


def _build_generic_input(config, workflow, calcdir=None):
    """Build one or two property passes for a generic/ESPF workflow."""
    has_strorb = _has_strorb(config, calcdir)
    warn_missing_orbital_guess(config, calcdir, has_strorb)
    plans = workflow.pass_plan or [{
        'espf': workflow.requires_esfp,
        'energy': True,
        'gradient': 'gradient' in workflow.tasks,
        'nac': 'nac' in workflow.tasks,
        'soc': 'soc' in workflow.tasks,
        'absorption': 'absorption' in workflow.tasks,
    }]
    texts = []
    for index, plan in enumerate(plans):
        if index:
            texts.append(ESPF_CLEANUP_TEMPLATE)
        texts.append(_build_generic_pass(
            config, workflow, plan, calcdir, has_strorb))
    return '\n'.join(texts)


def warn_missing_orbital_guess(config, calcdir, has_strorb):
    """Warn when a run starts RASSCF without a user orbital guess.

    One `{project}.RasOrb` per calculation directory is the design, not
    an oversight: the deck runs the singlet group before the triplet
    group, the parser copies the last pass's RasOrb back to StrOrb, and
    the next call starts from it. Without a staged guess no `fileorb`
    line is written and each pass keeps Molcas' own default orbitals -
    enough to keep the shipped cases runnable without an orbital file,
    but the default can converge on an active space other than the one
    the input describes.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str, optional
        Calculation directory the guess would be staged in.
    has_strorb : bool
        Whether a staged user guess was found.
    """
    if has_strorb or not calcdir:
        return
    project = config.project or 'MOLCAS'
    warnings.warn(
        f'Molcas: no {project}.StrOrb staged in {calcdir}; RASSCF runs on '
        'the program default orbitals (no fileorb line), which can converge '
        'on an active space other than the one the input declares. Provide '
        'an initial guess for production runs.',
        UserWarning, stacklevel=2,
    )


def _has_strorb(config, calcdir=None) -> bool:
    """Detect a staged user orbital guess ({project}.StrOrb).

    The StrOrb suffix locks the user-provided guess. When calcdir is
    given, the staged file in calcdir decides; otherwise the explicit
    guess_orbital setting is used (direct build_input callers).

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str, optional
        Calculation directory probed for {project}.StrOrb.

    Returns
    -------
    bool
        True when a user orbital guess is available.
    """
    if calcdir:
        project = config.project or 'MOLCAS'
        return os.path.exists(
            os.path.join(calcdir, f'{project}.StrOrb'))
    return bool(config.guess_orbital)


def _resolve_optimize_mode(config) -> str:
    """Derive the geometry optimization mode from the target states.

    Equal opt_state_i/opt_state_j give a single-state relaxation; a
    same-spin pair triggers MECI and a cross-spin pair triggers MECP.
    For MECP, opt_state_i must be the singlet state and opt_state_j
    the triplet state. The CI root count of each involved spin group
    must cover the requested roots.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    str
        'single', 'meci' or 'mecp'.

    Raises
    ------
    ValueError
        If the target state indices are out of range, the MECP spin
        assignment is inverted, or the CI roots do not cover the
        requested relaxation roots.
    """
    state_i = config.opt_state_i
    state_j = config.opt_state_j
    state_spin = config.state_spin

    for label, s in (('opt_state_i', state_i), ('opt_state_j', state_j)):
        if not 0 <= s < len(state_spin):
            raise ValueError(
                f'{label}={s} is out of range for state_spin of '
                f'length {len(state_spin)}'
            )

    n_singlet = sum(1 for s in state_spin if s == 0)

    if state_i == state_j:
        mode = 'single'
    elif state_spin[state_i] == state_spin[state_j]:
        mode = 'meci'
    else:
        mode = 'mecp'
        if state_spin[state_i] != 0:
            raise ValueError(
                'MECP optimization requires opt_state_i to be the '
                'singlet state and opt_state_j the triplet state'
            )

    # CI roots must cover the relaxed roots of each involved job.
    if mode == 'mecp':
        singlet_roots = (config.singlet_ci_roots or [1])[0]
        triplet_roots = (config.triplet_ci_roots or [1])[0]
        if singlet_roots < state_i + 1:
            raise ValueError(
                f'singlet_ci_roots[0]={singlet_roots} does not cover '
                f'Rlxroot={state_i + 1} (opt_state_i={state_i})'
            )
        if triplet_roots < state_j - n_singlet + 1:
            raise ValueError(
                f'triplet_ci_roots[0]={triplet_roots} does not cover '
                f'Rlxroot={state_j - n_singlet + 1} '
                f'(opt_state_j={state_j})'
            )
    else:
        ci_roots = (config.ci_roots or [1])[0]
        needed = max(state_i, state_j) + 1
        if ci_roots < needed:
            raise ValueError(
                f'ci_roots[0]={ci_roots} does not cover the requested '
                f'optimization roots (need >= {needed})'
            )

    return mode


def write_xyz_file(config, calcdir: str):
    """Write the XYZ coordinate file for Molcas.

    The file is named $MOLCAS_PROJECT.xyz and is referenced by
    the GATEWAY template via coord=$MOLCAS_PROJECT.xyz.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str
        Calculation directory.
    """
    project = config.project or 'MOLCAS'
    atoms = config.atoms
    coordinates = config.coordinates
    natom = len(atoms)

    xyz_path = os.path.join(calcdir, f'{project}.xyz')
    with open(xyz_path, 'w') as f:
        f.write(f'{natom}\n')
        f.write(f'{project} geometry\n')
        for atom, coord in zip(atoms, coordinates):
            f.write(f'{atom:>4s}  {coord[0]:16.10f}  {coord[1]:16.10f}  {coord[2]:16.10f}\n')


def write_velocity_file(config, calcdir: str):
    """Write the initial velocity file for Molcas dynamics.

    The file is named $MOLCAS_PROJECT.velocity.xyz and is read by
    DYNAMIX (velo=1) after being copied into the working directory.
    Despite the .xyz suffix the content is NOT standard xyz format:
    Molcas expects a plain N x 3 numeric block -- no atom count
    line, no comment line, no element-symbol column. Velocities
    are written in Bohr/au, matching the PyRAI3MD convention
    (title.velo).

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str
        Calculation directory.
    """
    project = config.project or 'MOLCAS'
    velocities = config.velocities

    velo_path = os.path.join(calcdir, f'{project}.velocity.xyz')
    with open(velo_path, 'w') as f:
        for velo in velocities:
            f.write(f'{velo[0]:16.10f}  {velo[1]:16.10f}  {velo[2]:16.10f}\n')


def manage_orbitals(config, calcdir: str):
    """Manage orbital file transfer between calculations.

    Handles the orbital guess staging:
    - The user guess (guess_orbital, or the {project}.StrOrb sitting
      in the current working directory when no explicit guess is
      given) is locked in calcdir under the StrOrb suffix; the input
      only ever references RasOrb-suffixed orbital files
    - The guess is additionally copied to calcdir RasOrb so the
      fileorb=$MOLCAS_PROJECT.RasOrb line resolves before the first
      RASSCF run; iterative loops continue from the RasOrb each run
      writes
    - Supports JobIph files for state-averaged calculations

    Existing StrOrb/RasOrb files in calcdir are never overwritten so
    that multi-step runs keep the orbitals converged by earlier
    steps.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str
        Calculation directory.
    """
    project = config.project or 'MOLCAS'
    guess = config.guess_orbital

    if guess and guess.endswith('.JobIph'):
        if os.path.exists(guess):
            jobiph_dst = os.path.join(calcdir, os.path.basename(guess))
            if not os.path.exists(jobiph_dst):
                shutil.copy2(guess, jobiph_dst)
        return

    # Resolve the guess source: the explicit guess_orbital path, or
    # the {project}.StrOrb locked next to the launching directory.
    source = None
    if guess and os.path.exists(guess):
        source = guess
    elif os.path.exists(f'{project}.StrOrb'):
        source = f'{project}.StrOrb'

    if source is None:
        return

    strorb_path = os.path.join(calcdir, f'{project}.StrOrb')
    rasorb_path = os.path.join(calcdir, f'{project}.RasOrb')
    if not os.path.exists(strorb_path):
        shutil.copy2(source, strorb_path)
    if not os.path.exists(rasorb_path):
        shutil.copy2(source, rasorb_path)
