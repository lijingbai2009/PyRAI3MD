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

This module assembles the complete OpenQP INI input file from the
compiled workflow and configuration. Each workflow section is
dispatched to a dedicated section builder.

State index conventions (verified against the OpenQP source):
- hf ground state: grad=0, energies[0]
- other methods: framework state k (0-based) maps to OpenQP index
  k+1 (grad=k+1, energies[k+1]); in MRSF grad=1 is S0.

"""

import os

from ..templates.openqp_templates import (
    INPUT_TEMPLATE,
    GUESS_TEMPLATE,
    SCF_TEMPLATE,
    TDHF_TEMPLATE,
    PROPERTIES_TEMPLATE,
    NAC_TEMPLATE,
    OPTIMIZE_TEMPLATE,
    HESS_TEMPLATE,
)


def write_input_file(config, workflow, calcdir: str) -> str:
    """Build and write the OpenQP input file to the calculation directory.

    Combines build_input() with file writing. The file is named
    $OPENQP_PROJECT.inp and placed in calcdir, so the OpenQP CLI
    produces the fixed-name log and orbital json files
    ($OPENQP_PROJECT.log, $OPENQP_PROJECT.json).

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
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
    return input_path


def build_input(config, workflow) -> str:
    """Build the complete OpenQP input file content.

    Iterates over the workflow sections and dispatches each to the
    appropriate section builder.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    str
        Complete OpenQP INI input file content.
    """
    section_builders = {
        'input': build_input_section,
        'guess': build_guess_section,
        'scf': build_scf_section,
        'tdhf': build_tdhf_section,
        'properties': build_properties_section,
        'nac': build_nac_section,
        'optimize': build_optimize_section,
        'hess': build_hess_section,
    }

    input_blocks = []
    for section in workflow.sections:
        builder = section_builders.get(section.lower())
        if builder is None:
            continue
        block = builder(config, workflow)
        if block:
            input_blocks.append(block)

    return '\n\n'.join(input_blocks) + '\n'


def build_input_section(config, workflow) -> str:
    """Build the [input] section.

    Method mapping:
    - hf    -> method=hf (no functional)
    - tddft -> method=tdhf + functional ([tdhf]type=rpa/tda)
    - mrsf  -> method=tdhf + functional ([tdhf]type=mrsf)

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    str
        [input] section content.
    """
    method = config.method
    project = config.project

    oqp_method = 'hf' if method == 'hf' else 'tdhf'

    functional_line = ''
    if method in ('tddft', 'mrsf'):
        functional_line = f"\nfunctional={config.functional}"

    # The two-electron spin-orbit term is not available with
    # spherical-harmonic functions: the native kernel aborts partway
    # through the run (upstream refuses it in the checker with
    # "set soc_2e=0 or ispher=false").  Every shipped OpenQP example that
    # computes SOC pins Cartesian functions for the same reason, so a
    # spin-orbit run does too rather than depending on which shells the
    # chosen basis happens to resolve.
    input_extra_line = ''
    if workflow.is_soc:
        input_extra_line = f"\nispher=false\nsoc_2e={config.soc_2e}"

    return INPUT_TEMPLATE.format(
        system=f'{project}.xyz',
        charge=config.charges,
        method=oqp_method,
        runtype=workflow.runtype,
        basis=config.basis,
        functional_line=functional_line,
        input_extra_line=input_extra_line,
    )


def build_guess_section(config, workflow) -> str:
    """Build the [guess] section.

    Fixed-name orbital continuation mechanism:
    - save_mol=true is always on: every calculation stores the orbital
      json as ${OPENQP_PROJECT}.json (named after the log file).
    - file=${OPENQP_PROJECT}.json is always declared; with type=auto
      OpenQP reads it when present and falls back to the huckel guess
      on the first step (no json yet).
    - file2=${OPENQP_PROJECT}.json is written ONLY for nacme
      calculations (nac_type=dcm on the runtype=prop path): OpenQP
      reads the previous-step orbitals from file2 to compute the MO
      overlap. Other workflows never write file2, so OpenQP cannot
      exit on a missing json.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    str
        [guess] section content.
    """
    project = config.project
    guess_type = config.guess_type or 'auto'
    guess_file = f'{project}.json'

    file2_line = ''
    if workflow.is_nac_profile and workflow.runtype == 'prop' and config.nac_type == 'dcm':
        file2_line = f"\nfile2={guess_file}"

    return GUESS_TEMPLATE.format(
        guess_type=guess_type,
        guess_file=guess_file,
        file2_line=file2_line,
    )


def build_scf_section(config, workflow) -> str:
    """Build the [scf] section.

    The mrsf method forces a triplet ROHF reference (multiplicity=3);
    otherwise the system multiplicity is used.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    str
        [scf] section content.
    """
    if config.method == 'mrsf':
        multiplicity = 3
    else:
        multiplicity = config.multiplicity

    # Each line is written only when the input states something other than
    # what OpenQP already does by default.  escalation in particular is not
    # a knob to turn on by habit: an empty value means the reactive ladder
    # diis -> soscf -> trah, while naming one stage replaces that ladder
    # with the named stage alone, so writing `escalation=trah` as a default
    # would quietly drop the soscf step.
    extra_lines = []
    if config.scf_converger:
        extra_lines.append(f"\nconverger_type={config.scf_converger}")
    if config.scal_rel:
        extra_lines.append(f"\nscal_rel={config.scal_rel}")
    if config.escalation:
        extra_lines.append(f"\nescalation={config.escalation}")

    return SCF_TEMPLATE.format(
        scf_type=config.scf_type,
        multiplicity=multiplicity,
        scf_maxit=config.scf_maxit,
        scf_converge=config.scf_converge,
        scf_extra_line=''.join(extra_lines),
    )


def build_tdhf_section(config, workflow) -> str:
    """Build the [tdhf] section (tddft and mrsf methods only).

    For mrsf, [tdhf]multiplicity=1 selects singlet target states
    from the triplet reference.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    str
        [tdhf] section content, empty for the hf method.
    """
    if config.method == 'hf':
        return ''

    multiplicity_line = ''
    if config.method == 'mrsf':
        multiplicity_line = '\nmultiplicity=1'

    # A two-manifold run sizes each ladder on its own.  [tdhf]nstate then
    # describes only the ladder the deck configures - the singlet one, since
    # [tdhf]multiplicity=1 above - and must not be the total across both,
    # which is what a caller reading `nstate` off the state list would write.
    nstate = config.nstate
    ladder_line = ''
    if workflow.two_manifold:
        nstate = config.nstate_s or config.nstate
        ladder_line = (f"\nnstate_s={config.nstate_s}"
                       f"\nnstate_t={config.nstate_t}")

    return TDHF_TEMPLATE.format(
        td_type=config.td_type,
        nstate=nstate,
        td_maxit=config.td_maxit,
        td_converge=config.td_converge,
        multiplicity_line=multiplicity_line,
        ladder_line=ladder_line,
    )


def spin_ladder_selectors(config):
    """Root lists per spin ladder for a two-manifold run.

    A spin-orbit run addresses its gradients by ladder (`grad_s` /
    `grad_t`), because a single index cannot say which of the two ladders
    it counts.  The framework names states in one flat list ordered
    singlet-then-triplet (`state_spin` is sorted that way), so a state k
    below the singlet count is root k+1 of the singlet ladder and one at
    or above it is root k-nstate_s+1 of the triplet ladder.  Roots count
    from 1: root 0 of a ladder is its internal MRSF reference, which no
    caller asks for.

    An explicitly written grad_s / grad_t list wins over the derived one,
    which is what a single-point or PES scan uses; with neither, the list
    comes from the states this step follows (the activestate channel).

    Parameters
    ----------
    config : SkillConfig
        Configuration container.

    Returns
    -------
    tuple of (list of int, list of int)
        (grad_s, grad_t) root numbers, each sorted and deduplicated.
    """
    if config.grad_s or config.grad_t:
        return sorted(set(config.grad_s)), sorted(set(config.grad_t))

    nsinglet = config.nstate_s
    singlet_roots = []
    triplet_roots = []
    for state in config.target_states:
        if state < nsinglet:
            singlet_roots.append(state + 1)
        else:
            triplet_roots.append(state - nsinglet + 1)
    return sorted(set(singlet_roots)), sorted(set(triplet_roots))


def build_properties_section(config, workflow) -> str:
    """Build the [properties] section.

    export=true with an empty title makes OpenQP dump results into
    fixed-name text files (energies, grad_<i>, nacme, dcme,
    nac_<i>_<j>, socme) that the output parser reads directly.

    grad index convention: hf ground state = 0; otherwise the target
    states k (0-based) map to k+1.  A two-manifold run replaces that one
    selector with grad_s / grad_t, one list per spin ladder.

    soc keyword: `soc=true` is what makes runtype=prop and runtype=data
    couple the two ladders they solved.  runtype=soc is the dedicated
    spin-orbit driver and needs no such switch -- and takes no gradient
    selector at all.

    nac keyword:
    - nac_type=dcm -> nac=nacme (dcme matrix via MO overlap, runtype=prop)
    - nac_type=nac -> nac=nac (numerical NAC vectors, answered by both
      runtype=prop and runtype=data)
    runtype=nac computes the NAC vector via the [nac] section and
    ignores this keyword.  A spin-orbit run carries no nac line: validate
    refuses the pairing before a deck is written.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    str
        [properties] section content.
    """
    method = config.method
    target_states = config.target_states

    selector_line = ''
    soc_line = ''
    if workflow.two_manifold:
        grad_s, grad_t = spin_ladder_selectors(config)
        # runtype=soc answers couplings only; a gradient request there is
        # refused by the engine, so the selectors stay off the deck.
        if workflow.runtype != 'soc':
            if grad_s:
                selector_line += 'grad_s=%s\n' % ','.join(str(r) for r in grad_s)
            if grad_t:
                selector_line += 'grad_t=%s\n' % ','.join(str(r) for r in grad_t)
            soc_line = 'soc=true'
    elif method == 'hf':
        selector_line = 'grad=0'
    else:
        selector_line = 'grad=%s' % ','.join(
            str(k + 1) for k in sorted(target_states))

    nac_line = ''
    if workflow.is_nac_profile and workflow.runtype in ('prop', 'data'):
        if config.nac_type == 'dcm':
            nac_line = 'nac=nacme'
        elif config.nac_type == 'nac':
            nac_line = 'nac=nac'

    # Assembled as lines so a selector set that names nothing leaves no
    # blank line inside the section.
    lines = [entry for entry in (selector_line.rstrip('\n'), soc_line, nac_line)
             if entry]

    return PROPERTIES_TEMPLATE.format(selector_lines='\n'.join(lines))


def build_nac_section(config, workflow) -> str:
    """Build the [nac] section (numerical NAC vector control).

    Read by the dedicated runtype=nac driver and, since OpenQP 1.3, also by
    a runtype=data run asked for NAC vectors ([properties] nac=nac).

    states carries 1-based pairs: the first entry of target_nac_pair
    (falling back to the spin-derived nac_pair_map) on runtype=nac, where
    OpenQP answers one pair only, and every declared pair on the runtype=data
    vector path, which writes one export file per pair.

    No worker count is written here. How many displaced structures run at once
    is set by how the job is launched -- `openqp_nproc` for the MPI
    ranks and `thread` / `openqp_thread` for OMP_NUM_THREADS per worker -- and
    neither of those reaches the deck.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    str
        [nac] section content, empty when not applicable.
    """
    if workflow.runtype != 'nac' and not workflow.nac_vectors:
        return ''

    pairs = [pair for pair in config.target_nac_pair if len(pair) == 2]
    if not pairs:
        pairs = [pair for pair in config.nac_pair_map if len(pair) == 2]
    if not pairs:
        return ''

    # runtype=nac answers one pair; runtype=data with a vector request names
    # one export file per pair, so every declared pair is listed (`1 2,2 3`).
    if workflow.runtype == 'nac':
        a, b = pairs[0]
        states = f'{a + 1} {b + 1}'
    else:
        states = ','.join(f'{a + 1} {b + 1}' for a, b in pairs)
    bp = 'true' if config.nac_bp else 'false'

    return NAC_TEMPLATE.format(
        dt=config.nac_dt,
        dx=config.nac_dx,
        states=states,
        bp=bp,
    )


def build_optimize_section(config, workflow) -> str:
    """Build the [optimize] section.

    istate follows the grad index convention: hf ground state = 0,
    otherwise the first target state k maps to k+1.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    str
        [optimize] section content.
    """
    target_states = config.target_states

    if config.method == 'hf':
        istate = 0
    else:
        istate = sorted(target_states)[0] + 1

    return OPTIMIZE_TEMPLATE.format(
        optimizer=config.opt_optimizer,
        maxit=config.opt_maxit,
        rmsd_grad=config.opt_rmsd_grad,
        max_grad=config.opt_max_grad,
        istate=istate,
    )


def build_hess_section(config, workflow) -> str:
    """Build the [hess] section.

    state follows the same index convention as optimize.istate.  No worker
    count is written: the displaced structures run on as many MPI ranks as the
    launcher starts (`openqp_nproc`), each with `thread` /
    `openqp_thread` OpenMP threads, and OpenQP's `[hess]nproc` applies only to
    a run the launcher left single-process.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    str
        [hess] section content.
    """
    target_states = config.target_states

    if config.method == 'hf':
        state = 0
    else:
        state = sorted(target_states)[0] + 1

    return HESS_TEMPLATE.format(
        state=state,
        dx=config.hess_dx,
    )


def write_xyz_file(config, calcdir: str):
    """Write the XYZ coordinate file for OpenQP.

    The file is named $OPENQP_PROJECT.xyz and is referenced by
    [input]system.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    calcdir : str
        Calculation directory.
    """
    project = config.project
    atoms = config.atoms
    coordinates = config.coordinates
    natom = len(atoms)

    xyz_path = os.path.join(calcdir, f'{project}.xyz')
    with open(xyz_path, 'w') as f:
        f.write(f'{natom}\n')
        f.write(f'{project} geometry\n')
        for atom, coord in zip(atoms, coordinates):
            f.write(f'{atom:>4s}  {coord[0]:16.10f}  {coord[1]:16.10f}  {coord[2]:16.10f}\n')
