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

This module provides individual Molcas module input builders.
Each function generates the input section for one Molcas module.

"""

from ..templates.molcas_templates import (
    GATEWAY_TEMPLATE,
    GATEWAY_SOC_TEMPLATE,
    GATEWAY_CONSTRAINT_TEMPLATE,
    XFIELD_TEMPLATE,
    SEWARD_TEMPLATE,
    SEWARD_SOC_TEMPLATE,
    RASSCF_TEMPLATE,
    RASSCF_FILEORB_TEMPLATE,
    RASSCF_ALTER_LINE,
    RASSCF_RLXROOT_LINE,
    RASSCF_FILEORB_LINE,
    RASSCF_MDRLXR_LINE,
    SCF_TEMPLATE,
    ALASKA_ROOT_TEMPLATE,
    ALASKA_NAC_TEMPLATE,
    RASSI_SOC_TEMPLATE,
    RASSI_ABSORPTION_TEMPLATE,
    RASSI_CASPT2_ABSORPTION_TEMPLATE,
    RASSI_SOC_CASPT2_HEFF_TEMPLATE,
    RASSI_SOC_CASPT2_EJOB_TEMPLATE,
    CASPT2_TEMPLATE,
    MCLR_TEMPLATE,
    SLAPAF_TEMPLATE,
    DYNAMIX_TEMPLATE,
    SURFACEHOP_TEMPLATE,
    MCKINLEY_TEMPLATE,
)
from ...schema.normalize import derive_alter_pairs


def build_gateway(config, soc: bool = False, constraint: str = None,
                  use_xfield: bool = True) -> str:
    """Build the GATEWAY module input section.

    Coordinates are referenced via $MOLCAS_PROJECT.xyz on the first
    line right after &GATEWAY. RICD and Group=c1 are included for
    all calculations. SOC calculations add AMMI and Angmom keywords.

    The basis line writes a plain basis name as-is (applies to all
    elements); a per-element combination (e.g. 'H.ano-rcc-vdzp,
    C.ano-rcc-vdzp') is passed through unchanged.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    soc : bool
        Whether this is a SOC calculation (adds relativistic keywords).
    constraint : str, optional
        Ediff constraint state suffix for MECI/MECP optimization:
        ' n m' (1-based roots, same-spin crossing) or '' (cross-spin
        crossing); None writes no Constraints block.

    Returns
    -------
    str
        GATEWAY input section.
    """
    project = config.project
    basis = config.basis
    atoms = config.atoms
    point_charges = config.point_charges

    # Dummy atoms ('X') cannot share a plain basis name: expand to a
    # per-element combination where every real element keeps the
    # plain basis and 'X' gets sto-3g.
    if '.' not in basis and 'X' in atoms:
        unique_atoms = list(dict.fromkeys(atoms))
        basis = ','.join(
            'X.sto-3g' if atom == 'X' else f'{atom}.{basis}'
            for atom in unique_atoms
        )

    title = config.project or project

    # Select SOC or standard template
    if soc:
        gateway_input = GATEWAY_SOC_TEMPLATE.format(
            title=title,
            basis=basis,
        )
    else:
        gateway_input = GATEWAY_TEMPLATE.format(
            title=title,
            basis=basis,
        )

    # Add XField block for background point charges (legacy path only;
# generic ESPF embedding passes use_xfield=False).
    if (use_xfield and point_charges and len(point_charges) > 0
            and len(point_charges[0]) > 0):
        charge_lines = []
        for pc in point_charges:
            if len(pc) >= 4:
                charge_lines.append(f' {pc[1]:16.10f} {pc[2]:16.10f} {pc[3]:16.10f} {pc[0]:10.6f}')
        if charge_lines:
            xfield_block = XFIELD_TEMPLATE.format(
                ncharges=len(charge_lines),
                charge_block='\n'.join(charge_lines),
            )
            gateway_input += xfield_block

    # MECI/MECP crossing constraint for SLAPAF optimization
    if constraint is not None:
        gateway_input += GATEWAY_CONSTRAINT_TEMPLATE.format(states=constraint)

    return gateway_input


def build_seward(config, soc: bool = False) -> str:
    """Build the SEWARD module input section.

    All calculations include doanalytic.
    SOC calculations add rela/R02O/relint/expert keywords.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    soc : bool
        Whether this is a SOC calculation.

    Returns
    -------
    str
        SEWARD input section.
    """
    if soc:
        return SEWARD_SOC_TEMPLATE
    return SEWARD_TEMPLATE


def build_rasscf(config, spin: str = None, rlxroot: int = None,
                 fileorb: bool = False, mdrlxr: int = None) -> str:
    """Build the RASSCF module input section.

    Outputs Spin, NactEl, Inactive, Ras2, ITERATIONS, CIRoot in the
    format recorded in dev/skills/MOLCAS_SKILL_DEVELOPMENT.md
    section 5.8.

    When the user selects active orbitals (config.active non-empty),
    ALTEr interchange pairs are derived (overlap with the default
    Ras2 window locked) and appended to the section so the selected
    orbitals are moved into the window.

    For SOC dual RASSCF, singlet reads singlet_ci_roots/singlet_ci_spin
    with Spin=1, triplet reads triplet_ci_roots/triplet_ci_spin with Spin=3.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    spin : str, optional
        Spin type for SOC: 'singlet' or 'triplet'. None for normal.
    rlxroot : int, optional
        CI root relaxed by SLAPAF (1-based within this RASSCF job);
        emitted as a Rlxroot keyword for geometry optimization.
    fileorb : bool
        Whether a staged user orbital guess ({project}.StrOrb) is
        available. Only then is fileorb=$MOLCAS_PROJECT.RasOrb
        emitted: the first iteration reads the guess copied to
        RasOrb, later loop iterations read the RasOrb each RASSCF
        run writes. Without a guess RASSCF falls back on its default
        orbitals and no fileorb line is written.
    mdrlxr : int, optional
        MD relaxation root (1-based) for dynamics: adds the MDRLXR
        keyword; the RasOrb continuation line follows the same
        fileorb enablement as above.

    Returns
    -------
    str
        RASSCF input section.
    """
    nact_elec = config.nact_elec
    ras1 = config.ras1_orb
    ras2 = config.ras2_orb or config.nact_orb
    ras3 = config.ras3_orb
    inactive = config.inactive_orb
    maxit_scf = config.maxit_scf or 200

    # Determine spin and CI root based on spin parameter
    if spin == 'singlet':
        spin_val = 1  # Molcas Spin=1 for singlet (2S+1=1)
        ci_roots = config.singlet_ci_roots or [3]
        ci_root = ci_roots[0] if ci_roots else 3
    elif spin == 'triplet':
        spin_val = 3  # Molcas Spin=3 for triplet (2S+1=3)
        ci_roots = config.triplet_ci_roots or [2]
        ci_root = ci_roots[0] if ci_roots else 2
    else:
        # Normal (non-SOC) calculation
        spin_val = config.multiplicity  # 2S+1
        ci_roots = config.ci_roots or [1]
        ci_root = ci_roots[0] if ci_roots else 1

    # Select template based on whether fileorb is needed. A
    # single-point job with a staged guess embeds the fileorb
    # (RasOrb) line in the template; the optimization loop appends
    # its own fileorb line after Rlxroot, and the dynamics loop
    # gates its continuation line on the same enablement.
    if (fileorb and spin is None and rlxroot is None
            and mdrlxr is None):
        rasscf_input = RASSCF_FILEORB_TEMPLATE.format(
            spin=spin_val,
            nact_el=nact_elec,
            ras1=ras1,
            ras3=ras3,
            inactive=inactive,
            ras2=ras2,
            maxit_scf=maxit_scf,
            ci_root=ci_root,
        )
    else:
        rasscf_input = RASSCF_TEMPLATE.format(
            spin=spin_val,
            nact_el=nact_elec,
            ras1=ras1,
            ras3=ras3,
            inactive=inactive,
            ras2=ras2,
            maxit_scf=maxit_scf,
            ci_root=ci_root,
        )

    # User-selected active orbitals: emit ALTEr interchange pairs so
    # the selected orbitals land in the Ras2 window. Pairs are
    # derived from the 0-based internal indices and written 1-based
    # (symmetry species fixed to 1); orbitals already inside the
    # window are locked and produce no ALTEr block.
    alter_pairs = derive_alter_pairs(config.active, inactive, ras1, ras2)
    if alter_pairs:
        alter_block = '\n'.join(
            ' %d %d %d' % pair for pair in alter_pairs)
        rasscf_input += RASSCF_ALTER_LINE.format(
            n_alter=len(alter_pairs), alter_block=alter_block)

    if rlxroot is not None:
        rasscf_input += RASSCF_RLXROOT_LINE.format(rlxroot=rlxroot)
        if fileorb:
            # The optimization loop starts from the guess copied to
            # RasOrb and continues from the RasOrb each iteration
            # writes.
            rasscf_input += RASSCF_FILEORB_LINE

    if mdrlxr is not None:
        if fileorb:
            rasscf_input += RASSCF_FILEORB_LINE
        rasscf_input += RASSCF_MDRLXR_LINE.format(mdrlxr=mdrlxr)

    if fileorb and rlxroot is None and mdrlxr is None and spin is not None:
        # A spin-labelled pass (the dual singlet/triplet decks) has no
        # Rlxroot or MDRLXR line to hang the continuation on, and the
        # fileorb template above only serves the unlabelled case, so a
        # staged guess used to be dropped silently here: the pass ran on
        # Molcas' default orbitals even though the deck asked for one.
        rasscf_input += RASSCF_FILEORB_LINE

    return rasscf_input


def build_scf(config) -> str:
    """Build the SCF module input section for Hartree-Fock calculation.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    str
        SCF input section.
    """
    return SCF_TEMPLATE.format(
        charge=config.charges,
        spin=config.multiplicity,
    )


def build_alaska(config, nac_mode: bool = False,
                 states: list = None, root_offset: int = 0,
                 nac_pairs: list = None, nocsf: bool = True) -> str:
    """Build the ALASKA module input section.

    Two modes:
    - NAC: one ALASKA block per requested NAC pair (NAC=i j)
    - Gradient: one ALASKA block per requested target state
      (ROOT=i).  In SOC workflows the singlet and triplet RASSCF
      jobs define separate root numberings, so callers pass the
      per-spin group state list and the root offset of that group
      (0 for singlet, number of singlet states for triplet).

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    nac_mode : bool
        Whether to generate NAC keywords instead of ROOT.
    states : list of int, optional
        Global (0-based) state indices to request gradients for.
        Defaults to config target_states.
    root_offset : int
        Number of states preceding the requested spin group; the
        ROOT/NAC keywords restart from 1 within each group.
    nac_pairs : list of list of int, optional
        Explicit NAC pairs (0-based global state indices); defaults
        to config target_nac_pair.
    nocsf : bool
        Add NOCSF to each NAC block.

    Returns
    -------
    str
        ALASKA input section (possibly several ALASKA blocks).
    """
    if nac_mode:
        return _build_alaska_nac(config, root_offset=root_offset,
                                 nac_pairs=nac_pairs, nocsf=nocsf)

    if states is None:
        states = config.target_states
    nstate = config.nstate

    # Filter out empty/invalid entries
    states = [s for s in states if isinstance(s, int) and 0 <= s < nstate]
    if not states:
        states = [0]

    # One ALASKA block per state: Molcas accepts a single ROOT per
    # module, so multiple gradients require repeated ALASKA modules.
    blocks = []
    for s in states:
        blocks.append(ALASKA_ROOT_TEMPLATE.format(
            root=s - root_offset + 1))
    return ''.join(blocks)


def _build_alaska_nac(config, root_offset: int = 0,
                      nac_pairs: list = None, nocsf: bool = True) -> str:
    """Build ALASKA for NAC computation between specific states.

    Uses target_nac_pair to generate one NAC=i j block per pair.
    State indices are renumbered from 1 within the requesting spin
    group: keyword = state index - root_offset + 1.  This keeps the
    triplet-group NAC keywords correct in dual-RASSCF SOC workflows.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    root_offset : int
        Number of states preceding the requesting spin group
        (0 for a single-RASSCF workflow or the singlet group, the
        number of singlet states for the triplet group).
    nac_pairs : list of list of int, optional
        Explicit NAC pairs (0-based global state indices); defaults
        to config target_nac_pair.

    Returns
    -------
    str
        ALASKA NAC input section.
    """
    pairs = nac_pairs if nac_pairs is not None else [
        p for p in config.target_nac_pair if len(p) == 2]

    if not pairs:
        # Nothing was declared, so nothing is asked for. This used to emit a
        # default `NAC=1 2` block, which made the engine compute - and the log
        # report - a coupling the input never named (#448).
        return ''

    # One ALASKA block per NAC pair
    blocks = []
    for pair in pairs:
        block = ALASKA_NAC_TEMPLATE.format(
            state_i=pair[0] - root_offset + 1,  # renumber within group
            state_j=pair[1] - root_offset + 1,
        )
        if nocsf:
            block += 'NOCSF\n'
        blocks.append(block)
    return ''.join(blocks)


def build_rassi(config, soc: bool = False, dual_rasscf: bool = False,
                absorption: bool = False, caspt2: bool = False) -> str:
    """Build the RASSI module input section.

    Three modes:
    - SOC: Nrof JobIph format with SpinOrbit/EJob/SOCOupling=0
    - Absorption: DIPRint=0 for transition properties; when the
      workflow also contains CASPT2, all nstate states are read back
      into RASSI first (Nrof JobIph with one job) after the
      multistate CASPT2 step.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    soc : bool
        Whether this is a SOC calculation.
    dual_rasscf : bool
        Whether dual RASSCF (singlet+triplet) was performed.
    absorption : bool
        Whether this is an absorption spectrum calculation.
    caspt2 : bool
        Whether the workflow contains a CASPT2 step (CASPT2-level
        absorption spectrum).

    Returns
    -------
    str
        RASSI input section.
    """
    if soc:
        return _build_rassi_soc(config, dual_rasscf)
    elif absorption and caspt2:
        nstate = config.nstate
        state_list = ' '.join(str(i + 1) for i in range(nstate))
        return RASSI_CASPT2_ABSORPTION_TEMPLATE.format(
            nstates=nstate,
            state_list=state_list,
        )
    elif absorption:
        return RASSI_ABSORPTION_TEMPLATE
    else:
        raise ValueError(
            'RASSI requires an explicit mode: SOC or absorption.'
        )


def _build_rassi_soc(config, dual_rasscf: bool) -> str:
    """Build RASSI for SOC computation with Nrof JobIph format.

    Handles singlet/triplet index mapping for dual RASSCF.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    dual_rasscf : bool
        Whether dual RASSCF was performed.

    Returns
    -------
    str
        RASSI SOC input section.
    """
    state_spin = config.state_spin

    # Count singlet and triplet states
    n_singlet = sum(1 for s in state_spin if s == 0)
    n_triplet = sum(1 for s in state_spin if s != 0)

    # Build Nrof JobIph line
    # Format: Nrof JobIph=njobs n_singlet n_triplet;singlet_states;triplet_states
    if dual_rasscf:
        njobs = 2
        singlet_states = ' '.join(str(i + 1) for i in range(n_singlet))
        triplet_states = ' '.join(str(i + 1) for i in range(n_triplet))
        states_line = f'{n_singlet} {n_triplet};{singlet_states};{triplet_states}'
    else:
        njobs = 1
        total = n_singlet + n_triplet
        states_line = f'{total};' + ' '.join(str(i + 1) for i in range(total))

    return RASSI_SOC_TEMPLATE.format(
        njobs=njobs,
        states_line=states_line,
    )


def build_caspt2(config, gradient: bool = False) -> str:
    """Build the CASPT2 input block for the selected variant.

    The state-specific density/gradient/NAC quantities are selected by
    the ALASKA blocks that follow (ROOT= / NAC=), not by CASPT2
    keywords. Only GRDT (and DORT for an IPEA shift) is emitted here.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration.
    gradient : bool
        Request the quantities needed by analytic gradients/NACs.

    Returns
    -------
    str
        CASPT2 input section.
    """
    ctype = str(getattr(config, 'caspt2_type', '') or '').lower()
    if not ctype:
        ctype = 'xms' if getattr(config, 'xms', True) else 'ss'

    lines = ['&CASPT2']
    if ctype == 'ss':
        pass
    elif ctype == 'ms':
        lines.extend(['MULTISTATE', 'all'])
    elif ctype == 'xms':
        lines.extend(['XMULTISTATE', 'all'])
    elif ctype == 'rms':
        lines.extend(['RMULTISTATE', 'all'])
    elif ctype == 'xdw':
        lines.extend(['XMULTISTATE', 'all'])
        lines.extend(['DWMS', str(getattr(config, 'dwms_zeta', 1.0))])
        lines.extend(['DWTYPE', str(getattr(config, 'dwtype', 3))])
    else:
        raise ValueError(
            "Unsupported caspt2_type %r; use ss, ms, xms, rms, or xdw."
            % (ctype,))

    lines.append('PROP')
    if gradient:
        lines.append('GRDT')
        if getattr(config, 'caspt2_ipea_shift', None) != 0.0:
            lines.append('DORT')

    ipea = getattr(config, 'caspt2_ipea_shift', None)
    if ipea is not None:
        lines.extend(['IPEA', str(float(ipea))])
    imag = getattr(config, 'caspt2_imaginary_shift', 0.2)
    if imag:
        lines.extend(['IMAG', str(float(imag))])
    real = getattr(config, 'caspt2_real_shift', 0.0)
    if real:
        lines.extend(['SHIFT', str(float(real))])
    return '\n'.join(lines) + '\n'


def build_mclr() -> str:
    """Build the MCLR module (CASPT2 NAC Lagrange multipliers).

    Reserved for the MCLR route; no production caller yet.
    """
    return MCLR_TEMPLATE


def build_rassi_caspt2_soc(config, multistate: bool,
                           jobiph_names=None) -> str:
    """Build RASSI-SO for CASPT2 JobIph files.

    Multistate CASPT2 uses HEff (the effective Hamiltonian stored on
    JobIph); single-state CASPT2 uses EJob (diagonal energies with the
    overlap-based off-diagonal approximation).
    """
    state_spin = list(getattr(config, 'state_spin', []) or [])
    n_singlet = sum(1 for s in state_spin if s == 0)
    n_triplet = len(state_spin) - n_singlet
    names = list(jobiph_names or [])
    if not names:
        names = ['JOBIPH_S']
        if n_triplet:
            names.append('JOBIPH_T')
    if len(names) == 1:
        states_line = 'all'
        jobiph_list = ''
    else:
        singlet_states = ' '.join(str(i + 1) for i in range(n_singlet))
        triplet_states = ' '.join(str(i + 1) for i in range(n_triplet))
        states_line = '%d %d' % (n_singlet, n_triplet)
        jobiph_list = ' %s\n %s\n' % (singlet_states, triplet_states)
    template = (RASSI_SOC_CASPT2_HEFF_TEMPLATE if multistate
                else RASSI_SOC_CASPT2_EJOB_TEMPLATE)
    return template.format(
        njobs=len(names),
        states_line=states_line,
        jobiph_list=jobiph_list,
        name_list='\n '.join(names),
    )


def build_slapaf(config) -> str:
    """Build the SLAPAF module input section for geometry optimization.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    str
        SLAPAF input section.
    """
    maxit = config.opt_maxit or 100
    return SLAPAF_TEMPLATE.format(maxit=maxit)


def build_dynamix(config) -> str:
    """Build the DYNAMIX module input section for molecular dynamics.

    The timestep comes from dynamics.size (au) and the thermostat
    from dynamics.thermostat: 'off'/'nve' map to ther=0 (NVE) and
    'nose'/'nose-hoover'/'nvt' map to ther=2 (Nose-Hoover NVT); any
    other thermostat (NPT, Langevin, ...) is unsupported.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    str
        DYNAMIX input section.
    """
    thermostat = (config.thermostat or 'off').strip().lower()
    ther_map = {
        'off': 0, 'nve': 0,
        'nose': 2, 'nose-hoover': 2, 'nvt': 2,
    }
    if thermostat not in ther_map:
        raise ValueError(
            f"Unsupported thermostat '{config.thermostat}' for Molcas "
            f"DYNAMIX: only NVE ('off'/'nve') and Nose-Hoover NVT "
            f"('nose'/'nose-hoover'/'nvt') are supported"
        )
    return DYNAMIX_TEMPLATE.format(
        size=config.size,
        ther=ther_map[thermostat],
        temperature=config.temperature,
    )


def build_surfacehop(config) -> str:
    """Build the SURFACEHOP module input section for surface hopping dynamics.

    The decoherence value comes from dynamics.deco.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    str
        SURFACEHOP input section.
    """
    return SURFACEHOP_TEMPLATE.format(deco=config.deco)


def build_mckinley(config) -> str:
    """Build the MCKINLEY module input section for vibrational frequency analysis.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    str
        MCKINLEY input section.
    """
    return MCKINLEY_TEMPLATE
