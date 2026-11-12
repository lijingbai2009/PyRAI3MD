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

This module provides the validation functions for the OpenQP skill,
which validate the input config for OpenQP computation,
including the calculation method settings and computing environment settings.

"""

import os


from .config import SkillConfig, Calculation, Environment
from .method import AVAILABLE_METHOD
from .method import PROFILE_METHOD_MAP
from .reference import AVAILABLE_BASIS
from .reference import AVAILABLE_FUNCTIONALS
from .reference import AVAILABLE_PROFILES


def validate(config: SkillConfig) -> SkillConfig:
    """Validate the input config for OpenQP computation.

    Bridges between the nested SkillConfig and the flat-dict
    validation functions: extracts a work dict from submodels,
    runs validation, then writes results back.

    Parameters
    ----------
    config : SkillConfig
        Normalized config model.

    Returns
    -------
    SkillConfig
        Validated config model.
    """
    # Molecular (non-periodic) backend: reject periodic runs explicitly
    # instead of silently ignoring the cell. The framework capability
    # gate (schema.validate) also refuses these; this is the
    # standalone/agent-entry backstop.
    _pbc = [int(x) for x in (getattr(config.system, 'pbc', None) or [])]
    if any(_pbc):
        raise ValueError(
            "OpenQP is a molecular (non-periodic) backend and does not "
            "support periodic boundary conditions (pbc=%r); disable pbc "
            "or use a periodic backend (cp2k / xtb / lammps)." % (_pbc,))

    # A state list that names a triplet is refused before an input file is
    # written, unless the run is the two-manifold kind that can ask for one.
    validate_requested_spins(config.system, is_two_manifold(config.calculation))

    # Spin-orbit runs carry the engine's own preconditions, checked here so an
    # unsupported combination fails at parse time instead of in a remote job.
    validate_soc(config.system, config.calculation)

    # Build a flat work dict from the submodels that validators touch
    work = {}
    work.update(config.calculation.model_dump())
    work.update(config.environment.model_dump())

    # Validate calculation method settings
    work = validate_method(work)

    # Validate computing environment settings
    work = validate_environment(work)

    # Write results back to the appropriate submodels
    calc_updates = {k: work[k] for k in Calculation.model_fields if k in work}
    env_updates = {k: work[k] for k in Environment.model_fields if k in work}

    return config.model_copy(update={
        'calculation': config.calculation.model_copy(update=calc_updates),
        'environment': config.environment.model_copy(update=env_updates),
    })


def soc_is_requested(calculation) -> bool:
    """True when this run computes spin-orbit coupling.

    Two spellings reach it: the profile a caller names outright, and the
    framework's &md dosoc flag, which the planner also turns into the 'soc'
    jobtype token. Both mean the same run.
    """
    profile = str(getattr(calculation, 'profile', '') or '').lower()
    return profile in ('mrsf_soc', 'mrsf_soc_gradient') or bool(
        getattr(calculation, 'dosoc', 0))


def is_two_manifold(calculation) -> bool:
    """True when the run solves and publishes both spin ladders.

    OpenQP does it for a spin-orbit run, and for any run that names a
    gradient on the ladder the deck did not configure; the second case is
    why an explicit grad_t list also earns a triplet row in state_spin.
    """
    return soc_is_requested(calculation) or bool(
        getattr(calculation, 'grad_t', None))


def validate_soc(system, calculation) -> None:
    """Check a spin-orbit run against the rules OpenQP 1.3 enforces.

    The engine reaches both spin manifolds from one triplet ROHF reference and
    refuses a request that cannot be answered that way; each refusal below
    names the constraint the checker reports, so an input that would be
    rejected remotely is rejected at parse time with the same reason.

    Parameters
    ----------
    system : System
        System submodel (state_spin, nstate, ladder sizes).
    calculation : Calculation
        Calculation submodel (profile, method, reference, selectors).

    Raises
    ------
    ValueError
        If a spin-orbit precondition is unmet or an unsupported coupling
        was requested alongside it.
    """
    if not soc_is_requested(calculation):
        return

    method = str(getattr(calculation, 'method', '') or '').lower()
    if method != 'mrsf':
        raise ValueError(
            "openqp: spin-orbit coupling is computed on the MRSF response, "
            "which answers the singlet and triplet ladders from one triplet "
            "ROHF reference; method=%r has no second ladder to couple. Use "
            "method 'mrsf'." % (method,))

    # The triplet ROHF reference is not the system's multiplicity: a molecule
    # like water stays multiplicity 1 (its ground state is what state_spin
    # labels), and the mrsf method forces [scf]multiplicity=3 when the deck is
    # written (input_builder.build_scf_section).  So the reference is checked
    # through the derived scf_type, not through the system.
    scf_type = str(getattr(calculation, 'scf_type', '') or '').lower()
    if scf_type != 'rohf':
        raise ValueError(
            "openqp: spin-orbit coupling needs a triplet ROHF reference, got "
            "scf_type=%r. OpenQP refuses the run with 'SOC requires an ROHF "
            "reference'; the mrsf method derives it, so check the method "
            "rather than setting scf_type by hand." % (scf_type,))

    spins = [int(value) for value in (getattr(system, 'state_spin', None) or [])]
    if 0 not in spins or 2 not in spins:
        raise ValueError(
            "openqp: spin-orbit coupling is the singlet-triplet block of the "
            "response, so state_spin %r names no pair to couple. Declare at "
            "least one singlet (0) and one triplet (2) row." % (spins,))

    # A NAC of either flavour alongside SOC is refused by the engine: the
    # analytic vector route answers the singlet fold only, and the numerical
    # route would sweep 3N displacements once per ladder into scratch that has
    # no ladder slot.  'dcm' (nacme) is refused by validate_method already,
    # for a reason of its own (it reads the previous step's orbitals).
    nac_type = str(getattr(calculation, 'nac_type', '') or '').lower()
    if nac_type == 'nac':
        raise ValueError(
            "openqp: spin-orbit coupling cannot be combined with NAC vectors. "
            "OpenQP computes the analytic couplings for the singlet fold only "
            "and its numerical route would run one 3N displacement sweep per "
            "spin ladder into scratch that has no ladder slot, so a run asking "
            "for both is refused before it starts. Drop the nac request to keep "
            "SOC, or drop SOC and take the NAC vectors on runtype=data.")

    # Ladder sizes.  OpenQP sizes each ladder from nstate_s / nstate_t and
    # falls back on [tdhf] nstate when unset, and it only measures a ladder
    # that carries a gradient request -- a ladder with no request is still
    # what sets the socme row count (nstate_s * nstate_t * 3), so a size that
    # contradicts the declared states has to be caught here.
    declared_s = spins.count(0)
    declared_t = spins.count(2)
    nstate_s = int(getattr(system, 'nstate_s', 0) or 0)
    nstate_t = int(getattr(system, 'nstate_t', 0) or 0)
    for name, size, declared in (('nstate_s', nstate_s, declared_s),
                                 ('nstate_t', nstate_t, declared_t)):
        if size and size < declared:
            raise ValueError(
                "openqp: %s = %d converges fewer roots than state_spin "
                "declares for that ladder (%d). Every declared state needs a "
                "row; raise %s or drop the state." % (name, size, declared, name))
    for name, roots, size in (('grad_s', getattr(calculation, 'grad_s', None),
                               nstate_s or declared_s),
                              ('grad_t', getattr(calculation, 'grad_t', None),
                               nstate_t or declared_t)):
        for root in roots or []:
            if int(root) > size:
                raise ValueError(
                    "openqp: %s names response root %d but that ladder "
                    "converges %d root(s). Roots count from 1 (root 0 is the "
                    "internal MRSF reference), so raise the ladder's nstate or "
                    "drop the root." % (name, int(root), size))


def validate_requested_spins(system, two_manifold=False) -> None:
    """Refuse a state list whose labels the written input cannot ask for.

    `state_spin` says per row what that row is: 0 a singlet, 2 a triplet. A 2 in a
    singlet system is a request for one surface of each spin, which is what a
    singlet-triplet crossing search reads. A single-block input carries one spin line
    for the whole calculation - `[tdhf]multiplicity=1` for the mrsf method and no
    multiplicity line at all otherwise, which is OpenQP's singlet block - so every
    row that comes back is a singlet root however the list was written, and the row
    count still matches. A search handed two singlets labelled S and T does not land
    on the wrong crossing; it lands on a crossing between surfaces it never sampled,
    and nothing downstream can see the difference.

    OpenQP 1.3 answers both blocks from one deck when the run is a two-manifold
    one: the driver solves the other ladder by switching [tdhf]multiplicity and
    sizes each from nstate_s / nstate_t, publishing `energies_s` and `energies_t`
    separately. That is what a spin-orbit run asks for, so a triplet row is
    legitimate there and the refusal would block the feature it was written to
    protect. It still applies to every single-block run.
    """
    spins = [int(value) for value in (getattr(system, 'state_spin', None) or [])]
    multiplicity = int(getattr(system, 'multiplicity', 1) or 1)
    if multiplicity == 1 and 2 in spins and not two_manifold:
        raise ValueError(
            "openqp: state_spin %r names a triplet row while multiplicity is 1, and "
            "the input asks the engine for one spin block only "
            "([tdhf]multiplicity=1 for mrsf, no multiplicity line otherwise), so "
            "every state returned is a singlet root and the triplet row would be a "
            "mislabelled singlet rather than a second surface. Ask for a "
            "two-manifold run (profile mrsf_soc, or &md dosoc, or an explicit "
            "grad_t list), or drop the triplet row" % (spins,))


def validate_method(data: dict) -> dict:
    """Verify that the profile, method, basis, and functional are valid.

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Updated configuration with validated fields.
    """
    profile = data['profile'].lower()
    method = data['method'].lower()
    basis = data['basis'].lower().replace(' ', '')
    functional = data['functional'].lower().replace(' ', '')

    if profile not in AVAILABLE_PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")

    if method not in AVAILABLE_METHOD:
        raise ValueError(f"Unsupported method: {method}")

    if basis not in AVAILABLE_BASIS:
        raise ValueError(f"Unsupported basis: {basis}")

    if functional not in AVAILABLE_FUNCTIONALS:
        raise ValueError(f"Unsupported functional: {functional}")

    data['profile'] = profile
    data['method'] = method
    data['basis'] = basis
    data['functional'] = functional

    # Check profile-method compatibility
    expected_method = PROFILE_METHOD_MAP.get(profile)
    if expected_method and expected_method != method:
        raise ValueError(
            f"Profile '{profile}' requires method '{expected_method}', "
            f"but method '{method}' was specified."
        )

    # Check method capability for the profile's task
    task = _profile_to_task(profile)
    if not AVAILABLE_METHOD[method].get(task, False):
        raise ValueError(
            f"Method '{method}' does not support task '{task}' "
            f"(required by profile '{profile}')."
        )

    # Functional requirements per method:
    # hf takes no functional; tddft/mrsf require an exchange-correlation functional.
    if method == 'hf' and functional:
        raise ValueError("hf method does not use a functional.")

    if method in ('tddft', 'mrsf') and not functional:
        raise ValueError(f"Method '{method}' requires a functional.")

    # NAC vector support: OpenQP only computes NAC vectors for the
    # mrsf method (check_nac_input exits on hf and non-mrsf tdhf).
    nac_type = data.get('nac_type', '')
    # `dcm` asks OpenQP to read the PREVIOUS step's orbitals (through
    # [guess]file2, which the builder always emits for it) in order to build
    # the MO overlap. A single calculation has no previous step of its own, so
    # the engine aborts -- `guess.file2: Second restart file is set but does
    # not exist` then `mol object OQP.json does not exist`, rc=1 (reproduced
    # in isolation in the test environment for the retired
    # `tddft_nonadiabatic_dynamics` case, round tmp_20261011_121148; it had
    # only ever 'passed' by finding an orbital json another case left in the
    # shared calcdir). Refused by name here until the two-step mechanism has a
    # real test to be opened against.
    if nac_type == 'dcm':
        raise ValueError(
            "openqp: nac_type 'dcm' is not supported for a single "
            "calculation -- it continues from the previous step's orbital "
            "json ([guess]file2) and nothing produces one in a one-shot run; "
            "take the dynamics path that writes a previous step before "
            "requesting it, or use nac_type 'nac' for a numerical NAC vector "
            "on a single state pair.")
    if method != 'mrsf' and nac_type == 'nac' and profile.endswith(('_nac', '_nonadiabatic_dynamics')):
        raise ValueError(
            f"OpenQP supports NAC vector calculations only for the "
            f"mrsf method, got method='{method}'."
        )

    return data


def validate_environment(data: dict) -> dict:
    """Verify the computing environment settings.

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Updated configuration with validated environment.
    """
    openqp_path = data.get('openqp_path', '')
    thread = data['thread']
    scratch_path = data.get('scratch_path', '')
    use_mpi = data.get('use_mpi', False)
    nproc = data.get('openqp_nproc', 1)

    # Check OpenQP installation path
    if openqp_path and not os.path.isdir(openqp_path):
        raise ValueError(f"OpenQP path does not exist: {openqp_path}")

    # The thread count is not checked here (#306): the machine this run
    # lands on is not the one that parsed the input, and oversubscription
    # is allowed, so the framework does the arithmetic and warns instead
    # of failing the job (workflow/dispatch/scheduler.py).

    # Check MPI settings
    if use_mpi and nproc < 1:
        raise ValueError(f"use_mpi requires openqp_nproc >= 1, got {nproc}.")

    # Check scratch path
    if scratch_path:
        scratch_path = os.path.expanduser(scratch_path)
        if not os.path.exists(scratch_path):
            try:
                os.makedirs(scratch_path)
            except Exception as e:
                raise ValueError(f"Failed to create scratch path {scratch_path}: {e}")

    return data


def _profile_to_task(profile: str) -> str:
    """Extract the task name from a profile string.

    All OpenQP profiles use the '<method>_<task>' naming, so the task
    is always the suffix after the first underscore
    (e.g. tddft_geometry_optimization -> geometry_optimization).

    Parameters
    ----------
    profile : str
        Profile name like 'tddft_energy'.

    Returns
    -------
    str
        Task name like 'energy'.
    """
    parts = profile.split('_', 1)
    if len(parts) == 2:
        task = parts[1]
        # A trailing `_gradient` is a modifier on a task, not a task of its
        # own: `mrsf_soc_gradient` asks for the spin-orbit couplings and the
        # gradients with them, and the capability the method has to answer is
        # `soc`.  `gradient` on its own stays the task it always was.
        if task.endswith('_gradient') and task != 'gradient':
            task = task[:-len('_gradient')]
        return task
    return profile
