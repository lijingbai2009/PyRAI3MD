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

This module provides the validation functions for the CP2K skill,
which validate the input config for CP2K computation,
including the calculation method settings and computing environment settings.

"""

import os
import warnings

import numpy as np

from .config import SkillConfig, Calculation, Environment
from .method import AVAILABLE_METHOD
from .method import PROFILE_METHOD_MAP
from .reference import AVAILABLE_FUNCTIONAL
from .reference import AVAILABLE_PROFILES
from .reference import DISPERSION_KEYWORD
from .reference import STRESS_METHODS
from .reference import STRESS_SCOPES

# Upper bound of the dimensionless strain step of the 'num' route;
# see validate_stress for why the value itself stays the caller's call.
STRAIN_DELTA_MAX = 1.0e-2


def validate(config: SkillConfig) -> SkillConfig:
    """Validate the input config for CP2K computation.

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
    # Build a flat work dict from the submodels that validators touch
    # (system fields are read-only here: state counts feed the
    # TDDFPT combination checks).
    work = {}
    work.update(config.system.model_dump())
    work.update(config.calculation.model_dump())
    work.update(config.environment.model_dump())

    # Validate calculation method settings
    work = validate_method(work)

    # Validate periodic boundary condition compatibility
    work = validate_system(work)

    # Validate the whole-system stress request (after the periodicity
    # check, which owns the lattice sanity the numerical route reads)
    work = validate_stress(work)

    # Validate the slab dipole correction against CP2K's cell-normal
    # requirement (after validate_stress, which resolves the route).
    work = validate_dipole_correction(work)

    # Validate computing environment settings
    work = validate_environment(work)

    # Write results back to the corresponding submodels
    calc_updates = {k: work[k] for k in Calculation.model_fields if k in work}
    env_updates = {k: work[k] for k in Environment.model_fields if k in work}

    return config.model_copy(update={
        'calculation': config.calculation.model_copy(update=calc_updates),
        'environment': config.environment.model_copy(update=env_updates),
    })


def validate_method(data: dict) -> dict:
    """Verify that the profile, method, functional, and keywords are valid.

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
    functional = data['functional'].lower()
    cutoff = data['cutoff']
    ref_cutoff = data['ref_cutoff']
    dispersion = str(data['dispersion']).strip().lower()
    tddft_nstates = data['tddft_nstates']
    current_state = data.get('current_state', 0)

    if profile not in AVAILABLE_PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")

    if method not in AVAILABLE_METHOD:
        raise ValueError(f"Unsupported method: {method}")

    if functional not in AVAILABLE_FUNCTIONAL:
        raise ValueError(f"Unsupported CP2K functional: {functional}")

    # Plane-wave cutoffs must be positive (Ry); both are written
    # explicitly into &MGRID.
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if ref_cutoff <= 0:
        raise ValueError(f"ref_cutoff must be positive, got {ref_cutoff}")

    # Grimme dispersion correction maps onto &VDW_POTENTIAL
    # POTENTIAL_TYPE; REFERENCE_FUNCTIONAL follows the functional.
    if dispersion in ('', 'none'):
        dispersion = ''
    elif dispersion not in DISPERSION_KEYWORD:
        raise ValueError(
            f"Unsupported dispersion correction: "
            f"{dispersion!r}. Supported: "
            f"{sorted(DISPERSION_KEYWORD)} or 'none'.")

    data['profile'] = profile
    data['method'] = method
    data['functional'] = functional
    data['dispersion'] = dispersion

    # A non-singlet reference is refused before the input is written, because the
    # input this skill emits cannot carry one. `MULTIPLICITY` goes into &DFT while
    # &SCF declares neither UKS nor any unrestricted keyword, and CP2K solves a
    # multiplicity other than 1 only from an unrestricted reference - so the file
    # that would be written is one the engine rejects, with the refusal arriving
    # from CP2K several seconds later instead of from the request. The open-shell
    # route needs its own &SCF block (UKS, and the occupation control that replaces
    # OT for a spin-polarized reference) before this can be opened.
    multiplicity = int(data.get('multiplicity', 1) or 1)
    if multiplicity != 1:
        raise ValueError(
            "cp2k: multiplicity %d asks for an open-shell reference, but the &SCF "
            "block this skill writes is restricted (no UKS keyword), so CP2K would "
            "refuse the input it is handed. Run the open-shell state from an "
            "unrestricted &SCF section once that block exists, or keep "
            "multiplicity 1 and take the excited states from TDDFPT"
            % multiplicity)

    # A triplet row is only meaningful if something solves a triplet. TDDFPT does
    # when `ntrip_root` is positive (the builder then writes RKS_TRIPLETS T);
    # a plain DFT run has no excited-state block at all, so the row would be a
    # label on nothing and the crossing search reading it would be stepping on a
    # surface the engine never produced.
    if method != 'tddft':
        spins = [int(x) for x in (data.get('state_spin') or [])]
        if 2 in spins:
            raise ValueError(
                "cp2k: state_spin %r names a triplet state while method is %r, "
                "which solves no excited states: only &TDDFPT with "
                "RKS_TRIPLETS T produces a triplet surface, so row %d would "
                "carry no energy or gradient of that spin"
                % (spins, method, spins.index(2)))

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

    # TDDFPT combination checks: the requested root count must cover
    # the active state, and excited-state runs need at least one
    # excited root.
    if method == 'tddft':
        if tddft_nstates <= 0:
            raise ValueError(
                f"TDDFPT profile '{profile}' requires at least one "
                f"excited state (tddft_nstates={tddft_nstates})."
            )
        if current_state > tddft_nstates:
            raise ValueError(
                f"Active state {current_state} exceeds the number of "
                f"computed excited states (tddft_nstates="
                f"{tddft_nstates}); raise tddft_nstates or nstate."
            )

    return data


def validate_system(data: dict) -> dict:
    """Verify periodic boundary condition compatibility for CP2K.

    pbc is the single normalized periodicity authority
    (dev/methods/PBC_DEVELOPMENT.md section 3.2): a non-zero lattice
    alone never turns a run periodic. Only full 3D periodicity is
    supported in this stage; partial periodicity (mixed 0/1) is
    rejected because the low-dimensional CP2K electrostatics solver
    mapping is not defined yet (stage E). A periodic run requires a
    valid, non-singular lattice_vector. The CP2K system fields are
    read-only in validate(), so this only raises on an invalid
    combination and never rewrites fields.

    Parameters
    ----------
    data : dict
        Flat configuration dictionary (system fields included).

    Returns
    -------
    dict
        The unchanged configuration dictionary.
    """
    method = str(data.get('method', '')).lower()
    pbc = [int(x) for x in (data.get('pbc') or [])]
    n_periodic = sum(1 for p in pbc if p)
    if n_periodic == 0:
        return data

    if not AVAILABLE_METHOD.get(method, {}).get('supports_pbc', False):
        raise ValueError(
            f"CP2K: method '{method}' does not support periodic "
            f"boundary conditions.")

    if n_periodic != 3:
        raise ValueError(
            "CP2K: partial periodicity (pbc=%r) is not supported in "
            "this version; use full 3D periodicity [1, 1, 1] or "
            "disable pbc (dev/methods/PBC_DEVELOPMENT.md stage E)."
            % (pbc,))

    lattice = np.asarray(data.get('lattice_vector') or [], dtype=np.float64)
    if (lattice.shape != (3, 3) or not np.all(np.isfinite(lattice))
            or abs(float(np.linalg.det(lattice))) <= 1e-8):
        raise ValueError(
            "CP2K: periodic run (pbc=%r) requires a valid, "
            "non-singular lattice_vector; got %r."
            % (pbc, (data.get('lattice_vector') or [])))

    return data


def validate_dipole_correction(data: dict) -> dict:
    """Verify the slab dipole correction against CP2K's normal requirement.

    CP2K's SURF_DIP correction (src/surface_dipole.F) aborts unless the
    slab normal is parallel to one Cartesian axis; the builder writes
    SURF_DIP_DIR Z, so c must lie on z and a/b must carry no z component.
    A non-periodic run has no slab to correct, and the 'full' strain
    route shears the cell (strain.py sets the off-diagonal slots), which
    would trip the same abort. 'iso' keeps the cell diagonal and is
    allowed. Nothing is rewritten here.

    Parameters
    ----------
    data : dict
        Flat configuration dictionary (system fields included).

    Returns
    -------
    dict
        The unchanged configuration dictionary.
    """
    if not data.get('dipole_correction'):
        return data

    pbc = [int(x) for x in (data.get('pbc') or [])]
    if not any(pbc):
        raise ValueError(
            "CP2K: dipole_correction requires a periodic slab cell; set "
            "pbc and lattice_vector (the molecular vacuum box is not a "
            "slab).")

    lattice = np.asarray(data.get('lattice_vector') or [], dtype=np.float64)
    if lattice.shape != (3, 3):
        raise ValueError(
            "CP2K: dipole_correction requires a 3x3 lattice_vector; got "
            "%r." % (data.get('lattice_vector') or []))

    a, b, c = lattice
    tol = 1.0e-6
    if (abs(a[2]) > tol or abs(b[2]) > tol
            or abs(c[0]) > tol or abs(c[1]) > tol):
        raise ValueError(
            "CP2K: dipole_correction writes SURF_DIP_DIR Z, so the slab "
            "normal must be the c vector along z (a_z=b_z=c_x=c_y=0); "
            "got lattice_vector=%r. CP2K aborts on a tilted cell: "
            "'Dipole correction only for surface perpendicular to one "
            "Cartesian axis'." % (lattice.tolist(),))

    if (data.get('stress')
            and str(data.get('stress_method') or '').lower() == 'num'
            and str(data.get('stress_scope') or 'full').lower() == 'full'):
        raise ValueError(
            "CP2K: dipole_correction cannot be combined with "
            "stress_method 'num' and stress_scope 'full': the shear "
            "strain points tilt the cell, and CP2K aborts the dipole "
            "correction on a tilted cell. Use stress_scope 'iso' or drop "
            "one request.")

    return data


def validate_stress(data: dict) -> dict:
    """Verify the whole-system stress request against its evaluation route.

    Three checks, each with a measured reason behind it:

    - The route name must be one of STRESS_METHODS. A misspelling would
      otherwise fall through `_stress_keyword` and emit no CP2K keyword
      at all, leaving a zero-filled tensor that looks like a result.
    - 'analytical' asks CP2K for a stress tensor, and the engine has none
      for a TDDFT run: it prints the TDDFT forces and then aborts in
      `response_solver.F:943` with "Stress Tensor not available for TDDFT
      calculations" (measured in the test environment over the same input in
      rounds tmp_20261010_133424 and tmp_20261010_190544). Refuse the
      combination where the request is made rather than spending an
      engine run on it. CP2K offers no numerical stress for a TDDFT run
      either (also measured), which is why this skill exposes one
      numerical route -- the external strain difference it drives itself
      -- instead of a second engine-side one.
    - 'num' (that external strain finite difference) sends no stress
      keyword to CP2K and differentiates the total energy instead, so it
      is the route that reaches an excited-state tensor. It perturbs the
      cell, which only means something for a full 3D periodic system
      with a non-singular lattice, and it needs a usable strain step.

    Parameters
    ----------
    data : dict
        Flat configuration dictionary (system fields included).

    Returns
    -------
    dict
        Updated configuration with the normalized stress fields.

    Raises
    ------
    ValueError
        On an unknown route, an unusable strain step or scope, a
        non-periodic or singular cell under 'num', or a stress keyword
        requested from a TDDFT run.
    """
    stress_method = str(data.get('stress_method') or '').lower()
    if stress_method and stress_method not in STRESS_METHODS:
        raise ValueError(
            f"cp2k: unknown stress_method {stress_method!r}; "
            f"expected one of {list(STRESS_METHODS)}.")

    method = str(data.get('method', '')).lower()
    if not stress_method:
        # The route is this skill's call, not the caller's, and it follows from
        # the method the run was given - which is a property of the calculation,
        # so it is resolved whether or not anything asks for a tensor. A
        # framework run that needs one says so through the `stress` capability
        # alone (planning/resolver.py::_derive_stress injects that flag and
        # knows nothing about engines), and the only spelling it could land on
        # here used to be 'analytical' - the combination CP2K aborts on for a
        # TDDFT run ('Stress Tensor not available for TDDFT calculations',
        # response_solver.F:943, measured in rounds tmp_20261010_133424 and
        # tmp_20261010_190544). An excited-state tensor therefore goes to the
        # strain route, which returns (nstate, 3, 3) - the shape a crossing
        # search consumes. A name the caller wrote is never overwritten.
        stress_method = 'num' if method == 'tddft' else 'analytical'
        if stress_method == 'num' and data.get('stress'):
            warnings.warn(
                "cp2k: stress_method was not written, so this skill resolved "
                "it to 'num' for a method 'tddft' run: CP2K prints no "
                "analytical stress tensor for TDDFT. Write stress_method "
                "explicitly to choose otherwise.",
                stacklevel=2,
            )
    data['stress_method'] = stress_method

    if not data.get('stress'):
        return data

    if stress_method == 'analytical':
        if method == 'tddft':
            raise ValueError(
                "cp2k: stress_method "
                f"{stress_method!r} asks CP2K for a stress tensor, which "
                "the engine refuses for a TDDFT run (it aborts with "
                "'Stress Tensor not available for TDDFT calculations'); "
                "request stress under method 'dft', or use "
                "stress_method 'num' for the finite-difference route.")
        return data

    scope = str(data.get('stress_scope') or 'full').lower()
    if scope not in STRESS_SCOPES:
        raise ValueError(
            f"cp2k: unknown stress_scope {scope!r} for the numerical "
            f"strain route; expected one of {list(STRESS_SCOPES)}.")
    delta = data.get('stress_delta')
    try:
        delta = float(delta)
    except (TypeError, ValueError):
        raise ValueError(
            f"cp2k: stress_delta must be a number, got {delta!r}.")
    # The recommended starting range is 1e-4 ~ 1e-3
    # (dev/skills/CP2K_SKILL_DEVELOPMENT.md section 6.6, step size); 1e-2
    # is refused as out
    # of range, not as inaccurate -- the step is the caller's call.
    if not 0.0 < delta <= STRAIN_DELTA_MAX:
        raise ValueError(
            f"cp2k: stress_delta must be in (0, {STRAIN_DELTA_MAX:g}] for "
            f"the numerical strain route, got {delta!r}.")

    pbc = [int(x) for x in (data.get('pbc') or [])]
    if sum(1 for p in pbc if p) != 3:
        raise ValueError(
            "cp2k: stress_method 'num' deforms the periodic cell, so it "
            f"needs full 3D periodicity; pbc={pbc!r} is not periodic in "
            "all three directions.")
    lattice = np.asarray(data.get('lattice_vector') or [], dtype=np.float64)
    if (lattice.shape != (3, 3) or not np.all(np.isfinite(lattice))
            or abs(float(np.linalg.det(lattice))) <= 1e-8):
        raise ValueError(
            "cp2k: stress_method 'num' requires a valid, non-singular "
            f"lattice_vector; got {(data.get('lattice_vector') or [])!r}.")

    data['stress_scope'] = scope
    data['stress_delta'] = delta
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
    cp2k_path = data.get('cp2k_path', '')
    thread = data['thread']
    scratch_path = data.get('scratch_path', '')

    # Check CP2K installation directory availability (empty defers
    # the check to env_check at run-script generation time)
    if cp2k_path and not os.path.isdir(cp2k_path):
        raise ValueError(f"CP2K path does not exist: {cp2k_path}")

    # The thread count is not checked here (#306): the machine this run
    # lands on is not the one that parsed the input, and oversubscription
    # is allowed, so the framework does the arithmetic and warns instead
    # of failing the job (workflow/dispatch/scheduler.py).

    # Check scratch path
    if scratch_path:
        scratch_path = os.path.expanduser(scratch_path)
        if not os.path.exists(scratch_path):
            try:
                os.makedirs(scratch_path)
            except Exception as e:
                raise ValueError(f"Failed to create scratch path {scratch_path}: {e}")

    return data


# Profile suffix to AVAILABLE_METHOD task key mapping.
# Most profiles use simple suffix extraction (e.g. dft_energy -> energy).
# Profiles where the suffix does not match the task key need explicit mapping.
_PROFILE_TASK_MAP = {
    'dft_opt_min': 'geometry_optimization',
    'tddft_opt_min': 'geometry_optimization',
    'dft_md': 'abinitio_dynamics',
    'tddft_md': 'abinitio_dynamics',
}


def _profile_to_task(profile: str) -> str:
    """Extract the task name from a profile string.

    Parameters
    ----------
    profile : str
        Profile name like 'dft_energy'.

    Returns
    -------
    str
        Task name like 'energy'.
    """
    if profile in _PROFILE_TASK_MAP:
        return _PROFILE_TASK_MAP[profile]
    parts = profile.split('_', 1)
    if len(parts) == 2:
        return parts[1]
    return profile
