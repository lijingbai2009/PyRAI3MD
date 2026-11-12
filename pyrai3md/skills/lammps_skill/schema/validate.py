"""
======================================================================
This is a standalone LAMMPS calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the validation functions for the LAMMPS skill,
which validate the input config for LAMMPS computation,
including the calculation method settings and computing environment settings.

"""

import math
import os
import sys
import warnings

from pydantic import BaseModel

from .config import SkillConfig, Calculation, Dynamics, Environment
from .method import AVAILABLE_METHOD
from .method import ENGINE_INTEGRATED_PROFILES
from .method import PROFILE_METHOD_MAP
from .reference import AVAILABLE_PROFILES, DEFAULT

# Supported LAMMPS units styles and their physical units
AVAILABLE_UNITS = {
    'real': {'energy': 'kcal/mol', 'force': 'kcal/mol/Angstrom'},
    'metal': {'energy': 'eV', 'force': 'eV/Angstrom'},
}

# Profiles the resident service implements. Framework-driven dynamics,
# optimization and variable-cell runs all query the mm layer as a
# per-step single point, so these two cover every hot loop; the
# internal-iteration profiles finish their whole propagation inside one
# calculate() call, where a resident instance would buy nothing.
# The one profile modifier spelled as a suffix (`ff_optimize_cell`).
_CELL_MODIFIER = '_cell'

# The lattice degrees of freedom the engine may own (`box_relax`).
BOX_RELAX_STYLES = ('off', 'iso', 'aniso')

PERSISTENT_PROFILES = ('ff_energy', 'ff_gradient',
                       'qmmm_field', 'qmmm_energy')

# Truthy spellings the resident switch accepts (yaml bools, classic
# input integers and the words the jobcards already use).
_PERSISTENT_ON = ('1', 'true', 'yes', 'on')
_PERSISTENT_OFF = ('0', 'false', 'no', 'off', '')

# Cell coupling choices of the ff_npt fix. The words are the framework's
# md.barostat_mode vocabulary, not a second dialect: the skill mirrors that
# field so one input deforms the cell the same way under either driver.
VALID_BAROSTAT_MODES = ('anisotropic', 'isotropic')

# Mapping from the profile task token (the last underscore-separated
# token of the profile name) to the capability key declared in
# AVAILABLE_METHOD and the manifest capabilities list.
_TASK_CAPABILITY_MAP = {
    'energy': 'energy',
    'gradient': 'gradient',
    'optimize': 'geometry_optimization',
    'nve': 'classical_dynamics',
    'nvt': 'classical_dynamics',
    'npt': 'classical_dynamics',
    # qmmm_field is a probe task. Its result is the qmmm_field block,
    # but it rides the single-point gradient capability gate.
    'field': 'gradient',
}


def validate(config: SkillConfig) -> SkillConfig:
    """Validate the input config for LAMMPS computation.

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
    work = {}
    work.update(config.calculation.model_dump())
    work.update(config.dynamics.model_dump())
    work.update(config.environment.model_dump())

    # The QM/MM region has its own decision table; it reads the system
    # submodel directly (natom, charges, atom_selection, qm_atoms).
    validate_qmmm(config)

    # Validate calculation method settings
    work = validate_method(work)

    # Validate internal-iteration parameters (optimize/MD profiles)
    work = validate_internal(work)

    # Validate shake constraint keywords
    work = validate_shake(work)

    # Validate computing environment settings
    work = validate_environment(work)

    # Validate the resident-execution switch after the mode/units/profile
    # checks it reads.
    work = validate_persistent(work)

    # Write results back to the appropriate submodels
    calc_updates = {k: work[k] for k in Calculation.model_fields if k in work}
    dyn_updates = {k: work[k] for k in Dynamics.model_fields if k in work}
    env_updates = {k: work[k] for k in Environment.model_fields if k in work}

    return config.model_copy(update={
        'calculation': config.calculation.model_copy(update=calc_updates),
        'dynamics': config.dynamics.model_copy(update=dyn_updates),
        'environment': config.environment.model_copy(update=env_updates),
    })


# The two QM/MM tasks are standalone profiles. They are deliberately not
# modifiers of ff_energy/ff_gradient: the old `qmmm` switch is retired.
QMMM_PROFILES = ('qmmm_field', 'qmmm_energy')


def validate_qmmm(config) -> None:
    """Validate the QM/MM task selected by profile.

    The topology itself is checked where it is read
    (service/builder/qmmm.py): the charge column, the partition charges
    and the cross-boundary bonded rows all need the Atoms and entity
    sections. What can be decided from the configuration alone is decided
    here, so a run that can never work fails before any file is written.

    Parameters
    ----------
    config : SkillConfig
        Normalized configuration (natom already derived).

    Raises
    ------
    ValueError
        On the retired switch, a missing/ill-formed partition, the wrong
        input fields for a task, an engine-side constraint, a multi-center
        row selection, or a charged system.
    """
    profile = str(config.calculation.profile).lower()
    qm_atoms = [int(index)
                for index in (getattr(config, 'qm_atoms', None) or [])]
    qm_probe = [int(index)
                for index in (getattr(config, 'qm_probe', None) or [])]
    qm_centers = [[int(index) for index in (group or [])]
                  for group in (getattr(config, 'qm_centers', None) or [])]
    q_qm = [float(value)
            for value in (getattr(config, 'q_qm', None) or [])]
    field_file = str(getattr(config, 'qmmm_field_file', '') or '')

    if profile not in QMMM_PROFILES:
        if qm_atoms or q_qm or field_file or qm_probe or qm_centers:
            raise ValueError(
                'QM/MM is selected by the profile `qmmm_field` or '
                '`qmmm_energy`; qm_atoms, qm_probe, qm_centers, q_qm and '
                'qmmm_field_file have '
                f'no effect on profile {profile!r}.')
        return

    if not qm_atoms:
        raise ValueError(
            f'{profile} needs a non-empty qm_atoms (0-based framework atom '
            'indices).')
    if len(set(qm_atoms)) != len(qm_atoms):
        raise ValueError(f'qm_atoms has duplicate indices: {qm_atoms}')
    natom = int(config.natom)
    outside = [index for index in qm_atoms if index < 0 or index >= natom]
    if outside:
        raise ValueError(
            f'qm_atoms index outside [0, {natom}): {outside}. The list is '
            f'0-based in the framework atom order, not 1-based LAMMPS ids.')
    if len(qm_atoms) >= natom:
        raise ValueError(
            'qm_atoms covers the whole system; that is the ordinary '
            'calculation, not a QM/MM partition (MM must be non-empty).')

    if profile == 'qmmm_field':
        # The probe sets group QMMM_QM charge 1.0 at run time, so the
        # energy task's q_qm/qmmm_field_file are simply unused here. They
        # are not refused: a composite shares one LAMMPS config across
        # the field and energy tasks, and the field result does not
        # depend on either value.
        if qm_centers:
            raise ValueError(
                'qm_centers belongs to qmmm_energy: a probe measures one '
                'center at a time through qm_probe and removes no Q-Q pair.')
        if qm_probe:
            if len(set(qm_probe)) != len(qm_probe):
                raise ValueError(
                    f'qm_probe has duplicate indices: {qm_probe}')
            outside = [index for index in qm_probe
                       if index not in set(qm_atoms)]
            if outside:
                raise ValueError(
                    f'qm_probe index outside qm_atoms {qm_atoms}: the probe '
                    f'measures one center of the QM set, not an atom outside '
                    f'it.')
            if sorted(qm_probe) == sorted(qm_atoms):
                raise ValueError(
                    'qm_probe equals qm_atoms: with one center measured '
                    'there is no unprobed environment, so leave qm_probe '
                    'empty and the probe keeps the whole QM set.')
    elif profile == 'qmmm_energy':
        if qm_probe:
            raise ValueError(
                'qm_probe belongs to qmmm_field only: qmmm_energy evaluates '
                'the whole force field over qm_atoms and measures no center.')
        if qm_centers:
            flat = [index for group in qm_centers for index in group]
            if any(not group for group in qm_centers):
                raise ValueError(
                    f'qm_centers holds an empty group: {qm_centers}; every '
                    f'center must name at least one QM atom.')
            if len(set(flat)) != len(flat):
                raise ValueError(
                    f'qm_centers share an atom: {flat}. A QM atom belongs to '
                    f'one center, or it carries two 0-order charges.')
            if sorted(set(flat)) != sorted(set(qm_atoms)):
                raise ValueError(
                    f'qm_centers {qm_centers} do not tile qm_atoms '
                    f'{qm_atoms}: the in-cell Q-Q removal is done inside each '
                    f'center, so an unclaimed QM atom would keep a pair its '
                    f'own calculation already contains.')
        if q_qm and len(q_qm) != len(qm_atoms):
            raise ValueError(
                f'q_qm has {len(q_qm)} values but qm_atoms has '
                f'{len(qm_atoms)}; the order is qm_atoms order.')
        if any(not math.isfinite(value) for value in q_qm):
            raise ValueError('q_qm contains a non-finite value.')
        if field_file and not os.path.isfile(field_file):
            raise ValueError(
                f'qmmm_field_file does not exist: {field_file}')

    if int(config.lammps_shake):
        raise ValueError(
            f'{profile} and lammps_shake=1 are mutually exclusive: a '
            'constraint inside the QM region belongs to the QM calculation, '
            'and the bonded rows it would act on are removed from the '
            'topology.')
    if getattr(config, 'atom_selection', None):
        raise ValueError(
            f'{profile} and atom_selection are mutually exclusive: qm_atoms '
            'holds 0-based indices in the framework order, atom_selection '
            'holds 1-based template rows, and a sliced subsystem is not the '
            'full system the QM/MM partition refers to.')
    # A charged QM region is admissible: the QM charge is declared per center
    # in the input and the MM charges come from the topology, and neither side
    # is forced neutral. The A/B probe cancels any uniform background between
    # its two legs (measured: 2.0e-5 residual on a cell charged by -0.4 e) and
    # a surviving offset would cancel between `E_QM` and `qmmm_el`, since both
    # are linear in V with the same QM net charge.
    charge_value = getattr(config, 'charges', 0) or 0
    if float(charge_value) != int(charge_value):
        raise ValueError(
            f'charges must be an integer electron count, got '
            f'{charge_value!r}')


def validate_method(data: dict) -> dict:
    """Verify that the profile and method are valid and compatible.

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

    if profile not in AVAILABLE_PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")

    if method not in AVAILABLE_METHOD:
        raise ValueError(f"Unsupported method: {method}")

    data['profile'] = profile
    data['method'] = method

    # Check profile-method compatibility
    expected_method = PROFILE_METHOD_MAP.get(profile)
    if expected_method and expected_method != method:
        raise ValueError(
            f"Profile '{profile}' requires method '{expected_method}', "
            f"but method '{method}' was specified."
        )

    # Check method capability for the profile's task
    task = _profile_to_task(profile)
    capability = _TASK_CAPABILITY_MAP.get(task, task)
    if not AVAILABLE_METHOD[method].get(capability, False):
        raise ValueError(
            f"Method '{method}' does not support task '{capability}' "
            f"(required by profile '{profile}')."
        )

    return data


def validate_internal(data: dict) -> dict:
    """Verify the internal-iteration parameters.

    Checks the minimization parameters for the ff_optimize profile
    and the MD parameters for the ff_nve/ff_nvt/ff_npt profiles.
    Single-point profiles skip the checks.

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Updated configuration with validated parameters.
    """
    profile = str(data['profile']).lower()
    task = _profile_to_task(profile)

    # Engine-side box relaxation. The style has three spellings, and the two
    # halves of the pairing must agree: a style named where no `fix` line is
    # written is a setting nobody reads, and a cell profile with no style is
    # `ff_optimize` wearing a name that promises a moved box.
    style = str(data.get('box_relax', 'off') or 'off').strip().lower()
    if style not in BOX_RELAX_STYLES:
        raise ValueError(
            "Unknown box_relax %r. Options: %s"
            % (style, ', '.join(BOX_RELAX_STYLES)))
    data['box_relax'] = style
    if style != 'off' and profile != 'ff_optimize_cell':
        raise ValueError(
            "box_relax %r needs profile ff_optimize_cell: profile %r lets the "
            "engine move its atoms only, so nothing would read the style"
            % (style, profile))
    if profile == 'ff_optimize_cell' and style == 'off':
        raise ValueError(
            "ff_optimize_cell with box_relax 'off' asks the engine to relax a "
            "box it has been told not to move; use ff_optimize, or name iso "
            "or aniso")

    if task == 'optimize':
        if int(data['min_maxiter']) <= 0 or int(data['min_maxeval']) <= 0:
            raise ValueError(
                "min_maxiter and min_maxeval must be positive for "
                "the ff_optimize profile."
            )
        if float(data['min_etol']) < 0 or float(data['min_ftol']) < 0:
            raise ValueError(
                "min_etol and min_ftol must be non-negative for "
                "the ff_optimize profile."
            )

    if task in ('nve', 'nvt', 'npt'):
        if int(data['md_nstep']) <= 0:
            raise ValueError("md_nstep must be positive for MD profiles.")
        if float(data['md_timestep']) <= 0:
            raise ValueError("md_timestep must be positive for MD profiles.")
        if int(data['dump_freq']) <= 0:
            raise ValueError("dump_freq must be positive for MD profiles.")
        if task in ('nvt', 'npt') and float(data['thermo_damp']) <= 0:
            raise ValueError("thermo_damp must be positive for MD profiles.")
        if task == 'npt':
            if float(data['pressure_damp']) <= 0:
                raise ValueError("pressure_damp must be positive for ff_npt.")
            # Same word list as the framework's md.barostat_mode, which this
            # field mirrors: an unknown coupling must not quietly fall back
            # to a different deformation than the one that was asked for.
            mode = str(data.get('barostat_mode', '')
                       or 'isotropic').strip().lower()
            if mode not in VALID_BAROSTAT_MODES:
                raise ValueError(
                    "Unknown barostat_mode '%s'. Options: %s"
                    % (mode, sorted(VALID_BAROSTAT_MODES)))

    return data


def validate_shake(data: dict) -> dict:
    """Verify the shake constraint keywords (&lammps section).

    A geometric constraint belongs to whichever party integrates the
    positions, so this group owns it only on the engine-integrated
    profiles; when the framework propagates, the framework's own
    ``constraint`` block does. The two are separate control
    sources and the switch must be well-formed before any calculation.

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Updated configuration with validated shake keywords.
    """
    lammps_shake = int(data.get('lammps_shake', 0))
    shake_fix_id = str(data.get('shake_fix_id', '')).strip()
    driver = str(data.get('driver', 'skill') or 'skill').strip().lower()
    if driver == 'main' and (lammps_shake or shake_fix_id
                             or str(data.get('shake_keywords', '')).strip()):
        # The framework propagates the positions on this driver, so there is no
        # motion here for a shake fix to hold: the request is inert rather than
        # wrong, and saying so is the difference between a debugged run and a
        # mystery. Cleared here so no downstream builder can act on it.
        warnings.warn(
            f'LAMMPS constraint keywords (lammps_shake={lammps_shake}, '
            f'shake_fix_id={shake_fix_id!r}, shake_keywords='
            f"{str(data.get('shake_keywords', '')).strip()!r}) are ignored: "
            f"control.driver='main' gives the framework integrator ownership "
            f"of the geometry, so the framework 'constraint' block is the only "
            f"constraint source of this run. Set driver=skill to let LAMMPS "
            f"integrate and constrain (D-27 ownership rule).")
        data['lammps_shake'] = 0
        data['shake_fix_id'] = ''
        data['shake_keywords'] = ''
        return data

    if lammps_shake not in (0, 1):
        raise ValueError(
            f"lammps_shake must be 0 (off) or 1 (on), got: {lammps_shake}"
        )

    if lammps_shake == 1 and not shake_fix_id and not str(data.get('shake_keywords', '')).strip():
        raise ValueError(
            "lammps_shake=1 requires either shake_fix_id (shake fix "
            "defined in the input template) or shake_keywords (raw "
            "fix command to inject)."
        )

    profile = str(data.get('profile', '')).lower()
    if lammps_shake == 1 and profile not in ENGINE_INTEGRATED_PROFILES:
        raise ValueError(
            f"lammps_shake=1 is inconsistent with profile '{profile}': a "
            f"geometric constraint has to be applied by whatever moves "
            f"the atoms, and this profile hands one force evaluation per "
            f"step to the framework integrator, which propagates the "
            f"geometry itself. There is no propagation in this input for a "
            f"shake fix to hold, so the request cannot be honoured here "
            f"rather than merely being ignored. Declare the constraint on "
            f"the side that integrates, under the framework "
            f"'constraint' block "
            f"(dev/methods/CONSTRAINT_DEVELOPMENT.md section 3.7), or use "
            f"an engine-integrated profile so LAMMPS does the propagating "
            f"({', '.join(ENGINE_INTEGRATED_PROFILES)})."
        )

    data['lammps_shake'] = lammps_shake
    data['shake_fix_id'] = shake_fix_id

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
    lammps_path = data.get('lammps_path', '')
    lammps_project = data.get('lammps_project', 'LAMMPS')
    lammps_units = str(data.get('lammps_units', 'real')).lower()
    thread = data['thread']
    scratch_path = data.get('scratch_path', '')

    # Check lammps executable availability (directory or binary path).
    # The resident path never executes lmp: it loads the shared library
    # through the LAMMPS Python module inside the service process, and the
    # service answers for that interpreter by importing it before it binds
    # a socket. Requiring the binary there would fail a perfectly good
    # resident run.
    if lammps_path and not is_persistent_enabled(data):
        if os.path.isdir(lammps_path):
            binary = os.path.join(lammps_path, 'bin', 'lmp')
            if not os.path.isfile(binary):
                raise ValueError(f"lammps executable not found at: {binary}")
        elif not os.path.isfile(lammps_path):
            raise ValueError(f"lammps path does not exist: {lammps_path}")

    # Check LAMMPS units style
    if lammps_units not in AVAILABLE_UNITS:
        raise ValueError(
            f"Unsupported lammps_units: '{lammps_units}'. "
            f"Available: {sorted(AVAILABLE_UNITS)}"
        )
    data['lammps_units'] = lammps_units

    # Check the reciprocal-sum accuracy. The token goes verbatim onto the
    # `kspace_style` line, so a value LAMMPS cannot parse would abort inside
    # the engine long after this configuration was accepted - the worst place
    # to discover it, because the run has already paid for the topology.
    # An empty value never arrives here: the config merge re-applies the
    # default for anything falsy, so "declare nothing" and "declare empty" are
    # one request. A case that wants the loose number its template carries
    # has to write that number.
    raw_accuracy = data.get('ewald_accuracy', DEFAULT['ewald_accuracy'])
    ewald_accuracy = str(raw_accuracy).strip().lower()
    try:
        value = float(ewald_accuracy)
    except (TypeError, ValueError):
        raise ValueError(
            f"ewald_accuracy must be a number in scientific notation such as "
            f"'1e-8', got: {raw_accuracy!r}"
        )
    if not 0.0 < value < 1.0:
        raise ValueError(
            f"ewald_accuracy must lie strictly between 0 and 1, got: "
            f"{ewald_accuracy!r}"
        )
    data['ewald_accuracy'] = ewald_accuracy

    # Check the user-provided LAMMPS topology file. This skill does
    # not generate force field templates or topology files (tooling
    # to be added later); the data file must exist in the launch
    # directory. The main input template '{project}.in' is optional:
    # when absent the input script is fully generated from the
    # workflow.
    data_file = f'{lammps_project}.data'
    if not os.path.exists(data_file):
        raise ValueError(
            f"LAMMPS topology file not found: {data_file}. "
            f"Provide '{data_file}' in the launch directory."
        )

    # The thread count is not checked here (#306). The machine this run
    # lands on is not the machine that parsed the input -- a submission
    # under Slurm gets cpus-per-task x ntasks, not os.cpu_count() of the
    # login node -- and oversubscription is allowed anyway, so the check
    # is the framework's, it reads the same two keywords, and it warns
    # instead of failing the job (workflow/dispatch/scheduler.py).

    # Check scratch path
    if scratch_path:
        scratch_path = os.path.expanduser(scratch_path)
        if not os.path.exists(scratch_path):
            try:
                os.makedirs(scratch_path)
            except Exception as e:
                raise ValueError(f"Failed to create scratch path {scratch_path}: {e}")

    return data


def _coerce_switch(value) -> bool:
    """Normalize the resident switch to a bool.

    Accepts the yaml bool, the classic-input integer and the words a
    jobcard may carry. Anything else is a typo, not a default.

    Parameters
    ----------
    value : bool or int or str or None
        Configured switch value.

    Returns
    -------
    bool
        Normalized switch.

    Raises
    ------
    ValueError
        When the value is neither an obvious bool nor a known spelling.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, int):
        return bool(value)
    text = str(value).strip().lower()
    if text in _PERSISTENT_ON:
        return True
    if text in _PERSISTENT_OFF:
        return False
    raise ValueError(
        f"python_enabled must be a bool (true/false, 1/0), got: {value!r}")


def is_persistent_enabled(source) -> bool:
    """Tell whether this config should run against a resident service.

    The single routing predicate, so no module can decide residency on
    its own. 'execution_mode=slurm' excludes it because the skill then
    submits and waits for one job per calculate() call (see
    service/runner/slurm_runner.py), which leaves nothing resident to
    reuse; the switch is simply inert there.

    Parameters
    ----------
    source : dict or SkillConfig
        Flat work dict or the validated config model.

    Returns
    -------
    bool
        True when the resident path applies.
    """
    if isinstance(source, dict):
        enabled = source.get('python_enabled', False)
        mode = source.get('execution_mode', '')
    elif isinstance(source, BaseModel):
        enabled = getattr(source, 'python_enabled', False)
        mode = getattr(source, 'execution_mode', '')
    else:
        return False
    try:
        wanted = _coerce_switch(enabled)
    except ValueError:
        return False
    return wanted and str(mode).lower() != 'slurm'


def validate_persistent(data: dict) -> dict:
    """Verify the resident-execution keywords.

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Configuration with the switch coerced to bool and the service
        interpreter filled in.

    Raises
    ------
    ValueError
        When the switch is malformed, a resident profile is not
        implemented, the rank count is unusable, or the configured
        interpreter does not exist.
    """
    data['python_enabled'] = _coerce_switch(data.get('python_enabled', False))
    # Normalized here and not only on the resident branch: a classic
    # input hands 'debug true' over as a string, and the flag has to mean
    # the same bool whichever way the step is finally executed.
    data['debug'] = _coerce_switch(data.get('debug', False))

    mode = str(data.get('execution_mode', '')).lower()
    if data['python_enabled'] and mode == 'slurm':
        # Not an error: residency cannot work when every step is its own
        # batch job, so the request is simply routed to the subprocess
        # path. The backend logs the decision once at initialization.
        data['python_path'] = str(data.get('python_path', '') or '')
        return data

    if data['python_enabled']:
        profile = str(data.get('profile', '')).lower()
        if profile not in PERSISTENT_PROFILES:
            raise ValueError(
                f"python_enabled=true is implemented for "
                f"{', '.join(PERSISTENT_PROFILES)} only; profile "
                f"'{profile}' runs the internal iteration in one call "
                f"and gains nothing from a resident instance. Set "
                f"python_enabled=false to keep the subprocess path.")

        # The folded `nproc` (max of nproc and lammps_nproc, resolved by
        # the schema) is what the service world size has to match.
        nproc = int(data.get('nproc', 1) or 1)
        if nproc < 1:
            raise ValueError(
                f"nproc/lammps_nproc must be >= 1 for a resident service, "
                f"got: {nproc}")
        data['nproc'] = nproc

        python_path = str(data.get('python_path', '') or '').strip()
        if not python_path:
            # Same-environment deployments (the LAMMPS wheel is installed
            # into the framework environment) need no extra path.
            python_path = sys.executable
        elif not os.path.isfile(python_path):
            raise ValueError(
                f"python_path is not an executable file: {python_path}. "
                f"Point it at the interpreter that can import lammps, or "
                f"leave it empty to use {sys.executable}.")
        data['python_path'] = python_path
    else:
        data['python_path'] = str(data.get('python_path', '') or '')

    return data


def _profile_to_task(profile: str) -> str:
    """Extract the task name from a profile string.

    The task is the last underscore-separated token.

    Parameters
    ----------
    profile : str
        Profile name like 'ff_energy' or 'ff_gradient'.

    Returns
    -------
    str
        Task name like 'energy' or 'gradient'.
    """
    parts = profile.rsplit('_', 1)
    if len(parts) == 2 and parts[1] == _CELL_MODIFIER.strip('_'):
        # `_cell` says which degrees of freedom the engine owns, not what the
        # step is; reading it as a task would send `ff_optimize_cell` to a task
        # called 'cell' that no builder writes.
        return _profile_to_task(parts[0])
    if len(parts) == 2:
        return parts[1]
    return profile
