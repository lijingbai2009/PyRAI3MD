"""
======================================================================
This module provides the unified configuration schema for PyRAI3MD.
It defines Pydantic v2 models for task control, system information,
dynamics settings, computed properties, calculation parameters, and
computational environment.

The Control, System, and Dynamics models contain skill-independent
parameters (from classic key_control, key_molecule, key_md).
The Properties, Calculation, and Environment models contain
skill-related parameters managed by individual skills.

All models are merged into a single Config model for flexible
input handling.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import math

import numpy as np
from typing import Any, Dict, List, Optional, Union
from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)

from ..utilities import tracker as _tracker

# Default values shared across configuration models
DEFAULT = {
    # Control defaults
    'title': 'pyrai3md',
    'jobtype': 'energy',
    'theory': [],
    'reference': [],
    'driver': 'main',
    # Two-theory-entry composite selector: 'qmmm' or 'oniom'.
    # Three theory entries are always ONIOM3 regardless of this value.
    'multiscale_type': '',
    'gl_seed': 1,
    'verbose': 2,
    'silent': 1,
    # Test-side switch: interrupt the trajectory when a monitored bound is
    # violated. Off means report and keep running. It is global on purpose
    # and outranks a skill's own `debug` of the same name (see
    # schema.adapter._merge_one_skill_config), which is what lets one
    # keyword drive every module under test.
    'debug': False,
    'remote_train': 0,
    'load': 1,
    'transfer': 0,
    'pop_step': 200,
    'maxiter': 1,
    'maxsample': 1,
    'record_step': 10,
    'keep_md': 1,
    'check_population': 1,
    'check_energy_order': 1,
    'gap_tolerance': 1e-6,
    # Analysis layer (dev/methods/ADAPTIVE_DEVELOPMENT.md section 16). All default off so
    # the adaptive loop stays byte-identical unless a switch is set.
    'analyze_pes': 0,
    'analyze_gap': 'off',
    'analyze_trace': 0,
    'analyze_coverage': 0,
    'analyze_window': 3,
    'analyze_local_tolerance': 0.0,
    'analyze_global_tolerance': 0.0,
    # Predictor selection (dev/methods/ADAPTIVE_DEVELOPMENT.md section 16): 0 returns the
    # member mean; k>0 is the 1-based ensemble member used for dynamics.
    'predict_mode': 0,
    # Threshold below which the gap analysis stops driving and the gap
    # exploration stops pushing (Hartree).
    'gap_threshold': 0.03,
    # Adaptive-private exploration (dev/methods/ADAPTIVE_DEVELOPMENT.md section 16).
    'explore_udd': 0,
    'explore_gap': 0,
    'udd_mode': 'force',
    'udd_amplitude': 0.0,
    'udd_width': 0.0,
    'udd_start_iteration': 1,
    'gap_method': 'cv',
    'gap_pair': [],
    'gap_height': 0.0,
    'gap_width': 0.0,
    'sample_energy': 1,
    'sample_grad': 1,
    'sample_nac': 0,
    'sample_soc': 0,
    'sample_dipole': 0,
    'sample_transition_dipole': 0,
    'sample_oscillator_strength': 0,
    'md_multitask': None,
    'md_remotetask': None,
    'md_multipool': None,
    'reference_multitask': None,
    'reference_remotetask': None,
    'reference_multipool': None,
    'dynsample': 0,
    'maxdiscard': 0,
    # Validation energy MAE ceiling of an adaptive fit, in eV (the unit the
    # `&nn validation mean absolute error` table prints). 0 disables the
    # training-data quality gate.
    'maxmae': 0.0,
    'refine': 0,
    'refine_num': 4,
    'refine_start': 0,
    'refine_end': 200,
    'refine_gap': 0.3,
    'maxenergy': 0.05,
    'minenergy': 0.02,
    'dynenergy': 0.1,
    'inienergy': 0.3,
    'fwdenergy': 1,
    'bckenergy': 1,
    'maxgrad': 0.15,
    'mingrad': 0.06,
    'dyngrad': 0.1,
    'inigrad': 0.3,
    'fwdgrad': 1,
    'bckgrad': 1,
    'maxnac': 0.15,
    'minnac': 0.06,
    'dynnac': 0.1,
    'ininac': 0.3,
    'fwdnac': 1,
    'bcknac': 1,
    'maxsoc': 50.0,
    'minsoc': 20.0,
    'dynsoc': 0.1,
    'inisoc': 0.3,
    'fwdsoc': 1,
    'bcksoc': 1,
    # System defaults
    'charges': 0,
    'multiplicity': 1,
    'ci': [1],
    'spin': [0],
    'coupling': [],
    'solvent': 'gas',
    'pbc': [0, 0, 0],
    'cell': [99.0, 99.0, 99.0, 90.0, 90.0, 90.0],
    'lattice_vector': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    'kpoint': [1, 1, 1],
    # Dynamics defaults
    'temperature': 300.0,
    'pressure': 1.0,
    'thermostat': 'off',
    'barostat': 'berendsen',
    'ensemble': 'nve',
    # Ensemble algorithm parameters (NVT thermostats / NPT barostats)
    'thermostat_chain': 1,
    'thermostat_tau': 100.0,
    'barostat_tau': 1000.0,
    'compressibility': 4.5e-5,
    'barostat_mode': 'isotropic',
    'thermostat_mass': None,
    'barostat_mass': None,
    'initcond': '0',
    'ninitcond': 1,
    'multipool': 1,
    'method': 'wigner',
    'format': 'molden',
    'step': 10,
    'size': 20.67,
    'root': 0,
    'sfhp': 'nosh',
    'nactype': 'ktdc',
    'deco': '0.1',
    'thermodelay': 200,
    # dissipate: 0 keeps the thermostat in charge from the first step (the
    # behavior every existing example was measured with). 1 propagates the
    # excited state on NVE and starts the thermostat only after the run has
    # been on the ground state for `thermodelay` steps.
    'dissipate': 0,
    # Output throttling: the first `direct` steps are recorded every
    # step, afterwards every `buffer`-th step (and the last one).
    'direct': 2000,
    'buffer': 500,
    'substep': 20,
    'maxh': 10,
    'gap': 0.5,
    'gapsoc': 0.5,
    'history': 4,
    # Detection thresholds of an NN-driven md run. None means the channel
    # is not judged; an adaptive md child has the four err_* thresholds
    # overwritten from the `&adaptive` max* values. They are md keywords so a
    # plain NN md run can ask for the resident guard without carrying an
    # &adaptive block.
    'md_maxenergy': None,
    'md_maxgrad': None,
    'md_maxnac': None,
    'md_maxsoc': None,
    # Population tolerance of an md run: a population outside
    # [-md_maxpop, 1 + md_maxpop] ends the trajectory. 0.01 is the legacy
    # bound and the default; None turns the guard off. It is the only
    # tolerance: adaptive reads the stop code and does not re-judge, and
    # `&adaptive pop_tolerance` is retired.
    'md_maxpop': 1e-2,
    # FSSH kernel implementation selection (numpy reference / numba
    # acceleration) and function-level profiling switch
    'fssh_rk4': 'numpy',
    'fssh_dpdt': 'numpy',
    'fssh_matb': 'numpy',
    'fssh_ktdc': 'numpy',
    'fssh_decoherence': 'numpy',
    'fssh_profile': 0,
    'fssh_force_hop': 0,
    # Write the end-of-trajectory framework phase table
    # (pyrai3md_step.perf). Off by default; a performance test turns it on.
    'perf_profile': 0,
    # Calculation defaults (skill-related)
    'project': 'pyrai3md',
    'task': 'energy',
    'profile': '',
    'functional': '',
    'basis': '',
    'chk_file': '',
    'save_mo': False,
    'save_norm_mode': False,
    'maxit_scf': 100,
    'maxit_opt': 100,
    'maxit_md': 1,
    'density_fitting': True,
    'non_local_correction': True,
    'dispersion_correction': '',
    'scf_converge': 1e-9,
    'td_converge': 1e-6,
    'opt_converge': False,
    'opt_energy': 1e-6,
    'opt_grms': 3e-4,
    'opt_gmax': 4.5e-4,
    'opt_drms': 1.2e-3,
    'opt_dmax': 1.8e-3,
    # A variable-cell run keeps its own two readings beside the atomic four, in
    # the units a box cares about: `opt_cs*` is the stress conjugate to a lattice
    # entry, in bar, and `opt_ce*` is how far a lattice entry moved divided by the
    # length of its own row (dimensionless strain). The numbers are what the
    # atomic thresholds above already mean on the eighth round's 25 A water box:
    # opt_gmax 4.5e-4 Hartree/Bohr on its x row is 59.8 bar and opt_grms 3e-4 is
    # 39.9 bar, while opt_dmax 1.8e-3 A and opt_drms 1.2e-3 A over a 25.3866 A row
    # are 7.1e-5 and 4.7e-5 of strain. Rounded to two digits, so a box of that
    # size is judged as it was; a box twice as long no longer asks for the same
    # force per lattice entry, which is the defect these four keys exist to fix.
    'opt_csrms': 40.0,
    'opt_csmax': 60.0,
    'opt_cerms': 5.0e-5,
    'opt_cemax': 7.0e-5,
    # The external pressure a variable-cell relaxation minimizes the enthalpy
    # `E + p V` against, in bar, tensile-positive like the stress it is added to:
    # the box is at rest where the stress equals `-p`, which is the same reading
    # and the same sign the NPT barostats use (`integrator/ensemble.py`). 0.0 asks
    # for a zero-stress box, which is what a relaxation without this keyword does.
    'opt_pressure': 0.0,
    # Which lattice components one pressure value is allowed to move: 'anisotropic'
    # lets each of the six relaxed components answer for itself, 'isotropic' asks
    # for a shape-preserving scale of the whole box. The two spellings are the ones
    # `dynamics.barostat_mode` already uses, so a run can be read against its NPT
    # twin without learning a second vocabulary.
    'opt_cell_coupling': 'anisotropic',
    # The largest step one search may take, in Bohr, the length that bounds both
    # the nuclear and the lattice half of a joint step. None means the search's
    # own value, which differs by algorithm and is not a mistake: `minima` and
    # `gdiis` start at 0.3 and the transition-state search at 0.05
    # (`workflow/optimizer/ts.py:170`), so a single framework default written
    # into every run would have moved the TS legs eight-fold. It is named here
    # because the measurements say the step size, not the iteration budget, is
    # what decides whether a pressurised box lands inside its bar band: round
    # `tmp_20261019_122659` records `optimize_cell_pressure` 181.5 bar from its
    # rest point at one step and 3.95 A^3 of box travelled on the next one, and
    # twenty-five steps of the same coupling on the LAMMPS `water25` box bottom
    # out at 1058.4 bar while still descending 0.096 Hartree of enthalpy a step.
    'opt_trust_radius': None,
    # --- The path family (`opt_method` mep / irc / neb, `workflow/path.py`) ---
    # These walks evaluate geometries that are not the state's own - a probe held
    # on a hypersphere, or a whole band of images - so `opt_method` is also what
    # resolves which of the two loops a run gets (`workflow.resolve_workflow`).
    # `opt_path_radius` is the arc length of one accepted path point on the
    # normalized mass-weighted coordinates, in Bohr: the same length the lattice
    # half of a joint step is measured in (`optimizer/base.py::_apply_trust_radius`),
    # so no amu factor enters the input.
    'opt_path_constraint': 'sphere',
    'opt_path_radius': 0.1,
    'opt_path_inner_maxit': 20,
    'opt_path_inner_tol': 3.0e-4,
    'opt_irc_direction': 'both',
    'opt_irc_mode': 'hessian',
    'opt_irc_mode_file': '',
    'opt_neb_reactant': '',
    'opt_neb_product': '',
    'opt_neb_ts': '',
    'opt_neb_images': 8,
    'opt_neb_interpolate': 'linear',
    'opt_neb_geodesic_threshold': 3.0,
    'opt_neb_geodesic_scaler': 1.7,
    'opt_neb_geodesic_min_neighbors': 4,
    'opt_neb_spring': 0.01,
    'opt_neb_climb': False,
    'opt_neb_climb_threshold': 1.0e-2,
    'opt_neb_step': 0.1,
    'opt_sp': True,
    'opt_freq': False,
    'ir_width': 100.0,
    'ir_scale': 1.0,
    'uv_width': 100.0,
    'nics_ring': [0],
    'nics_scan': [0.0],
    'target_states': [0],
    'target_nac_pair': [[]],
    'target_soc_pair': [[]],
    'grid_level': 3,
    # The external-potential defaults (`shape`, `factor`, `scale`) were the
    # archive's cavity keys. They left with the fields that read them: the same
    # law is a wall now, so its defaults are the `WallInput` fields.
    # Environment defaults
    'program': '',
    'version': '',
    'program_path': '',
    'scratch_path': '',
    'device': 'cpu',
    'num_gpu': 0,
    'slurm_time': '01:00:00',
    'slurm_partition': 'default',
    'slurm_ntasks': 1,
    'slurm_command': '',
    'slurm_memory': 4000,
    # Dispatch environment defaults
    'shell': '',
}


class Control(BaseModel):
    """Task control parameters.

    Corresponds to classic key_control section. Includes job control
    and resource allocation settings.

    `debug` is the one field that deliberately reaches past the framework:
    gating, monitoring and early interruption belong to the test side of
    the package, so a single global switch turns them on in every module
    under test, skills included. It outranks a skill's own `debug`
    (lammps_skill calls it that too, for its per-step artifact files);
    see workflow.records.diagnostics for what it decides.
    """

    # Job control
    title: str = DEFAULT['title']
    jobtype: str = DEFAULT['jobtype']
    theory: List[str] = DEFAULT['theory']          # Primary methods (formerly qm)
    reference: List[str] = DEFAULT['reference']    # Reference methods (formerly abinit)
    # Execution driver: 'main' lets the framework resolve the
    # workflow from jobtype (iterative or single); 'skill' or a
    # skill name hands the whole jobtype to the skill, which plans
    # and runs it internally (framework executes a single shot).
    driver: str = DEFAULT['driver']
    # Composite classification for exactly two theory entries:
    # 'qmmm' or 'oniom'. Empty keeps the legacy ONIOM2 meaning.
    # Three theory entries are always ONIOM3.
    multiscale_type: str = DEFAULT['multiscale_type']

    # Skill identification (used by agent mode)
    skill_name: str = ''
    skill_path: str = ''

    gl_seed: int = DEFAULT['gl_seed']
    verbose: int = DEFAULT['verbose']
    silent: int = DEFAULT['silent']
    debug: bool = DEFAULT['debug']


# Keys the adaptive section no longer reads. `pop_tolerance` moved to the md
# run: the guard that stops a trajectory is `&md md_maxpop`, and adaptive only
# decides whether the stopping structure is collected.
RETIRE_ADAPTIVE_KEYS = {
    'pop_tolerance': ('the population tolerance belongs to the md run; set '
                      '`&md md_maxpop` (default 0.01, `none` turns the guard '
                      'off). Adaptive only collects the structure md stopped '
                      'on, gated by check_population'),
}


class Adaptive(BaseModel):
    """Adaptive sampling parameters.

    Corresponds to classic &CONTROL adaptive sampling keywords.
    Moved to a dedicated &ADAPTIVE section for clarity.

    The tri-state execution keywords (multitask/remotetask) are
    modularized: they override the adaptive workflow class defaults
    during execution graph construction (None keeps the defaults).
    """

    @model_validator(mode='before')
    @classmethod
    def _refuse_retired_adaptive_keys(cls, data):
        """Turn a retired &adaptive keyword into an error, not a drop."""
        if not isinstance(data, dict):
            return data
        retired = sorted(str(key).lower() for key in data
                         if str(key).lower() in RETIRE_ADAPTIVE_KEYS)
        if retired:
            raise ValueError(
                'adaptive.%s %s no longer an input keyword; %s'
                % (', '.join('`%s`' % key for key in retired),
                   'is' if len(retired) == 1 else 'are',
                   '; '.join(RETIRE_ADAPTIVE_KEYS[key] for key in retired)))
        return data

    # Control
    remote_train: int = DEFAULT['remote_train']
    load: int = DEFAULT['load']
    transfer: int = DEFAULT['transfer']
    pop_step: int = DEFAULT['pop_step']
    maxiter: int = DEFAULT['maxiter']
    maxsample: int = DEFAULT['maxsample']
    # Screening window: the last N history frames of every trajectory.
    # 0 is normalized to 10 by the workflow (legacy semantics).
    record_step: int = DEFAULT['record_step']
    # Keep the round's MD directory after the round (0 deletes it).
    keep_md: int = DEFAULT['keep_md']
    # Physical screening switches (default on). `check_population` collects
    # the structure the md loop stopped on when its own `&md md_maxpop` guard
    # judged the population ill; the tolerance itself lives in `&md`, so this
    # switch only decides whether that structure goes to the reference
    # calculation (dev/methods/ADAPTIVE_DEVELOPMENT.md section 16).
    check_population: int = DEFAULT['check_population']
    check_energy_order: int = DEFAULT['check_energy_order']
    gap_tolerance: float = DEFAULT['gap_tolerance']
    # Property channels the adaptive loop labels and trains on. Energy and
    # gradient are the ML baseline; the others are opt-in and must be
    # supported by both the md/reference skills and the label extractor.
    sample_energy: int = DEFAULT['sample_energy']
    sample_grad: int = DEFAULT['sample_grad']
    sample_nac: int = DEFAULT['sample_nac']
    sample_soc: int = DEFAULT['sample_soc']
    sample_dipole: int = DEFAULT['sample_dipole']
    sample_transition_dipole: int = DEFAULT['sample_transition_dipole']
    sample_oscillator_strength: int = DEFAULT['sample_oscillator_strength']
    dynsample: int = DEFAULT['dynsample']
    maxdiscard: int = DEFAULT['maxdiscard']
    # Training-data quality gate: a fit whose validation energy MAE exceeds
    # this value (eV) is retried after the outlier frames of this round are
    # removed one at a time. 0 disables the gate.
    maxmae: float = DEFAULT['maxmae']

    # Analysis layer. Every switch defaults off; the adaptive workflow
    # writes analysis_iter-<k>.json only when at least one is on.
    analyze_pes: int = DEFAULT['analyze_pes']
    analyze_gap: str = DEFAULT['analyze_gap']
    analyze_trace: int = DEFAULT['analyze_trace']
    analyze_coverage: int = DEFAULT['analyze_coverage']
    analyze_window: int = DEFAULT['analyze_window']
    analyze_local_tolerance: float = DEFAULT['analyze_local_tolerance']
    analyze_global_tolerance: float = DEFAULT['analyze_global_tolerance']
    predict_mode: int = DEFAULT['predict_mode']
    gap_threshold: float = DEFAULT['gap_threshold']
    explore_udd: int = DEFAULT['explore_udd']
    explore_gap: int = DEFAULT['explore_gap']
    udd_mode: str = DEFAULT['udd_mode']
    udd_amplitude: float = DEFAULT['udd_amplitude']
    udd_width: float = DEFAULT['udd_width']
    udd_start_iteration: int = DEFAULT['udd_start_iteration']
    gap_method: str = DEFAULT['gap_method']
    gap_pair: list = Field(default_factory=list)
    gap_height: float = DEFAULT['gap_height']
    gap_width: float = DEFAULT['gap_width']

    # Modularized execution keywords (tri-state: None keeps the
    # workflow class defaults during graph construction)
    multitask: Optional[bool] = None
    remotetask: Optional[bool] = None
    # Multitask worker pool size of this workflow's dispatch fan-out
    # (1 keeps serial execution).
    multipool: int = Field(default=DEFAULT['multipool'], ge=1)

    # Per-stage execution keywords. They override the generic triple
    # above; a None flag falls back to the generic keyword and then to
    # the workflow class default. reference_remotetask does not submit
    # the framework runtime: it switches the reference skill itself to
    # its slurm execution mode.
    md_multitask: Optional[bool] = DEFAULT['md_multitask']
    md_remotetask: Optional[bool] = DEFAULT['md_remotetask']
    md_multipool: Optional[int] = DEFAULT['md_multipool']
    reference_multitask: Optional[bool] = DEFAULT['reference_multitask']
    reference_remotetask: Optional[bool] = DEFAULT['reference_remotetask']
    reference_multipool: Optional[int] = DEFAULT['reference_multipool']

    # Refinement
    refine: int = DEFAULT['refine']
    refine_num: int = DEFAULT['refine_num']
    refine_start: int = DEFAULT['refine_start']
    refine_end: int = DEFAULT['refine_end']
    refine_gap: float = DEFAULT['refine_gap']

    # Energy thresholds
    maxenergy: float = DEFAULT['maxenergy']
    minenergy: float = DEFAULT['minenergy']
    dynenergy: float = DEFAULT['dynenergy']
    inienergy: float = DEFAULT['inienergy']
    fwdenergy: int = DEFAULT['fwdenergy']
    bckenergy: int = DEFAULT['bckenergy']

    # Gradient thresholds
    maxgrad: float = DEFAULT['maxgrad']
    mingrad: float = DEFAULT['mingrad']
    dyngrad: float = DEFAULT['dyngrad']
    inigrad: float = DEFAULT['inigrad']
    fwdgrad: int = DEFAULT['fwdgrad']
    bckgrad: int = DEFAULT['bckgrad']

    # NAC thresholds
    maxnac: float = DEFAULT['maxnac']
    minnac: float = DEFAULT['minnac']
    dynnac: float = DEFAULT['dynnac']
    ininac: float = DEFAULT['ininac']
    fwdnac: int = DEFAULT['fwdnac']
    bcknac: int = DEFAULT['bcknac']

    # SOC thresholds
    maxsoc: float = DEFAULT['maxsoc']
    minsoc: float = DEFAULT['minsoc']
    dynsoc: float = DEFAULT['dynsoc']
    inisoc: float = DEFAULT['inisoc']
    fwdsoc: int = DEFAULT['fwdsoc']
    bcksoc: int = DEFAULT['bcksoc']


# ---------------------------------------------------------------------
# The system constraint block (dev/methods/CONSTRAINT_DEVELOPMENT.md).
#
# These models are the input-side vocabulary of `System.constraints`, and they
# sit here for the same reason `System` itself does: one place states what an
# input may declare. `schema/normalize.py` turns a validated block into a plan
# and `schema/adapter.py` reads the classic `&constraint` text into the mapping
# these models take.
#
# Style note. The rest of this file declares plain models without a
# model_config, because those absorb both the classic input and the normalized
# output and cannot afford to reject a field. These records are not in that
# position: they read only the constraint section, so they can and should be
# strict, which is why `extra='forbid'` appears here and nowhere else. The
# strictness belongs to the block, not to the flat namespace of `System`.
# ---------------------------------------------------------------------

class ConstraintInputError(ValueError):
    """Raised when a constraint declaration cannot be normalized."""


class UnsupportedCapabilityError(ConstraintInputError):
    """Raised when a declaration names something not implemented."""


def _strict():
    """Config shared by every record in this file."""
    return ConfigDict(extra='forbid')


LENGTH_UNITS = ('angstrom', 'bohr')
ANGLE_UNITS = ('degree', 'radian')
ENERGY_UNITS = ('hartree', 'ev', 'kcal', 'cm')
TARGET_KINDS = ('atom', 'com', 'cog')
WALL_GEOMETRIES = ('sphere', 'ellipsoid', 'box', 'triclinic')
# The two potential shapes a wall's geometry can carry. `hard` is the one-sided
# form this subsystem shipped with - nothing until the surface is crossed, then
# `k * d^p`. `soft` is the archived polynomial cavity, `f * (r/r0)^a` in the
# reduced distance: restoring on both sides of the center, and because its
# argument is a ratio of lengths its constant converts as a plain energy.
WALL_LAWS = ('hard', 'soft')
CV_KINDS = ('bond', 'angle', 'dihedral')
DIRECTIONS = ('x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz')

class UnitsInput(BaseModel):
    """Input unit spelling."""

    model_config = _strict()

    length: str = 'angstrom'
    angle: str = 'degree'
    energy: str = 'hartree'

    @field_validator('length')
    @classmethod
    def _check_length(cls, value):
        if value not in LENGTH_UNITS:
            raise ValueError('unknown length unit %r; expected %s'
                             % (value, ', '.join(LENGTH_UNITS)))
        return value

    @field_validator('angle')
    @classmethod
    def _check_angle(cls, value):
        if value not in ANGLE_UNITS:
            raise ValueError('unknown angle unit %r; expected %s'
                             % (value, ', '.join(ANGLE_UNITS)))
        return value

    @field_validator('energy')
    @classmethod
    def _check_energy(cls, value):
        if value not in ENERGY_UNITS:
            raise ValueError('unknown energy unit %r; expected %s'
                             % (value, ', '.join(ENERGY_UNITS)))
        return value


class _AtomList(BaseModel):
    """Base of a record that carries the atoms its own object line listed.

    `atoms` is what the flat grammar writes: `bias_bond 1 2`, `meta_rmsd 1 2 3`,
    `cv_bond 13 14`. A record whose selection can also be named by element or
    by a group inherits `_Selector` below.
    """

    model_config = _strict()

    atoms: list = Field(default_factory=list)

    @field_validator('atoms', mode='before')
    @classmethod
    def _coerce_atoms(cls, value):
        # Named uniquely on purpose: pydantic keys a validator by attribute, so
        # a subclass that defines its own `_as_list` would silently drop this
        # one and leave a string like '13-30' unconverted.
        if value is None:
            return []
        if isinstance(value, (str, int, np.integer)):
            return [value]
        return list(value)

    def selector(self, name):
        """The one selection source actually given, as a mapping."""
        if not self.atoms:
            raise ConstraintInputError('%s: no atom selection given' % name)
        return {'atoms': self.atoms}


class _Selector(_AtomList):
    """A record that may name its atoms by element or by a declared group.

    Only the families whose object line does not carry the atoms can use these:
    a wall and a fix take their geometry from `wall_type` or from `fix_group`
    and count their objects from a separate line, so a parameter line of
    variable length can align over them. A bias, a cv and a history potential
    take their atoms from their own object line, whose segments are what counts
    the objects, so an element list has nothing to align against.
    """

    element: list = Field(default_factory=list)
    group: str = Field(default='')

    @field_validator('element', mode='before')
    @classmethod
    def _coerce_elements(cls, value):
        if value is None:
            return []
        if isinstance(value, (str, int, np.integer)):
            return [value]
        return list(value)

    def selector(self, name):
        """The one selection source actually given, as a mapping."""
        given = []
        if self.atoms:
            given.append(('atoms', self.atoms))
        if self.element:
            given.append(('element', self.element))
        if self.group:
            given.append(('group', self.group))
        if not given:
            raise ConstraintInputError('%s: no atom selection given' % name)
        if len(given) > 1:
            raise ConstraintInputError(
                '%s: selection sources are mutually exclusive, got %s'
                % (name, ' and '.join(sorted(key for key, _ in given))))
        return dict(given)


class GroupInput(_Selector):
    """One named group of atoms."""

    name: str = Field(default='')


class CVInput(BaseModel):
    """One named collective variable.

    `atoms` are the row the object line listed: `cv_bond 13 14` is the bond of
    atoms 13 and 14. A cv is one variable, so one row.
    """

    model_config = _strict()

    name: str = Field(default='')
    kind: str = Field(default='')
    atoms: list = Field(default_factory=list)

    @field_validator('atoms', mode='before')
    @classmethod
    def _as_list(cls, value):
        if value is None:
            return []
        if isinstance(value, (str, int, np.integer)):
            return [value]
        return list(value)

    @field_validator('kind')
    @classmethod
    def _check_kind(cls, value):
        if value and value not in CV_KINDS:
            raise ValueError('unknown cv kind %r; expected %s'
                             % (value, ', '.join(CV_KINDS)))
        return value


class BiasInput(_AtomList):
    """A harmonic bias on one scalar collective variable.

    The law is harmonic and only harmonic: a bias kind that is not implemented
    has no field to hold its name.
    """

    model_config = _strict()

    id: str = Field(default='')
    cv: str = Field(default='')
    target: Optional[float] = Field(default=None, allow_inf_nan=False)
    force_constant: Optional[float] = Field(
        default=None, allow_inf_nan=False)
    region: str = Field(default='all')
    schedules: list = Field(default_factory=list)

    @field_validator('schedules', mode='before')
    @classmethod
    def _coerce_inline_schedules(cls, value):
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        return list(value)

    @field_validator('region')
    @classmethod
    def _check_region(cls, value):
        if value not in ('all', 'upper', 'lower', 'inside', 'outside'):
            raise ValueError('unknown region %r' % value)
        return value


class WallInput(_Selector):
    """An additive wall potential with a declared geometry and target."""

    model_config = _strict()

    id: str = Field(default='')
    geometry: str = Field(default='sphere')
    center: list = Field(default_factory=list)
    # The anchor of the wall, named by atoms instead of by coordinates: the
    # point is their center, captured on the reference geometry and fixed for
    # the run. `center` wins over this, and both fall back to the center of the
    # selection.
    ref_atoms: list = Field(default_factory=list)
    # Molecule groups the wall acts on, each a list of 0-based atom numbers.
    # The text form is `wall_groups`, whose last number per segment says how
    # many atoms a molecule holds. With one group the wall confines the center
    # of the whole selection, which is what it did before groups existed.
    molecules: list = Field(default_factory=list)
    # 'hard' is the one-sided wall: energy only past the surface. 'soft' is the
    # archived polynomial cavity, `f * (r/r0)^a` in the reduced axis ratio, so
    # its constant converts as an energy and not as an energy over a length to
    # the power.
    law: str = Field(default='hard')
    radius: Optional[float] = Field(default=None, allow_inf_nan=False)
    semiaxes: list = Field(default_factory=list)
    lengths: list = Field(default_factory=list)
    cell: list = Field(default_factory=list)
    power: int = Field(default=2)
    force_constant: Optional[float] = Field(
        default=None, allow_inf_nan=False)
    target: str = Field(default='atom')
    schedules: list = Field(default_factory=list)

    @field_validator('ref_atoms', 'molecules', mode='before')
    @classmethod
    def _as_rows(cls, value):
        """Atom numbers of the anchor, or the molecule groups, either shape."""
        if value is None:
            return []
        if isinstance(value, str):
            value = value.replace(',', ' ').split()
        if isinstance(value, (int, np.integer)):
            return [int(value)]
        return list(value)

    @field_validator('law')
    @classmethod
    def _check_law(cls, value):
        if value not in WALL_LAWS:
            raise ValueError('unknown wall law %r; expected %s'
                             % (value, ', '.join(WALL_LAWS)))
        return value

    @field_validator('schedules', mode='before')
    @classmethod
    def _coerce_wall_schedules(cls, value):
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        return list(value)

    @field_validator('center', 'semiaxes', 'lengths', 'cell',
                     mode='before')
    @classmethod
    def _as_list(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return value.replace(',', ' ').split()
        out = []
        for item in value:
            if isinstance(item, (list, tuple)):
                out.extend(item)
            else:
                out.append(item)
        return out

    @field_validator('geometry')
    @classmethod
    def _check_geometry(cls, value):
        if value not in WALL_GEOMETRIES:
            raise ValueError('unknown wall geometry %r; expected %s'
                             % (value, ', '.join(WALL_GEOMETRIES)))
        return value

    @field_validator('target')
    @classmethod
    def _check_target(cls, value):
        if value not in TARGET_KINDS:
            raise ValueError('unknown wall target %r; expected %s'
                             % (value, ', '.join(TARGET_KINDS)))
        return value


class FixInput(_Selector):
    """Frozen components of the coordinate, anchored at the start geometry."""

    model_config = _strict()

    id: str = Field(default='')
    direction: str = Field(default='xyz')

    @field_validator('direction')
    @classmethod
    def _check_direction(cls, value):
        if value not in DIRECTIONS:
            raise ValueError('unknown freeze direction %r; expected %s'
                             % (value, ', '.join(DIRECTIONS)))
        return value


class HardConstraintInput(BaseModel):
    """A strict geometric constraint solved by projection.

    The rows come from the family's own object line: `shake_bond 1 2` is one
    bond, `shake_bond_type O-H` expands to every perceived O-H pair through the
    plan's bond graph. `tolerance` and `max_iterations` belong to the position
    solve, and `velocity_constraint` says whether the velocities are projected
    onto the same manifold afterwards. All three travel to the plan as solver
    settings on the record, so a restart that reads a different setting is a
    different plan.
    """

    model_config = _strict()

    id: str = Field(default='')
    kind: str = Field(default='bond')
    bonds: list = Field(default_factory=list)
    bond_type: str = Field(default='')
    target: Optional[float] = Field(default=None, allow_inf_nan=False)
    tolerance: float = Field(default=1.0e-8, gt=0.0)
    max_iterations: int = Field(default=200, gt=0)
    # `none` projects positions only; `rattle` also projects velocities.
    velocity_constraint: str = Field(default='none')

    @field_validator('bonds', mode='before')
    @classmethod
    def _as_pairs(cls, value):
        """Every declared pair, from rows, from lists, or from DSL text.

        The block reader files `bond 13 14` as the string '13 14', so a text
        pair and a numeric pair have to reach the same row. Splitting a string by
        characters instead of by spaces would turn one declaration into three
        indices and a refusal that names neither atom.
        """
        if value is None:
            return []
        if isinstance(value, (str, int, np.integer)):
            value = [value]
        return [pair.split() if isinstance(pair, str) else list(pair)
                for pair in value]

    @model_validator(mode='before')
    @classmethod
    def _inline_bond(cls, data):
        """A strict constraint keeps every pair it was given.

        The selector base rewrites a lone 'bond 13 14' into an atom pair, which
        is what a bias or a cv needs because they bind exactly one variable. A
        constraint is a list of bonds by nature, so that rewriting must not run
        here; with it inherited, a two-bond SHAKE declaration was refused as
        "takes one inline bond". Overriding by name is what pydantic keys on.
        """
        return data

    @field_validator('kind')
    @classmethod
    def _check_kind(cls, value):
        if value not in ('bond', 'angle', 'dihedral'):
            raise ValueError('unknown constraint kind %r' % value)
        return value

    @field_validator('velocity_constraint')
    @classmethod
    def _check_velocity_constraint(cls, value):
        """Which half of the constraint the velocities get.

        `none` is the default and the measured case: positions are projected and
        velocities are left alone. `rattle` adds the velocity projection after
        the position solve. It does not replace the position path, which is why
        this is a switch on the record rather than a new kind of constraint.
        """
        if value not in ('none', 'rattle'):
            raise ValueError(
                "velocity_constraint must be 'none' or 'rattle', got %r"
                % value)
        return value


class MetadynamicsInput(_AtomList):
    """A history potential deposited along one or more scalars.

    With `kind` 'gaussian' the scalars are named collective variables; with
    'rmsd' there is exactly one, the squared deviation of the selected group
    after optimal superposition onto a reference geometry. The rmsd block is
    the atoms `meta_rmsd` listed, so its selection has no element or group
    spelling to align against.
    """

    model_config = _strict()

    id: str = Field(default='')
    kind: str = Field(default='gaussian')
    cvs: list = Field(default_factory=list)
    height: Optional[float] = Field(default=None, allow_inf_nan=False)
    width: list = Field(default_factory=list)
    pace: int = Field(default=100, gt=0)
    schedules: list = Field(default_factory=list)

    @field_validator('schedules', mode='before')
    @classmethod
    def _coerce_meta_schedules(cls, value):
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        return list(value)
    reference: list = Field(default_factory=list)
    weight: str = Field(default='uniform')
    # Well-tempered metadynamics (Laio and Parrinello, PNAS 99:12562, 2002):
    # each hill is deposited with its height scaled by exp(-V(s) / kB dT), where
    # dT = (bias_factor - 1) * temperature. Both are 0 by default, which is the
    # plain deposition this subsystem shipped with, and a bias factor is
    # meaningless without the temperature it is a ratio of.
    bias_factor: float = Field(default=0.0, allow_inf_nan=False)
    temperature: float = Field(default=0.0, allow_inf_nan=False)

    @field_validator('bias_factor', 'temperature', mode='after')
    @classmethod
    def _check_well_tempered(cls, value):
        if value < 0.0:
            raise ValueError(
                'a well-tempered setting cannot be negative, got %r' % value)
        return value

    @field_validator('cvs', 'width', 'reference', mode='before')
    @classmethod
    def _as_lists(cls, value):
        if value is None:
            return []
        if isinstance(value, (str, int, float, np.integer, np.floating)):
            return [value]
        return list(value)

    @field_validator('kind')
    @classmethod
    def _check_kind(cls, value):
        if value not in ('gaussian', 'rmsd'):
            raise ValueError('unknown metadynamics kind %r' % value)
        return value

    @field_validator('weight')
    @classmethod
    def _check_weight(cls, value):
        if value not in ('uniform', 'mass'):
            raise ValueError(
                "unknown rmsd weight %r; expected 'uniform' or 'mass'"
                % value)
        return value


class SystemConstraintConfig(BaseModel):
    """The whole `&constraint` or `system.constraints` declaration."""

    model_config = _strict()

    units: UnitsInput = Field(default_factory=UnitsInput)
    groups: list = Field(default_factory=list)
    cvs: list = Field(default_factory=list)
    biases: list = Field(default_factory=list)
    walls: list = Field(default_factory=list)
    fixes: list = Field(default_factory=list)
    constraints: list = Field(default_factory=list)
    metadynamics: list = Field(default_factory=list)
    # What the reader did that the input did not say: a parameter broadcast over
    # several objects, a ramp whose stop came from the length of the run, a wall
    # geometry `wall_type` does not read. The text front-end fills this in, the
    # schema appends to it, and the run record prints it, because a value that
    # reaches an object the input never named has to be readable somewhere.
    notes: list = Field(default_factory=list)

    @field_validator('notes', mode='before')
    @classmethod
    def _notes_are_text(cls, value):
        if value is None:
            return []
        return [str(item) for item in value]

    @field_validator('groups', 'cvs', 'biases', 'walls',
                     'fixes', 'constraints', 'metadynamics',
                     mode='before')
    @classmethod
    def _as_list(cls, value):
        if value is None:
            return []
        if isinstance(value, dict):
            # A single mapping is a one-item list; a mapping keyed by name is
            # not, because the id would move from the value into the key.
            return [value]
        return list(value)

    @field_validator('groups')
    @classmethod
    def _groups_are_records(cls, value):
        return [item if isinstance(item, GroupInput) else GroupInput(**item)
                for item in value]

    @field_validator('cvs')
    @classmethod
    def _cvs_are_records(cls, value):
        return [item if isinstance(item, CVInput) else CVInput(**item)
                for item in value]

    @field_validator('biases')
    @classmethod
    def _biases_are_records(cls, value):
        return [item if isinstance(item, BiasInput) else BiasInput(**item)
                for item in value]

    @field_validator('walls')
    @classmethod
    def _walls_are_records(cls, value):
        return [item if isinstance(item, WallInput) else WallInput(**item)
                for item in value]

    @field_validator('fixes')
    @classmethod
    def _fixes_are_records(cls, value):
        return [item if isinstance(item, FixInput) else FixInput(**item)
                for item in value]

    @field_validator('constraints')
    @classmethod
    def _constraints_are_records(cls, value):
        return [item if isinstance(item, HardConstraintInput)
                else HardConstraintInput(**item) for item in value]

    @field_validator('metadynamics')
    @classmethod
    def _metad_are_records(cls, value):
        return [item if isinstance(item, MetadynamicsInput)
                else MetadynamicsInput(**item) for item in value]

    def is_empty(self):
        """True when nothing is declared; the caller then gets an empty
        plan."""
        return not any([self.groups, self.cvs, self.biases,
                        self.walls, self.fixes,
                        self.constraints, self.metadynamics])


# The archive's flat constraint and external-potential spellings, refused by
# name. These are not fields of `System` any more: `&constraint` (and the
# structured `system.constraints` block) is the only way to declare a restraint.
# The table exists so that an input which still writes one of them is told where
# that declaration lives now, rather than having it dropped by a lenient model -
# a constraint that arrives nowhere is the most expensive failure this subsystem
# can have, and it is silent by construction.
RETIRE_CONSTRAINT_KEYS = {
    'freeze': ('frozen atoms are a fix: write `fix_atoms <indices>` in '
               '&constraint (`fix_dir xyz` is the default)'),
    'constrain': ('the archive used this as the selection the cavity potential '
                  'acted on: write `wall_atoms <indices>` on the wall, or leave '
                  'it out and let `wall_groups` name the atoms'),
    'cbond': 'write `shake_bond 1 2; 3 4`',
    'cangle': 'write `shake_angle 1 2 3`',
    'cdihedral': 'write `shake_dihedral 1 2 3 4`',
    'tbond': 'write `shake_bond_ref <length>`',
    'tangle': 'write `shake_angle_ref <angle>`',
    'tdihedral': 'write `shake_dihedral_ref <angle>`',
    'fbond': 'write `bias_bond_fc <k>` with `bias_bond i j`',
    'fangle': 'write `bias_angle_fc <k>` with `bias_angle i j k`',
    'fdihedral': 'write `bias_dihedral_fc <k>` with `bias_dihedral i j k l`',
    'shape': ("the polynomial cavity retired into the wall's own law: write "
              '`wall_type ellipsoid` (or `box`) with `wall_law soft`'),
    'factor': ('write `wall_power <p>` on a `wall_law soft` wall; the '
               'archived code floored it at 2 silently, this build refuses '
               'below 2'),
    'scale': ('write `wall_fc <value>`: with `wall_law soft` it converts as '
              'an energy, exactly as the cavity `scale` did'),
    'cavity': ('write `wall_sphere_radius`, `wall_ellipsoid_semiaxes` or '
               '`wall_box_lengths`; a cuboid radius is a box half edge, so '
               'double it'),
    'center': ('write `wall_center x y z` for the point the archived code '
               'called the cavity center'),
    'center_type': ("'mass' is `wall_target com`, 'geometric' is `wall_target "
                    "cog`, and 'origin' is `wall_center 0 0 0`"),
    'compress': ('write a geometry ramp: `wall_sphere_radius_end <ratio * '
                 'radius>` with `wall_sphere_radius_delay 0` and '
                 '`wall_sphere_radius_stop <steps>`'),
    'groups': ('two different things shared this name: fragment groups are '
               '`center_groups` now, and the cavity molecule partition is the '
               '`wall_groups` line of &constraint'),
}


class System(BaseModel):
    """System information including atomic structure, electronic settings,
    and molecular constraints (skill-independent).

    Corresponds to classic key_molecule section plus structural data.
    """

    # Atomic structure
    atoms: List[str] = []                      # (natom,) Element symbols
    mass: List[float] = []                     # (natom,) Atomic masses
    coordinates: List[List[float]] = []        # (natom, 3) Cartesian coordinates
    velocities: List[List[float]] = []         # (natom, 3) Cartesian velocities
    target_coord: List[List[float]] = []       # (natom, 3) Target coordinates
    # (natom, 4) [q, x, y, z] per-atom partial charges in framework
    # atom order (row i is atom i), Angstrom / e. Composite tasks read
    # the region slices; a skill never receives the whole array.
    point_charges: List[List[float]] = []
    # (natom, 4) [V, Fx, Fy, Fz] external electrostatic potential/field
    # at each QM atom, atomic units. Used by ESPF potential embedding.
    external_potential: List[List[float]] = []
    # Path to a legacy '.charge' file whose rows are arbitrary external
    # charges (ncharge / comment / 'q x y z'). It is background charge
    # for theory[0] / reference[0] only; it is never sliced and never
    # combined with a multiscale partition.
    external_charges: str = ''

    # Electronic settings
    # charges/multiplicity carry a center axis (docs
    # MULTICENTER_DEVELOPMENT.md section 4): a scalar is the
    # single-center (C=1) form and, for C>1, broadcasts to every
    # center; a list is the per-center form list[center]. Each center's
    # value is its region electronic structure; the full-system
    # L(real) aggregate (sum of charges, paired multiplicity) is
    # derived by data.center.build_center_plan.
    charges: Union[int, List[int]] = DEFAULT['charges']
    total_electron: int = 0
    nalpha: int = 0
    nbeta: int = 0
    multiplicity: Union[int, List[int]] = DEFAULT['multiplicity']
    natom: int = 0

    # State information (from key_molecule)
    # Center axis (dev/MULTICENTER_DEVELOPMENT.md section 4): every field
    # accepts the flat single-center form (C=1, backward compatible)
    # and the nested per-center form list[center][...]. The canonical
    # per-center record is derived by data.center.build_center_plan;
    # the flat fields below remain the C=1 projection.
    ci: Union[List[int], List[List[int]]] = DEFAULT['ci']              # Number of states per spin multiplicity (flat) or per center (nested)
    spin: Union[List[int], List[List[int]]] = DEFAULT['spin']          # Total spin S per multiplicity (flat) or per center (nested)
    coupling: Union[List[List[int]], List[List[List[int]]]] = DEFAULT['coupling']  # Transition pairs (flat) or per center (nested)
    nstate: int = 1
    current_state: int = 0                     # Currently active electronic state (0-based)
    state_label: List[str] = []                # (nstate,) State labels
    state_spin: List[int] = []                 # (nstate,) 2S per state (0=singlet, 2=triplet)
    # FSSH electronic state density matrix persisted across
    # timesteps: (nstate, nstate, 2) with each element stored as a
    # [re, im] pair (complex values stay JSON serializable).
    state_density: List[List[List[float]]] = []

    # Pair information
    npair: int = 0
    nnac: int = 0
    nsoc: int = 0
    transition_pair: List[List[int]] = [[]]    # (npair, 2) Transition pairs
    nac_pair_map: List[List[int]] = [[]]       # (nnac, 2) NAC pair mapping
    soc_pair_map: List[List[int]] = [[]]       # (nsoc, 2) SOC pair mapping

    # QM/MM settings (from key_molecule)
    qmmm_key: Optional[str] = None
    qmmm_xyz: str = 'Input'
    # QM 0-order charges in highlevel order (framework atom order). They
    # seed the LAMMPS qmmm_energy task; a QM backend that returns
    # computed_charges overrides them at run time. Empty means "take
    # them from the QM result".
    q_qm: List[float] = []
    highlevel: Union[List[int], List[List[int]]] = []  # High-level QM region indices (flat=single center, nested=multi-center)
    midlevel: Union[List[int], List[List[int]]] = []   # Mid-level region indices (flat=single center, nested=multi-center)
    # Reference partition for adaptive sampling. The reference method list
    # is independent of the theory queue: reference[0] is its high layer,
    # reference[1] (three-layer spelling) its mid layer, and the last entry
    # its low layer over the remaining atoms. These two fields define the
    # regions those layers cover; they are parsed exactly like
    # highlevel/midlevel (1-based, flat or center-group form) but never
    # enter the theory center plan. Empty inherits highlevel/midlevel when
    # the theory queue is layered; with one theory method the high layer is
    # the whole system unless a proper subset is written here.
    highref: Union[List[int], List[List[int]]] = []
    midref: Union[List[int], List[List[int]]] = []
    embedding: bool = False
    read_charge: bool = False
    # Who owns the 0-order field gradient term F_el = q_qm * E of a QM/MM
    # run. True is the Molcas ESPF contract: its analytic gradient already
    # contains -q_i * E_i, so composite removes F_el from the classical
    # Q-row correction to avoid counting it twice. False declares that the
    # QM provider does NOT contain it -- the fixed-potential derivative of
    # a polarizable NN (dev/methods/QMMM_NN_drafts.md 5.5) -- and the term
    # is then kept on the classical side. It never changes an energy.
    qm_contains_field_gradient: bool = True
    # Charge self-consistency of a multi-center QM/MM run. The field each
    # center is embedded in carries the *other* centers' 0-order charges, so
    # the charges and the field have to be iterated before anything is
    # asked of the wavefunction. False runs one charge round and then the
    # property in the field those charges produced (two QM solves per
    # center); True repeats the charge rounds until every center's charges
    # move by at most sc_charge_threshold, or until sc_charge_maxiter rounds
    # have run, and records which of the two stopped it. With one center the
    # field holds no QM charge, so there is nothing to iterate.
    sc_charge: bool = False
    sc_charge_maxiter: int = 10
    sc_charge_threshold: float = 1e-2
    # Weight of each round's fresh fit in the charges the next round is solved
    # in. 1.0 accepts the fit as it is, which is what every run did before this
    # key existed; a smaller weight damps a coupled iteration that oscillates
    # instead of settling (two centers polarizing each other can make the
    # undamped map non-contractive). The convergence test is taken on the raw
    # fit either way, so the damping changes the path and never what counts as
    # converged (`workflow/multiscale/charges.py::damp_charges`).
    sc_charge_damping: float = 1.0
    boundary: List[List[int]] = []             # Boundary atom groups

    # System constraint block (dev/methods/CONSTRAINT_DEVELOPMENT.md).
    # None is the default-closed signal: no block, no plan, no behaviour
    # change, and no keyword can enable a feature by being present.
    #
    # This is the only constraint surface `System` has. The archive's flat
    # spellings (the nineteen `freeze`/`cbond`/`cavity`/`shape`/... keys this
    # field replaced) are no longer fields here, and they are not folded: an
    # input that writes one is refused at the input boundary with the keyword of
    # the block that says what it meant, because a restraint that arrives
    # nowhere is the expensive direction.
    constraints: Optional[SystemConstraintConfig] = None

    # Center axis (dev/MULTICENTER_DEVELOPMENT.md): fragment groups, one list
    # of atom numbers per fragment, and the composite planner's cue for a
    # `fragment` task queue.
    #
    # The name says `center_` on purpose. While this field was called `groups`
    # it collided with the constraint block's named atom groups, which is a
    # different thing, and the collision was not hypothetical: the legacy cavity
    # read the same field as its molecule partition, so a multicenter input that
    # also declared a cavity had its fragment groups read as molecule groups,
    # with a different index base and a different meaning, and nothing raised
    # (CHANGELOG #272).
    #
    # The atom numbers here are 0-based, the convention the composite planner
    # has always read, while every atom number in the constraint block is
    # 1-based in the input and decremented once. That asymmetry is inherited, not
    # chosen, and is recorded here because it is the first thing a reader of
    # `center_groups 0 1; 2` should know.
    center_groups: List[List[int]] = []        # (nfrag, ...) 0-based atom numbers

    # Geometry tracking of the md trajectory (legacy GeomTracker). `dist`
    # compares each atom pair in track_index, `frag` the centers of exactly two
    # groups; the comparison is `distance > threshold`, in Angstrom (the unit
    # of the stored coordinates). Atom numbers are 1-based in every external
    # spelling and decremented once on the way in. A threshold list shorter
    # than the tracked objects is broadcast with its last value (legacy), and
    # `track_stop` 0 reports only while 1 ends the trajectory after the
    # offending frame is written; 2 (suppress surface hopping) is not
    # implemented and is refused (utilities/tracker.py::plan).
    track_type: Optional[str] = None
    track_index: List[List[int]] = []
    track_thrhd: List[float] = []
    track_stop: int = 0

    # Periodic settings
    solvent: str = DEFAULT['solvent']
    pbc: List[int] = DEFAULT['pbc']            # (3,) Periodic boundary conditions (0/1)
    cell: List[float] = DEFAULT['cell']        # (6,) Cell parameters
    lattice_vector: List[List[float]] = DEFAULT['lattice_vector']  # (3, 3)
    kpoint: List[int] = DEFAULT['kpoint']      # (3,) K-point grid
    volume: float = 0
    # image: (natom, 3) count of cell lengths each atom crossed, written
    # by the coordinate wrap. coordinates stay the wrapped (main image)
    # values, so this is what makes the continuous trajectory and the
    # cross-frame displacements recoverable. Runtime state like the
    # extended-ensemble block below: persisted with the history and the
    # checkpoint, never an input keyword, never synced to a backend.
    image: List[List[int]] = []

    # Extended-ensemble dynamical degrees of freedom (dev/methods/
    # ENSEMBLE_DEVELOPMENT.md section 4). These are RUNNING STATE persisted
    # across timesteps (checkpoint/restart, history) and are NEVER
    # synced to a backend (excluded from SYNC_SYSTEM_FIELDS).
    # thermostat_dof: Nose-Hoover chain [xi_1..xi_M, p_xi_1..p_xi_M].
    # barostat_strain: full-tensor cell strain epsilon (3, 3).
    # barostat_momentum: conjugate cell momentum p_epsilon (3, 3).
    # barostat_velocity: instantaneous cell velocity v_epsilon =
    # p_epsilon / W (3, 3), written by the barostat half-step and read
    # by the cell-aware Verlet drift of the same step (transient).
    thermostat_dof: List[float] = []
    barostat_strain: List[List[float]] = []
    barostat_momentum: List[List[float]] = []
    barostat_velocity: List[List[float]] = []
    # barostat_thermostat_dof: Nose-Hoover chain thermalizing the cell
    # momentum (MTTK NHC-baro), same layout as thermostat_dof.
    barostat_thermostat_dof: List[float] = []

    # Dispatch-layer execution keywords at molecule (center) scope.
    # Tri-state: None means "unspecified at molecule level", so the
    # planner inherits from the involved skills' declarations; an
    # explicit True/False overrides that inheritance and can resolve a
    # skill conflict (dev/MULTICENTER_DEVELOPMENT.md, molecule flags).
    # multipool=None likewise means "inherit", kept distinct from the
    # serial default 1 the workflow sections use.
    multitask: Optional[bool] = None
    remotetask: Optional[bool] = None
    multipool: Optional[int] = None

    # Structural checks only (shape / finiteness). The semantic rules
    # ("pbc on implies a valid cell", backend PBC support, partial
    # periodicity rejection) live in schema.validate; lenient input
    # forms are normalized in schema.normalize and schema.adapter.
    # The model does NOT enable validate_assignment: the NPT loop and
    # the backends assign these fields every step through
    # update_system / config.system, and revalidating on assignment
    # would only add cost without new safety.

    @model_validator(mode='before')
    @classmethod
    def _refuse_retired_constraint_keys(cls, data):
        """Turn a retired flat keyword into an error, not a dropped field.

        `System` is lenient about unknown keys on purpose - it absorbs the skill
        state and the legacy dumps - so the nineteen archive spellings of a
        restraint would otherwise vanish silently and the run would report an
        unconstrained trajectory that its input believed was constrained. Refusing
        here covers every route at once: classic text, YAML, JSON, the dotted
        agent form, a `.config` file and a programmatic `System(freeze=...)`.
        """
        if not isinstance(data, dict):
            return data
        retired = sorted(str(key).lower() for key in data
                         if str(key).lower() in RETIRE_CONSTRAINT_KEYS)
        if retired:
            raise ValueError(
                'system.%s %s no longer a%s input keyword; %s. The block is '
                '`system.constraints` (classic `&constraint`), and its '
                'keywords are listed in dev/methods/CONSTRAINT_DEVELOPMENT.md '
                'section 4.3'
                % (', '.join('`%s`' % key for key in retired),
                   'is' if len(retired) == 1 else 'are',
                   '' if len(retired) == 1 else 'n',
                   '; '.join('%s -> %s' % (key, RETIRE_CONSTRAINT_KEYS[key])
                             for key in retired)))
        return data

    @field_validator('embedding', 'read_charge', mode='before')
    @classmethod
    def _fold_embedding_vocabulary(cls, value, info):
        """Fold a skill's word for an embedding decision onto the bool switch.

        Both fields are framework switches. A sectioned config still hands the
        shared namespace the skill's own vocabulary ('charge', 'potential', a
        'legacy' file alias, 'none'), because one YAML key declares the
        decision on both sides, and the flatten step that builds `System`
        cannot tell which section a value came from. Both therefore fold in
        `mode='before'`: the bool is what `System` answers, and the skill's own
        word survives because its section is read separately on the way to the
        backend (`_merge_one_skill_config`), not through this model. A charge
        *file* name is in neither field's vocabulary and is refused rather than
        silently dropped: a standalone run takes its background charges from
        '{title}.charge', which is what the switch selects, and the name a skill
        wants stays in the skill's section.
        """
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ('', 'none', 'false', '0'):
                return False
            if text in ('true', '1', 'legacy', 'charge', 'field'):
                return True
            if text == 'potential':
                raise ValueError(
                    "%s='potential' is retired: write 'field' (the V/E block "
                    "per QM atom)." % info.field_name)
            raise ValueError(
                "%s does not read as a switch: %r; expected true or false, "
                "or the skill's charge/field/legacy/none vocabulary."
                % (info.field_name, value))
        return value

    @field_validator('pbc')
    @classmethod
    def _check_pbc(cls, value):
        if len(value) != 3:
            raise ValueError(
                "pbc must have exactly 3 elements (one per axis), "
                "got %d: %r" % (len(value), value))
        for flag in value:
            if flag not in (0, 1):
                raise ValueError(
                    "pbc values must be 0 or 1, got %r in %r"
                    % (flag, value))
        return value

    @field_validator('cell')
    @classmethod
    def _check_cell(cls, value):
        # Empty means "not filled"; the default [99, 99, 99, 90, 90,
        # 90] is the sentinel documented in schema.normalize.
        if value and len(value) not in (6, 9):
            raise ValueError(
                "cell must have 6 parameters or 9 components, got %d: "
                "%r" % (len(value), value))
        for item in value:
            if not math.isfinite(item):
                raise ValueError(
                    "cell contains a non-finite value: %r" % (value,))
        return value

    @field_validator('lattice_vector')
    @classmethod
    def _check_lattice_vector(cls, value):
        if not value:
            return value
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError(
                "lattice_vector must have shape (3, 3), got %r"
                % (value,))
        for row in value:
            for item in row:
                if not math.isfinite(item):
                    raise ValueError(
                        "lattice_vector contains a non-finite value: "
                        "%r" % (value,))
        return value

    @field_validator('sc_charge_damping')
    @classmethod
    def _check_sc_charge_damping(cls, value):
        # Open at zero: a weight of nothing would never move the charges and
        # the run would report a converged fixed point at its starting point.
        # Above one overshoots without any of the stability a damping term is
        # for, so that is refused too rather than read as "extra strong".
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(
                'sc_charge_damping must be in (0, 1], got %r' % (value,))
        return float(value)

    @field_validator('thermostat_dof', 'barostat_thermostat_dof',
                     'barostat_strain', 'barostat_momentum',
                     'barostat_velocity')
    @classmethod
    def _check_extended_dof(cls, value, info):
        # The extended-ensemble block is written by the integrator on
        # every step, so a NaN that appears here appears HERE and not
        # three steps later in a lattice nobody can read. Before this
        # validator a diverging anisotropic NPT run surfaced as
        # "lattice_vector contains a non-finite value" and named the
        # cell, which is the victim, not the source: the cell went NaN
        # because barostat_momentum did.
        if not value:
            return value
        for item in value:
            if isinstance(item, (list, tuple)):
                for component in item:
                    if not math.isfinite(component):
                        raise ValueError(
                            "%s contains a non-finite value: %r"
                            % (info.field_name, value))
            elif not math.isfinite(item):
                raise ValueError(
                    "%s contains a non-finite value: %r"
                    % (info.field_name, value))
        return value

    @model_validator(mode='after')
    def _check_track(self):
        # One reader for both entry points: `utilities/tracker.py::plan` is
        # what the md loop calls at step time, so an input that passes here
        # cannot fail halfway through a trajectory.
        _tracker.plan(self)
        return self


class Composite(BaseModel):
    """Composite workflow execution settings.

    Controls the execution mode of multi-task composite calculations
    (ONIOM, Fragment, QM/MM). Default is serial execution.
    """

    executor: str = 'serial'              # 'serial' or 'thread'
    max_workers: int = Field(default=1, ge=1)


# `&md` spellings that no longer answer to a reader. Every one of them was
# carried by the schema at some point, so the refusal states what took its
# place instead of calling the keyword unknown. `Dynamics` is lenient about
# unknown keys on purpose (it absorbs the classic input, the normalized
# config and the agent's dotted form), so without this table a retired
# keyword would vanish silently and the run would report a trajectory its
# input believed was restarted, throttled or reset.
RETIRE_DYNAMICS_KEYS = {
    'record': ('what one history frame stores is fixed by HistoryFrame; the '
               'number of frames kept is `history`'),
    'record_step': ('the snapshot window is `history`'),
    'restart': ('a run is continued by launching the runtime file it wrote '
                '(`pyrai3md <title>.rt`), which `checkpoint` names the '
                'cadence of'),
    'addstep': ('`step` is already the number of steps this process '
                'integrates, so a continued run adds it to the step the '
                'runtime file carries'),
    'thermo': ('the ensemble is written as `ensemble` together with '
               '`thermostat` and `barostat`'),
    'popreset': ('an out-of-range population is reported and propagated, '
                 'never collapsed (workflow/integrator/fssh.py)'),
    'graddesc': ('the dynamics integrates with velocity-Verlet only; no '
                 'gradient-descent integrator is provided'),
    'ref_energy': ('a reference calculation is a stage of the adaptive chain '
                   '(base.reference), not a keyword of &md'),
    'ref_grad': ('a reference calculation is a stage of the adaptive chain '
                 '(base.reference), not a keyword of &md'),
    'ref_nac': ('a reference calculation is a stage of the adaptive chain '
                '(base.reference), not a keyword of &md'),
    'ref_soc': ('a reference calculation is a stage of the adaptive chain '
                '(base.reference), not a keyword of &md'),
    'datapath': ('the reference records of the adaptive chain come from its '
                 'own stage configuration, not from a path keyword of &md'),
}


class Dynamics(BaseModel):
    """Dynamics settings for molecular dynamics simulations
    (skill-independent).

    Corresponds to classic key_md section.

    The tri-state execution keywords (multitask/remotetask) are
    modularized: they override the md workflow class defaults during
    execution graph construction (None keeps the defaults).
    """

    # Modularized execution keywords (tri-state: None keeps the
    # workflow class defaults during graph construction)
    multitask: Optional[bool] = None
    remotetask: Optional[bool] = None
    # Multitask worker pool size of this workflow's dispatch fan-out
    # (1 keeps serial execution).
    multipool: int = Field(default=DEFAULT['multipool'], ge=1)

    # State population (velocities live in System/Geometry)
    state_population: List[float] = [1.0]      # (nstate,)

    # Thermodynamic conditions
    temperature: float = DEFAULT['temperature']
    pressure: float = DEFAULT['pressure']
    thermostat: str = DEFAULT['thermostat']
    barostat: str = DEFAULT['barostat']
    ensemble: str = DEFAULT['ensemble']

    # Ensemble algorithm parameters (dev/methods/ENSEMBLE_DEVELOPMENT.md).
    # thermostat_chain: Nose-Hoover chain length M (1 = single NH).
    # thermostat_tau / barostat_tau: coupling time constants tau_T /
    # tau_p in fs (drive Q = tau_T^2 T0 / 4 pi^2 and W). compressibility
    # is the isothermal beta in bar^-1. barostat_mode selects the cell
    # coupling: 'isotropic' (scalar volume) or 'anisotropic' (full 3x3
    # tensor). thermostat_mass / barostat_mass optionally override the
    # tau-derived Q / W with an explicit value (atomic units). For
    # barostat 'scr' there is no W at all -- it is a first-order cell
    # equation, so barostat_tau is its volume relaxation time there and
    # not an oscillation period, which is why it needs no long tau_p.
    thermostat_chain: int = Field(default=DEFAULT['thermostat_chain'], ge=1)
    thermostat_tau: float = DEFAULT['thermostat_tau']
    barostat_tau: float = DEFAULT['barostat_tau']
    compressibility: float = DEFAULT['compressibility']
    barostat_mode: str = DEFAULT['barostat_mode']
    thermostat_mass: Optional[float] = DEFAULT['thermostat_mass']
    barostat_mass: Optional[float] = DEFAULT['barostat_mass']

    # Initial condition generation
    # initcond selects the initial-condition source:
    #   '0'/'none'         use the geometry/velocities carried by the
    #                      state (replicated across trajectories when
    #                      ninitcond > 1)
    #   'read'/'load'      read ninitcond frames from <title>.init.xyz
    #                      (coordinates + velocities per atom line)
    #   'wigner'/'boltzmann' vibrational sampling of the frequency
    #                      record '<title>.freq.<format>' in the cwd
    #                      (workflow/sampling.py; temperature +
    #                      control.gl_seed seed the draw)
    # ninitcond is the number of trajectories planned by the md
    # workflow (one task per initial condition).
    initcond: str = DEFAULT['initcond']
    ninitcond: int = DEFAULT['ninitcond']
    method: str = DEFAULT['method']            # legacy sampling method keyword (superseded by initcond)
    format: str = DEFAULT['format']            # frequency file format of the sampling
                                               # record: molden, g16, orca, bagel
    randvelo: int = 0

    # MD propagation
    step: int = DEFAULT['step']
    size: float = DEFAULT['size']              # Timestep in au
    # Initial state. Scalar for a single center; a list list[center]
    # for multi-center (docs section 4). The EXTERNAL spelling (yaml /
    # classic input / agent config, string or typed alike) is 1-based
    # like coupling/highlevel; the adapter decrements once, so this
    # field carries the internal 0-based index and is never
    # re-decremented downstream.
    root: Union[int, List[int]] = DEFAULT['root']
    activestate: int = 0                       # 0: all-state gradients; 1: current-state gradient only

    # History snapshot buffer: max number of pmd state snapshots kept
    # by PyRAIState (bounded list, oldest dropped first). Values
    # below 4 are normalized to 4 by the schema pipeline so
    # electronic dynamics algorithms always find enough past steps.
    history: int = Field(default=DEFAULT['history'], ge=1)

    # Surface hopping
    sfhp: str = DEFAULT['sfhp']                # nosh, gsh, fssh, lzsh
    nactype: str = DEFAULT['nactype']          # ktdc, etc.
    phasecheck: int = 1
    gap: float = DEFAULT['gap']                # IC hopping gap threshold
    gapsoc: float = DEFAULT['gapsoc']          # ISC hopping gap threshold
    # addsoc: constant bias added to every SOC component of one evaluation,
    # applied in the Calculator's result finalization so the biased matrix is
    # what the step records (workflow/calculator.py).
    addsoc: float = 0.0
    dosoc: int = 0
    # Property triggers consumed by planning: dodipole requests the
    # dipole capability and radiative requests transition dipoles.
    dodipole: int = 0
    radiative: str = '0'

    # Propagation details
    substep: int = DEFAULT['substep']
    rk4: str = '0'
    integrate: int = 0
    deco: str = DEFAULT['deco']                # Decoherence
    adjust: int = 1                            # Adjust velocity
    reflect: int = 1                           # Reflect velocity
    maxh: int = DEFAULT['maxh']                # Max hops

    # FSSH kernel implementation selection: every electronic
    # dynamics kernel runs either on the NumPy reference or the
    # Numba accelerated implementation (function-level choice,
    # dev/methods/FSSH_DEVELOPMENT.md). fssh_profile enables
    # the function-level timing summary.
    fssh_rk4: str = DEFAULT['fssh_rk4']
    fssh_dpdt: str = DEFAULT['fssh_dpdt']
    fssh_matb: str = DEFAULT['fssh_matb']
    fssh_ktdc: str = DEFAULT['fssh_ktdc']
    fssh_decoherence: str = DEFAULT['fssh_decoherence']
    fssh_profile: int = DEFAULT['fssh_profile']
    perf_profile: int = DEFAULT['perf_profile']
    # Debug/test control only: forces the hopping random number to
    # zero so any positive probability hops. Isolated from 'integrate'
    # (electronic substep integration mode), never used in production
    # runs.
    fssh_force_hop: int = DEFAULT['fssh_force_hop']

    # Thermodynamic control
    thermodelay: int = DEFAULT['thermodelay']
    # dissipate: 1 holds the thermostat off while the trajectory is on an
    # excited state and for `thermodelay` steps after it lands on the ground
    # state, so the excess energy leaves with the nuclear motion rather than
    # being pulled by the chain during the hop-dominated part of the run.
    # The barostat is not gated. (workflow/integrator/ensemble.py)
    dissipate: int = DEFAULT['dissipate']

    # Detection thresholds of the md chain. None leaves the channel
    # unjudged, which is what a QM-driven md run wants; an adaptive md
    # child gets the four err_* thresholds overwritten from `&adaptive`'s
    # max*. `md_maxpop` is the population tolerance: a population outside
    # [-t, 1 + t] ends the trajectory (the legacy 0.01 bound is the
    # default), and it is not overwritten by adaptive - the parent reads
    # the stop code and only decides whether to collect the structure.
    md_maxenergy: Optional[float] = DEFAULT['md_maxenergy']
    md_maxgrad: Optional[float] = DEFAULT['md_maxgrad']
    md_maxnac: Optional[float] = DEFAULT['md_maxnac']
    md_maxsoc: Optional[float] = DEFAULT['md_maxsoc']
    md_maxpop: Optional[float] = DEFAULT['md_maxpop']

    @field_validator('md_maxpop')
    @classmethod
    def _check_md_maxpop(cls, value):
        # The bound is [-t, 1 + t]: a negative tolerance would move both ends
        # inward and refuse every physical population, so it is refused here
        # rather than reinterpreted as |t|.
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                'md_maxpop must be a finite non-negative number or null, '
                'got %r' % (value,))
        return value

    # Output control
    direct: int = DEFAULT['direct']
    buffer: int = DEFAULT['buffer']
    checkpoint: int = 0

    # Velocity control. `excess` and `target` are kinetic energies in
    # Hartree (the quantity state.kinetic_energy() reports), `scale` is the
    # dimensionless factor they are reached with; `reset`/`resetstep` gate
    # the removal of the rigid motion (workflow/integrator/operations.py).
    excess: float = 0.0
    scale: float = 1.0
    target: float = 0.0
    reset: int = 0
    resetstep: int = 0

    @model_validator(mode='before')
    @classmethod
    def _refuse_retired_dynamics_keys(cls, data):
        """Turn a retired `&md` keyword into an error, not a dropped field."""
        if not isinstance(data, dict):
            return data
        retired = sorted(str(key).lower() for key in data
                         if str(key).lower() in RETIRE_DYNAMICS_KEYS)
        if retired:
            raise ValueError(
                'md.%s %s no longer an input keyword; %s. The live keywords '
                'of the block are listed in dev/methods/MD_DEVELOPMENT.md '
                'section 23'
                % (', '.join('`%s`' % key for key in retired),
                   'is' if len(retired) == 1 else 'are',
                   '; '.join('%s -> %s' % (key, RETIRE_DYNAMICS_KEYS[key])
                             for key in retired)))
        return data


class Properties(BaseModel):
    """Container for computed properties from quantum chemistry calculations.
    (Skill-related: managed by individual skills)
    """

    # Energy
    energy: List[float] = [0.0]                # (nstate,) Total energies
    kinetic_energy: float = 0
    pot_energy: List[float] = [0.0]            # (nstate,) Potential energies
    total_energy: List[float] = [0.0]          # (nstate,) Total energies

    # Final structure reported by skill internal iteration (optimize/md/...)
    fin_coord: List[List[float]] = []            # (natom, 3) Angstrom; empty = no internal iteration

    # Final velocities reported by skill internal MD propagation
    fin_velocity: List[List[float]] = []         # (natom, 3) Bohr/au; empty = no velocity output

    # Final simulation cell reported by a skill whose engine owns the cell
    # degrees of freedom (variable-cell NPT, cell optimization). Same shape
    # as system.cell so the value needs no interpretation: [a, b, c, alpha,
    # beta, gamma] in Angstrom and degrees; empty = the engine kept the cell
    # it was given.
    fin_cell: List[float] = []

    # Gradient and Hessian
    gradient: List[List[List[float]]] = [[[0.0, 0.0, 0.0]]]          # (nstate, natom, 3)
    # Whole-system stress/virial (dev/methods/ENSEMBLE_DEVELOPMENT.md
    # section 8) consumed by the NPT barostats, in the generic
    # (nstate, 3, 3) format (mirrors gradient's (nstate, natom, 3)) so a
    # future excited-state stress fits; the active state's 3x3 block is
    # projected by PyRAIState.active_stress().
    # The cp2k / xtb / lammps backends populate it (ground state; under a
    # cp2k TDDFT run the skill's external `num` strain route fills the
    # state-resolved pages, refusing only a degenerate spectrum) in their
    # native unit, and the calculator converts it centrally to
    # Hartree/Bohr^3 (convert_stress_to_atomic, driven by the provider
    # skill's manifest units.stress) before set_result. A backend that
    # does not emit the key leaves it absent, so the read path treats an
    # unfilled stress as unavailable (active_stress() returns None and
    # the barostat raises) rather than as a physical zero. When filled
    # it is the FULL tensor (not a scalar pressure) in Hartree/Bohr^3,
    # and only the whole-system provider task is required to supply it
    # (single center: theory[0]; multiscale / multi-center: theory[-1],
    # see resolve_system_task).
    stress_tensor: List[List[List[float]]] = [[[0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0]]]    # (nstate, 3, 3)
    hessian: List[List[List[float]]] = [[[0.0]]]                     # (nstate, 3*natom, 3*natom)
    # The mobile atoms of a partial Hessian, 0-based as every internal index
    # is; an empty list is the whole structure. The matrix keeps its full
    # (nstate, 3N, 3N) shape with held rows and columns at zero, so a reader
    # never has to know which case it is looking at.
    hessian_atoms: List[int] = []
    frequencies: List[List[float]] = [[0.0]]                         # (nstate, 3*natom)
    norm_mode: List[List[List[List[float]]]] = [[[[0.0, 0.0, 0.0]]]] # (nstate, 3*natom, natom, 3)
    reduced_mass: List[List[float]] = [[0.0]]                        # (nstate, 3*natom)
    rotation_constants: List[List[float]] = [[0.0]]                  # (nstate, 3)

    # Thermodynamic properties
    heat_capacity_cv: List[float] = [0.0]
    heat_capacity_cp: List[float] = [0.0]
    zero_point_energy: List[float] = [0.0]
    internal_energy: List[float] = [0.0]
    enthalpy: List[float] = [0.0]
    entropy: List[float] = [0.0]
    free_energy: List[float] = [0.0]

    # Dipole moments
    atom_dipole: List[List[List[float]]] = [[[0.0, 0.0, 0.0]]]
    dipole: List[List[float]] = [[0.0, 0.0, 0.0]]
    computed_charges: List[List[float]] = []            # (nstate, natom) atom charges computed by skill, extracted to update background charges

    # Transition properties
    atom_trans_dipole: List[List[List[float]]] = [[[0.0, 0.0, 0.0]]]
    transition_dipole: List[List[float]] = [[0.0, 0.0, 0.0]]
    osc_strength: List[float] = [0.0]
    excitation: List[float] = [0.0]

    # Nonadiabatic coupling and spin-orbit coupling
    nac: List[List[List[float]]] = [[[0.0, 0.0, 0.0]]]  # (nnac, natom, 3)
    soc: List[float] = [0.0]                            # (nsoc,) aligned with soc_pair_map

    # Orbital and density data
    mo: str = ''
    mo1: str = ''
    mo2: str = ''
    esp: str = ''
    nto_molden: List[str] = ['']
    nto_cube: List[List[dict]] = [[{}]]
    electronic_density: str = ''

    # Spectroscopy
    ir_width: float = DEFAULT['ir_width']
    ir_scale: float = DEFAULT['ir_scale']
    ir_intensity: List[float] = [0.0]
    ir_spectrum: str = ''
    ir_data: List[List[float]] = [[0, 0.0]]
    uv_width: float = DEFAULT['uv_width']
    uv_spectrum: str = ''
    uv_data: List[List[float]] = [[0, 0.0]]

    # NICS
    nics_ring: List[int] = DEFAULT['nics_ring']
    nics_scan: List[float] = DEFAULT['nics_scan']
    nics: List[List[float]] = [[0.0]]
    nics_zz: List[List[float]] = [[0.0]]
    shielding_tensors: List[List[List[List[float]]]] = [[[[0.0, 0.0, 0.0],
                                                          [0.0, 0.0, 0.0],
                                                          [0.0, 0.0, 0.0]]]] # (ns, na, 3, 3)
    shielding_constants: List[List[float]] = [[0.0]]


class Calculation(BaseModel):
    """Calculation parameters for quantum chemistry computations.
    (Skill-related: managed by individual skills)
    """

    project: str = DEFAULT['project']
    task: str = DEFAULT['task']
    profile: str = DEFAULT['profile']
    method: str = ''
    functional: str = DEFAULT['functional']
    basis: str = DEFAULT['basis']
    scratch_path: str = ''
    chk_file: str = DEFAULT['chk_file']
    save_mo: bool = DEFAULT['save_mo']
    save_norm_mode: bool = DEFAULT['save_norm_mode']

    target_states: List[int] = DEFAULT['target_states']
    target_nac_pair: List[List[int]] = DEFAULT['target_nac_pair']
    target_soc_pair: List[List[int]] = DEFAULT['target_soc_pair']
    maxit_scf: int = DEFAULT['maxit_scf']
    maxit_opt: int = DEFAULT['maxit_opt']
    maxit_md: int = DEFAULT['maxit_md']

    density_fitting: bool = DEFAULT['density_fitting']
    non_local_correction: bool = DEFAULT['non_local_correction']
    dispersion_correction: str = DEFAULT['dispersion_correction']
    grid_level: int = DEFAULT['grid_level']

    # Convergence criteria
    scf_converge: float = DEFAULT['scf_converge']
    td_converge: float = DEFAULT['td_converge']
    opt_converge: bool = DEFAULT['opt_converge']
    opt_energy: float = DEFAULT['opt_energy']
    opt_grms: float = DEFAULT['opt_grms']
    opt_gmax: float = DEFAULT['opt_gmax']
    opt_drms: float = DEFAULT['opt_drms']
    opt_dmax: float = DEFAULT['opt_dmax']
    opt_csrms: float = DEFAULT['opt_csrms']
    opt_csmax: float = DEFAULT['opt_csmax']
    opt_cerms: float = DEFAULT['opt_cerms']
    opt_cemax: float = DEFAULT['opt_cemax']
    opt_pressure: float = DEFAULT['opt_pressure']
    opt_cell_coupling: str = DEFAULT['opt_cell_coupling']
    opt_trust_radius: Optional[float] = DEFAULT['opt_trust_radius']
    opt_sp: bool = DEFAULT['opt_sp']
    opt_freq: bool = DEFAULT['opt_freq']

    verbose: int = DEFAULT['verbose']
    comment: str = ''


# Retired keys of the framework `&environment` section. They are refused
# on the input rather than absorbed, because all three read as a setting
# while nothing consumed them.
#
# `execution_mode` is a SKILL keyword: skills/<skill>/service/runner
# selects run_local or run_slurm from it, and every skill declares it in
# its own schema. The framework spells its own decision in
# multitask/remotetask/multipool, and planning.assembly.build_task_queue hands
# the resulting mode down through the task's skill config (#300), so a
# framework-level copy would be a second, unread source of one decision.
RETIRE_ENVIRONMENT_KEYS = {
    'execution_mode': ('write it in the skill section (`&pyscf`/`&lammps`'
                       '/...), or leave it to the framework: '
                       '`remotetask` makes it submit and the skill '
                       'receives execution_mode=local'),
    'workflow_execution': ('the framework dispatch is spelled '
                           '`multitask`/`remotetask`/`multipool` on the '
                           'workflow or skill section'),
    'workflow_workers': ('write `multipool <n>` on the workflow section; '
                         'the Scheduler sizes its pool from that'),
    'num_thread': ('threads are a property of the calculation, so they are '
                   'set in the skill section as `<skill>_thread` (e.g. '
                   '`lammps_thread`), next to `<skill>_nproc` for MPI ranks; '
                   'the framework keeps no run-wide thread setting (#306)'),
}


class Environment(BaseModel):
    """Computational environment settings of the framework run.

    The merged slurm/export sections and the shell commands travel with
    PyRAIState (state.pmd.environment) so the dispatch subsystem reads
    one single data source without extra environment variables:
    workflow/dispatch/slurm_run.py builds the child runtime submission
    script from this section.

    How the run is distributed is NOT declared here. The dispatch
    keywords multitask/remotetask/multipool belong to the molecule
    section and to each workflow/skill section, the skill-side
    execution_mode is carried by the skill's own config, and so are the
    compute-resource counts -- `<skill>_thread` and `<skill>_nproc` in the
    skill section (see RETIRE_ENVIRONMENT_KEYS, which now also refuses the
    old run-wide `num_thread`).
    """

    shell: str = DEFAULT['shell']
    environment: Dict[str, str] = Field(default_factory=dict)
    program: str = DEFAULT['program']
    version: str = DEFAULT['version']
    device: str = DEFAULT['device']
    program_path: str = DEFAULT['program_path']
    scratch_path: str = DEFAULT['scratch_path']
    slurm_time: str = DEFAULT['slurm_time']
    slurm_partition: str = DEFAULT['slurm_partition']
    slurm_ntasks: int = DEFAULT['slurm_ntasks']
    num_gpu: int = DEFAULT['num_gpu']
    slurm_memory: int = DEFAULT['slurm_memory']
    slurm_command: str = DEFAULT['slurm_command']
    # Merged slurm sbatch section carried through the state
    slurm: Dict[str, Any] = Field(default_factory=dict)


class Optimize(BaseModel):
    """Geometry optimization parameters.

    Framework-level section aligned with the optimize workflow name
    (&optimize). Contains the optimization method selector and
    convergence thresholds consumed by the Optimize workflow and
    the framework optimizers (minima, gdiis, etc.).

    Field names use the 'opt_' prefix to avoid collisions with
    skill sections during flat-section flattening (e.g. both
    optimize and pyscf define 'method'). Dispatch-layer keywords
    (multipool) keep their canonical names.
    """

    # Multitask worker pool size of this workflow's dispatch fan-out
    # (1 keeps serial execution).
    multipool: int = Field(default=DEFAULT['multipool'], ge=1)

    opt_method: str = 'minima'         # framework optimization algorithm
                                       # (minima, gdiis, ...); the execution
                                       # driver is control.driver
    opt_maxit: int = 100               # max optimization iterations
    opt_energy: float = 1e-6           # energy convergence
    opt_grms: float = 3e-4             # gradient RMS convergence
    opt_gmax: float = 4.5e-4           # gradient max convergence
    opt_drms: float = 1.2e-3           # displacement RMS convergence
    opt_dmax: float = 1.8e-3           # displacement max convergence
    # The cell half of `optimize_cell`, judged in bar and in strain rather than
    # in the atomic units above: a lattice entry and a nuclear coordinate are both
    # lengths, but the same force reading asks a different pressure of a big box
    # than of a small one. Taken from DEFAULT so the two models cannot drift.
    opt_csrms: float = DEFAULT['opt_csrms']
    opt_csmax: float = DEFAULT['opt_csmax']
    opt_cerms: float = DEFAULT['opt_cerms']
    opt_cemax: float = DEFAULT['opt_cemax']
    # The external pressure the cell half is judged against, in bar, and which
    # lattice components it may move. Both are read only while `optimize_cell` is
    # on, and the validator below refuses them otherwise rather than letting a run
    # carry a pressure it cannot apply.
    opt_pressure: float = DEFAULT['opt_pressure']
    opt_cell_coupling: str = DEFAULT['opt_cell_coupling']
    # The step bound of the search, in Bohr, and the reason it is a keyword: the
    # trust radius clips the joint nuclear-plus-lattice step
    # (`workflow/optimizer/base.py::_apply_trust_radius`), so the bar band a
    # pressurised box is judged against is reachable or not reachable depending
    # on this number, not on `maxit`. Unset leaves each algorithm on its own
    # starting radius (0.3 for `minima`/`gdiis`, 0.05 for the TS search), which
    # is what every shipped example currently runs; a value of 0 would freeze
    # the search, so the field refuses it rather than reporting a run that moved
    # nothing.
    opt_trust_radius: Optional[float] = Field(
        default=DEFAULT['opt_trust_radius'], gt=0)
    opt_use_rfo: bool = False          # use RFO step instead of BFGS Newton step
    # Target state to optimize (0-based, 0 = S0). A multi-center search names
    # one index PER CENTER - each center answers on its own ladder - so the
    # field takes a list as well as a scalar; a scalar is read by every center,
    # which is what a single-center run and a broadcast multi-center run mean.
    opt_state_i: Union[int, List[int]] = 0
    # The second surface of a pair, for `meci`/`mecp` only. Unset means there is
    # no second surface - it reads as `opt_state_i`, not as state 0 - so an
    # excited-state optimization that names one state does not appear to have
    # named a pair (`data/state.py::optimize_params` resolves it). Shaped like
    # `opt_state_i`: scalar or one index per center.
    opt_state_j: Optional[Union[int, List[int]]] = None
    # Cell optimization switch: when true the planning layer derives the
    # 'stress' capability so the whole-system stress tensor is computed
    # and passed to the skill config (framework-central unit conversion
    # to Hartree/Bohr^3), and the framework optimizer carries the six
    # movable lattice components as part of the optimized vector, driven
    # by the stress conjugate to them. A backend without a stress
    # capability is refused by the optimizer rather than relaxed in the
    # box it was given.
    optimize_cell: bool = False

    # --- The path family (`opt_method` mep / irc / neb, `workflow/path.py`) ---
    # Read only when `opt_method` names a path search; the validator below
    # refuses each group on a method that cannot use it, the way `opt_pressure`
    # is refused without `optimize_cell`. `opt_path_radius` is > 0 because a zero
    # radius is a walk that takes no step, and `opt_neb_images` counts the
    # *interior* images - the band carries `opt_neb_images + 2` frames, which is
    # also what keeps every band dispatch above the single-unit identity
    # allocation.
    opt_path_constraint: str = DEFAULT['opt_path_constraint']
    opt_path_radius: float = Field(default=DEFAULT['opt_path_radius'], gt=0)
    opt_path_inner_maxit: int = Field(
        default=DEFAULT['opt_path_inner_maxit'], ge=1)
    opt_path_inner_tol: float = Field(
        default=DEFAULT['opt_path_inner_tol'], gt=0)
    # Which side of the saddle an IRC walks. `both` runs one walker per side
    # from the one transition state.
    opt_irc_direction: str = DEFAULT['opt_irc_direction']
    opt_irc_mode: str = DEFAULT['opt_irc_mode']
    opt_irc_mode_file: str = DEFAULT['opt_irc_mode_file']
    # The band's two endpoints, and an optional intermediate guess. The files are
    # read when the band is built (`workflow/path.py::build_band`), which is also
    # where the atom order and - for a variable-cell run - the lattice comment
    # are checked, because a section validator has no file to look at.
    opt_neb_reactant: str = DEFAULT['opt_neb_reactant']
    opt_neb_product: str = DEFAULT['opt_neb_product']
    opt_neb_ts: str = DEFAULT['opt_neb_ts']
    opt_neb_images: int = Field(default=DEFAULT['opt_neb_images'], ge=1)
    opt_neb_interpolate: str = DEFAULT['opt_neb_interpolate']
    opt_neb_geodesic_threshold: float = Field(
        default=DEFAULT['opt_neb_geodesic_threshold'], gt=0)
    opt_neb_geodesic_scaler: float = Field(
        default=DEFAULT['opt_neb_geodesic_scaler'], gt=0)
    opt_neb_geodesic_min_neighbors: int = Field(
        default=DEFAULT['opt_neb_geodesic_min_neighbors'], ge=1)
    # The spring constant is in Hartree per Bohr^2, so that k times a segment
    # length in Bohr is the same Hartree per Bohr the true force is read in.
    opt_neb_spring: float = Field(default=DEFAULT['opt_neb_spring'], ge=0)
    opt_neb_climb: bool = DEFAULT['opt_neb_climb']
    opt_neb_climb_threshold: float = Field(
        default=DEFAULT['opt_neb_climb_threshold'], gt=0)
    opt_neb_step: float = Field(default=DEFAULT['opt_neb_step'], gt=0)

    @model_validator(mode='after')
    def _check_path_options(self):
        """Reject the path keys a method cannot read, and the pairings with no
        source behind them.

        The path family shares this section with the single-structure searches,
        so each of its keywords answers a question exactly one of the three walks
        asks: a constraint algorithm, an arc length and a probe bound belong to
        the hypersphere (mep/irc), an imaginary mode to the IRC, and endpoints,
        images, a spring and a timestep to the band. A key declared on a method
        that cannot read it is refused where the input is still cheap to change,
        rather than carried to an optimizer that would never look at it - the
        same reason `opt_pressure` is refused without `optimize_cell`.
        """
        def declared(key):
            return getattr(self, key) != DEFAULT[key]

        for key, allowed in (('opt_path_constraint', PATH_CONSTRAINT_MODELS),
                             ('opt_irc_direction', IRC_DIRECTIONS),
                             ('opt_irc_mode', IRC_MODE_SOURCES),
                             ('opt_neb_interpolate', PATH_INTERPOLATIONS)):
            value = str(getattr(self, key) or '').strip().lower()
            if value not in allowed:
                raise ValueError(
                    'Unknown %s %r. Options: %s'
                    % (key, getattr(self, key), ', '.join(allowed)))
            setattr(self, key, value)

        method = str(self.opt_method or '').strip().lower()
        groups = (
            ('The hypersphere keywords', ('mep', 'irc'),
             ('opt_path_constraint', 'opt_path_radius',
              'opt_path_inner_maxit', 'opt_path_inner_tol'),
             'they describe the constrained step a path point is found with'),
            ('The IRC keywords', ('irc',),
             ('opt_irc_direction', 'opt_irc_mode', 'opt_irc_mode_file'),
             'the imaginary mode of a transition state is what an intrinsic '
             'reaction coordinate starts on'),
            ('The band keywords', ('neb',),
             ('opt_neb_reactant', 'opt_neb_product', 'opt_neb_ts',
              'opt_neb_images', 'opt_neb_interpolate',
              'opt_neb_geodesic_threshold', 'opt_neb_geodesic_scaler',
              'opt_neb_geodesic_min_neighbors', 'opt_neb_spring',
              'opt_neb_climb', 'opt_neb_climb_threshold', 'opt_neb_step'),
             'a band is a chain of images interpolated between two endpoints, '
             'which no other search has'),
        )
        for label, wanted, keys, why in groups:
            if method in wanted:
                continue
            named = [key for key in keys if declared(key)]
            if named:
                raise ValueError(
                    '%s %s need optimize.method %s, and this run asks for %r: %s'
                    % (label, ', '.join(named), ' or '.join(wanted),
                       self.opt_method, why))

        if method == 'neb':
            if not (str(self.opt_neb_reactant).strip()
                    and str(self.opt_neb_product).strip()):
                raise ValueError(
                    'optimize.method neb needs both endpoints: the band is '
                    'interpolated between them, so a run that names no '
                    'neb_reactant/neb_product has nothing to relax')
            if self.opt_neb_interpolate == 'geodesic' and self.opt_neb_images < 2:
                raise ValueError(
                    'neb_interpolate geodesic needs neb_images >= 2: the '
                    'geodesic metric smooths the interior of the band, and one '
                    'interior image is a chord with no path between it and '
                    'anything to smooth')
            if declared('opt_neb_climb_threshold') and not self.opt_neb_climb:
                raise ValueError(
                    'neb_climb_threshold is the perpendicular force that arms '
                    'the climbing image, and this run does not ask for one: set '
                    'neb_climb true, or leave the threshold at its default')
        if (method == 'irc' and self.opt_irc_mode == 'molden'
                and not str(self.opt_irc_mode_file).strip()):
            raise ValueError(
                "irc_mode 'molden' needs irc_mode_file: the imaginary mode has "
                'to be read from somewhere, and this run names no file')
        return self

    @model_validator(mode='after')
    def _check_cell_options(self):
        # The two cell options answer questions only a moving box asks.
        if (self.opt_pressure != DEFAULT['opt_pressure']
                or self.opt_cell_coupling != DEFAULT['opt_cell_coupling']):
            if not self.optimize_cell:
                raise ValueError(
                    'opt_pressure and opt_cell_coupling need optimize_cell: '
                    'with the box held there is no lattice derivative for a '
                    'pressure to act on. Ask for a cell relaxation, or use '
                    'jobtype md with a barostat for a box under pressure')
        coupling = str(self.opt_cell_coupling).strip().lower()
        if coupling not in CELL_COUPLING_MODES:
            raise ValueError(
                'Unknown opt_cell_coupling %r. Options: %s'
                % (self.opt_cell_coupling, ', '.join(CELL_COUPLING_MODES)))
        self.opt_cell_coupling = coupling
        return self


# The lattice components one pressure value is allowed to move
# (`optimize.opt_cell_coupling`). The names are `dynamics.barostat_mode`'s, and so
# is the difference between them: isotropic scales the box, anisotropic lets each
# of the six relaxed components answer for itself.
CELL_COUPLING_MODES = ('isotropic', 'anisotropic')


# The searches that walk several structures. They are values of `optimize.method`
# like any other, but a path is not one geometry's next geometry: `mep` and `irc`
# integrate a sequence of constrained points and `neb` relaxes a whole band of
# images, so `workflow.resolve_workflow` routes these three to the Path workflow
# instead of the single-structure loop, and the optimizer registry's
# `implemented_optimizers()` is what the input layer checks the name against.
PATH_METHODS = ('mep', 'irc', 'neb')


# Which algorithm holds a path point on the hypersphere
# (`optimize.path_constraint`, mep/irc only): the classic Gonzalez-Schlegel
# mass-weighted constrained step, or the ReLU-gated constraint potential.
PATH_CONSTRAINT_MODELS = ('sphere', 'relu')


# Which side of a transition state an IRC walks (`optimize.irc_direction`).
# `both` runs one walker per side from the one saddle.
IRC_DIRECTIONS = ('backward', 'forward', 'both')


# Where the imaginary mode that seeds an IRC comes from
# (`optimize.irc_mode`): the framework's own Hessian computation, or a vibration
# file that already holds it.
IRC_MODE_SOURCES = ('hessian', 'molden')


# How a band's interior images are placed (`optimize.neb_interpolate`).
PATH_INTERPOLATIONS = ('linear', 'geodesic')


# Who produces the force constants of a Hessian request (`&hessian
# hess_driver`). The two names are the ones `control.driver` already uses:
# 'main' is the framework's own displacement batch, 'skill' is the matrix the
# backend computes for itself.
HESS_DRIVERS = ('main', 'skill')


def resolve_hessian_route(tokens, hessian_params=None):
    """Resolve the Hessian mode and producer from the input.

    One definition shared by planning and ``workflow.hessian`` so the
    multi-backend gate and the workflow cannot disagree about who owns
    the matrix. Order: an explicit ``hess_mode`` / ``hess_driver``
    wins; otherwise the jobtype token names the route (``num_hess`` /
    ``num_freq`` -> numerical -> main, ``hessian`` / ``frequency`` ->
    auto -> skill). An explicit ``hess_driver: main`` with no
    ``hess_mode`` resolves the mode to ``numerical``: the framework has
    no analytic second derivative, so main has exactly one producer.

    Parameters
    ----------
    tokens : sequence of str
        Canonical jobtype tokens.
    hessian_params : dict or None
        ``&hessian`` section values (hess_mode / hess_driver).

    Returns
    -------
    tuple of (str, str)
        (mode, driver). Mode is 'analytic' / 'numerical' / 'auto' /
        None; driver is 'main' or 'skill'.
    """
    params = dict(hessian_params or {})
    tokens = [str(token) for token in (tokens or [])]
    explicit_mode = params.get('hess_mode')
    mode = (str(explicit_mode).strip().lower()
            if explicit_mode is not None else None)
    if mode is None:
        if 'num_hess' in tokens or 'num_freq' in tokens:
            mode = 'numerical'
        elif 'hessian' in tokens or 'frequency' in tokens:
            mode = 'auto'
    driver = params.get('hess_driver')
    driver = (str(driver).strip().lower()
              if driver is not None else None)
    if driver is None:
        driver = 'main' if mode == 'numerical' else 'skill'
    if driver == 'main' and explicit_mode is None and mode != 'numerical':
        # main can only mean the displacement batch.
        mode = 'numerical'
    return mode, driver


class Hessian(BaseModel):
    """Numerical Hessian workflow parameters (&hessian).

    Dispatch-layer keywords of the hessian workflow node. The
    section travels nested through the adapter: its keyword names
    (multitask/remotetask/multipool) are shared by several workflow
    sections and cannot live in the flat namespace.
    """

    # Modularized execution keywords (tri-state: None keeps the
    # workflow class defaults during graph construction)
    multitask: Optional[bool] = None
    remotetask: Optional[bool] = None
    # Multitask worker pool size of this workflow's dispatch fan-out
    # (1 keeps serial execution).
    multipool: int = Field(default=DEFAULT['multipool'], ge=1)
    # Who produces the matrix. 'main' displaces the geometry and differences
    # the gradients the framework is already entitled to; 'skill' asks the
    # backend for the matrix it computes, and whether the backend answers with
    # an analytical second derivative or with displacements of its own is that
    # backend's keyword (ORCA's `num_freq`, for example, picks NumFreq over
    # AnFreq). None means 'the input said nothing', which lets the jobtype
    # token choose: `num_hess`/`num_freq` name the framework, `hessian`/
    # `frequency` name the backend. The same reason `hess_mode` below keeps
    # None as its default holds here: this section is always present with its
    # defaults by the time a workflow reads it, so a concrete default would
    # outrank the token on every run.
    hess_driver: Optional[str] = None
    # Which force constants to read: 'analytic' asks the backend for its own
    # matrix, 'numerical' displaces and differs here, 'auto' takes the
    # analytical one if the evaluation returned it. The jobtype tokens
    # (`hessian` / `num_hess`) supply the default; this overrides them.
    # None means 'the input said nothing', which lets the jobtype token
    # (hessian/num_hess/frequency/num_freq) choose the route. A concrete
    # default here would outrank the token on every run, since this section
    # is always present with its defaults by the time the workflow reads it.
    hess_mode: Optional[str] = None
    # Displacement of each mobile Cartesian coordinate, in Angstrom. It is a
    # distance while the gradients are Hartree per Bohr, and the difference
    # quotient carries that factor.
    hess_step: float = 0.005
    # Whether the matrix is read into frequencies and normal modes at all.
    hess_vibration: bool = True
    # Whether the modes are carried on into the thermochemical corrections.
    hess_thermo: bool = True
    # Write <title>.freq.molden, the record the initial-condition sampler
    # looks for by that exact name.
    hess_molden: bool = False
    # The mobile atoms of a partial Hessian, 1-based as every atom selection in
    # an input is; the atoms it does not name are held.
    hess_atoms: List[int] = Field(default_factory=list)
    hess_temperature: float = 298.15
    hess_pressure: float = 1.0
    hess_scale: float = 1.0
    hess_cutoff: float = 100.0
    hess_alpha: float = 4.0
    hess_symmetry: int = 1
    hess_linear: Optional[bool] = None

    @model_validator(mode='after')
    def _check_curvature_keys(self):
        """Reject the pairings of owner and method that have no source behind them.

        `hess_driver` answers who produces the force constants and `hess_mode`
        answers how they are produced. Two questions, but not every pair of
        answers exists: the framework has no analytical second derivative of its
        own, so naming it as the owner can only mean the displacement batch; and
        `auto` is the statement that no owner has been settled on yet (take the
        backend matrix if it arrives, displace if it does not), which cannot
        stand beside a named one. A `hess_mode` outside those three spellings is
        left alone: the vocabulary of the mode is not what this check is about.
        """
        if self.hess_driver is None:
            return self
        driver = str(self.hess_driver).strip().lower()
        if driver not in HESS_DRIVERS:
            raise ValueError(
                "unknown hess_driver %r; expected %s"
                % (self.hess_driver, ', '.join(HESS_DRIVERS)))
        mode = (str(self.hess_mode).strip().lower()
                if self.hess_mode is not None else None)
        if driver == 'main' and mode in ('analytic', 'auto'):
            raise ValueError(
                "hess_driver 'main' with hess_mode %r asks the framework for an "
                "analytical force constant, and it has none: it differentiates "
                "the gradients it is already given. Use hess_mode 'numerical' "
                "for the framework's own displacement batch, or hess_driver "
                "'skill' to have the backend produce the matrix"
                % self.hess_mode)
        self.hess_driver = driver
        return self


class Search(BaseModel):
    """Hyperparameter grid scan parameters (&search).

    The grid names the skill's own keywords and the values to try, so
    this section carries a mapping the framework never interprets: which
    keyword may be varied at all, and what a candidate means, is the
    skill's declaration (its manifest 'searchable' block, read through
    its query interface). The section travels nested through the
    adapter for the same reason the dispatch keywords do: its names
    belong to a skill's vocabulary and cannot live in the flat
    namespace.

    Legacy spellings map onto these fields: 'use_hpc 0' is multitask
    without remotetask, 'use_hpc 1' is both, 'retrieve 1' stays here, and
    'ml_ncpu' is the multipool of this node.
    """

    # Keyword name to its list of candidate values; the scan is the
    # Cartesian product of them, planned by workflow/grid/plan.py.
    grid: Dict[str, Any] = Field(default_factory=dict)
    # Read the folders a previous scan left instead of submitting
    # anything (the legacy retrieve mode): rank what is on disk.
    retrieve: int = 0
    # Channel the points are ranked on; empty takes the first channel
    # the trained profile reports, which the plan derives rather than
    # the user guessing it.
    sort_by: str = ''
    # 'max' ranks by the worst ensemble member, 'mean' by their average:
    # a folder is deployed as a whole, so its worst member is the surface
    # that gets served.
    sort_stat: str = 'max'
    # 'asc' puts the smallest error first.
    sort_order: str = 'asc'
    # Root directory of the scan: one folder per point under it, named
    # by the point's values so a restart re-derives them.
    grid_root: str = './grid-search'
    # Refuse a scan larger than this many points (0 states no limit):
    # a Cartesian product grows quietly, and the first one is rarely the
    # last one a user runs.
    max_grid: int = 0

    # Modularized execution keywords (tri-state: None keeps the
    # workflow class defaults during graph construction)
    multitask: Optional[bool] = None
    remotetask: Optional[bool] = None
    # Multitask worker pool size of this workflow's dispatch fan-out
    # (1 keeps serial execution).
    multipool: int = Field(default=DEFAULT['multipool'], ge=1)


class Train(BaseModel):
    """Model-training workflow parameters (&train).

    Dispatch-layer keywords of the training node. The section travels
    nested through the adapter: its keyword names are shared by the other
    workflow sections and cannot live in the flat namespace.

    A training run owns no calculation keyword of its own -- the data
    file, the hyperparameters and the continuation switch all belong to
    the skill section -- so this section carries how the one call is
    executed and the one preprocessing step the framework performs on
    its input. It exists at all because the fit is the longest call in
    the package: `remotetask` here is what hands a training to the
    cluster.

    `permutation_map` names a legacy one-based atom-order file. The
    expansion happens in the train workflow before the skill backend is
    initialized, so the skill only ever sees the resulting frames; an
    empty value or 'No' disables it.
    """

    # Modularized execution keywords (tri-state: None keeps the
    # workflow class defaults during graph construction)
    multitask: Optional[bool] = None
    remotetask: Optional[bool] = None
    # Multitask worker pool size of this workflow's dispatch fan-out
    # (1 keeps serial execution).
    multipool: int = Field(default=DEFAULT['multipool'], ge=1)
    # Legacy atom-order file applied to the training set before the
    # skill backend is initialized (workflow/train.py).
    permutation_map: str = ''


class Prediction(BaseModel):
    """Test-set evaluation workflow parameters (&prediction).

    Dispatch-layer keywords of the prediction node, same shape as
    &train: the error statistics, the test file and the raw comparison
    files are all the skill's business, and this section only says how
    the call is executed. The section travels nested through the adapter
    because its keyword names collide with the other workflow sections.
    """

    # Modularized execution keywords (tri-state: None keeps the
    # workflow class defaults during graph construction)
    multitask: Optional[bool] = None
    remotetask: Optional[bool] = None
    # Multitask worker pool size of this workflow's dispatch fan-out
    # (1 keeps serial execution).
    multipool: int = Field(default=DEFAULT['multipool'], ge=1)


class Read(BaseModel):
    """Read-only extraction parameters (&read).

    The section names the finished calculation artifacts to read and
    which labels the skill parser should ask for. It never starts an
    engine: `jobtype read` returns the extracted frames to its caller
    and writes no data artifact. The section travels nested through the
    adapter because its label names (`energy`, `gradient`, ...) collide
    with skill keyword names.

    Parameters
    ----------
    path : str or list of str
        A calculation directory, a calculation output file, or a list
        file holding one path per line.
    project : str
        Output file stem. Empty means the directory basename (or the
        file stem when `path` names a file).
    profile : str
        Skill profile used to parse iterative artifacts (MD/optimize).
        Empty derives the profile from the requested labels.
    energy, gradient, nac, soc : bool
        Labels to ask the parser for.
    """

    path: Union[str, List[str]] = ''
    project: str = ''
    profile: str = ''
    energy: bool = True
    gradient: bool = True
    nac: bool = False
    soc: bool = False


# Field-to-submodel mapping for adapter partitioning
# Maps each Config field name to its parent submodel name.
# Hessian/Search/Train/Prediction stay out of the flat mapping: their dispatch-layer
# keyword names collide with the other workflow sections, so they
# are routed by section name instead (NESTED_FRAMEWORK_SECTIONS).
FIELD_TO_SUBMODEL = {}
_FIELD_SUBMODEL_CANDIDATES = {}
for _model_cls, _model_name in [(Control, 'control'), (System, 'system'),
                                 (Dynamics, 'dynamics'), (Adaptive, 'adaptive'),
                                 (Optimize, 'optimize'), (Composite, 'composite')]:
    for _field_name in _model_cls.model_fields:
        FIELD_TO_SUBMODEL[_field_name] = _model_name
        _FIELD_SUBMODEL_CANDIDATES.setdefault(_field_name, []).append(
            _model_name)

# Keys several framework sections spell with the same name (the dispatch
# keywords). The flat namespace cannot say which section the user meant,
# and "last model wins" silently sends `multipool` to optimize even when
# it was written on `&md`. Nested sections route by section name and are
# unaffected; a bare top-level spelling is refused by the adapter.
AMBIGUOUS_FIELD_NAMES = {
    name: tuple(names)
    for name, names in _FIELD_SUBMODEL_CANDIDATES.items()
    if len(names) > 1
}

SUBMODEL_CLASSES = {
    'control': Control,
    'system': System,
    'dynamics': Dynamics,
    'adaptive': Adaptive,
    'optimize': Optimize,
    'hessian': Hessian,
    'search': Search,
    'train': Train,
    'prediction': Prediction,
    'read': Read,
    'composite': Composite,
}

def flatten_config_dump(nested_dict):
    """Flatten a nested Config model_dump() into a flat dict.

    Converts {'control': {'title': ...}, 'system': {...}, ...} back to
    {'title': ..., 'atoms': ..., ...}. One implementation lives here
    because this is the table both transformations read: `normalize` and
    `validate` each carried a copy of it, and the two copies could drift
    while the tables they walk could not.

    Parameters
    ----------
    nested_dict : dict
        A Config.model_dump() mapping.

    Returns
    -------
    dict
        The flat field mapping.
    """
    flat = {}
    for submodel_data in nested_dict.values():
        if isinstance(submodel_data, dict):
            flat.update(submodel_data)
    return flat


def partition_to_submodels(flat_dict):
    """Partition a flat dict back into submodel-grouped dict for Config.

    Parameters
    ----------
    flat_dict : dict
        The flat field mapping.

    Returns
    -------
    dict
        {submodel name: {field: value}}, only for fields the mapping names.
    """
    submodel_data = {name: {} for name in SUBMODEL_CLASSES}
    for key, value in flat_dict.items():
        parent = FIELD_TO_SUBMODEL.get(key)
        if parent is not None:
            submodel_data[parent][key] = value
    return submodel_data


# Framework sections kept nested through the pipeline (input section
# name -> submodel name). These sections share the dispatch-layer
# keyword names (multitask/remotetask/multipool), which cannot share
# the flattened namespace without silently overwriting each other.
# The molecule section maps to the system submodel and carries the
# molecule-scope dispatch keywords (dev/MULTICENTER_DEVELOPMENT.md); it
# stays nested for the same collision reason.
# Nested sections that own one field of their own instead of merging their
# inner keys into the submodel namespace. The constraint block's inner words
# (groups, cvs, walls, fixes, constraints, metadynamics) are the
# schema's vocabulary, and `System` already uses `groups` and `constraints`
# for something else, so merging them would either collide or be dropped.
# Section name -> (submodel name, field name).
# `constraint` is the name the classic input writes; `system_constraint` is
# the retired spelling of the same block and keeps working, because an
# archived input must run rather than fail on a name.
NESTED_SECTION_FIELDS = {
    'constraint': ('system', 'constraints'),
    'system_constraint': ('system', 'constraints'),
}

NESTED_FRAMEWORK_SECTIONS = {
    'molecule': 'system',
    'md': 'dynamics',
    'adaptive': 'adaptive',
    'optimize': 'optimize',
    'hessian': 'hessian',
    'search': 'search',
    'train': 'train',
    'prediction': 'prediction',
    'read': 'read',
    'composite': 'composite',
}

# Submodels restored from the model dump whenever the config is rebuilt
# through the flat namespace (schema.normalize, schema.validate): the
# framework input sections above plus the environment snapshot.
#
# The snapshot is not an input section -- build_config freezes it from
# the merged pmd_env.yaml and the run input -- and FIELD_TO_SUBMODEL
# routes none of its keys, because names like shell/program/device/
# scratch_path share the flat namespace with the workflow sections (the
# same collision the dispatch keywords above are restored for). Left out
# of this list, the rebuild silently replaces every machine setting of
# the run with its default, and the dispatch layer -- which documents
# state.pmd.environment as its single data source -- generates child
# runtime scripts without shell commands, exports or #SBATCH lines.
RESTORED_NESTED_SECTIONS = tuple(
    sorted(set(NESTED_FRAMEWORK_SECTIONS.values()) | {'environment'}))


class PyRAIConfig(BaseModel):
    """Unified configuration model with nested submodel structure.

    Skill-independent parameters are organized into four submodels:
    - control: job control and resource allocation
    - system: atomic structure, electronic settings, constraints
    - dynamics: MD simulation settings
    - adaptive: adaptive sampling parameters

    Supports both nested access (config.control.title) and flat
    access (config.title) for convenience. Flat kwargs in __init__
    are automatically routed to the appropriate submodel.
    """
    control: Control = Field(default_factory=Control)
    system: System = Field(default_factory=System)
    dynamics: Dynamics = Field(default_factory=Dynamics)
    adaptive: Adaptive = Field(default_factory=Adaptive)
    optimize: Optimize = Field(default_factory=Optimize)
    hessian: Hessian = Field(default_factory=Hessian)
    search: Search = Field(default_factory=Search)
    train: Train = Field(default_factory=Train)
    prediction: Prediction = Field(default_factory=Prediction)
    read: Read = Field(default_factory=Read)
    composite: Composite = Field(default_factory=Composite)
    # Computational environment snapshot (dispatch policy, slurm,
    # export variables). Populated by the adapter from the merged
    # environment source; travels with PyRAIState so the dispatch
    # subsystem never reads the kernel Environment singleton.
    environment: Environment = Field(default_factory=Environment)
    # Calculation results written back by skills (skill schema) or by
    # the Calculator composite aggregation (framework schema).
    results: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        # Separate submodel kwargs from flat field kwargs.
        # Declared nested fields (control/system/dynamics/adaptive and
        # any skill-specific submodels added by subclasses) are passed
        # through; remaining flat kwargs are routed via FIELD_TO_SUBMODEL.
        declared_fields = set(type(self).model_fields.keys())
        submodel_kwargs = {}
        flat_kwargs = {}
        for k, v in data.items():
            if k in declared_fields:
                submodel_kwargs[k] = v
            else:
                flat_kwargs[k] = v

        if flat_kwargs:
            # Partition flat fields into submodel groups
            for field_name, value in flat_kwargs.items():
                parent = FIELD_TO_SUBMODEL.get(field_name)
                if parent is not None:
                    if parent not in submodel_kwargs:
                        # Start from current defaults
                        submodel_kwargs[parent] = {}
                    if isinstance(submodel_kwargs[parent], dict):
                        submodel_kwargs[parent][field_name] = value

        super().__init__(**submodel_kwargs)

    def __getattr__(self, name):
        # Flat field lookup across submodels
        if name.startswith('model_') or name.startswith('__'):
            raise AttributeError(name)
        parent = FIELD_TO_SUBMODEL.get(name)
        if parent is not None:
            try:
                submodel = object.__getattribute__(self, parent)
                return getattr(submodel, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


# Backward compatibility alias
Config = PyRAIConfig
