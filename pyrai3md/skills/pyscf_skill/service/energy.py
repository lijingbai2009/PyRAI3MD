"""
======================================================================
This is a standalone quantum chemistry calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 30, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provide the energy calculation functions for the PySCF skill,
which is always the first step for all other functions

"""

import os
import copy
import warnings
import numpy as np

# Suppress PySCF module testing warnings (not indicative of errors)
warnings.filterwarnings('ignore', message='Module .* is under testing')
warnings.filterwarnings('ignore', message='Module .* is not fully tested')

# PySCF is imported where it is used, not here: a skill has to be importable
# (registered, its schema and helpers usable) on a host whose engine package
# is missing or broken -- the 12:16 round lost the whole pyscf skill, and with
# it 15 framework tests, to one `from pyscf import gto` at module scope. See
# skills/lammps_skill/service/persistent/server.py::_import_native.

from ..schema.method import AVAILABLE_METHOD
from .utils import current_stdout

# RUNTIME is a global variable to store the runtime information of the calculation
# RUNTIME is only intialized in build_meanfild
RUNTIME = {
    'mol': None,
    'mf': None,
    'post': None,
    'results': {},
}

def energy(config) -> dict:
    # Perform the energy calculation and return the results
    """
    return a runtime dictionary, where the results including:
        'energy',
        'dipole',
        'computed_charges',
        'excitation',
        'osc_strength',
        'transition_dipole'
    """
    runtime = build_meanfield(config)
    runtime = get_energy(runtime, config)

    return runtime


def get_energy(runtime: dict, config) -> dict:
    mf = runtime['mf']

    mf.kernel() # run scf calculation

    if not mf.converged:
        raise ValueError("SCF calculation did not converge.")

    pt = run_post_calculation(config, mf)

    results = get_results(config, mf, pt)

    mo_path = save_mo(config, mf)

    results['mo'] = mo_path
    runtime['post'] = pt
    runtime['results'].update(results)

    return runtime


def build_meanfield(config):
    # build mean-field object based on the input configuration,
    atoms = config.atoms
    from pyscf import gto
    coordinates = config.coordinates
    basis = config.basis
    charges = config.charges
    multiplicity = config.multiplicity
    verbose = config.verbose

    # Determine log file path: stored in runtime for reference.
    # The trace itself goes through the per-thread stdout the local
    # runner bound below (service/utils.current_stdout), never through a
    # process-wide sys.stdout swap. Log name follows Calculation.project
    # for consistency with the other scratch files (chk, json, etc.).
    scratch = config.scratch_path or os.getcwd()
    scratch = os.path.expanduser(scratch)
    os.makedirs(scratch, exist_ok=True)
    log_file = os.path.join(scratch, f'{config.project}.log')

    # Create a PySCF molecule object
    mol = gto.Mole()
    mol.atom = [[sym, coord] for sym, coord in zip(atoms, coordinates)]
    mol.charge = charges
    mol.spin = multiplicity - 1  # spin is multiplicity - 1
    _basis = {x: basis if x != 'X' else gto.basis.load('minao', 'H') for x in list(set(atoms))} # use minao basis for nics dummy atom,
    mol.basis = _basis
    mol.set_common_origin((0,0,0))
    mol.verbose = verbose
    # PySCF writes all calculation output through ``mol.stdout``. Bind
    # this term's own stream: the local runner attaches a tee to the
    # thread that runs it, so a concurrent term writes into its own log
    # instead of this one's (#302).
    mol.stdout = current_stdout()
    mol.build()

    # Set up the mean-field calculation based on the specified method and functional
    mf = set_method(config, mol)
    mf = set_functional(config, mf)
    mf = set_solvent(config, mf)
    mf = set_checkpoint(config, mf)

    runtime = copy.deepcopy(RUNTIME)
    runtime.update({
        'mol': mol,
        'mf': mf,
        'post': None,
        'results': {'analysis': {}},
        'log_file': log_file,
    })


    return runtime


def set_method(config, mol):
    # Perform the SCF calculation based on the specified method and functional
    method = config.method
    multiplicity = config.multiplicity
    is_open_shell = multiplicity > 1

    # important logic: method has been checked in validation, so we are sure the reference exists
    closed, open_ = AVAILABLE_METHOD[method]['reference']
    return open_(mol) if is_open_shell else closed(mol)


def set_functional(config, mf):
    # Set the functional for DFT calculations
    method = config.method
    functional = config.functional
    density_fitting = config.density_fitting
    non_local_correction = config.non_local_correction
    dispersion_correction = config.dispersion_correction
    scf_converge = config.scf_converge
    maxit_scf = config.maxit_scf

    if AVAILABLE_METHOD[method]['functional']:
        mf.xc = functional

    # PySCF does not support nuclear gradients for DF-TDDFT/TDA.
    # Automatically disable density fitting for TDDFT/TDA gradient profiles.
    profile = config.profile
    if density_fitting and method in ('tddft', 'tda') and profile.endswith('_gradient'):
        density_fitting = False

    if density_fitting:
        mf = mf.density_fit()

    if not non_local_correction:
        mf.nlc = False

    if len(dispersion_correction) > 0:
        mf.disp = dispersion_correction

    mf.max_cycle = maxit_scf
    mf.conv_tol = scf_converge

    return mf


def set_solvent(config, mf):
    # Set the solvent model for calculations.
    # Only real solvents trigger PCM; gas-phase keywords do not.
    solvent = config.solvent

    gas_phase = {'', 'vacuum', 'none', 'gas'}
    if solvent and solvent.lower() not in gas_phase:
        mf = mf.PCM()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.solvent = solvent

    return mf


def set_checkpoint(config, mf):
    project = config.project
    profile = config.profile
    method = config.method
    functional = config.functional
    basis = config.basis
    chk_file = config.chk_file
    scratch_path = config.scratch_path or os.getcwd()
    default_chk_file = os.path.join(scratch_path, f"{project}_{profile}_{method}_{functional}_{basis}.chk")

    if chk_file:
        chk_file = os.path.expanduser(chk_file)
    else:
        chk_file = default_chk_file

    mf.chkfile = chk_file

    if os.path.exists(chk_file):
        mf.init_guess = 'chkfile'

    return mf


def run_post_calculation(config, mf):
    # run post-method calculation if specified by the method, e.g. MP2 or TDDFT
    post_method = AVAILABLE_METHOD[config.method]['post_method']

    if post_method == 'mp2':
        return run_pt2(config, mf)

    elif post_method == 'tddft':
        return run_td(config, mf)

    else:
        return None


def run_pt2(config, mf):
    # compute mp2 correction for ground state
    pt = None
    method = config.method
    multiplicity = config.multiplicity

    closed, open_ = AVAILABLE_METHOD[method]['mp2']

    if multiplicity == 1:
        pt = closed(mf)
    else:
        pt = open_(mf)

    pt.scs_type = 'SCS'
    pt.kernel()

    return (pt, None)


def run_td(config, mf) -> tuple:
    # state_spin is already normalized to be either a list of same spins or combined singlet/triplet spins
    # so we don't need to worry about the first spin state and always assume triplet in the second spin state in td
    method = config.method
    tddft = AVAILABLE_METHOD[method]['tddft']
    nstate = config.nstate
    state_spin = config.state_spin
    multiplicity = config.multiplicity
    td_converge = config.td_converge

    if not tddft:
        return (None, None)

    if nstate < 2:
        return (None, None)

    # count number of triplet in spin state 2, the rest number is the remaining states with reference
    nspin_2 = state_spin.count(2)
    nspin_1 = nstate - nspin_2 - 1

    if nspin_1 > 0:
        td_1 = tddft(mf)
        td_1.nstates = nspin_1
        if multiplicity == 1: td_1.singlet = True # only works for singlet
        td_1.conv_tol = td_converge
        td_1.kernel()
    else:
        td_1 = None

    if nspin_2 > 0:
        td_2 = tddft(mf)
        td_2.nstates = nspin_2
        td_2.singlet = False # only works for triplet
        td_2.conv_tol = td_converge
        td_2.kernel()
        td_2.analyze()
    else:
        td_2 = None

    td = (td_1, td_2)

    return td


def get_results(config, mf, pt) -> dict:
    # extract results if applicable for different methods, and convert them to np.array for downstream tasks
    post_method = AVAILABLE_METHOD[config.method]['post_method']
    results = get_scf_results(mf)

    if post_method == 'mp2':
        return get_mp2_results(mf, pt)

    elif post_method == 'tddft':
        return get_td_results(mf, pt, config)

    else:
        return results


def get_scf_results(mf) -> dict:
    # Extract the computed properties from the SCF calculations
    energy = mf.e_tot
    pop_chg, dipole = mf.analyze()
    results = {
        'energy': np.array([energy]),
        'computed_charges': np.array([pop_chg[1]]),
        'dipole': np.array([dipole]),
        'excitation': np.array([0]),
        'osc_strength': np.array([0.0]),
        'transition_dipole': np.array([[0.0, 0.0, 0.0]]),
    }

    return results


def get_mp2_results(mf, pt) -> dict:
    # Extract the computed properties from the MP2 calculations
    # important logic: this method will only be called when mp2 is not None, which should always return non-None pt object
    mp = pt[0]
    energy = mp.e_tot
    # dipole = mf.dip_moment(dm=dm_re) * 0.393430 # convert Debye to a.u. <-- MP2 unrelax dipole failed in benzene test, not use so far
    pop_chg, dipole = mf.analyze()
    # convert all results to np.array for downstream tasks
    # all np.array will be converted to list before saving to json for AI agent to read
    results = {
        'energy': np.array([energy]),
        'computed_charges': np.array([pop_chg[1]]),
        'dipole': np.array([dipole]),
        'excitation': np.array([0]),
        'osc_strength': np.array([0.0]),
        'transition_dipole': np.array([[0.0, 0.0, 0.0]]),
    }

    return results


def get_td_results(mf, td, config) -> dict:
    # Extract the computed properties from the TDDFT calculations
    # important logic: this method will only be called when response is not None, which should always return non-None td object

    energy = [mf.e_tot]
    pop_chg, dip = mf.analyze()
    computed_charges = [pop_chg[1]]
    dipole = [dip]
    excitation = [0]

    # State-ordered transition data: index i stores the ground-to-state i
    # transition dipole and oscillator strength (entry 0 is zero).
    td_dipole = [[0.0, 0.0, 0.0]]
    td_osc = [0.0]

    t_1, t_2 = td[0:2]

    if t_1 is not None:
        energy += [x + energy[0] for x in t_1.e]
        computed_charges += [x for x in get_td_point_charges(mf, t_1)]
        dipole += [x for x in get_td_dipole(t_1)]
        excitation += [x for x in t_1.e]
        td_osc += [x for x in t_1.oscillator_strength()]
        td_dipole += [x for x in t_1.transition_dipole()]

    if t_2 is not None:
        energy += [x + energy[0] for x in t_2.e]
        computed_charges += [x for x in get_td_point_charges(mf, t_2)]
        dipole += [x for x in get_td_dipole(t_2)]
        excitation += [x for x in t_2.e]
        td_osc += [x for x in t_2.oscillator_strength()]
        td_dipole += [x for x in t_2.transition_dipole()]

    # Extract transition properties in transition_pair order (npair, 3).
    # Only ground-to-state pairs carry computed values; pyscf does not
    # compute excited-state to excited-state transitions. A declared pair the
    # engine cannot answer is refused, not zero-filled: a zero row would say
    # "this transition has no dipole" about a pair the run never reported.
    transition_dipole = []
    osc_strength = []
    for a, b in [pair for pair in config.transition_pair if len(pair) == 2]:
        if a != 0 or b >= len(td_dipole):
            raise ValueError(
                'PySCF computes only ground-to-first-N-excited transitions; '
                'declared transition pair (%d, %d) has no value to return '
                '(excited states computed: %d).'
                % (a, b, len(td_dipole)))
        transition_dipole.append(td_dipole[b])
        osc_strength.append(td_osc[b])

    # convert all results to np.array for downstream tasks
    # all np.array will be converted to list before saving to json for AI agent to read
    results = {
        'energy': np.array(energy),
        'dipole': np.array(dipole),
        'computed_charges': np.array(computed_charges),
        'excitation': np.array(excitation),
        'osc_strength': np.array(osc_strength),
        'transition_dipole': np.array(transition_dipole),
    }

    return results


def get_td_point_charges(mf, td) -> list:
    # pyscf does not provide point charge for excited states, we use the same point charge as ground state for all excited states as an approximation
    natom = mf.mol.natm
    nstates = td.nstates
    point_charges = [[0 for _ in range(natom)] for _ in range(nstates)]

    return point_charges

def get_td_dipole(td) -> list:
    # pyscf does not provide dipole moment for excited states
    nstates = td.nstates
    dipole = [[0, 0, 0] for _ in range(nstates)]

    return dipole


def save_mo(config, mf) -> str:
    # save mo coefficients to molden file for visualization if specified
    project = config.project
    from pyscf.tools import molden
    scratch_path = config.scratch_path
    save_mo = config.save_mo
    if save_mo:
        mo_path = f"{scratch_path}/{project}.mo.molden"
        with open(mo_path, 'w') as f:
            molden.header(mf.mol, f)
            molden.orbital_coeff(mf.mol, f, mf.mo_coeff)
    else:
        mo_path = ''

    return mo_path
