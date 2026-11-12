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

This module provides the geometry optimization functions for the PySCF skill,
inlcuding ground state local minimum and transition state optimization,
and excited state local minimum optimization.

"""

# PySCF is imported where it is used, not here: a skill has to be
# importable (registered, its schema and helpers usable) on a host
# whose engine package is missing or broken. See
# skills/lammps_skill/service/persistent/server.py::_import_native.

from ..schema.method import AVAILABLE_METHOD
from .energy import build_meanfield
from .energy import energy
from .hessian import get_hessian

def opt_min(config) -> dict:
    # important logic: target_state[0] is used for geometry optimization
    # target_state will be ordered in normalization step,
    # thus it is better to use single target_state for geometry optimization
    target_state = config.target_states[0]

    runtime = build_meanfield(config)

    if target_state == 0:
        runtime = optimize_ground_state(runtime, config)
    else:
        runtime = optimize_excited_state(runtime, config)

    return runtime


def opt_ts(config) -> dict:
    # important logic: ts optimization need initial hessian and assume the selected method provide analytical hessian
    # the availability of analytical hessian is checked in validation step,
    # thus we can directly call the optimize function from pyscf with hessian option
    maxit_opt = config.maxit_opt
    from pyscf.geomopt.geometric_solver import kernel as optimize

    conv_params = {
        'transition': True,
        'hessian': True,
        'convergence_energy': config.opt_energy,
        'convergence_grms': config.opt_grms,
        'convergence_gmax': config.opt_gmax,
        'convergence_drms': config.opt_drms,
        'convergence_dmax': config.opt_dmax,
    }

    callback=RecordOpt(config)
    runtime = build_meanfield(config)
    mf = runtime['mf']

    conv, mol = optimize(mf, maxsteps=maxit_opt, callback=callback, **conv_params)
    runtime = get_runtime(config, conv, mol)

    return runtime


def opt_irc(config) -> dict:
    raise NotImplementedError("IRC optimization is not implemented yet, as it requires more complex logic to handle the forward and backward IRC calculation, and the availability of IRC calculation is also limited for different methods. We will implement this in the future when we have more resources and time to test the implementation.")


def optimize_ground_state(runtime: dict, config) -> dict:
    # for ground state optimization, we can directly call the optimize function from pyscf
    # and we can use the energy function to get the energy and gradient
    maxit_opt = config.maxit_opt
    from pyscf.geomopt.geometric_solver import kernel as optimize
    conv_params = {
        'convergence_energy': config.opt_energy,
        'convergence_grms': config.opt_grms,
        'convergence_gmax': config.opt_gmax,
        'convergence_drms': config.opt_drms,
        'convergence_dmax': config.opt_dmax,
    }

    mf = runtime['mf']

    callback=RecordOpt(config)
    conv, mol = optimize(mf, maxsteps=maxit_opt, callback=callback, **conv_params)
    runtime = get_runtime(config, conv, mol)

    return runtime


def optimize_excited_state(runtime: dict, config) -> dict:
    # optimize excited state geometry
    # important logic: this function is called when user request target_state > 0,
    # and this function requires an excited state method
    # thus we must check if the method is available for excited state optimization
    maxit_opt = config.maxit_opt
    from pyscf.geomopt.geometric_solver import kernel as optimize
    conv_params = {
        'convergence_energy': config.opt_energy,
        'convergence_grms': config.opt_grms,
        'convergence_gmax': config.opt_gmax,
        'convergence_drms': config.opt_drms,
        'convergence_dmax': config.opt_dmax,
    }

    method = config.method
    target_state = config.target_states[0]
    state_spin = config.state_spin
    target_spin = state_spin[target_state]
    nstates = state_spin.count(target_spin)
    init_state = state_spin.index(target_spin) # the first state with the same spin as target state
    istate = target_state - init_state  # the index of target state in the excited state calculation
    ex_method = AVAILABLE_METHOD[method]['opt_min'] # this should be checked in validation step already

    mf = runtime['mf']
    mf.kernel()

    es = AVAILABLE_METHOD[method][ex_method](mf)
    es.nstates = nstates
    if target_spin == 2: es.singlet = False

    es_grad = es.nuc_grad_method().as_scanner(state=istate)
    callback=RecordOpt(config)
    conv, mol = optimize(es_grad, maxsteps=maxit_opt, callback=callback, **conv_params)
    runtime = get_runtime(config, conv, mol)

    return runtime


def get_runtime(config, conv: bool, mol) -> dict:
    # get the results from the optimization run, including energy, gradient, optimized geometry, etc.
    # this function can be called for both ground state and excited state optimization
    # for transition state optimization, we can also use this function to get the results after optimization
    config.system.coordinates = mol.atom_coords(unit='ANG').tolist()
    opt_sp = config.opt_sp
    opt_freq = config.opt_freq
    target_state = config.target_states[0]
    results = {
        'opt_converge': conv,
        # report the final structure to the framework through the
        # fin_coord channel so state.pmd.system stays in sync with
        # the skill-internal optimization
        'fin_coord': config.system.coordinates,
        }
    runtime = {'results': results}

    if opt_sp or opt_freq:
        runtime = energy(config)
        runtime['results'].update(results)

    if opt_freq and target_state == 0 and conv:
        runtime = get_hessian(runtime, config)
        runtime['results'].update(results)

    save_opt(config)

    return runtime


def save_opt(config):
    # save mo coefficients to molden file for visualization if specified
    project = config.project
    scratch_path = config.scratch_path
    natom = config.natom
    atoms = config.atoms
    coordinates = config.coordinates
    xyz = '%s\n\n' % natom
    for i in range(natom):
        xyz += '%-5s %24.16f %24.16f %24.16f\n' % (atoms[i], coordinates[i][0], coordinates[i][1], coordinates[i][2])

    with open('%s/%s.opt.xyz' % (scratch_path, project), 'w') as f:
        f.write(xyz)

    return 0


class RecordOpt:
    # this is a callback function for geometry optimization, which can be used to record the optimization process, such as energy, geometry, etc.
    # this function will be called in each optimization step, and the results will be saved to a log file for visualization in the frontend

    def __init__(self, config):
        self.project = config.project
        self.scratch_path = config.scratch_path
        self.step = 0


    def __call__(self, envs):
        self.step += 1
        mol = envs['mol']
        e = envs['energy']

        if self.step == 1:
            mode = 'w'
        else:
            mode = 'a'

        atoms = [atom[0] for atom in mol._atom]
        coord = mol.atom_coords(unit='ANG').tolist()
        with open('%s/%s.opt_traj.xyz' % (self.scratch_path, self.project), mode) as f:
            xyz = '%s\nStep: %s E: %s\n' % (len(atoms), self.step, e)
            for i in range(len(atoms)):
                xyz += '%-5s %24.16f %24.16f %24.16f\n' % (atoms[i], coord[i][0], coord[i][1], coord[i][2])
            f.write(xyz)
