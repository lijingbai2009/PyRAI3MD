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

This module provide the gradient calculation functions for the PySCF skill,
which return the energy and gradients for external use

"""

import numpy as np

from ..schema.method import AVAILABLE_METHOD

from .energy import energy


def gradient(config) -> dict:
    # Perform the gradient calculation and return the gradient vector
    """
    return a runtime dictionary, where the results including:
        'gradient',
    """
    runtime = energy(config)
    grad = get_gradient(runtime, config)
    results = get_results(grad)

    runtime['results'].update(results)

    return runtime


def get_gradient(runtime, config):
    # compute the gradient based on the specified method and functional
    # important logic: if td is not None, it must be a td calculation
    # important note: target_states is not necceessarily the same as the complete state order

    post_method = AVAILABLE_METHOD[config.method]['post_method']

    if post_method == 'mp2':
        grad = get_mp2_grad(runtime)

    elif post_method == 'tddft':
        grad = get_td_grad(runtime, config)

    else:
        grad = get_scf_grad(runtime)

    return grad



def get_scf_grad(runtime: dict):
    # for scf gradient, we can directly call the gradient function from pyscf
    mf = runtime['mf']
    return np.array([mf.Gradients().kernel()])


def get_mp2_grad(runtime: dict):
    # for mp2 gradient, we can directly call the gradient function from pyscf
    mp = runtime['post'][0]
    return np.array([mp.Gradients().kernel()])


def get_td_grad(runtime: dict, config):
    # important logic: for td gradient, we need to consider the spin state of the target states,
    # and compute the gradient for each target state separately,
    # and return a list of gradients for each target state
    nstate = config.nstate
    natom = config.natom
    target_states = config.target_states
    state_spin = config.state_spin
    mf = runtime['mf']
    pt = runtime['post']
    grad = np.zeros((nstate, natom, 3))
    g0 = [x for x in target_states if x == 0]  # ground state
    # singlet or other spins
    g1 = [x for x in target_states if state_spin[x] != 2]
    g2 = [x for x in target_states if state_spin[x] == 2]  # triplet

    td_1, td_2 = pt

    if len(g0) > 0:
        grad[0] = mf.Gradients().kernel()

    if td_1 and len(g1) > 0:
        for g in g1:
            grad[g] = td_1.Gradients().kernel(state=g)

    if td_2 and len(g2) > 0:
        for g in g2:
            grad[g] = td_2.Gradients().kernel(state=g-len(g1)-1)

    return grad

def get_results(grad):
    # Extract the gradient vector and format it for output

    return {
        'gradient': np.array(grad),
    }
