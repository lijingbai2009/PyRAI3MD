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

This module provides the spectrum calculation functions for the PySCF skill,
including IR, Raman(Not Implemented), UV-Vis, and NMR spectrum calculations.

"""

import numpy as np

from ..schema.method import AVAILABLE_METHOD
from .energy import energy
from .hessian import get_results as get_hessian_results


def ir(config) -> dict:
    # Perform IR spectrum calculation and return the results
    """
    return a runtime dictionary, where the results including:
        'hessian',
        'frequencies',
        'norm_mode',
        'reduced_mass',
        'rotation_constants',
        'heat_capacity_cv',
        'heat_capacity_cp',
        'zero_point_energy',
        'internal_energy',
        'enthalpy',
        'entropy',
        'free_energy',
        'ir_intensity',
        'ir_spectrum',
        'ir_data',
    """
    import matplotlib.pyplot as plt

    runtime = energy(config)
    mf = runtime['mf']

    ir_width = config.ir_width
    ir_scale = config.ir_scale
    method = config.method
    multiplicity = config.multiplicity
    is_open_shell = multiplicity > 1

    # important logic: method has been checked in validation, so we are sure the reference exists
    closed, open_ = AVAILABLE_METHOD[method]['infrared']
    mf_ir = open_(mf) if is_open_shell else closed(mf)

    mf_ir = mf_ir.run()
    mf_ir.summary()

    ir_inten = mf_ir.ir_inten
    ir_path = mf.chkfile.replace('.chk', '_IR.png')
    fig, ax, ax2 = mf_ir.plot_ir(w=ir_width, scale=ir_scale)
    try:
        fig.savefig(ir_path, bbox_inches='tight', dpi=400)
    finally:
        plt.close(fig)
    ir_data = np.array([ax.get_lines()[0].get_xdata(), ax.get_lines()[0].get_ydata()]).T
    np.savetxt(mf.chkfile.replace('.chk', '_IR.txt'), ir_data)

    results = get_hessian_results(runtime, config, mf_ir.mf_hess.de)
    results['ir_intensity'] = np.array(ir_inten)
    results['ir_spectrum'] = ir_path
    results['ir_data'] = ir_data
    runtime['results'].update(results)

    return runtime


def raman(config) -> dict:
    # Perform Raman spectrum calculation and return the results
    raise NotImplementedError("Raman spectrum calculation is not implemented yet.")


def uv_vis(config) -> dict:
    # Perform UV-Vis spectrum calculation and return the results
    """
    return a runtime dictionary, where the results including:
        'energy',
        'dipole',
        'excitation',
        'osc_strength',
        'transition_dipole',
        'uv_spectrum',
        'uv_data',
    """
    runtime = energy(config)
    mf = runtime['mf']
    uv_width = config.uv_width
    results = runtime['results']
    excitation = results['excitation'][1:] # skip the first state which is the ground state
    osc_strength = results['osc_strength'][1:] # skip the first state which is the ground state

    uv_path = mf.chkfile.replace('.chk', '_UVVis.png')

    uv_data = plot_uv(excitation, osc_strength, uv_width, uv_path)
    np.savetxt(mf.chkfile.replace('.chk', '_UVVis.txt'), uv_data)

    results['uv_spectrum'] = uv_path
    results['uv_data'] = uv_data
    runtime['results'].update(results)

    return runtime


def nmr(config) -> dict:
    # Perform NMR spectrum calculation and return the results
    """
    return a runtime dictionary, where the results including:
        'energy',
        'dipole',
        'excitation',
        'osc_strength',
        'transition_dipole',
        'shielding_tensors',
        'shielding_constants',
        'nics',
        'nics_zz',
    """
    natom = config.natom
    atoms = config.atoms
    coordinates = config.coordinates
    nics_ring = config.nics_ring
    nics_scan = config.nics_scan

    if len(nics_ring) > 1:
        do_nics = True
        nics_ring = [x - 1 for x in nics_ring] # convert to 0-based index
        nics_atoms, nics_coordinates = add_nics_coordinates(atoms, coordinates, nics_ring, nics_scan)
        config.system.atoms = nics_atoms
        config.system.coordinates = nics_coordinates
    else:
        do_nics = False

    runtime = shielding(config)

    if do_nics:
        results = runtime['results']
        nics_index = [natom + x for x in range(len(nics_scan))]
        shielding_tensors = results['shielding_tensors'][nics_index]
        nics = -np.trace(shielding_tensors, axis1=1, axis2=2) / 3
        nics_zz = -shielding_tensors[:, 2, 2]
        results['nics'] = nics[np.newaxis, :] # add state dimension
        results['nics_zz'] = nics_zz[np.newaxis, :] # add state dimension
        runtime['results'].update(results)

    return runtime

def shielding(config) -> dict:
    # compute nmr shielding constants and return the results
    runtime = energy(config)
    mf = runtime['mf']

    method = config.method
    multiplicity = config.multiplicity
    is_open_shell = multiplicity > 1

    # important logic: method has been checked in validation, so we are sure the reference exists
    classes = AVAILABLE_METHOD[method].get('shielding')
    if classes is None:
        raise RuntimeError(
            "the NMR shielding job needs the PySCF properties extension, which "
            "is not installed in this environment; install it with "
            "`pip install git+https://github.com/pyscf/properties` (see "
            "skills/pyscf_skill/skill_guide.md). HF and DFT energies, gradients "
            "and MD run without it.")
    closed, open_ = classes
    mf_nmr = open_(mf) if is_open_shell else closed(mf)

    shielding_tensor =mf_nmr(mf).kernel()

    results = runtime['results']
    results['shielding_tensors'] = np.array(shielding_tensor)[np.newaxis,:] # add state dimension
    results['shielding_constants'] = np.array([np.trace(x) / 3 for x in shielding_tensor])[np.newaxis,:] # add state dimension
    runtime['results'].update(results)

    return runtime


def plot_uv(excitation, osc_strength, uv_width, uv_path):
    grid = np.linspace(1, 1000, 1000)
    spectrum = np.zeros_like(grid)
    for exc, osc in zip(excitation, osc_strength):
        wav = 1239.84 / exc / 27.211 # convert excitation energy (Hartree) to wavelength (nm)
        spectrum += osc * np.exp(-(grid - wav)**2 / (2 * uv_width**2)) # Gaussian broadening

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Oscillator Strength')

    ax.plot(grid, spectrum)

    try:
        fig.savefig(uv_path, bbox_inches='tight', dpi=400)
    finally:
        plt.close(fig)

    return np.array([grid, spectrum]).T


def add_nics_coordinates(atoms, coordinates, nics_ring, nics_scan):
    # add nics coordinates to the original coordinates
    coordinates = np.array(coordinates)
    src = coordinates[nics_ring][0]
    dst = coordinates[nics_ring][1]
    center = coordinates[nics_ring].mean(axis=0)
    v1 = src - center
    v2 = dst - center
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])

    nx = np.dot(x, normal)
    ny = np.dot(y, normal)
    nz = np.dot(z, normal)

    ix = np.argmax(np.abs([nx, ny, nz]))
    ax = np.array([x, y, z])[ix]

    y = np.cross(normal, ax)
    y = y / np.linalg.norm(y)
    x = np.cross(y, normal)
    x = x / np.linalg.norm(x)
    z = normal

    coordinates = coordinates @ np.array([x, y, z]).T
    center = coordinates[nics_ring].mean(axis=0)
    nics_atoms = []
    nics_coordinates = []
    for scan in nics_scan:
        nics_atoms.append('X')
        nics_coordinates.append(center + np.array([0.0, 0.0, scan]))

    nics_coordinates = np.array(nics_coordinates)

    atoms = atoms + nics_atoms
    coordinates = coordinates.tolist() + nics_coordinates.tolist()

    return atoms, coordinates
