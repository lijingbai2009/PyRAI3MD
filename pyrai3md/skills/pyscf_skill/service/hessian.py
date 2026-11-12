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

This module provide the hessian calculation functions for the PySCF skill,
which return the hessian matrix and related vibrational analysis results
for thermochemsitry analysis and external use

"""


import numpy as np
# PySCF is imported where it is used, not here: a skill has to be
# importable (registered, its schema and helpers usable) on a host
# whose engine package is missing or broken. See
# skills/lammps_skill/service/persistent/server.py::_import_native.

from .energy import energy


def hessian(config) -> dict:
    # Perform the hessian calculation and return the hessian matrix
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
    """
    runtime = energy(config)
    runtime = get_hessian(runtime, config)

    return runtime


def get_hessian(runtime: dict, config) -> dict:
    # compute the hessian based on the specified method and functional
    mf = runtime['mf']

    # important logic: hessian calculation must be executated after energy calculation,
    # which has checked the availability of the selected method
    # so we don't check again here
    hessian_matrix = mf.Hessian().kernel()
    results = get_results(runtime, config, hessian_matrix)
    runtime['results'].update(results)

    save_freq(runtime, config)

    return runtime


def get_results(runtime: dict, config, hessian_matrix):
    mf = runtime['mf']
    from pyscf.hessian import thermo
    mol = mf.mol
    dim = len(hessian_matrix)
    temperature = config.temperature  # in K
    pressure = config.pressure # note pressure in bar
    freq_results = thermo.harmonic_analysis(mol, hessian_matrix, imaginary_freq=False)
    thermo_results = thermo.thermo(mf, freq_results['freq_au'], temperature, pressure*1e5)  # convert bar to Pa
    results = {
        'hessian': np.array(hessian_matrix).transpose(0, 2, 1, 3).reshape(3*dim, 3*dim),  # in Hartree/Bohr^2
        'frequencies': np.array(freq_results['freq_wavenumber']),  # in cm^-1
        'norm_mode': np.array(freq_results['norm_mode']),  # in amu^0.5 * bohr
        'reduced_mass': np.array(freq_results['reduced_mass']),  # in amu
        'rotation_constants': np.array(thermo_results['rot_const'][0]),  # in GHz
        'heat_capacity_cv': thermo_results['Cv_tot'][0],  # in Hartree/K
        'heat_capacity_cp': thermo_results['Cp_tot'][0],  # in Hartree/K
        'zero_point_energy': thermo_results['ZPE'][0],  # in Hartree, ZPE = 0.5 * sum(freq_au)
        'internal_energy': thermo_results['E_0K'][0],  # in Hartree， E_OK = E_electronic + ZPE
        'enthalpy': thermo_results['H_tot'][0],  # in Hartree, H_tot = E_0K + RT + dH
        'entropy': thermo_results['S_tot'][0],  # in Hartree/K
        'free_energy': thermo_results['G_tot'][0],  # in Hartree, G_tot = H_tot - T*S_tot
    }

    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results[key] = value[np.newaxis,:] # add state dimension for array results
        elif isinstance(value, list):
            results[key] = np.array(value)[np.newaxis,:] # add state dimension for array results
        elif isinstance(value, (int, float)):
            results[key] = np.array([value]) # convert scalar to array with state dimension
        else:
            raise ValueError(f"Unsupported result type for key '{key}': {type(value)}")

    return results


def format_freq_molden(atoms, coords, frequencies, modes):
    """Assemble a Molcas-layout frequency record.

    Parameters
    ----------
    atoms : list of str
        Element symbols, length natom.
    coords : array-like, shape (natom, 3)
        Equilibrium geometry in Bohr.
    frequencies : array-like, shape (nmode,)
        Harmonic frequencies in cm-1.
    modes : array-like, shape (nmode, natom, 3)
        Cartesian displacement vectors in Bohr, written exactly as they are
        given. `workflow.sampling.read_molden` assumes the Molcas convention
        of unnormalized, unmass-weighted vectors, so the caller converts
        PySCF's mass-weighted normalized `norm_mode` before handing them in
        (see `save_freq`).

    Returns
    -------
    str
        The record text.
    """
    natom = len(atoms)
    nmode = len(frequencies)
    frequency = '\n'.join('%10.2f' % float(value) for value in frequencies)
    coord = ''.join('%-5s %24.8f %24.8f %24.8f\n'
                    % (symbol, float(x), float(y), float(z))
                    for symbol, (x, y, z) in zip(atoms, coords))
    vibs = ''
    for index in range(nmode):
        rows = '\n'.join(' '.join('%24.8f' % float(value) for value in row)
                          for row in modes[index])
        vibs += '  vibration  %5s\n%s\n' % (index + 1, rows)
    return """[MOLDEN FORMAT]
[N_FREQ]
%s
[NATOM]
%s
[FREQ]
%s
[FR-COORD]
%s
[FR-NORM-COORD]
%s

""" % (nmode, natom, frequency, coord, vibs)


def save_freq(runtime: dict, config):
    # Save the normal modes when asked: the record serves the frontend
    # visualization AND the initial-condition sampler, which reads it through
    # `workflow.sampling.read_molden`. That reader follows the Molcas
    # convention of unnormalized, unmass-weighted Cartesian displacements,
    # while PySCF reports `norm_mode` mass-weighted and normalized
    # (sqrt(amu) Bohr). Dividing each atom's vector by sqrt of its mass
    # converts one convention into the other; without it the sampler would
    # displace the light atoms too far and the heavy ones too little, and a
    # sampled ensemble would look plausible while being wrong.
    project = config.project
    scratch_path = config.scratch_path
    save_norm_mode = config.save_norm_mode
    if not save_norm_mode:
        return
    frequencies = np.asarray(runtime['results']['frequencies'])
    norm_mode = np.asarray(runtime['results']['norm_mode'])
    mol = runtime['mf'].mol

    atoms = [atom[0] for atom in mol._atom]
    coords = mol.atom_coords(unit='Bohr')
    natom = len(atoms)
    # Results carry a leading state dimension, and the modes arrive as the
    # flattened (3N, 3N) block; the layout below is the one this file has
    # always written, only the scaling changes.
    block = norm_mode.reshape((-1, natom, 3))
    nmode = block.shape[0]
    masses = np.asarray(mol.atom_mass_list(isotope_avg=True), dtype=float)
    modes = block / np.sqrt(masses)[np.newaxis, :, np.newaxis]

    molden = format_freq_molden(atoms, coords,
                                frequencies.reshape(-1), modes)
    with open('%s/%s.freq.molden' % (scratch_path, project), 'w') as f:
        f.write(molden)
