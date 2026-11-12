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

This module provide the analysis functions for the PySCF skill,
such as mo, dipole, esp, density, nto, population, etc.

"""

# PySCF is imported where it is used, not here: a skill has to be
# importable (registered, its schema and helpers usable) on a host
# whose engine package is missing or broken. See
# skills/lammps_skill/service/persistent/server.py::_import_native.
from .energy import energy

def mo(config) -> dict:
    # save homo and lumo to cube
    config.calculation.save_mo = True
    runtime = energy(config)
    runtime = get_mo(runtime, config)

    return runtime


def dipole(config) -> dict:
    # pyscf cannot save dipole to cube, just return energy runtime
    runtime = energy(config)

    return runtime


def esp(config) -> dict:
    # save esp to cube
    runtime = energy(config)
    get_esp(runtime, config)

    return runtime




def population(config) -> dict:
    # pyscf cannot save population to cube, just return runtime
    runtime = energy(config)

    return runtime


def get_mo(runtime: dict, config) -> dict:
    # save mo to cube
    project = config.project
    from pyscf import tools
    scratch_path = config.scratch_path
    total_electron = config.total_electron
    basename = '%s/%s' % (scratch_path, project)
    mol = runtime['mol']
    mf = runtime['mf']
    if total_electron % 2 != 0:
        mo_1 = int(total_electron - 1) / 2
        mo_2 = int(total_electron + 1) / 2
        mo_1_name = 'somo_1'
        mo_2_name = 'somo_2'
    else:
        mo_1 = int(total_electron / 2)
        mo_2 = mo_1 + 1
        mo_1_name = 'homo'
        mo_2_name = 'lumo'

    file1 = '%s.%s.cube' % (basename, mo_1_name)
    file2 = '%s.%s.cube' % (basename, mo_2_name)

    tools.cubegen.orbital(mol, file1, mf.mo_coeff[:, mo_1 - 1])
    tools.cubegen.orbital(mol, file2, mf.mo_coeff[:, mo_2 - 1])

    runtime['results']['mo_1'] = file1
    runtime['results']['mo_2'] = file2

    return runtime


def get_esp(runtime: dict, config) -> dict:
    # save esp to cube
    project = config.project
    from pyscf import tools
    scratch_path = config.scratch_path
    basename = '%s/%s' % (scratch_path, project)
    mol = runtime['mol']
    mf = runtime['mf']

    file = '%s.esp.cube' % basename
    tools.cubegen.mep(mol, file, mf.make_rdm1())

    runtime['results']['esp'] = file

    return runtime
