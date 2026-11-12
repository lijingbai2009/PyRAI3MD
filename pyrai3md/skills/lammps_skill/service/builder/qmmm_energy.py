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

Input generation for the qmmm_energy (Flow 2) task.

One whole-cell full-force-field evaluation with the QM 0-order charges
written into the data file. The classical QM/MM decomposition is done in
the backend after the run: `in_cell_qq` supplies U_gas, and the Flow-1
V/E arrays supply qmmm_el. No QM/MM compute or extra run is added to the
LAMMPS script itself.
"""

import os

from ..utils import output_basename
from .input_builder import (build_script_head, build_step_tail,
                            write_input_package)
from .qmmm import _charge_index


def _rewrite_qm_charges(calcdir: str, project: str, model: dict,
                        qm_ids, q_qm) -> None:
    """Write q_qm into the QM rows of the rendered data file."""
    by_id = {int(atom_id): float(charge)
             for atom_id, charge in zip(qm_ids, q_qm)}
    path = os.path.join(calcdir, f'{project}.data')
    lines = open(path, 'r').read().splitlines()
    try:
        header = next(index for index, line in enumerate(lines)
                      if line.split()[:1] == ['Atoms'])
        start = header + 2
    except StopIteration:
        raise ValueError(f'{path} has no Atoms section')
    seen = set()
    for index in range(start, len(lines)):
        content, sep, comment = lines[index].partition('#')
        tokens = content.split()
        if not tokens:
            break
        try:
            atom_id = int(tokens[0])
        except ValueError:
            break
        if atom_id not in by_id:
            continue
        column = _charge_index(model, tokens)
        tokens[column] = f'{by_id[atom_id]:.16e}'
        lines[index] = ' '.join(tokens) + (
            ' ' + sep + comment if sep else '')
        seen.add(atom_id)
    if seen != set(by_id):
        raise ValueError(
            f'data file {path} lacks QM atom ids {sorted(set(by_id) - seen)}')
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')


def write_energy_input_file(config, workflow, calcdir: str, model,
                            q_qm) -> str:
    """Write the cold qmmm_energy input script and inject q_qm."""
    project = config.lammps_project
    basename = output_basename(project, workflow.profile)
    write_input_package(config, calcdir, model)
    _rewrite_qm_charges(calcdir, project, model,
                        model['qmmm']['qm_ids'], q_qm)
    head = build_script_head(config, workflow, calcdir, model)
    tail = build_step_tail(config, workflow)
    path = os.path.join(calcdir, f'{basename}.in')
    with open(path, 'w') as handle:
        handle.write(head)
        handle.write(tail)
    return basename
