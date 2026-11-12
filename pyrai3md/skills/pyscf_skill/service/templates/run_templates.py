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

This module provides SLURM script templates for PySCF skill execution.

"""

SLURM_SCRIPT_TEMPLATE = """#!/bin/bash
{partition_line}#SBATCH --time={time}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory}mb
#SBATCH --job-name={job_name}
#SBATCH --output={output}
#SBATCH --error={error}

{command}

export OMP_NUM_THREADS={cpus_per_task}
export PYTHONPATH={pythonpath}
{python_exec} -m {module} {config_file}
"""
