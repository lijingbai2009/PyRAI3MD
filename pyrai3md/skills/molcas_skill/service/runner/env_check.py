"""
======================================================================
This is a standalone OpenMolcas calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides environment validation and workdir resolution
for Molcas execution.

"""

import hashlib
import os


def validate_environment(config):
    """Check that the Molcas installation is accessible.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration container.

    Raises
    ------
    RuntimeError
        If molcas_path is not set, does not exist, or the
        pymolcas executable cannot be found.
    """
    molcas_path = config.molcas_path
    if not molcas_path:
        raise RuntimeError(
            'molcas_path is not set. '
            'Set it in pmd_env.yaml (e.g. molcas_path: /opt/openmolcas)'
        )
    if not os.path.isdir(molcas_path):
        raise RuntimeError(
            f'molcas_path directory does not exist: {molcas_path}'
        )
    pymolcas = os.path.join(molcas_path, 'bin', 'pymolcas')
    if not os.path.isfile(pymolcas):
        # OpenMolcas installs the driver at the prefix root in recent
        # builds; older layouts keep it under bin/.
        pymolcas = os.path.join(molcas_path, 'pymolcas')
    if not os.path.isfile(pymolcas):
        raise RuntimeError(
            f'pymolcas executable not found at: {pymolcas}'
        )


def resolve_workdir(config, calculation_scratch='', term='', run_id=''):
    """Resolve the Molcas working directory.

    The working directory comes from calculation.scratch_path
    (temporary-file control for calculation functions), which the
    framework syncs from environment.scratch_path by default.
    Molcas may override this decision for its own IO needs:
    - 'AUTO': prefer /srv/tmp, fallback to /tmp
    - empty: prefer the environment scratch_path, then AUTO
    - otherwise: use the specified path

    A per-term directory is appended underneath whichever root is
    resolved, because the run script both creates and `rm -r`s
    $MOLCAS_WORKDIR/$MOLCAS_PROJECT and MOLCAS_PROJECT is only the
    program label: two terms of this skill that share one root would
    otherwise delete each other's scratch.

    The AUTO root also carries a short hash of `run_id` when one is
    given. The term is only the calculation directory's basename, which
    repeats across examples and slurm jobs (`tmp_high_qm_MOLCAS`), so
    two jobs sharing a node wrote and deleted the same runfile and the
    gateway aborted with an internal inconsistency (round
    tmp_20261111_132628, 63 rows). The hash is stable for one
    calculation directory, so a restart keeps its scratch.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration container.
    calculation_scratch : str, optional
        Value of calculation.scratch_path from the nested config.
    term : str, optional
        Per-term directory stem (the calculation directory name),
        appended to keep concurrent terms apart. Empty keeps the
        historical single-directory layout.
    run_id : str, optional
        Unique run identity (the calculation directory). Empty keeps
        the historical shared AUTO path.

    Returns
    -------
    str
        Resolved working directory path.
    """
    workdir = calculation_scratch or config.scratch_path
    project = config.project or 'MOLCAS'

    if not workdir or str(workdir).upper() == 'AUTO':
        user = os.environ.get('USER', 'unknown')
        root = '/srv/tmp' if os.path.exists('/srv/tmp') else '/tmp'
        workdir = f'{root}/{user}/{project}'
        if run_id:
            digest = hashlib.sha1(
                os.path.abspath(str(run_id)).encode('utf-8')).hexdigest()[:10]
            workdir = os.path.join(workdir, digest)

    if term:
        workdir = os.path.join(workdir, str(term))

    return workdir
