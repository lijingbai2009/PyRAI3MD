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

This module handles Molcas orbital file management,
including StrOrb, RasOrb, and JobIph files.

"""

import os
import shutil


def manage_orbital_files(config, calcdir: str):
    """Manage orbital file transfer for the current calculation.

    Handles the following orbital file types:
    - StrOrb: starting orbital file
    - RasOrb: RASSCF output orbital file
    - JobIph: JobIph file for state-averaged calculations

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str
        Calculation directory.
    """
    project = config.project
    guess = config.guess_orbital

    strorb_dst = os.path.join(calcdir, f'{project}.StrOrb')
    jobiph_dst = os.path.join(calcdir, f'{project}.JobIph')

    # Copy starting orbital
    if guess and os.path.exists(guess):
        if not os.path.exists(strorb_dst):
            shutil.copy2(guess, strorb_dst)
    else:
        # Look for StrOrb in current directory
        local_strorb = f'{project}.StrOrb'
        if os.path.exists(local_strorb) and not os.path.exists(strorb_dst):
            shutil.copy2(local_strorb, strorb_dst)

    # Check for RasOrb from previous RASSCF step
    rasorb_src = os.path.join(calcdir, f'{project}.RasOrb')
    if os.path.exists(rasorb_src):
        shutil.copy2(rasorb_src, strorb_dst)

    # Check for JobIph
    local_jobiph = f'{project}.JobIph'
    if os.path.exists(local_jobiph) and not os.path.exists(jobiph_dst):
        shutil.copy2(local_jobiph, jobiph_dst)

    jobiph_new = os.path.join(calcdir, f'{project}.JobIph.new')
    if os.path.exists(jobiph_new):
        shutil.copy2(jobiph_new, jobiph_dst)


def save_orbital_files(config, calcdir: str, dest_dir: str):
    """Save orbital files from calculation directory to a destination.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str
        Source calculation directory.
    dest_dir : str
        Destination directory.
    """
    project = config.project

    for suffix in ['.StrOrb', '.RasOrb', '.JobIph', '.JobIph.new']:
        src = os.path.join(calcdir, f'{project}{suffix}')
        if os.path.exists(src):
            dst = os.path.join(dest_dir, f'{project}{suffix}')
            shutil.copy2(src, dst)
