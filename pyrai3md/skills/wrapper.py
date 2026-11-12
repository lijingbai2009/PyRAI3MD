"""
======================================================================
This module provides the pyrai3md_skill command entry point.

It is responsible for:
1. Reading the input .json config file
2. Extracting skill_name from the configuration
3. Executing the skill agent via subprocess

This module does NOT use the internal query module. It launches
the skill as an independent subprocess to verify the skill's
standalone execution capability.

Usage:
    pyrai3md_skill <input.json>

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import subprocess
import sys

from .query import list_skills


def resolve_skill_name(config):
    """Resolve the skill name from an input config dict.

    Flat skill configs carry 'skill_name' at the top level; older
    sectioned configs nest it in the 'environment' section, so fall
    back to that location.

    Parameters
    ----------
    config : dict
        Parsed input configuration.

    Returns
    -------
    str
        Skill name, or an empty string when not found.
    """
    skill_name = config.get('skill_name', '')
    if not skill_name:
        environment = config.get('environment')
        if isinstance(environment, dict):
            skill_name = environment.get('skill_name', '')
    return skill_name or ''


def main():
    """Entry point for the pyrai3md_skill command.

    Reads the input .json config, extracts skill_name, and executes
    the skill agent as an independent subprocess.

    The input .json config must contain a 'skill_name' field
    identifying the skill agent module.
    """
    if len(sys.argv) < 2:
        print('Usage: pyrai3md_skill <input.json>[JSON]')
        sys.exit(1)

    config_file = sys.argv[1]

    # Load input configuration
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Extract skill_name from config
    skill_name = resolve_skill_name(config)
    if not skill_name:
        print(f'Error: skill_name required in {config_file}')
        sys.exit(1)
    available = list_skills()
    if skill_name not in available:
        print(
            f'Error: unknown skill {skill_name!r} in {config_file}; '
            f'available skills: {", ".join(available)}')
        sys.exit(1)

    # Construct module path: <skills package>.<skill_name>.agent, with
    # the package taken from this module's own import path.
    skills_package = __name__.rpartition('.')[0] or 'pyrai3md.skills'
    full_module = f'{skills_package}.{skill_name}'

    # Execute skill agent as independent subprocess. The agent reads
    # its own execution_mode out of the config file; the command line
    # carries nothing but the config path.
    command = [sys.executable, '-m', f'{full_module}.agent', config_file]
    proc = subprocess.run(command)

    sys.exit(proc.returncode)
