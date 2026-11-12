"""
======================================================================
PyRAI3MD - Python Rapid Artificial Intelligence Ab Initio Molecular Dynamics

A modular scientific computing framework for quantum chemistry,
machine learning, molecular dynamics, and AI-driven workflows.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

__version__ = '3.0.0-dev'


def _ensure_user_config():
    """Create ~/.pyrai3md/pmd_env.yaml on first import after pip install."""
    import os
    import shutil

    user_dir = os.path.join(os.path.expanduser('~'), '.pyrai3md')
    user_config = os.path.join(user_dir, 'pmd_env.yaml')

    if os.path.isfile(user_config):
        return

    # Find package config (pyrai3md/qualification/pmd_env.yaml)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_config = os.path.join(pkg_dir, 'qualification', 'pmd_env.yaml')

    if not os.path.isfile(pkg_config):
        return

    try:
        os.makedirs(user_dir, exist_ok=True)
        shutil.copy2(pkg_config, user_config)
        print(f'  Created user config: {user_config}')
        print(f'  Edit this file to set molcas_path, bagel_path, etc.')
    except OSError:
        pass  # Silently ignore if we can't write


_ensure_user_config()
