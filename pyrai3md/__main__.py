"""
======================================================================
Python -m entry point for PyRAI3MD.

Usage:
    python -m pyrai3md <input_file> [--info]

Delegates to pmd.main() for the traditional CLI workflow.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .pmd import main

if __name__ == '__main__':
    main()
