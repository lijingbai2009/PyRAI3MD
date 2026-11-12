"""
======================================================================
Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

"""Integration tests: cross-mode and all-shipped-example checks.

These tests are not unit tests. They load the framework's shipped
examples and skills and compare the carriers against each other, so
they are slower and are controlled by the top-level
``test_integration`` switch. The unit suite lives in the sibling
``verification`` package and always runs.
"""
