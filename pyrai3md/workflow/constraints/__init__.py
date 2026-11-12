"""
======================================================================
Constraint shadow build: constraint components.

Mirror of pyrai3md/workflow/constraints. Each module owns one family of
numerics (schedule, bias, wall, projection, shake, metadynamics) and the
controller composes them. Nothing here imports a framework state object:
the inputs are arrays, records and an EvaluationContext.

Oct 1 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""
