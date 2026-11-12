"""
======================================================================
This module is the public API of the skill certification package.

Certification is the skill-dedicated testing infrastructure of
PyRAI3MD. It owns the skill test planning (planner) and the test
input generation tools (converter, handler) that were split out of
the former monolithic qualification converter. The qualification
manager and runner keep the unified planning and scheduling: they
consume plan_skill_tests() from this package as one of the three
test sources (verification / certification / benchmark).

Reference direction (one-way only):
    qualification.manager -> certification -> skills.query / framework

Deliberate lazy-export policy: planner is NOT imported here. It
imports the shared script writers from qualification.manager at
module level, so importing it while qualification.manager is still
initializing (qualification.manager imports certification.converter
at load time) would form a cycle. The manager therefore imports
certification.planner lazily inside generate_plan, and external
consumers import the submodules directly.

Aug 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .converter import (yaml_to_config, load_config, convert,
                    convert_to_pyrai_config)
