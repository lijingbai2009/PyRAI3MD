"""
======================================================================
This package collects the multiscale execution components of the
workflow layer: the shared machinery a composite (multi-region,
multi-center) job needs between the Calculator and the backends.

A multi-center job is one case of the family; later multiscale schemes
land here as additional modules rather than as new workflows, because
none of this decides execution order -- the workflow does.

    multiscale/
        regions.py       pure geometric region operations (atom
                         selection, gradient/NAC scatter-add, region
                         validation, per-center electronic projection)
        composite.py     contribution patterns of the composite schemes
                         and the task/outcome aggregation layer
        charges.py       embedding-charge sources, per-task embedding
                         and the outcome-driven charge update

Design rules:
- These are components: no module declares a Workflow subclass and none
  appears in WORKFLOW_REGISTRY or JOBTYPE_MAP.
- regions.py is the leaf of the layer: pure functions, no state, and it
  never imports Calculator, composite or the data package. composite.py
  builds on it, the Calculator builds on both.
- The MD chain never imports this package, so the two component families
  stay independent.

Oct 04 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .regions import (
    validate_region, materialize_region, project_gradient, slice_field,
    project_energy, project_nac, project_soc, project_dipole,
)
from .composite import (
    aggregate_results, term_label, build_composite_terms,
    TaskOutcome, invalidate_task_result, normalize_task_result,
    validate_plan, assemble_results, collect_center_electronic,
)
from .charges import (
    region_charge_sources, update_from_outcomes, embedding_for_task,
)

__all__ = [
    'validate_region',
    'materialize_region',
    'project_gradient',
    'slice_field',
    'project_energy',
    'project_nac',
    'project_soc',
    'project_dipole',
    'aggregate_results',
    'term_label',
    'build_composite_terms',
    'TaskOutcome',
    'invalidate_task_result',
    'normalize_task_result',
    'validate_plan',
    'assemble_results',
    'collect_center_electronic',
    'region_charge_sources',
    'update_from_outcomes',
    'embedding_for_task',
]
