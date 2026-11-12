"""
======================================================================
kTDC benchmark task module (flat benchmark layout).

One module per benchmark task type: ktdc.py serves the unified
ktdc.yaml case definition shipped next to it. The model
(tully1/2/3, lz1/2/3) is selected dynamically through the
production.method key of the case, so calibrating another model
only changes the method of the pmd_env benchmark entries -- no new
YAML file per model (a dedicated ktdc_<method>.yaml would take
precedence for one specific method; the lz calibrations ship such
dedicated cases with the analytic Landau-Zener verdict).

The kTDC calibration runs the FSSH algorithm with the energy-only
curvature coupling (nactype 'ktdc') instead of the exact qdm
transition gradients (nactype 'nac'): the production md input is
otherwise identical to the fssh calibration, so the two entries
isolate the NAC source as the single difference. The complete
calibration infrastructure is algorithm-agnostic and therefore
reused from the fssh task module verbatim (the same module
objects, never re-implemented copies):

- load_case / apply_overrides: the case YAML layout and the
  pmd_env override merging are identical;
- build_reference_input: the exact split-operator quantum
  reference depends on the model and the wavepacket setup only,
  never on the hopping NAC source under calibration;
- build_production_input: the shared builder forwards the case md
  keywords through the shared PRODUCTION_MD_KEYWORDS table (sfhp
  'fssh', nactype taken from the case; the ktdc case files pin
  nactype: ktdc while the fssh case files pin nactype: nac);
- evaluate_run: the shared evaluator reports the unified
  final-state metric set (the final trajectory / state / reference
  populations plus the final-state differences; the lz family
  additionally compares against the analytic single-passage
  Landau-Zener probability), saves the compared population tables
  and renders the S0 population plot exactly like the fssh
  calibration.

Sep 2 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from .fssh import (
    AMU_TO_AU,
    BOHR_TO_ANGSTROM,
    LZ_METHODS,
    PRODUCTION_MD_KEYWORDS,
    PRODUCTION_QDM_KEYWORDS,
    PRODUCTION_STRUCTURAL_KEYS,
    _find_case_yaml,
    _lz_analytic_probability,
    _save_population,
    apply_overrides,
    build_production_input,
    build_reference_input,
    collect_populations,
    emit,
    ensemble_average,
    evaluate_unified,
    load_case,
    parse_reference_population,
)
from .fssh import evaluate_run

__all__ = [
    'AMU_TO_AU',
    'BOHR_TO_ANGSTROM',
    'LZ_METHODS',
    'PRODUCTION_MD_KEYWORDS',
    'PRODUCTION_QDM_KEYWORDS',
    'PRODUCTION_STRUCTURAL_KEYS',
    '_find_case_yaml',
    '_lz_analytic_probability',
    '_save_population',
    'apply_overrides',
    'build_production_input',
    'build_reference_input',
    'collect_populations',
    'emit',
    'ensemble_average',
    'evaluate_run',
    'evaluate_unified',
    'load_case',
    'parse_reference_population',
]
