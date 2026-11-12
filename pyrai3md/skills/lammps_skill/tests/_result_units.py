"""
======================================================================
Local unit-conversion helper for the LAMMPS skill's own tests.

The tests used to import `workflow/records/conversion.py` to check the
native-to-framework conversion. That made the skill's test suite depend
on the framework, while a skill must be distributable on its own. The
helper below applies the same factors, read from this skill's own
`service/utils.py`, so the test still checks the parser's native output
against the skill's declared conversion.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from ..service.utils import FORCE_TO_HARTREE_PER_BOHR, KCAL_TO_HARTREE

NATIVE_UNITS_MARKER = '_native_units'


def apply_result_units(result, skill_name='lammps_skill'):
    """Apply the LAMMPS `real`/`metal` factors to one native result."""
    if not isinstance(result, dict) or NATIVE_UNITS_MARKER not in result:
        return result
    out = dict(result)
    unit_set = str(out.pop(NATIVE_UNITS_MARKER)).lower().strip()
    if unit_set == 'real':
        energy_factor = KCAL_TO_HARTREE
    else:
        raise ValueError(
            "test helper only carries the 'real' unit set, got %r"
            % unit_set)
    gradient_factor = FORCE_TO_HARTREE_PER_BOHR[unit_set]
    energy = out.get('energy')
    if energy is not None:
        out['energy'] = np.asarray(energy, dtype=np.float64) * energy_factor
    gradient = out.get('gradient')
    if gradient is not None:
        converted = np.asarray(gradient, dtype=np.float64) * gradient_factor
        out['gradient'] = (converted[np.newaxis, :, :]
                           if converted.ndim == 2 else converted)
    for key in list(out):
        if not str(key).startswith('alternative_'):
            continue
        channel = str(key)[len('alternative_'):]
        value = out.get(key)
        if value is None:
            continue
        if channel == 'energy':
            out[key] = np.asarray(value, dtype=np.float64) * energy_factor
        elif channel in ('gradient', 'nac'):
            out[key] = np.asarray(value, dtype=np.float64) * gradient_factor
    return out
