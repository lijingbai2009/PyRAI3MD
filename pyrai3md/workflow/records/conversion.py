"""
======================================================================
Skill result unit conversion for PyRAI3MD.

Every skill hands back the same labels (`energy`, `gradient`,
`stress_tensor`, ...), and a label says what it holds: `gradient` is
dU/dR, never a force. A skill may still compute in its own units and
declare them in its manifest:

    conversion:
      native_energy:   {real: kcal/mol,         metal: eV}
      native_gradient: {real: kcal/mol/Angstrom, metal: eV/Angstrom}

and marks one result with the unit set it used (`_native_units: 'real'`).
The framework then looks the declared unit name up in
`utilities.units` and applies the factor. It never imports a skill module
and never decides by inspection which label to read.

The QM/MM auxiliary blocks stay in native units by contract: they are
audit data and composite input, not framework Properties. Their arrays are
converted through the same factor lookup, so one unit set means one number
everywhere in a run.

Oct 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os

import numpy as np
import yaml

from ...utilities.units import (
    NATIVE_ENERGY_TO_HARTREE,
    NATIVE_GRADIENT_TO_HARTREE_PER_BOHR,
)

# The marker a backend writes to say "this result is in my native units,
# and here is which set". It carries no skill identity: the skill is known
# from the task descriptor.
NATIVE_UNITS_MARKER = '_native_units'

# One resolved (energy, gradient) factor pair per (skill, unit set). A
# manifest cannot change while a run computes, and reading it per result is
# what the measured MD step priced at milliseconds, so the pair is cached
# after the first lookup. An unknown unit name is NOT cached: it is a defect
# and must keep reporting itself.
_UNIT_FACTORS = {}


def _skills_root():
    """Path of the package's skills directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), 'skills')


def _manifest(skill_name):
    """One skill's manifest dictionary ({} when there is none)."""
    path = os.path.join(_skills_root(), skill_name, 'manifest.yaml')
    if not os.path.isfile(path):
        return {}
    with open(path, 'r') as handle:
        return yaml.safe_load(handle) or {}


def _resolve_unit(table, unit, kind, skill_name):
    """Look one unit name up in a framework table, refusing an unknown one."""
    factor = table.get(str(unit).lower().strip())
    if factor is None:
        raise ValueError(
            "skill '%s' declares %s unit %r, which the framework cannot "
            "convert; known units: %s."
            % (skill_name, kind, unit, ', '.join(sorted(table))))
    return factor


def resolve_result_units(skill_name, unit_set):
    """Resolve one skill's declared factors for one unit set.

    Parameters
    ----------
    skill_name : str
        Skill package directory name (e.g. 'lammps_skill').
    unit_set : str
        The run's unit-set name as the backend declared it (e.g. 'real').

    Returns
    -------
    tuple of float
        ``(energy_factor, gradient_factor)``; both 1.0 when the skill
        declares no native units (it already emits framework units).

    Raises
    ------
    ValueError
        When the unit set or one of its unit names is not declared.
    """
    key = (skill_name, str(unit_set).lower().strip())
    cached = _UNIT_FACTORS.get(key)
    if cached is not None:
        return cached

    spec = dict(_manifest(skill_name).get('conversion') or {})
    if not spec:
        _UNIT_FACTORS[key] = (1.0, 1.0)
        return _UNIT_FACTORS[key]

    # An empty unit set means the skill declared the maps but the backend
    # named no set for this result: that is a half-wired skill, not a
    # default, so it is refused instead of silently reading factors of 1.
    if not key[1]:
        raise ValueError(
            "skill '%s' declares native units but its result carries an "
            "empty %s marker." % (skill_name, NATIVE_UNITS_MARKER))

    def _declared(section):
        mapping = dict(spec.get(section) or {})
        unit = None
        for name, value in mapping.items():
            if str(name).lower().strip() == key[1]:
                unit = value
                break
        if unit is None:
            raise ValueError(
                "skill '%s' declares no conversion.%s for unit set %r; "
                "declared sets: %s."
                % (skill_name, section, unit_set,
                   ', '.join(sorted(str(name) for name in mapping))
                   or 'none'))
        return _resolve_unit(
            NATIVE_ENERGY_TO_HARTREE if section == 'native_energy'
            else NATIVE_GRADIENT_TO_HARTREE_PER_BOHR,
            unit, section, skill_name)

    factors = (_declared('native_energy'), _declared('native_gradient'))
    _UNIT_FACTORS[key] = factors
    return factors


def apply_result_units(result, skill_name):
    """Convert one native-unit result into framework units.

    Idempotence is not assumed and is not needed: the marker is removed on
    the first pass, so a second call is the documented no-op that returns
    the caller's own object.

    Parameters
    ----------
    result : dict
        Raw backend result.
    skill_name : str
        Skill package directory name of the task's backend.

    Returns
    -------
    dict
        Converted result, or the original object when it carries no marker.
    """
    if not isinstance(result, dict) or NATIVE_UNITS_MARKER not in result:
        return result
    out = dict(result)
    unit_set = out.pop(NATIVE_UNITS_MARKER)
    energy_factor, gradient_factor = resolve_result_units(skill_name,
                                                          unit_set)
    energy = out.get('energy')
    if energy is not None:
        out['energy'] = (np.asarray(energy, dtype=np.float64)
                         * energy_factor)
    gradient = out.get('gradient')
    if gradient is not None:
        converted = np.asarray(gradient, dtype=np.float64) * gradient_factor
        # The Properties contract is one state axis; a skill that reports a
        # single-state (natom, 3) array gets it promoted here, once.
        out['gradient'] = (converted[np.newaxis, :, :]
                           if converted.ndim == 2 else converted)
    # The comparison channel mirrors the prediction channels and carries
    # the same native units; a skill that reports it gets the same factors.
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


def convert_native_energy(native, unit_set, skill_name):
    """Convert one native energy scalar to Hartree."""
    energy_factor, _ = resolve_result_units(skill_name, unit_set)
    return float(np.asarray(native, dtype=np.float64).reshape(-1)[0]
                 * energy_factor)


def convert_native_gradient(native, unit_set, skill_name):
    """Convert one native-unit gradient array to Hartree/Bohr.

    The QM/MM auxiliary arrays are already gradients: the skill applies the
    sign, so this only scales. The result keeps the shape it was handed
    (per-QM-atom arrays carry no state axis).
    """
    _, gradient_factor = resolve_result_units(skill_name, unit_set)
    return (np.asarray(native, dtype=np.float64) * gradient_factor)
