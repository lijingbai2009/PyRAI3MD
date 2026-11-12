"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 04 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module turns the ensemble spread of one step into what the framework
and a reader are owed: one number per computed property, and a line in the
skill's own log.

An ensemble of neural network potentials has no error bar of its own, so
the spread between its members is used as one. The container returns the
standard deviation across the members with `ddof=1` -- for the two-member
ensemble that is the difference of the two predictions divided by the
square root of two, which is the scheme the original project reported. The
framework then compares it against the `&adaptive` thresholds and stops or
records a geometry when the surface is being asked to extrapolate.

What the framework receives per property is therefore a single scalar: the
largest member-to-member standard deviation over the components of that
property. A maximum rather than a mean because a threshold is a guard on
the worst component of the step, which is what the original project tested
with `np.amax`.

"""

import numpy as np

# One deviation key per computed property, named as the framework's
# reception channel names them (workflow/records/deviation.py:DEVIATION_KEYS).
DEVIATION_KEYS = {
    'energy': 'err_energy',
    'gradient': 'err_grad',
    'nac': 'err_nac',
    'soc': 'err_soc',
}

# The unit each deviation is expressed in, which is the unit the property
# itself is reported in. The number is a spread of the same quantity, so it
# carries the same unit and no conversion happens here.
DEVIATION_UNITS = {
    'err_energy': 'hartree',
    'err_grad': 'hartree/bohr',
    'err_nac': 'hartree/bohr',
    'err_soc': 'cm-1',
}


def deviation_metrics(spreads, members=2):
    """Reduce one step's ensemble spread to one number per property.

    Parameters
    ----------
    spreads : dict
        Property name to the member-to-member standard deviation array of
        this step, as the serving layer returns it.
    members : int, optional
        Ensemble size. A single network has no spread to report, and every
        deviation is then 0.0 rather than absent, so a consumer reads "no
        evidence of trouble" instead of a missing channel.

    Returns
    -------
    dict
        Deviation key to its scalar, for the properties that were
        computed. A property the profile did not produce is absent here,
        exactly like its value channel.
    """
    metrics = {}
    for property_name, key in DEVIATION_KEYS.items():
        spread = (spreads or {}).get(property_name)
        if spread is None:
            continue
        values = np.asarray(spread, dtype=np.float64)
        if values.size == 0 or members < 2:
            metrics[key] = 0.0
            continue
        metrics[key] = float(np.max(values))
    return metrics


def format_step(profile, atoms, results, metrics, natom=None):
    """Format one served step for the skill's own log.

    The log carries the numbers the backend returned, not a restatement of
    the configuration: what was computed, how big the member disagreement
    was on each of those channels, and in which unit. A property the
    profile does not produce gets no line, because a zero line would read
    as a computed value.

    Parameters
    ----------
    profile : str
        Profile that served the step.
    atoms : list of str
        Element symbols of the geometry.
    results : dict
        Property name to the reported values.
    metrics : dict
        Deviation key to its scalar, from deviation_metrics().
    natom : int, optional
        Atom count; derived from the geometry when not given.

    Returns
    -------
    str
        The step's log lines, ending in a newline.
    """
    natom = natom if natom is not None else len(atoms or [])
    results = results or {}
    # np.asarray(None) is a size-one array holding nan, so an absent energy
    # channel has to be recognized before the conversion: otherwise a step
    # that computed no energy at all (a training run) logs one state whose
    # value is nan.
    reported = results.get('energy')
    energy = (np.asarray(reported, dtype=np.float64).ravel()
              if reported is not None else np.empty(0))
    lines = ['step  profile=%s  atoms=%d  states=%d'
             % (profile, natom, int(energy.size))]
    if energy.size:
        lines.append('  %-24s %s  hartree'
                     % ('energy (per state)',
                        ' '.join('%.8f' % float(v) for v in energy)))
    for property_name in ('gradient', 'nac', 'soc'):
        values = results.get(property_name)
        if values is None:
            continue
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            continue
        unit = DEVIATION_UNITS[DEVIATION_KEYS[property_name]]
        lines.append('  %-24s %.8f  %s'
                     % ('%s (largest |component|)' % property_name,
                        float(np.max(np.abs(array))), unit))
    for key in sorted(metrics):
        lines.append('  %-24s %.8f  %s'
                     % ('%s (max member-to-member std)' % key,
                        float(metrics[key]), DEVIATION_UNITS[key]))
    return '\n'.join(lines) + '\n'
