"""
======================================================================
This module provides the NN prediction deviation reception channel
for iterative workflows.

The framework never selects the deviation method, distinguishes the
deviation type or computes deviations (the per-component RMS scheme
and the per-state standard deviation scheme both belong to the NN
skill): it always tries to fetch the deviation of every supported
property from the backend result and stays tolerant to missing
values. A deviation key that is absent or None is filled with the
scalar 0.0 so every recorded step carries the complete key set,
never interrupting the run.

Aug 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

# Deviation keys received from the backend, one per supported
# property (legacy naming convention).
DEVIATION_KEYS = (
    'err_energy',
    'err_grad',
    'err_nac',
    'err_soc',
    'err_dipole',
    'err_trans_dipole',
)


def fetch_deviations(result):
    """Fetch the prediction deviations from one backend result.

    Tolerant fetch: a key that is absent, None or unreadable is
    filled with the scalar 0.0; present values pass through as is
    (no shape assumption, no numerical processing). The input is
    never mutated.

    Parameters
    ----------
    result : Result or dict or None
        Backend result of the current step.

    Returns
    -------
    dict
        {deviation key: value} covering every DEVIATION_KEYS entry.
    """
    data = dict(result) if result is not None else {}
    deviations = {}
    for key in DEVIATION_KEYS:
        try:
            value = data.get(key)
        except (AttributeError, TypeError):
            value = None
        deviations[key] = value if value is not None else 0.0
    return deviations


def merge_deviations(results, deviations):
    """Fill missing deviation keys into a result dictionary.

    Keys already provided by the backend are kept untouched; only
    absent keys receive the fetched value (typically the 0.0 fill).

    Parameters
    ----------
    results : dict
        Result dictionary to update in place (e.g. pmd.results).
    deviations : dict
        Deviation values produced by fetch_deviations().
    """
    if results is None:
        return
    for key in DEVIATION_KEYS:
        if key not in results:
            results[key] = deviations.get(key, 0.0)


def judge_deviations(deviations, thresholds):
    """Judge received deviations against the input thresholds.

    The judgment rule follows the legacy _chkerror semantics: a
    deviation exceeding its threshold is flagged (error > threshold
    is True; the equality boundary stays False). The function only
    judges and never triggers any action.

    Parameters
    ----------
    deviations : dict
        {deviation key: value} from fetch_deviations().
    thresholds : dict or None
        {deviation key: threshold}; missing or non-numerical
        thresholds leave the entry unjudged.

    Returns
    -------
    dict
        {deviation key: (error, threshold, exceeded)} for every key
        of deviations; threshold is None and exceeded False when the
        threshold is unavailable or the error is not a number.
    """
    thresholds = thresholds or {}
    judged = {}
    for key, error in (deviations or {}).items():
        threshold = thresholds.get(key)
        exceeded = False
        try:
            exceeded = float(error) > float(threshold)
        except (TypeError, ValueError):
            threshold = None
        judged[key] = (error, threshold, exceeded)
    return judged
