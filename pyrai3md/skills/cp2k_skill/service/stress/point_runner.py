"""
======================================================================
This is a standalone CP2K calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module executes ONE strain single point of the 'num' route.

It is written as a module-level function taking a plain dictionary
because the point may run in another process: a bound method would drag
the parent backend -- and with it its compiled workflow, its live runner
and its calculation directory -- into every payload handed to a worker,
the same reason the framework's batch children are named by module and
function rather than by method.

The worker therefore rebuilds the backend from the plain config carried
in the payload and lets it own its directory end to end, including the
cleanup of that directory.

"""


import os


def run_energy_point(payload):
    """Run one perturbed single point and return its state energies.

    Parameters
    ----------
    payload : dict
        Task descriptor built by num_stress.build_payloads(), with:
        'config' -- plain skill config of this point (deformed cell and
        coordinates, energy-only profile, stress off, no nested pool);
        'calcdir' -- the point's own directory;
        'atoms' / 'coordinates' -- the deformed geometry to evaluate;
        'label' -- direction/sign label, e.g. 'xy_plus';
        'slurm_template' -- the user's submission script of the central
        run, if any, carried over under this point's own stem.

    Returns
    -------
    dict
        {'label', 'energy' (nstate Hartree), 'excitation' (nstate eV)}
        from the parsed result of the point.

    Raises
    ------
    RuntimeError
        Propagated from the backend when CP2K exits non-zero, so a
        failed point fails the stress evaluation instead of feeding a
        zero energy into the difference.
    """
    # Imported here rather than at module scope: the worker process
    # builds its own backend, and the driver module must stay importable
    # without pulling the whole backend chain into the parent.
    from ...backend import Cp2kBackend

    template = payload.get('slurm_template')
    if template:
        # Create the directory first: the runner looks for the template
        # inside it, and the backend only makes it when it writes.
        os.makedirs(payload['calcdir'], exist_ok=True)
        stem = payload['config']['calculation']['project']
        with open(os.path.join(payload['calcdir'], f'{stem}.slurm'),
                  'w') as handle:
            handle.write(template)

    backend = Cp2kBackend(payload['config'], calcdir=payload['calcdir'])
    results = backend.calculate(atoms=payload['atoms'],
                                 coordinates=payload['coordinates'])
    return {
        'label': payload['label'],
        'energy': results['energy'],
        'excitation': results.get('excitation'),
    }
