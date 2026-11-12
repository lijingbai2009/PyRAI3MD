"""
======================================================================
This is a standalone machine learning force field skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

A stand-in for the container, so a function test can stop paying for an
engine run.

The skill's real route -- train in subprocesses, serve through a compiled
Keras model -- is certified by the case files in this directory
(``00_mlp_eg_train.yaml`` and friends), which the qualification suite runs
through the three entry points. Those are where a real fit belongs.

Most of what this package also tests in python is not that route. Which
channels a prediction record carries, how a raw file is laid out, which
directory a calculation leaves behind when ``keep_tmp`` is off, whether a
serving request that names a test file is routed to the evaluator: every
one of those is a function of the framework's own code, and each becomes
*more* strongly checkable when the numbers are known analytically -- an
exact equality instead of a tolerance, a difference of two known columns
instead of a reconstruction. That is what the container below provides:
the same ``call()`` contract the real one offers, answering with a surface
whose derivative is its own energy, so the axis conversions under test are
the only thing left between an input and an expectation.

Patching the environment guard (``validate_environment``) is part of the
same decision, not an accident: the guard says whether TensorFlow can be
imported, and it has its own gate in
``tests/test_discovery_without_tensorflow.py``.

"""

import json

import numpy as np

from pyrai3md.skills.nn_pes_skill.schema.reference import BOHR_TO_ANGSTROM

# The offset between what the fake surface answers and what the label file
# it is compared against carries. One pair of numbers, so an error can be
# asserted rather than measured.
ENERGY_SHIFT = 1.0
GRADIENT_SHIFT = 0.5


def surface(coordinates):
    """The fake ensemble's energy and gradient for one geometry.

    ``energy`` is the summed square of the coordinates, lifted by the state
    index, and ``gradient`` is its exact derivative with respect to the
    Angstrom coordinates the container is fed. Keeping that pair consistent
    means a test cannot pass by accident on a per-Bohr/per-Angstrom mix-up:
    the factor has to be applied, and applied the right way round, for the
    asserted difference to come out at the shift below.

    Parameters
    ----------
    coordinates : numpy.ndarray
        (natom, 3) coordinates in Angstrom.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray)
        Energy (nstate,) in Hartree and gradient (nstate, natom, 3) in
        Hartree per Angstrom.
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    base = float(np.sum(coords ** 2))
    return base, 2.0 * coords


class FakeContainer:
    """A container-shaped object answering ``call()`` like the real one.

    Parameters
    ----------
    nstate : int
        Electronic states the energy family spans.
    natom : int
        Atoms in the geometries it will be asked about.
    nnac : int
        Coupling channels, for a profile that serves them.
    mult_nn : int
        Ensemble members; the spread is a fixed fraction of the answer per
        member, so a two-member ensemble reports a nonzero disagreement.
    """

    def __init__(self, nstate=2, natom=2, nnac=0, mult_nn=2):
        self.nstate = nstate
        self.natom = natom
        self.nnac = nnac
        self.mult_nn = mult_nn
        self.calls = 0
        # What the loader of a real folder would have filled in.
        self._models_hyper = {}
        self._models_scaler = {}

    def set_hyper(self, name, states=None, nnac=None):
        """Declare one ensemble's channels the way ``load`` would."""
        hyper = {'model': {'atoms': self.natom,
                           'states': self.nstate if states is None else states,
                           'nnac': self.nnac if nnac is None else nnac},
                 'general': {'model_type': 'mlp_eg'},
                 'training': {'initialize_weights': True},
                 'retraining': {'initialize_weights': False}}
        self._models_hyper[name] = [dict(hyper) for _ in range(self.mult_nn)]

    def call(self, geometries):
        """Answer a batch of geometries with the fake surface.

        Parameters
        ----------
        geometries : numpy.ndarray
            (nframe, natom, 3) coordinates in Angstrom, the shape the real
            container is called with.

        Returns
        -------
        tuple(dict, dict)
            (means, spreads) keyed by ensemble name: the energy family as
            ``(energy, gradient)`` and a coupling family as one array, with
            the shapes and units the real container reports.
        """
        self.calls += 1
        frames = np.atleast_2d(np.asarray(geometries, dtype=np.float64))
        if frames.ndim == 2:
            frames = frames.reshape(1, frames.shape[0], 3)
        count = frames.shape[0]

        energies = np.empty((count, self.nstate))
        gradients = np.empty((count, self.nstate, self.natom, 3))
        for index, geometry in enumerate(frames):
            base, gradient = surface(geometry)
            energies[index] = [base + state for state in range(self.nstate)]
            gradients[index] = np.repeat(gradient[np.newaxis, :, :],
                                         self.nstate, axis=0)

        spread = 0.05 if self.mult_nn > 1 else 0.0
        means = {'energy_grad': (energies, gradients)}
        spreads = {'energy_grad': (np.full(energies.shape, spread),
                                   np.full(gradients.shape, spread))}
        if self.nnac:
            couplings = np.empty((count, self.nnac, self.natom, 3))
            for index, geometry in enumerate(frames):
                _, gradient = surface(geometry)
                couplings[index] = np.repeat(gradient[np.newaxis, :, :],
                                             self.nnac, axis=0)
            means['nac'] = couplings
            spreads['nac'] = np.full(couplings.shape, spread)
        return means, spreads

    def call_with_members(self, geometries):
        """Means, spreads and per-member outputs of the fake surface.

        Each member is the surface plus a small member-dependent offset, so
        a test can assert which member the predictor selection returned and
        which members landed in ``alternative_*``.
        """
        frames = np.atleast_2d(np.asarray(geometries, dtype=np.float64))
        if frames.ndim == 2:
            frames = frames.reshape(1, frames.shape[0], 3)
        count = frames.shape[0]
        energies = []
        gradients = []
        couplings = [] if self.nnac else None
        for member in range(self.mult_nn):
            shift = 0.01 * member
            member_energy = np.empty((count, self.nstate))
            member_gradient = np.empty((count, self.nstate, self.natom, 3))
            for index, geometry in enumerate(frames):
                base, gradient = surface(geometry)
                member_energy[index] = [
                    base + state + shift for state in range(self.nstate)]
                member_gradient[index] = np.repeat(
                    (gradient + 0.001 * shift)[np.newaxis, :, :],
                    self.nstate, axis=0)
            energies.append(member_energy)
            gradients.append(member_gradient)
            if self.nnac:
                member_coupling = np.empty((count, self.nnac, self.natom, 3))
                for index, geometry in enumerate(frames):
                    _, gradient = surface(geometry)
                    member_coupling[index] = np.repeat(
                        (gradient + 0.001 * shift)[np.newaxis, :, :],
                        self.nnac, axis=0)
                couplings.append(member_coupling)

        def reduced(stack):
            array = np.asarray(stack)
            mean = np.mean(array, axis=0)
            spread = (np.std(array, axis=0, ddof=1)
                      if self.mult_nn > 1 else np.zeros_like(mean))
            return mean, spread

        mean_e, spread_e = reduced(energies)
        mean_g, spread_g = reduced(gradients)
        means = {'energy_grad': (mean_e, mean_g)}
        spreads = {'energy_grad': (spread_e, spread_g)}
        members = {'energy_grad': list(zip(energies, gradients))}
        if self.nnac:
            mean_c, spread_c = reduced(couplings)
            means['nac'] = mean_c
            spreads['nac'] = spread_c
            members['nac'] = list(couplings)
        return means, spreads, members


def label_file(path, geometries, energy_shift=ENERGY_SHIFT,
               gradient_shift=GRADIENT_SHIFT, nstate=2):
    """Write a test file whose labels are the fake surface plus a shift.

    The file is written in the units a label file carries -- Hartree
    energies and Hartree per Bohr derivatives -- so a test that expects an
    error of ``gradient_shift`` in eV per Angstrom is asserting the whole
    conversion chain, not restating it.

    Parameters
    ----------
    path : str
        Where to write the JSON file.
    geometries : numpy.ndarray
        (nframe, natom, 3) Angstrom coordinates the fake container answers.
    energy_shift : float
        Hartree added to every energy label.
    gradient_shift : float
        Hartree per Angstrom added to every gradient label.
    nstate : int
        Electronic states per frame.

    Returns
    -------
    tuple(float, float)
        The shifts as the test states them, so the expectation is written
        once.
    """
    frames = np.asarray(geometries, dtype=np.float64)
    energies = []
    gradients = []
    for geometry in frames:
        base, gradient = surface(geometry)
        energies.append([base + state + energy_shift
                         for state in range(nstate)])
        # Hartree per Angstrom -> the per-Bohr axis a file carries.
        gradients.append([[[value * BOHR_TO_ANGSTROM + gradient_shift
                            * BOHR_TO_ANGSTROM for value in row]
                          for row in gradient]
                          for _state in range(nstate)])
    with open(path, 'w') as handle:
        json.dump({'natom': int(frames.shape[1]), 'nstate': nstate,
                   'xyz': [[['%.10f' % value for value in row]
                            for row in frame] for frame in frames],
                   'energy': energies,
                   'grad': [[[list(row) for row in state] for state in frame]
                            for frame in gradients],
                   'unit': {'energy': 'hartree', 'gradient': 'hartree/bohr',
                            'xyz': 'angstrom'}}, handle)
    return energy_shift, gradient_shift


def coupling_label_file(path, geometries, nnac, nstate=2, shift=GRADIENT_SHIFT):
    """Write a coupling test file the same way.

    Parameters
    ----------
    path : str
        Where to write the JSON file.
    geometries : numpy.ndarray
        (nframe, natom, 3) Angstrom coordinates.
    nnac : int
        Coupling channels the profile serves.
    nstate : int
        Electronic states carried alongside the couplings.
    shift : float
        Hartree per Angstrom added to every coupling label.

    Returns
    -------
    float
        The shift, for the assertion that names it.
    """
    frames = np.asarray(geometries, dtype=np.float64)
    energies = []
    couplings = []
    for geometry in frames:
        base, gradient = surface(geometry)
        energies.append([base + state for state in range(nstate)])
        couplings.append([[[value * BOHR_TO_ANGSTROM + shift * BOHR_TO_ANGSTROM
                            for value in row] for row in gradient]
                          for _pair in range(nnac)])
    with open(path, 'w') as handle:
        json.dump({'natom': int(frames.shape[1]), 'nstate': nstate,
                   'nnac': nnac,
                   'xyz': [[['%.10f' % value for value in row]
                            for row in frame] for frame in frames],
                   'energy': energies,
                   'nac': [[list(row) for row in pair] for pair in couplings],
                   'unit': {'nac': 'hartree/bohr', 'energy': 'hartree',
                            'xyz': 'angstrom'}}, handle)
    return shift
