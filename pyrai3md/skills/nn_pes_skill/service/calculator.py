"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the serving side of the nn_pes skill: it opens the
ensembles a profile asks for and answers one geometry with the
properties those networks were trained to produce.

The container is opened once (backend initialization) and every later
call only swaps the geometry, so a dynamics step costs one compiled
forward pass per ensemble member. Each ensemble contributes exactly the
outputs its network family produces: an energy network gives energies,
an energy-gradient network gives energies and gradients, a coupling
network gives couplings. A dynamics run that needs all three asks a
composite method ('eg_nac', 'eg_nac2'), which loads both ensembles from
the same model folder.

"""

import os

import numpy as np

from ..schema.method import (PROFILE_ENSEMBLES, ROLE_ENERGY, ROLE_NAC,
                             ROLE_SOC)
from ..schema.reference import BOHR_TO_ANGSTROM

# Output unit of the served gradient and couplings, stated in the
# manifest. The container returns the derivative with respect to the
# coordinate unit it was fed (measured against a central difference of
# its own energy: median 0.99998, p10 0.99906, p90 1.00094), and that
# unit is Angstrom, so the boundary multiplies by BOHR_TO_ANGSTROM.
GRADIENT_TO_AU = BOHR_TO_ANGSTROM


def _model_channels(container, folder, model_type):
    """Return how many output channels a loaded ensemble produces.

    The width is a fact about the trained network, not about the request:
    `mlp_nac` builds one channel per electronic state, because its virtual
    potential scheme has one output per state, while `mlp_nac2` trains one
    output per declared coupling channel. Any network in the spin-orbit
    folder has one output per declared spin-orbit pair.

    Parameters
    ----------
    container : object
        Opened container.
    folder : str
        Ensemble folder name.
    model_type : str
        Container model type.

    Returns
    -------
    int or None
        Channel count, or None when the ensemble carries no hyper dict.
    """
    members = (getattr(container, '_models_hyper', None) or {}).get(folder)
    if not members:
        return None
    model = members[0].get('model', {})
    key = 'nnac' if (folder == ROLE_NAC and model_type == 'mlp_nac2') \
        else 'states'
    count = model.get(key)
    return int(count) if count is not None else None


def _declared_channels(config, folder):
    """Return the channel count the config declares for one role."""
    return {'nac': config.system.nnac, 'soc': config.system.nsoc,
            'energy_grad': config.system.nstate}[folder]


def _scalar_axis(array, pairs, pair_map):
    """Select spin-orbit channels in the order the caller declared.

    A spin-orbit value is one scalar per pair, so the reported shape is
    (npair, 1): a row per declared pair, matching what the propagator reads
    off the results of a spin-orbit capable backend.

    Parameters
    ----------
    array : numpy.ndarray
        Container output with the batch axis, shape (1, nchannel).
    pairs : list of list of int
        Requested spin-orbit pairs, in the order the result must report.
    pair_map : list of list of int
        The pair each trained channel corresponds to. Like the coupling
        channels, these are declared by the caller's coupling section and
        never enumerated here.

    Returns
    -------
    numpy.ndarray
        Array (npair, 1) in the requested order, in cm-1.

    Raises
    ------
    ValueError
        If a requested pair has no trained channel, or if pairs were
        requested while no pair map was declared.
    """
    data = np.asarray(array, dtype=np.float64)[0]
    if pair_map and len(pair_map) != data.shape[0]:
        raise ValueError(
            "The model produces %d spin-orbit channel(s) but the config "
            "declares %d pair(s): %s. Set the coupling section to the pairs "
            "the SOC values were calculated for."
            % (data.shape[0], len(pair_map),
               [list(pair) for pair in pair_map]))
    if not pairs:
        return np.asarray([[float(value)] for value in data])
    if not pair_map:
        raise ValueError(
            "Spin-orbit values were requested for %s but the config declares "
            "no spin-orbit pair map: without a coupling section the trained "
            "channels carry no pair names."
            % [list(pair) for pair in pairs])
    index = {tuple(sorted(pair)): position
             for position, pair in enumerate(pair_map)}
    rows = []
    for pair in pairs:
        key = tuple(sorted(int(state) for state in pair))
        if key not in index:
            raise ValueError(
                "Requested spin-orbit pair %s has no trained channel; the "
                "model folder declares channels for %s"
                % (list(key), [list(declared) for declared in pair_map]))
        rows.append([float(data[index[key]])])
    return np.asarray(rows)


def check_predict_mode(config, container):
    """Refuse a predictor selection the loaded folder cannot serve.

    This is the third layer: the skill schema checks the declared
    `mult_nn`, adaptive preflight checks the same declaration, and this
    function checks what the folder really loaded. A folder with one
    member cannot provide a comparison model even if the config says two.
    """
    mode = int(getattr(config.calculation, 'nn_predict_mode', 0) or 0)
    if mode == 0:
        return
    actual = getattr(container, '_addNN', None)
    if actual is None:
        models = getattr(container, '_models', None)
        if isinstance(models, dict) and models:
            actual = max(len(value) for value in models.values()
                         if isinstance(value, (list, tuple)))
    actual = int(actual or 0)
    if actual < 2:
        raise ValueError(
            "nn_predict_mode=%d needs at least 2 loaded ensemble members; "
            "the model folder holds %d" % (mode, actual))
    if mode > actual:
        raise ValueError(
            "nn_predict_mode=%d names a member outside the %d loaded "
            "ensemble members" % (mode, actual))


def open_container(config, NeuralNetPes):
    """Load the ensembles the profile asks for.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration.
    NeuralNetPes : class
        The container class, imported by the caller after the runtime
        environment has been checked.

    Returns
    -------
    object
        Container with the profile's ensembles loaded.
    """
    calculation = config.calculation
    folder = os.path.abspath(os.path.expanduser(calculation.modeldir))
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"modeldir does not exist: {folder}. Train the model with "
            "jobtype 'train' first, or point modeldir at a folder that "
            "holds hyper_v*.json, weights_v*.weights.h5 and scaler_v*.json.")

    ensembles = PROFILE_ENSEMBLES[calculation.profile]
    names = [name for name, _kind in ensembles]
    container = NeuralNetPes(folder, mult_nn=calculation.mult_nn)
    container.load(names)
    check_predict_mode(config, container)

    # A folder is only servable when its training finished. The parent
    # writes freshly initialized weights before it starts a fit, so a run
    # whose children died still leaves loadable .weights.h5 files that
    # would predict noise; the per-member fit error file is what the
    # training writes last, and its absence means the folder is a
    # half-product.
    for name in names:
        try:
            container._read_fit_error(name)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "Model folder %s for ensemble %r is not servable: %s"
                % (folder, name, error))

    # The coupling channels are read from the pairs the caller declared,
    # so the declaration has to match the network the folder holds. A
    # folder trained for another pair set would otherwise be reported
    # under the wrong state names.
    for name, model_type in ensembles:
        if name == ROLE_ENERGY:
            continue
        built = _model_channels(container, name, model_type)
        declared = int(_declared_channels(config, name))
        if built is not None and built != declared:
            raise ValueError(
                "Ensemble %r in %s produces %d channel(s) but the config "
                "declares %d for that role. Set the coupling section to the "
                "pairs this model was trained on."
                % (name, folder, built, declared))
    return container


def _state_axis(array, states):
    """Select state columns of a (1, nstate, ...) array.

    Parameters
    ----------
    array : numpy.ndarray
        Container output including the batch axis.
    states : list of int
        Requested state indices.

    Returns
    -------
    numpy.ndarray
        Array without the batch axis, holding the requested states.
    """
    data = np.asarray(array, dtype=np.float64)
    if data.ndim < 2:
        return data[0] if data.ndim == 1 and len(states) == 1 else data
    return data[0][states, ...]


def _coupling_axis(array, pairs, channel_map):
    """Select coupling channels in the order the caller asked for.

    Parameters
    ----------
    array : numpy.ndarray
        Coupling output with the batch axis, shape (1, nchannel, na, 3).
    pairs : list of list of int
        Requested state pairs, in the order the result must report.
    channel_map : list of list of int
        The pair each trained channel corresponds to. The model carries
        one channel per pair the caller declared in its coupling section,
        so this map is that declaration; nothing here enumerates pairs of
        its own. It can be empty when only the channel count was given,
        and then the channels are served in the order the model has them.

    Returns
    -------
    numpy.ndarray
        Array (npair, natom, 3) in the requested order.

    Raises
    ------
    ValueError
        If a requested pair is not one of the trained channels, or if a
        pair was requested while no pair map was declared.
    """
    data = np.asarray(array, dtype=np.float64)[0]
    if channel_map and len(channel_map) != data.shape[0]:
        raise ValueError(
            "The model produces %d coupling channel(s) but the config "
            "declares %d pair(s): %s. Set the coupling section to the pairs "
            "this model was trained on."
            % (data.shape[0], len(channel_map),
               [list(pair) for pair in channel_map]))
    if not pairs:
        return data
    if not channel_map:
        raise ValueError(
            "Couplings were requested for %s but the config declares no "
            "pair map: without a coupling section the trained channels "
            "carry no pair names, so they cannot be selected by pair. "
            "Declare the pairs the model was trained on."
            % [list(pair) for pair in pairs])
    index = {tuple(sorted(pair)): position
             for position, pair in enumerate(channel_map)}
    wanted = []
    for pair in pairs:
        key = tuple(sorted(int(state) for state in pair))
        if key not in index:
            raise ValueError(
                "Requested coupling pair %s has no trained channel; the "
                "model folder declares channels for %s"
                % (list(key), [list(declared) for declared in channel_map]))
        wanted.append(data[index[key]])
    return np.asarray(wanted)


def _member_channels(model_type, output, name, states, channel_map,
                     requested_pairs, requested_soc, soc_map):
    """The reported channels of one member's output, in framework units.

    The layout is the one ``evaluate`` reads from an ensemble mean, so the
    primary prediction and every alternative are produced by the same
    extraction. Adding a channel here adds it to both sides at once.
    """
    if model_type == 'mlp_eg':
        energies, gradients = output[0], output[1]
        return {
            'energy': _state_axis(energies, states),
            'gradient': _state_axis(gradients, states) * GRADIENT_TO_AU,
        }
    if model_type == 'mlp_e':
        if name == ROLE_SOC:
            # The spin-orbit folder holds the same network shape as the
            # energy one but its channels are pairs, not states, and the
            # values are cm-1 scalars rather than a derivative, so no unit
            # factor is applied to them.
            return {'soc': _scalar_axis(output, requested_soc, soc_map)}
        return {'energy': _state_axis(output, states)}
    if model_type == 'mlp_g2':
        return {'gradient': _state_axis(output, states) * GRADIENT_TO_AU}
    if model_type in ('mlp_nac', 'mlp_nac2'):
        return {'nac': _coupling_axis(output, requested_pairs, channel_map)
                * GRADIENT_TO_AU}
    raise ValueError(f"Unhandled model type {model_type!r}")


def evaluate(config, coordinates, container=None):
    """Answer one geometry with the properties of the profile.

    Parameters
    ----------
    config : SkillConfig
        Validated configuration; supplies profile, ensemble names, the
        requested states, the requested coupling pairs and the predictor
        selection (``calculation.nn_predict_mode``).
    coordinates : list of list of float
        Atomic coordinates (natom, 3) in Angstrom.
    container : object, optional
        An opened container. When omitted, one is opened here, which is
        the standalone path used by the agent.

    Returns
    -------
    dict
        Results keyed by the Properties field names: 'energy' (nstate,)
        Hartree, 'gradient' (nstate, natom, 3) Hartree/Bohr, 'nac'
        (npair, natom, 3) Hartree/Bohr in the requested pair order, and
        'soc' (npair, 1) cm-1. With ``nn_predict_mode=k>0`` the prediction
        is member k (1-based) and every reported channel gains an
        ``alternative_<channel>`` array whose leading axis is the other
        M-1 members; ``alternative_model_ids`` names them 1-based.
    dict
        Ensemble spread of the same shapes, for the log and the
        uncertainty file. It is not reported to the framework.
    """
    if container is None:
        from .runner.env_check import import_container
        container = open_container(config, import_container())

    calculation = config.calculation
    profile = calculation.profile
    geometry = np.asarray(coordinates, dtype=np.float32)
    if geometry.ndim != 2 or geometry.shape[1] != 3:
        raise ValueError("coordinates must be (natom, 3)")

    mode = int(getattr(calculation, 'nn_predict_mode', 0) or 0)
    member_map = None
    if mode > 0 and hasattr(container, 'call_with_members'):
        # One inference pass gives the mean, the spread and the members;
        # the mean/spread are still the full committee's, so err_* does not
        # change when the predictor selection changes.
        means, spreads, member_map = container.call_with_members(
            geometry.reshape(1, *geometry.shape))
    else:
        means, spreads = container.call(geometry.reshape(1, *geometry.shape))

    # Every state of the trained surface is reported: Properties.energy
    # is documented as (nstate,) and the propagator indexes it by state
    # number, so slicing it down to the requested states would misalign
    # the caller. target_states only bounds what must exist, which the
    # schema already checked.
    states = list(range(config.system.nstate))
    channel_map = config.system.nac_pair_map
    requested_pairs = calculation.target_nac_pair
    soc_map = config.system.soc_pair_map
    requested_soc = calculation.target_soc_pair

    results = {}
    uncertainty = {}
    for name, model_type in PROFILE_ENSEMBLES[profile]:
        mean = means[name]
        spread = spreads[name]
        outputs = member_map.get(name) if member_map else None
        if outputs and 0 < mode <= len(outputs):
            primary = outputs[mode - 1]
        else:
            primary = mean
            outputs = None

        channels = _member_channels(
            model_type, primary, name, states, channel_map,
            requested_pairs, requested_soc, soc_map)
        results.update(channels)

        if model_type == 'mlp_eg':
            uncertainty['energy'] = _state_axis(spread[0], states)
            uncertainty['gradient'] = (
                _state_axis(spread[1], states) * GRADIENT_TO_AU)
        elif model_type == 'mlp_e':
            if name == ROLE_SOC:
                uncertainty['soc'] = _scalar_axis(
                    spread, requested_soc, soc_map)
            else:
                uncertainty['energy'] = _state_axis(spread, states)
        elif model_type in ('mlp_g2', 'mlp_nac', 'mlp_nac2'):
            if model_type == 'mlp_g2':
                uncertainty['gradient'] = (
                    _state_axis(spread, states) * GRADIENT_TO_AU)
            else:
                uncertainty['nac'] = (
                    _coupling_axis(spread, requested_pairs, channel_map)
                    * GRADIENT_TO_AU)

        if outputs is not None:
            others = [output for index, output in enumerate(outputs, start=1)
                      if index != mode]
            for channel in channels:
                results['alternative_' + channel] = np.stack(
                    [_member_channels(
                        model_type, output, name, states, channel_map,
                        requested_pairs, requested_soc, soc_map)[channel]
                     for output in others], axis=0)
            results['alternative_model_ids'] = [
                index for index in range(1, len(outputs) + 1)
                if index != mode]
            results['predict_model_id'] = mode
            results['n_models'] = len(outputs)

    if mode == 0 or not any(key.startswith('alternative_')
                            for key in results):
        # The mean path (or a container that cannot expose members): the
        # comparison channels exist but carry no data. The adaptive layer
        # is the one that decides not_requested vs missing.
        for channel in ('energy', 'gradient', 'nac', 'soc'):
            if channel in results:
                results['alternative_' + channel] = None
        results['alternative_model_ids'] = None
        results['predict_model_id'] = 0
        actual = getattr(container, '_addNN', None)
        if actual is None:
            actual = getattr(container, 'mult_nn', calculation.mult_nn)
        results['n_models'] = int(actual or 1)

    return results, uncertainty
