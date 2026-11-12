"""
The main training script for NAC model. Called with ArgumentParse.
"""
import argparse
import contextlib
import json
import os
import pickle
import sys
import traceback

import numpy as np
import tensorflow as tf

import keras

from ...datasets.general import load_hyp
from ...datasets.general import split_validation_training_index
from ...models.mlp_nac import NACModel
from ...nn_pes_src.device import set_gpu
from ..repro import apply_seed

from ...scaler.general import SegmentStandardScaler
from ...scaler.nac import NACStandardScaler
from ...utils.callbacks import EarlyStopping, lr_lin_reduction, lr_exp_reduction, lr_step_reduction
from ...utils.loss import ScaledMeanAbsoluteError, get_lr_metric, r2_metric, NACphaselessLoss

def parse_args(argv=None):
    """
    Read the command line of one training call.

    Args:
        argv (list, optional): Arguments without the program name.

    Returns:
        dict: Parsed arguments.

    """
    parser = argparse.ArgumentParser(description='Train a nac model from data, '
                                                 'parameters given in a folder')
    parser.add_argument("-i", "--index", required=True, help="Index of the NN to train")
    parser.add_argument("-f", "--filepath", required=True, help="Filepath to weights, "
                                                                "hyperparameter, data etc. ")
    parser.add_argument("-g", "--gpus", default=-1, required=True, help="Index of gpu to use")
    parser.add_argument("-m", "--mode", default="training", required=True,
                        help="Which mode to use train or retrain")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="RNG seed of this instance: the framework sends its\n"
                             "control gl_seed advanced by the member index")
    return vars(parser.parse_args(argv))

def load_data(outdir):
    """
    Read the pickled training data written by the container.

    Args:
        outdir (str): Folder of the model instance.

    Returns:
        tuple: (x, y), or (None, None) when the data cannot be read.

    """
    try:
        with open(os.path.join(outdir, 'data_y'), 'rb') as f:
            y = pickle.load(f)
        with open(os.path.join(outdir, 'data_x'), 'rb') as f:
            x = pickle.load(f)
        return x, y
    except OSError:
        print("Error: Can not load data for fit", outdir)
        return None, None

def train_model_nac(i=0, outdir=None, mode='training'):
    """
    Train NAC model. Uses precomputed feature and model representation.

    Args:
        i (int, optional): Model index. The default is 0.
        outdir (str, optional): Direcotry for fit output. The default is None.
        mode (str, optional): Fitmode to take from hyperparameters. The default is 'training'.

    Raises:
        ValueError: Wrong input shape.

    Returns:
        error_val (list): Validation error for NAC.

    """
    i = int(i)
    # Load everything from folder
    x, y_in = load_data(outdir)
    if x is None:
        return None

    try:
        hyperall = load_hyp(os.path.join(outdir, 'hyper' + '_v%i' % i + ".json"))
    except Exception as error:
        print("Error: Can not load hyper for fit", outdir, error)
        return None

    scaler = NACStandardScaler()
    try:
        scaler.load(os.path.join(outdir, 'scaler' + '_v%i' % i + ".json"))
    except Exception as error:
        print("Error: Can not load scaler info for fit", outdir, error)
        return None

    # Model
    hypermodel = hyperall['model']
    num_outstates = int(hypermodel['states'])
    indim = int(hypermodel['atoms'])
    # plots
    # Fit
    hyper = hyperall[mode]
    phase_less_loss = hyper['phase_less_loss']
    epo = hyper['epo']
    batch_size = hyper['batch_size']
    epostep = hyper['epostep']
    pre_epo = hyper['pre_epo']
    val_disjoint = hyper['val_disjoint']
    val_split = hyper['val_split']
    initialize_weights = hyper['initialize_weights']
    learning_rate = hyper['learning_rate']
    auto_scale = hyper['auto_scaling']
    normalize_feat = int(hyper['normalization_mode'])
    # step
    use_step_callback = hyper['step_callback']
    use_linear_callback = hyper['linear_callback']
    use_exp_callback = hyper['exp_callback']
    use_early_callback = hyper['early_callback']

    # Data Check here:
    if (len(x.shape) != 3):
        raise ValueError("Input x-shape must be (batch,atoms,3)")
    else:
        print("Found x-shape of", x.shape)
    if (len(y_in.shape) != 4):
        raise ValueError("Input nac-shape must be (batch,states,atoms,3)")
    else:
        print("Found nac-shape of", y_in.shape)

    # Set stat dir
    dir_save = os.path.join(outdir, "fit_stats")
    os.makedirs(dir_save, exist_ok=True)

    # cbks,Learning rate schedule
    cbks = []
    if use_early_callback['use']:
        es_cbk = EarlyStopping(**use_early_callback)
        cbks.append(es_cbk)
    if use_linear_callback['use']:
        lr_sched = lr_lin_reduction(**use_linear_callback)
        lr_cbk = keras.callbacks.LearningRateScheduler(lr_sched)
        cbks.append(lr_cbk)
    if use_exp_callback['use']:
        lr_exp = lr_exp_reduction(**use_exp_callback)
        exp_cbk = keras.callbacks.LearningRateScheduler(lr_exp)
        cbks.append(exp_cbk)
    if use_step_callback['use']:
        lr_step = lr_step_reduction(**use_step_callback)
        step_cbk = keras.callbacks.LearningRateScheduler(lr_step)
        cbks.append(step_cbk)

    # Data selection
    lval = int(len(x) * val_split)
    allind = np.arange(0, len(x))
    i_train, i_val = split_validation_training_index(allind, lval, val_disjoint, i)
    print("Info: Train-Test split at Train:", len(i_train), "Test", len(i_val), "Total", len(x))

    # Make all Models
    out_model = NACModel(**hypermodel)
    out_model.precomputed_features = True

    npeps = np.finfo(float).eps
    if (initialize_weights == False):
        try:
            out_model.load_weights(os.path.join(outdir, "weights" + '_v%i' % i + '.weights.h5'))
            print("Info: Load old weights at:", os.path.join(outdir, "weights" + '_v%i' % i + '.weights.h5'))
        except Exception:
            print("Error: Can't load old weights...")
    else:
        print("Info: Making new initialized weights..")

    scaler.fit(x, y_in, auto_scale=auto_scale)
    x_rescale, y = scaler.transform(x=x, y=y_in)

    # Calculate features
    feat_x, feat_grad = out_model.precompute_feature_in_chunks(x_rescale, batch_size=batch_size)

    # Finding Normalization
    feat_x_mean, feat_x_std = out_model.set_const_normalization_from_features(feat_x,normalization_mode=normalize_feat)

    xtrain = [feat_x[i_train], feat_grad[i_train]]
    ytrain = y[i_train]
    xval = [feat_x[i_val], feat_grad[i_val]]
    yval = y[i_val]

    # Set Scaling
    scaled_metric = ScaledMeanAbsoluteError(scaling_shape=scaler.nac_std.shape)
    scaled_metric.set_scale( scaler.nac_std)
    scaler.print_params_info()
    print("Info: Using feature-scale", feat_x_std.shape, ":", feat_x_std)
    print("Info: Using feature-offset", feat_x_mean.shape, ":", feat_x_mean)
    print("")

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    lr_metric = get_lr_metric(optimizer)
    out_model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=[scaled_metric, lr_metric, r2_metric])

    # Pre -fit
    if (pre_epo > 0):
        print("Start Pre-fit without phaseless-loss.")
        print("Used loss:", out_model.loss)
        out_model.summary()
        out_model.fit(x=xtrain, y=ytrain, epochs=pre_epo, batch_size=batch_size, verbose=2)
        print("End fit.")
        print("")

    print("Start fit.")
    if phase_less_loss:
        print("Recompiling with phaseless loss.")
        out_model.compile(
            loss=NACphaselessLoss(number_state=num_outstates, shape_nac=(indim, 3), name='phaseless_loss'),
            optimizer=optimizer,
            metrics=[scaled_metric, lr_metric, r2_metric])
        print("Used loss:", out_model.loss)

    out_model.summary()
    hist = out_model.fit(x=xtrain, y=ytrain, epochs=epo, batch_size=batch_size, callbacks=cbks, validation_freq=epostep,
                         validation_data=(xval, yval), verbose=2)
    print("End fit.")
    print("")

    try:
        print("Info: Saving history...")
        # One history per version: every version trains into this directory.
        outname = os.path.join(dir_save, "history_v%i.json" % i)
        outhist = {a: np.array(b, dtype=np.float64).tolist() for a, b in hist.history.items()}
        with open(outname, 'w') as f:
            json.dump(outhist, f)
    except Exception:
        print("Warning: Cant save history")

    try:
        # Save Weights
        print("Info: Saving weights...")
        out_model.save_weights(os.path.join(outdir, "weights" + '_v%i' % i + '.weights.h5'))
    except Exception:
        print("Error: Cant save weights")

    try:
        print("Info: Saving auto-scaler to file...")
        scaler.save(os.path.join(outdir, "scaler" + '_v%i' % i + '.json'))
    except Exception:
        print("Error: Can not export scaler info. Model prediciton will be wrongly scaled.")

    # error out
    error_val = None

    try:
        print("Info: saving fitting error...")
        # Safe fitting Error MAE
        pval = out_model.predict(xval)
        ptrain = out_model.predict(xtrain)
        _, pval = scaler.inverse_transform(y=pval)
        _, ptrain = scaler.inverse_transform(y=ptrain)
        out_model.precomputed_features = False
        ptrain2 = out_model.predict(x_rescale[i_train])
        _, ptrain2 = scaler.inverse_transform(y=ptrain2)
        print("Info: MAE between precomputed and full keras model:")
        print("NAC", np.mean(np.abs(ptrain - ptrain2)))
        error_val = np.mean(np.abs(pval - y_in[i_val]))
        error_train = np.mean(np.abs(ptrain - y_in[i_train]))
        print("error_val:", error_val)
        print("error_train:", error_train)
        np.save(os.path.join(outdir, "fiterr_valid" + '_v%i' % i + ".npy"), error_val)
        np.save(os.path.join(outdir, "fiterr_train" + '_v%i' % i + ".npy"), error_train)
    except Exception:
        print("Error: Can not save fiterror")

    return error_val

def main(argv=None):
    """
    Run one training call with its output captured in the model folder.

    Args:
        argv (list, optional): Arguments without the program name.

    Returns:
        int: Process exit code, 0 when the fit completed.

    """
    args = parse_args(argv)
    # Seed before a single weight is drawn or a batch order picked: this
    # process is a child, so the parent's random state never arrived here
    # on its own, and the value the launcher put on the command line is
    # the only seed this fit has.
    apply_seed(args['seed'])
    logname = os.path.join(args['filepath'], "fitlog_" + str(args['index']) + ".txt")
    with open(logname, 'w') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            print("Input argparse:", args)
            set_gpu([int(args['gpus'])])
            print("Logic Devices:", tf.config.experimental.list_logical_devices('GPU'))
            try:
                fitted = train_model_nac(args['index'],
                                         args['filepath'],
                                         args['mode'])
                if fitted is None:
                    print("Fit produced no validation error: the training data, "
                          "hyper or scaler did not load for", args['filepath'])
                    return 1
            except Exception:
                traceback.print_exc()
                return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
