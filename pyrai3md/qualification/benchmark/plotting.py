"""
======================================================================
Population plot helper for the benchmark task modules.

Draws the S0 population curves of the compared population tables
persisted by the benchmark evaluators (production.act.pop /
production.pop / reference.pop) into one matplotlib figure saved as
population_s0.png inside the run directory, with the '<method>_<model>'
case key as the figure title (for example fssh_lz1 or lzsh_tully1;
plot_title() derives it from the case production section) and a
fixed S0 probability axis (y range 0 to 1, ticks every 0.2) so the
figures of different calibrations stay directly comparable. The task
modules call the helper right after the population tables are saved
and before the report returns (the fssh, gsh and lzsh evaluators;
the ktdc evaluator inherits it through the shared fssh evaluator);
the perf backend comparison carries no reference curve and never
plots.

The helper is diagnostic only: a missing matplotlib installation,
a missing or unreadable population table or any rendering failure
degrades to a 'skipped' entry in the report details and never
affects the calibration verdict. The Agg backend keeps the helper
headless-safe on compute nodes without a display.

Sep 2 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os

import numpy as np


# Population tables to overlay (file name, legend label, line
# color): the active-state ensemble, the electronic ensemble
# average and the exact reference, each plotted as its S0 column
# against the shared time column.
_POPULATION_SERIES = (
    ('production.act.pop', 'production active', 'goldenrod'),
    ('production.pop', 'production ensemble', 'royalblue'),
    ('reference.pop', 'reference', 'crimson'),
)


def plot_title(prod):
    """Compose the '<method>_<model>' figure title of one case.

    The method is the surface-hopping algorithm of the production
    input (the sfhp keyword: fssh/gsh/lzsh; the ktdc coupling
    variant reports its nactype instead, since its sfhp is fssh),
    the model is the qdm benchmark model (the method keyword:
    tully1/lz1/...), so the title matches the benchmark case key
    convention (fssh_lz1, ktdc_lz1, lzsh_tully1, ...) and figures
    of different calibrations stay directly comparable.

    Parameters
    ----------
    prod : dict
        The production section of the benchmark case YAML.

    Returns
    -------
    str
        The '<method>_<model>' title (sfhp defaults to 'fssh' for
        legacy production sections without the keyword).
    """
    method = 'ktdc' if str(prod.get('nactype', '')).lower() == 'ktdc' \
        else prod.get('sfhp', 'fssh')
    return '%s_%s' % (method, prod.get('method', 'unknown'))


def plot_population_s0(run_dir, title):
    """Draw the S0 population curves of one benchmark run.

    Reads whichever of the three persisted population tables exist
    in the run directory (time column + per-state population
    columns) and overlays their S0 columns on one figure saved as
    population_s0.png next to them.

    Parameters
    ----------
    run_dir : str
        Benchmark run directory holding the population tables.
    title : str
        Figure title: the '<method>_<model>' benchmark case key
        (see plot_title()).

    Returns
    -------
    dict
        {'path': <png path>} when the figure was saved, or
        {'skipped': <reason>} when matplotlib is unavailable, no
        population table exists or the rendering failed. The
        helper never raises: plotting is diagnostic only and must
        not affect the calibration verdict.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return {'skipped': 'matplotlib unavailable: %s' % exc}

    curves = []
    for fname, label, color in _POPULATION_SERIES:
        path = os.path.join(run_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            data = np.loadtxt(path)
        except (OSError, ValueError):
            continue  # unreadable table: plot the remaining curves
        data = np.atleast_2d(data)
        if data.shape[1] < 2:
            continue
        curves.append((label, color, data[:, 0], data[:, 1]))
    if not curves:
        return {'skipped': 'no population tables found in %s'
                % run_dir}

    try:
        fig, axis = plt.subplots(figsize=(8, 5))
        for label, color, time_col, pop_col in curves:
            axis.plot(time_col, pop_col, color=color, label=label)
        axis.set_title(title)
        axis.set_xlabel('time (au)')
        axis.set_ylabel('S0 population')
        # Fixed probability axis: the S0 population is a probability
        # in [0, 1]; pinning the range and the 0.2 ticks keeps the
        # figures of different calibrations directly comparable.
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks(np.arange(0.0, 1.01, 0.2))
        axis.legend()
        png_path = os.path.join(run_dir, 'population_s0.png')
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return {'skipped': 'plot failed: %s' % exc}
    return {'path': png_path}
