"""
Run FEM-BV-VARX on system with periodic parameter time-dependence.
"""


from __future__ import division

import argparse
from copy import deepcopy
from math import pi
import matplotlib.pyplot as plt
import numpy as np

from sklearn.utils import check_random_state
from reor.fembv_varx import FEMBVVARX


def generate_data(n_features=2, n_samples=500, period=10, order=1,
                  noise_variance=1, random_state=None):
    """Generate data from a VAR process with time-dependent parameters."""

    rng = check_random_state(random_state)

    X = np.empty((n_samples, n_features), dtype='f8')

    for i in range(order):
        X[i] = rng.normal(size=(n_features,))

    mu = rng.normal(size=(n_features,))

    if order > 0:
        A = rng.normal(size=(order, n_features, n_features))
    else:
        A = None

    phases = np.cos(2 * pi * np.arange(n_samples) / period)
    for t in range(order, n_samples):
        phase = phases[t]
        xt = mu * phase
        if order > 0:
            for m in range(1, order + 1):
                xt += np.dot(A[m - 1] * phase, X[t - m])

        X[t] = rng.multivariate_normal(
            xt, cov=(noise_variance*np.eye(n_features)))

    return X, mu, A, phases


def fit_fembv_varx(X, n_components=2, max_tv_norm=None, memory=0,
                   n_init=10, tolerance=1e-4, max_iterations=500,
                   verbose=0, random_state=None):
    """Fit FEM-BV-VARX model to data."""

    rng = check_random_state(random_state)

    best_fit = None
    best_weights = None

    for i in range(n_init):

        fit = FEMBVVARX(n_components=n_components, max_tv_norm=max_tv_norm,
                        memory=memory, tolerance=tolerance,
                        max_iterations=max_iterations, verbose=verbose,
                        random_state=rng)

        weights = fit.fit_transform(X)

        if best_fit is None or fit.cost_ < best_fit.cost_:
            best_fit = deepcopy(fit)
            best_weights = weights.copy()

    return best_fit, best_weights


def plot_data_timeseries(X, output_file=None):
    """Plot time-series of initial data."""

    n_features = X.shape[1]

    fig, ax = plt.subplots(nrows=n_features, figsize=(9, 4 * n_features),
                           sharex=True, squeeze=False)

    for i in range(n_features):
        ax[i, 0].plot(X[:, i], '-')

        ax[i, 0].set_ylabel(r'$x_{:d}$'.format(i + 1))

        ax[i, 0].grid(ls='--', color='gray', alpha=0.5)

        if i == n_features - 1:
            ax[i, 0].set_xlabel('Time')

    if output_file is not None and output_file:
        plt.savefig(output_file, bbox_inches='tight')

    plt.show()


def plot_weights_timeseries(weights, phases, output_file=None):
    """Plot time-series of FEM-BV-VARX weights."""

    n_components = weights.shape[1]

    fig, ax = plt.subplots(nrows=n_components, figsize=(9, 4 * n_components),
                           sharex=True, squeeze=False)

    for i in range(n_components):
        ax[i, 0].plot(weights[:, i], '-')
        ax[i, 0].plot(phases, '-', color='gray', alpha=0.7)

        ax[i, 0].set_ylabel(r'$\gamma_{:d}$'.format(i + 1))

        ax[i, 0].grid(ls='--', color='gray', alpha=0.5)

        if i == n_components - 1:
            ax[i, 0].set_xlabel('Time')

    if output_file is not None and output_file:
        plt.savefig(output_file, bbox_inches='tight')

    plt.show()


def parse_cmd_line_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description='Run FEM-BV-VARX on system with periodic time-dependence')

    parser.add_argument('--n-components', dest='n_components', type=int,
                        default=2, help='number of FEM-BV-VARX components')
    parser.add_argument('--n-init', dest='n_init', type=int, default=10,
                        help='number of initializations to try')
    parser.add_argument('--max-tv-norm', dest='max_tv_norm', type=float,
                        default=None, help='maximum TV norm bound')
    parser.add_argument('--memory', dest='memory', type=int, default=1,
                        help='maximum memory in FEM-BV-VARX models')
    parser.add_argument('--tolerance', dest='tolerance', type=float,
                        default=1e-4, help='convergence tolerance')
    parser.add_argument('--max-iterations', dest='max_iterations', type=int,
                        default=500, help='maximum number of iterations')
    parser.add_argument('--n-features', dest='n_features', type=int, default=2,
                        help='dimensionality of data')
    parser.add_argument('--n-samples', dest='n_samples', type=int, default=500,
                        help='length of time-series')
    parser.add_argument('--period', dest='period', type=float, default=10,
                        help='period of time-dependent parameters')
    parser.add_argument('--order', dest='order', type=int, default=1,
                        help='order of AR process to generate data with')
    parser.add_argument('--noise-variance', dest='noise_variance', type=float,
                        default=1.0, help='magnitude of noise variance')
    parser.add_argument('--random-seed', dest='random_seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--plot-data', dest='plot_data', action='store_true',
                        help='plot time-series of initial data')
    parser.add_argument('--plot-weights', dest='plot_weights', action='store_true',
                        help='plot FEM-BV-VARX weights')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='produce verbose output')

    args = parser.parse_args()

    if args.n_components < 1:
        raise ValueError('Number of components must be at least 1')

    if args.n_init < 1:
        raise ValueError('Number of initializations must be at least 1')

    if args.max_tv_norm is not None and args.max_tv_norm < 0:
        raise ValueError('Maximum TV norm bound must be non-negative')

    if args.memory < 0:
        raise ValueError('FEM-BV-VARX memory must be non-negative')

    if args.tolerance <= 0:
        raise ValueError('Convergence tolerance must be positive')

    if args.max_iterations < 1:
        raise ValueError('Maximum number of iterations must be at least 1')

    if args.n_features < 1:
        raise ValueError('Number of data features must be at least 1')

    if args.n_samples < 1:
        raise ValueError('Number of samples must be at least 1')

    if args.period <= 0:
        raise ValueError('Period must be positive')

    if args.order < 0:
        raise ValueError('Order of generating process must be non-negative')

    if args.noise_variance <= 0:
        raise ValueError('Noise variance must be positive')

    return args


def main():
    """Run FEM-BV-VARX on system with periodic time-dependence."""

    args = parse_cmd_line_args()

    random_state = check_random_state(args.random_seed)

    X, mu, A, phases = generate_data(args.n_features, n_samples=args.n_samples,
                                     period=args.period, order=args.order,
                                     noise_variance=args.noise_variance,
                                     random_state=random_state)

    if args.plot_data:
        plot_data_timeseries(X)

    best_fit, best_weights = fit_fembv_varx(
        X, n_components=args.n_components,
        max_tv_norm=args.max_tv_norm,
        memory=args.memory, n_init=args.n_init,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        verbose=args.verbose, random_state=random_state)

    if args.plot_weights:
        plot_weights_timeseries(best_weights, phases)


if __name__ == '__main__':
    main()
