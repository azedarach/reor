"""
Fit k-means based PWVARX model to SISO system.
"""


import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.utils import check_random_state


from reor.pwvarx import ClusterPWVARX


def generate_data(n_samples, random_state=None):
    """Generate time-series of data to fit."""

    rng = check_random_state(random_state)

    y = np.zeros((n_samples,))
    y[0] = rng.uniform(low=0.0, high=1.0)

    e = np.zeros((n_samples,))

    u = rng.uniform(size=(n_samples,), low=-5.0, high=5.0)

    mat_one = np.array([4, -1, 10])
    mat_two = np.array([5, 1, -6])

    for k in range(1, n_samples):

        e[k] = rng.uniform(low=-0.1, high=0.1)
        x_kbar = np.array([y[k - 1], u[k - 1], 1])

        if np.dot(mat_one, x_kbar) < 0:
            params = np.array([-0.4, 1, 1.5])
        elif np.dot(-mat_one, x_kbar) <= 0 and np.dot(mat_two, x_kbar) <= 0:
            params = np.array([0.5, -1, -0.5])
        elif np.dot(-mat_two, x_kbar) < 0:
            params = np.array([-0.3, 0.5, -1.7])
        else:
            raise ValueError('Invalid region')

        y[k] = np.dot(params, x_kbar) + e[k]

    return y, u, e


def plot_initial_data(y, u, e):
    """Plot initial data to be fitted."""

    fig, ax = plt.subplots(nrows=3, ncols=1, squeeze=False,
                           figsize=(7, 5))

    ax[0, 0].plot(y, 'b-')

    ax[0, 0].grid(ls='--', color='gray', alpha=0.5)

    ax[0, 0].set_xlabel(r'$k$')
    ax[0, 0].set_ylabel(r'$y_k$')

    ax[1, 0].plot(u, 'b-')

    ax[1, 0].grid(ls='--', color='gray', alpha=0.5)

    ax[1, 0].set_xlabel(r'$k$')
    ax[1, 0].set_ylabel(r'$u_k$')

    ax[2, 0].plot(e, 'b-')

    ax[2, 0].grid(ls='--', color='gray', alpha=0.5)

    ax[2, 0].set_xlabel(r'$k$')
    ax[2, 0].set_ylabel(r'$e_k$')

    plt.show()


def plot_cluster_assignments(y, u, p, s, labels):
    """Plot true and fitted cluster assignments."""

    unique_components = np.unique(labels)
    n_samples = np.size(y)
    presample_length = max(p, s)

    if np.size(labels) != n_samples - presample_length:
        raise ValueError(
            'Number of labels does not match number of fitted points')

    n_predictors = p + s

    if p > 0:
        y_lag = np.zeros((n_samples - presample_length, p))

        for i in range(1, p + 1):
            y_lag[:, i - 1] = y[presample_length - i:-i]
    else:
        y_lag = None

    if s > 0:
        u_lag = np.zeros((n_samples - presample_length, s))

        for i in range(1, s + 1):
            u_lag[:, i - 1] = u[presample_length - i:-i]
    else:
        u_lag = None

    n_plots = int(n_predictors * (n_predictors - 1) / 2)
    n_cols = 1
    n_rows = int(np.ceil(n_plots / n_cols))

    min_y = np.min(y)
    max_y = np.max(y)
    min_u = np.min(u)
    max_u = np.max(u)

    y_boundary = np.linspace(min_y, max_y, n_samples)
    boundary_1 = 4 * y_boundary + 10
    boundary_2 = -5 * y_boundary + 6

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False,
                           figsize=(7 * n_cols, 4 * n_rows))
    if n_plots > 1:
        fig.subplots_adjust(bottom=0.05, top=0.95, hspace=0.2)

    row_index = 0
    col_index = 0

    if p > 1:
        for i in range(p):
            for j in range(i + 1, p):
                y1 = y_lag[:, i]
                y2 = y_lag[:, j]

                markers = itertools.cycle(('.', 's', '+', 'x', 'd', 'o'))
                for k in unique_components:
                    mask = labels == k

                    ax[row_index, col_index].plot(
                        y1[mask], y2[mask], ls='none', marker=next(markers))

                ax[row_index, col_index].grid(ls='--', color='gray', alpha=0.5)

                ax[row_index, col_index].set_xlim(min_y, max_y)
                ax[row_index, col_index].set_ylim(min_y, max_y)

                ax[row_index, col_index].set_xlabel(
                    r'$y_{{k - {:d}}}$'.format(i + 1))
                ax[row_index, col_index].set_ylabel(
                    r'$y_{{k - {:d}}}$'.format(j + 1))

                col_index += 1
                if col_index >= n_cols:
                    col_index = 0
                    row_index += 1

    if s > 1:
        for i in range(s):
            for j in range(i + 1, s):
                u1 = u_lag[:, i]
                u2 = u_lag[:, j]

                markers = itertools.cycle(('.', 's', '+', 'x', 'd', 'o'))
                for k in unique_components:
                    mask = labels == k

                    ax[row_index, col_index].plot(
                        u1[mask], u2[mask], ls='none', marker=next(markers))

                ax[row_index, col_index].grid(ls='--', color='gray', alpha=0.5)

                ax[row_index, col_index].set_xlim(min_u, max_u)
                ax[row_index, col_index].set_ylim(min_u, max_u)

                ax[row_index, col_index].set_xlabel(
                    r'$u_{{k - {:d}}}$'.format(i + 1))
                ax[row_index, col_index].set_ylabel(
                    r'$u_{{k - {:d}}}$'.format(j + 1))

                col_index += 1
                if col_index >= n_cols:
                    col_index = 0
                    row_index += 1

    if p > 0 and s > 0:
        for i in range(p):
            for j in range(s):
                yi = y_lag[:, i]
                uj = u_lag[:, j]

                markers = itertools.cycle(('.', 's', '+', 'x', 'd', 'o'))
                for k in unique_components:
                    mask = labels == k

                    ax[row_index, col_index].plot(
                        yi[mask], uj[mask], ls='none', marker=next(markers))

                ax[row_index, col_index].plot(y_boundary, boundary_1, 'b--')
                ax[row_index, col_index].plot(y_boundary, boundary_2, 'b-.')

                ax[row_index, col_index].grid(ls='--', color='gray', alpha=0.5)

                ax[row_index, col_index].set_xlim(min_y, max_y)
                ax[row_index, col_index].set_ylim(min_u, max_u)

                ax[row_index, col_index].set_xlabel(
                    r'$y_{{k - {:d}}}$'.format(i + 1))
                ax[row_index, col_index].set_ylabel(
                    r'$u_{{k - {:d}}}$'.format(j + 1))

                col_index += 1
                if col_index >= n_cols:
                    col_index = 0
                    row_index += 1

    plt.show()


def parse_cmd_line_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description='Fit k-means based PWVARX model to SISO system')

    parser.add_argument('--n-samples', dest='n_samples', type=int,
                        default=500, help='number of samples')
    parser.add_argument('--n-components', dest='n_components', type=int,
                        default=2, help='number of components to fit')
    parser.add_argument('-p,--p', dest='p', type=int,
                        default=1, help='VAR memory')
    parser.add_argument('-s,--s', dest='s', type=int,
                        default=0, help='exogeneous factors memory')
    parser.add_argument('--n-init', dest='n_init', type=int, default=50,
                        help='number of times to repeat fit')
    parser.add_argument('--include-trend', dest='include_trend',
                        action='store_true', help='include intercept term')
    parser.add_argument('--plot-data', dest='plot_data',
                        action='store_true', help='show plots of initial data')
    parser.add_argument('--plot-cluster-assignments', dest='plot_cluster_assignments',
                        action='store_true', help='plot assigned clusters')

    return parser.parse_args()


def main():
    """Fit cluster based PWVARX model to SISO system."""

    args = parse_cmd_line_args()

    y, u, e = generate_data(args.n_samples)

    if args.plot_data:
        plot_initial_data(y, u, e)

    n_init = args.n_init
    if args.include_trend:
        fitted_mu = np.zeros((n_init, args.n_components, 1))
    else:
        fitted_mu = None

    if args.p > 0:
        fitted_A = np.zeros((n_init, args.n_components, args.p, 1, 1))
    else:
        fitted_A = None

    if args.s > 0:
        fitted_B = np.zeros((n_init, args.n_components, args.s, 1, 1))
    else:
        fitted_B = None

    cluster_model = GaussianMixture(n_components=args.n_components, n_init=10)
    for i in range(n_init):
        model = ClusterPWVARX(n_components=args.n_components,
                              p=args.p, s=args.s, include_trend=args.include_trend,
                              clusterer=cluster_model)

        labels = model.fit_transform(y, exog=u)

        component_ordering = None
        if args.include_trend:
            component_ordering = np.squeeze(np.argsort(model.mu, axis=0))
            fitted_mu[i] = model.mu[component_ordering]

        if args.p > 0:
            if component_ordering is None:
                component_ordering = np.squeeze(np.argsort(model.A, axis=0))
            fitted_A[i] = model.A[component_ordering]

        if args.s > 0:
            if component_ordering is None:
                component_ordering = np.squeeze(np.argsort(model.B, axis=0))
            fitted_B[i] = model.B[component_ordering]

    if args.include_trend:
        mean_mu = np.mean(fitted_mu, axis=0)
        std_mu = np.std(fitted_mu, axis=0)
    else:
        mean_mu = None
        std_mu = None

    if args.p > 0:
        mean_A = np.mean(fitted_A, axis=0)
        std_A = np.std(fitted_A, axis=0)
    else:
        mean_A = None
        std_A = None

    if args.s > 0:
        mean_B = np.mean(fitted_B, axis=0)
        std_B = np.std(fitted_B, axis=0)
    else:
        mean_B = None
        std_B = None

    for k in range(args.n_components):
        print(' *** Component {:d}:'.format(k + 1))
        if args.include_trend:
            print('    mu = {:16.8e} (sd = {:16.8e})'.format(
                mean_mu[k, 0], std_mu[k, 0]))
        if args.p > 0:
            for i in range(args.p):
                print('    A[{:d}] = {:16.8e} (sd = {:16.8e})'.format(
                    i + 1, mean_A[k, i, 0, 0], std_A[k, i, 0, 0]))
        if args.s > 0:
            for i in range(args.s):
                print('    B[{:d}] = {:16.8e} (sd = {:16.8e})'.format(
                    i + 1, mean_B[k, i, 0, 0], std_B[k, i, 0, 0]))

    if args.plot_cluster_assignments:
        plot_cluster_assignments(y, u, args.p, args.s, labels)


if __name__ == '__main__':
    main()
