"""
Run FEM-BV-k-means on data consisting of Gaussian clusters.
"""

from __future__ import division

import argparse
from copy import deepcopy
import itertools
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MultipleLocator
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from reor.fembv_kmeans import FEMBVKMeans


N_FEATURES = 2
N_CLUSTERS = 3
CLUSTER_1_MEAN = np.array([0, 0.5])
CLUSTER_1_COVARIANCE = np.array([[0.001, 0], [0, 0.02]])
CLUSTER_2_MEAN = np.array([0, -0.5])
CLUSTER_2_COVARIANCE = np.array([[0.001, 0], [0, 0.02]])
CLUSTER_3_MEAN = np.array([0.25, 0])
CLUSTER_3_COVARIANCE = np.array([[0.002, 0], [0, 0.3]])

CLUSTER_MEANS = np.vstack([CLUSTER_1_MEAN, CLUSTER_2_MEAN, CLUSTER_3_MEAN])
CLUSTER_COVARIANCES = np.stack(
    [CLUSTER_1_COVARIANCE, CLUSTER_2_COVARIANCE, CLUSTER_3_COVARIANCE], axis=0)


def generate_data(n_samples, n_switches=2, random_state=None):
    """Generate test data."""

    rng = check_random_state(random_state)

    run_length = int(np.floor(n_samples / (n_switches + 1)))

    Gamma = np.zeros((n_samples, N_CLUSTERS))
    cluster = 0
    for i in range(n_switches):
        Gamma[i * run_length:(i + 1) * run_length, cluster] = 1
        cluster = (cluster + 1) % N_CLUSTERS
    Gamma[(n_switches) * run_length:, cluster] = 1

    X = np.zeros((n_samples, N_FEATURES))
    for i in range(n_samples):
        r = np.zeros((N_CLUSTERS, N_FEATURES))
        for j in range(N_CLUSTERS):
            r[j] = rng.multivariate_normal(CLUSTER_MEANS[j],
                                           CLUSTER_COVARIANCES[j])
        X[i, :] = np.dot(Gamma[i, :], r)

    return X, Gamma


def run_kmeans(X, n_clusters=2, n_init=10, random_state=None):
    """Run k-means clustering."""

    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init,
                    random_state=random_state).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_


def fit_fembv_kmeans(X, n_components=2, max_tv_norm=None,
                     n_init=10, tolerance=1e-4, max_iterations=500,
                     verbose=0, random_state=None):
    """Fit FEM-BV-k-means model to data."""

    rng = check_random_state(random_state)

    best_fit = None
    best_weights = None

    for i in range(n_init):

        fit = FEMBVKMeans(n_components=n_components, max_tv_norm=max_tv_norm,
                          tolerance=tolerance,
                          max_iterations=max_iterations, verbose=verbose,
                          random_state=rng)

        weights = fit.fit_transform(X)

        if best_fit is None or fit.cost_ < best_fit.cost_:
            best_fit = deepcopy(fit)
            best_weights = weights.copy()

    return best_fit, best_weights


def plot_results(X, Gamma_true, kmeans_centroids, kmeans_labels,
                 fembv_centroids, fembv_affs, output_file=None):
    """Plot clustering results."""

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11, 8), squeeze=False)
    fig.subplots_adjust(hspace=0.3)

    n_samples = X.shape[0]
    true_affs = np.zeros(n_samples, dtype='i8')

    markers = itertools.cycle(('.', '+', 's'))

    for i in range(N_CLUSTERS):
        true_affs[Gamma_true[:, i] == 1] = i + 1
        cluster_data = X[Gamma_true[:, i] == 1]
        ax[0, 0].plot(cluster_data[:, 0], cluster_data[:, 1],
                      marker=next(markers), linestyle='none',
                      label='Cluster {:d}'.format(i + 1))

    ax[0, 0].set_xlabel(r'$x_1$')
    ax[0, 0].set_ylabel(r'$x_2$')
    ax[0, 0].legend(numpoints=1)
    ax[0, 0].set_title(r'Synthetic data')

    ax[0, 1].plot(true_affs)
    ax[0, 1].yaxis.set_major_locator(MultipleLocator(1))
    ax[0, 1].set_xlabel(r'Time')
    ax[0, 1].set_ylabel(r'Cluster affiliation')
    ax[0, 1].set_title(r'True affiliations')

    colors = itertools.cycle(('r', 'g', 'b', 'y', 'c', 'k'))
    kmeans_n_clusters = kmeans_centroids.shape[0]
    for j in range(kmeans_n_clusters):
        c = next(colors)
        mask = kmeans_labels == j
        for i in range(N_CLUSTERS):
            cluster_data = X[np.logical_and(true_affs == i + 1, mask)]
            ax[1, 0].plot(cluster_data[:, 0], cluster_data[:, 1],
                          marker=next(markers), color=c, linestyle='none')
        ax[1, 0].plot(kmeans_centroids[j, 0], kmeans_centroids[j, 1], 'kx')

    ax[1, 0].set_xlabel(r'$x_1$')
    ax[1, 0].set_ylabel(r'$x_2$')
    ax[1, 0].set_title(r'k-means clusters')

    colors = itertools.cycle(('r', 'g', 'b', 'y', 'c', 'k'))
    fembv_n_clusters = fembv_centroids.shape[0]
    vp = np.argmax(fembv_affs, axis=1)
    for j in range(fembv_n_clusters):
        c = next(colors)
        mask = vp == j
        for i in range(N_CLUSTERS):
            cluster_data = X[np.logical_and(true_affs == i + 1, mask)]
            ax[1, 1].plot(cluster_data[:, 0], cluster_data[:, 1],
                          marker=next(markers), color=c, linestyle='none')
        ax[1, 1].plot(fembv_centroids[j, 0], fembv_centroids[j, 1], 'kx')

    ax[1, 1].set_xlabel(r'$x_1$')
    ax[1, 1].set_ylabel(r'$x_2$')
    ax[1, 1].set_title(r'FEM-BV-k-means clusters')

    if output_file is not None and output_file:
        plt.savefig(output_file, bbox_inches='tight')

    plt.show()
    plt.close()


def parse_cmd_line_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description='Run FEM-BV-k-means on data composed of Gaussian clusters')

    parser.add_argument('--n-samples', dest='n_samples', type=int, default=500,
                        help='number of data points')
    parser.add_argument('--n-switches', dest='n_switches', type=int, default=2,
                        help='number of switches in hidden state')
    parser.add_argument('--n-components', dest='n_components', type=int, default=3,
                        help='number of FEM-BV clusters')
    parser.add_argument('--max-tv-norm', dest='max_tv_norm', type=float,
                        default=None, help='maximum TV norm')
    parser.add_argument('--n-init', dest='n_init', type=int, default=10,
                        help='number of random initializations to use')
    parser.add_argument('--tolerance', dest='tolerance', type=float, default=1e-4,
                        help='convergence tolerance')
    parser.add_argument('--max-iterations', dest='max_iterations', type=int,
                        default=500, help='maximum number of iterations')
    parser.add_argument('--random-seed', dest='random_seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='produce verbose output')

    args = parser.parse_args()

    if args.n_samples <= 0:
        raise ValueError('Number of data points must be a positive integer')

    if args.n_switches < 0:
        raise ValueError('Number of state switches must be non-negative')

    if args.n_components < 1:
        raise ValueError('Number of components must be a positive integer')

    if args.max_tv_norm is not None and args.max_tv_norm < 0:
        raise ValueError('Maximum TV norm must be non-negative')

    if args.n_init < 1:
        raise ValueError('Number of initializations must be a positive integer')

    if args.max_iterations < 1:
        raise ValueError('Maximum number of iterations must be a positive integer')

    if args.tolerance <= 0:
        raise ValueError('Stopping tolerance must be positive')

    return args


def main():
    """Run FEM-BV-k-means on data composed of Gaussian clusters."""

    args = parse_cmd_line_args()

    n_samples = args.n_samples
    n_switches = args.n_switches
    n_components = args.n_components
    n_init = args.n_init
    max_tv_norm = args.max_tv_norm

    random_state = check_random_state(args.random_seed)

    X, Gamma_true = generate_data(n_samples, n_switches=n_switches,
                                  random_state=random_state)

    kmeans_centroids, kmeans_labels = run_kmeans(
        X, n_clusters=N_CLUSTERS, n_init=n_init)
    model, weights = fit_fembv_kmeans(
        X, n_components=n_components, max_tv_norm=max_tv_norm,
        n_init=n_init, tolerance=args.tolerance,
        max_iterations=args.max_iterations, verbose=args.verbose)

    plot_results(X, Gamma_true, kmeans_centroids, kmeans_labels,
                 model.Theta, weights)


if __name__ == '__main__':
    main()
