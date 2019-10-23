"""
Provides routines for FEM-BV-k-means clustering method.
"""

import numbers
import numpy as np

from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from reor.fembv import FEMBV
from reor._validation import _check_array_shape


INTEGER_TYPES = (numbers.Integral, np.integer)


def _check_init_parameters(Theta, shape, whom):
    Theta = check_array(Theta)
    _check_array_shape(Theta, shape, whom)


def _initialize_fembv_kmeans_random(X, n_components, random_state=None):
    """Return random initial cluster parameters.

    The cluster parameters (centroids) are chosen as random rows of the
    data matrix X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to be fitted

    n_components : integer
        The number of clusters desired

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    Returns
    -------
    Theta : array-like, shape (n_components, n_features)
        Random initial guess for cluster centroids
    """

    n_samples, n_features = X.shape
    rng = check_random_state(random_state)

    rows = rng.choice(n_samples, n_components, replace=False)
    Theta = X[rows]

    return Theta


class FEMBVKMeans(FEMBV):
    """FEM-BV-k-means clustering of data."""

    def __init__(self, n_components, max_tv_norm, init='random',
                 max_iterations=500, tolerance=1e-4, verbose=0,
                 random_state=None, **kwargs):
        super().__init__(n_components=n_components, max_tv_norm=max_tv_norm,
                         init=init, max_iterations=max_iterations,
                         tolerance=tolerance, verbose=verbose,
                         random_state=random_state, **kwargs)

        self.name = 'FEM-BV-k-means'
        self.X = None
        self.Theta = None

    def _evaluate_distance_matrix(self):
        for j in range(self.n_components):
            self.distance_matrix[:, j] = np.linalg.norm(
                self.X - np.broadcast_to(self.Theta[j], self.X.shape),
                axis=1) ** 2

    def _initialize_components(self, data, parameters=None, init=None, **kwargs):
        """Generate initial guess for component parameters."""

        self.X = data.copy()

        n_samples, n_features = self.X.shape

        self.distance_matrix = np.empty(
            (n_samples, self.n_components), dtype=self.X.dtype)

        if init == 'custom' and parameters is not None:
            _check_init_parameters(
                parameters, shape=(self.n_components, n_features),
                whom='_initialize_components (input parameters)')

            self.Theta = parameters.copy()
        else:
            self.Theta = _initialize_fembv_kmeans_random(
                self.X, self.n_components, random_state=self.random_state)

        self._evaluate_distance_matrix()

    def _update_parameters(self):
        """Update k-means cluster assignments."""

        self.Theta = np.dot(self.Gamma.T, self.X)
        normalizations = np.sum(self.Gamma, axis=0)
        self.Theta = self.Theta / normalizations[:, np.newaxis]

        self._evaluate_distance_matrix()
