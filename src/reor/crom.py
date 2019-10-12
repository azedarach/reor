"""
Provides routines for cluster-based model order reduction.
"""

from __future__ import division

import numpy as np

from sklearn.cluster import KMeans


def kmeans_markov_crom(X, n_clusters=None, sample_weight=None, **kwargs):
    """Fit k-means CROM with Markov transitions.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to be used in calculating the reduction.

    n_clusters : integer or None
        If an integer, the number of clusters. If None,
        defaults to 2 clusters.

    sample_weight : array-like, shape (n_samples,), optional
        The weights for each observation in X. If None, all observations
        are assigned equal weight.

    **kwargs :
        Additional arguments to be passed to k-means clustering
        routine.

    Returns
    -------
    kmeans : KMeans object
        Result of performing k-means clustering.

    transition_matrix : array, shape (n_clusters, n_clusters)
        Fitted Markov transition matrix.
    """

    kmeans = KMeans(n_clusters=n_clusters, **kwargs).fit(
        X, sample_weight=sample_weight)

    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    n_samples = labels.size
    n_clusters = cluster_centers.shape[0]

    transition_matrix = np.zeros((n_clusters, n_clusters), dtype='f8')

    for j in range(n_clusters):
        Tj = np.array(labels == j, dtype='i8')
        for k in range(n_clusters):
            Tk = np.array(labels == k, dtype='i8')
            n_jk = (Tj[1:] * Tk[:-1]).sum()
            n_k = Tk[:-1].sum()
            transition_matrix[j, k] = n_jk / n_k

    return kmeans, transition_matrix


class KMeansMarkovCROM():
    """k-means based CROM with Markov transitions.

    Parameters
    ----------
    n_clusters : integer or None
        If an integer, the number of clusters. If None,
        defaults to 2 clusters.

    **kwargs :
        Additional arguments to be passed to k-means clustering
        routine.

    Attributes
    ----------
    cluster_centers : array, shape (n_clusters, n_features)
        Array containing the locations of the cluster centers.

    labels : array, shape (n_samples)
        Array containing the cluster assignment of each sample.

    inertia : float
        Sum of squared distances of samples to their closest
        cluster center.

    n_iter : int
        Number of iterations run.

    transition_matrix : array, shape (n_clusters, n_clusters)
        Fitted Markov transition matrix.

    Examples
    --------
    import numpy as np
    X = np.random.rand(10, 4)
    from reor.crom import KMeansMarkovCROM
    model = KMeansMarkovCROM(n_clusters=4, random_state=0).fit(X)
    """
    def __init__(self, n_clusters, **kwargs):
        self.n_clusters = n_clusters
        self.kmeans_kwargs = kwargs

    def fit(self, X, sample_weight=None):
        """Perform k-means based model reduction.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to be used to fit reduced order model.

        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        self
        """

        kmeans, transition_matrix = kmeans_markov_crom(
            X, n_clusters=self.n_clusters, sample_weight=sample_weight,
            **self.kmeans_kwargs)

        self.cluster_centers_ = kmeans.cluster_centers_
        self.labels_ = kmeans.labels_
        self.inertia_ = kmeans.inertia_
        self.n_iter_ = kmeans.n_iter_
        self.kmeans_ = kmeans
        self.transition_matrix_ = transition_matrix

        return self

    def fit_predict(self, X, sample_weight=None):
        """Perform model reduction and predict cluster index for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to be used to fit reduced order model.

        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : array, shape (n_samples,)
            Array of labels for each sample
        """

        return self.kmeans_.fit_predict(X, sample_weight=sample_weight)

    def fit_transform(self, X, sample_weight=None):
        """Perform model reduction and transform data to cluster-distance space.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to be used to fit reduced order model.

        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        distances : array, shape (n_clusters, n_samples)
            Array of distances of each sample to the cluster centroids.
        """

        return self.kmeans_.fit_transform(X, sample_weight=sample_weight)

    def predict(self, X, sample_weight=None, horizon=0):
        """Predict cluster assignment of each sample for the given horizon.

        If the requested forecast is horizon, then the label for each
        sample is computed. Otherwise, the probability distribution
        after the number of timesteps is computed using the fitted
        transition matrix.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to evaluate prediction for.

        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        horizon : integer
            If given, the number of timesteps over which to predict
            the distribution of cluster labels.

        Returns
        -------
        labels : array, shape (n_clusters, n_samples)
            Array containing the probability that each sample is
            associated with each cluster after the given number of
            time-steps.
        """

        # Get cluster assignments at the current time-step
        initial_labels = self.kmeans_.predict(X, sample_weight=sample_weight)

        n_clusters = self.n_clusters
        n_samples = initial_labels.size

        labels = np.zeros((n_clusters, n_samples), dtype='f8')
        for i in range(n_clusters):
            labels[i, initial_labels == i] = 1

        if horizon == 0:
            return labels

        for m in range(1, horizon + 1):
            labels = self.transition_matrix_.dot(labels)

        return labels

    def transform(self, X):
        """Transform X to a cluster-distance space.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to be transformed.

        Returns
        -------
        distances : array, shape (n_samples, n_clusters)
            Array containing distance of each sample to each cluster centroid.
        """

        return self.kmeans_.transform(X)
