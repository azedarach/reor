"""
Provides routines for fitting piecewise VARX models.
"""


from __future__ import division

from copy import deepcopy
import numpy as np

from sklearn.cluster import KMeans
from sklearn.svm import SVC


class ClusterPWVARX():
    """Piecewise VARX model with clustering of phase space.

    Parameters
    ----------
    n_components : integer or None
        If an integer, the number of components. If None, then all features
        are kept.

    max_iterations : integer, default: 500
        Maximum number of iterations before stopping.

    tolerance : float, default: 1e-4
        Tolerance of the stopping condition.

    verbose : integer, default: 0
        The verbosity level.

    random_state : integer, RandomState, or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    References
    ----------
    H. Nakada, K. Takaba, and T. Katayama, "Identification of piecewise
    affine systems based on statistical clustering technique",
    Automatica 41 (2005), 905 - 913 (doi:10.1016/j.automatica.2004.12.005)
    """

    def __init__(self, n_components, p, s=0, include_trend=True,
                 clusterer=None, classifier=None, **kwargs):

        self.n_components = n_components
        self.p = p
        self.s = s
        self.include_trend = include_trend
        self.clusterer = clusterer
        self.classifier = classifier

        self.mu = None
        self.A = None
        self.B = None
        self.Sigma_u = None
        self.residuals = None

    def _stack_predictors(self, endog, exog=None):

        n_samples = endog.shape[0]

        if endog.ndim == 1:
            n_endog = 1
            endog = np.expand_dims(endog, -1)
        else:
            n_endog = endog.shape[1]

        if exog is not None:
            if exog.ndim == 1:
                n_exog = 1
                exog = np.expand_dims(exog, -1)
            else:
                n_exog = exog.shape[1]
        else:
            n_exog = 0

        n_predictors = self.p * n_endog + self.s * n_exog

        presample_length = max(self.p, self.s)

        x = np.zeros((n_samples - presample_length, n_predictors))
        col_index = 0
        for i in range(1, self.p + 1):
            x[:, col_index : col_index + n_endog] = \
                endog[presample_length - i:-i]
            col_index += n_endog

        if exog is not None and self.s > 0:
            for i in range(1, self.s + 1):
                x[:, col_index : col_index + n_exog] = \
                    exog[presample_length - i:-i]
                col_index += n_exog

        return x

    def _cluster(self, y, x):

        z = np.hstack([x, y])

        if self.clusterer is None:
            self.clusterer = KMeans(n_clusters=self.n_components)

        self.cluster_labels_ = self.clusterer.fit_predict(z)

    def _fit_var_model(self, y, x, labels):

        n_endog = y.shape[1]

        if self.include_trend:
            self.mu = np.zeros((self.n_components, n_endog))

        if self.p > 0:
            self.A = np.zeros((self.n_components, self.p, n_endog, n_endog))

        n_lagged_exog = x.shape[1] - self.p * n_endog

        assert n_lagged_exog >= 0

        if n_lagged_exog > 0:
            n_exog = int(n_lagged_exog / self.s)
            self.B = np.zeros((self.n_components, self.s, n_endog, n_exog))
        else:
            n_exog = 0

        self.Sigma_u = np.zeros((self.n_components, n_endog, n_endog))
        self.residuals = np.zeros_like(y)
        for k in range(self.n_components):

            component_mask = labels == k
            n_component_samples = np.sum(component_mask)

            Y = y[component_mask]

            if self.include_trend:
                X = np.hstack([np.ones((n_component_samples, 1)),
                               x[component_mask]])
            else:
                X = x[component_mask]

            params = np.linalg.lstsq(X, Y, rcond=None)[0]

            row_start = 0
            if self.include_trend:
                self.mu[k] = params[0]
                row_start += 1

            if self.p > 0:
                A = np.reshape(params[row_start:row_start + n_endog * self.p],
                               (self.p, n_endog, n_endog))
                self.A[k] = A.swapaxes(1, 2).copy()
                row_start += n_endog * self.p

            if n_exog > 0 and self.s > 0:
                B = np.reshape(params[row_start:],
                               (self.s, n_exog, n_endog))
                self.B[k] = B.swapaxes(1, 2).copy()

            self.residuals[component_mask] = Y - np.dot(X, params)

            df = Y.shape[0] - self.p * n_endog
            if self.include_trend:
                df -= 1
            if n_exog > 0:
                df -= n_exog * self.s

            self.Sigma_u[k] = (np.dot(self.residuals[component_mask].T,
                                      self.residuals[component_mask]) / df)

    def fit_transform(self, endog, exog=None, **kwargs):
        """Fit k-means PWVARX model and return transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        endog : array-like, shape (n_samples, n_endog)
            Matrix of endogeneous (modelled) variables to be fit.

        exog : None or array-like, shape (n_samples, n_exog)
            If given, matrix of exogeneous variable values.

        Returns
        -------
        labels : array-like, shape (n_samples, n_components)
            Representation of the data.
        """

        if endog.ndim == 1:
            n_endog = 1
        else:
            n_endog = endog.shape[1]

        presample_length = max(self.p, self.s)

        if n_endog == 1:
            y = np.expand_dims(endog[presample_length:], -1)
        else:
            y = endog[presample_length:]

        x = self._stack_predictors(endog, exog=exog)

        self._cluster(y, x)

        if self.classifier is None:
            self.classifier = SVC()

        self.classifier.fit(x, self.cluster_labels_)
        labels = self.classifier.predict(x)

        self._fit_var_model(y, x, labels)

        return labels

    def fit(self, endog, exog=None, **kwargs):
        """Fit k-means PWVARX model.

        Parameters
        ----------
        endog : array-like, shape (n_samples, n_endog)
            Matrix of endogeneous variables to be fit.

        Returns
        -------
        self
        """

        self.fit_transform(endog, exog=exog, **kwargs)
        return self

    def transform(self, endog, exog=None):
        """Transform data according to the fitted classifier.

        Parameters
        ----------
        endog : array-like, shape (n_samples, n_endog)
            Matrix of endogeneous variables to be fit.

        Returns
        -------
        labels : array-like, shape (n_samples,)
            Labels of the given data.
        """

        x = self._stack_predictors(endog, exog=exog)

        return self.classifier.predict(x)
