cov"""
Provides routines for FEM-BV-VARX clustering method.
"""

import numbers
import numpy as np

from reor.fembv import FEMBV
from reor._validation import _check_array_shape


INTEGER_TYPES = (numbers.Integral, np.integer)


def _check_init_parameters(parameters, n_components, n_features,
                           n_external, lags, whom):

    if 'mu' not in parameters:
        raise ValueError('Initial guess for parameter mu missing in %s' % whom)

    if 'A' not in parameters:
        raise ValueError('Initial guess for parameter A missing in %s' % whom)

    if 'B' not in parameters and n_external != 0:
        raise ValueError('Initial guess for parameter B missing in %s' % whom)

    n_lags = lags.size

    _check_array_shape(parameters['mu'], (n_features,), whom)
    _check_array_shape(parameters['A'],
                       (n_components, n_lags, n_features, n_features), whom)
    if n_external != 0:
        _check_array_shape(parameters['B'],
                           (n_components, n_features, n_external), whom)


def _check_lags(lags):
    """Check given lags consists of a 1D integer array."""

    if np.isscalar(lags):
        lags_array = np.array([lags])
    else:
        lags_array = np.asarray(lags)

    if lags_array.ndim != 1:
        raise ValueError('lags must be a 1D array')

    for lag in lags_array:
        if not isinstance(lag, INTEGER_TYPES) or lag < 1:
            raise ValueError('Lags must only be positive integers;'
                             'got (lags=%r)' % lags)

    return lags_array


class FEMBVVARX(FEMBV):
    """FEM-BV-VARX clustering of data."""

    def __init__(self, n_components, max_tv_norm, lags=None, init='random',
                 max_iterations=500, tolerance=1e-4, verbose=0,
                 random_state=None, **kwargs):
        super().__init__(n_components=n_components, max_tv_norm=max_tv_norm,
                         init=init, max_iterations=max_iterations,
                         tolerance=tolerance, verbose=verbose,
                         random_state=random_state, **kwargs)

        if lags is None:
            self.order = 0
            self.lags = None
        else:
            self.lags = _check_lags(lags)
            self.order = lags.size

        self.X = None
        self.u = None

        self.gamma_LS = None
        self.gamma_EGLS = None
        self.cov_gamma_EGLS = None

        self.beta = None
        self.cov_beta = None
        self.Sigma_inv = None
        self.R = None

        self.mu = None
        self.cov_mu = None
        self.A = None
        self.cov_A = None
        self.B = None
        self.cov_B = None

        self.Z = None
        self.ZtkI = None
        self.residuals = None

    def _evaluate_residuals(self):
        n_samples, n_features = self.X.shape
        max_lag = self.lags.max()
        for i in range(self.n_components):
            self.residuals[i] = (self.X[max_lag:].ravel() -
                                 np.dot(self.ZtkI, self.beta[i])).reshape(
                                     n_samples - max_lag, n_features)

    def _evaluate_distance_matrix(self):
        for i in range(self.n_components):
            self.distance_matrix[:, i] = np.einsum(
                'ij,jk,ik->i', self.residuals[i], self.Sigma_inv[i],
                self.residuals[i])

    def _initialize_auxiliary_variables(self):

        n_samples, n_features = self.X.shape
        if self.u is None:
            n_external = 0
        else:
            n_external = self.u.shape[1]

        n_lags = self.lags.size
        max_lag = self.lags.max()

        rows_Z = n_lags * n_features + n_external + 1
        cols_Z = n_samples - max_lag
        self.Z = np.empty((rows_Z, cols_Z), dtype=self.X.dtype)

        self.Z[0, :] = 1.0
        for i, lag in enumerate(self.lags):
            start_row = i * n_features + 1
            end_row = (i + 1) * n_features + 1
            self.Z[start_row:end_row] = self.X[max_lag - lag:-lag, :].T
        self.Z[n_lags * n_features + 1, :] = self.u[max_lag:].T

        self.ZtkI = np.kron(self.Z.T, np.eye(n_features))
        self.residuals = np.empty(
            (self.n_components, n_samples - max_lag, n_features))

        self._evaluate_residuals()

    def _initialize_components(self, data, parameters=None, init=None, **kwargs):
        """Generate initial guess for component parameters."""

        self.X = data.copy()
        if 'external_factors' in kwargs:
            self.u = kwargs['external_factors'].copy()
        else:
            self.u = None

        n_samples, n_features = self.X.shape

        if self.u is not None:
            n_external = self.u.shape[1]
            if self.u.shape[0] != n_samples:
                raise ValueError(
                    'number of external variable samples does not match '
                    'number of samples')
        else:
            n_external = 0

        max_lag = self.lags.max()
        self.distance_matrix = np.empty((n_samples - max_lag, self.n_components),
                                        dtype=self.X.dtype)

        if init == 'custom' and parameters is not None:
            _check_init_parameters(
                parameters, n_components=self.n_components,
                n_features=n_features, n_external=n_external, lags=self.lags,
                whom='_initialize_parameters (input parameters)')

            self.mu = parameters['mu'].copy()
            self.A = parameters['A'].copy()
            self.B = parameters['B'].copy()
        else:
            self.mu = np.zeros((self.n_components, n_features,),
                               dtype=self.X.dtype)
            self.A = np.zeros((self.n_components, self.order, n_features, n_features),
                              dtype=self.X.dtype)
            if n_external > 0:
                self.B = np.zeros((self.n_components, n_features, n_external),
                                  dtype=self.X.dtype)

        self.cov_mu = np.empty_like(self.mu)
        self.cov_A = np.empty_like(self.A)

        n_parameters = self.mu[0].size + np.product(self.A[0].shape)

        if n_external > 0:
            self.cov_B = np.empty_like(self.B)
            n_parameters += np.product(self.B[0].shape)

        self.gamma_LS = np.empty((self.n_components, n_parameters,),
                                 dtype=self.X.dtype)
        self.gamma_EGLS = np.empty_like(self.gamma_LS)
        self.cov_gamma_EGLS = np.empty(
            (self.n_components, n_parameters, n_parameters), dtype=self.X.dtype)
        self.Sigma_inv = np.empty((self.n_components, n_features, n_features),
                                  dtype=self.X.dtype)
        for i in range(self.n_components):
            self.Sigma_inv[i] = np.eye(n_features)

        self.R = np.empty((self.n_components, n_parameters, n_parameters),
                          dtype=self.X.dtype)
        for i in range(self.n_components):
            self.R[i] = np.eye((n_parameters, n_parameters), dtype=self.X.dtype)

        self.beta = np.empty((self.n_components, n_parameters,),
                             dtype=self.X.dtype)
        self.cov_beta = np.empty_like(self.beta)

        n_lags = self.lags.size
        for i in range(self.n_components):
            self.beta[i, :n_features] = self.mu[i]
            start_index = n_features
            for p in range(n_lags):
                self.beta[i, start_index:start_index + n_features ** 2] = \
                    self.A[i, p].T.ravel()
                start_index += n_features ** 2
            self.beta[i, start_index:] = self.B[i].T.ravel()

        self._initialize_auxiliary_variables()

    def _fill_component_matrices(self):
        """Extract parameter matrices from parameter vector."""

        n_features = self.X.shape[1]
        n_lags = self.lags.size
        for i in range(self.n_components):
            self.mu[i] = self.beta[i, :n_features]
            self.cov_mu[i] = self.cov_beta[i, :n_features]

            index = n_features
            stride = n_features ** 2
            for p in range(n_lags):
                self.A[i, p] = self.beta[i, index:index + stride].reshape(
                    (n_features, n_features)).T
                self.cov_A[i, p] = self.cov_beta[i, index:index + stride].reshape(
                    (n_features, n_features)).T
                index += stride

            if self.B is not None:
                n_external = self.B[i].shape[1]
                self.B[i] = self.beta[i, index:].reshape((n_external, n_features)).T

    def _update_parameters(self):
        """Update component parameters."""

        n_samples, n_features = self.X.shape
        max_lag = self.lags.max()
        for i in range(self.n_components):
            weights = np.diag(np.sqrt(self.Gamma[:, i]))
            ZWtW = np.dot(self.Z, np.dot(weights.T, weights))
            ZWtWZt = np.dot(ZWtW, self.Z.T)

            lstsq_lhs = np.dot(
                self.R[i].T, np.dot(
                    np.kron(ZWtWZt, np.eye(n_features)), self.R[i]))
            lstsq_rhs = np.dot(
                self.R[i].T, np.dot(
                    np.kron(ZWtW, np.eye(n_features)), self.X.ravel()))

            self.gamma_LS[i] = np.linalg.lstsq(
                lstsq_lhs, lstsq_rhs, rcond=None)[0]

            residual = (self.X[max_lag:].ravel() - np.dot(
                self.ZtkI, np.dot(self.R[i], self.gamma_LS[i]))).reshape(
                    n_samples - max_lag, n_features)

            Sigma = np.dot(residual.T, residual) / (n_samples - max_lag)
            self.Sigma_inv[i] = np.linalg.pinv(Sigma)

            lstsq_lhs = np.dot(
                self.R[i].T, np.dot(
                    np.kron(ZWtWZt, self.Sigma_inv[i]), self.R[i]))
            lstsq_rhs = np.dot(
                self.R[i].T, np.dot(
                    np.kron(ZWtW, self.Sigma_inv[i]), self.X.ravel()))

            self.gamma_EGLS[i] = np.linalg.lstsq(
                lstsq_lhs, lstsq_rhs, rcond=None)[0]
            self.cov_gamma_EGLS[i] = np.linalg.pinv(
                np.dot(
                    self.R[i].T, np.dot(
                        np.kron(
                            self.ZZt / (n_samples - max_lag), self.Sigma_inv[i]),
                        self.R[i]))) / (n_samples - max_lag)

            self.beta[i] = np.dot(self.R[i], self.gamma_EGLS[i])
            self.cov_beta[i] = np.dot(
                self.R[i],
                np.dot(self.cov_gamma_EGLS, self.R[i].T))

        self._fill_component_matrices()
        self._evaluate_residuals()
