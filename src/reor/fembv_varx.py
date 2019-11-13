"""
Provides routines for FEM-BV-VARX clustering method.
"""

import numbers
import numpy as np

from sklearn.utils.validation import check_array

from reor.fembv import FEMBV
from reor._validation import _check_array_shape


INTEGER_TYPES = (numbers.Integral, np.integer)


class FEMBVVARXLocalLinearModel():
    """Local stationary linear VARX model."""

    def __init__(self, y, p=0, u=None, s=0):
        if not isinstance(p, INTEGER_TYPES) or p < 0:
            raise ValueError('VAR order must be a non-negative integer')

        self.p = p

        self._y = check_array(y)

        n_samples, n_features = self._y.shape

        self.mu = np.zeros((n_features,), dtype=self._y.dtype)
        self.Sigma_u = np.zeros((n_features, n_features), dtype=self._y.dtype)

        if p > 0:
            self.A = np.zeros((self.p, n_features, n_features),
                              dtype=self._y.dtype)
        else:
            self.A = None

        if u is not None:
            self._u = check_array(u)

            if self._u.shape[0] != n_samples:
                raise ValueError(
                    'Number of exogeneous factor samples does not match '
                    'number of endogeneous samples')

            n_exog = self._u.shape[1]

            self.B0 = np.zeros((n_features, n_external), dtype=self._y.dtype)

            if not isinstance(s, INTEGER_TYPES) or s < 0:
                raise ValueError(
                    'Exogeneous VAR order must be a non-negative integer')

            self.s = s

            if self.s > 0:
                self.B = np.zeros((self.s, n_features, n_external),
                                  dtype=self._y.dtype)
            else:
                self.B = None
        else:
            n_exog = 0
            self.s = 0
            self._u = None
            self.B0 = None
            self.B = None

        presample_length = max(self.p, self.s)

        n_cols = 1 + self.p * n_features + n_exog * (1 + self.s)
        self._z = np.empty((n_samples - presample_length, n_cols),
                           dtype=self._y.dtype)
        self._z[:, 0] = 1.0

        col_index = 1
        if self.p > 0:
            for i in range(1, self.p + 1):
                self._z[:, col_index:col_index + n_features] = self._y[presample_length - i:-i]
                col_index += n_features

        if self._u is not None:
            if self.s > 0:
                for i in range(1, self.s + 1):
                    self._z[:, col_index:col_index + n_exog] = self._u[presample_length - i:-i]
                    col_index += n_exog

            self._z[:, col_index:col_index + n_exog] = self._u[presample_length:]

        self.residuals = np.zeros_like(self._y)

    def fit(self, weights=None):
        n_samples, n_features = self._y.shape

        presample_length = max(self.p, self.s)

        if weights is None:
            params = np.linalg.lstsq(
                self._z, self._y[presample_length:], rcond=None)[0]
        else:
            if weights.shape != (n_samples,):
                raise ValueError('Number of weights does not match '
                                 'number of samples')

            w = np.sqrt(weights[presample_length:])

            params = np.linalg.lstsq(
                w[:, np.newaxis] * self._z,
                w[:, np.newaxis] * self._y[presample_length:],
                rcond=None)[0]

        self.residuals[presample_length:] = (
            self._y[presample_length:] - np.dot(self._z, params))

        self.mu = params[0]

        if self.p > 0:
            A = np.reshape(params[1:1 + n_features * self.p],
                           (self.p, n_features, n_features))
            self.A = A.swapaxes(1, 2).copy()

        if self._u is not None:
            n_exog = self._u.shape[1]
            if self.s > 0:
                row_start = 1 + n_features * self.p
                row_end = row_start + self.s * n_exog
                B = np.reshape(
                    params[row_start:row_end],
                    (self.s, n_exog, n_features))
                self.B = B.swapaxes(1, 2).copy()

            row_start = 1 + n_features * self.p + self.s * n_exog
            self.B0 = params[row_start:].swapaxes(1, 2).copy()

        df = n_samples - presample_length - self.p * n_features - 1
        if self._u is not None:
            df -= self._u.shape[1] * (1 + self.s)

        if weights is None:
            self.Sigma_u = np.dot(self.residuals[presample_length:].T,
                                  self.residuals[presample_length:]) / df
        else:
            weighted_residuals = (np.sqrt(weights[presample_length:, np.newaxis]) *
                                  self.residuals[presample_length:])
            self.Sigma_u = (np.dot(weighted_residuals[presample_length:].T,
                                   weighted_residuals[presample_length:]) /
                            np.sum(weights) - 1)

        self.Sigma_u_inv = np.linalg.pinv(self.Sigma_u)


class FEMBVVARX(FEMBV):
    """FEM-BV-VARX clustering of data.

    Performs a FEM-BV-VARX fit to the given data. Within each component,
    the data is assumed to be generated by a VARX(p, 0) process with
    memory p, with

        x_t = \mu_i + \sum_{\tau = 1}^p A_\tau^{(i)} x_{t - \tau} + B_0^{(i)} u_t

    and where u_t denotes any exogeneous factors to be included. The data
    to be fitted is assumed to be given in the form of a data matrix with
    shape (n_samples, n_features). Similarly, external factors may be provided
    as a matrix of (n_samples, n_external) values.

    Parameters
    ----------
    n_components : integer or None
        If an integer, the number of components. If None, then all features
        are kept.

    max_tv_norm : None or float, default: None
        If a number, the maximum total-variation norm allowed for the
        weights sequence. If None, no TV constraint is imposed.

    memory : None or integer, default: None
        If an integer, the maximum order of the AR process for each component.
        If None, defaults to 0.

    init : None | 'random' | 'custom'
        Method used to initialize the algorithm.
        Default: None
        Valid options:

        - None: falls back to 'random'.

        - 'random': weights and parameters are initialized to random values.

        - 'custom': use custom matrices for weights and parameters. If given,
          the values of the parameters must be provided as a dict with keys
          'mu', 'A', and 'B0', for which the corresponding values should be the
          initial guess for each parameter.

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

    Attributes
    ----------
    Gamma : array-like, shape (n_samples - memory, n_components)
        The fitted affiliations for the model.

    mu : array-like, shape (n_components, n_features)
        The fitted intercepts for each component.

    A : array-like, shape (n_components, memory, n_features, n_features)
        The fitted autoregressive coefficients for each component.

    B0 : array-like, shape (n_components, n_features, n_external)
        If external factors are included, the fitted external factor
        coefficients for each component. If no exogeneous factors are present,
        is None.

    cost_ : number
        Value of the cost function for the obtained clustering.

    n_iter_ : integer
        Actual number of iterations.

    Examples
    --------
    import numpy as np
    X = np.random.rand(10, 4)
    from reor.fembv_varx import FEMBVVARX
    model = FEMBVVARX(n_components=2, init='random', random_state=0)
    gamma = model.fit_transform(X)
    mu = model.mu
    A = model.A

    References
    ----------
    P. Metzner, L. Putzig, and I. Horenko, "Analysis of Persistent and
    Nonstationary Time Series and Applications", Commun. Appl. Math. Comput.
    Sci. 7, 2 (2012), 175 - 229
    """

    def __init__(self, n_components, max_tv_norm, memory=None, init='random',
                 max_iterations=500, tolerance=1e-4, verbose=0,
                 random_state=None, **kwargs):
        super().__init__(n_components=n_components, max_tv_norm=max_tv_norm,
                         init=init, max_iterations=max_iterations,
                         tolerance=tolerance, verbose=verbose,
                         random_state=random_state, **kwargs)

        if memory is None:
            self.memory = np.full((n_components,), 0, dtype='i8')
        else:
            if isinstance(memory, INTEGER_TYPES):
                if memory < 0:
                    raise ValueError(
                        'Maximum memory must be a non-negative integer;'
                        ' got (memory=%d)' % memory)

                self.memory = np.full((n_components,), memory, dtype='i8')
            elif isinstance(memory, (list, tuple)):
                self.memory = np.asarray(memory, dtype='i8')
                if self.memory.shape != (n_components,):
                    raise ValueError('Memory must be a 1-dimensional array; '
                                     'got shape %r' % list(self.memory.shape))
            else:
                raise ValueError(
                    'Memory must be a non-negative integer or tuple; '
                    ' got %r' % type(memory))

        self.name = 'FEM-BV-VARX'
        self.X = None
        self.u = None

        self.models = None
        self.mu = None
        self.A = None
        self.B0 = None

    def _evaluate_distance_matrix(self):
        presample_length = np.max(self.memory)
        for i in range(self.n_components):
            metric = self.models[i].Sigma_u_inv
            self.distance_matrix[:, i] = np.einsum(
                'ij,jk,ik->i',
                self.models[i].residuals[presample_length:],
                metric,
                self.models[i].residuals[presample_length:])

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

        presample_length = np.max(self.memory)

        self.mu = np.zeros((self.n_components, n_features),
                           dtype=self.X.dtype)
        if presample_length > 0:
            self.A = np.zeros((self.n_components, presample_length,
                               n_features, n_features),
                              dtype=self.X.dtype)
        if self.u is not None:
            self.B0 = np.zeros((self.n_components, n_features, n_external),
                               dtype=self.X.dtype)

        self.models = [FEMBVVARXLocalLinearModel(
            self.X[presample_length - self.memory[i]:],
            p=self.memory[i],
            u=self.u, s=0)
            for i in range(self.n_components)]

        # Initialize models by performing unweighted fits to
        # given data
        for i in range(self.n_components):
            self.models[i].fit()

        self.distance_matrix = np.empty(
            (n_samples - presample_length, self.n_components),
            dtype=self.X.dtype)

        self._evaluate_distance_matrix()

    def _update_parameters(self):
        """Update component parameters."""

        n_samples, n_features = self.X.shape

        presample_length = np.max(self.memory)
        for i in range(self.n_components):

            weights = np.ones(
                (n_samples - presample_length + self.memory[i],),
                dtype=self.X.dtype)
            weights[self.memory[i]:] = self.Gamma[:, i]

            self.models[i].fit(weights=weights)

            self.mu[i] = self.models[i].mu

            if self.memory[i] > 0:
                self.A[i, :self.memory[i]] = self.models[i].A

            if self.u is not None:
                self.B0[i] = self.models[i].B0
