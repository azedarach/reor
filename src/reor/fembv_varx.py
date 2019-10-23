"""
Provides routines for FEM-BV-VARX clustering method.
"""

import numbers
import numpy as np

from reor.fembv import FEMBV
from reor.var import linear_varx_EGLS
from reor._validation import _check_array_shape


INTEGER_TYPES = (numbers.Integral, np.integer)


def _check_init_parameters(parameters, n_components, n_features,
                           n_external, memory, whom):

    if 'mu' not in parameters:
        raise ValueError('Initial guess for parameter mu missing in %s' % whom)

    if memory > 0 and 'A' not in parameters:
        raise ValueError('Initial guess for parameter A missing in %s' % whom)

    if n_external != 0 and 'B0' not in parameters:
        raise ValueError('Initial guess for parameter B0 missing in %s' % whom)

    _check_array_shape(parameters['mu'], (n_features,), whom)

    if memory > 0:
        _check_array_shape(parameters['A'],
                           (n_components, memory, n_features, n_features), whom)

    if n_external != 0:
        _check_array_shape(parameters['B0'],
                           (n_components, n_features, n_external), whom)


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

    stderr_mu : array-like, shape (n_components, n_features)
        The estimated asymptotic standard errors for the fitted intercepts
        for each component.

    A : array-like, shape (n_components, memory, n_features, n_features)
        The fitted autoregressive coefficients for each component.

    stderr_A : array-like, shape (n_components, memory, n_features, n_features)
        The estimated asymptotic standard errors for the fitted autoregressive
        coefficients for each component.

    B0 : array-like, shape (n_components, n_features, n_external)
        If external factors are included, the fitted external factor
        coefficients for each component. If no exogeneous factors are present,
        is None.

    stderr_B0 : array-like, shape (n_components, n_features, n_external)
        If external factors are included, the estimated asymptotic standard
        errors for the fitted external coefficients for each component. If
        no exogeneous factors are present, is None.

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
    P. Metzner, L. PUtzig, and I. Horenko, "Analysis of Persistent and
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
            self.memory = 0
        else:
            if not isinstance(memory, INTEGER_TYPES) or memory < 0:
                raise ValueError(
                    'Memory must be a non-negative integer;'
                    ' got (memory=%d)' % memory)

            self.memory = memory

        self.name = 'FEM-BV-VARX'
        self.X = None
        self.u = None

        self.mu = None
        self.stderr_mu = None
        self.A = None
        self.stderr_A = None
        self.B0 = None
        self.stderr_B0 = None
        self.Sigma_inv = None

        self.residuals = None

    def _evaluate_residuals(self):
        n_external = 0 if self.u is None else self.u.shape[1]

        for i in range(self.n_components):
            self.residuals[i] = self.X[self.memory:, :] - self.mu[i]

            if self.memory > 0:
                for m in range(1, self.memory + 1):
                    self.residuals[i] -= np.dot(
                        self.X[self.memory - m:-m], self.A[i, m - 1].T)

            if n_external > 0:
                self.residuals[i] -= np.dot(self.u[self.memory:], self.B0[i].T)

    def _evaluate_distance_matrix(self):
        for i in range(self.n_components):
            self.distance_matrix[:, i] = np.einsum(
                'ij,jk,ik->i', self.residuals[i], self.Sigma_inv[i],
                self.residuals[i])

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

        self.distance_matrix = np.empty(
            (n_samples - self.memory, self.n_components), dtype=self.X.dtype)
        self.residuals = np.empty(
            (self.n_components, n_samples - self.memory, n_features),
            dtype=self.X.dtype)

        if init == 'custom' and parameters is not None:
            _check_init_parameters(
                parameters, n_components=self.n_components,
                n_features=n_features, n_external=n_external,
                memory=self.memory,
                whom='_initialize_components (input parameters)')

            self.mu = parameters['mu'].copy()

            if self.memory > 0:
                self.A = parameters['A'].copy()

            if n_external > 0:
                self.B0 = parameters['B0'].copy()
        else:
            self.mu = np.zeros((self.n_components, n_features,),
                               dtype=self.X.dtype)
            if self.memory > 0:
                self.A = np.zeros(
                    (self.n_components, self.memory, n_features, n_features),
                    dtype=self.X.dtype)

            if n_external > 0:
                self.B0 = np.zeros((self.n_components, n_features, n_external),
                                   dtype=self.X.dtype)

        self.stderr_mu = np.empty_like(self.mu)

        if self.memory > 0:
            self.stderr_A = np.empty_like(self.A)

        if n_external > 0:
            self.stderr_B0 = np.empty_like(self.stderr_B0)

        self.Sigma_inv = np.empty((self.n_components, n_features, n_features),
                                  dtype=self.X.dtype)
        for i in range(self.n_components):
            self.Sigma_inv[i] = np.eye(n_features)

        self._evaluate_residuals()

    def _update_parameters(self):
        """Update component parameters."""

        if self.u is not None:
            n_external = self.u.shape[1]
        else:
            n_external = 0

        for i in range(self.n_components):
            weights = np.ones((self.X.shape[0],))
            weights[self.memory:] = np.sqrt(self.Gamma[:, i])

            varx_kwargs = dict(p=self.memory, weights=weights,
                               include_intercept=True)
            if self.u is not None:
                varx_kwargs['x'] = self.u.T

            result = linear_varx_EGLS(self.X.T, **varx_kwargs)

            self.mu[i] = result['mu']
            self.stderr_mu[i] = result['stderr_mu']

            if self.memory > 0:
                self.A[i] = result['A']
                self.stderr_A[i] = result['stderr_A']

            if n_external > 0:
                self.B0[i] = result['B0']
                self.stderr_B0[i] = result['stderr_B0']

            self.Sigma_inv[i] = np.linalg.pinv(result['Sigma_LS'])

        self._evaluate_residuals()
