"""
Provides base class for FEM-BV clustering methods.
"""


from __future__ import print_function

import numbers
import time
import warnings
import numpy as np

from cvxopt import matrix, spmatrix
import cvxopt.solvers as cs
from sklearn.utils import check_array, check_random_state

from reor._random_matrix import right_stochastic_matrix
from reor._validation import _check_unit_axis_sums, _check_array_shape


INTEGER_TYPES = (numbers.Integral, np.integer)


def _check_init_weights(weights, shape, whom):
    weights = check_array(weights)
    _check_array_shape(weights, shape, whom)
    _check_unit_axis_sums(weights, whom, axis=1)


def _initialize_fembv_weights_random(n_samples, n_components, random_state=None):
    rng = check_random_state(random_state)

    return right_stochastic_matrix((n_samples, n_components), random_state=rng)


def _initialize_fembv_weights(n_samples, n_components, unused_init='random',
                              random_state=None):
    return _initialize_fembv_weights_random(
        n_samples, n_components, random_state=random_state)


class FEMBV():
    """Base class for FEM-BV clustering algorithms."""

    def __init__(self, n_components, max_tv_norm, init='random',
                 max_iterations=500,
                 tolerance=1e-4, verbose=0, random_state=None, **kwargs):
        self.n_components = n_components
        self.max_tv_norm = max_tv_norm
        self.init = init
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.require_monotonic_cost_decrease = kwargs.get(
            'require_monotonic_cost_decrease', True)
        self.name = 'Generic FEM-BV'

        self.Gamma = None
        self.Eta = None
        self.distance_matrix = None

        self.c = None
        self.A_eq = None
        self.b_eq = None
        self.A_ub = None
        self.b_ub = None
        self.bounds = None

    def _evaluate_distance_matrix(self):
        raise NotImplementedError()

    def _evaluate_cost(self):
        self._evaluate_distance_matrix()
        return (self.Gamma * self.distance_matrix).sum()

    def _has_max_tv_norm_constraint(self):
        return self.max_tv_norm is not None and self.max_tv_norm >= 0

    def _initialize_constraints(self, n_samples):

        n_weight_parameters = self.n_components * n_samples

        if self._has_max_tv_norm_constraint():
            n_weight_parameters += self.n_components * (n_samples - 1)

        self.c = matrix(0.0, size=(n_weight_parameters, 1), tc='d')

        if self.n_components > 1:
            eq_nnz_rows = np.repeat(np.arange(n_samples), self.n_components)
            eq_nnz_cols = np.arange(n_samples * self.n_components)
            self.A_eq = spmatrix(1.0, eq_nnz_rows, eq_nnz_cols,
                                 size=(n_samples, n_weight_parameters), tc='d')
            self.b_eq = matrix(1.0, size=(n_samples, 1), tc='d')

            # Number of non-negativity constraints
            n_non_neg_constraints = n_weight_parameters
            n_ub_constraints = n_non_neg_constraints

            # Non-negativity constraints on weight parameters
            nnz_coeffs = np.full((n_weight_parameters,), -1,
                                 dtype=self.Gamma.dtype)
            ub_nnz_rows = np.arange(n_weight_parameters)
            ub_nnz_cols = np.arange(n_weight_parameters)
            ub_values = np.zeros((n_weight_parameters,), dtype=self.Gamma.dtype)

            if self._has_max_tv_norm_constraint():
                # Number of auxiliary constraints
                n_aux_constraints = 2 * self.n_components * (n_samples - 1)

                nnz_aux_rows = np.repeat(
                    np.arange(n_ub_constraints,
                              n_ub_constraints + n_aux_constraints), 3)
                nnz_aux_cols = np.tile(np.array(
                    [[(t + 1) * self.n_components + i,
                      t * self.n_components + i,
                      self.n_components * n_samples + t * self.n_components + i]
                     for t in range(n_samples - 1)
                     for i in range(self.n_components)]).ravel(), 2)
                nnz_aux_coeffs = np.concatenate(
                    [np.tile(np.array([1.0, -1.0, -1.0]),
                             self.n_components * (n_samples - 1)),
                     np.tile(np.array([-1.0, 1.0, -1.0]),
                             self.n_components * (n_samples - 1))])

                n_ub_constraints += n_aux_constraints
                ub_values = np.concatenate(
                    [ub_values,
                     np.zeros((2 * self.n_components * (n_samples -1),))])

                # Number of total variation norm constraints
                n_tv_constraints = self.n_components

                nnz_tv_rows = np.repeat(
                    np.arange(n_ub_constraints,
                              n_ub_constraints + n_tv_constraints),
                    n_samples - 1)
                nnz_tv_cols = np.concatenate(
                    [np.array(
                        [self.n_components * n_samples + t * self.n_components + i
                         for t in range(n_samples - 1)])
                     for i in range(self.n_components)])
                nnz_tv_coeffs = np.ones((self.n_components * (n_samples - 1),),
                                        dtype=self.Gamma.dtype)

                n_ub_constraints += n_tv_constraints

                ub_values = np.concatenate(
                    [ub_values,
                     np.full((self.n_components,), self.max_tv_norm)])

                nnz_coeffs = np.concatenate(
                    [nnz_coeffs, nnz_aux_coeffs, nnz_tv_coeffs])
                ub_nnz_rows = np.concatenate(
                    [ub_nnz_rows, nnz_aux_rows, nnz_tv_rows])
                ub_nnz_cols = np.concatenate(
                    [ub_nnz_cols, nnz_aux_cols, nnz_tv_cols])

            self.A_ub = spmatrix(
                nnz_coeffs, ub_nnz_rows, ub_nnz_cols,
                size=(n_ub_constraints, n_weight_parameters), tc='d')
            self.b_ub = matrix(ub_values, size=(n_ub_constraints, 1), tc='d')

    def _initialize_weights(self, unused_data, n_samples, weights=None, init=None):
        """Generate initial guess for component weights."""

        if init == 'custom' and weights is not None:
            _check_init_weights(weights, (n_samples, self.n_components),
                                '_initialize_weights (input weights)')
            self.Gamma = weights.copy()
        else:
            self.Gamma = _initialize_fembv_weights(
                n_samples, n_components=self.n_components, unused_init=init,
                random_state=self.random_state)

        if self._has_max_tv_norm_constraint():
            self.Eta = np.empty(self.Gamma.shape, dtype=self.Gamma.dtype)

        self._initialize_constraints(n_samples)

    def _initialize_components(self, data, parameters=None, init=None, **kwargs):
        """Generate initial guess for component parameters."""
        raise NotImplementedError()

    def _initialize(self, data, parameters=None, weights=None, init=None,
                    **kwargs):
        """Generate initial guess for parameters and weights."""

        self._initialize_components(data, parameters=parameters, init=init,
                                    **kwargs)

        if (self.distance_matrix is None or
                self.distance_matrix.shape[1] != self.n_components):
            raise RuntimeError(
                'Incorrectly initialized distance matrix')

        n_samples = self.distance_matrix.shape[0]
        self._initialize_weights(data, n_samples, weights=weights, init=init)

    def _update_weights(self):
        """Update cluster weights."""

        if self.n_components == 1:
            return

        n_samples = self.Gamma.shape[0]

        self._evaluate_distance_matrix()

        assert self.distance_matrix.shape == (n_samples, self.n_components)

        self.c[:self.Gamma.size] = self.distance_matrix.ravel()

        if 'show_progress' in cs.options:
            old_show_progress = cs.options['show_progress']
        else:
            old_show_progress = True

        if self.verbose > 1:
            cs.options['show_progress'] = True
        else:
            cs.options['show_progress'] = False

        sol = cs.lp(self.c, G=self.A_ub, h=self.b_ub,
                    A=self.A_eq, b=self.b_eq)

        cs.options['show_progress'] = old_show_progress

        if sol['status'] != 'optimal':
            warnings.warn('Updating weights failed', UserWarning)
        else:
            self.Gamma = np.reshape(
                sol['x'][:self.n_components * n_samples],
                (n_samples, self.n_components))

            if self._has_max_tv_norm_constraint():
                self.Eta = np.reshape(
                    sol['x'][self.n_components * n_samples:],
                    (n_samples - 1, self.n_components))

    def _update_parameters(self):
        """Update cost function parameters."""
        raise NotImplementedError()

    def _iterate(self, update_weights=True, update_parameters=True):
        """Minimize cost function iteratively."""

        if self.verbose:
            print('*** {}: n_components = {:d} ***'.format(
                self.name, self.Gamma.shape[1]))
            print('{:<12s} | {:<13s} | {:<13s} | {:<12s}'.format(
                'Iteration', 'Cost', 'Cost delta', 'Time'))
            print(60 * '-')

        old_cost = self._evaluate_cost()
        new_cost = old_cost

        for n_iter in range(self.max_iterations):
            start_time = time.perf_counter()

            old_cost = new_cost

            if update_parameters:
                self._update_parameters()
                new_cost = self._evaluate_cost()

                if self.require_monotonic_cost_decrease:
                    update_failed = (new_cost > old_cost and
                                     abs(new_cost > old_cost) > self.tolerance)
                    if update_failed:
                        raise RuntimeError(
                            'cost increased after parameters update')

            if update_weights:
                self._update_weights()
                new_cost = self._evaluate_cost()

                if self.require_monotonic_cost_decrease:
                    update_failed = (new_cost > old_cost and
                                     abs(new_cost > old_cost) > self.tolerance)
                    if update_failed:
                        raise RuntimeError(
                            'cost increased after weights update')

            cost_delta = new_cost - old_cost

            end_time = time.perf_counter()

            if self.verbose:
                print('{:12d} | {: 12.6e} | {: 12.6e} | {: 12.6e}'.format(
                    n_iter + 1, new_cost, cost_delta, end_time - start_time))

            if abs(cost_delta) < self.tolerance:
                if self.verbose:
                    print('*** Converged at iteration {:d} ***'.format(
                        n_iter + 1))
                break

        return new_cost, n_iter

    def _fembv_fit(self, data, parameters=None, weights=None, init=None,
                   update_parameters=True, update_weights=True, **kwargs):
        """Calculate FEM-BV fit."""

        n_samples, n_features = data.shape

        if self.n_components is None:
            self.n_components = n_features

        if not isinstance(self.n_components, INTEGER_TYPES) or self.n_components <= 0:
            raise ValueError('Number of components must be a positive integer;'
                             ' got (n_components=%r)' % self.n_components)
        if not isinstance(self.max_iterations, INTEGER_TYPES) or self.max_iterations <= 0:
            raise ValueError('Maximum number of iterations must be a positive '
                             'integer; got (max_iterations=%r)' % self.max_iterations)
        if not isinstance(self.tolerance, numbers.Number) or self.tolerance < 0:
            raise ValueError('Tolerance for stopping criteria must be '
                             'positive; got (tolerance=%r)' % self.tolerance)

        self._initialize(data, parameters=parameters, weights=weights,
                         init=init, **kwargs)

        cost, n_iter = self._iterate(update_parameters=update_parameters,
                                     update_weights=update_weights)

        if n_iter == self.max_iterations and self.tolerance > 0:
            warnings.warn('Maximum number of iterations %d reached.' %
                          self.max_iterations, UserWarning)

        return cost, n_iter

    def fit_transform(self, data, parameters=None, weights=None, **kwargs):
        """Fit FEMBV model and return transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Data matrix to be fit.

        parameters :
            If init='custom', used as initial guess for solution.

        weights : array-like, shape (n_samples, n_components)
            If init='custom', used as initial guess for solution.

        Returns
        -------
        weights : array-like, shape (n_samples, n_components)
            Representation of the data.
        """

        cost, n_iter = self._fembv_fit(
            data, parameters=parameters, weights=weights, **kwargs)

        self.n_components_ = self.Gamma.shape[1]
        self.cost_ = cost
        self.n_iter_ = n_iter

        return self.Gamma

    def fit(self, data, **kwargs):
        """Fit FEMBV model to data.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Data matrix to be fit.

        Returns
        -------
        self
        """
        self.fit_transform(data, **kwargs)
        return self

    def transform(self, data):
        """Transform data according to the fitted factorization.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Data matrix for data to be transformed.

        Returns
        -------
        weights : array-like, shape (n_samples, n_components)
            Representation of the data.
        """

        self._fembv_fit(data=data, update_parameters=False,
                        update_weights=True)

        return self.Gamma
