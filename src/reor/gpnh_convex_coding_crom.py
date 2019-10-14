"""
Provides routines for cluster-based model order reduction.
"""


from collections import deque
import time
import numpy as np

from reor.gpnh_convex_coding import GPNHConvexCoding
from reor._random_matrix import left_stochastic_matrix
from reor.simplex_projection import (simplex_project_columns,
                                     simplex_project_vector)
from reor.spg import get_next_spg_alpha, get_next_spg_step_length


class GPNHConvexCodingCROM(GPNHConvexCoding):
    """GPNH convex coding based CROM.

    Parameters
    ----------
    n_components : integer or None
        If an integer, the number of components. If None, then all
        features are kept.

    epsilon_states : float, default: 0
        Regularization parameter for the dictionary, if chosen
        method includes a regularization for the dictionary.

    init : None | 'furthest_sum' | 'random' | 'custom'
        Method used to initialize the algorithm.
        Default: None
        Valid options:

        - None: falls back to 'random'.

        - 'furthest_sum': dictionary is initialized using FurthestSum
          method, and weights are initialized to a random stochastic
          matrix.

        - 'random': random matrix of dictionary elements scaled by
          sqrt(X.mean() / n_components), and a random stochastic
          matrix of weights.

        - 'custom': use custom matrices for dictionary and weights.

    tolerance : float, default: 1e-6
        Tolerance of the stopping condition.

    max_iterations : integer, default: 1000
        Maximum number of iterations before stopping.

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
    dictionary_ : array-like, shape (n_features, n_components)
        The dictionary of states.

    cost_ : number
        Value of the cost function for the obtained factorization.

    n_iter_ : integer
        Actual number of iterations.
    """
    def __init__(self, n_components, lags=None, **kwargs):
        super().__init__(n_components, **kwargs)

        if lags is None:
            self.lags = np.array([1], dtype='i8')
        else:
            self.lags = lags

        n_lags = self.lags.size
        self.order_weights = np.empty((n_lags,), dtype='f8')
        self.transition_matrices = np.empty(
            (n_lags, n_components, n_components), dtype='f8')
        self.Z = None

        self.order_weights_old = None
        self.grad_order_weights = None
        self.delta_grad_order_weights = None
        self.incr_order_weights = None

        self.transition_matrices_old = None
        self.grad_transition_matrices = None
        self.delta_grad_transition_matrices = None
        self.incr_transition_matrices = None

        self.alpha_parameters = 1.0
        self.f_parameters_mem = deque([0.0] * self.line_search_memory,
                                      maxlen=self.line_search_memory)

    def _initialize_parameters_workspace(self):
        """Initialize temporary variables."""

        self._update_predicted_weights()

        self.order_weights_old = self.order_weights.copy()
        self.grad_order_weights = np.empty_like(
            self.order_weights, dtype=self.order_weights.dtype)
        self.delta_grad_order_weights = np.empty_like(
            self.order_weights, dtype=self.order_weights.dtype)
        self.incr_order_weights = np.empty_like(
            self.order_weights, dtype=self.order_weights.dtype)

        self.transition_matrices_old = self.transition_matrices.copy()
        self.grad_transition_matrices = np.empty_like(
            self.transition_matrices, dtype=self.transition_matrices.dtype)
        self.delta_grad_transition_matrices = np.empty_like(
            self.transition_matrices, dtype=self.transition_matrices.dtype)
        self.incr_transition_matrices = np.empty_like(
            self.transition_matrices, dtype=self.transition_matrices.dtype)

        self.alpha_parameters = 1.0
        self.f_parameters_mem = deque([0.0] * self.line_search_memory,
                                      maxlen=self.line_search_memory)

    def _update_predicted_weights(self):
        """Update predicted weights."""

        self.Z = np.zeros_like(self.Gamma, dtype=self.Gamma.dtype)

        max_lag = self.lags.max()
        self.Z[:max_lag] = self.Gamma[:max_lag]
        for i, lag in enumerate(self.lags):
            self.Z[max_lag:] += (
                self.order_weights[i] *
                self.Gamma[max_lag - lag:-lag].dot(
                    self.transition_matrices[i].T))

    def _evaluate_fit_residual(self):
        """Evaluate fitted parameters residual."""

        value = 0.0

        if self.n_components == 1:
            return value

        n_samples = self.X.shape[0]
        max_lag = self.lags.max()

        normalization = 1.0 / (self.n_components * (n_samples - max_lag))

        value += np.linalg.norm(self.Gamma[max_lag:] - self.Z[max_lag:]) ** 2

        return normalization * value

    def _update_parameters_gradient(self):
        """Update gradient of cost function with respect to parameters."""

        max_lag = self.lags.max()
        n_samples = self.X.shape[0]
        normalization = 1.0 / (self.n_components * (n_samples - max_lag))

        residual = self.Gamma[max_lag:] - self.Z[max_lag:]

        self.grad_order_weights = np.zeros_like(self.order_weights)
        self.grad_transition_matrices = np.zeros_like(self.transition_matrices)

        for i, lag in enumerate(self.lags):
            self.grad_order_weights[i] = (
                -2 * normalization * residual.dot(
                    self.transition_matrices[i]).dot(
                        self.Gamma[max_lag - lag:-lag].T).trace())
            self.grad_transition_matrices[i] = (
                -2 * normalization *
                self.order_weights[i] *
                residual.T.dot(self.Gamma[max_lag - lag:-lag]))

    def _parameters_line_search(self):
        """Perform single step of line-search for parameters."""

        current_cost = self._evaluate_fit_residual()

        self.order_weights_old = self.order_weights.copy()
        self.transition_matrices_old = self.transition_matrices.copy()

        self.f_parameters_mem.popleft()
        self.f_parameters_mem.append(current_cost)

        f_max = None
        for f in self.f_parameters_mem:
            if f_max is None or f >= f_max:
                f_max = f

        incr_norm = ((self.incr_order_weights * self.grad_order_weights).sum() +
                     (self.incr_transition_matrices *
                      self.grad_transition_matrices).sum())
        step_size = 1

        self.order_weights += self.incr_order_weights
        self.transition_matrices += self.incr_transition_matrices
        self._update_predicted_weights()

        next_cost = self._evaluate_fit_residual()

        error = 0
        factor = self.line_search_gamma * incr_norm
        while next_cost > f_max + factor * step_size:
            step_size = get_next_spg_step_length(
                step_size, incr_norm, current_cost, next_cost,
                sigma_one=self.line_search_sigma_one,
                sigma_two=self.line_search_sigma_two)


            self.order_weights = (self.order_weights_old + step_size *
                                  self.incr_order_weights)
            self.transition_matrices = (self.transition_matrices_old + step_size *
                                        self.incr_transition_matrices)
            self._update_predicted_weights()

            next_cost = self._evaluate_fit_residual()

            if abs(step_size) < self.step_size_tolerance:
                break

        return error, step_size

    def _update_parameters(self):
        """Update parameters using line-search."""

        self._update_predicted_weights()
        self._update_parameters_gradient()

        self.incr_order_weights = (self.order_weights - self.alpha_parameters *
                                   self.grad_order_weights)
        self.incr_transition_matrices = (self.transition_matrices -
                                         self.alpha_parameters *
                                         self.grad_transition_matrices)

        self.incr_order_weights = simplex_project_vector(
            self.incr_order_weights)

        n_lags = self.lags.size
        for i in range(n_lags):
            simplex_project_columns(self.incr_transition_matrices[i])

        self.incr_order_weights -= self.order_weights
        self.incr_transition_matrices -= self.transition_matrices

        error, step_size = self._parameters_line_search()

        self.delta_grad_order_weights = self.grad_order_weights.copy()
        self.delta_grad_transition_matrices = self.grad_transition_matrices.copy()

        self._update_parameters_gradient()

        self.delta_grad_order_weights = (self.grad_order_weights -
                                         self.delta_grad_order_weights)
        self.delta_grad_transition_matrices = (self.grad_transition_matrices -
                                               self.delta_grad_transition_matrices)

        sksk = step_size ** 2 * (
            (self.incr_order_weights * self.incr_order_weights).sum() +
            (self.incr_transition_matrices * self.incr_transition_matrices).sum())
        beta = step_size * (
            (self.incr_order_weights * self.delta_grad_order_weights).sum() +
            (self.incr_transition_matrices * self.delta_grad_transition_matrices).sum())

        self.alpha_parameters = get_next_spg_alpha(
            beta, sksk,
            alpha_min=self.line_search_alpha_min,
            alpha_max=self.line_search_alpha_max)

        return error

    def _fit_parameters(self):
        """Fit Markov chain parameters for weights."""

        if self.verbose:
            print("*** GPNH convex coding CROM: n_components = {:d} ***".format(
                self.Gamma.shape[1]))
            print('{:<12s} | {:<13s} | {:<13s} | {:<12s}'.format(
                'Iteration', 'Cost', 'Cost delta', 'Time'))
            print(60 * '-')

        n_lags = self.lags.size
        self.order_weights = self.random_state.uniform(size=(n_lags,))
        self.order_weights /= self.order_weights.sum()

        self.transition_matrices = np.empty(
            (n_lags, self.n_components, self.n_components), dtype=self.X.dtype)
        for p in range(n_lags):
            self.transition_matrices[p] = left_stochastic_matrix(
                (self.n_components, self.n_components),
                random_state=self.random_state)

        self._initialize_parameters_workspace()

        old_cost = self._evaluate_fit_residual()
        new_cost = old_cost

        for n_iter in range(self.max_iterations):
            start_time = time.perf_counter()

            old_cost = new_cost

            self._update_parameters()
            new_cost = self._evaluate_fit_residual()
            if (new_cost > old_cost) and self.require_monotonic_cost_decrease:
                raise RuntimeError(
                    'factorization cost increased after parameters update')

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

    def fit_transform(self, data, **kwargs):
        """Fit convex coding and return transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Data matrix to be factorized.

        dictionary : array-like, shape (n_components, n_features)
            If init='custom', used as initial guess for solution.

        weights : array-like, shape (n_samples, n_components)
            If init='custom', used as initial guess for solution.

        Returns
        -------
        weights : array-like, shape (n_samples, n_components)
            Representation of the data.
        """

        super().fit_transform(data, **kwargs)

        self._initialize_parameters_workspace()
        self._fit_parameters()

        return self.Gamma

    def fit(self, data, **kwargs):
        """Fit convex coding to data.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Data matrix to perform analysis on.

        Returns
        -------
        self
        """
        self.fit_transform(data, **kwargs)
        return self

    def predict(self, data, horizon=0):
        """Predict values from sample for the given horizon.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Data matrix to evaluate prediction for.

        horizon : integer
            If given, the number of timesteps over which to predict.

        Returns
        -------
        forecast : array, shape (n_samples, n_features)
            Array containing the predicted values for the requested
            horizon.
        """

        initial_weights = self.transform(data)

        if horizon == 0:
            return self.inverse_transform(initial_weights)

        def predict_next_step_weights(current_weights):
            max_lag = self.lags.max()
            n_samples, n_components = current_weights.shape

            if n_samples < max_lag:
                return np.full((n_components,), np.NaN,
                               dtype=current_weights.dtype)

            predicted_weights = np.zeros(
                n_components, dtype=current_weights.dtype)

            for i, lag in enumerate(self.lags):
                predicted_weights += (
                    self.order_weights[i] *
                    self.transition_matrices[i].dot(current_weights[-lag]))

            return predicted_weights

        predicted_weights = np.zeros_like(
            initial_weights, dtype=initial_weights.dtype)

        max_lag = self.lags.max()
        predicted_weights[:max_lag - 1] = np.NaN

        n_samples = initial_weights.shape[0]
        for i in range(max_lag - 1, n_samples):
            lagged_weights = initial_weights[i - max_lag + 1:i + 1]
            for lead in range(horizon):
                next_weights = predict_next_step_weights(
                    lagged_weights)
                lagged_weights = np.roll(lagged_weights, -1, axis=0)
                lagged_weights[-1] = next_weights
            predicted_weights[i] = lagged_weights[-1]

        return self.inverse_transform(predicted_weights)
