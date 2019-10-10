"""
Provides routines for convex coding using GPNH regularization.
"""

from __future__ import print_function

from collections import deque
import numbers
import time
import warnings
import numpy as np

from sklearn.utils import check_array, check_random_state

from reor.furthest_sum import furthest_sum
from reor._random_matrix import right_stochastic_matrix
from reor.simplex_projection import simplex_project_rows
from reor.spg import get_next_spg_alpha, get_next_spg_step_length
from reor._validation import _check_unit_axis_sums, _check_array_shape


INTEGER_TYPES = (numbers.Integral, np.integer)

INITIALIZATION_METHODS = (None, 'random', 'furthest_sum')


def _check_init_weights(weights, shape, whom):
    weights = check_array(weights)
    _check_array_shape(weights, shape, whom)
    _check_unit_axis_sums(weights, whom, axis=0)


def _check_init_dictionary(dictionary, shape, whom):
    dictionary = check_array(dictionary)
    _check_array_shape(dictionary, shape, whom)


def _initialize_gpnh_convex_coding_dictionary_random(data, n_components,
                                                     random_state=None):
    rng = check_random_state(random_state)

    n_features = data.shape[1]
    avg = np.sqrt(np.abs(data).mean() / n_components)
    dictionary = avg * rng.randn(n_components, n_features)

    return dictionary


def _initialize_gpnh_convex_coding_dictionary_furthest_sum(
        data, n_components, start_index=None, n_extra_steps=10,
        exclude=None, random_state=None):
    rng = check_random_state(random_state)

    n_features = data.shape[1]
    kernel = data.dot(data.T)

    n_samples = kernel.shape[0]
    if start_index is None:
        start_index = rng.randint(n_samples)

    if exclude is None:
        exclude = np.array([], dtype='i8')

    kernel_diag = np.diag(kernel)
    dissimilarities = np.sqrt(
        np.tile(kernel_diag, (n_samples, 1)) -
        2 * kernel +
        np.tile(kernel_diag[:, np.newaxis], (1, n_samples)))

    selected = furthest_sum(
        dissimilarities, n_components, start_index, exclude, n_extra_steps)

    dictionary = np.zeros((n_components, n_features),
                          dtype=kernel.dtype)
    for i in range(n_components):
        dictionary[i] = data[selected[i]]

    return dictionary


def _initialize_gpnh_convex_coding_weights_random(data, n_components,
                                                  random_state=None):
    rng = check_random_state(random_state)

    n_samples = data.shape[0]

    return right_stochastic_matrix((n_samples, n_components), random_state=rng)


def _initialize_gpnh_convex_coding_dictionary(data, n_components, init='random',
                                              random_state=None, **kwargs):
    if init is None:
        init = 'random'

    if init == 'random':
        return _initialize_gpnh_convex_coding_dictionary_random(
            data, n_components, random_state=random_state)

    if init == 'furthest_sum':
        start_index = kwargs.get('start_index', None)
        n_extra_steps = kwargs.get('n_extra_steps', 10)
        exclude = kwargs.get('exclude', None)

        return _initialize_gpnh_convex_coding_dictionary_furthest_sum(
            data, n_components, start_index=start_index,
            n_extra_steps=n_extra_steps, exclude=exclude,
            random_state=random_state)

    raise ValueError(
        'Invalid init parameter: got %r instead of one of %r' %
        (init, INITIALIZATION_METHODS))


def _initialize_gpnh_convex_coding_weights(data, n_components, init='random',
                                           random_state=None):
    if init is None:
        init = 'random'

    if init in ('furthest_sum', 'random'):
        return _initialize_gpnh_convex_coding_weights_random(
            data, n_components, random_state=random_state)

    raise ValueError(
        'Invalid init parameter: got %r instead of one of %r' %
        (init, INITIALIZATION_METHODS))


def _initialize_gpnh_convex_coding(data, n_components, init='random',
                                   random_state=None, **kwargs):
    if init is None:
        init = 'random'

    rng = check_random_state(random_state)

    dictionary = _initialize_gpnh_convex_coding_dictionary(
        data, n_components, init=init, random_state=rng, **kwargs)
    weights = _initialize_gpnh_convex_coding_weights(
        data, n_components, init=init, random_state=rng)

    return dictionary, weights


class GPNHConvexCoding():
    """Convex encoding of data.

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

    Examples
    --------
    import numpy as np
    X = np.random.rand(4, 10)
    from reor.gpnh_convex_coding import GPNHConvexCoding
    model = GPNHConvexCoding(n_components=2, init='random', random_state=0)
    weights = model.fit_transform(X)
    dictionary = model.dictionary_

    References
    ----------
    S. Gerber, L. Pospisil, M. Navandard, and I. Horenko,
    "Low-cost scalable discretization, prediction and feature selection
    for complex systems" (2018)
    """
    def __init__(self, n_components,
                 epsilon_states=0, init=None,
                 tolerance=1e-6, max_iterations=1000,
                 verbose=0, random_state=None, **kwargs):
        self.n_components = n_components
        self.epsilon_states = epsilon_states
        self.init = init
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.random_state = random_state
        self.line_search_memory = kwargs.get('line_search_memory', 1)
        self.line_search_gamma = kwargs.get('gamma', 1e-4)
        self.line_search_alpha_min = kwargs.get('alpha_min', 1e-3)
        self.line_search_alpha_max = kwargs.get('alpha_max', 1e3)
        self.line_search_sigma_one = kwargs.get('sigma_one', 0.1)
        self.line_search_sigma_two = kwargs.get('sigma_two', 0.9)
        self.step_size_tolerance = kwargs.get('step_size_tolerance', 1e-12)
        self.require_monotonic_cost_decrease = kwargs.get(
            'require_monotonic_cost_decrease', True)

        self.X = None
        self.Gamma = None
        self.S = None

        self.trace_XtX = 0.0
        self.GtG = None
        self.GtX = None
        self.XSt = None
        self.SSt = None

        self.Gamma_old = None
        self.grad_Gamma = None
        self.delta_grad_Gamma = None
        self.incr_Gamma = None

        self.S_old = None
        self.grad_S = None
        self.delta_grad_S = None
        self.incr_S = None

        self.alpha_Gamma = 1.0
        self.alpha_S = 1.0
        self.f_S_mem = deque([0.0] * self.line_search_memory,
                             maxlen=self.line_search_memory)
        self.f_Gamma_mem = deque([0.0] * self.line_search_memory,
                                 maxlen=self.line_search_memory)

    def _initialize_workspace(self):
        """Initialize temporary variables."""

        self.trace_XtX = self.X.T.dot(self.X).trace()
        self.GtG = self.Gamma.T.dot(self.Gamma)
        self.GtX = self.Gamma.T.dot(self.X)
        self.XSt = self.X.dot(self.S.T)
        self.SSt = self.S.dot(self.S.T)

        self.Gamma_old = self.Gamma.copy()
        self.grad_Gamma = np.empty_like(self.Gamma, dtype=self.Gamma.dtype)
        self.delta_grad_Gamma = np.empty_like(self.Gamma, dtype=self.Gamma.dtype)
        self.incr_Gamma = np.empty_like(self.Gamma, dtype=self.Gamma.dtype)

        self.S_old = self.S.copy()
        self.grad_S = np.empty_like(self.S, dtype=self.S.dtype)
        self.delta_grad_S = np.empty_like(self.S, dtype=self.S.dtype)
        self.incr_S = np.empty_like(self.S, dtype=self.S.dtype)

        self.alpha_Gamma = 1.0
        self.alpha_S = 1.0
        self.f_S_mem = deque([0.0] * self.line_search_memory,
                             maxlen=self.line_search_memory)
        self.f_Gamma_mem = deque([0.0] * self.line_search_memory,
                                 maxlen=self.line_search_memory)

    def _evaluate_dictionary_penalty(self):
        """Evaluate dictionary penalty function."""

        value = 0.0

        if self.n_components == 1:
            return value

        n_features = self.X.shape[1]

        prefactor = 2.0 / (self.n_components * n_features *
                           (self.n_components - 1.0))

        self.SSt = self.S.dot(self.S.T)

        value += (prefactor * self.n_components * self.SSt.trace() -
                  prefactor * self.SSt.sum())

        return self.epsilon_states * value

    def _evaluate_loss_function(self):
        """Evaluate unregularized loss function."""

        n_samples, n_features = self.X.shape
        normalization = 1.0 / (n_features * n_samples)

        residual = np.linalg.norm(self.X - self.Gamma.dot(self.S))

        return normalization * residual ** 2

    def _evaluate_cost(self):
        """Evaluate cost function."""

        return (self._evaluate_loss_function() +
                self._evaluate_dictionary_penalty())

    def _dictionary_penalty_gradient(self):
        """Evaluate gradient of dictionary penalty."""

        if self.n_components == 1:
            self.grad_S = np.zeros_like(self.S)
            return

        n_features = self.X.shape[1]

        prefactor = (4.0 * self.epsilon_states /
                     (n_features * self.n_components * (self.n_components - 1)))

        D = self.n_components * np.eye(self.n_components) - 1

        self.grad_S = D.dot(prefactor * self.S)

    def _update_dictionary_gradient(self):
        """Update gradient of cost function with respect to dictionary."""

        self._dictionary_penalty_gradient()

        n_samples, n_features = self.X.shape
        normalization = 1.0 / (n_features * n_samples)

        self.grad_S += -2 * normalization * self.GtX
        self.grad_S += 2 * normalization * self.GtG.dot(self.S)

    def _dictionary_line_search(self):
        """Perform single step of line-search for dictionary."""

        current_cost = self._evaluate_cost()

        self.S_old = self.S.copy()

        self.f_S_mem.popleft()
        self.f_S_mem.append(current_cost)

        f_max = None
        for f in self.f_S_mem:
            if f_max is None or f >= f_max:
                f_max = f

        incr_norm = (self.incr_S * self.grad_S).sum()
        step_size = 1

        self.S += self.incr_S

        next_cost = self._evaluate_cost()

        error = 0
        factor = self.line_search_gamma * incr_norm
        while next_cost > f_max + factor * step_size:
            step_size = get_next_spg_step_length(
                step_size, incr_norm, current_cost, next_cost,
                sigma_one=self.line_search_sigma_one,
                sigma_two=self.line_search_sigma_two)

            self.S = self.S_old + step_size * self.incr_S

            next_cost = self._evaluate_cost()

            if abs(step_size) < self.step_size_tolerance:
                break

        return error, step_size

    def _update_dictionary(self):
        """Update dictionary using line-search."""

        self.GtX = self.Gamma.T.dot(self.X)
        self.GtG = self.Gamma.T.dot(self.Gamma)

        self._update_dictionary_gradient()

        self.incr_S = self.S - self.alpha_S * self.grad_S
        self.incr_S -= self.S

        error, step_size = self._dictionary_line_search()

        self.delta_grad_S = self.grad_S.copy()

        self._update_dictionary_gradient()

        self.delta_grad_S = self.grad_S - self.delta_grad_S

        sksk = step_size ** 2 * (self.incr_S * self.incr_S).sum()
        beta = step_size * (self.incr_S * self.delta_grad_S).sum()

        self.alpha_S = get_next_spg_alpha(
            beta, sksk,
            alpha_min=self.line_search_alpha_min,
            alpha_max=self.line_search_alpha_max)

        return error

    def _weights_penalty_gradient(self):
        """Evaluate gradient of weights penalty."""

        self.grad_Gamma = np.zeros_like(self.Gamma)

    def _update_weights_gradient(self):
        """Update gradient of cost function with respect to weights."""

        self._weights_penalty_gradient()

        n_samples, n_features = self.X.shape
        normalization = 1.0 / (n_features * n_samples)

        self.grad_Gamma += -2 * normalization * self.XSt
        self.grad_Gamma += 2 * normalization * self.Gamma.dot(self.SSt)

    def _weights_line_search(self):
        """Perform single step of line-search for weights."""

        current_cost = self._evaluate_cost()

        self.Gamma_old = self.Gamma.copy()

        self.f_Gamma_mem.popleft()
        self.f_Gamma_mem.append(current_cost)

        f_max = None
        for f in self.f_S_mem:
            if f_max is None or f >= f_max:
                f_max = f

        incr_norm = (self.incr_Gamma * self.grad_Gamma).sum()
        step_size = 1

        self.Gamma += self.incr_Gamma

        next_cost = self._evaluate_cost()

        error = 0
        factor = self.line_search_gamma * incr_norm
        while next_cost > f_max + factor * step_size:
            step_size = get_next_spg_step_length(
                step_size, incr_norm, current_cost, next_cost,
                sigma_one=self.line_search_sigma_one,
                sigma_two=self.line_search_sigma_two)

            self.Gamma = self.Gamma_old + step_size * self.incr_Gamma

            next_cost = self._evaluate_cost()

            if abs(step_size) < self.step_size_tolerance:
                break

        return error, step_size

    def _update_weights(self):
        """Update weights using line-search."""

        self.SSt = self.S.dot(self.S.T)
        self.XSt = self.X.dot(self.S.T)

        self._update_weights_gradient()

        self.incr_Gamma = self.Gamma - self.alpha_Gamma * self.grad_Gamma
        simplex_project_rows(self.incr_Gamma)
        self.incr_Gamma -= self.Gamma

        error, step_size = self._weights_line_search()

        self.delta_grad_Gamma = self.grad_Gamma.copy()

        self._update_weights_gradient()

        self.delta_grad_Gamma = self.grad_Gamma - self.delta_grad_Gamma

        sksk = step_size ** 2 * (self.incr_Gamma * self.incr_Gamma).sum()
        beta = step_size * (self.incr_Gamma * self.delta_grad_Gamma).sum()

        self.alpha_Gamma = get_next_spg_alpha(
            beta, sksk,
            alpha_min=self.line_search_alpha_min,
            alpha_max=self.line_search_alpha_max)

        return error

    def _iterate(self, update_dictionary=True, update_weights=True):
        """Minimize cost function iteratively."""

        if self.verbose:
            print("*** GPNH convex coding: n_components = {:d} ***".format(
                self.Gamma.shape[1]))
            print('{:<12s} | {:<13s} | {:<13s} | {:<12s}'.format(
                'Iteration', 'Cost', 'Cost delta', 'Time'))
            print(60 * '-')

        old_cost = self._evaluate_cost()
        new_cost = old_cost

        for n_iter in range(self.max_iterations):
            start_time = time.perf_counter()

            old_cost = new_cost

            if update_dictionary:
                self._update_dictionary()
                new_cost = self._evaluate_cost()
                if (new_cost > old_cost) and self.require_monotonic_cost_decrease:
                    raise RuntimeError(
                        'factorization cost increased after dictionary update')

            if update_weights:
                self._update_weights()
                new_cost = self._evaluate_cost()
                if (new_cost > old_cost) and self.require_monotonic_cost_decrease:
                    raise RuntimeError(
                        'factorization cost increased after weights update')

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

    def _gpnh_convex_coding(self, data, dictionary=None, weights=None,
                            update_dictionary=True, update_weights=True,
                            **kwargs):
        """Calculate GPNH-regularized convex coding of dataset."""

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

        if self.init == 'custom':
            _check_init_weights(weights, (n_samples, self.n_components),
                                '_gpnh_convex_coding (input weights)')
            _check_init_dictionary(dictionary, (self.n_components, n_features),
                                   '_gpnh_convex_coding (input dictionary)')
        elif not update_dictionary and update_weights:
            _check_init_dictionary(dictionary, (self.n_components, n_features),
                                   '_gpnh_convex_coding (input dictionary)')
            weights = _initialize_gpnh_convex_coding_weights(
                data, self.n_components, init=self.init,
                random_state=self.random_state, **kwargs)
        elif update_dictionary and not update_weights:
            _check_init_weights(weights, (n_samples, self.n_components),
                                '_gpnh_convex_coding (input weights)')
            dictionary = _initialize_gpnh_convex_coding_dictionary(
                data, self.n_components, init=self.init,
                random_state=self.random_state, **kwargs)
        else:
            dictionary, weights = _initialize_gpnh_convex_coding(
                data, self.n_components, init=self.init,
                random_state=self.random_state, **kwargs)

        self.X = data.copy()
        self.Gamma = weights.copy()
        self.S = dictionary.copy()

        self._initialize_workspace()

        cost, n_iter = self._iterate(
            update_dictionary=update_dictionary,
            update_weights=update_weights)

        if n_iter == self.max_iterations and self.tolerance > 0:
            warnings.warn('Maximum number of iterations %d reached.' %
                          self.max_iterations, UserWarning)

        return cost, n_iter

    def fit_transform(self, data, dictionary=None, weights=None, **kwargs):
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

        cost_, n_iter_ = self._gpnh_convex_coding(
            data,
            dictionary=dictionary,
            weights=weights, **kwargs)

        self.n_components_ = self.S.shape[0]
        self.cost_ = cost_
        self.dictionary_ = self.S
        self.n_iter_ = n_iter_

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

    def transform(self, data):
        """Transform the data according to the fitted factorization.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Data matrix for data to be transformed.

        Returns
        -------
        weights : array-like, shape (n_samples, n_features)
            Representation of the data.
        """

        self._gpnh_convex_coding(
            data=data,
            dictionary=self.S,
            update_dictionary=False, update_weights=True)

        return self.Gamma

    def inverse_transform(self, weights):
        """Transform data back into its original space.

        Parameters
        ----------
        weights : array-like, shape (n_samples, n_components)
            Representation of the data matrix.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
            Weights transformed to original space.
        """

        return weights.dot(self.S)
