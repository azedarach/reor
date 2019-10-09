"""
Provides routines for performing kernel AA.
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
    _check_unit_axis_sums(weights, whom, axis=1)


def _check_init_dictionary(dictionary, shape, whom):
    dictionary = check_array(dictionary)
    _check_array_shape(dictionary, shape, whom)
    _check_unit_axis_sums(dictionary, whom, axis=1)


def _initialize_kernel_aa_dictionary_random(
        kernel, n_components, random_state=None):
    rng = check_random_state(random_state)

    n_samples = kernel.shape[0]

    return right_stochastic_matrix((n_samples, n_components),
                                   random_state=rng)


def _initialize_kernel_aa_weights_random(
        kernel, n_components, random_state=None):
    rng = check_random_state(random_state)

    n_samples = kernel.shape[0]

    return right_stochastic_matrix((n_components, n_samples),
                                   random_state=rng)


def _initialize_kernel_aa_dictionary_furthest_sum(
        kernel, n_components, start_index=None, n_extra_steps=10,
        exclude=None, random_state=None):
    rng = check_random_state(random_state)

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

    dictionary = np.zeros((n_components, n_samples),
                          dtype=kernel.dtype)
    for i in range(n_components):
        dictionary[i, selected[i]] = 1

    return dictionary


def _initialize_kernel_aa_dictionary(kernel, n_components, init='furthest_sum',
                                     random_state=None, **kwargs):
    if init is None:
        init = 'furthest_sum'

    if init == 'furthest_sum':
        start_index = kwargs.get('start_index', None)
        n_extra_steps = kwargs.get('n_extra_steps', 10)
        exclude = kwargs.get('exclude', None)

        return _initialize_kernel_aa_dictionary_furthest_sum(
            kernel, n_components, start_index=start_index,
            n_extra_steps=n_extra_steps,
            exclude=exclude, random_state=random_state)

    if init == 'random':
        return _initialize_kernel_aa_dictionary_random(
            kernel, n_components, random_state=random_state)

    raise ValueError(
        'Invalid init parameter: got %r instead of one of %r' %
        (init, INITIALIZATION_METHODS))


def _initialize_kernel_aa_weights(kernel, n_components, init='furthest_sum',
                                  random_state=None):
    if init is None:
        init = 'furthest_sum'

    if init in ('furthest_sum', 'random'):
        return _initialize_kernel_aa_weights_random(
            kernel, n_components, random_state=random_state)

    raise ValueError(
        'Invalid init parameter: got %r instead of one of %r' %
        (init, INITIALIZATION_METHODS))


def _initialize_kernel_aa(kernel, n_components, init='furthest_sum',
                          random_state=None, **kwargs):
    if init is None:
        init = 'furthest_sum'

    rng = check_random_state(random_state)

    dictionary = _initialize_kernel_aa_dictionary(
        kernel, n_components, init=init, random_state=rng, **kwargs)

    weights = _initialize_kernel_aa_weights(
        kernel, n_components, init=init, random_state=rng, **kwargs)

    return dictionary, weights


class KernelAA():
    """Kernel archetypal analysis.

    Performs archetypal analysis given a kernel matrix computed
    from the original data.

    Parameters
    ----------
    n_components : integer or None
        If an integer, the number of archetypes. If None, then all
        samples are used.

    delta : float, default: 0
        Relaxation parameter for the dictionary.

    init : None | 'random' | 'furthest_sum' | 'custom'
        Method used to initialize the algorithm.
        Default: None
        Valid options:

        - None: falls back to 'furthest_sum'

        - 'random': dictionary and weights are initialized to
          random stochastic matrices.

        - 'furthest_sum': dictionary is initialized using FurthestSum
          method, and weights are initialized to a random stochastic
          matrix.

        - 'custom': use custom matrices for dictionary and weights.

    tolerance: float, default: 1e-6
        Tolerance of the stopping condition.

    max_iterations : integer, default: 1000
        Maximum number of iterations before stopping.

    verbose : integer, default: 0
        The verbosity level.

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    Attributes
    ----------
    dictionary_ : array-like, shape (n_samples, n_components)
        The dictionary containing the composition of the archetypes.

    cost_ : number
        Value of the cost function for the obtained factorization.

    n_iter_ : integer
        Actual number of iterations.

    Examples
    --------
    import numpy as np
    X = np.random.rand(10, 4)
    K = np.dot(X, X.T)
    from reor import KernelAA
    model = KernelAA(n_components=2, init='furthest_sum', random_state=0)
    weights = model.fit_transform(K)
    dictionary = model.dictionary_

    References
    ----------
    M. Morup and L. K. Hansen, "Archetypal analysis for machine learning
    and data mining", Neurocomputing 80 (2012) 54 - 63.
    """

    def __init__(self, n_components, **kwargs):
        self.n_components = n_components
        self.delta = kwargs.get('delta', 0.0)
        self.init = kwargs.get('init', None)
        self.tolerance = kwargs.get('tolerance', 1e-4)
        self.max_iterations = kwargs.get('max_iterations', 1000)
        self.verbose = kwargs.get('verbose', 0)
        self.random_state = kwargs.get('random_state', None)
        self.line_search_memory = kwargs.get('line_search_memory', 1)
        self.line_search_gamma = kwargs.get('gamma', 1e-4)
        self.line_search_alpha_min = kwargs.get('alpha_min', 1e-3)
        self.line_search_alpha_max = kwargs.get('alpha_max', 1e3)
        self.line_search_sigma_one = kwargs.get('sigma_one', 0.1)
        self.line_search_sigma_two = kwargs.get('sigma_two', 0.9)
        self.step_size_tolerance = kwargs.get('step_size_tolerance', 1e-12)
        self.require_monotonic_cost_decrease = kwargs.get(
            'require_monotonic_cost_decrease', True)

        self.K = None
        self.S = None
        self.C = None
        self.alpha = np.ones(n_components, dtype='f8')

        self.trace_K = 0.0
        self.CK = None
        self.CKCt = None
        self.CKS = None
        self.StS = None

        self.C_old = None
        self.grad_C = None
        self.delta_grad_C = None
        self.incr_C = None

        self.alpha_old = None
        self.grad_alpha = None
        self.delta_grad_alpha = None
        self.incr_alpha = None

        self.S_old = None
        self.grad_S = None
        self.delta_grad_S = None
        self.incr_S = None

        self.alpha_C = 1.0
        self.alpha_S = 1.0
        self.alpha_alpha = 1.0
        self.f_S_mem = deque([0.0] * self.line_search_memory,
                             maxlen=self.line_search_memory)
        self.f_C_mem = deque([0.0] * self.line_search_memory,
                             maxlen=self.line_search_memory)
        self.f_alpha_mem = deque([0.0] * self.line_search_memory,
                                 maxlen=self.line_search_memory)

    def _evaluate_cost(self):
        """Evaluate kernel AA cost function."""

        self.S.T.dot(self.S, out=self.StS)
        self.C.dot(self.K, out=self.CK)

        diag_alpha = np.diag(self.alpha)
        if self.delta > 0:
            diag_alpha.dot(self.CK, out=self.CK)

        self.CK.dot(self.C.T, out=self.CKCt)
        if self.delta > 0:
            self.CKCt.dot(diag_alpha, out=self.CKCt)

        return (self.StS.dot(self.CKCt).trace() -
                2 * self.S.dot(self.CK).trace() + self.K.trace())

    def _initialize_workspace(self):
        """Initialize temporary variables."""

        self.alpha = np.ones(self.n_components, dtype='f8')

        self.trace_K = self.K.trace()
        self.CK = self.C.dot(self.K)
        self.CKCt = self.CK.dot(self.C.T)
        self.CKS = self.CK.dot(self.S)
        self.StS = self.S.T.dot(self.S)

        self.C_old = self.C.copy()
        self.grad_C = np.empty_like(self.C)
        self.delta_grad_C = np.empty_like(self.C)
        self.incr_C = np.empty_like(self.C)

        self.alpha_old = self.alpha.copy()
        self.grad_alpha = np.empty_like(self.alpha)
        self.delta_grad_alpha = np.empty_like(self.alpha)
        self.incr_alpha = np.empty_like(self.alpha)

        self.S_old = self.S.copy()
        self.grad_S = np.empty_like(self.S)
        self.delta_grad_S = np.empty_like(self.S)
        self.incr_S = np.empty_like(self.S)

        self.alpha_C = 1.0
        self.alpha_S = 1.0
        self.alpha_alpha = 1.0
        self.f_S_mem = deque([0.0] * self.line_search_memory,
                             maxlen=self.line_search_memory)
        self.f_C_mem = deque([0.0] * self.line_search_memory,
                             maxlen=self.line_search_memory)
        self.f_alpha_mem = deque([0.0] * self.line_search_memory,
                                 maxlen=self.line_search_memory)

    def _update_weights_gradient(self):
        """Evaluate gradient of cost function with respect to weights."""

        self.K.T.dot(-2 * self.C.T, out=self.grad_S)

        if self.delta > 0:
            diag_alpha = np.diag(self.alpha)
            self.grad_S.dot(diag_alpha, out=self.grad_S)

        self.grad_S += self.S.dot(2 * self.CKCt)


    def _weights_line_search(self):
        """Perform single step of line-search for weights."""

        current_cost = (self.trace_K - 2 * self.S.dot(self.CK).trace() +
                        self.StS.dot(self.CKCt).trace())

        self.S_old = self.S.copy()

        self.f_S_mem.popleft()
        self.f_S_mem.append(current_cost)

        f_max = None
        for f in self.f_S_mem:
            if f_max is None or f > f_max:
                f_max = f

        incr_norm = (self.incr_S * self.grad_S).sum()
        step_size = 1

        self.S += self.incr_S

        self.S.T.dot(self.S, out=self.StS)

        next_cost = (self.trace_K - 2 * self.S.dot(self.CK).trace() +
                     self.StS.dot(self.CKCt).trace())

        error = 0
        factor = self.line_search_gamma * incr_norm
        while next_cost > f_max + factor * step_size:
            step_size = get_next_spg_step_length(
                step_size, incr_norm, current_cost, next_cost,
                sigma_one=self.line_search_sigma_one,
                sigma_two=self.line_search_sigma_two)

            self.S = self.S_old + step_size * self.incr_S

            self.S.T.dot(self.S, out=self.StS)

            next_cost = (self.trace_K - 2 * self.S.dot(self.CK).trace() +
                         self.StS.dot(self.CKCt).trace())

            if abs(step_size) < self.step_size_tolerance:
                break

        return error, step_size

    def _update_weights(self):
        """Update weights using line-search."""

#        self.S.T.dot(self.S, out=self.StS)

        self.C.dot(self.K, out=self.CK)

        diag_alpha = np.diag(self.alpha)
        if self.delta > 0:
            diag_alpha.dot(self.CK, out=self.CK)

        self.CK.dot(self.C.T, out=self.CKCt)
        if self.delta > 0:
            self.CKCt.dot(diag_alpha, out=self.CKCt)

        self._update_weights_gradient()

        self.incr_S = self.S - self.alpha_S * self.grad_S
        simplex_project_rows(self.incr_S)
        self.incr_S -= self.S

        error, step_size = self._weights_line_search()

        self.delta_grad_S = self.grad_S.copy()

        self._update_weights_gradient()

        self.delta_grad_S = self.grad_S - self.delta_grad_S

        sksk = step_size ** 2 * (self.incr_S * self.incr_S).sum()
        beta = step_size * (self.incr_S * self.delta_grad_S).sum()

        self.alpha_S = get_next_spg_alpha(
            beta, sksk,
            alpha_min=self.line_search_alpha_min,
            alpha_max=self.line_search_alpha_max)

        return error

    def _update_dictionary_gradient(self):
        """Update gradient of cost function with respect to dictionary."""

        self.S.T.dot(-2 * self.K.T, out=self.grad_C)

        self.grad_C += 2 * self.StS.dot(self.CK)
        if self.delta > 0:
            diag_alpha = np.diag(self.alpha)
            diag_alpha.dot(self.grad_C, out=self.grad_C)

    def _dictionary_line_search(self):
        """Perform single-step of line search for dictionary."""

        current_cost = (self.trace_K - 2 * self.S.dot(self.CK).trace() +
                        self.StS.dot(self.CKCt).trace())

        self.C_old = self.C.copy()

        self.f_C_mem.popleft()
        self.f_C_mem.append(current_cost)

        f_max = None
        for f in self.f_C_mem:
            if f_max is None or f > f_max:
                f_max = f

        incr_norm = (self.incr_C * self.grad_C).sum()
        step_size = 1

        self.C += self.incr_C

        self.C.dot(self.K, out=self.CK)

        diag_alpha = np.diag(self.alpha)
        if self.delta > 0:
            diag_alpha.dot(self.CK, out=self.CK)

        self.CK.dot(self.C.T, out=self.CKCt)
        if self.delta > 0:
            self.CKCt.dot(diag_alpha, out=self.CKCt)

        next_cost = (self.trace_K - 2 * self.S.dot(self.CK).trace() +
                     self.StS.dot(self.CKCt).trace())

        error = 0
        factor = self.line_search_gamma * incr_norm
        while next_cost > f_max + factor * step_size:
            step_size = get_next_spg_step_length(
                step_size, incr_norm, current_cost, next_cost,
                sigma_one=self.line_search_sigma_one,
                sigma_two=self.line_search_sigma_two)

            self.C = self.C_old + step_size * self.incr_C

            self.C.dot(self.K, out=self.CK)
            if self.delta > 0:
                diag_alpha.dot(self.CK, out=self.CK)

            self.CK.dot(self.C.T, out=self.CKCt)
            if self.delta > 0:
                self.CKCt.dot(diag_alpha, out=self.CKCt)

            next_cost = (self.trace_K - 2 * self.S.dot(self.CK).trace() +
                         self.StS.dot(self.CKCt).trace())

            if abs(step_size) < self.step_size_tolerance:
                break

        return error, step_size

    def _update_scale_factors_gradient(self):
        """Update cost function gradient with respect to scale factors."""

        diag_alpha = np.diag(self.alpha)

        grad_alpha_tmp = 2 * diag_alpha.dot(self.CKCt.T)
        self.StS.dot(grad_alpha_tmp, out=grad_alpha_tmp)

        grad_alpha_tmp += -2 * self.CKS.T

        self.grad_alpha = grad_alpha_tmp.diagonal()

    def _scale_factors_line_search(self):
        """Perform single step of line-search for scale factors."""

        CKCt_tmp = self.CKCt.copy()

        diag_alpha = np.diag(self.alpha)
        self.CKCt.dot(diag_alpha, out=CKCt_tmp)
        diag_alpha.dot(CKCt_tmp, out=CKCt_tmp)

        current_cost = (self.trace_K - 2 * self.CKS.dot(diag_alpha).trace() +
                        self.StS.dot(CKCt_tmp).trace())

        diag_alpha_old = diag_alpha.copy()

        self.f_alpha_mem.popleft()
        self.f_alpha_mem.append(current_cost)

        f_max = None
        for f in self.f_alpha_mem:
            if f_max is None or f > f_max:
                f_max = f

        incr_norm = (self.incr_alpha * self.grad_alpha).sum()
        step_size = 1

        diag_alpha = diag_alpha_old + self.incr_alpha

        self.CKCt.dot(diag_alpha, out=CKCt_tmp)
        diag_alpha.dot(CKCt_tmp, out=CKCt_tmp)

        next_cost = (self.trace_K - 2 * self.CKS.dot(diag_alpha).trace() +
                     self.StS.dot(CKCt_tmp).trace())

        error = 0
        factor = self.line_search_gamma * incr_norm
        while next_cost > f_max + factor * step_size:
            step_size = get_next_spg_step_length(
                step_size, incr_norm, current_cost, next_cost,
                sigma_one=self.line_search_sigma_one,
                sigma_two=self.line_search_sigma_two)

            diag_alpha = diag_alpha_old + step_size * self.incr_alpha

            self.CKCt.dot(diag_alpha, out=CKCt_tmp)
            diag_alpha.dot(CKCt_tmp, out=CKCt_tmp)

            next_cost = (self.trace_K - 2 * self.CKS.dot(diag_alpha).trace() +
                         self.StS.dot(CKCt_tmp).trace())

            if abs(step_size) < self.step_size_tolerance:
                break

        return error, step_size

    def _update_dictionary(self):
        """Update dictionary using line-search."""

        self.S.T.dot(self.S, out=self.StS)

        self.C.dot(self.K, out=self.CK)
        diag_alpha = np.diag(self.alpha)
        if self.delta > 0:
            diag_alpha.dot(self.CK, out=self.CK)

        self.CK.dot(self.C.T, out=self.CKCt)
        if self.delta > 0:
            self.CKCt.dot(diag_alpha, out=self.CKCt)

        self._update_dictionary_gradient()

        self.incr_C = self.C - self.alpha_C * self.grad_C
        simplex_project_rows(self.incr_C)
        self.incr_C -= self.C

        error, step_size = self._dictionary_line_search()

        self.delta_grad_C = self.grad_C.copy()

        self._update_dictionary_gradient()

        self.delta_grad_C = self.grad_C - self.delta_grad_C

        sksk = step_size ** 2 * (self.incr_C * self.incr_C).sum()
        beta = step_size * (self.incr_C * self.delta_grad_C).sum()

        self.alpha_C = get_next_spg_alpha(
            beta, sksk,
            alpha_min=self.line_search_alpha_min,
            alpha_max=self.line_search_alpha_max)

        if self.delta > 0:
            self.C.dot(self.K, out=self.CK)
            self.CK.dot(self.S, out=self.CKS)
            self.CK.dot(self.C.T, out=self.CKCt)

            self._update_scale_factors_gradient()

            self.incr_alpha = diag_alpha - self.alpha_alpha * self.grad_alpha
            np.fmin(self.incr_alpha, 1 - self.delta, out=self.incr_alpha)
            np.fmax(self.incr_alpha, 1 + self.delta, out=self.incr_alpha)
            self.incr_alpha -= diag_alpha

            scales_error, step_size = self._scale_factors_line_search()

            self.delta_grad_alpha = self.grad_alpha.copy()

            self._update_scale_factors_gradient()

            self.delta_grad_alpha = self.grad_alpha - self.delta_grad_alpha

            sksk = step_size ** 2 * (self.incr_alpha * self.incr_alpha).sum()
            beta = step_size * (self.incr_alpha * self.delta_grad_alpha).sum()

            self.alpha_alpha = get_next_spg_alpha(
                beta, sksk,
                alpha_min=self.line_search_alpha_min,
                alpha_max=self.line_search_alpha_max)

            if scales_error > error:
                error = scales_error

        return error

    def _iterate(self, update_dictionary=True, update_weights=True):
        """Iteratively minimize cost function."""

        if self.verbose:
            print("*** Kernel AA: n_components = {:d} ***".format(
                self.S.shape[1]))
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

    def _kernel_aa(self, kernel, dictionary=None, weights=None,
                   update_dictionary=True, update_weights=True, **kwargs):
        """Perform kernel archetypal analysis."""

        n_samples = kernel.shape[0]

        if kernel.shape[1] != n_samples:
            raise ValueError(
                'Expected square kernel matrix in %s. '
                'Got shape %s' % ('kernel_aa', kernel.shape))

        if self.n_components is None:
            self.n_components = n_samples

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
                                '_kernel_aa (input weights)')
            _check_init_dictionary(dictionary, (self.n_components, n_samples),
                                   '_kernel_aa (input dictionary)')
        elif not update_dictionary and update_weights:
            _check_init_dictionary(dictionary, (self.n_components, n_samples),
                                   '_kernel_aa (input dictionary)')
            weights = _initialize_kernel_aa_weights(
                kernel, self.n_components, init=self.init,
                random_state=self.random_state, **kwargs)
        elif update_dictionary and not update_weights:
            _check_init_weights(weights, (n_samples, self.n_components),
                                '_kernel_aa (input weights)')
            dictionary = _initialize_kernel_aa_dictionary(
                kernel, self.n_components, init=self.init,
                random_state=self.random_state, **kwargs)
        else:
            dictionary, weights = _initialize_kernel_aa(
                kernel, self.n_components, init=self.init,
                random_state=self.random_state, **kwargs)

        self.K = kernel.copy()
        self.S = weights.copy()
        self.C = dictionary.copy()

        self._initialize_workspace()

        cost, n_iter = self._iterate(
            update_dictionary=update_dictionary,
            update_weights=update_weights)

        if n_iter == self.max_iterations and self.tolerance > 0:
            warnings.warn('Maximum number of iterations %d reached.' %
                          self.max_iterations, UserWarning)

        return cost, n_iter

    def fit_transform(self, kernel, dictionary=None, weights=None, **kwargs):
        """Perform kernel archetypal analysis and return transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        kernel : array-like, shape (n_samples, n_samples)
            Kernel matrix to be factorized.

        dictionary : array-like, shape (n_samples, n_components)
            If init='custom', used as initial guess for solution.

        weights : array-like, shape (n_components, n_samples)
            If init='custom', used as initial guess for solution.

        Returns
        -------
        weights : array-like, shape (n_components, n_samples)
            Representation of the data.
        """

        cost_, n_iter_ = self._kernel_aa(
            kernel,
            dictionary=dictionary,
            weights=weights, **kwargs)

        self.n_components_ = self.C.shape[0]
        self.cost_ = cost_
        self.dictionary_ = self.C
        self.n_iter_ = n_iter_

        return self.S

    def fit(self, kernel, **kwargs):
        """Perform kernel archetypal analysis on given kernel.

        Parameters
        ----------
        kernel : array-like, shape (n_samples, n_samples)
            Kernel matrix to perform analysis on.

        Returns
        -------
        self
        """
        self.fit_transform(kernel, **kwargs)
        return self

    def transform(self, kernel):
        """Transform the data according to the fitted factorization.

        Parameters
        ----------
        kernel : array-like, shape (n_samples, n_samples)
            Kernel matrix for data to be transformed.

        Returns
        -------
        weights : array-like, shape (n_components, n_samples)
            Representation of the data.
        """

        self._kernel_aa(
            kernel=kernel,
            dictionary=self.C,
            update_dictionary=False, update_weights=True)

        return self.S

    def inverse_transform(self, weights):
        """Transform data back into its original space.

        Parameters
        ----------
        weights : array-like, shape (n_components, n_samples)
            Representation of the data matrix.

        Returns
        -------
        coefficients : array-like, shape (n_samples, n_samples)
            Matrix of coefficients for original data.
        """

        return weights.dot(self.C)


class ArchetypalAnalysis(KernelAA):
    """Standard archetypal analysis.

    Performs archetypal analysis by minimizing the cost function::

        ||X - S C X||_Fro^2

    by performing a series of alternating minimizations with
    respect to the dictionary C and weights S.

    Parameters
    ----------
    n_components : integer or None
        If an integer, the number of archetypes. If None, then all
        samples are used.

    kernel_func : None or callable
        If None, a kernel is computed from the given data
        using the Euclidean inner product. If callable, should
        take a data matrix with columns X_i and return a kernel
        matrix with elements K_{ij} = K(X_i, K_j).

    delta : float, default: 0
        Relaxation parameter for the dictionary.

    init : None | 'random' | 'furthest_sum' | 'custom'
        Method used to initialize the algorithm.
        Default: None
        Valid options:

        - None: falls back to 'furthest_sum'

        - 'random': dictionary and weights are initialized to
          random stochastic matrices.

        - 'furthest_sum': dictionary is initialized using FurthestSum
          method, and weights are initialized to a random stochastic
          matrix.

        - 'custom': use custom matrices for dictionary and weights.

    tolerance: float, default: 1e-6
        Tolerance of the stopping condition.

    max_iterations : integer, default: 1000
        Maximum number of iterations before stopping.

    verbose : integer, default: 0
        The verbosity level.

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    backend : 'eigen'
        The backend used by the factorization solver.

    Attributes
    ----------
    dictionary_ : array-like, shape (n_samples, n_components)
        The dictionary containing the composition of the archetypes.

    cost_ : number
        Value of the cost function for the obtained factorization.

    n_iter_ : integer
        Actual number of iterations.

    Examples
    --------
    import numpy as np
    X = np.random.rand(4, 10)
    from reor.archetypal_analysis import ArchetypalAnalysis
    model = ArchetypalAnalysis(n_components=2, init='furthest_sum',
                               random_state=0)
    weights = model.fit_transform(X)
    dictionary = model.dictionary_

    References
    ----------
    M. Morup and L. K. Hansen, "Archetypal analysis for machine learning
    and data mining", Neurocomputing 80 (2012) 54 - 63.
    """
    def __init__(self, n_components, kernel_func=None, **kwargs):
        super().__init__(n_components, **kwargs)

        self.kernel_func = kernel_func

    def fit_transform(self, data, dictionary=None, weights=None, **kwargs):
        """Perform archetypal analysis and return transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        data : array-like, shape (n_features, n_samples)
            Data matrix to be factorized.

        dictionary : array-like, shape (n_samples, n_components)
            If init='custom', used as initial guess for solution.

        weights : array-like, shape (n_components, n_samples)
            If init='custom', usede as initial guess for solution.

        Returns
        -------
        weights : array-like, shape (n_components, n_samples)
            Representation of the data.
        """
        if self.kernel_func is None:
            kernel = np.dot(data, data.T)
        else:
            kernel = self.kernel_func(data)

        cost_, n_iter_ = self._kernel_aa(
            kernel,
            dictionary=dictionary,
            weights=weights, **kwargs)

        self.n_components_ = self.C.shape[0]
        self.cost_ = cost_
        self.dictionary_ = self.C
        self.n_iter_ = n_iter_

        return self.S

    def transform(self, data):
        """Transform the data according to the fitted factorization.

        Parameters
        ----------
        data : array-like, shape (n_features, n_samples)
            Data matrix to be transformed.

        Returns
        -------
        weights : array-like, shape (n_components, n_samples)
            Representation of the data.
        """

        if self.kernel_func is None:
            kernel = data.dot(data.T)
        else:
            kernel = self.kernel_func(data)

        self._kernel_aa(kernel=kernel, dictionary=self.C,
                        update_dictionary=False, update_weights=True)

        return self.S
