"""
Provides routines for performing kernel AA.
"""


from __future__ import print_function

import numbers
import time
import warnings

import numpy as np

from sklearn.utils import check_array, check_random_state

from . import reor_ext
from ._random_matrix import left_stochastic_matrix
from ._validation import _check_backend


INTEGER_TYPES = (numbers.Integral, np.integer)

INITIALIZATION_METHODS = (None, 'random', 'furthest_sum')


def _check_unit_axis_sums(A, whom, axis=0):
    axis_sums = np.sum(A, axis=axis)
    if not np.all(np.isclose(axis_sums, 1)):
        raise ValueError(
            'Array with incorrect axis sums passed to %s. '
            'Expected sums along axis %d to be 1.'
            % (whom, axis))


def _check_array_shape(A, shape, whom):
    if np.shape(A) != shape:
        raise ValueError(
            'Array with wrong shape passed to %s. '
            'Expected %s, but got %s' % (whom, shape, np.shape(A)))


def _check_init_weights(weights, shape, whom):
    weights = check_array(weights)
    _check_array_shape(weights, shape, whom)
    _check_unit_axis_sums(weights, whom, axis=0)


def _check_init_dictionary(dictionary, shape, whom):
    dictionary = check_array(dictionary)
    _check_array_shape(dictionary, shape, whom)
    _check_unit_axis_sums(dictionary, whom, axis=0)


def _initialize_kernel_aa_dictionary_random(
        kernel, n_components, random_state=None):
    rng = check_random_state(random_state)

    n_samples = kernel.shape[0]

    return left_stochastic_matrix((n_samples, n_components),
                                  random_state=rng)


def _initialize_kernel_aa_weights_random(
        kernel, n_components, random_state=None):
    rng = check_random_state(random_state)

    n_samples = kernel.shape[0]

    return left_stochastic_matrix((n_components, n_samples),
                                  random_state=rng)


def _initialize_kernel_aa_dictionary_furthest_sum(
        kernel, n_components, start_index=None, n_extra_steps=10,
        exclude=None, random_state=None, backend='eigen'):
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

    if backend == 'eigen':
        selected = reor_ext.furthest_sum_eigen(
            dissimilarities, n_components, start_index, exclude, n_extra_steps)
    else:
        raise ValueError("Invalid backend parameter '%s'." % backend)

    dictionary = np.zeros((n_samples, n_components),
                          dtype=kernel.dtype)
    for i in range(n_components):
        dictionary[selected[i], i] = 1

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


def _iterate_kernel_aa_eigen(kernel, dictionary, weights,
                             delta=0,
                             update_dictionary=True,
                             update_weights=True,
                             tolerance=1e-6,
                             max_iterations=1000,
                             verbose=0,
                             require_monotonic_cost_decrease=True):
    _check_backend('eigen', '_iterate_kernel_aa_eigen')

    if verbose:
        print("*** Kernel AA: n_components = {:d} (backend = '{}') ***".format(
            weights.shape[0], 'eigen'))
        print('{:<12s} | {:<13s} | {:<13s} | {:<12s}'.format(
            'Iteration', 'Cost', 'Cost delta', 'Time'))
        print(60 * '-')

    solver = reor_ext.EigenKernelAA(kernel, dictionary, weights, delta)

    old_cost = solver.cost()
    new_cost = old_cost

    for n_iter in range(max_iterations):
        start_time = time.perf_counter()

        old_cost = new_cost

        if update_dictionary:
            solver.update_dictionary()
            new_cost = solver.cost()
            if (new_cost > old_cost) and require_monotonic_cost_decrease:
                raise RuntimeError(
                    'factorization cost increased after dictionary update')

        if update_weights:
            solver.update_weights()
            new_cost = solver.cost()
            if (new_cost > old_cost) and require_monotonic_cost_decrease:
                raise RuntimeError(
                    'factorization cost increased after weights update')

        cost_delta = new_cost - old_cost

        end_time = time.perf_counter()

        if verbose:
            print('{:12d} | {: 12.6e} | {: 12.6e} | {: 12.6e}'.format(
                n_iter + 1, new_cost, cost_delta, end_time - start_time))

        if abs(cost_delta) < tolerance:
            if verbose:
                print('*** Converged at iteration {:d} ***'.format(
                    n_iter + 1))
            break

    return solver.get_dictionary(), solver.get_weights(), n_iter


def kernel_aa(kernel, dictionary=None, weights=None,
              n_components=None, delta=0,
              update_dictionary=True, update_weights=True,
              init=None, tolerance=1e-6,
              max_iterations=1000,
              verbose=0, random_state=None, backend=None, **kwargs):
    r"""Perform kernel archetypal analysis.

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

    backend : 'eigen'
        The backend used by the factorization solver.

    Returns
    -------
    dictionary : array-like, shape (n_samples, n_components)
        Solution for the dictionary C in X ~= X C S.

    weights : array-like, shape (n_components, n_samples)
        Solution for the weights S in X ~= X C S.

    n_iter : integer
        The number of iterations done in the algorithm.
    """
    n_samples = kernel.shape[0]

    if kernel.shape[1] != n_samples:
        raise ValueError(
            'Expected square kernel matrix in %s. '
            'Got shape %s' % ('kernel_aa', kernel.shape))

    if n_components is None:
        n_components = n_samples

    if not isinstance(n_components, INTEGER_TYPES) or n_components <= 0:
        raise ValueError('Number of components must be a positive integer;'
                         ' got (n_components=%r)' % n_components)
    if not isinstance(max_iterations, INTEGER_TYPES) or max_iterations <= 0:
        raise ValueError('Maximum number of iterations must be a positive '
                         'integer; got (max_iterations=%r)' % max_iterations)
    if not isinstance(tolerance, numbers.Number) or tolerance < 0:
        raise ValueError('Tolerance for stopping criteria must be '
                         'positive; got (tolerance=%r)' % tolerance)

    if init == 'custom' and update_dictionary and update_weights:
        n_features = dictionary.shape[0]
        _check_init_weights(weights, (n_components, n_samples),
                            'kernel_aa (input weights)')
        _check_init_dictionary(dictionary, (n_features, n_components),
                               'kernel_aa (input dictionary)')
    elif not update_dictionary and update_weights:
        n_features = dictionary.shape[0]
        _check_init_dictionary(dictionary, (n_features, n_components),
                               'kernel_aa (input dictionary)')
        weights = _initialize_kernel_aa_weights(kernel, n_components, init=init,
                                                random_state=random_state,
                                                **kwargs)
    elif update_dictionary and not update_weights:
        _check_init_weights(weights, (n_components, n_samples),
                            'kernel_aa (input weights)')
        dictionary = _initialize_kernel_aa_dictionary(
            kernel_aa, n_components, init=init,
            random_state=random_state, **kwargs)
    else:
        dictionary, weights = _initialize_kernel_aa(kernel, n_components,
                                                    init=init,
                                                    random_state=random_state,
                                                    **kwargs)

    if backend == 'eigen':
        dictionary, weights, n_iter = _iterate_kernel_aa_eigen(
            kernel, dictionary, weights, delta=delta,
            update_dictionary=update_dictionary,
            update_weights=update_weights,
            tolerance=tolerance, max_iterations=max_iterations,
            verbose=verbose)
    else:
        raise ValueError("Invalid backend parameter '%s'." % backend)

    if n_iter == max_iterations and tolerance > 0:
        warnings.warn('Maximum number of iterations %d reached.' %
                      max_iterations,
                      UserWarning)

    return dictionary, weights, n_iter


class KernelAA():
    r"""Kernel archetypal analysis.

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
    K = np.dot(X.T, X)
    from pyreor import KernelAA
    model = KernelAA(n_components=2, init='furthest_sum', random_state=0)
    weights = model.fit_transform(K)
    dictionary = model.dictionary_

    References
    ----------
    M. Morup and L. K. Hansen, "Archetypal analysis for machine learning
    and data mining", Neurocomputing 80 (2012) 54 - 63.
    """
    def __init__(self, n_components, delta=0, init=None,
                 tolerance=1e-6, max_iterations=1000,
                 verbose=0, random_state=None,
                 backend='builtin'):
        self.n_components = n_components
        self.delta = delta
        self.init = init
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.random_state = random_state
        self.backend = backend
    
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
        dictionary_, weights_, n_iter_ = kernel_aa(
            kernel,
            dictionary=dictionary,
            weights=weights,
            n_components=self.n_components,
            delta=self.delta,
            init=self.init,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            random_state=self.random_state,
            backend=self.backend, **kwargs)

        self.n_components_ = dictionary_.shape[1]
        self.dictionary_ = dictionary_
        self.n_iter_ = n_iter_

        return weights_

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

        _, weights, _ = kernel_aa(
            kernel=kernel,
            dictionary=self.dictionary_,
            n_components=self.n_components_,
            delta=self.delta,
            init=self.init,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            random_state=self.random_state,
            backend=self.backend)

        return weights

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

        return np.dot(self.dictionary_, weights)


class ArchetypalAnalysis():
    r"""Standard archetypal analysis.

    Performs archetypal analysis by minimizing the cost function::

        ||X - X C S ||_Fro^2

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
    from pyreor import ArchetypalAnalysis
    model = KernelAA(n_components=2, init='furthest_sum', random_state=0)
    weights = model.fit_transform(X)
    dictionary = model.dictionary_

    References
    ----------
    M. Morup and L. K. Hansen, "Archetypal analysis for machine learning
    and data mining", Neurocomputing 80 (2012) 54 - 63.
    """
    def __init__(self, n_components, kernel_func=None,
                 delta=0, init=None,
                 tolerance=1e-6, max_iterations=1000,
                 verbose=0, random_state=None,
                 backend='builtin'):
        self.n_components = n_components
        self.kernel_func = kernel_func
        self.delta = delta
        self.init = init
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.random_state = random_state
        self.backend = backend

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
            kernel = np.dot(data.T, data)
        else:
            kernel = self.kernel_func(data)

        dictionary_, weights_, n_iter_ = kernel_aa(
            kernel,
            dictionary=dictionary,
            weights=weights,
            n_components=self.n_components,
            delta=self.delta,
            init=self.init,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            random_state=self.random_state,
            backend=self.backend, **kwargs)

        self.n_components_ = dictionary_.shape[1]
        self.dictionary_ = dictionary_
        self.n_iter_ = n_iter_

        return weights_

    def fit(self, data, **kwargs):
        """Perform archetypal analysis on given data.

        Parameters
        ----------
        data : array-like, shape (n_features, n_samples)
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
        data : array-like, shape (n_features, n_samples)
            Data matrix to be transformed.

        Returns
        -------
        weights : array-like, shape (n_components, n_samples)
            Representation of the data.
        """

        if self.kernel_func is None:
            kernel = np.dot(data.T, data)
        else:
            kernel = self.kernel_func(data)

        _, weights, _ = kernel_aa(
            kernel=kernel,
            dictionary=self.dictionary_,
            n_components=self.n_components_,
            delta=self.delta,
            init=self.init,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            random_state=self.random_state,
            backend=self.backend)

        return weights

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

        return np.dot(self.dictionary_, weights)
