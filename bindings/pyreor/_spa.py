from __future__ import print_function

import numbers
import warnings

import numpy as np

from sklearn.utils import check_array, check_random_state

from . import pyreor_ext
from ._random_matrix import left_stochastic_matrix
from ._validation import _check_backend

INTEGER_TYPES = (numbers.Integral, np.integer)

INITIALIZATION_METHODS = (None, 'random')


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


def _initialize_l2_spa_dictionary_random(data, n_components, random_state=None):
    rng = check_random_state(random_state)

    n_features = data.shape[0]
    avg = np.sqrt(np.abs(data).mean() / n_components)
    dictionary = avg * rng.randn(n_features, n_components)

    return dictionary


def _initialize_l2_spa_weights_random(data, n_components, random_state=None):
    rng = check_random_state(random_state)

    n_samples = data.shape[1]

    return left_stochastic_matrix((n_components, n_samples), random_state=rng)


def _initialize_l2_spa_dictionary(data, n_components, init='random',
                                  random_state=None):
    if init is None:
        init = 'random'

    if init == 'random':
        return _initialize_l2_spa_dictionary_random(data, n_components,
                                                    random_state=random_state)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, INITIALIZATION_METHODS))


def _initialize_l2_spa_weights(data, n_components, init='random',
                               random_state=None):
    if init is None:
        init = 'random'

    if init == 'random':
        return _initialize_l2_spa_weights_random(data, n_components,
                                                 random_state=random_state)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, INITIALIZATION_METHODS))


def _initialize_l2_spa(data, n_components, init='random', random_state=None):
    if init is None:
        init = 'random'

    rng = check_random_state(random_state)

    dictionary = _initialize_l2_spa_dictionary(data, n_components, init=init,
                                               random_state=rng)
    weights = _initialize_l2_spa_weights(data, n_components, init=init,
                                         random_state=rng)

    return dictionary, weights


def _iterate_l2_spa_gpnh_eigen(data, dictionary, weights,
                               epsilon_states=0,
                               update_dictionary=True,
                               update_weights=True,
                               tolerance=1e-6,
                               max_iterations=1000,
                               verbose=0,
                               require_monotonic_cost_decrease=True):
    _check_backend('eigen', '_iterate_l2_spa_gpnh_eigen')

    solver = pyreor_ext.EigenL2SPAGPNH(data, dictionary, weights)
    solver.epsilon_states = epsilon_states

    old_cost = solver.cost()
    new_cost = old_cost
    initial_cost = old_cost

    for n_iter in range(max_iterations):
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

        if verbose:
            print('Iteration %d:' % (n_iter + 1))
            print('-----------------')
            print('Cost function = %.5e' % new_cost)
            print('Current cost delta = %.5e' % cost_delta)
            print('Total cost ratio = %.5e' % (new_cost / initial_cost))

        if abs(cost_delta) < tolerance:
            if verbose:
                print('*** Converged at iteration %d ***' % (n_iter + 1))
            break

    return solver.get_dictionary(), solver.get_weights(), n_iter


def l2_spa_gpnh(data, dictionary=None, weights=None,
                n_components=None, epsilon_states=0,
                update_dictionary=True, update_weights=True,
                init=None, tolerance=1e-6,
                max_iterations=1000,
                verbose=0, random_state=None, backend=None):
    r"""Compute l2-SPA factorization with GPNH regularization.

    The objective function is:

        ||X - S Gamma||_Fro^2 / (n_samples * n_features)
        + epsilon_states * \Phi_{GPNH}(S)

    where the data matrix X has shape (n_features, n_samples) and::

        \Phi_{GPNH}(S) = \sum_{i, j = 1}^{n_components} ||s_i - s_j||_2^2
                         / (n_features * n_components * (n_components - 1))

    The objective function is minimized with an alternating minimization
    with respect to the dictionary S and weights Gamma. If
    update_dictionary=False or update_weights=False, it solves only for
    the weights or dictionary, respectively.

    Parameters
    ----------
    data : array-like, shape (n_features, n_samples)
        The data matrix to be factorized.

    dictionary : array-like, shape (n_features, n_components)
        If init='custom', used as initial guess for the solution.
        If update_dictionary=False, used as a constant to solve for
        the weights only.

    weights : array-like, shape (n_components, n_samples)
        If init='custom', used as initial guess for the solution.
        If update_weights=False, used as a constant to solve for
        the dictionary only.

    n_components : integer or None
        If an integer, the number of components. If None, then all
        features are kept.

    epsilon_states : float, default: 0
        Regularization parameter for the dictionary.

    update_dictionary : boolean, default: True
        If True, the dictionary will be estimated from initial guesses.
        If False, the given dictionary will be used as a constant.

    update_weights : boolean, default: True
        If True, the weights will be estimated from initial guesses.
        If False, the given weights will be used as a constant.

    init : None | 'random' | 'custom'
        Method used to initialize the algorithm.
        Default: None
        Valid options:

        - None: falls back to 'random'.

        - 'random': random matrix of dictionary elements scaled by
          sqrt(X.mean() / n_components), and a random stochastic
          matrix of weights.

        - 'custom': use custom matrices for dictionary and weights.

    tolerance : float, default : 1e-6
        Tolerance of the stopping condition.

    max_iterations : integer, default: 1000
        Maximum number of iterations before stopping.

    verbose : integer, default : 0
        The verbosity level.

    random_state : integer, RandomState, or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    backend : 'eigen'
        The backend used by the factorization solver.

    Returns
    -------
    dictionary : array-like, shape (n_features, n_components)
        Solution for the dictionary S in X ~= S Gamma.

    weights : array-like, shape (n_components, n_samples)
        Solution for the weights Gamma in X ~= S Gamma.

    n_iter : integer
        The number of iterations done in the algorithm.
    """

    if data.ndim == 1:
        n_samples = data.shape[0]
        n_features = 1
        X = np.reshape(data, (n_features, n_samples))
    else:
        n_features, n_samples = data.shape
        X = data

    if n_components is None:
        n_components = n_features

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
        _check_init_weights(weights, (n_components, n_samples),
                            'l2_spa_gpnh (input weights)')
        _check_init_dictionary(dictionary, (n_features, n_components),
                               'l2_spa_gpnh (input dictionary)')
    elif not update_dictionary and update_weights:
        _check_init_dictionary(dictionary, (n_features, n_components),
                               'l2_spa_gpnh (input dictionary)')
        weights = _initialize_l2_spa_weights(X, n_components, init=init,
                                             random_state=random_state)
    elif update_dictionary and not update_weights:
        _check_init_weights(weights, (n_components, n_samples),
                            'l2_spa_gpnh (input weights)')
        dictionary = _initialize_l2_spa_dictionary(
            X, n_components, init=init,
            random_state=random_state)
    else:
        dictionary, weights = _initialize_l2_spa(X, n_components, init=init,
                                                 random_state=random_state)

    if backend == 'eigen':
        dictionary, weights, n_iter = _iterate_l2_spa_gpnh_eigen(
            X, dictionary, weights, epsilon_states=epsilon_states,
            update_dictionary=update_dictionary,
            update_weights=update_weights,
            tolerance=tolerance, max_iterations=max_iterations,
            verbose=verbose)
    else:
        raise ValueError("Invalid backend parameter '%s'." % backend)

    if n_iter == max_iterations and tolerance > 0:
        warnings.warn('Maximum number of iterations %d reached.' %
                      max_iterations,
                      warnings.UserWarning)

    return dictionary, weights, n_iter


class L2SPAGPNH(object):
    r"""l2-SPA factorization with GPNH regularization of dictionary.

    The objective function is::

        ||X - S Gamma||_Fro^2 / (n_samples * n_features)
        + epsilon_states * \Phi_{GPNH}(S)

    where the data matrix X has shape (n_features, n_samples) and::

        \Phi_{GPNH}(S) = \sum_{i, j = 1}^{n_components} ||s_i - s_j||_2^2
                         / (n_features * n_components * (n_components - 1))

    The objective function is minimized with an alternating minimization
    with respect to the dictionary S and weights Gamma.

    Parameters
    ----------
    n_components : integer or None
        If an integer, the number of components. If None, then all
        features are kept.

    epsilon_states : float, default: 0
        Regularization parameter for the dictionary.

    init : None | 'random' | 'custom'
        Method used to initialize the algorithm.
        Default: None
        Valid options:

        - None: falls back to 'random'.

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

    backend : 'eigen'
        The backend used by the factorization solver.

    Attributes
    ----------
    dictionary_ : array-like, shape (n_features, n_components)
        The dictionary of states.

    cost_ : number
        Value of the l2-SPA cost function for the obtained factorization.

    n_iter_ : integer
        Actual number of iterations.

    Examples
    --------
    import numpy as np
    X = np.random.rand(4, 10)
    from pyreor import L2SPAGPNH
    model = L2SPAGPNH(n_components=2, init='random', random_state=0)
    weights = model.fit_transform(X)
    dictionary = model.dictionary_

    References
    ----------
    S. Gerber, L. Pospisil, M. Navandard, and I. Horenko,
    "Low-cost scalable discretization, prediction and feature selection
    for complex systems" (2018)
    """
    def __init__(self, n_components, epsilon_states=0, init=None,
                 tolerance=1e-6, max_iterations=1000,
                 verbose=0, random_state=None,
                 backend='builtin'):
        self.n_components = n_components
        self.epsilon_states = epsilon_states
        self.init = init
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.random_state = random_state
        self.backend = backend

    def fit_transform(self, data, dictionary=None, weights=None):
        """Calculate l2-SPA factorization and return transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        data : array-like, shape (n_features, n_samples)
            Data matrix to be factorized.

        dictionary : array-like, shape (n_features, n_components)
            If init='custom', used as initial guess for solution.

        weights : array-like, shape (n_components, n_samples)
            If init='custom', used as initial guess for solution.

        Returns
        -------
        weights : array-like, shape (n_components, n_samples)
            Representation of the data.
        """
        dictionary_, weights_, n_iter_ =  l2_spa_gpnh(
            data,
            dictionary=dictionary,
            weights=weights,
            n_components=self.n_components,
            epsilon_states=self.epsilon_states,
            init=self.init,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            random_state=self.random_state,
            backend=self.backend)

        self.n_components_ = dictionary_.shape[1]
        self.dictionary_ = dictionary_
        self.n_iter_ = n_iter_

        return weights_

    def fit(self, data, **kwargs):
        """Calculate l2-SPA factorization for the data.

        Parameters
        ----------
        data : array-like, shape (n_features, n_samples)
            Data matrix to compute factorization for.

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
            Data matrix to compute factorization for.

        Returns
        -------
        weights : array-like, shape (n_components, n_samples)
            Representation of the data.
        """
        check_is_fitted(self, 'n_components_')

        _, weights, _ = l2_spa_gpnh(
            data=data,
            dictionary=self.dictionary_,
            n_components=self.n_components_,
            epsilon_states=self.epsilon_states,
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
            Representation of data matrix.

        Returns
        -------
        data : array-like, shape (n_features, n_samples)
            Data matrix with original shape.
        """
        check_is_fitted(self, 'n_components_')
        return np.dot(self.dictionary_, weights)

