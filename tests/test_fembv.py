"""
Provides test routines for generic FEM-BV routines.
"""

import unittest
import cvxopt
import numpy as np

from sklearn.utils import check_random_state
from reor.fembv import FEMBV


class FEMBVDummy(FEMBV):
    """Dummy FEM-BV class with trivial parameters updates."""

    def __init__(self, n_components, max_tv_norm, init='random',
                 max_iterations=500, tolerance=1e-4, verbose=0,
                 random_state=None, **kwargs):
        super().__init__(n_components=n_components, max_tv_norm=max_tv_norm,
                         init=init, max_iterations=max_iterations,
                         tolerance=tolerance, verbose=verbose,
                         random_state=random_state, **kwargs)

        self.name = 'FEM-BV-dummy'
        self.n_samples = 0

    def _evaluate_distance_matrix(self):
        self.distance_matrix = self.random_state.uniform(
            size=(self.n_samples, self.n_components))

    def _initialize_components(self, data, parameters=None, init=None, **kwargs):
        self.n_samples = data.shape[0]
        self.distance_matrix = self.random_state.uniform(
            size=(self.n_samples, self.n_components))

    def _update_parameters(self):
        return


class TestFEMBVWeightsConstraints(unittest.TestCase):
    """Provides unit tests for FEM-BV weight constraints."""

    def test_equality_constraints_with_no_max_tv_norm(self):
        """Test equality constraints match expected form."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_samples = 4
        n_features = 3
        n_components = 3
        max_tv_norm = None

        X = random_state.uniform(size=(n_samples, n_features))

        model = FEMBVDummy(n_components=n_components, max_tv_norm=max_tv_norm,
                           random_state=random_state)

        model._initialize_components(X)
        model._initialize_weights(X, n_samples)

        A_eq = np.array(cvxopt.matrix(model.A_eq.real()))
        b_eq = np.array(model.b_eq)

        expected_A_eq = np.array(
            [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
        expected_b_eq = np.array([[1.0], [1.0], [1.0], [1.0]])

        self.assertTrue(np.allclose(A_eq, expected_A_eq))
        self.assertTrue(np.allclose(b_eq, expected_b_eq))

    def test_upper_bound_constraints_with_no_max_tv_norm(self):
        """Test inequality constraints match expected form."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_samples = 3
        n_features = 2
        n_components = 3
        max_tv_norm = None

        X = random_state.uniform(size=(n_samples, n_features))

        model = FEMBVDummy(n_components=n_components, max_tv_norm=max_tv_norm,
                           random_state=random_state)

        model._initialize_components(X)
        model._initialize_weights(X, n_samples)

        A_ub = np.array(cvxopt.matrix(model.A_ub.real()))
        b_ub = np.array(model.b_ub)

        expected_A_ub = -np.eye(n_components * n_samples)
        expected_b_ub = np.zeros((n_components * n_samples,))

        self.assertTrue(np.allclose(A_ub, expected_A_ub))
        self.assertTrue(np.allclose(b_ub, expected_b_ub))

    def test_equality_constraints_with_max_tv_norm(self):
        """Test equality constraints match expected form."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_samples = 4
        n_features = 3
        n_components = 3
        max_tv_norm = 5

        X = random_state.uniform(size=(n_samples, n_features))

        model = FEMBVDummy(n_components=n_components, max_tv_norm=max_tv_norm,
                           random_state=random_state)

        model._initialize_components(X)
        model._initialize_weights(X, n_samples)

        A_eq = np.array(cvxopt.matrix(model.A_eq.real()))
        b_eq = np.array(model.b_eq)

        n_parameters = n_components * (2 * n_samples - 1)
        expected_A_eq = np.zeros((n_samples, n_parameters), dtype='f8')
        expected_A_eq[0, 0:3] = 1.0
        expected_A_eq[1, 3:6] = 1.0
        expected_A_eq[2, 6:9] = 1.0
        expected_A_eq[3, 9:12] = 1.0

        expected_b_eq = np.array([[1.0], [1.0], [1.0], [1.0]])

        self.assertTrue(np.allclose(A_eq, expected_A_eq))
        self.assertTrue(np.allclose(b_eq, expected_b_eq))

    def test_upper_bound_constraints_with_max_tv_norm(self):
        """Test inequality constraints match expected form."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_samples = 3
        n_features = 2
        n_components = 2
        max_tv_norm = 5

        X = random_state.uniform(size=(n_samples, n_features))

        model = FEMBVDummy(n_components=n_components, max_tv_norm=max_tv_norm,
                           random_state=random_state)

        model._initialize_components(X)
        model._initialize_weights(X, n_samples)

        A_ub = np.array(cvxopt.matrix(model.A_ub.real()))
        b_ub = np.array(model.b_ub)

        n_parameters = n_components * (2 * n_samples - 1)
        n_constraints = (n_parameters + 2 * n_components * (n_samples - 1) +
                         n_components)

        self.assertTrue(A_ub.shape == (n_constraints, n_parameters))
        self.assertTrue(b_ub.shape == (n_constraints, 1))

        expected_A_ub = np.zeros((n_constraints, n_parameters))
        expected_A_ub[:n_parameters, :n_parameters] = -np.eye(n_parameters)

        expected_A_ub = np.array(
            [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
             [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
             [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
             [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
             [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
             [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]])

        expected_b_ub = np.zeros((n_constraints, 1), dtype='f8')
        expected_b_ub[-n_components:] = max_tv_norm

        self.assertTrue(np.allclose(A_ub, expected_A_ub))
        self.assertTrue(np.allclose(b_ub, expected_b_ub))
