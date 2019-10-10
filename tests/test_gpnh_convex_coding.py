"""
Provides unit tests for GPNH regularized convex coding.
"""


import unittest
import numpy as np

from sklearn.utils import check_random_state

from reor.gpnh_convex_coding import GPNHConvexCoding
from reor._random_matrix import right_stochastic_matrix


class TestGPNHConvexCodingCostFunction(unittest.TestCase):
    """Provides unit tests for GPNH convex coding cost function."""

    def test_returns_zero_for_perfect_reconstruction(self):
        """Test cost is zero for perfect factorization."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 5
        n_components = 3
        n_samples = 30
        tolerance = 1e-14

        S = random_state.uniform(size=(n_components, n_features))
        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, tolerance))

        X = Gamma.dot(S)

        model = GPNHConvexCoding(n_components=n_components, epsilon_states=0)

        model.X = X
        model.Gamma = Gamma
        model.S = S

        model._initialize_workspace()

        cost = model._evaluate_cost()
        expected_cost = 0

        self.assertEqual(cost, expected_cost)


class TestGPNHConvexCodingDictionaryUpdate(unittest.TestCase):
    """Provides unit tests for GPNH convex coding dictionary update."""

    def test_single_update_reduces_cost_function_with_zero_epsilon(self):
        """Test single dictionary update reduces cost function."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 7
        n_components = 5
        n_samples = 450

        X = random_state.uniform(size=(n_samples, n_features))
        S = random_state.uniform(size=(n_components, n_features))
        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, 1e-14))

        model = GPNHConvexCoding(n_components=n_components, epsilon_states=0)

        model.X = X
        model.Gamma = Gamma
        model.S = S

        model._initialize_workspace()

        initial_cost = model._evaluate_cost()

        error = model._update_dictionary()

        self.assertEqual(error, 0)

        final_cost = model._evaluate_cost()

        self.assertTrue(final_cost <= initial_cost)

    def test_single_update_reduces_cost_function_with_nonzero_epsilon(self):
        """Test single dictionary update reduces cost function."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 11
        n_components = 6
        n_samples = 230

        X = random_state.uniform(size=(n_samples, n_features))
        S = random_state.uniform(size=(n_components, n_features))
        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, 1e-14))

        model = GPNHConvexCoding(n_components=n_components, epsilon_states=3.2)

        model.X = X
        model.Gamma = Gamma
        model.S = S

        model._initialize_workspace()

        initial_cost = model._evaluate_cost()

        error = model._update_dictionary()

        self.assertEqual(error, 0)

        final_cost = model._evaluate_cost()

        self.assertTrue(final_cost <= initial_cost)

    def test_exact_solution_is_fixed_point(self):
        """Test exact solution is a fixed point of update step."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 10
        n_components = 6
        n_samples = 40
        tolerance = 1e-12

        S = random_state.uniform(size=(n_components, n_features))

        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, tolerance))

        X = Gamma.dot(S)

        model = GPNHConvexCoding(n_components=n_components, epsilon_states=0)

        model.X = X
        model.Gamma = Gamma
        model.S = S

        model._initialize_workspace()

        initial_cost = model._evaluate_cost()

        error = model._update_dictionary()

        self.assertEqual(error, 0)

        final_cost = model._evaluate_cost()

        updated_S = model.S

        self.assertTrue(final_cost <= initial_cost)
        self.assertTrue(np.allclose(S, updated_S, tolerance))

    def test_repeated_updates_converge_with_zero_epsilon(self):
        """Test repeated updates converge to fixed point."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 13
        n_components = 3
        n_samples = 50
        max_iter = 100
        tolerance = 1e-6

        X = random_state.uniform(size=(n_samples, n_features))
        S = random_state.uniform(size=(n_components, n_features))
        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, 1e-12))

        model = GPNHConvexCoding(n_components=n_components, epsilon_states=0,
                                 tolerance=tolerance, max_iterations=max_iter)

        model.X = X
        model.Gamma = Gamma
        model.S = S

        model._initialize_workspace()

        cost_delta = 1 + tolerance
        old_cost = model._evaluate_cost()
        new_cost = old_cost
        n_iter = 0

        while abs(cost_delta) > tolerance and n_iter < max_iter:
            old_cost = new_cost
            error = model._update_dictionary()
            self.assertEqual(error, 0)
            new_cost = model._evaluate_cost()

            cost_delta = new_cost - old_cost

            self.assertTrue(cost_delta <= 0)

            n_iter += 1

        self.assertTrue(n_iter < max_iter)

    def test_repeated_updates_converge_with_nonzero_epsilon(self):
        """Test repeated updates converge to fixed point."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 27
        n_components = 13
        n_samples = 500
        max_iter = 100
        tolerance = 1e-6

        X = random_state.uniform(size=(n_samples, n_features))
        S = random_state.uniform(size=(n_components, n_features))
        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, 1e-12))

        model = GPNHConvexCoding(n_components=n_components, epsilon_states=4.3,
                                 tolerance=tolerance, max_iterations=max_iter)

        model.X = X
        model.Gamma = Gamma
        model.S = S

        model._initialize_workspace()

        cost_delta = 1 + tolerance
        old_cost = model._evaluate_cost()
        new_cost = old_cost
        n_iter = 0

        while abs(cost_delta) > tolerance and n_iter < max_iter:
            old_cost = new_cost
            error = model._update_dictionary()
            self.assertEqual(error, 0)
            new_cost = model._evaluate_cost()

            cost_delta = new_cost - old_cost

            self.assertTrue(cost_delta <= 0)

            n_iter += 1

        self.assertTrue(n_iter < max_iter)


class TestGPNHConvexCodingWeightsUpdate(unittest.TestCase):
    """Provides unit tests for GPNH convex coding weights update."""

    def test_single_update_reduces_cost_function_with_zero_epsilon(self):
        """Test single weights update reduces cost function."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 25
        n_components = 6
        n_samples = 300

        X = random_state.uniform(size=(n_samples, n_features))
        S = random_state.uniform(size=(n_components, n_features))
        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, 1e-14))

        model = GPNHConvexCoding(n_components=n_components, epsilon_states=0)

        model.X = X
        model.Gamma = Gamma
        model.S = S

        model._initialize_workspace()

        initial_cost = model._evaluate_cost()

        error = model._update_weights()

        self.assertEqual(error, 0)

        final_cost = model._evaluate_cost()

        self.assertTrue(final_cost <= initial_cost)

    def test_single_update_reduces_cost_function_with_nonzero_epsilon(self):
        """Test single dictionary update reduces cost function."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 19
        n_components = 6
        n_samples = 230

        X = random_state.uniform(size=(n_samples, n_features))
        S = random_state.uniform(size=(n_components, n_features))
        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, 1e-14))

        model = GPNHConvexCoding(n_components=n_components, epsilon_states=7)

        model.X = X
        model.Gamma = Gamma
        model.S = S

        model._initialize_workspace()

        initial_cost = model._evaluate_cost()

        error = model._update_weights()

        self.assertEqual(error, 0)

        final_cost = model._evaluate_cost()

        self.assertTrue(final_cost <= initial_cost)

    def test_exact_solution_is_fixed_point_with_zero_epsilon(self):
        """Test exact solution is a fixed point of update step."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 5
        n_components = 6
        n_samples = 40
        tolerance = 1e-12

        S = random_state.uniform(size=(n_components, n_features))

        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, tolerance))

        X = Gamma.dot(S)

        model = GPNHConvexCoding(n_components=n_components, epsilon_states=0)

        model.X = X
        model.Gamma = Gamma
        model.S = S

        model._initialize_workspace()

        initial_cost = model._evaluate_cost()

        error = model._update_weights()

        self.assertEqual(error, 0)

        final_cost = model._evaluate_cost()

        updated_Gamma = model.Gamma

        self.assertTrue(final_cost <= initial_cost)
        self.assertTrue(np.allclose(Gamma, updated_Gamma, tolerance))

    def test_exact_solution_is_fixed_point_with_nonzero_epsilon(self):
        """Test exact solution is a fixed point of update step."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 10
        n_components = 7
        n_samples = 40
        tolerance = 1e-12

        S = random_state.uniform(size=(n_components, n_features))

        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, tolerance))

        X = Gamma.dot(S)

        model = GPNHConvexCoding(n_components=n_components, epsilon_states=4.0)

        model.X = X
        model.Gamma = Gamma
        model.S = S

        model._initialize_workspace()

        initial_cost = model._evaluate_cost()

        error = model._update_weights()

        self.assertEqual(error, 0)

        final_cost = model._evaluate_cost()

        updated_Gamma = model.Gamma

        self.assertTrue(final_cost <= initial_cost)
        self.assertTrue(np.allclose(Gamma, updated_Gamma, tolerance))

    def test_repeated_updates_converge_with_zero_epsilon(self):
        """Test repeated updates converge to fixed point."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 10
        n_components = 9
        n_samples = 500
        max_iter = 100
        tolerance = 1e-6

        X = random_state.uniform(size=(n_samples, n_features))
        S = random_state.uniform(size=(n_components, n_features))
        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, 1e-12))

        model = GPNHConvexCoding(n_components=n_components, epsilon_states=0,
                                 tolerance=tolerance, max_iterations=max_iter)

        model.X = X
        model.Gamma = Gamma
        model.S = S

        model._initialize_workspace()

        cost_delta = 1 + tolerance
        old_cost = model._evaluate_cost()
        new_cost = old_cost
        n_iter = 0

        while abs(cost_delta) > tolerance and n_iter < max_iter:
            old_cost = new_cost
            error = model._update_weights()
            self.assertEqual(error, 0)
            new_cost = model._evaluate_cost()

            cost_delta = new_cost - old_cost

            self.assertTrue(cost_delta <= 0)

            n_iter += 1

        self.assertTrue(n_iter < max_iter)

    def test_repeated_updates_converge_with_nonzero_epsilon(self):
        """Test repeated updates converge to fixed point."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 35
        n_components = 4
        n_samples = 500
        max_iter = 100
        tolerance = 1e-6

        X = random_state.uniform(size=(n_samples, n_features))
        S = random_state.uniform(size=(n_components, n_features))
        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, 1e-12))

        model = GPNHConvexCoding(n_components=n_components, epsilon_states=1.3,
                                 tolerance=tolerance, max_iterations=max_iter)

        model.X = X
        model.Gamma = Gamma
        model.S = S

        model._initialize_workspace()

        cost_delta = 1 + tolerance
        old_cost = model._evaluate_cost()
        new_cost = old_cost
        n_iter = 0

        while abs(cost_delta) > tolerance and n_iter < max_iter:
            old_cost = new_cost
            error = model._update_weights()
            self.assertEqual(error, 0)
            new_cost = model._evaluate_cost()

            cost_delta = new_cost - old_cost

            self.assertTrue(cost_delta <= 0)

            n_iter += 1

        self.assertTrue(n_iter < max_iter)
