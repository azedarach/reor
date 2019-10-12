"""
Provides unit tests for MTD regularized convex coding.
"""


import unittest
import numpy as np

from sklearn.utils import check_random_state

from reor.mtd_convex_coding import MTDConvexCoding
from reor._random_matrix import left_stochastic_matrix, right_stochastic_matrix


class TestMTDConvexCodingCostFunction(unittest.TestCase):
    """Provides unit tests for MTD convex coding cost function."""

    def test_returns_zero_for_perfect_reconstruction(self):
        """Test cost is zero for perfect factorization."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 19
        n_components = 5
        n_samples = 30
        tolerance = 1e-14

        S = random_state.uniform(size=(n_components, n_features))
        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, tolerance))

        X = Gamma.dot(S)

        lags = np.arange(1, 3)
        n_lags = lags.size
        model = MTDConvexCoding(n_components=n_components, epsilon_states=0,
                                epsilon_weights=0, lags=lags)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = random_state.uniform(size=(n_components,))
        model.order_weights /= model.order_weights.sum()

        model.transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            model.transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        model._initialize_workspace()

        cost = model._evaluate_cost()
        expected_cost = 0

        self.assertEqual(cost, expected_cost)

    def test_returns_zero_for_perfect_reconstruction_and_exact_weights(self):
        """Test cost is zero for perfect factorization and Markov weights."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 19
        n_components = 5
        n_samples = 30
        tolerance = 1e-14

        lags = [1, 3]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_lags,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        S = random_state.uniform(size=(n_components, n_features))

        Gamma = np.zeros((n_samples, n_components), dtype='f8')
        Gamma[0] = random_state.uniform(size=(n_components,))
        Gamma[0] /= Gamma[0].sum()
        Gamma[1] = random_state.uniform(size=(n_components,))
        Gamma[1] /= Gamma[1].sum()
        Gamma[2] = random_state.uniform(size=(n_components,))
        Gamma[2] /= Gamma[2].sum()
        for t in range(3, n_samples):
            Gamma[t] = (
                order_weights[0] * transition_matrices[0].dot(
                    Gamma[t - 1]) +
                order_weights[1] * transition_matrices[1].dot(
                    Gamma[t - 3]))

        self.assertTrue(np.allclose(order_weights.sum(), 1, tolerance))
        for i in range(n_lags):
            self.assertTrue(
                np.allclose(transition_matrices[i].sum(axis=0), 1, tolerance))
        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, tolerance))

        X = Gamma.dot(S)

        model = MTDConvexCoding(n_components=n_components, epsilon_states=0,
                                epsilon_weights=2.3, lags=lags)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

        model._initialize_workspace()
        cost = model._evaluate_cost()
        expected_cost = 0

        self.assertTrue(abs(cost - expected_cost) < tolerance)


class TestMTDConvexCodingDictionaryUpdate(unittest.TestCase):
    """Provides unit tests for MTD convex coding dictionary update."""

    def test_analytic_gradient_matches_numerical_gradient_with_zero_epsilon(self):
        """Test analytical gradient matches finite difference approximation."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 3
        n_components = 5
        n_samples = 10

        X = random_state.uniform(size=(n_samples, n_features))
        S = random_state.uniform(size=(n_components, n_features))
        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, 1e-12))

        lags = [1, 3, 5, 8]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_components,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        model = MTDConvexCoding(n_components=n_components, epsilon_states=0,
                                epsilon_weights=1.7, lags=lags)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

        model._initialize_workspace()

        model._update_dictionary_gradient()

        analytic_grad_S = model.grad_S

        def central_difference_deriv(i, j, h=1e-4):
            old_x = S[i, j]

            xmh = old_x - h
            model.S[i, j] = xmh
            model._initialize_workspace()
            fmh = model._evaluate_cost()

            xph = old_x + h
            model.S[i, j] = xph
            model._initialize_workspace()
            fph = model._evaluate_cost()

            model.S[i, j] = old_x

            return (fph - fmh) / (2 * h)

        numeric_grad_S = np.zeros((n_components, n_features))
        for i in range(n_components):
            for j in range(n_features):
                numeric_grad_S[i, j] = central_difference_deriv(i, j)

        self.assertTrue(np.allclose(analytic_grad_S, numeric_grad_S, 1e-4))

    def test_single_update_reduces_cost_function_with_zero_epsilon(self):
        """Test single dictionary update reduces cost function."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 10
        n_components = 8
        n_samples = 500

        X = random_state.uniform(size=(n_samples, n_features))
        S = random_state.uniform(size=(n_components, n_features))
        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        lags = [1, 3, 5, 8]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_components,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, 1e-14))

        model = MTDConvexCoding(n_components=n_components, epsilon_states=0,
                                epsilon_weights=1.7, lags=lags)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

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
        n_components = 3
        n_samples = 300

        X = random_state.uniform(size=(n_samples, n_features))
        S = random_state.uniform(size=(n_components, n_features))
        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, 1e-14))

        lags = [2, 3, 5]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_components,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        model = MTDConvexCoding(n_components=n_components, epsilon_states=3.2,
                                epsilon_weights=4.3, lags=lags)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

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

        lags = [1]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_components,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        model = MTDConvexCoding(n_components=n_components, epsilon_states=0,
                                epsilon_weights=0.5, lags=lags)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

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

        lags = [6, 7]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_components,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        model = MTDConvexCoding(n_components=n_components, epsilon_states=0,
                                epsilon_weights=3.4, lags=lags,
                                tolerance=tolerance, max_iterations=max_iter)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

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

        lags = [1, 12]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_components,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        model = MTDConvexCoding(n_components=n_components, epsilon_states=4.3,
                                epsilon_weights=2.0, lags=lags,
                                tolerance=tolerance, max_iterations=max_iter)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

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


class TestMTDConvexCodingWeightsUpdate(unittest.TestCase):
    """Provides unit tests for MTD convex coding weights update."""

    def test_analytic_gradient_matches_numerical_gradient_with_zero_epsilon(self):
        """Test analytical gradient matches finite difference approximation."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 15
        n_components = 5
        n_samples = 10

        X = random_state.uniform(size=(n_samples, n_features))
        S = random_state.uniform(size=(n_components, n_features))
        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, 1e-12))

        lags = [1, 5]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_lags,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        model = MTDConvexCoding(n_components=n_components, epsilon_states=1.0,
                                epsilon_weights=0, lags=lags)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

        model._initialize_workspace()

        model._update_weights_gradient()

        analytic_grad_Gamma = model.grad_Gamma

        def central_difference_deriv(i, j, h=1e-4):
            old_x = model.Gamma[i, j]

            xmh = old_x - h
            model.Gamma[i, j] = xmh
            model._initialize_workspace()
            fmh = model._evaluate_cost()

            xph = old_x + h
            model.Gamma[i, j] = xph
            model._initialize_workspace()
            fph = model._evaluate_cost()

            model.Gamma[i, j] = old_x

            return (fph - fmh) / (2 * h)

        numeric_grad_Gamma = np.zeros((n_samples, n_components))
        for i in range(n_samples):
            for j in range(n_components):
                numeric_grad_Gamma[i, j] = central_difference_deriv(i, j)

        self.assertTrue(
            np.allclose(analytic_grad_Gamma, numeric_grad_Gamma, 1e-4))

    def test_analytic_gradient_matches_numerical_gradient_with_nonzero_epsilon(self):
        """Test analytical gradient matches finite difference approximation."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 10
        n_components = 3
        n_samples = 10

        X = random_state.uniform(size=(n_samples, n_features))
        S = random_state.uniform(size=(n_components, n_features))
        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, 1e-12))

        lags = [1, 2, 4]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_lags,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        model = MTDConvexCoding(n_components=n_components, epsilon_states=1.0,
                                epsilon_weights=3.2, lags=lags)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

        model._initialize_workspace()

        model._update_weights_gradient()

        analytic_grad_Gamma = model.grad_Gamma

        def central_difference_deriv(i, j, h=1e-4):
            old_x = model.Gamma[i, j]

            xmh = old_x - h
            model.Gamma[i, j] = xmh
            model._initialize_workspace()
            fmh = model._evaluate_cost()

            xph = old_x + h
            model.Gamma[i, j] = xph
            model._initialize_workspace()
            fph = model._evaluate_cost()

            model.Gamma[i, j] = old_x

            return (fph - fmh) / (2 * h)

        numeric_grad_Gamma = np.zeros((n_samples, n_components))
        for i in range(n_samples):
            for j in range(n_components):
                numeric_grad_Gamma[i, j] = central_difference_deriv(i, j)

        self.assertTrue(
            np.allclose(analytic_grad_Gamma, numeric_grad_Gamma, 1e-4))

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

        lags = [1, 12]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_components,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        model = MTDConvexCoding(n_components=n_components, epsilon_states=2.3,
                                epsilon_weights=0, lags=lags)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

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

        lags = [1, 2, 3]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_components,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        model = MTDConvexCoding(n_components=n_components, epsilon_states=7,
                                epsilon_weights=2.3)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

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

        lags = [1, 12]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_components,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        model = MTDConvexCoding(n_components=n_components, epsilon_states=0,
                                epsilon_weights=0, lags=lags)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

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

        lags = [1,]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_lags,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        S = random_state.uniform(size=(n_components, n_features))

        Gamma = np.zeros((n_samples, n_components), dtype='f8')
        Gamma[0] = random_state.uniform(size=(n_components,))
        Gamma[0] /= Gamma[0].sum()
        for t in range(1, n_samples):
            Gamma[t] = order_weights[0] * transition_matrices[0].dot(
                Gamma[t - 1])

        self.assertTrue(np.allclose(order_weights.sum(), 1, tolerance))
        for i in range(n_lags):
            self.assertTrue(
                np.allclose(transition_matrices[i].sum(axis=0), 1, tolerance))
        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, tolerance))

        X = Gamma.dot(S)

        model = MTDConvexCoding(n_components=n_components, epsilon_states=4.0,
                                epsilon_weights=1.2, lags=lags)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

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

        lags = [1, 12]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_components,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        model = MTDConvexCoding(n_components=n_components, epsilon_states=5.3,
                                epsilon_weights=0, lags=lags,
                                tolerance=tolerance, max_iterations=max_iter)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

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

        lags = [1, 12]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_components,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        model = MTDConvexCoding(n_components=n_components, epsilon_states=1.3,
                                epsilon_weights=3.4, lags=lags,
                                tolerance=tolerance, max_iterations=max_iter)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

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


class TestMTDConvexCodingParametersUpdate(unittest.TestCase):
    """Provides unit tests for MTD convex coding parameters update."""

    def test_analytic_gradient_matches_numerical_gradient(self):
        """Test analytical gradient matches finite difference approximation."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 3
        n_components = 2
        n_samples = 10

        X = random_state.uniform(size=(n_samples, n_features))
        S = random_state.uniform(size=(n_components, n_features))
        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, 1e-12))

        lags = [1, 3]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_lags,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        model = MTDConvexCoding(n_components=n_components, epsilon_states=1.0,
                                epsilon_weights=1.2, lags=lags)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

        model._initialize_workspace()

        model._update_parameters_gradient()

        analytic_grad_order_weights = model.grad_order_weights
        analytic_grad_transition_matrices = model.grad_transition_matrices

        def central_difference_order_weights_deriv(i, h=1e-4):
            old_x = model.order_weights[i]

            xmh = old_x - h
            model.order_weights[i] = xmh
            model._initialize_workspace()
            fmh = model._evaluate_cost()

            xph = old_x + h
            model.order_weights[i] = xph
            model._initialize_workspace()
            fph = model._evaluate_cost()

            model.order_weights[i] = old_x

            return (fph - fmh) / (2 * h)

        def central_difference_transition_matrices_deriv(i, j, k, h=1e-4):
            old_x = model.transition_matrices[i, j, k]

            xmh = old_x - h
            model.transition_matrices[i, j, k] = xmh
            model._initialize_workspace()
            fmh = model._evaluate_cost()

            xph = old_x + h
            model.transition_matrices[i, j, k] = xph
            model._initialize_workspace()
            fph = model._evaluate_cost()

            model.transition_matrices[i, j, k] = old_x

            return (fph - fmh) / (2 * h)

        numeric_grad_order_weights = np.zeros((n_lags,))
        for i in range(n_lags):
            numeric_grad_order_weights[i] = central_difference_order_weights_deriv(i)

        self.assertTrue(
            np.allclose(analytic_grad_order_weights,
                        numeric_grad_order_weights, 1e-4))

        numeric_grad_transition_matrices = np.zeros(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            for j in range(n_components):
                for k in range(n_components):
                    numeric_grad_transition_matrices[i, j, k] = \
                        central_difference_transition_matrices_deriv(i, j, k)

        self.assertTrue(
            np.allclose(analytic_grad_transition_matrices,
                        numeric_grad_transition_matrices, 1e-4))

    def test_single_update_reduces_cost_function(self):
        """Test single parameters update reduces cost function."""

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

        lags = [2, 7]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_lags,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        model = MTDConvexCoding(n_components=n_components, epsilon_states=2.3,
                                epsilon_weights=1.3, lags=lags)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

        model._initialize_workspace()

        initial_cost = model._evaluate_cost()

        error = model._update_parameters()

        self.assertEqual(error, 0)

        final_cost = model._evaluate_cost()

        self.assertTrue(final_cost <= initial_cost)

    def test_exact_solution_is_fixed_point_with_nonzero_epsilon(self):
        """Test exact solution is a fixed point of update step."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 10
        n_components = 7
        n_samples = 40
        tolerance = 1e-12

        lags = [1, 2,]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_lags,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        S = random_state.uniform(size=(n_components, n_features))

        Gamma = np.zeros((n_samples, n_components), dtype='f8')
        Gamma[0] = random_state.uniform(size=(n_components,))
        Gamma[0] /= Gamma[0].sum()
        Gamma[1] = random_state.uniform(size=(n_components,))
        Gamma[1] /= Gamma[1].sum()
        for t in range(2, n_samples):
            Gamma[t] = (
                order_weights[0] * transition_matrices[0].dot(
                    Gamma[t - 1]) +
                order_weights[1] * transition_matrices[1].dot(
                    Gamma[t - 2]))

        self.assertTrue(np.allclose(order_weights.sum(), 1, tolerance))
        for i in range(n_lags):
            self.assertTrue(
                np.allclose(transition_matrices[i].sum(axis=0), 1, tolerance))
        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, tolerance))

        X = Gamma.dot(S)

        model = MTDConvexCoding(n_components=n_components, epsilon_states=4.0,
                                epsilon_weights=2.2, lags=lags)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

        model._initialize_workspace()

        initial_cost = model._evaluate_cost()

        error = model._update_parameters()

        self.assertEqual(error, 0)

        final_cost = model._evaluate_cost()

        updated_order_weights = model.order_weights
        updated_transition_matrices = model.transition_matrices

        self.assertTrue(final_cost <= initial_cost)

        self.assertTrue(np.allclose(updated_order_weights.sum(), 1, tolerance))
        for i in range(n_lags):
            self.assertTrue(
                np.allclose(updated_transition_matrices[i].sum(axis=0), 1, tolerance))

        self.assertTrue(
            np.allclose(order_weights, updated_order_weights, tolerance))
        self.assertTrue(
            np.allclose(transition_matrices, updated_transition_matrices,
                        tolerance))

    def test_repeated_updates_converge_with_nonzero_epsilon(self):
        """Test repeated updates converge to fixed point."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 35
        n_components = 4
        n_samples = 500
        max_iter = 200
        tolerance = 1e-6

        X = random_state.uniform(size=(n_samples, n_features))
        S = random_state.uniform(size=(n_components, n_features))
        Gamma = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(Gamma.sum(axis=1), 1, 1e-12))

        lags = [1, 3, 4]
        n_lags = len(lags)

        order_weights = random_state.uniform(size=(n_components,))
        order_weights /= order_weights.sum()

        transition_matrices = np.empty(
            (n_lags, n_components, n_components))
        for i in range(n_lags):
            transition_matrices[i] = left_stochastic_matrix(
                (n_components, n_components), random_state=random_state)

        model = MTDConvexCoding(n_components=n_components, epsilon_states=1.3,
                                epsilon_weights=6.4, lags=lags,
                                tolerance=tolerance, max_iterations=max_iter)

        model.X = X
        model.Gamma = Gamma
        model.S = S
        model.order_weights = order_weights
        model.transition_matrices = transition_matrices

        model._initialize_workspace()

        cost_delta = 1 + tolerance
        old_cost = model._evaluate_cost()
        new_cost = old_cost
        n_iter = 0

        while abs(cost_delta) > tolerance and n_iter < max_iter:
            old_cost = new_cost
            error = model._update_parameters()
            self.assertEqual(error, 0)
            new_cost = model._evaluate_cost()

            cost_delta = new_cost - old_cost
            self.assertTrue(cost_delta <= 0)

            self.assertTrue(np.allclose(model.order_weights.sum(), 1, tolerance))
            for i in range(n_lags):
                self.assertTrue(
                    np.allclose(model.transition_matrices[i].sum(axis=0), 1, tolerance))

            n_iter += 1

        self.assertTrue(n_iter < max_iter)
