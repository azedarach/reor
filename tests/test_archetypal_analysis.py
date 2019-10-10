"""
Provides unit tests for archetypal analysis routines.
"""


import unittest
import numpy as np

from sklearn.utils import check_random_state

from reor.archetypal_analysis import KernelAA
from reor._random_matrix import right_stochastic_matrix


class TestKernelAADictionaryUpdate(unittest.TestCase):
    """Provides unit tests for kernel AA dictionary update."""

    def test_single_dictionary_update_reduces_cost_with_zero_delta(self):
        """Test single update step reduces cost function."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 10
        n_components = 5
        n_samples = 400

        X = random_state.uniform(size=(n_samples, n_features))
        K = X.dot(X.T)

        C = right_stochastic_matrix(
            (n_components, n_samples), random_state=random_state)
        S = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(C.sum(axis=1), 1, 1e-12))
        self.assertTrue(np.allclose(S.sum(axis=1), 1, 1e-12))

        delta = 0
        aa = KernelAA(n_components=n_components, delta=delta)

        aa.K = K
        aa.C = C
        aa.S = S

        aa._initialize_workspace()

        initial_cost = aa._evaluate_cost()

        error = aa._update_dictionary()

        self.assertEqual(error, 0)

        final_cost = aa._evaluate_cost()

        self.assertTrue(final_cost <= initial_cost)

        updated_C = aa.C

        self.assertTrue(np.allclose(updated_C.sum(axis=1), 1, 1e-12))

    def test_single_dictionary_update_reduces_cost_with_nonzero_delta(self):
        """Test single update step reduces cost function."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 10
        n_components = 5
        n_samples = 400

        X = random_state.uniform(size=(n_samples, n_features))
        K = X.dot(X.T)

        C = right_stochastic_matrix(
            (n_components, n_samples), random_state=random_state)
        S = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(C.sum(axis=1), 1, 1e-12))
        self.assertTrue(np.allclose(S.sum(axis=1), 1, 1e-12))

        delta = 1.2
        aa = KernelAA(n_components=n_components, delta=delta)

        aa.K = K
        aa.C = C
        aa.S = S

        aa._initialize_workspace()

        initial_cost = aa._evaluate_cost()

        error = aa._update_dictionary()

        self.assertEqual(error, 0)

        final_cost = aa._evaluate_cost()

        self.assertTrue(final_cost <= initial_cost)

        updated_C = aa.C

        self.assertTrue(np.allclose(updated_C.sum(axis=1), 1, 1e-12))

    def test_exact_solution_with_zero_delta_is_fixed_point(self):
        """Test exact solution is a fixed point of the dictionary update."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 10
        n_components = 6
        n_samples = 100
        tolerance = 1e-12

        basis = random_state.uniform(size=(n_components, n_features))

        S = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        archetype_indices = np.zeros(n_components, dtype='i8')
        for i in range(n_components):
            new_index = False
            current_index = 0

            while not new_index:
                new_index = True

                current_index = random_state.randint(
                    low=0, high=n_samples)

                for index in archetype_indices:
                    if current_index == index:
                        new_index = False

            archetype_indices[i] = current_index

        C = np.zeros((n_components, n_samples))
        component = 0
        for index in archetype_indices:
            C[component, index] = 1.0
            for i in range(n_components):
                if i == component:
                    S[index, i] = 1.0
                else:
                    S[index, i] = 0.0
            component += 1

        X = S.dot(basis)
        basis_projection = C.dot(X)

        self.assertTrue(np.allclose(basis_projection, basis, tolerance))
        self.assertTrue(np.linalg.norm(X - S.dot(C.dot(X))) < tolerance)

        K = X.dot(X.T)

        delta = 0
        aa = KernelAA(n_components=n_components, delta=delta)

        aa.K = K
        aa.C = C
        aa.S = S

        aa._initialize_workspace()

        initial_cost = aa._evaluate_cost()

        error = aa._update_dictionary()

        self.assertEqual(error, 0)

        final_cost = aa._evaluate_cost()

        self.assertTrue(abs(final_cost - initial_cost) < tolerance)

        updated_C = aa.C

        self.assertTrue(np.allclose(updated_C.sum(axis=1), 1, 1e-12))
        self.assertTrue(np.allclose(updated_C, C, tolerance))

    def test_repeated_updates_converge_with_zero_delta(self):
        """Test repeated updates converge to a fixed point with delta = 0."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 20
        n_components = 15
        n_samples = 600
        max_iter = 500
        tolerance = 1e-6

        X = random_state.uniform(size=(n_samples, n_features))
        K = X.dot(X.T)

        C = right_stochastic_matrix(
            (n_components, n_samples), random_state=random_state)
        S = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(C.sum(axis=1), 1, tolerance))
        self.assertTrue(np.allclose(S.sum(axis=1), 1, tolerance))

        delta = 0
        aa = KernelAA(n_components=n_components, delta=delta)

        aa.K = K
        aa.C = C
        aa.S = S

        aa._initialize_workspace()

        cost_delta = 1 + tolerance
        old_cost = aa._evaluate_cost()
        new_cost = old_cost
        n_iter = 0
        while abs(cost_delta) > tolerance and n_iter < max_iter:
            old_cost = new_cost
            error = aa._update_dictionary()
            self.assertEqual(error, 0)
            new_cost = aa._evaluate_cost()

            cost_delta = new_cost - old_cost

            self.assertTrue(cost_delta <= 0)

            n_iter += 1

        self.assertTrue(n_iter < max_iter)

        updated_C = aa.C
        self.assertTrue(np.allclose(updated_C.sum(axis=1), 1, 1e-12))

    def test_repeated_updates_converge_with_nonzero_delta(self):
        """Test repeated updates converge to a fixed point with non-zero delta."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 30
        n_components = 11
        n_samples = 320
        max_iter = 1000
        tolerance = 1e-4

        X = random_state.uniform(size=(n_samples, n_features))
        K = X.dot(X.T)

        C = right_stochastic_matrix(
            (n_components, n_samples), random_state=random_state)
        S = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(C.sum(axis=1), 1, tolerance))
        self.assertTrue(np.allclose(S.sum(axis=1), 1, tolerance))

        delta = 3.2
        aa = KernelAA(n_components=n_components, delta=delta)

        aa.K = K
        aa.C = C
        aa.S = S

        aa._initialize_workspace()

        cost_delta = 1 + tolerance
        old_cost = aa._evaluate_cost()
        new_cost = old_cost
        n_iter = 0
        while abs(cost_delta) > tolerance and n_iter < max_iter:
            old_cost = new_cost
            error = aa._update_dictionary()
            self.assertEqual(error, 0)
            new_cost = aa._evaluate_cost()

            cost_delta = new_cost - old_cost

            self.assertTrue(cost_delta <= 0)

            n_iter += 1

        self.assertTrue(n_iter < max_iter)

        updated_C = aa.C
        self.assertTrue(np.allclose(updated_C.sum(axis=1), 1, 1e-12))

        updated_alpha = aa.alpha
        for i in range(n_components):
            self.assertTrue(1 - delta <= updated_alpha[i] <= 1 + delta)

class TestKernelAAWeightsUpdate(unittest.TestCase):
    """Provides unit tests for kernel AA weights update."""

    def test_analytic_gradient_matches_numerical_gradient_with_zero_delta(self):
        """Test analytical gradient matches finite difference approximation."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 3
        n_components = 5
        n_samples = 10

        X = random_state.uniform(size=(n_samples, n_features))
        K = X.dot(X.T)

        C = right_stochastic_matrix(
            (n_components, n_samples), random_state=random_state)
        S = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(C.sum(axis=1), 1, 1e-12))
        self.assertTrue(np.allclose(S.sum(axis=1), 1, 1e-12))

        delta = 0
        aa = KernelAA(n_components=n_components, delta=delta)

        aa.K = K
        aa.C = C
        aa.S = S

        aa._initialize_workspace()
        aa._update_weights_gradient()

        analytic_grad_S = aa.grad_S

        def central_difference_deriv(i, j, h=1e-4):
            old_x = S[i, j]

            xmh = old_x - h
            aa.S[i, j] = xmh
            aa._initialize_workspace()
            fmh = aa._evaluate_cost()

            xph = old_x + h
            aa.S[i, j] = xph
            aa._initialize_workspace()
            fph = aa._evaluate_cost()

            aa.S[i, j] = old_x

            return (fph - fmh) / (2 * h)

        numeric_grad_S = np.zeros((n_samples, n_components))
        for i in range(n_samples):
            for j in range(n_components):
                numeric_grad_S[i, j] = central_difference_deriv(i, j)

        self.assertTrue(np.allclose(analytic_grad_S, numeric_grad_S, 1e-4))

    def test_analytic_gradient_matches_numerical_gradient_with_nonzero_delta(self):
        """Test analytical gradient matches finite difference approximation."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 3
        n_components = 5
        n_samples = 10

        X = random_state.uniform(size=(n_samples, n_features))
        K = X.dot(X.T)

        C = right_stochastic_matrix(
            (n_components, n_samples), random_state=random_state)
        S = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(C.sum(axis=1), 1, 1e-12))
        self.assertTrue(np.allclose(S.sum(axis=1), 1, 1e-12))

        delta = 1.2
        aa = KernelAA(n_components=n_components, delta=delta)

        aa.K = K
        aa.C = C
        aa.S = S

        aa._initialize_workspace()

        aa.alpha[2] = 0.95
        aa.C.dot(aa.K, out=aa.CK)

        diag_alpha = np.diag(aa.alpha)
        diag_alpha.dot(aa.CK, out=aa.CK)

        aa.CK.dot(aa.C.T, out=aa.CKCt)
        aa.CKCt.dot(diag_alpha, out=aa.CKCt)

        aa._update_weights_gradient()

        analytic_grad_S = aa.grad_S

        def central_difference_deriv(i, j, h=1e-4):
            old_x = S[i, j]

            xmh = old_x - h
            aa.S[i, j] = xmh
            fmh = aa._evaluate_cost()

            xph = old_x + h
            aa.S[i, j] = xph
            fph = aa._evaluate_cost()

            aa.S[i, j] = old_x

            return (fph - fmh) / (2 * h)

        numeric_grad_S = np.zeros((n_samples, n_components))
        for i in range(n_samples):
            for j in range(n_components):
                numeric_grad_S[i, j] = central_difference_deriv(i, j)

        self.assertTrue(np.allclose(analytic_grad_S, numeric_grad_S, 1e-4))

    def test_single_weights_update_reduces_cost_with_zero_delta(self):
        """Test single weights update reduces cost function with delta = 0."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 13
        n_components = 7
        n_samples = 100

        X = random_state.uniform(size=(n_samples, n_features))
        K = X.dot(X.T)

        C = right_stochastic_matrix(
            (n_components, n_samples), random_state=random_state)
        S = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(C.sum(axis=1), 1, 1e-12))
        self.assertTrue(np.allclose(S.sum(axis=1), 1, 1e-12))

        delta = 0
        aa = KernelAA(n_components=n_components, delta=delta)

        aa.K = K
        aa.C = C
        aa.S = S

        aa._initialize_workspace()

        initial_cost = aa._evaluate_cost()

        error = aa._update_weights()

        self.assertEqual(error, 0)

        final_cost = aa._evaluate_cost()

        self.assertTrue(final_cost <= initial_cost)

    def test_single_weights_update_reduces_cost_with_nonzero_delta(self):
        """Test single weights update reduces cost function with non-zero delta."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 50
        n_components = 5
        n_samples = 400

        X = random_state.uniform(size=(n_samples, n_features))
        K = X.dot(X.T)

        C = right_stochastic_matrix(
            (n_components, n_samples), random_state=random_state)
        S = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(C.sum(axis=1), 1, 1e-12))
        self.assertTrue(np.allclose(S.sum(axis=1), 1, 1e-12))

        delta = 2.3
        aa = KernelAA(n_components=n_components, delta=delta)

        aa.K = K
        aa.C = C
        aa.S = S

        aa._initialize_workspace()

        initial_cost = aa._evaluate_cost()

        error = aa._update_weights()

        self.assertEqual(error, 0)

        final_cost = aa._evaluate_cost()

        self.assertTrue(final_cost <= initial_cost)

    def test_exact_solution_with_zero_delta_is_fixed_point(self):
        """Test exact solution for weights is fixed point of update step."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 30
        n_components = 10
        n_samples = 130
        tolerance = 1e-12

        basis = random_state.uniform(size=(n_components, n_features))

        S = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        archetype_indices = np.zeros(n_components, dtype='i8')
        for i in range(n_components):
            new_index = False
            current_index = 0

            while not new_index:
                new_index = True

                current_index = random_state.randint(
                    low=0, high=n_samples)

                for index in archetype_indices:
                    if current_index == index:
                        new_index = False

            archetype_indices[i] = current_index

        C = np.zeros((n_components, n_samples))
        component = 0
        for index in archetype_indices:
            C[component, index] = 1.0
            for i in range(n_components):
                if i == component:
                    S[index, i] = 1.0
                else:
                    S[index, i] = 0.0
            component += 1

        X = S.dot(basis)
        basis_projection = C.dot(X)

        self.assertTrue(np.allclose(basis_projection, basis, tolerance))
        self.assertTrue(np.linalg.norm(X - S.dot(C.dot(X))) < tolerance)

        K = X.dot(X.T)

        delta = 0
        aa = KernelAA(n_components=n_components, delta=delta)

        aa.K = K
        aa.C = C
        aa.S = S

        aa._initialize_workspace()

        initial_cost = aa._evaluate_cost()

        error = aa._update_weights()

        self.assertEqual(error, 0)

        final_cost = aa._evaluate_cost()

        self.assertTrue(abs(final_cost - initial_cost) < tolerance)

        updated_S = aa.S

        self.assertTrue(np.allclose(updated_S.sum(axis=1), 1, 1e-12))
        self.assertTrue(np.allclose(updated_S, S, tolerance))

    def test_repeated_updates_converge_with_zero_delta(self):
        """Test repeated updates converge to a fixed point with delta = 0."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 10
        n_components = 3
        n_samples = 600
        max_iter = 100
        tolerance = 1e-6

        X = random_state.uniform(size=(n_samples, n_features))
        K = X.dot(X.T)

        C = right_stochastic_matrix(
            (n_components, n_samples), random_state=random_state)
        S = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(C.sum(axis=1), 1, tolerance))
        self.assertTrue(np.allclose(S.sum(axis=1), 1, tolerance))

        delta = 0
        aa = KernelAA(n_components=n_components, delta=delta)

        aa.K = K
        aa.C = C
        aa.S = S

        aa._initialize_workspace()

        cost_delta = 1 + tolerance
        old_cost = aa._evaluate_cost()
        new_cost = old_cost
        n_iter = 0
        while abs(cost_delta) > tolerance and n_iter < max_iter:
            old_cost = new_cost
            error = aa._update_weights()
            self.assertEqual(error, 0)
            new_cost = aa._evaluate_cost()

            cost_delta = new_cost - old_cost

            self.assertTrue(cost_delta <= 0)

            n_iter += 1

        self.assertTrue(n_iter < max_iter)

        updated_S = aa.S
        self.assertTrue(np.allclose(updated_S.sum(axis=1), 1, 1e-12))

    def test_repeated_updates_converge_with_nonzero_delta(self):
        """Test repeated updates converge to a fixed point with non-zero delta."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 30
        n_components = 11
        n_samples = 320
        max_iter = 1000
        tolerance = 1e-4

        X = random_state.uniform(size=(n_samples, n_features))
        K = X.dot(X.T)

        C = right_stochastic_matrix(
            (n_components, n_samples), random_state=random_state)
        S = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(C.sum(axis=1), 1, tolerance))
        self.assertTrue(np.allclose(S.sum(axis=1), 1, tolerance))

        delta = 0.3
        aa = KernelAA(n_components=n_components, delta=delta)

        aa.K = K
        aa.C = C
        aa.S = S

        aa._initialize_workspace()

        cost_delta = 1 + tolerance
        old_cost = aa._evaluate_cost()
        new_cost = old_cost
        n_iter = 0
        while abs(cost_delta) > tolerance and n_iter < max_iter:
            old_cost = new_cost
            error = aa._update_weights()
            self.assertEqual(error, 0)
            new_cost = aa._evaluate_cost()

            cost_delta = new_cost - old_cost

            self.assertTrue(cost_delta <= 0)

            n_iter += 1

        self.assertTrue(n_iter < max_iter)

        updated_S = aa.S
        self.assertTrue(np.allclose(updated_S.sum(axis=1), 1, 1e-12))


class TestKernelAASolution(unittest.TestCase):
    """Provides unit tests for kernel AA solution."""

    def test_finds_elements_of_3_point_convex_hull(self):
        """Test finds archetypes in convex hull for 2D example."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 2
        n_samples = 50
        n_components = 3
        max_iter = 500
        tolerance = 1e-6

        basis = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        expected_S = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        assignments = np.array([5, 27, 32])
        for i in range(n_components):
            expected_S[assignments[i]] = np.zeros(n_components)
            expected_S[assignments[i], i] = 1

        X = expected_S.dot(basis)
        K = X.dot(X.T)

        C = right_stochastic_matrix(
            (n_components, n_samples), random_state=random_state)
        S = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(C.sum(axis=1), 1, 1e-12))
        self.assertTrue(np.allclose(S.sum(axis=1), 1, 1e-12))

        delta = 0
        aa = KernelAA(n_components=n_components, delta=delta, init='custom',
                      max_iterations=max_iter, tolerance=tolerance)

        solution_S = aa.fit_transform(K, dictionary=C, weights=S)
        solution_C = aa.dictionary_

        self.assertTrue(aa.n_iter_ < max_iter)

        self.assertTrue(np.allclose(solution_C.sum(axis=1), 1, 1e-12))
        self.assertTrue(np.allclose(solution_S.sum(axis=1), 1, 1e-12))

        main_components = solution_C.argmax(axis=1)
        main_components = sorted(main_components)
        for i in range(n_components):
            self.assertEqual(main_components[i], assignments[i])

    def test_finds_elements_of_4_point_convex_hull(self):
        """Test finds archetypes in convex hull for 3D example."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 3
        n_samples = 123
        n_components = 4
        max_iter = 500
        tolerance = 1e-12

        basis = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

        expected_S = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        assignments = np.array([8, 9, 56, 90])
        for i in range(n_components):
            expected_S[assignments[i]] = np.zeros(n_components)
            expected_S[assignments[i], i] = 1

        expected_C = np.zeros((n_components, n_samples), dtype='f8')
        for i in range(n_components):
            expected_C[i, assignments[i]] = 1

        X = expected_S.dot(basis)

        self.assertTrue(
            np.linalg.norm(X - expected_S.dot(expected_C.dot(X))) < tolerance)

        K = X.dot(X.T)

        C = right_stochastic_matrix(
            (n_components, n_samples), random_state=random_state)
        S = right_stochastic_matrix(
            (n_samples, n_components), random_state=random_state)

        self.assertTrue(np.allclose(C.sum(axis=1), 1, 1e-12))
        self.assertTrue(np.allclose(S.sum(axis=1), 1, 1e-12))

        delta = 0
        aa = KernelAA(n_components=n_components, delta=delta, init='custom',
                      max_iterations=max_iter, tolerance=tolerance)

        solution_S = aa.fit_transform(K, dictionary=C, weights=S)
        solution_C = aa.dictionary_

        self.assertTrue(aa.n_iter_ < max_iter)

        self.assertTrue(np.allclose(solution_C.sum(axis=1), 1, 1e-12))
        self.assertTrue(np.allclose(solution_S.sum(axis=1), 1, 1e-12))

        main_components = solution_C.argmax(axis=1)
        main_components = sorted(main_components)
        for i in range(n_components):
            self.assertEqual(main_components[i], assignments[i])
