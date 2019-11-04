"""
Provides unit tests for PPCA routines.
"""


import unittest
import numpy as np
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.utils import check_random_state

from reor.ppca import ppca_max_likelihood_estimate


class TestPPCAMaxLikelihoodEstimate(unittest.TestCase):
    """Provides unit tests for PPCA maximum likelihood estimates."""

    def test_correct_estimate_for_mean(self):
        """Test returns correct ML estimate for feature means."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 9
        n_components = 4
        n_samples = 500

        mu = random_state.normal(size=(n_features,))
        w = random_state.normal(size=(n_features, n_components))
        z = random_state.normal(size=(n_components, n_samples))
        # Ensure latent variables have zero mean
        z -= z.mean(axis=1, keepdims=True)

        sigma_sq = 1e-3
        noise = random_state.normal(size=(n_features, n_samples),
                                    scale=np.sqrt(sigma_sq))
        noise -= noise.mean(axis=1, keepdims=True)

        x = mu[:, np.newaxis] + np.dot(w, z) + noise

        mu_ml, w_ml, z_ml, sigma_sq_ml, _ = ppca_max_likelihood_estimate(
            x, n_components=n_components)

        self.assertTrue(np.allclose(mu, mu_ml))

    def test_correct_reconstruction_when_no_features_dropped(self):
        """Test returns correct reconstruction."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 10
        n_components = 10
        n_samples = 1000

        mu = random_state.normal(size=(n_features,))
        w = random_state.normal(size=(n_features, n_components))
        z = random_state.normal(size=(n_components, n_samples))

        sigma_sq = 1e-2
        noise = random_state.normal(size=(n_features, n_samples),
                                    scale=np.sqrt(sigma_sq))

        x = mu[:, np.newaxis] + np.dot(w, z) + noise

        mu_ml, w_ml, z_ml, sigma_sq_ml, _ = ppca_max_likelihood_estimate(
            x, n_components=n_components)

        x_rec = tf.matmul(w_ml, z_ml) + mu_ml[:, tf.newaxis]

        self.assertTrue(np.allclose(x, x_rec))

    def test_approximate_reconstruction_when_features_dropped(self):
        """Test reconstruction is approximately correct."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 20
        n_components = 15
        n_samples = 1000

        mu = random_state.normal(size=(n_features,))
        w = random_state.normal(size=(n_features, n_components))
        z = random_state.normal(size=(n_components, n_samples))
        z -= z.mean(axis=1, keepdims=True)

        sigma_sq = 1e-3
        noise = random_state.normal(size=(n_features, n_samples),
                                    scale=np.sqrt(sigma_sq))
        noise -= noise.mean(axis=1, keepdims=True)

        x = mu[:, np.newaxis] + np.dot(w, z) + noise

        mu_ml, w_ml, z_ml, sigma_sq_ml, _ = ppca_max_likelihood_estimate(
            x, n_components=n_components)

        x_rec = tf.matmul(w_ml, z_ml) + mu_ml[:, tf.newaxis]

        self.assertTrue(np.allclose(x, x_rec, atol=1e-1))

    def test_ppca_matches_pca(self):
        """Test PPCA result reduces to standard PCA."""

        random_seed = 0
        random_state = check_random_state(random_seed)

        n_features = 15
        n_components = 10
        n_samples = 1000

        mu = np.zeros((n_features,))
        w = random_state.normal(size=(n_features, n_components))
        z = random_state.normal(size=(n_components, n_samples))
        z -= z.mean(axis=1, keepdims=True)

        sigma_sq = 1e-2
        noise = random_state.normal(size=(n_features, n_samples),
                                    scale=np.sqrt(sigma_sq))
        noise -= noise.mean(axis=1, keepdims=True)

        x = mu[:, np.newaxis] + np.dot(w, z) + noise

        self.assertTrue(np.allclose(x.mean(axis=1), 0))

        mu_ml, w_ml, z_ml, sigma_sq_ml, _ = ppca_max_likelihood_estimate(
            x, n_components=n_components)

        model = PCA(n_components=n_components)
        z_pca = model.fit_transform(x.T)
        # Note that scikit-learn adopts a different normalization
        # for the latent features
        w_pca = np.dot(model.components_.T,
                       np.diag(model.singular_values_) / np.sqrt(n_samples))

        for i in range(n_components):
            self.assertTrue(
                np.abs(np.dot(w_ml[:, i], w_pca[:, i]) - 1), 1e-3)
