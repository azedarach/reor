"""
Provides routines for carrying out convex coding.
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_probability as tfp

@tf.function
def _convex_coding_log_likelihood_fn(data, mu, w, z, sigma_sq, concentration):

    n_features, n_samples = data.shape
    n_components = w.shape[1]

    log_likelihood = 0.0

    z_dist = tfp.distributions.Dirichlet(concentration=concentration)

    log_likelihood += tf.reduce_sum(z_dist.log_prob(z))

    x_dist = tfp.distributions.Normal(
        loc=tf.dtypes.cast(mu[:, tf.newaxis] + tf.matmul(w, z), data.dtype),
        scale=sigma_sq * tf.ones([n_features, n_samples], dtype=data.dtype))

    log_likelihood += tf.reduce_sum(x_dist.log_prob(data))

    return log_likelihood


@tf.function
def _convex_coding_log_joint_fn(data, mu, w, z, sigma_sq, concentration,
                                sigma_mu=-1, sigma_w=1, beta=1):

    n_features, n_samples = data.shape
    n_components = w.shape[1]

    log_joint = 0.0

    if sigma_mu > 0:
        mu_dist = tfp.distributions.Normal(
            loc=tf.zeros([n_features,], dtype=mu.dtype),
            scale=sigma_mu * tf.ones([n_features,], dtype=mu.dtype))

        log_joint += tf.reduce_sum(mu_dist.log_prob(mu))

    if beta > 0:
        sigma_sq_dist = tfp.distributions.Gamma(
            concentration=beta, rate=beta)

    w_dist = tfp.distributions.Normal(
        loc=tf.zeros([n_features, n_components], dtype=w.dtype),
        scale=sigma_w * tf.ones([n_features, n_components], dtype=w.dtype))

    log_joint += tf.reduce_sum(w_dist.log_prob(w))

    z_dist = tfp.distributions.Dirichlet(
        concentration=concentration)

    log_joint += tf.reduce_sum(z_dist.log_prob(tf.transpose(z)))

    x_dist = tfp.distributions.Normal(
        loc=tf.dtypes.cast(mu[:, tf.newaxis] + tf.matmul(w, z), data.dtype),
        scale=sigma_sq * tf.ones([n_features, n_samples], dtype=data.dtype))

    log_joint += tf.reduce_sum(x_dist.log_prob(data))

    return log_joint


def convex_coding_map_estimate(data, n_components, concentration,
                               mu_init=None,
                               w_init=None, z_init=None, sigma_sq_init=None,
                               sigma_mu=-1, sigma_w=1.0, beta=0.1,
                               optimizer=None, max_epochs=200, tolerance=-1,
                               callbacks=None):
    """Evaluate MAP estimate for convex coding on dataset.

    Parameters
    ----------
    data : array-like, shape (n_features, n_samples)
        Data to perform convex coding on.

    n_components : integer or None
        If a positive integer, the number of latent dimensions. If None
        or a non-positive integer, the number of latent dimensions is
        set to the number of features (i.e., no dimensions are dropped).

    concentration : Tensor, shape (n_components,)
        Concentration parameter of the Dirichlet prior for the
        latent variables.

    mu_init : None or Tensor, shape (n_features,)
        If a Tensor, used as initial guess for solution.

    w_init : None or Tensor, shape (n_features, n_components)
        If a Tensor, used as initial guess for solution.

    z_init : None or Tensor, shape (n_components, n_samples)
        If a Tensor, used as initial guess for solution.

    sigma_sq_init : None or Tensor, shape (1,)
        If a Tensor, used as initial guess for solution.

    sigma_mu : scalar, default: -1
        Variance hyperparameter for feature means prior. If less than
        or equal to zero, an improper uniform prior is used. Otherwise,
        a normal prior with variance sigma_mu is used.

    sigma_w : scalar, default: 1
        Variance hyperparameter for latent features.

    beta : scalar, default: -1
        Rate hyperparameter for noise variance prior. If less than or
        equal to zero, an improper uniform prior is used. Otherwise,
        a gamma prior with rate beta and concentration beta is used.

    optimizer : None or instance of tf.optimizers.Optimizer
        Optimizer to be used to calculate MAP estimate.

    max_epochs : integer, default: 200
        Maximum number of training epochs.

    tolerance : None or scalar
        If a positive scalar, the stopping condition tolerance. If None
        or less than or equal to zero, iterates for the maximum number
        of epochs.

    callbacks : None or list of callbacks
        If a list, should contain a list of callbacks to be called.

    Returns
    -------
    mu_map : Tensor, shape (n_features,)
        MAP estimate for the mean of each feature.

    w_map : Tensor, shape (n_features, n_components)
        MAP estimate for the latent features.

    z_map : Tensor, shape (n_components, n_samples)
        MAP estimate for latent variables.

    sigma_sq_map : Tensor, shape (1,)
        MAP estimate of noise variance.

    loss_value : scalar
        Value of the negative log joint density.
    """

    n_features, n_samples = data.shape

    if n_components is None or n_components <= 0:
        n_components = n_features

    if w_init is None:
        w_map = tf.Variable(
            tf.random.normal([n_features, n_components]),
            dtype=data.dtype)
    else:
        w_map = tf.Variable(w_init, dtype=data.dtype)

    if mu_init is None:
        mu_map = tf.Variable(tf.random.normal([n_features,]),
                             dtype=data.dtype)
    else:
        mu_map = tf.Variable(mu_init, dtype=data.dtype)

    if sigma_sq_init is None:
        sigma_sq_map = tf.Variable(
            tf.random.uniform([1,], maxval=tf.math.reduce_variance(data)),
            dtype=data.dtype)
    else:
        sigma_sq_map = tf.Variable(sigma_sq_init, dtype=data.dtype)

    if z_init is None:
        z_vals = tf.random.uniform([n_components, n_samples],
                                   dtype=data.dtype)
        z_vals /= tf.reduce_sum(z_vals, axis=0, keepdims=True)
        z_map = tf.Variable(z_vals)
    else:
        z_map = tf.Variable(z_init, dtype=data.dtype)

    if optimizer is None:
        optimizer = tf.optimizers.Adam(learning_rate=0.05)

    if callbacks is not None:
        for c in callbacks:
            c.on_train_begin(logs=None)

    old_loss = None
    for epoch in range(max_epochs):

        if callbacks is not None:
            for c in callbacks:
                c.on_epoch_begin(epoch, logs=None)

        with tf.GradientTape() as tape:
            loss_value = -_convex_coding_log_joint_fn(
                data, mu_map, w_map, z_map, sigma_sq_map,
                concentration,
                sigma_mu=sigma_mu, sigma_w=sigma_w, beta=beta)

        grads = tape.gradient(loss_value, [mu_map, w_map, z_map, sigma_sq_map])

        optimizer.apply_gradients(
            zip(grads, [mu_map, w_map, z_map, sigma_sq_map]))

        if callbacks is not None:
            logs = {'loss': loss_value}
            for c in callbacks:
                c.on_epoch_end(epoch, logs)

        if tolerance is not None and tolerance > 0 and epoch > 0:
            loss_delta = tf.abs(loss_value - old_loss)
            if loss_delta < tolerance:
                break

        old_loss = loss_value

    if callbacks is not None:
        for c in callbacks:
            c.on_train_end(logs=None)

    return mu_map, w_map, z_map, sigma_sq_map, loss_value


class ConvexCoding():
    """Run convex coding on a dataset."""

    def __init__(self, n_components=None, concentration=None,
                 fit_method='map', verbose=0,
                 sigma_w=1, sigma_mu=-1, beta=-1):

        self.valid_fit_methods = ['map']

        self.n_components = n_components
        self.fit_method = fit_method
        self.verbose = verbose

        if concentration is None:
            self.concentration = tf.ones([n_components,])
        else:
            self.concentration = concentration

        self.sigma_w = sigma_w
        self.sigma_mu = sigma_mu
        self.beta = beta

        self.mu = None
        self.w = None
        self.z = None
        self.sigma_sq = None
        self.loss = None

    def fit_transofmr(self, data, **kwargs):
        """Perform convex coding of data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        data : array-like, shape (n_features, n_samples)
            Data matrix to be fitted.

        Returns
        -------
        z : Tensor, shape (n_components, n_samples)
            Latent representation of the data.
        """

        self.fit(data, **kwargs)
        return self.z

    def fit(self, data, mu_init=None, w_init=None, z_init=None,
            sigma_sq_init=None, optimizer=None, max_epochs=200,
            tolerance=-1, callbacks=None):
        """Fit convex coding to data.

        Parameters
        ----------
        data : array-like, shape (n_features, n_samples)
            Data matrix to be fitted.

        Returns
        -------
        self
        """

        if self.fit_method == 'map':
            mu, w, z, sigma_sq, loss = convex_coding_map_estimate(
                data, n_components=self.n_components,
                concentration=self.concentration,
                mu_init=mu_init, w_init=w_init, z_init=z_init,
                sigma_sq_init=sigma_sq_init, sigma_mu=self.sigma_mu,
                sigma_w=self.sigma_w, beta=self.beta, optimizer=optimizer,
                max_epochs=max_epochs, tolerance=tolerance,
                callbacks=callbacks)

            self.loss = loss

        else:
            raise ValueError("Invalid fit method '%r'" % self.fit_method)

        self.mu = mu
        self.w = w
        self.z = z
        self.sigma_sq = sigma_sq

        return self

    def transform(self, data, optimizer=None, max_epochs=200, tolerance=-1):
        """Transform the data according to the fitted model.

        The returned latent variables are represented by their
        MAP estimate under the fitted parameters.

        Parameters
        ----------
        data : array-like, shape (n_features, n_samples)
            Data matrix for data to be transformed.

        Returns
        -------
        z : array-like, shape (n_components, n_samples)
            Latent representation of the data.
        """

        n_features, n_samples = data.shape

        z_vals = tf.random.uniform([self.n_components, n_samples],
                                   dtype=data.dtype)
        z_vals /= tf.reduce_sum(z_vals, axis=0, keepdims=True)
        z_map = tf.Variable(z_vals)

        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=0.05)

        old_loss = None
        for epoch in range(max_epochs):

            with tf.GradientTape() as tape:
                loss_value = -_convex_coding_log_joint_fn(
                    data, self.mu, self.w, z_map, self.sigma_sq,
                    self.concentration,
                    sigma_mu=self.sigma_mu, sigma_w=self.sigma_w,
                    beta=self.beta)

            grads = tape.gradient(loss_value, [z_map])

            optimizer.apply_gradients(zip(grads, [z_map]))

            if tolerance is not None and tolerance > 0 and epoch > 0:
                loss_delta = tf.abs(loss_value - old_loss)
                if loss_delta < tolerance:
                    break

        return z_map

    def inverse_transform(self, z):
        """Transform data back into its original space.

        Parameters
        ----------
        z : array-like, shape (n_components, n_samples)
            Latent representation of the data.

        Returns
        -------
        x_new : Tensor, shape (n_features, n_samples)
            Latent variables transformed to original space.
        """

        return self.mu[:, tf.newaxis] + tf.matmul(self.w, z)
