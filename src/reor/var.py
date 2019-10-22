"""
Provides routines for fitting VAR models.
"""

from __future__ import division

import numbers
import numpy as np


INTEGER_TYPES = (numbers.Integral, np.integer)


def _vec(A):
    """Return vector obtained from matrix by stacking columns."""
    return np.reshape(A, (A.size,), order='F')


def _linear_varx_leastsq_impl(Y, Z, R=None, r=None, W=None):
    """Perform linear VARX model fit using least-squares.

    Parameters
    ----------
    Y : array-like, shape (n_features, n_samples)
        Array containing the outcome values.

    Z : array-like, shape (n_features, n_samples)
        Array containing the stacked lagged predictors.

    R : array-like, shape (n_parameters, n_parameters)
        Array containing restrictions on parameters.

    r : array-like, shape (n_parameters,)
        Array containing the values of the restrictions.

    W : array-like, shape (n_samples, n_samples)
        Array containing weights to use in fitting.

    Returns
    -------
    gamma_LS : array-like, shape (n_parameters,)
        Least-squares estimate for the parameters.

    Sigma_LS : array-like, shape (n_features, n_features)
        Least-squares estimate for the noise covariances.
    """

    if Y.ndim != 2:
        raise ValueError('Outcomes matrix must be two-dimensional')

    if Z.ndim != 2:
        raise ValueError('Predictors matrix must be two-dimensional')

    n_features, n_samples = Y.shape

    if Z.shape[1] != n_samples:
        raise ValueError('Number of outcome samples does not match'
                         ' number of predictor samples')

    n_parameters = n_features * Z.shape[0]

    if R is None and r is None:
        R = np.eye(n_parameters, dtype=Y.dtype)
        r = np.zeros(n_parameters, dtype=Y.dtype)
    else:
        raise ValueError(
            'Either both or neither restriction parameters R and r'
            ' must be present')

    if W is None:
        ZkI = np.kron(Z, np.eye(n_features))
        ZZt = np.dot(Z, Z.T)
    else:
        WtW = np.dot(W.T, W)
        ZkI = np.kron(np.dot(Z, WtW), np.eye(n_features))
        ZZt = np.dot(Z, np.dot(WtW, Z.T))

    y = _vec(Y)
    if r is not None:
        y = y - np.dot(np.kron(Z.T, np.eye(n_features)), r)

    lhs = np.dot(R.T, np.dot(np.kron(ZZt, np.eye(n_features)), R))
    rhs = np.dot(np.dot(R.T, ZkI), y)

    gamma_LS = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    beta_LS = np.dot(R, gamma_LS) + r
    B_LS = np.reshape(beta_LS, (n_features, Z.shape[0]), order='F')
    residuals = Y - np.dot(B_LS, Z)

    Sigma_LS = np.dot(residuals, residuals.T) / n_samples

    return gamma_LS, Sigma_LS


def _linear_varx_EGLS_impl(Y, Z, R=None, r=None, W=None, ddof=None):
    """Perform linear VARX model fit using estimated generalized least-squares.

    Parameters
    ----------
    Y : array-like, shape (n_features, n_samples)
        Array containing the outcome values.

    Z : array-like, shape (n_features, n_samples)
        Array containing the stacked lagged predictors.

    R : array-like, shape (n_parameters, n_parameters)
        Array containing restrictions on parameters.

    Returns
    -------
    gamma_EGLS : array-like, shape (n_parameters,)
        Least-squares estimate for the parameters.

    cov_gamma_EGLS : array-like, shape (n_parameters, n_parameters)
        Estimated covariance matrix for model parameters.

    Sigma_LS : array-like, shape (n_features, n_features)
        Least-squares estimate for the noise covariances.
    """

    if Y.ndim != 2:
        raise ValueError('Outcomes matrix must be two-dimensional')

    if Z.ndim != 2:
        raise ValueError('Predictors matrix must be two-dimensional')

    n_features, n_samples = Y.shape

    if Z.shape[1] != n_samples:
        raise ValueError('Number of outcome samples does not match'
                         ' number of predictor samples')

    n_parameters = n_features * Z.shape[0]

    if R is None:
        R = np.eye(n_parameters, dtype=Y.dtype)
        r = np.zeros(n_parameters, dtype=Y.dtype)
    else:
        raise ValueError(
            'Either both or neither restriction parameters R and r'
            ' must be present')

    # Obtain covariance estimate by initial least-squares fit
    gamma_LS, Sigma_LS = _linear_varx_leastsq_impl(Y, Z=Z, R=R, r=r, W=W)

    if ddof is not None:
        Sigma_LS *= n_samples / ddof

    # Re-estimate with updated covariance estimate
    Sigma_LS_inv = np.linalg.pinv(Sigma_LS)

    if W is None:
        ZkSigma = np.kron(Z, Sigma_LS_inv)
        ZZt = np.dot(Z, Z.T)
    else:
        WtW = np.dot(W.T, W)
        ZkSigma = np.kron(np.dot(Z, WtW), Sigma_LS_inv)
        ZZt = np.dot(Z, np.dot(WtW, Z.T))

    y = _vec(Y)
    if r is not None:
        y = y - np.dot(np.kron(Z.T, np.eye(n_features)), r)

    lhs = np.dot(R.T, np.dot(np.kron(ZZt, Sigma_LS_inv), R))
    rhs = np.dot(np.dot(R.T, ZkSigma), y)

    gamma_EGLS = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    beta_EGLS = np.dot(R, gamma_EGLS) + r

    EZZt = np.dot(Z, Z.T) / n_samples
    cov_gamma_EGLS_inv = np.dot(
        R.T, np.dot(np.kron(EZZt, Sigma_LS_inv), R)) / n_samples
    cov_gamma_EGLS = np.linalg.pinv(cov_gamma_EGLS_inv)

    return gamma_EGLS, cov_gamma_EGLS, Sigma_LS


def _assemble_lagged_data_matrices(y, x=None, p=1, s=0, include_intercept=True):
    """Assemble given data into stacked matrices used for fitting."""

    y = np.atleast_2d(y)
    n_features, n_samples = y.shape

    if x is not None:
        x = np.atleast_2d(x)
        n_external = x.shape[0]
    else:
        n_external = 0

    n_lagged_features = p * n_features
    if x is not None:
        n_lagged_features += n_external
        if s > 0:
            n_lagged_features += s * n_external

    if include_intercept:
        n_lagged_features += 1

    presample_length = max(p, s)
    Y = y[:, presample_length:]
    Z = np.empty((n_lagged_features, n_samples - presample_length),
                 dtype=y.dtype)

    if include_intercept:
        Z[0, :] = 1.0
        row_index = 1
    else:
        row_index = 0

    for i in range(1, p + 1):
        Z[row_index:row_index + n_features, :] = y[:, presample_length - i:-i]
        row_index += n_features

    if x is not None:
        if s > 0:
            for i in range(1, s + 1):
                Z[row_index:row_index + n_external, :] = x[:, presample_length - i:-i]
                row_index += n_external
        Z[row_index:row_index + n_external, :] = x[:, presample_length:]

    return Y, Z


def linear_varx_leastsq(y, x=None, p=1, s=0, weights=None,
                        include_intercept=True, bias=True):
    """Fit VARX model using least-squares.

    Parameters
    ----------
    y : array-like, shape (n_samples,) or (n_features, n_samples)
        Array containing the values of the modelled (endogeneous) variables.

    x : array-like, shape (n_samples,) or (n_external, n_samples)
        If given, array containing the values of the unmodelled
        (exogeneous) variables.

    Returns
    -------
    mu : scalar or array-like, shape (n_features,)
        Array containing the fitted intercepts.

    A : array-like, shape (p,) or (p, n_features, n_features)
        Array containing the values of the autoregressive coefficients.

    B : array-like, shape (s,) or (s, n_features, n_external)
        Array containing the values of the lagged external factor coefficients.

    B0 : array-like, shape (n_features, n_external)
        Array containing the values of the instantaneous external factor
        coefficients.

    Sigma_LS : array-like, shape (n_features, n_features)
        Array containing the estimated residual covariances.
    """

    if not isinstance(p, INTEGER_TYPES) or p < 0:
        raise ValueError('Lag order must be a non-negative integer;'
                         ' got (p=%r)' % p)
    if not isinstance(s, INTEGER_TYPES) or s < 0:
        raise ValueError('External lag order must be a non-negative integer;'
                         ' got (s=%r)' % s)

    y = np.atleast_2d(y)
    n_features, n_samples = y.shape

    if n_samples <= p:
        raise ValueError('Too few samples to fit at requested order;'
                         ' got (p=%r) but only have n_samples=%d samples' %
                         (p, n_samples))

    if x is not None:
        x = np.atleast_2d(x)
        n_external, n_external_samples = x.shape

        if n_external_samples != n_samples:
            raise ValueError('Number of samples does not match number of'
                             ' external variable values; got n_samples=%d but'
                             ' n_external_samples=%d' %
                             (n_samples, n_external_samples))

        if n_external_samples <= s:
            raise ValueError('Too few samples to fit at requested order;'
                             ' got (s=%r) but only have n_samples=%d samples' %
                             (s, n_samples))
    else:
        n_external = 0

    presample_length = max(p, s)
    Y, Z = _assemble_lagged_data_matrices(y, x=x, p=p, s=s,
                                          include_intercept=include_intercept)

    if weights is None:
        W = np.eye(n_samples - presample_length, dtype=y.dtype)
    else:
        W = np.diag(weights[presample_length:])

    # Perform simple least-squares estimation
    gamma_LS, Sigma_LS = _linear_varx_leastsq_impl(Y, Z, R=None, r=None, W=W)

    if not bias:
        Sigma_LS *= (Y.shape[1] / (Y.shape[1] - n_features * p - 1.0))

    # Reshape vector of fitted parameters into separate intercept
    # and coefficient matrices
    parameter_index = 0
    if include_intercept:
        mu = gamma_LS[:n_features]
        parameter_index += n_features
    else:
        mu = np.zeros((n_features,), dtype=y.dtype)

    A = np.empty((p, n_features, n_features), dtype=y.dtype)
    stride = n_features ** 2
    for i in range(p):
        A[i] = np.reshape(
            gamma_LS[parameter_index:parameter_index + stride],
            (n_features, n_features), order='F')
        parameter_index += stride

    if x is not None:
        stride = n_features * n_external
        if s > 0:
            B = np.empty((s, n_features, n_external), dtype=y.dtype)
            for i in range(s):
                B[i] = np.reshape(
                    gamma_LS[parameter_index:parameter_index + stride],
                    (n_features, n_external), order='F')
                parameter_index += stride
        else:
            B = None

        B0 = np.reshape(
            gamma_LS[parameter_index:parameter_index + stride],
            (n_features, n_external), order='F')
        parameter_index += stride
    else:
        B = None
        B0 = None

    # For scalar time-series, return fitted parameters as scalar
    # values
    if n_features == 1:
        mu = np.asscalar(mu)
        A = np.ravel(A)
        Sigma_LS = np.asscalar(Sigma_LS)

        if x is not None:
            if n_external == 1:
                B0 = np.asscalar(B0)
                if s > 1:
                    B = np.ravel(B)
            else:
                B0 = np.ravel(B0)
                if s > 1:
                    B = np.squeeze(B)

    return mu, A, B, B0, Sigma_LS


def linear_varx_EGLS(y, x=None, p=1, s=0, weights=None,
                     include_intercept=True, bias=True):
    """Fit VARX model using single iteration of EGLS.

    Parameters
    ----------
    y : array-like, shape (n_samples,) or (n_features, n_samples)
        Array containing the values of the modelled (endogeneous) variables.

    x : array-like, shape (n_samples,) or (n_external, n_samples)
        If given, array containing the values of the unmodelled
        (exogeneous) variables.

    Returns
    -------
    A dictionary containing the following fields:

    'mu' : scalar or array-like, shape (n_features,)
        Array containing the fitted intercepts.

    'A' : array-like, shape (p,) or (p, n_features, n_features)
        Array containing the values of the autoregressive coefficients.

    'B' : array-like, shape (s,) or (s, n_features, n_external)
        Array containing the values of the lagged external factor coefficients.

    'B0' : array-like, shape (n_features, n_external)
        Array containing the values of the instantaneous external factor
        coefficients.

    'Sigma_LS' : array-like, shape (n_features, n_features)
        Array containing the estimated residual covariances.

    'stderr_mu' : scalar or array-like, shape (n_features,)
        Array containing the estimated standard error of the intercept.

    'stderr_A' : array-like, shape (p,) or (p, n_features, n_features)
        Array containing the estimated standard error of the autoregressive
        coefficients.

    'stderr_B' : array-like, shape (s,) or (s, n_features, n_external)
        Array containing the estimated standard error of the lagged external
        factor coefficients.

    'stderr_B0' : array-like, shape (n_features, n_external)
        Array containing the estimated standard error of the lagged
        instantaneous external factor coefficients.

    'cov_params' : array-like, shape (n_parameters, n_parameters)
        Array containing estimated covariances of model parameters,
        where the parameters are assumed to be stacked column-wise
        into a single vector.
    """

    if not isinstance(p, INTEGER_TYPES) or p < 0:
        raise ValueError('Lag order must be a non-negative integer;'
                         ' got (p=%r)' % p)
    if not isinstance(s, INTEGER_TYPES) or s < 0:
        raise ValueError('External lag order must be a non-negative integer;'
                         ' got (s=%r)' % s)

    y = np.atleast_2d(y)
    n_features, n_samples = y.shape

    if n_samples <= p:
        raise ValueError('Too few samples to fit at requested order;'
                         ' got (p=%r) but only have n_samples=%d samples' %
                         (p, n_samples))

    if x is not None:
        x = np.atleast_2d(x)
        n_external, n_external_samples = x.shape

        if n_external_samples != n_samples:
            raise ValueError('Number of samples does not match number of'
                             ' external variable values; got n_samples=%d but'
                             ' n_external_samples=%d' %
                             (n_samples, n_external_samples))

        if n_external_samples <= s:
            raise ValueError('Too few samples to fit at requested order;'
                             ' got (s=%r) but only have n_samples=%d samples' %
                             (s, n_samples))
    else:
        n_external = 0

    presample_length = max(p, s)
    Y, Z = _assemble_lagged_data_matrices(y, x=x, p=p, s=s,
                                          include_intercept=include_intercept)

    if weights is None:
        W = np.eye(n_samples - presample_length, dtype=y.dtype)
    else:
        W = np.diag(weights[presample_length:])

    # Perform simple least-squares estimation
    if not bias:
        residuals_ddof = Y.shape[1] - n_features * p - 1.0
    else:
        residuals_ddof = Y.shape[1]

    gamma_EGLS, cov_gamma_EGLS, Sigma_LS = _linear_varx_EGLS_impl(
        Y, Z, R=None, r=None, W=W, ddof=residuals_ddof)

    stderr_gamma_EGLS = np.sqrt(np.diag(cov_gamma_EGLS))

    # Reshape vector of fitted parameters into separate intercept
    # and coefficient matrices
    parameter_index = 0
    if include_intercept:
        mu = gamma_EGLS[:n_features]
        stderr_mu = stderr_gamma_EGLS[:n_features]
        parameter_index += n_features
    else:
        mu = np.zeros((n_features,), dtype=y.dtype)
        stderr_mu = 0.0

    A = np.empty((p, n_features, n_features), dtype=y.dtype)
    stderr_A = np.empty_like(A)
    stride = n_features ** 2
    for i in range(p):
        A[i] = np.reshape(
            gamma_EGLS[parameter_index:parameter_index + stride],
            (n_features, n_features), order='F')
        stderr_A[i] = np.reshape(
            stderr_gamma_EGLS[parameter_index:parameter_index + stride],
            (n_features, n_features), order='F')
        parameter_index += stride

    if x is not None:
        stride = n_features * n_external
        if s > 0:
            B = np.empty((s, n_features, n_external), dtype=y.dtype)
            stderr_B = np.empty_like(B)
            for i in range(s):
                B[i] = np.reshape(
                    gamma_EGLS[parameter_index:parameter_index + stride],
                    (n_features, n_external), order='F')
                stderr_B[i] = np.reshape(
                    stderr_gamma_EGLS[parameter_index:parameter_index + stride],
                    (n_features, n_external), order='F')
                parameter_index += stride
        else:
            B = None
            stderr_B = None

        B0 = np.reshape(
            gamma_EGLS[parameter_index:parameter_index + stride],
            (n_features, n_external), order='F')
        stderr_B0 = np.reshape(
            stderr_gamma_EGLS[parameter_index:parameter_index + stride],
            (n_features, n_external), order='F')
        parameter_index += stride
    else:
        B = None
        stderr_B = None
        B0 = None
        stderr_B0 = None

    # For scalar time-series, return fitted parameters as scalar
    # values
    # For scalar time-series, return fitted parameters as scalar
    # values
    if n_features == 1:
        mu = np.asscalar(mu)
        stderr_mu = np.asscalar(stderr_mu)
        A = np.ravel(A)
        stderr_A = np.ravel(stderr_A)

        Sigma_LS = np.asscalar(Sigma_LS)

        if x is not None:
            if n_external == 1:
                B0 = np.asscalar(B0)
                stderr_B0 = np.asscalar(stderr_B0)
                if s > 1:
                    B = np.ravel(B)
                    stderr_B = np.ravel(stderr_B)
            else:
                B0 = np.ravel(B0)
                stderr_B0 = np.ravel(stderr_B0)
                if s > 1:
                    B = np.squeeze(B)
                    stderr_B = np.squeeze(stderr_B)

    return dict(
        mu=mu, A=A, B=B, B0=B0,
        stderr_mu=stderr_mu, stderr_A=stderr_A,
        stderr_B=stderr_B, stderr_B0=stderr_B0,
        Sigma_LS=Sigma_LS,
        cov_gamma_EGLS=cov_gamma_EGLS)
