"""
Provides helper routines for validation and error checking.
"""


import numpy as np


def _check_unit_axis_sums(A, whom, axis=0):
    axis_sums = A.sum(axis=axis)
    if not np.all(np.isclose(axis_sums, 1)):
        raise ValueError(
            'Array with incorrect axis sums passed to %s. '
            'Expected sums along axis %d to be 1.'
            % (whom, axis))


def _check_array_shape(A, shape, whom):
    if A.shape != shape:
        raise ValueError(
            'Array with wrong shape passed to %s. '
            'Expected %s, but got %s' % (whom, shape, A.shape))
