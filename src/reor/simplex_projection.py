"""
Provides routines for projections onto probability simplices.
"""


import numpy as np


def simplex_project_vector(x, out=None):
    """Project vector onto standard simplex."""

    sorted_x = np.sort(x)

    n = sorted_x.size

    t_hat = 0
    for i in range(n - 2, -2, -1):
        t_hat = (sorted_x[-(n - 1 - i):].sum() - 1) / (n - 1 - i)
        if t_hat >= sorted_x[i]:
            break

    return np.fmax(x - t_hat, 0, out=out)


def simplex_project_columns(A):
    """Project columns of matrix onto standard simplex."""

    n_cols = A.shape[1]
    for i in range(n_cols):
        simplex_project_vector(A[:, i], out=A[:, i])


def simplex_project_rows(A):
    """Project rows of matrix onto standard simplex."""

    n_rows = A.shape[0]
    for i in range(n_rows):
        simplex_project_vector(A[i], out=A[i])
