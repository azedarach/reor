"""
Provides implementation of FurthestSum initialization strategy.
"""


__all__ = ['furthest_sum']


import numpy as np


def _update_distance_sums(dissimilarity_matrix, new_index, current_distances):
    for i, qi in enumerate(current_distances):
        dij = dissimilarity_matrix[new_index, qi[0]]
        current_distances[i][1] += dij


def _get_furthest_index(current_distances):
    current_distances.sort(key=lambda x: x[1])
    furthest_index = current_distances.pop(-1)[0]
    return furthest_index


def _furthest_sum_impl(dissimilarity_matrix, n_components, start_index,
                       exclude=None, extra_steps=1):
    """Performs FurthestSum initialization.

    Parameters
    ----------
    dissimilarity_matrix : array-like, shape (n_samples, n_samples)
        Array containing dissimilarities between samples.

    n_components : integer
        Number of elements to select.

    start_index: integer
        Index of initial point.

    exclude : None or list, optional
        If given, a list of integers corresponding to the indices
        of the sample points to exclude.

    extra_steps : integer, optional
        Number of extra steps to take in selecting sample elements.

    Returns
    -------
    selected : array-like, shape (n_components,)
        Indices of selected sample points.
    """

    if n_components == 0:
        return []

    if exclude is None:
        exclude = []

    n_samples = dissimilarity_matrix.shape[0]
    n_excluded = len(exclude)

    if start_index >= n_samples:
        raise ValueError('Start index %r is out of bounds (n_samples = %d)' %
                         (start_index, n_samples))

    for index in exclude:
        if index == start_index:
            raise ValueError('Start index %r is excluded' % start_index)

    if n_excluded < n_samples and n_components > n_samples - n_excluded:
        raise ValueError(
            'Too few point available to select requested number of components '
            '(n_components=%d, n_samples=%d, n_excluded=%d)' %
            (n_components, n_samples, n_excluded))

    selected = np.full((n_components,), start_index)

    def is_valid_candidate(index):
        is_valid = True

        for i in selected:
            if index == i:
                is_valid = False
                break

        for i in exclude:
            if index == i:
                is_valid = False
                break

        return is_valid

    q = [None] * (n_samples - n_excluded - 1)
    pos = 0
    for i in range(n_samples):
        if is_valid_candidate(i):
            q[pos] = [i, dissimilarity_matrix[i, start_index]]
            pos += 1

    for i in range(1, n_components):
        selected[i] = _get_furthest_index(q)
        _update_distance_sums(dissimilarity_matrix, selected[i], q)

    if extra_steps > 0:
        for i in range(extra_steps):
            update_index = i % n_components
            index_to_replace = selected[update_index]

            for i, qi in enumerate(q):
                dij = dissimilarity_matrix[qi[0], index_to_replace]
                q[i][1] -= dij

            qi = 0
            for index in selected:
                if index != index_to_replace:
                    dij = dissimilarity_matrix[index_to_replace, index]
                    qi += dij
            q.append([index_to_replace, qi])

            selected[update_index] = _get_furthest_index(q)
            _update_distance_sums(
                dissimilarity_matrix, selected[update_index], q)

    return selected


def furthest_sum(dissimilarity_matrix, n_components, start_index,
                 exclude=None, extra_steps=1):
    """Run FurthestSum initialization on given dissimilarity matrix.

    Parameters
    ----------
    dissimilarity_matrix : array-like, shape (n_samples, n_samples)
        Array containing dissimilarities between samples.

    n_components : integer
        Number of elements to select.

    start_index: integer
        Index of initial point.

    exclude : None or list, optional
        If given, a list of integers corresponding to the indices
        of the sample points to exclude.

    extra_steps : integer, optional
        Number of extra steps to take in selecting sample elements.

    Returns
    -------
    selected : array-like, shape (n_components,)
        Indices of selected sample points.
    """

    if dissimilarity_matrix.shape[0] != dissimilarity_matrix.shape[1]:
        raise ValueError(
            'Dissimilarity matrix must be square, but got shape %r' %
            list(dissimilarity_matrix.shape))

    return _furthest_sum_impl(
        dissimilarity_matrix, n_components, start_index,
        exclude=exclude, extra_steps=extra_steps)
