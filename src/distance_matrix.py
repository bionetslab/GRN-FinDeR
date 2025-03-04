
import pandas as pd
import numpy as np
from numba import njit, prange, set_num_threads, types


@njit(nopython=True, nogil=True)
def _merge_sorted_arrays(a, b):
    lenA,lenB = a.shape[0], b.shape[0]
    # Get searchsorted indices
    idx = np.searchsorted(a,b)
    # Offset each searchsorted indices with ranged array to get new positions of b in output array
    b_pos = np.arange(lenB) + idx
    lenTotal = lenA+lenB
    mask = np.ones(lenTotal,dtype=types.boolean)
    out = np.empty(lenTotal,dtype=types.float64)
    mask[b_pos] = False
    out[b_pos] = b
    out[mask] = a
    return out


@njit(nopython=True, parallel=True, nogil=True)
def _pairwise_wasserstein_dists(sorted_matrix, num_threads):
    if num_threads != -1:
        set_num_threads(num_threads)
    num_cols = sorted_matrix.shape[1]
    num_rows = sorted_matrix.shape[0]
    distance_mat = np.zeros((num_cols, num_cols))
    for col1 in prange(num_cols):
        for col2 in range(col1 + 1, num_cols):
            all_values = _merge_sorted_arrays(sorted_matrix[:, col1], sorted_matrix[:, col2])
            # Compute the differences between pairs of successive values of u and v.
            deltas = np.diff(all_values)
            # Get the respective positions of the values of u and v among the values of
            # both distributions.
            col1_cdf_indices = np.searchsorted(sorted_matrix[:, col1], all_values[:-1], 'right')
            col2_cdf_indices = np.searchsorted(sorted_matrix[:, col2], all_values[:-1], 'right')
            # Calculate the CDFs of u and v using their weights, if specified.
            col1_cdf = col1_cdf_indices / num_rows
            col2_cdf = col2_cdf_indices / num_rows
            # Compute the value of the integral based on the CDFs.
            distance = np.sum(np.multiply(np.abs(col1_cdf - col2_cdf), deltas))
            distance_mat[col1, col2] = distance
            distance_mat[col2, col1] = distance
    return distance_mat


def compute_wasserstein_distance_matrix(expression_mat : pd.DataFrame, num_threads: int = -1):
    numpy_mat = expression_mat.to_numpy()
    numpy_mat = np.sort(numpy_mat, axis=0)
    distance_mat = _pairwise_wasserstein_dists(numpy_mat, num_threads)
    distance_mat = pd.DataFrame(distance_mat, columns=expression_mat.columns)
    return distance_mat