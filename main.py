

def time_wiki_wasserstein():

    # Using the definition from Wikipedia: L_1-distance of sorted expression vectors
    # https://en.wikipedia.org/wiki/Wasserstein_metric#Empirical_distributions

    import time
    import numpy as np
    import pandas as pd

    from src.distance_matrix import compute_wasserstein_presort

    expr_matrix = np.random.normal(0, 1, (2000, 15000))

    expr_df = pd.DataFrame(
        data=expr_matrix,
        index=[f"gene{i}" for i in range(expr_matrix.shape[0])],
    )

    print('# ### Starting distance matrix computation ...')
    st = time.time()

    distance_matrix = compute_wasserstein_presort(expression_matrix=expr_df)

    et = time.time()

    print(f"# ### Time: {et - st}")

    print(f"# ### Distance matrix shape: {distance_matrix.shape}")


def compare_wiki_scipy_wasserstein():

    import numpy as np
    import pandas as pd
    from src.distance_matrix import compute_wasserstein_scipy_numba
    from scipy.stats import wasserstein_distance


    a = np.random.normal(0, 1, (15000, ))
    b = np.random.normal(1, 1, (15000,))
    c = np.random.normal(2, 1, (15000,))

    expr_matrix = pd.DataFrame(np.vstack((a, b, c)).T.copy())

    wd_wiki = compute_wasserstein_scipy_numba(expr_matrix, 4)

    wd_scipy_ab = wasserstein_distance(a, b)
    wd_scipy_ac = wasserstein_distance(a, c)
    wd_scipy_bc = wasserstein_distance(b, c)

    print(f"WD wiki: {wd_wiki}")
    print(f"WD scipy: {wd_scipy_ab, wd_scipy_ac, wd_scipy_bc}")



def numpy_vs_numba_sorting():

    import time
    import numpy as np
    from numba import njit, prange, set_num_threads

    @njit(parallel=True, nogil=True)
    def numba_sort(matrix: np.ndarray, n: int = 4, inplace: bool = True):
        set_num_threads(n)
        if not inplace:
            matrix = matrix.copy()

        for i in prange(matrix.shape[1]):
            matrix[:, i] = np.sort(matrix[:, i])

        return matrix

    mtrx = np.random.normal(0, 1, (10000, 15000))

    st = time.time()
    sorted_matrix_numba = numba_sort(mtrx, inplace=False)
    et = time.time()

    t_numba = et - st

    st = time.time()
    sorted_matrix_numpy = np.sort(mtrx, axis=0)
    et = time.time()

    t_npy = et - st

    print(f"# ### Time numba: {t_numba}")
    print(f"# ### Time numpy: {t_npy}")


if __name__ == '__main__':

    # time_wiki_wasserstein()

    compare_wiki_scipy_wasserstein()

    # numpy_vs_numba_sorting()

    print("done")
