

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
    from src.distance_matrix import compute_wasserstein_scipy_numba, pairwise_wasserstein_dists
    from scipy.stats import wasserstein_distance

    def wasserstein_distance_wiki(u: np.ndarray, v: np.ndarray) -> float:

        u_sorted = np.sort(u)
        v_sorted = np.sort(v)

        return np.abs(u_sorted - v_sorted).sum()

    a = np.random.normal(0, 1, (15000, ))
    b = np.random.normal(1, 1, (15000,))

    wd_wiki = pairwise_wasserstein_dists(np.vstack((a, b)).T.copy())

    wd_scipy = wasserstein_distance(a, b)

    print(f"WD wiki: {wd_wiki}")
    print(f"WD scipy: {wd_scipy}")
    # Not the same ...

    # Problem probably is that in the 1st definition/here:
    # - https://en.wikipedia.org/wiki/Wasserstein_metric#Empirical_distributions
    # the Wasserstein distance is defined for empirical measures.
    # This implicitly assumes that the entries of the data vectors are iid
    # (see: https://en.wikipedia.org/wiki/Empirical_measure).
    # However, that is not the case in our setting. (At least I think so ???)

    # We should probably use this definition:
    # https://en.wikipedia.org/wiki/Wasserstein_metric#Higher_dimensions

    # Todo:
    #  Copy scipies code:
    #  https://github.com/scipy/scipy/blob/v1.15.1/scipy/stats/_stats_py.py#L9872
    #  Line 9872, _cdf_distance
    #  BUT, optimize for pairwise calculation given a matrix input
    #  e.g. sort matrices once at the beginning ...


if __name__ == '__main__':

    # time_wiki_wasserstein()

    compare_wiki_scipy_wasserstein()

    print("done")
