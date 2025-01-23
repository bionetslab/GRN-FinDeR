

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

    from scipy.stats import wasserstein_distance

    def wasserstein_distance_wiki(u: np.ndarray, v: np.ndarray) -> float:

        u_sorted = np.sort(u)
        v_sorted = np.sort(v)

        return np.abs(u_sorted - v_sorted).sum()

    a = np.random.normal(0, 1, (15000, ))
    b = np.random.normal(1, 1, (15000,))

    wd_wiki = wasserstein_distance_wiki(a, b)

    wd_scipy = wasserstein_distance(a, b)

    print(f"WD wiki: {wd_wiki}")
    print(f"WD scipy: {wd_scipy}")

    # Not the same ...


if __name__ == '__main__':

    # time_wiki_wasserstein()

    compare_wiki_scipy_wasserstein()

    print("done")
