

def compare_wiki_scipy_wasserstein():

    import numpy as np
    import pandas as pd
    from src.distance_matrix import compute_wasserstein_distance_matrix
    from scipy.stats import wasserstein_distance


    a = np.random.normal(0, 1, (15000, ))
    b = np.random.normal(1, 1, (15000,))
    c = np.random.normal(2, 1, (15000,))

    expr_matrix = pd.DataFrame(np.vstack((a, b, c)).T.copy())

    wd_wiki = compute_wasserstein_distance_matrix(expr_matrix, 4)

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


def time_wasserstein():

    import time
    import numpy as np
    import pandas as pd
    from src.distance_matrix import compute_wasserstein_distance_matrix, _pairwise_wasserstein_dists

    mtrx = pd.DataFrame(np.random.normal(0, 1, (10000, 15000)))

    # st = time.time()

    # dist_mtrx = compute_wasserstein_scipy_numba(mtrx, 16)

    # et = time.time()

    # print(f'# ### Computation time distance matrix: {et - st}')

    # (10000, 15000), 16 threads: 32799.50227713585

    # (1000, 15000), 16 threads: 2345.9375002384186


    st_sorting = time.time()

    mtrx_sorted = np.sort(mtrx, axis=0)

    et_sorting = time.time()


    st_dist = time.time()

    distance_mat = _pairwise_wasserstein_dists(sorted_matrix=mtrx_sorted, num_threads=16)

    et_dist = time.time()


    sorting_time = et_sorting - st_sorting
    dist_time = et_dist - st_dist

    print(f"# ### Input matrix shape: {mtrx.shape}")
    print(f"# ### Time sorting: {sorting_time} s")
    print(f"# ### Time wasserstein: {dist_time} s")

    # shape: (1000, 15000), Time sorting: 0.20757675170898438 s, Time wasserstein: 2745.7698554992676 s
    # shape: (10000, 15000), Time sorting: 1.796863317489624 s, Time wasserstein: 30009.90110206604 s


def example_workflow():
    import os
    import time
    import numpy as np
    import pandas as pd
    from arboreto.algo import grnboost2

    from src.distance_matrix import compute_wasserstein_distance_matrix
    from src.clustering import cluster_genes_to_dict
    from src.fdr_calculation import approximate_fdr

    n_tfs = 10
    n_genes = 10
    n_cells = 10
    tfs = [f'TF{i}' for i in range(n_tfs)]
    genes = [f'Gene{i}' for i in range(n_genes)]
    # Construct dummy example

    np.random.seed(42)

    expr_matrix = pd.DataFrame(
        # np.random.normal(0, 1, (n_cells, n_tfs + n_genes)),
        np.random.poisson(lam=np.random.gamma(shape=2, scale=1, size=(n_cells, n_tfs + n_genes))),
        columns=tfs + genes,
    )

    print(expr_matrix)

    grn = grnboost2(expression_data=expr_matrix, tf_names=tfs, verbose=True, seed=777)

    print(grn)

    dist_mat = compute_wasserstein_distance_matrix(expr_matrix, 4)

    print(dist_mat)

    gene_to_clust = cluster_genes_to_dict(dist_mat, num_clusters=3)
    print(gene_to_clust)

    grn_w_pvals = approximate_fdr(
        expression_mat=expr_matrix, grn=grn, gene_to_cluster=gene_to_clust, num_permutations=2)

    print(grn_w_pvals)


def run_approximate_fdr_control(expression_file_path : str, tf_file_path : str, grn_file_path : str,
                                num_permutations : int, num_clusters : int, num_threads : int,
                                output_path : str):
    """Computes approximate FDR control for GRNs based on empirical P-value computation.

    Args:
        expression_file_path (str): Path to input file containig preprocessed expression matrix. Should be
            tab-separated csv file, with gene symbols as columns.
        grn (str): Path to dataframe containing edges of to-be-corrected GRN.
        tf_file_path (str): Path to txt file containig newline-separated list of TFs as gene symbols.
        num_permutations (int): Number of permutations to run for empirical P-value computation.
        num_clusters (int): Number of clusters to cluster genes into and draw representatives from.
        num_threads (int): How many threads to use for numba-based parallelized computation of 
            Wasserstein distance matrix.
    """
    import os
    import time
    import numpy as np
    import pandas as pd
    from arboreto.algo import grnboost2
    import pickle

    from src.distance_matrix import compute_wasserstein_distance_matrix
    from src.clustering import cluster_genes_to_dict
    from src.fdr_calculation import approximate_fdr
    
    # Read preprocessed expression matrix and TF list.
    exp_matrix = pd.read_csv(expression_file_path, sep='\t', index_col=0)
    with open(tf_file_path) as tf:
        tfs = [line.strip() for line in tf]
        
    # Make sure that all TFs are actually contained in expression matrix [check if this is necessary?].
    tfs = list(set(tfs).intersection(list(exp_matrix.columns)))
        
    # Read GRN dataframe.
    # grn = pd.read_csv(grn_file_path, sep='\t')
    grn = grnboost2(expression_data=exp_matrix, tf_names=tfs, verbose=True, seed=777)
    grn.to_csv(output_path + 'input_grn.csv')
    
    # Compute Wasserstein distance matrix.
    print("Computing Wasserstein distance matrix...")
    dist_start = time.time()
    dist_mat = compute_wasserstein_distance_matrix(expr_matrix, num_threads)
    dist_end = time.time()
    dist_mat.to_csv(output_path + "distance_matrix.csv", sep='\t')
    print(f'Wasserstein distance matrix computation took {dist_end-dist_start} seconds.')
    
    # Cluster genes based on Wasserstein distance.
    print("Clustering genes...")
    clust_start = time.time()
    gene_to_cluster = cluster_genes_to_dict(dist_mat, num_clusters=num_clusters)
    clust_end = time.time()
    with open(output_path + "clustering.pkl", 'wb') as f:
        pickle.dump(gene_to_clust, f)
    print(f'Gene clustering took {clust_end-clust_start} seconds.')
    
    # Run approximate empirical P-value computation.
    print("Running approximate FDR control...")
    fdr_start = time.time()
    grn_pvals = approximate_fdr(expression_mat=exp_matrix, grn=grn, gene_to_cluster=gene_to_cluster,
                                num_permutations=num_permutations)
    fdr_end = time.time()
    grn_pvals.to_csv(output_path + "grn_pvalues.csv", sep='\t')
    print(f'Approximate FDR control took {fdr_end-fdr_start} seconds.')
    
    logger = pd.DataFrame()
    logger['distance_mat'] = [dist_end-dist_start]
    logger['clustering'] = [clust_end-clust_start]
    logger['fdr'] = [fdr_end-fdr_start]
    logger.to_csv(output_path + 'logger.csv')
    

if __name__ == '__main__':

    # Adjust paths (orig. GTEX data path: /data/bionets/datasets/gtex/<tissue>)
    # TF file URL: https://resources.aertslab.org/cistarget/tf_lists/
    expression_file_path = "/data/bionets/xa39zypy/GTEX/GTEX_Prostate_filtered_standardized.tsv"
    tf_file_path = "/data/bionets/xa39zypy/GTEX/allTFs_hg38.txt"
    grn_file_path = ""
    num_permutations = 100
    num_clusters = 100
    num_threads = 32
    output_path = "/data/bionets/xa39zypy/GRN-FinDeR/data/Prostate/"
    run_approximate_fdr_control(expression_file_path, tf_file_path, grn_file_path, 
                                num_permutations, num_clusters, num_threads, output_path)
    
    print("done")
