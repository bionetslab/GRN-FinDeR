
from typing import Union

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
    # print(expr_matrix)

    grn = grnboost2(expression_data=expr_matrix, tf_names=tfs, verbose=True, seed=777)
    # print(grn)

    dist_mat_all = compute_wasserstein_distance_matrix(expr_matrix, -1)
    # print(dist_mat_all)

    tf_bool = [True if gene in tfs else False for gene in dist_mat_all.columns]
    gene_bool = [False if gene in tfs else True for gene in dist_mat_all.columns]
    dist_mat_tfs = dist_mat_all.loc[tf_bool, tf_bool]
    dist_mat_genes = dist_mat_all.loc[gene_bool, gene_bool]

    gene_to_clust = cluster_genes_to_dict(dist_mat_genes, num_clusters=3)
    # print(gene_to_clust)

    tf_to_clust = cluster_genes_to_dict(dist_mat_tfs, num_clusters=3)
    # print(tf_to_clust)

    grn_w_pvals = approximate_fdr(
        expression_mat=expr_matrix, grn=grn, gene_to_cluster=(tf_to_clust, gene_to_clust), num_permutations=2)

    print(grn_w_pvals)


def generate_input_multiple_tissues(root_directory : str, num_threads : int,
                                    num_clusters : list[int]):
    import pickle
    from src.distance_matrix import compute_wasserstein_distance_matrix
    from src.clustering import cluster_genes_to_dict
    import time
    import os
    import pandas as pd
    from arboreto.algo import grnboost2

    subdirectories = [os.path.join(root_directory, d) for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
    # Process all tissues.
    for subdir in subdirectories:
        print(f'Processing tissue {subdir}...')
        file_names = os.listdir(subdir)
        sorted_file_names = sorted(file_names, key=len)
        expression_file = sorted_file_names[0]
        print(expression_file)
        targets_file = sorted_file_names[1]
        print(targets_file)
        expression_mat = pd.read_csv(os.path.join(subdir, expression_file), sep='\t', index_col=0)
        targets = set(pd.read_csv(os.path.join(subdir, targets_file), index_col=0)['target_gene'].tolist())

        runtimes = []
        # Run GRN inference once.
        all_genes = set(expression_mat.columns.tolist())
        tfs = list(all_genes - targets)

        start_grn = time.time()
        grn = grnboost2(expression_data=expression_mat, tf_names=tfs, verbose=True, seed=42)
        end_grn = time.time()
        runtimes.append(end_grn - start_grn)
        grn.to_csv(os.path.join(subdir, 'input_grn.csv'))

        print("Computing Wasserstein distance matrix...")
        start_distance = time.time()
        dist_mat = compute_wasserstein_distance_matrix(expression_mat, num_threads)
        end_distance = time.time()
        dist_mat.to_csv(os.path.join(subdir, 'distance_matrix.csv'))
        runtimes.append(end_distance - start_distance)

        print("Clustering genes...")
        for n in num_clusters:
            start_cluster = time.time()
            gene_to_cluster = cluster_genes_to_dict(dist_mat, num_clusters=n)
            end_cluster = time.time()
            runtimes.append(end_cluster - start_cluster)
            os.makedirs(os.path.join(subdir, 'clusterings'), exist_ok=True)
            with open(os.path.join(subdir, 'clusterings', f'clustering_{n}.pkl'), 'wb') as f:
                pickle.dump(gene_to_cluster, f)

        column_names = ['grn', 'distance'] + [f'clustering_{x}' for x in num_clusters]
        runtimes_df = pd.DataFrame(columns=column_names)
        runtimes_df.loc['time'] = runtimes
        runtimes_df.to_csv(os.path.join(subdir, 'runtimes.csv'))


def compute_cluster_metrics(root_directory : str, num_clusters : list[int]):
    import pickle
    import os
    import pandas as pd
    import numpy as np
    from sklearn.metrics import silhouette_score

    subdirectories = [os.path.join(root_directory, d) for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
    tissues = [str(d) for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
    # Process all tissues.
    for tissue, subdir in zip(tissues, subdirectories):
        print(f'# ### Processing tissue {tissue}...')
        # expression_file = f'{tissue}.tsv'
        # targets_file = f'{tissue}_target_genes.tsv'
        # expression_mat = pd.read_csv(os.path.join(subdir, expression_file), sep='\t', index_col=0)
        # targets = set(pd.read_csv(os.path.join(subdir, targets_file), index_col=0)['target_gene'].tolist())

        # Read distance matrix.
        distance_file = os.path.join(subdir, 'distance_matrix.csv')
        distance_df = pd.read_csv(distance_file, index_col=0)
        distance_df.index = distance_df.columns

        # Iterate over desired cluster sizes.
        cluster_sizes_dict = dict()
        num_singletons_dict = dict()
        cluster_diam_dict = dict()
        median_cluster_member_distances_dict = dict()
        silhouette_score_dict = dict()
        for n in num_clusters:
            print(f'# ## Number of clusters: {n}...')
            # Read clustering from respective file.
            cluster_file = os.path.join(subdir, 'clusterings', f'clustering_{n}.pkl')
            with open(cluster_file, 'rb') as handle:
                gene_to_cluster = pickle.load(handle)
            # Invert gene-to-cluster dictionary to obtain cluster-to-gene dictionary.
            cluster_to_gene = dict()
            for key, val in gene_to_cluster.items():
                if val in cluster_to_gene:
                    cluster_to_gene[val].append(key)
                else:
                    cluster_to_gene[val] = [key]
            # ### Compute sizes of each cluster.
            sizes_per_cluster = [len(genes) for _, genes in cluster_to_gene.items()]
            cluster_sizes_dict[n] = sizes_per_cluster
            # print(f'# Cluster sizes:\n{sizes_per_cluster}')
            # ### Compute number of singleton clusters.
            num_singletons = sum([1 for _, genes in cluster_to_gene.items() if len(genes)==1])
            num_singletons_dict[n] = num_singletons
            print(f'# Number of singletons: {num_singletons}')
            # ### Compute diameter and median distance of clusters, i.e. maximum/median Wasserstein distance of pairs.
            cluster_diameters = []
            median_cluster_member_distances = []
            for _, genes in cluster_to_gene.items():
                # Subset distance matrix to given genes in cluster.
                subset_matrix = distance_df.loc[genes, genes].to_numpy()
                # Look up cluster diameter
                cluster_diam = subset_matrix.max()
                cluster_diameters.append(cluster_diam)
                # Compute median distance between cluster members
                upper_tri_elements = subset_matrix[np.triu_indices(subset_matrix.shape[0], k=1)]
                if upper_tri_elements.size == 0:
                    # Singleton has only self distance which is 0
                    cluster_median_dist = 0.0
                else:
                    cluster_median_dist = np.median(upper_tri_elements)

                median_cluster_member_distances.append(cluster_median_dist)

            cluster_diam_dict[n] = cluster_diameters
            median_cluster_member_distances_dict[n] = median_cluster_member_distances
            # print(f'# Cluster diameters:\n{cluster_diameters}')
            # print(f'# Cluster median distances:\n{median_cluster_member_distances_dict}')
            # ### Compute silhouette score
            # Create label vector
            cluster_label_vec = [gene_to_cluster[gene] for gene in distance_df.columns.tolist()]
            # Compute silhouette score
            sil_score = silhouette_score(distance_df.to_numpy(), cluster_label_vec, metric='precomputed')
            silhouette_score_dict[n] = sil_score
            print(f'# Clustering silhouette score: {sil_score}')

        # Save assemble dictionaries to file.
        savedir = os.path.join(subdir, 'clustering_metrics')
        os.makedirs(savedir, exist_ok=True)
        with open(os.path.join(savedir, "sizes_per_clustering.pkl"), 'wb') as f:
            pickle.dump(cluster_sizes_dict, f)

        with open(os.path.join(savedir, 'num_singletons_per_clustering.pkl'), 'wb') as f:
            pickle.dump(num_singletons_dict, f)

        with open(os.path.join(savedir, 'diameters_per_clustering.pkl'), 'wb') as f:
            pickle.dump(cluster_diam_dict, f)

        with open(os.path.join(savedir, 'median_distances_per_clustering.pkl'), 'wb') as f:
            pickle.dump(median_cluster_member_distances_dict, f)

        with open(os.path.join(savedir, 'silhouette_score_per_clustering.pkl'), 'wb') as f:
            pickle.dump(silhouette_score_dict, f)


def run_fdr_permutations_per_tissue(root_directory : str, num_clusters : list[int],
                                    num_permutations : int = 1000):
    import pickle
    import time
    import os
    import pandas as pd
    from src.fdr_calculation import approximate_fdr

    subdirectories = [os.path.join(root_directory, d) for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
    tissues = [str(d) for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
    # Process all tissues.
    for tissue, subdir in zip(tissues, subdirectories):
        print(f'Processing tissue {tissue}...')
        expression_file = f'{tissue}.tsv'
        targets_file = f'{tissue}_target_genes.tsv'
        expression_mat = pd.read_csv(os.path.join(subdir, expression_file), sep='\t', index_col=0)
        targets = set(pd.read_csv(os.path.join(subdir, targets_file), index_col=0)['target_gene'].tolist())

        # Read GRN to-be-pruned.
        grn_file = os.path.join(subdir, 'input_grn.csv')
        original_grn = pd.read_csv(grn_file, index_col=0)

        # Iterate over desired cluster sizes.
        runtimes = []
        for n in num_clusters:
            # Read clustering from respective file.
            cluster_file = os.path.join(subdir, 'clusterings', f'clustering_{n}.pkl')
            with open(cluster_file, 'rb') as handle:
                gene_to_cluster = pickle.load(handle)
            fdr_start = time.time()
            _ = approximate_fdr(expression_mat=expression_mat, grn=original_grn, gene_to_cluster=gene_to_cluster,
                                        num_permutations=num_permutations)
            fdr_end = time.time()
            runtimes.append((fdr_end - fdr_start)/num_permutations)

        # Save runtimes per cluster size to file.
        column_names = [f'clusters_{x}' for x in num_clusters]
        runtimes_df = pd.DataFrame(columns=column_names)
        runtimes_df.loc['time'] = runtimes
        runtimes_df.to_csv(os.path.join(subdir, 'times_per_num_clusters.csv'))


def run_approximate_fdr_control(
        expression_file_path: str,
        num_permutations: int = 1000,
        grn_file_path: Union[str, None] = None,  # Either load or infer input GRN
        target_file_path: Union[str, None] = None,  # Needed if input GRN is to be inferred, if None all genes are viewed as TFs
        clustering_file_path: Union[str, None] = None,  # Either load precomputed clustering or compute Wasserstein distance matrix and clustering
        num_clusters: Union[int, None] = None,  # Needed if clustering is to be computed, defaults to 100
        num_threads: Union[int, None] = None,  # Needed if clustering is to be computed, defaults to 6
        output_path: Union[str, None] = None,
) -> None:

    """Computes approximate FDR control for Gene Regulatory Networks (GRNs) based on empirical P-value computation.


        Args:
            expression_file_path (str): Path to the input file containing the preprocessed expression matrix.
                The file should be a tab-separated CSV with gene symbols as column headers.
            num_permutations (int): Number of permutations for empirical P-value computation.
            grn_file_path (Union[str, None]): Path to the input GRN file. If None, the GRN will be inferred.
            target_file_path (Union[str, None]): Path to a TSV file containing a newline-separated list of target genes.
                Required if `grn_file_path` is None. If None, all genes are considered as transcription factors (TFs).
            clustering_file_path (Union[str, None]): Path to a precomputed clustering file.
                If None, the Wasserstein distance matrix and clustering will be computed.
            num_clusters (Union[int, None]): Number of clusters for gene grouping. Required if clustering is computed;
                defaults to 100.
            num_threads (Union[int, None]): Number of threads for parallel computation of the Wasserstein distance matrix;
                defaults to 6.
            output_path (Union[str, None]): Path to save output files. If None, the current working directory is used.

        Outputs:
            - `distance_matrix.csv`: Wasserstein distance matrix (if clustering is computed).
            - `clustering.pkl`: Dictionary mapping genes to clusters (if clustering is computed).
            - `grn_pvalues.csv`: Empirical P-values for the GRN.
            - `times.csv`: Log of execution times for different steps.
        """

    import warnings
    import time
    import pandas as pd
    from arboreto.algo import grnboost2
    import pickle

    from src.distance_matrix import compute_wasserstein_distance_matrix
    from src.clustering import cluster_genes_to_dict
    from src.fdr_calculation import approximate_fdr

    if output_path is None:
        output_path = os.getcwd()

    # Read preprocessed expression matrix and TF list.
    exp_matrix = pd.read_csv(expression_file_path, sep='\t', index_col=0)

    if grn_file_path is None:
        if target_file_path is None:
            warnings.warn(
                "'target_file_path' should not be None if 'grn_file_path' is None. "
                "Running grn inference with all genes as possible targets"
            )
            targets = set()
        else:
            targets = set(pd.read_csv(target_file_path, index_col=0)['target_gene'].tolist())
        all_genes = set(exp_matrix.columns.tolist())
        tfs = list(all_genes - targets)

        grn = grnboost2(expression_data=exp_matrix, tf_names=tfs, verbose=True, seed=42)
        grn.to_csv(output_path + 'input_grn.csv')
    else:
        # Read GRN dataframe.
        grn = pd.read_csv(grn_file_path, index_col=0)

    if clustering_file_path is None:
        # Compute Wasserstein distance matrix.
        print("Computing Wasserstein distance matrix...")
        dist_start = time.time()
        dist_mat = compute_wasserstein_distance_matrix(exp_matrix, num_threads)
        dist_end = time.time()
        dist_mat.to_csv(output_path + "distance_matrix.csv", sep='\t')
        print(f'Wasserstein distance matrix computation took {dist_end-dist_start} seconds.')

        # Cluster genes based on Wasserstein distance.
        if num_clusters is None:
            num_clusters = 100
        print("Clustering genes...")
        clust_start = time.time()
        gene_to_cluster = cluster_genes_to_dict(dist_mat, num_clusters=num_clusters)
        clust_end = time.time()
        with open(output_path + "clustering.pkl", 'wb') as f:
            pickle.dump(gene_to_cluster, f)
        print(f'Gene clustering took {clust_end-clust_start} seconds.')
    else:
        with open(clustering_file_path, "rb") as f:
            gene_to_cluster = pickle.load(f)

    # Run approximate empirical P-value computation.
    print("Running approximate FDR control...")
    fdr_start = time.time()
    grn_pvals = approximate_fdr(expression_mat=exp_matrix, grn=grn, gene_to_cluster=gene_to_cluster,
                                num_permutations=num_permutations)
    fdr_end = time.time()
    grn_pvals.to_csv(os.path.join(output_path, 'grn_pvalues.csv'))
    print(f'Approximate FDR control took {fdr_end-fdr_start} seconds.')

    logger = pd.DataFrame()
    if clustering_file_path is None:
        logger['distance_mat'] = [dist_end-dist_start]
        logger['clustering'] = [clust_end-clust_start]
    logger['fdr'] = [fdr_end-fdr_start]
    logger.to_csv(os.path.join(output_path, 'times.csv'))


def approximate_fdr_validation(
        root_directory: str,
        num_clusters: list[int],
        num_permutations: int = 1000,
        verbosity: int = 0
):
    import pickle
    import time
    import os
    import warnings
    import pandas as pd
    from src.distance_matrix import compute_wasserstein_distance_matrix
    from src.clustering import cluster_genes_to_dict
    from src.fdr_calculation import approximate_fdr

    # Get subdirectories with expression and ground truth grn data for all tissues
    subdirectories = [
        os.path.join(root_directory, d) for d in os.listdir(root_directory)
        if os.path.isdir(os.path.join(root_directory, d))
    ]
    tissues = [str(d) for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]

    # Iterate over tissues
    for tissue, subdir in zip(tissues, subdirectories):

        if verbosity > 0:
            print(f'# ###### Processing tissue {tissue}...')

        # Load GRN to-be-pruned.
        grn_file = os.path.join(subdir, 'fdr_grn.tsv')
        try:
            original_grn = pd.read_csv(grn_file, sep='\t')
        except FileNotFoundError:
            warnings.warn(f'Ground truth GRN for tissue {tissue} not found. Continue.', UserWarning)
            continue

        # Load expression matrix
        expression_file = f'{tissue}.tsv'
        expression_mat = pd.read_csv(os.path.join(subdir, expression_file), sep='\t', index_col=0)

        # Compute and save distance matrix
        if verbosity > 0:
            print('# ### Computing Wasserstein distance matrix...')
        distance_mat_st = time.time()
        distance_mat = compute_wasserstein_distance_matrix(expression_mat=expression_mat, num_threads=-1)
        distance_mat_et = time.time()
        distance_mat_time = distance_mat_et - distance_mat_st
        distance_mat.to_csv(os.path.join(subdir, 'distance_matrix.csv'))
        if verbosity > 0:
            print(f'# ### Wasserstein distance matrix computation took {distance_mat_time} seconds.')

        # Compute clusterings and approximate fdr
        runtimes_idx = ['distance_matrix', ]
        runtimes = [distance_mat_time, ]
        for n in num_clusters:

            # Compute and save clustering
            if verbosity:
                print(f'# ### Computing clustering, n = {n} ...')
            gene_to_clust_st = time.time()
            gene_to_clust = cluster_genes_to_dict(distance_matrix=distance_mat, num_clusters=n)
            gene_to_clust_et = time.time()
            gene_to_clust_time = gene_to_clust_et - gene_to_clust_st
            runtimes_idx.append(f'clustering_{n}')
            runtimes.append(gene_to_clust_time)
            if verbosity > 0:
                print(f'# ### Clustering took {gene_to_clust_time} seconds.')

            with open(os.path.join(subdir, f'clustering_{n}.pkl'), 'wb') as f:
                pickle.dump(gene_to_clust, f)

            if verbosity > 0:
                print(f'# ### Approximate FDR, n = {n} ...')
            fdr_st = time.time()
            dummy_grn = approximate_fdr(
                expression_mat=expression_mat,
                grn=original_grn,
                gene_to_cluster=gene_to_clust,
                num_permutations=num_permutations
            )
            fdr_et = time.time()
            fdr_time = fdr_et - fdr_st
            runtimes_idx.append(f'fdr_{n}')
            runtimes.append(fdr_time)
            if verbosity > 0:
                print(f'# ### Approximate FDR computation took {fdr_time} seconds.')

            original_grn[f'count_{n}'] = dummy_grn['count'].to_numpy()
            original_grn[f'pvalue_{n}'] = dummy_grn['pvalue'].to_numpy()

            runtimes_df = pd.DataFrame(index=runtimes_idx)
            runtimes_df['runtimes'] = runtimes

            runtimes_df.to_csv(os.path.join(subdir, 'runtimes.csv'))

            original_grn.to_csv(os.path.join(subdir, 'grn.csv'))

def assess_approximation_quality(
        root_directory: str,
        num_clusters: list[int],
        num_permutations: int = 1000,
        filter_tissues : list[str] = []
):
    import os
    import pandas as pd
    from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, f1_score

    # Get subdirectories with expression and ground truth grn data for all tissues
    subdirectories = [
        os.path.join(root_directory, d) for d in os.listdir(root_directory)
        if os.path.isdir(os.path.join(root_directory, d))
    ]
    tissues = [str(d) for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]

    # Iterate over tissues
    for tissue, subdir in zip(tissues, subdirectories):

        if len(filter_tissues)>0 and (not tissue in filter_tissues):
            # Ignore current tissue subdirectory.
            continue 
        
        print(f'Processing tissue {tissue}...')
        # Load approximate Pvalues and counts.
        grn_file = os.path.join(subdir, 'grn.csv')
        grn = pd.read_csv(grn_file, index_col=0)
        
        results_dict = dict()
        for clusters in num_clusters:
            # Compute MSE between groundtruth counts and respective approx. counts.
            gt_counts = grn['counter'].to_list()
            approx_counts = grn[f'count_{clusters}'].to_list()
            mse_counts = mean_squared_error(gt_counts, approx_counts)
            # Threshold empirical Pvalues and compute respective metrics.
            gt_pvals = [(1.0+count)/(1.0+num_permutations) for count in gt_counts]
            approx_pvals = [(1.0+count)/(1.0+num_permutations) for count in approx_counts]
            gt_signif_005 = [1 if pval <= 0.05 else 0 for pval in gt_pvals]
            approx_signif_005 = [1 if pval <= 0.05 else 0 for pval in approx_pvals]
            gt_signif_001 = [1 if pval <= 0.01 else 0 for pval in gt_pvals]
            approx_signif_001 = [1 if pval <= 0.01 else 0 for pval in approx_pvals]
            # Compute approximation metrics.
            accuracy_pvals_005 = accuracy_score(gt_signif_005, approx_signif_005)
            hamming_pvals_005 = hamming_loss(gt_signif_005, approx_signif_005)
            f1_pvals_005 = f1_score(gt_signif_005, approx_signif_005)
            accuracy_pvals_001 = accuracy_score(gt_signif_001, approx_signif_001)
            hamming_pvals_001 = hamming_loss(gt_signif_001, approx_signif_001)
            f1_pvals_001 = f1_score(gt_signif_001, approx_signif_001)
            results_dict[f'clusters_{clusters}'] = [mse_counts, hamming_pvals_005, hamming_pvals_001,
                                                    accuracy_pvals_005, accuracy_pvals_001,
                                                    f1_pvals_005, f1_pvals_001]
        
        results_df = pd.DataFrame(results_dict)
        results_df.index = ['mse_counts', 'hamming_pvals005', 'hamming_pvals001', 
                            'accuracy_pvals005', 'accuracy_pvals001', 'f1_pvals005',
                            'f1_pvals001']
        results_df.to_csv(os.path.join(subdir, 'approx_results.csv'), index=True)
            


if __name__ == '__main__':

    # For GRNboost2 use: pip install dask-expr==0.5.3 distributed==2024.2.1

    import os

    generate_fdr_control_input = False
    cluster_metrics = False
    plot_clust_metrics = False
    fdr = False
    mwe = True
    approx_quality = False

    if generate_fdr_control_input:
        # ### Compute input to FDR control for all tissues (GRN, distance matrix, clustering)
        root_dir = os.path.join(os.getcwd(), 'data/gtex_tissues_preprocessed')
        n_threads = 20
        num_clusters_list = list(range(100, 5001, 100))
        generate_input_multiple_tissues(
            root_directory=root_dir,
            num_threads=n_threads,
            num_clusters=num_clusters_list,
        )

    elif cluster_metrics:
        # ### Compute cluster metrics for all tissues
        root_dir = os.path.join(os.getcwd(), 'data/gtex_tissues_preprocessed')
        num_clusters_list = list(range(100, 5001, 100))
        compute_cluster_metrics(
            root_directory=root_dir,
            num_clusters=num_clusters_list,
        )

    elif plot_clust_metrics:
        from src.postprocessing import plot_cluster_metrics
        root_dir = os.path.join(os.getcwd(), 'data/gtex_tissues_preprocessed')
        num_clusters_list = list(range(100, 5001, 100))
        plot_cluster_metrics(
            file_path=root_dir,
            num_clusters=num_clusters_list,
            plt_umap=True,
        )

    elif fdr:
        # ### Run approximate FDR control for Adipose_Tissue
        root_dir = os.path.join(os.getcwd(), 'data/gtex_tissues_preprocessed/Adipose_Tissue')
        expr_fp = os.path.join(root_dir, 'Adipose_Tissue.tsv')
        grn_fp = os.path.join(root_dir, 'input_grn.csv')

        n_permut = 1000
        n_clusters = [100, 200, 300, 400, 500, 600]

        for n in n_clusters:

            print(f'# ### Approximate FDR for {n} clusters ...')

            clust_fp = os.path.join(root_dir, 'clusterings', f'clustering_{n}.pkl')

            out_p = os.path.join(root_dir, 'approx_fdr_control', f'npermut{n_permut}_nclust{n}')
            os.makedirs(out_p, exist_ok=True)

            run_approximate_fdr_control(
                expression_file_path=expr_fp,
                num_permutations=n_permut,
                grn_file_path=grn_fp,
                target_file_path=None,
                clustering_file_path=clust_fp,
                num_clusters=None,
                num_threads=None,
                output_path=out_p,
            )
    if mwe:
        example_workflow()

    elif approx_quality:
        assess_approximation_quality(
        root_directory="/home/woody/iwbn/iwbn106h/gtex",
        num_clusters=[700],
        num_permutations=1000,
        filter_tissues=["Adipose_Tissue"])
    
    else:
        root_directory  = "/home/woody/iwbn/iwbn106h/gtex"
        num_clusters = [100, 300]

        approximate_fdr_validation(
            root_directory=root_directory,
            num_clusters=num_clusters,
            verbosity=1,
        )

    print("done")






