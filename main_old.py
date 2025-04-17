
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
        tissue_list : list = [],
        include_tfs : bool = False,
        num_permutations: int = 1000,
        verbosity: int = 0,
        keep_tfs_singleton : bool = False,
        scale_importances : bool = False,
        use_cluster_medoids : bool = False
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
        
        if len(tissue_list)>0 and (not tissue in tissue_list):
            continue

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
        
        # Load target genes file.
        if include_tfs:
            target_file = f'{tissue}_target_genes.tsv'
            target_df = pd.read_csv(os.path.join(subdir, target_file), index_col=0)
            tf_list = set(expression_mat.columns) - set(target_df['target_gene'])

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
            # Check whether to separately cluster TFs.
            if include_tfs:
                tf_bool = [True if gene in tf_list else False for gene in distance_mat.columns]
                gene_bool = [False if gene in tf_list else True for gene in distance_mat.columns]
                dist_mat_tfs = distance_mat.loc[tf_bool, tf_bool]
                dist_mat_genes = distance_mat.loc[gene_bool, gene_bool]
                if keep_tfs_singleton:
                    # Ensure to process all TFs, by creating dummy-singleton clusters for TFs.
                    tfs_to_clust = {tf : counter for (counter, tf) in enumerate(tf_list)}
                else:
                    tfs_to_clust = cluster_genes_to_dict(distance_matrix=dist_mat_tfs, num_clusters=n)
                genes_to_clust = cluster_genes_to_dict(distance_matrix=dist_mat_genes, num_clusters=n)
            else:
                gene_to_clust = cluster_genes_to_dict(distance_matrix=distance_mat, num_clusters=n)
            gene_to_clust_et = time.time()
            gene_to_clust_time = gene_to_clust_et - gene_to_clust_st
            runtimes_idx.append(f'clustering_{n}')
            runtimes.append(gene_to_clust_time)
            if verbosity > 0:
                print(f'# ### Clustering took {gene_to_clust_time} seconds.')

            if include_tfs:
                with open(os.path.join(subdir, f'tf_clustering_{n}.pkl'), 'wb') as f:
                    pickle.dump(tfs_to_clust, f)
                with open(os.path.join(subdir, f'gene_clustering_{n}.pkl'), 'wb') as f:
                    pickle.dump(genes_to_clust, f)
            else:
                with open(os.path.join(subdir, f'clustering_{n}.pkl'), 'wb') as f:
                    pickle.dump(gene_to_clust, f)

            cluster_medoid_dict = None
            if use_cluster_medoids:
                if include_tfs:
                    # Iterate over all target clusters and compute medoid based on Wasserstein dist.
                    cluster_medoid_dict = {}
                    cluster_gene_dict = {}
                    for gene, cluster in genes_to_clust.items():
                        if cluster in cluster_gene_dict:
                            cluster_gene_dict[cluster].append(gene)
                        else:
                            cluster_gene_dict[cluster] = [gene]
                    for cluster, gene_list in cluster_gene_dict.items():
                        # Subset distance matrix to cluster gene set.
                        subset_bool = [True if gene in gene_list else False for gene in distance_mat.columns]
                        subset_dists = distance_mat.loc[subset_bool, subset_bool]
                        distance_sums = subset_dists.sum(axis=1)
                        medoid_gene = distance_sums.idxmin()
                        cluster_medoid_dict[cluster] = medoid_gene
                    # TODO: Implement medoid also for TF clusters.
                else:
                    raise ValueError("Medoid representatives not yet implemented for inclusive TF clustering!")
            
            if verbosity > 0:
                print(f'# ### Approximate FDR, n = {n} ...')
            fdr_st = time.time()
            
            if include_tfs:
                dummy_grn = approximate_fdr(
                    expression_mat=expression_mat,
                    grn=original_grn,
                    gene_to_cluster=(tfs_to_clust, genes_to_clust),
                    num_permutations=num_permutations,
                    scale_importances=scale_importances,
                    cluster_medoid_dict=cluster_medoid_dict)
            else:
                dummy_grn = approximate_fdr(
                    expression_mat=expression_mat,
                    grn=original_grn,
                    gene_to_cluster=gene_to_clust,
                    num_permutations=num_permutations,
                    scale_importances=scale_importances
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
    from sklearn.metrics import (
        mean_squared_error, accuracy_score, hamming_loss, f1_score, precision_score, recall_score
    )

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
        grn_file = os.path.join(subdir, 'approx_grn_with_tf_clustering.csv')
        # grn_file = os.path.join(subdir, 'grn.csv')
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
            prec_pvals_005 = precision_score(gt_signif_005, approx_signif_005)
            rec_pvals_005 = recall_score(gt_signif_005, approx_signif_005)

            accuracy_pvals_001 = accuracy_score(gt_signif_001, approx_signif_001)
            hamming_pvals_001 = hamming_loss(gt_signif_001, approx_signif_001)
            f1_pvals_001 = f1_score(gt_signif_001, approx_signif_001)
            prec_pvals_001 = precision_score(gt_signif_001, approx_signif_001)
            rec_pvals_001 = recall_score(gt_signif_001, approx_signif_001)

            results_dict[f'clusters_{clusters}'] = [
                mse_counts, hamming_pvals_005, hamming_pvals_001, accuracy_pvals_005, accuracy_pvals_001, f1_pvals_005,
                f1_pvals_001, prec_pvals_005, prec_pvals_001, rec_pvals_005, rec_pvals_001
            ]
        
        results_df = pd.DataFrame(results_dict)
        results_df.index = [
            'mse_counts', 'hamming_pvals005', 'hamming_pvals001', 'accuracy_pvals005', 'accuracy_pvals001',
            'f1_pvals005', 'f1_pvals001', 'prec_pvals005', 'prec_pvals001', 'rec_pvals005', 'rec_pvals001'
        ]

        results_df.to_csv(os.path.join(subdir, 'approx_results.csv'), index=True)
            
def compute_distance_metainfo(root_directory: str,
        filter_tissues : list[str] = []):
    import os
    import pandas as pd
    import numpy as np

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
        dist_file = os.path.join(subdir, 'distance_matrix.csv')
        dist_df = pd.read_csv(dist_file, index_col=0)
        
        dist_mat = dist_df.to_numpy()
        mask = np.eye(dist_mat.shape[0], dtype=bool)
        non_diag_values = dist_mat[~mask]
        unique_values = np.unique(non_diag_values)
        info_dict = {'min' : [unique_values.min()], 'max' : [unique_values.max()], 'median' : [np.median(unique_values)], 'mean' : [np.mean(unique_values)]}
        info_df = pd.DataFrame(info_dict)
        info_df.to_csv(os.path.join(subdir, 'distance_info.csv'))

def debug_exampe0():

    import os
    import time
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from arboreto.algo import grnboost2
    from src.fdr_calculation import classical_fdr, approximate_fdr
    from src.distance_matrix import compute_wasserstein_distance_matrix
    from src.clustering import cluster_genes_to_dict
    from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

    # ### Set all flags and vars here ##################################################################################
    grnboost2_random_seed = 42
    n_permutations = 1000
    sig_thresh_plot = 0.05
    # n_clusters = list(range(120, 200, 20))
    n_clusters = list(range(10, 101, 20)) + list(range(100, 200, 20)) + list(range(200, 401, 20))
    fdr_thresholds = [0.05, 0.01]

    mean_case1 = 0
    mean_case2 = 11
    scale_case1_case2 = 1.0

    n_cells = 300

    n_tfs_case1 = 100
    n_tfs_case2 = 100
    n_non_tfs_case1 = 200
    n_non_tfs_case2 = 200

    inference = True
    just_approx = False

    # ### Define path where results are stored
    save_p = os.path.join(os.getcwd(), 'results/debug0_w_fixed_seed')
    os.makedirs(save_p, exist_ok=True)

    if inference:
        if not just_approx:
            # Simulate data, infer input GRN, classical FDR, compute distance matrix

            # ### Simulate simple dataset:
            # - 2 Normal distributions (I/case1, II/case2)
            # - I) Mean low, unit variance, II) Mean high, unit variance
            # - Some TFs with I) some with II) same for non TFs
            # - I) TFs should be predictors for I) genes, same for II)

            np.random.seed(42)

            case1_tf_data = np.random.normal(loc=mean_case1, scale=scale_case1_case2, size=(n_cells, n_tfs_case1))
            case2_tf_data = np.random.normal(loc=mean_case2, scale=scale_case1_case2, size=(n_cells, n_tfs_case2))
            case1_non_tf_data = np.random.normal(loc=mean_case1, scale=scale_case1_case2, size=(n_cells, n_non_tfs_case1))
            case2_non_tf_data = np.random.normal(loc=mean_case2, scale=scale_case1_case2, size=(n_cells, n_non_tfs_case2))

            x = np.concatenate((case1_tf_data, case2_tf_data, case1_non_tf_data, case2_non_tf_data), axis=1)

            tf_names = [f'TF_{i}_c1' for i in range(n_tfs_case1)] + [f'TF_{i}_c2' for i in range(n_tfs_case2)]
            non_tf_names = [f'Gene_{i}_c1' for i in range(n_non_tfs_case1)] + [f'Gene_{i}_c2' for i in range(n_non_tfs_case2)]

            expression_df = pd.DataFrame(x, columns=tf_names + non_tf_names)

            # print(expression_df)
            print(
                f"Mean expr. TFs c1: {expression_df[[f'TF_{i}_c1' for i in range(n_tfs_case1)]].to_numpy().mean()}\n"
                f"Mean expr. TFs c2: {expression_df[[f'TF_{i}_c2' for i in range(n_tfs_case2)]].to_numpy().mean()}\n"
                f"Mean expr. non TFs c1: {expression_df[[f'Gene_{i}_c1' for i in range(n_non_tfs_case1)]].to_numpy().mean()}\n"
                f"Mean expr. non TFs c1: {expression_df[[f'Gene_{i}_c2' for i in range(n_non_tfs_case2)]].to_numpy().mean()}"
            )

            expression_df.to_csv(os.path.join(save_p, 'expression_df.csv'))

            # ### Infer input GRN
            print('# ### Inferring input GRN ...')
            st_input_grn = time.time()
            input_grn = grnboost2(
                expression_data=expression_df,
                tf_names=tf_names,
                verbose=True,
                seed=grnboost2_random_seed
            )
            et_input_grn = time.time()
            print(f'# took {et_input_grn - st_input_grn} seconds')

            # print(input_grn)

            c1_reg_c1_count = ((input_grn['TF'].str.endswith('c1')) & (input_grn['target'].str.endswith('c1'))).sum()
            c2_reg_c2_count = ((input_grn['TF'].str.endswith('c2')) & (input_grn['target'].str.endswith('c2'))).sum()

            print(
                f'Class 1 regulated class 1, count: {c1_reg_c1_count}\n'
                f'Class 2 regulated class 2, count: {c2_reg_c2_count}\n'
                f'Rest: {input_grn.shape[0] - (c1_reg_c1_count + c2_reg_c2_count)}\n'
            )

            c1_reg_c1_or_c2_reg_c2_fractions = []
            top_k = list(range(1, input_grn.shape[0] + 1))
            for i in top_k:
                count = (
                        ((input_grn.iloc[:i]['TF'].str.endswith('c1')) & (input_grn.iloc[:i]['target'].str.endswith('c1'))) |
                        ((input_grn.iloc[:i]['TF'].str.endswith('c2')) & (input_grn.iloc[:i]['target'].str.endswith('c2')))
                ).sum()

                c1_reg_c1_or_c2_reg_c2_fractions.append(count / (i + 1))

            fig, ax = plt.subplots(dpi=300)
            ax.plot(top_k, c1_reg_c1_or_c2_reg_c2_fractions, c='r', linewidth=0.5)
            ax.set_ylabel('Fraction c1-c1 or c2-c2 edges')
            ax.set_xlabel('Top k edges (ranked by importance)')
            plt.savefig(os.path.join(save_p, 'frac_of_c1_c1_or_c2_c2_regulation_in_top_k.png'), dpi=300)
            plt.close('all')

            input_grn.to_csv(os.path.join(save_p, 'input_grn.csv'))

            # ### Perform classical FDR control
            print('# ### Performing classical FDR ...')
            st_classical_fdr = time.time()
            ground_truth_grn = classical_fdr(
                expression_mat=expression_df,
                grn=input_grn,
                tf_names=tf_names,
                num_permutations=n_permutations,
                grnboost2_random_seed=grnboost2_random_seed,
                verbosity=1,
            )
            et_classical_fdr = time.time()
            print(f'# took {et_classical_fdr - st_classical_fdr} seconds')

            # print(ground_truth_grn)
            sig_bool = ground_truth_grn['p_value'].to_numpy() <= sig_thresh_plot
            n_sig = np.cumsum(sig_bool) / np.array(list(range(1, ground_truth_grn.shape[0] + 1)))

            fig, ax = plt.subplots(dpi=300)
            ax.plot(top_k, n_sig, c='g', linewidth=0.5)
            ax.set_ylabel('Fraction significant edges')
            ax.set_xlabel('Top k edges (ranked by importance)')
            plt.savefig(os.path.join(save_p, 'frac_of_significant_in_top_k.png'), dpi=300)
            plt.close('all')

            ground_truth_grn.to_csv(os.path.join(save_p, 'ground_truth_grn.csv'))

            # ### Perform approximate FDR control
            print('# ### Computing Wasserstein distance matrix...')
            st_dist_mat = time.time()
            distance_mat = compute_wasserstein_distance_matrix(expression_mat=expression_df, num_threads=-1)
            et_dist_mat = time.time()
            print(f'# took {et_dist_mat - st_dist_mat} seconds')

            distance_mat.to_csv(os.path.join(save_p, 'distance_mat.csv'))

        else:
            # Load expression matrix, input grn, ground truth grn
            expression_df = pd.read_csv(os.path.join(save_p, 'expression_df.csv'), index_col=0)
            input_grn = pd.read_csv(os.path.join(save_p, 'input_grn.csv'), index_col=0)
            ground_truth_grn = pd.read_csv(os.path.join(save_p, 'approx_fdr_grn.csv'), index_col=0)
            distance_mat = pd.read_csv(os.path.join(save_p, 'distance_mat.csv'), index_col=0)

            tf_names = [s for s in expression_df.columns if s.startswith('TF')]

        # Cluster TFs and non TFs separately
        tf_bool = [True if gene in tf_names else False for gene in distance_mat.columns]
        gene_bool = [not b for b in tf_bool]
        distance_mat_tfs = distance_mat.loc[tf_bool, tf_bool]
        distance_mat_non_tfs = distance_mat.loc[gene_bool, gene_bool]

        for n_clust in n_clusters:
            print(f'# ### Approx. FDR control with {n_clust} clusters ...')

            if len(tf_names) > n_clust:
                tfs_to_clust = cluster_genes_to_dict(distance_matrix=distance_mat_tfs, num_clusters=n_clust)
            else:
                tfs_to_clust = {tfn: i for i, tfn in enumerate(tf_names)}  # No clustering
            non_tfs_to_clust = cluster_genes_to_dict(distance_matrix=distance_mat_non_tfs, num_clusters=n_clust)

            # Perform FDR control
            dummy_grn = approximate_fdr(
                expression_mat=expression_df,
                grn=input_grn,
                gene_to_cluster=(tfs_to_clust, non_tfs_to_clust),
                num_permutations=n_permutations,
                grnboost2_random_seed=grnboost2_random_seed,
            )

            # Append results to groundtruth GRN
            ground_truth_grn[f'count_{n_clust}'] = dummy_grn['count'].to_numpy()
            ground_truth_grn[f'pvalue_{n_clust}'] = dummy_grn['pvalue'].to_numpy()

            ground_truth_grn.to_csv(os.path.join(save_p, 'approx_fdr_grn.csv'))

        # print(ground_truth_grn)

    # ### Evaluate approximation performance
    else:
        ground_truth_grn = pd.read_csv(os.path.join(save_p, 'approx_fdr_grn.csv'), index_col=0)
        # print(ground_truth_grn)

        # Sort columns
        gtgrn_part1 = ground_truth_grn.iloc[:, :5].copy()
        gtgrn_part2 = ground_truth_grn.iloc[:, 5:].copy()

        sorted_cols = sorted(gtgrn_part2.columns, key=lambda x: int(x.split("_")[1]))
        gtgrn_part2 = gtgrn_part2[sorted_cols]

        ground_truth_grn = pd.concat([gtgrn_part1, gtgrn_part2], axis=1)


    res_df = pd.DataFrame(index=['mse', 'acc', 'prec', 'rec', 'f1'])
    for thresh in fdr_thresholds:
        ground_truth_p_vals = ground_truth_grn['p_value'].to_numpy()
        ground_truth_sig_bool = ground_truth_p_vals <= thresh
        for n_clust in n_clusters:
            approx_p_vals = ground_truth_grn[f'pvalue_{n_clust}'].to_numpy()
            approx_sig_bool =  approx_p_vals <= thresh

            mse = mean_squared_error(ground_truth_p_vals, approx_p_vals)
            acc = accuracy_score(ground_truth_sig_bool, approx_sig_bool)
            prec = precision_score(ground_truth_sig_bool, approx_sig_bool)
            rec = recall_score(ground_truth_sig_bool, approx_sig_bool)
            f1 = f1_score(ground_truth_sig_bool, approx_sig_bool)

            res_df[f'nclust{n_clust}_thresh{thresh}'] = [mse, acc, prec, rec, f1]

    k = len(n_clusters)
    res_df_005 = res_df.iloc[:, 0:k]
    res_df_001 = res_df.iloc[:, 0:k]
    print(res_df_005)
    print(res_df_001)

    res_df.to_csv(os.path.join(save_p, 'res_df.csv'))

    fig, ax = plt.subplots(dpi=300)
    ax.plot(n_clusters, res_df_005.loc['f1'], c='g', label='F1, thresh: 0.05', linewidth=1.0, marker='o')
    ax.plot(n_clusters, res_df_001.loc['f1'], c='b', label='F1, thresh: 0.01', linewidth=1.0, marker='o')
    ax.set_ylabel('F1 score')
    ax.set_xlabel('N clusters')
    ax.axvline(x=n_tfs_case1 + n_tfs_case1, color='r', linestyle='--', label='number of TFs')
    plt.legend()
    plt.savefig(os.path.join(save_p, 'eval_f1.png'), dpi=300)
    plt.close('all')

    fig, ax = plt.subplots(dpi=300)
    ax.plot(n_clusters, res_df_005.loc['rec'], c='g', label='Rec, thresh: 0.05', linewidth=1.0, marker='o')
    ax.plot(n_clusters, res_df_001.loc['rec'], c='b', label='Rec, thresh: 0.01', linewidth=1.0, marker='o')
    ax.set_ylabel('Recall score')
    ax.set_xlabel('N clusters')
    ax.axvline(x=n_tfs_case1 + n_tfs_case1, color='r', linestyle='--', label='number of TFs')
    plt.legend()
    plt.savefig(os.path.join(save_p, 'eval_rec.png'), dpi=300)
    plt.close('all')

    fig, ax = plt.subplots(dpi=300)
    ax.plot(n_clusters, res_df_005.loc['prec'], c='g', label='Prec, thresh: 0.05', linewidth=1.0, marker='o')
    ax.plot(n_clusters, res_df_001.loc['prec'], c='b', label='Prec, thresh: 0.01', linewidth=1.0, marker='o')
    ax.set_ylabel('Precision score')
    ax.set_xlabel('N clusters')
    ax.axvline(x=n_tfs_case1 + n_tfs_case1, color='r', linestyle='--', label='number of TFs')
    plt.legend()
    plt.savefig(os.path.join(save_p, 'eval_prec.png'), dpi=300)
    plt.close('all')

    fig, ax = plt.subplots(dpi=300)
    ax.plot(n_clusters, res_df_005.loc['acc'], c='g', label='Acc, thresh: 0.05', linewidth=1.0, marker='o')
    ax.plot(n_clusters, res_df_001.loc['acc'], c='b', label='Acc, thresh: 0.01', linewidth=1.0, marker='o')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('N clusters')
    ax.axvline(x=n_tfs_case1 + n_tfs_case1, color='r', linestyle='--', label='number of TFs')
    plt.legend()
    plt.savefig(os.path.join(save_p, 'eval_acc.png'), dpi=300)
    plt.close('all')

    fig, ax = plt.subplots(dpi=300)
    ax.plot(n_clusters, res_df_005.loc['mse'], c='g', label='MSE', linewidth=1.0, marker='o')
    ax.set_ylabel('MSE')
    ax.set_xlabel('N clusters')
    ax.axvline(x=n_tfs_case1 + n_tfs_case1, color='r', linestyle='--', label='number of TFs')
    plt.legend()
    plt.savefig(os.path.join(save_p, 'eval_mse.png'), dpi=300)
    plt.close('all')


def debug_exampe1():

    import os
    import time
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from arboreto.algo import grnboost2
    from src.fdr_calculation import classical_fdr, approximate_fdr
    from src.distance_matrix import compute_wasserstein_distance_matrix
    from src.clustering import cluster_genes_to_dict
    from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

    # ### Set all flags and vars here ##################################################################################
    grnboost2_random_seed = 42
    n_permutations = 1000
    sig_thresh_plot = 0.05
    n_clusters = [10, ] + list(range(20, 301, 20))  # 300 as upper bound, draw 2 representatives (2 x 300 <= 600)
    fdr_thresholds = [0.05, 0.01]

    mean_case1 = 0
    mean_case2 = 11
    scale_case1_case2 = 1.0

    n_cells = 300

    n_genes_case1 = 300
    n_genes_case2 = 300

    inference = True
    just_approx = False

    # ### Define path where results are stored
    save_p = os.path.join(os.getcwd(), 'results/debug1_w_fixed_seed')
    os.makedirs(save_p, exist_ok=True)

    if inference:
        if not just_approx:
            # Simulate data, infer input GRN, classical FDR, compute distance matrix

            # ### Simulate simple dataset:
            # - 2 Normal distributions (I/case1, II/case2)
            # - I) Mean low, unit variance, II) Mean high, unit variance
            # - Some TFs with I) some with II) same for non TFs
            # - I) TFs should be predictors for I) genes, same for II)

            np.random.seed(42)

            case1_data = np.random.normal(loc=mean_case1, scale=scale_case1_case2, size=(n_cells, n_genes_case1))
            case2_data = np.random.normal(loc=mean_case2, scale=scale_case1_case2, size=(n_cells, n_genes_case2))

            x = np.concatenate((case1_data, case2_data), axis=1)

            gene_names = [f'G_{i}_c1' for i in range(n_genes_case1)] + [f'G_{i}_c2' for i in range(n_genes_case2)]

            expression_df = pd.DataFrame(x, columns=gene_names)

            # print(expression_df)
            print(
                f"Mean expr. genes c1: {expression_df[[f'G_{i}_c1' for i in range(n_genes_case1)]].to_numpy().mean()}\n"
                f"Mean expr. genes c2: {expression_df[[f'G_{i}_c2' for i in range(n_genes_case2)]].to_numpy().mean()}\n"
            )
            expression_df.to_csv(os.path.join(save_p, 'expression_df.csv'))

            # ### Infer input GRN
            print('# ### Inferring input GRN ...')
            st_input_grn = time.time()
            input_grn = grnboost2(
                expression_data=expression_df,
                tf_names=None,   # No TFs !!!
                verbose=True,
                seed=grnboost2_random_seed
            )
            et_input_grn = time.time()
            print(f'# took {et_input_grn - st_input_grn} seconds')

            # print(input_grn)

            c1_reg_c1_count = ((input_grn['TF'].str.endswith('c1')) & (input_grn['target'].str.endswith('c1'))).sum()
            c2_reg_c2_count = ((input_grn['TF'].str.endswith('c2')) & (input_grn['target'].str.endswith('c2'))).sum()

            print(
                f'Class 1 regulated class 1, count: {c1_reg_c1_count}\n'
                f'Class 2 regulated class 2, count: {c2_reg_c2_count}\n'
                f'Rest: {input_grn.shape[0] - (c1_reg_c1_count + c2_reg_c2_count)}\n'
            )

            c1_reg_c1_or_c2_reg_c2_fractions = []
            top_k = list(range(1, input_grn.shape[0] + 1))
            for i in top_k:
                count = (
                        ((input_grn.iloc[:i]['TF'].str.endswith('c1')) & (input_grn.iloc[:i]['target'].str.endswith('c1'))) |
                        ((input_grn.iloc[:i]['TF'].str.endswith('c2')) & (input_grn.iloc[:i]['target'].str.endswith('c2')))
                ).sum()

                c1_reg_c1_or_c2_reg_c2_fractions.append(count / (i + 1))

            fig, ax = plt.subplots(dpi=300)
            ax.plot(top_k, c1_reg_c1_or_c2_reg_c2_fractions, c='r', linewidth=0.5)
            ax.set_ylabel('Fraction c1-c1 or c2-c2 edges')
            ax.set_xlabel('Top k edges (ranked by importance)')
            plt.savefig(os.path.join(save_p, 'frac_of_c1_c1_or_c2_c2_regulation_in_top_k.png'), dpi=300)
            plt.close('all')

            input_grn.to_csv(os.path.join(save_p, 'input_grn.csv'))

            # ### Perform classical FDR control
            print('# ### Performing classical FDR ...')
            st_classical_fdr = time.time()
            ground_truth_grn = classical_fdr(
                expression_mat=expression_df,
                grn=input_grn,
                tf_names=None,  # No TFs !!!
                num_permutations=n_permutations,
                grnboost2_random_seed=grnboost2_random_seed,
                verbosity=1,
            )
            et_classical_fdr = time.time()
            print(f'# took {et_classical_fdr - st_classical_fdr} seconds')

            # print(ground_truth_grn)
            sig_bool = ground_truth_grn['p_value'].to_numpy() <= sig_thresh_plot
            n_sig = np.cumsum(sig_bool) / np.array(list(range(1, ground_truth_grn.shape[0] + 1)))

            fig, ax = plt.subplots(dpi=300)
            ax.plot(top_k, n_sig, c='g', linewidth=0.5)
            ax.set_ylabel('Fraction significant edges')
            ax.set_xlabel('Top k edges (ranked by importance)')
            plt.savefig(os.path.join(save_p, 'frac_of_significant_in_top_k.png'), dpi=300)
            plt.close('all')

            ground_truth_grn.to_csv(os.path.join(save_p, 'ground_truth_grn.csv'))

            # ### Perform approximate FDR control
            print('# ### Computing Wasserstein distance matrix...')
            st_dist_mat = time.time()
            distance_mat = compute_wasserstein_distance_matrix(expression_mat=expression_df, num_threads=-1)
            et_dist_mat = time.time()
            print(f'# took {et_dist_mat - st_dist_mat} seconds')

            distance_mat.to_csv(os.path.join(save_p, 'distance_mat.csv'))

        else:
            # Load expression matrix, input grn, ground truth grn
            expression_df = pd.read_csv(os.path.join(save_p, 'expression_df.csv'), index_col=0)
            input_grn = pd.read_csv(os.path.join(save_p, 'input_grn.csv'), index_col=0)
            ground_truth_grn = pd.read_csv(os.path.join(save_p, 'approx_fdr_grn.csv'), index_col=0)
            distance_mat = pd.read_csv(os.path.join(save_p, 'distance_mat.csv'), index_col=0)


        for n_clust in n_clusters:
            print(f'# ### Approx. FDR control with {n_clust} clusters ...')

            gene_to_clust = cluster_genes_to_dict(distance_matrix=distance_mat, num_clusters=n_clust)

            # Perform FDR control
            dummy_grn = approximate_fdr(
                expression_mat=expression_df,
                grn=input_grn,
                gene_to_cluster=gene_to_clust,  # No TFs !!!
                num_permutations=n_permutations,
                grnboost2_random_seed=grnboost2_random_seed,
            )

            # Append results to groundtruth GRN
            ground_truth_grn[f'count_{n_clust}'] = dummy_grn['count'].to_numpy()
            ground_truth_grn[f'pvalue_{n_clust}'] = dummy_grn['pvalue'].to_numpy()

            ground_truth_grn.to_csv(os.path.join(save_p, 'approx_fdr_grn.csv'))

        # print(ground_truth_grn)

    # ### Evaluate approximation performance
    else:
        ground_truth_grn = pd.read_csv(os.path.join(save_p, 'approx_fdr_grn.csv'), index_col=0)
        # print(ground_truth_grn)

    res_df = pd.DataFrame(index=['mse', 'acc', 'prec', 'rec', 'f1'])
    for thresh in fdr_thresholds:
        ground_truth_p_vals = ground_truth_grn['p_value'].to_numpy()
        ground_truth_sig_bool = ground_truth_p_vals <= thresh
        for n_clust in n_clusters:
            approx_p_vals = ground_truth_grn[f'pvalue_{n_clust}'].to_numpy()
            approx_sig_bool =  approx_p_vals <= thresh

            mse = mean_squared_error(ground_truth_p_vals, approx_p_vals)
            acc = accuracy_score(ground_truth_sig_bool, approx_sig_bool)
            prec = precision_score(ground_truth_sig_bool, approx_sig_bool)
            rec = recall_score(ground_truth_sig_bool, approx_sig_bool)
            f1 = f1_score(ground_truth_sig_bool, approx_sig_bool)

            res_df[f'nclust{n_clust}_thresh{thresh}'] = [mse, acc, prec, rec, f1]

    k = len(n_clusters)
    res_df_005 = res_df.iloc[:, 0:k]
    res_df_001 = res_df.iloc[:, 0:k]
    print(res_df_005)
    print(res_df_001)

    res_df.to_csv(os.path.join(save_p, 'res_df.csv'))

    fig, ax = plt.subplots(dpi=300)
    ax.plot(n_clusters, res_df_005.loc['f1'], c='g', label='F1, thresh: 0.05', linewidth=1.0, marker='o')
    ax.plot(n_clusters, res_df_001.loc['f1'], c='b', label='F1, thresh: 0.01', linewidth=1.0, marker='o')
    ax.set_ylabel('F1 score')
    ax.set_xlabel('N clusters')
    plt.legend()
    plt.savefig(os.path.join(save_p, 'eval_f1.png'), dpi=300)
    plt.close('all')

    fig, ax = plt.subplots(dpi=300)
    ax.plot(n_clusters, res_df_005.loc['rec'], c='g', label='Rec, thresh: 0.05', linewidth=1.0, marker='o')
    ax.plot(n_clusters, res_df_001.loc['rec'], c='b', label='Rec, thresh: 0.01', linewidth=1.0, marker='o')
    ax.set_ylabel('Recall score')
    ax.set_xlabel('N clusters')
    plt.legend()
    plt.savefig(os.path.join(save_p, 'eval_rec.png'), dpi=300)
    plt.close('all')

    fig, ax = plt.subplots(dpi=300)
    ax.plot(n_clusters, res_df_005.loc['prec'], c='g', label='Prec, thresh: 0.05', linewidth=1.0, marker='o')
    ax.plot(n_clusters, res_df_001.loc['prec'], c='b', label='Prec, thresh: 0.01', linewidth=1.0, marker='o')
    ax.set_ylabel('Precision score')
    ax.set_xlabel('N clusters')
    plt.legend()
    plt.savefig(os.path.join(save_p, 'eval_prec.png'), dpi=300)
    plt.close('all')

    fig, ax = plt.subplots(dpi=300)
    ax.plot(n_clusters, res_df_005.loc['acc'], c='g', label='Acc, thresh: 0.05', linewidth=1.0, marker='o')
    ax.plot(n_clusters, res_df_001.loc['acc'], c='b', label='Acc, thresh: 0.01', linewidth=1.0, marker='o')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('N clusters')
    plt.legend()
    plt.savefig(os.path.join(save_p, 'eval_acc.png'), dpi=300)
    plt.close('all')

    fig, ax = plt.subplots(dpi=300)
    ax.plot(n_clusters, res_df_005.loc['mse'], c='g', label='MSE', linewidth=1.0, marker='o')
    ax.set_ylabel('MSE')
    ax.set_xlabel('N clusters')
    plt.legend()
    plt.savefig(os.path.join(save_p, 'eval_mse.png'), dpi=300)
    plt.close('all')


def main_time_shuffling():

    import time
    import numpy as np
    import pandas as pd

    from numba import njit, prange

    expr_mat = np.random.normal(loc=1, scale=6, size=(2000, 20000))
    expr_df = pd.DataFrame(expr_mat, columns=[f'gene_{i}' for i in range(expr_mat.shape[1])])

    def shuffle_columns_numba_fisher_yates(df):
        matrix = df.to_numpy()
        shuffled = shuffle_columns_numba_fisher_yates_worker(matrix)
        return pd.DataFrame(shuffled, columns=df.columns, index=df.index)

    @njit(parallel=True)
    def shuffle_columns_numba_fisher_yates_worker(matrix):
        rows, cols = matrix.shape
        out = matrix.copy()
        for col in prange(cols):  # Parallelize over columns
            for i in range(rows - 1, 0, -1):  # Fisher-Yates
                j = np.random.randint(0, i + 1)
                out[i, col], out[j, col] = out[j, col], out[i, col]
        return out

    def shuffle_columns_numba_np(df):
        matrix = df.to_numpy()
        shuffled = shuffle_columns_numba_np_worker(matrix)
        return pd.DataFrame(shuffled, columns=df.columns, index=df.index)

    @njit(parallel=True)
    def shuffle_columns_numba_np_worker(matrix):
        rows, cols = matrix.shape
        out = np.empty_like(matrix)
        for col in prange(cols):
            out[:, col] = np.random.permutation(matrix[:, col])
        return out

    def shuffle_columns_numpy(df):
        matrix = df.to_numpy()
        shuffled = np.empty_like(matrix)
        for i in range(matrix.shape[1]):
            shuffled[:, i] = np.random.permutation(matrix[:, i])
        return pd.DataFrame(shuffled, columns=df.columns, index=df.index)

    def shuffle_columns_pandas(df):
        return df.apply(np.random.permutation, axis=0)

    def time_shuffling(df, perm_method, n):

        st = time.time()
        for i in range(n):
            perm_method(df)
        et = time.time()

        return et -st

    n_permut = 20
    print('# ### Numba fisher-yates ...')
    t_numba_fisher = time_shuffling(df=expr_df, perm_method=shuffle_columns_numba_fisher_yates, n=n_permut)
    print('# ### Numba fisher-yates sek per iter: ', t_numba_fisher / n_permut)

    print('# ### Numba np ...')
    t_numba_np = time_shuffling(df=expr_df, perm_method=shuffle_columns_numba_np, n=n_permut)
    print('# ### Numba np sek per iter: ', t_numba_np / n_permut)

    print('# ### Np  ...')
    t_np = time_shuffling(df=expr_df, perm_method=shuffle_columns_numpy, n=n_permut)
    print('# ### Np sek per iter: ', t_np / n_permut)

    print('# ### Pd ...')
    t_pd = time_shuffling(df=expr_df, perm_method=shuffle_columns_pandas, n=n_permut)
    print('# ### Pd sek per iter: ', t_pd / n_permut)


def approx_fdr_scanpy_data_mwe():

    import os
    import time
    import pickle
    import warnings
    import pandas as pd
    import matplotlib.pyplot as plt

    from src.utils import DebugDataSuite
    from arboreto.algo import grnboost2
    from src.distance_matrix import compute_wasserstein_distance_matrix
    from src.fdr_calculation import classical_fdr, approximate_fdr
    from src.clustering import cluster_genes_to_dict
    from src.utils import compute_evaluation_metrics, plot_metric

    # ### Set flags ####################################################################################################
    save_p = os.path.join(os.getcwd(), 'results/tfs_not_clustered')
    os.makedirs(save_p, exist_ok=True)

    process_data = False
    downsampling_frac_cells = 0.1
    downsampling_frac_genes = 0.2
    # cells: 0.1, genes: 0.2; 263 cells, 373 genes, 25 TFs
    # No ds: 2638 cells, 1868 genes, 159 TFs

    compute_input_grn = False
    use_tf_info = True

    compute_distance_mat = False

    compute_classical_fdr = False
    n_permutations = 1000

    compute_clusterings = False
    n_clusters_clustering = list(range(20, 101, 20)) + list(range(125, 326, 25)) # + [348, ]
    cluster_tfs = False

    compute_approximate_fdr = False
    n_clusters_approx_fdr = list(range(20, 101, 20)) + list(range(125, 326, 25)) # + [348, ]

    evaluate = True
    fdr_thresholds = [0.01, 0.05]

    ####################################################################################################################

    # Load, process, and save data
    if process_data:
        dds = DebugDataSuite(cache_dir=save_p, verbosity=1)
        dds.load_and_preprocess()
        dds.downsample_scale(fraction_cells=downsampling_frac_cells, fraction_genes=downsampling_frac_genes, seed=42)
        expression_mat = dds.expression_mat_.copy()
    else:
        expression_mat = pd.read_csv(os.path.join(save_p, 'pbmc3k_prepr_downsampled.csv'), index_col=0)

    # Compute the input grn
    if compute_input_grn:

        if use_tf_info:
            # Download the TF list
            if not os.path.exists(os.path.join(save_p, 'allTFs_hg38.txt')):
                url = 'https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt'
                os.system(f'wget -P {save_p} {url}')

            # Load the TF list
            with open(os.path.join(save_p, 'allTFs_hg38.txt'), 'r') as file:
                tfs = [line.strip() for line in file]
                genes = expression_mat.columns.tolist()
                intersection = list(set(tfs) & set(genes))

                n_genes = len(genes)
                n_tfs = len(intersection)
                print(f'# ### Out of the {n_genes} genes {n_tfs} ({n_tfs / n_genes * 100:.2f}%) are TFs')
        else:
            intersection = 'all'

        print('# ### Inferring input GRN ...')
        st_input_grn = time.time()
        input_grn = grnboost2(
            expression_data=expression_mat,
            tf_names=intersection,
            verbose=True,
            seed=42
        )
        et_input_grn = time.time()
        print(f'# took {et_input_grn - st_input_grn} seconds')

        print(expression_mat.shape)
        print(len(intersection))

        input_grn.to_csv(os.path.join(save_p, 'input_grn.csv'))

    else:
        input_grn = pd.read_csv(os.path.join(save_p, 'input_grn.csv'), index_col=0)

    # Compute the distance matrix
    if compute_distance_mat:
        print('# ### Computing Wasserstein distance matrix...')
        st_dist_mat = time.time()
        distance_mat = compute_wasserstein_distance_matrix(expression_mat=expression_mat, num_threads=-1)
        et_dist_mat = time.time()
        print(f'# took {et_dist_mat - st_dist_mat} seconds')
        distance_mat.to_csv(os.path.join(save_p, 'distance_mat.csv'))
    else:
        distance_mat = pd.read_csv(os.path.join(save_p, 'distance_mat.csv'), index_col=0)

    # Compute classical FDR
    if compute_classical_fdr:

        if use_tf_info:
            # Load the TF list
            with open(os.path.join(save_p, 'allTFs_hg38.txt'), 'r') as file:
                tfs = [line.strip() for line in file]

                genes = expression_mat.columns.tolist()
                intersection = list(set(tfs) & set(genes))
        else:
            intersection = 'all'

        print('# ### Performing classical FDR ...')
        st_classical_fdr = time.time()
        ground_truth_grn = classical_fdr(
            expression_mat=expression_mat,
            grn=input_grn,
            tf_names=intersection,
            num_permutations=n_permutations,
            grnboost2_random_seed=42,
            verbosity=1,
        )
        et_classical_fdr = time.time()
        t_classical_fdr = et_classical_fdr - st_classical_fdr
        print(f'# took {t_classical_fdr} seconds')

        time_df_classical = pd.DataFrame(columns=['total', 'per_iter'])
        time_df_classical.loc['classical', :] = [t_classical_fdr, t_classical_fdr / n_permutations]
        time_df_classical.to_csv(os.path.join(save_p, 'time_classical_fdr.csv'))

        ground_truth_grn.to_csv(os.path.join(save_p, 'ground_truth_grn.csv'))
    else:
        ground_truth_grn = pd.read_csv(os.path.join(save_p, 'ground_truth_grn.csv'), index_col=0)

    if compute_clusterings:
        if use_tf_info:

            # Subset distance matrix
            with open(os.path.join(save_p, 'allTFs_hg38.txt'), 'r') as file:
                tfs = [line.strip() for line in file]
                genes = expression_mat.columns.tolist()
                intersection = list(set(tfs) & set(genes))

            tf_bool = [True if gene in intersection else False for gene in distance_mat.columns]
            gene_bool = [not b for b in tf_bool]
            distance_mat_tfs = distance_mat.loc[tf_bool, tf_bool]
            distance_mat_non_tfs = distance_mat.loc[gene_bool, gene_bool]

            clusterings_tfs = []
            clusterings_non_tfs = []

            for n in n_clusters_clustering:

                n_leq_n_tfs = n <= len(intersection)

                if not n_leq_n_tfs:
                    warnings.warn(f"Number of clusters > number of TFs. Not clustering TFs.")

                if cluster_tfs and n_leq_n_tfs:
                    tfs_to_clust = cluster_genes_to_dict(distance_matrix=distance_mat_tfs, num_clusters=n)
                else:
                    tfs_to_clust = {tfn: i for i, tfn in enumerate(intersection)}  # No clustering

                non_tfs_to_clust = cluster_genes_to_dict(distance_matrix=distance_mat_non_tfs, num_clusters=n)

                clusterings_tfs.append(tfs_to_clust)
                clusterings_non_tfs.append(non_tfs_to_clust)

                cluster_p = os.path.join(save_p, 'clusterings')
                os.makedirs(cluster_p, exist_ok=True)
                with open(os.path.join(cluster_p, f'{n}_clusters_tfs{'_not_clustered' if not cluster_tfs else ''}.pkl'), 'wb') as f:
                    pickle.dump(tfs_to_clust, f)

                with open(os.path.join(cluster_p, f'{n}_clusters_non_tfs.pkl'), 'wb') as f:
                    pickle.dump(non_tfs_to_clust, f)

        else:

            clusterings = []

            for n in n_clusters_clustering:
                gene_to_clust = cluster_genes_to_dict(distance_matrix=distance_mat, num_clusters=n)

                clusterings.append(gene_to_clust)

                cluster_p = os.path.join(save_p, 'clusterings')
                os.makedirs(cluster_p, exist_ok=True)
                with open(os.path.join(cluster_p, f'{n}_clusters.pkl'), 'wb') as f:
                    pickle.dump(gene_to_clust, f)
    else:

        cluster_p = os.path.join(save_p, 'clusterings')

        if use_tf_info:

            clusterings_tfs = []
            clusterings_non_tfs = []

            for n in n_clusters_clustering:

                with open(
                        os.path.join(
                            cluster_p, f'{n}_clusters_tfs{'_not_clustered' if not cluster_tfs else ''}.pkl'
                        ),
                        'rb'
                ) as f:
                    tfs_to_clust = pickle.load(f)
                clusterings_tfs.append(tfs_to_clust)

                with open(os.path.join(cluster_p, f'{n}_clusters_non_tfs.pkl'), 'rb') as f:
                    non_tfs_to_clust = pickle.load(f)
                clusterings_non_tfs.append(non_tfs_to_clust)

        else:
            clusterings = []
            for n in n_clusters_clustering:
                with open(os.path.join(cluster_p, f'{n}_clusters.pkl'), 'rb') as f:
                    gene_to_clust = pickle.load(f)
                clusterings.append(gene_to_clust)

    if compute_approximate_fdr:

        if use_tf_info:
            clusterings_tfs_fdr = []
            clusterings_non_tfs_fdr = []
            n_clusters_approx_fdr_new = []
            for i, n in enumerate(n_clusters_clustering):
                if n in n_clusters_approx_fdr:
                    clusterings_tfs_fdr.append(clusterings_tfs[i])
                    clusterings_non_tfs_fdr.append(clusterings_non_tfs[i])
                    n_clusters_approx_fdr_new.append(n)
        else:
            clusterings_fdr = []
            n_clusters_approx_fdr_new = []
            for i, n in enumerate(n_clusters_clustering):
                if n in n_clusters_approx_fdr:
                    clusterings_fdr.append(clusterings[i])
                    n_clusters_approx_fdr_new.append(n)

        for n in n_clusters_approx_fdr:
            if n not in n_clusters_approx_fdr_new:
                print(f'No clustering computed for n = {n}. Cannot compute approximate FDR.')

        approx_fdr_grn = ground_truth_grn.copy()
        time_df_approx = pd.DataFrame(columns=['total', 'per_iter'])
        for i, n in enumerate(n_clusters_approx_fdr_new):
            print(f'# ### Approx. FDR control with {n} clusters ...')
            if use_tf_info:
                cluster_input = (clusterings_tfs_fdr[i], clusterings_non_tfs_fdr[i])
            else:
                cluster_input = clusterings_fdr[i]

            # Perform FDR control
            st = time.time()
            dummy_grn = approximate_fdr(
                expression_mat=expression_mat,
                grn=input_grn,
                gene_to_cluster=cluster_input,
                num_permutations=n_permutations,
                grnboost2_random_seed=42
            )
            et = time.time()
            t_approx = et - st
            print(f'# took {t_approx} seconds')

            time_df_approx.loc[f'{n}_clusters', :] = [t_approx, t_approx / n_permutations]
            time_df_approx.to_csv(os.path.join(save_p, 'time_approx_fdr.csv'))

            # Append results to groundtruth GRN
            approx_fdr_grn[f'count_{n}'] = dummy_grn['count'].to_numpy()
            approx_fdr_grn[f'pvalue_{n}'] = dummy_grn['pvalue'].to_numpy()

            approx_fdr_grn.to_csv(os.path.join(save_p, 'approx_fdr_grn.csv'))

    else:
        n_clusters_approx_fdr_new = []
        for i, n in enumerate(n_clusters_clustering):
            if n in n_clusters_approx_fdr:
                n_clusters_approx_fdr_new.append(n)
        approx_fdr_grn = pd.read_csv(os.path.join(save_p, 'approx_fdr_grn.csv'), index_col=0)

    if evaluate:
        res_df = compute_evaluation_metrics(
            grn=approx_fdr_grn,
            fdr_thresholds=fdr_thresholds,
            n_clusters=None,  # Fet from df
        )

        with open(os.path.join(save_p, 'allTFs_hg38.txt'), 'r') as file:
            tfs = [line.strip() for line in file]

            genes = expression_mat.columns.tolist()
            intersection = list(set(tfs) & set(genes))

        save_p_plots = os.path.join(save_p, 'plots')
        os.makedirs(save_p_plots, exist_ok=True)

        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(len(fdr_thresholds))]

        for metric in ['mse', 'acc', 'prec', 'rec', 'f1']:
            fig, ax = plt.subplots()
            for i, (fdr_threshold, col) in enumerate(zip(fdr_thresholds, colors)):
                plot_metric(
                    res_df=res_df,
                    fdr_threshold=fdr_threshold,
                    metric=metric,
                    n_tfs=len(intersection) if i == len(fdr_thresholds) - 1 else None,
                    line_color=col,
                    ax=ax
                )
            plt.savefig(os.path.join(save_p_plots, f'{metric}.png'))
            plt.close('all')





if __name__ == '__main__':

    # Todo: when classical is done for not downsampled, stop and restart with other n clusters
    approx_fdr_scanpy_data_mwe()

    # import pandas as pd
    # grn0 = pd.read_csv('results/tfs_not_clustered/approx_fdr_grn_0.csv', index_col=0)
    # grn1 = pd.read_csv('results/tfs_not_clustered/approx_fdr_grn_1.csv', index_col=0)
    # grn = pd.concat([grn0, grn1.iloc[:, 6:]], axis=1)
    # grn.to_csv('results/tfs_not_clustered/approx_fdr_grn.csv')

    quit()

    import os
    import pickle
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    grn = pd.read_csv('results/tfs_not_clustered/approx_fdr_grn.csv', index_col=0)

    sp = './results/tfs_not_clustered/plots/'
    os.makedirs(sp, exist_ok=True)

    fig ,ax = plt.subplots(dpi=300)
    ax.scatter(
        grn['importance'].to_numpy(),
        -np.log10(grn['p_value'].to_numpy()),
        s=11,
        c='skyblue',
        edgecolors='black',
        linewidth=0.5,
    )
    ax.set_xlabel('Importance')
    ax.set_ylabel('- log10(p_value)')
    plt.savefig(sp + 'scatter.png', dpi=300)
    plt.close('all')

    fig, ax = plt.subplots(dpi=300)
    ax.hist(grn['p_value'].to_numpy(), bins=30, color='skyblue', edgecolor='black')
    ax.set_xlabel('p_value')
    ax.set_ylabel('count')
    plt.savefig(sp + 'hist.png', dpi=300)
    plt.close('all')

    for i in list(range(20, 101, 20)) + list(range(125, 326, 25)):
        fig, ax = plt.subplots(dpi=300)
        ax.hist(grn[f'pvalue_{i}'].to_numpy(), bins=30, color='skyblue', edgecolor='black')
        ax.set_xlabel('p_value')
        ax.set_ylabel('count')
        plt.savefig(sp + f'hist_{i}.png', dpi=300)
        plt.close('all')

    for i in list(range(20, 101, 20)) + list(range(125, 326, 25)):
        fig, ax = plt.subplots(dpi=300)
        ax.scatter(
            grn['p_value'].to_numpy(),
            grn[f'pvalue_{i}'].to_numpy(),
            s=11,
            c='skyblue',
            edgecolors='black',
            linewidth=0.5,
        )
        ax.set_xlabel('p-val ground truth')
        ax.set_ylabel(f'p-val approx {i}')
        plt.savefig(sp + f'scatter_pval_{i}.png', dpi=300)
        plt.close('all')

    fig, ax = plt.subplots(dpi=300)

    cmap = plt.get_cmap('magma')

    for j, i in enumerate(list(range(20, 101, 20)) + list(range(125, 326, 25)) + [348, ]):
        with open(f'./results/tfs_not_clustered/clusterings/{i}_clusters_non_tfs.pkl', 'rb') as f:
            clustering = pickle.load(f)

        labels = [val for _, val in clustering.items()]

        from collections import Counter
        counts = sorted([val for _, val in Counter(labels).items()])

        ax.plot(list(range(len(counts))), counts, c=cmap(1/(j + 1)), label=f'n_{i}')

    ax.set_xlabel('cluster')
    ax.set_ylabel('n genes')
    plt.legend()
    plt.savefig(sp + 'cluster_sizes.png', dpi=300)
    plt.close('all')

    print('done')

    quit()

    # For GRNboost2 use: pip install dask-expr==0.5.3 distributed==2024.2.1

    import os

    generate_fdr_control_input = False
    cluster_metrics = False
    plot_clust_metrics = False
    fdr = False
    mwe = False
    approx_quality = False
    plot_approx_quality = False
    debug_example0_flag = False
    debug_example1_flag = False

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
    elif mwe:
        example_workflow()

    elif approx_quality:

        assess_approximation_quality(
        root_directory=os.path.join(os.getcwd(), "data"),
        num_clusters=[100, 200, 300],
        num_permutations=1000,
        filter_tissues=["Adipose_Tissue"])

    elif plot_approx_quality:
        import pandas as pd
        import matplotlib.pyplot as plt
        df = pd.read_csv(os.path.join(os.getcwd(), 'data/Adipose_Tissue/approx_results.csv'), index_col=0)

        fig, ax = plt.subplots(dpi=300)
        ax.plot([0, 1, 2], df.loc['mse_counts', :].to_numpy(), label='mse', color='r', marker='o')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('MSE')
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([100, 200, 300])
        plt.savefig(os.path.join(os.getcwd(), 'data/Adipose_Tissue/mse.png'), dpi=300)
        plt.close('all')

        fig, ax = plt.subplots(dpi=300)
        ax.plot([0, 1, 2], df.loc['accuracy_pvals005', :].to_numpy(), label='Accuracy 0.05', color='c', marker='x')
        ax.plot([0, 1, 2], df.loc['accuracy_pvals001', :].to_numpy(), label='Accuracy 0.01', color='y', marker='x')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Accuracy')
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([100, 200, 300])
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), 'data/Adipose_Tissue/acc.png'), dpi=300)
        plt.close('all')

        fig, ax = plt.subplots(dpi=300)
        ax.plot([0, 1, 2], df.loc['prec_pvals005', :].to_numpy(), label='Prec 0.05', color='g', marker='x')
        ax.plot([0, 1, 2], df.loc['rec_pvals005', :].to_numpy(), label='Rec 0.05', color='m', marker='x')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Prec and rec')
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([100, 200, 300])
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), 'data/Adipose_Tissue/prec_rec_005.png'), dpi=300)
        plt.close('all')

        fig, ax = plt.subplots(dpi=300)
        ax.plot([0, 1, 2], df.loc['prec_pvals001', :].to_numpy(), label='Prec 0.01', color='g', marker='x')
        ax.plot([0, 1, 2], df.loc['rec_pvals001', :].to_numpy(), label='Rec 0.01', color='m', marker='x')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Prec and rec')
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([100, 200, 300])
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), 'data/Adipose_Tissue/prec_rec_001.png'), dpi=300)
        plt.close('all')

    elif debug_example0_flag:
        debug_exampe0()

    elif debug_example1_flag:
        debug_exampe1()

    else:
        print("Running FDR comparison...")
        root_directory  = "/home/woody/iwbn/iwbn106h/gtex"
        num_clusters = list(range(100, 1001, 100))
        tissue_list = ['Liver']

        approximate_fdr_validation(
            root_directory=root_directory,
            num_clusters=num_clusters,
            tissue_list=tissue_list,
            include_tfs=True,
            num_permutations=1000,
            keep_tfs_singleton=True,
            scale_importances=False,
            verbosity=1,
            use_cluster_medoids=True
        )

    print("done")






