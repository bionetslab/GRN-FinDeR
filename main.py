
def debug_arboreto_fdr_random_tfs_clustered():
    import numpy as np
    import pandas as pd
    from arboreto_fdr.algo import grnboost2_fdr# Import from arboreto_fdr !!!
    from arboreto.algo import grnboost2

    from src.distance_matrix import compute_wasserstein_distance_matrix
    from src.clustering import cluster_genes_to_dict
    import random

    n_tfs = 10
    n_non_tfs = 10
    n_cells = 10
    tfs = [f'TF{i}' for i in range(n_tfs)]
    non_tfs = [f'Gene{i}' for i in range(n_non_tfs)]
    # Construct dummy example

    np.random.seed(42)

    expr_mat = pd.DataFrame(
        # np.random.normal(0, 1, (n_cells, n_tfs + n_genes)),
        np.random.poisson(lam=np.random.gamma(shape=2, scale=1, size=(n_cells, n_tfs + n_non_tfs))),
        columns=tfs + non_tfs,
    )

    input_grn_df = grnboost2(expression_data=expr_mat, tf_names=tfs, verbose=True, seed=777)
    # print(input_grn_df)
    # Transform input GRN to dict-format.
    input_grn = dict()
    for tf, target, importance in zip(input_grn_df['TF'], input_grn_df['target'], input_grn_df['importance']):
        input_grn[(tf, target)] = {'importance' : importance}

    dist_mat_all = compute_wasserstein_distance_matrix(expr_mat, -1)
    # print(dist_mat_all)

    tf_bool = [True if gene in tfs else False for gene in dist_mat_all.columns]
    non_tf_bool = [False if gene in tfs else True for gene in dist_mat_all.columns]
    dist_mat_non_tfs = dist_mat_all.loc[non_tf_bool, non_tf_bool]
    dist_mat_tfs = dist_mat_all.loc[non_tf_bool, tf_bool]

    NUM_CLUSTERS=3
    non_tf_to_clust = cluster_genes_to_dict(dist_mat_non_tfs, num_clusters=NUM_CLUSTERS)
    tf_to_clust = cluster_genes_to_dict(dist_mat_tfs, num_clusters=NUM_CLUSTERS)

    def merge_gene_clusterings(clustering1 : dict, clustering2 : dict):
        num_clusters1 = max({clusterID for _, clusterID in clustering1.items()})+1
        updated_clustering2 = {gene : clusterID+num_clusters1 for gene, clusterID in clustering2.items()}
        return clustering1 | updated_clustering2

    gene_to_clust = merge_gene_clusterings(non_tf_to_clust, tf_to_clust)

    # Compute medoids for each target cluster.
    non_tf_representatives = non_tfs
    tf_representatives = tfs

    are_tfs_clustered = True
    fdr_mode = 'random'

    corrected_grn = grnboost2_fdr(expr_mat,
                                  are_tfs_clustered,
                                  tf_representatives,
                                  non_tf_representatives,
                                  gene_to_clust,
                                  input_grn,
                                  fdr_mode)

    print(corrected_grn)

def debug_arboreto_fdr_random_tfs_unclustered():
    import numpy as np
    import pandas as pd
    from arboreto_fdr.algo import grnboost2_fdr# Import from arboreto_fdr !!!
    from arboreto.algo import grnboost2

    from src.distance_matrix import compute_wasserstein_distance_matrix
    from src.clustering import cluster_genes_to_dict
    import random

    n_tfs = 10
    n_non_tfs = 10
    n_cells = 10
    tfs = [f'TF{i}' for i in range(n_tfs)]
    non_tfs = [f'Gene{i}' for i in range(n_non_tfs)]
    # Construct dummy example

    np.random.seed(42)

    expr_mat = pd.DataFrame(
        # np.random.normal(0, 1, (n_cells, n_tfs + n_genes)),
        np.random.poisson(lam=np.random.gamma(shape=2, scale=1, size=(n_cells, n_tfs + n_non_tfs))),
        columns=tfs + non_tfs,
    )

    input_grn_df = grnboost2(expression_data=expr_mat, tf_names=tfs, verbose=True, seed=777)
    # print(input_grn_df)
    # Transform input GRN to dict-format.
    input_grn = dict()
    for tf, target, importance in zip(input_grn_df['TF'], input_grn_df['target'], input_grn_df['importance']):
        input_grn[(tf, target)] = {'importance' : importance}

    dist_mat_all = compute_wasserstein_distance_matrix(expr_mat, -1)
    # print(dist_mat_all)

    tf_bool = [True if gene in tfs else False for gene in dist_mat_all.columns]
    non_tf_bool = [False if gene in tfs else True for gene in dist_mat_all.columns]
    dist_mat_non_tfs = dist_mat_all.loc[non_tf_bool, non_tf_bool]

    NUM_CLUSTERS=3
    non_tf_to_clust = cluster_genes_to_dict(dist_mat_non_tfs, num_clusters=NUM_CLUSTERS)
    # Create dummy clustering for TFs, since they are not being clustered.
    tf_to_clust = {tf : index+NUM_CLUSTERS for index, tf in enumerate(tfs)}
    gene_to_clust = non_tf_to_clust | tf_to_clust

    # Compute medoids for each target cluster.
    tf_representatives = tfs
    non_tf_representatives = non_tfs

    are_tfs_clustered = False
    fdr_mode = 'random'

    corrected_grn = grnboost2_fdr(expr_mat,
                                  are_tfs_clustered,
                                  tf_representatives,
                                  non_tf_representatives,
                                  gene_to_clust,
                                  input_grn,
                                  fdr_mode)

    print(corrected_grn)

def debug_arboreto_fdr_medoids_tfs_clustered():
    import numpy as np
    import pandas as pd
    from arboreto_fdr.algo import grnboost2_fdr# Import from arboreto_fdr !!!
    from arboreto.algo import grnboost2

    from src.distance_matrix import compute_wasserstein_distance_matrix
    from src.clustering import cluster_genes_to_dict
    import random

    n_tfs = 10
    n_non_tfs = 10
    n_cells = 10
    tfs = [f'TF{i}' for i in range(n_tfs)]
    non_tfs = [f'Gene{i}' for i in range(n_non_tfs)]
    # Construct dummy example

    np.random.seed(42)

    expr_mat = pd.DataFrame(
        # np.random.normal(0, 1, (n_cells, n_tfs + n_genes)),
        np.random.poisson(lam=np.random.gamma(shape=2, scale=1, size=(n_cells, n_tfs + n_non_tfs))),
        columns=tfs + non_tfs,
    )

    input_grn_df = grnboost2(expression_data=expr_mat, tf_names=tfs, verbose=True, seed=777)
    # print(input_grn_df)
    # Transform input GRN to dict-format.
    input_grn = dict()
    for tf, target, importance in zip(input_grn_df['TF'], input_grn_df['target'], input_grn_df['importance']):
        input_grn[(tf, target)] = {'importance' : importance}

    dist_mat_all = compute_wasserstein_distance_matrix(expr_mat, -1)
    # print(dist_mat_all)

    tf_bool = [True if gene in tfs else False for gene in dist_mat_all.columns]
    non_tf_bool = [False if gene in tfs else True for gene in dist_mat_all.columns]
    dist_mat_non_tfs = dist_mat_all.loc[non_tf_bool, non_tf_bool]
    dist_mat_tfs = dist_mat_all.loc[non_tf_bool, tf_bool]

    NUM_CLUSTERS=3
    non_tf_to_clust = cluster_genes_to_dict(dist_mat_non_tfs, num_clusters=NUM_CLUSTERS)
    tf_to_clust = cluster_genes_to_dict(dist_mat_tfs, num_clusters=NUM_CLUSTERS)

    def merge_gene_clusterings(clustering1 : dict, clustering2 : dict):
        num_clusters1 = max({clusterID for _, clusterID in clustering1.items()})+1
        updated_clustering2 = {gene : clusterID+num_clusters1 for gene, clusterID in clustering2.items()}
        return clustering1 | updated_clustering2

    gene_to_clust = merge_gene_clusterings(non_tf_to_clust, tf_to_clust)

    # Compute medoids for each target cluster.
    tf_representatives_dict = dict()
    non_tf_representatives_dict = dict()
    for non_tf, cluster in non_tf_to_clust.items():
        if not cluster in non_tf_representatives_dict:
            non_tf_representatives_dict[cluster] = non_tf
    non_tf_representatives = [non_tf for _, non_tf in non_tf_representatives_dict.items()]
    for tf, cluster in tf_to_clust.items():
        if not cluster in tf_representatives_dict:
            tf_representatives_dict[cluster] = tf
    tf_representatives = [tf for _, tf in tf_representatives_dict.items()]

    are_tfs_clustered = True
    fdr_mode = 'medoid'

    corrected_grn = grnboost2_fdr(expr_mat,
                                  are_tfs_clustered,
                                  tf_representatives,
                                  non_tf_representatives,
                                  gene_to_clust,
                                  input_grn,
                                  fdr_mode)

    print(corrected_grn)

def debug_arboreto_fdr_medoids_tfs_unclustered():
    import numpy as np
    import pandas as pd
    from arboreto_fdr.algo import grnboost2_fdr# Import from arboreto_fdr !!!
    from arboreto.algo import grnboost2

    from src.distance_matrix import compute_wasserstein_distance_matrix
    from src.clustering import cluster_genes_to_dict
    import random

    n_tfs = 10
    n_non_tfs = 10
    n_cells = 10
    tfs = [f'TF{i}' for i in range(n_tfs)]
    non_tfs = [f'Gene{i}' for i in range(n_non_tfs)]
    # Construct dummy example

    np.random.seed(42)

    expr_mat = pd.DataFrame(
        # np.random.normal(0, 1, (n_cells, n_tfs + n_genes)),
        np.random.poisson(lam=np.random.gamma(shape=2, scale=1, size=(n_cells, n_tfs + n_non_tfs))),
        columns=tfs + non_tfs,
    )

    input_grn_df = grnboost2(expression_data=expr_mat, tf_names=tfs, verbose=True, seed=777)
    # print(input_grn_df)
    # Transform input GRN to dict-format.
    input_grn = dict()
    for tf, target, importance in zip(input_grn_df['TF'], input_grn_df['target'], input_grn_df['importance']):
        input_grn[(tf, target)] = {'importance' : importance}

    dist_mat_all = compute_wasserstein_distance_matrix(expr_mat, -1)
    # print(dist_mat_all)

    tf_bool = [True if gene in tfs else False for gene in dist_mat_all.columns]
    non_tf_bool = [False if gene in tfs else True for gene in dist_mat_all.columns]
    dist_mat_non_tfs = dist_mat_all.loc[non_tf_bool, non_tf_bool]

    NUM_CLUSTERS=2
    non_tf_to_clust = cluster_genes_to_dict(dist_mat_non_tfs, num_clusters=NUM_CLUSTERS)
    # Create dummy clustering for TFs, since they are not being clustered.
    tf_to_clust = {tf : index+NUM_CLUSTERS for index, tf in enumerate(tfs)}
    gene_to_clust = non_tf_to_clust | tf_to_clust

    # Compute medoids for each target cluster.
    tf_representatives = tfs
    non_tf_representatives_dict = dict()
    for non_tf, cluster in non_tf_to_clust.items():
        if not cluster in non_tf_representatives_dict:
            non_tf_representatives_dict[cluster] = non_tf
    non_tf_representatives = [non_tf for _, non_tf in non_tf_representatives_dict.items()]

    are_tfs_clustered = False
    fdr_mode = 'medoid'

    corrected_grn = grnboost2_fdr(expr_mat,
                                  are_tfs_clustered,
                                  tf_representatives,
                                  non_tf_representatives,
                                  gene_to_clust,
                                  input_grn,
                                  fdr_mode)

    print(corrected_grn)

def main_mwe():

    import numpy as np
    import pandas as pd
    from arboreto.algo import grnboost2

    from src.distance_matrix import compute_wasserstein_distance_matrix
    from src.clustering import cluster_genes_to_dict
    from src.fdr_calculation import approximate_fdr

    n_tfs = 10
    n_non_tfs = 10
    n_cells = 10
    tfs = [f'TF{i}' for i in range(n_tfs)]
    non_tfs = [f'Gene{i}' for i in range(n_non_tfs)]
    # Construct dummy example

    np.random.seed(42)

    expr_mat = pd.DataFrame(
        # np.random.normal(0, 1, (n_cells, n_tfs + n_genes)),
        np.random.poisson(lam=np.random.gamma(shape=2, scale=1, size=(n_cells, n_tfs + n_non_tfs))),
        columns=tfs + non_tfs,
    )
    # print(expr_matrix)

    input_grn = grnboost2(expression_data=expr_mat, tf_names=tfs, verbose=True, seed=777)
    # print(grn)

    dist_mat_all = compute_wasserstein_distance_matrix(expr_mat, -1)
    # print(dist_mat_all)

    tf_bool = [True if gene in tfs else False for gene in dist_mat_all.columns]
    non_tf_bool = [False if gene in tfs else True for gene in dist_mat_all.columns]
    dist_mat_non_tfs = dist_mat_all.loc[non_tf_bool, non_tf_bool]

    non_tf_to_clust = cluster_genes_to_dict(dist_mat_non_tfs, num_clusters=3)
    # print(gene_to_clust)

    grn_w_pvals = approximate_fdr(
        expression_mat=expr_mat,
        grn=input_grn,
        gene_to_cluster=non_tf_to_clust,
        num_permutations=2
    )

    print(grn_w_pvals)




if __name__ == '__main__':

    debug_arboreto_fdr_random_tfs_clustered()
    # main_mwe()

    print('done')



