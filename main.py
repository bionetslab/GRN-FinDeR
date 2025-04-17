

def main_check_arboreto_fdr():
    import numpy as np
    import pandas as pd
    from arboreto_fdr.algo import grnboost2  # Import from arboreto_fdr !!!

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

    grn = grnboost2(expression_data=expr_mat, tf_names=tfs, verbose=True, seed=777)
    print(grn)


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

    main_check_arboreto_fdr()

    # main_mwe()

    print('done')



