

def load_and_process_data():

    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='scanpy')

    import os
    import numpy as np
    import scanpy as sc
    import cellrank as cr
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.sparse as sp

    from scipy.stats import median_abs_deviation

    def is_outlier(data, metric: str, nmads: int):
        m = data.obs[metric]
        outlier = (m < np.median(m) - nmads * median_abs_deviation(m)) | (
                np.median(m) + nmads * median_abs_deviation(m) < m
        )
        return outlier


    # Load dataset from scanpy
    adata = sc.datasets.paul15()

    # Load dataset from cellrank
    bdata = cr.datasets.bone_marrow('./data/bone_marrow.h5ad')

    names = ['paul15', 'setty19']
    datas = [adata, bdata]
    res_ps = ['./results/paul15', './results/setty19']
    mito_prefixes = ['mt-', 'MT-']

    for name, xdata, res_p, mito_prefix in zip(names, datas, res_ps, mito_prefixes):

        print(f'# --- {name} --- #')

        os.makedirs(res_p, exist_ok=True)

        x = xdata.X
        if sp.issparse(x):
            xdata.X = x.toarray()

        # --- Quality control
        xdata.layers['counts'] = xdata.X.copy()

        xdata.var['mt'] = xdata.var_names.str.startswith(mito_prefix)
        sc.pp.calculate_qc_metrics(xdata, qc_vars=['mt', ], inplace=True, percent_top=[20], log1p=True)

        facetgrid = sns.displot(xdata.obs['total_counts'], bins=100, kde=False)
        facetgrid.savefig(os.path.join(res_p, 'total_counts.png'))
        plt.close(facetgrid.fig)
        fig, ax = plt.subplots(dpi=300)
        sc.pl.violin(xdata, 'pct_counts_mt', show=False, ax=ax)
        fig.savefig(os.path.join(res_p, 'pct_counts_mt.png'))
        plt.close(fig)
        fig, ax = plt.subplots(dpi=300)
        sc.pl.scatter(xdata, 'total_counts', 'n_genes_by_counts', color='pct_counts_mt', show=False, ax=ax)
        plt.savefig(os.path.join(res_p, 'total_counts_scatter.png'))
        plt.close(fig)

        xdata.obs['outlier'] = (
                is_outlier(xdata, 'log1p_total_counts', 5)
                | is_outlier(xdata, 'log1p_n_genes_by_counts', 5)
                | is_outlier(xdata, 'pct_counts_in_top_20_genes', 5)
        )
        print(f'Number of count-based outliers:\n{xdata.obs.outlier.value_counts()}')

        xdata.obs['mt_outlier'] = is_outlier(xdata, 'pct_counts_mt', 3) | (
                xdata.obs['pct_counts_mt'] > 8
        )
        print(f'Number of count mitochondrial count-based outliers:\n{xdata.obs.outlier.value_counts()}')

        print(f'Total number of cells before filtering: {xdata.n_obs}')
        xdata = xdata[(~xdata.obs.outlier) & (~xdata.obs.mt_outlier)].copy()

        print(f'Number of cells after filtering of low quality cells: {xdata.n_obs}')

        fig, ax = plt.subplots(dpi=300)
        sc.pl.scatter(xdata, 'total_counts', 'n_genes_by_counts', color='pct_counts_mt', show=False, ax=ax)
        plt.savefig(os.path.join(res_p, 'total_counts_scatter_after_filtering.png'))
        plt.close(fig)

        print(f'Total number of genes before filtering: {xdata.n_vars}')
        sc.pp.filter_genes(xdata, min_cells=20)
        print(f'Number of genes after cell filter: {xdata.n_vars}')

        # --- Normalization
        sc.pp.normalize_total(xdata)
        sc.pp.log1p(xdata)

        xdata.write(os.path.join(res_p, f'{name}_preprocessed.h5ad'))


def input_grn_inference():

    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='scanpy')

    import os
    import scanpy as sc

    # Add our arboreto version to path
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from arboreto.utils import load_tf_names
    from arboreto.algo import grnboost2


    names = ['paul15', 'setty19']
    cluster_keys = ['paul15_clusters', 'clusters']
    tf_files = ['./data/allTFs_mm.txt', './data/allTFs_hg38.txt']
    res_ps = ['./results/paul15', './results/setty19']

    for name, cluster_key, tf_file, res_p in zip(names, cluster_keys, tf_files, res_ps):

        print(f'# --- {name} --- #')

        # Load preprocessed data
        adata = sc.read(os.path.join(res_p, f'{name}_preprocessed.h5ad'))

        # Load TF names
        tf_names = load_tf_names(tf_file)

        # Check intersection of TF names and gene names
        gene_names = set(adata.var_names)
        tf_names_set = set(tf_names)
        intersection = tf_names_set & gene_names
        print((
            f'# Number of Genes: {len(gene_names)}, '
            f'Number of TFs: {len(tf_names_set)}, '
            f'Number of genes in intersection: {len(intersection)}'
        ))

        print(f'# Cluster sizes:\n{adata.obs[cluster_key].value_counts()}')
        clusters = list(set(adata.obs[cluster_key]))
        for cluster in clusters:

            # Subset to cell-type cluster
            adata_sub = adata[adata.obs[cluster_key] == cluster].copy()

            # Scale to unit variance
            sc.pp.scale(adata_sub)

            grn = grnboost2(expression_data=adata_sub.to_df(), tf_names=tf_names, verbose=True)

            grn.to_csv(os.path.join(res_p, f'{cluster}_grn.csv'))


def p_value_calculation():
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='scanpy')

    import os
    import pandas as pd
    import scanpy as sc

    # Add our arboreto version to path
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from arboreto.algo import grnboost2_fdr

    names = ['paul15', 'setty19']
    cluster_keys = ['paul15_clusters', 'clusters']
    tf_files = ['./data/allTFs_mm.txt', './data/allTFs_hg38.txt']
    res_ps = ['./results/paul15', './results/setty19']

    for name, cluster_key, tf_file, res_p in zip(names, cluster_keys, tf_files, res_ps):

        print(f'# --- {name} --- #')

        # Load preprocessed data
        adata = sc.read(os.path.join(res_p, f'{name}_preprocessed.h5ad'))

        print(f'# Cluster sizes:\n{adata.obs[cluster_key].value_counts()}')
        clusters = list(set(adata.obs[cluster_key]))
        for cluster in clusters:

            # Subset to cell-type cluster and scale to unit variance
            adata_sub = adata[adata.obs[cluster_key] == cluster].copy()
            sc.pp.scale(adata_sub)

            # Load input GRN
            grn = pd.read_csv(os.path.join(res_p, f'{cluster}_grn.csv'), index_col=0)

            # Compute empirical P-values
            grn = grnboost2_fdr(
                expression_data=adata_sub.to_df(),
                cluster_representative_mode='random',
                num_target_clusters=100,
                num_tf_clusters=-1,
                target_cluster_mode='wasserstein',
                input_grn=grn,
                seed=42,
                verbose=True,
                num_permutations=1000,
                scale_for_tf_sampling=True,
                inference_mode='grnboost2',
            )

            grn.to_csv(os.path.join(res_p, f'{cluster}_grn_w_pvals.csv'))


def compute_grn_similarity_and_cluster():

    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='scanpy')

    import os
    import itertools
    import pandas as pd
    import scanpy as sc

    names = ['paul15', 'setty19']
    res_ps = ['./results/paul15', './results/setty19']

    name_to_cluster_names = {
        'paul15': [
            '1Ery', '2Ery', '3Ery', '4Ery', '5Ery', '6Ery', '7MEP', '8Mk', '9GMP', '10GMP',
            '11DC', '12Baso', '13Baso', '14Mo', '15Mo', '16Neu', '17Neu', '18Eos', '19Lymph'
        ],
        'setty19': ['CLP', 'DCs', 'Ery_1', 'Ery_2', 'HSC_1', 'HSC_2', 'Mega', 'Mono_1', 'Mono_2', 'Precursors'],
    }

    fdr_threshold = 0.05

    def compute_jaccard_similarity_matrix(cluster_name_to_edge_set: dict) -> pd.DataFrame:
        """
        Compute Jaccard similarity matrix from a dict mapping cluster name -> set of edges.
        Each edge set is a set of (TF, target) tuples.
        """
        clusters = list(cluster_name_to_edge_set.keys())
        sim_mat = pd.DataFrame(0.0, index=clusters, columns=clusters)

        for cl1, cl2 in itertools.combinations_with_replacement(clusters, 2):
            edges1 = cluster_name_to_edge_set.get(cl1, set())
            edges2 = cluster_name_to_edge_set.get(cl2, set())

            if len(edges1) == 0 and len(edges2) == 0:
                sim = 1.0
            elif len(edges1) == 0 or len(edges2) == 0:
                sim = 0.0
            else:
                inter = len(edges1 & edges2)
                union = len(edges1 | edges2)
                sim = inter / union if union > 0 else 0.0

            sim_mat.loc[cl1, cl2] = sim
            sim_mat.loc[cl2, cl1] = sim

        return sim_mat

    for name, res_p in zip(names, res_ps):

        clusters_names = name_to_cluster_names[name]

        cluster_name_to_edge_set_base = dict()
        cluster_name_to_edge_set_fdr = dict()
        for cluster_name in clusters_names:

            # Load the GRN
            grn = pd.read_csv(os.path.join(res_p, f'{cluster_name}_grn_w_pvals.csv'), index_col=0)

            # Scale P-values
            max_occurence = max(grn['shuffled_occurences'])
            grn['pvalue_scaled'] = grn['pvalue'] * (max_occurence + 1.0) / (grn['shuffled_occurences'] + 1.0)

            # Threshold and prune edges
            grn_fdr = grn[grn['pvalue_scaled'] <= fdr_threshold].copy()

            # Todo: benjamini-hochberg

            # Extract edge sets
            cluster_name_to_edge_set_base[cluster_name] = set(zip(grn['TF'], grn['target']))
            cluster_name_to_edge_set_fdr[cluster_name] = set(zip(grn_fdr['TF'], grn_fdr['target']))

        sim_mat_base = compute_jaccard_similarity_matrix(cluster_name_to_edge_set_base)
        sim_mat_fdr = compute_jaccard_similarity_matrix(cluster_name_to_edge_set_fdr)

        print(f'# --- {name} --- #')
        print(f'# Sim mat base:\n{sim_mat_base}')
        print(f'# Sim mat fdr:\n{sim_mat_fdr}')


if __name__ == '__main__':

    # load_and_process_data()

    # input_grn_inference()

    # p_value_calculation()

    compute_grn_similarity_and_cluster()

    print('done')


