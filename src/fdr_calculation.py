
import time
import pandas as pd
import numpy as np
import gc
from dask.distributed import Client, LocalCluster
from statsmodels.stats.multitest import multipletests
from arboreto.algo import grnboost2
from arboreto.utils import load_tf_names
from typing import Union, List

from src.preprocessing import preprocess_data
import random
import itertools


def approximate_fdr(
        expression_mat : pd.DataFrame,
        grn : pd.DataFrame,
        gene_to_cluster : Union[dict, tuple[dict, dict]],
        num_permutations : int = 1000,
        grnboost2_random_seed: Union[int, None] = None
) -> pd.DataFrame:

    if isinstance(gene_to_cluster, dict):
        fdr_grn = _approximate_fdr_no_tfs(
            expression_mat=expression_mat,
            grn=grn,
            gene_to_cluster=gene_to_cluster,
            num_permutations=num_permutations,
            grnboost2_random_seed=grnboost2_random_seed,
        )
    else:
        fdr_grn = _approximate_fdr_with_tfs(
            expression_mat=expression_mat,
            grn=grn,
            tf_to_cluster=gene_to_cluster[0],
            gene_to_cluster=gene_to_cluster[1],
            num_permutations=num_permutations,
            grnboost2_random_seed=grnboost2_random_seed,
        )
    return fdr_grn


def _approximate_fdr_no_tfs(
        expression_mat: pd.DataFrame,
        grn: pd.DataFrame,
        gene_to_cluster: dict,
        num_permutations: int = 1000,
        grnboost2_random_seed: Union[int, None] = None,
) -> pd.DataFrame:

    # Invert gene to cluster dictionary.
    cluster_to_gene = _invert_gene_cluster_dictionary(gene_to_cluster)

    # Compute set of singleton genes, i.e. genes that are in one-element-cluster.
    singleton_genes = {genes[0] for _, genes in cluster_to_gene.items() if len(genes) == 1}

    # Create dict from original GRN {('TF', 'target'): ('importance', 'counter')}
    grn_zipped = zip(grn['TF'].to_list(), grn['target'].to_list(), grn['importance'].to_list())
    grn_dict = {(tf, target): (importance, 0.0, 0.0) for tf, target, importance in grn_zipped}

    # Init data structure for counting between and withing cluster empirical counts.
    cluster_cluster_counts = {cluster_edge: 0.0 for cluster_edge in itertools.product(cluster_to_gene.keys(), repeat=2)}

    for i in range(num_permutations):
        # Sample representatives from each cluster.
        representatives = _draw_representatives(cluster_to_gene, 2)
        represent_expression = expression_mat[representatives].copy()
        # Shuffle expression matrix.
        represent_permuted = represent_expression.sample(frac=1, axis=1)
        shuffled_grn = grnboost2(
            expression_data=represent_permuted,
            tf_names=represent_permuted.columns.to_list(),
            seed=grnboost2_random_seed,
        )
        # Compute adjusted count values for each shuffled edge.
        for tf, target, factor in zip(shuffled_grn['TF'], shuffled_grn['target'], shuffled_grn['importance']):
            if (tf, target) in grn_dict:
                count_value = int(factor >= grn_dict[(tf, target)][0])

                tf_is_singleton = (tf in singleton_genes)
                target_is_singleton = (target in singleton_genes)

                if tf_is_singleton and target_is_singleton:
                    # Two singleton clusters, keep count value as is.
                    count_value = 1.0 * count_value
                elif tf_is_singleton or target_is_singleton:
                    # One singleton cluster, two possible edges.
                    count_value /= 2.0
                elif gene_to_cluster[tf] == gene_to_cluster[target]:
                    # No singleton clusters; intra-cluster edge can occur twice.
                    count_value /= 2.0
                else:
                    # No singleton clusters; inter-cluster edge can occur four times.
                    count_value /= 4.0
                # Update corresponding cluster counts.
                cluster_cluster_counts[(gene_to_cluster[tf], gene_to_cluster[target])] += count_value

    # Write cluster-cluster Pvalues into original genes.
    for key in grn_dict.keys():
        cluster_tuple = (gene_to_cluster[key[0]], gene_to_cluster[key[1]])
        p_value = (cluster_cluster_counts[cluster_tuple] + 1) / (num_permutations + 1)
        grn_dict[key] = (grn_dict[key][0], cluster_cluster_counts[cluster_tuple], p_value)

    grn_transposed = {'tf': [], 'target': [], 'importance': [], 'count': [], 'pvalue': []}
    for key, val in grn_dict.items():
        grn_transposed['tf'].append(key[0])
        grn_transposed['target'].append(key[1])
        grn_transposed['importance'].append(val[0])
        grn_transposed['count'].append(val[1])
        grn_transposed['pvalue'].append(val[2])

    grn_df = pd.DataFrame.from_dict(grn_transposed)
    return grn_df


def _approximate_fdr_with_tfs(
        expression_mat: pd.DataFrame,
        grn: pd.DataFrame,
        tf_to_cluster: dict,
        gene_to_cluster: dict,
        num_permutations: int = 1000,
        grnboost2_random_seed: Union[int, None] = None
) -> pd.DataFrame:
    target_to_cluster = _merge_clusterings(tf_to_cluster, gene_to_cluster)
    cluster_to_target = _invert_gene_cluster_dictionary(target_to_cluster)
    cluster_to_gene = _invert_gene_cluster_dictionary(gene_to_cluster)
    cluster_to_tf = _invert_gene_cluster_dictionary(tf_to_cluster)

    # Create dict representation from CSV input.
    grn_zipped = zip(grn['TF'].to_list(), grn['target'].to_list(), grn['importance'].to_list())
    grn_dict = {(tf, target): (importance, 0.0, 0.0) for tf, target, importance in grn_zipped}

    # Edge count structure on cluster level with structure {(TFclusterID, TargetclusterID) : count}.
    cluster_cluster_counts = {
        cluster_edge: 0.0 for cluster_edge in itertools.product(
            list(cluster_to_tf.keys()),
            list(cluster_to_target.keys())
        )
    }

    for i in range(num_permutations):
        # Sample representatives from each cluster.
        tf_representatives = _draw_representatives(cluster_to_tf, 1)
        gene_representatives = _draw_representatives(cluster_to_gene, 1)

        joint_representatives = list(set(tf_representatives + gene_representatives))
        represent_expression = expression_mat[joint_representatives].copy()
        # Shuffle expression matrix.
        represent_permuted = represent_expression.sample(frac=1, axis=1)
        shuffled_grn = grnboost2(
            expression_data=represent_permuted,
            tf_names=tf_representatives,
            seed=grnboost2_random_seed,
        )

        # Compute adjusted count values for each shuffled edge.
        for tf, target, factor in zip(shuffled_grn['TF'], shuffled_grn['target'], shuffled_grn['importance']):
            if (tf, target) in grn_dict:
                count_value = int(factor >= grn_dict[(tf, target)][0])
                # Update corresponding cluster counts.
                cluster_cluster_counts[(tf_to_cluster[tf], target_to_cluster[target])] += count_value

    # Map cluster-cluster Pvalues into original genes.
    for key in grn_dict.keys():
        cluster_tuple = (tf_to_cluster[key[0]], target_to_cluster[key[1]])
        p_value = (cluster_cluster_counts[cluster_tuple] + 1) / (num_permutations + 1)
        grn_dict[key] = (grn_dict[key][0], cluster_cluster_counts[cluster_tuple], p_value)

    grn_transposed = {'tf': [], 'target': [], 'importance': [], 'count': [], 'pvalue': []}
    for key, val in grn_dict.items():
        grn_transposed['tf'].append(key[0])
        grn_transposed['target'].append(key[1])
        grn_transposed['importance'].append(val[0])
        grn_transposed['count'].append(val[1])
        grn_transposed['pvalue'].append(val[2])

    grn_df = pd.DataFrame.from_dict(grn_transposed)
    return grn_df


def _invert_gene_cluster_dictionary(gene_to_cluster : dict):
    cluster_to_gene = dict()
    for key, val in gene_to_cluster.items():
        if val in cluster_to_gene:
            cluster_to_gene[val].append(key)
        else:
            cluster_to_gene[val] = [key]
    return cluster_to_gene


def _merge_clusterings(tf_to_cluster : dict, gene_to_cluster : dict):
    tf_to_cluster = tf_to_cluster.copy()
    num_tf_categories = max(tf_to_cluster.values())+1
    gene_to_cluster_updated = {gene : cluster+num_tf_categories for gene, cluster in gene_to_cluster.items()}
    tf_to_cluster.update(gene_to_cluster_updated)
    return tf_to_cluster


def _draw_representatives(cluster_to_gene : dict, num_representatives : int = 2):
    representatives = [random.sample(val, min(len(val), num_representatives)) for val in cluster_to_gene.values()]
    representatives = list(itertools.chain.from_iterable(representatives))
    return representatives


def classical_fdr(
        expression_mat: pd.DataFrame,
        grn: pd.DataFrame,
        tf_names: Union[List[str], None] = None,
        num_permutations: int = 1000,
        grnboost2_random_seed: Union[int, None] = None,
        verbosity: int = 0,
) -> pd.DataFrame:

    # Create dict representation from CSV input.
    grn_zipped = zip(grn['TF'].to_list(), grn['target'].to_list(), grn['importance'].to_list())
    grn_dict = {
        (tf, target): {'importance': importance, 'count': 0, 'p_value': 0.0} for tf, target, importance in grn_zipped
    }

    # Iterate for num_permutations, compute GRN on input matrix with shuffled genes
    for i in range(num_permutations):

        st = time.time()

        if verbosity >= 1:
            print(f'# ### Iteration {i} ...')

        # Shuffle expression matrix
        shuffled_expression_mat = expression_mat.sample(frac=1, axis=1, random_state=i)

        # Compute GRN with shuffled expression matrix
        shuffled_grn = grnboost2(
            expression_data=shuffled_expression_mat,
            tf_names=tf_names if tf_names is not None else 'all',
            seed=grnboost2_random_seed,
            verbose=False,
        )

        # Compute adjusted count values for each shuffled edge.
        shuffled_grn_zipped = zip(shuffled_grn['TF'], shuffled_grn['target'], shuffled_grn['importance'])
        for tf_shuffled, target_shuffled, importance_shuffled in shuffled_grn_zipped:
            if (tf_shuffled, target_shuffled) in grn_dict:
                grn_dict[(tf_shuffled, target_shuffled)]['count'] += int(
                    importance_shuffled >= grn_dict[(tf_shuffled, target_shuffled)]['importance']
                )

        et = time.time()

        if verbosity >= 1:
            print(f'# took {et-st:.2f} seconds')

    # Compute the p-values
    st = time.time()
    if verbosity >= 1:
        print('# ### Computing FDR p-values ...')

    for key, val in grn_dict.items():
        grn_dict[key]['p_value'] = (grn_dict[key]['count'] + 1) / (num_permutations + 1)

    et = time.time()
    if verbosity >= 1:
        print(f'# took {et-st:.2f} seconds')

    # Turn dict into dataframe
    grn_df = pd.DataFrame([
        {'TF': tf, 'target': target, **values} for (tf, target), values in grn_dict.items()
    ])

    return grn_df


def classical_fdr_wout_merge(expression_matrix: pd.DataFrame,
                  tf_names_file: str,
                  grn: pd.DataFrame,
                  output_dir: str,
                  num_permutations=1000) -> pd.DataFrame:


    # Set up a Dask local cluster for parallel computing
    local_cluster = LocalCluster(n_workers=60, threads_per_worker=16, memory_limit='20GB')
    custom_client = Client(local_cluster)

    # Preprocess the expression matrix
    filtered_matrix_vst = preprocess_data(expression_matrix=expression_matrix)

    # Load transcription factor (TF) names
    tf_names = load_tf_names(tf_names_file)

    # Create dict from original GRN {('TF', 'target'): ('importance', 'counter')}
    grn_zipped = zip(grn['TF'].to_list(), grn['target'].to_list(), grn['importance'].to_list())
    grn_dict = {(tf, target): (importance, 0) for tf, target, importance in grn_zipped}

    # Permutation test
    for i in range(num_permutations):
        shuffled_matrix = filtered_matrix_vst.apply(np.random.permutation, axis=0)
        shuffled_grn = grnboost2(expression_data=shuffled_matrix, tf_names=tf_names, client_or_address=custom_client,
                                 verbose=False, seed=777)

        for tf, target, importance in zip(
                shuffled_grn['TF'].tolist(), shuffled_grn['target'].tolist(), shuffled_grn['importance'].to_list()):
            if (tf, target) in grn_dict:
                grn_dict[(tf, target)][1] += int(importance >= grn_dict[(tf, target)][0])

        del shuffled_matrix, shuffled_grn
        gc.collect()

    # Calculate p-values
    grn['p_value'] = grn.apply(
        lambda row: (grn_dict[(row['TF'], row['target'])][1] + 1) / (num_permutations + 1),
        axis=1
    )

    # Apply FDR correction
    p_values = grn['p_value'].to_numpy()
    grn['fdr'] = multipletests(p_values, method='fdr_bh')[1]

    # Save the results
    output_file_path = f"{output_dir}/final_grn_with_pvalues.tsv"
    grn.to_csv(output_file_path, sep='\t', index=False)

    # Close the Dask client and local cluster
    custom_client.close()
    local_cluster.close()

    # Return the final GRN with p-values
    return grn

