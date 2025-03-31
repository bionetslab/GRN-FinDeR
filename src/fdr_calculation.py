
import time
import pandas as pd
import numpy as np
from arboreto.algo import grnboost2
from typing import Union, List
from numba import njit, prange

import random
import itertools


def approximate_fdr(
        expression_mat : pd.DataFrame,
        grn : pd.DataFrame,
        gene_to_cluster : Union[dict, tuple[dict, dict]],
        num_permutations : int = 1000,
        grnboost2_random_seed: Union[int, None] = None,
        scale_importances : bool = False
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
        if scale_importances:
            fdr_grn = _approximate_fdr_with_tfs_with_scaling(
                expression_mat=expression_mat,
                grn=grn,
                tf_to_cluster=gene_to_cluster[0],
                gene_to_cluster=gene_to_cluster[1],
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
        represent_permuted = _shuffle_column_wise(df=represent_expression)
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
        represent_permuted = _shuffle_column_wise(df=represent_expression)
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

def _approximate_fdr_with_tfs_with_scaling(
        expression_mat: pd.DataFrame,
        grn: pd.DataFrame,
        tf_to_cluster: dict,
        gene_to_cluster: dict,
        num_permutations: int = 1000,
        grnboost2_random_seed: Union[int, None] = None
) -> pd.DataFrame:
    print("Using approx. FDR with edge scaling...")
    cluster_to_gene = _invert_gene_cluster_dictionary(gene_to_cluster)
    cluster_to_tf = _invert_gene_cluster_dictionary(tf_to_cluster)

    # Create dict representation from CSV input.
    grn_zipped = zip(grn['TF'].to_list(), grn['target'].to_list(), grn['importance'].to_list())
    grn_dict = {(tf, target): [importance, 0.0, 0.0] for tf, target, importance in grn_zipped}

    # Compute sum of per-target incoming feature importances for scaling shuffled edges later.
    all_targets = set(grn['target'])
    target_sum_dict = {target : 0.0 for target in all_targets}
    for _, target, importance in grn_zipped:
        target_sum_dict[target] += importance

    for i in range(num_permutations):
        # Sample representatives from each cluster.
        tf_representatives = _draw_representatives(cluster_to_tf, 1)
        gene_representatives = _draw_representatives(cluster_to_gene, 1)

        joint_representatives = list(set(tf_representatives + gene_representatives))
        represent_expression = expression_mat[joint_representatives].copy()
        # Shuffle expression matrix.
        represent_permuted = _shuffle_column_wise(df=represent_expression)
        shuffled_grn = grnboost2(
            expression_data=represent_permuted,
            tf_names=tf_representatives,
            seed=grnboost2_random_seed,
        )

        # Compute per-target importance factor sums on shuffled edges. If target does not exist
        # in to-be-pruned GRN, we can already ignore it for efficiency reasons.
        shuffled_target_dict = {target : 0.0 for target in set(shuffled_grn['target']) if target in all_targets}
        for target, factor in zip(shuffled_grn['target'], shuffled_grn['importance']):
            if target in all_targets:
                shuffled_target_dict[target] += factor            

        # Compute count values based on scaled edge importances for each shuffled edge.
        for tf, target, factor in zip(shuffled_grn['TF'], shuffled_grn['target'], shuffled_grn['importance']):
            if (tf, target) in grn_dict:
                adjusted_factor = factor * (target_sum_dict[target] / shuffled_target_dict[target])
                count_value = int(adjusted_factor >= grn_dict[(tf, target)][0])
                # Update corresponding edge count.
                grn_dict[(tf, target)][1] += count_value

    # Compute edge-wise empirical P-values from counts.
    grn_dict_final = {key : (val[0], val[1], (1+val[1])/(1+num_permutations)) for key, val in grn_dict.items()}

    grn_transposed = {'tf': [], 'target': [], 'importance': [], 'count': [], 'pvalue': []}
    for key, val in grn_dict_final.items():
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


def _shuffle_column_wise(df: pd.DataFrame):
    matrix = df.to_numpy()
    shuffled = _shuffle_column_wise_worker(matrix=matrix)
    return pd.DataFrame(shuffled, columns=df.columns, index=df.index)


@njit(parallel=True)
def _shuffle_column_wise_worker(matrix: np.ndarray):
    n_cols = matrix.shape[1]
    out = np.empty_like(matrix)
    for col in prange(n_cols):
        out[:, col] = np.random.permutation(matrix[:, col])
    return out


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
        shuffled_expression_mat = _shuffle_column_wise(df=expression_mat)

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

