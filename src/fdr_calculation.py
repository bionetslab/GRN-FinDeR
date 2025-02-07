
import pandas as pd
import numpy as np
import gc
from dask.distributed import Client, LocalCluster
from statsmodels.stats.multitest import multipletests
from arboreto.algo import grnboost2
from arboreto.utils import load_tf_names

from src.clustering import cluster_genes_to_dict
from src.preprocessing import preprocess_data
import random
import itertools


def approximate_fdr(expression_mat : pd.DataFrame,
                    grn : pd.DataFrame,
                    gene_to_cluster : dict,
                    num_permutations : int = 1000):
    
    # Invert gene to cluster dictionary.
    cluster_to_gene = dict()
    for key, val in gene_to_cluster.items():
        if val in cluster_to_gene:
            cluster_to_gene[val].append(key)
        else:
            cluster_to_gene[val] = [key]

    # Compute set of singleton genes, i.e. genes that are in one-element-cluster.
    singleton_genes = {genes[0] for _, genes in cluster_to_gene.items() if len(genes)==1}

    # Create dict from original GRN {('TF', 'target'): ('importance', 'counter')}
    grn_zipped = zip(grn['TF'].to_list(), grn['target'].to_list(), grn['importance'].to_list())
    grn_dict = {(tf, target): (importance, 0.0) for tf, target, importance in grn_zipped}
    
    # Init data structure for counting between and withing cluster empirical counts.
    cluster_cluster_counts = {cluster_edge : 0.0 for cluster_edge in itertools.product(cluster_to_gene.keys(), repeat=2)}

    for i in range(num_permutations):
        # Sample representatives from each cluster.
        representatives = _draw_representatives(cluster_to_gene, 2)
        represent_expression = expression_mat[representatives].copy()
        # Shuffle expression matrix.
        represent_permuted = represent_expression.sample(frac=1, axis=1)
        shuffled_grn = grnboost2(expression_data=represent_permuted, 
                                 tf_names=represent_permuted.columns.to_list())
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
        p_value = (cluster_cluster_counts[cluster_tuple]+1) / (num_permutations+1)
        grn_dict[key] = (grn_dict[key][0], p_value)
    
    grn_transposed = {'tf': [], 'target' : [], 'importance' : [], 'pvalue': []}
    for key, val in grn_dict.items():
        grn_transposed['tf'].append(key[0])
        grn_transposed['target'].append(key[1])
        grn_transposed['importance'].append(val[0])
        grn_transposed['pvalue'].append(val[1])
    
    grn_df = pd.DataFrame.from_dict(grn_transposed)
    return grn_df


def _draw_representatives(cluster_to_gene : dict, num_representatives : int = 2):
    representatives = [random.sample(val, min(len(val), num_representatives)) for val in cluster_to_gene.values()]
    representatives = list(itertools.chain.from_iterable(representatives))
    return representatives


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

