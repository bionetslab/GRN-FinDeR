import pandas as pd
import numpy as np
import gc
import time
from dask.distributed import Client, LocalCluster
from statsmodels.stats.multitest import multipletests
from arboreto.algo import grnboost2
from arboreto.utils import load_tf_names
from src.preprocessing import preprocess_data
from src.clustering import cluster_genes




def classical_fdr(expression_matrix: pd.DataFrame,
                  tf_names_file: str, 
                  grn: pd.DataFrame,
                  output_dir: str, 
                  num_permutations=1000) -> pd.DataFrame:
    """
    Statistically evaluate Gene Regulatory Networks (GRNs) using RNA-seq expression data,
    compute p-values and FDR for the edges in the GRN.

    This function:
    1.  Preprocesses the input expression matrix using the `preprocess_data` function.
    2.  Initializes a local Dask cluster for parallel processing.
    3.  For a specified number of permutations, it shuffles the expression matrix, computes the GRN,
        and counts how many times the importance of each edge in the real GRN is less than the importance of that edge
        in the shuffled GRNs.
    4.  Calculates p-values for the edges in the original GRN based on the counts from the permutations.
    5.  Applies FDR correction to the calculated p-values.
    6.  Saves the final GRN with p-values to the specified output directory.

    Parameters:
    ----------
    expression_matrix : pd.DataFrame
        The input gene expression matrix where rows represent samples and columns represent genes.
    tf_names_file : str
        Path to the file containing the list of transcription factors (TFs) in `.tsv` format.
    grn : pd.DataFrame
        The original Gene Regulatory Network DataFrame containing columns ['TF', 'target', 'importance'].
    output_dir : str
        Directory to save the final GRN output file with p-values.
    num_permutations : int, optional
        Number of permutations to perform for the edge significance testing (default is 1000).

    Returns:
    -------
    pd.DataFrame
        DataFrame containing the final GRN with p-values and FDR correction.

    Example:
    --------
    >>> import pandas as pd
    >>> from src.grn_computation import compute_grn
    >>> from src.fdr_calculation import generate_and_evaluate_grns_rnaseq_final
    >>> expression_matrix = pd.read_csv('preprocessed_expression_matrix.csv')
    >>> tf_names_file = 'genenametfs.tsv'
    >>> output_dir = '/path/to/output/directory'
    >>> grn = compute_grn(expression_matrix, tf_names_file, output_dir)
    >>> final_grn = classical_fdr(expression_matrix, tf_names_file, grn, output_dir)
    >>> print(final_grn.head())

    Notes:
    ------
    - Requires a Dask local cluster for parallel computation; adjust `n_workers` and 
      `threads_per_worker` based on available resources.
    """

    start_time = time.time()

    # Set up a Dask local cluster for parallel computing
    local_cluster = LocalCluster(n_workers=60, threads_per_worker=16, memory_limit='20GB')
    custom_client = Client(local_cluster)

    # Preprocess the expression matrix
    filtered_matrix_vst = preprocess_data(expression_matrix=expression_matrix)

    print('Computing Classical FDR')

    # Load transcription factor (TF) names
    tf_names = load_tf_names(tf_names_file)

    # Prepare edge counts dictionary from the original GRN
    edge_counts = {tuple(row): 0 for row in grn[['TF', 'target']].values}

    # Permutation test
    for i in range(num_permutations):
        shuffled_matrix = filtered_matrix_vst.apply(np.random.permutation, axis=0)
        shuffled_grn = grnboost2(expression_data=shuffled_matrix, tf_names=tf_names, client_or_address=custom_client, verbose=False, seed=777)

        # Count the matches between the original and shuffled GRNs
        merged_grn = pd.merge(grn, shuffled_grn, on=['TF', 'target'], suffixes=('_real', '_shuffled'))
        merged_grn['importance_match'] = (merged_grn['importance_shuffled'] >= merged_grn['importance_real']).fillna(False)
        
        local_edge_counts = merged_grn.groupby(['TF', 'target'])['importance_match'].sum().to_dict()
        
        for key, value in local_edge_counts.items():
            edge_counts[key] += value

        del shuffled_matrix, shuffled_grn, merged_grn
        gc.collect()

    # Calculate p-values
    grn['p_value'] = grn.apply(lambda row: edge_counts[(row['TF'], row['target'])] / num_permutations, axis=1)

    # Apply FDR correction
    p_values = grn['p_value'].values
    grn['fdr'] = multipletests(p_values, method='fdr_bh')[1]

    # Save the results
    output_file_path = f"{output_dir}/final_grn_with_pvalues.tsv"
    grn.to_csv(output_file_path, sep='\t', index=False)

    elapsed_time = time.time() - start_time
    print(f"Time taken for classical FDR: {elapsed_time} seconds")

    with open(f"{output_dir}/time_log.txt", 'w') as log_file:
        log_file.write(f'Time elapsed: {elapsed_time:.2f} seconds\n')

    # Close the Dask client and local cluster
    custom_client.close()
    local_cluster.close()

    # Return the final GRN with p-values
    return grn


def classical_fdr_wout_merge(expression_matrix: pd.DataFrame,
                  tf_names_file: str,
                  grn: pd.DataFrame,
                  output_dir: str,
                  num_permutations=1000) -> pd.DataFrame:

    start_time = time.time()

    # Set up a Dask local cluster for parallel computing
    local_cluster = LocalCluster(n_workers=60, threads_per_worker=16, memory_limit='20GB')
    custom_client = Client(local_cluster)

    # Preprocess the expression matrix
    filtered_matrix_vst = preprocess_data(expression_matrix=expression_matrix)

    print('Computing Classical FDR')

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

    elapsed_time = time.time() - start_time
    print(f"Time taken for classical FDR: {elapsed_time} seconds")

    with open(f"{output_dir}/time_log.txt", 'w') as log_file:
        log_file.write(f'Time elapsed: {elapsed_time:.2f} seconds\n')

    # Close the Dask client and local cluster
    custom_client.close()
    local_cluster.close()

    # Return the final GRN with p-values
    return grn



def fdr_centroid(expression_matrix: pd.DataFrame,
                 tf_names_file: str,
                 grn: pd.DataFrame,
                 output_dir: str,
                 num_permutations= 1000,
                 batch_size: int = 5000,
                 num_workers:int = 15 ,
                 memory_limit: str = '200GB', 
                 m: int = 10) -> pd.DataFrame:
    """
    Statistically evaluate Gene Regulatory Networks (GRNs) using centroid-based representative genes,
    gemerated from clustering, compute p-values and FDR for the edges in the GRN.

    This function:
    1. Preprocesses the input expression matrix using the `preprocess_data` function.
    2. Calls the `cluster_genes` function to obtain hierarchical clustering mapping and clusters.
    3. Selects the centroid gene of each cluster as representative genes to create a subset of the expression matrix.
    4. For a specified number of permutations, it shuffles the subset expression matrix, computes the GRN.
    5. Extrapolates the shuffled importance score of the representative genes edges to the respective cluster genes edges.
    6. Counts how many times the importance of each edge in the real GRN is less than the importance of that edge
        in the shuffled GRNs.
    7.  Calculates p-values for the edges in the original GRN based on the counts from the permutations.
    8.  Applies FDR correction to the calculated p-values.
    9.  Saves the final GRN with p-values to the specified output directory.

    Parameters:
    ----------
    expression_matrix : pd.DataFrame
        The input gene expression matrix where rows represent samples and columns represent genes.
    tf_names_file     : str
        Path to the file containing the list of transcription factors (TFs) in `.tsv` format.
    grn              : pd.DataFrame
        The original Gene Regulatory Network DataFrame containing columns ['TF', 'target', 'importance'].
    output_dir       : str
        Directory to save the final GRN output file with p-values.
    num_permutations : int, optional
        Number of permutations to perform for the edge significance testing (default is 1000).
    batch_size       :  int, optional 
        Number of pairs to process in a single batch (default is 5000).
    num_workers      :  int, optional
        Number of Dask workers (default is 15).
    memory_limit     :  str, optional
        Memory limit for Dask workers (default is '200GB').
    m                :int
         The divisor to calculate the number of clusters.

    Returns:
    -------
    pd.DataFrame
        DataFrame containing the final GRN with p-values and FDR correction.   

    Example:
    --------
    >>> import pandas as pd
    >>> from src.grn_computation import compute_grn
    >>> from src.fdr_calculation import generate_and_evaluate_grns_rnaseq_final
    >>> expression_matrix = pd.read_csv('preprocessed_expression_matrix.csv')
    >>> tf_names_file = 'genenametfs.tsv'
    >>> output_dir = '/path/to/output/directory'
    >>> grn = compute_grn(expression_matrix, tf_names_file, output_dir)
    >>> final_grn = fdr_centroid(expression_matrix, tf_names_file, grn, output_dir)
    >>> print(final_grn.head())
    
    Notes:
    ------
    - Requires a Dask local cluster for parallel computation; adjust `n_workers` and 
      `threads_per_worker` based on available resources.
    - This function requires sufficient computational resources to handle large gene expression datasets.
    - Adjust `batch_size`, `n_workers`, and `memory_limit` to optimize memory usage and processing time depending on dataset size.
    - The function assumes that the input expression_matrix is already filtered for protein-coding genes.
    - Ensure that the tf_names_file_path is correct and points to a valid file.
   

    """

    # Preprocess the expression matrix
    filtered_matrix_vst = preprocess_data(expression_matrix=expression_matrix)

    # Perform Clustering
    hclust_gene_mapping, clusters, time_clustering = cluster_genes(filtered_matrix_vst, tf_names_file, 
                                                                   batch_size, num_workers, 
                                                                   memory_limit, m)
    
    start_time = time.time()

    # Set up a Dask local cluster for parallel computing
    local_cluster = LocalCluster(n_workers=60, threads_per_worker=16, memory_limit='20GB')
    custom_client = Client(local_cluster)

    print('Computing FDR approximation with Centroid Genes')

    # Load transcription factor (TF) names
    tf_names = load_tf_names(tf_names_file)

    # Filter the matrix to only include the representative genes (centroids)
    representative_genes = list(hclust_gene_mapping.keys())
    filtered_matrix_representatives = filtered_matrix_vst[representative_genes]

    edge_counts = {tuple(row): 0 for row in grn[['TF', 'target']].values}

    cluster_df = pd.DataFrame({'cluster': clusters})
    cluster_df = cluster_df.explode('cluster').reset_index().rename(columns={'index': 'cluster_id', 'cluster': 'gene'})

    for i in range(num_permutations):

        local_edge_counts = {key: 0 for key in edge_counts.keys()}
        shuffled_matrix_representatives = filtered_matrix_representatives.apply(np.random.permutation, axis=0)

        shuffled_grn = grnboost2(expression_data=shuffled_matrix_representatives, tf_names=tf_names, client_or_address=custom_client, verbose=False, seed=777)
        matching_edges = pd.merge(shuffled_grn, cluster_df, left_on='target', right_on ='gene',how='left')
        represented_edges = pd.merge(matching_edges[['TF', 'cluster_id','importance']], cluster_df[['cluster_id', 'gene']],
                                     on='cluster_id', how='inner')    
        merged_df = pd.merge(grn, represented_edges[['TF','gene','importance']],
                             left_on=['TF','target'], right_on=['TF','gene'],
                             suffixes=('_real', '_shuffled'))
        
        merged_df['match'] = (merged_df['importance_shuffled'] >= merged_df['importance_real']).fillna(False)
        local_edge_counts = merged_df.groupby(['TF', 'target'])['match'].sum().to_dict()
        for key, value in local_edge_counts.items():
            edge_counts[key] += value

        del shuffled_matrix_representatives, shuffled_grn, merged_df, matching_edges, represented_edges
        gc.collect()
    
    # Calculate p-values
    grn['p_value'] = grn.apply(lambda row: edge_counts[(row['TF'], row['target'])] / num_permutations, axis=1)

    # Apply FDR correction
    p_values = grn['p_value'].values
    grn['fdr'] = multipletests(p_values, method='fdr_bh')[1]

    # Save the results
    output_file_path = f"{output_dir}/centroid_grn_with_pvalues.tsv"
    grn.to_csv(output_file_path, sep='\t', index=False)

    elapsed_time = time.time() - start_time

    total_time = time_clustering + elapsed_time
    print(f"Time taken for FDR approximation using centroid: {total_time} seconds")

    with open(f"{output_dir}/time_log.txt", 'w') as log_file:
        log_file.write(f'Time elapsed: {total_time:.2f} seconds\n')

    # Close the Dask client and local cluster
    custom_client.close()
    local_cluster.close()

    # Return the final GRN with p-values
    return grn





def fdr_rotation(expression_matrix: pd.DataFrame,
                 tf_names_file: str,
                 grn: pd.DataFrame,
                 output_dir: str,
                 num_permutations= 1000,
                 batch_size: int = 5000,
                 num_workers:int = 15 ,
                 memory_limit: str = '200GB', 
                 m: int = 10) -> pd.DataFrame:
    """
    Statistically evaluate Gene Regulatory Networks (GRNs) using rotation-based representative genes,
    gemerated from clustering, and compute p-values and FDR for the edges in the GRN.

    This function:
    1. Preprocesses the input expression matrix using the `preprocess_data` function.
    2. Calls the `cluster_genes` function to obtain hierarchical clustering mapping and clusters.
    3. For a specified number of permutations, it selects one randomly sampled gene from each clusters 
        as representative genes to create a subset of the expression matrix.
    4. For a specified number of permutations, it shuffles the subset expression matrix, computes the GRN.
    5. Extrapolates the shuffled importance score of the representative genes edges to the respective cluster genes edges.
    6. Counts how many times the importance of each edge in the real GRN is less than the importance of that edge
        in the shuffled GRNs.
    7.  Calculates p-values for the edges in the original GRN based on the counts from the permutations.
    8.  Applies FDR correction to the calculated p-values.
    9.  Saves the final GRN with p-values to the specified output directory.

    Parameters:
    ----------
    expression_matrix : pd.DataFrame
        The input gene expression matrix where rows represent samples and columns represent genes.
    tf_names_file     : str
        Path to the file containing the list of transcription factors (TFs) in `.tsv` format.
    grn              : pd.DataFrame
        The original Gene Regulatory Network DataFrame containing columns ['TF', 'target', 'importance'].
    output_dir       : str
        Directory to save the final GRN output file with p-values.
    num_permutations : int, optional
        Number of permutations to perform for the edge significance testing (default is 1000).
    batch_size       :  int, optional 
        Number of pairs to process in a single batch (default is 5000).
    num_workers      :  int, optional
        Number of Dask workers (default is 15).
    memory_limit     :  str, optional
        Memory limit for Dask workers (default is '200GB').
    m                :int
         The divisor to calculate the number of clusters.

    Returns:
    -------
    pd.DataFrame
        DataFrame containing the final GRN with p-values and FDR correction.   

    Example:
    --------
    >>> import pandas as pd
    >>> from src.grn_computation import compute_grn
    >>> from src.fdr_calculation import generate_and_evaluate_grns_rnaseq_final
    >>> expression_matrix = pd.read_csv('preprocessed_expression_matrix.csv')
    >>> tf_names_file = 'genenametfs.tsv'
    >>> output_dir = '/path/to/output/directory'
    >>> grn = compute_grn(expression_matrix, tf_names_file, output_dir)
    >>> final_grn = fdr_rotation(expression_matrix, tf_names_file, grn, output_dir)
    >>> print(final_grn.head())
    
    Notes:
    ------
    - Requires a Dask local cluster for parallel computation; adjust `n_workers` and 
      `threads_per_worker` based on available resources.
    - This function requires sufficient computational resources to handle large gene expression datasets.
    - Adjust `batch_size`, `n_workers`, and `memory_limit` to optimize memory usage and processing time depending on dataset size.
    - The function assumes that the input expression_matrix is already filtered for protein-coding genes.
    - Ensure that the tf_names_file_path is correct and points to a valid file.
   

    """

    # Preprocess the expression matrix
    filtered_matrix_vst = preprocess_data(expression_matrix=expression_matrix)

    # Perform Clustering
    hclust_gene_mapping, clusters, time_clustering = cluster_genes(filtered_matrix_vst, tf_names_file, 
                                                                   batch_size, num_workers, 
                                                                   memory_limit, m)
    
    start_time = time.time()

    # Set up a Dask local cluster for parallel computing
    local_cluster = LocalCluster(n_workers=60, threads_per_worker=16, memory_limit='20GB')
    custom_client = Client(local_cluster)

    print('Computing FDR approximation with Rotational Genes')

    # Load transcription factor (TF) names
    tf_names = load_tf_names(tf_names_file)

    edge_counts = {tuple(row): 0 for row in grn[['TF', 'target']].values}

    cluster_df = pd.DataFrame({'cluster': clusters})
    cluster_df = cluster_df.explode('cluster').reset_index().rename(columns={'index': 'cluster_id', 'cluster': 'gene'})

    for i in range(num_permutations):

        local_edge_counts = {key: 0 for key in edge_counts.keys()}
        representative_genes = cluster_df.groupby('cluster_id')['gene'].apply(lambda x: x.sample(n=1).values[0]).tolist()
        filtered_matrix_representatives = filtered_matrix_vst[representative_genes]
        shuffled_matrix_representatives = filtered_matrix_representatives.apply(np.random.permutation, axis=0)

        shuffled_grn = grnboost2(expression_data=shuffled_matrix_representatives, tf_names=tf_names, client_or_address=custom_client, verbose=False, seed=777)
        matching_edges = pd.merge(shuffled_grn, cluster_df, left_on='target', right_on ='gene',how='left')
        represented_edges = pd.merge(matching_edges[['TF', 'cluster_id','importance']], cluster_df[['cluster_id', 'gene']],
                                     on='cluster_id', how='inner')
        merged_df = pd.merge(grn, represented_edges[['TF','gene','importance']],
                             left_on=['TF','target'], right_on=['TF','gene'],
                             suffixes=('_real', '_shuffled'))
        merged_df['match'] = (merged_df['importance_shuffled'] >= merged_df['importance_real']).fillna(False)

        local_edge_counts = merged_df.groupby(['TF', 'target'])['match'].sum().to_dict()
        for key, value in local_edge_counts.items():
            edge_counts[key] += value

        del shuffled_matrix_representatives, shuffled_grn, merged_df, matching_edges, represented_edges
        gc.collect()

    # Calculate p-values
    grn['p_value'] = grn.apply(lambda row: edge_counts[(row['TF'], row['target'])] / num_permutations, axis=1)

    # Apply FDR correction
    p_values = grn['p_value'].values
    grn['fdr'] = multipletests(p_values, method='fdr_bh')[1]

    # Save the results
    output_file_path = f"{output_dir}/rotation_grn_with_pvalues.tsv"
    grn.to_csv(output_file_path, sep='\t', index=False)

    elapsed_time = time.time() - start_time

    total_time = time_clustering + elapsed_time
    print(f"Time taken for FDR approximation using rotation: {total_time} seconds")

    with open(f"{output_dir}/time_log.txt", 'w') as log_file:
        log_file.write(f'Time elapsed: {total_time:.2f} seconds\n')

    # Close the Dask client and local cluster
    custom_client.close()
    local_cluster.close()

    # Return the final GRN with p-values
    return grn