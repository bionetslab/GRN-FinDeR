import numpy as np
import pandas as pd
import time
from sklearn.cluster import AgglomerativeClustering
from arboreto.utils import load_tf_names
from distance_matrix import compute_wasserstein_distances_hexa_split 

def cluster_genes(expression_matrix: pd.DataFrame,
                  tf_names_file_path: str , 
                  batch_size: int = 5000,
                  num_workers: int =15 ,
                  memory_limit: str = '200GB', 
                  m: int = 10) -> tuple[dict, list, float]:
    
    """
    Cluster genes based on the provided expression matrix and perform distance calculations using Wasserstein distances.

    This function computes the Wasserstein distance matrix for the given expression matrix, applies 
    hierarchical clustering on the computed distance matrix, and returns gene clusters and their mappings 
    without saving to files.

    Steps:
    ----------
    1. Compute the Wasserstein distance matrix using the provided expression matrix.
    2. Load the transcription factor names from the specified file.
    3. Filter the distance matrix to include only those genes that are not transcription factors.
    4. Perform agglomerative hierarchical clustering on the filtered distance matrix to cluster genes with similar expression profiles.
    5. Calculate the centroid of each cluster and create a mapping of centroid genes to their associated genes.
    
    Parameters:
    ----------
    expression_matrix (pd.DataFrame): The preprocessed expression matrix.

    tf_names_file_path (str): The file path to the TF names file.

    output_dir (str): The directory where outputs (if any) are saved.

    batch_size (int, optional): Number of pairs to process in a single batch (default is 5000).

    num_workers (int, optional): Number of Dask workers (default is 15).

    memory_limit (str, optional): Memory limit for Dask workers (default is '200GB').

    m (int): The divisor to calculate the number of clusters.

    Returns:
    ----------
    tuple: A tuple containing:
        - hclust_gene_mapping (dict): A dictionary mapping centroid genes to their associated genes.
        - clusters (list): A list of clusters, where each cluster is a list of gene names.
        - total_time (float): The total time taken for clustering.

    Example:
    ----------
    >>> expression_matrix = pd.DataFrame(...)  # Preprocessed expression data
    >>> tf_names_file_path = '/path/to/tfnames.tsv'
    >>> output_dir = '/path/to/output'
    >>> hclust_gene_mapping, clusters, total_time = cluster_genes(expression_matrix, tf_names_file_path, output_dir)

    Notes:
    ----------
    - This function requires sufficient computational resources to handle large gene expression datasets.
    - Adjust `batch_size`, `n_workers`, and `memory_limit` to optimize memory usage and processing time depending on dataset size.
    - The function assumes that the input expression_matrix is already filtered for protein-coding genes.
    - Ensure that the tf_names_file_path is correct and points to a valid file.
    """

    # Compute the Wasserstein distance matrix
    print('Computing Distance Matrix')

    distance_matrix, time_scipy = compute_wasserstein_distances_hexa_split(
        expression_matrix=expression_matrix, batch_size=batch_size, n_workers = num_workers, 
        memory_limit=memory_limit)
    
    start_time = time.time()

    print('Clustering Distances')

    # Load transcription factors
    tf_names = load_tf_names(tf_names_file_path)
    
    gene_names = distance_matrix.index.tolist()
    common_genes = list(set(gene_names) & set(tf_names))
    
    # Filter out the common genes from the distance matrix
    filtered_gene_names = [gene for gene in gene_names if gene not in common_genes]
    filtered_distances_df = distance_matrix.loc[filtered_gene_names, filtered_gene_names]
    filtered_distances = filtered_distances_df.values

    # Calculate number of clusters based on filtered data
    n_clusters = len(filtered_gene_names) // m
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
    hclust_labels = agg_clustering.fit_predict(filtered_distances)

    # Create gene mapping based on clustering
    hclust_gene_mapping = {}
    gene_names = filtered_distances_df.index

    for i in range(n_clusters):
        cluster_genes = gene_names[hclust_labels == i].tolist()
        cluster_indices = np.where(hclust_labels == i)[0]

        # Calculate the centroid of the cluster
        cluster_distances = filtered_distances[cluster_indices][:, cluster_indices]
        centroid_index = np.argmin(np.mean(cluster_distances, axis=1))
        
        centroid_gene = gene_names[cluster_indices[centroid_index]]
        cluster_genes.remove(centroid_gene)
        hclust_gene_mapping[centroid_gene] = cluster_genes
    
    for tf_gene in common_genes:
        hclust_gene_mapping[tf_gene] = []

    # Create final clusters
    clusters = [[] for _ in range(n_clusters)]
    for gene, label in zip(filtered_gene_names, hclust_labels):
        clusters[label].append(gene)

    # Add the TF genes to the clusters
    for tf_gene in common_genes:
        clusters.append([tf_gene])

    time_cluster = time.time() - start_time
    total_time = time_scipy + time_cluster

    return hclust_gene_mapping, clusters, total_time