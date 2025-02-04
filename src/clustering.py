
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def cluster_genes_to_dict(distance_matrix : pd.DataFrame, num_clusters : int = 100):
    # Create clusters.
    agg_clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage='complete')
    cluster_labels = agg_clustering.fit_predict(distance_matrix.to_numpy())
    # Map clustering output to dictionary representation.
    gene_names = distance_matrix.columns.to_list()
    gene_to_cluster = {name : id for name, id in zip(gene_names, cluster_labels)}
    return gene_to_cluster
