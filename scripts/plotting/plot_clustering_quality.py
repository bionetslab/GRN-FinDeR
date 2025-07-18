import pandas as pd
from arboreto.fdr_utils import compute_wasserstein_distance_matrix, cluster_genes_to_dict
from sklearn.metrics import silhouette_score
import os
import matplotlib.pyplot as plt

def compute_silhouette_scores(expression_df,
                              target_list : list[str],
                              num_tf_cluster_list : list[int],
                              num_non_tf_cluster_list : list[int]):

    non_tf_silhouettes = []
    tf_silhouettes = []

    all_genes = list(expression_df.columns)
    tf_names = [gene for gene in all_genes if not gene in target_list]

    # Compute full distance matrix between all pairs of input genes.
    dist_matrix_all = compute_wasserstein_distance_matrix(expression_df, num_threads=-1)

    # Separate TF and non-TF distances and cluster both types individually.
    tf_bool = [True if gene in tf_names else False for gene in dist_matrix_all.columns]
    non_tf_bool = [False if gene in tf_names else True for gene in dist_matrix_all.columns]
    dist_mat_non_tfs = dist_matrix_all.loc[non_tf_bool, non_tf_bool]
    dist_mat_tfs = dist_matrix_all.loc[tf_bool, tf_bool]

    for num_tf_clusters in num_tf_cluster_list:
        tf_to_clust = cluster_genes_to_dict(dist_mat_tfs, num_clusters=num_tf_clusters)
        # Prepare input for TF clusters to silhouette score computation.
        tf_cluster_labels = [tf_to_clust[gene] for gene in dist_mat_tfs.columnms]
        dist_mat_tfs_numpy = dist_mat_tfs.copy().to_numpy()
        tf_silhouette_score = silhouette_score(X=dist_mat_tfs_numpy, labels=tf_cluster_labels, metric='precomputed')
        tf_silhouettes.append(tf_silhouette_score)

    for num_non_tf_clusters in num_non_tf_cluster_list:
        non_tf_to_clust = cluster_genes_to_dict(dist_mat_non_tfs, num_clusters=num_non_tf_clusters)
        # Prepare input for non-TF clusters to silhouette score computation.
        non_tf_cluster_labels = [non_tf_to_clust[gene] for gene in dist_mat_non_tfs.columns]
        dist_mat_non_tfs_numpy = dist_mat_non_tfs.copy().to_numpy()
        non_tf_silhouette_score = silhouette_score(X=dist_mat_non_tfs_numpy, labels=non_tf_cluster_labels, metric='precomputed')
        non_tf_silhouettes.append(non_tf_silhouette_score)

    return tf_silhouettes, non_tf_silhouettes

def plot_silhouette_scores(num_clusters, silhouette_scores, x_label, file_name):
    plt.figure(figsize=(8, 5))
    plt.plot(num_clusters, silhouette_scores, marker='o', linestyle='-', color='blue')

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel('Averaged Silhouette Score')
    #plt.title('Silhouette Score vs. Number of Clusters')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{file_name}.png')

def compute_silhouettes_for_all_tissues(all_tissues_dir, output_dir, subset_tissues : list[str]):

    non_tf_cluster_list = list(range(1,10,1)) + list(range(10, 100, 10)) + list(range(100, 1001, 100))
    tf_cluster_list = list(range(1,10,1)) + list(range(10, 100, 10)) + list(range(100, 1501, 100))

    # Iterate over subdirectories in the parent directory
    for tissue_dir in os.listdir(all_tissues_dir):
        if len(subset_tissues) > 0 and tissue_dir not in subset_tissues:
            continue
        full_dir_path = os.path.join(all_tissues_dir, tissue_dir)

        if os.path.isdir(full_dir_path):
            tsv_file = os.path.join(full_dir_path, f'{tissue_dir}.tsv')
            exp_mat = pd.read_csv(tsv_file, sep='\t', index_col=0)
            target_file = os.path.join(full_dir_path, f'{tissue_dir}_target_genes.tsv')
            target_df = pd.read_csv(target_file, index_col=0)
            target_list = list(target_df['target_gene'])

            tf_silhouettes, non_tf_silhouettes = compute_silhouette_scores(
                exp_mat,
                target_list,
                tf_cluster_list,
                non_tf_cluster_list
            )

            plot_file_tf = os.path.join(output_dir, tissue_dir, 'silhouette_tf_clusters.png')
            plot_silhouette_scores(tf_cluster_list, tf_silhouettes, 'Number TF Clusters', plot_file_tf)

            plot_file_non_tf = os.path.join(output_dir, tissue_dir, 'silhouette_non_tf_clusters.png')
            plot_silhouette_scores(non_tf_cluster_list, non_tf_silhouettes, 'Number non_TF Clusters', plot_file_non_tf)

if __name__ == "__main__":
    all_tissue_dir = "work/"
    output_dir = "gtex_fdr_results/"
    subset_tissues = ['Liver']

    compute_silhouettes_for_all_tissues(all_tissue_dir,
                                        output_dir,
                                        subset_tissues)