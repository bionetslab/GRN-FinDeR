import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt

from umap import UMAP


def plot_cluster_metrics(file_path : str, num_clusters : list[int], plt_umap: bool = False):
    """Plot different precomputed characteristics of clusters for all GTEX tissues.

    Args:
        file_path (str): Path to file containing all GTEX tissues.
        cluster_sizes (list[int]): Clustering sizes to analyze, i.e. plot on x-axis.
    """
    subdirectories = [os.path.join(file_path, d) for d in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, d))]
    tissues = [str(d) for d in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, d))]
    # Process all tissues.
    for tissue, subdir in zip(tissues, subdirectories):

        print(f'Processing tissue {tissue}...')
        subdir_metrics = os.path.join(subdir, 'clustering_metrics')
        # Open precompute cluster metrics.
        with open(os.path.join(subdir_metrics, 'diameters_per_clustering.pkl'), 'rb') as f:
            diams_per_cluster_dict = pickle.load(f)

        with open(os.path.join(subdir_metrics, 'median_distances_per_clustering.pkl'), 'rb') as f:
            median_distances_per_clustering = pickle.load(f)

        with open(os.path.join(subdir_metrics, 'num_singletons_per_clustering.pkl'), 'rb') as f:
            singletons_per_clustering = pickle.load(f)

        with open(os.path.join(subdir_metrics, 'sizes_per_clustering.pkl'), 'rb') as f:
            sizes_per_clustering = pickle.load(f)

        with open(os.path.join(subdir_metrics, 'silhouette_score_per_clustering.pkl'), 'rb') as f:
            silhouette_score_per_clustering = pickle.load(f)

        subset_diams_per_clustering = {key : val for key,val in diams_per_cluster_dict.items() if key in num_clusters}
        subset_median_distances_per_clustering = {key: val for key, val in median_distances_per_clustering.items() if key in num_clusters}
        subset_singletons_per_clustering = {key : val for key,val in singletons_per_clustering.items() if key in num_clusters}
        subset_sizes_per_clustering = {key : val for key, val in sizes_per_clustering.items() if key in num_clusters}
        subset_silhouette_score_per_clustering = {key: val for key, val in silhouette_score_per_clustering.items() if key in num_clusters}

        # Plot diameters, singletons, and sizes.
        x_labels = num_clusters
        diams_data = list(subset_diams_per_clustering.values())
        median_dist_data = list(subset_median_distances_per_clustering.values())
        sizes_data = list(subset_sizes_per_clustering.values())
        singletons_list = [val for _, val in subset_singletons_per_clustering.items()]
        silhouette_list = [val for _, val in subset_silhouette_score_per_clustering.items()]

        dpi = 300
        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax.boxplot(sizes_data, tick_labels=[str(i) for i in x_labels])
        ax.set_xticks(list(range(1, len(diams_data) + 1)))
        ax.set_xticklabels([str(x) if x % 500 == 0 or x == 100 else "" for x in x_labels])
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Cluster Sizes")
        ax.grid(True, linestyle='-', color='grey', alpha=0.6)
        plt.savefig(os.path.join(subdir_metrics, 'cluster_sizes.png'), dpi=dpi)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax.violinplot(diams_data)
        ax.set_xticks(list(range(1, len(diams_data) + 1)))
        ax.set_xticklabels([str(x) if x % 500 == 0 or x == 100 else "" for x in x_labels])
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Cluster Diameters")
        plt.savefig(os.path.join(subdir_metrics, 'cluster_diameters.png'), dpi=dpi)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax.violinplot(median_dist_data)
        ax.set_xticks(list(range(1, len(median_dist_data) + 1)))
        ax.set_xticklabels([str(x) if x % 500 == 0 or x == 100 else "" for x in x_labels])
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Cluster Median Distance")
        plt.savefig(os.path.join(subdir_metrics, 'cluster_median_distance.png'), dpi=dpi)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax.plot(x_labels, singletons_list, marker='o')
        # ax.set_xticks(list(range(1, len(diams_data) + 1)))
        # ax.set_xticklabels(x_labels)
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Number of Singletons")
        plt.savefig(os.path.join(subdir_metrics, 'number_of_singletons.png'), dpi=dpi)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax.plot(x_labels, silhouette_list, marker='o')
        # ax.set_xticks(list(range(1, len(diams_data) + 1)))
        # ax.set_xticklabels(x_labels)
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Silhouette Score")
        plt.savefig(os.path.join(subdir_metrics, 'silhouette_score.png'), dpi=dpi)
        plt.close(fig)

        if plt_umap:
            # Create directory for saving plots
            subdir_umaps = os.path.join(subdir_metrics, 'umaps')
            os.makedirs(subdir_umaps, exist_ok=True)

            # Load expression matrix
            # expression_file = f'{tissue}.tsv'
            # expression_df = pd.read_csv(os.path.join(subdir, expression_file), sep='\t', index_col=0)

            # Load distance matrix.
            distance_file = os.path.join(subdir, 'distance_matrix.csv')
            distance_df = pd.read_csv(distance_file, index_col=0)
            distance_df.index = distance_df.columns

            # Create Anndata object for plotting
            # expression_mat = expression_df.to_numpy().T
            # adata = sc.AnnData(X=expression_mat)
            # adata.obs_names = expression_df.columns.to_numpy().copy()

            # Compute Umap
            print('# ### Computing UMAP ...')
            umap_reducer = UMAP(metric='precomputed')
            umap_embedding = umap_reducer.fit_transform(distance_df.to_numpy())

            print('# ### Plotting UMAPs ...')
            for n in num_clusters:
                # Read clustering from respective file.
                cluster_file = os.path.join(subdir, 'clusterings', f'clustering_{n}.pkl')
                with open(cluster_file, 'rb') as handle:
                    gene_to_cluster = pickle.load(handle)

                # Annotate with cluster ids
                cluster_ids = [gene_to_cluster[gene] for gene in distance_df.columns.to_numpy()]

                # Plot Umap
                fig, ax = plt.subplots(dpi=dpi)
                sc_plot = ax.scatter(
                    umap_embedding[:, 0],
                    umap_embedding[:, 1],
                    c=cluster_ids,
                    cmap='gist_ncar',
                    alpha=0.8,
                    edgecolors='none',
                    s=120000 / (umap_embedding.shape[0] * 2),
                )
                cbar = fig.colorbar(sc_plot, ax=ax)
                cbar.set_label('Cluster ID', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel('Umap 1', fontsize=12)
                ax.set_ylabel('Umap 2', fontsize=12)
                plt.savefig(os.path.join(subdir_umaps, f'umap_n_clusters_{n}.png'), dpi=dpi)
                plt.close(fig)

