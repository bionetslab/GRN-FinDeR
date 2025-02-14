import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt


def plot_cluster_metrics(file_path : str, clustering_sizes : list[int]):
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
        
        # Open precompute cluster metrics.
        with open(os.path.join(subdir, 'diameters_per_clustering.pkl'), 'rb') as f:
            diams_per_cluster_dict = pickle.load(f)
        
        with open(os.path.join(subdir, 'num_singletons_per_clustering.pkl'), 'rb') as f:
            singletons_per_clustering = pickle.load(f)
            
        with open(os.path.join(subdir, 'sizes_per_clustering.pkl'), 'rb') as f:
            sizes_per_clustering = pickle.load(f)
            
        subset_diams_per_clustering = {key : val for key,val in diams_per_cluster_dict.items() if key in clustering_sizes}
        subset_singletons_per_clustering = {key : val for key,val in singletons_per_clustering.items() if key in clustering_sizes}
        subset_sizes_per_clustering = {key : val for key, val in sizes_per_clustering.items() if key in clustering_sizes}
        
        # Plot diameters, singletons, and sizes.
        x_labels = clustering_sizes
        diams_data = list(subset_diams_per_clustering.values())
        sizes_data = list(subset_sizes_per_clustering.values())
        singletons_list = [val for _, val in subset_singletons_per_clustering.items()]

        dpi = 300
        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax.boxplot(sizes_data, tick_labels=[str(i) for i in x_labels])  # Assign labels to x-axis
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Cluster Sizes")
        ax.grid(True, linestyle='-', color='grey', alpha=0.6)
        plt.savefig(os.path.join(subdir, 'cluster_sizes.png'), dpi=dpi)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax.violinplot(diams_data)
        ax.set_xticks(list(range(1, len(diams_data) + 1)))
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Cluster Diameters")
        plt.savefig(os.path.join(subdir, 'cluster_diameters.png'), dpi=dpi)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax.plot(x_labels, singletons_list, marker='o')
        # ax.set_xticks(list(range(1, len(diams_data) + 1)))
        # ax.set_xticklabels(x_labels)
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Number of Singletons")
        plt.savefig(os.path.join(subdir, 'number_of_singletons.png'), dpi=dpi)
        
        
if __name__ == "__main__":
    plot_cluster_metrics('/data/bionets/ac07izid/grn/data/gtex_cluster_data', list(range(100, 5001, 100)))
    