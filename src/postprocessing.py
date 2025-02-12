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
    for tissue, subdir in zip(tissues[0], subdirectories[0]):
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
        
        plt.figure(figsize=(8, 5))
        plt.boxplot(sizes_data, labels=x_labels)  # Assign labels to x-axis
        plt.xlabel("Clustering Size")
        plt.ylabel("Cluster Sizes")
        plt.title("Box Plot of Cluster Sizes by Clustering Size")
        plt.show()
        
        
if __name__ == "__main__":
    plot_cluster_metrics("~/Projects/GRN-FinDeR.git/data/gtex_cluster_data/", list(range(100, 5001, 100)))
    