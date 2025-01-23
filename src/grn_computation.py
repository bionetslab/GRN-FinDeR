import pandas as pd
from arboreto.algo import grnboost2
from arboreto.utils import load_tf_names
from dask.distributed import Client, LocalCluster
from preprocessing import preprocess_data

def compute_grn(expression_matrix: pd.DataFrame, tf_names_file: str, output_dir: str) -> pd.DataFrame:
    """
    Compute the Gene Regulatory Network (GRN) from a preprocessed expression matrix
    using the GRNBoost2 algorithm and save the output in the specified format.

    This function:
    1. Loads a list of transcription factors (TFs) from a specified file.
    2. Initializes a local Dask cluster for parallel processing.
    3. Preprocesses Expression matrix
    4. Computes the GRN using the GRNBoost2 algorithm with the preprocessed expression matrix.
    5. Saves the computed GRN to the specified output directory in the format:
       TF, target, importance.

    Parameters:
    ----------
    expression_matrix : pd.DataFrame
        The input gene expression matrix where rows represent samples and columns represent genes..
    tf_names_file : str
        Path to the file containing the list of TF(Transcription Factor) gene HGNC symbols in `.tsv` format.
    output_dir : str
        Directory to save the computed GRN output file.

    Returns:
    -------
    pd.DataFrame
        DataFrame containing the computed GRN with columns ['TF', 'target', 'importance'].

    Example:
    --------
    >>> import pandas as pd
    >>> from src.grn_computation import compute_grn
    >>> expression_matrix = pd.read_csv('preprocessed_expression_matrix.csv')
    >>> tf_names_file = 'genenametfs.tsv'
    >>> output_dir = '/path/to/output/directory'
    >>> grn_df = compute_grn(expression_matrix, tf_names_file, output_dir)
    >>> print(grn_df.head())

    Notes:
    ------
    - Requires a Dask local cluster for parallel computation; adjust `n_workers` and 
      `threads_per_worker` based on available resources.
    """
    
    # Load transcription factor (TF) names
    tf_names = load_tf_names(tf_names_file)

    # Set up a Dask local cluster for parallel computing
    local_cluster = LocalCluster(n_workers=60, threads_per_worker=16)
    custom_client = Client(local_cluster)

    #  Preprocess expression matrix
    preprocessed_matrix = preprocess_data(expression_matrix=expression_matrix)

    print('Computing GRN')
    # Compute the GRN using GRNBoost2
    grn = grnboost2(expression_data=preprocessed_matrix, tf_names=tf_names,
                    client_or_address=custom_client, verbose=True, seed=777)


    # Save the GRN as a .tsv file in the output directory with specified format
    grn_output_path = f"{output_dir}/computed_GRN.tsv"
    grn.to_csv(grn_output_path, sep='\t', index=False)

    custom_client.close()
    local_cluster.close()

    # Return the GRN as a DataFrame
    return grn