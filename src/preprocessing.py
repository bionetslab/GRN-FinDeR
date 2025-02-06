import pandas as pd
import numpy as np
import scanpy as sc
from pybiomart import Dataset
from sklearn.preprocessing import StandardScaler

def preprocess_gtex(input_file_path : str, output_file_path : str):
    """
        Load, preprocess and save GTEX gene expression matrices for given tissue.
            Parameters:
            ----------
            input_file_path (str): Path to file containing raw count expression matrices.
            output_file_path (str): Name and path of output file.
    """
    # Open file and transform into gene-column based format.
    exp_df = pd.read_csv(input_file_path, sep='\t', index_col='Description')
    exp_df.drop(columns=['Name'], inplace=True)
    exp_df = exp_df.T
    print(exp_df)
    
    # Fetch protein-coding genes from BioMart.
    dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
    genes = dataset.query(attributes=['hgnc_symbol', 'gene_biotype'])
    protein_coding_genes = genes[genes['Gene type'] == 'protein_coding']['HGNC symbol']
    print(protein_coding_genes)
    
    # Subset expression matrix only to protein coding genes.
    intersection_genes = set(exp_df.columns).intersection(protein_coding_genes)
    filtered_matrix = exp_df[list(intersection_genes)]
    filtered_matrix = filtered_matrix.astype(np.float32)
    print(filtered_matrix.shape)
    
    # Remove genes with low expression.
    RATIO_THRESHOLD = 0.1 * filtered_matrix.shape[0]
    filtered_matrix = filtered_matrix.loc[:, (filtered_matrix != 0).sum(axis=0) > RATIO_THRESHOLD]
    print(filtered_matrix.shape)
    
    # Scale gene columns to zero-mean and unit variance.
    scaler = StandardScaler()
    scaled_matrix = pd.DataFrame(scaler.fit_transform(filtered_matrix), columns=filtered_matrix.columns)
    print(scaled_matrix)
    
    # Save preprocessed expression matrix.
    scaled_matrix.to_csv(output_file_path, sep='\t')    
    

def preprocess_data(expression_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input gene expression matrix by filtering for protein-coding genes,
    normalizing, and scaling the data.

    Steps:
    ----------
    1. Filters for protein-coding genes using the HGNC symbol from BioMart.
    2. Filters genes based on minimum expression across cells.
    3. Applies a threshold to remove genes expressed in fewer than 10% of cells.
    4. Normalizes the total expression per cell to a target sum and scales the matrix.

    Parameters:
    ----------
    expression_matrix (pd.DataFrame): 
        The input gene expression matrix, where columns represent gene HGNC symbols and rows represent 
        samples or conditions or cells or metacells.

    Returns:
    ----------
    pd.DataFrame: Preprocessed, filtered, and normalized expression matrix.

    Example:
    --------
    >>> import pandas as pd
    >>> from src.preprocessing import preprocess_data
    >>> # Load or define an expression matrix
    >>> expression_matrix = pd.DataFrame({
    ...     'Gene1': [0, 1, 3, 0, 2],
    ...     'Gene2': [5, 2, 0, 0, 3],
    ...     'Gene3': [0, 0, 0, 1, 0]
    ... })
    >>> # Preprocess the expression matrix
    >>> preprocessed_matrix = preprocess_data(expression_matrix)
    >>> print(preprocessed_matrix)

    Notes:
    -----
    This function requires an internet connection to fetch gene HGNC symbols from the Ensembl database.

    """
    print('Preprocessing Data')

    # Fetch protein-coding genes from BioMart
    dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
    genes = dataset.query(attributes=['hgnc_symbol', 'gene_biotype'])
    protein_coding_genes = genes[genes['Gene type'] == 'protein_coding']['HGNC symbol']

    # Filter for protein-coding genes in the expression matrix
    adata = sc.AnnData(expression_matrix)
    sc.pp.filter_genes(adata, min_cells=1)
    filtered_ex_matrix = adata.to_df()
    intersection_genes = set(filtered_ex_matrix.columns).intersection(protein_coding_genes)
    filtered_matrix = filtered_ex_matrix[list(intersection_genes)]
    filtered_matrix = filtered_matrix.astype(np.float32)

    # Apply threshold to remove genes with low expression
    threshold = 0.1 * filtered_matrix.shape[0]
    filtered_matrix = filtered_matrix.loc[:, (filtered_matrix != 0).sum(axis=0) > threshold]
    
    # Normalize and scale the matrix
    adata_filtered = sc.AnnData(filtered_matrix)
    sc.pp.normalize_total(adata_filtered, target_sum=1e4, exclude_highly_expressed=True)
    sc.pp.scale(adata_filtered, zero_center=True)
    filtered_matrix_vst = pd.DataFrame(adata_filtered.X, index=filtered_matrix.index, columns=filtered_matrix.columns)

    return filtered_matrix_vst

if __name__ == '__main__':
    input_file_name = "/data/bionets/datasets/gtex/Colon/GTEX_Colon_counts.tsv"
    output_file_name = "/data/bionets/xa39zypy/GTEX/GTEX_Colon_filtered_standardized.tsv"
    preprocess_gtex(input_file_name, output_file_name)