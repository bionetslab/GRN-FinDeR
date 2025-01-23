# GRN_FDR

## Overview

This package provides tools for preprocessing gene expression data, computing gene regulatory networks (GRNs), calculating distance matrices, clustering genes, and performing False Discovery Rate (FDR) calculations.

## Installation

To install the package, you can use pip:

```bash
pip install -r requirements.txt
```

Make sure you have Python version 3.12.2 installed.

## Requirements

Here are the required packages for this project:

```plaintext
python==3.12.2
pandas>=1.0
numpy>=1.18
dask[distributed]>=2022.02.0
scipy>=1.5
scanpy>=1.8.0
pybiomart>=0.2.6
arboreto>=0.1.0
statsmodels>=0.12.0
scikit-learn>=0.24.0
```

## Usage

Below are some examples of how to use the package's functionalities:

### Preprocessing Data

This section shows how to preprocess the expression matrix.

```python
import pandas as pd
from src.preprocessing import preprocess_data

# Load or define an expression matrix
expression_matrix = pd.DataFrame({
    'Gene1': [0, 1, 3, 0, 2],
    'Gene2': [5, 2, 0, 0, 3],
    'Gene3': [0, 0, 0, 1, 0]
})

# Preprocess the expression matrix
preprocessed_matrix = preprocess_data(expression_matrix)
print(preprocessed_matrix)
```

### Computing Gene Regulatory Networks (GRNs)

This section shows how to compute GRNs from the preprocessed expression matrix.

```python
import pandas as pd
from src.grn_computation import compute_grn

# Load preprocessed expression matrix and TF names
expression_matrix = pd.read_csv('preprocessed_expression_matrix.csv')
tf_names_file = 'genenametfs.tsv'
output_dir = '/path/to/output/directory'

# Compute the GRN
grn_df = compute_grn(expression_matrix, tf_names_file, output_dir)
print(grn_df.head())
```

### Calculating Distance Matrices

This section demonstrates how to compute distance matrices using Wasserstein distances.

```python
import pandas as pd
from src.distance_matrix import compute_wasserstein_distances_rna_hexa_split

# Load the filtered expression matrix
expression_matrix = pd.read_csv('filtered_expression_matrix.csv')

# Compute the Wasserstein distance matrix
distance_matrix_df = compute_wasserstein_distances_rna_hexa_split(
    expression_matrix, batch_size=5000, n_workers=10, memory_limit='150GB'
)
print(distance_matrix_df.head())
print(f"Computation Time: {computation_time} seconds")
```

### Clustering Genes

This section explains how to cluster genes based on the expression data.

```python
import pandas as pd
from src.clustering import cluster_genes

# Preprocessed expression data
expression_matrix = pd.DataFrame(...)  # Replace with your data
tf_names_file_path = '/path/to/tfnames.tsv'
output_dir = '/path/to/output'

# Cluster genes
hclust_gene_mapping, clusters, total_time = cluster_genes(expression_matrix, tf_names_file_path, output_dir)
```

### FDR Calculation

This section covers different methods for calculating False Discovery Rates (FDR).

#### Classical Method

```python
import pandas as pd
from src.grn_computation import compute_grn
from src.fdr_calculation import classical_fdr

# Load preprocessed expression matrix and TF names
expression_matrix = pd.read_csv('preprocessed_expression_matrix.csv')
tf_names_file = 'genenametfs.tsv'
output_dir = '/path/to/output/directory'

# Compute the GRN
grn = compute_grn(expression_matrix, tf_names_file, output_dir)

# Perform classical FDR calculation
final_grn = classical_fdr(expression_matrix, tf_names_file, grn, output_dir)
print(final_grn.head())
```

#### Centroid Method

```python
import pandas as pd
from src.grn_computation import compute_grn
from src.fdr_calculation import fdr_centroid

# Load preprocessed expression matrix and TF names
expression_matrix = pd.read_csv('preprocessed_expression_matrix.csv')
tf_names_file = 'genenametfs.tsv'
output_dir = '/path/to/output/directory'

# Compute the GRN
grn = compute_grn(expression_matrix, tf_names_file, output_dir)

# Perform FDR calculation using the centroid method
final_grn = fdr_centroid(expression_matrix, tf_names_file, grn, output_dir)
print(final_grn.head())
```

#### Rotation Method

```python
import pandas as pd
from src.grn_computation import compute_grn
from src.fdr_calculation import fdr_rotation;

# Load preprocessed expression matrix and TF names
expression_matrix = pd.read_csv('preprocessed_expression_matrix.csv')
tf_names_file = 'genenametfs.tsv'
output_dir = '/path/to/output/directory';

# Compute the GRN
grn = compute_grn(expression_matrix, tf_names_file, output_dir);

# Perform FDR calculation using the rotation method
final_grn = fdr_rotation(expression_matrix, tf_names_file, grn, output_dir);
print(final_grn.head());
```

## Contributing

Contributions are welcome! If you would like to contribute to this package, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## Contact

For any inquiries or feedback, please contact:

- **Souptik Sen** - [souptik.sen@fau.de](mailto:souptik.sen@fau.de)
- GitHub: [My GitHub Profile](https://github.com/souptik69)

