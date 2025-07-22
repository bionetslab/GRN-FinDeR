# GRN-FinDeR: Efficient P-value Computation for Gene-Regulatory Networks

## Overview

We here provide the software implementing our novel strategy for reducing the computational cost of tree-based GRN inference while maintaining reasonable runtime. The method clusters genes to create a fixed number of background distributions, enabling the computation of significance values for multiple genes simultaneously and reducing computational cost linearly.

Our software extends the popular GRNBoost2/arboreto package and can be used as an add-on. It outputs a list of regulatory interactions, their importance scores, and empirical p-values, computed up to a user-defined significance threshold.



![Schematic of the workflow](/img/flowchart_grn_finder.png)

## Installation

As of now, we recommend installing the arboreto package separately (https://github.com/aertslab/arboreto) and copying the content of this repos `arboreto/` into arboreto's `arboreto/` directory:
```
cp -R ./arboreto/* <PATH_TO_ARBORETO_REPO>/arboreto
```

## Usage examples

An example call to our FDR control without an input GRN includes the following steps:

```python
import pandas as pd
from arboreto.algo import grnboost2_fdr

# Load expression matrix - in this case simulate one.
exp_data = np.random.randn(100, 10)
exp_df = pd.DataFrame(data, columns=columns)

# Run approximate FDR control.
fdr_grn = grnboost2_fdr(
            expression_data=exp_df,
            cluster_representative_mode="random",
            num_target_clusters=5,
            num_tf_clusters=-1
)
```

## Parameter Details & IO Format

A more detailed description of the function parameters of our `grnboost2_fdr` function can be found below as well as in the docstring of `arboreto/algo.py`:
- expression_data: Expression matrix as pandas dataframe with genes as columns, samples as columns.
- **cluster_representative_mode** (str): How to draw representatives from gene clusters, either randomly ('random'),
            or always take medoid of cluster ('medoid'). In case of full FDR, i.e. without using gene clusters, use
            mode 'all_genes'.
- **num_target_clusters** (int, optional): Number of clusters for target genes; set to -1 if no clustering desired.
- **num_tf_clusters** (int, optional): Number of clusters for TFs; set to -1 if no clustering is desired.
- **target_cluster_mode** (str, optional): How to cluster targets, either based on Wasserstein distance('wasserstein'),
            or using KMedoids clustering on PCAs of expression matrix ('kmeans').
- **tf_cluster_mode** (str, optional): How to cluster TFs, either by using Pearson correlation as distance ('correlation'),
            or using Wasserstein distance ('wasserstein').
- **input_grn** (pd.DataFrame, optional): If an input GRN to perform FDR control on is given, pass this here as dataframe
            with columns 'TF', 'target', 'importance'. Otherwise an input GRN is inferred from the given expression data.
- **target_subset** (list, optional): Subset of target genes to perform FDR control on.
- tf_names: Optional list of transcription factors only used for input GRN inference.
            If None or 'all', the list of gene_names will be used.
- client_or_address: one of:
           * None or 'local': a new Client(LocalCluster()) will be used to perform the computation.
           * string address: a new Client(address) will be used to perform the computation.
           * a Client instance: the specified Client instance will be used to perform the computation.
- early_stop_window_length: early stop window length. Default 25.
- seed: optional random seed for the regressors. Default None.
- verbose: print info.
- param num_permutations: Number of permutations to run for empirical P-value computation.
- output_dir: Directory where to write intermediate results to.

All parameters that have been added / changed compared to the grnboost2() function in `arboreto/algo.py` are listed above in **bold** font. 

Our grnboost2_fdr() function returns a pandas DataFrame with columns `TF, target, importance, pvalue` representing the FDR-controlled gene regulatory links. In case you have given a GRN as input, only the `pvalue` column is added to the input dataframe, otherwise a new GRN is inferred from the given expression data.
