
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from typing import Union, List, Literal
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

def compute_evaluation_metrics(
        grn: pd.DataFrame,
        fdr_thresholds: Union[List[float], None] = None,
        n_clusters: Union[List[int], None] = None,
):
    """
    Computes evaluation metrics comparing ground truth gene regulatory network (GRN) p-values
    to approximate GRN results generated with different cluster counts.

    Parameters:
    ----------
    grn : pd.DataFrame
        A DataFrame containing ground truth and approximate GRN results. Expected columns:
        - 'TF': Transcription factor name
        - 'target': Target gene name
        - 'importance': Edge importance score
        - 'count': Edge count in ground truth
        - 'p_value': p-value in ground truth
        - For approximations with n clusters, columns named 'count_{n}' and 'pvalue_{n}' should be present

    fdr_thresholds : list of float, optional
        List of false discovery rate (FDR) thresholds to evaluate significance (default is [0.01, 0.05])

    n_clusters : list of int, optional
        List of cluster counts to evaluate. If None, inferred from column names with the pattern 'count_{n}'

    Returns:
    -------
    pd.DataFrame
        A DataFrame with evaluation metrics as rows ('mse', 'acc', 'prec', 'rec', 'f1') and
        columns for each combination of cluster count and FDR threshold.
    """

    if fdr_thresholds is None:
        fdr_thresholds = [0.01, 0.05]

    if n_clusters is None:
        n_clusters = [
            int(col.removeprefix('count_')) for col in grn.columns if col not in [
                'TF', 'target', 'importance', 'count', 'p_value'
            ] and col.startswith('count_')
        ]

    res_df = pd.DataFrame(index=['mse', 'acc', 'prec', 'rec', 'f1'])

    for threshold in fdr_thresholds:
        ground_truth_p_vals = grn['p_value'].to_numpy()
        ground_truth_sig_bool = ground_truth_p_vals <= threshold
        for n_clust in n_clusters:
            approx_p_vals = grn[f'pvalue_{n_clust}'].to_numpy()
            approx_sig_bool = approx_p_vals <= threshold

            mse = mean_squared_error(ground_truth_p_vals, approx_p_vals)
            acc = accuracy_score(ground_truth_sig_bool, approx_sig_bool)
            prec = precision_score(ground_truth_sig_bool, approx_sig_bool)
            rec = recall_score(ground_truth_sig_bool, approx_sig_bool)
            f1 = f1_score(ground_truth_sig_bool, approx_sig_bool)

            res_df[f'nclust{n_clust}_fdrthresh{threshold}'] = [mse, acc, prec, rec, f1]

    return res_df


def plot_metric(
        res_df: pd.DataFrame,
        fdr_threshold: float = 0.01,
        metric: Literal['mse', 'acc', 'prec', 'rec', 'f1'] = 'mse',
        n_tfs: Union[int, None] = None,
        line_color: str = 'b',
        ax: Union[plt.Axes, None] = None,
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots()

    # Subset dataframe
    cols = [col for col in res_df.columns if col.endswith(str(fdr_threshold))]
    res_df_subset = res_df[cols]

    if len(cols) == 0:
        raise ValueError(f"The 'fdr_threshold' {fdr_threshold} is not present in the res_df.")

    n_clusters = [int(col.removeprefix('nclust').removesuffix(f'_fdrthresh{fdr_threshold}')) for col in cols]

    ax.plot(
        n_clusters, res_df_subset.loc[metric],
        c=line_color, label=f'{metric}, thresh. {fdr_threshold}', linewidth=1.0, marker='o'
    )
    ax.set_ylabel(metric)
    ax.set_xlabel('n clusters')
    if n_tfs is not None:
        ax.axvline(x=n_tfs, color='r', linestyle='--', label='number of TFs')
    plt.legend()

    return ax


class DebugDataSuite:
    def __init__(self, cache_dir: str, verbosity: int = 0):
        super(DebugDataSuite, self).__init__()
        self.cache_dir = cache_dir
        self.verbosity = verbosity

        self.adata_ = None
        self.expression_mat_ = None

    def load_and_preprocess(self):
        # Set datasetdir variable for scanpy
        os.makedirs(self.cache_dir, exist_ok=True)
        sc.settings.datasetdir = self.cache_dir

        # Download raw pbmc3k data set (https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc3k.html)
        self.adata_ = sc.datasets.pbmc3k()

        if self.verbosity >= 1:
            print(f'# ### adata raw:\n{self.adata_}\n')

        # Preprocess according to: https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering-2017.html
        self.adata_.var_names_make_unique()
        sc.pp.filter_cells(self.adata_, min_genes=200)
        sc.pp.filter_genes(self.adata_, min_cells=3)

        # Annotate the group of mitochondrial genes as "mt"
        self.adata_.var['mt'] = self.adata_.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(
            self.adata_, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True
        )

        self.adata_ = self.adata_[self.adata_.obs.n_genes_by_counts < 2500, :].copy()
        self.adata_ = self.adata_[self.adata_.obs.pct_counts_mt < 5, :].copy()

        sc.pp.log1p(self.adata_)

        sc.pp.highly_variable_genes(self.adata_, min_mean=0.0125, max_mean=3, min_disp=0.5)

        self.adata_.raw = self.adata_.copy()

        self.adata_ = self.adata_[:, self.adata_.var.highly_variable].copy()

        sc.pp.regress_out(self.adata_, ['total_counts', 'pct_counts_mt'])

        self.adata_.write_h5ad(os.path.join(self.cache_dir, 'pbmc3k_preprocessed.h5ad'))

        if self.verbosity >= 1:
            print(f'# ### adata preprocessed:\n{self.adata_}\n')


    def downsample_scale(self, fraction_cells: float, fraction_genes: float, seed: int = 42):

        if fraction_cells < 1.0:

            if self.verbosity >= 1:
                print(f'# ### n cells before downsampling:\n{self.adata_.n_obs}\n')

            self.adata_ = DebugDataSuite.downsample_anndata(
                adata=self.adata_, fraction=fraction_cells, axis=0, seed=seed
            )

            if self.verbosity >= 1:
                print(f'# ### n cells after downsampling:\n{self.adata_.n_obs}\n')

        if fraction_genes < 1.0:
            if self.verbosity >= 1:
                print(f'# ### n genes before downsampling:\n{self.adata_.n_vars}\n')

            self.adata_ = DebugDataSuite.downsample_anndata(
                adata=self.adata_, fraction=fraction_genes, axis=1, seed=seed
            )

            if self.verbosity >= 1:
                print(f'# ### n genes after downsampling:\n{self.adata_.n_vars}\n')

        sc.pp.scale(self.adata_, max_value=fraction_cells)

        self.adata_.write_h5ad(os.path.join(self.cache_dir, 'pbmc3k_prepr_downsampled.h5ad'))

        self.expression_mat_ = self.adata_.to_df()

        self.expression_mat_.to_csv(os.path.join(self.cache_dir, 'pbmc3k_prepr_downsampled.csv'))


    @staticmethod
    def downsample_anndata(adata: sc.AnnData, fraction: float, axis: int, seed: int = 42):

        np.random.seed(seed)

        n_start = adata.shape[axis]

        n_sample = int(n_start * fraction)

        indices = np.random.choice(n_start, size=n_sample, replace=False)

        if axis == 0:
            adata = adata[indices, :].copy()
        elif axis == 1:
            adata = adata[:, indices].copy()
        else:
            raise ValueError(f'axis must be 0 or 1, but got {axis}')

        return adata




