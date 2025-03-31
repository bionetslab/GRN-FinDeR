
import pandas as pd
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
        c='b', label=f'{metric}, thresh. {fdr_threshold}', linewidth=1.0, marker='o'
    )
    ax.set_ylabel(metric)
    ax.set_xlabel('n clusters')
    if n_tfs is not None:
        ax.axvline(x=n_tfs, color=line_color, linestyle='--', label='number of TFs')
    plt.legend()

    return ax













