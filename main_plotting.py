
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import seaborn as sns
from typing import Literal
from itertools import product


def annotate_mosaic(fig: plt.Figure, axd: dict[str, plt.Axes], fontsize: float | None = None):
    # Annotate subplot mosaic tiles with labels
    for label, ax in axd.items():
        # ax = fig.add_subplot(axd[label])
        # ax.annotate(label, xy=(0.1, 1.1), xycoords='axes fraction', ha='center', fontsize=16)
        # label physical distance to the left and up:
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(
            0.0,
            0.95,
            label,
            transform=ax.transAxes + trans,
            fontsize=fontsize,
            va='bottom',
            fontfamily='sans-serif',
            fontweight='bold'
        )


def plot_mae(
        results_df: pd.DataFrame,
        x_key: Literal['num_non_tfs', 'num_tfs'],
        col_names_mapping: dict[str, str],
        log10_x: bool = False,
        tissue_to_color: dict[str, tuple[float, float, float]] | None = None,
        exclude_tissues: list[str] | None = None,
        ax: plt.Axes = None,
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    col_names_mapping = col_names_mapping.copy()
    results_df = results_df.copy()
    results_df = results_df.rename(columns=col_names_mapping)

    if exclude_tissues is not None:
        keep_bool = [True if t not in exclude_tissues else False for t in results_df[col_names_mapping['tissue']]]
        results_df = results_df[keep_bool]

    if x_key == 'num_non_tfs':
        x_label = 'Number of non-TF Clusters'
    else:
        x_label = 'Number of TF Clusters'

    # Convert x-axis values to log10-scale if flag is set
    if log10_x:

        results_df[f'log10({x_label})'] = np.log10(results_df[col_names_mapping[x_key]])
        x_key = f'log10({x_label})'
        x_label = f'log10({x_label})'
        col_names_mapping[x_key] = x_key

    sns.lineplot(
        data=results_df,
        x=col_names_mapping[x_key],
        y=col_names_mapping['mae'],
        hue=col_names_mapping['tissue'],
        palette=tissue_to_color,
        ax=ax,
    )

    # Set log10-scale x-ticks
    if log10_x:
        # Remove existing x-ticks
        ax.set_xticks([])

        # Set custom x-ticks
        labels = np.array(list(range(1, 10, 1)) + list(range(10, 100, 10)) + list(range(100, 1001, 100)))
        pos = np.log10(labels)

        ax.set_xticks(pos)
        ax.set_xticklabels([str(l) if l in [10, 100, 1000] else '' for l in labels])

    ax.legend().remove()

    ax.set_xlabel(x_label)

    # ax.set_title('Mean Absolute Error (MAE)')

    return ax


def plot_mae_vs_n_samples(
        results_df: pd.DataFrame,
        col_names_mapping: dict[str, str],
        tissue_to_n_samples: dict[str, int],
        tissue_to_color: dict[str, tuple[float, float, float]] | None = None,
        ax: plt.Axes = None,
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    col_names_mapping = col_names_mapping.copy()
    results_df = results_df.copy()
    results_df = results_df.rename(columns=col_names_mapping)

    results_df = results_df[results_df[col_names_mapping['alpha_level']] == 0.01]

    results_df = results_df.groupby(
        [col_names_mapping['tissue']],
        as_index=False
    ).agg({
        col_names_mapping['mae']: 'mean',
    })

    results_df['Number of Samples'] = [
        tissue_to_n_samples[tissue] for tissue in results_df[col_names_mapping['tissue']]
    ]

    sns.scatterplot(
        data=results_df,
        x='Number of Samples',
        y=col_names_mapping['mae'],
        hue=col_names_mapping['tissue'],
        palette=tissue_to_color,
        ax=ax,
    )

    ax.legend().remove()

    ax.set_ylabel('Avg ' + col_names_mapping['mae'] + ' across Num. Clusters)')

    return ax


def plot_f1_vs_n_samples(
        results_df: pd.DataFrame,
        col_names_mapping: dict[str, str],
        tissue_to_n_samples: dict[str, int],
        tissue_to_color: dict[str, tuple[float, float, float]] | None = None,
        ax: plt.Axes = None,
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    col_names_mapping = col_names_mapping.copy()
    results_df = results_df.copy()
    results_df = results_df.rename(columns=col_names_mapping)

    results_df = results_df.groupby(
        [col_names_mapping['tissue'], col_names_mapping['alpha_level']],
        as_index=False
    ).agg({
        col_names_mapping['f1_score']: 'mean',
    })

    results_df['Number of Samples'] = [
        tissue_to_n_samples[tissue] for tissue in results_df[col_names_mapping['tissue']]
    ]

    sns.scatterplot(
        data=results_df,
        x='Number of Samples',
        y=col_names_mapping['f1_score'],
        hue=col_names_mapping['tissue'],
        palette=tissue_to_color,
        style=col_names_mapping['alpha_level'],
        ax=ax,
    )

    ax.legend().remove()

    ax.set_ylabel('Avg ' + col_names_mapping['f1_score'] + ' across Num. Clusters)')

    return ax


def plot_f1(
        results_df: pd.DataFrame,
        x_key: Literal['num_non_tfs', 'num_tfs'],
        col_names_mapping: dict[str, str],
        alpha: float = 0.01,
        log10_x: bool = False,
        tissue_to_color: dict[str, tuple[float, float, float]] | None = None,
        ax: plt.Axes = None,
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    col_names_mapping = col_names_mapping.copy()
    results_df = results_df.copy()
    results_df = results_df.rename(columns=col_names_mapping)

    results_df = results_df[results_df[col_names_mapping['alpha_level']] == alpha]

    if x_key == 'num_non_tfs':
        x_label = 'Number of non-TF Clusters'
    else:
        x_label = 'Number of TF Clusters'

    # Convert x-axis values to log10-scale if flag is set
    if log10_x:

        results_df[f'log10({x_label})'] = np.log10(results_df[col_names_mapping[x_key]])
        x_key = f'log10({x_label})'
        x_label = f'log10({x_label})'
        col_names_mapping[x_key] = x_key

    sns.lineplot(
        data=results_df,
        x=col_names_mapping[x_key],
        y=col_names_mapping['f1_score'],
        hue=col_names_mapping['tissue'],
        palette=tissue_to_color,
        ax=ax,
    )

    # Set log10-scale x-ticks
    if log10_x:
        # Remove existing x-ticks
        ax.set_xticks([])

        # Set custom x-ticks
        labels = np.array(list(range(1, 10, 1)) + list(range(10, 100, 10)) + list(range(100, 1001, 100)))
        pos = np.log10(labels)

        ax.set_xticks(pos)
        ax.set_xticklabels([str(l) if l in [10, 100, 1000] else '' for l in labels])

    ax.set_ylabel(f'F1 Score, alpha = {alpha}')

    ax.set_xlabel(x_label)

    ax.legend().remove()

    return ax


def plot_runtimes(
        results_df: pd.DataFrame,
        x_key: Literal['num_non_tfs', 'num_tfs'],
        col_names_mapping: dict[str, str],
        log10_x: bool = False,
        tissue_to_color: dict[str, tuple[float, float, float]] | None = None,
        ax: plt.Axes = None,
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    col_names_mapping = col_names_mapping.copy()
    results_df = results_df.copy()
    results_df = results_df.rename(columns=col_names_mapping)

    # Convert y-axis values from seconds to hours
    results_df[col_names_mapping['total_runtime']] = results_df[col_names_mapping['total_runtime']] / 3600

    if x_key == 'num_non_tfs':
        x_label = 'Number of non-TF Clusters'
    else:
        x_label = 'Number of TF Clusters'

    # Convert x-axis values to log10-scale if flag is set
    if log10_x:

        results_df[f'log10({x_label})'] = np.log10(results_df[col_names_mapping[x_key]])
        x_key = f'log10({x_label})'
        x_label = f'log10({x_label})'
        col_names_mapping[x_key] = x_key

    sns.lineplot(
        data=results_df,
        x=col_names_mapping[x_key],
        y=col_names_mapping['total_runtime'],
        hue=col_names_mapping['tissue'],
        palette=tissue_to_color,
        ax=ax,
    )

    # Set log10-scale x-ticks
    if log10_x:
        # Remove existing x-ticks
        ax.set_xticks([])

        # Set custom x-ticks
        labels = np.array(list(range(1,10, 1)) + list(range(10,100, 10)) + list(range(100,1001, 100)))
        pos = np.log10(labels)

        ax.set_xticks(pos)
        ax.set_xticklabels([str(l) if l in [10, 100, 1000] else '' for l in labels])

    ax.legend().remove()

    ax.set_title('Runtime')

    ax.set_xlabel(x_label)

    return ax


def plot_saved_runtime(
        results_df: pd.DataFrame,
        x_key: Literal['num_non_tfs', 'num_tfs'],
        col_names_mapping: dict[str, str],
        log10_x: bool = False,
        tissue_to_color: dict[str, tuple[float, float, float]] | None = None,
        ax: plt.Axes = None,
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    col_names_mapping = col_names_mapping.copy()
    results_df = results_df.copy()
    results_df = results_df.rename(columns=col_names_mapping)

    # Convert y-axis values from seconds to hours
    results_df[col_names_mapping['abs_time_saving']] = results_df[col_names_mapping['abs_time_saving']] / 3600

    if x_key == 'num_non_tfs':
        x_label = 'Number of non-TF Clusters'
    else:
        x_label = 'Number of TF Clusters'

    # Convert x-axis values to log10-scale if flag is set
    if log10_x:

        results_df[f'log10({x_label})'] = np.log10(results_df[col_names_mapping[x_key]])
        x_key = f'log10({x_label})'
        x_label = f'log10({x_label})'
        col_names_mapping[x_key] = x_key

    sns.lineplot(
        data=results_df,
        x=col_names_mapping[x_key],
        y=col_names_mapping['abs_time_saving'],
        hue=col_names_mapping['tissue'],
        palette=tissue_to_color,
        ax=ax,
    )

    # Set log10-scale x-ticks
    if log10_x:
        # Remove existing x-ticks
        ax.set_xticks([])

        # Set custom x-ticks
        labels = np.array(list(range(1,10, 1)) + list(range(10,100, 10)) + list(range(100,1001, 100)))
        pos = np.log10(labels)

        ax.set_xticks(pos)
        ax.set_xticklabels([str(l) if l in [10, 100, 1000] else '' for l in labels])

    ax.legend().remove()

    ax.set_title('Saved Runtime')

    ax.set_xlabel(x_label)

    return ax


def plot_speedup(
        results_df: pd.DataFrame,
        x_key: Literal['num_non_tfs', 'num_tfs'],
        col_names_mapping: dict[str, str],
        log10_x: bool = False,
        tissue_to_color: dict[str, tuple[float, float, float]] | None = None,
        ax: plt.Axes = None,
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    col_names_mapping = col_names_mapping.copy()
    results_df = results_df.copy()
    results_df = results_df.rename(columns=col_names_mapping)

    if x_key == 'num_non_tfs':
        x_label = 'Number of non-TF Clusters'
    else:
        x_label = 'Number of TF Clusters'

    # Convert x-axis values to log10-scale if flag is set
    if log10_x:

        results_df[f'log10({x_label})'] = np.log10(results_df[col_names_mapping[x_key]])
        x_key = f'log10({x_label})'
        x_label = f'log10({x_label})'
        col_names_mapping[x_key] = x_key

    sns.lineplot(
        data=results_df,
        x=col_names_mapping[x_key],
        y=col_names_mapping['rel_time_saving'],
        hue=col_names_mapping['tissue'],
        palette=tissue_to_color,
        ax=ax,
    )

    # Set log10-scale x-ticks
    if log10_x:
        # Remove existing x-ticks
        ax.set_xticks([])

        # Set custom x-ticks
        labels = np.array(list(range(1,10, 1)) + list(range(10,100, 10)) + list(range(100,1001, 100)))
        pos = np.log10(labels)

        ax.set_xticks(pos)
        ax.set_xticklabels([str(l) if l in [10, 100, 1000] else '' for l in labels])

    ax.legend().remove()

    ax.set_title('Speedup')

    ax.set_xlabel(x_label)

    return ax


def plot_saved_emissions(
        results_df: pd.DataFrame,
        x_key: Literal['num_non_tfs', 'num_tfs'],
        col_names_mapping: dict[str, str],
        tissue_to_color: dict[str, tuple[float, float, float]] | None = None,
        log10_x: bool = False,
        ax: plt.Axes = None,
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    col_names_mapping = col_names_mapping.copy()
    results_df = results_df.copy()
    results_df = results_df.rename(columns=col_names_mapping)

    if x_key == 'num_non_tfs':
        x_label = 'Number of non-TF Clusters'
    else:
        x_label = 'Number of TF Clusters'

    # Convert x-axis values to log10-scale if flag is set
    if log10_x:

        results_df[f'log10({x_label})'] = np.log10(results_df[col_names_mapping[x_key]])
        x_key = f'log10({x_label})'
        x_label = f'log10({x_label})'
        col_names_mapping[x_key] = x_key

    sns.lineplot(
        data=results_df,
        x=col_names_mapping[x_key],
        y=col_names_mapping['abs_emission_saving'],
        hue=col_names_mapping['tissue'],
        palette=tissue_to_color,
        ax=ax,
    )

    # Set log10-scale x-ticks
    if log10_x:
        # Remove existing x-ticks
        ax.set_xticks([])

        # Set custom x-ticks
        labels = np.array(list(range(1,10, 1)) + list(range(10,100, 10)) + list(range(100,1001, 100)))
        pos = np.log10(labels)

        ax.set_xticks(pos)
        ax.set_xticklabels([str(l) if l in [10, 100, 1000] else '' for l in labels])

    ax.legend().remove()

    # ax.set_title('Saved Emissions')

    ax.set_xlabel(x_label)

    return ax


def plot_runtime_meta(
        res_df: pd.DataFrame,
        x_key: Literal['num_non_tfs', 'num_tfs'],
        save_path: str
):

    res_df = res_df.copy()

    # Melt the dataframe to long format for F1 scores, add column with alpha level
    res_df = res_df.melt(
        id_vars=[
            "tissue", "num_non_tfs", "num_tfs", "mae", "abs_time_saving", "rel_time_saving", "abs_emission_saving",
            "rel_emission_saving", "total_runtime"
        ],
        value_vars=["f1_005", "f1_001"],
        var_name="alpha_level",
        value_name="f1_score"
    )
    res_df["alpha_level"] = res_df["alpha_level"].str.extract(r"f1_(\d+)").astype(float) / 1000

    old_to_new_col_names = {
        'tissue': 'Tissue',
        'num_non_tfs': 'Number of non-TF Clusters', 'num_tfs': 'Number of TF Clusters',
        'mae': 'MAE',
        'f1_score': 'F1 Score', 'alpha_level': 'Alpha',
        'abs_time_saving': 'Saved Time [hours]', 'rel_time_saving': 'Speedup Factor',
        'abs_emission_saving': 'Saved Emissions [gram CO2]', 'rel_emission_saving': 'Emissions Saved',
        'total_runtime': 'Runtime [hours]'
    }

    fig = plt.figure(figsize=(8, 6), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        """
    )

    # Define color mapping
    all_tissues = sorted(res_df['tissue'].unique())
    palette = sns.color_palette('tab20', n_colors=len(all_tissues))  # Or your preferred palette
    tissue_to_color = dict(zip(all_tissues, palette))

    plot_runtimes(
        results_df=res_df,
        x_key=x_key,
        col_names_mapping=old_to_new_col_names,
        log10_x=True,
        tissue_to_color=tissue_to_color,
        ax=axd['A']
    )
    plot_saved_runtime(
        results_df=res_df,
        x_key=x_key,
        col_names_mapping=old_to_new_col_names,
        log10_x=True,
        tissue_to_color=tissue_to_color,
        ax=axd['B']
    )
    plot_speedup(
        results_df=res_df,
        x_key=x_key,
        col_names_mapping=old_to_new_col_names,
        log10_x=True,
        tissue_to_color=tissue_to_color,
        ax=axd['C']
    )

    # Build a legend
    handles, labels = axd['A'].get_legend_handles_labels()
    axd['D'].legend(
        handles,
        labels,
        title='Tissue',
        frameon=False,
        ncol=2,
        loc='center'
    )
    axd['D'].axis('off')

    annotate_mosaic(fig=fig, axd=axd, fontsize=None)

    plt.savefig(save_path, dpi=300)
    plt.close('all')


def plot_performance_meta(
        res_df: pd.DataFrame,
        tissue_to_n_samples: dict[str, int],
        x_key: Literal['num_non_tfs', 'num_tfs'],
        save_path: str
):

    res_df = res_df.copy()

    # Melt the dataframe to long format for F1 scores, add column with alpha level
    res_df = res_df.melt(
        id_vars=[
            "tissue", "num_non_tfs", "num_tfs", "mae", "abs_time_saving", "rel_time_saving", "abs_emission_saving",
            "rel_emission_saving", "total_runtime"
        ],
        value_vars=["f1_005", "f1_001"],
        var_name="alpha_level",
        value_name="f1_score"
    )
    res_df["alpha_level"] = res_df["alpha_level"].str.extract(r"f1_(\d+)").astype(float) / 100

    old_to_new_col_names = {
        'tissue': 'Tissue',
        'num_non_tfs': 'Number of non-TF Clusters', 'num_tfs': 'Number of TF Clusters',
        'mae': 'MAE',
        'f1_score': 'F1 Score', 'alpha_level': 'Alpha',
        'abs_time_saving': 'Saved Time [hours]', 'rel_time_saving': 'Speedup Factor',
        'abs_emission_saving': 'Saved Emissions [gram CO2]', 'rel_emission_saving': 'Emissions Saved',
        'total_runtime': 'Runtime [hours]'
    }

    fig = plt.figure(figsize=(8, 9), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        EF
        """
    )

    # Define color mapping
    all_tissues = sorted(res_df['tissue'].unique())
    palette = sns.color_palette('tab20', n_colors=len(all_tissues))
    tissue_to_color = dict(zip(all_tissues, palette))

    plot_mae_vs_n_samples(
        results_df=res_df,
        col_names_mapping=old_to_new_col_names,
        tissue_to_n_samples=tissue_to_n_samples,
        tissue_to_color=tissue_to_color,
        ax=axd['A']
    )

    plot_mae(
        results_df=res_df,
        x_key=x_key,
        col_names_mapping=old_to_new_col_names,
        log10_x=True,
        exclude_tissues=['Fallopian_Tube', 'Bladder', 'Cervix_Uteri'],
        tissue_to_color=tissue_to_color,
        ax=axd['B']
    )

    plot_f1_vs_n_samples(
        results_df=res_df,
        col_names_mapping=old_to_new_col_names,
        tissue_to_n_samples=tissue_to_n_samples,
        tissue_to_color=tissue_to_color,
        ax=axd['C']
    )

    # Build a legend
    handles, labels = axd['C'].get_legend_handles_labels()
    axd['D'].legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc='center'
    )
    axd['D'].axis('off')

    plot_f1(
        results_df=res_df,
        x_key=x_key,
        col_names_mapping=old_to_new_col_names,
        alpha=0.05,
        log10_x=True,
        tissue_to_color=tissue_to_color,
        ax=axd['E']
    )

    plot_f1(
        results_df=res_df,
        x_key=x_key,
        col_names_mapping=old_to_new_col_names,
        alpha=0.01,
        log10_x=True,
        tissue_to_color=tissue_to_color,
        ax=axd['F']
    )

    annotate_mosaic(fig=fig, axd=axd, fontsize=None)

    plt.savefig(save_path, dpi=300)
    plt.close('all')


def plot_emission_meta(
        res_df: pd.DataFrame,
        x_key: Literal['num_non_tfs', 'num_tfs'],
        save_path: str
):

    res_df = res_df.copy()

    # Melt the dataframe to long format for F1 scores, add column with alpha level
    res_df = res_df.melt(
        id_vars=[
            "tissue", "num_non_tfs", "num_tfs", "mae", "abs_time_saving", "rel_time_saving", "abs_emission_saving",
            "rel_emission_saving", "total_runtime"
        ],
        value_vars=["f1_005", "f1_001"],
        var_name="alpha_level",
        value_name="f1_score"
    )
    res_df["alpha_level"] = res_df["alpha_level"].str.extract(r"f1_(\d+)").astype(float) / 1000

    old_to_new_col_names = {
        'tissue': 'Tissue',
        'num_non_tfs': 'Number of non-TF Clusters', 'num_tfs': 'Number of TF Clusters',
        'mae': 'MAE',
        'f1_score': 'F1 Score', 'alpha_level': 'Alpha',
        'abs_time_saving': 'Saved Time [hours]', 'rel_time_saving': 'Speedup Factor',
        'abs_emission_saving': 'Saved Emissions [gram CO2]', 'rel_emission_saving': 'Emissions Saved',
        'total_runtime': 'Runtime [hours]'
    }

    fig = plt.figure(figsize=(8, 3), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        """
    )

    # Define color mapping
    all_tissues = sorted(res_df['tissue'].unique())
    palette = sns.color_palette('tab20', n_colors=len(all_tissues))  # Or your preferred palette
    tissue_to_color = dict(zip(all_tissues, palette))

    plot_saved_emissions(
        results_df=res_df,
        x_key=x_key,
        col_names_mapping=old_to_new_col_names,
        log10_x=True,
        tissue_to_color=tissue_to_color,
        ax=axd['A']
    )

    # Build a legend
    handles, labels = axd['A'].get_legend_handles_labels()
    axd['B'].legend(
        handles,
        labels,
        title='Tissue',
        frameon=False,
        ncol=2,
        loc='center'
    )
    axd['B'].axis('off')

    annotate_mosaic(fig=fig, axd=axd, fontsize=None)

    plt.savefig(save_path, dpi=300)
    plt.close('all')


def main_plot_rt_perf_emissions():
    res_dir = './results/gtex_up_to_breast'

    save_dir = './results/plots'
    os.makedirs(save_dir, exist_ok=True)

    for mode in ['nontf_medoid', 'nontf_random', 'tf_medoid', 'tf_random']:
        x_key = 'num_non_tfs' if mode in {'nontf_medoid', 'nontf_random'} else 'num_tfs'

        mode_to_file = {
            'nontf_medoid': 'approximate_fdr_grns_medoid_nonTF_results.csv',
            'nontf_random': 'approximate_fdr_grns_random_nonTF_results.csv',
            'tf_medoid': 'approximate_fdr_grns_medoid_TF_results.csv',
            'tf_random': 'approximate_fdr_grns_random_TF_results.csv',
        }

        res_df = pd.read_csv(os.path.join(res_dir, mode_to_file[mode]))

        # Load sample sizes dictionary.
        with open(os.path.join(res_dir, 'samples_per_tissue.pkl'), 'rb') as f:
            tissue_to_n_samples = pickle.load(f)

        plot_runtime_meta(
            res_df=res_df,
            x_key=x_key,
            save_path=os.path.join(save_dir, f'runtime_{mode}.png'),
        )

        plot_performance_meta(
            res_df=res_df,
            tissue_to_n_samples=tissue_to_n_samples,
            x_key=x_key,
            save_path=os.path.join(save_dir, f'performance_{mode}.png'))

        plot_emission_meta(
            res_df=res_df,
            x_key=x_key,
            save_path=os.path.join(save_dir, f'emission_{mode}.png')
        )


def plot_pval_gt_vs_approx(
        grn: pd.DataFrame,
        ax: plt.Axes | None = None,
):

    pvals_gt = grn['pvalue_gt'].to_numpy()
    pvals_approx = grn['pvalue_approx'].to_numpy()
    importances = grn['importance'].to_numpy()

    plot_df = pd.DataFrame({
        'GT p-value': pvals_gt,
        'Approx p-value': pvals_approx,
        'Importance': importances
    })

    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    # Create the scatterplot
    sns.scatterplot(
        data=plot_df,
        x='GT p-value',
        y='Approx p-value',
        hue='Importance',
        palette='magma',
        ax=ax,
        edgecolor=None,
        s=0.5,
    )

    # Add y = x diagonal line
    min_val = min(plot_df['GT p-value'].min(), plot_df['Approx p-value'].min(), 0)
    max_val = max(plot_df['GT p-value'].max(), plot_df['Approx p-value'].max())
    ax.plot([min_val, max_val], [min_val, max_val], ls='--', color='gray', linewidth=1)

    # ax.set_xscale('log')
    # ax.set_yscale('log')

    ax.legend(loc='upper right')

    return ax


def plot_pval_hists(
    grn: pd.DataFrame,
    bins: int = 50,
    ax: plt.Axes | None = None
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    sns.histplot(
        grn['pvalue_gt'],
        bins=bins,
        color='tab:blue',
        label='GT p-value',
        kde=False,
        ax=ax,
        stat='density',
        alpha=0.5
    )

    sns.histplot(
        grn['pvalue_approx'],
        bins=bins,
        color='tab:orange',
        label='Approx p-value',
        kde=False,
        ax=ax,
        stat='density',
        alpha=0.5
    )

    ax.set_xlabel("P-value")
    ax.set_ylabel("Density")
    ax.legend(loc='upper right')

    return ax


def plot_pval_vs_importance(
        grn: pd.DataFrame,
        mode: Literal['GT', 'Approx'],
        ax: plt.Axes | None = None,
):

    if mode == 'GT':
        pvals = -np.log10(grn['pvalue_gt'].to_numpy())
    else:
        pvals = -np.log10(grn['pvalue_approx'].to_numpy())

    importances = grn['importance'].to_numpy()

    plot_df = pd.DataFrame({
        '-log10(P-value)': pvals,
        'Importance': importances
    })

    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    # Create the scatterplot
    sns.scatterplot(
        data=plot_df,
        x='Importance',
        y='-log10(P-value)',
        ax=ax,
        edgecolor=None,
        s=0.5,
    )

    # Add y = x diagonal line
    min_val = min(plot_df['-log10(P-value)'].min(), plot_df['Importance'].min(), 0)
    max_val = max(plot_df['-log10(P-value)'].max(), plot_df['Importance'].max())
    ax.plot([min_val, max_val], [min_val, max_val], ls='--', color='gray', linewidth=1)

    # ax.set_xscale('log')
    # ax.set_yscale('log')

    # ax.legend(loc='upper right')

    return ax


def main_plot_pval_gt_vs_approx():

    res_dir = './results/grn_files'

    save_dir = './results/plots'
    os.makedirs(save_dir, exist_ok=True)

    importance_percentile = None

    fig = plt.figure(figsize=(8, 3), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        ABC
        """
    )

    plot_labels = list('ABC')

    for tissue, plot_label in zip(['breast', 'kidney', 'testis'], plot_labels):  # 'kidney', 'testis'

        grn_gt = pd.read_feather(os.path.join(res_dir, f'aggregated_groundtruth_{tissue}.feather'))
        grn_approx = pd.read_feather(os.path.join(res_dir, f'fdr_grn_{tissue}_nontf_100_numtf_-1.feather'))

        grn_merged = pd.merge(
            grn_gt[['TF', 'target', 'pvalue', 'importance']],
            grn_approx[['TF', 'target', 'pvalue']],
            on=['TF', 'target'],
            suffixes=('_gt', '_approx')
        )

        if importance_percentile is not None:
            importances = grn_merged['importance'].to_numpy()
            threshold = np.percentile(importances, importance_percentile)
            mask = importances >= threshold
            grn_merged = grn_merged[mask]

        print(grn_merged)

        plot_pval_gt_vs_approx(
            grn=grn_merged,
            ax=axd[plot_label],
        )

        axd[plot_label].set_title(tissue.capitalize())

    plt.savefig(os.path.join(save_dir, 'pvals_gt_vs_approx.png'), dpi=300)
    plt.close('all')


def main_plot_pval_dist():

    res_dir = './results/grn_files'

    save_dir = './results/plots'
    os.makedirs(save_dir, exist_ok=True)

    importance_percentile = None

    fig = plt.figure(figsize=(8, 3), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        ABC
        """
    )

    plot_labels = list('ABC')

    for tissue, plot_label in zip(['breast', 'kidney', 'testis'], plot_labels):  # 'kidney', 'testis'

        grn_gt = pd.read_feather(os.path.join(res_dir, f'aggregated_groundtruth_{tissue}.feather'))
        grn_approx = pd.read_feather(os.path.join(res_dir, f'fdr_grn_{tissue}_nontf_100_numtf_-1.feather'))

        grn_merged = pd.merge(
            grn_gt[['TF', 'target', 'pvalue', 'importance']],
            grn_approx[['TF', 'target', 'pvalue']],
            on=['TF', 'target'],
            suffixes=('_gt', '_approx')
        )

        print(grn_merged)

        if importance_percentile is not None:
            importances = grn_merged['importance'].to_numpy()
            threshold = np.percentile(importances, importance_percentile)
            mask = importances >= threshold
            grn_merged = grn_merged[mask]

            print(grn_merged)

        plot_pval_hists(grn=grn_merged, bins=100, ax=axd[plot_label])

        axd[plot_label].set_title(tissue.capitalize())

    plt.savefig(os.path.join(save_dir, 'pvals_histogram.png'), dpi=300)
    plt.close('all')


def main_plot_pval_vs_importance():

    res_dir = './results/grn_files'

    save_dir = './results/plots'
    os.makedirs(save_dir, exist_ok=True)

    importance_percentile = None

    fig = plt.figure(figsize=(8, 9), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        EF
        """
    )

    plot_labels = list('ABCDEF')

    plot_combinations = list(product(['breast', 'kidney', 'testis'], ['GT', 'Approx']))

    for (tissue, mode), plot_label in zip(plot_combinations, plot_labels):  # 'kidney', 'testis'

        grn_gt = pd.read_feather(os.path.join(res_dir, f'aggregated_groundtruth_{tissue}.feather'))
        grn_approx = pd.read_feather(os.path.join(res_dir, f'fdr_grn_{tissue}_nontf_100_numtf_-1.feather'))

        grn_merged = pd.merge(
            grn_gt[['TF', 'target', 'pvalue', 'importance']],
            grn_approx[['TF', 'target', 'pvalue']],
            on=['TF', 'target'],
            suffixes=('_gt', '_approx')
        )

        print(grn_merged)

        if importance_percentile is not None:
            importances = grn_merged['importance'].to_numpy()
            threshold = np.percentile(importances, importance_percentile)
            mask = importances >= threshold
            grn_merged = grn_merged[mask]

            print(grn_merged)

        plot_pval_vs_importance(grn=grn_merged, mode=mode, ax=axd[plot_label])

        axd[plot_label].set_title(f'{tissue.capitalize()} | {mode.capitalize()}')

    plt.savefig(os.path.join(save_dir, 'pvals_vs_importance.png'), dpi=300)
    plt.close('all')




if __name__ == '__main__':

    # main_plot_rt_perf_emissions()

    main_plot_pval_gt_vs_approx()

    main_plot_pval_dist()

    main_plot_pval_vs_importance()

    print('done')


# Todo:
#  - Same clustering strategy as before: change number of non TF representative clusters to 10, run TFs clustered + random analysis again
#  - Replace Wasserstein clustering of target space by voronoi partitioning (PCA + K-means) -> Hypothesis: Just need to represent input space
#  - Structure-preserving clustering for TFs (e.g. K-means, correlation based), Wasserstein-based clustering of target space (TF + non TF)
