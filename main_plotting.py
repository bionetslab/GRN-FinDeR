
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import seaborn as sns


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

    # Convert x-axis values to log10-scale if flag is set
    x_key = 'num_non_tfs'
    if log10_x:
        results_df['log10(Number of non-TF Clusters)'] = np.log10(results_df[col_names_mapping[x_key]])
        x_key = 'log10(Number of non-TF Clusters)'
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

    ax.set_title('Mean Absolute Error (MAE)')

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

    print(results_df)

    results_df['Number of Samples'] = [
        tissue_to_n_samples[tissue] for tissue in results_df[col_names_mapping['tissue']]
    ]

    sns.scatterplot(
        data=results_df,
        x='Number of Samples',
        y=col_names_mapping['mae'],
        hue=col_names_mapping['tissue'],
        palette=tissue_to_color,
        # style=col_names_mapping['alpha_level'],
        ax=ax,
    )

    ax.legend().remove()

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

    print(results_df)

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

    return ax

def plot_f1(
        results_df: pd.DataFrame,
        col_names_mapping: dict[str, str],
        ax: plt.Axes = None,
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    return ax


def plot_runtimes(
        results_df: pd.DataFrame,
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

    # Convert x-axis values to log10-scale if flag is set
    x_key = 'num_non_tfs'
    if log10_x:
        results_df['log10(Number of non-TF Clusters)'] = np.log10(results_df[col_names_mapping[x_key]])
        x_key = 'log10(Number of non-TF Clusters)'
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

    return ax


def plot_saved_runtime(
        results_df: pd.DataFrame,
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

    # Convert x-axis values to log10-scale if flag is set
    x_key = 'num_non_tfs'
    if log10_x:
        results_df['log10(Number of non-TF Clusters)'] = np.log10(results_df[col_names_mapping[x_key]])
        x_key = 'log10(Number of non-TF Clusters)'
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

    return ax


def plot_speedup(
        results_df: pd.DataFrame,
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

    # Convert x-axis values to log10-scale if flag is set
    x_key = 'num_non_tfs'
    if log10_x:
        results_df['log10(Number of non-TF Clusters)'] = np.log10(results_df[col_names_mapping[x_key]])
        x_key = 'log10(Number of non-TF Clusters)'
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

    return ax


def plot_emissions(
        results_df: pd.DataFrame,
        col_names_mapping: dict[str, str],
        ax: plt.Axes = None,
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    return ax


def plot_runtime_meta():

    res_dir = './results/gtex_up_to_breast'

    res_df = pd.read_csv(os.path.join(res_dir, 'approximate_fdr_grns_medoid_results.csv'))

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
        'abs_emission_saving': 'Saved Emissions', 'rel_emission_saving': 'Emissions Saved',
        'total_runtime': 'Runtime [hours]'
    }

    save_dir = './results/plots'
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(8, 6), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        """
    )

    # Define color mapping
    all_tissues = sorted(res_df['tissue'].unique())
    palette = sns.color_palette('husl', n_colors=len(all_tissues))  # Or your preferred palette
    tissue_to_color = dict(zip(all_tissues, palette))

    plot_runtimes(
        results_df=res_df,
        col_names_mapping=old_to_new_col_names,
        log10_x=True,
        tissue_to_color=tissue_to_color,
        ax=axd['A']
    )
    plot_saved_runtime(
        results_df=res_df,
        col_names_mapping=old_to_new_col_names,
        log10_x=True,
        tissue_to_color=tissue_to_color,
        ax=axd['B']
    )
    plot_speedup(
        results_df=res_df,
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

    plt.savefig(os.path.join(save_dir, 'runtime.png'))
    plt.close('all')


def plot_performance_meta():

    res_dir = './results/gtex_up_to_breast'

    res_df = pd.read_csv(os.path.join(res_dir, 'approximate_fdr_grns_medoid_results.csv'))

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
        'abs_emission_saving': 'Saved Emissions', 'rel_emission_saving': 'Emissions Saved',
        'total_runtime': 'Runtime [hours]'
    }

    # Load sample sizes dictionary.
    with open(os.path.join(res_dir, 'samples_per_tissue.pkl'), 'rb') as f:
        tissue_to_n_samples = pickle.load(f)

    save_dir = './results/plots'
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(8, 6), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        """
    )

    # Define color mapping
    all_tissues = sorted(res_df['tissue'].unique())
    palette = sns.color_palette('husl', n_colors=len(all_tissues))  # Or your preferred palette
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

    plt.savefig(os.path.join(save_dir, 'performance.png'))
    plt.close('all')





if __name__ == '__main__':

    plot_runtime_meta()

    plot_performance_meta()

    print('done')
