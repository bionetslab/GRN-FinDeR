
import os
import copy
import yaml
import argparse
import time
import pandas as pd

from codecarbon import OfflineEmissionsTracker
from arboreto.algo import grnboost2_fdr


def generate_configs(
        gtex_dir: str,
        num_clusters_non_tfs: list[int],
        num_clusters_tfs: list[int] | None = None,
        config_dir: str | None = None,
        results_dir: str | None = None,
        verbosity: int = 0
) -> None:

    if config_dir is None:
        config_dir = os.getcwd()
    os.makedirs(config_dir, exist_ok=True)

    if results_dir is None:
        results_dir = os.path.join(os.getcwd(), 'results')

    if num_clusters_tfs is None:
        num_clusters_tfs = [-1, ]

    config = dict()

    tissue_dirs = sorted(os.listdir(gtex_dir))

    for tissue_dir in tissue_dirs:

        if verbosity > 0:
            print(f'# ###### Tissue {tissue_dir} ###### #')

        tissue_dir_path = os.path.join(gtex_dir, tissue_dir)

        expression_mat_filename = f'{tissue_dir}.tsv'
        input_grn_filename = f'{tissue_dir}_input_grn.csv'

        config['tissue_name'] = tissue_dir
        config['tissue_data_path'] = tissue_dir_path
        config['expression_mat_filename'] = expression_mat_filename
        config['results_dir'] = results_dir
        config['input_grn_filename'] = input_grn_filename

        for i in num_clusters_non_tfs:
            for j in num_clusters_tfs:

                param_combination_config = copy.deepcopy(config)

                param_combination_config['num_clusters_non_tfs'] = i
                param_combination_config['num_clusters_tfs'] = j

                config_filename = f'{config['tissue_name']}_config_{i}_{j}.yaml'
                save_path = os.path.join(config_dir, config_filename)

                with open(save_path, 'w') as f:
                    yaml.dump(param_combination_config, f, default_flow_style=False)


def compute_approximate_fdr(config: dict, verbosity: int = 0) -> pd.DataFrame:

    # Load the expression data
    tissue_name = config['tissue_name']  # Same as tissue_dir
    tissue_dir_path = config['tissue_data_path']  # gtex_dir + tissue_dir (where expression matrix is saved)

    # Load the expression data
    expression_mat_filename = config['expression_mat_filename']
    expression_mat = pd.read_csv(os.path.join(tissue_dir_path, expression_mat_filename), sep='\t', index_col=0)

    # Load the input GRN
    results_dir = config['results_dir']
    grn_path = os.path.join(results_dir, tissue_name, config['input_grn_filename'])
    input_grn = pd.read_csv(grn_path)


    num_clusters_non_tfs = config['num_clusters_non_tfs']
    num_clusters_tfs = config['num_clusters_tfs']

    if verbosity > 0:
        print(
            f'# ### Computing approximate FDR for tissue: {tissue_name}, '
            f'num non-TF clusters: {num_clusters_non_tfs}, num TF clusters: {num_clusters_tfs}'
        )

    # Create subdir for saving
    fdr_mode = 'random'
    save_dir = os.path.join(results_dir, tissue_name, f'approximate_fdr_grns_{fdr_mode}')
    os.makedirs(save_dir, exist_ok=True)

    emissions_file = os.path.join(save_dir, f'emissions_nontf_{num_clusters_non_tfs}_numtf_{num_clusters_tfs}.csv')

    with OfflineEmissionsTracker(
            country_iso_code="DEU", output_file=emissions_file, log_level='error', measure_power_secs=600
    ) as tracker:

        st = time.time()
        fdr_grn = grnboost2_fdr(
            expression_data=expression_mat,
            cluster_representative_mode=fdr_mode,
            num_non_tf_clusters=num_clusters_non_tfs,
            num_tf_clusters=num_clusters_tfs,
            input_grn=input_grn,
            tf_names=None,
            target_subset=None,
            client_or_address = 'local',
            seed=42,
            verbose=False,
            num_permutations=1000,
            output_dir=None
        )
        et = time.time()

    elapsed_time = et - st

    time_file = os.path.join(save_dir, f'time_nontf_{num_clusters_non_tfs}_numtf_{num_clusters_tfs}.txt')
    with open(time_file, 'w') as f:
        f.write(str(elapsed_time))

    # Save fdr controlled grn for target batch
    fdr_grn.to_csv(
        os.path.join(save_dir, f'fdr_grn_nontf_{num_clusters_non_tfs}_numtf_{num_clusters_tfs}.csv'),
        index=False
    )

    return fdr_grn


if __name__ == '__main__':

    # Set flag whether to do input GRN computation and config generation or run classical FDR control
    fdr = True

    if not fdr:

        gtex_path = './data/gtex_tissues_preprocessed'
        alternate_gtex_path = f'/home/woody/iwbn/iwbn106h/gtex'

        if not os.path.exists(gtex_path):
            gtex_path = alternate_gtex_path

        res_dir = '/home/woody/iwbn/iwbn106h/gtex_fdr_results'

        # Generate the config files
        cfg_dir = '/home/woody/iwbn/iwbn106h/configs_approx_fdr'

        nc_non_tfs = list(range(1200, 4001, 200))
        nc_tfs = [-1]

        generate_configs(
            gtex_dir=gtex_path,
            num_clusters_non_tfs=nc_non_tfs,
            num_clusters_tfs=nc_tfs,
            config_dir=cfg_dir,
            results_dir=res_dir,
            verbosity=1
        )

        print('done')

    else:
        parser = argparse.ArgumentParser(description="Process a config file from the command line.")

        # Add the file argument
        parser.add_argument('-f', type=str, help='The config file to process')

        # Parse the arguments
        args = parser.parse_args()

        with open(args.f, 'r') as f:
            cfg = yaml.safe_load(f)

        compute_approximate_fdr(config=cfg, verbosity=1)
