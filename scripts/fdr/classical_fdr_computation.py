
# Info needed in configs:
#  - tissue name
#  - path to preprocessed expression matrix
#  - target list (batch)

import os
import copy
import yaml
import argparse
import time
import pandas as pd

from codecarbon import OfflineEmissionsTracker
from arboreto.algo import grnboost2, grnboost2_fdr


def compute_input_grns(gtex_dir: str, results_dir: str | None, verbosity: int = 0):

    if results_dir is None:
        results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)

    tissue_dirs = sorted(os.listdir(gtex_dir))

    for tissue_dir in tissue_dirs:

        if verbosity > 0:
            print(f'# ### Computing GRN for tissue: {tissue_dir}')

        tissue_dir_path = os.path.join(gtex_dir, tissue_dir)

        # Load the expression matrix
        expression_mat_filename = f'{tissue_dir}.tsv'
        expression_mat = pd.read_csv(os.path.join(tissue_dir_path, expression_mat_filename), sep='\t', index_col=0)

        # Load the targets and get the TFs as the complement
        all_genes = expression_mat.columns.tolist()

        target_filename = f'{tissue_dir}_target_genes.tsv'
        target_df = pd.read_csv(os.path.join(tissue_dir_path, target_filename), index_col=0)
        tf_list = set(all_genes) - set(target_df['target_gene'])

        input_grn = grnboost2(
            expression_data=expression_mat,
            gene_names=None,
            tf_names=tf_list,
            seed=42,
            verbose=False,
        )

        output_dir_path = os.path.join(results_dir, tissue_dir)
        os.makedirs(output_dir_path, exist_ok=True)

        input_grn.to_csv(os.path.join(output_dir_path, f'{tissue_dir}_input_grn.csv'), index=False)


def generate_batch_configs(
        gtex_dir: str,
        batch_size: int,
        config_dir: str | None,
        results_dir: str | None,
        verbosity: int = 0
) -> None:

    if config_dir is None:
        config_dir = os.getcwd()
    os.makedirs(config_dir, exist_ok=True)

    if results_dir is None:
        results_dir = os.path.join(os.getcwd(), 'results')

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

        expression_mat = pd.read_csv(os.path.join(tissue_dir_path, expression_mat_filename), sep='\t', index_col=0)

        all_genes = expression_mat.columns.tolist()

        batched_genes = _batch_genes(genes=all_genes, batch_size=batch_size)

        for i, batch in enumerate(batched_genes):

            if verbosity > 0:
                print(f'# ### Batch {str(i).zfill(3)}')

            batch_config = copy.deepcopy(config)

            batch_config['targets'] = batch

            batch_id = str(i).zfill(3)

            batch_config['batch_id'] = batch_id

            batch_config_filename = f'{config['tissue_name']}_{batch_id}.yaml'
            save_path = os.path.join(config_dir, batch_config_filename)

            with open(save_path, 'w') as f:
                yaml.dump(batch_config, f, default_flow_style=False)


def _batch_genes(genes: list[str], batch_size: int) -> list[list[str]]:

    batch_list = [genes[i:i+batch_size] for i in range(0, len(genes), batch_size)]

    return batch_list


def compute_classical_fdr(config: dict, verbosity: int = 0) -> pd.DataFrame:

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

    # Get the targets
    targets = config['targets']  # List of gene names ['gene0', 'gene1', ...]

    # Get the batch id
    batch_id = config['batch_id']

    if verbosity > 0:
        print(f'# ### Computing classical FDR for tissue: {tissue_name}, batch: {batch_id}')


    # Create subdir for saving
    save_dir = os.path.join(results_dir, tissue_name, 'batch_wise_fdr_grns')
    os.makedirs(save_dir, exist_ok=True)

    emissions_file = os.path.join(save_dir, f'emissions_batch_{batch_id}.csv')

    with OfflineEmissionsTracker(
            country_iso_code="DEU", output_file=emissions_file, log_level='error', measure_power_secs=600
    ) as tracker:

        st = time.time()
        fdr_grn = grnboost2_fdr(
            expression_data=expression_mat,
            cluster_representative_mode='all_genes',
            num_non_tf_clusters=-1,
            num_tf_clusters=-1,
            input_grn=input_grn,
            tf_names=None,
            target_subset=targets,
            client_or_address = 'local',
            seed=42,
            verbose=False,
            num_permutations=1000,
            output_dir=None
        )
        et = time.time()

    elapsed_time = et - st

    time_file = os.path.join(save_dir, f'time_batch_{batch_id}.txt')
    with open(time_file, 'w') as f:
        f.write(str(elapsed_time))

    # Save fdr controlled grn for target batch
    fdr_grn.to_csv(os.path.join(save_dir, f'fdr_grn_batch_{batch_id}.csv'), index=False)

    return fdr_grn


if __name__ == '__main__':

    # Set flag whether to do input GRN computation and config generation or run classical FDR control
    fdr = True

    if not fdr:

        gtex_path = './data/gtex_tissues_preprocessed'
        alternate_gtex_path = f'/home/woody/iwbn/iwbn107h/gtex'

        if not os.path.exists(gtex_path):
            gtex_path = alternate_gtex_path

        res_dir = './results'

        # Compute the input GRNs
        compute_input_grns(gtex_dir=gtex_path, results_dir=res_dir, verbosity=1)

        # Generate the config files
        cfg_dir = './config'
        bs = 100

        generate_batch_configs(gtex_dir=gtex_path, batch_size=bs, config_dir=cfg_dir, results_dir=res_dir, verbosity=1)

        print('done')

    else:
        parser = argparse.ArgumentParser(description="Process a config file from the command line.")

        # Add the file argument
        parser.add_argument('-f', type=str, help='The config file to process')

        # Parse the arguments
        args = parser.parse_args()

        with open(args.f, 'r') as f:
            cfg = yaml.safe_load(f)

        compute_classical_fdr(config=cfg, verbosity=1)
