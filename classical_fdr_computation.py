
# Info needed in configs:
#  - tissue name
#  - path to preprocessed expression matrix
#  - target list (batch)

import os
import copy
import yaml
import pandas as pd




def generate_batch_configs(gtex_dir: str, batch_size: int, save_dir: str | None, verbosity: int = 0) -> None:

    if save_dir is None:
        save_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)

    config = dict()

    tissue_dirs = sorted(os.listdir(gtex_dir))

    for tissue_dir in tissue_dirs:

        if verbosity > 0:
            print(f'# ###### Tissue {tissue_dir} ###### #')

        tissue_dir_path = os.path.join(gtex_dir, tissue_dir)

        expression_mat_filename = f'{tissue_dir}.tsv'

        config['tissue_name'] = tissue_dir
        config['tissue_data_path'] = tissue_dir_path
        config['expression_mat_filename'] = expression_mat_filename

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
            save_path = os.path.join(save_dir, batch_config_filename)

            with open(save_path, 'w') as f:
                yaml.dump(batch_config, f, default_flow_style=False)


def _batch_genes(genes: list[str], batch_size: int) -> list[list[str]]:

    batch_list = [genes[i:i+batch_size] for i in range(0, len(genes), batch_size)]

    return batch_list


def compute_classical_fdr_grn(config: dict, verbosity: int = 0) -> pd.DataFrame:

    # Load the expression data
    tissue_name = config['tissue_name']  # Same as tissue_dir
    tissue_dir_path = config['tissue_data_path']  # gtex_dir + tissue_dir (where expression matrix is saved)
    expression_mat_filename = config['expression_mat_filename']

    expression_mat = pd.read_csv(os.path.join(tissue_dir_path, expression_mat_filename), sep='\t', index_col=0)

    targets = config['targets']  # List of gene names ['gene0', 'gene1', ...]

    batch_id = config['batch_id']

    if verbosity > 0:
        print(f'# ### Computing classical FDR for tissue: {tissue_name}, batch: {batch_id}')

    fdr_grn = None

    return


if __name__ == '__main__':

    gtex_path = os.path.join(os.getcwd(), 'data/gtex_tissues_preprocessed')
    config_dir = os.path.join(os.getcwd(), 'config')
    bs = 100

    generate_batch_configs(gtex_dir=gtex_path, batch_size=bs, save_dir=config_dir, verbosity=1)

    print('done')
