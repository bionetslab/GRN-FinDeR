#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import arboreto as a
import os.path as op
import re
import os
import numpy as np

from dask.distributed import Client, LocalCluster
from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2
import json
import codecarbon as co

from codecarbon import OfflineEmissionsTracker

from preprocess_gtex import create_GTEX_data, read_tf_list


def compute_and_save_network(data, tf_list, client, file, use_tf=False, n_permutations = 1000, output_dir = '/tmp/grnboost2', bootstrap_fdr_fraction = 1.0):
    print('Computing network')
    # compute the GRN
    if not use_tf:
        network = grnboost2(expression_data=data,
                            client_or_address=client,
                            n_permutations = n_permutations,
                            output_directory = output_dir,
                            bootstrap_fdr_fraction = bootstrap_fdr_fraction)
    else:
        network = grnboost2(expression_data=data,
                            tf_names=tf_list,
                            client_or_address=client,
                            n_permutations = n_permutations,
                            output_directory = output_dir,
                            bootstrap_fdr_fraction = bootstrap_fdr_fraction)

    # write the GRN to file
    network.to_csv(file, sep='\t', index=False, header=False)
    return network


def inference_pipeline_GTEX(config):

    biomart = pd.read_csv(config['biomart'], sep='\t')
    tf_list = read_tf_list(config['tf_list'], biomart)


    ## RUN INFERENCE for transcript based network
    tissue_string = config['tissue'].replace(' ', '_')
    results_dir = op.join(config['results_dir'], tissue_string)
    results_dir_grn = op.join(results_dir, config['subnetwork_dir'])
    os.makedirs(results_dir_grn, exist_ok=True)


    tissue_output_file = op.join(results_dir, f"{config['tissue'].replace(' ', '_')}.tsv")
    data_gex = create_GTEX_data(tissue=config['tissue'], 
                                gtex_count_file=config['gtex_count_data'],
                                gtex_sample_attribute_file=config['gtex_sample_attributes'], 
                                processed_output_file = tissue_output_file, 
                                biomart=biomart, 
                                standardize_data= config['standardize_data'])
 
        
    genes = list(config['selected_genes'])+list(tf_list['Gene stable ID'].unique().tolist())
    genes = list(set(genes).intersection(set(data_gex.columns.unique().tolist())))
    data_gex = data_gex.loc[:,genes]

    cluster = LocalCluster()
    client = Client(cluster)


    print('Running inferrence pipeline')
    if op.isdir(config['temp_dir']):
        grn_temp_dir = op.join(config['temp_dir'], tissue_string)
        os.makedirs(grn_temp_dir, exist_ok=True)
        print(f"Storing intermediate results in {op.join(config['temp_dir'], tissue_string)}")
    else:
        grn_temp_dir = results_dir_grn

    if config['bootstrap']:
        # Bootstrap is set to a target number of samples, need to compute required percentage
        # Make sure it is between 2 and 100 percent.
        bootstrap_percentage = np.max(np.min(1.0, np.floor(config['bootstrap_size']/data_gex.shape[0], decimals = 2)), 0.02)
    else:
        bootstrap_percentage = 1.0

    file_gene = op.join(results_dir, f"{tissue_string}_gene_tf.network.tsv")
    network = compute_and_save_network(data_gex,
                                    tf_list['Gene stable ID'].unique().tolist(),
                                    client,
                                    file_gene,
                                    use_tf=True, 
                                    output_dir = grn_temp_dir,
                                    n_permutations=config['fdr_samples'],
                                    bootstrap_fdr_fraction= bootstrap_percentage)

    




if __name__ == "__main__":
    import yaml
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Process a file from the command line.")
    
    # Add the file argument
    parser.add_argument('-f', type=str, help='The file to process')
    
    # Parse the arguments
    args = parser.parse_args()
    
    with open(args.f, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        config = {key: os.path.expandvars(value) if isinstance(value, str) else value for key, value in config.items()}
        print(f"Temp dir: {config['temp_dir']}")
    except:
        raise ValueError('Issue parsing yaml file')

    emissions_file = op.join(config['results_dir'], config['tissue'].replace(' ', '_'), 'emissions.csv')
    
    with OfflineEmissionsTracker(country_iso_code="DEU", output_file = emissions_file, log_level = 'error', measure_power_secs = 600) as tracker:
        inference_pipeline_GTEX(config)

