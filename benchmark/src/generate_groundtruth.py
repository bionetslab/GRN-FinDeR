#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import arboreto as a
import os.path as op
import re
import os
import numpy as np

from dask_mpi import initialize
from dask.distributed import Client, LocalCluster
from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2
import json
import codecarbon as co

from codecarbon import OfflineEmissionsTracker
from sklearn.preprocessing import scale
from dask_jobqueue.slurm import SLURMRunner
class TissueNotFoundException(Exception):
    pass


def retrieve_GTEX_tissue_sampleids(gtex_annotation_file, tissue='Liver'):
    print('Retrieving tissue sample IDs')
    sample_meta = pd.read_csv(gtex_annotation_file, sep='\t')
    sub = sample_meta.loc[:, ['SAMPID', 'SMTS']]
    try:
        sub = sub[sub.SMTS == tissue]
    except:
        raise TissueNotFoundException(f'{tissue} is not a valid tissue identifier in annotation file')
    return sub.SAMPID.tolist()


def read_GTEX_transcript_expression(path, sample_ids, headers=['transcript_id', 'gene_id']):
    print('Reading Gene expression data')
    columns_to_read = headers + sample_ids
    data = pd.read_csv(path, sep='\t', comment='#', skiprows=2, skipinitialspace=True, nrows=1, header=0)
    columns = data.columns
    columns_to_read = set(columns_to_read).intersection(set(columns))
    data = pd.read_csv(path, sep='\t', comment='#', skiprows=2, skipinitialspace=True, usecols=columns_to_read,
                       header=0)
    return data




def remove_version_id(data, transcript_column='transcript_id'):
    """
    Remove all version ids from the transcript and gene columns
    """
    data[transcript_column] = data[transcript_column].apply(lambda x: re.sub(r"\.[0-9]", "", x))

    return data





def subset_protein_coding(biomart, data, transcript_column='Gene stable ID', mapping_column='Gene type',
                          mapping_keyword='protein_coding'):
    """
    Remove protein
    """
    protein_coding = data[
        data.transcript_id.isin(biomart[biomart[mapping_column] == mapping_keyword][transcript_column].tolist())]
    return protein_coding




def clean_GTEX_tissue_gene_counts(data_gex, transcript_column, drop_column, biomart, biomart_column = 'Gene stable ID', new_colname='gene_id'):
    print('Cleaning up counts')
    data_gex = remove_version_id(data_gex, transcript_column=transcript_column)
    data_gex = data_gex.rename(columns={transcript_column: new_colname})
    data_gex = data_gex[
        data_gex[new_colname].isin(biomart[biomart['Gene type'] == 'protein_coding'][biomart_column].tolist())]
    if drop_column is not None and drop_column in data_gex.columns:
        data_gex = data_gex.drop([drop_column], axis=1)
    data_gex = data_gex.set_index(new_colname)
    return data_gex


def read_tf_list(tf_path, biomart):
    tf_list = pd.read_csv(tf_path, sep='\t', header=None)
    tf_list.columns = ['TF']
    tf_list = tf_list.merge(biomart, left_on='TF', right_on='Gene name')
    tf_list = tf_list.loc[:, ['TF', 'Gene stable ID']].drop_duplicates()
    tf_list = tf_list.reset_index()
    return tf_list


def compute_and_save_network(data, tf_list, client, file, use_tf=False):
    print('Computing network')
    # compute the GRN
    if not use_tf:
        network = grnboost2(expression_data=data,
                            client_or_address=client)
    else:
        network = grnboost2(expression_data=data,
                            tf_names=tf_list,
                            client_or_address=client)

    # write the GRN to file
    network.to_csv(file, sep='\t', index=False, header=False)
    return network


def save_randomization_counts(count, iteration, filename, yaml_file, tissue):
    l = []
    for k in count:
        l.append([k[0], k[1], count[k]['counter']])
    r = pd.DataFrame(l)
    r.columns = ['source', 'target', 'count']
    r.to_csv(filename, sep='\t', index=None)

    data = {'tissue': tissue, 'iteration': iteration}
    with open(yaml_file, 'w') as f:
        json.dump(data, f)
    return r


# def compute_background_model(data, tf_list, client, grn, output_file, yaml_file, tissue, samples, tf_column='TF',
#                              target_column='target', importance_column='importance', use_tf=True):
#     print('Computing background model')
#     count = {}
#     for i, row in grn.iterrows():
#         count[(row[tf_column], row[target_column])] = {'score': row[importance_column], 'counter': 0}
#     np.random.seed(22)
#     for i in range(0, samples):
#         data_permuted = data.sample(frac=1, axis=1)
#         if not use_tf:
#             network = grnboost2(expression_data=data_permuted,
#                                 client_or_address=client)
#         else:
#             network = grnboost2(expression_data=data_permuted,
#                                 tf_names=tf_list,
#                                 client_or_address=client)
#         for row in network.itertuples():
#             if (row[1], row[2]) in count and (row[3] >= count[(row[1], row[2])]['score']):
#                 count[(row[1], row[2])]['counter'] += 1

#         rcount = save_randomization_counts(count, i, output_file, yaml_file, tissue)

#     rcount = save_randomization_counts(count, i, output_file, yaml_file, tissue)
#     return rcount

import threading
import numpy as np

def compute_background_model(data, tf_list, client, grn, output_file, yaml_file, tissue, samples, tf_column='TF',
                             target_column='target', importance_column='importance', use_tf=True):
    print('Computing background model')
    count = {}
    lock = threading.Lock()  # Lock to protect the shared count object during updates

    # Initialize the count dictionary for each TF-target pair
    for i, row in grn.iterrows():
        count[(row[tf_column], row[target_column])] = {'score': row[importance_column], 'counter': 0}
    
    np.random.seed(22)

    # Function to update the count object in a thread-safe manner
    def update_count(network):
        nonlocal count
        with lock:  # Ensure only one thread modifies `count` at a time
            for row in network.itertuples():
                if (row[1], row[2]) in count and (row[3] >= count[(row[1], row[2])]['score']):
                    count[(row[1], row[2])]['counter'] += 1

    # List to store results for each iteration
    for i in range(0, samples):
        # Permute the data
        print(f'Iteration {i}')
        data_permuted = data.sample(frac=1, axis=1)

        # Perform the GRNboost2 operation in the main thread (OpenMP handles parallelism)
        if not use_tf:
            network = grnboost2(expression_data=data_permuted, client_or_address=client)
        else:
            network = grnboost2(expression_data=data_permuted, tf_names=tf_list, client_or_address=client)

        # Update the count in the current thread
        update_count(network)
        
        save_interval = 50
        if (i + 1) % save_interval == 0:
            print(f"Saving after iteration {i+1}...")
            save_randomization_counts(count, i + 1, output_file, yaml_file, tissue)

    # After all iterations are done, write the final count to the file
    save_randomization_counts(count, samples, output_file, yaml_file, tissue)
    
    return count


def read_result(outfile, outfile_perm, resultfile_fdr):
    print('Adding empirical p-values to result')
    real_network = pd.read_csv(outfile, sep='\t', header=None)
    real_network.columns = ['source', 'target', 'importance']
    rand_network = pd.read_csv(outfile_perm, sep='\t', header=0)
    rand_network.columns = ['source', 'target', 'pval']
    fdr_network = real_network.merge(rand_network, on=['source', 'target'])
    fdr_network['pval'] = fdr_network['pval'] / 1000
    fdr_network.to_csv(resultfile_fdr, sep='\t', index=False)
    return fdr_network


def add_gene_names(biomart, bb, tissue):
    print('Adding human readable gene names')
    # Merge source column (only contains TF isoforms (transcript IDs))
    bb = bb.merge(biomart, left_on='source', right_on='Transcript stable ID')
    # Merge target column, containing either TF isoform ID, or gene ID
    t = [x.startswith('ENST') for x in bb.target]
    b1 = bb[t].merge(biomart, left_on='target', right_on='Transcript stable ID', suffixes=('_s', '_t'))
    t2 = [not x.startswith('ENST') for x in bb.target]
    # gene id needs to be deduplicated in biomart for this merge
    b2 = bb[t2].merge(biomart[~biomart.loc[:, ['Gene stable ID', 'Gene name']].duplicated()], left_on='target',
                      right_on='Gene stable ID', suffixes=('_s', '_t'))
    bb = pd.concat([b1, b2])
    bb = bb.drop(['Transcript stable ID_s'], axis=1)
    bb = bb.drop(['Transcript stable ID_t'], axis=1)
    bb['tissue'] = tissue
    return bb


def create_GTEX_data(config, biomart, tf_list):
    # Load GTEX sammple attributes
    # path = op.join(data_dir, 'GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt')
    tissue_ids = retrieve_GTEX_tissue_sampleids(config['gtex_sample_attributes'], tissue=config['tissue'])

    # load gene expression data
    data_gex = read_GTEX_transcript_expression(config['gtex_count_data'], tissue_ids, ['Name', 'Description'])
    data_gex = clean_GTEX_tissue_gene_counts(data_gex, 'Name', 'Description', biomart)
    threshold = data_gex.shape[1] * 0.1
    mask = (data_gex == 0).sum(axis=1) > threshold
    # Filter the DataFrame
    data_gex = data_gex[~mask]

    return data_gex


def inference_pipeline_GTEX(config):

    biomart = pd.read_csv(config['biomart'], sep='\t')
    tf_list = read_tf_list(config['tf_list'], biomart)

    print(biomart.head())
    print(tf_list)

    data_gex = create_GTEX_data(config, biomart, tf_list)
 
    # genes need to be columns
    data_gex = data_gex.T

    if config['standardize_data']:
        data_gex[data_gex.columns] = scale(data_gex.values)

    print(f'Full data shape:{data_gex.head()}')

    # instantiate a custom Dask distributed Client
    from dask_jobqueue import SLURMCluster

    cluster = SLURMCluster(queue='work', account="iwbn", cores=30, memory="200 GB", walltime='24:00:00')
    #cluster = SLURMRunner()
    cluster.scale(jobs=1)  # ask for 10 jobs
    client = Client(cluster)


    ## RUN INFERENCE for transcript based network
    results_dir = op.join(config['results_dir'], config['tissue'])
    results_dir_grn = op.join(results_dir, 'grn')
    os.makedirs(results_dir_grn, exist_ok=True)
    results_dir_permutation = op.join(results_dir, 'permuted')
    os.makedirs(results_dir_permutation, exist_ok=True)

    file_gene = op.join(results_dir_grn, f"{config['tissue']}_gene_tf.network.tsv")
    grn = compute_and_save_network(data_gex,
                                    tf_list['Gene stable ID'].unique().tolist(),
                                    client,
                                    file_gene,
                                    use_tf=True)

    ## RUN FDR control
    if config['fdr_samples'] > 0:
        file_gene_fdr = op.join(results_dir_permutation, f"{config['tissue']}_gene_tf.count.tsv", )
        yaml_file_gene = op.join(results_dir_permutation, f"{config['tissue']}_gene_metadata.json")
        compute_background_model(data_gex,
                                    tf_list['Gene stable ID'].unique().tolist(),
                                    client,
                                    grn,
                                    file_gene_fdr,
                                    yaml_file_gene,
                                    config['tissue'],
                                    config['fdr_samples'],
                                    use_tf=True
                                    )

        resultfile_frd_gene = op.join(results_dir, f"{config['tissue']}_gene_tf.fdr_network.tsv")
        read_result(file_gene, file_gene_fdr, resultfile_frd_gene)

    if config['run_full_network']:
        ## Infer Gene based networks with all genes
        file_gene_all = f"{config['tissue']}_gene_all.network.tsv"
        compute_and_save_network(data_gex,
                                        tf_list['Gene stable ID'].unique().tolist(),
                                        client,
                                        file_gene_all,
                                        use_tf=False)

        print(f'Gene data shape:{data_gex.shape}')

        if config['fdr_samples'] > 0:
            file_gene_fdr_all = op.join(results_dir_permutation, f"{config['tissue']}_gene_all.count.tsv", )
            yaml_file_gene = op.join(results_dir_permutation, f"{config['tissue']}_gene_metadata.json")
            compute_background_model(data_gex,
                                        tf_list['Gene stable ID'].unique().tolist(),
                                        client,
                                        grn,
                                        file_gene_fdr_all,
                                        yaml_file_gene,
                                        config['tissue'],
                                        config['fdr_samples'],
                                        use_tf=False
                                        )

            resultfile_frd_gene = op.join(results_dir, f"{config['tissue']}_gene_all.fdr_network.tsv")
            read_result(file_gene, file_gene_fdr, resultfile_frd_gene)





if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description="Process a file from the command line.")
    
    # Add the file argument
    parser.add_argument('-f', type=str, help='The file to process')
    
    # Parse the arguments
    args = parser.parse_args()
    
    with open(args.f, 'r') as f:
        config = yaml.safe_load(f)

    emissions_file = op.join(config['results_dir'], config['tissue'], 'emissions.csv')
    
    with OfflineEmissionsTracker(country_iso_code="DEU", output_file = emissions_file) as tracker:
        inference_pipeline_GTEX(config)

