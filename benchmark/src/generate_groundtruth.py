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
from sklearn.preprocessing import scale
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


def compute_and_save_network(data, tf_list, client, file, use_tf=False, n_permutations = 1000, output_dir = '/tmp/grnboost2'):
    print('Computing network')
    # compute the GRN
    if not use_tf:
        network = grnboost2(expression_data=data,
                            client_or_address=client,
                            n_permutations = n_permutations,
                            output_directory = output_dir)
    else:
        network = grnboost2(expression_data=data,
                            tf_names=tf_list,
                            client_or_address=client,
                            n_permutations = n_permutations,
                            output_directory = output_dir)

    # write the GRN to file
    network.to_csv(file, sep='\t', index=False, header=False)
    return network





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

    tissue_gex_file = op.join(config['results_dir'], f"{config['tissue'].replace(' ', '_')}.tsv")
    if not op.isfile(tissue_gex_file):
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
            # genes need to be columns
        data_gex = data_gex.T

        if config['standardize_data']:
            data_gex[data_gex.columns] = scale(data_gex.values)
    
        data_gex.to_csv(tissue_gex_file, sep = '\t')
    else:
        data_gex = pd.read_csv(tissue_gex_file, sep = '\t')

    return data_gex


def inference_pipeline_GTEX(config):

    biomart = pd.read_csv(config['biomart'], sep='\t')
    tf_list = read_tf_list(config['tf_list'], biomart)

    print(biomart.head())
    print(tf_list)

    data_gex = create_GTEX_data(config, biomart, tf_list)
 

    print(f'Full data shape:{data_gex.head()}')


    cluster = LocalCluster()
    client = Client(cluster)
    print(client)
    
    print(f'Client {client.dashboard_link}')

    ## RUN INFERENCE for transcript based network
    results_dir = op.join(config['results_dir'], config['tissue'])
    results_dir_grn = op.join(results_dir, 'grn')
    os.makedirs(results_dir_grn, exist_ok=True)

    file_gene = op.join(results_dir, f"{config['tissue']}_gene_tf.network.tsv")
    network = compute_and_save_network(data_gex,
                                    tf_list['Gene stable ID'].unique().tolist(),
                                    client,
    
                                    file_gene,
                                    use_tf=True, 
                                    output_dir = results_dir_grn)



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
    
    #with OfflineEmissionsTracker(country_iso_code="DEU", output_file = emissions_file) as tracker:
    inference_pipeline_GTEX(config)

