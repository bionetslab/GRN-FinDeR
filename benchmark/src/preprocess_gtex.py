#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import os.path as op
import re
import os
import numpy as np
import json
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
    data[transcript_column] = data[transcript_column].str.split('.').str[0]
    return data



def subset_protein_coding(biomart, data, transcript_column='Gene stable ID', mapping_column='Gene type',
                          mapping_keyword='protein_coding'):
    """
    Select protein coding genes
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


def create_GTEX_data(tissue, gtex_count_file, gtex_sample_attribute_file, processed_output_file, biomart, standardize_data =True):

    if not op.isfile(processed_output_file):
        # Load GTEX sammple attributes
        # path = op.join(data_dir, 'GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt')
        tissue_ids = retrieve_GTEX_tissue_sampleids(gtex_sample_attribute_file, tissue=tissue)

        # load gene expression data
        data_gex = read_GTEX_transcript_expression(gtex_count_file, tissue_ids, ['Name', 'Description'])
        data_gex = clean_GTEX_tissue_gene_counts(data_gex, 'Name', 'Description', biomart)
        threshold = data_gex.shape[1] * 0.1
        mask = (data_gex == 0).sum(axis=1) > threshold
        # Filter the DataFrame
        data_gex = data_gex[~mask]
            # genes need to be columns
        data_gex = data_gex.T

        if standardize_data:
            data_gex[data_gex.columns] = scale(data_gex.values)
        
        print(f"Gene expression data shape {data_gex.shape}")
        print(data_gex.columns)
        data_gex.to_csv(processed_output_file, sep = '\t')
    else:
        data_gex = pd.read_csv(processed_output_file, sep = '\t', header = 0, index_col=0)

    return data_gex


def preprocessing_pipeline_GTEX(config):

    biomart = pd.read_csv(config['biomart'], sep='\t')
    tf_list = read_tf_list(config['tf_list'], biomart)


    ## RUN INFERENCE for transcript based network
    results_dir = op.join(config['results_dir'], config['tissue'])
    

    tissue_output_file = op.join(results_dir, f"{config['tissue'].replace(' ', '_')}.tsv")
    data_gex = create_GTEX_data(tissue=config['tissue'], 
                                gtex_count_file=config['gtex_count_data'],
                                gtex_sample_attribute_file=config['gtex_sample_attributes'], 
                                processed_output_file = tissue_output_file, 
                                biomart=biomart, 
                                standardize_data= config['standardize_data'])
 
    target_gene_names = set(data_gex.column.tolist()).intersection(set(tf_list))
    target_gene_output_file = op.join(results_dir, f"{config['tissue'].replace(' ', '_')}_target_genes.tsv")
    pd.DataFrame({'target_gene': list(target_gene_names)}).to_csv(target_gene_output_file)

    return tissue_output_file, target_gene_output_file


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
    except:
        raise ValueError('Issue parsing yaml file')

    emissions_file = op.join(config['results_dir'], config['tissue'], 'emissions.csv')
    
    with OfflineEmissionsTracker(country_iso_code="DEU", output_file = emissions_file, log_level = 'error') as tracker:
        tissue_output_file, gene_name_output_file = preprocessing_pipeline_GTEX(config)


    # Update configuration file
    config['preprocessed_data_file'] = tissue_output_file
    config['target_gene_name_file'] = gene_name_output_file

    with open(filename, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)


    

