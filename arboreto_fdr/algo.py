"""
Top-level functions.
"""
from multiprocessing.managers import Value

import pandas as pd
from distributed import Client, LocalCluster
from numba.cuda.cudadecl import integer_numba_types
from requests.packages import target

from arboreto_fdr.core import create_graph, create_graph_fdr, SGBM_KWARGS, RF_KWARGS, EARLY_STOP_WINDOW_LENGTH, DEFAULT_PERMUTATIONS, DEFAULT_TMP_DIR, BOOTSTRAP_FDR_FRACTION
import os.path as op
import os

def grnboost2(expression_data,
              gene_names=None,
              tf_names='all',
              client_or_address='local',
              early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
              limit=None,
              seed=None,
              verbose=False,
              ):

    """
    Launch arboreto with [GRNBoost2] profile.

    :param expression_data: one of:
           * a pandas DataFrame (rows=observations, columns=genes)
           * a dense 2D numpy.ndarray
           * a sparse scipy.sparse.csc_matrix
    :param gene_names: optional list of gene names (strings). Required when a (dense or sparse) matrix is passed as
                       'expression_data' instead of a DataFrame.
    :param tf_names: optional list of transcription factors. If None or 'all', the list of gene_names will be used.
    :param client_or_address: one of:
           * None or 'local': a new Client(LocalCluster()) will be used to perform the computation.
           * string address: a new Client(address) will be used to perform the computation.
           * a Client instance: the specified Client instance will be used to perform the computation.
    :param early_stop_window_length: early stop window length. Default 25.
    :param limit: optional number (int) of top regulatory links to return. Default None.
    :param seed: optional random seed for the regressors. Default None.
    :param verbose: print info.
    :return: a pandas DataFrame['TF', 'target', 'importance'] representing the inferred gene regulatory links.
    """

    return diy(expression_data=expression_data, regressor_type='GBM', regressor_kwargs=SGBM_KWARGS,
               gene_names=gene_names, tf_names=tf_names, client_or_address=client_or_address,
               early_stop_window_length=early_stop_window_length, limit=limit, seed=seed, verbose=verbose,)

def grnboost2_fdr(expression_data,
              are_tfs_clustered : bool,
              tf_representatives : list,
              non_tf_representatives : list,
              gene_to_cluster : dict,
              input_grn : dict,
              gene_names=None,
              client_or_address='local',
              early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
              seed=None,
              verbose=False,
              n_permutations=DEFAULT_PERMUTATIONS,
              output_directory = DEFAULT_TMP_DIR,
              ):

    """
    Perform FDR control for given input GRN with grnboost2.

    :param expression_data: one of:
           * a pandas DataFrame (rows=observations, columns=genes)
           * a dense 2D numpy.ndarray
           * a sparse scipy.sparse.csc_matrix
    :param are_tfs_clustered: True if TFs have also been clustered for FDR control.
    :param tf_representatives: Either list of pre-chosen TF representatives or simply all TFs.
    :param non_tf_representatives: Either list of pre-chosen non-TF representatives or all non-TFs.
    :param gene_to_cluster: Keys are gene names and values are cluster IDs as integers.
    :param input_grn: Dict storing input GRN for FDR control with keys as edge tuples, and as values dicts with
        {'importance' : <float>} structure.
    :param gene_names: optional list of gene names (strings). Required when a (dense or sparse) matrix is passed as
                       'expression_data' instead of a DataFrame.
    :param client_or_address: one of:
           * None or 'local': a new Client(LocalCluster()) will be used to perform the computation.
           * string address: a new Client(address) will be used to perform the computation.
           * a Client instance: the specified Client instance will be used to perform the computation.
    :param early_stop_window_length: early stop window length. Default 25.
    :param seed: optional random seed for the regressors. Default None.
    :param verbose: print info.
    :return: a pandas DataFrame['TF', 'target', 'importance'] representing the inferred gene regulatory links.
    """

    return diy_fdr(expression_data=expression_data,
               regressor_type='GBM',
               regressor_kwargs=SGBM_KWARGS,
               gene_names=gene_names,
               are_tfs_clustered=are_tfs_clustered,
               tf_representatives=tf_representatives,
               non_tf_representatives=non_tf_representatives,
               gene_to_cluster=gene_to_cluster,
               input_grn=input_grn,
               client_or_address=client_or_address,
               early_stop_window_length=early_stop_window_length,
               seed=seed,
               verbose=verbose,
               n_permutations=n_permutations,
               output_directory=output_directory,
               )

def genie3(expression_data,
           gene_names=None,
           tf_names='all',
           client_or_address='local',
           limit=None,
           seed=None,
           verbose=False):
    """
    Launch arboreto with [GENIE3] profile.

    :param expression_data: one of:
           * a pandas DataFrame (rows=observations, columns=genes)
           * a dense 2D numpy.ndarray
           * a sparse scipy.sparse.csc_matrix
    :param gene_names: optional list of gene names (strings). Required when a (dense or sparse) matrix is passed as
                       'expression_data' instead of a DataFrame.
    :param tf_names: optional list of transcription factors. If None or 'all', the list of gene_names will be used.
    :param client_or_address: one of:
           * None or 'local': a new Client(LocalCluster()) will be used to perform the computation.
           * string address: a new Client(address) will be used to perform the computation.
           * a Client instance: the specified Client instance will be used to perform the computation.
    :param limit: optional number (int) of top regulatory links to return. Default None.
    :param seed: optional random seed for the regressors. Default None.
    :param verbose: print info.
    :return: a pandas DataFrame['TF', 'target', 'importance'] representing the inferred gene regulatory links.
    """

    return diy(expression_data=expression_data, regressor_type='RF', regressor_kwargs=RF_KWARGS,
               gene_names=gene_names, tf_names=tf_names, client_or_address=client_or_address,
               limit=limit, seed=seed, verbose=verbose)


def diy(expression_data,
        regressor_type,
        regressor_kwargs,
        gene_names=None,
        tf_names='all',
        client_or_address='local',
        early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
        limit=None,
        seed=None,
        verbose=False):
    """
    :param expression_data: one of:
           * a pandas DataFrame (rows=observations, columns=genes)
           * a dense 2D numpy.ndarray
           * a sparse scipy.sparse.csc_matrix
    :param regressor_type: string. One of: 'RF', 'GBM', 'ET'. Case insensitive.
    :param regressor_kwargs: a dictionary of key-value pairs that configures the regressor.
    :param gene_names: optional list of gene names (strings). Required when a (dense or sparse) matrix is passed as
                       'expression_data' instead of a DataFrame.
    :param tf_names: optional list of transcription factors. If None or 'all', the list of gene_names will be used.
    :param early_stop_window_length: early stopping window length.
    :param client_or_address: one of:
           * None or 'local': a new Client(LocalCluster()) will be used to perform the computation.
           * string address: a new Client(address) will be used to perform the computation.
           * a Client instance: the specified Client instance will be used to perform the computation.
    :param limit: optional number (int) of top regulatory links to return. Default None.
    :param seed: optional random seed for the regressors. Default 666. Use None for random seed.
    :param verbose: print info.
    :return: a pandas DataFrame['TF', 'target', 'importance'] representing the inferred gene regulatory links.
    """
    if verbose:
        print('preparing dask client')

    client, shutdown_callback = _prepare_client(client_or_address)

    try:
        if verbose:
            print('parsing input')

        expression_matrix, gene_names, tf_names = _prepare_input(expression_data, gene_names, tf_names)

        if verbose:
            print('creating dask graph')

        graph = create_graph(expression_matrix,
                             gene_names,
                             tf_names,
                             client=client,
                             regressor_type=regressor_type,
                             regressor_kwargs=regressor_kwargs,
                             early_stop_window_length=early_stop_window_length,
                             limit=limit,
                             seed=seed)

        if verbose:
            print('{} partitions'.format(graph.npartitions))
            print('computing dask graph')

        return client \
            .compute(graph, sync=True) \
            .sort_values(by='importance', ascending=False)

    finally:
        shutdown_callback(verbose)

        if verbose:
            print('finished')


def diy_fdr(expression_data,
        regressor_type,
        regressor_kwargs,
        are_tfs_clustered,
        tf_representatives,
        non_tf_representatives,
        gene_to_cluster,
        input_grn,
        gene_names=None,
        client_or_address='local',
        early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
        seed=None,
        verbose=False,
        n_permutations = DEFAULT_PERMUTATIONS,
        output_directory = DEFAULT_TMP_DIR,
        ):
    """
    :param are_tfs_clustered: True if TFs have also been clustered for FDR control.
    :param tf_representatives: Either list of pre-chosen TF representatives or simply all TFs.
    :param non_tf_representatives: Either list of pre-chosen non-TF representatives or all non-TFs.
    :param gene_to_cluster: Keys are gene names and values are cluster IDs as integers.
    :param input_grn: Dict storing input GRN for FDR control with keys as edge tuples, and as values dicts with
        {'importance' : <float>} structure.
    :param expression_data: one of:
           * a pandas DataFrame (rows=observations, columns=genes)
           * a dense 2D numpy.ndarray
           * a sparse scipy.sparse.csc_matrix
    :param regressor_type: string. One of: 'RF', 'GBM', 'ET'. Case insensitive.
    :param regressor_kwargs: a dictionary of key-value pairs that configures the regressor.
    :param gene_names: optional list of gene names (strings). Required when a (dense or sparse) matrix is passed as
                       'expression_data' instead of a DataFrame.
    :param early_stop_window_length: early stopping window length.
    :param client_or_address: one of:
           * None or 'local': a new Client(LocalCluster()) will be used to perform the computation.
           * string address: a new Client(address) will be used to perform the computation.
           * a Client instance: the specified Client instance will be used to perform the computation.
    :param seed: optional random seed for the regressors. Default 666. Use None for random seed.
    :param verbose: print info.
    :return: a pandas DataFrame['TF', 'target', 'importance'] representing the inferred gene regulatory links.
    """
    if verbose:
        print('preparing dask client')
    
    if output_directory is not None:
        if not os.path.exists(output_directory):
            print('output directory does not exist, creating!')
            os.makedirs(output_directory, exist_ok=True)
    else:
        print('No output directory specified')

    client, shutdown_callback = _prepare_client(client_or_address)

    try:
        if verbose:
            print('parsing input')

        # TF names do not matter in FDR mode, hence can be set to dummy list.
        tf_names = None
        expression_matrix, gene_names, _ = _prepare_input(expression_data, gene_names, tf_names)

        if verbose:
            print('creating dask graph')

        if gene_to_cluster is None:
            raise ValueError(f'Clustering is None, but needs to be passed in FDR mode.')
        if input_grn is None:
            raise ValueError(f'Input GRN is None, but needs to be passed in FDR mode.')
        if tf_representatives is None or non_tf_representatives is None:
            raise ValueError(f'TF or non-TF representatives are None, but need to passed in FDR mode.')

        graph = create_graph_fdr(expression_matrix,
                                 gene_names=gene_names,
                                 are_tfs_clustered=are_tfs_clustered,
                                 tf_representatives=tf_representatives,
                                 non_tf_representatives=non_tf_representatives,
                                 gene_to_cluster=gene_to_cluster,
                                 input_grn=input_grn,
                                 regressor_type=regressor_type,
                                 regressor_kwargs=regressor_kwargs,
                                 client=client,
                                 early_stop_window_length=early_stop_window_length,
                                 seed=seed,
                                 n_permutations=n_permutations)

        if verbose:
            print('{} partitions'.format(graph.npartitions))
            print('computing dask graph')

        return client \
            .compute(graph, sync=True) \
            .sort_values(by='importance', ascending=False)

    finally:
        shutdown_callback(verbose)

        if verbose:
            print('finished')


def _prepare_client(client_or_address):
    """
    :param client_or_address: one of:
           * None
           * verbatim: 'local'
           * string address
           * a Client instance
    :return: a tuple: (Client instance, shutdown callback function).
    :raises: ValueError if no valid client input was provided.
    """

    if client_or_address is None or str(client_or_address).lower() == 'local':
        local_cluster = LocalCluster(diagnostics_port=None)
        client = Client(local_cluster)

        def close_client_and_local_cluster(verbose=False):
            if verbose:
                print('shutting down client and local cluster')

            client.close()
            local_cluster.close()

        return client, close_client_and_local_cluster

    elif isinstance(client_or_address, str) and client_or_address.lower() != 'local':
        client = Client(client_or_address)

        def close_client(verbose=False):
            if verbose:
                print('shutting down client')

            client.close()

        return client, close_client

    elif isinstance(client_or_address, Client):

        def close_dummy(verbose=False):
            if verbose:
                print('not shutting down client, client was created externally')

            return None

        return client_or_address, close_dummy

    else:
        raise ValueError("Invalid client specified {}".format(str(client_or_address)))


def _prepare_input(expression_data,
                   gene_names,
                   tf_names):
    """
    Wrangle the inputs into the correct formats.

    :param expression_data: one of:
                            * a pandas DataFrame (rows=observations, columns=genes)
                            * a dense 2D numpy.ndarray
                            * a sparse scipy.sparse.csc_matrix
    :param gene_names: optional list of gene names (strings).
                       Required when a (dense or sparse) matrix is passed as 'expression_data' instead of a DataFrame.
    :param tf_names: optional list of transcription factors. If None or 'all', the list of gene_names will be used.
    :return: a triple of:
             1. a np.ndarray or scipy.sparse.csc_matrix
             2. a list of gene name strings
             3. a list of transcription factor name strings.
    """

    if isinstance(expression_data, pd.DataFrame):
        expression_matrix = expression_data.to_numpy()
        gene_names = list(expression_data.columns)
    else:
        expression_matrix = expression_data
        assert expression_matrix.shape[1] == len(gene_names)

    if tf_names is None:
        tf_names = gene_names
    elif tf_names == 'all':
        tf_names = gene_names
    else:
        if len(tf_names) == 0:
            raise ValueError('Specified tf_names is empty')

        if not set(gene_names).intersection(set(tf_names)):
            raise ValueError('Intersection of gene_names and tf_names is empty.')

    return expression_matrix, gene_names, tf_names
