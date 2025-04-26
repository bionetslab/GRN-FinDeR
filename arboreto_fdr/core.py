"""
Core functional building blocks, composed in a Dask graph for distributed computation.
"""
import itertools
from multiprocessing.managers import Value

import numpy as np
import pandas as pd
import logging

import scipy
from fontTools.subset import subset
from requests.packages import target
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from dask import delayed
from dask.dataframe import from_delayed
from dask.dataframe.utils import make_meta
import os.path as op
import random

logger = logging.getLogger(__name__)

DEMON_SEED = 666
ANGEL_SEED = 777
EARLY_STOP_WINDOW_LENGTH = 25
DEFAULT_PERMUTATIONS = 1000
DEFAULT_TMP_DIR = None
BOOTSTRAP_FDR_FRACTION = 1.0

SKLEARN_REGRESSOR_FACTORY = {
    'RF': RandomForestRegressor,
    'ET': ExtraTreesRegressor,
    'GBM': GradientBoostingRegressor
}

# scikit-learn random forest regressor
RF_KWARGS = {
    'n_jobs': 1,
    'n_estimators': 1000,
    'max_features': 'sqrt'
}

# scikit-learn extra-trees regressor
ET_KWARGS = {
    'n_jobs': 1,
    'n_estimators': 1000,
    'max_features': 'sqrt'
}

# scikit-learn gradient boosting regressor
GBM_KWARGS = {
    'learning_rate': 0.01,
    'n_estimators': 500,
    'max_features': 0.1
}

# scikit-learn stochastic gradient boosting regressor
SGBM_KWARGS = {
    'learning_rate': 0.01,
    'n_estimators': 5000,  # can be arbitrarily large
    'max_features': 0.1,
    'subsample': 0.9
}


def is_sklearn_regressor(regressor_type):
    """
    :param regressor_type: string. Case insensitive.
    :return: whether the regressor type is a scikit-learn regressor, following the scikit-learn API.
    """
    return regressor_type.upper() in SKLEARN_REGRESSOR_FACTORY.keys()


def is_xgboost_regressor(regressor_type):
    """
    :param regressor_type: string. Case insensitive.
    :return: boolean indicating whether the regressor type is the xgboost regressor.
    """
    return regressor_type.upper() == 'XGB'


def is_oob_heuristic_supported(regressor_type, regressor_kwargs):
    """
    :param regressor_type: on
    :param regressor_kwargs:
    :return: whether early stopping heuristic based on out-of-bag improvement is supported.

    """
    return \
        regressor_type.upper() == 'GBM' and \
        'subsample' in regressor_kwargs and \
        regressor_kwargs['subsample'] < 1.0


def to_tf_matrix(expression_matrix,
                 gene_names,
                 tf_names):
    """
    :param expression_matrix: numpy matrix. Rows are observations and columns are genes.
    :param gene_names: a list of gene names. Each entry corresponds to the expression_matrix column with same index.
    :param tf_names: a list of transcription factor names. Should be a subset of gene_names.
    :return: tuple of:
             0: A numpy matrix representing the predictor matrix for the regressions.
             1: The gene names corresponding to the columns in the predictor matrix.
    """

    tuples = [(index, gene) for index, gene in enumerate(gene_names) if gene in tf_names]

    tf_indices = [t[0] for t in tuples]
    tf_matrix_names = [t[1] for t in tuples]

    return expression_matrix[:, tf_indices], tf_matrix_names


def fit_model(regressor_type,
              regressor_kwargs,
              tf_matrix,
              target_gene_expression,
              early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
              seed=DEMON_SEED):
    """
    :param regressor_type: string. Case insensitive.
    :param regressor_kwargs: a dictionary of key-value pairs that configures the regressor.
    :param tf_matrix: the predictor matrix (transcription factor matrix) as a numpy array.
    :param target_gene_expression: the target (y) gene expression to predict in function of the tf_matrix (X).
    :param early_stop_window_length: window length of the early stopping monitor.
    :param seed: (optional) random seed for the regressors.
    :return: a trained regression model.
    """
    regressor_type = regressor_type.upper()


    if isinstance(target_gene_expression, scipy.sparse.spmatrix):
        target_gene_expression = target_gene_expression.A.flatten()

    assert tf_matrix.shape[0] == target_gene_expression.shape[0]


    def do_sklearn_regression():
        regressor = SKLEARN_REGRESSOR_FACTORY[regressor_type](random_state=seed, **regressor_kwargs)

        with_early_stopping = is_oob_heuristic_supported(regressor_type, regressor_kwargs)

    
        if with_early_stopping:
            regressor.fit(tf_matrix, target_gene_expression, monitor=EarlyStopMonitor(early_stop_window_length))
        else:
            regressor.fit(tf_matrix, target_gene_expression)

        return regressor

    if is_sklearn_regressor(regressor_type):
        return do_sklearn_regression()
    # elif is_xgboost_regressor(regressor_type):
    #     raise ValueError('XGB regressor not yet supported')
    else:
        raise ValueError('Unsupported regressor type: {0}'.format(regressor_type))


def to_feature_importances(regressor_type,
                           regressor_kwargs,
                           trained_regressor):
    """
    Motivation: when the out-of-bag improvement heuristic is used, we cancel the effect of normalization by dividing
    by the number of trees in the regression ensemble by multiplying again by the number of trees used.

    This enables prioritizing links that were inferred in a regression where lots of

    :param regressor_type: string. Case insensitive.
    :param regressor_kwargs: a dictionary of key-value pairs that configures the regressor.
    :param trained_regressor: the trained model from which to extract the feature importances.
    :return: the feature importances inferred from the trained model.
    """

    if is_oob_heuristic_supported(regressor_type, regressor_kwargs):
        n_estimators = len(trained_regressor.estimators_)

        denormalized_importances = trained_regressor.feature_importances_ * n_estimators

        return denormalized_importances
    else:
        return trained_regressor.feature_importances_


def to_meta_df(trained_regressor,
               target_gene_name):
    """
    :param trained_regressor: the trained model from which to extract the meta information.
    :param target_gene_name: the name of the target gene.
    :return: a Pandas DataFrame containing side information about the regression.
    """
    n_estimators = len(trained_regressor.estimators_)

    return pd.DataFrame({'target': [target_gene_name], 'n_estimators': [n_estimators]})


def to_links_df(regressor_type,
                regressor_kwargs,
                trained_regressor,
                tf_matrix_gene_names,
                target_gene_name):
    """
    :param regressor_type: string. Case insensitive.
    :param regressor_kwargs: dict of key-value pairs that configures the regressor.
    :param trained_regressor: the trained model from which to extract the feature importances.
    :param tf_matrix_gene_names: the list of names corresponding to the columns of the tf_matrix used to train the model.
    :param target_gene_name: the name of the target gene.
    :return: a Pandas DataFrame['TF', 'target', 'importance'] representing inferred regulatory links and their
             connection strength.
    """

    def pythonic():
        # feature_importances = trained_regressor.feature_importances_
        feature_importances = to_feature_importances(regressor_type, regressor_kwargs, trained_regressor)

        links_df = pd.DataFrame({'TF': tf_matrix_gene_names, 'importance': feature_importances})
        links_df['target'] = target_gene_name

        clean_links_df = links_df[links_df.importance > 0].sort_values(by='importance', ascending=False)

        return clean_links_df[['TF', 'target', 'importance']]

    if is_sklearn_regressor(regressor_type):
        return pythonic()
    elif is_xgboost_regressor(regressor_type):
        raise ValueError('XGB regressor not yet supported')
    else:
        raise ValueError('Unsupported regressor type: ' + regressor_type)


def clean(tf_matrix,
          tf_matrix_gene_names,
          target_gene_name):
    """
    :param tf_matrix: numpy array. The full transcription factor matrix.
    :param tf_matrix_gene_names: the full list of transcription factor names, corresponding to the tf_matrix columns.
    :param target_gene_name: the target gene to remove from the tf_matrix and tf_names.
    :return: a tuple of (matrix, names) equal to the specified ones minus the target_gene_name if the target happens
             to be one of the transcription factors. If not, the specified (tf_matrix, tf_names) is returned verbatim.
    """

    if target_gene_name not in tf_matrix_gene_names:
        clean_tf_matrix = tf_matrix
    else:
        ix = tf_matrix_gene_names.index(target_gene_name)
        if isinstance(tf_matrix, scipy.sparse.spmatrix):
            clean_tf_matrix = scipy.sparse.hstack([tf_matrix[:, :ix],
                                                   tf_matrix[:, ix+1:]])
        else:
            clean_tf_matrix = np.delete(tf_matrix, ix, 1)

    clean_tf_names = [tf for tf in tf_matrix_gene_names if tf != target_gene_name]

    assert clean_tf_matrix.shape[1] == len(clean_tf_names)  # sanity check

    return clean_tf_matrix, clean_tf_names



def retry(fn, max_retries=10, warning_msg=None, fallback_result=None):
    """
    Minimalistic retry strategy to compensate for failures probably caused by a thread-safety bug in scikit-learn:
    * https://github.com/scikit-learn/scikit-learn/issues/2755
    * https://github.com/scikit-learn/scikit-learn/issues/7346

    :param fn: the function to retry.
    :param max_retries: the maximum number of retries to attempt.
    :param warning_msg: a warning message to display when an attempt fails.
    :param fallback_result: result to return when all attempts fail.
    :return: Returns the result of fn if one attempt succeeds, else return fallback_result.
    """
    nr_retries = 0

    result = fallback_result

    for attempt in range(max_retries):
        try:
            result = fn()
        except Exception as cause:
            nr_retries += 1

            msg_head = '' if warning_msg is None else repr(warning_msg) + ' '
            msg_tail = "Retry ({1}/{2}). Failure caused by {0}.".format(repr(cause), nr_retries, max_retries)

            logger.warning(msg_head + msg_tail)
        else:
            break

    return result


# def infer_partial_network(regressor_type,
#                           regressor_kwargs,
#                           tf_matrix,
#                           tf_matrix_gene_names,
#                           target_gene_name,
#                           target_gene_expression,
#                           include_meta=False,
#                           early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
#                           seed=DEMON_SEED):
#     """
#     Ties together regressor model training with regulatory links and meta data extraction.

#     :param regressor_type: string. Case insensitive.
#     :param regressor_kwargs: dict of key-value pairs that configures the regressor.
#     :param tf_matrix: numpy matrix. The feature matrix X to use for the regression.
#     :param tf_matrix_gene_names: list of transcription factor names corresponding to the columns of the tf_matrix used to
#                                  train the regression model.
#     :param target_gene_name: the name of the target gene to infer the regulatory links for.
#     :param target_gene_expression: the expression profile of the target gene. Numpy array.
#     :param include_meta: whether to also return the meta information DataFrame.
#     :param early_stop_window_length: window length of the early stopping monitor.
#     :param seed: (optional) random seed for the regressors.
#     :return: if include_meta == True, return links_df, meta_df

#              link_df: a Pandas DataFrame['TF', 'target', 'importance'] containing inferred regulatory links and their
#              connection strength.

#              meta_df: a Pandas DataFrame['target', 'meta', 'value'] containing meta information regarding the trained
#              regression model.
#     """
#     def fn():
#         (clean_tf_matrix, clean_tf_matrix_gene_names) = clean(tf_matrix, tf_matrix_gene_names, target_gene_name)

#         # special case in which only a single TF is passed and the target gene
#         # here is the same as the TF (clean_tf_matrix is empty after cleaning):
#         if clean_tf_matrix.size==0:
#             raise ValueError("Cleaned TF matrix is empty, skipping inference of target {}.".format(target_gene_name))

#         try:
#             trained_regressor = fit_model(regressor_type, regressor_kwargs, clean_tf_matrix, target_gene_expression,
#                                           early_stop_window_length, seed)
#         except ValueError as e:
#             raise ValueError("Regression for target gene {0} failed. Cause {1}.".format(target_gene_name, repr(e)))

#         links_df = to_links_df(regressor_type, regressor_kwargs, trained_regressor, clean_tf_matrix_gene_names,
#                                target_gene_name)

#         if include_meta:
#             meta_df = to_meta_df(trained_regressor, target_gene_name)

#             return links_df, meta_df
#         else:
#             return links_df

#     fallback_result = (_GRN_SCHEMA, _META_SCHEMA) if include_meta else _GRN_SCHEMA

#     return retry(fn,
#                  fallback_result=fallback_result,
#                  warning_msg='WARNING: infer_data failed for target {0}'.format(target_gene_name))

_GRN_SCHEMA = make_meta({'TF': str, 'target': str, 'importance': float, 'counter': int})
_META_SCHEMA = make_meta({'target': str, 'n_estimators': int})

# ### TODO: infer partial network for original, medoid, and random
#       - infer_partial_network for each case

def count_computation_medoid_representative(
        regressor_type,
        regressor_kwargs,
        tf_matrix,
        tf_matrix_gene_names: list[str],
        target_gene_name: str,
        target_gene_expression: np.ndarray,
        partial_input_grn: dict,  # {(TF, target): {'importance': float}}
        gene_to_clust: dict[str, int],
        n_permutations: int,
        early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
        seed=DEMON_SEED,
):
    # Remove target from TF-list and TF-expression matrix if target itself is a TF
    (clean_tf_matrix, clean_tf_matrix_gene_names) = clean(tf_matrix, tf_matrix_gene_names, target_gene_name)

    # Special case in which only a single TF is passed and the target gene
    # here is the same as the TF (clean_tf_matrix is empty after cleaning):
    if clean_tf_matrix.size == 0:
        raise ValueError("Cleaned TF matrix is empty, skipping inference of target {}.".format(target_gene_name))

    # Initialize counts
    for _, val in partial_input_grn.items():
        val.update({'count': 0.0})

    # Iterate for num permutations
    for _ in range(n_permutations):

        # Shuffle target gene expression vector
        permuted_target_gene_expression = np.random.permutation(target_gene_expression)

        # Train the random forest regressor
        try:
            trained_regressor = fit_model(
                regressor_type,
                regressor_kwargs,
                clean_tf_matrix,
                permuted_target_gene_expression,
                early_stop_window_length,
                seed
            )
        except ValueError as e:
            raise ValueError(
                "Count_computation_medoid: regression for target gene {0} failed. Cause {1}.".format(target_gene_name, repr(e))
            )

        # Construct the shuffled GRN dataframe from the trained regressor
        shuffled_grn_df = to_links_df(
            regressor_type,
            regressor_kwargs,
            trained_regressor,
            clean_tf_matrix_gene_names,
            target_gene_name
        )

        # Update the count values of the partial input GRN
        _count_helper(shuffled_grn_df, partial_input_grn, gene_to_clust)

    # Change partial input GRN format from dict to df
    partial_input_grn_fdr_df = pd.DataFrame(
        [(TF, target, v['importance'], v['count']) for (TF, target), v in partial_input_grn.items()],
        columns=['TF', 'target', 'importance', 'count']
    )

    return partial_input_grn_fdr_df


def count_computation_sampled_representative(
        regressor_type,
        regressor_kwargs,
        tf_matrix,
        tf_matrix_gene_names,
        cluster_to_tfs,
        target_gene_names,
        target_gene_idxs,
        target_gene_expressions,
        partial_input_grn: dict,
        gene_to_cluster: dict[str, int],
        n_permutations = DEFAULT_PERMUTATIONS,
        early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
        seed=DEMON_SEED,

):
    # Initialize counts on input GRN edges.
    for _, val in partial_input_grn.items():
        val.update({'count': 0.0})

    for perm in range(n_permutations):
        # Retrieve "random" target gene from cluster.
        perm_index = perm % len(target_gene_names)
        target_gene_name = target_gene_names[perm_index]
        target_gene_index = target_gene_idxs[perm_index]
        target_expression = target_gene_expressions[:, target_gene_index]

        # Remove target from TF-list and TF-expression matrix if target itself is a TF
        # TODO: dicuss, whether TF should remain in TF matrix if it is representative and not individual TF
        (clean_tf_matrix, clean_tf_matrix_gene_names) = clean(tf_matrix, tf_matrix_gene_names, target_gene_name)

        # Sample one TF per TF-cluster and subset TF expression matrix in case of TFs having been clustered.
        if not cluster_to_tfs is None:
            tf_representatives = []
            for cluster, tf_list in cluster_to_tfs.items():
                representative = random.sample(tf_list, k=1)
                tf_representatives.append(representative[0])
            # Subset TF expression matrix.
            clean_tf_matrix, clean_tf_matrix_gene_names = _subset_tf_matrix(clean_tf_matrix,
                                                                    clean_tf_matrix_gene_names,
                                                                    tf_representatives)

        # Special case in which only a single TF is passed and the target gene
        # here is the same as the TF (clean_tf_matrix is empty after cleaning):
        if clean_tf_matrix.size == 0:
            raise ValueError("Cleaned TF matrix is empty, skipping inference of target {}.".format(target_gene_name))

        # Shuffle target gene expression vector
        permuted_target_gene_expression = np.random.permutation(target_expression)

        # Train the random forest regressor.
        try:
            trained_regressor = fit_model(
                regressor_type,
                regressor_kwargs,
                clean_tf_matrix,
                permuted_target_gene_expression,
                early_stop_window_length,
                seed
            )
        except ValueError as e:
            raise ValueError(
                "Count_computation_sampled: regression for target gene {0} failed. Cause {1}.".format(target_gene_name, repr(e))
            )

        # Construct the shuffled GRN dataframe from the trained regressor
        shuffled_grn_df = to_links_df(
            regressor_type,
            regressor_kwargs,
            trained_regressor,
            clean_tf_matrix_gene_names,
            target_gene_name
        )

        # Update the count values of the partial input GRN.
        _count_helper(shuffled_grn_df, partial_input_grn, gene_to_cluster)

    # Change partial input GRN format from dict to df
    partial_input_grn_fdr_df = pd.DataFrame(
        [(TF, target, v['importance'], v['count']) for (TF, target), v in partial_input_grn.items()],
        columns=['TF', 'target', 'importance', 'count']
    )

    return partial_input_grn_fdr_df

def _subset_tf_matrix(tf_matrix : np.ndarray,
                      tf_gene_names : list[str],
                      tf_subset_names : list[str]):
    tf_subset_indices = [index for index, tf in enumerate(tf_gene_names) if tf in tf_subset_names]
    tf_subset_matrix = tf_matrix[:, tf_subset_indices]
    tf_subset_names = [tf for tf in tf_gene_names if tf in tf_subset_names]
    return tf_subset_matrix, tf_subset_names

def _count_helper(
        shuffled_grn: pd.DataFrame,
        partial_input_grn: dict[str, dict[str, float]],
        gene_to_clust: dict[str, int],
) -> None:
    """
    Computes empirical counts for all edges in input GRN based on given decoy edges.
    Args:
        shuffled_grn: Decoy edges based on shuffled expression matrix.
        partial_input_grn: "Groundtruth" edges from input GRN.
        gene_to_clust: Clustering of genes, with gene names as keys and cluster IDs as values.

    Returns: None. Partial_input_GRN is updated in-place.
    """

    # {id of cluster of TF: importance}
    # Note that each TF-cluster-ID (i.e. key in the following dict) still only occurs exactly once, since either
    # TFs have not been clustered (i.e. each TF has one dummy cluster) or if TFs have been clustered, then
    # each TF cluster has exactly one representative and hence can only appear once in shuffled output GRN.
    shuffled_grn_tf_cluster_to_importance = {
        gene_to_clust[tf]: imp for tf, imp in zip(shuffled_grn['TF'], shuffled_grn['importance'])
    }

    for (tf, _), val in partial_input_grn.items():
        if gene_to_clust[tf] in shuffled_grn_tf_cluster_to_importance:
            importance_input = val['importance']
            importance_shuffled = shuffled_grn_tf_cluster_to_importance[gene_to_clust[tf]]

            if importance_shuffled >= importance_input:
                val['count'] += 1



def infer_partial_network(regressor_type,
                          regressor_kwargs,
                          tf_matrix,
                          tf_matrix_gene_names,
                          target_gene_name,
                          target_gene_expression,
                          include_meta=False,
                          early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
                          seed=DEMON_SEED,
                          n_permutations = DEFAULT_PERMUTATIONS,
                          output_directory = DEFAULT_TMP_DIR,
                          bootstrap_fdr_fraction = BOOTSTRAP_FDR_FRACTION):
    """
    Ties together regressor model training with regulatory links and meta data extraction.

    :param regressor_type: string. Case insensitive.
    :param regressor_kwargs: dict of key-value pairs that configures the regressor.
    :param tf_matrix: numpy matrix. The feature matrix X to use for the regression.
    :param tf_matrix_gene_names: list of transcription factor names corresponding to the columns of the tf_matrix used to
                                 train the regression model.
    :param target_gene_name: the name of the target gene to infer the regulatory links for.
    :param target_gene_expression: the expression profile of the target gene. Numpy array.
    :param include_meta: whether to also return the meta information DataFrame.
    :param early_stop_window_length: window length of the early stopping monitor.
    :param seed: (optional) random seed for the regressors.
    :return: if include_meta == True, return links_df, meta_df

             link_df: a Pandas DataFrame['TF', 'target', 'importance'] containing inferred regulatory links and their
             connection strength.

             meta_df: a Pandas DataFrame['target', 'meta', 'value'] containing meta information regarding the trained
             regression model.
    """
    def fn():
        rng = np.random.default_rng()

        (clean_tf_matrix, clean_tf_matrix_gene_names) = clean(tf_matrix, tf_matrix_gene_names, target_gene_name)

        # ### TODO: remove start (input grn is part of input)
        #       - Transform input GRN to dict format beforehand

        # special case in which only a single TF is passed and the target gene
        # here is the same as the TF (clean_tf_matrix is empty after cleaning):
        if clean_tf_matrix.size==0:
            raise ValueError("Cleaned TF matrix is empty, skipping inference of target {}.".format(target_gene_name))

        try:
            trained_regressor_df = fit_model(regressor_type, regressor_kwargs, clean_tf_matrix, target_gene_expression,
                                            early_stop_window_length, seed)
        except ValueError as e:
            raise ValueError("Initial Regression for target gene {0} failed. Cause {1}.".format(target_gene_name, repr(e)))

        links_df = to_links_df(regressor_type, regressor_kwargs, trained_regressor_df, clean_tf_matrix_gene_names,
                                target_gene_name)

        count = {}
        for tf, importance in zip(links_df['TF'], links_df['importance']):
            count[tf] = {'score': importance, 'counter': 0}

        # ### TODO: remove end (input grn is part of input)

        for _ in range(n_permutations):
            exp = np.random.permutation(target_gene_expression)

            if bootstrap_fdr_fraction<1.0:
                boot = np.where(rng.uniform(size=clean_tf_matrix.shape[0])<=bootstrap_fdr_fraction)[0]
                tf_matrix_subs = clean_tf_matrix[boot, :]
                exp = exp[boot]

            else:
                tf_matrix_subs = clean_tf_matrix

            try:
                trained_regressor = fit_model(regressor_type, regressor_kwargs, tf_matrix_subs, exp,
                                            early_stop_window_length, seed)
            except ValueError as e:
                raise ValueError("Regression for target gene {0} failed. Cause {1}.".format(target_gene_name, repr(e)))


            links_df_dc = to_links_df(regressor_type, regressor_kwargs, trained_regressor, clean_tf_matrix_gene_names,
                                target_gene_name)
            
            for tf, importance in zip(links_df_dc['TF'], links_df_dc['importance']):
                if tf in count and importance >= count[tf]['score']:
                    count[tf]['counter'] += 1


        fdr = pd.DataFrame.from_dict(count).T
        fdr = fdr.reset_index()
        fdr.columns = ['TF', 'score', 'counter']
        fdr = fdr[['TF', 'counter']]
        fdr = pd.merge(links_df, fdr, left_on = 'TF', right_on='TF')

        if output_directory is not None:

            try:
                fdr.to_feather(op.join(output_directory, f'{target_gene_name}.feather'))
            except:
                print('Failed to save file, continuing...')


        if include_meta:
            meta_df = to_meta_df(trained_regressor_df, target_gene_name)

            return fdr, meta_df
        else:
            return fdr

    fallback_result = (_GRN_SCHEMA, _META_SCHEMA) if include_meta else _GRN_SCHEMA

    return retry(fn,
                    fallback_result=fallback_result,
                    warning_msg='WARNING: infer_data failed for target {0}'.format(target_gene_name))


def target_gene_indices(gene_names,
                        target_genes):
    """
    :param gene_names: list of gene names.
    :param target_genes: either int (the top n), 'all', or a collection (subset of gene_names).
    :return: the (column) indices of the target genes in the expression_matrix.
    """

    if isinstance(target_genes, list) and len(target_genes) == 0:
        return []

    if isinstance(target_genes, str) and target_genes.upper() == 'ALL':
        return list(range(len(gene_names)))

    elif isinstance(target_genes, int):
        top_n = target_genes
        assert top_n > 0

        return list(range(min(top_n, len(gene_names))))

    elif isinstance(target_genes, list):
        if not target_genes:  # target_genes is empty
            return target_genes
        elif all(isinstance(target_gene, str) for target_gene in target_genes):
            return [index for index, gene in enumerate(gene_names) if gene in target_genes]
        elif all(isinstance(target_gene, int) for target_gene in target_genes):
            return target_genes
        else:
            raise ValueError("Mixed types in target genes.")

    else:
        raise ValueError("Unable to interpret target_genes.")


def create_graph(expression_matrix,
                 gene_names,
                 tf_names,
                 regressor_type,
                 regressor_kwargs,
                 client,
                 target_genes='all',
                 limit=None,
                 include_meta=False,
                 early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
                 repartition_multiplier=1,
                 seed=DEMON_SEED,
                 n_permutations = DEFAULT_PERMUTATIONS,
                 output_directory = DEFAULT_TMP_DIR,
                 bootstrap_fdr_fraction = BOOTSTRAP_FDR_FRACTION):
    """
    Main API function. Create a Dask computation graph.

    Note: fixing the GC problems was fixed by 2 changes: [1] and [2] !!!

    :param expression_matrix: numpy matrix. Rows are observations and columns are genes.
    :param gene_names: list of gene names. Each entry corresponds to the expression_matrix column with same index.
    :param tf_names: list of transcription factor names. Should have a non-empty intersection with gene_names.
    :param regressor_type: regressor type. Case insensitive.
    :param regressor_kwargs: dict of key-value pairs that configures the regressor.
    :param client: a dask.distributed client instance.
                   * Used to scatter-broadcast the tf matrix to the workers instead of simply wrapping in a delayed().
    :param target_genes: either int, 'all' or a collection that is a subset of gene_names.
    :param limit: optional number of top regulatory links to return. Default None.
    :param include_meta: Also return the meta DataFrame. Default False.
    :param early_stop_window_length: window length of the early stopping monitor.
    :param repartition_multiplier: multiplier
    :param seed: (optional) random seed for the regressors. Default 666.
    :return: if include_meta is False, returns a Dask graph that computes the links DataFrame.
             If include_meta is True, returns a tuple: the links DataFrame and the meta DataFrame.
    """

    assert expression_matrix.shape[1] == len(gene_names)
    assert client, "client is required"

    tf_matrix, tf_matrix_gene_names = to_tf_matrix(expression_matrix, gene_names, tf_names)

    future_tf_matrix = client.scatter(tf_matrix, broadcast=True)
    # [1] wrap in a list of 1 -> unsure why but Matt. Rocklin does this often...
    [future_tf_matrix_gene_names] = client.scatter([tf_matrix_gene_names], broadcast=True)

    delayed_link_dfs = []  # collection of delayed link DataFrames
    delayed_meta_dfs = []  # collection of delayed meta DataFrame

    for target_gene_index in target_gene_indices(gene_names, target_genes):
        target_gene_name = delayed(gene_names[target_gene_index], pure=True)
        target_gene_expression = delayed(expression_matrix[:, target_gene_index], pure=True)

        if include_meta:
            delayed_link_df, delayed_meta_df = delayed(infer_partial_network, pure=True, nout=2)(
                regressor_type, regressor_kwargs,
                future_tf_matrix, future_tf_matrix_gene_names,
                target_gene_name, target_gene_expression, include_meta, early_stop_window_length, seed, n_permutations, output_directory, bootstrap_fdr_fraction)

            if delayed_link_df is not None:
                delayed_link_dfs.append(delayed_link_df)
                delayed_meta_dfs.append(delayed_meta_df)
        else:
            delayed_link_df = delayed(infer_partial_network, pure=True)(
                regressor_type, regressor_kwargs,
                future_tf_matrix, future_tf_matrix_gene_names,
                target_gene_name, target_gene_expression, include_meta, early_stop_window_length, seed, n_permutations, output_directory, bootstrap_fdr_fraction)

            if delayed_link_df is not None:
                delayed_link_dfs.append(delayed_link_df)

    # gather the DataFrames into one distributed DataFrame
    all_links_df = from_delayed(delayed_link_dfs, meta=_GRN_SCHEMA)

    # optionally limit the number of resulting regulatory links, descending by top importance
    if limit:
        maybe_limited_links_df = all_links_df.nlargest(limit, columns=['importance'])
    else:
        maybe_limited_links_df = all_links_df

    # [2] repartition to nr of workers -> important to avoid GC problems!
    # see: http://dask.pydata.org/en/latest/dataframe-performance.html#repartition-to-reduce-overhead
    n_parts = len(client.ncores()) * repartition_multiplier

    if include_meta:
        all_meta_df = from_delayed(delayed_meta_dfs, meta=_META_SCHEMA)
        return maybe_limited_links_df.repartition(npartitions=n_parts), \
               all_meta_df.repartition(npartitions=n_parts)
    else:
        return maybe_limited_links_df.repartition(npartitions=n_parts)


def partition_input_grn(input_grn, clustering_dict):
    grn_subsets = dict()
    genes_per_cluster = dict()
    for (tf, target), val in input_grn.items():
        target_cluster = clustering_dict[target]
        if target_cluster in grn_subsets:
            grn_subsets[target_cluster].update({(tf, target): val})
            genes_per_cluster[target_cluster].add(target)
        else:
            grn_subsets[target_cluster] = {(tf, target): val}
            genes_per_cluster[target_cluster] = {target}
    genes_per_cluster = {cluster : list(genes) for cluster, genes in genes_per_cluster.items()}
    return grn_subsets, genes_per_cluster

def invert_tf_to_cluster_dict(tf_representatives : list[str],
                              gene_to_cluster : dict[str, int]):
    """
    Computes per cluster list of corresponding TFs names.
    Args:
        tf_representatives: List of TF names as strings.
        gene_to_cluster: Whole clustering of all input genes.

    Returns:
        Dict with cluster ID as key and list of TF names as values.
    """
    cluster_to_tf = dict()
    # Subset and invert gene-cluster dict to only contain TFs.
    for gene, cluster in gene_to_cluster.items():
        if gene in tf_representatives:
            if not cluster in cluster_to_tf:
                cluster_to_tf[cluster] = []
            cluster_to_tf[cluster].append(gene)
    return cluster_to_tf


def create_graph_fdr(expression_matrix : np.ndarray,
                 gene_names : list[str],
                 are_tfs_clustered : bool,
                 tf_representatives : list[str],
                 non_tf_representatives : list[str],
                 gene_to_cluster : dict[str, int],
                 input_grn : dict,
                 regressor_type,
                 regressor_kwargs,
                 client,
                 early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
                 repartition_multiplier=1,
                 seed=DEMON_SEED,
                 n_permutations = DEFAULT_PERMUTATIONS,
                 ):
    """
    Main API function for FDR control. Create a Dask computation graph.

    Note: fixing the GC problems was fixed by 2 changes: [1] and [2] !!!

    :param expression_matrix: numpy matrix. Rows are observations and columns are genes.
    :param gene_names : list[str]. List of gene names, in the order as they appear in the expression matrix columns.
    :param fdr_mode: str. One of 'medoid', 'random' to indicate representative drawing mode.
    :param are_tfs_clustered: bool. True if TFs have also been clustered, False otherwise.
    :param tf_representatives: list[str]. List of TFs, either only medoids, or all TFs in 'random' mode.
    :param non_tf_representatives: list[str]. List of non-TFs, either only medoids, or all non-TFs in 'random' mode.
    :param gene_to_cluster: dict[str, int]. Keys are gene names, values are cluster IDs they belong to.
    :param input_grn: dict. Input GRN to be FDR-controlled in dict format with (tf,target) as keys.
    :param regressor_type: regressor type. Case insensitive.
    :param regressor_kwargs: dict of key-value pairs that configures the regressor.
    :param client: a dask.distributed client instance.
                   * Used to scatter-broadcast the tf matrix to the workers instead of simply wrapping in a delayed().
    :param early_stop_window_length: window length of the early stopping monitor.
    :param repartition_multiplier: multiplier
    :param n_permutations: int. Number of random permutations to run for FDR control.
    :param seed: (optional) random seed for the regressors. Default 666.
    :return: if include_meta is False, returns a Dask graph that computes the links DataFrame.
             If include_meta is True, returns a tuple: the links DataFrame and the meta DataFrame.
    """
    '''
    4 cases
    A) TF matrix (normal) + medoid gene expression VECTOR (TFs are passed as is)
    Pass variables
        - full TF matrix
        - subset target gene expression matrix (medoid)
        - per target cluster: input GRN subsetted to all target gene represented by the medoid 
    Loop logic the same.
    Counting logic must be modified in infer_partial_network:
        Count logic A.
        - for every edge in input_grn:
            iterate over all edges in input grn and compare importance values for every edge with permuted importance value.
            It does not matter whether it is TF or medoid, because the input GRN is already subset to only include the 
            relevant edges.

    B) TF matrix (normal) + sampling gene expression MATRIX
    Pass variables
        - full TF matrix
        - gex_mat: subset target gene expression matrix (submatrix containing all genes in the cluster)
        - per target cluster: input GRN subsetted to all target genes in the cluster
    Loop logic needs to be modified: We need to loop over/'sample from' gex_mat and permute the current vector of gene expression
    Count logic: Count logic A

    C) TF matrix clustered medoid + medioid gene expression VECTOR
    Conceptual output: Edge between TF cluster and target gene cluster
    Pass variables:
        - medoid TF matrix (only medioid TFs can be targets)
        - subset target gene expression matrix (medoid)
        - per target cluster and TF cluster: input GRN subsetted to all trancription factors 
            and target genes represented by the medoid tfs and target medoids.
    Loop logic: same as A
    Count logic C: For every edge in input_GRN increase counter if permuted value is larger that input value.

    D) TF matrix clustered sampling + sampleing gene expression MATRIX
    Pass variables:
        - Full TF matrix + cluster information (shared globally with all nodes in dansk)
        - gex_mat: subset target gene expression matrix (submatrix containing all genes in the cluster)
        - per target cluster: input GRN subsetted to all target genes in the cluster
    Loop logic needs to be modified: 
    in each iteration
     -  over/'sample from' gex_mat and permute the current vector of gene expression
     -  sample one TF per TF cluster as predictors.
    Count logic: count logic C

    Functions we want to implement:

    Groundtruth = everything is singleton cluster
    1) Subset input_grn: get the relevant tf-target edges. takes lists as input, TFs always all
        targets always the ones represented by the current medioid (A+c)/present in the current cluster (B+C)

    2) Counting logic: takes input GRN, shuffled GNR and mappings between clusters and genes for genes and/or TFs.

    3) Two versions of infer_partial network one for A+C and one for B+D

    4) Reset GRNboost to standard for input_GRN computation and create alternative versions for FDR functionalities with different names.
    

    '''
    assert client, "Client not given, but is required in create_graph_fdr!"
    # Check if gene_to_cluster is complete, i.e. if for every gene in expression matrix, a corresponding cluster has
    # been precomputed.
    all_genes = {gene for gene, _ in gene_to_cluster.items()}
    assert expression_matrix.shape[1] == len(all_genes), "Size of expression matrix does not match gene names."
    assert len(gene_names)==len(all_genes), "Number of clustered genes and genes in expression matrix do not match."

    # Extract FDR mode information from TF and non-TF representative lists.
    fdr_mode = None
    if len(tf_representatives)+len(non_tf_representatives) == len(all_genes):
        fdr_mode = 'random'
    else:
        fdr_mode = 'medoid'

    # Subset expression matrix to TF representatives ('medoid' mode). Leave as is, if TFs have not been clustered or
    # FDR mode is 'random'.
    tf_matrix, tf_matrix_gene_names = to_tf_matrix(expression_matrix, gene_names, tf_representatives)

    # Partition input GRN into dict storing target-cluster IDs as keys and edge dicts (as in input GRN) as values.
    # Second data structure stores target genes per cluster for 'random' FDR mode.
    grn_subsets_per_target, genes_per_target_cluster = partition_input_grn(input_grn, gene_to_cluster)

    #future_tf_matrix = client.scatter(tf_matrix, broadcast=True)
    future_tf_matrix = tf_matrix
    # [1] wrap in a list of 1 -> unsure why but Matt. Rocklin does this often...
    #[future_tf_matrix_gene_names] = client.scatter([tf_matrix_gene_names], broadcast=True)
    future_tf_matrix_gene_names = tf_matrix_gene_names

    delayed_link_dfs = []  # collection of delayed link DataFrames

    # Use pre-computed medoid representatives for TFs and/or non-TFs.
    if fdr_mode == 'medoid':
        # Loop over all representative targets, i.e. non-TF medoids.
        for target_gene_index in target_gene_indices(gene_names, non_tf_representatives+tf_representatives):
            target_gene_name = gene_names[target_gene_index]
            target_gene_expression = expression_matrix[:, target_gene_index]
            target_subset_grn = grn_subsets_per_target[gene_to_cluster[target_gene_name]]

            # Pass subset of GRN which is represented by the medoids.
            delayed_link_df = count_computation_medoid_representative(
                    regressor_type,
                    regressor_kwargs,
                    future_tf_matrix,
                    future_tf_matrix_gene_names,
                    target_gene_name,
                    target_gene_expression,
                    target_subset_grn,
                    gene_to_cluster,
                    n_permutations,
                    early_stop_window_length,
                    seed,
                )

            if delayed_link_df is not None:
                delayed_link_dfs.append(delayed_link_df)

    # Loop over all genes of cluster, i.e. simulate random drawing of genes from clusters.
    elif fdr_mode == 'random':
        # Loop over all target clusters (that includes TF clusters).
        for cluster_id, cluster_targets in genes_per_target_cluster.items():
            target_cluster_idxs = target_gene_indices(gene_names, cluster_targets)
            # Like this, order of cluster gene names should be consistent with order in cluster_idxs and hence with
            # gene column order in expression matrix.
            target_cluster_gene_names = [gene for index, gene in enumerate(gene_names) if index in target_cluster_idxs]
            target_cluster_expression = expression_matrix
            target_subset_grn = grn_subsets_per_target[cluster_id]

            # If TFs have been clustered in 'random' mode, then per permutation, one TF per cluster needs to be
            # drawn. Precompute the necessary cluster-TF relationships here such that keys are cluster IDs
            # and values are list of TFs.
            cluster_to_tfs = None
            if are_tfs_clustered:
                cluster_to_tfs = invert_tf_to_cluster_dict(tf_representatives, gene_to_cluster)

            delayed_link_df = count_computation_sampled_representative(
                    regressor_type,
                    regressor_kwargs,
                    future_tf_matrix,
                    future_tf_matrix_gene_names,
                    cluster_to_tfs,
                    target_cluster_gene_names,
                    target_cluster_idxs,
                    target_cluster_expression,
                    target_subset_grn,
                    gene_to_cluster,
                    n_permutations,
                    early_stop_window_length,
                    seed,
                )

            if delayed_link_dfs is not None:
                delayed_link_dfs.append(delayed_link_df)

    else:
        raise ValueError(f'Unknown FDR mode: {fdr_mode}.')

    # gather the DataFrames into one distributed DataFrame
    # all_links_df = from_delayed(delayed_link_dfs, meta=_GRN_SCHEMA)
    all_links_df = pd.concat(delayed_link_dfs, ignore_index=True)

    # [2] repartition to nr of workers -> important to avoid GC problems!
    # see: http://dask.pydata.org/en/latest/dataframe-performance.html#repartition-to-reduce-overhead
    n_parts = len(client.ncores()) * repartition_multiplier

    #return all_links_df.repartition(npartitions=n_parts)
    return all_links_df


class EarlyStopMonitor:

    def __init__(self, window_length=EARLY_STOP_WINDOW_LENGTH):
        """
        :param window_length: length of the window over the out-of-bag errors.
        """

        self.window_length = window_length

    def window_boundaries(self, current_round):
        """
        :param current_round:
        :return: the low and high boundaries of the estimators window to consider.
        """

        lo = max(0, current_round - self.window_length + 1)
        hi = current_round + 1

        return lo, hi

    def __call__(self, current_round, regressor, _):
        """
        Implementation of the GradientBoostingRegressor monitor function API.

        :param current_round: the current boosting round.
        :param regressor: the regressor.
        :param _: ignored.
        :return: True if the regressor should stop early, else False.
        """

        if current_round >= self.window_length - 1:
            lo, hi = self.window_boundaries(current_round)
            return np.mean(regressor.oob_improvement_[lo: hi]) < 0
        else:
            return False
