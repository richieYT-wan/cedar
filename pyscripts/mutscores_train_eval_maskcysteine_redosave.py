import pandas as pd
import sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
from functools import partial
import multiprocessing
import itertools

from tqdm.auto import tqdm

import warnings
from datetime import datetime as dt
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Custom fct imports
from src.utils import pkl_load, pkl_dump
import argparse
from src.data_processing import BL62_VALUES, BL62FREQ_VALUES, get_dataset, AA_KEYS
from src.utils import str2bool, mkdirs, convert_path
from src.metrics import get_metrics, get_mean_roc_curve, get_nested_feature_importance
from src.bootstrap import bootstrap_eval
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

N_CORES = 30

# BOOTSTRAP FUNCTIONS
from src.metrics import get_metrics
"""
HERE keeps custom definition to set cysteine to zero (x[:, 4]=0)
"""

def assert_encoding_kwargs(encoding_kwargs, mode_eval=False):
    """
    Assertion / checks for encoding kwargs and verify all the necessary key-values 
    are in
    """
    # Making a deep copy since dicts are mutable between fct calls
    encoding_kwargs = deepcopy(encoding_kwargs)
    if encoding_kwargs is None:
        encoding_kwargs = {'max_len': 12,
                           'encoding': 'onehot',
                           'blosum_matrix': None,
                           'standardize': False}
    essential_keys = ['max_len', 'encoding', 'blosum_matrix', 'standardize']
    assert all([x in encoding_kwargs.keys() for x in
                essential_keys]), f'Encoding kwargs don\'t contain the essential key-value pairs! ' \
                                  f"{'max_len', 'encoding', 'blosum_matrix', 'standardize'} are required."

    if mode_eval:
        if any([(x not in encoding_kwargs.keys()) for x in ['seq_col', 'hla_col', 'target_col', 'rank_col']]):
            if 'seq_col' not in encoding_kwargs.keys():
                encoding_kwargs.update({'seq_col': 'Peptide'})
            if 'hla_col' not in encoding_kwargs.keys():
                encoding_kwargs.update({'hla_col': 'HLA'})
            if 'target_col' not in encoding_kwargs.keys():
                encoding_kwargs.update({'target_col': 'agg_label'})
            if 'rank_col' not in encoding_kwargs.keys():
                encoding_kwargs.update({'seq_col': 'trueHLA_EL_rank'})

        # This KWARGS not needed in eval mode since I'm using Pipeline and Wrapper
        del encoding_kwargs['standardize']
    return encoding_kwargs


def get_predictions(df, models, ics_dict, encoding_kwargs):
    """
    Args:
        models (list) : list of all the models for a given fold. Should be a LIST
        ics_dict (dict): weights or None
        encoding_kwargs: the kwargs needed to process the df
        metrics (dict):

    Returns:
        predictions_df (pd
        df (pd.DataFrame): DataFrame containing the Peptide-HLA pairs to evaluate
        models (list): A.DataFrame): Original DataFrame + a column predictions which are the scores + y_true
    """

    df = df.copy()

    # HERE NEED TO DO SWITCH CASES
    x, y = get_dataset(df, ics_dict, **encoding_kwargs)
    x[:, 4]=0
    # Take the first model in the list and get its class
    model_class = models[0].__class__

    # If model is a scikit-learn model, get pred prob
    # if issubclass(model_class, sklearn.base.BaseEstimator):

    # THESE ARE SCORES
    average_predictions = [model.predict_proba(x)[:, 1] \
                           for model in models]

    average_predictions = np.mean(np.stack(average_predictions), axis=0)
    # assert len(average_predictions)==len(df), f'Wrong shapes passed preds:{len(average_predictions)},df:{len(df)}'
    output_df = df.copy(deep=True)
    output_df['pred'] = average_predictions
    return output_df


# TRAIN WITH PARALLEL WRAPPER
def parallel_inner_train_wrapper(train_dataframe, x_test, base_model, ics_dict,
                                 encoding_kwargs, standardize, fold_out, fold_in):
    seed = fold_out * 10 + fold_in
    # Copy the base model, resets the seed
    model = sklearn.base.clone(base_model)
    model.set_params(random_state=seed)
    if standardize:
        model = Pipeline([('scaler', StandardScaler()), ('model', model)])

    # Here resets model weight at every fold, using the fold number (range(0, n_folds*(n_folds-1)) ) as seed
    # Query subset dataframe and get encoded data and targets
    train = train_dataframe.query('fold != @fold_out and fold != @fold_in').reset_index(drop=True)
    valid = train_dataframe.query('fold == @fold_in').reset_index(drop=True)
    # Get datasets
    x_train, y_train = get_dataset(train, ics_dict, **encoding_kwargs)
    x_valid, y_valid = get_dataset(valid, ics_dict, **encoding_kwargs)
    # SETTING CYSTEINE TO 0
    x_train[:, 4]=0
    x_valid[:, 4]=0
    x_test[:, 4]=0
    # Fit the model and append it to the list
    model.fit(x_train, y_train)

    # Get the prediction values on both the train and validation set
    y_train_pred, y_train_score = model.predict(x_train), model.predict_proba(x_train)[:, 1]
    y_val_pred, y_val_score = model.predict(x_valid), model.predict_proba(x_valid)[:, 1]
    # Get the metrics and save them
    train_metrics = get_metrics(y_train, y_train_score, y_train_pred)
    valid_metrics = get_metrics(y_valid, y_val_score, y_val_pred)
    y_pred_test = model.predict_proba(x_test)[:, 1]

    return model, train_metrics, valid_metrics, y_pred_test


def nested_kcv_train_mut(dataframe, base_model, ics_dict, encoding_kwargs: dict = None, n_jobs=None):
    """
    Args:
        dataframe:
        base_model:
        ics_dict:
        encoding_kwargs:

    Returns:
        models_fold
        train_results
        test_results
    """
    encoding_kwargs = assert_encoding_kwargs(encoding_kwargs, mode_eval=False)
    #
    models_dict = {}
    test_metrics = {}
    train_metrics = {}
    folds = sorted(dataframe.fold.unique())
    std = encoding_kwargs.pop('standardize')
    for fold_out in tqdm(folds, leave=False, desc='Outer fold', position=2):
        # Get test set & init models list to house all models trained in inner fold
        test = dataframe.query('fold == @fold_out').reset_index(drop=True)
        x_test, y_test = get_dataset(test, ics_dict, **encoding_kwargs)
        # For a given fold, all the models that are trained should be appended to this list
        inner_folds = sorted([f for f in folds if f != fold_out])
        n_jobs = len(inner_folds) if n_jobs is None else n_jobs
        # Create the sub-dict and put the model into the models dict
        train_wrapper_ = partial(parallel_inner_train_wrapper, train_dataframe=dataframe, x_test=x_test,
                                 base_model=base_model, ics_dict=ics_dict, encoding_kwargs=encoding_kwargs,
                                 standardize=std, fold_out=fold_out)
        output = Parallel(n_jobs=n_jobs)(
            delayed(train_wrapper_)(fold_in=fold_in) for fold_in in tqdm(inner_folds,
                                                                         desc='Inner Folds',
                                                                         leave=False, position=1))
        models_dict[fold_out] = [x[0] for x in output]
        train_tmp = [x[1] for x in output]
        valid_tmp = [x[2] for x in output]
        avg_prediction = [x[3] for x in output]
        avg_prediction = np.mean(np.stack(avg_prediction), axis=0)
        train_metrics[fold_out] = {k: {'train': v_train,
                                       'valid': v_valid} for k, v_train, v_valid in
                                   zip(inner_folds, train_tmp, valid_tmp)}
        test_metrics[fold_out] = get_metrics(y_test, avg_prediction)

    return models_dict, train_metrics, test_metrics


def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to crossvalidate and evaluate methods that use aa frequency as encoding')

    parser.add_argument('-datadir', type=str, default='../data/mutant/',
                        help='Path to directory containing the pre-partitioned data')
    parser.add_argument('-outdir', type=str, default='../output/221028_new_core_mutscores/')
    parser.add_argument('-trainset', type=str, default='cedar')
    parser.add_argument('-icsdir', type=str, default='../data/ic_dicts/',
                        help='Path containing the pre-computed ICs dicts.')
    parser.add_argument('-ncores', type=int, default=36,
                        help='N cores to use in parallel, by default will be multiprocesing.cpu_count() * 3/4')
    return parser.parse_args()


def main():
    start = dt.now()
    args = vars(args_parser())
    args['outdir'], args['datadir'], args['icsdir'] = convert_path(args['outdir']), convert_path(
        args['datadir']), convert_path(args['icsdir'])
    print('Making dirs')
    #mkdirs(args['outdir'])
    #mkdirs(f'{args["outdir"]}raw/')
    #mkdirs(f'{args["outdir"]}bootstrapping/')
    N_CORES = int(multiprocessing.cpu_count() * 3 / 4) + int(multiprocessing.cpu_count() * 0.05) if (
            args['ncores'] is None) else args['ncores']

    # LOADING DATA AND STUFF
    cedar_dataset = pd.read_csv(f'{args["datadir"]}221028_cedar_related_newcore_fold.csv')
    prime_dataset = pd.read_csv(f'{args["datadir"]}221117_prime_related_newcore_fold.csv')
    merged_dataset = pd.read_csv(f'{args["datadir"]}221112_cedar_prime_merged_fold.csv')
    ibel_dataset = pd.read_csv(f'{args["datadir"]}221117_ibel_merged_fold.csv')
    prime_switch_dataset = pd.read_csv(f'{args["datadir"]}221122_prime_AC_switch.csv')
    ics_shannon = pkl_load(f'{args["icsdir"]}ics_shannon.pkl')
    ics_kl = pkl_load(f'{args["icsdir"]}ics_kl.pkl')

    # Setting trainset
    assert (args['trainset'].lower() in ['cedar', 'prime',
                                         'merged']), 'please input -trainset as either "cedar", "prime" or "merged"'
    trainmap = {'cedar': cedar_dataset,
                'prime': prime_dataset,
                'merged': merged_dataset}
    train_dataset = trainmap[args['trainset']]

    # DEFINING COLS
    aa_cols = ['aliphatic_index', 'boman', 'hydrophobicity', 'isoelectric_point', 'VHSE1', 'VHSE3', 'VHSE7', 'VHSE8']
    mcs = []
    cols_ = ['dissimilarity_score', 'blsm_mut_score', 'mutation_score', 'ratio_rank']
    for L in range(0, len(cols_) + 1):
        for mc in itertools.combinations(cols_, L):
            mcs.append(list(mc))

    cols_ = ['dissimilarity_score', 'core_blsm_mut_score', 'core_mutation_score', 'ratio_rank']
    for L in range(0, len(cols_) + 1):
        for mc in itertools.combinations(cols_, L):
            mcs.append(list(mc))
    mcs.append(aa_cols)
    mcs = list(np.unique(mcs))
    # DEFINING KWARGS
    encoding_kwargs = {'max_len': 12,
                       'encoding': 'onehot',
                       'blosum_matrix': None,
                       'mask': False,  # Using Shannon ICs, true if both mask and name is "shannon"
                       'add_rank': True,
                       'add_aaprop': False,
                       'remove_pep': False,
                       'standardize': True}
    results_related = {}
    mega_df = pd.DataFrame()
    print('Starting loops')
    for rank_col in ['trueHLA_EL_rank', 'EL_rank_mut']:
        results_related[rank_col] = {}
        encoding_kwargs['rank_col'] = rank_col
        for pep_col in ['Peptide', 'icore_mut']:
            results_related[rank_col][pep_col] = {}
            encoding_kwargs['seq_col'] = pep_col
            for mut_cols in tqdm(mcs, position=0, leave=True, desc='cols'):
                key = '-'.join(mut_cols)
                if key == '':
                    key = 'only_rank'
                elif key == 'aliphatic_index-boman-hydrophobicity-isoelectric_point-VHSE1-VHSE3-VHSE7-VHSE8':
                    key = 'aa_props'

                results_related[rank_col][pep_col][key] = {}
                encoding_kwargs['mut_col'] = mut_cols
                # megaloops for encoding-weighting
                for encoding, blosum_matrix, blsm_name in tqdm(zip(['onehot', 'blosum', 'blosum'],
                                                                   [None, BL62_VALUES, BL62FREQ_VALUES],
                                                                   ['onehot', 'BL62LO', 'BL62FREQ']),
                                                               desc='encoding', leave=False, position=1):
                    if blsm_name=='onehot':continue
                    encoding_kwargs['encoding'] = encoding
                    encoding_kwargs['blosum_matrix'] = blosum_matrix
                    results_related[rank_col][pep_col][key][blsm_name] = {}
                    for invert in [True, False]:
                        for ic_name, ics_dict in tqdm(
                                zip(['Mask', 'KL', 'None', 'Shannon'], [ics_shannon, ics_kl, None, ics_shannon]),
                                desc='Weighting', leave=False, position=2):
                            # Set args
                            encoding_kwargs['invert'] = invert
                            encoding_kwargs['mask'] = True if 'Mask' in ic_name else False
                            # Load params
                            if invert:
                                if ic_name == 'None':
                                    continue
                                else:
                                    ic_name = 'Inverted ' + ic_name
                            # Make result dict
                            results_related[rank_col][pep_col][key][blsm_name][ic_name] = {}
                            # Using the same model and hyperparameters
                            model = RandomForestClassifier(n_jobs=1, min_samples_leaf=7, n_estimators=300,
                                                           max_depth=8, ccp_alpha=9.945e-6)
                            # Training model and getting feature importances
                            print('Training')
                            trained_models, train_metrics, _ = nested_kcv_train_mut(train_dataset, model,
                                                                                    ics_dict=ics_dict,
                                                                                    encoding_kwargs=encoding_kwargs,
                                                                                    n_jobs=10)
                            fi = get_nested_feature_importance(trained_models)
                            fn = AA_KEYS + ['rank'] + mut_cols
                            # Saving Feature importances as dataframe
                            df_fi = pd.DataFrame(fi, index=fn).T
                            df_fi.to_csv(
                                f'{args["outdir"]}raw/featimps_{blsm_name}_{"-".join(ic_name.split(" "))}_{pep_col}_{rank_col}_{key}.csv',
                                index=False)

if __name__ == '__main__':
    main()
