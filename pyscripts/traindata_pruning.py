import pandas as pd
import numpy as np
import multiprocessing
import itertools

from tqdm.auto import tqdm
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from functools import partial

from datetime import datetime as dt
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Custom fct imports
from src.utils import pkl_load, pkl_dump
import argparse
from src.data_processing import BL62_VALUES, BL62FREQ_VALUES, AA_KEYS, get_dataset
from src.utils import mkdirs, convert_path, str2bool
from src.metrics import get_nested_feature_importance, get_metrics
from src.bootstrap import bootstrap_eval
from src.sklearn_train_eval import nested_kcv_train_sklearn, evaluate_trained_models_sklearn
import copy
from copy import deepcopy

N_CORES = 39


def assert_encoding_kwargs(encoding_kwargs, mode_eval=False):
    """
    Assertion / checks for encoding kwargs and verify all the necessary key-values
    are in
    """
    # Making a deep copy since dicts are mutable between fct calls
    encoding_kwargs = copy.deepcopy(encoding_kwargs)
    if encoding_kwargs is None:
        encoding_kwargs = {'max_len': 12,
                           'encoding': 'onehot',
                           'blosum_matrix': None,
                           'standardize': False}
    essential_keys = ['max_len', 'encoding', 'blosum_matrix', 'standardize']
    keys_check = [x in encoding_kwargs.keys() for x in essential_keys]
    keys_check_dict = {k: v for (k, v) in zip(essential_keys, keys_check) if v == False}
    assert all(keys_check), f'Encoding kwargs don\'t contain the essential key-value pairs! ' \
                            f"{list(keys_check_dict.keys())} are missing!"

    if mode_eval:
        if any([(x not in encoding_kwargs.keys()) for x in ['seq_col', 'hla_col', 'target_col', 'rank_col']]):
            if 'seq_col' not in encoding_kwargs.keys():
                encoding_kwargs.update({'seq_col': 'icore_mut'})
            if 'hla_col' not in encoding_kwargs.keys():
                encoding_kwargs.update({'hla_col': 'HLA'})
            if 'target_col' not in encoding_kwargs.keys():
                encoding_kwargs.update({'target_col': 'agg_label'})
            if 'rank_col' not in encoding_kwargs.keys():
                encoding_kwargs.update({'rank_col': 'EL_rank_mut'})

        # This KWARGS not needed in eval mode since I'm using Pipeline and Pipeline
        del encoding_kwargs['standardize']
    return encoding_kwargs


def final_bootstrap_wrapper(preds_df, args, filename, percentile, pruned, evalset,
                            n_rounds=10000, n_jobs=36):
    scores = preds_df.pred.values if 'pred' in preds_df.columns else preds_df['mean_pred'].values
    targets = preds_df.agg_label.values if 'agg_label' in preds_df.columns else preds_df['Immunogenicity'].values

    bootstrapped_df, mean_rocs = bootstrap_eval(y_score=scores,
                                                y_true=targets,
                                                n_rounds=n_rounds, n_jobs=n_jobs)
    bootstrapped_df['condition'] = percentile
    bootstrapped_df['pruned'] = pruned
    bootstrapped_df['evalset'] = evalset.upper()

    bootstrapped_df.to_csv(
        f'{args["outdir"]}bootstrapping/{evalset}_bootstrapped_df_{filename}.csv',
        index=False)
    pkl_dump(mean_rocs,
             f'{args["outdir"]}bootstrapping/{evalset}_mean_rocs_{filename}.pkl')

    return bootstrapped_df


def flatten_list(list_of_list):
    return [x for list_ in list_of_list for x in list_]


######### TRAIN FCTS DEF
# For each

# TRAIN WITH PARALLEL WRAPPER

# TRAIN WITH PARALLEL WRAPPER
def parallel_inner_train_wrapper_prune(train_dataframe, x_test, base_model, ics_dict,
                                 encoding_kwargs, standardize, fold_out, fold_in):
    seed = fold_out * 10 + fold_in
    # Copy the base model, resets the seed
    model = sklearn.base.clone(base_model)
    model.set_params(random_state=seed)
    if standardize:
        model = Pipeline([('scaler', StandardScaler()), ('model', model)])

    # Here resets model weight at every fold, using the fold number (range(0, n_folds*(n_folds-1)) ) as seed
    # Query subset dataframe and get encoded data and targets and remove misclassified validation from training set
    train = train_dataframe.query('fold != @fold_out and fold != @fold_in and not misclassified').reset_index(drop=True)
    valid = train_dataframe.query('fold == @fold_in').reset_index(drop=True)
    # Get datasets
    x_train, y_train = get_dataset(train, ics_dict, **encoding_kwargs)
    x_valid, y_valid = get_dataset(valid, ics_dict, **encoding_kwargs)

    # Fit the model and append it to the list
    model.fit(x_train, y_train)

    # Get the prediction values on both the train and validation set
    y_train_pred, y_train_score = model.predict(x_train), model.predict_proba(x_train)[:, 1]
    y_val_pred, y_val_score = model.predict(x_valid), model.predict_proba(x_valid)[:, 1]
    # Get the metrics and save them
    try:
        train_metrics = get_metrics(y_train, y_train_score, y_train_pred)
    except:
        print(train_dataframe.head())
        raise ValueError(f'{encoding_kwargs}')
    try:
        valid_metrics = get_metrics(y_valid, y_val_score, y_val_pred)
    except:
        print(train_dataframe.head())
        raise ValueError(f'{encoding_kwargs}')
    y_pred_test = model.predict_proba(x_test)[:, 1]

    return model, train_metrics, valid_metrics, y_pred_test


def nested_kcv_train_sklearn_prune(dataframe, base_model, ics_dict, encoding_kwargs: dict = None, n_jobs: int = None):
    """
    Args:
        dataframe:
        base_model:
        ics_dict:
        encoding_kwargs:
        n_jobs (int): number of parallel processes. If None, will use len(inner_folds)

    Returns:
        models_fold
        train_results
        test_results
    """
    encoding_kwargs = assert_encoding_kwargs(encoding_kwargs, mode_eval=False)
    models_dict = {}
    test_metrics = {}
    train_metrics = {}
    folds = sorted(dataframe.fold.unique())
    std = encoding_kwargs.pop('standardize')
    for fold_out in tqdm(folds, leave=False, desc='Train:Outer fold', position=2):
        # Get test set & init models list to house all models trained in inner fold
        test = dataframe.query('fold == @fold_out').reset_index(drop=True)
        x_test, y_test = get_dataset(test, ics_dict, **encoding_kwargs)
        # For a given fold, all the models that are trained should be appended to this list
        inner_folds = sorted([f for f in folds if f != fold_out])
        # N jobs must be lower than cpu_count
        n_jobs = min(multiprocessing.cpu_count() - 1, len(inner_folds)) if n_jobs is None \
            else n_jobs if (n_jobs is not None and n_jobs <= multiprocessing.cpu_count()) \
            else multiprocessing.cpu_count() - 1
        # Create the sub-dict and put the model into the models dict
        train_wrapper_ = partial(parallel_inner_train_wrapper_prune, train_dataframe=dataframe, x_test=x_test,
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


def get_misclassified(pred, label, bot, top):
    if pred<bot and label == 1:
        return 'FN'
    if pred>top and label == 0:
        return 'FP'
    else:
        return 'Normal'



######## ARGS PARSE & MAIN
def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to crossvalidate and evaluate methods that use aa frequency as encoding')

    parser.add_argument('-datadir', type=str, default='../data/pepx/',
                        help='Path to directory containing the pre-partitioned data')
    parser.add_argument('-outdir', type=str, default='../output/230721_TrainDataPruning/')
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
    mkdirs(args['outdir'])
    mkdirs(f'{args["outdir"]}raw/')
    mkdirs(f'{args["outdir"]}bootstrapping/')
    N_CORES = int(multiprocessing.cpu_count() * 3 / 4) + int(multiprocessing.cpu_count() * 0.05) if (
            args['ncores'] is None) else args['ncores']

    # LOADING DATA AND STUFF
    cedar_dataset = pd.read_csv(f'{args["datadir"]}230418_cedar_aligned_pepx.csv')
    prime_dataset = pd.read_csv(f'{args["datadir"]}230418_prime_aligned_pepx.csv')
    nepdb_dataset = pd.read_csv(f'{args["datadir"]}230418_nepdb_aligned_pepx.csv')

    preds_100k = pd.read_csv('../data/human_proteome/preds_100k.txt', header=None)
    ics_kl = pkl_load(f'{args["icsdir"]}ics_kl_new.pkl')

    # Setting trainset
    trainmap = {'cedar': cedar_dataset,
                'prime': prime_dataset}

    assert (args['trainset'].lower() in trainmap.keys()), f'please input -trainset as either one of {trainmap.keys()}'

    train_dataset = trainmap[args['trainset']]

    # DEFINING COLS
    aa_cols = ['aliphatic_index', 'boman', 'hydrophobicity', 'isoelectric_point', 'VHSE1', 'VHSE3', 'VHSE7', 'VHSE8']

    kwargs, ics = dict(max_len=12, encoding='onehot', blosum_matrix='None', add_rank=True,
                       seq_col='icore_mut', rank_col='EL_rank_mut', target_col='agg_label', hla_col='HLA',
                       add_aaprop=False, remove_pep=False, standardize=True,
                       mask=True, invert=False, threshold = 0.2,
                       mut_col=['icore_dissimilarity_score', 'icore_blsm_mut_score', 'Total_Gene_TPM']), ics_kl
    mega_df = pd.DataFrame()

    # Top and bottom X percentiles
    for percentile_thr in range(1,16):
        bot, top = [percentile_thr/100, (100-percentile_thr)/100]
        bot, top = preds_100k.describe(percentiles=[bot, top]).loc[[f'{bot:.0%}', f'{top:.0%}']].values

        model = RandomForestClassifier(n_jobs=1, min_samples_leaf=7, n_estimators=300,
                                       max_depth=8, ccp_alpha=9.945e-6)

        # First training, with KCV preds + test preds (PRIME) and no pruning
        trained_models, _, _ = nested_kcv_train_sklearn(cedar_dataset, model, ics, kwargs, n_jobs=10)
        _, kcv_preds = evaluate_trained_models_sklearn(cedar_dataset, trained_models, ics, None,
                                                       kwargs, concatenated=False, only_concat=False, n_jobs=10,
                                                       kcv_eval=True)
        _, prime_preds = evaluate_trained_models_sklearn(prime_dataset.query('Peptide not in @cedar_dataset.Peptide.values'),
                                                        trained_models, ics, None,
                                                        kwargs, concatenated=False, only_concat=False, n_jobs=10,
                                                        kcv_eval=False)
        _, nepdb_preds = evaluate_trained_models_sklearn(nepdb_dataset.query('Peptide not in @cedar_dataset.Peptide.values'),
                                                        trained_models, ics, None,
                                                        kwargs, concatenated=False, only_concat=False, n_jobs=10,
                                                        kcv_eval=False)
        # Second training, with pruned datapoints
        kcv_preds['class'] = kcv_preds.apply(lambda x: get_misclassified(x['mean_pred'], x['agg_label'], bot, top), axis=1)
        kcv_preds['misclassified'] = kcv_preds['class'] != 'Normal'

        trained_models_prune, _, _ = nested_kcv_train_sklearn_prune(kcv_preds, model, ics, kwargs, n_jobs=10)

        _, kcv_preds_prune = evaluate_trained_models_sklearn(cedar_dataset, trained_models_prune, ics, None,
                                                       kwargs, concatenated=False, only_concat=False, n_jobs=10,
                                                       kcv_eval=True)

        _, prime_preds_prune = evaluate_trained_models_sklearn(prime_dataset.query('Peptide not in @cedar_dataset.Peptide.values'),
                                                        trained_models_prune, ics, None,
                                                        kwargs, concatenated=False, only_concat=False, n_jobs=10,
                                                        kcv_eval=False)

        _, nepdb_preds_prune = evaluate_trained_models_sklearn(nepdb_dataset.query('Peptide not in @cedar_dataset.Peptide.values'),
                                                        trained_models_prune, ics, None,
                                                        kwargs, concatenated=False, only_concat=False, n_jobs=10,
                                                        kcv_eval=False)

        for preds, evalname, pruned in [(kcv_preds, 'KCV', False),
                                        (prime_preds, 'PRIME', False),
                                        (nepdb_preds, 'NEPDB', False),
                                        (kcv_preds_prune, 'KCV', True),
                                        (prime_preds_prune, 'PRIME', True),
                                        (nepdb_preds_prune, 'NEPDB', True)]:
            filename = f'Pruning{pruned}_Percentile_{percentile_thr:02}'
            preds.to_csv(f'{args["outdir"]}/raw/{evalname}_preds_{filename}.csv', index=False)
            bdf = final_bootstrap_wrapper(preds, args, filename, percentile_thr, pruned, evalname,
                                          n_rounds=10000, n_jobs=args['ncores'])
            mega_df = mega_df.append(bdf)
        mega_df.to_csv(f'{args["outdir"]}/total_df.csv', index=False)

if __name__ == '__main__':
    main()
