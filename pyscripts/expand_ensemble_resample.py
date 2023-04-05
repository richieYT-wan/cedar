import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import multiprocessing
import itertools

from tqdm.auto import tqdm
import sklearn
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
from src.metrics import get_nested_feature_importance
from src.bootstrap import bootstrap_eval
from src.sklearn_train_eval import nested_kcv_train_sklearn, evaluate_trained_models_sklearn
from copy import deepcopy

N_CORES = 39

def final_bootstrap_wrapper(preds_df, args, filename, expand, condition, evalset,
                            n_rounds=10000, n_jobs=36):
    scores = preds_df.pred.values if 'pred' in preds_df.columns else preds_df['mean_pred'].values
    targets = preds_df.agg_label.values if 'agg_label' in preds_df.columns else preds_df['Immunogenicity'].values

    bootstrapped_df, mean_rocs = bootstrap_eval(y_score=scores,
                                                y_true=targets,
                                                n_rounds=n_rounds, n_jobs=n_jobs)
    bootstrapped_df['condition'] = condition
    bootstrapped_df['expand'] = expand
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
def parallel_inner_train_wrapper(train_dataframe, x_test, base_model, ics_dict,
                                 encoding_kwargs, standardize, fold_out, fold_in):
    seed = fold_out * 10 + fold_in
    
    # Here resets model weight at every fold, using the fold number (range(0, n_folds*(n_folds-1)) ) as seed
    # Query subset dataframe and get encoded data and targets
    train = train_dataframe.query('fold != @fold_out and fold != @fold_in').reset_index(drop=True)
    valid = train_dataframe.query('fold == @fold_in').reset_index(drop=True)
    
    expanded_models = []
    # expanded_train_metrics = []
    # expanded_valid_metrics = []
    expanded_y_pred_test = []

    # Here re-sample 10x
    for resample in range(1,11):
        model = sklearn.base.clone(base_model)
        model.set_params(random_state=(resample*seed)+resample+seed)
        tmp = train.sample(len(train), random_state=resample, replace=True)
        # Get datasets
        x_train, y_train = get_dataset(train, ics_dict, **encoding_kwargs)
        x_valid, y_valid = get_dataset(valid, ics_dict, **encoding_kwargs)
        # Fit the model and append it to the list
        model.fit(x_train, y_train)
        expanded_models.append(model)

    return expanded_models

def nested_kcv_train_sklearn_expand():
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
    for fold_out in tqdm(folds, leave=False, desc='Outer fold', position=2):
        # Get test set & init models list to house all models trained in inner fold
        test = dataframe.query('fold == @fold_out').reset_index(drop=True)
        x_test, y_test = get_dataset(test, ics_dict, **encoding_kwargs)
        # For a given fold, all the models that are trained should be appended to this list
        inner_folds = sorted([f for f in folds if f != fold_out])
        # N jobs must be lower than cpu_count
        n_jobs = min(multiprocessing.cpu_count()-1, len(inner_folds)) if n_jobs is None\
            else n_jobs if (n_jobs is not None and n_jobs <= multiprocessing.cpu_count()) \
            else multiprocessing.cpu_count()-1
        # Create the sub-dict and put the model into the models dict
        train_wrapper_ = partial(parallel_inner_train_wrapper, train_dataframe=dataframe, x_test=x_test,
                                 base_model=base_model, ics_dict=ics_dict, encoding_kwargs=encoding_kwargs,
                                 standardize=std, fold_out=fold_out)
        output = Parallel(n_jobs=n_jobs)(
            delayed(train_wrapper_)(fold_in=fold_in) for fold_in in tqdm(inner_folds,
                                                                         desc='Inner Folds',
                                                                         leave=False, position=1))
        models_dict[fold_out] = [flatten_list(x[0]) for x in output]

    return models_dict, None, None

######## ARGS PARSE & MAIN
def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to crossvalidate and evaluate methods that use aa frequency as encoding')

    parser.add_argument('-datadir', type=str, default='../data/pepx/',
                        help='Path to directory containing the pre-partitioned data')
    parser.add_argument('-outdir', type=str, default='../output/230405_ExpandEnsembleResample/')
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
    cedar_dataset = pd.read_csv(f'{args["datadir"]}230308_cedar_aligned_pepx_fold.csv')
    prime_dataset = pd.read_csv(f'{args["datadir"]}230308_prime_aligned_pepx.csv')
    nepdb_dataset = pd.read_csv(f'{args["datadir"]}230308_nepdb_aligned_pepx.csv')

    ics_shannon = pkl_load(f'{args["icsdir"]}ics_shannon.pkl')
    ics_kl = pkl_load(f'{args["icsdir"]}ics_kl.pkl')

    # Setting trainset
    trainmap = {'cedar': cedar_dataset,
                'prime': prime_dataset}
    expandmap = {True: (nested_kcv_train_sklearn_expand, evaluate_trained_models_sklearn),
                 False: (nested_kcv_train_sklearn, evaluate_trained_models_sklearn)}

    assert (args['trainset'].lower() in trainmap.keys()), f'please input -trainset as either one of {trainmap.keys()}'

    train_dataset = trainmap[args['trainset']]


    # DEFINING COLS
    aa_cols = ['aliphatic_index', 'boman', 'hydrophobicity', 'isoelectric_point', 'VHSE1', 'VHSE3', 'VHSE7', 'VHSE8']
    
    # Defining cdt = (kwargs, ics_dict, name)
    cdt_base = (dict(max_len=12, encoding='onehot', blosum_matrix=None, mask=False, 
                       add_rank=True, seq_col='Peptide', rank_col='trueHLA_EL_rank', hla_col='HLA',
                       target_col = 'agg_label', add_aaprop=False, remove_pep=False, standardize=True),
                None, 'Base')
    cdt_cedar = (dict(max_len=12, encoding='onehot', blosum_matrix='None', add_rank=True, seq_col='icore_mut', rank_col='EL_rank_mut', target_col ='agg_label', hla_col='HLA',
                        add_aaprop=False, remove_pep=False, standardize=True,
                        mask=False, invert=True,
                        mut_col = aa_cols+['EL_rank_wt_aligned','icore_dissimilarity_score','ratio_rank','Total_Gene_TPM']
                        ), # Here it should be icore similarity score because it's not dissimilarity that we have
                    ics_shannon, 'OptCEDAR')

    cdt_prime = (dict(max_len=12, encoding='onehot', blosum_matrix='None', add_rank=True, seq_col='icore_mut', rank_col='EL_rank_mut', target_col ='agg_label', hla_col='HLA',
                        add_aaprop=False, remove_pep=False, standardize=True,
                        mask=True, invert=False,
                        mut_col = ['icore_dissimilarity_score', 'icore_blsm_mut_score']),
                    ics_shannon, 'OptPRIME')

    cdt_general = (dict(max_len=12, encoding='onehot', blosum_matrix='None', add_rank=True, seq_col='icore_mut', rank_col='EL_rank_mut', target_col ='agg_label', hla_col='HLA',
                          add_aaprop=False, remove_pep=False, standardize=True,
                          mask=False, invert=False,
                          mut_col = ['icore_dissimilarity_score', 'icore_blsm_mut_score', 'ratio_rank', 'Total_Gene_TPM']),
                     None, 'General')


    print('Starting loops')
    for encoding_kwargs, ics_dict, condition in [cdt_general,cdt_base, cdt_cedar, cdt_prime]:
        mut_cols = encoding_kwargs['mut_col'] if condition!='Base' else None
        for expand_ensemble in [False, True]:
            train_fct, eval_fct = expandmap[expand_ensemble]
            filename = f'expandEnsemble{expand_ensemble}_Condition{condition}'
            # Using the same model and hyperparameters
            model = RandomForestClassifier(n_jobs=1, min_samples_leaf=7, n_estimators=300,
                                               max_depth=8, ccp_alpha=9.945e-6)
            # Training model and getting feature importances
            print('Training')
            trained_models, train_metrics, _ = train_fct(train_dataset, model,
                                                                        ics_dict=ics_dict,
                                                                        encoding_kwargs=encoding_kwargs,
                                                                        n_jobs=args["ncores"])
            fi = get_nested_feature_importance(trained_models)
            fn = AA_KEYS + ['rank'] + mut_cols if mut_cols is not None else AA_KEYS+['rank']
            # Saving Feature importances as dataframe
            df_fi = pd.DataFrame(fi, index=fn).T
            df_fi.to_csv(
                f'{args["outdir"]}raw/featimps_{filename}.csv',
                index=False)

            for evalset, evalname in zip([cedar_dataset, prime_dataset, nepdb_dataset],
                                         ['CEDAR', 'PRIME', 'NEPDB']):
                # FULLY FILTERED + Mean_pred
                if not evalset.equals(train_dataset):
                    evalset = evalset.query('Peptide not in @train_dataset.Peptide.values')
                _, preds = eval_fct(evalset.drop_duplicates(subset=['Peptide','HLA','agg_label']),
                                                           trained_models, ics_dict,
                                                           train_dataset,
                                                           encoding_kwargs, concatenated=False,
                                                           only_concat=True, n_jobs=args["ncores"])

                # p_col = 'pred' if 'pred' in preds.columns else 'mean_pred'
                preds.drop(columns=aa_cols).to_csv(f'{args["outdir"]}raw/{evalname}_preds_{filename}.csv', index=False)

                bootstrapped_df = final_bootstrap_wrapper(preds, args, filename, expand_ensemble, condition, 
                                                          evalname, n_rounds=10000, n_jobs = args["ncores"])
                mega_df = mega_df.append(bootstrapped_df)

    mega_df.to_csv(f'{args["outdir"]}/total_df.csv', index=False)


if __name__ == '__main__':
    main()
