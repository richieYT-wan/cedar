import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import multiprocessing
import itertools

from tqdm.auto import tqdm
from datetime import datetime as dt
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Custom fct imports
from src.utils import pkl_load, pkl_dump
import argparse
from src.data_processing import BL62_VALUES, BL62FREQ_VALUES, AA_KEYS
from src.utils import mkdirs, convert_path, str2bool
from src.metrics import get_nested_feature_importance
from src.bootstrap import bootstrap_eval
from src.sklearn_train_eval import nested_kcv_train_sklearn, evaluate_trained_models_sklearn

from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from joblib import Parallel, delayed
from functools import partial
N_CORES = 36


def random_search(param, model_constructor, train_dataset, ics_dict, encoding_kwargs):
    model = model_constructor(**param)
    _, train_metrics, _ = nested_kcv_train_sklearn(train_dataset, model, ics_dict, encoding_kwargs, n_jobs=9)
    mean_valid_auc = np.mean([np.mean([train_metrics[i][j]['valid']['auc'] for j in train_metrics[i].keys()]) \
                              for i in train_metrics.keys()])
    return param, mean_valid_auc


def final_bootstrap_wrapper(preds_df, args, id_name, model_name,
                            ic_name, pep_col, rank_col, key, evalset,
                            n_rounds=10000, n_jobs=40):
    scores = preds_df.pred.values if 'pred' in preds_df.columns else preds_df['mean_pred'].values
    targets = preds_df.agg_label.values if 'agg_label' in preds_df.columns else preds_df['Immunogenicity'].values

    bootstrapped_df, mean_rocs = bootstrap_eval(y_score=scores,
                                                y_true=targets,
                                                n_rounds=n_rounds, n_jobs=n_jobs)
    bootstrapped_df['best_cdt'] = id_name
    bootstrapped_df['encoding'] = 'onehot'
    bootstrapped_df['weight'] = ic_name
    bootstrapped_df['pep_col'] = pep_col
    bootstrapped_df['rank_col'] = rank_col
    bootstrapped_df['key'] = key
    bootstrapped_df['model']=model_name
    bootstrapped_df['evalset'] = evalset.upper()

    bootstrapped_df.to_csv(
        f'{args["outdir"]}bootstrapping/{evalset}_bootstrapped_df_onehot_{"-".join(ic_name.split(" "))}_{pep_col}_{rank_col}_{key}_{model_name}.csv',
        index=False)
    pkl_dump(mean_rocs,
             f'{args["outdir"]}bootstrapping/{evalset}_mean_rocs_onehot_{"-".join(ic_name.split(" "))}_{pep_col}_{rank_col}_{key}_{model_name}.pkl')

    return bootstrapped_df


def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to crossvalidate and evaluate methods that use aa frequency as encoding')

    parser.add_argument('-datadir', type=str, default='../data/mutant/',
                        help='Path to directory containing the pre-partitioned data')
    parser.add_argument('-outdir', type=str, default='../output/230109_hyperparams_retune/')
    parser.add_argument('-icsdir', type=str, default='../data/ic_dicts/',
                        help='Path containing the pre-computed ICs dicts.')
    parser.add_argument('-ncores', type=int, default=40,
                        help='N cores to use in parallel, by default will be multiprocesing.cpu_count() * 3/4')
    parser.add_argument('-mask_aa', type=str, default='false', help='Which AA to mask. has to be Capital letters and '
                                                                    'within the standard amino acid alphabet. To '
                                                                    'disable, input "false". "false" by default.')
    parser.add_argument('-add_rank', type=str2bool, default=True, help='Whether to add rank as a feature or not')
    parser.add_argument('-frac_iter', type=float, default=.67, help='The proportion of how many hyperparams '
                                                                    'combination (out of the total) to test. By '
                                                                    'default, 0.67 of the grid will be randomly '
                                                                    'sampled')
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
    cedar_dataset = pd.read_csv(f'{args["datadir"]}221028_cedar_related_newcore_fold.csv')
    prime_dataset = pd.read_csv(f'{args["datadir"]}221117_prime_related_newcore_fold.csv')
    ibel_dataset = pd.read_csv(f'{args["datadir"]}221117_ibel_merged_fold.csv')
    ics_shannon = pkl_load(f'{args["icsdir"]}ics_shannon.pkl')
    ics_kl = pkl_load(f'{args["icsdir"]}ics_kl.pkl')

    aa_cols = ['aliphatic_index', 'boman', 'hydrophobicity', 'isoelectric_point', 'VHSE1', 'VHSE3', 'VHSE7', 'VHSE8']

    # Getting "best" encodings here
    best_cedar = (dict(max_len=12, encoding='onehot', blosum_matrix=None, mask=False, add_rank=True,
                       add_aaprop=False, remove_pep=False, standardize=True, invert=True,
                       target_col='agg_label', seq_col='Peptide', rank_col='EL_rank_mut', hla_col='HLA',
                       mut_col=aa_cols, mask_aa=False),
                  ics_kl, 'Inverted KL', 'Best CEDAR')

    best_prime = (dict(max_len=12, encoding='onehot', blosum_matrix=None, mask=True, add_rank=True,
                       add_aaprop=False, remove_pep=False, standardize=True, invert=False,
                       target_col='agg_label', seq_col='Peptide', rank_col='EL_rank_mut', hla_col='HLA',
                       mut_col=['dissimilarity_score', 'blsm_mut_score'], mask_aa=False),
                  ics_shannon, 'Mask', 'Best PRIME')

    best_avg = (dict(max_len=12, encoding='onehot', blosum_matrix=None, mask=True, add_rank=True,
                     add_aaprop=False, remove_pep=False, standardize=True, invert=False,
                     target_col='agg_label', seq_col='Peptide', rank_col='EL_rank_mut', hla_col='HLA',
                     mut_col=['core_blsm_mut_score', 'core_mutation_score'], mask_aa=False),
                ics_shannon, 'Mask', 'Best AVG')

    best_extra = (dict(max_len=12, encoding='onehot', blosum_matrix=None, mask=False, add_rank=True,
                       add_aaprop=False, remove_pep=False, standardize=True, invert=False,
                       target_col='agg_label', seq_col='Peptide', rank_col='EL_rank_mut', hla_col='HLA',
                       mut_col=['dissimilarity_score', 'blsm_mut_score'], mask_aa=False),
                  ics_kl, 'KL', 'Extra KL Peptide')

    best_extra2 = (dict(max_len=12, encoding='onehot', blosum_matrix=None, mask=False, add_rank=True,
                        add_aaprop=False, remove_pep=False, standardize=True, invert=False,
                        target_col='agg_label', seq_col='icore_mut', rank_col='EL_rank_mut', hla_col='HLA',
                        mut_col=['dissimilarity_score', 'blsm_mut_score'], mask_aa=False),
                   ics_kl, 'KL', 'Extra KL icore_mut')
    # Add one extra condition using the best avg kwargs but with a KL info content

    mega_df = []
    print('Starting loops')
    rf_hp = {'n_estimators': [10, 50, 100, 200, 300],
             'max_depth': [3, 5, 7, 8, 9],
             'criterion': ['gini', 'entropy'],
             'ccp_alpha': np.logspace(-12, -3, 7),
             'min_samples_leaf': [1, 3, 5, 7, 9]}  # 5*3*2*4*3 = 360 combi

    xgb_hp = {'n_estimators': [10, 50, 100, 200, 300],
              'max_depth': [4, 6, 8],
              'learning_rate': [.1, .3, .5],
              'alpha': np.logspace(-15, -2, 7),
              'lambda': np.logspace(-15, -2, 7),
              'colsample_by_tree': np.linspace(0.5, 1.1, 3),
              'subsample': [0.6, .8, 1]
              }
    for best_kwargs, ics_dict, ic_name, id_name in [best_cedar, best_prime, best_avg, best_extra, best_extra2]:
        for hp, model in zip([rf_hp, xgb_hp], [RandomForestClassifier, XGBClassifier]):
            n_iter = int(args['frac_iter']*len(list(ParameterGrid(hp))))
            params_list = list(ParameterSampler(hp, n_iter=n_iter, random_state=13))
            random_search_wrapper = partial(random_search, model_constructor=model,
                                            train_dataset= cedar_dataset, ics_dict=ics_dict,encoding_kwargs=best_kwargs)
            output = Parallel(n_jobs=1)(delayed(random_search_wrapper)(param=param) for param in params_list)
            best_hyperparams = list(reversed(sorted(output, key=lambda x: x[1])))[0][0]
            best_model = model(**best_hyperparams)
            trained_models, _, _ = nested_kcv_train_sklearn(cedar_dataset, best_model, ics_dict, best_kwargs, n_jobs=9)
            key = 'aa_props' if best_kwargs['mut_col'][0] == aa_cols[0] else '-'.join(best_kwargs['mut_col'])
            fn = f'{id_name}_{ic_name}_{model.__name__}'
            pkl_dump(best_hyperparams, f'{fn}_hyperparams.pkl')

            for evalset, evalname in zip([cedar_dataset, prime_dataset, ibel_dataset],['CEDAR','PRIME','IBEL']):
                _, preds = evaluate_trained_models_sklearn(evalset, trained_models, ics_dict, train_dataframe=cedar_dataset,
                                                           encoding_kwargs=best_kwargs, concatenated=True, only_concat=True)
                preds.to_csv(f'{args["outdir"]}/raw/{evalname}_preds_{fn}.csv', index=False)
                bootstrapped_df = final_bootstrap_wrapper(preds, args, id_name = id_name, model_name = model.__name__, ic_name=ic_name, pep_col = best_kwargs['seq_col'],
                                                          rank_col = best_kwargs['rank_col'], key = key, evalset=evalname, n_rounds=10000, n_jobs=N_CORES)
                mega_df.append(bootstrapped_df)

    pd.concat(mega_df).to_csv(f'{args["outdir"]}/total_df.csv', index=False)


if __name__ == '__main__':
    main()
