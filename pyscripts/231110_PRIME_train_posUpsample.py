import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import multiprocessing
import itertools
from functools import partial
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from datetime import datetime as dt
import os, sys
import copy
import tracemalloc

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Custom fct imports
from src.utils import pkl_load, pkl_dump
import argparse
from src.data_processing import BL62_VALUES, BL62FREQ_VALUES, AA_KEYS, get_aa_properties
from src.utils import mkdirs, convert_path, str2bool
from src.metrics import get_nested_feature_importance
from src.bootstrap import bootstrap_eval, get_pval_wrapper
from src.sklearn_train_eval import nested_kcv_train_sklearn, evaluate_trained_models_sklearn
from copy import deepcopy

N_CORES = 39


def final_bootstrap_wrapper(preds_df, args, filename,
                            ic_name, replacement, upsample, key, evalset,
                            n_rounds=10000, n_jobs=36):
    scores = preds_df.pred.values if 'pred' in preds_df.columns else preds_df['mean_pred'].values
    targets = preds_df.agg_label.values if 'agg_label' in preds_df.columns else preds_df['Immunogenicity'].values

    bootstrapped_df = bootstrap_eval(y_score=scores, y_true=targets, n_rounds=n_rounds,
                                     n_jobs=n_jobs, add_roc=False, reduced=True)
    bootstrapped_df['encoding'] = 'onehot'
    bootstrapped_df['weight'] = ic_name
    bootstrapped_df['input_type'] = 'icore_mut'
    bootstrapped_df['with_replacement'] = replacement
    bootstrapped_df['n_upsample'] = upsample
    bootstrapped_df['key'] = key
    bootstrapped_df['evalset'] = evalset.upper()

    bootstrapped_df.to_csv(
        f'{args["outdir"]}bootstrapping/{evalset}_bootstrapped_df_{filename}.csv',
        index=False)

    return bootstrapped_df


def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to crossvalidate and evaluate methods that use aa frequency as encoding')

    parser.add_argument('-datadir', type=str, default='../data/pepx/',
                        help='Path to directory containing the pre-partitioned data')
    parser.add_argument('-outdir', type=str,
                        default='../output/231110_TrainPRIME_Upsample_Prime_Positives/')
    parser.add_argument('-replace', type=str2bool, default=False)
    parser.add_argument('-trainset', type=str, default='cp_merged')
    parser.add_argument('-icsdir', type=str, default='../data/ic_dicts/',
                        help='Path containing the pre-computed ICs dicts.')
    parser.add_argument('-ncores', type=int, default=36,
                        help='N cores to use in parallel, by default will be multiprocesing.cpu_count() * 3/4')
    parser.add_argument('-condition', type=str, default='None',
                        help='Inverted-Shannon, Mask or None. Must be string, so "None"')
    parser.add_argument('-mc_index', type=int, default=None, help='sample a single condition')
    parser.add_argument('-key', type=str, default=None,
                        help='a - separated string (i.e. the key) to make the mutcols from. Can\'t be used together with mc_index!!')
    parser.add_argument('-wc', type=str2bool, default=False,
                        help='Wildcard : Bypass everything and train harmonic model')
    parser.add_argument('-input_type', type=str, default='icore_mut', help='icore_mut, expanded_input, or Peptide')
    parser.add_argument('-debug', type=str2bool, default=False)
    return parser.parse_args()


def main():
    tracemalloc.start()
    start = dt.now()
    args = vars(args_parser())
    args['outdir'], args['datadir'], args['icsdir'] = convert_path(args['outdir']), convert_path(
        args['datadir']), convert_path(args['icsdir'])
    print('Making dirs')
    print('Sanity check')
    mkdirs(args['outdir'])
    mkdirs(f'{args["outdir"]}raw/')
    mkdirs(f'{args["outdir"]}bootstrapping/')
    print('Using ', args['ncores'], ' cores')
    N_CORES = int(multiprocessing.cpu_count() * 3 / 4) + int(multiprocessing.cpu_count() * 0.05) if (
            args['ncores'] is None) else args['ncores']

    # LOADING DATA AND STUFF
    cedar_dataset = pd.read_csv(f'{args["datadir"]}230418_cedar_aligned_pepx.csv')
    prime_dataset = pd.read_csv(f'{args["datadir"]}230418_prime_aligned_pepx.csv')
    nepdb_dataset = pd.read_csv(f'{args["datadir"]}230418_nepdb_aligned_pepx.csv')
    cp_dataset = pd.read_csv(f'{args["datadir"]}231018_cedar_prime_merged_fold.csv')
    ics_kl = pkl_load(f'{args["icsdir"]}ics_kl_new.pkl')

    encoding_kwargs = {'max_len': 12,
                       'encoding': 'onehot',
                       'blosum_matrix': None,
                       'mask': False,  # Using Shannon ICs, true if both mask and name is "shannon"
                       'add_rank': True,
                       'seq_col': 'icore_mut',
                       'rank_col': 'EL_rank_mut',
                       'hla_col': 'HLA',
                       'add_aaprop': False,
                       'remove_pep': False,
                       'standardize': True}

    conditions_list = {'KL-Mask': (False, 'KL-Mask', ics_kl, True)}
    invert, ic_name, ics_dict, mask = conditions_list['KL-Mask']
    # megaloops for encoding-weighting
    mut_cols = ['hydrophobicity', 'icore_blsm_mut_score', 'ratio_rank']
    encoding_kwargs['mut_col'] = mut_cols
    key = '-'.join(mut_cols).replace(' ', '-')

    # You fucking moron ; ICname should've just been args["condition"] from the start instead of this hardcoded nonsense bullshit FUCK I hate you so much

    if 'KL' in args["condition"]:
        encoding_kwargs['threshold'] = 0.200
    encoding_kwargs['invert'] = invert
    encoding_kwargs['mask'] = mask

    for upsample in tqdm([0] + list(range(5, 21)), desc='ups'):
        if args['replace']:
            train_dataset = pd.concat([cp_dataset.query('in_prime and not in_cedar'),
                                       cp_dataset.query('in_prime and not in_cedar and agg_label==1') \
                                      .sample(frac=upsample, replace=True)])
        else:
            train_dataset = pd.concat([cp_dataset.query('in_prime and not in_cedar')] + \
                                      [cp_dataset.query('in_prime and not in_cedar and agg_label==1')] * upsample)

        filename = f'trainPrimeUpsamplePos_n_{upsample:02}_replacement{args["replace"]}_onehot_KL-Mask_{key}'
        # Using the same model and hyperparameters
        model = RandomForestClassifier(n_jobs=1, min_samples_leaf=7, n_estimators=300,
                                       max_depth=8, ccp_alpha=9.945e-6)
        # Training model and getting feature importances
        print('Training')
        trained_models, _, _ = nested_kcv_train_sklearn(train_dataset, model,
                                                        ics_dict=ics_dict,
                                                        encoding_kwargs=encoding_kwargs,
                                                        n_jobs=min(10, args['ncores']))
        fi = get_nested_feature_importance(trained_models)
        fn = AA_KEYS + ['rank'] + mut_cols
        # Saving Feature importances as dataframe
        df_fi = pd.DataFrame(fi, index=fn).T
        df_fi.to_csv(
            f'{args["outdir"]}raw/featimps_{filename}.csv',
            index=False)

        for evalset, evalname in zip([prime_dataset, cedar_dataset],
                                     ['PRIME_KCV', 'CEDAR']):

            _, preds = evaluate_trained_models_sklearn(evalset.drop_duplicates(), trained_models, ics_dict, train_dataset,
                                                       encoding_kwargs, False, True, min(10, args['ncores']),
                                                       kcv_eval=(evalname == "PRIME_KCV"))

            p_col = 'pred' if 'pred' in preds.columns else 'mean_pred'
            preds.to_csv(f'{args["outdir"]}raw/{evalname}_preds_{filename}.csv', index=False,
                         columns=['HLA', 'Peptide', 'agg_label', 'icore_mut', 'icore_wt_aligned', 'EL_rank_mut',
                                  'EL_rank_wt_aligned'] + mut_cols + [p_col])
            bootstrapped_df = final_bootstrap_wrapper(preds, args, filename, ic_name, args['replace'], upsample,
                                                      key, evalname, n_rounds=10000, n_jobs=args['ncores'])


    end = dt.now()
    elapsed = divmod((end - start).seconds, 60)
    print(f'Elapsed: {elapsed[0]} minutes, {elapsed[1]} seconds. ; Memory used: {tracemalloc.get_traced_memory()}')
    tracemalloc.stop()


if __name__ == '__main__':
    main()
