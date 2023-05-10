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

N_CORES = 40


def final_bootstrap_wrapper(preds_df, args, filename,
                            ic_name, key, evalset,
                            n_rounds=10000, n_jobs=1):
    scores = preds_df.pred.values if 'pred' in preds_df.columns else preds_df['mean_pred'].values
    targets = preds_df.agg_label.values if 'agg_label' in preds_df.columns else preds_df['Immunogenicity'].values

    bootstrapped_df = bootstrap_eval(y_score=scores, y_true=targets, n_rounds=n_rounds,
                                     n_jobs=n_jobs, add_roc=False, reduced=True)
    bootstrapped_df['encoding'] = 'onehot'
    bootstrapped_df['weight'] = ic_name
    bootstrapped_df['input_type'] = args['input_type']
    bootstrapped_df['key'] = key
    bootstrapped_df['evalset'] = evalset.upper()
    fn = f'{args["outdir"]}bootstrapping/{evalset}_bootstrapped_df_{filename}.csv'
    print(f'Saved at {fn}')
    bootstrapped_df.to_csv(fn, index=False)

    return bootstrapped_df


def parallel_cdt(args, ic_name, ics_dict, mut_cols, encoding_kwargs, baseline, train_dataset, cedar_dataset, prime_dataset,
                 nepdb_dataset):
    AA_KEYS = [x for x in 'ARNDCQEGHILKMFPSTWYV']
    scol = 'Peptide' if args['input_type'] == 'Peptide' else 'icore_mut'
    prefix = 'fullpep_' if scol == 'Peptide' else 'icore_'
    rankcol = 'trueHLA_EL_rank' if args['input_type'] == 'Peptide' else 'EL_rank_mut'
    cols_ = [f'{prefix}aliphatic_index', f'{prefix}boman', f'{prefix}hydrophobicity', f'{prefix}isoelectric_point',
             'icore_dissimilarity_score', 'icore_blsm_mut_score', 'ratio_rank',
             'EL_rank_wt_aligned', 'foreignness_score', 'Total_Gene_TPM']
    encoding_kwargs['mut_col'] = mut_cols
    key = '-'.join(mut_cols).replace(' ', '-')
    if key == '':
        key = 'only_rank'
    # Hotfix for filename length...
    key = 'all_feats' if key == '-'.join(cols_) else key
    key = key.replace('icore_', '')
    filename = f'{args["trainset"]}_onehot_{ic_name}_{args["input_type"]}_{key}'.replace('Inverted', 'Inv')
    model = RandomForestClassifier(n_jobs=1, min_samples_leaf=7, n_estimators=300,
                                   max_depth=8, ccp_alpha=9.945e-6)
    # Training model and getting feature importances
    print('Training')
    trained_models, _, _ = nested_kcv_train_sklearn(train_dataset, model,
                                                    ics_dict=ics_dict,
                                                    encoding_kwargs=encoding_kwargs,
                                                    n_jobs=1)
    fi = get_nested_feature_importance(trained_models)
    fn = AA_KEYS + ['rank'] + mut_cols
    # Saving Feature importances as dataframe
    df_fi = pd.DataFrame(fi, index=fn).T
    df_fi.to_csv(
        f'{args["outdir"]}raw/featimps_{filename}.csv',
        index=False)

    pval_df = pd.DataFrame([[ic_name, args['input_type'], key]],
                           columns=['weight', 'input_type', 'key'])
    for evalset, evalname in zip([cedar_dataset, prime_dataset, nepdb_dataset],
                                 ['CEDAR', 'PRIME', 'NEPDB']):
        # FULLY FILTERED + Mean_pred
        if not evalset.equals(train_dataset):
            evalset = evalset.query('Peptide not in @train_dataset.Peptide.values').copy()

        # print(evalname, len(evalset), evalset.columns)
        _, preds = evaluate_trained_models_sklearn(evalset.drop_duplicates(subset=['Peptide', 'HLA', 'agg_label']),
                                                   trained_models, ics_dict,
                                                   train_dataset,
                                                   encoding_kwargs, concatenated=False,
                                                   only_concat=True, n_jobs=1)
        p_col = 'pred' if 'pred' in preds.columns else 'mean_pred'
        preds.to_csv(f'{args["outdir"]}raw/{evalname}_preds_{filename}.csv', index=False,
                     columns=['HLA', 'Peptide', 'agg_label', 'icore_mut', 'icore_wt_aligned'] + mut_cols + [p_col])
        bootstrapped_df = final_bootstrap_wrapper(preds, args, filename, ic_name,
                                                  key, evalname, n_rounds=10000, n_jobs=1)
        # print('#'*100,'\n\n\n', evalname, bootstrapped_df['auc'].mean(), '#'*100, '\n\n\n\n')

        if evalname == "NEPDB":
            continue
        else:
            for xx in baseline.keys():
                df_base = baseline[xx][evalname]
                pval, _ = get_pval_wrapper(bootstrapped_df[['id', 'auc']], df_base[['id', 'auc']], column='auc')
                pval_df[f'pval_{xx}_{evalname}'] = pval
    pval_fn=f'{args["outdir"]}raw/pvals_{filename}.csv'
    print(f'Pval df saved at {pval_fn}')
    pval_df.to_csv(pval_fn)



def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to crossvalidate and evaluate methods that use aa frequency as encoding')

    parser.add_argument('-datadir', type=str, default='../data/pepx/',
                        help='Path to directory containing the pre-partitioned data')
    parser.add_argument('-outdir', type=str,
                        default='../output/230427_MutExpr_Final_input_type/')
    parser.add_argument('-trainset', type=str, default='cedar')
    parser.add_argument('-icsdir', type=str, default='../data/ic_dicts/',
                        help='Path containing the pre-computed ICs dicts.')
    parser.add_argument('-ncores', type=int, default=36,
                        help='N cores to use in parallel, by default will be multiprocesing.cpu_count() * 3/4')
    parser.add_argument('-input_type', type=str, default='icore_mut', help='icore_mut, expanded_input, or Peptide')
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
    ics_shannon = pkl_load(f'{args["icsdir"]}ics_shannon.pkl')
    ics_kl = pkl_load(f'{args["icsdir"]}ics_kl_new.pkl')
    baseline = pkl_load(f'{args["outdir"]}baseline_bootstrapped.pkl')
    # print('fit check xd')
    # DEFINING COLS
    mcs = []
    cols_ = ['icore_aliphatic_index', 'icore_boman', 'icore_hydrophobicity', 'icore_isoelectric_point',
             'icore_dissimilarity_score',
             'icore_blsm_mut_score', 'ratio_rank', 'EL_rank_wt_aligned', 'foreignness_score', 'Total_Gene_TPM']

    for L in range(0, len(cols_) + 1):
        for mc in itertools.combinations(cols_, L):
            mcs.append(list(mc))

    # Define various stuff depending on the input columns
    scol = 'Peptide' if args['input_type'] == 'Peptide' else 'icore_mut'
    prefix = 'fullpep_' if scol == 'Peptide' else 'icore_'
    rankcol = 'trueHLA_EL_rank' if args['input_type'] == 'Peptide' else 'EL_rank_mut'

    cols_ = [f'{prefix}aliphatic_index', f'{prefix}boman', f'{prefix}hydrophobicity', f'{prefix}isoelectric_point',
             'icore_dissimilarity_score', 'icore_blsm_mut_score', 'ratio_rank',
             'EL_rank_wt_aligned', 'foreignness_score', 'Total_Gene_TPM']

    # Setting trainset
    trainmap = {'cedar': cedar_dataset,
                'prime': prime_dataset}
    assert (args['trainset'].lower() in trainmap.keys()), f'please input -trainset as either one of {trainmap.keys()}'

    train_dataset = trainmap[args['trainset']]

    # DEFINING KWARGS
    encoding_kwargs = {'max_len': 12,
                       'encoding': 'onehot',
                       'blosum_matrix': None,
                       'mask': False,  # Using Shannon ICs, true if both mask and name is "shannon"
                       'add_rank': True,
                       'seq_col': args['input_type'],
                       'rank_col': rankcol,
                       'hla_col': 'HLA',
                       'add_aaprop': False,
                       'remove_pep': False,
                       'standardize': True}

    # megaloops for encoding-weighting
    encoding_kwargs['threshold'] = 0.201
    encoding_kwargs['encoding'] = 'onehot'
    encoding_kwargs['blosum_matrix'] = None
    # Doing only Inverted Shannon, Mask, None
    cedar_dataset, _ = get_aa_properties(cedar_dataset, seq_col=scol, do_vhse=False, prefix=prefix)
    prime_dataset, _ = get_aa_properties(prime_dataset, seq_col=scol, do_vhse=False, prefix=prefix)
    nepdb_dataset, _ = get_aa_properties(nepdb_dataset, seq_col=scol, do_vhse=False, prefix=prefix)

    encoding_kwargs['mask'] = False

    conditions_list = [(False, 'KL', ics_kl, False), (True, 'Inverted-KL', ics_kl, False)]

    for invert, ic_name, ics_dict, mask in conditions_list:
        encoding_kwargs['invert'] = invert

        wrapper = partial(parallel_cdt, args=args, ic_name=ic_name, ics_dict= ics_dict,encoding_kwargs= encoding_kwargs, baseline=baseline, train_dataset=train_dataset,
            cedar_dataset=cedar_dataset,prime_dataset=prime_dataset,nepdb_dataset=nepdb_dataset)
        _ = Parallel(n_jobs=39)(delayed(wrapper)(mut_cols = mut_cols) for mut_cols in tqdm(mcs))

    end = dt.now()
    elapsed = divmod((end - start).seconds, 60)
    print(f'Elapsed: {elapsed[0]} minutes, {elapsed[1]} seconds. ; Memory used: {tracemalloc.get_traced_memory()}')
    tracemalloc.stop()


if __name__ == '__main__':
    main()