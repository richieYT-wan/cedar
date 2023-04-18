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
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Custom fct imports
from src.utils import pkl_load, pkl_dump
import argparse
from src.data_processing import BL62_VALUES, BL62FREQ_VALUES, AA_KEYS, get_aa_properties
from src.utils import mkdirs, convert_path, str2bool
from src.metrics import get_nested_feature_importance
from src.bootstrap import bootstrap_eval
from src.sklearn_train_eval import nested_kcv_train_sklearn, evaluate_trained_models_sklearn
from copy import deepcopy

N_CORES = 39

def final_bootstrap_wrapper(preds_df, args, filename,
                            ic_name, key, evalset,
                            n_rounds=10000, n_jobs=36):
    scores = preds_df.pred.values if 'pred' in preds_df.columns else preds_df['mean_pred'].values
    targets = preds_df.agg_label.values if 'agg_label' in preds_df.columns else preds_df['Immunogenicity'].values

    bootstrapped_df = bootstrap_eval(y_score=scores, y_true=targets, n_rounds=n_rounds, n_jobs=n_jobs, add_roc=False)
    bootstrapped_df['encoding'] = 'onehot'
    bootstrapped_df['weight'] = ic_name
    bootstrapped_df['pep_col'] = 'icore_mut'
    bootstrapped_df['rank_col'] = 'EL_rank_mut'
    bootstrapped_df['key'] = key
    bootstrapped_df['evalset'] = evalset.upper()

    bootstrapped_df.to_csv(
        f'{args["outdir"]}bootstrapping/{evalset}_bootstrapped_df_{filename}.csv',
        index=False)

    return bootstrapped_df.drop(columns=['AP', 'f1'])

def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to crossvalidate and evaluate methods that use aa frequency as encoding')

    parser.add_argument('-datadir', type=str, default='../data/pepx/',
                        help='Path to directory containing the pre-partitioned data')
    parser.add_argument('-outdir', type=str, default='../output/230414_aligned_icore_exprscore_mutscore_filtered_aaprops_exp8mers/')
    parser.add_argument('-trainset', type=str, default='cedar')
    parser.add_argument('-icsdir', type=str, default='../data/ic_dicts/',
                        help='Path containing the pre-computed ICs dicts.')
    parser.add_argument('-ncores', type=int, default=36,
                        help='N cores to use in parallel, by default will be multiprocesing.cpu_count() * 3/4')
    parser.add_argument('-condition', type=str, default='None', help = 'Inverted-Shannon, Mask or None. Must be string, so "None"')

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

    ics_shannon = pkl_load(f'{args["icsdir"]}ics_shannon.pkl')

    # DEFINING COLS
    mcs = []
    cols_ = ['icore_aliphatic_index', 'icore_boman', 'icore_hydrophobicity', 'icore_isoelectric_point', 'icore_dissimilarity_score', 'icore_blsm_mut_score', 'ratio_rank', 'EL_rank_wt_aligned', 'foreignness_score', 'Total_Gene_TPM'] 
    
    
    for L in range(0, len(cols_) + 1):
        for mc in itertools.combinations(cols_, L):
           mcs.append(list(mc))
    cedar_dataset, _ = get_aa_properties(cedar_dataset, seq_col='icore_mut', do_vhse=False, prefix='icore_')
    prime_dataset, _ = get_aa_properties(prime_dataset, seq_col='icore_mut', do_vhse=False, prefix='icore_')
    nepdb_dataset, _ = get_aa_properties(nepdb_dataset, seq_col='icore_mut', do_vhse=False, prefix='icore_')

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
                       'seq_col':'expanded_input',
                       'rank_col':'EL_rank_mut',
                       'hla_col':'HLA',
                       'add_aaprop': False,
                       'remove_pep': False,
                       'standardize': True}

    conditions_list = {'Inverted-Shannon':(True, 'Shannon', ics_shannon, False),
                       'None':(False, 'None', None, False),
                       'Mask':(False, 'Mask', ics_shannon, True)}


        # for invert, ic_name, ics_dict, mask in [(True, 'Shannon', ics_shannon, False),
        #                                   (False, 'None', None, False),
        #                                   (False, 'Mask', ics_shannon, True)]:
    invert, ic_name, ics_dict, mask = conditions_list[args["condition"]]
    # megaloops for encoding-weighting

    encoding_kwargs['encoding'] = 'onehot'
    encoding_kwargs['blosum_matrix'] = None
    # Doing only Inverted Shannon, Mask, None

    encoding_kwargs['invert'] = invert
    encoding_kwargs['mask'] = mask
    if invert:
        if ic_name == 'None':
            pass
        else:
            ic_name = 'Inverted ' + ic_name

    mega_df = pd.DataFrame()
    print('Starting loops')
    for mut_cols in tqdm(reversed(mcs), position=0, leave=True, desc='cols'):
        encoding_kwargs['mut_col'] = mut_cols
        key = '-'.join(mut_cols).replace(' ', '-')
        if key == '':
            key = 'only_rank'
        # Hotfix for filename length...
        key = 'all_feats' if key == '-'.join(cols_) else key
        key = key.replace('icore_','')
        filename = f'{args["trainset"]}_onehot_{"-".join(ic_name.split(" "))}_icore_mut_EL_rank_mut_{key}'.replace('Inverted','Inv')
        
        # Using the same model and hyperparameters
        model = RandomForestClassifier(n_jobs=1, min_samples_leaf=7, n_estimators=300,
                                       max_depth=8, ccp_alpha=9.945e-6)
        # Training model and getting feature importances
        print('Training')
        trained_models, train_metrics, _ = nested_kcv_train_sklearn(train_dataset, model,
                                                                    ics_dict=ics_dict,
                                                                    encoding_kwargs=encoding_kwargs,
                                                                    n_jobs=10)
        fi = get_nested_feature_importance(trained_models)
        fn = AA_KEYS + ['rank'] + mut_cols
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
            _, preds = evaluate_trained_models_sklearn(evalset.drop_duplicates(subset=['Peptide','HLA','agg_label']),
                                                       trained_models, ics_dict,
                                                       train_dataset,
                                                       encoding_kwargs, concatenated=False,
                                                       only_concat=True, n_jobs=10)
            p_col = 'pred' if 'pred' in preds.columns else 'mean_pred'
            preds.to_csv(f'{args["outdir"]}raw/{evalname}_preds_{filename}.csv', index=False,
                         columns = ['HLA','Peptide','agg_label', 'icore_mut', 'icore_wt_aligned']+mut_cols+[p_col])

            bootstrapped_df = final_bootstrap_wrapper(preds, args, filename, ic_name,
                                                      key, evalname, n_rounds=10000, n_jobs = 38)
            mega_df = mega_df.append(bootstrapped_df)

    mega_df.to_csv(f'{args["outdir"]}/total_df_{args["condition"]}.csv', index=False)


if __name__ == '__main__':
    main()
