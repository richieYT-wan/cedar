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
from src.metrics import get_nested_feature_importance
from src.bootstrap import bootstrap_eval
from src.sklearn_train_eval import nested_kcv_train_sklearn, evaluate_trained_models_sklearn
import copy
from copy import deepcopy

N_CORES = 39

def final_bootstrap_wrapper(preds_df, args, filename, add_expression, condition, key, evalset,
                            n_rounds=10000, n_jobs=36):
    scores = preds_df.pred.values if 'pred' in preds_df.columns else preds_df['mean_pred'].values
    targets = preds_df.agg_label.values if 'agg_label' in preds_df.columns else preds_df['Immunogenicity'].values

    bootstrapped_df, mean_rocs = bootstrap_eval(y_score=scores,
                                                y_true=targets,
                                                n_rounds=n_rounds, n_jobs=n_jobs)
    bootstrapped_df['condition'] = condition
    bootstrapped_df['add_expression'] = expand
    bootstrapped_df['key'] = key
    bootstrapped_df['evalset'] = evalset.upper()

    bootstrapped_df.to_csv(
        f'{args["outdir"]}bootstrapping/{evalset}_bootstrapped_df_{filename}.csv',
        index=False)
    pkl_dump(mean_rocs,
             f'{args["outdir"]}bootstrapping/{evalset}_mean_rocs_{filename}.pkl')

    return bootstrapped_df

######## ARGS PARSE & MAIN
def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to crossvalidate and evaluate methods that use aa frequency as encoding')

    parser.add_argument('-datadir', type=str, default='../data/pepx/',
                        help='Path to directory containing the pre-partitioned data')
    parser.add_argument('-outdir', type=str, default='../output/230410_redo_AddExpression_BestModels/')
    parser.add_argument('-trainset', type=str, default='cedar')
    parser.add_argument('-icsdir', type=str, default='../data/ic_dicts/',
                        help='Path containing the pre-computed ICs dicts.')
    parser.add_argument('-ncores', type=int, default=40,
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
                        mut_col = aa_cols+['EL_rank_wt_aligned','icore_dissimilarity_score','ratio_rank']
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
                          mut_col = ['icore_dissimilarity_score', 'icore_blsm_mut_score', 'ratio_rank']),
                     None, 'General')

    mega_df=pd.DataFrame()
    print('Starting loops')
    for encoding_kwargs, ics_dict, condition in [cdt_general,cdt_base, cdt_cedar, cdt_prime]:
        for add_expression in [False, True]:
            if add_expression:
                encoding_kwargs['mut_col'] = encoding_kwargs['mut_col'] + ['Total_Gene_TPM']
            filename = f'addExpression{add_expression}_Condition{condition}'
            mut_cols = encoding_kwargs['mut_col'] if (condition!='Base' or add_expression) else None
            # Using the same model and hyperparameters
            model = RandomForestClassifier(n_jobs=1, min_samples_leaf=7, n_estimators=300,
                                               max_depth=8, ccp_alpha=9.945e-6)
            # Training model and getting feature importances
            print('Training')
            trained_models, train_metrics, _ = nested_kcv_train_sklearn(train_dataset, model,
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
            encoding_kwargs['standardize']=True
            key = '-'.join(encoding_kwargs['mut_col']).replace('-'.join(aa_cols), 'aa_props')
            for evalset, evalname in zip([cedar_dataset, prime_dataset, nepdb_dataset],
                                         ['CEDAR', 'PRIME', 'NEPDB']):
                # FULLY FILTERED + Mean_pred
                if not evalset.equals(train_dataset):
                    evalset = evalset.query('Peptide not in @train_dataset.Peptide.values')
                _, preds = evaluate_trained_models_sklearn(evalset.drop_duplicates(subset=['Peptide','HLA','agg_label']),
                                                           trained_models, ics_dict,
                                                           train_dataset,
                                                           encoding_kwargs, concatenated=False,
                                                           only_concat=True, n_jobs=args["ncores"])

                # p_col = 'pred' if 'pred' in preds.columns else 'mean_pred'
                preds.drop(columns=aa_cols).to_csv(f'{args["outdir"]}raw/{evalname}_preds_{filename}.csv', index=False)

                bootstrapped_df = final_bootstrap_wrapper(preds, args, filename, add_expression, condition, key,
                                                          evalname, n_rounds=10000, n_jobs = args["ncores"])
                mega_df = mega_df.append(bootstrapped_df)

    mega_df.to_csv(f'{args["outdir"]}/total_df.csv', index=False)


if __name__ == '__main__':
    main()
