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
from datetime import datetime as dt

today = dt.today().strftime("%Y%m%d")

N_CORES = 36


def final_bootstrap_wrapper(preds_df, args, filename, blsm_name,
                            ic_name, pep_col, rank_col, key, evalset,
                            n_rounds=10000, n_jobs=36):
    scores = preds_df.pred.values if 'pred' in preds_df.columns else preds_df['mean_pred'].values
    targets = preds_df.agg_label.values if 'agg_label' in preds_df.columns else preds_df['Immunogenicity'].values

    bootstrapped_df, mean_rocs = bootstrap_eval(y_score=scores,
                                                y_true=targets,
                                                n_rounds=n_rounds, n_jobs=n_jobs)
    bootstrapped_df['encoding'] = blsm_name
    bootstrapped_df['weight'] = ic_name
    bootstrapped_df['pep_col'] = pep_col
    bootstrapped_df['rank_col'] = rank_col
    bootstrapped_df['key'] = key
    bootstrapped_df['evalset'] = evalset.upper()

    bootstrapped_df.to_csv(
        f'{args["outdir"]}bootstrapping/{evalset}_bootstrapped_df_{filename}.csv',
        index=False)
    pkl_dump(mean_rocs,
             f'{args["outdir"]}bootstrapping/{evalset}_mean_rocs_{filename}.pkl')

    return bootstrapped_df


def parse_name(name):
    name = name.split('/')[-1].replace('Peptide', '').replace('EL_rank_mut', '').replace('trueHLA_EL_rank', '') \
        .replace('icore_mut', '').replace('None', '').replace('Mask', '').replace('Inverted-', '') \
        .replace('KL', '').replace('Shannon', '').replace('____', 'XX')
    return name


def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to crossvalidate and evaluate methods that use aa frequency as encoding')

    parser.add_argument('-datadir', type=str, default='../data/mutant/',
                        help='Path to directory containing the pre-partitioned data')
    parser.add_argument('-outdir', type=str, default=f'../output/{today}_new_core_mutscores_norank/')
    parser.add_argument('-icsdir', type=str, default='../data/ic_dicts/',
                        help='Path containing the pre-computed ICs dicts.')

    parser.add_argument('-cdtdir', type=str, default='../output/best_conditions/',
                        help='Path containing the "best_conditions"')
    parser.add_argument('-ncores', type=int, default=36,
                        help='N cores to use in parallel, by default will be multiprocesing.cpu_count() * 3/4')
    parser.add_argument('-mask_aa', type=str, default='false', help='Which AA to mask. has to be Capital letters and '
                                                                    'within the standard amino acid alphabet. To '
                                                                    'disable, input "false". "false" by default.')
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
    merged_dataset = pd.read_csv(f'{args["datadir"]}221112_cedar_prime_merged_fold.csv')
    ibel_dataset = pd.read_csv(f'{args["datadir"]}221117_ibel_merged_fold.csv')
    prime_switch_dataset = pd.read_csv(f'{args["datadir"]}221122_prime_AC_switch.csv')
    ics_shannon = pkl_load(f'{args["icsdir"]}ics_shannon.pkl')
    ics_kl = pkl_load(f'{args["icsdir"]}ics_kl.pkl')
    # Get best cdts for each trainset-evalset
    cdts = set(parse_name([x for x in os.listdir(f'{args["cdtdir"]}{trainfolder}/{evalfolder}') \
                           if 'csv' not in x and 'pkl' not in x][0]) \
               for trainfolder in ['cedar', 'prime', 'merged'] \
               for evalfolder in ['best_cedar', 'best_prime'])
    cdts = [x.split('XX') for x in cdts]
    # Set default arguments : Do not add rank, use onehot, set default columns
    encoding_kwargs = {'max_len': 12,
                       'encoding': 'onehot',
                       'blosum_matrix': None,
                       'mask': False,  # Using Shannon ICs, true if both mask and name is "shannon"
                       'add_rank': False,
                       'add_aaprop': False,
                       'remove_pep': False,
                       'standardize': True,
                       'seq_col': 'Peptide',
                       'hla_col': 'HLA',
                       'target_col': 'agg_label',
                       'rank_col': 'EL_rank_mut'}

    results = {}
    aa_cols = ['aliphatic_index', 'boman', 'hydrophobicity', 'isoelectric_point', 'VHSE1', 'VHSE3', 'VHSE7', 'VHSE8']
    print('Starting loops')
    mega_df = pd.DataFrame()
    # Set some params hardcoded to not rewrite code
    pep_col = 'Peptide'
    blsm_name = 'onehot'
    rank_col = 'None'
    # For each training set
    for train_dataset, train_nm in zip([cedar_dataset, prime_dataset, merged_dataset], ['cedar', 'prime', 'merged']):
        results[train_nm] = {}
        # For each
        for condition in cdts:
            key = condition[-1]
            mut_cols = key if key != 'aa_props' else aa_cols
            encoding_kwargs['mut_col'] = mut_cols
            results[train_nm][key] = {}
            for invert in [True, False]:
                for ic_name, ics_dict in zip(['Mask', 'KL', 'None', 'Shannon'],
                                             [ics_shannon, ics_kl, None, ics_shannon]):
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
                    results[train_nm][key][ic_name] = {}
                    # Using the same model and hyperparameters
                    model = RandomForestClassifier(n_jobs=1, min_samples_leaf=7, n_estimators=300,
                                                   max_depth=8, ccp_alpha=9.945e-6)
                    filename = f'train_{train_nm}XXenc_{blsm_name}XXweight_{"-".join(ic_name.split(" "))}XX{key}'
                    # Training model and getting feature importances
                    print('Training')
                    trained_models, train_metrics, _ = nested_kcv_train_sklearn(train_dataset, model,
                                                                                ics_dict=ics_dict,
                                                                                encoding_kwargs=encoding_kwargs,
                                                                                n_jobs=15)
                    fi = get_nested_feature_importance(trained_models)
                    print(AA_KEYS)
                    print(mut_cols)
                    fn = AA_KEYS + mut_cols
                    # Saving Feature importances as dataframe
                    df_fi = pd.DataFrame(fi, index=fn).T
                    df_fi.to_csv(
                        f'{args["outdir"]}raw/featimps_{filename}.csv',
                        index=False)

                    # EVAL AND BOOTSTRAPPING ON CEDAR
                    _, cedar_preds_df = evaluate_trained_models_sklearn(cedar_dataset,
                                                                        trained_models,
                                                                        ics_dict, train_dataset,
                                                                        encoding_kwargs,
                                                                        concatenated=True,
                                                                        only_concat=True)
                    #
                    cedar_preds_df.drop(columns=aa_cols).to_csv(
                        f'{args["outdir"]}raw/cedar_preds_{filename}.csv',
                        index=False)
                    # Bootstrapping (CEDAR)
                    cedar_bootstrapped_df = final_bootstrap_wrapper(cedar_preds_df, args, filename, blsm_name, ic_name,
                                                                    pep_col, rank_col, key,
                                                                    evalset='CEDAR', n_rounds=10000,
                                                                    n_jobs=N_CORES)
                    mega_df = mega_df.append(cedar_bootstrapped_df)

                    # EVAL AND BOOTSTRAPPING ON PRIME
                    _, prime_preds_df = evaluate_trained_models_sklearn(prime_dataset,
                                                                        trained_models,
                                                                        ics_dict, train_dataset,
                                                                        encoding_kwargs,
                                                                        concatenated=False,
                                                                        only_concat=False)

                    # Pre-saving results before bootstrapping
                    prime_preds_df.drop(columns=aa_cols).to_csv(
                        f'{args["outdir"]}raw/prime_preds_{filename}.csv',
                        index=False)
                    # Bootstrapping (PRIME)
                    prime_bootstrapped_df = final_bootstrap_wrapper(prime_preds_df, args, filename, blsm_name, ic_name,
                                                                    pep_col, rank_col, key,
                                                                    evalset='PRIME', n_rounds=10000,
                                                                    n_jobs=N_CORES)
                    mega_df = mega_df.append(prime_bootstrapped_df)

                    # EVAL AND BOOTSTRAPPING ON PRIME SWITCH
                    _, prime_switch_preds_df = evaluate_trained_models_sklearn(prime_switch_dataset,
                                                                               trained_models,
                                                                               ics_dict, train_dataset,
                                                                               encoding_kwargs,
                                                                               concatenated=False,
                                                                               only_concat=False)

                    # Pre-saving results before bootstrapping
                    prime_switch_preds_df.drop(columns=aa_cols).to_csv(
                        f'{args["outdir"]}raw/prime_switch_preds_{filename}.csv',
                        index=False)
                    # Bootstrapping (PRIME)
                    prime_switch_bootstrapped_df = final_bootstrap_wrapper(prime_switch_preds_df, args, filename,
                                                                           blsm_name, ic_name,
                                                                           pep_col, rank_col, key,
                                                                           evalset='PRIME_AC', n_rounds=10000,
                                                                           n_jobs=N_CORES)
                    mega_df = mega_df.append(prime_switch_bootstrapped_df)

                    # ///////////////////////////
                    # EVAL AND BOOTSTRAPPING ON IBEL
                    _, ibel_preds_df = evaluate_trained_models_sklearn(ibel_dataset,
                                                                       trained_models,
                                                                       ics_dict, train_dataset,
                                                                       encoding_kwargs,
                                                                       concatenated=False,
                                                                       only_concat=False)

                    # Pre-saving results before bootstrapping
                    ibel_preds_df.drop(columns=aa_cols).to_csv(
                        f'{args["outdir"]}raw/ibel_preds_{filename}.csv',
                        index=False)
                    # Bootstrapping (PRIME)
                    ibel_bootstrapped_df = final_bootstrap_wrapper(ibel_preds_df, args, filename, blsm_name, ic_name,
                                                                   pep_col, rank_col, key,
                                                                   evalset='IBEL', n_rounds=10000,
                                                                   n_jobs=N_CORES)
                    mega_df = mega_df.append(ibel_bootstrapped_df)

                    # /////////////////////////// only if trainset == MERGED should I evaluate on it
                    if args['trainset'].lower() == 'merged':
                        # EVAL AND BOOTSTRAPPING ON IBEL
                        _, merged_preds_df = evaluate_trained_models_sklearn(merged_dataset,
                                                                             trained_models,
                                                                             ics_dict, train_dataset,
                                                                             encoding_kwargs,
                                                                             concatenated=False,
                                                                             only_concat=False)

                        # Pre-saving results before bootstrapping
                        merged_preds_df.drop(columns=aa_cols).to_csv(
                            f'{args["outdir"]}raw/merged_preds_{filename}.csv',
                            index=False)
                        # Bootstrapping (PRIME)
                        merged_bootstrapped_df = final_bootstrap_wrapper(merged_preds_df, args, filename, blsm_name,
                                                                         ic_name, pep_col, rank_col, key,
                                                                         evalset='MERGED', n_rounds=10000,
                                                                         n_jobs=N_CORES)
                        mega_df = mega_df.append(merged_bootstrapped_df)

    mega_df.to_csv(f'{args["outdir"]}/total_df.csv', index=False)


if __name__ == '__main__':
    main()
