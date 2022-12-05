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
from src.utils import mkdirs, convert_path
from src.metrics import get_nested_feature_importance
from src.bootstrap import bootstrap_eval
from src.sklearn_train_eval import nested_kcv_train_sklearn, evaluate_trained_models_sklearn


N_CORES = 36


def final_bootstrap_wrapper(preds_df, args, blsm_name,
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
        f'{args["outdir"]}bootstrapping/{evalset}_bootstrapped_df_{blsm_name}_{"-".join(ic_name.split(" "))}_{pep_col}_{rank_col}_{key}.csv',
        index=False)
    pkl_dump(mean_rocs,
             f'{args["outdir"]}bootstrapping/{evalset}_mean_rocs_{blsm_name}_{"-".join(ic_name.split(" "))}_{pep_col}_{rank_col}_{key}.pkl')

    return bootstrapped_df


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
                            trained_models, train_metrics, _ = nested_kcv_train_sklearn(train_dataset, model,
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

                            # EVAL AND BOOTSTRAPPING ON CEDAR
                            _, cedar_preds_df = evaluate_trained_models_sklearn(cedar_dataset,
                                                                            trained_models,
                                                                            ics_dict, train_dataset,
                                                                            encoding_kwargs,
                                                                            concatenated=True,
                                                                            only_concat=True)
                            #
                            cedar_preds_df.drop(columns=aa_cols).to_csv(
                                f'{args["outdir"]}raw/cedar_preds_{blsm_name}_{"-".join(ic_name.split(" "))}_{pep_col}_{rank_col}_{key}.csv',
                                index=False)
                            # Bootstrapping (CEDAR)
                            cedar_bootstrapped_df = final_bootstrap_wrapper(cedar_preds_df, args, blsm_name, ic_name,
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
                                f'{args["outdir"]}raw/prime_preds_{blsm_name}_{"-".join(ic_name.split(" "))}_{pep_col}_{rank_col}_{key}.csv',
                                index=False)
                            # Bootstrapping (PRIME)
                            prime_bootstrapped_df = final_bootstrap_wrapper(prime_preds_df, args, blsm_name, ic_name,
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
                                f'{args["outdir"]}raw/prime_switch_preds_{blsm_name}_{"-".join(ic_name.split(" "))}_{pep_col}_{rank_col}_{key}.csv',
                                index=False)
                            # Bootstrapping (PRIME)
                            prime_switch_bootstrapped_df = final_bootstrap_wrapper(prime_switch_preds_df, args,
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
                                f'{args["outdir"]}raw/ibel_preds_{blsm_name}_{"-".join(ic_name.split(" "))}_{pep_col}_{rank_col}_{key}.csv',
                                index=False)
                            # Bootstrapping (PRIME)
                            ibel_bootstrapped_df = final_bootstrap_wrapper(ibel_preds_df, args, blsm_name, ic_name,
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
                                    f'{args["outdir"]}raw/merged_preds_{blsm_name}_{"-".join(ic_name.split(" "))}_{pep_col}_{rank_col}_{key}.csv',
                                    index=False)
                                # Bootstrapping (PRIME)
                                merged_bootstrapped_df = final_bootstrap_wrapper(merged_preds_df, args, blsm_name,
                                                                                 ic_name, pep_col, rank_col, key,
                                                                                 evalset='MERGED', n_rounds=10000,
                                                                                 n_jobs=N_CORES)
                                mega_df = mega_df.append(merged_bootstrapped_df)

    mega_df.to_csv(f'{args["outdir"]}/total_df.csv', index=False)


if __name__ == '__main__':
    main()
