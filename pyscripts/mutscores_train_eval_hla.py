import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
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

N_CORES = 36


def final_bootstrap_wrapper(preds_df, args, filename, hla,
                            ic_name, evalset,
                            n_rounds=10000, n_jobs=36):
    scores = preds_df.pred.values if 'pred' in preds_df.columns else preds_df['mean_pred'].values
    targets = preds_df.agg_label.values if 'agg_label' in preds_df.columns else preds_df['Immunogenicity'].values

    bootstrapped_df, mean_rocs = bootstrap_eval(y_score=scores,
                                                y_true=targets,
                                                n_rounds=n_rounds, n_jobs=n_jobs)
    bootstrapped_df['encoding'] = 'onehot'
    bootstrapped_df['weight'] = ic_name
    bootstrapped_df['hla'] = hla
    bootstrapped_df['evalset'] = evalset.upper()

    bootstrapped_df.to_csv(
        f'{args["outdir"]}bootstrapping/{evalset}_bootstrapped_df_{filename}.csv',
        index=False)
    pkl_dump(mean_rocs,
             f'{args["outdir"]}bootstrapping/{evalset}_mean_rocs_{filename}.pkl')

    return bootstrapped_df


def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to crossvalidate and evaluate methods that use aa frequency as encoding')

    parser.add_argument('-datadir', type=str, default='../data/mutant/',
                        help='Path to directory containing the pre-partitioned data')
    parser.add_argument('-outdir', type=str, default='../output/221212_HLA_specific_model/')
    parser.add_argument('-icsdir', type=str, default='../data/ic_dicts/',
                        help='Path containing the pre-computed ICs dicts.')
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

    ics_shannon = pkl_load(f'{args["icsdir"]}ics_shannon.pkl')
    ics_kl = pkl_load(f'{args["icsdir"]}ics_kl.pkl')

    # Setting trainset
    mega_df = pd.DataFrame()
    print('Starting loops')
    encoding_kwargs = dict(max_len=12, encoding='onehot', blosum_matrix=None, mask=False, add_rank=True,
                           add_aaprop=False, remove_pep=False, standardize=True,
                           target_col='agg_label', seq_col='Peptide', rank_col='EL_rank_mut', hla_col='HLA',
                           mask_aa=False)
    results = {}
    for dataset, trainname in zip([cedar_dataset, prime_dataset], ['cedar', 'prime']):
        results[trainname] = {}
        for hla in ['HLA-A0201', 'HLA-A1101', 'HLA-A2402', 'HLA-B0702', 'HLA-B1501', 'HLA-B3501']:
            # Manually gets split everytime
            kf = KFold(5, shuffle=True, random_state=13)
            train_dataset = dataset.query('HLA==@hla').copy()
            for i, (train_idx, test_idx) in enumerate(kf.split(train_dataset['Peptide'], train_dataset['agg_label'])):
                train_dataset.iloc[test_idx, list(train_dataset.columns).index('fold')] = i
            # Do weights loops
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
                    # creating filename
                    filename = f'{trainname}_onehot_{"-".join(ic_name.split(" "))}_{hla}'
                    # Using the same model and hyperparameters
                    model = RandomForestClassifier(n_estimators=100, max_depth=5, ccp_alpha=5e-7)
                    # Training model and getting feature importances
                    print('Training')
                    trained_models, train_metrics, _ = nested_kcv_train_sklearn(train_dataset, model,
                                                                                ics_dict=ics_dict,
                                                                                encoding_kwargs=encoding_kwargs,
                                                                                n_jobs=10)
                    fi = get_nested_feature_importance(trained_models)
                    fn = AA_KEYS + ['rank']
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
                    cedar_preds_df.to_csv(
                        f'{args["outdir"]}raw/cedar_preds_{filename}.csv',
                        index=False)
                    # Bootstrapping (CEDAR)
                    cedar_bootstrapped_df = final_bootstrap_wrapper(cedar_preds_df, args, filename, hla, ic_name,
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
                    prime_preds_df.to_csv(
                        f'{args["outdir"]}raw/prime_preds_{filename}.csv',
                        index=False)
                    # Bootstrapping (PRIME)
                    prime_bootstrapped_df = final_bootstrap_wrapper(prime_preds_df, args, filename, hla, ic_name,
                                                                    evalset='PRIME', n_rounds=10000,
                                                                    n_jobs=N_CORES)
                    mega_df = mega_df.append(prime_bootstrapped_df)

    mega_df.to_csv(f'{args["outdir"]}/total_df.csv', index=False)


if __name__ == '__main__':
    main()
