import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import argparse
from joblib import Parallel, delayed
from functools import partial
import multiprocessing
from itertools import product
from torch import nn
from tqdm.auto import tqdm
import warnings
from datetime import datetime as dt
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Custom fct imports
from src.utils import pkl_load, pkl_dump, flatten_product
from src.data_processing import BL62_VALUES, BL62FREQ_VALUES
from src.utils import str2bool, mkdirs, convert_path
from src.sklearn_train_eval import kcv_tune_sklearn

warnings.filterwarnings('ignore')
N_CORES = 1 + (multiprocessing.cpu_count() // 3)


def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to crossvalidate and evaluate methods that use aa frequency as encoding')

    parser.add_argument('-datadir', type=str, default='../data/partitioned_traindata_redo/',
                        help='Path to directory containing the pre-partitioned data')
    parser.add_argument('-outdir', type=str, default='../output/tuning/')
    parser.add_argument('-icsdir', type=str, default='../data/ic_dicts/', help='Path containing the pre-computed ICs dicts.')
    parser.add_argument('-debug', type=str2bool, default=False)

    return parser.parse_args()


def tune_wrapper(args, model, hyperparams, train_dataset, ics_dict, ics_name,
                 encoding, blosum_matrix, mask, add_rank, add_aaprop,
                 remove_pep, standardize):
    """
    ARGUMENT LIST SHOULD REFLECT THE PRODUCT ORDER
    Args:
        train_dataset:
        hyperparams: the hyperparameter grid (i.e. dict with multiple HP and a list of values for each HP)
        model:
        ics_dict:
        ics_name:
        encoding:
        blosum_matrix:
        mask:
        add_rank:
        add_aaprop:
        remove_pep:
        standardize:

    Returns:

    """
    encoding_kwargs = {'max_len': 12,
                       'encoding': encoding,
                       'blosum_matrix': blosum_matrix,
                       'mask': mask,  # Using Shannon ICs, true if both mask and name is "shannon"
                       'add_rank': add_rank,
                       'add_aaprop': add_aaprop,
                       'remove_pep': remove_pep,
                       'standardize': standardize}
    # HERE CONDITION TO EARLY EXIT IF REMOVE_PEP IS TRUE, AS WE DON'T NEED 100 CONDITIONS with
    # IC dicts, blosum, etc.
    # If remove_pep is true, we should just keep a single condition with Onehot and no weight and no blosum
    # This drops 176 conditions for each of the 576 loops per model
    # --> Leaves 400 runs per model
    if remove_pep:
        if encoding != 'onehot' or ics_name != 'None':
            return None

    # Here sets the output tag (names, variables and their values)
    # quickly rename the model name so it's shorter
    mapping = {'RandomForestClassifier': 'RF',
               'LogisticRegression': 'LogReg',
               'XGBClassifier': 'XGB',
               'LinearBlock': 'LinearBlock'}
    # Save the parameter name as a massive string to tag the output.
    # Set the "trainset" name in the beginning of the for loop by assigning name to a train_dataset column
    # i.e. train_dataset['trainset'] = 'name'
    outdict = {'trainset': train_dataset['trainset'].unique()[0],
               'model': mapping[model.__class__.__name__],
               'weight': ics_name}
    outdict.update({k: v for k, v in encoding_kwargs.items() if k != 'blosum_matrix'})
    if blosum_matrix is not None:
        bl_name = 'BL62LO' if blosum_matrix['A'].dtype == np.int64 else 'BL62FREQ'
        outdict['blsm'] = bl_name
    outdict.update({k: v for k, v in encoding_kwargs.items() if k != 'blosum_matrix'})
    # Merging each tag into one single string
    outname = '_'.join([f'{k}{v}' for k, v in outdict.items()])

    # HERE THE ACTUAL FUNCTION CALL TO GET THE TUNING OUTPUT
    # TODO : redo the fct for KCV tune nn
    if issubclass(model.__class__, nn.Module):
        # nested_kcv_train_nn(kwargs)
        results_df = pd.DataFrame()
        pass
    else:
        results_df, list_roc_curves = kcv_tune_sklearn(train_dataset, model, ics_dict, encoding_kwargs, hyperparams)
    results_df.reset_index(drop=True, inplace=True)
    best_idx = results_df['score_avg_valid_auc'].argmax()
    best_model = results_df.iloc[best_idx]
    params_cols = [x for x in results_df.columns if 'score' not in x]
    best_params = results_df.iloc[best_idx][params_cols].to_dict()
    # Saving the best params
    pkl_dump(best_params, os.path.join(args['outdir'], outname + '.pkl'))
    outdict['best_params'] = best_params
    results_df.to_csv(os.path.join(args['outdir'], outname + '.csv'), index=False)

    return results_df, outname, outdict


def main():
    start = dt.now()
    run_id = f'run_{start.strftime(f"%y%m%d_%Hh%Mm%Ss")}/'
    args = vars(args_parser())
    args['outdir'], args['datadir'], args['icsdir'] = convert_path(args['outdir']), convert_path(
        args['datadir']), convert_path(args['icsdir'])
    mkdirs(args['outdir'])
    args['outdir'] = convert_path(os.path.join(args['outdir'], run_id))
    mkdirs(args['outdir'])
    # Read data and assign train set name
    dataset_prime = pd.read_csv(f'{args["datadir"]}prime_5fold.csv')
    dataset_prime['trainset'] = 'prime'
    # Read data and assign train set name
    dataset_cedar = pd.read_csv(f'{args["datadir"]}cedar_10fold.csv')
    dataset_cedar['trainset'] = 'cedar'
    # Read data and assign train set name
    dataset_viral = pd.read_csv(f'{args["datadir"]}viral_only_5fold.csv')
    dataset_viral['trainset'] = 'virus'

    dataset_cedar_viral = pd.read_csv(f'{args["datadir"]}new_cedar_viral_5fold.csv')
    dataset_cedar_viral['trainset'] = 'cedar_virus'

    ics_shannon = pkl_load(f'{args["icsdir"]}ics_shannon.pkl')
    ics_kl = pkl_load(f'{args["icsdir"]}ics_kl.pkl')
    ics_none = None  # bit stupid but need to have the variable name somewhere :-)
    if args["debug"]:
        encode_blosum_zip = zip(['onehot', 'blosum'], [None, BL62FREQ_VALUES])
        ics_mask_zip = zip([ics_shannon, ics_none, ics_shannon],
                           ['Shannon', 'None', 'Mask'],
                           [False, False, True])
        features_zip = zip([True], [True], [False])
        train_datasets = [dataset_prime]
        standardize_ = [True]
    else:
        # Creating the condition/products/zips

        # encoding, blosum matrix
        encode_blosum_zip = zip(['onehot', 'blosum', 'blosum'], [None, BL62FREQ_VALUES, BL62_VALUES])
        # Weighting zip (ic dict, ic name, mask bool)
        ics_mask_zip = zip([ics_shannon, ics_kl, ics_none, ics_shannon], ['Shannon', 'KL', 'None', 'Mask'],
                           [False, False, False, True])
        # True/False zips for add_rank, add_aaprop, remove_pep
        features_zip = zip([True, True, False, False],
                           [True, False, True, False],
                           [False, False, False, False])

        # Lone conditions (to be producted so not zipped)
        train_datasets = [dataset_cedar, dataset_prime, dataset_viral, dataset_cedar_viral]
        standardize_ = [True]

    # Here make the conditions that don't rely on hyperparams, i.e. all dataset and dataprocessing conditions
    conditions = product(train_datasets,
                         product(encode_blosum_zip,
                                 product(ics_mask_zip,
                                         product(features_zip,
                                                 product(standardize_)))))
    # Flatten the list of products
    # Gets the argument in the order :
    # train_dataset, encoding, blosum matrix, ics dict, ics name, mask, add rank, add prop, remove pep, standardize
    conditions = list(list(flatten_product(x)) for x in conditions)
    tune_results_model = {}

    # Manual tuning and their hyperparameters to tune
    if args['debug']:
        models_params_grid = zip([RandomForestClassifier(), XGBClassifier()],
                                 [{'n_estimators': [20], 'max_depth': [3, None],
                                   'ccp_alpha': [1e-10]},
                                  {'n_estimators':[20], 'max_depth':[3,None]}])

    else:
        models_params_grid = zip([RandomForestClassifier(), XGBClassifier()],
                                 [{'n_estimators': [200, 250, 300], 'max_depth': [3, 7, None],
                                   'ccp_alpha': np.logspace(-9, -1, 4)},
                                  {'n_estimators': [200, 250], 'max_depth': [3, 7, None],
                                   'learning_rate': [0.12],
                                   'reg_alpha': np.logspace(-9, -1, 4), 'reg_lambda': np.logspace(-9, -1, 4)}])
    print(f'\n\n\n\tCPU COUNT: {multiprocessing.cpu_count()}, using {multiprocessing.cpu_count() - 1}\n\n\n')
    # Loop with parallelized jobs
    for model, hyperparameters in tqdm(models_params_grid, desc='models', leave=False):
        tune_results = []
        wrapper = partial(tune_wrapper, args=args, model=model, hyperparams=hyperparameters)

        output = Parallel(n_jobs=multiprocessing.cpu_count()-6)(delayed(wrapper)(train_dataset=train_dataset, encoding=encoding,

                                                     blosum_matrix=blosum_matrix, ics_dict=ics_dict, ics_name=ics_name,
                                                     mask=mask, add_rank=add_rank, add_aaprop=add_aaprop,
                                                     remove_pep=remove_pep, standardize=standardize) for \
                                    (train_dataset, encoding, blosum_matrix, ics_dict, ics_name, mask, add_rank,
                                     add_aaprop, remove_pep, standardize) in conditions)
        tune_results.extend(output)
        tune_results_model[model.__class__.__name__] = tune_results
    pkl_dump(tune_results_model, os.path.join(args['outdir'], 'tune_results_models.pkl'))
    end = dt.now()
    elapsed = (end - start).seconds
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    print(f'Time elapsed: {hours} hours, {minutes} minutes, {seconds} seconds.')


if __name__ == '__main__':
    main()
