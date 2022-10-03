import pandas as pd
import numpy as np
import pathlib
import argparse
from joblib import Parallel, delayed
from functools import partial
import multiprocessing
from itertools import product
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

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
from src.models import FFN
from src.train_eval import kcv_tune_nn_freq

warnings.filterwarnings('ignore')
N_CORES = 1 + (multiprocessing.cpu_count() // 3)


def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to crossvalidate and evaluate methods that use aa frequency as encoding')

    parser.add_argument('-datadir', type=str, default='../data/partitioned_traindata/',
                        help='Path to directory containing the pre-partitioned data')
    parser.add_argument('-outdir', type=str, default='../output/tuning/')
    parser.add_argument('-icsdir', type=str, default='../data/ic_dicts/',
                        help='Path containing the pre-computed ICs dicts.')
    parser.add_argument('-outname', type=str, default='_NN', help='additional name (appended to the end)')
    parser.add_argument('-ncores', type=int, default=2, help='N cores to use, default 2')
    parser.add_argument('-gpu', type=str2bool, default=False, help='Enabling gpu or not')
    parser.add_argument('-debug', type=str2bool, default=False)

    return parser.parse_args()


def tune_wrapper(args, hyperparams, device, train_dataset, ics_dict, ics_name,
                 encoding, blosum_matrix, mask, add_rank, add_aaprop,
                 remove_pep, standardize,
                 learning_rate, weight_decay):
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
    # Save the parameter name as a massive string to tag the output.
    # Set the "trainset" name in the beginning of the for loop by assigning name to a train_dataset column
    # i.e. train_dataset['trainset'] = 'name'
    outdict = {'trainset': train_dataset['trainset'].unique()[0],
               'model': 'FFN',
               'weight': ics_name}
    outdict.update({k: v for k, v in encoding_kwargs.items() if k != 'blosum_matrix'})
    if blosum_matrix is not None:
        bl_name = 'BL62LO' if blosum_matrix['A'].dtype == np.int64 else 'BL62FREQ'
        outdict['blsm'] = bl_name
    outdict.update({k: v for k, v in encoding_kwargs.items() if k != 'blosum_matrix'})
    outdict['lr'] = learning_rate
    outdict['wd'] = weight_decay
    # Merging each tag into one single string
    outname = '_'.join([f'{k}{v}' for k, v in outdict.items()])

    # HERE THE ACTUAL FUNCTION CALL TO GET THE TUNING OUTPUT
    results_df, list_roc_curves = kcv_tune_nn_freq(train_dataset, FFN, ics_dict, encoding_kwargs, hyperparams,
                                              criterion=nn.BCELoss(), optimizer=optim.Adam(FFN(n_in=1).parameters(),
                                                                                           lr=learning_rate,
                                                                                           weight_decay=weight_decay),
                                              device=device, outdir=args['outdir'])

    results_df.reset_index(drop=True, inplace=True)
    best_idx = results_df['score_avg_valid_auc'].argmax()
    best_model = results_df.iloc[best_idx]
    params_cols = [x for x in results_df.columns if 'score' not in x]
    best_params = results_df.iloc[best_idx][params_cols].to_dict()
    # Saving the best params
    # savepath = pathlib.Path(os.path.join(args['outdir'], outname + args['outname']+'.pkl')).absolute()
    pkl_dump(best_params, os.path.join(args['outdir'], outname + args['outname']+'.pkl'))
    outdict['best_params'] = best_params
    results_df.to_csv(os.path.join(args['outdir'], outname + args['outname']+'.csv'), index=False)

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

    device = 'cuda' if torch.cuda.is_available() and args['gpu'] else 'cpu'
    print(f'Using {device}')
    # Read data and assign train set name
    dataset_cedar = pd.read_csv(f'{args["datadir"]}cedar_10fold.csv')
    dataset_cedar['trainset'] = 'cedar'

    dataset_cedar_hp_rank_low = pd.read_csv(f'{args["datadir"]}cedar_hp_mixed_10fold.csv').query('Peptide!="YTKDGIGL"')
    dataset_cedar_hp_rank_low['trainset'] = 'cedar_hp_rank_low'

    xl = ['HAGLLQTV', 'KSISALPV', 'VLDASKAL']
    dataset_cedar_hp_rank_uni = pd.read_csv(f'{args["datadir"]}cedar_hp_mixed_rank120_10fold.csv').query(
        'Peptide not in @xl')
    dataset_cedar_hp_rank_uni['trainset'] = 'cedar_hp_rank_uni'

    dataset_cedar_virus = pd.read_csv(f'{args["datadir"]}cedar_viral_10fold.csv')
    dataset_cedar_virus['trainset'] = 'cedar_virus'

    ics_shannon = pkl_load(f'{args["icsdir"]}ics_shannon.pkl')
    ics_kl = pkl_load(f'{args["icsdir"]}ics_kl.pkl')
    ics_none = None  # bit stupid but need to have the variable name somewhere :-)
    if args["debug"]:
        encode_blosum_zip = zip(['onehot', 'blosum'], [None, BL62FREQ_VALUES])
        ics_mask_zip = zip([ics_shannon, ics_none, ics_shannon],
                           ['Shannon', 'None', 'Mask'],
                           [False, False, True])
        features_zip = zip([True], [True], [False])
        train_datasets = [dataset_cedar]
        standardize_ = [True]
        lrs = [1e-4]
        wds = [1e-2]
    else:
        # Creating the condition/products/zips

        # encoding, blosum matrix
        encode_blosum_zip = zip(['onehot', 'blosum', 'blosum'], [None, BL62FREQ_VALUES, BL62_VALUES])
        # Weighting zip (ic dict, ic name, mask bool)
        ics_mask_zip = zip([ics_shannon, ics_kl, ics_none, ics_shannon], ['Shannon', 'KL', 'None', 'Mask'],
                           [False, False, False, True])
        # True/False zips for add_rank, add_aaprop, remove_pep
        features_zip = zip([True, True, True, False, False, False], [True, True, False, True, True, False],
                           [False, True, False, False, True, False])
        # Lone conditions (to be producted so not zipped)
        train_datasets = [dataset_cedar, dataset_cedar_hp_rank_low, dataset_cedar_hp_rank_uni, dataset_cedar_virus]
        standardize_ = [True, False]
        lrs = [5e-4, 1e-3]
        wds = [1e-6]


    # Here make the conditions that don't rely on hyperparams, i.e. all dataset and dataprocessing conditions
    conditions = product(train_datasets,
                         product(encode_blosum_zip,
                                 product(ics_mask_zip,
                                         product(features_zip,
                                                 product(standardize_,
                                                         product(lrs,
                                                                 product(wds)))))))
    # Flatten the list of products
    # Gets the argument in the order :
    # train_dataset, encoding, blosum matrix, ics dict, ics name, mask, add rank, add prop, remove pep, standardize
    conditions = list(list(flatten_product(x)) for x in conditions)

    # Manual tuning and their hyperparameters to tune
    if args['debug']:
        hyperparams_grid = {'n_hidden': [12], 'n_layers': [1], 'act': [nn.ReLU()]}
    else:
        hyperparams_grid = {'n_hidden': [16, 24, 32], 'n_layers': [1, 2],
                                   'act': [nn.ReLU(), nn.SELU()]}

    tune_results_model = {}
    print(f'\n\n\n\tCPU COUNT: {multiprocessing.cpu_count()}, using {args["ncores"]}\n\n\n')
    # Loop with parallelized jobs
    wrapper = partial(tune_wrapper, args=args, hyperparams=hyperparams_grid, device=device)
    print(f'Running {len(conditions)-2*176} ish conditions')
    output = Parallel(n_jobs=args['ncores'])(delayed(wrapper)(train_dataset=train_dataset, encoding=encoding,
                                                 blosum_matrix=blosum_matrix, ics_dict=ics_dict, ics_name=ics_name,
                                                 mask=mask, add_rank=add_rank, add_aaprop=add_aaprop,
                                                 remove_pep=remove_pep, standardize=standardize, learning_rate=learning_rate,
                                                 weight_decay=weight_decay) for \
                                (train_dataset, encoding, blosum_matrix, ics_dict, ics_name, mask, add_rank,
                                 add_aaprop, remove_pep, standardize, learning_rate, weight_decay) in conditions)
    output = [(x[0].sort_values('score_avg_valid_auc'), x[-1]) for x in output if x is not None]
    pkl_dump(output, os.path.join(args['outdir'], 'tune_results_models.pkl'))
    end = dt.now()
    elapsed = (end - start).seconds
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    print(f'Time elapsed: {hours} hours, {minutes} minutes, {seconds} seconds.')


if __name__ == '__main__':
    main()
