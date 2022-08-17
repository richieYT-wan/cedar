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
from src.utils import pkl_load, pkl_dump
from torch import nn


from src.models import LinearBlock
from src.data_processing import BL62_VALUES, BL62FREQ_VALUES
from src.utils import str2bool
from src.train_eval import nested_kcv_train_nn, nested_kcv_train_sklearn, evaluate_trained_models_sklearn, evaluate_trained_models_nn

N_CORES = 1+(multiprocessing.cpu_count()//3)

def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to crossvalidate and evaluate methods that use aa frequency as encoding')

    parser.add_argument('-datadir', type=str, default='./data/script_traindata/',
                        help='Path to directory containing the pre-partitioned data')
    parser.add_argument('-outdir', type=str, default='./output/')
    parser.add_argument('-icsdir', type=str, default='./output/', help='Path containing the pre-computed ICs dicts.')

    return parser.parse_args()


def train_wrapper(train_dataset, model, ics_dict, ics_name,
                  encoding, blosum_matrix, mask, add_rank, add_aaprop,
                  remove_pep, standardize):

    encoding_kwargs = {'max_len':12,
                       'encoding':encoding,
                       'blosum_matrix':blosum_matrix,
                       'mask': mask, # Using Shannon ICs, true if both mask and name is "shannon"
                       'add_rank': add_rank,
                       'add_aaprop': add_aaprop,
                       'remove_pep':remove_pep,
                       'standardize': standardize}

    # TODO : redo the fct for nested kcv train nn
    if issubclass(model.__class__, nn.Module):
        # nested_kcv_train_nn(kwargs)
        pass
    else:
        models, train_results, test_results = nested_kcv_train_sklearn(train_dataset, model, ics_dict, encoding_kwargs)
    # quickly rename the model name so it's shorter
    mapping = {'RandomForestClassifier':'RF',
               'LogisticRegression':'LogReg',
               'XGBClassifier':'XGB',
               'LinearBlock':'LinearBlock'}
    # Save the parameter name as a massive string to tag the output.
    # Set the "trainset" name in the beginning of the for loop by assigning name to a train_dataset column
    # i.e. train_dataset['trainset'] = 'name'
    outname = {'trainset':train_dataset['trainset'].unique()[0],
               'model': mapping[model.__class__.__name__],
               'weight': ics_name}
    outname.update({k: v for k, v in encoding_kwargs.items() if k != 'blosum_matrix'})
    if blosum_matrix is not None:
        bl_name = 'BL62LO' if blosum_matrix.dtype == np.int64 else 'BL62FREQ'
        outname['blsm'] = bl_name

    outname.update({k:v for k,v in encoding_kwargs.items() if k!='blosum_matrix'})
    # Merging each tag into one
    outname = '_'.join([f'{k}{v}' for k,v in outname.items()])

    return (models, train_results, test_results), outname

def evaluate_wrapper(test_dataset, model_dict, encoding, blosum_matrix, mask, add_rank, add_aaprop,
                     remove_pep, standardize, seq_col, hla_col, target_col, rank_col, train_results:dict):
    """
    Only needed to evaluate on prime because I can just get the test performance from the train output
    Args:
        test_dataset:
        model_dict:
        encoding:
        blosum_matrix:
        mask:
        add_rank:
        add_aaprop:
        remove_pep:
        standardize:
        seq_col:
        hla_col:
        target_col:
        rank_col:
        train_results:

    Returns:

    """
    encoding_kwargs = {'max_len':12,
                       'encoding':encoding,
                       'blosum_matrix':blosum_matrix,
                       'mask': mask,
                       'add_rank': add_rank,
                       'add_aaprop': add_aaprop,
                       'remove_pep':remove_pep,
                       'standardize':standardize,
                       # Putting these here just in case ; Not really needed (will be needed in eval wrapper)
                       'seq_col':seq_col,
                       'hla_col':hla_col,
                       'target_col':target_col,
                       'rank_col':rank_col}
    # Removing the unnecessary items in the dictionary so to only keep mus and sigmas
    for k1, v1 in train_results.items():
        for k2, v2 in v1.items():
            del v2['train'], v2['valid']
    if blosum_matrix is not None:
        bl_name = 'BL62LO' if blosum_matrix.dtype == np.int64 else 'BL62FREQ'
        encoding_kwargs['blsm'] = bl_name
    # Naming each tag using a stupid for loop
    outname = '_'.join([f'{k}{v}' for k,v in encoding_kwargs.items() if k!='blosum_matrix' and '_col' not in k])
    return 0

def main():
    args = args_parser()
    # Read data and assign train set name
    dataset_cedar = pd.read_csv(f'{args["datadir"]}cedar_10fold.csv')
    dataset_cedar_hp = pd.read_csv(f'{args["datadir"]}cedar_hp_mixed_10fold.csv')
    ics_shannon = pkl_load(f'{args["icsdir"]}ics_shannon.pkl')
    ics_kl = pkl_load(f'{args["icsdir"]}ics_kl.pkl')
    ics_none = None  # bit stupid but need to have the variable name somewhere :-)

    results = []
    # FIRST LOOP FOR THE TRAIN WRAPPER
    for train_dataset, dataset_name in zip([dataset_cedar, dataset_cedar_hp], ['cedar', 'cedar_hp']):
        for standardize in [True, False]:
            for remove_pep in [True, False]:
                for encoding in ['onehot', 'blosum']:
                    if encoding == 'blosum':
                        for
        for model in [RandomForestClassifier, XGBClassifier, LogisticRegression, LinearBlock]:
            kcv_wrapper_ = partial(train_wrapper, dataset=train_dataset, dataset_name = dataset_name,
                                  model_type= model)
            output = Parallel(n_jobs=4)(
                delayed(kcv_wrapper_)(weighting=weighting, weighting_name=weighting_name, add_rank=add_rank) for
                (weighting, weighting_name), add_rank in \
                product(zip([ics_shannon, ics_kl, ics_none], ['shannon', 'kl', 'none']), [True, False]))
            results.extend(output)

    # SECOND LOOP BY RE-READING THE TRAIN_RESULTS, AND GET THE EVALUATION PERFORMANCE
    # Doing this makes it easier to re-group the run parameters together as it re-uses the `outname` variable
    # present in x[1] for x in train_results ; Easier but also the script is slower since I have to
    # Iterate a second time through the ~2300 run conditions, but will save me time when saving results


