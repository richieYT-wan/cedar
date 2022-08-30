import copy

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import argparse
from joblib import Parallel, delayed, parallel_backend
from functools import partial
import multiprocessing
from itertools import product
import torch
from torch import nn
from torch import optim

from tqdm.auto import tqdm
import warnings
from datetime import datetime as dt
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Custom fct imports
from src.models import FFN
from src.utils import pkl_load, pkl_dump, flatten_product
from src.data_processing import BL62_VALUES, BL62FREQ_VALUES
from src.utils import str2bool, mkdirs, convert_path
from src.train_eval import nested_kcv_train_nn_freq, evaluate_trained_models_nn_freq
from src.train_eval import nested_kcv_train_sklearn, evaluate_trained_models_sklearn

warnings.filterwarnings('ignore')
N_CORES = 1 + (multiprocessing.cpu_count() // 3)


def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to crossvalidate and evaluate methods that use aa frequency as encoding')

    parser.add_argument('-datadir', type=str, default='../data/partitioned_traindata/',
                        help='Path to directory containing the pre-partitioned data')
    parser.add_argument('-outdir', type=str, default='../output/traineval/')
    parser.add_argument('-icsdir', type=str, default='../data/ic_dicts/',
                        help='Path containing the pre-computed ICs dicts.')
    parser.add_argument('-tunedir', type=str, default='../output/tuning/run_220819_03h13m45s/')
    parser.add_argument('-model', type=str, default='', help='"rf", "log" or "xgb_normal", "xgb_hp" or "nn"')
    parser.add_argument('-namesuffix', type=str, default='', help='Additional suffix that may be added to your filenames, '
                                                                  'ex _NN ; Otherwise won\'t be able to properly read your files.')
    parser.add_argument('-ncores', type=int, default=None, help='N cores to use in parallel, by default will be multiprocesing.cpu_count() * 3/4')
    parser.add_argument('-gpu', type=str2bool, default=False, help='Enable GPU (default False)')
    parser.add_argument('-debug', type=str2bool, default=False)

    return parser.parse_args()


def train_eval_wrapper(args, model, train_dataset, cedar_dataset, prime_dataset,
                       ics_dict, ics_name, encoding, blosum_matrix, mask, add_rank, add_aaprop,
                       remove_pep, standardize, learning_rate=None, weight_decay=None, device='cpu'):
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

    # HERE CONDITION TO EARLY EXIT IF REMOVE_PEP IS TRUE, AS WE DON'T NEED 100 CONDITIONS with
    # IC dicts, blosum, etc.
    # If remove_pep is true, we should just keep a single condition with Onehot and no weight and no blosum
    # This drops 176 conditions for each of the 576 loops per model
    # --> Leaves 400 runs per model
    if remove_pep:
        if encoding != 'onehot' or ics_name != 'None':
            return None

    if not issubclass(model.__class__, nn.Module):
        learning_rate = None
        weight_decay = None


    # Creating the encoding kwargs dict
    encoding_kwargs = {'max_len': 12,
                       'encoding': encoding,
                       'blosum_matrix': blosum_matrix,
                       'mask': mask,  # Using Shannon ICs, true if both mask and name is "shannon"
                       'add_rank': add_rank,
                       'add_aaprop': add_aaprop,
                       'remove_pep': remove_pep,
                       'standardize': standardize}

    # Here sets the output tag (names, variables and their values)
    # quickly rename the model name so it's shorter
    mapping = {'RandomForestClassifier': 'RF',
               'LogisticRegression': 'LogReg',
               'XGBClassifier': 'XGB',
               'FFN': 'FFN'}
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
    if weight_decay not in [-1, None] and learning_rate not in [-1, None]:
        outdict['lr']=learning_rate
        outdict['wd']=weight_decay

    outname = '_'.join([f'{k}{v}' for k, v in outdict.items()])+args['namesuffix']

    # Here use the outname to look for the corresponding filename for the best hyperparams
    try:
        pkl_name = convert_path(os.path.join(args['tunedir'],(outname+'.pkl')).replace('//','/'))
        hyperparams = pkl_load(pkl_name)
        if 'max_depth' in hyperparams:
            if np.isnan(hyperparams['max_depth']):
                hyperparams['max_depth']=None
            else:
                hyperparams['max_depth']=int(hyperparams['max_depth'])

        for k in [x for x in hyperparams.keys() if x.startswith('n_')]:
            # converts n_estimators, n_layers, n_in, n_hidden, to int
            hyperparams[k] = int(hyperparams[k])
    except:
        print(f'Couldn\'t reload the params for {outname} in {args["tunedir"]}! Please check the input.')
        print(f'Please check the contents of {args["tunedir"]}')
        raise Exception

    # Training part of the wrapper after setting up the options
    # TODO
    if issubclass(model.__class__, nn.Module):
        # nested_kcv_train_nn(kwargs)
        # Variable n_in so this is fine to just give the class constructor as it is
        model_constructor = model.__class__
        models_dict, train_metrics, test_metrics = nested_kcv_train_nn_freq(train_dataset, model_constructor, ics_dict,
                                                                            encoding_kwargs, hyperparams, criterion=nn.BCELoss(),
                                                                            optimizer=optim.Adam(model.parameters()), device=device,
                                                                            filename=outname)
        print('Evaluating models')
        # Evaluation part of the wrapper
        cedar_results = evaluate_trained_models_nn_freq(cedar_dataset, models_dict, ics_dict, device, train_dataset,
                                                        train_metrics, encoding_kwargs=encoding_kwargs,
                                                        concatenated=True)

        prime_results = evaluate_trained_models_nn_freq(prime_dataset, models_dict, ics_dict, device,
                                                        train_metrics=train_metrics,
                                                        encoding_kwargs=encoding_kwargs, concatenated=True)

    else:
        # not saving the fucking models because it's TOO BIG
        model.set_params(**hyperparams)
        models_dict, train_metrics, test_metrics = nested_kcv_train_sklearn(train_dataset, model, ics_dict,
                                                                            encoding_kwargs)
        print('Evaluating models')
        # Evaluation part of the wrapper
        cedar_results = evaluate_trained_models_sklearn(cedar_dataset, models_dict, ics_dict, train_dataset,
                                                        train_metrics, encoding_kwargs=encoding_kwargs, concatenated=True)

        prime_results = evaluate_trained_models_sklearn(prime_dataset, models_dict, ics_dict, train_metrics=train_metrics,
                                                        encoding_kwargs=encoding_kwargs, concatenated=True)

    # Big mess to save results :-)
    outdict['score_avg_prime_auc'] = np.mean([prime_results[k]['auc'] for k in prime_results.keys() if k != 'concatenated'])
    outdict['score_avg_prime_auc_01'] = np.mean([prime_results[k]['auc_01'] for k in prime_results.keys() if k != 'concatenated'])
    outdict['score_avg_prime_f1'] = np.mean([prime_results[k]['f1'] for k in prime_results.keys() if k != 'concatenated'])

    outdict['score_concat_prime_auc'] = prime_results['concatenated']['auc']
    outdict['score_concat_prime_auc_01'] = prime_results['concatenated']['auc_01']
    outdict['score_concat_prime_f1'] = prime_results['concatenated']['f1']

    outdict['score_avg_cedar_auc'] = np.mean([cedar_results[k]['auc'] for k in cedar_results.keys() if k != 'concatenated'])
    outdict['score_avg_cedar_auc_01'] = np.mean([cedar_results[k]['auc_01'] for k in cedar_results.keys() if k != 'concatenated'])
    outdict['score_avg_cedar_f1'] = np.mean([cedar_results[k]['f1'] for k in cedar_results.keys() if k != 'concatenated'])

    outdict['score_concat_cedar_auc'] = cedar_results['concatenated']['auc']
    outdict['score_concat_cedar_auc_01'] = cedar_results['concatenated']['auc_01']
    outdict['score_concat_cedar_f1'] = cedar_results['concatenated']['f1']

    if issubclass(model.__class__, nn.Module):
        outdict['score_avg_valid_auc'] = np.mean([v2['valid']['auc'][-1] for _, v1 in train_metrics.items() for _, v2 in v1.items()])
        outdict['score_avg_valid_auc_01'] = np.mean([v2['valid']['auc_01'][-1] for _, v1 in train_metrics.items() for _, v2 in v1.items()])
        outdict['score_avg_valid_f1'] = np.mean([v2['valid']['f1'][-1] for _, v1 in train_metrics.items() for _, v2 in v1.items()])
    else:
        outdict['score_avg_valid_auc'] = np.mean([v2['valid']['auc'] for _, v1 in train_metrics.items() for _, v2 in v1.items()])
        outdict['score_avg_valid_auc_01'] = np.mean([v2['valid']['auc_01'] for _, v1 in train_metrics.items() for _, v2 in v1.items()])
        outdict['score_avg_valid_f1'] = np.mean([v2['valid']['f1'] for _, v1 in train_metrics.items() for _, v2 in v1.items()])

    # Single line dataframe
    results_df = pd.DataFrame(outdict, index=[0])
    train_metrics['kwargs'] = outdict
    cedar_results['kwargs'] = outdict
    prime_results['kwargs'] = outdict

    # NOT SAVING THE TRAINED MODEL BECAUSE IT'S SHIT
    # models_dict['kwargs'] = outdict
    return results_df, train_metrics, cedar_results, prime_results


def main():
    start = dt.now()
    run_id = f'run_{start.strftime(f"%y%m%d_%Hh%Mm%Ss")}/'
    args = vars(args_parser())
    assert args['model'] in ['rf', 'log', 'xgb_normal', 'xgb_hp', 'nn'], f"Undefined model specified {args['model']}. Should be in ['rf', " \
                                                  f"'log', 'xgb_normal', 'xgb_hp', 'nn']! "
    args['outdir'], args['datadir'], args['icsdir'] = convert_path(args['outdir']), convert_path(
        args['datadir']), convert_path(args['icsdir'])
    args['outdir'] = convert_path(os.path.join(args['outdir'], run_id+str(args['model'])))
    args['outdir'] = args['outdir']
    # Will make all the nested dirs
    mkdirs(args['outdir'])
    N_CORES = int(multiprocessing.cpu_count()*3/4)+int(multiprocessing.cpu_count()*0.05) if (args['ncores'] is None or 'xgb' in args['model']) else args['ncores']
    N_CORES = 12 if args['debug'] else N_CORES

    device = 'cuda' if (args['gpu'] and torch.cuda.is_available()) else 'cpu'
    tree_method = 'gpu_hist' if args['gpu'] else 'hist'
    # ============ DATA LOADING ============ #
    # Read data and assign train set name
    dataset_cedar = pd.read_csv(f'{args["datadir"]}cedar_10fold.csv')
    dataset_cedar['trainset'] = 'cedar'
    # Low rank cedar-hp dataset
    dataset_cedar_hp_rank_low = pd.read_csv(f'{args["datadir"]}cedar_hp_mixed_10fold.csv').query('Peptide!="YTKDGIGL"')
    dataset_cedar_hp_rank_low['trainset'] = 'cedar_hp_rank_low'
    # Uniform rank (0-20) cedar-hp dataset
    xl = ['HAGLLQTV', 'KSISALPV', 'VLDASKAL']
    dataset_cedar_hp_rank_uni = pd.read_csv(f'{args["datadir"]}cedar_hp_mixed_rank120_10fold.csv').query(
        'Peptide not in @xl')
    dataset_cedar_hp_rank_uni['trainset'] = 'cedar_hp_rank_uni'
    # Cedar + Virus dataset
    dataset_cedar_virus = pd.read_csv(f'{args["datadir"]}cedar_viral_10fold.csv')
    dataset_cedar_virus['trainset'] = 'cedar_virus'

    # PRIME dataset as external test set
    cedar_peps = dataset_cedar.Peptide.values
    prime_dataset = pd.read_excel(f'{args["datadir"]}PRIME_dataset.xlsx', skiprows=2, comment='#')
    # Filter to avoid evaluating on trained peptides
    prime_dataset = prime_dataset.query('StudyOrigin!="Random" and Mutant not in @cedar_peps')
    # RENAME COLUMNS to avoid re-creating a custom prime kwargs every in function call
    prime_dataset.rename(columns={'Mutant': 'Peptide', 'Allele': 'HLA',
                                  'Immunogenicity': 'agg_label', 'NetMHCpanEL': 'trueHLA_EL_rank'},
                         inplace=True)

    # IC weights
    ics_shannon = pkl_load(f'{args["icsdir"]}ics_shannon.pkl')
    ics_kl = pkl_load(f'{args["icsdir"]}ics_kl.pkl')
    ics_none = None  # bit stupid but need to have the variable name somewhere :-)

    # ============ Conditions Product/Zip ============ #
    # Making the conditions
    # Could / should've used sklearn.model_processing ParameterGrid instead and re-think the dictionary/wrapper
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
        encode_blosum_zip = zip(['onehot', 'blosum', 'blosum'], [None, BL62FREQ_VALUES, BL62_VALUES])
        # Weighting zip
        ics_mask_zip = zip([ics_shannon, ics_kl, ics_none, ics_shannon], ['Shannon', 'KL', 'None', 'Mask'],
                           [False, False, False, True])
        # True/False zips for add_rank, add_aaprop, remove_pep
        features_zip = zip([True, True, True, False, False, False], [True, True, False, True, True, False],
                           [False, True, False, False, True, False])

        # Lone conditions (to be producted so not zipped)
        if 'xgb' not in args['model']:
            train_datasets = [dataset_cedar, dataset_cedar_hp_rank_low, dataset_cedar_hp_rank_uni, dataset_cedar_virus]
        elif args['model'] == 'xgb_normal':
            train_datasets = [dataset_cedar, dataset_cedar_virus]
        elif args['model'] == 'xgb_hp':
            train_datasets = [dataset_cedar_hp_rank_low, dataset_cedar_hp_rank_uni]
        standardize_ = [True, False]
        lrs = [5e-4, 1e-3] if args['model'] == 'nn' else [-1]
        wds = [ 1e-6, 1e-2] if args['model'] == 'nn' else [-1]



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

    # ============ Setting up models ============= #
    # Manual tuning and their hyperparameters to tune

    models = {'rf': RandomForestClassifier(),
              'log': LogisticRegression(tol=1e-5, max_iter=250, solver='saga'),
              'xgb_hp': XGBClassifier(tree_method=tree_method),
              'xgb_normal': XGBClassifier(tree_method=tree_method),
              'nn': FFN(n_in=20)}

    # ============ Parallel run ============ #
    print(f'\n\n\n\tCPU COUNT: {multiprocessing.cpu_count()}; Using {N_CORES}\n\n\n')
    # Loop with parallelized jobs
    print(f'Running train & evaluation for {args["model"]}')
    # for model in tqdm(models_params_grid, desc='models', leave=False):
    model = models[args['model']]
    wrapper = partial(train_eval_wrapper, args=args, model=model,
                      cedar_dataset=dataset_cedar, prime_dataset=prime_dataset, device=device)
    output = Parallel(n_jobs=N_CORES)(
        delayed(wrapper)(train_dataset=train_dataset, encoding=encoding,
                         blosum_matrix=blosum_matrix, ics_dict=ics_dict, ics_name=ics_name,
                         mask=mask, add_rank=add_rank, add_aaprop=add_aaprop,
                         remove_pep=remove_pep, standardize=standardize,
                         learning_rate=lr, weight_decay=wd ) for \
        (train_dataset, encoding, blosum_matrix, ics_dict, ics_name, mask, add_rank,
         add_aaprop, remove_pep, standardize, lr, wd) in conditions)
    output = [x for x in output if x is not None]
    print('Saving results')
    # DF
    df = pd.concat([x[0] for x in output]).reset_index(drop=True)
    df.to_csv(os.path.join(args['outdir'], f'df_results_{args["model"]}.csv'), index=False)

    # models = [x[1] for x in output]
    train_metrics = [x[1] for x in output]
    cedar_results = [x[2] for x in output]
    prime_results = [x[3] for x in output]
    # pkl_dump(models, f'models_{args["model"]}.pkl', dirname=args['outdir'])
    pkl_dump(train_metrics, f'train_metrics_{args["model"]}.pkl', dirname=args['outdir'])
    pkl_dump(cedar_results, f'cedar_results_{args["model"]}.pkl', dirname=args['outdir'])
    pkl_dump(prime_results, f'prime_results_{args["model"]}.pkl', dirname=args['outdir'])


    # pkl_dump(tune_results_model, os.path.join(args['outdir'], 'tune_results_models.pkl'))
    end = dt.now()
    elapsed = (end - start).seconds
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    print(f'Time elapsed: {hours} hours, {minutes} minutes, {seconds} seconds.')


if __name__ == '__main__':
    main()
