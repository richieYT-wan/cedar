import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import torch
from torch import optim, nn
from itertools import product
from src.utils import mkdirs, convert_path, flatten_product
from src.utils import pkl_load, pkl_dump
from joblib import Parallel, delayed
from functools import partial
from src.models import FFNetPipeline
from src.nn_train_eval import nested_kcv_train_nn, evaluate_trained_models_nn

ics_kl = pkl_load('../data/ic_dicts/ics_kl.pkl')
train_dataset = pd.read_csv('../data/mutant/221028_cedar_related_newcore_fold.csv')
eval_dataset = pd.read_csv('../data/mutant/221119_prime_related_10fold.csv')
ics_dict = ics_kl

DIR_ = f'../output/nn_manual_tune/'
mkdirs(DIR_)
mkdirs(f'{DIR_}checkpoints/')
training_kwargs = dict(n_epochs=1000, early_stopping=True, patience=500, delta=1e-6,
                       filename=f'{DIR_}checkpoint_test',
                       verbosity=1)
encoding_kwargs = dict(max_len=12, encoding='onehot', blosum_matrix=None, standardize=True, seq_col='Peptide',
                       hla_col='HLA', target_col='agg_label', rank_col='EL_rank_mut',
                       mut_col=['mutation_score', 'ratio_rank'],
                       adaptive=False, mask=True, add_rank=True, add_aaprop=False, remove_pep=False)

criterion = nn.BCELoss()
device = 'cpu'
n_jobs = 1


def parallel_wrapper(lr, nh, nl):
    model = FFNetPipeline(n_in=23, n_hidden=nh, n_layers=nl, dropout=0.25)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
    training_kwargs['filename'] = f'{DIR_}checkpoints/lr{lr}_nh{nh}_nl{nl}_chkpt'
    models_dict, train_metrics, test_metrics = nested_kcv_train_nn(train_dataset, model, optimizer, criterion, device,
                                                                   ics_dict, encoding_kwargs, training_kwargs, n_jobs)

    test_results, predictions_df = evaluate_trained_models_nn(eval_dataset, models_dict, ics_dict, device,
                                                              train_dataset,
                                                              encoding_kwargs, concatenated=True, only_concat=True,
                                                              n_jobs=n_jobs)
    print(f'Eta:\t{lr},\tHidden:\t{nh},\tLayers:\t{nl}')
    print('\ntrain',
          np.mean([x[-1] for x in [v2['train']['auc'] for k1, v1 in train_metrics.items() for _, v2 in v1.items()]]))
    print('valid',
          np.mean([x[-1] for x in [v2['valid']['auc'] for k1, v1 in train_metrics.items() for _, v2 in v1.items()]]))
    print('test', np.mean([x['auc'] for _, x in test_metrics.items()]))
    print('prime concat', test_results['concatenated']['auc'])

    results = [lr, nh, nl, np.mean(
        [x[-1] for x in [v2['train']['auc'] for k1, v1 in train_metrics.items() for _, v2 in v1.items()]]),
               np.mean(
                   [x[-1] for x in [v2['valid']['auc'] for k1, v1 in train_metrics.items() for _, v2 in v1.items()]]),
               np.mean([x['auc'] for _, x in test_metrics.items()]),
               test_results['concatenated']['auc']]

    train_auc = np.stack([train_metrics[k1][k2]['train']['auc'] for k1 in train_metrics for k2 in train_metrics[k1]]),
    valid_auc = np.stack([train_metrics[k1][k2]['valid']['auc'] for k1 in train_metrics for k2 in train_metrics[k1]]),
    train_losses = np.stack(
        [train_metrics[k1][k2]['train']['losses'] for k1 in train_metrics for k2 in train_metrics[k1]]),
    valid_losses = np.stack(
        [train_metrics[k1][k2]['valid']['losses'] for k1 in train_metrics for k2 in train_metrics[k1]])

    mean_train_losses = np.mean(train_losses, axis=0)
    mean_valid_losses = np.mean(valid_losses, axis=0)
    std_train_losses = np.std(train_losses, axis=0)
    std_valid_losses = np.std(valid_losses, axis=0)
    low_train_losses = mean_train_losses - std_train_losses
    high_train_losses = mean_train_losses + std_train_losses
    low_valid_losses = mean_valid_losses - std_valid_losses
    high_valid_losses = mean_valid_losses + std_valid_losses

    mean_train_auc = np.mean(train_auc, axis=0)
    mean_valid_auc = np.mean(valid_auc, axis=0)
    std_train_auc = np.std(train_auc, axis=0)
    std_valid_auc = np.std(valid_auc, axis=0)
    low_train_auc = mean_train_auc - std_train_auc
    high_train_auc = mean_train_auc + std_train_auc
    low_valid_auc = mean_valid_auc - std_valid_auc
    high_valid_auc = mean_valid_auc + std_valid_auc

    f, a = plt.subplots(1, 2, figsize=(12, 4))
    f.suptitle(f'lr: {lr},   nh: {nh},   nl: {nl}')
    x = np.arange(1, len(mean_train_auc) + 1, 1)
    a[0].plot(x, mean_train_losses, label='mean_train_loss')
    a[0].fill_between(x, y1=low_train_losses,
                      y2=high_train_losses, alpha=0.175)

    a[0].plot(x, mean_valid_losses, label='mean_valid_loss')
    a[0].fill_between(x, y1=low_valid_losses,
                      y2=high_valid_losses, alpha=0.175)
    a[0].legend()
    a[0].set_title('Losses')
    a[0].set_xlabel('Epoch')
    a[1].plot(x, mean_train_auc, label='mean_train_auc')
    a[1].fill_between(x, y1=low_train_auc,
                      y2=high_train_auc, alpha=0.175)

    a[1].plot(x, mean_valid_auc, label='mean_valid_auc')
    a[1].fill_between(x, y1=low_valid_auc,
                      y2=high_valid_auc, alpha=0.175)
    a[1].legend(loc='lower right')
    a[1].set_title('AUCs')
    a[1].set_xlabel('Epoch')
    f.savefig(f'{DIR_}lr{lr}_nh{nh}_nl{nl}_fig.pn', bbox_inches='tight')
    return results


lrs = [1e-5, 5e-5, 1e-4]
nhs = [10, 20, 30, 40]
nls = [1, 2, 3, 4]
conditions = product(lrs, product(nhs, product(nls)))
conditions = list(list(flatten_product(x)) for x in conditions)

output = Parallel(n_jobs=4)(delayed(parallel_wrapper)(lr=lr, nh=nh, nl=nl) for lr, nh, nl in conditions)
results = pd.DataFrame(output, columns=['lr', 'n_hidden', 'n_layers', 'train_auc', 'valid_auc', 'test_auc',
                                        'external_prime_auc'])
results.to_csv(f'{DIR_}manual_tune_results.csv', index=False)
