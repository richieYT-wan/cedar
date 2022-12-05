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

from src.utils import mkdirs, convert_path, pkl_dump, pkl_load, display_side
from src.data_processing import get_dataset, BL62_VALUES, BL62FREQ_VALUES, HLAS, AA_KEYS
from src.utils import pkl_load, pkl_dump

from src.models import FFNetPipeline
from src.nn_train_eval import nested_kcv_train_nn, evaluate_trained_models_nn


ics_kl = pkl_load('../data/ic_dicts/ics_kl.pkl')
train_dataset = pd.read_csv('../data/mutant/221028_cedar_related_newcore_fold.csv')
eval_dataset = pd.read_csv('../data/mutant/221119_prime_related_10fold.csv')
ics_dict = ics_kl


training_kwargs = dict(n_epochs=500, early_stopping=True, patience=500, delta=1e-6,
                       filename='../output/nn_test/checkpoint_test',
                       verbosity=1)
encoding_kwargs = dict(max_len=12, encoding='onehot', blosum_matrix=None, standardize=True, seq_col='Peptide',
                       hla_col='HLA', target_col='agg_label', rank_col='EL_rank_mut', mut_col=['mutation_score'],
                       adaptive=False, mask=True, add_rank=True, add_aaprop=False, remove_pep=False)

model = FFNetPipeline(n_in=22, n_hidden=25, n_layers=3, dropout=0.25)

lr=5e-5
wd=5e-3
DIR_ = f'../output/nn_test/{lr}_{wd}_Stop/'
mkdirs(DIR_)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd, amsgrad=True)

criterion = nn.BCELoss()
device = 'cpu'
n_jobs = 1

# def parallel_wrapper(lr, nh):
#     model = FFNetPipeline(n_in=22, n_hidden=nh, n_layers=2, dropout=0.25)
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
#
#     models_dict, train_metrics, test_metrics = nested_kcv_train_nn(train_dataset, model, optimizer, criterion, device,
#                                                                    ics_dict, encoding_kwargs, training_kwargs, n_jobs)
#
#     test_results, predictions_df = evaluate_trained_models_nn(eval_dataset, models_dict, ics_dict, device,
#                                                               train_dataset,
#                                                               encoding_kwargs, concatenated=True, only_concat=True,
#                                                               n_jobs=n_jobs)
#     print('\ntrain',
#           np.mean([x[-1] for x in [v2['train']['auc'] for k1, v1 in train_metrics.items() for _, v2 in v1.items()]]))
#     print('valid',
#           np.mean([x[-1] for x in [v2['valid']['auc'] for k1, v1 in train_metrics.items() for _, v2 in v1.items()]]))
#     print('test', np.mean([x['auc'] for _, x in test_metrics.items()]))
#     print('prime concat', test_results['concatenated']['auc'])
#
#
#
# for lr in [1e-4, 5e-4, 1e-3, 5e-3]:
#     wr = partial(parallel_wrapper, lr=lr)
#     output = Parallel(n_jobs=4)(delayed(wr)(nh=nh) for nh in [10, 20, 30, 40])
#


models_dict, train_metrics, test_metrics = nested_kcv_train_nn(train_dataset, model, optimizer, criterion, device,
                                                               ics_dict, encoding_kwargs, training_kwargs, n_jobs)
test_results, predictions_df = evaluate_trained_models_nn(eval_dataset, models_dict, ics_dict, device, train_dataset,
                                                          encoding_kwargs, concatenated=True, only_concat=True,
                                                          n_jobs=n_jobs)
print('\ntrain',
      np.mean([x[-1] for x in [v2['train']['auc'] for _, v1 in train_metrics.items() for _, v2 in v1.items()]]))
print('valid',
      np.mean([x[-1] for x in [v2['valid']['auc'] for _, v1 in train_metrics.items() for _, v2 in v1.items()]]))
print('test', np.mean([x['auc'] for _, x in test_metrics.items()]))
print('prime concat', test_results['concatenated']['auc'])

dump = dict(
    train_auc=np.stack([train_metrics[k1][k2]['train']['auc'] for k1 in train_metrics for k2 in train_metrics[k1]]),
    valid_auc=np.stack([train_metrics[k1][k2]['valid']['auc'] for k1 in train_metrics for k2 in train_metrics[k1]]),
    train_losses=np.stack(
        [train_metrics[k1][k2]['train']['losses'] for k1 in train_metrics for k2 in train_metrics[k1]]),
    valid_losses=np.stack(
        [train_metrics[k1][k2]['valid']['losses'] for k1 in train_metrics for k2 in train_metrics[k1]]))

for k in list(dump.keys()):
    dump[f'mean_{k}'] = np.mean(dump[k], axis=0)
    std = np.std(dump[k])
    dump[f'std_{k}'] = std
    dump[f'low_{k}'] = dump[f'mean_{k}'] - std
    dump[f'high_{k}'] = dump[f'mean_{k}'] + std

pkl_dump(dump, f'{DIR_}train_metrics.pkl')
