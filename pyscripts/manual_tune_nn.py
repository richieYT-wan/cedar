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

training_kwargs = dict(n_epochs=1000, early_stopping=True, patience=500, delta=1e-6,
                       filename='../output/nn_test/checkpoint_test',
                       verbosity=1)
encoding_kwargs = dict(max_len=12, encoding='onehot', blosum_matrix=None, standardize=True, seq_col='Peptide',
                       hla_col='HLA', target_col='agg_label', rank_col='EL_rank_mut',
                       mut_col=['mutation_score', 'ratio_rank'],
                       adaptive=False, mask=True, add_rank=True, add_aaprop=False, remove_pep=False)

model = FFNetPipeline(n_in=22, n_hidden=25, n_layers=3, dropout=0.25)

lr = 5e-5
wd = 5e-3
DIR_ = f'../output/nn_test/{lr}_{wd}_Stop/'
mkdirs(DIR_)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd, amsgrad=True)

criterion = nn.BCELoss()
device = 'cpu'
n_jobs = 1



def parallel_wrapper(lr, nh, nl):
    model = FFNetPipeline(n_in=23, n_hidden=nh, n_layers=nl, dropout=0.25)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)

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

    results=[lr, nh, nl, np.mean([x[-1] for x in [v2['train']['auc'] for k1, v1 in train_metrics.items() for _, v2 in v1.items()]]),
             np.mean([x[-1] for x in [v2['valid']['auc'] for k1, v1 in train_metrics.items() for _, v2 in v1.items()]]),
             np.mean([x['auc'] for _, x in test_metrics.items()]),
             test_results['concatenated']['auc']]
    return results

lrs = [1e-5, 5e-5, 1e-4]
nhs = [10, 20, 30, 40]
nls = [1, 2, 3, 4]
conditions = product(lrs, product(nhs, product(nls)))
conditions = list(list(flatten_product(x)) for x in conditions)

output = Parallel(n_jobs=4)(delayed(parallel_wrapper)(lr=lr, nh=nh, nl=nl) for lr, nh, nl in conditions)
results = pd.DataFrame(output, columns = ['lr', 'n_hidden', 'n_layers', 'train_auc', 'valid_auc', 'test_auc', 'external_prime_auc'])
results.to_csv('./manual_tune_results.csv', index=False)