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

from joblib import Parallel, delayed
from functools import partial
import multiprocessing
import itertools
from copy import deepcopy
from src.models import FFNetWrapper, FFN, LinearBlock
from src.nn_train_eval import nested_kcv_train_nn, evaluate_trained_models_nn

mkdirs('../output/nn_test/')
training_kwargs = dict(n_epochs=500, early_stopping=False, patience=50, delta=1e-10, filename='../output/nn_test/checkpoint_test',
                       verbosity=1)
encoding_kwargs = dict(max_len=12, encoding='onehot', blosum_matrix=None, standardize=True, seq_col='Peptide',
                       hla_col='HLA', target_col='agg_label', rank_col='EL_rank_mut', mut_col=['dissimilarity_score'],
                       adaptive=False, mask=False, add_rank=True, add_aaprop=False, remove_pep=False)
ics_kl = pkl_load('../data/ic_dicts/ics_kl.pkl')
train_dataset = pd.read_csv('../data/mutant/221028_cedar_related_newcore_fold.csv')
eval_dataset = pd.read_csv('../data/mutant/221119_prime_related_10fold.csv')

ics_dict = ics_kl
model = FFNetWrapper(n_in=22, n_hidden=15, n_layers=2, dropout=0.15)
optimizer = optim.Adam(model.parameters(), lr=6.67e-5, weight_decay=1e-5)
criterion = nn.BCELoss()
device = 'cpu'
n_jobs = 8

models_dict, train_metrics, test_metrics = nested_kcv_train_nn(train_dataset, model, optimizer, criterion, device,
                                                     ics_dict, encoding_kwargs, training_kwargs, n_jobs)
test_results, predictions_df = evaluate_trained_models_nn(eval_dataset, models_dict, ics_dict, device, train_dataset,
                                                          encoding_kwargs, concatenated=True, only_concat=True, n_jobs=n_jobs)
pkl_dump(train_metrics, '../output/nn_test/train_metrics.pkl')
print('\ntrain', np.mean([x[-1] for x in [v2['train']['auc'] for k1, v1 in train_metrics.items() for _, v2 in v1.items()]]))
print('valid', np.mean([x[-1] for x in [v2['valid']['auc'] for k1, v1 in train_metrics.items() for _, v2 in v1.items()]]))
print('test', np.mean([x['auc'] for _, x in test_metrics.items()]))
print('prime concat', test_results['concatenated']['auc'])