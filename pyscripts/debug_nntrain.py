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
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tqdm.auto import tqdm
from datetime import datetime as dt
from src.utils import mkdirs, convert_path, pkl_dump, pkl_load, display_side
from src.data_processing import get_dataset, BL62_VALUES, BL62FREQ_VALUES, HLAS, AA_KEYS
from src.metrics import get_predictions, get_roc, get_metrics, plot_roc_auc_fold, get_mean_roc_curve
from src.metrics import get_nested_feature_importance, plot_feature_importance
from src.utils import pkl_load, pkl_dump
from src.baselines import plot_baseline_roc, plot_related_baseline_roc

from joblib import Parallel, delayed
from functools import partial
import multiprocessing
import itertools
from copy import deepcopy
from src.models import FFNetWrapper, FFN, LinearBlock
from src.nn_train_eval import nested_kcv_train_nn

mkdirs('../output/nn_test/')
training_kwargs = dict(n_epochs=300, early_stopping=False, patience=100, delta=1e-5, filename='../output/nn_test/checkpoint_test',
                       verbosity=1)
encoding_kwargs = dict(max_len=12, encoding='onehot', blosum_matrix=None, standardize=True, seq_col='Peptide',
                       hla_col='HLA', target_col='agg_label', rank_col='EL_rank_mut', mut_col=['dissimilarity_score'],
                       adaptive=False, mask=False, add_rank=True, add_aaprop=False, remove_pep=False)
ics_kl = pkl_load('../data/ic_dicts/ics_kl.pkl')
train_dataset = pd.read_csv('../data/mutant/221028_cedar_related_newcore_fold.csv')
eval_dataset = pd.read_csv('../data/mutant/221119_prime_related_10fold.csv')

ics_dict = ics_kl
model = FFNetWrapper(n_in=22, n_hidden=30, dropout=0)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.BCELoss()
device = 'cpu'
n_jobs = 8

models_dict, train_metrics, test_metrics = nested_kcv_train_nn(train_dataset, model, optimizer, criterion, device,
                                                               ics_dict, encoding_kwargs, training_kwargs, n_jobs)
