from datetime import datetime as dt
import os, sys
import copy
import pandas as pd
import numpy as np
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils import pkl_load
from src.sklearn_train_eval import evaluate_trained_models_sklearn
from mutation_tools import pipeline_mutation_scores

unpickle = pkl_load('/path/to/model.pkl')
models, kwargs, ics = unpickle['model'], unpickle['kwargs'], unpickle['ics']

data = pd.read_csv('/path/to/data', sep=' ')
pepx = pd.read_csv('/path/to/pepx')
merged=pd.merge(data, pepx.rename(columns={'peptide':'icore_wt_aligned'}), how='left', left_on='icore_wt_aligned', right_on='icore_wt_aligned')
merged.fillna(merged.median(skipna=True, numeric_only=True), inplace=True)
merged = pipeline_mutation_scores(merged, 'icore_mut','icore_wt_aligned', ics, threshold=0.2, prefix='icore_')

_, predictions = evaluate_trained_models_sklearn(merged, models, ics, encoding_kwargs=kwargs, n_jobs=N_CORES)
predictions.to_csv('Peptide', ascending=False).to_csv('/path/to/output', index=False)

