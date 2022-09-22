import pandas as pd
import numpy as np
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Custom fct imports
from src.utils import pkl_load, pkl_dump, flatten_product
from src.utils import str2bool, mkdirs, convert_path
from src.train_eval import nested_kcv_train_tmp, evaluate_trained_models_tmp

print(os.getcwd())
cedar = pd.read_csv('../data/debug/cedar_10fold.csv')
prime = pd.read_csv('../data/debug/prime_5fold.csv')
models = pkl_load('../data/debug/models.pkl')
kwargs = pkl_load('../data/debug/kwargs.pkl')
ics = None

test_cedar, df_cedar = evaluate_trained_models_tmp(cedar, models, ics, cedar,
                                                                               kwargs, concatenated=True, only_concat=False)
print(len(df_cedar))

test_prime, df_prime = evaluate_trained_models_tmp(prime, models, ics, cedar,
                                                   kwargs, concatenated=False, only_concat=False)
print(len(df_prime))

sys.exit()
