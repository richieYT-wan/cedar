import os, sys
import pandas as pd
import numpy as np
import torch
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils import pkl_load, pkl_dump

# THIS IS A LIST
train_metrics = pkl_load('/home/projects/vaccine/people/yatwan/cedar/output/train_eval/run_220826_13h45m05s_nn/train_metrics_nn.pkl')
# Re-make the list, dropping all epochs except the first and last...
new_train_metrics = []

for x in train_metrics:
    # tmp = {}
    for fold_out, dict_fold_out in x.items():
        # tmp[fold_out] =
        for fold_in, dict_fold_in in dict_fold_out.items():
            for partition, results in dict_fold_in.items():
                if partition=='mu' or partition=='sigma':
                    continue
                for metric, values in results.items():
                    if metric == 'losses': continue
                    results[metric] = ([values[0], values[-1]], len(values))

pkl_dump(train_metrics,
         '/home/projects/vaccine/people/yatwan/cedar/output/train_eval/run_220826_13h45m05s_nn/train_metrics_nn_new.pkl')


