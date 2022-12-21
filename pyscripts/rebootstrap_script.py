import pandas as pd 

import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.bootstrap import bootstrap_eval
from src.utils import pkl_dump
# Parsing and re-bootstrapping all results in a HLA specific manner
PATH = '../output/rebootstrap_copy/'
# top_hlas list taken from `221213_allele_specific_model.ipynb`
top_hlas = ['HLA-A0201', 'HLA-A1101', 'HLA-A2402', 'HLA-B0702', 'HLA-B1501', 'HLA-B3501']

concat_df = []
mean_rocs_list = []
mean_rocs_ids = []

for trainset in ['cedar', 'prime']:
    subpath = f'{PATH}trainset_{trainset}/'
    for file in os.listdir(subpath):
        preds_df = pd.read_csv(f'{subpath}{file}')
        name = file.replace('_Peptide_EL_rank_mut_only_rank.csv','').replace('preds_onehot_','')
        evalset = name.split('_')[0]
        pred_col = 'mean_pred' if 'prime' in evalset.lower() else 'pred'
        weight = name.split('_')[1]
        for hla in top_hlas+['non_top']:
            if hla != 'non_top':
                tmp = preds_df.query('HLA==@hla')
            elif hla == 'non_top':
                tmp = preds_df.query('HLA not in @top_hlas')
            scores = tmp[pred_col].values
            labels = tmp['agg_label'].values
            bootstrapped_df, mean_rocs = bootstrap_eval(scores, labels, n_rounds=10000, n_jobs = 8)
            bootstrapped_df['trainset']=trainset
            bootstrapped_df['evalset']=evalset
            bootstrapped_df['weight']=weight
            bootstrapped_df['hla']=hla
            bootstrapped_df['encoding']='onehot'
            bootstrapped_df['features']='only_rank'
            concat_df.append(bootstrapped_df)
            mean_rocs_list.append(mean_rocs)
            mean_rocs_ids.append(dict(trainset=trainset, evalset=evalset, weight=weight, hla=hla))

pd.concat(concat_df).to_csv(f"{PATH}concat_df.csv", index=False)
pkl_dump(mean_rocs_list, f'{PATH}rocs_list.pkl')
pkl_dump(mean_rocs_ids, f'{PATH}rocs_ids.pkl')

