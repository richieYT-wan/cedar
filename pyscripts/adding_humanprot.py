import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import multiprocessing
from joblib import Parallel, delayed
from functools import partial
import itertools

from tqdm.auto import tqdm
from datetime import datetime as dt
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Custom fct imports
from src.utils import pkl_load, pkl_dump
import argparse
from src.data_processing import BL62_VALUES, BL62FREQ_VALUES, AA_KEYS
from src.utils import mkdirs, convert_path, str2bool
from src.metrics import get_nested_feature_importance
from src.sklearn_train_eval import nested_kcv_train_sklearn, evaluate_trained_models_sklearn

def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to crossvalidate and evaluate methods that use aa frequency as encoding')

    parser.add_argument('-datadir', type=str, default='../data/neoepi_human/',
                        help='Path to directory containing the pre-partitioned data')
    parser.add_argument('-outdir', type=str, default='../output/230426_human_featimp/')
    parser.add_argument('-icsdir', type=str, default='../data/ic_dicts/',
                        help='Path containing the pre-computed ICs dicts.')
    parser.add_argument('-ncores', type=int, default=40,
                        help='N cores to use in parallel, by default will be multiprocesing.cpu_count() * 3/4')
    return parser.parse_args()




def flatten_list(list_of_lists):
    return [x for list_ in list_of_lists for x in list_]

def parallel_npep_wrapper(npep, cedar_dataset, human_dataset, seed, ic_name, ics_dict, encoding_kwargs, args):
    p_human = 100*npep / (npep+len(cedar_dataset))
    filename = f"NeoepiHuman_Seed{seed}_{ic_name}_ProportionHuman{p_human:.2f}".replace(".","_")
    dataset = pd.concat([cedar_dataset, human_dataset.sample(npep, random_state=seed)])
    kf = KFold(n_splits=5, shuffle=True,random_state=seed)
    dataset['fold'] = 0
    for i, (train_idx, test_idx) in enumerate(
            kf.split(dataset['sequence'].values, dataset['agg_label'].values, groups=dataset['agg_label'])):
        dataset.iloc[test_idx, dataset.columns.get_loc('fold')] = i
    dataset.fold = dataset.fold.astype(int)
    model = RandomForestClassifier(n_jobs=1, min_samples_leaf=7, n_estimators=100,
                                                   max_depth=6, ccp_alpha=9.945e-6)
    # Training model and getting feature importances
    print('Training')
    trained_models, train_metrics, _ = nested_kcv_train_sklearn(dataset, model,
                                                                ics_dict=ics_dict,
                                                                encoding_kwargs=encoding_kwargs,
                                                                n_jobs=9)
    fi = get_nested_feature_importance(trained_models)
    fn = AA_KEYS + ['rank']
    # Saving Feature importances as dataframe
    df_fi = pd.DataFrame(fi, index=fn).T
    df_fi.to_csv(
        f'{args["outdir"]}raw/featimps_{filename}.csv',
        index=False)

    df_fi['Proportionhuman'] = p_human
    df_fi['Npephuman'] = npep
    df_fi['Weight'] = ic_name
    df_fi['seed'] = seed
    df_fi['Proportion human'] = df_fi.apply(lambda x: f"{x['Proportionhuman']/100:.1%}", axis=1)
    df_fi['Tryptophan (W) Feat. Importance'] = df_fi.apply(lambda x: 100*x['W'], axis=1)
    # Crossval roc_auc_score
    _, preds = evaluate_trained_models_sklearn(dataset.drop_duplicates(subset=['sequence','HLA','agg_label']),
                                                               trained_models, ics_dict,
                                                               dataset,
                                                               encoding_kwargs, concatenated=False,
                                                               only_concat=False, n_jobs=9)
    pcol = 'mean_pred' if 'mean_pred' in preds.columns else 'pred'
    auc = roc_auc_score(preds['agg_label'].values, preds[pcol])
    df_fi['kcv_auc']=auc
    # Only CEDAR AUC
    _, preds = evaluate_trained_models_sklearn(cedar_dataset.drop_duplicates(subset=['sequence','HLA','agg_label']),
                                                    trained_models, ics_dict,
                                                dataset,
                                                encoding_kwargs, concatenated=False,
                                                only_concat=False, n_jobs=9)
    pcol = 'mean_pred' if 'mean_pred' in preds.columns else 'pred'
    auc = roc_auc_score(preds['agg_label'].values, preds[pcol])
    df_fi['neoepi_auc']=auc
    df_fi.to_csv(f'{args["outdir"]}/bootstrapping/df_fi_{filename}.csv',index=False)
    return df_fi


def main():
    start = dt.now()
    args = vars(args_parser())
    args['outdir'], args['datadir'], args['icsdir'] = convert_path(args['outdir']), convert_path(
        args['datadir']), convert_path(args['icsdir'])
    print('Making dirs')
    mkdirs(args['outdir'])
    mkdirs(f'{args["outdir"]}raw/')
    mkdirs(f'{args["outdir"]}bootstrapping/')
    N_CORES = int(multiprocessing.cpu_count() * 3 / 4) + int(multiprocessing.cpu_count() * 0.05) if (
            args['ncores'] is None) else args['ncores']

    # LOADING DATA AND STUFF
    cedar_dataset = pd.read_csv(f'{args["datadir"]}cedar.csv')
    human_dataset = pd.read_csv(f'{args["datadir"]}human.csv').rename(columns={'icore_mut':'sequence'}).assign(agg_label=0)[['sequence','HLA','EL_rank_mut','agg_label']]

    ics_shannon = pkl_load(f'{args["icsdir"]}ics_shannon.pkl')


    # DEFINING KWARGS
    encoding_kwargs = dict(max_len=12, encoding='onehot', blosum_matrix=None, mask=False, add_rank=True,
                           seq_col='sequence', rank_col='EL_rank_mut', hla_col = 'HLA', target_col='agg_label',
                           add_aaprop=False, remove_pep=False, standardize=True)

    n_human = [int(x*len(cedar_dataset)) for x in np.arange(0, 10, 0.1) if x*len(cedar_dataset) <= len(human_dataset)]
    p_human = [round(100 * x / (x+len(cedar_dataset)),2) for x in n_human]
    feat_imps_df = []
    for seed in tqdm([0,1,2,3,4,5,6,7,8,10], desc='seed', leave=True):
        for ic_name, ics_dict, invert, mask in tqdm([('Inverted-Shannon', ics_shannon, True, False),
                                                     (['Mask', ics_shannon, False, True]),
                                                     (['None', None, False, False])],
                                                    desc = 'weighting', leave=True):
            encoding_kwargs['mask']=mask
            encoding_kwargs['invert']=invert
            partial_wrapper = partial(parallel_npep_wrapper, cedar_dataset=cedar_dataset, human_dataset=human_dataset, 
                                      seed=seed, ic_name=ic_name, ics_dict=ics_dict, encoding_kwargs=encoding_kwargs,args=args)
            output = Parallel(n_jobs=4)(delayed(partial_wrapper)(npep=npep) for npep in tqdm(n_human, desc='nhuman', leave=True))
            feat_imps_df.append(output)
    pd.concat(flatten_list(feat_imps_df)).to_csv(f'{args["outdir"]}/feat_imps_df.csv', index=False)


if __name__ == '__main__':
    main()
