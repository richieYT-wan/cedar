import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import multiprocessing
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

    parser.add_argument('-datadir', type=str, default='../data/neoepi_viral/',
                        help='Path to directory containing the pre-partitioned data')
    parser.add_argument('-outdir', type=str, default='../output/230403_viral_featimp/')
    parser.add_argument('-icsdir', type=str, default='../data/ic_dicts/',
                        help='Path containing the pre-computed ICs dicts.')
    parser.add_argument('-ncores', type=int, default=36,
                        help='N cores to use in parallel, by default will be multiprocesing.cpu_count() * 3/4')
    return parser.parse_args()


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
    viral_dataset = pd.read_csv(f'{args["datadir"]}viral.csv')

    ics_shannon = pkl_load(f'{args["icsdir"]}ics_shannon.pkl')


    # DEFINING KWARGS
    encoding_kwargs = dict(max_len=12, encoding='onehot', blosum_matrix=None, mask=False, add_rank=True,
                           seq_col='sequence', rank_col='EL_rank_mut', hla_col = 'HLA', target_col='agg_label',
                           add_aaprop=False, remove_pep=False, standardize=True)

    n_viral = [int(x*len(cedar_dataset)) for x in np.arange(0, 10, 0.1) if x*len(cedar_dataset) <= len(viral_dataset)]
    p_viral = [round(100 * x / (x+len(cedar_dataset)),2) for x in n_viral]

    feat_imps_df = []

    for seed in tqdm(np.arange(0, 11, 1), desc='seed', position=0,leave=True):
        for ic_name, ics_dict, invert, mask in tqdm([('Inverted-Shannon', None, True, False),(['Mask', None, False, True]),(['None', None, False, False])],
                                                    desc = 'weighting', position=1, leave=True):
            encoding_kwargs['mask']=mask
            encoding_kwargs['invert']=invert
            for npep in tqdm(n_viral, desc='nviral', position=2, leave=True):
                p_viral = 100*npep / (npep+len(cedar_dataset))

                filename = f"NeoepiViral_Seed{seed}_{ic_name}_ProportionViral{p_viral:.2f}".replace(".","_")
                dataset = pd.concat([cedar_dataset, viral_dataset.sample(npep, random_state=seed)])
                kf = KFold(n_splits=5, shuffle=True, random_state=seed)
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
                                                                            n_jobs=N_CORES)
                fi = get_nested_feature_importance(trained_models)
                fn = AA_KEYS + ['rank']
                # Saving Feature importances as dataframe
                df_fi = pd.DataFrame(fi, index=fn).T
                df_fi.to_csv(
                    f'{args["outdir"]}raw/featimps_{filename}.csv',
                    index=False)

                df_fi['ProportionViral'] = p_viral
                df_fi['NpepViral'] = npep
                df_fi['Weight'] = ic_name
                df_fi['seed'] = seed

                _, preds = evaluate_trained_models_sklearn(evalset.drop_duplicates(subset=['Peptide','HLA','agg_label']),
                                                                           trained_models, ics_dict,
                                                                           train_dataset,
                                                                           encoding_kwargs, concatenated=False,
                                                                           only_concat=False)
                pcol = 'mean_pred' if 'mean_pred' in preds.columns else 'pred'
                auc = roc_auc_score(preds['agg_label'].values, pcol[pcol])
                df_fi['auc']=auc
                feat_imps_df.append(df_fi)
    pd.concat(feat_imps_df).to_csv(f'{args["outdir"]}/feat_imps_df.csv', index=False)


if __name__ == '__main__':
    main()
