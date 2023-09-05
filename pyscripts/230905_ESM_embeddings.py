import pandas as pd
import os, sys


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Custom fct imports
N_CORES = 39

from src.sklearn_train_eval import nested_kcv_train_sklearn, evaluate_trained_models_sklearn
from sklearn.ensemble import RandomForestClassifier
from src.data_processing import AA_KEYS
from src.metrics import get_metrics, get_nested_feature_importance
from src.bootstrap import bootstrap_eval
from tqdm.auto import tqdm

cedar_merged = pd.read_csv('../data/esm/230905_cedar_esm_pca.csv')
prime_merged = pd.read_csv('../data/esm/230905_prime_esm_pca.csv')
nepdb_merged = pd.read_csv('../data/esm/230905_nepdb_esm_pca.csv')

# Here, I can do remove pep and don't add rank
encoding_kwargs = {'max_len': 12,
                   'encoding': 'onehot',
                   'blosum_matrix': None,
                   'mask': False,
                   'threshold': 0.2,
                   'invert': False,
                   'add_rank': False,
                   'seq_col': 'icore_mut',
                   'rank_col': 'EL_rank_mut',
                   'hla_col': 'HLA',
                   'add_aaprop': False,
                   'remove_pep': True,
                   'standardize': True}


esm_cols = [x for x in cedar_merged.columns if x.startswith('dim_')]
pca_cols = [f'ESM_PCA_{i:02}' for i in range(1, 26)]
outdir = '../output/230905_ESM_embeddings/'
os.makedirs(outdir, exist_ok=True)

mega_df = []
for add_rank in [True, False]:
    for remove_pep in [True, False]:
        for mut_cols, name in zip([pca_cols, esm_cols], ['ESM_PCA', 'ESM_mean']):
            encoding_kwargs['mut_col'] = mut_cols
            encoding_kwargs['add_rank'] = add_rank
            encoding_kwargs['remove_pep'] = remove_pep
            filename = f'removePep{remove_pep}_addRank{add_rank}_{name}'

            model = RandomForestClassifier(n_jobs=1, min_samples_leaf=7, n_estimators=300, max_depth=9,
                                           ccp_alpha=9.945e-6)
            trained_models, _, _ = nested_kcv_train_sklearn(cedar_merged, model, None, encoding_kwargs, n_jobs=10)
            fi = get_nested_feature_importance(trained_models)
            fn = []
            if not remove_pep:
                fn.extend(AA_KEYS)
            if add_rank:
                fn.extend(['%Rank'])
            fn.extend(mut_cols)
            # Saving Feature importances as dataframe
            df_fi = pd.DataFrame(fi, index=fn).T
            df_fi.to_csv(f'{outdir}featimps_{filename}.csv', index=True)
            for evalset, evalname in zip([cedar_merged, prime_merged, nepdb_merged],
                                         ['CEDAR', 'PRIME', 'NEPDB']):
                # FULLY FILTERED + Mean_pred
                if evalname =='CEDAR':
                    kcv_eval = True
                else:
                    kcv_eval = False
                    evalset = evalset.query('Peptide not in @cedar_merged.Peptide.values').copy()

                # print(evalname, len(evalset), evalset.columns)
                _, preds = evaluate_trained_models_sklearn(
                    evalset.drop_duplicates(subset=['Peptide', 'HLA', 'agg_label']),
                    trained_models, None,
                    cedar_merged,
                    encoding_kwargs, concatenated=False,
                    only_concat=False, n_jobs=8, kcv_eval=kcv_eval)
                p_col = 'pred' if 'pred' in preds.columns else 'mean_pred'
                preds.to_csv(f'{outdir}/{evalname}_preds_{filename}.csv', index=False,
                             columns=['HLA', 'Peptide', 'agg_label', 'icore_mut', 'icore_wt_aligned'] + mut_cols + [
                                 p_col])
                bootstrapped_df = bootstrap_eval(preds[p_col], preds['agg_label'], n_rounds=10000, n_jobs=40,
                                                 reduced=True)
                bootstrapped_df['remove_pep'] = remove_pep
                bootstrapped_df['add_rank'] = add_rank

                bootstrapped_df.to_csv(f'{outdir}/bootstrapped_df_{evalname}_{filename}.csv', index=False)
                mega_df.append(bootstrapped_df)

pd.concat(mega_df).to_csv(f'{outdir}total_df.csv')