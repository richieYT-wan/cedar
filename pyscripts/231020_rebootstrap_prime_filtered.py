import pandas as pd
import numpy as np
import multiprocessing
import itertools
from functools import partial
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from datetime import datetime as dt
import os, sys
import copy
import tracemalloc

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Custom fct imports
import argparse
from src.utils import mkdirs, convert_path
from src.bootstrap import get_pval_wrapper, bootstrap_eval

N_CORES = 39


def get_cdt(basename):
    print(basename)
    basename = basename.split('onehot_')[1].split('.csv')[0]
    weight, key = basename.split('_icore_mut_')
    
    return weight, key


def get_in_ref(peptide, hla, ref):
    return len(ref.query('Peptide==@peptide and HLA == @hla')) == 1


def bootstrap_pipeline(filename, args, ref):
    basename = os.path.basename(filename)
    weight, key = get_cdt(basename)
    if os.path.exists(f'{args["outdir"]}PRIME_bootstrapped_df_cpn_merged_onehot_{weight}_icore_mut_{key}.csv'):
        return None
    df = pd.read_csv(filename)
    df['in_nepdb'] = df.apply(lambda x: get_in_ref(x['Peptide'], x['HLA'], ref=ref), axis=1)
    df = df.query('not in_nepdb')
    pcol = 'pred' if 'pred' in df.columns else 'mean_pred'
    bdf = bootstrap_eval(df[pcol].values, df['agg_label'].values, 10000, 40, True, False, True) \
        .assign(evalset='PRIME', input_type='icore_mut', weight=weight, key=key)
    bdf.to_csv(f'{args["outdir"]}PRIME_bootstrapped_df_cpn_merged_onehot_{weight}_icore_mut_{key}.csv')
    return None


def compare_baseline(df, baseline):
    evalset = df.evalset.unique()[0]
    b = baseline.query('evalset==@evalset')
    if len(b) == 0:
        pval_icore = 1
        pval_pep = 1
    else:
        baseline_icore = b.query('input_type=="icore_mut"')
        baseline_pep = b.query('input_type=="Peptide"')
        pval_icore, sig_icore = get_pval_wrapper(df, baseline_icore, 'auc')
        pval_pep, sig_pep = get_pval_wrapper(df, baseline_pep, 'auc')
    gb = df.groupby(['weight', 'key']).agg(mean_auc=('auc', 'mean'), mean_auc01=('auc_01', 'mean'))
    gb['pval_icore'] = pval_icore
    gb['pval_pep'] = pval_pep
    gb.columns = [f'{x}_{evalset}' for x in gb.columns]
    return gb


def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to crossvalidate and evaluate methods that use aa frequency as encoding')

    parser.add_argument('-datadir', type=str,
                        default='/home/projects/vaccine/people/yatwan/cedar/output/230427_MutExpr_Final_input_type/prime_filtered/',
                        help='Path to directory containing the results')
    parser.add_argument('-savename', type=str, default='recompare_harmonic')
    parser.add_argument('-outdir', type=str,
                        default='/home/projects/vaccine/people/yatwan/cedar/output/230427_MutExpr_Final_input_type/cp_filtered_outdir/')
    parser.add_argument('-ncores', type=int, default=40,
                        help='N cores to use in parallel, by default will be multiprocesing.cpu_count() * 3/4')
    return parser.parse_args()


def main():
    tracemalloc.start()
    start = dt.now()
    args = vars(args_parser())
    args['outdir'], args['datadir'] = convert_path(args['outdir']), convert_path(args['datadir'])
    print('Making dirs')
    print('Sanity check')
    mkdirs(args['outdir'])
    baselines = os.path.join(args['datadir'], 'baselines/')
    baseline_df = pd.concat([pd.read_csv(f'{baselines}{x}') for x in os.listdir(baselines)])
    nepdb = pd.read_csv(f'{args["datadir"]}NEPDB_preds_cpn_merged_onehot_None_icore_mut_only_rank.csv')
    files = [f'{args["datadir"]}{x}' for x in os.listdir(args["datadir"]) if x.endswith('.csv') and 'PRIME' in x]
    # This will be a list of lists initially, then concat along axis = 1 for each evalset
    print('Doing BDFs')
    # bs_wrapper = partial(bootstrap_pipeline, args=args, ref=nepdb)
    # output = Parallel(n_jobs=4)(
    #     delayed(bs_wrapper)(filename=x) for x in
    #     tqdm(files, desc='files', position=1, leave=False))
    for f in reversed(files):
        bootstrap_pipeline(f, args, nepdb)

    print('Doing baseline comparisons')
    baseline_wrapper = partial(compare_baseline, baseline=baseline_df)
    bdf_files = [x for x in os.listdir(args["outdir"]) if x.endswith('.csv')]
    res_list = []
    for evalset in tqdm(['KCV','CEDAR', 'PRIME', 'NEPDB'], desc='evalset', position=0, leave=True):
        filtered_files = list(filter(lambda x: evalset in x and x.startswith(evalset), bdf_files))
        if len(list(filtered_files)) == 0: continue
        output = Parallel(n_jobs=args['ncores'])(
            delayed(baseline_wrapper)(df=pd.read_csv(f'{args["outdir"]}{x}')) for x in
            tqdm(filtered_files, desc='files', position=1, leave=False))
        res_list.append(pd.concat(output, axis=0))
    print(len(res_list[0]))
    print(len(res_list[1]))
    pd.concat(res_list, axis=1).to_csv(f'{args["outdir"]}../{args["savename"]}_gb_results.csv')
    print(len(pd.concat(res_list)))

    end = dt.now()
    elapsed = divmod((end - start).seconds, 60)
    print(f'Elapsed: {elapsed[0]} minutes, {elapsed[1]} seconds. ; Memory used: {tracemalloc.get_traced_memory()}')
    tracemalloc.stop()


if __name__ == '__main__':
    main()
