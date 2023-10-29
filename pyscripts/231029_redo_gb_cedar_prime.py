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
    basename = basename.split('onehot_')[1].split('.csv')[0]
    weight, key = basename.split('_icore_mut_')
    return weight, key


def get_in_ref(peptide, hla, ref):
    return len(ref.query('Peptide==@peptide and HLA == @hla')) == 1


def compare_baseline(df, baseline):
    evalset = df.evalset.unique()[0]
    b = baseline.query('evalset==@evalset')
    if len(b)==0:
        pval_icore = 100
        pval_pep = 100
    else:
        baseline_icore = b.query('input_type=="icore_mut"')
        baseline_pep = b.query('input_type=="Peptide"')
        try:
            pval_icore, sig_icore = get_pval_wrapper(df, baseline_icore, 'auc')
        except:
            pval_icore = 100
        try:
            pval_pep, sig_pep = get_pval_wrapper(df, baseline_pep, 'auc')
        except:
            pval_pep = 100
    gb = df.groupby(['weight', 'key']).agg(mean_auc=('auc', 'mean'), mean_auc01=('auc_01', 'mean'))
    gb['pval_icore'] = pval_icore
    gb['pval_pep'] = pval_pep
    gb.columns = [f'{x}_{evalset}' for x in gb.columns]
    return gb


def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to crossvalidate and evaluate methods that use aa frequency as encoding')

    parser.add_argument('-datadir', type=str,
                        default='/home/projects/vaccine/people/yatwan/cedar/output/230427_MutExpr_Final_input_type/cp_filtered_outdir/',
                        help='Path to directory containing the results')
    parser.add_argument('-savename', type=str, default='dunno')
    parser.add_argument('-outdir', type=str,
                        default='/home/projects/vaccine/people/yatwan/cedar/output/230427_MutExpr_Final_input_type/redo_cp_gb/')
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
    files = [f'{args["datadir"]}{x}' for x in os.listdir(args["datadir"]) if x.endswith('.csv') and 'PRIME' in x]
    # This will be a list of lists initially, then concat along axis = 1 for each evalset

    print('Doing baseline comparisons')
    baseline_wrapper = partial(compare_baseline, baseline=baseline_df)
    bdf_files = [x for x in os.listdir(args["datadir"]) if x.endswith('.csv') and 'bootstrapped_df' in x]
    res_list = []
    for evalset in tqdm(['CEDAR','PRIME'], desc='evalset', position=0, leave=True):
        filtered_files = list(filter(lambda x: evalset in x and x.startswith(evalset), bdf_files))
        if len(list(filtered_files)) == 0: continue
        output = Parallel(n_jobs=args['ncores'])(
            delayed(baseline_wrapper)(df=pd.read_csv(f'{args["datadir"]}{x}')) for x in
            tqdm(filtered_files, desc='files', position=1, leave=False))
        cat = pd.concat(output, axis=0)
        res_list.append(cat)
        cat.to_csv(f'{args["outdir"]}{evalset}_gb_results.csv')

    pd.concat(res_list, axis=1).to_csv(f'{args["outdir"]}{args["savename"]}_gb_results.csv')


    end = dt.now()
    elapsed = divmod((end - start).seconds, 60)
    print(f'Elapsed: {elapsed[0]} minutes, {elapsed[1]} seconds. ; Memory used: {tracemalloc.get_traced_memory()}')
    tracemalloc.stop()


if __name__ == '__main__':
    main()
