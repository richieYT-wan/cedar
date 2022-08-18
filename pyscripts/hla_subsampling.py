import pandas as pd
import random
import os, sys
import argparse
from tqdm import tqdm
from datetime import datetime as dt
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils import mkdirs, convert_path, str2bool
from joblib import Parallel, delayed
from functools import partial
from tqdm.auto import tqdm
# To be used for Ks
N_CORES_K = 5
# To be used for HLAs
N_CORES_HLA = 3
# Computed from the datasize of cedar 220701 dump after filtering
LEN_WEIGHTS = {8: 0.05718390804597701,
               9: 0.475,
               10: 0.2775862068965517,
               11: 0.1896551724137931,
               12: 0.0005747126436781609}


def args_parser():
    parser = argparse.ArgumentParser(description='K-mers extraction script')
    parser.add_argument('-datadir', type=str, default='../output_xls/filtered_rank20/', help='path to txt files')
    parser.add_argument('-hla', type=str, default='all',
                        help='for which HLA to subsample and group. \
                        Default is "all" which will list and subsample for all HLAs present \
                        in the data directory. Else, format should be like in the txt filenames.')
    parser.add_argument('-outdir', type=str, default='../output_xls/subsampled/')
    parser.add_argument('-n', type=int, default=100, help='Subsampling N sequences from each file')
    parser.add_argument('-seed', type=int, default=13, help='Seed for the random sampling')
    parser.add_argument('-conserved', type=str2bool, default=False,
                        help='Whether to randomly sample or sample conserved peptides (default False)')
    parser.add_argument('-conserved_threshold', type=int, default=2,
                        help='How many occurrences in human proteome file to define a peptide as "conserved"')
    parser.add_argument('-rank_weighting', dest='rw', type=str2bool, default=False,
                        help='Whether to use weighting to try and follow a HLA rank distribution.')
    parser.add_argument('-rank_range', nargs='+', default = [0, 0.5], help='Rank-range (inclusive) in which to select peptide. [0, 0.5] by default')
    return parser.parse_args()


def weight_peps(k, hla, args):
    if len([x for x in os.listdir(args['datadir']) if hla in x and f'{k}mer' in x]) == 0:
        return pd.DataFrame()

    tmp = pd.concat([pd.read_csv(os.path.join(args['datadir'],x), sep='\t', skiprows=1) \
                     for x in os.listdir(args['datadir']) if hla in x and f'{k}mer' in x])
    tmp['HLA'] = hla
    lower = float(args['rank_range'][0])
    upper = float(args['rank_range'][1])
    tmp = tmp.query("EL_Rank>=@lower and EL_Rank<=@upper")
    # Here first does a downsampling wrt. len proportions
    if args['rw']:
        tmp['wt'] = 1 / tmp['EL_Rank']
        return tmp.sample(int(LEN_WEIGHTS[k] * len(tmp)), weights='wt', random_state=args['seed'])
    else:
        tmp['wt'] = 1
        return tmp.sample(int(LEN_WEIGHTS[k] * len(tmp)), random_state=args['seed'])


def sample_peps(hla, args):
    """
    to parallelize sampling for a given hla
    :param hla:
    :return:
    """

    weight_peps_ = partial(weight_peps, hla=hla, args=args)
    output = Parallel(n_jobs=N_CORES_K)(delayed(weight_peps_)(k) for k in [8, 9, 10, 11, 12])
    return pd.concat(output).sample(n=args['n'], weights='wt', random_state=args['seed'])


def main():
    args = vars(args_parser())
    args['outdir'], args['datadir'] = convert_path(args['outdir']), convert_path(args['datadir'])
    assert len(args['rank_range'])==2, f'Rank range does not contain 2 numbers! {args["rank_range"]} with numbers of type {type(args["rank_range"][0])}.'
    mkdirs(args['outdir'])
    hlas = sorted(set(hla.strip('.txt') for z in os.listdir(args['datadir']) \
                      for i, hla in enumerate(z.split('_')) if "HLA" in hla)) if args['hla'] == 'all' else [args['hla']]
    if args['conserved']:
        n = args['conserved_threshold']
        ref_peps = {}
        print('Reading reference peptides')
        for k in tqdm(LEN_WEIGHTS.keys()):
            tmp = pd.read_csv(f'../output/whole_proteome/{k}mers_human_proteome.txt', usecols=['Peptide', 'count'])
            ref_peps[k] = tmp.query('count>@n').Peptide.values
    add_name = f'_cons_{args["conserved"]}_{n}' if args['conserved'] else ''

    sample_peps_ = partial(sample_peps, args=args)
    df_results = Parallel(n_jobs=N_CORES_HLA)(delayed(sample_peps_)(hla) for hla in tqdm(hlas))
    df_results = pd.concat(df_results, ignore_index=True)
    now = dt.now().microsecond
    id = str(now)+str(int(random.random()*1e5))
    fname = f"humanprot_sub_N{args['n']}_seed{args['seed']}{add_name}_{id}_scored.txt"
    df_results['len'] = df_results.Peptide.apply(len)
    df_results['agg_label'] = 0
    df_results['percentage_pos'] = 0
    df_results['dataset'] = 'hp'
    df_results['total_count'] = int(args['conserved_threshold'])
    df_results.to_csv(os.path.join(args['outdir'], fname), index=False)


if __name__ == '__main__':
    main()
