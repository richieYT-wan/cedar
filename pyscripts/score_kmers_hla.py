"""
Reads two files (kmers file) and NetMHCpan score files
Keep only the ones that score above a given threshold
"""

import os, sys
import pandas as pd
import argparse
import tqdm

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils import mkdirs, convert_path, read_netmhcpan_results, set_hla


def args_parser():
    parser = argparse.ArgumentParser(description='K-mers extraction script')
    parser.add_argument('-filepath', type=str, help='path to txt file')
    parser.add_argument('-resultspath', type=str, default=None, help='path to the folder containing the NetMHCpan outputs (.xls format)')
    parser.add_argument('-outdir', type=str, default='../output/')
    parser.add_argument('-rank_filter', type=str, default='EL_Rank',
                        help='Which rank to filter by; Takes value (BA_Rank or EL_Rank), '
                             'case insensitive')
    parser.add_argument('-rank_thr', type=float, default=10.0, help='Threshold for rank filtering')
    return parser.parse_args()


def main():
    # set args & asserting errs
    args = vars(args_parser())
    assert args["rank_filter"] in ['EL_Rank', 'BA_Rank'], f'{args["rank_filter"]} should be EL_Rank or BA_rank!'
    args['outdir'], args['filepath'] = convert_path(args['outdir']), convert_path(args['filepath'])
    mkdirs(args['outdir'])
    threshold = args['rank_thr']
    print(os.listdir())
    # Reading df_pep
    fn_txt = args['filepath']
    df_pep = pd.read_csv(fn_txt)#, header=None)
    #df_pep.columns = ['Peptide']
    # listing/reading XLS
    dirname = os.path.dirname(fn_txt)+'/' if args['resultspath'] is None else args['resultspath']
    # there should be 10 xls here for a given chunk
    xls = [os.path.join(dirname,x) for x in os.listdir(dirname) if fn_txt.replace(dirname,'').replace('.txt','') in x and 'xls' in x]
    dfs = []
    # read all files, query threshold, melt&append
    for fn in xls:
        df = read_netmhcpan_results(fn)
        df = set_hla(df)
        dfs.append(df.query(f'{args["rank_filter"]}<@threshold').melt(id_vars = ['Peptide', 'HLA']))
    # Concat outputs, find best scoring (idxmin for rank), and merge to pep
    output = pd.concat(dfs)
    output = output.pivot(index=['Peptide','HLA'], columns = ['variable'], values='value')
    output[args['rank_filter']] = output[args['rank_filter']].astype(float)
    output = output.loc[output.groupby(level=0)[args['rank_filter']].idxmin()]
    # Resets the index to merge both dfs on peptide column
    output = df_pep.merge(output.reset_index(), left_on='Peptide', right_on='Peptide')
    output.to_csv(os.path.join(args['outdir'], fn_txt.replace(dirname,'').replace('.txt','scored.txt')), index=False)


if __name__ == '__main__':
    main()
