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

from src.sequence_handling import get_fasta_kmers
from src.utils import str2bool, mkdirs, convert_path, parse_netmhcpan_header, filter_rank, get_filtered_df


def args_parser():
    parser = argparse.ArgumentParser(description='K-mers extraction script')
    parser.add_argument('-filepath', type=str, help='path to pep file')
    parser.add_argument('-resultspath', type=str, help='path to the NetMHCpan output (.xls format)')
    parser.add_argument('-outdir', type=str, default='./output/')
    parser.add_argument('-rank_filter', type=str, default='el_rank',
                        help='Which rank to filter by; Takes value (BA_Rank or EL_Rank), '
                             'case insensitive')
    parser.add_argument('-rank_thr', type=float, default=2.0, help='Threshold for rank filtering')
    return parser.parse_args()


def main():
    # set args
    args = vars(args_parser())
    args['outdir'], args['filepath'] = convert_path(args['outdir']), convert_path(args['filepath'])
    mkdirs(args['outdir'])
    fn_pep = args['filepath']
    fn_mhc = args['resultspath']
    savename = f"{args['filepath'][args['filepath'].rfind('/') + 1:args['filepath'].find('.pep')]}_scored.csv"
    # read csvs and parse columns/index
    peptides = pd.read_csv(fn_pep, header=None)
    peptides['columns'] = 'Peptide'
    df_netmhcpan = pd.read_csv(fn_mhc, header=[0, 1], sep='\t')
    df_netmhcpan.columns = parse_netmhcpan_header(df_netmhcpan.columns)
    df_netmhcpan.set_index(('base', 'Peptide'), inplace=True)
    df_netmhcpan.index.name = 'Peptide'
    # Get the df containing the sequence and its "best binding" HLA
    df_output = filter_rank(df_netmhcpan, args['rank_filter'])
    # Get the final output DF merging the best HLA with its features (BA, EL score and ranks)
    df_output = get_filtered_df(df_output, df_netmhcpan)
    print('xd all gud')
    print(os.path.join(args['outdir'], savename))
    df_output.to_csv(os.path.join(args['outdir'], savename))


if __name__ == '__main__':
    main()
