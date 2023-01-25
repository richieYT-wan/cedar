import argparse
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import pandas as pd

from src.utils import convert_path, convert_hla, mkdirs
from src.data_processing import AA_KEYS, HLAS


def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to process the data into something useable by the tool\nInput format should be a tab separated dataframe, with no header : Peptide\tWild-type\tHLA\tLabel(optional)')

    parser.add_argument('-input_path', type=str, help='abs or relative path to the input file')
    parser.add_argument('-outdir', type=str, default='../output/test/',
                        help='Directory where intermediate and final files will be saved')
    return parser.parse_args()


def main():
    args = vars(args_parser())
    args['outdir'], args['input_path'], = convert_path(args['outdir']), convert_path(args['input_path'])
    mkdirs(args['outdir'])
    fn = os.path.basename(args['input_path'])
    # Reading and filtering
    df = pd.read_csv(f'{args["input_path"]}', sep='\t', header=None)

    assert len(df.columns) in [3, 4], f'Wrong number of columns passed. Expected 3 or 4, got {len(df.columns)}'
    df.columns = ['Peptide', 'wild_type', 'HLA'] if len(df.columns) == 3 else ['Peptide', 'wild_type', 'HLA',
                                                                               'agg_label']
    df['mut_len'] = df['Peptide'].apply(len)
    df['wt_len'] = df['wild_type'].apply(len)
    df = df.query('(mut_len>=8 and mut_len<=12) and (wt_len>=8 and wt_len<=12)')
    # Filtering datapoints where Mut or WT aren't in AA_keys
    # Checks if any seq not in alphabet
    df = df.drop(df.loc[df['Peptide'].apply(lambda x: any([z not in AA_KEYS for z in x]))].index).reset_index(drop=True)
    df = df.drop(df.loc[df['wild_type'].apply(lambda x: any([z not in AA_KEYS for z in x]))].index).reset_index(
        drop=True)

    # filter by HLAs
    df['HLA'] = df['HLA'].apply(convert_hla)
    df = df.query(f'HLA in @HLAS')
    # Sort values
    df.sort_values('Peptide', ascending=True, inplace=True)

    df['seq_id'] = [f'seq_{i}' for i in range(1, len(df) + 1)]

    # For NetMHCpan (core-shift)
    print(f'Saving files for NetMHCpan (core-shift) at {args["outdir"]}{fn}_mut.fa and {args["outdir"]}{fn}_wt.fa')
    with open(f'{args["outdir"]}{fn}_mut.fa', 'w') as f, \
            open(f'{args["outdir"]}{fn}_wt.fa', 'w') as g:
        for i, row in df.iterrows():
            f.write(f'>{row["seq_id"]}\n')
            f.write(f'{row["Peptide"]}\n')
            g.write(f'>{row["seq_id"]}\n')
            g.write(f'{row["wild_type"]}\n')
    # For NetMHCpan ("trueHLA_EL_rank")
    print(f'Saving files for NetMHCpan (""true"") at {args["outdir"]}{fn}_mut.pep')
    df[['Peptide', 'wild_type']].to_csv(f'{args["outdir"]}{fn}_mut.pep', index=False, header=False, sep='\t')

    print(f'Saving files for dissimilarity_score at {args["outdir"]}{fn}_mut_wt.pep')
    # For Kern_dist
    df[['Peptide', 'wild_type']].to_csv(f'{args["outdir"]}{fn}_mut_wt.pep', index=False, header=False, sep='\t')

    df.to_csv(f'{args["outdir"]}{fn}_full_df.csv',index=False)
    sys.exit(0)


if __name__ == '__main__':
    main()
