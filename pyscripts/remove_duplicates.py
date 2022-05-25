"""
Reads all kmers txt files and dedupes, save to .txt and .pep
"""
import os, sys
import pandas as pd
import numpy as np
import argparse

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.sequence_handling import remove_dupe_kmers
from src.utils import mkdirs, convert_path, str2bool


def args_parser():
    parser = argparse.ArgumentParser(description='K-mers de-duping script')
    parser.add_argument('-filepath', type=str, help='path to merged txt file')
    parser.add_argument('-split', type=str2bool, help='whether to split the output into chunks')
    parser.add_argument('-n', type=int, help='number N chunks to get')
    return parser.parse_args()


def main():
    # set args
    args = vars(args_parser())
    args['filepath'] = convert_path(args['filepath'])
    # This should be the concatenated df
    df = pd.read_csv(args['filepath'])
    # drops the duplicated header rows
    df.drop(index=df.query('Peptide=="Peptide"').index, inplace=True)
    print('Removing duplicates')
    df = remove_dupe_kmers(df)

    if args['split']:
        print(f'Splitting into {args["n"]} chunks and saving to .txt and .pep')
        indices = np.array_split(df.index.tolist(), args['n'])
        for i, idx in enumerate(indices):
            savename = f'{args["filepath"].replace(".txt", f"_chunk_{i}")}'
            df.loc[idx].to_csv(f'{savename}.txt', index=False)
            df.loc[idx][['peptide']].to_csv(f'{savename}.pep', header=None, index=False)

    else:
        print('Saving to .txt and .pep')
        df.to_csv(args['filepath'])
        df[['peptide']].to_csv(args['filepath'].replace('.txt','.pep'), header=None, index=False)


if __name__ == '__main__':
    main()
