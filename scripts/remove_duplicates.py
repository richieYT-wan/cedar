"""
Reads all kmers txt files and dedupes, save to .txt and .pep
"""
import os, sys
import pandas as pd
import argparse

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.sequence_handling import remove_dupe_kmers
from src.utils import mkdirs, convert_path


def args_parser():
    parser = argparse.ArgumentParser(description='K-mers de-duping script')
    parser.add_argument('-filepath', type=str, help='path to merged txt file')
    return parser.parse_args()


def main():
    # set args
    args = vars(args_parser())
    args['filepath'] = convert_path(args['filepath'])
    df = pd.read_csv(args['filepath'])
    print('Removing duplicates')
    df = remove_dupe_kmers(df)
    print('Saving to .txt and .pep')
    df.to_csv(args['filepath'])
    df[['peptide']].to_csv(args['filepath'].replace('.txt','.pep'), header=None, index=False)


if __name__ == '__main__':
    main()
