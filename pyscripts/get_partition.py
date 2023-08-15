from src.partition_tools import pipeline_stratified_kfold
from src.utils import str2bool
import argparse
import pandas as pd
import numpy as np
import random


def args_parser():
    parser = argparse.ArgumentParser(description='K-mers extraction script')
    parser.add_argument('-filepath', type=str, help='path to fasta-file or directory containing fastafiles')
    parser.add_argument('-outdir', type=str, default='../output/')
    parser.add_argument('-k', type=int, help='Extracts k-mers of length k')
    parser.add_argument('-description_verbose', type=str2bool, default=False,
                        help='Keep additional description information; By default, will only keep the UniProt ID')
    parser.add_argument('-drop_sequence', type=str2bool, default=True,
                        help='Remove original sequence from output dataframe')
    return parser.parse_args()


def main():
    args = vars(args_parser())
    pass

if __name__ == '__main__':
    main()