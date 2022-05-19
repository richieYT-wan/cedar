import os
from src.sequence_handling import get_fasta_kmers
import argparse

def args_parser():
    parser = argparse.ArgumentParser(description = 'K-mers extraction script')
    parser.add_argument('-filepath', type='str', help= 'path to fasta-file or directory containing fastafiles')
    parser.add_argument('-k', type=int, help = 'Extracts k-mers of length k')
    parser.add_argument('-verbose', type=str_to_bool)



