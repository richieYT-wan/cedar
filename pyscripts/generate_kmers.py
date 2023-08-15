"""
Reads a uniprot fasta file, export all the KMers to a dataframe.
"""

import os, sys
import argparse
import tqdm
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.sequence_handling import get_fasta_kmers
from src.utils import str2bool, mkdirs, convert_path


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
    # handling Windows pathing (\ to /)
    args['outdir'], args['filepath'] = convert_path(args['outdir']), convert_path(args['filepath'])
    mkdirs(args['outdir'])
    fn = args['filepath']
    # If provided filepath is a directory, will take all fasta files inside
    if os.path.isdir(fn):
        files = sorted([os.path.join(fn, x) for x in os.listdir(fn) if '.fa' in x])
    # Else if it's a direct filepath, will try to read it
    elif os.path.isfile(fn):
        if '.fa' not in fn:
            raise ValueError(f'{fn} file provided is not in a fasta format.')
        files = sorted([fn])

    for i, f in enumerate(tqdm.tqdm(files, desc = 'File', leave=True)):
        output_kmers = get_fasta_kmers(f, args['k'], args['description_verbose'], args['drop_sequence'])
        outname = f'{args["k"]}mers_{f[f.rfind("/") + 1:f.find(".fasta")]}.txt'
        # Keep all headers and simply remove them when concatenating in remove dupe...
        output_kmers.to_csv(os.path.join(args['outdir'], outname), index=False, header=True)


if __name__ == '__main__':
    main()
