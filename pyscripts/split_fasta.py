from Bio import SeqIO
import argparse
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils import mkdirs, convert_path


def args_parser():
    parser = argparse.ArgumentParser(description='Splits a single fasta file into smaller files'
                                                 'containing N sequences each or fewer for the last file\n'
                                                 'Example: Splitting a file of length 1000 sequences with N = 100 will result in 10 files')
    parser.add_argument('-filepath', type=str, help='path to fasta file')
    parser.add_argument('-outdir', type=str, help='output directory')
    parser.add_argument('-n', type=int, help='N sequences to get per file')
    return parser.parse_args()


def main():
    args = vars(args_parser())
    # handling Windows pathing (\ to /)
    args['outdir'], args['filepath'] = convert_path(args['outdir']), convert_path(args['filepath'])
    mkdirs(args['outdir'])
    fn = args['filepath']
    data_dict = SeqIO.to_dict(SeqIO.parse(fn, 'fasta'))
    keys = list(data_dict.keys())
    split_keys = [keys[i:i + args['n']] for i in range(0, len(keys), args['n'])]
    sample_name = fn[fn.rfind("/") + 1:fn.find(".fasta")]
    for i, keys in enumerate(split_keys):
        out_name = os.path.join(args['outdir'], f'{sample_name}_split_{i * args["n"]}.fasta')
        split_dict = {k: data_dict[k] for k in keys}
        with open(out_name, 'w') as out_name:
            SeqIO.write(split_dict.values(), out_name, 'fasta')


if __name__ == '__main__':
    main()
