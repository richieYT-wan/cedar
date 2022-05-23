import modin.pandas as pd
from Bio import SeqIO
from joblib import Parallel, delayed
from functools import partial
import multiprocessing
import math

VAL = math.floor(4 + (multiprocessing.cpu_count() / 1.5))
N_CORES = VAL if VAL <= multiprocessing.cpu_count() else int(multiprocessing.cpu_count() - 2)


def kmerize(seq: str, k: int):
    if len(seq) < k:
        raise Exception(f'Provided K {k} is shorter than sequence length {len(seq)}')
    else:
        return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def kmerize_dict(seq: str, k: int = 9):
    """
    Same as kmerize but returns a dict with start position included
    """
    if type(k) != int:
        k = int(k)

    if len(seq) < k:
        raise Exception(f'Provided K {k} is shorter than sequence length {len(seq)}')
    else:
        return [{'peptide': seq[i:i + k],
                 'start_pos': i} for i in range(len(seq) - k + 1)]


def extract_fasta_sequence(fasta_seq, verbose=True):
    """
    IMPORTANT: FASTA SHOULD BE IN UNIPROT FORMAT
    TODO: handle other format (like GCRh38_protein.faa)
    """
    desc = fasta_seq.description
    output = {'sequence': str(fasta_seq.seq),
              'uniprot_id': desc[desc.find('|') + 1:desc.rfind('|')]}
    if verbose:
        output.update({'db': desc[:desc.find('|')],
                       'name': desc[desc.find('_') + 1:desc.find(' OS=')],
                       'species': desc[desc.find('OS=') + 3:desc.find(' OX=')],
                       'species_id': desc[desc.find('OX=') + 3:desc.find(' GN=')],
                       'gene_name': desc[desc.find('GN=') + 3:desc.find(' PE=')],
                       'PE': desc[desc.find('PE=') + 3:desc.find(' SV=')],
                       'SV': desc[desc.find('SV=') + 3:]})
    return output


def read_fasta(fn, min_len: int = None, description_verbose=True):
    """
    Reads a uniprot formatted fasta file and returns a dataframe containing
    the sequences and related informations
    """
    sequences = SeqIO.parse(open(fn), 'fasta')
    lst = []
    if min_len:
        sequences = [seq for seq in sequences if len(seq.seq) >= min_len]
    extract_fasta_sequences_ = partial(extract_fasta_sequence, verbose=description_verbose)
    output = Parallel(n_jobs=N_CORES)(delayed(extract_fasta_sequences_)(seq) for seq in sequences)
    return pd.DataFrame(output)


def get_sequence_kmers(parsed_sequence, k: int = 9, description_verbose=False, drop_sequence=True):
    """
    For a single parsed Fasta sequence, get the kmers of length {k} into a dataframe
    """
    # if len above K, then do kmerize
    extracted_seq = extract_fasta_sequence(parsed_sequence, description_verbose)
    if len(extracted_seq['sequence']) >= k:
        kmers = kmerize_dict(extracted_seq['sequence'], k)
        if drop_sequence:
            extracted_seq.pop('sequence')
        data = [kmer | extracted_seq for kmer in kmers]
    elif len(extracted_seq['sequence']) < k:
        pass
    return pd.DataFrame(data)


def remove_dupe_kmers(df):
    """
    From a sequence Kmers dataframe, remove the duplicates and merge-keep the ID+start position
    """
    # Copy & get merged ID+Start position
    tmp = df.copy()
    tmp['id_position'] = tmp['uniprot_id'] + '_' + tmp['start_pos'].astype(str)
    # Find all duplicates and first duplicates index
    total_index = tmp.loc[tmp['peptide'].duplicated(keep=False)].index
    first_index = total_index.difference(tmp.loc[tmp.duplicated('peptide', keep='first')].index)
    # Merge ID for total, set_index as first_index
    merged_tmp = tmp.loc[total_index].groupby('peptide')['id_position'].agg(','.join) \
        .reset_index().set_index(first_index)  # resets peptides from index and set as first_index for querying
    tmp.drop_duplicates('peptide', keep='first', inplace=True)
    tmp.loc[first_index, 'id_position'] = merged_tmp['id_position']
    return tmp


def get_fasta_kmers(fn, k: int = 9, description_verbose=False, drop_sequence=True):
    """
    Parallelized code to read sequences and extract all the kmers of length K
    """
    sequences = [s for s in SeqIO.parse(open(fn), 'fasta') if len(s.seq) >= k]
    get_sequence_kmers_ = partial(get_sequence_kmers, k=k, description_verbose=description_verbose,
                                  drop_sequence=drop_sequence)
    output = Parallel(n_jobs=N_CORES)(delayed(get_sequence_kmers_)(seq) for seq in sequences)
    return remove_dupe_kmers(pd.concat(output, ignore_index=True)).reset_index(drop=True)
