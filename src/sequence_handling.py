import pandas as pd
from Bio import SeqIO
from joblib import Parallel, delayed
import numpy as np
from functools import partial
import multiprocessing
import math

VAL = math.floor(4 + (multiprocessing.cpu_count() / 1.5))
N_CORES = VAL if VAL <= multiprocessing.cpu_count() else int(multiprocessing.cpu_count() - 2)

AA_KEYS = [x for x in 'ARNDCQEGHILKMFPSTWYV']

CHAR_TO_INT = dict((c,i) for i,c in enumerate(AA_KEYS))
INT_TO_CHAR = dict((i,c) for i,c in enumerate(AA_KEYS))

CHAR_TO_INT = dict((c,i) for i,c in enumerate(AA_KEYS))
INT_TO_CHAR = dict((i,c) for i,c in enumerate(AA_KEYS))


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
        return [{'Peptide': seq[i:i + k],
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
    total_index = tmp.loc[tmp['Peptide'].duplicated(keep=False)].index
    first_index = total_index.difference(tmp.loc[tmp.duplicated('Peptide', keep='first')].index)
    # Merge ID for total, set_index as first_index
    merged_tmp = tmp.loc[total_index].groupby('Peptide')['id_position'].agg(','.join) \
        .reset_index().set_index(first_index)  # resets Peptides from index and set as first_index for querying
    tmp.drop_duplicates('Peptide', keep='first', inplace=True)
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


def onehot_encode(sequence):
    int_encoded = [CHAR_TO_INT[char] for char in sequence]
    onehot_encoded = list()
    for value in int_encoded:
        letter = [0 for _ in range(len(AA_KEYS))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return np.array(onehot_encoded)


def onehot_decode(onehot_sequence):
    return ''.join([INT_TO_CHAR[k.item()] for k in onehot_sequence.argmax(axis=1)])


def onehot_batch_encode(sequences):
    return np.stack([onehot_encode(x) for x in sequences])


def onehot_batch_decode(onehot_sequences):
    return np.stack([onehot_decode(x) for x in onehot_sequences])


def compute_pfm(sequences):
    """
    Computes the position frequency matrix given a list of sequences
    """
    N = len(sequences)
    onehot_seqs = onehot_batch_encode(sequences)
    return onehot_seqs.sum(axis=0) / N


def compute_ic_position(matrix, position):
    """
    Computes the information content at a given position for a given position-frequency matrix
    :param matrix:
    :param position:
    :return:
    """
    row = matrix[position]
    row_log20 = np.nan_to_num(np.log(row) / np.log(20), neginf=0)
    IC = 1 + np.sum(row * row_log20)
    return IC


def compute_ic_sequence(matrix):
    """
    returns the IC for sequences of a given length based on the frequency matrix
    """
    return np.array([compute_ic_position(matrix, pos) for pos in range(matrix.shape[0])])


def get_mia(IC_array, threshold=0.3):
    return np.where(IC_array<threshold)[0]