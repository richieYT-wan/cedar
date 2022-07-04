import pandas as pd
import numpy as np
import torch
import multiprocessing
import math

#### ==== CONST (blosum, multiprocessing, keys, etc) ==== ####
VAL = math.floor(4 + (multiprocessing.cpu_count() / 1.5))
N_CORES = VAL if VAL <= multiprocessing.cpu_count() else int(multiprocessing.cpu_count() - 2)

DATADIR = '../data/Matrices/'
AA_KEYS = [x for x in 'ARNDCQEGHILKMFPSTWYV']

CHAR_TO_INT = dict((c, i) for i, c in enumerate(AA_KEYS))
INT_TO_CHAR = dict((i, c) for i, c in enumerate(AA_KEYS))

BG = np.loadtxt(f'{DATADIR}bg.freq.fmt', dtype=float)
BG = dict((k, v) for k, v in zip(AA_KEYS, BG))
# BLOSUMS 62 FREQS
_blosum62 = np.loadtxt(f'{DATADIR}blosum62.freq_rownorm', dtype=float).T
BL62FREQ = {}
for i, letter_1 in enumerate(AA_KEYS):
    BL62FREQ[letter_1] = {}
    for j, letter_2 in enumerate(AA_KEYS):
        BL62FREQ[letter_1][letter_2] = _blosum62[i, j]
# BLOSUMS 50 
BL50 = {}
_blosum50 = np.loadtxt(f'{DATADIR}BLOSUM50', dtype=float).T
for i, letter_1 in enumerate(AA_KEYS):
    BL50[letter_1] = {}
    for j, letter_2 in enumerate(AA_KEYS):
        BL50[letter_1][letter_2] = _blosum50[i, j]
# BLOSUMS 62
BL62 = pd.read_csv(f'{DATADIR}BLOSUM62', sep='\s+', comment='#', index_col=0).to_dict()


#### ==== SEQUENCES ENCODING ==== ####

def encode(sequence, max_len=None, how='onehot', blosum_matrix=None):
    """
    encodes a single peptide into a matrix, using 'onehot' or 'blosum'
    if 'blosum', then need to provide the blosum dictionary as argument
    """

    # One hot encode by setting 1 to positions where amino acid is present, 0 elsewhere
    size = len(sequence)
    if how == 'onehot':
        int_encoded = [CHAR_TO_INT[char] for char in sequence]
        onehot_encoded = list()
        for value in int_encoded:
            letter = [0 for _ in range(len(AA_KEYS))]
            letter[value] = 1
            onehot_encoded.append(letter)
        tmp = np.array(onehot_encoded)
    # BLOSUM encode
    if how == 'blosum':
        if blosum_matrix is None or not isinstance(blosum_matrix, dict):
            raise Exception('No BLOSUM matrix provided!')

        tmp = np.zeros([size, len(AA_KEYS)], dtype=np.float32)
        for idx in range(size):
            tmp[idx, :] = blosum_matrix[sequence[idx]]

    # Paddding if max_len is provided
    if max_len is not None and max_len > size:
        diff = max_len - size
        tmp = np.concatenate([tmp, np.zeros([diff, len(AA_KEYS)], dtype=np.float32)],
                             axis=0)
    return torch.from_numpy(tmp).float()


def encode_batch(sequences, max_len=None, how='onehot', blosum_matrix=None):
    """
    Encode multiple sequences at once.
    """
    if max_len is None:
        max_len = max([len(x) for x in sequences])

    return torch.stack([encode(seq, max_len, how, blosum_matrix) for seq in sequences])


def onehot_decode(onehot_sequence):
    return ''.join([INT_TO_CHAR[k.item()] for k in onehot_sequence.argmax(axis=1)])


def onehot_batch_decode(onehot_sequences):
    return np.stack([onehot_decode(x) for x in onehot_sequences])


def standardize(x_train, x_eval, x_test=None):
    """
    Sets mean to 0 and variance to 1 wrt the mean/variance of the training set.
    """
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    x_train_std = (x_train - mu) / sigma
    x_eval_std = (x_eval - mu) / sigma
    if x_test is not None:
        x_test_std = (x_test - mu) / sigma
        return x_train_std, x_eval_std, x_test_std
    else:
        return x_train_std, x_eval_std, mu, sigma


#### ==== PSSM, pfm etc ==== ####

def compute_pfm(sequences):
    """
    Computes the position frequency matrix given a list of sequences
    """
    max_len = max([len(x) for x in sequences])
    N = len(sequences)
    onehot_seqs = encode_batch(sequences, max_len, how='onehot', blosum_matrix=None).numpy()
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
    ic = 1 + np.sum(row * row_log20)
    return ic


def compute_ic(sequences):
    """
    returns the IC for sequences of a given length based on the frequency matrix
    Args:
        sequences (list) : list of strings (sequences) from which to compute the IC
    Returns:
        ic_array (np.ndarray) : A Numpy array of the information content at each position (of shape max([len(seq) for seq in sequences]), 1)
    """
    # if type(sequences) == np.ndarray:
    #     return np.array([compute_ic_position(sequences, pos) for pos in range(sequences.shape[0])])
    pfm = compute_pfm(sequences)
    ic_array = np.array([compute_ic_position(pfm, pos) for pos in range(pfm.shape[0])])
    return ic_array
    
    
def get_mia(ic_array, threshold=0.3):
    return np.where(ic_array < threshold)[0]
