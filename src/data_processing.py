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

    # Padding if max_len is provided
    if max_len is not None and max_len > size:
        diff = int(max_len) - int(size)
        try:
            # print()
            tmp = np.concatenate([tmp, np.zeros([diff, len(AA_KEYS)], dtype=np.float32)],
                                 axis=0)
        except:
            print(type(tmp), tmp.shape, len(AA_KEYS), type(diff), type(max_len), type(size), sequence)
            #     return tmp, diff, len(AA_KEYS)
            raise Exception
    return torch.from_numpy(tmp).float()


def encode_batch(sequences, max_len=None, how='onehot', blosum_matrix=None):
    """
    Encode multiple sequences at once.
    """
    if max_len is None:
        max_len = max([len(x) for x in sequences])

    return torch.stack([encode(seq, max_len, how, blosum_matrix) for seq in sequences])


def onehot_decode(onehot_sequence):
    if type(onehot_sequence) == np.ndarray:
        return ''.join([INT_TO_CHAR[x.item()] for x in onehot_sequence.nonzero()[1]])
    elif type(onehot_sequence) == torch.Tensor:
        return ''.join([INT_TO_CHAR[x.item()] for x in onehot_sequence.nonzero()[:, 1]])


def onehot_batch_decode(onehot_sequences):
    return np.stack([onehot_decode(x) for x in onehot_sequences])


def get_ic_weights(df, ics_dict:dict, max_len=None, seq_col='Peptide', hla_col='HLA', rank_thr=0.25):
    """

    Args:
        df:
        ics_dict:
        max_len:
        seq_col:
        hla_col:
        rank_thr:

    Returns:

    """
    if 'len' not in df.columns:
        df['len'] = df[seq_col].apply(len)
    if max_len is not None:
        df = df.query('len<=@max_len')
    else:
        max_len = df['len'].max()
    # Weighting the encoding wrt len and HLA
    lens = df['len'].values
    pads = [max_len - x for x in lens]
    hlas = df[hla_col].str.replace('*', '').str.replace(':', '').values
    weights = 1 - np.stack([np.pad(ics_dict[l][hla][rank_thr], pad_width=(0, pad), constant_values=(1, 1)) \
                            for l, hla, pad in zip(lens, hlas, pads)])
    weights = np.expand_dims(weights, axis=2).repeat(len(AA_KEYS), axis=2)
    return weights


def encode_batch_weighted(df, ics_dict, max_len=None,
                          seq_col='Peptide', hla_col='HLA', rank_thr=0.25):
    """
    Takes as input a df containing sequence, len, HLA;
    Batch onehot-encode all sequences & weights them with (1-IC) depending on the ICs dict given

    Args:
        df (pandas.DataFrame): DF containing pep sequence, HLA, optionally 'len'
        ics_dict (dict): Dictionary containing the ICs
        max_len (int): Maximum length to consider
        seq_col (str): Name of the column containing the Peptide sequences (default = 'Peptide')
        hla_col (str): Name of the column containing the HLA alleles (default = 'HLA')
        rank_thr (float): %Rank threshold for the IC selection [0.25, 0.5] (default = 0.25)

    Returns:

    """
    if 'len' not in df.columns:
        df['len'] = df[seq_col].apply(len)
    if max_len is not None:
        df = df.query('len<=@max_len')
    else:
        max_len = df['len'].max()

    # Encoding the sequences
    encoded_sequences = encode_batch(df[seq_col].values, max_len, how='onehot', blosum_matrix=None)

    weights = get_ic_weights(df, ics_dict, max_len, seq_col, hla_col, rank_thr)

    weighted_sequences = torch.from_numpy(weights) * encoded_sequences
    return weighted_sequences.float()


def compute_frequency(onehot_sequence):
    """

    Args:
        onehot_sequence:

    Returns:

    """
    # counts == onehot, use nonzero to get true length (and not padded length)
    non_zero = onehot_sequence.nonzero()
    true_len = len(non_zero[:, 0]) if type(onehot_sequence) == torch.Tensor else len(non_zero[0])
    # Weighted Frequencies for each amino acid = sum of column (aa) divided by true_len
    # If onehot is not weighted, then it's just the true frequency
    frequencies = onehot_sequence.sum(axis=0) / true_len
    return frequencies


def batch_compute_frequency(onehot_sequences):
    """

    currently this stack thing is suboptimal, should probly use the true len
    with bincount and just do it in a vectorized manner

    Args:
        onehot_sequences: (

    Returns:

    """
    # old stuff
    # if type(onehot_sequences) == np.ndarray:
    #     return np.stack([compute_frequency(x) for x in onehot_sequences])
    # elif type(onehot_sequences) == torch.Tensor:
    #     return torch.stack([compute_frequency(x) for x in onehot_sequences])

    # new manner that doesn't use the compute_frequency fct
    non_zeros = onehot_sequences.nonzero()
    true_lens = np.expand_dims(np.bincount(non_zeros[0]), 1) if type(onehot_sequences) == np.ndarray \
        else torch.bincount(non_zeros[:, 0]).unsqueeze(1)
    frequencies = onehot_sequences.sum(axis=1) / true_lens
    return frequencies


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
def get_weights(onehot_seqs):
    """
    Compute the sequences weights using heuristics (1/rs)
    :param onehot_seqs:
    :param counts:
    :return:
    """
    # Get counts
    counts = onehot_seqs.sum(axis=0)
    # Get absolute counts (i.e. # of diff AA in position K --> r)
    abs_counts = counts.copy()
    abs_counts[abs_counts > 0] = 1
    rs = abs_counts.sum(axis=1)
    # Get total count of each aa per position --> s
    ss = (onehot_seqs * counts).sum(axis=2)
    # weights = 1/sum(r*s)
    weights = (1 / (np.multiply(rs, ss))).sum(axis=1)
    n_eff = rs.sum() / onehot_seqs.shape[1]
    # Reshaping to fit the right shape to allow pointwise mul with onehot
    weights = np.expand_dims(np.tile(weights, (onehot_seqs.shape[1], 1)).T, axis=2).repeat(20, axis=2)
    return weights, n_eff


def compute_pfm(sequences, how='shannon', seq_weighting=False, beta=50):
    """
    Computes the position frequency matrix or pseudofrequency given a list of sequences
    """
    max_len = max([len(x) for x in sequences])
    N = len(sequences)
    onehot_seqs = encode_batch(sequences, max_len, how='onehot', blosum_matrix=None).numpy()

    if how == 'shannon':
        freq_matrix = onehot_seqs.sum(axis=0) / N
        return freq_matrix

    elif how == 'kl':
        weights, neff = get_weights(onehot_seqs) if seq_weighting else (1, len(sequences))
        # return weights, neff
        onehot_seqs = weights * onehot_seqs
        alpha = neff - 1
        freq_matrix = onehot_seqs.sum(axis=0) / N
        g_matrix = np.matmul(_blosum62, freq_matrix.T).T
        p_matrix = (alpha * freq_matrix + beta * g_matrix) / (alpha + beta)
        return p_matrix


def compute_ic_position(matrix, position):
    """

    Args:
        matrix:
        position:

    Returns:

    """
    row = matrix[position]
    row_log20 = np.nan_to_num(np.log(row) / np.log(20), neginf=0)
    ic = 1 + np.sum(row * row_log20)
    return ic


def compute_ic(sequences, how='shannon', seq_weighting=True, beta=50):
    """
    returns the IC for sequences of a given length based on the frequency matrix
    Args:
        sequences (list) : list of strings (sequences) from which to compute the IC
        how (str): 'shannon' or 'kl' for either shannon or kullback leibler PFM
    Returns:
        ic_array (np.ndarray) : A Numpy array of the information content at each position (of shape max([len(seq) for seq in sequences]), 1)
    """
    # if type(sequences) == np.ndarray:
    #     return np.array([compute_ic_position(sequences, pos) for pos in range(sequences.shape[0])])
    pfm = compute_pfm(sequences, how, seq_weighting, beta)
    ic_array = np.array([compute_ic_position(pfm, pos) for pos in range(pfm.shape[0])])
    return ic_array


def get_mia(ic_array, threshold=0.3):
    return np.where(ic_array < threshold)[0]
