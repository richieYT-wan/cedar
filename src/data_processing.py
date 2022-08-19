import copy

import pandas as pd
import numpy as np
import torch
import multiprocessing
import math
from torch.utils.data import TensorDataset
from src.utils import pkl_load, pkl_dump
import os
import warnings
import peptides

warnings.filterwarnings('ignore')

DATADIR = '../data/' if os.path.exists('../data/') else './data/'
OUTDIR = '../output/' if os.path.exists('../output/') else './output/'


def _init(DATADIR):
    #### ==== CONST (blosum, multiprocessing, keys, etc) ==== ####
    VAL = math.floor(4 + (multiprocessing.cpu_count() / 1.5))
    N_CORES = VAL if VAL <= multiprocessing.cpu_count() else int(multiprocessing.cpu_count() - 2)

    DATADIR = f'{DATADIR}Matrices/'
    AA_KEYS = [x for x in 'ARNDCQEGHILKMFPSTWYV']

    CHAR_TO_INT = dict((c, i) for i, c in enumerate(AA_KEYS))
    INT_TO_CHAR = dict((i, c) for i, c in enumerate(AA_KEYS))

    BG = np.loadtxt(f'{DATADIR}bg.freq.fmt', dtype=float)
    BG = dict((k, v) for k, v in zip(AA_KEYS, BG))

    # BLOSUMS 50
    BL50 = {}
    _blosum50 = np.loadtxt(f'{DATADIR}BLOSUM50', dtype=float).T
    for i, letter_1 in enumerate(AA_KEYS):
        BL50[letter_1] = {}
        for j, letter_2 in enumerate(AA_KEYS):
            BL50[letter_1][letter_2] = _blosum50[i, j]
    # BLOSUMS 62
    BL62_DF = pd.read_csv(f'{DATADIR}BLOSUM62', sep='\s+', comment='#', index_col=0)
    BL62 = BL62_DF.to_dict()
    BL62_VALUES = BL62_DF.drop(columns=['B', 'Z', 'X', '*'], index=['B', 'Z', 'X', '*'])
    BL62_VALUES = dict((x, BL62_VALUES.loc[x].values) for x in BL62_VALUES.index)

    # BLOSUMS 62 FREQS
    _blosum62 = np.loadtxt(f'{DATADIR}BLOSUM62.freq_rownorm', dtype=float).T
    BL62FREQ = {}
    BL62FREQ_VALUES = {}
    for i, letter_1 in enumerate(AA_KEYS):
        BL62FREQ[letter_1] = {}
        BL62FREQ_VALUES[letter_1] = _blosum62[i]
        for j, letter_2 in enumerate(AA_KEYS):
            BL62FREQ[letter_1][letter_2] = _blosum62[i, j]
    HLAS = pkl_load(DATADIR.strip('Matrices/')+'ic_dicts/' + 'ics_shannon.pkl')[9].keys()
    return VAL, N_CORES, DATADIR, AA_KEYS, CHAR_TO_INT, INT_TO_CHAR, BG, BL62FREQ, BL62FREQ_VALUES, BL50, BL62, BL62_VALUES, HLAS


VAL, N_CORES, DATADIR, AA_KEYS, CHAR_TO_INT, INT_TO_CHAR, BG, BL62FREQ, BL62FREQ_VALUES, BL50, BL62, BL62_VALUES, HLAS = _init(
    DATADIR)


######################################
####      SEQUENCES ENCODING      ####
######################################


def get_aa_properties(df, seq_col='Peptide'):
    """
    Compute some AA properties that I have selected
    Args:
        df (pandas.DataFrame) : input dataframe, should contain at least the peptide sequences
        seq_col (str) : column name containing the peptide sequences

    Returns:
        out (pandas.DataFrame) : The same dataframe but + the computed AA properties

    """
    out = df.copy()
    out['aliphatic_index'] = out[seq_col].apply(lambda x: peptides.Peptide(x).aliphatic_index())
    out['boman'] = out[seq_col].apply(lambda x: peptides.Peptide(x).boman())
    out['charge_7_4'] = out[seq_col].apply(lambda x: peptides.Peptide(x).charge(pH=7.4))
    out['charge_6_65'] = out[seq_col].apply(lambda x: peptides.Peptide(x).charge(pH=6.65))
    out['hydrophobicity'] = out[seq_col].apply(lambda x: peptides.Peptide(x).hydrophobicity())
    out['isoelectric_point'] = out[seq_col].apply(lambda x: peptides.Peptide(x).isoelectric_point())
    out['PD2'] = out[seq_col].apply(lambda x: peptides.Peptide(x).physical_descriptors()[1])
    vhse = out[seq_col].apply(lambda x: peptides.Peptide(x).vhse_scales())
    for i in range(1, 9):
        out[f'VHSE{i}'] = [x[i - 1] for x in vhse]
    return out, [x for x in out.columns if x not in df.columns]


def encode(sequence, max_len=None, encoding='onehot', blosum_matrix=BL62_VALUES):
    """
    encodes a single peptide into a matrix, using 'onehot' or 'blosum'
    if 'blosum', then need to provide the blosum dictionary as argument
    """

    # One hot encode by setting 1 to positions where amino acid is present, 0 elsewhere
    size = len(sequence)
    if encoding == 'onehot':
        int_encoded = [CHAR_TO_INT[char] for char in sequence]
        onehot_encoded = list()
        for value in int_encoded:
            letter = [0 for _ in range(len(AA_KEYS))]
            letter[value] = 1
            onehot_encoded.append(letter)
        tmp = np.array(onehot_encoded)

    # BLOSUM encode
    if encoding == 'blosum':
        if blosum_matrix is None or not isinstance(blosum_matrix, dict):
            raise Exception('No BLOSUM matrix provided!')

        tmp = np.zeros([size, len(AA_KEYS)], dtype=np.float32)
        for idx in range(size):
            tmp[idx, :] = blosum_matrix[sequence[idx]]

    # Padding if max_len is provided
    if max_len is not None and max_len > size:
        diff = int(max_len) - int(size)
        try:
            tmp = np.concatenate([tmp, np.zeros([diff, len(AA_KEYS)], dtype=np.float32)],
                                 axis=0)
        except:
            print(type(tmp), tmp.shape, len(AA_KEYS), type(diff), type(max_len), type(size), sequence)
            #     return tmp, diff, len(AA_KEYS)
            raise Exception
    return torch.from_numpy(tmp).float()


def encode_batch(sequences, max_len=None, encoding='onehot', blosum_matrix=None):
    """
    Encode multiple sequences at once.
    """
    if max_len is None:
        max_len = max([len(x) for x in sequences])

    return torch.stack([encode(seq, max_len, encoding, blosum_matrix) for seq in sequences])


def onehot_decode(onehot_sequence):
    if type(onehot_sequence) == np.ndarray:
        return ''.join([INT_TO_CHAR[x.item()] for x in onehot_sequence.nonzero()[1]])
    elif type(onehot_sequence) == torch.Tensor:
        return ''.join([INT_TO_CHAR[x.item()] for x in onehot_sequence.nonzero()[:, 1]])


def onehot_batch_decode(onehot_sequences):
    return np.stack([onehot_decode(x) for x in onehot_sequences])


def get_ic_weights(df, ics_dict: dict, max_len=None, seq_col='Peptide', hla_col='HLA', rank_thr=0.25, mask=False):
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
    # If mask is true, then the weight is just a 0-1 mask filter
    # Using the conserved / MIAs positions instead of the ICs
    if mask:
        # Get mask for where the values should be thresholded to 0 and 1
        weights = np.stack([np.pad(ics_dict[l][hla][rank_thr], pad_width=(0, pad), constant_values=(1, 1)) \
                            for l, hla, pad in zip(lens, hlas, pads)])
        # IC > 0.3 goes to 0 because anchor position
        # IC <= 0.3 goes to 1 because "MIA" position
        idx_min = (weights > 0.3)
        idx_max = (weights <= 0.3)
        weights[idx_min] = 0
        weights[idx_max] = 1
    # Else we get the weight with the 1-IC depending on the IC dict provided
    else:
        weights = 1 - np.stack([np.pad(ics_dict[l][hla][rank_thr], pad_width=(0, pad), constant_values=(1, 1)) \
                                for l, hla, pad in zip(lens, hlas, pads)])

    weights = np.expand_dims(weights, axis=2).repeat(len(AA_KEYS), axis=2)
    return weights


def encode_batch_weighted(df, ics_dict=None, device=None, max_len=None, encoding='onehot', blosum_matrix=BL62_VALUES,
                          seq_col='Peptide', hla_col='HLA', target_col='agg_label', rank_thr=0.25, mask=False):
    """
    Takes as input a df containing sequence, len, HLA;
    Batch onehot-encode all sequences & weights them with (1-IC) depending on the ICs dict given

    Args:
        target_col:
        df (pandas.DataFrame): DF containing pep sequence, HLA, optionally 'len'
        ics_dict (dict): Dictionary containing the ICs
        device (str) : device for cpu or cuda transfer
        max_len (int): Maximum length to consider
        encoding (str) : 'onehot' or 'blosum'
        blosum_matrix : The blosum matrix dictionary; Should just use the BL62_VALUES that's initialized by default
        seq_col (str): Name of the column containing the Peptide sequences (default = 'Peptide')
        hla_col (str): Name of the column containing the HLA alleles (default = 'HLA')
        rank_thr (float): %Rank threshold for the IC selection [0.25, 0.5] (default = 0.25)

    Returns:
        weighted_sequence (torch.Tensor): Tensor containing the weighted onehot-encoded peptide sequences.
    """
    df = verify_df_(df, seq_col, hla_col, target_col)
    if 'len' not in df.columns:
        df['len'] = df[seq_col].apply(len)
    if max_len is not None:
        df = df.query('len<=@max_len')
    else:
        max_len = df['len'].max()

    # Encoding the sequences
    encoded_sequences = encode_batch(df[seq_col].values, max_len, encoding=encoding, blosum_matrix=blosum_matrix)
    if ics_dict is not None:
        weights = get_ic_weights(df, ics_dict, max_len, seq_col, hla_col, rank_thr, mask)
    else:
        # Here, if no ics_dict is provided, the normal weight will just be ones everywhere
        # In case we are not doing weighted sequence (either for onehot-input or frequency computation)
        weights = np.ones(encoded_sequences.shape)
    weighted_sequences = torch.from_numpy(weights) * encoded_sequences

    if device is None:
        return weighted_sequences.float()
    else:
        return weighted_sequences.to(device).float()


def query_dataframe_fold(dataframe, fold_inner, fold_outer):
    train_data = dataframe.query('fold != @fold_inner and fold != @fold_outer')
    valid_data = dataframe.query('fold == @fold_inner')
    return train_data, valid_data


def verify_df_(df, seq_col, hla_col, target_col):
    df = copy.deepcopy(df)
    unique_labels = sorted(df[target_col].dropna().unique())
    # Checks binary label
    assert ([int(x) for x in sorted(unique_labels)]) in [[0, 1], [0], [1]], f'Labels are not 0, 1! {unique_labels}'
    # Checks if any seq not in alphabet
    df = df.drop(df.loc[df[seq_col].apply(lambda x: any([z not in AA_KEYS for z in x]))].index)
    # Checks if HLAs have correct format
    if all(df[hla_col].apply(lambda x: not x.startswith('HLA-'))):
        df[hla_col] = df[hla_col].apply(lambda x: 'HLA-' + x)
    df[hla_col] = df[hla_col].apply(lambda x: x.replace('*', '').replace(':', ''))
    # Check HLA only in subset
    df = df.query(f'{hla_col} in @HLAS')

    return df


def get_tensor_dataset(df, ics_dict, device, max_len=12, encoding='onehot', blosum_matrix=BL62_VALUES,
                       seq_col='Peptide', hla_col='HLA', target_col='agg_label', rank_thr=0.25,
                       mask=False, add_rank=False, add_chem=False):
    """
        Takes as input a df containing sequence, len, HLA;
        Batch onehot-encode all sequences & weights them with (1-IC) depending on the ICs dict given
        Stacks it with the targets in another dimension
    Args:
        df:
        ics_dict:
        device:
        max_len:
        encoding:
        blosum_matrix:
        seq_col:
        hla_col:
        target_col:
        rank_thr:

    Returns:
        tensor_dataset (torch.utils.data.TensorDataset): Dataset containing the tensors X and y
    """
    df = verify_df_(df, seq_col, hla_col, target_col)
    x = encode_batch_weighted(df, ics_dict, device, max_len, encoding, blosum_matrix,
                              seq_col, hla_col, target_col, rank_thr, mask)
    y = torch.from_numpy(df[target_col].values).float().unsqueeze(1).to(device)
    dataset = TensorDataset(x, y)
    return dataset


def get_array_dataset(df, ics_dict, max_len=12, encoding='onehot', blosum_matrix=BL62_VALUES,
                      seq_col='Peptide', hla_col='HLA', target_col='agg_label', rank_col='trueHLA_EL_rank',
                      rank_thr=0.25, mask=False, add_rank=False, add_aaprop=False, remove_pep=False, standardize=False):
    """
        Computes the frequencies as the main features
        Takes as input a df containing sequence, len, HLA;
        Batch encode all sequences & weights them with (1-IC) depending on the ICs dict given
        Stacks it with the targets in another dimension, return as a np.ndarray
        Big mess of a fct to be honest :-)
    Args:
        df:
        ics_dict:
        max_len:
        encoding:
        blosum_matrix:
        seq_col:
        hla_col:
        target_col:
        rank_thr:
        add_rank:
        mask:
        add_aaprop:
        remove_pep: Boolean switch to discard the AA sequence/freq in features (e.g. keep only rank or only chem props)
        standardize: Exists here because I'm bad at coding so it doesn't throw an error if I had thought about this in a better way

    Returns:
        tensor_dataset (torch.utils.data.TensorDataset): Dataset containing the tensors X and y
    """
    df = verify_df_(df, seq_col, hla_col, target_col)
    x = batch_compute_frequency(encode_batch_weighted(df, ics_dict, 'cpu', max_len, encoding, blosum_matrix,
                                                      seq_col, hla_col, target_col, rank_thr, mask).numpy())

    if add_rank:
        ranks = np.expand_dims(df[rank_col].values, 1)
        x = np.concatenate([x, ranks], axis=1)

    if add_aaprop:
        df_props, columns = get_aa_properties(df, seq_col)
        aa_props = df_props[columns].values
        x = np.concatenate([x, aa_props], axis=1)

    y = df[target_col].values
    # Queries whatever is above 20, and only keeps that as feature
    if remove_pep and (add_rank or add_aaprop):
        x = x[:, 20:]
    return x, y


def compute_frequency(encoded_sequence):
    """
    ACTUALLY HERE COULD ALSO USE BLOSUM ENCODED SEQ not just onehot
    Compute some kind of combined pseudo frequency with BL62 replacement

    Then have it weighted (or not!)
    Args:
        encoded_sequence:

    Returns:

    """

    # THIS IS THE OLD WAY WITH BINCOUNT WHICH ONLY WORKS WITH ONE HOT ENCODING
    # counts == onehot, use nonzero to get true length (and not padded length)
    # non_zero = encoded_sequence.nonzero()

    # Pretty convoluted way to get it but works for OH and blosum :-)
    mask = (encoded_sequence == 0).all(1)
    true_len = torch.where(mask)[0][0].item() if type(mask) == torch.Tensor else \
        np.where(mask)[0][0].item()

    # Weighted Frequencies for each amino acid = sum of column (aa) divided by true_len
    # If onehot is not weighted, then it's just the true frequency
    frequencies = encoded_sequence.sum(axis=0) / true_len
    return frequencies


def batch_compute_frequency(encoded_sequences):
    """

    Args:
        encoded_sequences:
    Returns:

    """
    # old stuff
    # if type(onehot_sequences) == np.ndarray:
    #     return np.stack([compute_frequency(x) for x in onehot_sequences])
    # elif type(onehot_sequences) == torch.Tensor:
    #     return torch.stack([compute_frequency(x) for x in onehot_sequences])

    # THIS IS THE OLD WAY WITH BINCOUNT WHICH ONLY WORKS WITH ONE HOT ENCODING
    # non_zeros = onehot_sequences.nonzero()
    # true_lens = np.expand_dims(np.bincount(non_zeros[0]), 1) if type(onehot_sequences) == np.ndarray \
    #     else torch.bincount(non_zeros[:, 0]).unsqueeze(1)

    # This is the new way with mask and .all(dim=2) which works with both BLOSUM and OH
    mask = (encoded_sequences == 0).all(2)  # checking on second dim
    true_lens = (mask.shape[1] - torch.bincount(torch.where(mask)[0])).unsqueeze(1) if type(mask) == torch.Tensor else \
        np.expand_dims(mask.shape[1] - np.bincount(np.where(mask)[0]), 1)

    frequencies = encoded_sequences.sum(axis=1) / true_lens

    return frequencies


def standardize(x_train, x_eval, x_test=None):
    """
    Sets mean to 0 and variance to 1 wrt the mean/variance of the training set.
    Args:
        x_train (np.ndarray, torch.Tensor): Train input
        x_eval  (np.ndarray, torch.Tensor): Validation input
        x_test  (np.ndarray, torch.Tensor): Test input

    Returns:

    """
    assert x_train.shape[1:] == x_eval.shape[1:], f'Input have different shapes!' \
                                                  f'train/eval: {x_train.shape[1:], x_eval.shape[1:]}'
    if x_test is not None:
        assert x_train.shape[1:] == x_test.shape[1:], 'Input have different shapes!' \
                                                      f'train/test: {x_train.shape[1:], x_test.shape[1:]}'

    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    x_train_std = (x_train - mu) / sigma
    x_eval_std = (x_eval - mu) / sigma
    if x_test is not None:
        x_test_std = (x_test - mu) / sigma
        return x_train_std, x_eval_std, x_test_std
    else:
        return x_train_std, x_eval_std


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
    onehot_seqs = encode_batch(sequences, max_len, encoding='onehot', blosum_matrix=None).numpy()

    if how == 'shannon':
        freq_matrix = onehot_seqs.sum(axis=0) / N
        return freq_matrix

    elif how == 'kl':
        weights, neff = get_weights(onehot_seqs) if seq_weighting else (1, len(sequences))
        # return weights, neff
        onehot_seqs = weights * onehot_seqs
        alpha = neff - 1
        freq_matrix = onehot_seqs.sum(axis=0) / N
        g_matrix = np.matmul(BL62FREQ, freq_matrix.T).T
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


def get_mia(ic_array, threshold=1/3):
    return np.where(ic_array < threshold)[0]
