import pandas as pd
import numpy as np
from src.data_processing import AA_KEYS, BL50, BL62

# Alignment stuff used in mutation type
def smith_waterman_alignment(query="VLLP", database="VLILP", scoring_scheme=BL62, gap_open=-2, gap_extension=-1):
    # Matrix imensions
    M = len(query)
    N = len(database)

    # D matrix change to float
    D_matrix = np.zeros((M + 1, N + 1), np.int)

    # P matrix
    P_matrix = np.zeros((M + 1, N + 1), np.int)

    # Q matrix
    Q_matrix = np.zeros((M + 1, N + 1), np.int)

    # E matrix
    E_matrix = np.zeros((M + 1, N + 1), dtype=object)

    # Initialize matrices
    for i in range(M, 0, -1):
        # Here you might include  penalties for end gaps, i.e
        # alignment_matrix[i-1, N] = alignment_matrix[i, N] + gap_open
        D_matrix[i - 1, N] = 0
        P_matrix[i - 1, N] = 0
        Q_matrix[i - 1, N] = 0
        E_matrix[i - 1, N] = 0

    for j in range(N, 0, -1):
        # Here you might include  penalties for end gaps, i.e
        # alignment_matrix[M, j-1] = alignment_matrix[M, j] + gap_open
        D_matrix[M, j - 1] = 0
        P_matrix[M, j - 1] = 0
        Q_matrix[M, j - 1] = 0
        E_matrix[M, j - 1] = 0

    # Main loop
    D_matrix_max_score, D_matrix_i_max, D_matrix_i_max = -9, -9, -9
    for i in range(M - 1, -1, -1):
        for j in range(N - 1, -1, -1):

            # Q_matrix[i,j] entry
            gap_open_database = D_matrix[i + 1, j] + gap_open
            gap_extension_database = Q_matrix[i + 1, j] + gap_extension
            max_gap_database = max(gap_open_database, gap_extension_database)

            Q_matrix[i, j] = max_gap_database

            # P_matrix[i,j] entry
            gap_open_query = D_matrix[i, j + 1] + gap_open
            gap_extension_query = P_matrix[i, j + 1] + gap_extension
            max_gap_query = max(gap_open_query, gap_extension_query)

            P_matrix[i, j] = max_gap_query

            # D_matrix[i,j] entry
            diagonal_score = D_matrix[i + 1, j + 1] + scoring_scheme[database[j]][query[i]]

            # E_matrix[i,j] entry
            candidates = [(1, diagonal_score),
                          (2, gap_open_database),
                          (4, gap_open_query),
                          (3, gap_extension_database),
                          (5, gap_extension_query)]

            direction, max_score = max(candidates, key=lambda x: x[1])

            # check entry sign
            if max_score > 0:
                E_matrix[i, j] = direction
                D_matrix[i, j] = max_score
            else:
                E_matrix[i, j] = 0
                D_matrix[i, j] = 0

            # fetch global max score
            if max_score > D_matrix_max_score:
                D_matrix_max_score = max_score
                D_matrix_i_max = i
                D_matrix_j_max = j

    return P_matrix, Q_matrix, D_matrix, E_matrix, D_matrix_i_max, D_matrix_j_max, D_matrix_max_score


def smith_waterman_traceback(E_matrix, D_matrix, i_max, j_max, query="VLLP", database="VLILP", gap_open=-2,
                             gap_extension=-1):
    M = len(query)
    N = len(database)

    aligned_query = []
    aligned_database = []
    positions = []
    matches = 0

    # start from max_i, max_j
    i, j = i_max, j_max
    while i < M and j < N:

        positions.append([i, j])

        # E[i,j] = 0, stop back tracking
        if E_matrix[i, j] == 0:
            break

        # E[i,j] = 1, match
        if E_matrix[i, j] == 1:
            aligned_query.append(query[i])
            aligned_database.append(database[j])
            if (query[i] == database[j]):
                matches += 1
            i += 1
            j += 1

        # E[i,j] = 2, gap opening in database
        if E_matrix[i, j] == 2:
            aligned_database.append("-")
            aligned_query.append(query[i])
            i += 1

        # E[i,j] = 3, gap extension in database
        if E_matrix[i, j] == 3:

            count = i + 2
            score = D_matrix[count, j] + gap_open + gap_extension

            # Find length of gap (check if score == D_matrix[i, j])
            while ((score - D_matrix[i, j]) * (score - D_matrix[i, j]) >= 0.00001):
                count += 1
                score = D_matrix[count, j] + gap_open + (count - i - 1) * gap_extension

            for k in range(i, count):
                aligned_database.append("-")
                aligned_query.append(query[i])
                i += 1

        # E[i,j] = 4, gap opening in query
        if E_matrix[i, j] == 4:
            aligned_query.append("-")
            aligned_database.append(database[j])
            j += 1

        # E[i,j] = 5, gap extension in query
        if E_matrix[i, j] == 5:

            count = j + 2
            score = D_matrix[i, count] + gap_open + gap_extension

            # Find length of gap (check if score == D_matrix[i, j])
            while ((score - D_matrix[i, j]) * (score - D_matrix[i, j]) >= 0.0001):
                count += 1
                score = D_matrix[i, count] + gap_open + (count - j - 1) * gap_extension

            for k in range(j, count):
                aligned_query.append("-")
                aligned_database.append(database[j])
                j += 1
    return aligned_query, aligned_database, matches


def pipeline_align(query, database, blosum=BL62, gap_open=-1, gap_extension=-1, print_=False):
    P_matrix, Q_matrix, D_matrix, E_matrix, i_max, j_max, max_score = smith_waterman_alignment(query, database, BL62,
                                                                                               gap_open, gap_extension)
    aligned_query, aligned_database, matches = smith_waterman_traceback(E_matrix, D_matrix, i_max, j_max, query,
                                                                        database, gap_open, gap_extension)
    if print_:
        return ''.join(aligned_query), ''.join(aligned_database), matches  # /len(aligned_query)
    else:
        return matches / len(aligned_query)


# Only keep Wild Type which are peptides (& not some protein name)
def check_wt(string):
    if ' ' in string:
        return False
    if any([(x not in AA_KEYS) for x in string]):
        return False
    if string != string.upper():
        return False
    else:
        return True


def get_mutation_type(mutant, wildtype):
    if mutant == wildtype:
        return 'same'
    if len(mutant) == len(wildtype):
        # Allowing up to 3 substitution (some of them could be on anchors)
        if sum(1 for a, b in zip(mutant, wildtype) if a != b) / len(mutant) <= 3 / 8:
            return 'substitution'
        else:
            return 'else'
    elif len(mutant) != len(wildtype):
        try:
            aligned_query, aligned_database, matches = pipeline_align(mutant, wildtype, print_=True)
        except:
            print(mutant, wildtype)
        len_align = len(aligned_query)
        if len_align == 0:
            return 'else'
        if matches / len_align < 0.6:
            return 'else'
        else:
            # If mutant longer than wildtype, has to be insertion, except frameshift which
            # makes the mutant very different
            if len(mutant) > len(wildtype):
                return 'insertion'
            # Missing amino acids, should be deletions ; could also be frameshift if very different
            if len(mutant) < len(wildtype):
                return 'deletion'


def get_mutation_pos(mutant, wildtype, mut_type):
    if mutant == wildtype:
        return str(-1)
    if mut_type == 'substitution':
        # Here just use the full mutant & wt, and not the core
        return ','.join([str(i) for i, z in enumerate([x != y for x, y in zip(mutant, wildtype)]) if z])
    else:
        return str(-1)


def get_anchor(allele, ic_dict, threshold=0.1615):
    """
    rank is the % rank at which we take the IC ; will use .25
    threshold is the threshold for info content to count as anchor
    ONLY USE 9MER MOTIF ; Use core/wt_core to check if anchor mutation if len!=9
    """
    try:
        info_content = ic_dict[9][allele][.25]
    except: print(allele)
    return ','.join(np.where(info_content >= threshold)[0].astype(str))


def get_anchor_mutation(mutant, wildtype, mut_core, wt_core, len_, anchor, mut_type):
    """
    Check if len==9 ; If not, use the core / wt_core to get mutation anchor or not
    """
    if mut_type != 'substitution':
        return False
    if len_ == 9:
        return str(get_mutation_pos(mutant, wildtype, mut_type)) in anchor
    else:
        if mut_core == wt_core:
            return False
        else:
            return any([x in anchor for x in get_mutation_pos(mut_core, wt_core, mut_type)])


def get_binder_type(mut_rank, wt_rank):
    """
    Based on the fact that Improved binder should mean that the immune system hasn't
    tolerized against the wild-type peptide, i.e. wt_rank>2%
    So WT should technically be a non-binder to get improved ;
    If WT is a non binder, can check that the ratio of of wt_rank/mut_rank > 1.35 (value decided empirically)

    If WT is a binder, can still consider a 5-fold ratio to be an improvement.
    ex: wt_rank = 0.5%, mut_rank = 0.1%
    """
    ratio = wt_rank / mut_rank
    if wt_rank >= 2:
        if mut_rank < 2:
            if ratio < 3:  #
                return 'Conserved'
            else:
                return 'Improved'
        elif mut_rank >= 2:
            return 'Conserved'

    elif wt_rank < 2:
        if mut_rank >= 2:
            return 'Conserved'
        elif mut_rank < 2:
            if ratio >= 4:  # Chose 4 because this adds exactly 101 peptides to the Improved class lol
                return 'Improved'
            else:
                return 'Conserved'