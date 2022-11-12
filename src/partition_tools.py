import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold


def read_hobohm(filename, original_df, pep_col='Peptide', hla_col='HLA', elrank_col='trueHLA_EL_rank',
                target_col='agg_label'):
    """
    Reads the output of a hobohm reduced file
    (ex would read "dataset.out" from ./hobohm1_pepkernel input.pep > dataset.out
    Args:
        filename: Filepath to the output file of the hobohm script
        original_df: DataFrame or the path to the original dataframe used to run Hobohm
        pep_col : Column of DataFrame containing the Peptide
        hla_col : Column of DataFrame containing the HLA
        elrank_col : Column of DataFrame containing the ELrank
    Returns:
        unique_df (pd.DataFrame): DataFrame containing the unique sequences
        not_unique_df (pd.DataFrame): DataFrame containing the non-unique sequences
    """
    assert type(original_df) in [pd.DataFrame, str], 'original_df must be a Pandas DataFrame or a filepath!'
    if type(original_df) != pd.DataFrame:
        try:
            original_df = pd.read_csv(original_df)
        except:
            raise ValueError(f"Couldn't read or access {original_df} of type {type(original_df)}!")
    # Get unique df and merge to original df;
    # Given Hobohm, all identical sequences with different HLA should re-appear and will need to be
    # addressed later in another function
    unique_df = pd.read_csv(filename, header=None, comment='#')
    unique_df.columns = [pep_col]
    unique_df = unique_df.merge(original_df[[pep_col, hla_col, elrank_col, target_col]],
                                left_on=pep_col, right_on=pep_col)

    with open(filename, 'r') as f:
        lines = [l.strip('\n') for l in f.readlines()]
    # Get the non-unique saved somewhere to re-assign them later
    values = []
    for l in lines:
        if 'Not unique' not in l: continue
        l = l.replace('\n', '')
        l = [x.split(' ') for x in l.split('# Not unique. ')[1].split(' is similar to ')]
        idx = l[0][0]
        discarded = l[0][1]
        similar = l[1][0]
        similarity_score = l[1][1]
        values.append([idx, similar, discarded, similarity_score])
    not_unique_df = pd.DataFrame(values, columns=['drop_idx', 'similar', 'discarded', 'similarity_score'])
    not_unique_df['self'] = not_unique_df['similar'] == not_unique_df['discarded']
    return unique_df, not_unique_df


def manually_reassign_identical(k, unique_df, pep_col='Peptide'):
    """
    Assumes the unique_df contains identical sequences with different HLAs due to the previous merge operation.
    Assumes the unique_df has already been KFold separated. Then, identical sequences with different HLAs will be reassigned
    to the same fold.
    Args:
        k: K in K fold
        unique_df: The unique_df that has previously been kfold split
        pep_col: column containing the Peptide

    Returns:
        unique_df: The unique_df that has undergone re-assignment
    """

    assignment_counts = {x: 0 for x in range(k)}
    # Go through all the duplicated peps
    for pep in unique_df.loc[unique_df.duplicated(pep_col, keep=False)][pep_col].unique():
        tmp = unique_df.loc[unique_df[pep_col] == pep]
        # if already all the same fold then it's fine, continue
        if len(tmp.fold.unique()) == 1: continue
        # otherwise: assign to a fold that has the least assignments (starts at 0)
        counts = [assignment_counts[k] for k in tmp.fold.values]
        new_assignment = tmp.fold.values[counts.index(min(counts))]
        unique_df.loc[unique_df[pep_col] == pep, 'fold'] = new_assignment
        assignment_counts[new_assignment] += 1
    return unique_df


def manually_reassign_related(unique_df, not_unique_df, pep_col='Peptide', hla_col='HLA', elrank_col='trueHLA_EL_rank',
                              target_col='agg_label'):
    """
    Manually reassigns non-unique (discarded) peptide to the same fold as their related pep
    Args:
        unique_df (pd.DataFrame): DataFrame containing the unique sequences
        not_unique_df (pd.DataFrame): DataFrame containing the non-unique sequences
        pep_col: Column of DataFrame containing the Peptide
        hla_col: Column of DataFrame containing the HLA
        elrank_col: Column of DataFrame containing the ELrank
    Returns:
        not_unique-df (pd.DataFrame): Non-unique DataFrame with sequences re-assigned to the correct fold
    """
    # Re-assigning
    not_unique_df['fold'] = not_unique_df.apply(lambda x: unique_df.query(f'{pep_col}==@x.similar')['fold'].unique()[0],
                                                axis=1)
    not_unique_df = not_unique_df[['discarded', target_col, hla_col, elrank_col, 'fold']].rename(
        columns={'discarded': pep_col})

    return not_unique_df


def stratified_kfold_unique(unique_df, not_unique_df, original_df,
                     k=5, shuffle=True, seed=13,
                     pep_col='Peptide', hla_col='HLA', elrank_col='trueHLA_EL_rank', target_col='agg_label'):
    # Stratify KFold on the unique set, based on duplicated counts as stratifying group.
    # Then repopulate the folds with
    not_unique_df['drop_idx'] = not_unique_df['drop_idx'].astype(int)
    # Get the strat KF object
    stratkf = StratifiedKFold(k, shuffle=shuffle, random_state=seed)
    # Merge the not unique (to get agg_label, i.e. "y" for stratkf to split and get the duplicated counts
    not_unique_df = not_unique_df.merge(original_df.reset_index()[['index', target_col, hla_col, elrank_col]],
                                     left_on='drop_idx', right_on='index')

    dup_counts = not_unique_df.groupby('similar').agg({'self': 'count'})\
                              .sort_values('self', ascending=False).reset_index()

    dup_counts = dup_counts.merge(unique_df[[pep_col, target_col]], left_on='similar',
                                  right_on=pep_col).drop_duplicates(['similar', target_col])
    # Merge and assign the duplicated counts, to be used as stratify groups
    unique_df['counts'] = 0
    tmp = unique_df.reset_index().merge(dup_counts[[pep_col, 'self']], left_on=pep_col, right_on=pep_col)
    unique_df.iloc[tmp['index'].values, unique_df.columns.get_loc('counts')] = tmp['self']

    # Ready to stratify and set the folds
    unique_df['fold'] = np.nan
    for i, (train_idx, test_idx) in enumerate(
            stratkf.split(unique_df[pep_col].values, unique_df[target_col], groups=unique_df['counts'])):
        unique_df.iloc[test_idx, unique_df.columns.get_loc('fold')] = i
    unique_df.fold = unique_df.fold.astype(int)
    return unique_df, not_unique_df


def pipeline_stratified_kfold(hobohm_filename, original_df, k=5, shuffle=True, seed=13,
                              pep_col='Peptide', hla_col='HLA', elrank_col='trueHLA_EL_rank', target_col='agg_label'):
    """

    Args:
        hobohm_filename:
        original_df:
        k:
        shuffle:
        seed:
        pep_col:
        hla_col:
        elrank_col:

    Returns:
        dataset (pd.DataFrame): dataset with assigned folds
    """
    # original_df = original_df.sort_values(pep_col).reset_index(drop=True)
    unique_df, not_unique_df = read_hobohm(hobohm_filename, original_df, pep_col, hla_col, elrank_col, target_col)
    print('read hobohm', len(unique_df), len(not_unique_df))
    unique_df, not_unique_df = stratified_kfold_unique(unique_df, not_unique_df, original_df, k, shuffle, seed,
                                                pep_col, hla_col, elrank_col, target_col )
    print('strat kfold unique', len(unique_df), len(not_unique_df))
    unique_df = manually_reassign_identical(k, unique_df, pep_col)
    print('manually reassign identical', len(unique_df), len(not_unique_df))
    not_unique_df = manually_reassign_related(unique_df, not_unique_df, pep_col, hla_col, elrank_col, target_col)
    print('manually reassign related', len(unique_df), len(not_unique_df))
    dataset = pd.concat([unique_df, not_unique_df], ignore_index=True) \
        .sort_values(pep_col, ascending=True).reset_index(drop=True).drop(columns=['counts'])
    print('concat', len(dataset))
    merge_cols = [pep_col, hla_col]
    merge_cols.extend(original_df.columns.difference(dataset.columns))
    dataset = dataset.merge(original_df[merge_cols], left_on=[pep_col, hla_col], right_on=[pep_col, hla_col])
    print('merge', len(dataset))

    # Bugfix: Drops some unique that were duplicated during the manual re-assignment
    dataset = dataset.drop(dataset.loc[dataset.duplicated(keep='first')].index)
    print('drop duplicated', len(dataset))
    return dataset
