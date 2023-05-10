import multiprocessing
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from functools import partial
from tqdm.auto import tqdm
from src.metrics import get_metrics, get_mean_roc_curve
import math
N_CORES = multiprocessing.cpu_count() - 2


def bootstrap_wrapper(y_score, y_true, seed, auc01=False, add_roc=False, reduced=True):
    np.random.seed(seed)
    sample_idx = np.random.randint(0, len(y_score), len(y_score))
    sample_score = y_score[sample_idx]
    sample_true = y_true[sample_idx]

    try:
        test_results = get_metrics(sample_true, sample_score, reduced=reduced)
    except:
        return pd.DataFrame(), (None, None, None, None)

    # Save to get mean curves after
    if add_roc or not reduced:
        roc_curve = (test_results.pop('roc_curve'), test_results['auc'], test_results['auc_01']) if auc01 \
            else (test_results.pop('roc_curve'), test_results['auc'])
        # Delete PR curve and not saving because we don't use it at the moment
        _ = (test_results.pop('pr_curve'), test_results['prauc'])
    else:
        roc_curve = None

    bootstrapped_df = pd.DataFrame(test_results, index=[0])
    # bootstrapped_df['seed'] = seed

    unique_id = [seed]+ [str(sample_idx[0]), str(sample_idx[-1])] + \
                [str(sample_idx[math.floor(-1+len(sample_idx)/x)])[-1] for x in range(1,7)]
    bootstrapped_df['id'] = str(unique_id[0])+'_'+''.join([x for x in unique_id[1:]])
    return bootstrapped_df, roc_curve


def bootstrap_downsample_wrapper(df, downsample_label, downsample_number, score_col, target_col, seed):
    """
    used to downsample positives or negatives
    Args:
        downsample_label:
        downsample_number:
        df:
        score_col:
        target_col:
        seed:

    Returns:

    """
    np.random.seed(seed)
    # Downsampling
    downsample = df.query(f'{target_col} == @downsample_label').sample(int(downsample_number), random_state=seed)
    sample_df = pd.concat([df.query(f'{target_col} != @downsample_label'), downsample])
    y_score = sample_df[score_col].values
    y_score = -1 * y_score if 'rank' in score_col.lower() else y_score
    y_true = sample_df[target_col].values
    sample_idx = np.random.randint(0, len(y_score), len(y_score))
    sample_score = y_score[sample_idx]
    sample_true = y_true[sample_idx]

    try:
        test_results = get_metrics(sample_true, sample_score)
    except:
        return pd.DataFrame(), (None, None, None, None)

    # Save to get mean curves after
    roc_curve = (test_results.pop('roc_curve'), test_results['auc'])
    # Delete PR curve and not saving because we don't use it at the moment
    _ = (test_results.pop('pr_curve'), test_results['prauc'])
    return pd.DataFrame(test_results, index=[0]), roc_curve


def bootstrap_downsample(df, downsample_label, downsample_number, score_col, target_col='agg_label', n_rounds=10000,
                         n_jobs=N_CORES):
    wrapper = partial(bootstrap_downsample_wrapper,
                      df, downsample_label=downsample_label, downsample_number=downsample_number,
                      score_col=score_col, target_col=target_col)
    print('Sampling')
    output = Parallel(n_jobs=n_jobs)(delayed(wrapper)(seed=seed) for seed in
                                     tqdm(range(n_rounds), desc='Bootstrapping rounds', position=1, leave=False))

    print('Making results DF and curves')
    result_df = pd.concat([x[0] for x in output])
    mean_roc_curve = get_mean_roc_curve([x[1] for x in output if x[1][0] is not None])
    # mean_pr_curve = get_mean_pr_curve([x[2] for x in output])
    return result_df, mean_roc_curve


def bootstrap_eval(y_score, y_true, n_rounds=10000, n_jobs=N_CORES, auc01=False, add_roc=False, reduced=True):
    """
    Takes the score, true labels, returns bootstrapped DF + mean rocs
    Args:
        y_score:
        y_true:
        n_rounds:
        n_jobs:
        auc01:

    Returns:
        bootstrapped_df
        mean_roc
    """
    wrapper = partial(bootstrap_wrapper, reduced=reduced, add_roc=add_roc,
                      y_score=y_score, y_true=y_true, auc01=auc01)
    print('Sampling')
    output = Parallel(n_jobs=n_jobs)(delayed(wrapper)(seed=seed) for seed in
                                     tqdm(range(n_rounds), desc='Bootstrapping rounds', position=1, leave=False))

    print('Making results DF and curves')
    result_df = pd.concat([x[0] for x in output])
    if add_roc:
        mean_roc_curve = get_mean_roc_curve([x[1] for x in output if x[1][0] is not None], auc01=auc01)
        # mean_pr_curve = get_mean_pr_curve([x[2] for x in output])
        return result_df, mean_roc_curve
    else:
        return result_df


def bootstrap_df_score(df, score_col, target_col='agg_label', n_rounds=10000, n_jobs=N_CORES, auc01=False):
    """
    Does the same as bootstrap_eval but with a custom score_columns instead of taking as input the arrays
    of scores and labels
    Args:
        df: df containing the true labels and predictions/scores/whichever
        score_col: the name of the score columns (ex: 'pred', 'MixMHCrank', etc)
        target_col: the name of the target columns
        n_rounds: # of bootstrapping rounds
        n_jobs: # of parallel jobs

    Returns:

    """
    scores = -1 * df[score_col].values if 'rank' in score_col.lower() else df[score_col].values
    labels = df[target_col].values
    wrapper = partial(bootstrap_wrapper, y_score=scores, y_true=labels, auc01=auc01)
    output = Parallel(n_jobs=n_jobs)(delayed(wrapper)(seed=seed) for seed in
                                     tqdm(range(n_rounds), desc='Bootstrapping rounds', position=1, leave=False))

    print('Making results DF and curves')
    result_df = pd.concat([x[0] for x in output])
    mean_roc_curve = get_mean_roc_curve([x[1] for x in output if x[1][0] is not None])
    # mean_pr_curve = get_mean_pr_curve([x[2] for x in output])
    return result_df, mean_roc_curve

def get_pval_wrapper(df_a, df_b, column='auc'):
    df_a.sort_values('id', inplace=True)
    df_a.reset_index(drop=True, inplace=True)
    df_b.sort_values('id', inplace=True)
    df_b.reset_index(drop=True, inplace=True)
    assert all(df_a.id==df_b.id), 'wrong IDs!'
    return get_pval(df_a[column].values, df_b[column].values)

def get_pval(sample_a, sample_b):
    """
    Null hypothesis : Sample A !>= Sample B
    Alt. hypothesis : Sample A >> Sample B

    Returns the bootstrapped pval that sample_a > sample_b

    Ex: sample_a is the AUCs for a given cdt
        sample_b is the AUCs for another condition
        --> Check that condition A works better than B
    Args:
        sample_a: an array-like of values of size N
        sample_b: an array-like of values of size N

    Returns:
        pval : P value
        sig : significance symbol
    """
    # If both are not the same size can't do the comparison
    assert len(sample_a) == len(sample_b), 'Provided samples don\'t have the same length!' \
                                           f'Sample A: {len(sample_a)}, Sample B: {len(sample_b)}'

    pval = 1 - (len((sample_a > sample_b).astype(int).nonzero()[0]) / len(sample_a))

    sig = '*' if pval < .05 and pval >= 0.01 else '**' if pval < .01 and pval >= 0.001 \
        else '***' if pval < 0.001 and pval >= 0.0001 else '****' if pval < 0.0001 else 'ns'
    return pval, sig


def plot_pval(axis, pval, sig, x0, x1, y, h=0.015, color='k'):
    # Rounds the label to the relevant decimal
    pvstr = str(pval)
    if sig == '****':
        print(pval)
        # label = f'{sig}, p={pval:.1e}'
        label = f'{sig}, p<1e-4'
    else:
        label = f'{sig}, p={round(pval, pvstr.rfind(pvstr.lstrip("0.")))}'
    # Drawing Pval */ns rectangles
    # x1, x2 = 0, 1
    # y, h, col = df['similarity'].max() + 0.015, 0.015, 'k'
    axis.plot([x0, x0, x1, x1], [y, y + h / 1.25, y + h / 1.25, y], lw=1.5, c=color)
    axis.text((x0 + x1) * .5, y + h, label, ha='center', va='bottom', color=color)
