import multiprocessing
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from functools import partial
from tqdm.auto import tqdm
from src.metrics import get_metrics, get_mean_roc_curve

N_CORES = multiprocessing.cpu_count() - 2


def bootstrap_wrapper(y_score, y_true, seed):
    np.random.seed(seed)
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


def bootstrap_eval(y_score, y_true, n_rounds=10000, n_jobs=N_CORES):
    wrapper = partial(bootstrap_wrapper,
                      y_score=y_score, y_true=y_true)
    print('Sampling')
    output = Parallel(n_jobs=n_jobs)(delayed(wrapper)(seed=seed) for seed in
                                     tqdm(range(n_rounds), desc='Bootstrapping rounds', position=1, leave=False))

    print('Making results DF and curves')
    result_df = pd.concat([x[0] for x in output])
    mean_roc_curve = get_mean_roc_curve([x[1] for x in output if x[1][0] is not None])
    # mean_pr_curve = get_mean_pr_curve([x[2] for x in output])
    return result_df, mean_roc_curve


def bootstrap_df_score(df, score_col, target_col='agg_label', n_rounds=10000, n_jobs=N_CORES):
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
    wrapper = partial(bootstrap_wrapper, y_score=scores, y_true=labels)
    output = Parallel(n_jobs=n_jobs)(delayed(wrapper)(seed=seed) for seed in
                                     tqdm(range(n_rounds), desc='Bootstrapping rounds', position=1, leave=False))

    print('Making results DF and curves')
    result_df = pd.concat([x[0] for x in output])
    mean_roc_curve = get_mean_roc_curve([x[1] for x in output if x[1][0] is not None])
    # mean_pr_curve = get_mean_pr_curve([x[2] for x in output])
    return result_df, mean_roc_curve


def get_pval(sample_a, sample_b):
    """
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
    assert len(sample_a)==len(sample_b), 'Provided samples don\'t have the same length!'\
                                        f'Sample A: {len(sample_a)}, Sample B: {len(sample_b)}'

    pval = 1 - (len((sample_a > sample_b).astype(int).nonzero()[0]) / len(sample_a))

    sig = '*' if pval < .05 and pval >= 0.01 else '**' if pval < .01 and pval >= 0.001 \
        else '***' if pval < 0.001 and pval >= 0.0001 else '****' if pval < 0.0001 else 'ns'
    return pval, sig


def plot_pval(axis, pval, sig, x1, x2, y, h=0.015):
    # Rounds the label to the relevant decimal
    pvstr = str(pval)
    label = f'{sig}, p={round(pval, pvstr.rfind(pvstr.lstrip("0.")))}'
    # Drawing Pval */ns rectangles
    # x1, x2 = 0, 1
    # y, h, col = df['similarity'].max() + 0.015, 0.015, 'k'
    axis.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=2, c='k')
    axis.text((x1 + x2) * .5, y + h, label, ha='center', va='bottom', color='k')