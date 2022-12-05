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
    sample_idx = np.random.randint(0, len(y_score)+1, len(y_score))
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
    y_score = -1*y_score if 'rank' in score_col.lower() else y_score
    y_true = sample_df[target_col].values
    sample_idx = np.random.randint(0, len(y_score)+1, len(y_score))
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
